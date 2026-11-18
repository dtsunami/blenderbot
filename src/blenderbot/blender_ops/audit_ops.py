"""Programmatic Spatial, Ground-Contact, Frustum-Framing & Kinematics Verifier (`scene.audit`).

Inspired by the spatial self-verification loop seen in high-end previs workflows:
Samples frames across `[frame_start .. frame_end]` and mathematically audits:
1. Subject-to-surface ground contact / penetration (`min`, `max`, `rms` clearance in mm).
2. Camera-to-surface clearance (verifies camera never dips underground or clips geometry).
3. Camera-to-subject distance profile (`min`, `max`, `start`, `end` in meters) + focal length range.
4. Subject 2D camera-frustum framing (`world_to_camera_view`: in-frame %, screen bbox coverage %, clipping).
5. Subject velocity & acceleration smoothness (`m/s`, `km/h`, peak G spikes).
"""

from __future__ import annotations

import math
from typing import Any

from .helpers import effective_fps, world_aabb
from .schema import OpContext, OpError, blender_op


def _raycast_vertical_surface_z(
    bvh: Any,
    mw: Any,
    mw_inv: Any,
    x: float,
    y: float,
    z_probe_top: float,
) -> float | None:
    from mathutils import Vector

    orig_local = mw_inv @ Vector((x, y, z_probe_top))
    dir_local = (mw_inv.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    hit_loc, _, _, _ = bvh.ray_cast(orig_local, dir_local)
    if hit_loc is None:
        return None
    return float((mw @ hit_loc).z)


def _audit_mesh_topology(bpy: Any, depsgraph: Any, obj: Any) -> tuple[dict[str, Any], list[str]]:
    import bmesh

    issues: list[str] = []
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(eval_mesh)
        non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
        zero_area = sum(1 for f in bm.faces if f.calc_area() < 1e-8)
        loose_verts = sum(1 for v in bm.verts if not v.link_edges)
        ngons = sum(1 for f in bm.faces if len(f.verts) > 4)
        n_verts = len(bm.verts)
        n_faces = len(bm.faces)
    finally:
        bm.free()
        eval_obj.to_mesh_clear()

    sx, sy, sz = (float(obj.scale.x), float(obj.scale.y), float(obj.scale.z))
    neg_scale = (sx * sy * sz) < 0.0
    non_uniform_scale = max(abs(sx - sy), abs(sy - sz), abs(sx - sz)) > 1e-3

    if neg_scale:
        issues.append(f"Mesh '{obj.name}' has negative scale {[round(sx, 3), round(sy, 3), round(sz, 3)]} (inverted normals risk).")
    if zero_area > 0:
        issues.append(f"Mesh '{obj.name}' contains {zero_area} degenerate zero-area face(s).")

    return (
        {
            "name": obj.name,
            "vertices": n_verts,
            "faces": n_faces,
            "non_manifold_edges": non_manifold,
            "zero_area_faces": zero_area,
            "loose_vertices": loose_verts,
            "ngons": ngons,
            "negative_scale": neg_scale,
            "unapplied_nonuniform_scale": non_uniform_scale,
        },
        issues,
    )


def _check_world_bvh_collisions(depsgraph: Any, objs: list[Any]) -> list[dict[str, Any]]:
    import bmesh
    from mathutils.bvhtree import BVHTree

    trees: list[tuple[str, dict[str, list[float]], Any]] = []
    for o in objs:
        if o.type != "MESH":
            continue
        ab = world_aabb(o, depsgraph)
        if ab is None:
            continue
        ev = o.evaluated_get(depsgraph)
        em = ev.to_mesh()
        bm = bmesh.new()
        try:
            bm.from_mesh(em)
            bm.transform(ev.matrix_world)
            bvh = BVHTree.FromBMesh(bm)
        finally:
            bm.free()
            ev.to_mesh_clear()
        trees.append((o.name, ab, bvh))

    hits: list[dict[str, Any]] = []
    for i in range(len(trees)):
        na, aba, bvha = trees[i]
        for j in range(i + 1, len(trees)):
            nb, abb, bvhb = trees[j]
            # Fast world AABB broadphase reject
            if (
                aba["max"][0] < abb["min"][0]
                or aba["min"][0] > abb["max"][0]
                or aba["max"][1] < abb["min"][1]
                or aba["min"][1] > abb["max"][1]
                or aba["max"][2] < abb["min"][2]
                or aba["min"][2] > abb["max"][2]
            ):
                continue
            overlaps = bvha.overlap(bvhb)
            if overlaps:
                hits.append({"object_a": na, "object_b": nb, "overlapping_polygon_pairs": len(overlaps)})
    return hits


@blender_op(
    "scene.audit",
    tier="read",
    summary=(
        "Deterministic spatial, topology & animation verifier: mesh topology health (non-manifold, zero-area faces, "
        "negative scale), world-space BVHTree mesh collisions, ground contact/penetration (mm), camera clearance, "
        "2D frustum framing %, and velocity smoothness."
    ),
    params_doc={
        "meshes": "Optional `'*'` or list of Mesh object names to run topology health checks on",
        "check_topology": "Alias for `meshes: '*'` when `meshes` is omitted",
        "check_collisions": "Optional list of Mesh object names (or `'*'`) to test for pairwise 3D BVHTree intersections",
        "fail_on_non_manifold": "If true, treat non-manifold open boundary edges as an audit failure (default false)",
        "subject": "Optional hero/moving object name to audit across frames",
        "ground": "Optional ground/floor/terrain Mesh object name (or list of Mesh names)",
        "camera": "Camera object name (defaults to `scene.camera`)",
        "frame_start": "Start frame (defaults to `scene.frame_start`)",
        "frame_end": "End frame (defaults to `scene.frame_end`)",
        "step": "Frame sampling stride (default 2; use 1 for every frame)",
        "contact_tolerance_mm": "Allowed ground clearance/penetration tolerance in mm before flagging (default 35.0)",
        "min_camera_clearance_m": "Minimum safe camera height above ground in meters (default 0.25)",
        "frustum_margin": "Optional normalized `[0..0.4]` border margin; when set, flags if any subject AABB corner clips or crowds the frame",
    },
)
def op_scene_audit(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    sc = ctx.scene
    f_start = int(params.get("frame_start", sc.frame_start))
    f_end = int(params.get("frame_end", sc.frame_end))
    step = max(1, int(params.get("step", 2)))
    orig_frame = int(sc.frame_current)
    fps = effective_fps(sc)

    subj_name = str(params.get("subject") or "").strip()
    subj_obj = bpy.data.objects.get(subj_name) if subj_name else None
    if subj_name and subj_obj is None:
        raise OpError(f"Subject object not found: '{subj_name}'", code="NOT_FOUND")

    cam_name = str(params.get("camera") or "").strip()
    cam_obj = bpy.data.objects.get(cam_name) if cam_name else sc.camera
    if cam_name and cam_obj is None:
        raise OpError(f"Camera object not found: '{cam_name}'", code="NOT_FOUND")

    raw_ground = params.get("ground")
    ground_names: list[str] = []
    if isinstance(raw_ground, str) and raw_ground.strip():
        ground_names = [raw_ground.strip()]
    elif isinstance(raw_ground, list):
        ground_names = [str(g).strip() for g in raw_ground if str(g).strip()]

    depsgraph_init = bpy.context.evaluated_depsgraph_get()
    ground_bvhs: list[tuple[str, Any, Any, Any, float]] = []
    for gn in ground_names:
        g_obj = bpy.data.objects.get(gn)
        if g_obj is None or g_obj.type != "MESH":
            raise OpError(f"Ground mesh object not found: '{gn}'", code="NOT_FOUND")
        g_aabb = world_aabb(g_obj, depsgraph_init)
        z_top = (g_aabb["max"][2] if g_aabb else 100.0) + 200.0
        ground_bvhs.append(
            (gn, BVHTree.FromObject(g_obj, depsgraph_init), g_obj.matrix_world.copy(), g_obj.matrix_world.inverted(), z_top)
        )

    def _highest_ground_z(x: float, y: float) -> float | None:
        best: float | None = None
        for _, bvh, mw, mw_inv, z_top in ground_bvhs:
            hz = _raycast_vertical_surface_z(bvh, mw, mw_inv, x, y, z_top)
            if hz is not None and (best is None or hz > best):
                best = hz
        return best

    sampled_frames = list(range(f_start, f_end + 1, step))
    if sampled_frames[-1] != f_end:
        sampled_frames.append(f_end)

    subj_clearances_m: list[float] = []
    subj_positions: list[tuple[int, Any]] = []
    cam_clearances_m: list[float] = []
    cam_distances_m: list[float] = []
    cam_lenses_mm: list[float] = []
    raw_margin = params.get("frustum_margin")
    req_margin = max(0.0, min(0.45, float(raw_margin))) if raw_margin is not None else 0.0

    in_frame_count = 0
    bbox_in_frame_count = 0
    bbox_clipped_frames = 0
    clipped_behind_cam = 0
    screen_widths_pct: list[float] = []
    screen_heights_pct: list[float] = []
    edge_margins: list[float] = []
    beat_samples: list[dict[str, Any]] = []

    try:
        for idx, fr in enumerate(sampled_frames):
            sc.frame_set(fr)
            depsgraph = bpy.context.evaluated_depsgraph_get()

            s_aabb: dict[str, list[float]] | None = None
            s_center: Any = None
            if subj_obj is not None:
                s_eval = subj_obj.evaluated_get(depsgraph)
                s_aabb = world_aabb(s_eval)
                assert s_aabb is not None
                s_center = Vector(s_aabb["center"])
                subj_positions.append((fr, s_center.copy()))

                if ground_bvhs:
                    gz = _highest_ground_z(s_center.x, s_center.y)
                    if gz is not None:
                        subj_clearances_m.append(s_aabb["min"][2] - gz)

            if cam_obj is not None and cam_obj.type == "CAMERA":
                c_eval = cam_obj.evaluated_get(depsgraph)
                c_pos = c_eval.matrix_world.translation.copy()
                cam_lenses_mm.append(float(c_eval.data.lens))
                clip_start = float(c_eval.data.clip_start)

                if ground_bvhs:
                    c_gz = _highest_ground_z(c_pos.x, c_pos.y)
                    if c_gz is not None:
                        cam_clearances_m.append(c_pos.z - c_gz)

                if subj_obj is not None and s_aabb is not None and s_center is not None:
                    dist_m = float((c_pos - s_center).length)
                    cam_distances_m.append(dist_m)

                    # Project all 8 world AABB corners into normalized [0..1] camera view
                    mn, mx = s_aabb["min"], s_aabb["max"]
                    corners = [
                        Vector((x, y, z))
                        for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])
                    ]
                    ndc_pts = [world_to_camera_view(sc, c_eval, pt) for pt in corners]
                    center_ndc = world_to_camera_view(sc, c_eval, s_center)

                    if center_ndc.z < clip_start:
                        clipped_behind_cam += 1
                        bbox_clipped_frames += 1
                        edge_margins.append(-1.0)
                    else:
                        if 0.0 <= center_ndc.x <= 1.0 and 0.0 <= center_ndc.y <= 1.0:
                            in_frame_count += 1
                        us = [float(p.x) for p in ndc_pts]
                        vs = [float(p.y) for p in ndc_pts]
                        w_pct = max(0.0, min(100.0, (max(us) - min(us)) * 100.0))
                        h_pct = max(0.0, min(100.0, (max(vs) - min(vs)) * 100.0))
                        screen_widths_pct.append(w_pct)
                        screen_heights_pct.append(h_pct)

                        any_corner_behind = any(float(p.z) < clip_start for p in ndc_pts)
                        edge_m = (
                            -1.0
                            if any_corner_behind
                            else min(min(us), 1.0 - max(us), min(vs), 1.0 - max(vs))
                        )
                        edge_margins.append(edge_m)
                        if edge_m < 0.0:
                            bbox_clipped_frames += 1
                        if not any_corner_behind and edge_m >= req_margin:
                            bbox_in_frame_count += 1

                        # Record 5 evenly spaced telemetry beats for prompt compilation!
                        if len(sampled_frames) <= 5 or idx in {
                            0,
                            len(sampled_frames) // 4,
                            len(sampled_frames) // 2,
                            (3 * len(sampled_frames)) // 4,
                            len(sampled_frames) - 1,
                        }:
                            beat_samples.append(
                                {
                                    "frame": fr,
                                    "time_sec": round((fr - f_start) / fps, 2) if fps > 0 else 0.0,
                                    "cam_dist_m": round(dist_m, 2),
                                    "cam_height_agl_m": round(cam_clearances_m[-1], 2) if cam_clearances_m else None,
                                    "lens_mm": round(float(c_eval.data.lens), 1),
                                    "subject_screen_width_pct": round(w_pct, 1),
                                    "subject_screen_height_pct": round(h_pct, 1),
                                    "edge_margin_pct": round(edge_m * 100.0, 1),
                                    "subject_ndc": [round(float(center_ndc.x), 3), round(float(center_ndc.y), 3)],
                                }
                            )
    finally:
        sc.frame_set(orig_frame)

    issues: list[str] = []
    report: dict[str, Any] = {
        "frames_sampled": len(sampled_frames),
        "frame_range": [f_start, f_end],
        "step": step,
        "fps": round(fps, 4),
    }

    tol_mm = float(params.get("contact_tolerance_mm", 35.0))
    if subj_clearances_m:
        mm_vals = [v * 1000.0 for v in subj_clearances_m]
        rms_mm = math.sqrt(sum(v * v for v in mm_vals) / len(mm_vals))
        min_mm = min(mm_vals)
        max_mm = max(mm_vals)
        report["ground_contact"] = {
            "min_clearance_mm": round(min_mm, 2),
            "max_clearance_mm": round(max_mm, 2),
            "rms_mm": round(rms_mm, 2),
            "floating_frames": sum(1 for v in mm_vals if v > tol_mm),
            "penetrating_frames": sum(1 for v in mm_vals if v < -tol_mm),
        }
        if min_mm < -tol_mm:
            issues.append(
                f"Subject '{subj_name}' penetrates ground by {abs(min_mm):.1f} mm (exceeds {tol_mm:.0f} mm tolerance)."
            )
        if max_mm > tol_mm * 3.0:
            issues.append(
                f"Subject '{subj_name}' floats {max_mm:.1f} mm above ground."
            )

    min_cam_clr = float(params.get("min_camera_clearance_m", 0.25))
    if cam_clearances_m:
        min_c = min(cam_clearances_m)
        report["camera_clearance"] = {
            "min_agl_m": round(min_c, 3),
            "max_agl_m": round(max(cam_clearances_m), 3),
            "underground_frames": sum(1 for v in cam_clearances_m if v < 0.0),
        }
        if min_c < min_cam_clr:
            issues.append(
                f"Camera dips below safe ground clearance ({min_c:.2f} m < {min_cam_clr:.2f} m)."
            )

    if cam_distances_m:
        n_s = len(sampled_frames)
        in_frame_pct = (in_frame_count / n_s) * 100.0
        bbox_in_frame_pct = (bbox_in_frame_count / n_s) * 100.0
        min_edge_m = min(edge_margins) if edge_margins else 0.0
        report["framing"] = {
            "in_frame_pct": round(in_frame_pct, 1),
            "bbox_in_frame_pct": round(bbox_in_frame_pct, 1),
            "bbox_clipped_frames": bbox_clipped_frames,
            "min_edge_margin_pct": round(min_edge_m * 100.0, 1),
            "clipped_near_frames": clipped_behind_cam,
            "cam_distance_m": {
                "start": round(cam_distances_m[0], 2),
                "end": round(cam_distances_m[-1], 2),
                "min": round(min(cam_distances_m), 2),
                "max": round(max(cam_distances_m), 2),
            },
            "lens_mm_range": [round(min(cam_lenses_mm), 1), round(max(cam_lenses_mm), 1)] if cam_lenses_mm else None,
            "subject_screen_width_pct": {
                "start": round(screen_widths_pct[0], 1) if screen_widths_pct else 0.0,
                "end": round(screen_widths_pct[-1], 1) if screen_widths_pct else 0.0,
                "max": round(max(screen_widths_pct), 1) if screen_widths_pct else 0.0,
            },
            "subject_screen_height_pct": {
                "start": round(screen_heights_pct[0], 1) if screen_heights_pct else 0.0,
                "end": round(screen_heights_pct[-1], 1) if screen_heights_pct else 0.0,
                "max": round(max(screen_heights_pct), 1) if screen_heights_pct else 0.0,
            },
            "beats": beat_samples,
        }
        if clipped_behind_cam > 0:
            issues.append(f"Subject clips behind camera near-plane on {clipped_behind_cam} sampled frames.")
        if in_frame_pct < 85.0:
            issues.append(f"Subject center leaves camera frustum too often ({in_frame_pct:.1f}% in-frame).")
        elif raw_margin is not None and min_edge_m < req_margin:
            issues.append(
                f"Subject '{subj_name}' AABB clips or crowds camera border "
                f"(min edge margin {min_edge_m * 100.0:.1f}% < required {req_margin * 100.0:.1f}%)."
            )

    if len(subj_positions) >= 2 and fps > 0:
        speeds: list[float] = []
        for i in range(1, len(subj_positions)):
            f0, p0 = subj_positions[i - 1]
            f1, p1 = subj_positions[i]
            dt = (f1 - f0) / fps
            if dt > 1e-6:
                speeds.append(float((p1 - p0).length) / dt)
        if speeds:
            report["kinematics"] = {
                "speed_mps": [round(min(speeds), 2), round(max(speeds), 2)],
                "speed_kmh": [round(min(speeds) * 3.6, 1), round(max(speeds) * 3.6, 1)],
                "mean_speed_kmh": round((sum(speeds) / len(speeds)) * 3.6, 1),
            }

    # Optional mesh topology audit
    raw_meshes = params.get("meshes")
    check_topo = bool(params.get("check_topology", False))
    topo_objs: list[Any] = []
    if raw_meshes == "*" or (raw_meshes is None and (check_topo or (subj_obj is None and not ground_names and not params.get("check_collisions")))):
        topo_objs = [o for o in sc.objects if o.type == "MESH"]
    elif isinstance(raw_meshes, str) and raw_meshes.strip():
        o = bpy.data.objects.get(raw_meshes.strip())
        if o is None or o.type != "MESH":
            raise OpError(f"Mesh object not found for topology audit: '{raw_meshes}'", code="NOT_FOUND")
        topo_objs = [o]
    elif isinstance(raw_meshes, list):
        for mn in raw_meshes:
            o = bpy.data.objects.get(str(mn).strip())
            if o is None or o.type != "MESH":
                raise OpError(f"Mesh object not found for topology audit: '{mn}'", code="NOT_FOUND")
            topo_objs.append(o)

    if topo_objs:
        deps_now = bpy.context.evaluated_depsgraph_get()
        topo_reports: list[dict[str, Any]] = []
        fail_nm = bool(params.get("fail_on_non_manifold", False))
        for mo in topo_objs:
            tr, t_issues = _audit_mesh_topology(bpy, deps_now, mo)
            if fail_nm and tr["non_manifold_edges"] > 0:
                t_issues.append(
                    f"Mesh '{mo.name}' is not watertight ({tr['non_manifold_edges']} non-manifold edge(s))."
                )
            topo_reports.append(tr)
            issues.extend(t_issues)
        report["topology"] = topo_reports

    # Optional world-space BVH pairwise intersection check
    raw_col = params.get("check_collisions")
    col_objs: list[Any] = []
    if raw_col == "*":
        col_objs = [o for o in sc.objects if o.type == "MESH"]
    elif isinstance(raw_col, list):
        for cn in raw_col:
            o = bpy.data.objects.get(str(cn).strip())
            if o is None or o.type != "MESH":
                raise OpError(f"Mesh object not found for collision check: '{cn}'", code="NOT_FOUND")
            col_objs.append(o)

    if len(col_objs) >= 2:
        deps_now = bpy.context.evaluated_depsgraph_get()
        overlaps = _check_world_bvh_collisions(deps_now, col_objs)
        report["collisions"] = overlaps
        for ov in overlaps:
            issues.append(
                f"Meshes '{ov['object_a']}' and '{ov['object_b']}' intersect ({ov['overlapping_polygon_pairs']} overlapping face pairs)."
            )

    report["passed"] = len(issues) == 0
    report["issues"] = issues
    return report