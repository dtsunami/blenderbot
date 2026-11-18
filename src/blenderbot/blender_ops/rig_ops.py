"""Parametric 3D Previs Rigs: arc-length-solved splines, path followers, pursuit cameras & BVH scatter."""

from __future__ import annotations

import math
import random
from typing import Any

from .helpers import (
    assign_solid_material,
    ensure_channelbag_fcurves,
    ensure_collection,
    link_exclusively,
)
from .schema import OpContext, OpError, blender_op


def _sample_polyline_arc_table(pts: list[Any]) -> tuple[list[float], float]:
    """Return cumulative arc lengths `[0.0, ...]` and total length for a list of `Vector` points."""
    cum = [0.0]
    total = 0.0
    for i in range(1, len(pts)):
        total += float((pts[i] - pts[i - 1]).length)
        cum.append(total)
    return cum, total


def evaluate_curve_world_points(curve_obj: Any, depsgraph: Any) -> tuple[list[Any], list[float], float]:
    """Evaluate `curve_obj` to dense world-space vertices and build its cumulative arc-length table."""
    from mathutils import Vector

    eval_obj = curve_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        mw = eval_obj.matrix_world
        pts = [mw @ Vector(v.co) for v in mesh.vertices]
        if len(pts) < 2:
            raise OpError(f"Curve '{curve_obj.name}' has fewer than 2 evaluated points", code="INVALID_CURVE")
        # If cyclic, close loop in arc table
        is_cyclic = bool(curve_obj.data.splines and (curve_obj.data.splines[0].use_cyclic_u))
        if is_cyclic:
            pts.append(pts[0].copy())
        cum, total = _sample_polyline_arc_table(pts)
        return pts, cum, total
    finally:
        eval_obj.to_mesh_clear()


@blender_op(
    "rig.curve_path",
    tier="mutate",
    snapshot_blend=True,
    summary="Create a smooth 3D Bezier/Catmull-Rom spline Curve with dense resolution and return its exact arc length.",
    params_doc={
        "name": "Curve object name (e.g. 'TRACK_Centerline' or 'CAM_Path')",
        "points": "List of `[x, y, z]` control points in meters",
        "cyclic": "Close loop smoothly (default false)",
        "collection": "Collection name (default 'RIGS')",
        "resolution_u": "Subdivisions per segment for smooth motion (default 32)",
        "handle_type": "'AUTO' (smooth Catmull-like) or 'VECTOR' (default 'AUTO')",
    },
)
def op_rig_curve_path(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    name = str(params.get("name") or "").strip()
    raw_pts = params.get("points")
    if not name or not isinstance(raw_pts, list) or len(raw_pts) < 2:
        raise OpError("rig.curve_path requires 'name' and at least 2 `[x, y, z]` 'points'", code="INVALID_PARAM")

    cyclic = bool(params.get("cyclic", False))
    res_u = max(4, int(params.get("resolution_u", 32)))
    handle_type = str(params.get("handle_type", "AUTO")).upper()
    col_name = str(params.get("collection") or "RIGS")

    col = ensure_collection(col_name)

    existing = bpy.data.objects.get(name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = res_u
    curve_data.use_path = True
    curve_data.path_duration = max(1, ctx.scene.frame_end - ctx.scene.frame_start + 1)

    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(len(raw_pts) - 1)
    spline.use_cyclic_u = cyclic

    for i, pt in enumerate(raw_pts):
        bp = spline.bezier_points[i]
        bp.co = (float(pt[0]), float(pt[1]), float(pt[2]))
        bp.handle_left_type = handle_type
        bp.handle_right_type = handle_type

    curve_obj = bpy.data.objects.new(name, curve_data)
    link_exclusively(curve_obj, col)
    ctx.view_layer.update()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    _, _, total_len = evaluate_curve_world_points(curve_obj, depsgraph)

    return {
        "name": curve_obj.name,
        "collection": col.name,
        "control_points": len(raw_pts),
        "cyclic": cyclic,
        "arc_length_m": round(total_len, 3),
    }


def _interpolate_along_polyline(
    pts: list[Any],
    cum: list[float],
    total_len: float,
    dist: float,
    *,
    cyclic: bool = False,
) -> tuple[Any, Any, float]:
    """Sample world position, unit forward tangent, and horizontal signed curvature (1/m) at `dist`."""
    import bisect

    if total_len <= 1e-6:
        return pts[0].copy(), (pts[-1] - pts[0]).normalized(), 0.0

    s = (dist % total_len) if cyclic else max(0.0, min(total_len, dist))
    idx = bisect.bisect_right(cum, s) - 1
    idx = max(0, min(len(pts) - 2, idx))
    seg_len = cum[idx + 1] - cum[idx]
    alpha = (s - cum[idx]) / seg_len if seg_len > 1e-8 else 0.0
    pos = pts[idx].lerp(pts[idx + 1], alpha)

    # Smooth finite-difference tangent & curvature over a 2.5m window
    ds = min(2.5, total_len * 0.02)
    s_prev = ((s - ds) % total_len) if cyclic else max(0.0, s - ds)
    s_next = ((s + ds) % total_len) if cyclic else min(total_len, s + ds)

    def _pos_at(d_val: float) -> Any:
        j = max(0, min(len(pts) - 2, bisect.bisect_right(cum, d_val) - 1))
        sl = cum[j + 1] - cum[j]
        a = (d_val - cum[j]) / sl if sl > 1e-8 else 0.0
        return pts[j].lerp(pts[j + 1], a)

    p_prev = _pos_at(s_prev)
    p_next = _pos_at(s_next)
    tangent = p_next - p_prev
    if tangent.length < 1e-8:
        tangent = pts[idx + 1] - pts[idx]
    tangent.normalize()

    # Signed horizontal curvature for banking/steering
    t1 = pos - p_prev
    t2 = p_next - pos
    curvature = 0.0
    if t1.length > 1e-5 and t2.length > 1e-5:
        t1_n = t1.normalized()
        t2_n = t2.normalized()
        cross_z = t1_n.x * t2_n.y - t1_n.y * t2_n.x
        curvature = cross_z / max(0.25, ds)

    return pos, tangent, curvature


@blender_op(
    "rig.follow_path",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Bake physically smooth arc-length-parameterized motion for `object` along `curve`, "
        "with optional curvature speed modulation, roll banking, and exact wheel rotation (`s / radius`)."
    ),
    params_doc={
        "object": "Object name to animate along the curve",
        "curve": "Target Curve object name",
        "start_dist_m": "Starting distance along curve in meters (default 0.0)",
        "end_dist_m": "Ending distance along curve in meters (defaults to full curve arc length)",
        "frame_start": "Start frame (defaults to scene.frame_start)",
        "frame_end": "End frame (defaults to scene.frame_end)",
        "forward_axis": "'+Y' (default) or '-Y' or '+X'",
        "z_offset_m": "Vertical height offset above curve in meters (default 0.0)",
        "corner_slowdown": "0.0..0.6 curvature-based corner braking factor (default 0.2)",
        "bank_gain_deg": "Roll banking gain on corners in degrees (default 0.0)",
        "wheels": "Optional list of wheel object names to spin seamlessly (`angle = dist / wheel_radius`)",
        "wheel_radius_m": "Wheel radius in meters for exact zero-slip rolling (default 0.34)",
        "wheel_spin_axis": "Local axis index 0=X, 1=Y, 2=Z for wheel spin (default 0 = local X)",
    },
)
def op_rig_follow_path(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    from mathutils import Matrix, Vector

    obj_name = str(params.get("object") or "").strip()
    curve_name = str(params.get("curve") or "").strip()
    obj = bpy.data.objects.get(obj_name)
    curve_obj = bpy.data.objects.get(curve_name)
    if obj is None:
        raise OpError(f"Object not found: '{obj_name}'", code="NOT_FOUND")
    if curve_obj is None or curve_obj.type != "CURVE":
        raise OpError(f"Curve object not found: '{curve_name}'", code="NOT_FOUND")

    sc = ctx.scene
    f_start = int(params.get("frame_start", sc.frame_start))
    f_end = int(params.get("frame_end", sc.frame_end))
    if f_end <= f_start:
        raise OpError("frame_end must be > frame_start", code="INVALID_RANGE")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    pts, cum, total_len = evaluate_curve_world_points(curve_obj, depsgraph)
    cyclic = bool(curve_obj.data.splines and curve_obj.data.splines[0].use_cyclic_u)

    s0 = float(params.get("start_dist_m", 0.0))
    s1 = float(params.get("end_dist_m", total_len))
    total_travel = s1 - s0
    z_off = float(params.get("z_offset_m", 0.0))
    slowdown = max(0.0, min(0.75, float(params.get("corner_slowdown", 0.2))))
    bank_gain = float(params.get("bank_gain_deg", 0.0))
    fwd_axis = str(params.get("forward_axis", "+Y")).upper()

    n_frames = f_end - f_start + 1
    fps = float(sc.render.fps) / float(sc.render.fps_base or 1.0)

    # Pre-solve curvature-modulated distance schedule normalized strictly to [s0 .. s1]
    weights: list[float] = []
    for i in range(n_frames):
        u = i / max(1, n_frames - 1)
        approx_s = s0 + u * total_travel
        _, _, curv = _interpolate_along_polyline(pts, cum, total_len, approx_s, cyclic=cyclic)
        w = 1.0 / (1.0 + slowdown * min(3.0, abs(curv) * 25.0))
        weights.append(w)

    cum_w = [0.0]
    for i in range(1, n_frames):
        cum_w.append(cum_w[-1] + 0.5 * (weights[i - 1] + weights[i]))
    w_total = cum_w[-1] if cum_w[-1] > 1e-8 else 1.0
    dists = [s0 + (cw / w_total) * total_travel for cw in cum_w]

    # Clear existing location/rotation fcurves on obj and bake clean Blender 5.x channelbag curves
    fcurves = ensure_channelbag_fcurves(obj)
    for fc in list(fcurves):
        if fc.data_path in ("location", "rotation_euler"):
            fcurves.remove(fc)

    obj.rotation_mode = "XYZ"
    loc_fcs = [fcurves.new(data_path="location", index=ax) for ax in range(3)]
    rot_fcs = [fcurves.new(data_path="rotation_euler", index=ax) for ax in range(3)]
    for fc in loc_fcs + rot_fcs:
        fc.keyframe_points.add(n_frames)

    wheel_names = params.get("wheels") or []
    wheel_radius = max(0.02, float(params.get("wheel_radius_m", 0.34)))
    wheel_axis = int(params.get("wheel_spin_axis", 0))
    wheel_fcs: list[Any] = []
    for wn in wheel_names:
        w_obj = bpy.data.objects.get(str(wn))
        if w_obj is not None:
            w_obj.rotation_mode = "XYZ"
            wfcs = ensure_channelbag_fcurves(w_obj)
            for existing_fc in list(wfcs):
                if existing_fc.data_path == "rotation_euler" and existing_fc.array_index == wheel_axis:
                    wfcs.remove(existing_fc)
            wfc = wfcs.new(data_path="rotation_euler", index=wheel_axis)
            wfc.keyframe_points.add(n_frames)
            wheel_fcs.append(wfc)

    world_up = Vector((0.0, 0.0, 1.0))
    prev_euler = None
    speeds_mps: list[float] = []

    for i in range(n_frames):
        fr = f_start + i
        d_now = dists[i]
        if i > 0:
            speeds_mps.append((dists[i] - dists[i - 1]) * fps)

        pos, tangent, curv = _interpolate_along_polyline(pts, cum, total_len, d_now, cyclic=cyclic)
        pos = pos + Vector((0.0, 0.0, z_off))

        fwd = tangent if fwd_axis != "-Y" else -tangent
        right = fwd.cross(world_up)
        if right.length < 1e-6:
            right = Vector((1.0, 0.0, 0.0))
        else:
            right.normalize()
        up = right.cross(fwd).normalized()

        if fwd_axis == "+X":
            rot_mat = Matrix((fwd, -right, up)).transposed()
        else:
            rot_mat = Matrix((right, fwd, up)).transposed()

        eul = rot_mat.to_euler("XYZ", prev_euler) if prev_euler is not None else rot_mat.to_euler("XYZ")
        if abs(bank_gain) > 1e-5:
            bank_rad = math.radians(max(-25.0, min(25.0, curv * bank_gain * 10.0)))
            eul.y += bank_rad
        prev_euler = eul

        for ax in range(3):
            kp_l = loc_fcs[ax].keyframe_points[i]
            kp_l.co = (float(fr), float(pos[ax]))
            kp_l.interpolation = "LINEAR"

            kp_r = rot_fcs[ax].keyframe_points[i]
            kp_r.co = (float(fr), float(eul[ax]))
            kp_r.interpolation = "LINEAR"

        spin_angle = (d_now - s0) / wheel_radius
        for wfc in wheel_fcs:
            kp_w = wfc.keyframe_points[i]
            kp_w.co = (float(fr), float(spin_angle))
            kp_w.interpolation = "LINEAR"

    for fc in loc_fcs + rot_fcs + wheel_fcs:
        fc.update()

    ctx.view_layer.update()
    min_v = min(speeds_mps) if speeds_mps else 0.0
    max_v = max(speeds_mps) if speeds_mps else 0.0

    return {
        "object": obj.name,
        "curve": curve_obj.name,
        "frames_baked": n_frames,
        "distance_m": round(total_travel, 3),
        "speed_mps_range": [round(min_v, 2), round(max_v, 2)],
        "speed_kmh_range": [round(min_v * 3.6, 1), round(max_v * 3.6, 1)],
        "wheels_animated": len(wheel_fcs),
    }


@blender_op(
    "rig.camera_pursuit",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Build or update a smooth Pursuit/Tracking Camera with damped look-at target (`Track To`) "
        "and keyframed relative offsets `[right_m, back_m, up_m, lens_mm]` across shot beats."
    ),
    params_doc={
        "name": "Camera object name (default 'CAM_Main')",
        "target": "Target object name to track (e.g. 'CAR_Body')",
        "beats": (
            "List of beat dicts: `[{'frame': 1, 'offset': [right_m, back_m, up_m], "
            "'look_ahead': [dx, dy, dz], 'lens_mm': 28}, ...]` interpolated smoothly with Catmull-Rom/Bezier"
        ),
        "collection": "Collection name (default 'CAMERA')",
        "set_active": "Make this the active scene render camera (default true)",
    },
)
def op_rig_camera_pursuit(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    from mathutils import Vector

    cam_name = str(params.get("name") or "CAM_Main").strip()
    target_name = str(params.get("target") or "").strip()
    beats = params.get("beats")
    if not target_name or not isinstance(beats, list) or len(beats) < 1:
        raise OpError("rig.camera_pursuit requires 'target' and non-empty 'beats' list", code="INVALID_PARAM")

    target_obj = bpy.data.objects.get(target_name)
    if target_obj is None:
        raise OpError(f"Target object not found: '{target_name}'", code="NOT_FOUND")

    col = ensure_collection(str(params.get("collection") or "CAMERA"))

    # Create or reuse camera + look target Empty
    cam_obj = bpy.data.objects.get(cam_name)
    if cam_obj is None or cam_obj.type != "CAMERA":
        if cam_obj is not None:
            bpy.data.objects.remove(cam_obj, do_unlink=True)
        cam_data = bpy.data.cameras.new(name=f"{cam_name}_Data")
        cam_obj = bpy.data.objects.new(cam_name, cam_data)
    else:
        cam_data = cam_obj.data
    cam_data.clip_start = 0.1
    cam_data.clip_end = 5000.0
    link_exclusively(cam_obj, col)

    aim_name = f"{cam_name}_Aim"
    aim_obj = bpy.data.objects.get(aim_name)
    if aim_obj is None:
        aim_obj = bpy.data.objects.new(aim_name, None)
        aim_obj.empty_display_size = 0.5
        aim_obj.empty_display_type = "PLAIN_AXES"
    link_exclusively(aim_obj, col)

    # Ensure clean Track To constraint on camera -> Aim
    cam_obj.constraints.clear()
    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = aim_obj
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    sorted_beats = sorted(beats, key=lambda b: int(b["frame"]))

    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _eval_beat_channels(fr: int) -> tuple[Vector, Vector, float]:
        if fr <= int(sorted_beats[0]["frame"]):
            b0 = sorted_beats[0]
            return (
                Vector([float(x) for x in b0.get("offset", [0, -12, 4])]),
                Vector([float(x) for x in b0.get("look_ahead", [0, 2, 0.6])]),
                float(b0.get("lens_mm", 35.0)),
            )
        if fr >= int(sorted_beats[-1]["frame"]):
            b1 = sorted_beats[-1]
            return (
                Vector([float(x) for x in b1.get("offset", [0, -12, 4])]),
                Vector([float(x) for x in b1.get("look_ahead", [0, 2, 0.6])]),
                float(b1.get("lens_mm", 35.0)),
            )
        for idx in range(len(sorted_beats) - 1):
            fa = int(sorted_beats[idx]["frame"])
            fb = int(sorted_beats[idx + 1]["frame"])
            if fa <= fr <= fb:
                alpha = _smoothstep((fr - fa) / max(1, fb - fa))
                oa = Vector([float(x) for x in sorted_beats[idx].get("offset", [0, -12, 4])])
                ob = Vector([float(x) for x in sorted_beats[idx + 1].get("offset", [0, -12, 4])])
                la = Vector([float(x) for x in sorted_beats[idx].get("look_ahead", [0, 2, 0.6])])
                lb = Vector([float(x) for x in sorted_beats[idx + 1].get("look_ahead", [0, 2, 0.6])])
                lensa = float(sorted_beats[idx].get("lens_mm", 35.0))
                lensb = float(sorted_beats[idx + 1].get("lens_mm", 35.0))
                return oa.lerp(ob, alpha), la.lerp(lb, alpha), lensa + (lensb - lensa) * alpha
        b_last = sorted_beats[-1]
        return (
            Vector([float(x) for x in b_last.get("offset", [0, -12, 4])]),
            Vector([float(x) for x in b_last.get("look_ahead", [0, 2, 0.6])]),
            float(b_last.get("lens_mm", 35.0)),
        )

    sc = ctx.scene
    f_start, f_end = int(sc.frame_start), int(sc.frame_end)
    n_frames = f_end - f_start + 1
    orig_frame = int(sc.frame_current)

    cam_fcs = ensure_channelbag_fcurves(cam_obj)
    aim_fcs = ensure_channelbag_fcurves(aim_obj)
    lens_fcs = ensure_channelbag_fcurves(cam_data, action_name=f"{cam_name}_LensAction")

    for fcs, dp in ((cam_fcs, "location"), (aim_fcs, "location"), (lens_fcs, "lens")):
        for fc in list(fcs):
            if fc.data_path == dp:
                fcs.remove(fc)

    cam_loc_fcs = [cam_fcs.new(data_path="location", index=ax) for ax in range(3)]
    aim_loc_fcs = [aim_fcs.new(data_path="location", index=ax) for ax in range(3)]
    lens_fc = lens_fcs.new(data_path="lens", index=0)
    for fc in [*cam_loc_fcs, *aim_loc_fcs, lens_fc]:
        fc.keyframe_points.add(n_frames)

    world_up = Vector((0.0, 0.0, 1.0))

    try:
        for i in range(n_frames):
            fr = f_start + i
            sc.frame_set(fr)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            t_eval = target_obj.evaluated_get(depsgraph)
            mw = t_eval.matrix_world
            t_pos = mw.translation.copy()

            # Horizontal heading frame of the target (decoupled from pitch/roll bounce for steady camera!)
            fwd = (mw.to_3x3() @ Vector((0.0, 1.0, 0.0)))
            fwd.z = 0.0
            if fwd.length < 1e-5:
                fwd = Vector((0.0, 1.0, 0.0))
            else:
                fwd.normalize()
            right = fwd.cross(world_up).normalized()

            rel_off, look_off, lens_mm = _eval_beat_channels(fr)
            cam_world = t_pos + right * rel_off.x + fwd * rel_off.y + world_up * rel_off.z
            aim_world = t_pos + right * look_off.x + fwd * look_off.y + world_up * look_off.z

            for ax in range(3):
                k_c = cam_loc_fcs[ax].keyframe_points[i]
                k_c.co = (float(fr), float(cam_world[ax]))
                k_c.interpolation = "LINEAR"

                k_a = aim_loc_fcs[ax].keyframe_points[i]
                k_a.co = (float(fr), float(aim_world[ax]))
                k_a.interpolation = "LINEAR"

            k_l = lens_fc.keyframe_points[i]
            k_l.co = (float(fr), float(lens_mm))
            k_l.interpolation = "LINEAR"
    finally:
        for fc in [*cam_loc_fcs, *aim_loc_fcs, lens_fc]:
            fc.update()
        sc.frame_set(orig_frame)

    if params.get("set_active", True):
        sc.camera = cam_obj

    return {
        "camera": cam_obj.name,
        "aim_target": aim_obj.name,
        "tracked_object": target_obj.name,
        "beats_count": len(sorted_beats),
        "frames_baked": n_frames,
        "active": sc.camera == cam_obj,
    }


@blender_op(
    "terrain.scatter",
    tier="mutate",
    snapshot_blend=True,
    summary="Scatter linked instances onto a ground/terrain mesh via exact `BVHTree` vertical raycasting + exclusion corridor.",
    params_doc={
        "terrain": "Ground/terrain Mesh object name",
        "prototype": "Prototype object name to instance (or omit to auto-build low-poly stylized pine trees)",
        "count": "Requested scatter attempts (default 80)",
        "bounds": "Optional `[xmin, xmax, ymin, ymax]` scatter region (defaults to terrain AABB)",
        "exclude_curve": "Optional road/track Curve name to keep clear",
        "exclude_margin_m": "Minimum horizontal distance from `exclude_curve` in meters (default 8.0)",
        "scale_range": "`[min_scale, max_scale]` uniform random variation (default [0.8, 1.35])",
        "seed": "Deterministic RNG seed (default 42)",
        "collection": "Collection for scattered instances (default 'ENV_Scatter')",
    },
)
def op_terrain_scatter(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    import bmesh
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    terrain_name = str(params.get("surface") or params.get("terrain") or "").strip()
    terrain_obj = bpy.data.objects.get(terrain_name)
    if terrain_obj is None or terrain_obj.type != "MESH":
        raise OpError(f"Surface/terrain mesh object not found: '{terrain_name}'", code="NOT_FOUND")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(terrain_obj, depsgraph)
    mw = terrain_obj.matrix_world
    mw_inv = mw.inverted()

    # Optional road/curve exclusion KD-lookup
    excl_pts: list[Any] = []
    excl_margin = float(params.get("exclude_margin_m", 8.0))
    if params.get("exclude_curve"):
        c_obj = bpy.data.objects.get(str(params["exclude_curve"]))
        if c_obj is not None and c_obj.type == "CURVE":
            excl_pts, _, _ = evaluate_curve_world_points(c_obj, depsgraph)

    col = ensure_collection(str(params.get("collection") or "ENV_Scatter"))

    proto_name = str(params.get("prototype") or "").strip()
    proto_obj = bpy.data.objects.get(proto_name) if proto_name else None
    if proto_obj is None:
        # Build a clean low-poly previs pine tree prototype
        proto_mesh = bpy.data.meshes.new("PrevisPine_Mesh")
        bm = bmesh.new()
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=7,
            radius1=1.4,
            radius2=0.05,
            depth=5.5,
        )
        for v in bm.verts:
            v.co.z += 3.25
        bm.to_mesh(proto_mesh)
        bm.free()
        proto_obj = bpy.data.objects.new("_Proto_PrevisPine", proto_mesh)
        assign_solid_material(proto_obj, "Mat_PrevisPine", (0.12, 0.28, 0.14, 1.0), roughness=0.8)

    # Compute scatter XY bounds from terrain world AABB if not passed
    from .helpers import world_aabb

    t_aabb = world_aabb(terrain_obj, depsgraph)
    assert t_aabb is not None
    raw_bounds = params.get("bounds")
    if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 4:
        xmin, xmax, ymin, ymax = [float(v) for v in raw_bounds]
    else:
        xmin, ymin = t_aabb["min"][0], t_aabb["min"][1]
        xmax, ymax = t_aabb["max"][0], t_aabb["max"][1]
    z_top = t_aabb["max"][2] + 100.0

    rng = random.Random(int(params.get("seed", 42)))
    s_min, s_max = [float(x) for x in (params.get("scale_range") or [0.8, 1.35])[:2]]
    attempts = max(1, int(params.get("count", 80)))
    placed = 0
    excl_sq = excl_margin * excl_margin

    down_world = Vector((0.0, 0.0, -1.0))
    for i in range(attempts * 3):
        if placed >= attempts:
            break
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if excl_pts:
            too_close = any((x - p.x) ** 2 + (y - p.y) ** 2 < excl_sq for p in excl_pts[::4])
            if too_close:
                continue

        ray_orig_local = mw_inv @ Vector((x, y, z_top))
        ray_dir_local = (mw_inv.to_3x3() @ down_world).normalized()
        hit_loc, hit_norm, _, _ = bvh.ray_cast(ray_orig_local, ray_dir_local)
        if hit_loc is None:
            continue
        world_hit = mw @ hit_loc
        world_norm = (mw.to_3x3() @ hit_norm).normalized()
        # Skip near-vertical cliffs (>50 deg slope)
        if world_norm.z < math.cos(math.radians(50.0)):
            continue

        inst = bpy.data.objects.new(f"{col.name}_{placed:03d}", proto_obj.data)
        inst.location = world_hit
        sc_val = rng.uniform(s_min, s_max)
        inst.scale = (sc_val, sc_val, sc_val * rng.uniform(0.9, 1.15))
        inst.rotation_euler = (0.0, 0.0, rng.uniform(0.0, math.tau))
        inst.color = proto_obj.color
        col.objects.link(inst)
        placed += 1
        _ = i

    ctx.view_layer.update()
    return {
        "collection": col.name,
        "placed": placed,
        "requested": attempts,
        "surface": terrain_obj.name,
        "terrain": terrain_obj.name,
        "exclude_margin_m": excl_margin if excl_pts else 0.0,
    }


@blender_op(
    "scatter.surface",
    tier="mutate",
    snapshot_blend=True,
    summary="Scatter linked instances across any target Mesh `surface` using exact `BVHTree` raycasting, slope filtering & curve exclusion.",
    params_doc={
        "surface": "Target Mesh object name to scatter onto (alias: `terrain`)",
        "prototype": "Prototype object name to instance (auto-creates stylized pine if omitted)",
        "count": "Number of instances to place (default 80)",
        "bounds": "Optional `[xmin, xmax, ymin, ymax]` XY bounds",
        "exclude_curve": "Optional Curve object name to keep clear",
        "exclude_margin_m": "Exclusion radius around `exclude_curve` in meters (default 8.0)",
        "scale_range": "`[min_scale, max_scale]` (default `[0.8, 1.35]`)",
        "seed": "Deterministic RNG seed (default 42)",
        "collection": "Destination collection (default 'ENV_Scatter')",
    },
)
def op_scatter_surface(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    return op_terrain_scatter(ctx, params)


def _point_segment_dist_sq(pt: Any, a: Any, b: Any) -> float:
    ab = b - a
    ab_len_sq = float(ab.length_squared)
    if ab_len_sq < 1e-10:
        return float((pt - a).length_squared)
    t = max(0.0, min(1.0, float((pt - a).dot(ab)) / ab_len_sq))
    proj = a + ab * t
    return float((pt - proj).length_squared)


def _bind_mesh_to_armature_smooth(bpy: Any, mesh_obj: Any, arm_obj: Any) -> None:
    """Bind `mesh_obj` to `arm_obj` with smooth distance-falloff vertex weights (context-free, works on any topology)."""
    if mesh_obj.type != "MESH" or mesh_obj.data is None:
        raise OpError(f"Cannot skin non-mesh object '{mesh_obj.name}'", code="INVALID_TARGET")

    orig_mw = mesh_obj.matrix_world.copy()
    mesh_obj.parent = arm_obj
    mesh_obj.matrix_parent_inverse = arm_obj.matrix_world.inverted()
    mesh_obj.matrix_world = orig_mw

    # Ensure Armature modifier exists
    arm_mod = next((m for m in mesh_obj.modifiers if m.type == "ARMATURE" and m.object == arm_obj), None)
    if arm_mod is None:
        arm_mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
        arm_mod.object = arm_obj

    arm_mw = arm_obj.matrix_world
    bones_world: list[tuple[str, Any, Any, float]] = []
    for bone in arm_obj.data.bones:
        if not bone.use_deform:
            continue
        w_head = arm_mw @ bone.head_local
        w_tail = arm_mw @ bone.tail_local
        b_len = max(0.05, float((w_tail - w_head).length))
        bones_world.append((bone.name, w_head, w_tail, b_len))

    if not bones_world:
        return

    vgroups = {}
    for bname, _, _, _ in bones_world:
        vg = mesh_obj.vertex_groups.get(bname) or mesh_obj.vertex_groups.new(name=bname)
        vgroups[bname] = vg

    mesh_mw = mesh_obj.matrix_world
    for v in mesh_obj.data.vertices:
        v_world = mesh_mw @ v.co
        scored: list[tuple[str, float]] = []
        for bname, wh, wt, blen in bones_world:
            d = math.sqrt(_point_segment_dist_sq(v_world, wh, wt))
            sigma = max(0.08, blen * 0.55)
            w = 1.0 / ((d / sigma) ** 4 + 1e-4)
            scored.append((bname, w))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:4]
        w_sum = sum(w for _, w in top) or 1.0
        for bname, w in top:
            norm_w = w / w_sum
            if norm_w >= 0.01:
                vgroups[bname].add([v.index], norm_w, "REPLACE")


@blender_op(
    "rig.armature_create",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Create an Armature rig from hierarchical bone definitions (`head`, `tail`, `parent`, `connected`, `roll_deg`) "
        "and optionally bind/skin target meshes."
    ),
    params_doc={
        "name": "Armature object name (default 'Armature')",
        "bones": (
            "List of bone dicts: `[{'name': 'Root', 'head': [0,0,0], 'tail': [0,0,1]}, "
            "{'name': 'UpperArm', 'head': [0,0,1], 'tail': [0.6,0,1], 'parent': 'Root', 'connected': true}]`"
        ),
        "location": "Optional `[x, y, z]` world location of the armature object (default `[0, 0, 0]`)",
        "collection": "Collection name (default 'RIGS')",
        "show_in_front": "Display bones X-ray in front of geometry in viewport (default true)",
        "bind_meshes": "Optional list of Mesh object names to parent & smooth-weight-bind to this armature",
    },
)
def op_rig_armature_create(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    name = str(params.get("name") or "Armature").strip()
    bones_spec = params.get("bones")
    if not isinstance(bones_spec, list) or not bones_spec:
        raise OpError("rig.armature_create requires non-empty 'bones' list", code="MISSING_PARAM")

    existing = bpy.data.objects.get(name)
    if existing is not None:
        old_d = existing.data if existing.type == "ARMATURE" else None
        bpy.data.objects.remove(existing, do_unlink=True)
        if old_d is not None and old_d.users == 0:
            bpy.data.armatures.remove(old_d)

    arm_data = bpy.data.armatures.new(name=f"{name}_Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    arm_obj.show_in_front = bool(params.get("show_in_front", True))

    col = ensure_collection(str(params.get("collection") or "RIGS"))
    link_exclusively(arm_obj, col)

    if params.get("location") is not None:
        loc = params["location"]
        arm_obj.location = (float(loc[0]), float(loc[1]), float(loc[2]))

    ctx.view_layer.update()
    prev_active = ctx.view_layer.objects.active
    if prev_active is not None and getattr(prev_active, "mode", "OBJECT") != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    for o in ctx.view_layer.objects:
        o.select_set(False)
    arm_obj.select_set(True)
    ctx.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    try:
        ebones = arm_data.edit_bones
        created_ebones: dict[str, Any] = {}
        # Pass 1: create bones + head/tail/roll
        for bspec in bones_spec:
            bname = str(bspec.get("name") or "").strip()
            head = bspec.get("head")
            tail = bspec.get("tail")
            if not bname or not isinstance(head, (list, tuple)) or not isinstance(tail, (list, tuple)):
                raise OpError(
                    "Each bone in rig.armature_create requires 'name', 'head' `[x,y,z]`, and 'tail' `[x,y,z]`",
                    code="INVALID_PARAM",
                )
            eb = ebones.new(bname)
            eb.head = (float(head[0]), float(head[1]), float(head[2]))
            eb.tail = (float(tail[0]), float(tail[1]), float(tail[2]))
            if (eb.tail - eb.head).length < 1e-5:
                raise OpError(f"Bone '{bname}' has zero length (head == tail)", code="INVALID_PARAM")
            if bspec.get("roll_deg") is not None:
                eb.roll = math.radians(float(bspec["roll_deg"]))
            eb.use_deform = bool(bspec.get("deform", True))
            created_ebones[bname] = eb

        # Pass 2: wire parent relationships
        for bspec in bones_spec:
            bname = str(bspec["name"]).strip()
            parent_name = str(bspec.get("parent") or "").strip()
            if parent_name:
                parent_eb = created_ebones.get(parent_name)
                if parent_eb is None:
                    raise OpError(f"Parent bone '{parent_name}' for '{bname}' not found", code="NOT_FOUND")
                eb = created_ebones[bname]
                eb.parent = parent_eb
                eb.use_connect = bool(bspec.get("connected", False))
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")

    # Initialize pose bones to Euler XYZ for intuitive degree-based posing & keyframing
    if arm_obj.pose:
        for pb in arm_obj.pose.bones:
            pb.rotation_mode = "XYZ"

    ctx.view_layer.update()

    bound_meshes: list[str] = []
    for m_name in params.get("bind_meshes") or []:
        m_obj = bpy.data.objects.get(str(m_name).strip())
        if m_obj is None:
            raise OpError(f"Mesh object to bind not found: '{m_name}'", code="NOT_FOUND")
        _bind_mesh_to_armature_smooth(bpy, m_obj, arm_obj)
        bound_meshes.append(m_obj.name)

    ctx.view_layer.update()
    return {
        "name": arm_obj.name,
        "collection": col.name,
        "bones_count": len(arm_data.bones),
        "bones": [b.name for b in arm_data.bones],
        "bound_meshes": bound_meshes,
    }


@blender_op(
    "rig.armature_inspect",
    tier="read",
    summary="Inspect an Armature's bone hierarchy, rest lengths, local pose channels, and evaluated world-space head/tail positions.",
    params_doc={"name": "Armature object name"},
)
def op_rig_armature_inspect(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    name = str(params.get("name") or "").strip()
    arm_obj = bpy.data.objects.get(name)
    if arm_obj is None or arm_obj.type != "ARMATURE" or arm_obj.data is None:
        raise OpError(f"Armature object not found: '{name}'", code="NOT_FOUND")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_arm = arm_obj.evaluated_get(depsgraph)
    mw = eval_arm.matrix_world

    bones_info: list[dict[str, Any]] = []
    for bone in arm_obj.data.bones:
        pb = eval_arm.pose.bones.get(bone.name) if eval_arm.pose else None
        if pb is not None:
            w_head = mw @ pb.head
            w_tail = mw @ pb.tail
            rot_eul = (
                pb.rotation_quaternion.to_euler("XYZ")
                if pb.rotation_mode == "QUATERNION"
                else pb.rotation_euler
            )
            pose_loc = [round(float(v), 4) for v in pb.location]
            pose_rot = [round(math.degrees(float(v)), 3) for v in rot_eul]
            pose_scl = [round(float(v), 4) for v in pb.scale]
        else:
            w_head = mw @ bone.head_local
            w_tail = mw @ bone.tail_local
            pose_loc = [0.0, 0.0, 0.0]
            pose_rot = [0.0, 0.0, 0.0]
            pose_scl = [1.0, 1.0, 1.0]

        bones_info.append(
            {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "connected": bool(bone.use_connect),
                "length": round(float(bone.length), 4),
                "rest_head": [round(float(v), 4) for v in bone.head_local],
                "rest_tail": [round(float(v), 4) for v in bone.tail_local],
                "world_head": [round(float(v), 4) for v in w_head],
                "world_tail": [round(float(v), 4) for v in w_tail],
                "pose_location": pose_loc,
                "pose_rotation_deg": pose_rot,
                "pose_scale": pose_scl,
            }
        )

    return {
        "name": arm_obj.name,
        "bones_count": len(bones_info),
        "bones": bones_info,
    }


@blender_op(
    "rig.pose",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Pose one or more bones on an Armature (`rotation_deg`, `location`, `scale`) and optionally insert "
        "Blender 5.3 slotted-Action keyframes at `frame`."
    ),
    params_doc={
        "armature": "Armature object name",
        "poses": (
            "Dict mapping bone name -> `{'rotation_deg'?: [rx,ry,rz], 'location'?: [x,y,z], 'scale'?: [sx,sy,sz]}`"
        ),
        "action": "Optional shared Multi-ID Slotted Action name (e.g. `'Shot_01'`)",
        "frame": "Optional frame number to set before posing/keyframing",
        "keyframe": "If true (default when `frame` is specified), insert keyframes on posed channels",
        "interpolation": "Keyframe interpolation ('BEZIER' default, 'LINEAR', 'CONSTANT', 'BACK', etc.)",
        "reset_unlisted": "If true, reset all unlisted bones back to rest pose first (default false)",
    },
)
def op_rig_pose(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    arm_name = str(params.get("armature") or params.get("name") or "").strip()
    arm_obj = bpy.data.objects.get(arm_name)
    if arm_obj is None or arm_obj.type != "ARMATURE" or arm_obj.pose is None:
        raise OpError(f"Armature object not found: '{arm_name}'", code="NOT_FOUND")

    poses = params.get("poses")
    if not isinstance(poses, dict) or not poses:
        raise OpError("rig.pose requires a non-empty 'poses' dict mapping bone names to transforms", code="MISSING_PARAM")

    has_frame = params.get("frame") is not None
    if has_frame:
        ctx.scene.frame_set(int(params["frame"]))
    target_frame = float(ctx.scene.frame_current)
    do_key = bool(params.get("keyframe", has_frame))
    interp = str(params.get("interpolation") or "BEZIER").upper()
    act_name = str(params.get("action") or "").strip() or None

    if bool(params.get("reset_unlisted", False)):
        for pb in arm_obj.pose.bones:
            if pb.name not in poses:
                pb.location = (0.0, 0.0, 0.0)
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = (0.0, 0.0, 0.0)
                pb.scale = (1.0, 1.0, 1.0)

    fcurves = ensure_channelbag_fcurves(arm_obj, action_name=act_name) if do_key else None
    posed_bones: list[str] = []

    def _insert_bone_vec_keys(bname: str, prop: str, vec3: tuple[float, float, float]) -> None:
        assert fcurves is not None
        dp = f'pose.bones["{bname}"].{prop}'
        for ax in range(3):
            fc = next((c for c in fcurves if c.data_path == dp and c.array_index == ax), None)
            if fc is None:
                fc = fcurves.new(data_path=dp, index=ax)
            kp = fc.keyframe_points.insert(target_frame, float(vec3[ax]), options={"FAST"})
            kp.interpolation = interp
            if interp == "BEZIER":
                kp.handle_left_type = "AUTO_CLAMPED"
                kp.handle_right_type = "AUTO_CLAMPED"
            fc.update()

    for bname, tf in poses.items():
        pb = arm_obj.pose.bones.get(str(bname))
        if pb is None:
            raise OpError(f"Bone '{bname}' not found on armature '{arm_obj.name}'", code="NOT_FOUND")
        if pb.rotation_mode == "QUATERNION":
            pb.rotation_euler = pb.rotation_quaternion.to_euler("XYZ")
            pb.rotation_mode = "XYZ"

        if isinstance(tf, dict):
            if tf.get("location") is not None:
                loc = tf["location"]
                pb.location = (float(loc[0]), float(loc[1]), float(loc[2]))
                if do_key:
                    _insert_bone_vec_keys(pb.name, "location", tuple(pb.location))
            if tf.get("rotation_deg") is not None:
                rot = tf["rotation_deg"]
                pb.rotation_euler = tuple(math.radians(float(a)) for a in rot[:3])
                if do_key:
                    _insert_bone_vec_keys(pb.name, "rotation_euler", tuple(pb.rotation_euler))
            if tf.get("scale") is not None:
                scl = tf["scale"]
                if isinstance(scl, (int, float)):
                    s = float(scl)
                    pb.scale = (s, s, s)
                else:
                    pb.scale = (float(scl[0]), float(scl[1]), float(scl[2]))
                if do_key:
                    _insert_bone_vec_keys(pb.name, "scale", tuple(pb.scale))
        posed_bones.append(pb.name)

    ctx.view_layer.update()
    return {
        "armature": arm_obj.name,
        "posed_bones": posed_bones,
        "frame": int(target_frame),
        "keyframed": do_key,
    }