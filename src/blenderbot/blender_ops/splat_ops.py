"""Native Blender 5.3+ 3D Gaussian Splat Operations (`blenderbot.blender_ops.splat_ops`).

Pure-stdlib + `bpy` / `mathutils` implementations (zero third-party dependencies inside Blender):
- `splat.capture_views`: Multi-view Blender -> known-pose `.tar.gz` bundle (exact `R_w2c, t_w2c, K`
  + surface-sampled `init_points.bin` + RGBA/RGB stills) for zero-drift Modal H100 3DGS training.
- `splat.import`: Fast strided binary `.ply` parser -> native Blender 5.3 `bpy.data.pointclouds`
  (`position`, `radius`, `splat_color`, `splat_scale`, `splat_rot`), optional Geometry Nodes
  anisotropic 3D ellipsoids, `SeedanceSourceCam` Slotted Action, and `Z=0` shadow-catcher ground.
- `splat.crop_clean`: Prune floaters/outliers on a splat `PointCloud` by AABB, opacity, or radius.
- `splat.project_video`: Camera-projected `MOVIE`/`IMAGE` shader onto proxy geometry.
"""

from __future__ import annotations

import io
import json
import math
import random
import struct
import tarfile
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any

from .helpers import ensure_channelbag_fcurves, world_aabb
from .schema import OpContext, OpError, blender_op

_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199


def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return max(0.0, c / 12.92)
    return ((c + 0.055) / 1.055) ** 2.4


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _cam_matrix_world_to_opencv_w2c(mw: Any) -> tuple[list[list[float]], list[float]]:
    """Convert Blender camera `matrix_world` (`+X right, +Y up, -Z fwd`) to OpenCV `(R_w2c, t_w2c)`."""
    r_w2c = [
        [float(mw[0][0]), float(mw[1][0]), float(mw[2][0])],
        [-float(mw[0][1]), -float(mw[1][1]), -float(mw[2][1])],
        [-float(mw[0][2]), -float(mw[1][2]), -float(mw[2][2])],
    ]
    lx, ly, lz = float(mw[0][3]), float(mw[1][3]), float(mw[2][3])
    t_w2c = [
        -(r_w2c[0][0] * lx + r_w2c[0][1] * ly + r_w2c[0][2] * lz),
        -(r_w2c[1][0] * lx + r_w2c[1][1] * ly + r_w2c[1][2] * lz),
        -(r_w2c[2][0] * lx + r_w2c[2][1] * ly + r_w2c[2][2] * lz),
    ]
    return r_w2c, t_w2c


def _resolve_target_objects(ctx: OpContext, params: dict[str, Any]) -> tuple[list[Any], bool]:
    bpy = ctx.bpy
    sc = ctx.scene
    raw_targets = params.get("targets")
    coll_name = params.get("collection")

    render_types = {"MESH", "CURVE", "SURFACE", "FONT", "META", "POINTCLOUD", "VOLUME"}
    if raw_targets:
        names = [raw_targets] if isinstance(raw_targets, str) else [str(n) for n in raw_targets]
        objs = []
        for nm in names:
            o = bpy.data.objects.get(nm)
            if o is None:
                raise OpError(f"Target object '{nm}' not found", code="OBJECT_NOT_FOUND")
            objs.append(o)
        return objs, True

    if coll_name:
        coll = bpy.data.collections.get(str(coll_name))
        if coll is None:
            raise OpError(f"Collection '{coll_name}' not found", code="COLLECTION_NOT_FOUND")
        objs = [o for o in coll.all_objects if o.type in render_types and not o.hide_render]
        if not objs:
            raise OpError(f"Collection '{coll_name}' has no renderable objects", code="EMPTY_TARGETS")
        return objs, True

    objs = [o for o in sc.objects if o.type in render_types and not o.hide_render]
    if not objs:
        raise OpError("No visible renderable objects in scene", code="EMPTY_SCENE")
    return objs, False


def _material_slot_rgb_u8(obj: Any, mat_index: int) -> tuple[int, int, int]:
    slots = getattr(obj, "material_slots", None)
    if slots and 0 <= mat_index < len(slots):
        mat = slots[mat_index].material
        if mat is not None:
            if getattr(mat, "use_nodes", False) and mat.node_tree is not None:
                for node in mat.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        col = node.inputs["Base Color"].default_value
                        return (
                            max(0, min(255, round((col[0] ** (1.0 / 2.2)) * 255.0))),
                            max(0, min(255, round((col[1] ** (1.0 / 2.2)) * 255.0))),
                            max(0, min(255, round((col[2] ** (1.0 / 2.2)) * 255.0))),
                        )
                    if node.type == "EMISSION":
                        col = node.inputs["Color"].default_value
                        return (
                            max(0, min(255, round((col[0] ** (1.0 / 2.2)) * 255.0))),
                            max(0, min(255, round((col[1] ** (1.0 / 2.2)) * 255.0))),
                            max(0, min(255, round((col[2] ** (1.0 / 2.2)) * 255.0))),
                        )
            dc = getattr(mat, "diffuse_color", None)
            if dc is not None:
                return (
                    max(0, min(255, round((dc[0] ** (1.0 / 2.2)) * 255.0))),
                    max(0, min(255, round((dc[1] ** (1.0 / 2.2)) * 255.0))),
                    max(0, min(255, round((dc[2] ** (1.0 / 2.2)) * 255.0))),
                )
    return (192, 192, 192)


def _sample_surface_points_bin(
    ctx: OpContext,
    targets: list[Any],
    bmin: list[float],
    bmax: list[float],
    max_points: int = 40_000,
) -> tuple[bytes, int]:
    """Sample surface seed points `(x, y, z, r, g, b)` from evaluated target geometry into binary."""
    depsgraph = ctx.bpy.context.evaluated_depsgraph_get()
    rng = random.Random(42)

    tri_pool: list[tuple[float, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]]] = []
    direct_pts: list[tuple[float, float, float, int, int, int]] = []
    total_area = 0.0

    for obj in targets:
        if obj.type == "POINTCLOUD":
            pos_attr = obj.data.attributes.get("position")
            col_attr = obj.data.attributes.get("splat_color")
            if pos_attr is not None:
                mw = obj.matrix_world
                n_pc = len(pos_attr.data)
                step_pc = max(1, n_pc // max(1, max_points // max(1, len(targets))))
                for idx in range(0, n_pc, step_pc):
                    v = mw @ pos_attr.data[idx].vector
                    if col_attr is not None and idx < len(col_attr.data):
                        c = col_attr.data[idx].color
                        rgb = (
                            max(0, min(255, int(c[0] * 255))),
                            max(0, min(255, int(c[1] * 255))),
                            max(0, min(255, int(c[2] * 255))),
                        )
                    else:
                        rgb = (192, 192, 192)
                    direct_pts.append((float(v.x), float(v.y), float(v.z), *rgb))
            continue

        eval_obj = obj.evaluated_get(depsgraph)
        mesh = None
        try:
            mesh = eval_obj.to_mesh()
        except RuntimeError:
            mesh = None
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            verts = mesh.vertices
            mw = obj.matrix_world
            slot_colors = [_material_slot_rgb_u8(obj, i) for i in range(max(1, len(obj.material_slots)))]
            for tri in mesh.loop_triangles:
                p0 = mw @ verts[tri.vertices[0]].co
                p1 = mw @ verts[tri.vertices[1]].co
                p2 = mw @ verts[tri.vertices[2]].co
                area = 0.5 * (p1 - p0).cross(p2 - p0).length
                if area <= 1e-10:
                    continue
                col = slot_colors[tri.material_index] if tri.material_index < len(slot_colors) else (192, 192, 192)
                total_area += area
                tri_pool.append((
                    total_area,
                    (float(p0.x), float(p0.y), float(p0.z)),
                    (float(p1.x), float(p1.y), float(p1.z)),
                    (float(p2.x), float(p2.y), float(p2.z)),
                    col,
                ))
        finally:
            eval_obj.to_mesh_clear()

    buf = bytearray(4)
    count = 0

    if direct_pts:
        if len(direct_pts) > max_points:
            direct_pts = rng.sample(direct_pts, max_points)
        for px, py, pz, cr, cg, cb in direct_pts:
            buf.extend(struct.pack("<3f3B", px, py, pz, cr, cg, cb))
            count += 1

    rem = max(0, max_points - count)
    if tri_pool and total_area > 0.0 and rem > 0:
        import bisect

        cum_areas = [t[0] for t in tri_pool]
        for _ in range(rem):
            r_a = rng.random() * total_area
            idx = min(len(tri_pool) - 1, bisect.bisect_left(cum_areas, r_a))
            _, p0, p1, p2, col = tri_pool[idx]
            u = math.sqrt(rng.random())
            v = rng.random()
            w0 = 1.0 - u
            w1 = u * (1.0 - v)
            w2 = u * v
            px = w0 * p0[0] + w1 * p1[0] + w2 * p2[0]
            py = w0 * p0[1] + w1 * p1[1] + w2 * p2[1]
            pz = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
            buf.extend(struct.pack("<3f3B", px, py, pz, col[0], col[1], col[2]))
            count += 1

    if count == 0:
        n_fb = min(4_000, max_points)
        for _ in range(n_fb):
            px = rng.uniform(bmin[0], bmax[0])
            py = rng.uniform(bmin[1], bmax[1])
            pz = rng.uniform(bmin[2], bmax[2])
            buf.extend(struct.pack("<3f3B", px, py, pz, 180, 180, 180))
            count += 1

    struct.pack_into("<I", buf, 0, count)
    return bytes(buf), count


@blender_op(
    "splat.capture_views",
    tier="read",
    summary=(
        "Render multi-view 2D stills at known Blender camera poses + surface seed points "
        "into a `.tar.gz` bundle for zero-drift Modal H100 3DGS generation."
    ),
    params_doc={
        "output_path": "Destination `.tar.gz` bundle path (defaults to temp `views_bundle.tar.gz`)",
        "targets": "Optional object name or list of object names to isolate (omitted = entire scene)",
        "collection": "Optional collection name to isolate",
        "cameras": "Optional list of existing camera names to capture instead of generating an orbital rig",
        "num_views": "Orbital views to generate when `cameras` is omitted (default 36)",
        "elevations_deg": "Optional elevation angles in degrees (default `[12, 32, 56]`)",
        "distance_scale": "Orbit radius multiplier relative to bounding radius (default 2.6)",
        "lens_mm": "Focal length in mm for orbital camera (default 50.0)",
        "resolution": "`[width, height]` in pixels (default `[768, 768]`)",
        "engine": "`'BLENDER_EEVEE'` (default), `'CYCLES'`, or `'BLENDER_WORKBENCH'`",
        "transparent_bg": "Render RGBA transparent background for clean isolated objects (default True if `targets`/`collection` set)",
        "max_init_points": "Surface seed point cap sampled into `init_points.bin` (default 40000)",
    },
)
def op_splat_capture_views(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    from mathutils import Vector

    bpy = ctx.bpy
    sc = ctx.scene
    r = sc.render

    raw_out = str(params.get("output_path") or "").strip()
    if raw_out:
        out_path = Path(raw_out).resolve()
        if not str(out_path).endswith(".tar.gz"):
            out_path = Path(str(out_path.with_suffix("")) + ".tar.gz")
    else:
        out_path = Path(tempfile.gettempdir()) / "blenderbot_views_bundle.tar.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets, is_isolated = _resolve_target_objects(ctx, params)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    bmin = [float("inf"), float("inf"), float("inf")]
    bmax = [float("-inf"), float("-inf"), float("-inf")]
    for obj in targets:
        ab = world_aabb(obj, depsgraph=depsgraph)
        if ab is None:
            continue
        for i in range(3):
            bmin[i] = min(bmin[i], ab["min"][i])
            bmax[i] = max(bmax[i], ab["max"][i])
    if bmin[0] == float("inf"):
        bmin, bmax = [-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]

    center = Vector(((bmin[0] + bmax[0]) * 0.5, (bmin[1] + bmax[1]) * 0.5, (bmin[2] + bmax[2]) * 0.5))
    ext = Vector((bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]))
    radius = max(0.15, 0.5 * ext.length)

    res = params.get("resolution") or [768, 768]
    width, height = max(128, int(res[0])), max(128, int(res[1]))
    lens_mm = float(params.get("lens_mm", 50.0))
    dist_scale = float(params.get("distance_scale", 2.6))
    transparent_bg = bool(params.get("transparent_bg", is_isolated))
    max_init_pts = int(params.get("max_init_points", 40_000))
    engine = str(params.get("engine") or "BLENDER_EEVEE").upper()

    init_pts_bin, n_init_pts = _sample_surface_points_bin(ctx, targets, bmin, bmax, max_points=max_init_pts)

    # Snapshot render state for spotless restoration
    orig_cam = sc.camera
    orig_engine = str(r.engine)
    orig_res = (int(r.resolution_x), int(r.resolution_y), int(r.resolution_percentage))
    orig_filepath = str(r.filepath)
    orig_media = getattr(r.image_settings, "media_type", None)
    orig_fmt = str(r.image_settings.file_format)
    orig_color_mode = str(r.image_settings.color_mode)
    orig_film_trans = bool(getattr(r, "film_transparent", False))

    target_set = {o.name for o in targets}
    hidden_restore: list[tuple[Any, bool]] = []
    temp_cam_obj = None
    temp_cam_data = None

    t0 = time.perf_counter()
    views_manifest: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="bb_cap_") as tmp_dir_s:
        tmp_dir = Path(tmp_dir_s)
        try:
            if is_isolated:
                for o in sc.objects:
                    if (
                        o.type in {"MESH", "CURVE", "SURFACE", "FONT", "META", "POINTCLOUD", "VOLUME"}
                        and o.name not in target_set
                        and not o.hide_render
                    ):
                        hidden_restore.append((o, bool(o.hide_render)))
                        o.hide_render = True

            r.engine = engine
            r.resolution_x = width
            r.resolution_y = height
            r.resolution_percentage = 100
            if hasattr(r, "film_transparent"):
                r.film_transparent = transparent_bg
            if orig_media is not None:
                r.image_settings.media_type = "IMAGE"
            r.image_settings.file_format = "PNG"
            r.image_settings.color_mode = "RGBA" if transparent_bg else "RGB"

            custom_cams = params.get("cameras")
            if custom_cams:
                cam_list: list[tuple[Any, Any]] = []
                for cname in custom_cams:
                    co = bpy.data.objects.get(str(cname))
                    if co is None or co.type != "CAMERA":
                        raise OpError(f"Camera '{cname}' not found", code="CAMERA_NOT_FOUND")
                    cam_list.append((co, None))
            else:
                num_views = max(6, int(params.get("num_views", 36)))
                elevs = [float(e) for e in (params.get("elevations_deg") or [12.0, 32.0, 56.0])]
                temp_cam_data = bpy.data.cameras.new("_SplatCaptureCamData")
                temp_cam_data.sensor_fit = "HORIZONTAL"
                temp_cam_data.sensor_width = 36.0
                temp_cam_data.lens = lens_mm
                temp_cam_data.clip_start = max(0.01, radius * 0.02)
                temp_cam_data.clip_end = max(100.0, radius * 40.0)
                temp_cam_obj = bpy.data.objects.new("_SplatCaptureCam", temp_cam_data)
                sc.collection.objects.link(temp_cam_obj)

                orbit_r = radius * dist_scale
                poses: list[Vector] = []
                n_rings = max(1, len(elevs))
                per_ring = max(1, num_views // n_rings)
                for r_i, el_deg in enumerate(elevs):
                    el_rad = math.radians(el_deg)
                    n_here = num_views - len(poses) if r_i == n_rings - 1 else per_ring
                    az_offset = (math.pi / max(1, n_here)) * (r_i % 2)
                    for k in range(n_here):
                        az = az_offset + (2.0 * math.pi * k) / max(1, n_here)
                        px = center.x + orbit_r * math.cos(el_rad) * math.cos(az)
                        py = center.y + orbit_r * math.cos(el_rad) * math.sin(az)
                        pz = center.z + orbit_r * math.sin(el_rad)
                        poses.append(Vector((px, py, pz)))
                cam_list = [(temp_cam_obj, pos) for pos in poses]

            for idx, (cam_obj, pos_vec) in enumerate(cam_list):
                if pos_vec is not None:
                    cam_obj.location = pos_vec
                    direction = center - pos_vec
                    cam_obj.rotation_mode = "QUATERNION"
                    cam_obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")
                else:
                    # Normalize sensor fit temporarily on custom camera? Keep existing intrinsics accurate!
                    pass
                ctx.view_layer.update()

                cd = cam_obj.data
                sw = float(getattr(cd, "sensor_width", 36.0))
                sh = float(getattr(cd, "sensor_height", 24.0))
                fit = str(getattr(cd, "sensor_fit", "AUTO"))
                lens = float(getattr(cd, "lens", 50.0))
                if fit == "VERTICAL" or (fit == "AUTO" and height > width):
                    fx = fy = lens * float(height) / max(1e-6, sh)
                else:
                    fx = fy = lens * float(width) / max(1e-6, sw)
                cx = width * 0.5 - float(getattr(cd, "shift_x", 0.0)) * width
                cy = height * 0.5 + float(getattr(cd, "shift_y", 0.0)) * width

                r_w2c, t_w2c = _cam_matrix_world_to_opencv_w2c(cam_obj.matrix_world)

                img_name = f"view_{idx:04d}.png"
                img_path = tmp_dir / img_name
                sc.camera = cam_obj
                r.filepath = str(img_path)
                bpy.ops.render.render(write_still=True)

                views_manifest.append({
                    "frame_index": idx,
                    "image_name": img_name,
                    "width": width,
                    "height": height,
                    "fx": round(fx, 4),
                    "fy": round(fy, 4),
                    "cx": round(cx, 4),
                    "cy": round(cy, 4),
                    "R_w2c": r_w2c,
                    "t_w2c": t_w2c,
                })

            manifest_doc = {
                "coordinate_system": "BLENDER_RIGHT_HANDED_Z_UP",
                "transparent_bg": transparent_bg,
                "bounds_min": [round(v, 5) for v in bmin],
                "bounds_max": [round(v, 5) for v in bmax],
                "center": [round(float(center.x), 5), round(float(center.y), 5), round(float(center.z), 5)],
                "radius": round(float(radius), 5),
                "init_points_count": n_init_pts,
                "views": views_manifest,
            }
            manifest_bytes = json.dumps(manifest_doc, indent=2).encode("utf-8")

            with tarfile.open(out_path, "w:gz", compresslevel=2) as tf:
                ti_m = tarfile.TarInfo("manifest.json")
                ti_m.size = len(manifest_bytes)
                tf.addfile(ti_m, io.BytesIO(manifest_bytes))

                ti_p = tarfile.TarInfo("init_points.bin")
                ti_p.size = len(init_pts_bin)
                tf.addfile(ti_p, io.BytesIO(init_pts_bin))

                for v in views_manifest:
                    vp = tmp_dir / v["image_name"]
                    tf.add(vp, arcname=v["image_name"])

        finally:
            for obj_r, was_hidden in hidden_restore:
                obj_r.hide_render = was_hidden
            sc.camera = orig_cam
            if temp_cam_obj is not None:
                bpy.data.objects.remove(temp_cam_obj, do_unlink=True)
            if temp_cam_data is not None:
                bpy.data.cameras.remove(temp_cam_data, do_unlink=True)
            r.engine = orig_engine
            r.resolution_x, r.resolution_y, r.resolution_percentage = orig_res
            if orig_media is not None:
                r.image_settings.media_type = orig_media
            r.image_settings.file_format = orig_fmt
            r.image_settings.color_mode = orig_color_mode
            if hasattr(r, "film_transparent"):
                r.film_transparent = orig_film_trans
            r.filepath = orig_filepath

    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "output_path": str(out_path),
        "views": len(views_manifest),
        "resolution": [width, height],
        "transparent_bg": transparent_bg,
        "init_points": n_init_pts,
        "bounds_min": [round(v, 4) for v in bmin],
        "bounds_max": [round(v, 4) for v in bmax],
        "size_bytes": out_path.stat().st_size if out_path.is_file() else 0,
        "elapsed_s": elapsed_s,
    }


def _parse_gaussian_ply_stdlib(
    ply_path: Path,
    *,
    min_opacity: float = 0.025,
    max_points: int = 250_000,
    point_scale: float = 1.0,
    max_world_scale: float = 0.28,
    max_anisotropy: float = 12.0,
) -> dict[str, Any]:
    """Vectorized binary `.ply` parser (`<60 ms` for 700k+ Gaussians via `numpy`, with stdlib fallback)."""
    raw = ply_path.read_bytes()
    hdr_end = raw.find(b"end_header\n")
    if hdr_end < 0:
        hdr_end = raw.find(b"end_header\r\n")
        if hdr_end < 0:
            raise OpError(f"Invalid PLY header in {ply_path.name}", code="INVALID_PLY")
        body_offset = hdr_end + len(b"end_header\r\n")
    else:
        body_offset = hdr_end + len(b"end_header\n")

    header = raw[:hdr_end].decode("ascii", errors="replace").splitlines()
    is_little = False
    n_verts = 0
    in_vertex = False
    props: list[tuple[str, str]] = []
    for ln in header:
        toks = ln.strip().split()
        if not toks:
            continue
        if toks[0] == "format":
            is_little = toks[1] == "binary_little_endian"
        elif toks[0] == "element":
            in_vertex = toks[1] == "vertex"
            if in_vertex:
                n_verts = int(toks[2])
        elif toks[0] == "property" and in_vertex and len(toks) >= 3:
            props.append((toks[1], toks[2]))

    if not is_little or n_verts <= 0:
        raise OpError(f"Unsupported PLY format or empty vertex count in {ply_path.name}", code="INVALID_PLY")

    all_floats = all(dt in ("float", "float32") for dt, _ in props)
    pidx = {name: i for i, (_, name) in enumerate(props)}

    if all_floats:
        import numpy as np

        stride = len(props)
        data = np.frombuffer(
            raw, dtype="<f4", offset=body_offset, count=n_verts * stride
        ).reshape(n_verts, stride)

        if "opacity" in pidx:
            op_logit = np.clip(data[:, pidx["opacity"]], -20.0, 20.0)
            op = 1.0 / (1.0 + np.exp(-op_logit))
        else:
            op = np.ones(n_verts, dtype=np.float32)

        has_scale = "scale_0" in pidx
        mask = op >= min_opacity
        if has_scale and (max_world_scale > 0 or max_anisotropy > 0):
            s_lin_all = np.exp(
                np.clip(
                    data[:, [pidx["scale_0"], pidx["scale_1"], pidx["scale_2"]]],
                    -16.0,
                    4.0,
                )
            )
            if max_world_scale > 0:
                mask &= s_lin_all.max(axis=1) <= max_world_scale
            if max_anisotropy > 0:
                s_part = np.sort(s_lin_all, axis=1)
                aniso = s_part[:, 2] / (s_part[:, 1] + 1e-6)
                mask &= aniso <= max_anisotropy

        keep = np.flatnonzero(mask)
        if keep.size == 0:
            return {
                "count": 0,
                "raw_count": n_verts,
                "pos": np.empty(0, dtype=np.float32),
                "rad": np.empty(0, dtype=np.float32),
                "col": np.empty(0, dtype=np.float32),
                "scale": np.empty(0, dtype=np.float32),
                "rot": np.empty(0, dtype=np.float32),
                "bounds_min": [0.0, 0.0, 0.0],
                "bounds_max": [0.0, 0.0, 0.0],
            }

        if max_points > 0 and keep.size > max_points:
            op_k = op[keep]
            if has_scale:
                s_max_log = np.maximum(
                    np.maximum(data[keep, pidx["scale_0"]], data[keep, pidx["scale_1"]]),
                    data[keep, pidx["scale_2"]],
                )
                sc_max = np.exp(np.clip(s_max_log, -20.0, 5.0))
            else:
                sc_max = 0.02
            weights = op_k * np.minimum(sc_max, 0.25)
            top_local = np.argpartition(weights, -max_points)[-max_points:]
            keep = keep[top_local]

        n = int(keep.size)
        pos = np.empty((n, 3), dtype=np.float32)
        pos[:, 0] = data[keep, pidx["x"]]
        pos[:, 1] = data[keep, pidx["y"]]
        pos[:, 2] = data[keep, pidx["z"]]

        bmin = [float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 2].min())]
        bmax = [float(pos[:, 0].max()), float(pos[:, 1].max()), float(pos[:, 2].max())]

        col = np.empty((n, 4), dtype=np.float32)
        dc_srgb: Any = None
        sh1_0: Any = None
        sh1_1: Any = None
        sh1_2: Any = None
        has_sh1 = all(f"f_rest_{k}" in pidx for k in range(9))

        if "f_dc_0" in pidx:
            dc_srgb = np.empty((n, 3), dtype=np.float32)
            dc_srgb[:, 0] = 0.5 + _SH_C0 * data[keep, pidx["f_dc_0"]]
            dc_srgb[:, 1] = 0.5 + _SH_C0 * data[keep, pidx["f_dc_1"]]
            dc_srgb[:, 2] = 0.5 + _SH_C0 * data[keep, pidx["f_dc_2"]]
            srgb_clamped = np.clip(dc_srgb, 0.0, 1.0)
            col[:, :3] = np.where(
                srgb_clamped <= 0.04045,
                srgb_clamped / 12.92,
                np.power((srgb_clamped + 0.055) / 1.055, 2.4),
            )
            if has_sh1:
                sh1_0 = np.empty((n, 3), dtype=np.float32)
                sh1_1 = np.empty((n, 3), dtype=np.float32)
                sh1_2 = np.empty((n, 3), dtype=np.float32)
                sh1_0[:, 0] = _SH_C1 * data[keep, pidx["f_rest_0"]]
                sh1_0[:, 1] = _SH_C1 * data[keep, pidx["f_rest_1"]]
                sh1_0[:, 2] = _SH_C1 * data[keep, pidx["f_rest_2"]]
                sh1_1[:, 0] = _SH_C1 * data[keep, pidx["f_rest_3"]]
                sh1_1[:, 1] = _SH_C1 * data[keep, pidx["f_rest_4"]]
                sh1_1[:, 2] = _SH_C1 * data[keep, pidx["f_rest_5"]]
                sh1_2[:, 0] = _SH_C1 * data[keep, pidx["f_rest_6"]]
                sh1_2[:, 1] = _SH_C1 * data[keep, pidx["f_rest_7"]]
                sh1_2[:, 2] = _SH_C1 * data[keep, pidx["f_rest_8"]]
        else:
            col[:, :3] = 0.5225
        col[:, 3] = op[keep]

        if has_scale:
            scl = np.empty((n, 3), dtype=np.float32)
            scl[:, 0] = np.exp(np.clip(data[keep, pidx["scale_0"]], -16.0, 4.0)) * point_scale
            scl[:, 1] = np.exp(np.clip(data[keep, pidx["scale_1"]], -16.0, 4.0)) * point_scale
            scl[:, 2] = np.exp(np.clip(data[keep, pidx["scale_2"]], -16.0, 4.0)) * point_scale
            np.maximum(scl, 1e-4, out=scl)
            # Scale pointcloud radius by middle axis * sqrt(opacity) so sparse haze points stay tiny
            # and dense surfaces pack tightly in fast PointCloud mode!
            s_part_k = np.sort(scl, axis=1)
            rad = np.clip(s_part_k[:, 1] * np.sqrt(op[keep]) * 1.35, 0.0015, 0.045).astype(np.float32)
        else:
            scl = np.full((n, 3), 0.015 * point_scale, dtype=np.float32)
            rad = np.full(n, 0.015 * point_scale, dtype=np.float32)

        rot = np.empty((n, 4), dtype=np.float32)
        if "rot_0" in pidx:
            rot[:, 0] = data[keep, pidx["rot_0"]]
            rot[:, 1] = data[keep, pidx["rot_1"]]
            rot[:, 2] = data[keep, pidx["rot_2"]]
            rot[:, 3] = data[keep, pidx["rot_3"]]
            qnorm = np.linalg.norm(rot, axis=1, keepdims=True)
            qnorm = np.where(qnorm > 1e-8, qnorm, 1.0)
            rot /= qnorm
        else:
            rot[:] = (1.0, 0.0, 0.0, 0.0)

        return {
            "count": n,
            "raw_count": n_verts,
            "pos": pos.ravel(),
            "rad": rad,
            "col": col.ravel(),
            "scale": scl.ravel(),
            "rot": rot.ravel(),
            "has_sh1": has_sh1 and dc_srgb is not None,
            "dc_srgb": dc_srgb.ravel() if dc_srgb is not None else None,
            "sh1_0": sh1_0.ravel() if sh1_0 is not None else None,
            "sh1_1": sh1_1.ravel() if sh1_1 is not None else None,
            "sh1_2": sh1_2.ravel() if sh1_2 is not None else None,
            "bounds_min": bmin,
            "bounds_max": bmax,
        }

    # Mixed-dtype fallback (e.g. uchar red/green/blue PLYs)
    type_fmt = {
        "float": "f", "float32": "f", "double": "d", "float64": "d",
        "uchar": "B", "uint8": "B", "char": "b", "int8": "b",
        "ushort": "H", "uint16": "H", "short": "h", "int16": "h",
        "uint": "I", "uint32": "I", "int": "i", "int32": "i",
    }
    fmt = "<" + "".join(type_fmt.get(dt, "f") for dt, _ in props)
    rec_sz = struct.calcsize(fmt)
    n_take = min(n_verts, max_points if max_points > 0 else n_verts)
    stride_idx = max(1, n_verts // max(1, n_take))
    pos_out = array("f")
    rad_out = array("f")
    col_out = array("f")
    scl_out = array("f")
    rot_out = array("f")
    bmin = [float("inf"), float("inf"), float("inf")]
    bmax = [float("-inf"), float("-inf"), float("-inf")]
    ix, iy, iz = pidx["x"], pidx["y"], pidx["z"]
    ir, ig, ib = pidx.get("red", -1), pidx.get("green", -1), pidx.get("blue", -1)

    for i in range(0, n_verts, stride_idx):
        tup = struct.unpack_from(fmt, raw, body_offset + i * rec_sz)
        x, y, z = float(tup[ix]), float(tup[iy]), float(tup[iz])
        pos_out.extend((x, y, z))
        bmin[0] = min(bmin[0], x)
        bmin[1] = min(bmin[1], y)
        bmin[2] = min(bmin[2], z)
        bmax[0] = max(bmax[0], x)
        bmax[1] = max(bmax[1], y)
        bmax[2] = max(bmax[2], z)
        cr = (_srgb_to_linear(tup[ir] / 255.0) if ir >= 0 else 0.7)
        cg = (_srgb_to_linear(tup[ig] / 255.0) if ig >= 0 else 0.7)
        cb = (_srgb_to_linear(tup[ib] / 255.0) if ib >= 0 else 0.7)
        col_out.extend((cr, cg, cb, 1.0))
        rad_out.append(0.015 * point_scale)
        scl_out.extend((0.015 * point_scale, 0.015 * point_scale, 0.015 * point_scale))
        rot_out.extend((1.0, 0.0, 0.0, 0.0))

    n = len(rad_out)
    return {
        "count": n,
        "raw_count": n_verts,
        "pos": pos_out,
        "rad": rad_out,
        "col": col_out,
        "scale": scl_out,
        "rot": rot_out,
        "has_sh1": False,
        "bounds_min": bmin if n > 0 else [0.0, 0.0, 0.0],
        "bounds_max": bmax if n > 0 else [0.0, 0.0, 0.0],
    }


def _build_splat_material(
    bpy: Any,
    name: str,
    *,
    instancer: bool = False,
    unlit: bool = True,
    has_sh1: bool = False,
    alpha_mode: str = "DITHERED",
    cutoff_sigma: float = 2.5,
) -> Any:
    """Build native 3DGS shader: analytic Mahalanobis ray-integral `exp(-0.5 * rho^2)` + live L=1 SH."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = alpha_mode
    if instancer and hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (760, 0)

    attr_type = "INSTANCER" if instancer else "GEOMETRY"
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "splat_color"
    attr.attribute_type = attr_type
    attr.location = (-680, 80)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    geom.location = (-900, 360)

    # Color path: live view-dependent L=1 SH if present, else pre-converted linear DC
    color_socket = attr.outputs["Color"]
    if unlit and has_sh1:
        sep_i = nt.nodes.new("ShaderNodeSeparateXYZ")
        sep_i.location = (-680, -240)
        nt.links.new(geom.outputs["Incoming"], sep_i.inputs["Vector"])

        dc_node = nt.nodes.new("ShaderNodeAttribute")
        dc_node.attribute_name = "splat_dc"
        dc_node.attribute_type = attr_type
        dc_node.location = (-680, -380)

        sh0_node = nt.nodes.new("ShaderNodeAttribute")
        sh0_node.attribute_name = "splat_sh1_0"
        sh0_node.attribute_type = attr_type
        sh0_node.location = (-680, -520)

        sh1_node = nt.nodes.new("ShaderNodeAttribute")
        sh1_node.attribute_name = "splat_sh1_1"
        sh1_node.attribute_type = attr_type
        sh1_node.location = (-680, -660)

        sh2_node = nt.nodes.new("ShaderNodeAttribute")
        sh2_node.attribute_name = "splat_sh1_2"
        sh2_node.attribute_type = attr_type
        sh2_node.location = (-680, -800)

        # srgb = dc + iy * sh1_0 - iz * sh1_1 + ix * sh1_2
        t0 = nt.nodes.new("ShaderNodeVectorMath")
        t0.operation = "SCALE"
        nt.links.new(sh0_node.outputs["Vector"], t0.inputs[0])
        nt.links.new(sep_i.outputs["Y"], t0.inputs["Scale"])

        t1 = nt.nodes.new("ShaderNodeVectorMath")
        t1.operation = "SCALE"
        nt.links.new(sh1_node.outputs["Vector"], t1.inputs[0])
        nt.links.new(sep_i.outputs["Z"], t1.inputs["Scale"])

        t2 = nt.nodes.new("ShaderNodeVectorMath")
        t2.operation = "SCALE"
        nt.links.new(sh2_node.outputs["Vector"], t2.inputs[0])
        nt.links.new(sep_i.outputs["X"], t2.inputs["Scale"])

        sum_a = nt.nodes.new("ShaderNodeVectorMath")
        sum_a.operation = "ADD"
        nt.links.new(dc_node.outputs["Vector"], sum_a.inputs[0])
        nt.links.new(t0.outputs["Vector"], sum_a.inputs[1])

        sum_b = nt.nodes.new("ShaderNodeVectorMath")
        sum_b.operation = "SUBTRACT"
        nt.links.new(sum_a.outputs["Vector"], sum_b.inputs[0])
        nt.links.new(t1.outputs["Vector"], sum_b.inputs[1])

        sum_c = nt.nodes.new("ShaderNodeVectorMath")
        sum_c.operation = "ADD"
        nt.links.new(sum_b.outputs["Vector"], sum_c.inputs[0])
        nt.links.new(t2.outputs["Vector"], sum_c.inputs[1])

        vmax = nt.nodes.new("ShaderNodeVectorMath")
        vmax.operation = "MAXIMUM"
        vmax.inputs[1].default_value = (0.0, 0.0, 0.0)
        nt.links.new(sum_c.outputs["Vector"], vmax.inputs[0])

        vmin = nt.nodes.new("ShaderNodeVectorMath")
        vmin.operation = "MINIMUM"
        vmin.inputs[1].default_value = (1.0, 1.0, 1.0)
        nt.links.new(vmax.outputs["Vector"], vmin.inputs[0])

        gam = nt.nodes.new("ShaderNodeGamma")
        gam.inputs["Gamma"].default_value = 2.2
        nt.links.new(vmin.outputs["Vector"], gam.inputs["Color"])
        color_socket = gam.outputs["Color"]

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (260, 180)

    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (520, 40)

    if unlit:
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        em.location = (260, -40)
        nt.links.new(color_socket, em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], mix.inputs[2])
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.85
        bsdf.location = (260, -80)
        nt.links.new(color_socket, bsdf.inputs["Base Color"])
        nt.links.new(bsdf.outputs["BSDF"], mix.inputs[2])

    if instancer:
        # Exact analytic 3DGS ray impact parameter in per-instance Mahalanobis space:
        # u_hit = TexCoord.Object = S^-1 R^T (x_world - mu)
        # w_hat = normalize(VectorTransform(WORLD->OBJECT, Incoming))
        # u_perp = u_hit - dot(u_hit, w_hat) * w_hat ; rho2 = dot(u_perp, u_perp)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        tc.location = (-900, 560)

        vt = nt.nodes.new("ShaderNodeVectorTransform")
        vt.vector_type = "VECTOR"
        vt.convert_from = "WORLD"
        vt.convert_to = "OBJECT"
        vt.location = (-680, 440)
        nt.links.new(geom.outputs["Incoming"], vt.inputs["Vector"])

        w_hat = nt.nodes.new("ShaderNodeVectorMath")
        w_hat.operation = "NORMALIZE"
        w_hat.location = (-480, 440)
        nt.links.new(vt.outputs["Vector"], w_hat.inputs[0])

        u_dot_w = nt.nodes.new("ShaderNodeVectorMath")
        u_dot_w.operation = "DOT_PRODUCT"
        u_dot_w.location = (-300, 520)
        nt.links.new(tc.outputs["Object"], u_dot_w.inputs[0])
        nt.links.new(w_hat.outputs["Vector"], u_dot_w.inputs[1])

        u_par = nt.nodes.new("ShaderNodeVectorMath")
        u_par.operation = "SCALE"
        u_par.location = (-120, 480)
        nt.links.new(w_hat.outputs["Vector"], u_par.inputs[0])
        nt.links.new(u_dot_w.outputs["Value"], u_par.inputs["Scale"])

        u_perp = nt.nodes.new("ShaderNodeVectorMath")
        u_perp.operation = "SUBTRACT"
        u_perp.location = (60, 520)
        nt.links.new(tc.outputs["Object"], u_perp.inputs[0])
        nt.links.new(u_par.outputs["Vector"], u_perp.inputs[1])

        rho2 = nt.nodes.new("ShaderNodeVectorMath")
        rho2.operation = "DOT_PRODUCT"
        rho2.location = (220, 520)
        nt.links.new(u_perp.outputs["Vector"], rho2.inputs[0])
        nt.links.new(u_perp.outputs["Vector"], rho2.inputs[1])

        neg_half = nt.nodes.new("ShaderNodeMath")
        neg_half.operation = "MULTIPLY"
        neg_half.inputs[1].default_value = -0.5
        nt.links.new(rho2.outputs["Value"], neg_half.inputs[0])

        gauss_k = nt.nodes.new("ShaderNodeMath")
        gauss_k.operation = "EXPONENT"
        nt.links.new(neg_half.outputs["Value"], gauss_k.inputs[0])

        in_cut = nt.nodes.new("ShaderNodeMath")
        in_cut.operation = "LESS_THAN"
        in_cut.inputs[1].default_value = float(cutoff_sigma * cutoff_sigma)
        nt.links.new(rho2.outputs["Value"], in_cut.inputs[0])

        front_mask = nt.nodes.new("ShaderNodeMath")
        front_mask.operation = "SUBTRACT"
        front_mask.inputs[0].default_value = 1.0
        nt.links.new(geom.outputs["Backfacing"], front_mask.inputs[1])

        a1 = nt.nodes.new("ShaderNodeMath")
        a1.operation = "MULTIPLY"
        nt.links.new(attr.outputs["Alpha"], a1.inputs[0])
        nt.links.new(gauss_k.outputs["Value"], a1.inputs[1])

        a2 = nt.nodes.new("ShaderNodeMath")
        a2.operation = "MULTIPLY"
        nt.links.new(a1.outputs["Value"], a2.inputs[0])
        nt.links.new(in_cut.outputs["Value"], a2.inputs[1])

        eff_alpha = nt.nodes.new("ShaderNodeMath")
        eff_alpha.operation = "MULTIPLY"
        nt.links.new(a2.outputs["Value"], eff_alpha.inputs[0])
        nt.links.new(front_mask.outputs["Value"], eff_alpha.inputs[1])
        nt.links.new(eff_alpha.outputs["Value"], mix.inputs["Fac"])
    else:
        nt.links.new(attr.outputs["Alpha"], mix.inputs["Fac"])

    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _build_shadow_ground_material(bpy: Any, name: str) -> Any:
    """Dual Cycles/EEVEE shadow ground: native raytraced shadow catcher in Cycles, invisible in EEVEE."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    nt = mat.node_tree
    nt.nodes.clear()

    out_cycles = nt.nodes.new("ShaderNodeOutputMaterial")
    out_cycles.target = "CYCLES"
    out_cycles.location = (320, 100)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.location = (20, 120)
    nt.links.new(bsdf.outputs["BSDF"], out_cycles.inputs["Surface"])

    out_eevee = nt.nodes.new("ShaderNodeOutputMaterial")
    out_eevee.target = "EEVEE"
    out_eevee.location = (320, -120)

    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (20, -120)
    nt.links.new(transp.outputs["BSDF"], out_eevee.inputs["Surface"])
    return mat


def _attach_ellipsoid_geonodes(
    bpy: Any,
    obj: Any,
    mat_inst: Any,
    *,
    mat_pc: Any | None = None,
    ellipsoid_radius: float = 2.5,
    viewport_fast_pointcloud: bool = True,
) -> None:
    """Instance 3DGS bounding cubes for F12 Render while keeping interactive viewport 60fps via IsViewport!"""
    mod = obj.modifiers.new(name="SplatEllipsoids", type="NODES")
    ng = bpy.data.node_groups.new(name=f"{obj.name}_SplatEllipsoids", type="GeometryNodeTree")
    mod.node_group = ng

    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    n_in = ng.nodes.new("NodeGroupInput")
    n_in.location = (-620, 0)
    n_out = ng.nodes.new("NodeGroupOutput")
    n_out.location = (620, 0)

    cube = ng.nodes.new("GeometryNodeMeshCube")
    d = float(ellipsoid_radius) * 2.0
    cube.inputs["Size"].default_value = (d, d, d)
    cube.location = (-360, -200)

    set_mat = ng.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat_inst
    set_mat.location = (-120, -200)

    attr_scale = ng.nodes.new("GeometryNodeInputNamedAttribute")
    attr_scale.data_type = "FLOAT_VECTOR"
    attr_scale.inputs["Name"].default_value = "splat_scale"
    attr_scale.location = (-320, -380)

    attr_rot = ng.nodes.new("GeometryNodeInputNamedAttribute")
    attr_rot.data_type = "QUATERNION"
    attr_rot.inputs["Name"].default_value = "splat_rot"
    attr_rot.location = (-320, -540)

    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    inst.location = (160, -120)

    ng.links.new(cube.outputs["Mesh"], set_mat.inputs["Geometry"])
    ng.links.new(n_in.outputs["Geometry"], inst.inputs["Points"])
    ng.links.new(set_mat.outputs["Geometry"], inst.inputs["Instance"])
    ng.links.new(attr_rot.outputs["Attribute"], inst.inputs["Rotation"])
    ng.links.new(attr_scale.outputs["Attribute"], inst.inputs["Scale"])

    if viewport_fast_pointcloud:
        is_vp = ng.nodes.new("GeometryNodeIsViewport")
        is_vp.location = (160, 140)
        vp_geom = n_in.outputs["Geometry"]
        if mat_pc is not None:
            set_mat_pc = ng.nodes.new("GeometryNodeSetMaterial")
            set_mat_pc.inputs["Material"].default_value = mat_pc
            set_mat_pc.location = (-120, 80)
            ng.links.new(n_in.outputs["Geometry"], set_mat_pc.inputs["Geometry"])
            vp_geom = set_mat_pc.outputs["Geometry"]
        sw = ng.nodes.new("GeometryNodeSwitch")
        sw.input_type = "GEOMETRY"
        sw.location = (400, 0)
        ng.links.new(is_vp.outputs["Is Viewport"], sw.inputs["Switch"])
        ng.links.new(inst.outputs["Instances"], sw.inputs["False"])
        ng.links.new(vp_geom, sw.inputs["True"])
        ng.links.new(sw.outputs["Output"], n_out.inputs["Geometry"])
    else:
        ng.links.new(inst.outputs["Instances"], n_out.inputs["Geometry"])


def _import_cameras_json(
    ctx: OpContext,
    cam_json_path: Path,
    name: str = "SeedanceSourceCam",
    *,
    sync_resolution: bool = True,
    spawn_frustum_rig: bool = True,
    bg_media_path: Path | None = None,
) -> dict[str, Any]:
    from mathutils import Euler, Quaternion

    bpy = ctx.bpy
    sc = ctx.scene
    doc = json.loads(cam_json_path.read_text(encoding="utf-8"))
    cams = doc.get("cameras") or []
    if not cams:
        return {"imported": False}

    # Replace stale camera / rig from prior re-imports cleanly
    old_cam = bpy.data.objects.get(name)
    if old_cam is not None:
        bpy.data.objects.remove(old_cam, do_unlink=True)

    c0 = cams[0]
    use_quat = "rotation_quaternion" in c0
    cam_data = bpy.data.cameras.new(f"{name}_Data")
    cam_data.sensor_fit = str(c0.get("sensor_fit", "HORIZONTAL"))
    cam_data.sensor_width = float(c0.get("sensor_width_mm", 36.0))
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    cam_data.shift_x = float(c0.get("shift_x", 0.0))
    cam_data.shift_y = float(c0.get("shift_y", 0.0))
    cam_data.display_size = 0.35

    if sync_resolution and c0.get("width") and c0.get("height"):
        sc.render.resolution_x = int(c0["width"])
        sc.render.resolution_y = int(c0["height"])
        sc.render.resolution_percentage = 100

    # Attach GT video / image as semi-transparent Foreground Camera Overlay for instant alignment check!
    bg_attached = None
    if bg_media_path is not None and bg_media_path.is_file():
        try:
            img = bpy.data.images.load(str(bg_media_path), check_existing=True)
            is_vid = bg_media_path.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")
            if is_vid:
                img.source = "MOVIE"
            cam_data.show_background_images = True
            bg_slot = cam_data.background_images.new()
            bg_slot.image = img
            bg_slot.alpha = 0.38
            bg_slot.display_depth = "FRONT"
            bg_slot.frame_method = "CROP"
            if is_vid and hasattr(bg_slot, "image_user"):
                bg_slot.image_user.use_auto_refresh = True
                bg_slot.image_user.frame_duration = max(len(cams), int(getattr(img, "frame_duration", len(cams)) or len(cams)))
                bg_slot.image_user.frame_start = 1
            bg_attached = bg_media_path.name
        except Exception:  # noqa: BLE001, S110
            pass

    cam_obj = bpy.data.objects.new(name, cam_data)
    cam_obj.rotation_mode = "QUATERNION" if use_quat else "XYZ"
    sc.collection.objects.link(cam_obj)

    fcurves = ensure_channelbag_fcurves(cam_obj, action_name=f"{name}_Action")

    def _get_or_new_fc(dp: str, idx: int) -> Any:
        for fc in fcurves:
            if fc.data_path == dp and fc.array_index == idx:
                return fc
        return fcurves.new(data_path=dp, index=idx)

    loc_fcs = [_get_or_new_fc("location", i) for i in range(3)]
    rot_dp = "rotation_quaternion" if use_quat else "rotation_euler"
    rot_dim = 4 if use_quat else 3
    rot_fcs = [_get_or_new_fc(rot_dp, i) for i in range(rot_dim)]

    # Hemisphere-align consecutive quaternions (`dot(q_prev, q_cur) >= 0`) so LINEAR F-curves
    # NEVER suffer a 360-degree spin flip when Shepperd's sign flips (`q` vs `-q`)!
    prev_q: list[float] | None = None
    aligned_quats: list[list[float]] = []
    if use_quat:
        for c in cams:
            rq = [float(x) for x in c["rotation_quaternion"]]
            if prev_q is not None:
                dot = prev_q[0] * rq[0] + prev_q[1] * rq[1] + prev_q[2] * rq[2] + prev_q[3] * rq[3]
                if dot < 0.0:
                    rq = [-rq[0], -rq[1], -rq[2], -rq[3]]
            prev_q = rq
            aligned_quats.append(rq)

    for i, c in enumerate(cams):
        frame = int(c.get("frame_index", i)) + 1
        loc = c["location"]
        if i == 0:
            cam_data.lens = float(c.get("lens_mm", 50.0))
            cam_obj.location = (float(loc[0]), float(loc[1]), float(loc[2]))
            if use_quat:
                rq0 = aligned_quats[0]
                cam_obj.rotation_quaternion = Quaternion((rq0[0], rq0[1], rq0[2], rq0[3]))
            else:
                re = c["rotation_euler"]
                cam_obj.rotation_euler = Euler((float(re[0]), float(re[1]), float(re[2])), "XYZ")
        for ax in range(3):
            kp_l = loc_fcs[ax].keyframe_points.insert(frame, float(loc[ax]))
            kp_l.interpolation = "LINEAR"
        rot_vals = aligned_quats[i] if use_quat else c["rotation_euler"]
        for ax in range(rot_dim):
            kp_r = rot_fcs[ax].keyframe_points.insert(frame, float(rot_vals[ax]))
            kp_r.interpolation = "LINEAR"

    # Optionally spawn COLMAP/VGGT static camera frustum markers + trajectory curve in a Collection
    rig_col_name = None
    if spawn_frustum_rig and len(cams) >= 2:
        rig_col_name = f"{name}_Poses"
        old_col = bpy.data.collections.get(rig_col_name)
        if old_col is not None:
            for o in list(old_col.objects):
                bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.collections.remove(old_col)
        rig_col = bpy.data.collections.new(rig_col_name)
        sc.collection.children.link(rig_col)

        # 3D Polyline trajectory connecting all solved camera centers
        cu = bpy.data.curves.new(f"{name}_PathCurve", type="CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.006
        sp = cu.splines.new("POLY")
        sp.points.add(len(cams) - 1)
        for idx_p, c in enumerate(cams):
            lx, ly, lz = c["location"]
            sp.points[idx_p].co = (float(lx), float(ly), float(lz), 1.0)
        path_obj = bpy.data.objects.new(f"{name}_Trajectory", cu)
        rig_col.objects.link(path_obj)

        # Spawn lightweight camera frustums along the arc (capped to ~24 markers for clean viewport)
        step_f = max(1, len(cams) // 24)
        shared_cd = cam_data.copy()
        shared_cd.show_background_images = False
        shared_cd.display_size = 0.16
        for idx_c in range(0, len(cams), step_f):
            c = cams[idx_c]
            fo = bpy.data.objects.new(f"Pose_{idx_c + 1:03d}_{c.get('image_name', '')}", shared_cd)
            fo.location = tuple(float(v) for v in c["location"])
            if use_quat:
                q_i = aligned_quats[idx_c]
                fo.rotation_mode = "QUATERNION"
                fo.rotation_quaternion = Quaternion((q_i[0], q_i[1], q_i[2], q_i[3]))
            else:
                fo.rotation_euler = Euler(tuple(float(v) for v in c["rotation_euler"]), "XYZ")
            fo.hide_render = True
            rig_col.objects.link(fo)

    sc.frame_start = 1
    sc.frame_end = max(sc.frame_end, len(cams))
    sc.camera = cam_obj
    return {
        "imported": True,
        "camera": cam_obj.name,
        "frames": len(cams),
        "pose_backend": doc.get("pose_backend", "vggt"),
        "frustum_collection": rig_col_name,
        "background_gt": bg_attached,
        "resolution": [int(sc.render.resolution_x), int(sc.render.resolution_y)],
    }


@blender_op(
    "splat.import",
    tier="mutate",
    summary=(
        "Import a 3DGS `.ply` into a native Blender 5.3 `PointCloud` with optional Geometry Nodes "
        "anisotropic ellipsoids, `cameras.json` Slotted Action camera, and `Z=0` shadow-catcher ground."
    ),
    params_doc={
        "filepath": "Path to binary little-endian `.ply` file",
        "name": "Object name for imported splat (default PLY stem)",
        "cameras_json": "Optional path to `cameras.json` to animate `SeedanceSourceCam`",
        "render_mode": "`'pointcloud'` (default, ultra-fast) or `'ellipsoids'` (anisotropic GeoNodes 3D ellipsoids)",
        "max_points": "Maximum Gaussians to keep sorted by visual weight (default 250000)",
        "min_opacity": "Opacity pruning threshold `[0..1]` (default 0.015)",
        "point_scale": "Scale multiplier for radii/ellipsoids (default 1.0)",
        "ellipsoid_radius": "Unit-sphere σ multiplier for `ellipsoids` mode (default 1.65)",
        "alpha_mode": "`'DITHERED'` (default, order-independent stochastic) or `'BLENDED'`",
        "unlit": "Use pure Emission shader + `Standard` view transform (True, default) or Principled BSDF (False)",
        "shadow_catcher_ground": "Add a `Z=0` Cycles shadow-catcher ground plane (default True)",
    },
)
def op_splat_import(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    ply_path = Path(str(params["filepath"])).resolve()
    if not ply_path.is_file():
        raise OpError(f"PLY file not found: {ply_path}", code="FILE_NOT_FOUND")

    t0 = time.perf_counter()
    parsed = _parse_gaussian_ply_stdlib(
        ply_path,
        min_opacity=float(params.get("min_opacity", 0.025)),
        max_points=int(params.get("max_points", 250_000)),
        point_scale=float(params.get("point_scale", 1.0)),
        max_world_scale=float(params.get("max_world_scale", 0.28)),
        max_anisotropy=float(params.get("max_anisotropy", 12.0)),
    )
    t_parsed = time.perf_counter()
    n = parsed["count"]
    obj_name = str(params.get("name") or ply_path.stem)
    old_splat = bpy.data.objects.get(obj_name)
    if old_splat is not None and old_splat.type == "POINTCLOUD":
        old_pc_data = old_splat.data
        bpy.data.objects.remove(old_splat, do_unlink=True)
        if old_pc_data and old_pc_data.users == 0:
            bpy.data.pointclouds.remove(old_pc_data)

    raw_mode = str(params.get("render_mode") or "pointcloud").lower()
    if raw_mode in ("gaussian_quads", "ellipsoids_viewport"):
        render_mode = "ellipsoids"
        viewport_fast_pc = False
    elif raw_mode in ("hybrid", "fast_viewport"):
        render_mode = "ellipsoids"
        viewport_fast_pc = True
    elif raw_mode == "ellipsoids":
        render_mode = "ellipsoids"
        viewport_fast_pc = bool(params.get("viewport_fast_pointcloud", False))
    else:
        render_mode = "pointcloud"
        viewport_fast_pc = False
    unlit = bool(params.get("unlit", True))
    alpha_mode = str(params.get("alpha_mode") or "DITHERED").upper()
    ellipsoid_radius = float(params.get("ellipsoid_radius", 2.5))
    use_sh1 = bool(parsed.get("has_sh1")) and bool(params.get("use_sh", True))

    pc = bpy.data.pointclouds.new(name=f"{obj_name}_PC")
    if n > 0:
        pc.resize(n)
        pos_attr = pc.attributes.get("position") or pc.attributes.new("position", "FLOAT_VECTOR", "POINT")
        pos_attr.data.foreach_set("vector", parsed["pos"])

        rad_attr = pc.attributes.get("radius") or pc.attributes.new("radius", "FLOAT", "POINT")
        rad_attr.data.foreach_set("value", parsed["rad"])

        col_attr = pc.color_attributes.new("splat_color", "FLOAT_COLOR", "POINT")
        col_attr.data.foreach_set("color", parsed["col"])
        pc.color_attributes.active = col_attr

        scl_attr = pc.attributes.new("splat_scale", "FLOAT_VECTOR", "POINT")
        scl_attr.data.foreach_set("vector", parsed["scale"])

        rot_attr = pc.attributes.new("splat_rot", "QUATERNION", "POINT")
        rot_attr.data.foreach_set("value", parsed["rot"])

        if use_sh1:
            pc.attributes.new("splat_dc", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", parsed["dc_srgb"])
            pc.attributes.new("splat_sh1_0", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", parsed["sh1_0"])
            pc.attributes.new("splat_sh1_1", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", parsed["sh1_1"])
            pc.attributes.new("splat_sh1_2", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", parsed["sh1_2"])
    t_foreach = time.perf_counter()

    # Hide default startup Cube in fresh scenes so it never sits inside imported splat clouds
    if bool(params.get("hide_default_cube", True)):
        def_cube = bpy.data.objects.get("Cube")
        if def_cube is not None and def_cube.type == "MESH" and len(def_cube.data.vertices) == 8:
            def_cube.hide_viewport = True
            def_cube.hide_render = True

    obj = bpy.data.objects.new(obj_name, pc)
    sc.collection.objects.link(obj)

    use_instancer = render_mode == "ellipsoids"
    mat = _build_splat_material(
        bpy,
        f"{obj_name}_SplatMat",
        instancer=use_instancer,
        unlit=unlit,
        has_sh1=use_sh1,
        alpha_mode=alpha_mode,
        cutoff_sigma=ellipsoid_radius,
    )
    pc.materials.append(mat)
    if use_instancer and n > 0:
        mat_pc = None
        if viewport_fast_pc:
            mat_pc = _build_splat_material(
                bpy,
                f"{obj_name}_ViewportPCMat",
                instancer=False,
                unlit=unlit,
                has_sh1=use_sh1,
                alpha_mode=alpha_mode,
            )
            pc.materials.append(mat_pc)
        _attach_ellipsoid_geonodes(
            bpy,
            obj,
            mat,
            mat_pc=mat_pc,
            ellipsoid_radius=ellipsoid_radius,
            viewport_fast_pointcloud=viewport_fast_pc,
        )

    if unlit:
        if hasattr(sc, "view_settings"):
            sc.view_settings.view_transform = str(params.get("view_transform", "Standard"))
            sc.view_settings.look = "None"
            sc.view_settings.exposure = 0.0
        if bool(params.get("dark_world", True)):
            if sc.world is None:
                sc.world = bpy.data.worlds.new("SplatWorld")
            sc.world.use_nodes = True
            bg = next((nd for nd in sc.world.node_tree.nodes if nd.type == "BACKGROUND"), None)
            if bg is not None:
                bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
                bg.inputs["Strength"].default_value = 0.0

    if hasattr(sc, "cycles") and hasattr(sc.cycles, "transparent_max_bounces"):
        sc.cycles.transparent_max_bounces = max(int(sc.cycles.transparent_max_bounces), 1024)
    if hasattr(sc, "eevee") and hasattr(sc.eevee, "taa_render_samples"):
        sc.eevee.taa_render_samples = max(int(sc.eevee.taa_render_samples), 64)

    cam_info: dict[str, Any] = {"imported": False}
    if params.get("cameras_json"):
        cj = Path(str(params["cameras_json"])).resolve()
        if cj.is_file():
            # Auto-discover GT frame 1 or source Seedance MP4 for camera overlay inspection
            bg_p: Path | None = None
            if params.get("gt_overlay_path"):
                cand = Path(str(params["gt_overlay_path"])).resolve()
                if cand.is_file():
                    bg_p = cand
            if bg_p is None:
                gt1 = cj.parent / "previews" / "gt_frame_00001.jpg"
                if gt1.is_file():
                    bg_p = gt1
            cam_info = _import_cameras_json(
                ctx,
                cj,
                sync_resolution=bool(params.get("sync_resolution", True)),
                spawn_frustum_rig=bool(params.get("spawn_frustum_rig", True)),
                bg_media_path=bg_p,
            )

    # In headed Blender GUI, switch 3D viewports to Material Preview (EEVEE) so 3DGS shaders evaluate!
    if bool(params.get("activate_viewport_shading", True)) and not bpy.app.background:
        try:
            wm = getattr(bpy.context, "window_manager", None)
            for win in (wm.windows if wm else ()):
                for area in win.screen.areas:
                    if area.type != "VIEW_3D":
                        continue
                    for sp in area.spaces:
                        if sp.type != "VIEW_3D":
                            continue
                        if sp.shading.type in ("SOLID", "WIREFRAME"):
                            sp.shading.type = "MATERIAL"
                        sp.shading.use_scene_lights = True
                        sp.shading.use_scene_world = True
                        if cam_info.get("imported"):
                            sp.region_3d.view_perspective = "CAMERA"
        except Exception:  # noqa: BLE001, S110
            pass

    ground_name = None
    if bool(params.get("shadow_catcher_ground", True)):
        bmin, bmax = parsed["bounds_min"], parsed["bounds_max"]
        span = max(8.0, max(bmax[0] - bmin[0], bmax[1] - bmin[1]) * 1.6)
        cx = (bmin[0] + bmax[0]) * 0.5
        cy = (bmin[1] + bmax[1]) * 0.5
        bpy.ops.mesh.primitive_plane_add(size=span, location=(cx, cy, 0.0))
        g_obj = bpy.context.active_object
        if g_obj is not None:
            g_obj.name = f"{obj_name}_ShadowGround"
            if hasattr(g_obj, "is_shadow_catcher"):
                g_obj.is_shadow_catcher = True
            g_mat = _build_shadow_ground_material(bpy, f"{obj_name}_ShadowGroundMat")
            g_obj.data.materials.append(g_mat)
            ground_name = g_obj.name

    return {
        "object": obj.name,
        "render_mode": render_mode,
        "points_imported": n,
        "raw_points": parsed["raw_count"],
        "bounds_min": [round(v, 4) for v in parsed["bounds_min"]],
        "bounds_max": [round(v, 4) for v in parsed["bounds_max"]],
        "cameras": cam_info,
        "shadow_catcher": ground_name,
        "parse_ms": round((t_parsed - t0) * 1000.0, 2),
        "foreach_ms": round((t_foreach - t_parsed) * 1000.0, 2),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


@blender_op(
    "splat.crop_clean",
    tier="mutate",
    summary="Crop/prune floaters on a Blender `PointCloud` splat object by AABB, opacity, or radius.",
    params_doc={
        "object_name": "Name of `POINTCLOUD` object in scene",
        "bounds_min": "Optional `[xmin, ymin, zmin]` crop box",
        "bounds_max": "Optional `[xmax, ymax, zmax]` crop box",
        "min_opacity": "Optional minimum opacity `[0..1]`",
        "max_radius": "Optional maximum point radius in meters",
    },
)
def op_splat_crop_clean(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    bpy = ctx.bpy
    name = str(params["object_name"])
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "POINTCLOUD":
        raise OpError(f"PointCloud object '{name}' not found", code="OBJECT_NOT_FOUND")

    pc = obj.data
    n = len(pc.points)
    if n == 0:
        return {"object": name, "kept": 0, "removed": 0}

    pos = np.empty((n, 3), dtype=np.float32)
    pc.attributes["position"].data.foreach_get("vector", pos.ravel())
    rad = np.zeros(n, dtype=np.float32)
    if "radius" in pc.attributes:
        pc.attributes["radius"].data.foreach_get("value", rad)

    has_col = "splat_color" in pc.attributes
    col = np.empty((n, 4), dtype=np.float32) if has_col else None
    if col is not None:
        pc.attributes["splat_color"].data.foreach_get("color", col.ravel())

    mask = np.ones(n, dtype=bool)
    bmin = params.get("bounds_min")
    bmax = params.get("bounds_max")
    if bmin is not None:
        mask &= (pos[:, 0] >= bmin[0]) & (pos[:, 1] >= bmin[1]) & (pos[:, 2] >= bmin[2])
    if bmax is not None:
        mask &= (pos[:, 0] <= bmax[0]) & (pos[:, 1] <= bmax[1]) & (pos[:, 2] <= bmax[2])
    if params.get("min_opacity") is not None and col is not None:
        mask &= col[:, 3] >= float(params["min_opacity"])
    if params.get("max_radius") is not None:
        mask &= rad <= float(params["max_radius"])

    keep = np.flatnonzero(mask)
    k = int(keep.size)
    if k == n:
        return {"object": name, "kept": n, "removed": 0}

    vec3_attrs: dict[str, Any] = {}
    for aname in ("splat_scale", "splat_dc", "splat_sh1_0", "splat_sh1_1", "splat_sh1_2"):
        if aname in pc.attributes:
            buf = np.empty((n, 3), dtype=np.float32)
            pc.attributes[aname].data.foreach_get("vector", buf.ravel())
            vec3_attrs[aname] = buf[keep].ravel()

    rot_sliced = None
    if "splat_rot" in pc.attributes:
        rbuf = np.empty((n, 4), dtype=np.float32)
        pc.attributes["splat_rot"].data.foreach_get("value", rbuf.ravel())
        rot_sliced = rbuf[keep].ravel()

    new_pc = bpy.data.pointclouds.new(name=f"{name}_CleanPC")
    for m in pc.materials:
        new_pc.materials.append(m)
    if k > 0:
        new_pc.resize(k)
        (new_pc.attributes.get("position") or new_pc.attributes.new("position", "FLOAT_VECTOR", "POINT")).data.foreach_set("vector", pos[keep].ravel())
        (new_pc.attributes.get("radius") or new_pc.attributes.new("radius", "FLOAT", "POINT")).data.foreach_set("value", rad[keep])
        if col is not None:
            new_pc.attributes.new("splat_color", "FLOAT_COLOR", "POINT").data.foreach_set("color", col[keep].ravel())
        for aname, aval in vec3_attrs.items():
            new_pc.attributes.new(aname, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", aval)
        if rot_sliced is not None:
            new_pc.attributes.new("splat_rot", "QUATERNION", "POINT").data.foreach_set("value", rot_sliced)

    obj.data = new_pc
    if pc.users == 0:
        bpy.data.pointclouds.remove(pc)

    return {"object": name, "kept": k, "removed": n - k}


@blender_op(
    "splat.project_video",
    tier="mutate",
    summary=(
        "Camera-project a video/image onto proxy geometry from `projection_camera` with "
        "transparent frustum-border feathering so 3DGS parallax splats fill off-axis views."
    ),
    params_doc={
        "video_path": "Path to `.mp4` or image file",
        "target_object": "Mesh object name to receive camera-projected material",
        "projection_camera": "Optional camera name (defaults to `scene.camera`)",
        "feather": "Soft UV frustum-border blend width `[0..0.25]` into 3DGS background (default 0.05)",
    },
)
def op_splat_project_video(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    vid_path = Path(str(params["video_path"])).resolve()
    if not vid_path.is_file():
        raise OpError(f"Media file not found: {vid_path}", code="FILE_NOT_FOUND")

    obj = bpy.data.objects.get(str(params["target_object"]))
    if obj is None or obj.type != "MESH":
        raise OpError(f"Target mesh '{params['target_object']}' not found", code="OBJECT_NOT_FOUND")

    cam_name = params.get("projection_camera")
    cam = bpy.data.objects.get(str(cam_name)) if cam_name else sc.camera
    if cam is None or cam.type != "CAMERA":
        raise OpError("Projection camera not found", code="NO_CAMERA")

    feather = max(0.0, min(0.25, float(params.get("feather", 0.05))))

    if len(obj.data.vertices) < 256:
        sub = obj.modifiers.new(name="CamProjSubdiv", type="SUBSURF")
        sub.subdivision_type = "SIMPLE"
        sub.levels = 4
        sub.render_levels = 4

    uv_layer = obj.data.uv_layers.active or obj.data.uv_layers.new(name="CamProjUV")
    mod = obj.modifiers.new(name="CamProjUV", type="UV_PROJECT")
    mod.uv_layer = uv_layer.name
    mod.projector_count = 1
    mod.projectors[0].object = cam
    mod.aspect_x = float(sc.render.resolution_x)
    mod.aspect_y = float(sc.render.resolution_y)

    img = bpy.data.images.load(str(vid_path), check_existing=True)
    if vid_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        img.source = "MOVIE"

    mat = bpy.data.materials.new(name=f"{obj.name}_CamProjMat")
    mat.use_nodes = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "CLIP"
    if hasattr(tex, "image_user") and img.source == "MOVIE":
        tex.image_user.use_auto_refresh = True
        tex.image_user.frame_duration = max(1, int(getattr(img, "frame_duration", sc.frame_end) or sc.frame_end))
        tex.image_user.frame_start = 1
        tex.image_user.frame_offset = 0

    uv_node = nt.nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = uv_layer.name

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    front_mask = nt.nodes.new("ShaderNodeMath")
    front_mask.operation = "SUBTRACT"
    front_mask.inputs[0].default_value = 1.0
    nt.links.new(geom.outputs["Backfacing"], front_mask.inputs[1])

    alpha_sock = tex.outputs["Alpha"]
    if feather > 1e-4:
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(uv_node.outputs["UV"], sep.inputs["Vector"])

        one_u = nt.nodes.new("ShaderNodeMath")
        one_u.operation = "SUBTRACT"
        one_u.inputs[0].default_value = 1.0
        nt.links.new(sep.outputs["X"], one_u.inputs[1])

        one_v = nt.nodes.new("ShaderNodeMath")
        one_v.operation = "SUBTRACT"
        one_v.inputs[0].default_value = 1.0
        nt.links.new(sep.outputs["Y"], one_v.inputs[1])

        min_u = nt.nodes.new("ShaderNodeMath")
        min_u.operation = "MINIMUM"
        nt.links.new(sep.outputs["X"], min_u.inputs[0])
        nt.links.new(one_u.outputs["Value"], min_u.inputs[1])

        min_v = nt.nodes.new("ShaderNodeMath")
        min_v.operation = "MINIMUM"
        nt.links.new(sep.outputs["Y"], min_v.inputs[0])
        nt.links.new(one_v.outputs["Value"], min_v.inputs[1])

        min_uv = nt.nodes.new("ShaderNodeMath")
        min_uv.operation = "MINIMUM"
        nt.links.new(min_u.outputs["Value"], min_uv.inputs[0])
        nt.links.new(min_v.outputs["Value"], min_uv.inputs[1])

        div_f = nt.nodes.new("ShaderNodeMath")
        div_f.operation = "DIVIDE"
        div_f.inputs[1].default_value = feather
        div_f.use_clamp = True
        nt.links.new(min_uv.outputs["Value"], div_f.inputs[0])

        mul_clip = nt.nodes.new("ShaderNodeMath")
        mul_clip.operation = "MULTIPLY"
        nt.links.new(tex.outputs["Alpha"], mul_clip.inputs[0])
        nt.links.new(div_f.outputs["Value"], mul_clip.inputs[1])
        alpha_sock = mul_clip.outputs["Value"]

    eff_alpha = nt.nodes.new("ShaderNodeMath")
    eff_alpha.operation = "MULTIPLY"
    nt.links.new(alpha_sock, eff_alpha.inputs[0])
    nt.links.new(front_mask.outputs["Value"], eff_alpha.inputs[1])

    nt.links.new(uv_node.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(em.outputs["Emission"], mix.inputs[2])
    nt.links.new(eff_alpha.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    if hasattr(sc, "view_settings"):
        sc.view_settings.view_transform = "Standard"
        sc.view_settings.look = "None"
        sc.view_settings.exposure = 0.0

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return {"object": obj.name, "camera": cam.name, "image": img.name, "material": mat.name, "feather": feather}


@blender_op(
    "splat.import_cameras",
    tier="mutate",
    summary=(
        "Import solved Seedance / COLMAP / VGGT `cameras.json` into Blender 5.3 as an animated "
        "`SeedanceSourceCam` + `SeedanceSourceCam_Poses` 3D frustum trajectory rig + optional GT/Error camera overlay."
    ),
    params_doc={
        "cameras_json": "Path to `cameras.json` inside `assets/splats/<job_id>/`",
        "name": "Camera name (default `'SeedanceSourceCam'`)",
        "spawn_frustum_rig": "Create `*_Poses` Collection with 3D trajectory curve & pose frustums (default True)",
        "overlay_path": "Optional path to source `.mp4`, `gt_frame_00001.jpg`, or `latest_diff.jpg` for Camera View overlay",
    },
)
def op_splat_import_cameras(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    cj = Path(str(params["cameras_json"])).resolve()
    if not cj.is_file():
        raise OpError(f"cameras.json not found: {cj}", code="FILE_NOT_FOUND")

    bg_p: Path | None = None
    if params.get("overlay_path"):
        cand = Path(str(params["overlay_path"])).resolve()
        if cand.is_file():
            bg_p = cand
    if bg_p is None:
        gt1 = cj.parent / "previews" / "gt_frame_00001.jpg"
        if gt1.is_file():
            bg_p = gt1

    res = _import_cameras_json(
        ctx,
        cj,
        name=str(params.get("name") or "SeedanceSourceCam"),
        sync_resolution=bool(params.get("sync_resolution", True)),
        spawn_frustum_rig=bool(params.get("spawn_frustum_rig", True)),
        bg_media_path=bg_p,
    )

    if not bpy.app.background and res.get("imported"):
        try:
            wm = getattr(bpy.context, "window_manager", None)
            for win in (wm.windows if wm else ()):
                for area in win.screen.areas:
                    if area.type == "VIEW_3D":
                        for sp in area.spaces:
                            if sp.type == "VIEW_3D":
                                sp.region_3d.view_perspective = "CAMERA"
        except Exception:  # noqa: BLE001, S110
            pass
    return res