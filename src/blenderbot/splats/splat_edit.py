"""Fast pure-NumPy 3DGS `.ply` <-> `.splat` I/O, Default Cube Benchmark Splat, and Surgical Editor.

Enables interactive Minecraft-style 3D Gaussian Splat viewing, brush culling, 3D AABB cropping,
smart floater filtering, recoloring/scaling, undo-safe `.orig.ply` backups, and lossless SH-preserving
round-trip saves back to `splat.ply` + `splat.splat` + live Blender 5.3.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

_SH_C0 = 0.28209479177387814


def parse_ply_gaussians(raw: bytes) -> dict[str, Any]:
    """Parse a binary_little_endian 3DGS `.ply` into mutable float32 `(N, P)` array + property map."""
    end_marker = b"end_header\n"
    idx = raw.find(end_marker)
    if idx < 0:
        end_marker = b"end_header\r\n"
        idx = raw.find(end_marker)
    if idx < 0:
        raise ValueError("Invalid binary 3DGS PLY: missing end_header")

    hdr = raw[:idx].decode("ascii", errors="ignore")
    if "binary_little_endian" not in hdr:
        raise ValueError("Only binary_little_endian 3DGS PLY files are supported")

    data_offset = idx + len(end_marker)
    n_verts = 0
    props: list[str] = []
    in_vertex = False
    for ln in hdr.splitlines():
        parts = ln.strip().split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                n_verts = int(parts[2])
        elif in_vertex and parts[0] == "property":
            props.append(parts[-1])

    if n_verts <= 0 or not props:
        raise ValueError("PLY contains zero vertices or properties")

    stride_f = len(props)
    expected_bytes = n_verts * stride_f * 4
    if len(raw) - data_offset < expected_bytes:
        n_verts = max(0, (len(raw) - data_offset) // (stride_f * 4))

    arr = (
        np.frombuffer(raw, dtype="<f4", count=n_verts * stride_f, offset=data_offset)
        .reshape(n_verts, stride_f)
        .astype(np.float32, copy=True)
    )
    pmap = {name: i for i, name in enumerate(props)}
    for req in ("x", "y", "z"):
        if req not in pmap:
            raise ValueError(f"PLY missing required vertex property '{req}'")
    return {"props": props, "pmap": pmap, "data": arr}


def encode_ply_gaussians(props: list[str], data: np.ndarray) -> bytes:
    """Serialize `(N, P)` float32 Gaussian attributes back to binary_little_endian `.ply`."""
    arr = np.ascontiguousarray(data, dtype="<f4")
    n = int(arr.shape[0])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "\n".join(f"property float {p}" for p in props)
        + "\nend_header\n"
    ).encode("ascii")
    return header + arr.tobytes()


def ply_data_to_splat_bytes(
    props: list[str],
    data: np.ndarray,
    *,
    max_gaussians: int = 0,
) -> bytes:
    """Serialize `(N, P)` PLY data to 32-byte `.splat` in exact row order (`i_splat == i_ply`)."""
    pmap = {name: i for i, name in enumerate(props)}
    arr = data[:max_gaussians] if (max_gaussians > 0 and len(data) > max_gaussians) else data
    n = len(arr)
    if n == 0:
        return b""

    means = np.stack([arr[:, pmap["x"]], arr[:, pmap["y"]], arr[:, pmap["z"]]], axis=-1).astype("<f4")

    if all(k in pmap for k in ("scale_0", "scale_1", "scale_2")):
        s_log = np.clip(
            np.stack([arr[:, pmap["scale_0"]], arr[:, pmap["scale_1"]], arr[:, pmap["scale_2"]]], axis=-1),
            -20.0,
            5.0,
        )
        scales_lin = np.exp(s_log).astype("<f4")
    else:
        scales_lin = np.full((n, 3), 0.025, dtype="<f4")

    if all(k in pmap for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh0 = np.stack([arr[:, pmap["f_dc_0"]], arr[:, pmap["f_dc_1"]], arr[:, pmap["f_dc_2"]]], axis=-1)
        rgb = np.clip(0.5 + _SH_C0 * sh0, 0.0, 1.0)
    else:
        rgb = np.full((n, 3), 0.75, dtype=np.float32)

    if "opacity" in pmap:
        op_logit = np.clip(arr[:, pmap["opacity"]], -20.0, 20.0)
        opa = 1.0 / (1.0 + np.exp(-op_logit))
    else:
        opa = np.ones(n, dtype=np.float32)

    if all(k in pmap for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
        quats = np.stack(
            [arr[:, pmap["rot_0"]], arr[:, pmap["rot_1"]], arr[:, pmap["rot_2"]], arr[:, pmap["rot_3"]]],
            axis=-1,
        )
        q_norm = np.linalg.norm(quats, axis=-1, keepdims=True)
        quats = np.where(q_norm > 1e-8, quats / np.maximum(q_norm, 1e-8), np.array([1, 0, 0, 0], dtype=np.float32))
    else:
        quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))

    rgba_u8 = np.clip(np.concatenate([rgb, opa[:, None]], axis=1) * 255.0, 0.0, 255.0).astype(np.uint8)
    quat_u8 = np.clip((quats + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)

    rec = np.empty(
        n,
        dtype=[("pos", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rot", "u1", (4,))],
    )
    rec["pos"] = means
    rec["scale"] = scales_lin
    rec["rgba"] = rgba_u8
    rec["rot"] = quat_u8
    return rec.tobytes()


def splat_bytes_to_ply_bytes(splat_bytes: bytes) -> tuple[bytes, int]:
    """Convert 32-byte `.splat` binary buffer into a standard 3DGS binary `.ply`."""
    n = len(splat_bytes) // 32
    if n <= 0:
        raise ValueError("Empty .splat buffer")
    rec = np.frombuffer(
        splat_bytes,
        dtype=[("pos", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rot", "u1", (4,))],
        count=n,
    )
    means = rec["pos"].astype(np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)
    rgb = rec["rgba"][:, :3].astype(np.float32) / 255.0
    sh0 = (rgb - 0.5) / _SH_C0
    alpha = np.clip(rec["rgba"][:, 3].astype(np.float32) / 255.0, 1e-4, 1.0 - 1e-4)
    op_logit = np.log(alpha / (1.0 - alpha))[:, None].astype(np.float32)
    scales_log = np.log(np.maximum(rec["scale"].astype(np.float32), 1e-6))
    quats = (rec["rot"].astype(np.float32) - 127.5) / 127.5
    qn = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = np.where(qn > 1e-8, quats / np.maximum(qn, 1e-8), np.array([1, 0, 0, 0], dtype=np.float32))

    props = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    data = np.concatenate([means, normals, sh0, op_logit, scales_log, quats], axis=1).astype(np.float32)
    return encode_ply_gaussians(props, data), n


def ensure_aligned_splat_file(ply_path: Path, splat_path: Path, max_gaussians: int = 350_000) -> dict[str, Any]:
    """Ensure `splat.splat` exists and has 1:1 vertex index alignment with `splat.ply`."""
    parsed = parse_ply_gaussians(ply_path.read_bytes())
    props, pmap, data = parsed["props"], parsed["pmap"], parsed["data"]
    n_ply = len(data)
    target_n = min(n_ply, max_gaussians) if max_gaussians > 0 else n_ply

    need_rebuild = not splat_path.is_file()
    if not need_rebuild:
        raw_sp = splat_path.read_bytes()
        n_sp = len(raw_sp) // 32
        if n_sp != target_n:
            need_rebuild = True
        elif n_sp > 0:
            check_k = min(32, n_sp)
            sp_rec = np.frombuffer(
                raw_sp,
                dtype=[("pos", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rot", "u1", (4,))],
                count=check_k,
            )
            ply_xyz = np.stack([data[:check_k, pmap["x"]], data[:check_k, pmap["y"]], data[:check_k, pmap["z"]]], axis=-1)
            if not np.allclose(sp_rec["pos"], ply_xyz, atol=1e-4):
                need_rebuild = True

    if need_rebuild:
        splat_bytes = ply_data_to_splat_bytes(props, data, max_gaussians=max_gaussians)
        splat_path.write_bytes(splat_bytes)

    return {
        "n_ply": n_ply,
        "n_splat": target_n,
        "rebuilt": need_rebuild,
        "has_orig": (ply_path.parent / "splat.orig.ply").is_file(),
    }


def build_default_cube_splat(
    variant: str = "courtyard",
    seed: int = 42,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Build a metric-calibrated Default Cube (`2m x 2m x 2m` on `Z=0`) + Courtyard 3DGS benchmark."""
    rng = np.random.default_rng(seed)
    pos_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    opa_parts: list[np.ndarray] = []

    def _add_box_shell(
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        base_rgb: tuple[float, float, float],
        edge_rgb: tuple[float, float, float] | None = None,
        step: float = 0.085,
        skip_bottom: bool = False,
    ) -> None:
        cx, cy, cz = center
        sx, sy, sz = size
        hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
        u_x = np.linspace(-hx + step * 0.4, hx - step * 0.4, max(2, round(sx / step)))
        u_y = np.linspace(-hy + step * 0.4, hy - step * 0.4, max(2, round(sy / step)))
        u_z = np.linspace(-hz + step * 0.4, hz - step * 0.4, max(2, round(sz / step)))

        faces = [
            # (axis_fixed, sign, grid_a, grid_b, shade_mult, scales_xyz)
            (0, +1.0, u_y, u_z, 0.92, (0.008, step * 0.68, step * 0.68)),
            (0, -1.0, u_y, u_z, 0.72, (0.008, step * 0.68, step * 0.68)),
            (1, +1.0, u_x, u_z, 0.80, (step * 0.68, 0.008, step * 0.68)),
            (1, -1.0, u_x, u_z, 0.96, (step * 0.68, 0.008, step * 0.68)),
            (2, +1.0, u_x, u_y, 1.06, (step * 0.68, step * 0.68, 0.008)),
        ]
        if not skip_bottom:
            faces.append((2, -1.0, u_x, u_y, 0.58, (step * 0.68, step * 0.68, 0.008)))

        for axis, sgn, ga, gb, shade, sc_xyz in faces:
            aa, bb = np.meshgrid(ga, gb, indexing="ij")
            aa_f, bb_f = aa.ravel(), bb.ravel()
            m = len(aa_f)
            pts = np.zeros((m, 3), dtype=np.float32)
            if axis == 0:
                pts[:, 0] = cx + sgn * hx
                pts[:, 1] = cy + aa_f
                pts[:, 2] = cz + bb_f
                near_edge = (np.abs(aa_f) > hy - step * 1.15) | (np.abs(bb_f) > hz - step * 1.15)
            elif axis == 1:
                pts[:, 0] = cx + aa_f
                pts[:, 1] = cy + sgn * hy
                pts[:, 2] = cz + bb_f
                near_edge = (np.abs(aa_f) > hx - step * 1.15) | (np.abs(bb_f) > hz - step * 1.15)
            else:
                pts[:, 0] = cx + aa_f
                pts[:, 1] = cy + bb_f
                pts[:, 2] = cz + sgn * hz
                near_edge = (np.abs(aa_f) > hx - step * 1.15) | (np.abs(bb_f) > hy - step * 1.15)

            col = np.tile(np.clip(np.array(base_rgb, dtype=np.float32) * shade, 0.0, 1.0), (m, 1))
            if edge_rgb is not None:
                col[near_edge] = np.array(edge_rgb, dtype=np.float32)

            pos_parts.append(pts)
            rgb_parts.append(col)
            scale_parts.append(np.tile(np.array(sc_xyz, dtype=np.float32), (m, 1)))
            opa_parts.append(np.full(m, 3.4, dtype=np.float32))

    # 1. Centerpiece Blender Default Cube (`2m x 2m x 2m` resting cleanly on `Z = 0`)
    _add_box_shell(
        center=(0.0, 0.0, 1.0),
        size=(2.0, 2.0, 2.0),
        base_rgb=(0.76, 0.79, 0.84),
        edge_rgb=(1.0, 0.48, 0.16),
        step=0.075,
    )

    # 2. Z=0 Metric Checkerboard Floor Pad
    tile_ext = 6.0 if variant == "courtyard" else 3.0
    f_step = 0.22 if variant == "courtyard" else 0.16
    fx = np.arange(-tile_ext + 0.5 * f_step, tile_ext, f_step, dtype=np.float32)
    gx, gy = np.meshgrid(fx, fx, indexing="ij")
    gx_f, gy_f = gx.ravel(), gy.ravel()
    # Skip floor directly underneath the solid cube footprint
    vis_floor = (np.abs(gx_f) > 0.96) | (np.abs(gy_f) > 0.96)
    gx_f, gy_f = gx_f[vis_floor], gy_f[vis_floor]
    nf = len(gx_f)
    f_pts = np.stack([gx_f, gy_f, np.zeros(nf, dtype=np.float32)], axis=-1)
    checker = ((np.floor(gx_f) + np.floor(gy_f)).astype(np.int32) & 1) == 0
    f_col = np.where(
        checker[:, None],
        np.array([[0.22, 0.26, 0.34]], dtype=np.float32),
        np.array([[0.14, 0.17, 0.23]], dtype=np.float32),
    )
    # Subtle X/Y axis lines on floor
    f_col[np.abs(gy_f) < 0.12] = np.array([0.85, 0.25, 0.28], dtype=np.float32)
    f_col[np.abs(gx_f) < 0.12] = np.array([0.24, 0.80, 0.42], dtype=np.float32)
    pos_parts.append(f_pts)
    rgb_parts.append(f_col)
    scale_parts.append(np.tile(np.array([[f_step * 0.72, f_step * 0.72, 0.007]], dtype=np.float32), (nf, 1)))
    opa_parts.append(np.full(nf, 3.1, dtype=np.float32))

    if variant == "courtyard":
        # 4 Parallax Courtyard Pillars with 1.65m eye-level cyan band
        for px in (-3.8, 3.8):
            for py in (-3.8, 3.8):
                _add_box_shell(
                    center=(px, py, 1.3),
                    size=(0.65, 0.65, 2.6),
                    base_rgb=(0.38, 0.46, 0.58),
                    edge_rgb=(0.22, 0.74, 0.96),
                    step=0.12,
                    skip_bottom=True,
                )
        # Overhead Background Archway Lintel at Y = +3.8
        _add_box_shell(
            center=(0.0, 3.8, 2.85),
            size=(8.25, 0.65, 0.50),
            base_rgb=(0.44, 0.50, 0.62),
            edge_rgb=(1.0, 0.62, 0.22),
            step=0.14,
        )

    # Deliberate "Floater Cluster" hovering at `(2.3, -1.7, 1.85)` so users can test Brush Erase & Smart Filter!
    n_floaters = 160
    fl_pts = rng.normal(loc=[2.3, -1.7, 1.85], scale=[0.42, 0.42, 0.36], size=(n_floaters, 3)).astype(np.float32)
    fl_col = np.clip(
        np.array([[0.92, 0.52, 0.86]], dtype=np.float32) + rng.normal(0.0, 0.08, (n_floaters, 3)).astype(np.float32),
        0.0,
        1.0,
    )
    # Mix of wispy low-opacity puffs and spiky high-anisotropy needles
    fl_sc = np.full((n_floaters, 3), 0.055, dtype=np.float32)
    fl_sc[:60, 0] = 0.28  # spiky needles!
    fl_sc[:60, 1] = 0.012
    fl_opa = np.concatenate([
        np.full(60, -0.4, dtype=np.float32),   # sigmoid ~ 0.40 needles
        np.full(100, -1.9, dtype=np.float32),  # sigmoid ~ 0.13 wispy mist floaters
    ])
    pos_parts.append(fl_pts)
    rgb_parts.append(fl_col)
    scale_parts.append(fl_sc)
    opa_parts.append(fl_opa)

    means = np.concatenate(pos_parts, axis=0)
    rgb = np.concatenate(rgb_parts, axis=0)
    scales_lin = np.concatenate(scale_parts, axis=0)
    op_logit = np.concatenate(opa_parts, axis=0)[:, None]
    n_total = len(means)

    normals = np.zeros((n_total, 3), dtype=np.float32)
    sh0 = ((rgb - 0.5) / _SH_C0).astype(np.float32)
    scales_log = np.log(np.maximum(scales_lin, 1e-5)).astype(np.float32)
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n_total, 1))

    props = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    data = np.concatenate([means, normals, sh0, op_logit, scales_log, quats], axis=1).astype(np.float32)
    ply_b = encode_ply_gaussians(props, data)
    splat_b = ply_data_to_splat_bytes(props, data)
    return ply_b, splat_b, {"variant": variant, "n_gaussians": n_total, "floaters_seeded": n_floaters}


def _points_in_stroke_spheres(xyz: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    """Fast vectorized union-of-spheres membership mask for continuous 3D brush strokes."""
    r2 = float(radius) * float(radius)
    if len(centers) == 0:
        return np.zeros(len(xyz), dtype=bool)
    # Broadphase AABB reject first so 350k-point strokes run in <3ms
    bmin = centers.min(axis=0) - radius
    bmax = centers.max(axis=0) + radius
    in_box = (
        (xyz[:, 0] >= bmin[0]) & (xyz[:, 0] <= bmax[0])
        & (xyz[:, 1] >= bmin[1]) & (xyz[:, 1] <= bmax[1])
        & (xyz[:, 2] >= bmin[2]) & (xyz[:, 2] <= bmax[2])
    )
    cand_idx = np.flatnonzero(in_box)
    if len(cand_idx) == 0:
        return in_box
    sub = xyz[cand_idx]
    hit_sub = np.zeros(len(cand_idx), dtype=bool)
    # Chunk centers by 32 to bound peak memory
    for start in range(0, len(centers), 32):
        chunk = centers[start : start + 32]  # (K, 3)
        diff = sub[:, None, :] - chunk[None, :, :]  # (M, K, 3)
        d2 = np.einsum("mkd,mkd->mk", diff, diff)
        hit_sub |= (d2 <= r2).any(axis=1)
    out = np.zeros(len(xyz), dtype=bool)
    out[cand_idx] = hit_sub
    return out


def apply_splat_edits(
    ply_path: Path,
    splat_path: Path,
    edits: dict[str, Any],
    *,
    out_ply_path: Path | None = None,
    out_splat_path: Path | None = None,
    backup_orig: bool = True,
    max_splat_gaussians: int = 350_000,
) -> dict[str, Any]:
    """Apply culling masks, 3D brush strokes, AABB crops, filters & color/scale edits to `.ply` + `.splat`."""
    parsed = parse_ply_gaussians(ply_path.read_bytes())
    props: list[str] = parsed["props"]
    pmap: dict[str, int] = parsed["pmap"]
    data: np.ndarray = parsed["data"]
    n_before = len(data)

    dst_ply = out_ply_path or ply_path
    dst_splat = out_splat_path or splat_path

    if backup_orig and dst_ply == ply_path:
        orig_backup = ply_path.parent / "splat.orig.ply"
        if not orig_backup.is_file():
            shutil.copy2(ply_path, orig_backup)

    keep = np.ones(n_before, dtype=bool)
    ix, iy, iz = pmap["x"], pmap["y"], pmap["z"]
    has_dc = all(k in pmap for k in ("f_dc_0", "f_dc_1", "f_dc_2"))
    has_sc = all(k in pmap for k in ("scale_0", "scale_1", "scale_2"))
    has_op = "opacity" in pmap
    rest_cols = [idx for name, idx in pmap.items() if name.startswith("f_rest_")]

    # 1. Replay structured geometric / paint operations (`ops`) across the full `.ply`
    ops_list = list(edits.get("ops") or [])
    for op in ops_list:
        kind = str(op.get("op") or op.get("type") or "").strip().lower()
        xyz = np.stack([data[:, ix], data[:, iy], data[:, iz]], axis=-1)

        if kind in ("sphere_cull", "brush_cull"):
            pts_raw = op.get("points") or ([op["center"]] if op.get("center") else [])
            centers = np.asarray(pts_raw, dtype=np.float32).reshape(-1, 3)
            rad = max(0.005, float(op.get("radius") or 0.25))
            keep &= ~_points_in_stroke_spheres(xyz, centers, rad)

        elif kind in ("sphere_restore", "brush_restore"):
            pts_raw = op.get("points") or ([op["center"]] if op.get("center") else [])
            centers = np.asarray(pts_raw, dtype=np.float32).reshape(-1, 3)
            rad = max(0.005, float(op.get("radius") or 0.25))
            keep |= _points_in_stroke_spheres(xyz, centers, rad)

        elif kind == "sphere_keep":
            center = np.asarray(op["center"], dtype=np.float32).reshape(1, 3)
            rad = max(0.01, float(op.get("radius") or 2.0))
            keep &= _points_in_stroke_spheres(xyz, center, rad)

        elif kind == "aabb_cull":
            bmin = np.asarray(op["bounds_min"], dtype=np.float32)
            bmax = np.asarray(op["bounds_max"], dtype=np.float32)
            lo, hi = np.minimum(bmin, bmax), np.maximum(bmin, bmax)
            inside = (
                (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0])
                & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
                & (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2])
            )
            keep &= ~inside

        elif kind == "aabb_keep":
            bmin = np.asarray(op["bounds_min"], dtype=np.float32)
            bmax = np.asarray(op["bounds_max"], dtype=np.float32)
            lo, hi = np.minimum(bmin, bmax), np.maximum(bmin, bmax)
            inside = (
                (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0])
                & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
                & (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2])
            )
            keep &= inside

        elif kind == "filter_prune":
            if op.get("min_opacity") is not None and has_op:
                min_op = float(op["min_opacity"])
                opa_lin = 1.0 / (1.0 + np.exp(-np.clip(data[:, pmap["opacity"]], -20.0, 20.0)))
                keep &= opa_lin >= min_op
            if op.get("max_world_scale") is not None and has_sc:
                max_ws = float(op["max_world_scale"])
                sc_lin = np.exp(
                    np.clip(
                        np.stack([data[:, pmap["scale_0"]], data[:, pmap["scale_1"]], data[:, pmap["scale_2"]]], axis=-1),
                        -20.0,
                        5.0,
                    )
                )
                keep &= sc_lin.max(axis=-1) <= max_ws
            if op.get("max_anisotropy") is not None and has_sc:
                max_an = float(op["max_anisotropy"])
                sc_lin = np.sort(
                    np.exp(
                        np.clip(
                            np.stack([data[:, pmap["scale_0"]], data[:, pmap["scale_1"]], data[:, pmap["scale_2"]]], axis=-1),
                            -20.0,
                            5.0,
                        )
                    ),
                    axis=-1,
                )
                ratio = sc_lin[:, 2] / np.maximum(sc_lin[:, 1], 1e-6)
                keep &= ratio <= max_an
            if op.get("min_z") is not None:
                keep &= xyz[:, 2] >= float(op["min_z"])
            if op.get("max_z") is not None:
                keep &= xyz[:, 2] <= float(op["max_z"])
            if op.get("max_radius") is not None:
                c_ref = np.median(xyz[keep], axis=0) if keep.any() else np.zeros(3, dtype=np.float32)
                keep &= np.linalg.norm(xyz - c_ref[None, :], axis=1) <= float(op["max_radius"])

        elif kind in ("sphere_paint", "brush_paint", "box_transform"):
            if kind == "box_transform":
                bmin = np.asarray(op["bounds_min"], dtype=np.float32)
                bmax = np.asarray(op["bounds_max"], dtype=np.float32)
                lo, hi = np.minimum(bmin, bmax), np.maximum(bmin, bmax)
                hit = keep & (
                    (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0])
                    & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
                    & (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2])
                )
            else:
                pts_raw = op.get("points") or ([op["center"]] if op.get("center") else [])
                centers = np.asarray(pts_raw, dtype=np.float32).reshape(-1, 3)
                rad = max(0.005, float(op.get("radius") or 0.25))
                hit = keep & _points_in_stroke_spheres(xyz, centers, rad)

            if hit.any():
                if op.get("color_rgb") is not None and has_dc:
                    target_rgb = np.clip(np.asarray(op["color_rgb"], dtype=np.float32)[:3], 0.0, 1.0)
                    blend = max(0.0, min(1.0, float(op.get("blend", 0.65))))
                    cur_sh0 = np.stack(
                        [data[hit, pmap["f_dc_0"]], data[hit, pmap["f_dc_1"]], data[hit, pmap["f_dc_2"]]],
                        axis=-1,
                    )
                    cur_rgb = np.clip(0.5 + _SH_C0 * cur_sh0, 0.0, 1.0)
                    new_rgb = cur_rgb * (1.0 - blend) + target_rgb[None, :] * blend
                    new_sh0 = (new_rgb - 0.5) / _SH_C0
                    data[hit, pmap["f_dc_0"]] = new_sh0[:, 0]
                    data[hit, pmap["f_dc_1"]] = new_sh0[:, 1]
                    data[hit, pmap["f_dc_2"]] = new_sh0[:, 2]
                    if rest_cols:
                        # Dampen higher-order SH view lobes proportionally to recolor blend
                        data[np.ix_(hit, rest_cols)] *= (1.0 - 0.75 * blend)

                if op.get("scale_mult") is not None and has_sc:
                    sm = max(0.05, min(10.0, float(op["scale_mult"])))
                    dl = float(np.log(sm))
                    for sk in ("scale_0", "scale_1", "scale_2"):
                        data[hit, pmap[sk]] = np.clip(data[hit, pmap[sk]] + dl, -18.0, 4.0)

                if op.get("opacity_mult") is not None and has_op:
                    om = max(0.02, min(10.0, float(op["opacity_mult"])))
                    cur_op = 1.0 / (1.0 + np.exp(-np.clip(data[hit, pmap["opacity"]], -20.0, 20.0)))
                    nxt_op = np.clip(cur_op * om, 1e-4, 1.0 - 1e-4)
                    data[hit, pmap["opacity"]] = np.log(nxt_op / (1.0 - nxt_op))

                if op.get("translate") is not None:
                    tr = np.asarray(op["translate"], dtype=np.float32)[:3]
                    data[hit, ix] += tr[0]
                    data[hit, iy] += tr[1]
                    data[hit, iz] += tr[2]

    # 2. Apply exact final WebGL culled-index mask if provided (guarantees 100% Undo/Redo fidelity!)
    raw_drop = edits.get("drop_indices")
    if isinstance(raw_drop, list) and len(raw_drop) > 0:
        # When `ops` were NOT used or to reconcile exact undo/redo final state over the loaded prefix:
        if edits.get("authoritative_mask", False):
            loaded_cap = int(edits.get("loaded_count") or n_before)
            if loaded_cap <= n_before:
                keep[:loaded_cap] = True
        drop_arr = np.asarray(raw_drop, dtype=np.int64)
        valid_drop = drop_arr[(drop_arr >= 0) & (drop_arr < n_before)]
        keep[valid_drop] = False

    kept_data = data[keep]
    n_after = len(kept_data)
    if n_after == 0:
        raise ValueError("Edit would remove 100% of Gaussians — keeping file unchanged (Undo first).")

    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    dst_ply.write_bytes(encode_ply_gaussians(props, kept_data))
    dst_splat.parent.mkdir(parents=True, exist_ok=True)
    dst_splat.write_bytes(ply_data_to_splat_bytes(props, kept_data, max_gaussians=max_splat_gaussians))

    return {
        "n_before": n_before,
        "n_after": n_after,
        "removed": n_before - n_after,
        "has_orig": (dst_ply.parent / "splat.orig.ply").is_file(),
    }


def restore_original_ply(job_dir: Path, max_splat_gaussians: int = 350_000) -> dict[str, Any]:
    """Restore `splat.orig.ply` -> `splat.ply` and rebuild aligned `splat.splat`."""
    orig_p = job_dir / "splat.orig.ply"
    ply_p = job_dir / "splat.ply"
    splat_p = job_dir / "splat.splat"
    if not orig_p.is_file():
        raise FileNotFoundError(f"No backup splat.orig.ply found in {job_dir.name}")
    shutil.copy2(orig_p, ply_p)
    parsed = parse_ply_gaussians(ply_p.read_bytes())
    splat_p.write_bytes(
        ply_data_to_splat_bytes(parsed["props"], parsed["data"], max_gaussians=max_splat_gaussians)
    )
    return {"restored_gaussians": len(parsed["data"]), "has_orig": True}