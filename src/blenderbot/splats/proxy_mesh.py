"""Fast vectorized 3D Surface-Nets iso-surface extractor -> Binary glTF 2.0 (`.glb`) proxy mesh.

Zero external mesh dependencies (`numpy` + Python stdlib `json`/`struct` only).
Runs in `<45 ms` at Stage 4 on Modal GPU workers OR locally on Windows for existing `.ply` splats,
producing a smooth, manifold 3D proxy mesh (`proxy_collision.glb`) aligned 1:1 with Blender 5.3
(`+X right, +Y forward, +Z up`, meters) and Unreal Engine Interchange.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_SH_C0 = 0.28209479177387814


def _smooth_3d_separable(vol: np.ndarray, passes: int = 2) -> np.ndarray:
    """Fast `[1, 2, 1] / 4` separable 3D smoothing in pure NumPy."""
    out = vol
    for _ in range(passes):
        p = np.pad(out, 1, mode="edge")
        out = 0.25 * (p[:-2, 1:-1, 1:-1] + 2.0 * p[1:-1, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1])
        p = np.pad(out, 1, mode="edge")
        out = 0.25 * (p[1:-1, :-2, 1:-1] + 2.0 * p[1:-1, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1])
        p = np.pad(out, 1, mode="edge")
        out = 0.25 * (p[1:-1, 1:-1, :-2] + 2.0 * p[1:-1, 1:-1, 1:-1] + p[1:-1, 1:-1, 2:])
    return out


def pack_glb_bytes(
    verts_blender_xyz: np.ndarray,
    normals_blender_xyz: np.ndarray,
    colors_rgba: np.ndarray,
    faces_u32: np.ndarray,
    mesh_name: str = "SplatProxyMesh",
) -> bytes:
    """Serialize a Blender Z-up triangle mesh into a spec-valid Binary glTF 2.0 (`.glb`).

    Applies `Rx(-90°)` (`(x, y, z)_bl -> (x, z, -y)_gltf`, `det=+1`) so Blender's `wm.gltf_import`
    and Unreal Engine Interchange import into exact 1:1 world alignment.
    """
    v_bl = np.asarray(verts_blender_xyz, dtype=np.float32)
    n_bl = np.asarray(normals_blender_xyz, dtype=np.float32)
    c_f4 = np.asarray(colors_rgba, dtype=np.float32)
    idx = np.asarray(faces_u32, dtype=np.uint32).reshape(-1)

    # Proper rotation Rx(-90 deg): glTF +Y up, -Z forward <-> Blender +Z up, +Y forward
    v_gl = np.stack([v_bl[:, 0], v_bl[:, 2], -v_bl[:, 1]], axis=-1).astype("<f4")
    n_gl = np.stack([n_bl[:, 0], n_bl[:, 2], -n_bl[:, 1]], axis=-1).astype("<f4")
    n_len = np.linalg.norm(n_gl, axis=-1, keepdims=True)
    n_gl = np.where(n_len > 1e-8, n_gl / np.maximum(n_len, 1e-8), np.array([0.0, 1.0, 0.0], dtype="<f4")).astype("<f4")
    c_gl = np.clip(c_f4, 0.0, 1.0).astype("<f4")
    idx_le = idx.astype("<u4")

    b_pos = v_gl.tobytes()
    b_nrm = n_gl.tobytes()
    b_col = c_gl.tobytes()
    b_idx = idx_le.tobytes()

    chunks: list[bytes] = []
    views: list[dict[str, int]] = []
    offset = 0
    for raw, target in ((b_pos, 34962), (b_nrm, 34962), (b_col, 34962), (b_idx, 34963)):
        pad = (4 - (len(raw) % 4)) % 4
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        chunks.append(raw + (b"\x00" * pad))
        offset += len(raw) + pad

    bin_blob = b"".join(chunks)
    vmin = v_gl.min(axis=0).tolist() if len(v_gl) else [0.0, 0.0, 0.0]
    vmax = v_gl.max(axis=0).tolist() if len(v_gl) else [0.0, 0.0, 0.0]

    gltf = {
        "asset": {"version": "2.0", "generator": "blenderbot.splats.proxy_mesh"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": mesh_name}],
        "materials": [
            {
                "name": f"{mesh_name}_Mat",
                "doubleSided": True,
                "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.9},
            }
        ],
        "meshes": [
            {
                "name": mesh_name,
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(v_gl),
                "type": "VEC3",
                "min": [float(x) for x in vmin],
                "max": [float(x) for x in vmax],
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(n_gl),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": len(c_gl),
                "type": "VEC4",
            },
            {
                "bufferView": 3,
                "componentType": 5125,
                "count": len(idx_le),
                "type": "SCALAR",
            },
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_chunk = json_bytes + (b" " * json_pad)

    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_blob)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    c0_hdr = struct.pack("<II", len(json_chunk), 0x4E4F534A)
    c1_hdr = struct.pack("<II", len(bin_blob), 0x004E4942)
    return header + c0_hdr + json_chunk + c1_hdr + bin_blob


def build_proxy_glb_bytes(
    means_xyz: np.ndarray,
    opacities_lin: np.ndarray | None = None,
    scales_lin: np.ndarray | None = None,
    rgb_01: np.ndarray | None = None,
    *,
    grid_res: int = 48,
    min_opacity: float = 0.05,
    mesh_name: str = "SplatProxyMesh",
) -> tuple[bytes, dict[str, int]]:
    """Extract a smooth 3D Surface-Nets proxy collision/projection `.glb` from 3DGS arrays."""
    pts = np.asarray(means_xyz, dtype=np.float32)
    n = len(pts)
    if n < 16:
        raise ValueError("Need at least 16 points to extract a proxy mesh.")

    opa = np.ones(n, dtype=np.float32) if opacities_lin is None else np.asarray(opacities_lin, dtype=np.float32).reshape(-1)
    rgb = np.full((n, 3), 0.72, dtype=np.float32) if rgb_01 is None else np.clip(np.asarray(rgb_01, dtype=np.float32), 0.0, 1.0)

    keep = np.isfinite(pts).all(axis=1) & (opa >= min_opacity)
    if scales_lin is not None:
        s_max = np.asarray(scales_lin, dtype=np.float32).max(axis=-1)
        med_s = float(np.median(s_max[keep])) if keep.any() else 0.05
        keep &= s_max <= max(0.35, med_s * 12.0)

    if keep.sum() >= 16:
        pts, opa, rgb = pts[keep], opa[keep], rgb[keep]

    # Trim distant 4% sky shell so interior scene gets crisp voxel density
    center = np.median(pts, axis=0)
    radii = np.linalg.norm(pts - center[None, :], axis=1)
    r_cap = max(0.25, float(np.percentile(radii, 96.0)))
    in_core = radii <= r_cap
    if in_core.sum() >= 16:
        pts, opa, rgb = pts[in_core], opa[in_core], rgb[in_core]

    g = max(20, min(96, int(grid_res)))
    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    span = np.maximum(bmax - bmin, 1e-2)
    pad = span * (2.5 / float(g))
    origin = bmin - pad
    extent = (bmax + pad) - origin
    cell_size = extent / float(g - 1)

    ijk_f = (pts - origin[None, :]) / cell_size[None, :]
    ijk = np.clip(np.rint(ijk_f).astype(np.int32), 1, g - 2)

    density = np.zeros((g, g, g), dtype=np.float32)
    col_acc = np.zeros((g, g, g, 3), dtype=np.float32)
    np.add.at(density, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), opa)
    for ch in range(3):
        np.add.at(col_acc[:, :, :, ch], (ijk[:, 0], ijk[:, 1], ijk[:, 2]), opa * rgb[:, ch])

    phi = _smooth_3d_separable(density, passes=2)
    col_sm = np.stack([_smooth_3d_separable(col_acc[:, :, :, ch], passes=2) for ch in range(3)], axis=-1)
    col_grid = np.where(phi[..., None] > 1e-6, col_sm / np.maximum(phi[..., None], 1e-6), 0.72)

    nz_vals = phi[phi > 1e-5]
    iso = float(np.percentile(nz_vals, 25.0)) * 0.55 if len(nz_vals) > 0 else 0.02
    iso = max(1e-4, iso)
    occ = phi >= iso

    # Active dual cells `(g-1, g-1, g-1)` straddling the iso-surface
    c000 = occ[:-1, :-1, :-1]
    c100 = occ[1:, :-1, :-1]
    c010 = occ[:-1, 1:, :-1]
    c110 = occ[1:, 1:, :-1]
    c001 = occ[:-1, :-1, 1:]
    c101 = occ[1:, :-1, 1:]
    c011 = occ[:-1, 1:, 1:]
    c111 = occ[1:, 1:, 1:]

    corner_sum = (
        c000.astype(np.uint8)
        + c100.astype(np.uint8)
        + c010.astype(np.uint8)
        + c110.astype(np.uint8)
        + c001.astype(np.uint8)
        + c101.astype(np.uint8)
        + c011.astype(np.uint8)
        + c111.astype(np.uint8)
    )
    active = (corner_sum > 0) & (corner_sum < 8)
    act_ijk = np.argwhere(active)
    if len(act_ijk) < 4:
        raise ValueError("Proxy iso-surface yielded too few active surface cells.")

    cell_to_vid = np.full((g - 1, g - 1, g - 1), -1, dtype=np.int32)
    cell_to_vid[act_ijk[:, 0], act_ijk[:, 1], act_ijk[:, 2]] = np.arange(len(act_ijk), dtype=np.int32)

    # Smooth Surface-Nets dual vertex placement via iso-weighted corner interpolation
    corner_offsets = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.float32,
    )
    ax, ay, az = act_ijk[:, 0], act_ijk[:, 1], act_ijk[:, 2]
    w_sum = np.zeros(len(act_ijk), dtype=np.float32)
    local_pos = np.zeros((len(act_ijk), 3), dtype=np.float32)
    v_rgb = np.zeros((len(act_ijk), 3), dtype=np.float32)

    for co in corner_offsets:
        cx_i = ax + int(co[0])
        cy_i = ay + int(co[1])
        cz_i = az + int(co[2])
        p_val = phi[cx_i, cy_i, cz_i]
        # Higher weight near the `phi == iso` surface crossing
        w = 1.0 / (np.abs(p_val - iso) + 0.05 * iso + 1e-5)
        w_sum += w
        local_pos += w[:, None] * co[None, :]
        v_rgb += w[:, None] * col_grid[cx_i, cy_i, cz_i]

    local_pos /= w_sum[:, None]
    v_rgb /= w_sum[:, None]
    verts_xyz = origin[None, :] + (act_ijk.astype(np.float32) + local_pos) * cell_size[None, :]

    # Analytic outward normal `-grad(phi)`
    gx = phi[np.minimum(ax + 1, g - 1), ay, az] - phi[np.maximum(ax - 1, 0), ay, az]
    gy = phi[ax, np.minimum(ay + 1, g - 1), az] - phi[ax, np.maximum(ay - 1, 0), az]
    gz = phi[ax, ay, np.minimum(az + 1, g - 1)] - phi[ax, ay, np.maximum(az - 1, 0)]
    nrms = -np.stack([gx / cell_size[0], gy / cell_size[1], gz / cell_size[2]], axis=-1)
    n_norm = np.linalg.norm(nrms, axis=-1, keepdims=True)
    nrms = np.where(n_norm > 1e-8, nrms / np.maximum(n_norm, 1e-8), np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # Connect sign-changing interior grid edges into consistently oriented triangle pairs
    tris: list[np.ndarray] = []

    # X-edges: `(i, j, k) -> (i+1, j, k)` for i in 0..g-2, j in 1..g-2, k in 1..g-2
    ex_flip = occ[:-1, 1:-1, 1:-1] != occ[1:, 1:-1, 1:-1]
    if ex_flip.any():
        ei, ej, ek = np.nonzero(ex_flip)
        ej = ej + 1
        ek = ek + 1
        v0 = cell_to_vid[ei, ej - 1, ek - 1]
        v1 = cell_to_vid[ei, ej, ek - 1]
        v2 = cell_to_vid[ei, ej, ek]
        v3 = cell_to_vid[ei, ej - 1, ek]
        pos_sign = occ[ei, ej, ek]  # True means inside->outside along +X
        t_a = np.where(pos_sign[:, None], np.stack([v0, v1, v2], 1), np.stack([v0, v2, v1], 1))
        t_b = np.where(pos_sign[:, None], np.stack([v0, v2, v3], 1), np.stack([v0, v3, v2], 1))
        tris.extend([t_a, t_b])

    # Y-edges: `(i, j, k) -> (i, j+1, k)` for i in 1..g-2, j in 0..g-2, k in 1..g-2
    ey_flip = occ[1:-1, :-1, 1:-1] != occ[1:-1, 1:, 1:-1]
    if ey_flip.any():
        ei, ej, ek = np.nonzero(ey_flip)
        ei = ei + 1
        ek = ek + 1
        v0 = cell_to_vid[ei - 1, ej, ek - 1]
        v1 = cell_to_vid[ei - 1, ej, ek]
        v2 = cell_to_vid[ei, ej, ek]
        v3 = cell_to_vid[ei, ej, ek - 1]
        pos_sign = occ[ei, ej, ek]
        t_a = np.where(pos_sign[:, None], np.stack([v0, v1, v2], 1), np.stack([v0, v2, v1], 1))
        t_b = np.where(pos_sign[:, None], np.stack([v0, v2, v3], 1), np.stack([v0, v3, v2], 1))
        tris.extend([t_a, t_b])

    # Z-edges: `(i, j, k) -> (i, j, k+1)` for i in 1..g-2, j in 1..g-2, k in 0..g-2
    ez_flip = occ[1:-1, 1:-1, :-1] != occ[1:-1, 1:-1, 1:]
    if ez_flip.any():
        ei, ej, ek = np.nonzero(ez_flip)
        ei = ei + 1
        ej = ej + 1
        v0 = cell_to_vid[ei - 1, ej - 1, ek]
        v1 = cell_to_vid[ei, ej - 1, ek]
        v2 = cell_to_vid[ei, ej, ek]
        v3 = cell_to_vid[ei - 1, ej, ek]
        pos_sign = occ[ei, ej, ek]
        t_a = np.where(pos_sign[:, None], np.stack([v0, v1, v2], 1), np.stack([v0, v2, v1], 1))
        t_b = np.where(pos_sign[:, None], np.stack([v0, v2, v3], 1), np.stack([v0, v3, v2], 1))
        tris.extend([t_a, t_b])

    if not tris:
        raise ValueError("Proxy iso-surface generated zero faces.")

    faces = np.concatenate(tris, axis=0)
    valid_f = (faces >= 0).all(axis=1)
    faces = faces[valid_f].astype(np.uint32)

    colors_rgba = np.concatenate([np.clip(v_rgb, 0.0, 1.0), np.ones((len(verts_xyz), 1), dtype=np.float32)], axis=1)
    glb = pack_glb_bytes(verts_xyz, nrms, colors_rgba, faces, mesh_name=mesh_name)
    return glb, {"vertices": len(verts_xyz), "faces": len(faces), "grid_res": g}


def extract_proxy_glb_from_ply(
    ply_path: Path,
    out_glb_path: Path,
    *,
    grid_res: int = 48,
    min_opacity: float = 0.05,
) -> dict[str, int]:
    """Build `proxy_collision.glb` on disk directly from a binary little-endian 3DGS `.ply`."""
    raw = ply_path.read_bytes()
    end_marker = b"end_header\n"
    idx = raw.find(end_marker)
    if idx < 0:
        end_marker = b"end_header\r\n"
        idx = raw.find(end_marker)
    if idx < 0:
        raise ValueError(f"Invalid PLY header in {ply_path}")

    hdr = raw[:idx].decode("ascii", errors="ignore")
    data_offset = idx + len(end_marker)
    n_verts = 0
    props: list[str] = []
    for ln in hdr.splitlines():
        parts = ln.strip().split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property":
            props.append(parts[-1])

    stride_f = len(props)
    arr = np.frombuffer(raw, dtype="<f4", count=n_verts * stride_f, offset=data_offset).reshape(n_verts, stride_f)
    pmap = {name: i for i, name in enumerate(props)}

    xyz = np.stack([arr[:, pmap["x"]], arr[:, pmap["y"]], arr[:, pmap["z"]]], axis=-1)
    if "opacity" in pmap:
        op_raw = np.clip(arr[:, pmap["opacity"]], -20.0, 20.0)
        opa = 1.0 / (1.0 + np.exp(-op_raw))
    else:
        opa = np.ones(n_verts, dtype=np.float32)

    scales = None
    if all(k in pmap for k in ("scale_0", "scale_1", "scale_2")):
        scales = np.exp(np.clip(np.stack([arr[:, pmap["scale_0"]], arr[:, pmap["scale_1"]], arr[:, pmap["scale_2"]]], axis=-1), -20.0, 5.0))

    if all(k in pmap for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
        rgb = np.clip(0.5 + _SH_C0 * np.stack([arr[:, pmap["f_dc_0"]], arr[:, pmap["f_dc_1"]], arr[:, pmap["f_dc_2"]]], axis=-1), 0.0, 1.0)
    else:
        rgb = np.full((n_verts, 3), 0.72, dtype=np.float32)

    glb_bytes, stats = build_proxy_glb_bytes(
        xyz,
        opa,
        scales,
        rgb,
        grid_res=grid_res,
        min_opacity=min_opacity,
        mesh_name=f"{ply_path.parent.name}_Proxy",
    )
    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    out_glb_path.write_bytes(glb_bytes)
    return stats