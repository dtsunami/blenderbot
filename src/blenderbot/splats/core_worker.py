"""Cloud GPU 3DGS worker core (`extract -> pose/align -> VRAM gsplat -> export + surgical resume`).

Runs inside the Modal CUDA container (or any RunPod / Vast.ai GPU pod via `pod_server.py`).
Zero Modal imports here so the core remains 100% portable across cloud GPU hosts.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import math
import os
import random
import shutil
import struct
import subprocess
import tarfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import ExtractConfig, PoseConfig, SplatConfig, SurgerySpec
from .sim3 import (
    Sim3Transform,
    apply_sim3_to_camera_w2c,
    estimate_floor_aligned_sim3,
    opencv_w2c_to_blender_camera,
)

_SH_C0 = 0.28209479177387814

# Cached VGGT model singleton across warm bursts inside the same GPU container
_VGGT_CACHE: dict[str, Any] = {}


# ─── Stage 1: Fast FFmpeg Frame Extraction ────────────────────────────────────


def _probe_video_duration_fps(video_path: Path) -> tuple[float, float]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate:format=duration",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    dur = 5.0
    fps = 24.0
    try:
        info = json.loads(res.stdout or "{}")
        dur = float((info.get("format") or {}).get("duration") or 5.0)
        rate_str = ((info.get("streams") or [{}])[0]).get("r_frame_rate", "24/1")
        num, den = rate_str.split("/")
        fps = float(num) / max(float(den), 1.0)
    except Exception:  # noqa: BLE001, S110
        pass
    return dur, fps


def extract_frames(video_path: Path, frames_dir: Path, cfg: ExtractConfig) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    dur, _native_fps = _probe_video_duration_fps(video_path)
    sample_fps = cfg.fps
    if dur > 0 and cfg.max_frames > 0 and (dur * sample_fps) > cfg.max_frames:
        sample_fps = cfg.max_frames / dur

    vf_parts = [f"fps={sample_fps:.4f}"]
    if cfg.dedup_frames:
        vf_parts.append("mpdecimate=hi=64*8:lo=64*3:frac=0.25")
    if cfg.max_width > 0:
        vf_parts.append(f"scale='min({cfg.max_width},iw)':-2:flags=lanczos")

    qscale = max(1, min(10, 32 - cfg.quality // 3))
    out_pat = str(frames_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", ",".join(vf_parts),
        "-vsync", "vfr" if cfg.dedup_frames else "passthrough",
        "-q:v", str(qscale),
        "-hide_banner", "-loglevel", "error",
        out_pat,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed: {proc.stderr[-600:]}")

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if cfg.max_frames > 0 and len(frames) > cfg.max_frames:
        # Uniform subsample if mpdecimate kept more than max_frames
        idxs = np.linspace(0, len(frames) - 1, cfg.max_frames, dtype=int)
        keep_set = {frames[i] for i in idxs}
        for f in frames:
            if f not in keep_set:
                f.unlink(missing_ok=True)
        frames = sorted(frames_dir.glob("frame_*.jpg"))

    if len(frames) < 3:
        raise RuntimeError(f"Too few frames extracted ({len(frames)}). Need at least 3 frames.")
    return frames


# ─── Stage 2A: Feed-Forward Metric Pose + Dense Cloud via VGGT-1B ─────────────


def _get_vggt_model(device: torch.device, models_dir: Path | None = None):
    if "model" in _VGGT_CACHE:
        return _VGGT_CACHE["model"]
    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(models_dir / "hf"))
    from vggt.models.vggt import VGGT

    dtype_cap = torch.cuda.get_device_capability(device)[0] >= 8 if device.type == "cuda" else False
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    _VGGT_CACHE["model"] = model
    _VGGT_CACHE["bf16"] = dtype_cap
    return model


def solve_poses_vggt(
    frame_paths: list[Path],
    cfg: PoseConfig,
    device: torch.device,
    models_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Solve camera intrinsics/extrinsics + dense RGB point cloud via VGGT-1B on GPU."""
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Subsample uniformly if frame count exceeds vggt_max_frames (keeps attention fast & crisp)
    if cfg.vggt_max_frames > 0 and len(frame_paths) > cfg.vggt_max_frames:
        sel_idx = np.linspace(0, len(frame_paths) - 1, cfg.vggt_max_frames, dtype=int)
        used_paths = [frame_paths[int(i)] for i in sel_idx]
    else:
        used_paths = list(frame_paths)

    with Image.open(used_paths[0]) as im0:
        orig_w, orig_h = im0.width, im0.height

    model = _get_vggt_model(device, models_dir=models_dir)
    amp_dtype = torch.bfloat16 if _VGGT_CACHE.get("bf16", True) else torch.float16

    # Use mode="pad" so portrait (9:16), square (1:1), and ultrawide (21:9) Seedance clips NEVER
    # suffer top/bottom center-crop truncation — VGGT sees 100% of every frame!
    try:
        imgs = load_and_preprocess_images([str(p) for p in used_paths], mode="pad").to(device)
        used_pad = True
    except TypeError:
        imgs = load_and_preprocess_images([str(p) for p in used_paths]).to(device)
        used_pad = False
    if imgs.ndim == 4:
        imgs_b = imgs.unsqueeze(0)
    else:
        imgs_b = imgs
    _, _s_cnt, _, vh, vw = imgs_b.shape

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
        preds = model(imgs_b)
        extrinsic_b, intrinsic_b = pose_encoding_to_extri_intri(preds["pose_enc"], (vh, vw))

    ext_np = extrinsic_b[0].float().cpu().numpy()   # (S, 3, 4) OpenCV w2c [R|t]
    int_np = intrinsic_b[0].float().cpu().numpy()   # (S, 3, 3) at (vw, vh)

    if "depth" in preds and "depth_conf" in preds:
        depth_np = preds["depth"][0].float().cpu().numpy()        # (S, vh, vw, 1) or (S, vh, vw)
        conf_np = preds["depth_conf"][0].float().cpu().numpy()    # (S, vh, vw)
        pts_map = unproject_depth_map_to_point_map(depth_np, ext_np, int_np)  # (S, vh, vw, 3)
    else:
        pts_map = preds["world_points"][0].float().cpu().numpy()
        conf_np = preds["world_points_conf"][0].float().cpu().numpy()

    rgb_map = (imgs_b[0].permute(0, 2, 3, 1).float().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(
        np.uint8
    )  # (S, vh, vw, 3)

    # Exact intrinsic un-mapping for both `mode="pad"` (longest side = 518, symmetric letterbox/pillarbox)
    # and fallback `mode="crop"` (width = 518, vertical crop):
    if used_pad:
        if orig_w >= orig_h:
            scale_px = orig_w / float(vw)
            pad_x = 0.0
            pad_y = max(0.0, (float(vh) - orig_h / scale_px) * 0.5)
        else:
            scale_px = orig_h / float(vh)
            pad_y = 0.0
            pad_x = max(0.0, (float(vw) - orig_w / scale_px) * 0.5)
        # Zero out confidence on white letterbox/pillarbox bars so zero border floaters spawn!
        px_i = round(pad_x)
        py_i = round(pad_y)
        if px_i > 0:
            conf_np[:, :, :px_i] = -1e9
            conf_np[:, :, vw - px_i :] = -1e9
        if py_i > 0:
            conf_np[:, :py_i, :] = -1e9
            conf_np[:, vh - py_i :, :] = -1e9
    else:
        scale_px = orig_w / float(vw)
        pad_x = 0.0
        pad_y = -max(0.0, (orig_h / scale_px - float(vh)) * 0.5)

    cam_records: list[dict[str, Any]] = []
    for i, fp in enumerate(used_paths):
        r_w2c = ext_np[i, :3, :3].astype(np.float64)
        u_svd, _, vt_svd = np.linalg.svd(r_w2c)
        r_orth = u_svd @ vt_svd
        if np.linalg.det(r_orth) < 0:
            u_svd[:, -1] *= -1.0
            r_orth = u_svd @ vt_svd
        t_w2c = ext_np[i, :3, 3].astype(np.float64)

        k_v = int_np[i]
        fx = float(k_v[0, 0]) * scale_px
        fy = float(k_v[1, 1]) * scale_px
        cx = (float(k_v[0, 2]) - pad_x) * scale_px
        cy = (float(k_v[1, 2]) - pad_y) * scale_px

        cam_records.append({
            "frame_index": i,
            "image_name": fp.name,
            "image_path": str(fp),
            "width": orig_w,
            "height": orig_h,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "R_w2c": r_orth.tolist(),
            "t_w2c": t_w2c.tolist(),
        })

    # Confidence-filter & subsample initial dense point cloud
    flat_pts = pts_map.reshape(-1, 3)
    flat_rgb = rgb_map.reshape(-1, 3)
    flat_conf = conf_np.reshape(-1)
    finite_mask = np.isfinite(flat_pts).all(axis=1) & np.isfinite(flat_conf)
    flat_pts = flat_pts[finite_mask]
    flat_rgb = flat_rgb[finite_mask]
    flat_conf = flat_conf[finite_mask]

    if len(flat_pts) > 1000 and cfg.conf_percentile > 0.0:
        thresh = float(np.percentile(flat_conf, cfg.conf_percentile))
        keep = flat_conf >= thresh
        if keep.sum() >= 500:
            flat_pts = flat_pts[keep]
            flat_rgb = flat_rgb[keep]

    if cfg.max_init_points > 0 and len(flat_pts) > cfg.max_init_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(flat_pts), size=cfg.max_init_points, replace=False)
        flat_pts = flat_pts[sel]
        flat_rgb = flat_rgb[sel]

    return cam_records, flat_pts.astype(np.float64), flat_rgb.astype(np.uint8)


# ─── Stage 2B: COLMAP Sequential Fallback (`arcgs` Binary Parsers) ────────────


_COLMAP_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


def solve_poses_colmap(
    frames_dir: Path,
    work_dir: Path,
    cfg: PoseConfig,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Run COLMAP sequential SfM + undistort and parse `cameras.bin`, `images.bin`, `points3D.bin`."""
    sfm_dir = work_dir / "colmap"
    db_path = sfm_dir / "database.db"
    sparse_dir = sfm_dir / "sparse"
    undist_dir = sfm_dir / "undistorted"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    def _colmap(args: list[str], step_name: str) -> None:
        proc = subprocess.run(["colmap", *args], capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0:
            tail = (proc.stdout + "\n" + proc.stderr)[-800:]
            raise RuntimeError(f"COLMAP {step_name} failed:\n{tail}")

    _colmap([
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", "16",
    ], "feature_extractor")
    _colmap([
        "sequential_matcher",
        "--database_path", str(db_path),
        "--SequentialMatching.overlap", "12",
        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", "16",
    ], "sequential_matcher")
    _colmap([
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
    ], "mapper")

    sub_models = sorted(
        (p for p in sparse_dir.iterdir() if p.is_dir() and (p / "images.bin").is_file()),
        key=lambda p: (p / "images.bin").stat().st_size,
        reverse=True,
    )
    if not sub_models:
        raise RuntimeError("COLMAP mapper failed to register video frames (not enough rigid parallax).")
    sparse_0 = sub_models[0]

    undist_dir.mkdir(parents=True, exist_ok=True)
    _colmap([
        "image_undistorter",
        "--image_path", str(frames_dir),
        "--input_path", str(sparse_0),
        "--output_path", str(undist_dir),
        "--output_type", "COLMAP",
    ], "image_undistorter")

    sp = undist_dir / "sparse"
    img_root = undist_dir / "images"

    # Parse cameras.bin
    cams_map: dict[int, tuple[int, int, int, tuple[float, ...]]] = {}
    with open(sp / "cameras.bin", "rb") as f:
        (n_cams,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_cams):
            cid = struct.unpack("<I", f.read(4))[0]
            mid = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            npars = _COLMAP_CAMERA_MODELS[mid][1]
            pars = struct.unpack(f"<{npars}d", f.read(8 * npars))
            cams_map[cid] = (mid, w, h, pars)

    # Parse images.bin
    raw_imgs = []
    with open(sp / "images.bin", "rb") as f:
        (n_im,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_im):
            _iid = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<I", f.read(4))[0]
            nb = bytearray()
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                nb.extend(c)
            name = nb.decode("utf-8", errors="replace")
            (n_pts2d,) = struct.unpack("<Q", f.read(8))
            f.seek(n_pts2d * 24, os.SEEK_CUR)
            raw_imgs.append((name, (qw, qx, qy, qz), (tx, ty, tz), cid))

    raw_imgs.sort(key=lambda r: r[0])
    cam_records: list[dict[str, Any]] = []
    from .sim3 import quat_to_rotmat

    for idx, (name, qvec, tvec, cid) in enumerate(raw_imgs):
        mid, w, h, p = cams_map[cid]
        mname = _COLMAP_CAMERA_MODELS[mid][0]
        if mname in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        r_mat = quat_to_rotmat(qvec)
        cam_records.append({
            "frame_index": idx,
            "image_name": Path(name).name,
            "image_path": str(img_root / Path(name).name),
            "width": int(w),
            "height": int(h),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "R_w2c": [list(row) for row in r_mat],
            "t_w2c": list(tvec),
        })

    # Parse points3D.bin
    xyz_list, rgb_list = [], []
    with open(sp / "points3D.bin", "rb") as f:
        (n_pts,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_pts):
            f.seek(8, os.SEEK_CUR)
            xyz_list.append(struct.unpack("<3d", f.read(24)))
            rgb_list.append(struct.unpack("<3B", f.read(3)))
            f.seek(8, os.SEEK_CUR)
            (tlen,) = struct.unpack("<Q", f.read(8))
            f.seek(tlen * 8, os.SEEK_CUR)

    xyz = np.array(xyz_list, dtype=np.float64)
    rgb = np.array(rgb_list, dtype=np.uint8)
    if cfg.max_init_points > 0 and len(xyz) > cfg.max_init_points:
        sel = np.random.default_rng(42).choice(len(xyz), size=cfg.max_init_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return cam_records, xyz, rgb


# ─── Stage 3: VRAM-Preloaded CUDA `gsplat` Trainer + Surgical Resume ──────────


def _preload_frames_u8(
    cam_records: list[dict[str, Any]], load_alpha: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Decode all training images in parallel once onto contiguous uint8 tensors.

    Returns `(frames_rgb_u8 [M,H,W,3], alphas_u8 [M,H,W,1] | None)`.
    """
    w0, h0 = cam_records[0]["width"], cam_records[0]["height"]

    def _read_one(rec: dict[str, Any]) -> np.ndarray:
        with Image.open(rec["image_path"]) as im:
            mode = "RGBA" if load_alpha else "RGB"
            im_conv = im.convert(mode)
            if im_conv.width != w0 or im_conv.height != h0:
                im_conv = im_conv.resize((w0, h0), Image.LANCZOS)
            return np.asarray(im_conv, dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        arrs = list(pool.map(_read_one, cam_records))
    stacked = np.stack(arrs, axis=0)
    if load_alpha:
        rgb_t = torch.from_numpy(np.ascontiguousarray(stacked[:, :, :, :3]))
        alpha_t = torch.from_numpy(np.ascontiguousarray(stacked[:, :, :, 3:4]))
        return rgb_t, alpha_t
    return torch.from_numpy(stacked), None


def _unpack_known_views_bundle(
    tar_gz_bytes: bytes, work_dir: Path, max_init_points: int = 180_000
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, bool]:
    """Unpack a Blender `splat.capture_views` `.tar.gz` bundle with exact ground-truth poses & seed cloud."""
    with tarfile.open(fileobj=io.BytesIO(tar_gz_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            # Prevent path traversal
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                continue
            tf.extract(member, path=work_dir)

    manifest_path = work_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("views_bundle.tar.gz is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    has_alpha = bool(manifest.get("transparent_bg", False))

    cam_records: list[dict[str, Any]] = []
    for i, v in enumerate(manifest.get("views") or []):
        img_p = work_dir / str(v["image_name"])
        if not img_p.is_file():
            continue
        cam_records.append({
            "frame_index": int(v.get("frame_index", i)),
            "image_name": str(v["image_name"]),
            "image_path": str(img_p),
            "width": int(v["width"]),
            "height": int(v["height"]),
            "fx": float(v["fx"]),
            "fy": float(v["fy"]),
            "cx": float(v["cx"]),
            "cy": float(v["cy"]),
            "R_w2c": v["R_w2c"],
            "t_w2c": v["t_w2c"],
        })
    if len(cam_records) < 3:
        raise RuntimeError(f"Known-views bundle only contained {len(cam_records)} valid images (need >= 3).")

    pts_bin = work_dir / "init_points.bin"
    if pts_bin.is_file():
        raw = pts_bin.read_bytes()
        (n_pts,) = struct.unpack_from("<I", raw, 0)
        dt = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
        arr = np.frombuffer(raw, dtype=dt, offset=4, count=n_pts)
        xyz = arr["xyz"].astype(np.float64)
        rgb = arr["rgb"].astype(np.uint8)
    else:
        bmin = np.array(manifest.get("bounds_min") or [-1.0, -1.0, 0.0], dtype=np.float64)
        bmax = np.array(manifest.get("bounds_max") or [1.0, 1.0, 2.0], dtype=np.float64)
        rng = np.random.default_rng(42)
        xyz = rng.uniform(bmin, bmax, size=(8_000, 3))
        rgb = np.full((8_000, 3), 180, dtype=np.uint8)

    if max_init_points > 0 and len(xyz) > max_init_points:
        sel = np.random.default_rng(42).choice(len(xyz), size=max_init_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return cam_records, xyz, rgb, has_alpha


def _init_gaussians_cuda(
    xyz_bl: np.ndarray,
    rgb_u8: np.ndarray,
    sh_degree: int,
    device: torch.device,
) -> torch.nn.ParameterDict:
    n = len(xyz_bl)
    means = torch.from_numpy(xyz_bl.astype(np.float32)).to(device)

    # Local 3-NN initial scale (evaluated on random 16k subset against 16k subset for speed)
    rng_idx = torch.randperm(n, device=device)[: min(n, 16_384)]
    sub = means[rng_idx]
    dists = torch.cdist(sub, sub)
    dists.fill_diagonal_(float("inf"))
    k_nn = min(3, max(1, sub.shape[0] - 1))
    knn_d, _ = dists.topk(k_nn, dim=1, largest=False)
    # Approximate per-point spacing; if n <= 16k use exact per-point 3-NN!
    if n <= 16_384:
        init_r = knn_d.mean(dim=1, keepdim=True).clamp(1e-4, 0.25).expand(n, 3)
        scales = torch.log(init_r).contiguous()
    else:
        med_r = float(knn_d.median().item())
        med_r = max(1e-4, min(0.15, med_r))
        scales = torch.full((n, 3), math.log(med_r), device=device)

    quats = torch.zeros((n, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((n,), math.log(0.15 / 0.85), dtype=torch.float32, device=device)

    col = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).to(device)
    sh0 = ((col - 0.5) / _SH_C0).unsqueeze(1)
    k_sh = (max(0, sh_degree) + 1) ** 2
    shn = torch.zeros((n, max(0, k_sh - 1), 3), dtype=torch.float32, device=device)

    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shn),
    })


def _make_optimizers(params: torch.nn.ParameterDict, scene_scale: float = 1.0) -> dict[str, torch.optim.Adam]:
    lr_map = {
        "means": 1.6e-4 * max(0.25, min(4.0, scene_scale)),
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    return {
        k: torch.optim.Adam([params[k]], lr=lr_map[k], eps=1e-15)
        for k in params
    }


def _ssim_cuda(pred: torch.Tensor, gt: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Fast 11x11 Gaussian SSIM on `(H, W, 3)` float32 tensors using a pre-allocated window."""
    p = pred.permute(2, 0, 1).unsqueeze(0)
    g = gt.permute(2, 0, 1).unsqueeze(0)
    mu1 = F.conv2d(p, window, padding=5, groups=3)
    mu2 = F.conv2d(g, window, padding=5, groups=3)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sig1_sq = F.conv2d(p * p, window, padding=5, groups=3) - mu1_sq
    sig2_sq = F.conv2d(g * g, window, padding=5, groups=3) - mu2_sq
    sig12 = F.conv2d(p * g, window, padding=5, groups=3) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2.0 * mu12 + c1) * (2.0 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2))
    return smap.mean()


def _build_ssim_window(device: torch.device) -> torch.Tensor:
    x = torch.arange(11, dtype=torch.float32, device=device) - 5.0
    k1 = torch.exp(-x * x / (2.0 * 1.5 * 1.5))
    k1 = k1 / k1.sum()
    k2 = (k1.unsqueeze(0) * k1.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return k2.expand(3, 1, 11, 11).contiguous()


def _encode_jpeg_b64(arr_hwc_f32: np.ndarray, max_w: int = 640, quality: int = 82) -> str:
    u8 = (np.clip(arr_hwc_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(u8)
    if im.width > max_w:
        nh = max(2, round(im.height * (max_w / float(im.width))))
        im = im.resize((max_w, nh), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_diff_b64(render: np.ndarray, gt: np.ndarray, max_w: int = 640) -> str:
    lum_r = 0.299 * render[:, :, 0] + 0.587 * render[:, :, 1] + 0.114 * render[:, :, 2]
    lum_g = 0.299 * gt[:, :, 0] + 0.587 * gt[:, :, 1] + 0.114 * gt[:, :, 2]
    d = np.abs(lum_r - lum_g)
    fire = np.stack([np.clip(d * 3.0, 0, 1), np.clip(d * 3.0 - 1.0, 0, 1), np.clip(d * 3.0 - 2.0, 0, 1)], axis=-1)
    return _encode_jpeg_b64(fire.astype(np.float32), max_w=max_w, quality=80)


def slice_params_and_adam(
    params: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Adam],
    strategy_state: dict[str, Any],
    keep: torch.Tensor,
) -> int:
    """Prune Gaussians while preserving Adam `exp_avg`/`exp_avg_sq` and `DefaultStrategy` state."""
    n_before = int(params["means"].shape[0])
    n_after = int(keep.sum().item())
    if n_after == n_before or n_after == 0:
        return 0
    for name in list(params.keys()):
        old_p = params[name]
        opt = optimizers[name]
        old_st = opt.state.pop(old_p, None)
        new_p = torch.nn.Parameter(old_p.detach()[keep].contiguous().requires_grad_(True))
        params[name] = new_p
        opt.param_groups[0]["params"] = [new_p]
        if old_st is not None:
            new_st: dict[str, Any] = {}
            for sk, sv in old_st.items():
                if torch.is_tensor(sv) and sv.ndim > 0 and sv.shape[0] == n_before:
                    new_st[sk] = sv[keep].contiguous()
                else:
                    new_st[sk] = sv
            opt.state[new_p] = new_st
    for sk, sv in list(strategy_state.items()):
        if torch.is_tensor(sv) and sv.ndim > 0 and sv.shape[0] == n_before:
            strategy_state[sk] = sv[keep].contiguous()
    return n_before - n_after


def apply_surgeries(
    params: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Adam],
    strategy_state: dict[str, Any],
    surgery: SurgerySpec,
) -> dict[str, Any]:
    """Execute pre-burst surgical tensor edits (`prune_opa`, `drop_aabb`, `clamp_log_scale`, etc.)."""
    n_start = int(params["means"].shape[0])
    details: list[str] = []
    with torch.no_grad():
        keep = torch.ones((n_start,), dtype=torch.bool, device=params["means"].device)
        if surgery.prune_opa is not None:
            keep &= torch.sigmoid(params["opacities"]) >= float(surgery.prune_opa)
            details.append(f"prune_opa>={surgery.prune_opa}")
        if surgery.max_world_scale is not None:
            max_lin = torch.exp(params["scales"]).max(dim=-1).values
            keep &= max_lin <= float(surgery.max_world_scale)
            details.append(f"max_world_scale<={surgery.max_world_scale}")
        if surgery.keep_aabb is not None:
            lo = torch.tensor(surgery.keep_aabb[0], dtype=torch.float32, device=keep.device)
            hi = torch.tensor(surgery.keep_aabb[1], dtype=torch.float32, device=keep.device)
            m = params["means"]
            keep &= ((m >= lo) & (m <= hi)).all(dim=-1)
            details.append("keep_aabb")
        if surgery.drop_aabb is not None:
            lo = torch.tensor(surgery.drop_aabb[0], dtype=torch.float32, device=keep.device)
            hi = torch.tensor(surgery.drop_aabb[1], dtype=torch.float32, device=keep.device)
            m = params["means"]
            inside = ((m >= lo) & (m <= hi)).all(dim=-1)
            keep &= ~inside
            details.append("drop_aabb")
        if surgery.keep_radius is not None:
            r2 = (params["means"] ** 2).sum(dim=-1)
            keep &= r2 <= float(surgery.keep_radius) ** 2
            details.append(f"keep_radius<={surgery.keep_radius}")

        removed = slice_params_and_adam(params, optimizers, strategy_state, keep)

        if surgery.clamp_log_scale is not None:
            params["scales"].data.clamp_(max=float(surgery.clamp_log_scale))
            details.append(f"clamp_log_scale<={surgery.clamp_log_scale}")
        if surgery.reset_opacities is not None:
            v = max(1e-4, min(0.99, float(surgery.reset_opacities)))
            params["opacities"].data.fill_(math.log(v / (1.0 - v)))
            details.append(f"reset_opacities={v}")
        if surgery.lr_scale != 1.0:
            for opt in optimizers.values():
                for pg in opt.param_groups:
                    pg["lr"] *= float(surgery.lr_scale)
            details.append(f"lr_scale*{surgery.lr_scale}")

    return {
        "n_before": n_start,
        "n_after": int(params["means"].shape[0]),
        "removed": removed,
        "ops": details,
    }


# ─── Stage 4: Binary `.ply` + 32-byte `.splat` Exporters (`arcgs` Format) ─────


def export_ply_bytes(
    params: torch.nn.ParameterDict,
    max_gaussians: int = 0,
    sort_by_opacity: bool = True,
) -> tuple[bytes, int]:
    """Serialize standard binary_little_endian 3DGS `.ply` (already in Blender `+Z` up, `Z=0` coords)."""
    with torch.no_grad():
        means = params["means"].detach().float().cpu().numpy()
        scales = params["scales"].detach().float().cpu().numpy()
        quats = F.normalize(params["quats"].detach().float(), dim=-1).cpu().numpy()
        opas = params["opacities"].detach().float().cpu().numpy()
        n = means.shape[0]
        sh0 = params["sh0"].detach().float().cpu().numpy().reshape(n, 3)
        k_rest = params["shN"].shape[1]
        shn = params["shN"].detach().float().cpu().numpy().reshape(n, k_rest * 3)

    if sort_by_opacity or (max_gaussians > 0 and n > max_gaussians):
        order = np.argsort(-opas)
        if max_gaussians > 0 and n > max_gaussians:
            order = order[:max_gaussians]
        means, scales, quats, opas, sh0, shn = (
            means[order], scales[order], quats[order], opas[order], sh0[order], shn[order]
        )
        n = means.shape[0]

    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(k_rest * 3)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    normals = np.zeros((n, 3), dtype=np.float32)
    packed = np.concatenate(
        [means, normals, sh0, shn, opas[:, None], scales, quats], axis=1
    ).astype("<f4")

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "\n".join(f"property float {p}" for p in props)
        + "\nend_header\n"
    ).encode("ascii")
    return header + packed.tobytes(), n


def export_splat_bytes(params: torch.nn.ParameterDict, max_gaussians: int = 0) -> bytes:
    """Vectorized 32-byte `.splat` binary serializer (SuperSplat / WebGL format, <20 ms for 300k)."""
    with torch.no_grad():
        means = params["means"].detach().float().cpu().numpy()
        scales_lin = torch.exp(params["scales"].detach().float()).cpu().numpy()
        quats = F.normalize(params["quats"].detach().float(), dim=-1).cpu().numpy()
        opa = torch.sigmoid(params["opacities"].detach().float()).cpu().numpy()
        sh0 = params["sh0"].detach().float().cpu().numpy().reshape(-1, 3)

    order = np.argsort(-opa)
    if max_gaussians > 0 and len(order) > max_gaussians:
        order = order[:max_gaussians]
    means = means[order].astype("<f4")
    scales_lin = scales_lin[order].astype("<f4")
    rgb = np.clip(0.5 + _SH_C0 * sh0[order], 0.0, 1.0)
    rgba_u8 = np.clip(np.concatenate([rgb, opa[order, None]], axis=1) * 255.0, 0, 255).astype(np.uint8)
    quat_u8 = np.clip((quats[order] + 1.0) * 127.5, 0, 255).astype(np.uint8)

    n = len(order)
    rec = np.empty(n, dtype=[("pos", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("rot", "u1", (4,))])
    rec["pos"] = means
    rec["scale"] = scales_lin
    rec["rgba"] = rgba_u8
    rec["rot"] = quat_u8
    return rec.tobytes()


# ─── End-to-End Job Runner & Stateful Burst Trainer ───────────────────────────


def run_splat_job_stream(
    *,
    job_id: str,
    jobs_root: Path,
    models_dir: Path | None = None,
    mp4_bytes: bytes | None = None,
    views_tar_gz: bytes | None = None,
    cfg: SplatConfig | None = None,
    add_steps: int | None = None,
    surgery: SurgerySpec | None = None,
    masks_b64: dict[str, str] | None = None,
    exclude_frames: list[str] | None = None,
    gpu_hourly_usd: float = 3.95,
    ram_cache: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Generator yielding live telemetry events and final compressed artifacts.

    Supports:
      1. `from_video` (`mp4_bytes` provided): FFmpeg -> VGGT/COLMAP -> RANSAC Z=0 Sim3 -> gsplat
      2. `from_blender` (`views_tar_gz` provided): Ground-truth Blender cameras + surface seed cloud -> gsplat
      3. `resume` (both None): Reloads `/jobs/{job_id}/frames_u8.pt` + `ckpt_latest.pt` (~0.3s or 0.0s HBM)
    """
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    t_job_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    meta_path = job_dir / "meta.json"
    ckpt_path = job_dir / "ckpt_latest.pt"
    frames_pt_path = job_dir / "frames_u8.pt"
    cams_json_path = job_dir / "cameras.json"
    telem_path = job_dir / "telemetry.jsonl"
    masks_dir = job_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    def _append_telem(row: dict[str, Any]) -> None:
        clean = {k: v for k, v in row.items() if not k.endswith("_b64") and k not in ("ply_gz", "splat_gz")}
        with open(telem_path, "a", encoding="utf-8") as tf:
            tf.write(json.dumps(clean) + "\n")

    if masks_b64:
        for mname, mb64 in masks_b64.items():
            raw_m = base64.b64decode(mb64)
            (masks_dir / mname).write_bytes(raw_m)

    is_resume = (mp4_bytes is None) and (views_tar_gz is None)
    eff_cfg = cfg or SplatConfig()
    macro_timings: dict[str, float] = {}
    alphas_gpu: torch.Tensor | None = None

    if not is_resume:
        work_shm = Path("/dev/shm") / f"splat_{job_id}" if Path("/dev/shm").is_dir() else job_dir / "tmp"
        if work_shm.exists():
            shutil.rmtree(work_shm, ignore_errors=True)
        work_shm.mkdir(parents=True, exist_ok=True)

        if views_tar_gz is not None:
            # ── Known-Pose Blender Capture Mode (0.0px pose error, exact Blender world space!) ──
            t0 = time.perf_counter()
            cam_records, xyz_bl, raw_rgb, has_alpha = _unpack_known_views_bundle(
                views_tar_gz, work_shm, max_init_points=eff_cfg.pose.max_init_points
            )
            sim3 = Sim3Transform.identity()
            pose_backend_label = "blender_exact"
            macro_timings["extract_s"] = round(time.perf_counter() - t0, 3)
            macro_timings["pose_s"] = 0.0
            yield {
                "event": "stage",
                "stage": "extract",
                "job_id": job_id,
                "device": dev_name,
                "source": "blender_views",
                "frames": len(cam_records),
                "init_points": len(xyz_bl),
                "transparent_bg": has_alpha,
                "elapsed_s": macro_timings["extract_s"],
            }
        else:
            # ── Video Reconstruction Mode: Stage 1 (Extract) + Stage 2 (Pose + Floor Z=0 Sim3) ──
            has_alpha = False
            mp4_path = work_shm / "source.mp4"
            mp4_path.write_bytes(mp4_bytes)  # type: ignore[arg-type]

            t0 = time.perf_counter()
            frames_dir = work_shm / "frames"
            frame_paths = extract_frames(mp4_path, frames_dir, eff_cfg.extract)
            macro_timings["extract_s"] = round(time.perf_counter() - t0, 3)
            yield {
                "event": "stage",
                "stage": "extract",
                "job_id": job_id,
                "device": dev_name,
                "frames": len(frame_paths),
                "elapsed_s": macro_timings["extract_s"],
            }

            t0 = time.perf_counter()
            pose_backend_label = eff_cfg.pose.backend
            if eff_cfg.pose.backend == "colmap":
                try:
                    cam_records, raw_xyz, raw_rgb = solve_poses_colmap(frames_dir, work_shm, eff_cfg.pose)
                    if len(cam_records) < max(3, len(frame_paths) // 4):
                        raise RuntimeError(f"COLMAP registered only {len(cam_records)}/{len(frame_paths)} frames")
                except Exception as col_err:  # noqa: BLE001
                    # Graceful fallback to VGGT-1B if AI video is non-rigid / low-parallax for SIFT
                    pose_backend_label = "vggt (colmap_fallback)"
                    yield {
                        "event": "stage",
                        "stage": "pose_fallback",
                        "job_id": job_id,
                        "reason": str(col_err)[:240],
                    }
                    cam_records, raw_xyz, raw_rgb = solve_poses_vggt(
                        frame_paths, eff_cfg.pose, device, models_dir=models_dir
                    )
            else:
                cam_records, raw_xyz, raw_rgb = solve_poses_vggt(
                    frame_paths, eff_cfg.pose, device, models_dir=models_dir
                )
            macro_timings["pose_s"] = round(time.perf_counter() - t0, 3)

            t0 = time.perf_counter()
            sim3 = estimate_floor_aligned_sim3(
                [c["R_w2c"] for c in cam_records],
                [c["t_w2c"] for c in cam_records],
                raw_xyz.tolist() if len(raw_xyz) <= 15_000 else raw_xyz[:: max(1, len(raw_xyz) // 15_000)].tolist(),
                align_floor=eff_cfg.pose.align_floor_z0,
                target_cam_height_m=eff_cfg.pose.target_cam_height_m,
            )
            r_np = np.array(sim3.R, dtype=np.float64)
            t_np = np.array(sim3.t, dtype=np.float64)
            xyz_bl = sim3.scale * (raw_xyz @ r_np.T) + t_np[None, :]

        # Build aligned OpenCV viewmats/Ks & Blender `cameras.json`
        t_align = time.perf_counter()
        bl_cameras_list: list[dict[str, Any]] = []
        viewmats_list: list[np.ndarray] = []
        ks_list: list[np.ndarray] = []
        for c in cam_records:
            rc_new, tc_new = apply_sim3_to_camera_w2c(sim3, c["R_w2c"], c["t_w2c"])
            vm = np.eye(4, dtype=np.float32)
            vm[:3, :3] = np.array(rc_new, dtype=np.float32)
            vm[:3, 3] = np.array(tc_new, dtype=np.float32)
            viewmats_list.append(vm)
            k_mat = np.array(
                [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            ks_list.append(k_mat)

            bl_cam = opencv_w2c_to_blender_camera(
                rc_new, tc_new, c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"]
            )
            bl_cam["frame_index"] = c["frame_index"]
            bl_cam["image_name"] = c["image_name"]
            bl_cameras_list.append(bl_cam)

        cameras_doc = {
            "job_id": job_id,
            "coordinate_system": "BLENDER_RIGHT_HANDED_Z_UP",
            "floor_z": 0.0,
            "pose_backend": pose_backend_label,
            "sim3": sim3.to_dict(),
            "low_parallax_warning": (views_tar_gz is None) and (sim3.baseline_ratio < 0.03),
            "cameras": bl_cameras_list,
        }
        cams_json_path.write_text(json.dumps(cameras_doc, indent=2), encoding="utf-8")

        frames_u8_cpu, alphas_u8_cpu = _preload_frames_u8(cam_records, load_alpha=has_alpha)
        shutil.rmtree(work_shm, ignore_errors=True)

        torch.save(
            {
                "frames_u8": frames_u8_cpu,
                "alphas_u8": alphas_u8_cpu,
                "viewmats": torch.from_numpy(np.stack(viewmats_list, axis=0)),
                "Ks": torch.from_numpy(np.stack(ks_list, axis=0)),
                "image_names": [c["image_name"] for c in cam_records],
                "width": int(cam_records[0]["width"]),
                "height": int(cam_records[0]["height"]),
            },
            frames_pt_path,
        )
        macro_timings["align_s"] = round(time.perf_counter() - t_align, 3)

        if views_tar_gz is None:
            yield {
                "event": "stage",
                "stage": "pose",
                "job_id": job_id,
                "cameras": len(cam_records),
                "init_points": len(xyz_bl),
                "floor_inlier_ratio": round(sim3.floor_inlier_ratio, 3),
                "baseline_ratio": round(sim3.baseline_ratio, 4),
                "elapsed_s": round(macro_timings["pose_s"] + macro_timings["align_s"], 3),
            }

        frames_gpu = frames_u8_cpu.to(device, non_blocking=True)
        alphas_gpu = alphas_u8_cpu.to(device, non_blocking=True) if alphas_u8_cpu is not None else None
        viewmats_gpu = torch.from_numpy(np.stack(viewmats_list, axis=0)).to(device)
        ks_gpu = torch.from_numpy(np.stack(ks_list, axis=0)).to(device)
        image_names = [c["image_name"] for c in cam_records]
        img_w, img_h = int(cam_records[0]["width"]), int(cam_records[0]["height"])

        params = _init_gaussians_cuda(xyz_bl, raw_rgb, eff_cfg.train.sh_degree, device)
        scene_scale = max(1.0, float(np.percentile(np.linalg.norm(xyz_bl, axis=1), 90)))
        optimizers = _make_optimizers(params, scene_scale=scene_scale)
        eff_densify_stop = min(
            int(eff_cfg.train.densify_until),
            max(int(eff_cfg.train.densify_from) + 200, int(eff_cfg.train.iterations * 0.72)),
        )
        strategy = DefaultStrategy(
            prune_opa=eff_cfg.train.prune_opa,
            grow_grad2d=eff_cfg.train.densify_grad_threshold,
            grow_scale3d=0.01,
            prune_scale3d=float(getattr(eff_cfg.train, "prune_scale3d", 0.05)),
            refine_start_iter=eff_cfg.train.densify_from,
            refine_stop_iter=eff_densify_stop,
            refine_every=eff_cfg.train.densify_every,
            absgrad=True,
            verbose=False,
        )
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        start_step = 0
        target_step = eff_cfg.train.iterations
        burst_id = 1
        cum_gpu_sec_prev = 0.0
        excluded_set: set[str] = set(exclude_frames or [])

        meta = {
            "job_id": job_id,
            "config": eff_cfg.to_dict(),
            "burst_id": burst_id,
            "step": 0,
            "cum_gpu_sec": 0.0,
            "scene_scale": round(scene_scale, 4),
            "excluded": sorted(excluded_set),
            "macro_timings": macro_timings,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if telem_path.exists():
            telem_path.unlink()

    else:
        # ── Stateful Burst Resume: Reload from HBM `ram_cache` (0.0s) or NVMe Volume (~0.3s) ──
        if not meta_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(f"Job '{job_id}' has no saved checkpoint in {job_dir}.")

        t0 = time.perf_counter()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_cfg = SplatConfig.from_dict(meta.get("config"))
        eff_cfg = cfg if cfg is not None else saved_cfg
        scene_scale = float(meta.get("scene_scale", 2.5))

        if ram_cache is not None and ram_cache.get("job_id") == job_id:
            frames_gpu = ram_cache["frames_gpu"]
            alphas_gpu = ram_cache.get("alphas_gpu")
            viewmats_gpu = ram_cache["viewmats_gpu"]
            ks_gpu = ram_cache["ks_gpu"]
            image_names = ram_cache["image_names"]
            img_w, img_h = ram_cache["width"], ram_cache["height"]
            params = ram_cache["params"]
            optimizers = ram_cache["optimizers"]
            strategy_state = ram_cache["strategy_state"]
            start_step = int(ram_cache["step"])
            cache_hit = True
        else:
            pack = torch.load(frames_pt_path, map_location="cpu", weights_only=False)
            frames_gpu = pack["frames_u8"].to(device, non_blocking=True)
            alphas_u8_loaded = pack.get("alphas_u8")
            alphas_gpu = alphas_u8_loaded.to(device, non_blocking=True) if alphas_u8_loaded is not None else None
            viewmats_gpu = pack["viewmats"].to(device)
            ks_gpu = pack["Ks"].to(device)
            image_names = list(pack["image_names"])
            img_w, img_h = int(pack["width"]), int(pack["height"])

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            start_step = int(ckpt["step"])
            params = torch.nn.ParameterDict({
                k: torch.nn.Parameter(v.to(device).requires_grad_(True))
                for k, v in ckpt["params"].items()
            })
            optimizers = _make_optimizers(params, scene_scale=scene_scale)
            for k, st in ckpt.get("opt_states", {}).items():
                if k in optimizers:
                    optimizers[k].load_state_dict(st)
            strategy_state = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in (ckpt.get("strategy_state") or {}).items()
            }
            cache_hit = False

        want_k_rest = (max(0, eff_cfg.train.sh_degree) + 1) ** 2 - 1
        cur_k_rest = params["shN"].shape[1]
        if want_k_rest > cur_k_rest:
            pad = torch.zeros(
                (params["shN"].shape[0], want_k_rest - cur_k_rest, 3),
                dtype=torch.float32,
                device=device,
            )
            new_shn = torch.nn.Parameter(torch.cat([params["shN"].detach(), pad], dim=1))
            params["shN"] = new_shn
            optimizers["shN"] = torch.optim.Adam([new_shn], lr=2.5e-3 / 20.0, eps=1e-15)

        eff_densify_stop = start_step + int((add_steps or 2000) * 0.65)
        strategy = DefaultStrategy(
            prune_opa=eff_cfg.train.prune_opa,
            grow_grad2d=eff_cfg.train.densify_grad_threshold,
            grow_scale3d=0.01,
            prune_scale3d=float(getattr(eff_cfg.train, "prune_scale3d", 0.05)),
            refine_start_iter=eff_cfg.train.densify_from,
            refine_stop_iter=eff_densify_stop,
            refine_every=eff_cfg.train.densify_every,
            absgrad=True,
            verbose=False,
        )
        if not strategy_state:
            strategy_state = strategy.initialize_state(scene_scale=scene_scale)

        burst_id = int(meta.get("burst_id", 1)) + 1
        cum_gpu_sec_prev = float(meta.get("cum_gpu_sec", 0.0))
        excluded_set = set(meta.get("excluded") or [])
        if exclude_frames is not None:
            excluded_set = set(exclude_frames)

        target_step = start_step + int(add_steps if add_steps is not None else 2_000)
        macro_timings["reload_s"] = round(time.perf_counter() - t0, 3)

        yield {
            "event": "stage",
            "stage": "resume_load",
            "job_id": job_id,
            "burst_id": burst_id,
            "start_step": start_step,
            "target_step": target_step,
            "vram_cache_hit": cache_hit,
            "elapsed_s": macro_timings["reload_s"],
        }

        if surgery is not None:
            surg_res = apply_surgeries(params, optimizers, strategy_state, surgery)
            surg_ev = {
                "event": "surgery",
                "job_id": job_id,
                "burst_id": burst_id,
                "step": start_step,
                **surg_res,
            }
            _append_telem(surg_ev)
            yield surg_ev

    # ── Load optional masks onto GPU ──
    mask_tensors: dict[int, torch.Tensor] = {}
    global_mask_t: torch.Tensor | None = None
    gm_path = masks_dir / "global_mask.png"
    if gm_path.exists():
        gm_arr = np.asarray(Image.open(gm_path).convert("L").resize((img_w, img_h)), dtype=np.float32) / 255.0
        global_mask_t = torch.from_numpy(gm_arr).to(device).unsqueeze(-1)
    for idx, nm in enumerate(image_names):
        mp = masks_dir / f"{Path(nm).stem}.png"
        if mp.exists():
            m_arr = np.asarray(Image.open(mp).convert("L").resize((img_w, img_h)), dtype=np.float32) / 255.0
            mt = torch.from_numpy(m_arr).to(device).unsqueeze(-1)
            mask_tensors[idx] = torch.minimum(mt, global_mask_t) if global_mask_t is not None else mt

    active_indices = [i for i, nm in enumerate(image_names) if nm not in excluded_set]
    if not active_indices:
        raise RuntimeError("All training frames are excluded!")

    # ── Stage 3: High-Throughput VRAM Training Loop ──
    ssim_win = _build_ssim_window(device)
    t_train_start = time.perf_counter()
    last_preview_t = 0.0
    min_preview_dt = 1.0 / max(0.25, eff_cfg.train.preview_hz)
    sent_gt_cams: set[str] = set()
    rng_py = random.Random(1337 + burst_id)

    use_cuda_ev = device.type == "cuda"
    if use_cuda_ev:
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_fwd = torch.cuda.Event(enable_timing=True)
        ev_loss = torch.cuda.Event(enable_timing=True)
        ev_bwd = torch.cuda.Event(enable_timing=True)
        ev_opt = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

    focus_idx: int | None = None
    if eff_cfg.train.focus_camera:
        for i, nm in enumerate(image_names):
            if nm == eff_cfg.train.focus_camera:
                focus_idx = i
                break

    for step in range(start_step, target_step):
        is_log = (step == start_step) or ((step + 1) % max(1, eff_cfg.train.log_every) == 0) or (step == target_step - 1)
        t_step_cpu = time.perf_counter() if is_log else 0.0

        cam_i = focus_idx if focus_idx is not None else rng_py.choice(active_indices)
        viewmat = viewmats_gpu[cam_i : cam_i + 1]
        k_mat = ks_gpu[cam_i : cam_i + 1]
        fg_img = frames_gpu[cam_i].float() * (1.0 / 255.0)
        gt_alpha = (alphas_gpu[cam_i].float() * (1.0 / 255.0)) if alphas_gpu is not None else None

        # Random solid background compositing when RGBA silhouette views are provided:
        # forces exact 0 opacity outside object silhouettes with zero background floaters!
        bg_col: torch.Tensor | None = None
        if gt_alpha is not None:
            bg_col = torch.zeros((1, 3), device=device) if is_log else torch.rand((1, 3), device=device)
            gt_img = fg_img * gt_alpha + bg_col.view(1, 1, 3) * (1.0 - gt_alpha)
        else:
            gt_img = fg_img

        sh_cur = min(eff_cfg.train.sh_degree, step // 1000)
        k_use = (sh_cur + 1) ** 2

        if is_log and use_cuda_ev:
            ev_s.record()

        sh_all = torch.cat([params["sh0"], params["shN"]], dim=1)[:, :k_use, :]
        opa_sig = torch.sigmoid(params["opacities"])
        scales_lin = torch.exp(params["scales"])

        renders, ren_alphas, info = rasterization(
            means=params["means"],
            quats=F.normalize(params["quats"], dim=-1),
            scales=scales_lin,
            opacities=opa_sig,
            colors=sh_all,
            viewmats=viewmat,
            Ks=k_mat,
            width=img_w,
            height=img_h,
            sh_degree=sh_cur,
            backgrounds=bg_col,
            packed=True,
            absgrad=True,
        )
        render_img = renders[0]

        if is_log and use_cuda_ev:
            ev_fwd.record()

        mask_t = mask_tensors.get(cam_i, global_mask_t)
        if mask_t is not None:
            l1_term = ((render_img - gt_img).abs() * mask_t).sum() / (mask_t.sum() * 3.0 + 1e-8)
            masked_r = render_img * mask_t + gt_img * (1.0 - mask_t)
            ssim_val = _ssim_cuda(masked_r, gt_img, ssim_win) if eff_cfg.train.lambda_dssim > 0 else torch.tensor(1.0, device=device)
        else:
            l1_term = (render_img - gt_img).abs().mean()
            ssim_val = _ssim_cuda(render_img, gt_img, ssim_win) if eff_cfg.train.lambda_dssim > 0 else torch.tensor(1.0, device=device)

        lam = eff_cfg.train.lambda_dssim
        photo_loss = (1.0 - lam) * l1_term + lam * (1.0 - ssim_val) if lam > 0 else l1_term
        alpha_sil_loss = (ren_alphas[0] - gt_alpha).abs().mean() * 0.15 if gt_alpha is not None else 0.0
        opa_reg_term = eff_cfg.train.opa_reg * opa_sig.mean() if eff_cfg.train.opa_reg > 0 else 0.0
        scale_reg_term = eff_cfg.train.scale_reg * scales_lin.mean() if eff_cfg.train.scale_reg > 0 else 0.0
        aniso_w = float(getattr(eff_cfg.train, "aniso_reg", 0.0))
        if aniso_w > 0.0:
            s_sorted, _ = torch.sort(scales_lin, dim=-1)
            aniso_ratio = s_sorted[:, 2] / (s_sorted[:, 1] + 1e-6)
            max_aniso = float(getattr(eff_cfg.train, "max_anisotropy", 6.0))
            aniso_term = aniso_w * F.relu(aniso_ratio - max_aniso).mean()
        else:
            aniso_term = 0.0
        loss = photo_loss + alpha_sil_loss + opa_reg_term + scale_reg_term + aniso_term

        if is_log and use_cuda_ev:
            ev_loss.record()

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()

        if is_log and use_cuda_ev:
            ev_bwd.record()

        # Exponential position LR decay
        t_prog = min(1.0, step / float(max(1, target_step)))
        pos_lr = 1.6e-4 * ((1.6e-6 / 1.6e-4) ** t_prog)
        for pg in optimizers["means"].param_groups:
            pg["lr"] = pos_lr

        for opt in optimizers.values():
            opt.step()

        if is_log and use_cuda_ev:
            ev_opt.record()

        n_pre_dens = int(params["means"].shape[0])
        force_now = bool(surgery and surgery.densify_now and step == start_step)
        eff_step_for_strat = (
            eff_cfg.train.densify_from
            if force_now
            else step
        )
        strategy.step_post_backward(
            params, optimizers, strategy_state, eff_step_for_strat, info, packed=True
        )
        n_post_dens = int(params["means"].shape[0])

        if (
            eff_cfg.train.opacity_reset_every > 0
            and step > 0
            and step % eff_cfg.train.opacity_reset_every == 0
            and step < eff_densify_stop
            and (target_step - step) >= 400
        ):
            with torch.no_grad():
                params["opacities"].data.clamp_(max=math.log(0.01 / 0.99))

        if n_post_dens != n_pre_dens:
            dens_ev = {
                "event": "densify",
                "job_id": job_id,
                "burst_id": burst_id,
                "step": step + 1,
                "n_before": n_pre_dens,
                "n_after": n_post_dens,
                "delta": n_post_dens - n_pre_dens,
            }
            _append_telem(dens_ev)
            yield dens_ev

        if is_log:
            if use_cuda_ev:
                ev_end.record()
                torch.cuda.synchronize(device)
                fwd_ms = ev_s.elapsed_time(ev_fwd)
                loss_ms = ev_fwd.elapsed_time(ev_loss)
                bwd_ms = ev_loss.elapsed_time(ev_bwd)
                adam_ms = ev_bwd.elapsed_time(ev_opt)
                dens_ms = ev_opt.elapsed_time(ev_end)
                gpu_step_ms = ev_s.elapsed_time(ev_end)
                wall_step_ms = (time.perf_counter() - t_step_cpu) * 1e3
                host_ms = max(0.0, wall_step_ms - gpu_step_ms)
                stage_timings = {
                    "fwd_raster_ms": round(fwd_ms, 3),
                    "loss_ssim_ms": round(loss_ms, 3),
                    "bwd_ms": round(bwd_ms, 3),
                    "adam_ms": round(adam_ms, 3),
                    "densify_ms": round(dens_ms, 3),
                    "host_ms": round(host_ms, 3),
                    "step_ms": round(wall_step_ms, 3),
                }
            else:
                wall_step_ms = (time.perf_counter() - t_step_cpu) * 1e3
                stage_timings = {"step_ms": round(wall_step_ms, 3)}

            with torch.no_grad():
                r_det = render_img.detach()
                mse_val = float(((r_det - gt_img) ** 2).mean().item())
                psnr_val = -10.0 * math.log10(max(mse_val, 1e-10))
                l1_val = float(l1_term.detach().item())
                ssim_f = float(ssim_val.detach().item())
                loss_f = float(loss.detach().item())

            elapsed_train = time.perf_counter() - t_train_start
            steps_run = max(1, (step + 1) - start_step)
            it_s = steps_run / max(elapsed_train, 1e-4)
            cum_gpu = cum_gpu_sec_prev + elapsed_train
            cam_name = image_names[cam_i]

            tick: dict[str, Any] = {
                "event": "telemetry",
                "job_id": job_id,
                "burst_id": burst_id,
                "step": step + 1,
                "total_steps": target_step,
                "cum_gpu_sec": round(cum_gpu, 3),
                "it_per_sec": round(it_s, 1),
                "loss": round(loss_f, 5),
                "l1": round(l1_val, 5),
                "ssim": round(ssim_f, 5),
                "mse": round(mse_val, 6),
                "psnr": round(psnr_val, 2),
                "n_gaussians": n_post_dens,
                "sh_degree_cur": sh_cur,
                "pos_lr": round(pos_lr, 7),
                "camera_name": cam_name,
                "stage_timings": stage_timings,
            }

            now_t = time.perf_counter()
            is_snap = (
                eff_cfg.train.snapshot_every > 0
                and (step + 1) % eff_cfg.train.snapshot_every == 0
            )
            want_preview = (
                step == start_step
                or step == target_step - 1
                or is_snap
                or (now_t - last_preview_t) >= min_preview_dt
            )
            if want_preview:
                last_preview_t = now_t
                r_np_im = r_det.float().cpu().numpy()
                g_np_im = gt_img.float().cpu().numpy()
                tick["render_b64"] = _encode_jpeg_b64(r_np_im)
                tick["diff_b64"] = _encode_diff_b64(r_np_im, g_np_im)
                if cam_name not in sent_gt_cams:
                    tick["gt_b64"] = _encode_jpeg_b64(g_np_im)
                    sent_gt_cams.add(cam_name)
                tick["is_snapshot"] = is_snap or (step == target_step - 1)

            _append_telem(tick)
            yield tick

    macro_timings["train_s"] = round(time.perf_counter() - t_train_start, 3)

    # ── Save Resumable Checkpoint + Update In-VRAM Cache ──
    t_exp = time.perf_counter()
    torch.save(
        {
            "step": target_step,
            "params": {k: v.detach().cpu() for k, v in params.items()},
            "opt_states": {k: opt.state_dict() for k, opt in optimizers.items()},
            "strategy_state": {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in strategy_state.items()
            },
        },
        ckpt_path,
    )
    if ram_cache is not None:
        ram_cache.clear()
        ram_cache.update({
            "job_id": job_id,
            "frames_gpu": frames_gpu,
            "viewmats_gpu": viewmats_gpu,
            "ks_gpu": ks_gpu,
            "image_names": image_names,
            "width": img_w,
            "height": img_h,
            "params": params,
            "optimizers": optimizers,
            "strategy_state": strategy_state,
            "step": target_step,
        })

    total_cum_gpu = cum_gpu_sec_prev + macro_timings["train_s"] + macro_timings.get("pose_s", 0.0)
    meta.update({
        "config": eff_cfg.to_dict(),
        "burst_id": burst_id,
        "step": target_step,
        "cum_gpu_sec": round(total_cum_gpu, 3),
        "n_gaussians": int(params["means"].shape[0]),
        "excluded": sorted(excluded_set),
        "macro_timings": macro_timings,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ── Stage 4: Export `.ply` + `.splat` + `cameras.json` ──
    ply_bytes, n_exported = export_ply_bytes(
        params,
        max_gaussians=eff_cfg.export.max_gaussians,
        sort_by_opacity=eff_cfg.export.sort_by_opacity,
    )
    splat_bytes = (
        export_splat_bytes(params, max_gaussians=eff_cfg.export.max_gaussians)
        if eff_cfg.export.export_splat
        else b""
    )
    macro_timings["export_s"] = round(time.perf_counter() - t_exp, 3)

    wall_s = round(time.perf_counter() - t_job_start, 3)
    cost_usd = round((wall_s / 3600.0) * gpu_hourly_usd, 5)
    cams_text = cams_json_path.read_text(encoding="utf-8")

    done_ev: dict[str, Any] = {
        "event": "done",
        "job_id": job_id,
        "burst_id": burst_id,
        "step": target_step,
        "n_gaussians": n_exported,
        "device": dev_name,
        "wall_s": wall_s,
        "cost_usd": cost_usd,
        "macro_timings": macro_timings,
        "cameras_json": cams_text,
        "ply_gz": gzip.compress(ply_bytes, compresslevel=2),
        "splat_gz": gzip.compress(splat_bytes, compresslevel=2) if splat_bytes else b"",
        "ply_raw_bytes": len(ply_bytes),
        "splat_raw_bytes": len(splat_bytes),
    }
    _append_telem(done_ev)
    yield done_ev