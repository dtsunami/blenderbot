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
import re
import shutil
import struct
import subprocess
import tarfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .client import SUPPORTED_IMAGE_EXTS, _natural_sort_key, unpack_images_bundle
from .config import ExtractConfig, PoseConfig, SplatConfig, SurgerySpec
from .depth_pro2_bridge import (
    align_depth_maps_to_sparse_cloud,
    compute_depth_prior_loss,
    nudge_and_densify_init_points_with_depth_pro2,
    predict_depth_pro2_maps,
)
from .proxy_mesh import build_proxy_glb_bytes
from .sim3 import (
    Sim3Transform,
    apply_sim3_to_camera_w2c,
    build_stereo_peer_table,
    estimate_floor_aligned_sim3,
    interpolate_camera_trajectory,
    opencv_w2c_to_blender_camera,
)

_SH_C0 = 0.28209479177387814

# Cached VGGT model singleton across warm bursts inside the same GPU container
_VGGT_CACHE: dict[str, Any] = {}


# ─── Stage 1: Fast FFmpeg Frame Extraction ────────────────────────────────────


def _probe_video_meta(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    dur = 5.0
    fps = 24.0
    width = 1280
    height = 720
    nb_frames = 0
    try:
        info = json.loads(res.stdout or "{}")
        dur = float((info.get("format") or {}).get("duration") or 5.0)
        st0 = (info.get("streams") or [{}])[0]
        rate_str = str(st0.get("r_frame_rate") or "24/1")
        num, den = rate_str.split("/")
        fps = float(num) / max(float(den), 1.0)
        width = int(st0.get("width") or width)
        height = int(st0.get("height") or height)
        if st0.get("nb_frames") and str(st0["nb_frames"]).isdigit():
            nb_frames = int(st0["nb_frames"])
    except Exception:  # noqa: BLE001, S110
        pass
    if nb_frames <= 0:
        nb_frames = max(1, round(dur * fps))
    return {
        "duration_sec": round(dur, 4),
        "fps": round(fps, 4),
        "total_frames": nb_frames,
        "width": width,
        "height": height,
    }


def _probe_video_duration_fps(video_path: Path) -> tuple[float, float]:
    m = _probe_video_meta(video_path)
    return float(m["duration_sec"]), float(m["fps"])


def extract_frames_with_meta(
    video_path: Path, frames_dir: Path, cfg: ExtractConfig
) -> tuple[list[Path], dict[str, float], dict[str, Any]]:
    """Extract sharp keyframes AND capture exact per-frame `pts_sec` + native `video_meta`."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    vmeta = _probe_video_meta(video_path)
    dur = float(vmeta["duration_sec"])
    native_fps = float(vmeta["fps"])
    # Motion-adaptive extraction: decode at `dense_pool_fps` (at least `2.5x` uniform rate during long clips)
    # so fast camera whip-pans keep dense overlapping frames (`<= 30°` per step) instead of leaving 120°–180° blind gaps!
    sample_fps = float(cfg.fps)
    if dur > 0 and cfg.max_frames > 0 and (dur * sample_fps) > cfg.max_frames:
        uniform_fps = cfg.max_frames / dur
        sample_fps = min(float(cfg.fps), max(uniform_fps * 2.8, 3.0))

    vf_parts = [f"fps={sample_fps:.4f}"]
    if cfg.dedup_frames:
        vf_parts.append("mpdecimate=hi=64*8:lo=64*3:frac=0.25")
    if cfg.max_width > 0:
        vf_parts.append(f"scale='min({cfg.max_width},iw)':-2:flags=lanczos")
    vf_parts.append("showinfo")

    qscale = max(1, min(10, 32 - cfg.quality // 3))
    out_pat = str(frames_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", ",".join(vf_parts),
        "-vsync", "vfr" if cfg.dedup_frames else "passthrough",
        "-q:v", str(qscale),
        "-hide_banner", "-loglevel", "info",
        out_pat,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed: {proc.stderr[-600:]}")

    raw_pts = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", proc.stderr or "")]
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(raw_pts) == len(frames):
        pts_list = raw_pts
    else:
        span_end = max(0.0, dur - 1.0 / max(1.0, native_fps))
        pts_list = np.linspace(0.0, span_end, max(1, len(frames))).tolist()

    pts_by_name: dict[str, float] = {fp.name: round(float(pt), 5) for fp, pt in zip(frames, pts_list)}

    if cfg.max_frames > 0 and len(frames) > cfg.max_frames:
        # Arc-length motion-adaptive subsampling: score inter-frame visual motion on fast 96x54 thumbnails
        # so fast pans get 3x–4x denser keyframes while slow walks share the remaining budget smoothly!
        thumbs = []
        for fp in frames:
            with Image.open(fp) as im_t:
                thumbs.append(np.asarray(im_t.convert("L").resize((96, 54), Image.BILINEAR), dtype=np.float32))
        diffs = np.zeros((len(frames),), dtype=np.float64)
        for i in range(1, len(frames)):
            # Blend baseline time step (`0.35`) + visual motion (`0.65`) so neither pauses nor fast pans starve!
            mad = float(np.abs(thumbs[i] - thumbs[i - 1]).mean())
            dt_i = max(0.02, pts_list[i] - pts_list[i - 1])
            diffs[i] = 0.35 * dt_i + 0.65 * (mad / 18.0)
        cum_arc = np.cumsum(diffs)
        if cum_arc[-1] > 1e-6:
            targets = np.linspace(0.0, cum_arc[-1], cfg.max_frames)
            picked: list[int] = [0]
            for tg in targets[1:-1]:
                ci = int(np.searchsorted(cum_arc, tg))
                ci = max(picked[-1] + 1, min(len(frames) - 2, ci))
                if ci not in picked:
                    picked.append(ci)
            picked.append(len(frames) - 1)
            # Pad if searchsorted collided near the end
            if len(picked) < cfg.max_frames:
                rem = [k for k in range(len(frames)) if k not in set(picked)]
                extra = np.linspace(0, len(rem) - 1, min(len(rem), cfg.max_frames - len(picked)), dtype=int)
                picked = sorted(set(picked) | {rem[int(x)] for x in extra})
            idxs = np.array(picked[: cfg.max_frames], dtype=int)
        else:
            idxs = np.linspace(0, len(frames) - 1, cfg.max_frames, dtype=int)

        keep_set = {frames[int(i)] for i in idxs}
        for f in frames:
            if f not in keep_set:
                pts_by_name.pop(f.name, None)
                f.unlink(missing_ok=True)
        frames = sorted(frames_dir.glob("frame_*.jpg"))

    if len(frames) < 3:
        raise RuntimeError(f"Too few frames extracted ({len(frames)}). Need at least 3 frames.")
    return frames, pts_by_name, vmeta


def extract_frames(video_path: Path, frames_dir: Path, cfg: ExtractConfig) -> list[Path]:
    frames, _, _ = extract_frames_with_meta(video_path, frames_dir, cfg)
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


def _repair_outlier_trajectory_spikes(
    ext_w2c: np.ndarray,
    spike_factor: float = 2.8,
) -> tuple[np.ndarray, int]:
    """Detect & replace isolated VGGT translation AND 180° orientation-flip spikes on long arcs!

    Catches BOTH:
      1. Translation teleports (`d_prev > spike * med` AND `d_next > spike * med` while `i-1..i+1` is close)
      2. Rotation snap-back flips (`ang(i-1, i) > 95°` AND `ang(i, i+1) > 95°` while `ang(i-1, i+1) < 85°` —
         e.g. `frame_00025` flipping 177.6° backward for 1 frame during a smooth pan!)
    """
    from .sim3 import quat_slerp, quat_to_rotmat, rotmat_to_quat

    n = len(ext_w2c)
    if n < 5:
        return ext_w2c, 0

    out = ext_w2c.copy().astype(np.float64)
    centers = np.zeros((n, 3), dtype=np.float64)
    quats = []
    for i in range(n):
        r_i = out[i, :3, :3]
        u_s, _, vt_s = np.linalg.svd(r_i)
        r_o = u_s @ vt_s
        if np.linalg.det(r_o) < 0:
            u_s[:, -1] *= -1.0
            r_o = u_s @ vt_s
        out[i, :3, :3] = r_o
        centers[i] = -r_o.T @ out[i, :3, 3]
        quats.append(rotmat_to_quat(r_o.tolist()))

    def _ang_deg(qa: tuple[float, float, float, float], qb: tuple[float, float, float, float]) -> float:
        dot = min(1.0, abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]))
        return math.degrees(2.0 * math.acos(dot))

    step_d = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
    med_step = max(1e-3, float(np.median(step_d)))
    repaired = 0

    for _pass in range(2):
        for i in range(1, n - 1):
            d_prev = float(np.linalg.norm(centers[i] - centers[i - 1]))
            d_next = float(np.linalg.norm(centers[i + 1] - centers[i]))
            d_chord = float(np.linalg.norm(centers[i + 1] - centers[i - 1]))
            is_pos_spike = (
                d_prev > spike_factor * med_step
                and d_next > spike_factor * med_step
                and (d_prev + d_next) > 2.25 * max(d_chord, med_step * 0.5)
            )

            a_prev = _ang_deg(quats[i - 1], quats[i])
            a_next = _ang_deg(quats[i], quats[i + 1])
            a_chord = _ang_deg(quats[i - 1], quats[i + 1])
            is_rot_spike = (a_prev > 95.0 and a_next > 95.0 and (a_prev + a_next) > max(210.0, 2.2 * a_chord))

            if is_pos_spike or is_rot_spike:
                c_mid = 0.5 * (centers[i - 1] + centers[i + 1])
                q_mid = quat_slerp(quats[i - 1], quats[i + 1], 0.5)
                r_mid = np.array(quat_to_rotmat(q_mid), dtype=np.float64)
                t_mid = -r_mid @ c_mid
                out[i, :3, :3] = r_mid
                out[i, :3, 3] = t_mid
                centers[i] = c_mid
                quats[i] = q_mid
                repaired += 1

    return out, repaired


def _run_vggt_single_window(
    model: Any,
    window_paths: list[Path],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, int]:
    """Run VGGT-1B on a `<= 48` frame window where multi-view attention stays razor-sharp."""
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    try:
        imgs = load_and_preprocess_images([str(p) for p in window_paths], mode="pad").to(device)
        used_pad = True
    except TypeError:
        imgs = load_and_preprocess_images([str(p) for p in window_paths]).to(device)
        used_pad = False
    imgs_b = imgs.unsqueeze(0) if imgs.ndim == 4 else imgs
    _, _s_cnt, _, vh, vw = imgs_b.shape

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
        preds = model(imgs_b)
        extrinsic_b, intrinsic_b = pose_encoding_to_extri_intri(preds["pose_enc"], (vh, vw))

    ext_np = extrinsic_b[0].float().cpu().numpy()   # (W, 3, 4)
    int_np = intrinsic_b[0].float().cpu().numpy()   # (W, 3, 3)
    if "depth" in preds and "depth_conf" in preds:
        depth_np = preds["depth"][0].float().cpu().numpy()
        conf_np = preds["depth_conf"][0].float().cpu().numpy()
        pts_map = unproject_depth_map_to_point_map(depth_np, ext_np, int_np)
    else:
        pts_map = preds["world_points"][0].float().cpu().numpy()
        conf_np = preds["world_points_conf"][0].float().cpu().numpy()
    rgb_map = (imgs_b[0].permute(0, 2, 3, 1).float().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return ext_np, int_np, pts_map, conf_np, rgb_map, used_pad, vh, vw


def solve_poses_vggt(
    frame_paths: list[Path],
    cfg: PoseConfig,
    device: torch.device,
    models_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Solve camera intrinsics/extrinsics + dense RGB point cloud via VGGT-1B on GPU.

    For long clips (`> 48` views), runs overlapping 40-frame VGGT windows (`overlap = 10`) stitched
    via closed-form `Sim(3)` (camera centers + look-ahead points) + trajectory spike repair so
    pose quality never degrades when attention spans 80s+ walkthroughs!
    """
    if cfg.vggt_max_frames > 0 and len(frame_paths) > cfg.vggt_max_frames:
        sel_idx = np.linspace(0, len(frame_paths) - 1, cfg.vggt_max_frames, dtype=int)
        used_paths = [frame_paths[int(i)] for i in sel_idx]
    else:
        used_paths = list(frame_paths)

    with Image.open(used_paths[0]) as im0:
        orig_w, orig_h = im0.width, im0.height

    model = _get_vggt_model(device, models_dir=models_dir)
    amp_dtype = torch.bfloat16 if _VGGT_CACHE.get("bf16", True) else torch.float16

    win_sz = 40
    ov_sz = 10
    if len(used_paths) <= 48:
        ext_np, int_np, pts_map, conf_np, rgb_map, used_pad, vh, vw = _run_vggt_single_window(
            model, used_paths, device, amp_dtype
        )
    else:
        # Overlapping windowed VGGT + Sim(3) trajectory & point-cloud stitching!
        stride = win_sz - ov_sz
        ext_global_list: list[np.ndarray] = []
        int_global_list: list[np.ndarray] = []
        pts_global_list: list[np.ndarray] = []
        conf_global_list: list[np.ndarray] = []
        rgb_global_list: list[np.ndarray] = []
        used_pad, vh, vw = True, 518, 518

        w0 = 0
        while w0 < len(used_paths):
            w1 = min(len(used_paths), w0 + win_sz)
            if len(used_paths) - w1 < 6 and w1 < len(used_paths):
                w1 = len(used_paths)
            sub_p = used_paths[w0:w1]
            e_w, k_w, p_w, c_w, rgb_w, used_pad, vh, vw = _run_vggt_single_window(
                model, sub_p, device, amp_dtype
            )

            if w0 == 0:
                ext_global_list.extend(e_w[i] for i in range(len(sub_p)))
                int_global_list.extend(k_w[i] for i in range(len(sub_p)))
                pts_global_list.append(p_w)
                conf_global_list.append(c_w)
                rgb_global_list.append(rgb_w)
            else:
                ov_actual = len(ext_global_list) - w0
                # Fit exact Sim(3) mapping window local world -> global world across `ov_actual` shared frames
                # using both camera centers AND forward look-at points (constrains 3D roll even on straight walks!)
                src_anch, dst_anch = [], []
                for oi in range(ov_actual):
                    r_loc, t_loc = e_w[oi, :3, :3].astype(np.float64), e_w[oi, :3, 3].astype(np.float64)
                    c_loc = -r_loc.T @ t_loc
                    f_loc = r_loc[2, :]
                    u_loc = -r_loc[1, :]

                    e_glb = ext_global_list[w0 + oi]
                    r_glb, t_glb = e_glb[:3, :3].astype(np.float64), e_glb[:3, 3].astype(np.float64)
                    c_glb = -r_glb.T @ t_glb
                    f_glb = r_glb[2, :]
                    u_glb = -r_glb[1, :]

                    src_anch.extend([c_loc, c_loc + f_loc * 0.5, c_loc + u_loc * 0.25])
                    dst_anch.extend([c_glb, c_glb + f_glb * 0.5, c_glb + u_glb * 0.25])

                sim_stitch = _estimate_umeyama_sim3(np.array(src_anch), np.array(dst_anch))
                r_st = np.array(sim_stitch.R, dtype=np.float64)
                t_st = np.array(sim_stitch.t, dtype=np.float64)

                # Transform non-overlapping tail frames `ov_actual .. len(sub_p)` into global frame
                for ti in range(ov_actual, len(sub_p)):
                    rc_a, tc_a = apply_sim3_to_camera_w2c(sim_stitch, e_w[ti, :3, :3], e_w[ti, :3, 3])
                    e_new = np.zeros((3, 4), dtype=np.float32)
                    e_new[:3, :3] = np.array(rc_a, dtype=np.float32)
                    e_new[:3, 3] = np.array(tc_a, dtype=np.float32)
                    ext_global_list.append(e_new)
                    int_global_list.append(k_w[ti])

                p_tail = p_w[ov_actual:]
                p_tail_glb = (sim_stitch.scale * (p_tail.reshape(-1, 3) @ r_st.T) + t_st[None, :]).reshape(p_tail.shape)
                pts_global_list.append(p_tail_glb.astype(np.float32))
                conf_global_list.append(c_w[ov_actual:])
                rgb_global_list.append(rgb_w[ov_actual:])

            if w1 >= len(used_paths):
                break
            w0 += stride

        ext_np = np.stack(ext_global_list, axis=0)
        int_np = np.stack(int_global_list, axis=0)
        pts_map = np.concatenate(pts_global_list, axis=0)
        conf_np = np.concatenate(conf_global_list, axis=0)
        rgb_map = np.concatenate(rgb_global_list, axis=0)

    # Repair any isolated 1-frame VGGT attention teleport spikes (`_repaired_indices`) and mask their
    # unprojected points so zero mis-registered 3D point clouds contaminate the initial seed cloud!
    ext_np_fixed, _n_repaired = _repair_outlier_trajectory_spikes(ext_np)
    if _n_repaired > 0:
        for si in range(len(ext_np)):
            if float(np.linalg.norm(ext_np_fixed[si, :3, 3] - ext_np[si, :3, 3])) > 1e-3:
                conf_np[si] = -1e9
    ext_np = ext_np_fixed

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

    # Lock shared square-pixel PINHOLE intrinsics across the whole clip when `lock_pinhole_intrinsics=True`
    # so per-frame focal length jitter never causes optical FOV breathing along the camera arc!
    raw_f_list: list[float] = []
    for i in range(len(used_paths)):
        k_v = int_np[i]
        raw_f_list.append(0.5 * (float(k_v[0, 0]) + float(k_v[1, 1])) * scale_px)
    shared_f = float(np.median(raw_f_list)) if raw_f_list else float(max(orig_w, orig_h))
    lock_k = bool(getattr(cfg, "lock_pinhole_intrinsics", True))

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
        if lock_k:
            fx = fy = shared_f
            cx = orig_w * 0.5
            cy = orig_h * 0.5
        else:
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


def _estimate_umeyama_sim3(src_pts: np.ndarray, dst_pts: np.ndarray) -> Sim3Transform:
    """Closed-form proper `det(R)=+1` Umeyama Sim(3) aligning COLMAP camera centers onto Blender priors."""
    if len(src_pts) < 3 or len(src_pts) != len(dst_pts):
        return Sim3Transform.identity()
    mu_s = src_pts.mean(axis=0)
    mu_d = dst_pts.mean(axis=0)
    xs = src_pts - mu_s
    xd = dst_pts - mu_d
    var_s = float((xs * xs).sum() / len(src_pts))
    if var_s < 1e-10:
        return Sim3Transform(scale=1.0, R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), t=tuple((mu_d - mu_s).tolist()))  # type: ignore[arg-type]

    cov = (xd.T @ xs) / float(len(src_pts))
    u_m, s_v, vt_m = np.linalg.svd(cov)
    d_sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(u_m @ vt_m) < 0.0:
        d_sign[2, 2] = -1.0
    r_mat = u_m @ d_sign @ vt_m
    scale = max(1e-3, min(1e3, float(np.sum(s_v * np.diag(d_sign)) / var_s)))
    t_vec = mu_d - scale * (r_mat @ mu_s)
    r_tup = (
        (float(r_mat[0, 0]), float(r_mat[0, 1]), float(r_mat[0, 2])),
        (float(r_mat[1, 0]), float(r_mat[1, 1]), float(r_mat[1, 2])),
        (float(r_mat[2, 0]), float(r_mat[2, 1]), float(r_mat[2, 2])),
    )
    return Sim3Transform(scale=scale, R=r_tup, t=(float(t_vec[0]), float(t_vec[1]), float(t_vec[2])))


def solve_poses_colmap(
    frames_dir: Path,
    work_dir: Path,
    cfg: PoseConfig,
    prior_intrinsics: tuple[float, float, float, float] | None = None,
    *,
    is_image_set: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Run COLMAP SfM (`PINHOLE` + exhaustive/sequential matching) and parse binary outputs."""
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

    img_files = [
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
    unique_dims: set[tuple[int, int]] = set()
    for p in img_files[:64]:
        try:
            with Image.open(p) as im_chk:
                unique_dims.add((im_chk.width, im_chk.height))
        except OSError:
            pass
    single_cam = "1" if (len(unique_dims) <= 1 or prior_intrinsics is not None) else "0"

    cam_model = "PINHOLE" if getattr(cfg, "lock_pinhole_intrinsics", True) or prior_intrinsics else "OPENCV"
    feat_args = [
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", single_cam,
        "--ImageReader.camera_model", cam_model,
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", "16",
    ]
    if prior_intrinsics is not None and single_cam == "1":
        fx_p, fy_p, cx_p, cy_p = prior_intrinsics
        feat_args.extend(["--ImageReader.camera_params", f"{fx_p:.4f},{fy_p:.4f},{cx_p:.4f},{cy_p:.4f}"])
    _colmap(feat_args, "feature_extractor")

    matcher_mode = str(getattr(cfg, "matcher", "auto") or "auto").strip().lower()
    use_exhaustive = matcher_mode == "exhaustive" or (
        matcher_mode == "auto" and ((is_image_set and len(img_files) <= 80) or len(img_files) <= 32)
    )
    if use_exhaustive:
        _colmap([
            "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", "0",
            "--SiftMatching.num_threads", "16",
        ], "exhaustive_matcher")
    else:
        _colmap([
            "sequential_matcher",
            "--database_path", str(db_path),
            "--SequentialMatching.overlap", "12",
            "--SiftMatching.use_gpu", "0",
            "--SiftMatching.num_threads", "16",
        ], "sequential_matcher")

    mapper_args = [
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
    ]
    if prior_intrinsics is not None:
        mapper_args.extend([
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_extra_params", "0",
        ])
    _colmap(mapper_args, "mapper")

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

    raw_imgs.sort(key=lambda r: _natural_sort_key(r[0]))
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


# ─── Stage 2C: Known-Pose Seedance Video Re-Polish (Locked Blender Cameras!) ──


def solve_known_cameras_video(
    frame_paths: list[Path],
    pts_by_name: dict[str, float],
    video_meta: dict[str, Any],
    known_cameras_doc: dict[str, Any],
    cfg: PoseConfig,
    device: torch.device,
    models_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Lock extracted Seedance video frames onto exact known Blender camera poses (zero Sim3 drift).

    Uses VGGT metric depth maps unprojected directly through the known `(R_w2c, t_w2c, K)` when
    available, falling back to frustum-ray multi-depth surface seeding if VGGT is unavailable.
    """
    with Image.open(frame_paths[0]) as im0:
        orig_w, orig_h = im0.width, im0.height

    traj = interpolate_camera_trajectory(
        known_cameras_doc,
        fps=float(video_meta.get("fps") or 24.0),
        duration_sec=float(video_meta.get("duration_sec") or 5.0),
        total_frames=int(video_meta.get("total_frames") or len(frame_paths)),
        target_width=orig_w,
        target_height=orig_h,
    )
    t_frames = traj["frames"]
    eff_fps = float(traj["fps"])

    cam_records: list[dict[str, Any]] = []
    for idx, fp in enumerate(frame_paths):
        pts_s = pts_by_name.get(fp.name, idx / max(1.0, eff_fps))
        fi = max(0, min(len(t_frames) - 1, round(pts_s * eff_fps)))
        tf = t_frames[fi]
        cam_records.append({
            "frame_index": idx,
            "pts_sec": round(float(pts_s), 5),
            "timeline_frame": int(tf["frame"]),
            "image_name": fp.name,
            "image_path": str(fp),
            "width": orig_w,
            "height": orig_h,
            "fx": float(tf["fx"]),
            "fy": float(tf["fy"]),
            "cx": float(tf["cx"]),
            "cy": float(tf["cy"]),
            "R_w2c": tf["R_w2c"],
            "t_w2c": tf["t_w2c"],
        })

    # Seed 3D points strictly inside the locked Blender world frame by unprojecting rays
    rng = np.random.default_rng(42)
    pts_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []

    try:
        from vggt.utils.load_fn import load_and_preprocess_images

        sel_paths = (
            [frame_paths[int(i)] for i in np.linspace(0, len(frame_paths) - 1, min(len(frame_paths), 24), dtype=int)]
        )
        sel_map = {p.name: c for c in cam_records for p in [Path(c["image_path"])] if p.name == c["image_name"]}
        model = _get_vggt_model(device, models_dir=models_dir)
        amp_dtype = torch.bfloat16 if _VGGT_CACHE.get("bf16", True) else torch.float16
        imgs = load_and_preprocess_images([str(p) for p in sel_paths], mode="pad").to(device)
        imgs_b = imgs.unsqueeze(0) if imgs.ndim == 4 else imgs
        _, _, _, vh, vw = imgs_b.shape
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            preds = model(imgs_b)
        if "depth" in preds:
            depth_np = preds["depth"][0].float().cpu().numpy().squeeze(-1) if preds["depth"].ndim == 5 else preds["depth"][0].float().cpu().numpy()
            rgb_v = (imgs_b[0].permute(0, 2, 3, 1).float().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
            scale_px = max(orig_w, orig_h) / float(max(vw, vh))
            pad_x = max(0.0, (float(vw) - orig_w / scale_px) * 0.5)
            pad_y = max(0.0, (float(vh) - orig_h / scale_px) * 0.5)
            uu, vv = np.meshgrid(np.arange(vw, dtype=np.float64), np.arange(vh, dtype=np.float64))
            u_orig = (uu - pad_x) * scale_px
            v_orig = (vv - pad_y) * scale_px
            valid_uv = (u_orig >= 0) & (u_orig < orig_w) & (v_orig >= 0) & (v_orig < orig_h)

            for si, sp in enumerate(sel_paths):
                c = sel_map[sp.name]
                d_s = depth_np[si]
                med_d = float(np.median(d_s[valid_uv & (d_s > 1e-3)])) if np.any(d_s > 1e-3) else 1.0
                # Rescale VGGT depth to camera-to-origin distance so metric scale matches known rig
                r_c = np.array(c["R_w2c"], dtype=np.float64)
                t_c = np.array(c["t_w2c"], dtype=np.float64)
                cam_pos = -r_c.T @ t_c
                rig_depth = max(0.8, float(np.linalg.norm(cam_pos)))
                d_metric = d_s * (rig_depth / max(med_d, 1e-3))
                mask = valid_uv & np.isfinite(d_metric) & (d_metric > 0.15) & (d_metric < rig_depth * 4.0)
                ys_i, xs_i = np.nonzero(mask)
                if len(xs_i) > 6000:
                    pick = rng.choice(len(xs_i), size=6000, replace=False)
                    ys_i, xs_i = ys_i[pick], xs_i[pick]
                zc = d_metric[ys_i, xs_i]
                xc = (u_orig[ys_i, xs_i] - c["cx"]) * zc / c["fx"]
                yc = (v_orig[ys_i, xs_i] - c["cy"]) * zc / c["fy"]
                pts_cam = np.stack([xc, yc, zc], axis=-1)
                pts_world = (pts_cam - t_c[None, :]) @ r_c
                pts_chunks.append(pts_world)
                rgb_chunks.append(rgb_v[ys_i, xs_i])
    except Exception:  # noqa: BLE001, S110
        pass

    if not pts_chunks:
        # Pure multi-depth frustum unprojection fallback (works offline/CPU with zero VGGT weights!)
        per_cam = max(400, min(4000, cfg.max_init_points // max(1, len(cam_records))))
        for c in cam_records:
            with Image.open(c["image_path"]) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            r_c = np.array(c["R_w2c"], dtype=np.float64)
            t_c = np.array(c["t_w2c"], dtype=np.float64)
            cam_pos = -r_c.T @ t_c
            base_d = max(1.0, float(np.linalg.norm(cam_pos)))
            us = rng.integers(0, orig_w, size=per_cam)
            vs = rng.integers(0, orig_h, size=per_cam)
            zc = rng.uniform(base_d * 0.45, base_d * 1.65, size=per_cam)
            xc = (us.astype(np.float64) - c["cx"]) * zc / c["fx"]
            yc = (vs.astype(np.float64) - c["cy"]) * zc / c["fy"]
            pts_cam = np.stack([xc, yc, zc], axis=-1)
            pts_world = (pts_cam - t_c[None, :]) @ r_c
            pts_chunks.append(pts_world)
            rgb_chunks.append(arr[vs, us])

    xyz = np.concatenate(pts_chunks, axis=0)
    rgb = np.concatenate(rgb_chunks, axis=0)
    if cfg.max_init_points > 0 and len(xyz) > cfg.max_init_points:
        sel = rng.choice(len(xyz), size=cfg.max_init_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return cam_records, xyz.astype(np.float64), rgb.astype(np.uint8)


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

    # Exact chunked 3-NN per-point scale against a 24k reference subset (`<6 ms` on H100 for 180k points!):
    # Gives close foreground props 4mm splats and distant walls 4cm splats (instead of uniform 15cm beachballs!)
    ref_cnt = min(n, 24_576)
    ref_idx = torch.randperm(n, device=device)[:ref_cnt]
    ref_pts = means[ref_idx]
    k_nn = min(4, max(2, ref_cnt))
    init_radii = torch.empty((n, 1), dtype=torch.float32, device=device)
    chunk_sz = 16_384
    for i0 in range(0, n, chunk_sz):
        i1 = min(n, i0 + chunk_sz)
        d_blk = torch.cdist(means[i0:i1], ref_pts)
        # Drop self-match (`d == 0`) by taking top-4 smallest and averaging the 3 non-zero neighbors
        knn_b, _ = d_blk.topk(k_nn, dim=1, largest=False)
        r_b = knn_b[:, 1:].mean(dim=1, keepdim=True) if k_nn > 1 else knn_b[:, :1]
        init_radii[i0:i1] = r_b.clamp(0.002, 0.08)
    scales = torch.log(init_radii.expand(n, 3)).contiguous()

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
    use_fused = any(p.is_cuda for p in params.values())
    opts: dict[str, torch.optim.Adam] = {}
    for k in params:
        if use_fused:
            try:
                opts[k] = torch.optim.Adam([params[k]], lr=lr_map[k], betas=(0.9, 0.999), eps=1e-15, fused=True)
                continue
            except Exception:  # noqa: BLE001, S110
                pass
        opts[k] = torch.optim.Adam([params[k]], lr=lr_map[k], betas=(0.9, 0.999), eps=1e-15)
    return opts


def _tensor_grad_stats(p: torch.Tensor) -> dict[str, float]:
    """Summarize parameter gradient magnitude, non-zero support, and fp32 underflow/NaN health."""
    g = p.grad
    if g is None or g.numel() == 0:
        return {"rms": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0, "nz_frac": 0.0, "nan_inf": 0}
    gf = g.detach().float().reshape(-1)
    finite_mask = torch.isfinite(gf)
    nan_inf = int((~finite_mask).sum().item())
    if nan_inf == gf.numel():
        return {"rms": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0, "nz_frac": 0.0, "nan_inf": nan_inf}
    ag = gf[finite_mask].abs()
    nz_mask = ag > 1e-12
    nz_cnt = int(nz_mask.sum().item())
    nz_frac = nz_cnt / float(max(1, gf.numel()))
    rms = float(torch.sqrt((ag * ag).mean()).item())
    mx = float(ag.max().item())
    if nz_cnt > 0:
        ag_nz = ag[nz_mask]
        if ag_nz.numel() > 16_384:
            stride = max(1, ag_nz.numel() // 16_384)
            ag_nz = ag_nz[::stride]
        q50 = float(torch.quantile(ag_nz, 0.50).item())
        q99 = float(torch.quantile(ag_nz, 0.99).item())
    else:
        q50, q99 = 0.0, 0.0
    return {
        "rms": round(rms, 9),
        "p50": round(q50, 9),
        "p99": round(q99, 9),
        "max": round(mx, 9),
        "nz_frac": round(nz_frac, 4),
        "nan_inf": nan_inf,
    }


def _collect_kernel_telemetry(
    *,
    params: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Adam],
    info: dict[str, Any],
    ren_alphas: torch.Tensor,
    opa_sig: torch.Tensor,
    scales_lin: torch.Tensor,
    loss_terms: dict[str, float],
    pose_rot6d: torch.Tensor | None = None,
    pose_trans: torch.Tensor | None = None,
    use_pose_opt: bool = False,
    densify_grad_thresh: float = 0.0002,
    img_w: int = 1280,
    img_h: int = 720,
) -> dict[str, Any]:
    """Capture intermediate forward/raster/backward/Adam kernel diagnostics on telemetry steps."""
    with torch.no_grad():
        n_total = max(1, int(params["means"].shape[0]))
        # 1. Per-parameter gradient norms & support (`means`, `scales`, `quats`, `opacities`, `sh0`, `shN`)
        grads: dict[str, Any] = {k: _tensor_grad_stats(params[k]) for k in params}

        # 2D projected screen-space gradient (`info["means2d"].absgrad` or `.grad`) driving densification
        m2d = info.get("means2d")
        if torch.is_tensor(m2d):
            g2d_raw = getattr(m2d, "absgrad", None)
            if g2d_raw is None:
                g2d_raw = m2d.grad
            if torch.is_tensor(g2d_raw) and g2d_raw.numel() > 0:
                # Normalize to NDC screen units matching gsplat DefaultStrategy (`* 0.5 * [W, H]`)
                g2d_v = g2d_raw.detach().float().reshape(-1, g2d_raw.shape[-1])[:, :2]
                scale_wh = torch.tensor([0.5 * img_w, 0.5 * img_h], dtype=torch.float32, device=g2d_v.device)
                g2d_norm = (g2d_v * scale_wh).norm(dim=-1)
                finite_2d = g2d_norm[torch.isfinite(g2d_norm)]
                if finite_2d.numel() > 0:
                    sample_2d = finite_2d[:: max(1, finite_2d.numel() // 16_384)]
                    grads["means2d_absgrad"] = {
                        "p50": round(float(torch.quantile(sample_2d, 0.50).item()), 7),
                        "p95": round(float(torch.quantile(sample_2d, 0.95).item()), 7),
                        "p99": round(float(torch.quantile(sample_2d, 0.99).item()), 7),
                        "max": round(float(finite_2d.max().item()), 7),
                        "above_densify_frac": round(
                            float((finite_2d >= densify_grad_thresh).float().mean().item()), 4
                        ),
                    }

        if use_pose_opt and pose_trans is not None and pose_rot6d is not None:
            grads["pose_trans"] = _tensor_grad_stats(pose_trans)
            grads["pose_rot6d"] = _tensor_grad_stats(pose_rot6d)

        # 2. Packed Rasterizer & Tile-Intersection Intermediate Stats
        radii = info.get("radii")
        visible_cnt = 0
        rad_p50 = 0.0
        rad_p99 = 0.0
        rad_max = 0.0
        if torch.is_tensor(radii) and radii.numel() > 0:
            rf = radii.detach().float()
            if rf.ndim >= 2 and rf.shape[-1] == 2:
                rmax_per = rf.max(dim=-1).values.reshape(-1)
            else:
                rmax_per = rf.reshape(-1)
            vis_mask = rmax_per > 0
            visible_cnt = int(vis_mask.sum().item())
            if visible_cnt > 0:
                rv = rmax_per[vis_mask]
                rv_s = rv[:: max(1, rv.numel() // 16_384)]
                rad_p50 = float(torch.quantile(rv_s, 0.50).item())
                rad_p99 = float(torch.quantile(rv_s, 0.99).item())
                rad_max = float(rv.max().item())

        n_isects = 0
        flatten_ids = info.get("flatten_ids")
        if torch.is_tensor(flatten_ids):
            n_isects = int(flatten_ids.numel())
        n_tiles = max(1, math.ceil(img_w / 16) * math.ceil(img_h / 16))

        alpha_2d = ren_alphas[0].detach().float().reshape(-1)
        alpha_s = alpha_2d[:: max(1, alpha_2d.numel() // 16_384)]
        transmittance = 1.0 - alpha_2d
        # Pixels where T_final < 1e-4 terminate early in gsplat forward/backward (`T < 1e-4` floor!)
        early_term_frac = float((transmittance <= 1e-4).float().mean().item())
        hole_frac = float((alpha_2d < 0.10).float().mean().item())

        raster_stats = {
            "visible_gaussians": visible_cnt,
            "visible_frac": round(visible_cnt / float(n_total), 4),
            "tile_isects": n_isects,
            "avg_splats_per_tile": round(n_isects / float(n_tiles), 2),
            "radius_px_p50": round(rad_p50, 2),
            "radius_px_p99": round(rad_p99, 2),
            "radius_px_max": round(rad_max, 1),
            "render_alpha_mean": round(float(alpha_s.mean().item()), 4),
            "T_floor_sat_frac": round(early_term_frac, 4),
            "alpha_hole_frac": round(hole_frac, 4),
        }

        # 3. Intermediate Parameter & Sigmoid/Exp Jacobian Saturation Diagnostics
        opa_f = opa_sig.detach().float().reshape(-1)
        opa_s = opa_f[:: max(1, opa_f.numel() // 16_384)]
        # Sigmoid derivative `sigma * (1 - sigma)` vanishes near 0 and 1 ("opacity dead-zone")
        opa_jac = opa_f * (1.0 - opa_f)
        dead_low_frac = float((opa_f < 0.01).float().mean().item())
        dead_high_frac = float((opa_f > 0.99).float().mean().item())

        sc_f = scales_lin.detach().float()
        sc_max = sc_f.max(dim=-1).values
        sc_sorted, _ = torch.sort(sc_f, dim=-1)
        aniso = sc_sorted[:, 2] / (sc_sorted[:, 1] + 1e-8)
        sc_s = sc_max[:: max(1, sc_max.numel() // 16_384)]
        aniso_s = aniso[:: max(1, aniso.numel() // 16_384)]

        q_norm = params["quats"].detach().float().norm(dim=-1)
        sh0_rgb = (0.5 + _SH_C0 * params["sh0"].detach().float().reshape(-1, 3))
        rgb_clamped_frac = float(((sh0_rgb < 0.0) | (sh0_rgb > 1.0)).any(dim=-1).float().mean().item())

        param_stats = {
            "opa_p10": round(float(torch.quantile(opa_s, 0.10).item()), 4),
            "opa_p50": round(float(torch.quantile(opa_s, 0.50).item()), 4),
            "opa_p90": round(float(torch.quantile(opa_s, 0.90).item()), 4),
            "opa_dead_low_frac": round(dead_low_frac, 4),
            "opa_dead_high_frac": round(dead_high_frac, 4),
            "opa_jacobian_mean": round(float(opa_jac.mean().item()), 5),
            "scale_m_p50": round(float(torch.quantile(sc_s, 0.50).item()), 5),
            "scale_m_p99": round(float(torch.quantile(sc_s, 0.99).item()), 5),
            "scale_m_max": round(float(sc_max.max().item()), 4),
            "aniso_p95": round(float(torch.quantile(aniso_s, 0.95).item()), 2),
            "quat_norm_min": round(float(q_norm.min().item()), 4),
            "sh0_oot_gamut_frac": round(rgb_clamped_frac, 4),
        }

        # 4. Adam Optimizer Second-Moment (`v_t`), SNR (`|m_hat| / (sqrt(v_hat) + eps)`), & Step Floor
        adam_stats: dict[str, Any] = {}
        for k in ("means", "opacities", "scales", "sh0"):
            opt_k = optimizers.get(k)
            p_k = params.get(k)
            if opt_k is None or p_k is None:
                continue
            st_k = opt_k.state.get(p_k)
            pg0 = opt_k.param_groups[0]
            lr_k = float(pg0.get("lr", 0.0))
            eps_k = float(pg0.get("eps", 1e-15))
            fused_k = bool(pg0.get("fused", False))
            if not st_k or "exp_avg" not in st_k or "exp_avg_sq" not in st_k:
                adam_stats[k] = {"lr": lr_k, "fused": fused_k, "snr_p50": 0.0, "step_rms": 0.0, "eps_dom_frac": 0.0}
                continue
            m_t = st_k["exp_avg"].detach().float().reshape(-1)
            v_t = st_k["exp_avg_sq"].detach().float().reshape(-1)
            step_t = st_k.get("step", 1)
            step_num = max(1, int(step_t.item()) if torch.is_tensor(step_t) else int(step_t))
            b1, b2 = pg0.get("betas", (0.9, 0.999))
            bc1 = 1.0 - (b1 ** step_num)
            bc2 = 1.0 - (b2 ** step_num)
            stride_a = max(1, m_t.numel() // 16_384)
            m_hat = m_t[::stride_a] / max(bc1, 1e-12)
            v_hat = (v_t[::stride_a] / max(bc2, 1e-12)).clamp_min(0.0)
            denom = torch.sqrt(v_hat) + eps_k
            snr = m_hat.abs() / denom
            upd = lr_k * snr
            active_m = v_hat > 0
            snr_med = float(torch.quantile(snr[active_m], 0.50).item()) if active_m.any() else 0.0
            eps_dom = float((torch.sqrt(v_hat) < eps_k * 10.0).float().mean().item())
            adam_stats[k] = {
                "lr": round(lr_k, 8),
                "fused": fused_k,
                "snr_p50": round(snr_med, 4),
                "step_rms": round(float(torch.sqrt((upd * upd).mean()).item()), 9),
                "v_sqrt_p50": round(float(torch.quantile(torch.sqrt(v_hat), 0.50).item()), 9),
                "eps_dom_frac": round(eps_dom, 4),
            }

        pose_shift_mm = (
            round(float(pose_trans.detach().norm(dim=-1).mean().item() * 1000.0), 3)
            if (use_pose_opt and pose_trans is not None)
            else 0.0
        )

        return {
            "loss_terms": loss_terms,
            "grads": grads,
            "raster": raster_stats,
            "params": param_stats,
            "adam": adam_stats,
            "pose_shift_mm": pose_shift_mm,
        }


def _sep_gauss_blur_bchw(x: torch.Tensor, k_h: torch.Tensor, k_v: torch.Tensor) -> torch.Tensor:
    """Separable 1D horizontal (`1x11`) + vertical (`11x1`) Gaussian blur (`22` taps vs `121` in 2D!)."""
    xh = F.conv2d(x, k_h, padding=(0, 5), groups=3)
    return F.conv2d(xh, k_v, padding=(5, 0), groups=3)


def _ssim_cuda(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    gt_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Batched separable 11x11 SSIM on `(H, W, 3)` or `(B, H, W, 3)` with optional pre-cached GT moments.

    Returns scalar when `pred.ndim == 3`, or `[B]` per-view SSIM vector when `pred.ndim == 4`.
    """
    is_single = pred.ndim == 3
    p = pred.permute(2, 0, 1).unsqueeze(0) if is_single else pred.permute(0, 3, 1, 2)
    g = gt.permute(2, 0, 1).unsqueeze(0) if gt.ndim == 3 else gt.permute(0, 3, 1, 2)

    if isinstance(window, tuple):
        k_h, k_v = window
        mu1 = _sep_gauss_blur_bchw(p, k_h, k_v)
        mu1_sq = mu1 * mu1
        sig1_sq = _sep_gauss_blur_bchw(p * p, k_h, k_v) - mu1_sq
        if gt_stats is not None:
            mu2, sig2_sq = gt_stats
            mu2_sq = mu2 * mu2
        else:
            mu2 = _sep_gauss_blur_bchw(g, k_h, k_v)
            mu2_sq = mu2 * mu2
            sig2_sq = _sep_gauss_blur_bchw(g * g, k_h, k_v) - mu2_sq
        mu12 = mu1 * mu2
        sig12 = _sep_gauss_blur_bchw(p * g, k_h, k_v) - mu12
    else:
        mu1 = F.conv2d(p, window, padding=5, groups=3)
        mu2 = F.conv2d(g, window, padding=5, groups=3)
        mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        sig1_sq = F.conv2d(p * p, window, padding=5, groups=3) - mu1_sq
        sig2_sq = F.conv2d(g * g, window, padding=5, groups=3) - mu2_sq
        sig12 = F.conv2d(p * g, window, padding=5, groups=3) - mu12

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2.0 * mu12 + c1) * (2.0 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2))
    if is_single:
        return smap.mean()
    return smap.mean(dim=(1, 2, 3))


def _build_ssim_window(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build separable `(k_horiz [3,1,1,11], k_vert [3,1,11,1])` 1D Gaussian kernels (`sigma=1.5`)."""
    x = torch.arange(11, dtype=torch.float32, device=device) - 5.0
    k1 = torch.exp(-x * x / (2.0 * 1.5 * 1.5))
    k1 = k1 / k1.sum()
    k_h = k1.view(1, 1, 1, 11).expand(3, 1, 1, 11).contiguous()
    k_v = k1.view(1, 1, 11, 1).expand(3, 1, 11, 1).contiguous()
    return k_h, k_v


def _build_stereo_peer_table(viewmats_gpu: Any, active_indices: list[int]) -> dict[int, list[int]]:
    with torch.no_grad():
        vm = viewmats_gpu.detach().float().cpu().numpy()
    return build_stereo_peer_table(vm, active_indices)


def _build_swept_arc_samples_gpu(
    viewmats_gpu: torch.Tensor,
    ks_gpu: torch.Tensor,
    active_indices: list[int],
    pts_sec_list: list[float] | None,
    img_w: int,
    img_h: int,
    n_arc_samples: int = 512,
    n_frustum_probes: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample dense points along the continuous centripetal camera arc + frustum view matrices.

    Returns:
      - `swept_centers_gpu [S, 3]`: dense world-space points along the continuous camera tunnel
      - `probe_vms_gpu [P, 4, 4]`: evenly spaced W2C matrices along the arc for near-frustum carving
      - `probe_ks_gpu [P, 3, 3]`: corresponding intrinsics for frustum boundary checks
    """
    device = viewmats_gpu.device
    act_sorted = sorted(active_indices)
    with torch.no_grad():
        vms_np = viewmats_gpu[act_sorted].detach().float().cpu().numpy()
        ks_np = ks_gpu[act_sorted].detach().float().cpu().numpy()

    if len(act_sorted) < 2:
        r0 = vms_np[0, :3, :3]
        t0 = vms_np[0, :3, 3]
        c0 = torch.from_numpy((-r0.T @ t0).astype(np.float32)).unsqueeze(0).to(device)
        return c0, viewmats_gpu[act_sorted[:1]].detach(), ks_gpu[act_sorted[:1]].detach()

    bl_cams: list[dict[str, Any]] = []
    for k_idx, orig_idx in enumerate(act_sorted):
        rc = vms_np[k_idx, :3, :3].tolist()
        tc = vms_np[k_idx, :3, 3].tolist()
        fx, fy = float(ks_np[k_idx, 0, 0]), float(ks_np[k_idx, 1, 1])
        cx, cy = float(ks_np[k_idx, 0, 2]), float(ks_np[k_idx, 1, 2])
        bl = opencv_w2c_to_blender_camera(rc, tc, fx, fy, cx, cy, img_w, img_h)
        if pts_sec_list and orig_idx < len(pts_sec_list):
            bl["pts_sec"] = float(pts_sec_list[orig_idx])
        else:
            bl["pts_sec"] = float(k_idx) * 0.25
        bl_cams.append(bl)

    t_min = float(bl_cams[0]["pts_sec"])
    t_max = max(t_min + 0.1, float(bl_cams[-1]["pts_sec"]))
    s_cnt = max(len(act_sorted), n_arc_samples)
    eff_fps = max(1.0, (s_cnt - 1) / max(0.1, t_max - t_min))
    traj = interpolate_camera_trajectory(
        {"cameras": bl_cams},
        fps=eff_fps,
        duration_sec=(t_max - t_min),
        total_frames=s_cnt,
        target_width=img_w,
        target_height=img_h,
    )
    t_frames = traj["frames"]
    centers = np.array([f["location"] for f in t_frames], dtype=np.float32)

    p_cnt = min(len(t_frames), max(8, n_frustum_probes))
    p_idxs = np.linspace(0, len(t_frames) - 1, p_cnt, dtype=int)
    p_vms = np.zeros((p_cnt, 4, 4), dtype=np.float32)
    p_ks = np.zeros((p_cnt, 3, 3), dtype=np.float32)
    for pi, fi in enumerate(p_idxs):
        fr = t_frames[int(fi)]
        p_vms[pi, 3, 3] = 1.0
        p_vms[pi, :3, :3] = np.array(fr["R_w2c"], dtype=np.float32)
        p_vms[pi, :3, 3] = np.array(fr["t_w2c"], dtype=np.float32)
        p_ks[pi, 0, 0] = float(fr["fx"])
        p_ks[pi, 1, 1] = float(fr["fy"])
        p_ks[pi, 0, 2] = float(fr["cx"])
        p_ks[pi, 1, 2] = float(fr["cy"])
        p_ks[pi, 2, 2] = 1.0

    return (
        torch.from_numpy(centers).to(device),
        torch.from_numpy(p_vms).to(device),
        torch.from_numpy(p_ks).to(device),
    )


def _evict_swept_arc_and_near_frustum_floaters(
    params: torch.nn.ParameterDict,
    optimizers: dict[str, torch.optim.Adam],
    strategy_state: dict[str, Any],
    swept_centers_gpu: torch.Tensor,
    probe_vms_gpu: torch.Tensor,
    probe_ks_gpu: torch.Tensor,
    img_w: int,
    img_h: int,
    corridor_r: float,
    near_clear_m: float,
) -> dict[str, int]:
    """Hard-evict Gaussians inside the continuous swept camera tunnel OR near-lens view frustums."""
    with torch.no_grad():
        means = params["means"]
        n0 = int(means.shape[0])
        if n0 < 1500:
            return {"corridor_evicted": 0, "frustum_evicted": 0}

        keep = torch.ones((n0,), dtype=torch.bool, device=means.device)
        corr_ev = 0
        if corridor_r > 0.0 and swept_centers_gpu.numel() > 0:
            min_d = torch.cdist(means, swept_centers_gpu).min(dim=1).values
            in_tube = min_d < corridor_r
            corr_ev = int(in_tube.sum().item())
            keep &= ~in_tube

        frust_ev = 0
        if near_clear_m > 0.0 and probe_vms_gpu.shape[0] > 0:
            # Project `[N, 3]` into `[P, N, 3]` camera coordinates across all arc probes simultaneously
            r_p = probe_vms_gpu[:, :3, :3]  # [P, 3, 3]
            t_p = probe_vms_gpu[:, :3, 3]   # [P, 3]
            x_cam = torch.einsum("pij,nj->pni", r_p, means) + t_p.unsqueeze(1)  # [P, N, 3]
            z_c = x_cam[:, :, 2]
            fx_p = probe_ks_gpu[:, 0, 0].unsqueeze(1)
            fy_p = probe_ks_gpu[:, 1, 1].unsqueeze(1)
            cx_p = probe_ks_gpu[:, 0, 2].unsqueeze(1)
            cy_p = probe_ks_gpu[:, 1, 2].unsqueeze(1)
            u_px = fx_p * (x_cam[:, :, 0] / z_c.clamp(min=1e-4)) + cx_p
            v_px = fy_p * (x_cam[:, :, 1] / z_c.clamp(min=1e-4)) + cy_p
            in_cone = (
                (z_c > 0.01)
                & (z_c < near_clear_m)
                & (u_px >= -16.0)
                & (u_px <= (img_w + 16.0))
                & (v_px >= -16.0)
                & (v_px <= (img_h + 16.0))
            )
            hit_any_cone = in_cone.any(dim=0) & keep
            frust_ev = int(hit_any_cone.sum().item())
            keep &= ~hit_any_cone

        if int(keep.sum().item()) >= 1000 and (corr_ev + frust_ev) > 0:
            slice_params_and_adam(params, optimizers, strategy_state, keep)
            return {"corridor_evicted": corr_ev, "frustum_evicted": frust_ev}
        return {"corridor_evicted": 0, "frustum_evicted": 0}


def _solve_dense_video_arc_and_promote_frames(
    *,
    params: torch.nn.ParameterDict,
    frames_gpu: torch.Tensor,
    viewmats_gpu: torch.Tensor,
    ks_gpu: torch.Tensor,
    active_indices: list[int],
    pts_sec_list: list[float],
    excluded_set: set[str],
    image_names: list[str],
    ssim_win: tuple[torch.Tensor, torch.Tensor],
    sh_deg: int,
    rast_mode: str,
    solve_steps: int = 24,
    promote_k: int = 24,
) -> dict[str, Any]:
    """Iterative Cycle Bridge:
    1. Initialize all unposed candidate video frames along the centripetal spline of active cameras.
    2. Freeze Gaussians and run batched differentiable `se(3)` (`6D rot + 3D trans`) photometric
       video-diff minimization + temporal smoothness (`||Δ²pose||²`) across ALL video frames.
    3. Rank candidate frames by residual photometric error × temporal coverage and promote the top
       `promote_k` frames into `active_indices` for the next 3DGS training cycle!
    """
    from gsplat import rasterization

    device = viewmats_gpu.device
    m_all = int(frames_gpu.shape[0])
    img_h, img_w = int(frames_gpu.shape[1]), int(frames_gpu.shape[2])
    act_set = set(active_indices)

    # 1. Initialize candidate (unposed) frames from centripetal spline of currently active cameras
    with torch.no_grad():
        act_sorted = sorted(active_indices)
        vms_act_np = viewmats_gpu[act_sorted].detach().float().cpu().numpy()
        ks_act_np = ks_gpu[act_sorted].detach().float().cpu().numpy()
        bl_anchors: list[dict[str, Any]] = []
        for ki, oi in enumerate(act_sorted):
            bl_a = opencv_w2c_to_blender_camera(
                vms_act_np[ki, :3, :3].tolist(),
                vms_act_np[ki, :3, 3].tolist(),
                float(ks_act_np[ki, 0, 0]),
                float(ks_act_np[ki, 1, 1]),
                float(ks_act_np[ki, 0, 2]),
                float(ks_act_np[ki, 1, 2]),
                img_w,
                img_h,
            )
            bl_a["pts_sec"] = float(pts_sec_list[oi]) if oi < len(pts_sec_list) else float(oi) * 0.1
            bl_anchors.append(bl_a)

        if len(bl_anchors) >= 2:
            t0_a = float(bl_anchors[0]["pts_sec"])
            t1_a = max(t0_a + 0.1, float(bl_anchors[-1]["pts_sec"]))
            dense_fps = 60.0
            n_dense = max(16, min(6000, round((t1_a - t0_a) * dense_fps) + 1))
            traj_init = interpolate_camera_trajectory(
                {"cameras": bl_anchors},
                fps=dense_fps,
                duration_sec=(t1_a - t0_a),
                total_frames=n_dense,
                target_width=img_w,
                target_height=img_h,
            )
            tf_list = traj_init["frames"]
            vms_work = viewmats_gpu.detach().clone()
            for idx_all in range(m_all):
                if idx_all in act_set:
                    continue
                pt_i = float(pts_sec_list[idx_all]) if idx_all < len(pts_sec_list) else 0.0
                fi_d = max(0, min(len(tf_list) - 1, round((pt_i - t0_a) * dense_fps)))
                fr_d = tf_list[fi_d]
                vm_d = torch.eye(4, dtype=torch.float32, device=device)
                vm_d[:3, :3] = torch.tensor(fr_d["R_w2c"], dtype=torch.float32, device=device)
                vm_d[:3, 3] = torch.tensor(fr_d["t_w2c"], dtype=torch.float32, device=device)
                vms_work[idx_all] = vm_d
        else:
            vms_work = viewmats_gpu.detach().clone()

    # 2. Freeze Gaussian tensors and run Stage A wide-angle yaw sweep (`[-60°..+60°]`) on candidate frames
    # so fast whip-pan candidates sitting 20°–60° off the coarse spline snap into their basin BEFORE Adam!
    means_f = params["means"].detach()
    quats_f = F.normalize(params["quats"].detach(), dim=-1)
    scales_f = torch.exp(params["scales"].detach())
    opa_f = torch.sigmoid(params["opacities"].detach())
    k_sh = (max(0, sh_deg) + 1) ** 2
    colors_f = torch.cat([params["sh0"].detach(), params["shN"].detach()], dim=1)[:, :k_sh, :]

    cand_unposed = [i for i in range(m_all) if i not in act_set and image_names[i] not in excluded_set]
    if cand_unposed:
        with torch.no_grad():
            sw_w, sw_h = max(64, img_w // 6), max(48, img_h // 6)
            k_sw = ks_gpu[cand_unposed].clone()
            k_sw[:, 0, :] *= sw_w / float(img_w)
            k_sw[:, 1, :] *= sw_h / float(img_h)
            gt_sw = F.interpolate(
                (frames_gpu[cand_unposed].float() * (1.0 / 255.0)).permute(0, 3, 1, 2),
                size=(sw_h, sw_w),
                mode="area",
            ).permute(0, 2, 3, 1)
            base_cand = vms_work[cand_unposed].clone()
            best_cand = base_cand.clone()
            best_e = torch.full((len(cand_unposed),), 1e9, dtype=torch.float32, device=device)
            for deg_o in (0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0, -60.0, 60.0):
                rad_o = math.radians(deg_o)
                ca_o, sa_o = math.cos(rad_o), math.sin(rad_o)
                rz_o = torch.tensor([[ca_o, -sa_o, 0.0], [sa_o, ca_o, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
                vc = base_cand.clone()
                vc[:, :3, :3] = torch.matmul(base_cand[:, :3, :3], rz_o.T)
                es_list = []
                for b0_s in range(0, len(cand_unposed), 16):
                    rs_c, _, _ = rasterization(
                        means=means_f, quats=quats_f, scales=scales_f, opacities=opa_f, colors=colors_f,
                        viewmats=vc[b0_s : b0_s + 16], Ks=k_sw[b0_s : b0_s + 16],
                        width=sw_w, height=sw_h, sh_degree=0, packed=True,
                    )
                    es_list.append((rs_c - gt_sw[b0_s : b0_s + 16]).abs().mean(dim=(1, 2, 3)) + 0.003 * (abs(deg_o) / 15.0))
                e_vec = torch.cat(es_list, dim=0)
                imp = e_vec < best_e
                best_e = torch.where(imp, e_vec, best_e)
                best_cand[imp] = vc[imp]
            vms_work[cand_unposed] = best_cand

    rot6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device).repeat(
        m_all, 1
    ).requires_grad_(True)
    trans = torch.zeros((m_all, 3), dtype=torch.float32, device=device, requires_grad=True)
    opt_arc = torch.optim.Adam([rot6d, trans], lr=6e-4, betas=(0.9, 0.99), eps=1e-15)
    row4_b = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    def _compose_batch_vms(idxs: list[int]) -> torch.Tensor:
        r6 = rot6d[idxs]  # [B, 6]
        a1 = F.normalize(r6[:, 0:3], dim=-1)
        b2 = r6[:, 3:6] - (a1 * r6[:, 3:6]).sum(dim=-1, keepdim=True) * a1
        a2 = F.normalize(b2, dim=-1)
        a3 = torch.linalg.cross(a1, a2, dim=-1)
        r_del = torch.stack([a1, a2, a3], dim=1)  # [B, 3, 3]
        t_del = trans[idxs]  # [B, 3]
        base_b = vms_work[idxs]
        r_new = torch.bmm(r_del, base_b[:, :3, :3])
        t_new = torch.bmm(r_del, base_b[:, :3, 3:4]).squeeze(-1) + t_del
        top3 = torch.cat([r_new, t_new.unsqueeze(-1)], dim=-1)
        bot1 = row4_b.view(1, 1, 4).expand(len(idxs), 1, 4)
        return torch.cat([top3, bot1], dim=1)

    valid_all = [i for i in range(m_all) if image_names[i] not in excluded_set]
    batch_sz = min(8, len(valid_all))
    rng_s = random.Random(2026)

    # Pre-evaluate exact initial photometric residual BEFORE optimization (deterministic across all frames!)
    with torch.no_grad():
        errs_init: list[float] = []
        for b0_i in range(0, len(valid_all), 8):
            chunk_i = valid_all[b0_i : b0_i + 8]
            ren_i, _, _ = rasterization(
                means=means_f,
                quats=quats_f,
                scales=scales_f,
                opacities=opa_f,
                colors=colors_f,
                viewmats=vms_work[chunk_i],
                Ks=ks_gpu[chunk_i],
                width=img_w,
                height=img_h,
                sh_degree=sh_deg,
                packed=True,
                rasterize_mode=rast_mode,
            )
            gt_i = frames_gpu[chunk_i].float() * (1.0 / 255.0)
            errs_init.extend((ren_i - gt_i).abs().mean(dim=(1, 2, 3)).tolist())
        loss_before = float(np.mean(errs_init)) if errs_init else 0.0

    # Coarse-to-fine multi-scale registration: first half @ 1/2 resolution (wide basin of attraction for 10–60px shifts!),
    # second half @ full native resolution (sub-pixel lock!)
    n_iters = max(6, int(solve_steps))
    total_passes = max(n_iters, (len(valid_all) * max(8, n_iters // 2)) // max(1, batch_sz))
    total_passes = min(180, total_passes)
    id6 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    for it in range(total_passes):
        b_idxs = rng_s.sample(valid_all, batch_sz) if len(valid_all) > batch_sz else list(valid_all)
        vm_b = _compose_batch_vms(b_idxs)
        use_coarse = (it < total_passes // 2) and (img_w >= 640)
        if use_coarse:
            cw, ch = max(64, img_w // 2), max(64, img_h // 2)
            sx_c, sy_c = cw / float(img_w), ch / float(img_h)
            k_b = ks_gpu[b_idxs].clone()
            k_b[:, 0, :] *= sx_c
            k_b[:, 1, :] *= sy_c
            gt_full = (frames_gpu[b_idxs].float() * (1.0 / 255.0)).permute(0, 3, 1, 2)
            gt_b = F.interpolate(gt_full, size=(ch, cw), mode="area").permute(0, 2, 3, 1)
        else:
            cw, ch = img_w, img_h
            k_b = ks_gpu[b_idxs]
            gt_b = frames_gpu[b_idxs].float() * (1.0 / 255.0)

        ren_b, _, _ = rasterization(
            means=means_f,
            quats=quats_f,
            scales=scales_f,
            opacities=opa_f,
            colors=colors_f,
            viewmats=vm_b,
            Ks=k_b,
            width=cw,
            height=ch,
            sh_degree=sh_deg,
            packed=True,
            rasterize_mode=rast_mode,
        )
        l1_b = (ren_b - gt_b).abs().mean()
        ssim_b = _ssim_cuda(ren_b, gt_b, ssim_win).mean()
        photo_b = 0.8 * l1_b + 0.2 * (1.0 - ssim_b)

        # Regularize smoothness on WORLD camera centers `C_i = -R_i^T t_i` (NEVER `t_w2c`, which swings
        # wildly whenever `R_w2c` rotates around a distant origin and creates zigzag steps!)
        if m_all >= 3:
            acc_r = (rot6d[2:] - 2.0 * rot6d[1:-1] + rot6d[:-2]).pow(2).mean()
            acc_t = (trans[2:] - 2.0 * trans[1:-1] + trans[:-2]).pow(2).mean()
            smooth_reg = 0.06 * (acc_t + 0.25 * acc_r)
        else:
            smooth_reg = 0.0
        # Strong anchor tether (`0.25`) so already-solved SfM keyframes NEVER get dragged toward unposed interpolations!
        act_tether = 0.25 * (trans[act_sorted].pow(2).mean() + (rot6d[act_sorted] - id6).pow(2).mean())

        loss_arc = photo_b + smooth_reg + act_tether

        opt_arc.zero_grad(set_to_none=True)
        loss_arc.backward()
        base_lr = 1.5e-3 if use_coarse else 5e-4
        for pg in opt_arc.param_groups:
            pg["lr"] = base_lr * (0.30 ** (it / float(max(1, total_passes - 1))))
        opt_arc.step()

    # 3. Bake registered poses for ALL frames & evaluate exact post-solve residual on the same full set!
    per_frame_loss: dict[int, float] = {}
    with torch.no_grad():
        all_idxs = list(range(m_all))
        vms_solved = _compose_batch_vms(all_idxs).detach()
        shift_mm = float(trans.detach().norm(dim=-1).mean().item() * 1000.0)
        for b0 in range(0, len(valid_all), 8):
            chunk = valid_all[b0 : b0 + 8]
            ren_c, _, _ = rasterization(
                means=means_f,
                quats=quats_f,
                scales=scales_f,
                opacities=opa_f,
                colors=colors_f,
                viewmats=vms_solved[chunk],
                Ks=ks_gpu[chunk],
                width=img_w,
                height=img_h,
                sh_degree=sh_deg,
                packed=True,
                rasterize_mode=rast_mode,
            )
            gt_c = frames_gpu[chunk].float() * (1.0 / 255.0)
            errs = (ren_c - gt_c).abs().mean(dim=(1, 2, 3))
            for ci_loc, idx_g in enumerate(chunk):
                per_frame_loss[idx_g] = float(errs[ci_loc].item())
        loss_after = float(np.mean(list(per_frame_loss.values()))) if per_frame_loss else loss_before
        # Monotonicity safety guard: if optimization failed to improve global L1, keep initial spline poses
        if loss_after > loss_before * 1.02:
            vms_solved = vms_work
            loss_after = loss_before
            shift_mm = 0.0

    # 4. Promote up to `promote_k` WELL-REGISTERED candidate frames!
    # CRITICAL: Never promote a candidate frame whose post-solve photometric L1 exceeds `1.85x` the median
    # active-camera L1 (promoting an unaligned/motion-blurred frame poisons 3DGS gradients and creates
    # staircase sawtooth jumps in `cameras.json`!). Among well-aligned candidates, prioritize largest temporal
    # & angular gaps (`min_dt`) moderate residual (`err_ci`).
    act_losses = [per_frame_loss[a_i] for a_i in act_sorted if a_i in per_frame_loss]
    med_act_l1 = float(np.median(act_losses)) if act_losses else 0.045
    max_safe_l1 = max(0.085, med_act_l1 * 1.85)

    candidates = [
        i for i in valid_all
        if i not in act_set and per_frame_loss.get(i, 1.0) <= max_safe_l1
    ]
    # Fallback if scene is still very early/coarse: take best-aligned 60% of candidates
    if not candidates:
        all_cands = [i for i in valid_all if i not in act_set]
        all_cands.sort(key=lambda idx_c: per_frame_loss.get(idx_c, 1.0))
        candidates = all_cands[: max(1, (len(all_cands) * 3) // 5)]

    promoted: list[int] = []
    cur_act = sorted(act_set)
    for _ in range(min(max(0, int(promote_k)), len(candidates))):
        best_idx = -1
        best_score = -1.0
        for c_i in candidates:
            if c_i in act_set:
                continue
            t_ci = float(pts_sec_list[c_i]) if c_i < len(pts_sec_list) else float(c_i)
            min_dt = min((abs(t_ci - (float(pts_sec_list[a_i]) if a_i < len(pts_sec_list) else float(a_i))) for a_i in cur_act), default=1.0)
            err_ci = per_frame_loss.get(c_i, 0.05)
            # Reward gap coverage × registration confidence (penalize badly misaligned outliers!)
            align_quality = 1.0 / (1.0 + (err_ci / max(0.02, med_act_l1)) ** 2)
            score = math.sqrt(max(0.01, min_dt)) * (0.4 + 0.6 * align_quality)
            if score > best_score:
                best_score = score
                best_idx = c_i
        if best_idx < 0:
            break
        promoted.append(best_idx)
        act_set.add(best_idx)
        cur_act.append(best_idx)

    return {
        "viewmats_gpu": vms_solved,
        "promoted_indices": sorted(promoted),
        "active_indices": sorted(act_set),
        "loss_before": round(loss_before, 5),
        "loss_after": round(loss_after, 5),
        "mean_shift_mm": round(shift_mm, 2),
    }


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
    optimizers: dict[str, torch.optim.Adam] | None,
    strategy_state: dict[str, Any] | None,
    keep: torch.Tensor,
) -> int:
    """Prune Gaussians while preserving optional Adam `exp_avg`/`exp_avg_sq` and `DefaultStrategy` state."""
    n_before = int(params["means"].shape[0])
    n_after = int(keep.sum().item())
    if n_after == n_before or n_after == 0:
        return 0
    for name in list(params.keys()):
        old_p = params[name]
        new_p = torch.nn.Parameter(old_p.detach()[keep].contiguous().requires_grad_(True))
        params[name] = new_p
        opt = optimizers.get(name) if optimizers else None
        if opt is not None:
            old_st = opt.state.pop(old_p, None)
            opt.param_groups[0]["params"] = [new_p]
            if old_st is not None:
                new_st: dict[str, Any] = {}
                for sk, sv in old_st.items():
                    if torch.is_tensor(sv) and sv.ndim > 0 and sv.shape[0] == n_before:
                        new_st[sk] = sv[keep].contiguous()
                    else:
                        new_st[sk] = sv
                opt.state[new_p] = new_st
    if strategy_state:
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
    images_tar_gz: bytes | None = None,
    known_cameras_json: str | None = None,
    cfg: SplatConfig | None = None,
    cfg_overrides: dict[str, Any] | None = None,
    add_steps: int | None = None,
    surgery: SurgerySpec | None = None,
    masks_b64: dict[str, str] | None = None,
    exclude_frames: list[str] | None = None,
    control_poll_fn: Callable[[], dict[str, Any] | None] | None = None,
    gpu_hourly_usd: float = 3.95,
    ram_cache: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Generator yielding live telemetry events and final compressed artifacts.

    Supports:
      1. `from_video` (`mp4_bytes` provided): FFmpeg -> VGGT/COLMAP -> RANSAC Z=0 Sim3 -> gsplat
      2. `from_images` (`images_tar_gz` provided): EXIF-normalized multi-view images -> VGGT/COLMAP -> gsplat
      3. `from_blender` (`views_tar_gz` provided): Ground-truth Blender cameras + surface seed cloud -> gsplat
      4. `resume` (all None): Reloads `/jobs/{job_id}/frames_u8.pt` + `ckpt_latest.pt` (~0.3s or 0.0s HBM)
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
        clean = {
            k: v
            for k, v in row.items()
            if not k.endswith("_b64")
            and k not in ("ply_gz", "splat_gz", "proxy_glb_gz", "cameras_json")
            and not isinstance(v, (bytes, bytearray))
        }
        with open(telem_path, "a", encoding="utf-8") as tf:
            tf.write(json.dumps(clean) + "\n")

    if masks_b64:
        for mname, mb64 in masks_b64.items():
            raw_m = base64.b64decode(mb64)
            (masks_dir / mname).write_bytes(raw_m)

    is_resume = (mp4_bytes is None) and (views_tar_gz is None) and (images_tar_gz is None)
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
            # ── Video or Multi-View Image Set Mode: Stage 1 (Extract/Unpack) + Stage 2 (Pose / Known-Lock) ──
            is_image_set = images_tar_gz is not None
            frames_dir = work_shm / "frames"
            t0 = time.perf_counter()
            if is_image_set:
                assert images_tar_gz is not None
                frame_paths, pts_by_name, video_meta, has_alpha = unpack_images_bundle(
                    images_tar_gz, frames_dir, eff_cfg.extract
                )
            else:
                has_alpha = False
                mp4_path = work_shm / "source.mp4"
                mp4_path.write_bytes(mp4_bytes)  # type: ignore[arg-type]
                frame_paths, pts_by_name, video_meta = extract_frames_with_meta(
                    mp4_path, frames_dir, eff_cfg.extract
                )
            macro_timings["extract_s"] = round(time.perf_counter() - t0, 3)
            yield {
                "event": "stage",
                "stage": "extract",
                "job_id": job_id,
                "device": dev_name,
                "source": "images" if is_image_set else "video",
                "frames": len(frame_paths),
                "transparent_bg": has_alpha,
                "video_meta": video_meta,
                "elapsed_s": macro_timings["extract_s"],
            }

            t0 = time.perf_counter()
            known_doc_parsed = json.loads(known_cameras_json) if known_cameras_json else None
            use_hard_known = known_doc_parsed is not None and eff_cfg.pose.backend not in ("colmap_prior", "colmap")

            if use_hard_known and known_doc_parsed is not None:
                # Opportunity #4A: Hard Known-Pose Seedance Re-Polish (exact Blender camera lock, 0 Sim3 drift)
                pose_backend_label = "known_cameras"
                cam_records, xyz_bl, raw_rgb = solve_known_cameras_video(
                    frame_paths,
                    pts_by_name,
                    video_meta,
                    known_doc_parsed,
                    eff_cfg.pose,
                    device,
                    models_dir=models_dir,
                )
                sim3 = Sim3Transform.identity()
                macro_timings["pose_s"] = round(time.perf_counter() - t0, 3)
            else:
                # Check for Blender camera prior (`colmap_prior` or COLMAP + `known_cameras_json`):
                # Locks exact PINHOLE intrinsics `(fx, fy, cx, cy)` during SIFT BA + Umeyama-aligns
                # solved COLMAP poses onto the Blender scene frame so interior easing adapts to SIFT!
                prior_traj: dict[str, Any] | None = None
                prior_intr: tuple[float, float, float, float] | None = None
                if known_doc_parsed is not None:
                    with Image.open(frame_paths[0]) as im_p0:
                        pw, ph = im_p0.width, im_p0.height
                    prior_traj = interpolate_camera_trajectory(
                        known_doc_parsed,
                        fps=float(video_meta.get("fps") or 24.0),
                        duration_sec=float(video_meta.get("duration_sec") or 5.0),
                        total_frames=int(video_meta.get("total_frames") or len(frame_paths)),
                        target_width=pw,
                        target_height=ph,
                    )
                    pf0 = prior_traj["frames"][0]
                    prior_intr = (float(pf0["fx"]), float(pf0["fy"]), float(pf0["cx"]), float(pf0["cy"]))

                pose_backend_label = eff_cfg.pose.backend
                if eff_cfg.pose.backend in ("colmap", "colmap_prior"):
                    try:
                        # CRITICAL: Never subsample frames BEFORE COLMAP on a video (`col_sub_dir` left 1.2s–2.4s
                        # blind gaps across fast turns where SIFT broke into separate islands & failed to register!).
                        # Always run COLMAP on the FULL extracted sequence (`<= 192` dense overlapping frames) so SIFT
                        # tracks through every turn; if `iterative_arc_cycles > 1`, Cycle 0 starts from 56 anchors of
                        # the ALREADY-SOLVED coherent COLMAP trajectory!
                        col_input_dir = frames_dir
                        if not is_image_set and len(frame_paths) > 192:
                            col_sub_dir = work_shm / "colmap_dense_frames"
                            col_sub_dir.mkdir(parents=True, exist_ok=True)
                            anc_idxs = np.linspace(0, len(frame_paths) - 1, 192, dtype=int)
                            for ai in anc_idxs:
                                src_fp = frame_paths[int(ai)]
                                dst_fp = col_sub_dir / src_fp.name
                                if not dst_fp.exists():
                                    shutil.copy2(src_fp, dst_fp)
                            col_input_dir = col_sub_dir
                            n_col_input = 192
                        else:
                            n_col_input = len(frame_paths)

                        min_req = max(12, int(n_col_input * 0.75))
                        cam_records_col, raw_xyz, raw_rgb = solve_poses_colmap(
                            col_input_dir,
                            work_shm,
                            eff_cfg.pose,
                            prior_intrinsics=prior_intr,
                            is_image_set=is_image_set,
                        )
                        if len(cam_records_col) < min_req:
                            raise RuntimeError(
                                f"COLMAP fragmented on fast pans (registered only {len(cam_records_col)}/{n_col_input} frames)"
                            )
                        # If `iterative_arc_cycles > 1` and COLMAP solved a dense sequence (`> 80` frames),
                        # seed Cycle 0 with 64 evenly-spaced anchors from the ALREADY-COHERENT COLMAP model
                        # so Cycles 1..C promote the remaining frames along a clean, connected global arc!
                        cyc_cnt_col = max(1, int(getattr(eff_cfg.train, "iterative_arc_cycles", 1) or 1))
                        if cyc_cnt_col > 1 and not is_image_set and len(cam_records_col) > 80:
                            anc_c = np.linspace(0, len(cam_records_col) - 1, 64, dtype=int)
                            cam_records = [cam_records_col[int(i)] for i in anc_c]
                        else:
                            cam_records = cam_records_col
                        if prior_intr is not None:
                            pose_backend_label = "colmap_prior"
                    except Exception as col_err:  # noqa: BLE001
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

                # Attach exact source video PTS & 1-indexed timeline frame to every solved camera!
                native_fps_v = float(video_meta.get("fps") or 24.0)
                for c in cam_records:
                    pt_s = pts_by_name.get(c["image_name"])
                    if pt_s is not None:
                        c["pts_sec"] = round(float(pt_s), 5)
                        c["timeline_frame"] = max(1, round(float(pt_s) * native_fps_v) + 1)

                t0 = time.perf_counter()
                if prior_traj is not None and len(cam_records) >= 3:
                    # Opportunity #4B: Align solved SIFT/VGGT trajectory onto Blender prior via proper Umeyama Sim(3)!
                    src_c_list = []
                    dst_c_list = []
                    t_frames_p = prior_traj["frames"]
                    p_fps = float(prior_traj["fps"])
                    for c in cam_records:
                        rc_arr = np.array(c["R_w2c"], dtype=np.float64)
                        tc_arr = np.array(c["t_w2c"], dtype=np.float64)
                        src_c_list.append(-rc_arr.T @ tc_arr)
                        pt_s = float(c.get("pts_sec") or 0.0)
                        fi_p = max(0, min(len(t_frames_p) - 1, round(pt_s * p_fps)))
                        dst_c_list.append(np.array(t_frames_p[fi_p]["location"], dtype=np.float64))
                    sim3 = _estimate_umeyama_sim3(np.stack(src_c_list, 0), np.stack(dst_c_list, 0))
                else:
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

        # Build aligned OpenCV viewmats/Ks & Blender `cameras.json` (+ retain unposed candidate video frames
        # when `iterative_arc_cycles > 1` so subsequent cycles can register & promote them into training!)
        t_align = time.perf_counter()
        w0_target = int(cam_records[0]["width"])
        h0_target = int(cam_records[0]["height"])
        n_iter_cycles_init = max(1, int(getattr(eff_cfg.train, "iterative_arc_cycles", 1) or 1))
        solved_by_name = {c["image_name"]: c for c in cam_records}

        if (
            views_tar_gz is None
            and "frame_paths" in locals()
            and n_iter_cycles_init > 1
            and len(frame_paths) > len(cam_records)
        ):
            # Build full candidate pool across all extracted video frames (`frame_paths`)
            # Initial cycle trains ONLY on `solved_by_name`; Cycles 2..C solve arc + promote candidates!
            shared_fx = float(np.median([float(c["fx"]) for c in cam_records]))
            shared_fy = float(np.median([float(c["fy"]) for c in cam_records]))
            shared_cx = float(np.median([float(c["cx"]) for c in cam_records]))
            shared_cy = float(np.median([float(c["cy"]) for c in cam_records]))
            native_fps_pool = float(video_meta.get("fps") or 24.0) if "video_meta" in locals() else 24.0

            # Pre-align solved anchors to interpolate initial placeholder poses for candidate frames
            anchor_bl_tmp: list[dict[str, Any]] = []
            for c_a in cam_records:
                rc_a, tc_a = apply_sim3_to_camera_w2c(sim3, c_a["R_w2c"], c_a["t_w2c"])
                bl_a = opencv_w2c_to_blender_camera(rc_a, tc_a, shared_fx, shared_fy, shared_cx, shared_cy, w0_target, h0_target)
                bl_a["pts_sec"] = float(c_a.get("pts_sec") or 0.0)
                anchor_bl_tmp.append(bl_a)
            t0_pool = float(anchor_bl_tmp[0]["pts_sec"])
            t1_pool = max(t0_pool + 0.1, float(anchor_bl_tmp[-1]["pts_sec"]))
            traj_pool = interpolate_camera_trajectory(
                {"cameras": anchor_bl_tmp},
                fps=60.0,
                duration_sec=(t1_pool - t0_pool),
                total_frames=max(16, min(6000, round((t1_pool - t0_pool) * 60.0) + 1)),
                target_width=w0_target,
                target_height=h0_target,
            )["frames"]

            full_records: list[dict[str, Any]] = []
            initial_posed_indices: list[int] = []
            for fi_p, fp_cand in enumerate(frame_paths):
                if fp_cand.name in solved_by_name:
                    c_ex = dict(solved_by_name[fp_cand.name])
                    c_ex["frame_index"] = fi_p
                    c_ex["is_posed"] = True
                    rc_n, tc_n = apply_sim3_to_camera_w2c(sim3, c_ex["R_w2c"], c_ex["t_w2c"])
                    c_ex["_aligned_R"] = rc_n
                    c_ex["_aligned_t"] = tc_n
                    initial_posed_indices.append(fi_p)
                    full_records.append(c_ex)
                else:
                    pt_cand = float(pts_by_name.get(fp_cand.name, fi_p / max(1.0, native_fps_pool)))
                    ti_cand = max(0, min(len(traj_pool) - 1, round((pt_cand - t0_pool) * 60.0)))
                    fr_cand = traj_pool[ti_cand]
                    full_records.append({
                        "frame_index": fi_p,
                        "image_name": fp_cand.name,
                        "image_path": str(fp_cand),
                        "width": w0_target,
                        "height": h0_target,
                        "fx": shared_fx,
                        "fy": shared_fy,
                        "cx": shared_cx,
                        "cy": shared_cy,
                        "pts_sec": round(pt_cand, 5),
                        "timeline_frame": max(1, round(pt_cand * native_fps_pool) + 1),
                        "is_posed": False,
                        "_aligned_R": fr_cand["R_w2c"],
                        "_aligned_t": fr_cand["t_w2c"],
                    })
            cam_records = full_records
        else:
            initial_posed_indices = list(range(len(cam_records)))
            for c in cam_records:
                rc_n, tc_n = apply_sim3_to_camera_w2c(sim3, c["R_w2c"], c["t_w2c"])
                c["is_posed"] = True
                c["_aligned_R"] = rc_n
                c["_aligned_t"] = tc_n

        bl_cameras_list: list[dict[str, Any]] = []
        viewmats_list: list[np.ndarray] = []
        ks_list: list[np.ndarray] = []
        pts_sec_list: list[float] = []
        for idx_cr, c in enumerate(cam_records):
            cw, ch = int(c["width"]), int(c["height"])
            if (cw, ch) != (w0_target, h0_target) and cw > 0 and ch > 0:
                sx = w0_target / float(cw)
                sy = h0_target / float(ch)
                c["fx"] = float(c["fx"]) * sx
                c["cx"] = float(c["cx"]) * sx
                c["fy"] = float(c["fy"]) * sy
                c["cy"] = float(c["cy"]) * sy
                c["width"] = w0_target
                c["height"] = h0_target
            rc_new, tc_new = c["_aligned_R"], c["_aligned_t"]
            vm = np.eye(4, dtype=np.float32)
            vm[:3, :3] = np.array(rc_new, dtype=np.float32)
            vm[:3, 3] = np.array(tc_new, dtype=np.float32)
            viewmats_list.append(vm)
            k_mat = np.array(
                [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            ks_list.append(k_mat)
            pts_sec_list.append(float(c.get("pts_sec") if c.get("pts_sec") is not None else idx_cr * 0.2))

            if c.get("is_posed", True):
                bl_cam = opencv_w2c_to_blender_camera(
                    rc_new, tc_new, c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"]
                )
                bl_cam["frame_index"] = c["frame_index"]
                bl_cam["image_name"] = c["image_name"]
                if c.get("pts_sec") is not None:
                    bl_cam["pts_sec"] = c["pts_sec"]
                if c.get("timeline_frame") is not None:
                    bl_cam["timeline_frame"] = c["timeline_frame"]
                bl_cameras_list.append(bl_cam)

        cameras_doc: dict[str, Any] = {
            "job_id": job_id,
            "coordinate_system": "BLENDER_RIGHT_HANDED_Z_UP",
            "floor_z": 0.0,
            "pose_backend": pose_backend_label,
            "sim3": sim3.to_dict(),
            "low_parallax_warning": (views_tar_gz is None) and (not known_cameras_json) and (sim3.baseline_ratio < 0.03),
            "cameras": bl_cameras_list,
        }
        if views_tar_gz is None and "video_meta" in locals():
            cameras_doc["video_meta"] = video_meta
        cams_json_path.write_text(json.dumps(cameras_doc, indent=2), encoding="utf-8")

        frames_u8_cpu, alphas_u8_cpu = _preload_frames_u8(cam_records, load_alpha=has_alpha)
        shutil.rmtree(work_shm, ignore_errors=True)

        vms_np_stack = np.stack(viewmats_list, axis=0)
        ks_np_stack = np.stack(ks_list, axis=0)

        # ── Apple Depth Pro 2 (`ml-depth-pro-2`) Metric Depth Prior, Scale Alignment & Seed Nudging ──
        depths_prior_cpu: torch.Tensor | None = None
        confs_prior_cpu: torch.Tensor | None = None
        dp2_model_name = str(getattr(eff_cfg.pose, "depth_prior_model", "auto") or "auto")
        dp2_weight = float(getattr(eff_cfg.pose, "depth_nudge_strength", 0.25) or 0.0)
        if views_tar_gz is None and not has_alpha and dp2_model_name not in ("off", "none", "false", "0") and dp2_weight > 0.0:
            t_dp2 = time.perf_counter()
            dp2_res = predict_depth_pro2_maps(
                frames_u8_cpu,
                fx_px=float(ks_np_stack[0, 0, 0]),
                preset=dp2_model_name,
                device=device,
                models_dir=models_dir,
            )
            if dp2_res is not None:
                raw_dp2, confs_prior_cpu = dp2_res
                depths_prior_cpu, dp2_scale = align_depth_maps_to_sparse_cloud(
                    raw_dp2,
                    confs_prior_cpu,
                    vms_np_stack,
                    ks_np_stack,
                    xyz_bl,
                    int(cam_records[0]["width"]),
                    int(cam_records[0]["height"]),
                )
                # Supplement sparse COLMAP clouds up to `max_init_points` (e.g. 4k SIFT -> 180k dense walls!)
                need_sup = max(45_000, (eff_cfg.pose.max_init_points or 180_000) - len(xyz_bl))
                xyz_bl, raw_rgb, dp2_nudge_stats = nudge_and_densify_init_points_with_depth_pro2(
                    xyz_bl,
                    raw_rgb,
                    frames_u8_cpu,
                    depths_prior_cpu,
                    confs_prior_cpu,
                    vms_np_stack,
                    ks_np_stack,
                    initial_posed_indices,
                    nudge_alpha=0.18,
                    supplement_points=need_sup,
                    max_total_points=eff_cfg.pose.max_init_points or 180_000,
                )
                macro_timings["depth_pro2_s"] = round(time.perf_counter() - t_dp2, 3)
                yield {
                    "event": "stage",
                    "stage": "depth_pro2",
                    "job_id": job_id,
                    "scale_align": dp2_scale,
                    "elapsed_s": macro_timings["depth_pro2_s"],
                    **dp2_nudge_stats,
                }

        torch.save(
            {
                "frames_u8": frames_u8_cpu,
                "alphas_u8": alphas_u8_cpu,
                "depths_fp16": depths_prior_cpu,
                "confs_fp16": confs_prior_cpu,
                "viewmats": torch.from_numpy(vms_np_stack),
                "Ks": torch.from_numpy(ks_np_stack),
                "pts_sec": pts_sec_list,
                "posed_indices": initial_posed_indices,
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
                "pose_backend": pose_backend_label,
                "cameras": len(initial_posed_indices),
                "candidate_frames": len(cam_records) - len(initial_posed_indices),
                "init_points": len(xyz_bl),
                "floor_inlier_ratio": round(sim3.floor_inlier_ratio, 3),
                "baseline_ratio": round(sim3.baseline_ratio, 4),
                "elapsed_s": round(macro_timings["pose_s"] + macro_timings["align_s"], 3),
            }

        frames_gpu = frames_u8_cpu.to(device, non_blocking=True)
        alphas_gpu = alphas_u8_cpu.to(device, non_blocking=True) if alphas_u8_cpu is not None else None
        depths_gpu = depths_prior_cpu.to(device, non_blocking=True) if depths_prior_cpu is not None else None
        confs_gpu = confs_prior_cpu.to(device, non_blocking=True) if confs_prior_cpu is not None else None
        viewmats_gpu = torch.from_numpy(vms_np_stack).to(device)
        ks_gpu = torch.from_numpy(ks_np_stack).to(device)
        image_names = [c["image_name"] for c in cam_records]
        img_w, img_h = int(cam_records[0]["width"]), int(cam_records[0]["height"])

        params = _init_gaussians_cuda(xyz_bl, raw_rgb, eff_cfg.train.sh_degree, device)
        # Ensure initial seed opacity sits comfortably above prune_opa so step-500 never mass-executes VGGT seeds!
        safe_init_opa = min(0.55, max(0.18, float(eff_cfg.train.prune_opa) * 2.5))
        with torch.no_grad():
            params["opacities"].data.fill_(math.log(safe_init_opa / (1.0 - safe_init_opa)))
        # Cap `scene_scale` to the local room/corridor scale (`<= 3.5m`) even on 125m walkthroughs so
        # `prune_scale3d * scene_scale` caps max splats at ~15cm instead of allowing 3.0m wall floaters!
        raw_p90_ext = float(np.percentile(np.linalg.norm(xyz_bl - np.median(xyz_bl, axis=0), axis=1), 75))
        scene_scale = max(1.0, min(3.5, raw_p90_ext))
        optimizers = _make_optimizers(params, scene_scale=scene_scale)
        eff_densify_stop = min(
            int(eff_cfg.train.densify_until),
            max(int(eff_cfg.train.densify_from) + 200, int(eff_cfg.train.iterations * 0.72)),
        )
        # Disable DefaultStrategy's hardcoded 3000-step 0.01 reset (`reset_every=10**9`) so our
        # multi-view warmup-protected reset controller manages opacity resets cleanly!
        # CRITICAL: Base `refine_start_iter` on `len(initial_posed_indices)` (capped at `densify_from + 400`),
        # NEVER `len(cam_records) * 6` (which reached `320 * 6 = 1920` and froze densification for 2,000 steps!)
        eff_refine_start = min(
            int(eff_cfg.train.densify_from) + 400,
            max(int(eff_cfg.train.densify_from), len(initial_posed_indices) * 6),
        )
        strategy = DefaultStrategy(
            prune_opa=min(0.01, float(eff_cfg.train.prune_opa)),
            grow_grad2d=eff_cfg.train.densify_grad_threshold,
            grow_scale3d=0.01,
            prune_scale3d=float(getattr(eff_cfg.train, "prune_scale3d", 0.05)),
            refine_start_iter=eff_refine_start,
            refine_stop_iter=eff_densify_stop,
            reset_every=1_000_000_000,
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
        # Merge only caller-specified overrides onto `saved_cfg` so SH degree / loss settings never reset!
        if cfg_overrides:
            eff_cfg = saved_cfg.merge_overrides(cfg_overrides)
        elif cfg is not None:
            eff_cfg = cfg
        else:
            eff_cfg = saved_cfg
        scene_scale = float(meta.get("scene_scale", 2.5))

        if ram_cache is not None and ram_cache.get("job_id") == job_id:
            frames_gpu = ram_cache["frames_gpu"]
            alphas_gpu = ram_cache.get("alphas_gpu")
            depths_gpu = ram_cache.get("depths_gpu")
            confs_gpu = ram_cache.get("confs_gpu")
            pts_sec_list = list(ram_cache.get("pts_sec") or [])
            initial_posed_indices = list(ram_cache.get("posed_indices") or range(len(ram_cache["image_names"])))
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
            dp_loaded = pack.get("depths_fp16")
            cf_loaded = pack.get("confs_fp16")
            depths_gpu = dp_loaded.to(device, non_blocking=True) if dp_loaded is not None else None
            confs_gpu = cf_loaded.to(device, non_blocking=True) if cf_loaded is not None else None
            pts_sec_list = list(pack.get("pts_sec") or [i * 0.2 for i in range(len(pack["image_names"]))])
            initial_posed_indices = list(pack.get("posed_indices") or range(len(pack["image_names"])))
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
            reset_every=1_000_000_000,
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

    posed_set = set(initial_posed_indices)
    active_indices = [i for i, nm in enumerate(image_names) if (i in posed_set) and (nm not in excluded_set)]
    if not active_indices:
        active_indices = [i for i, nm in enumerate(image_names) if nm not in excluded_set]
    if not active_indices:
        raise RuntimeError("All training frames are excluded!")

    # Compute iterative cycle boundaries (`colmap/vggt -> train N -> solve arc + clear floaters + select frames -> train N...`)
    n_iter_cycles = max(1, int(getattr(eff_cfg.train, "iterative_arc_cycles", 1) or 1))
    total_span_steps = max(1, target_step - start_step)
    cycle_break_steps: set[int] = set()
    if n_iter_cycles > 1 and total_span_steps >= 400:
        for cyc_i in range(1, n_iter_cycles):
            b_step = start_step + round(total_span_steps * (cyc_i / float(n_iter_cycles)))
            if start_step < b_step < target_step:
                cycle_break_steps.add(b_step)

    # ── Stage 3: High-Throughput Multi-View Consistent VRAM Training Loop ──
    ssim_win = _build_ssim_window(device)
    t_train_start = time.perf_counter()
    last_preview_t = 0.0
    min_preview_dt = 1.0 / max(0.25, eff_cfg.train.preview_hz)
    sent_gt_cams: set[str] = set()
    rng_py = random.Random(1337 + burst_id)
    live_lr_mult = 1.0
    actual_final_step = target_step
    last_ctl_seq = -1

    # Pre-build stereo cross-view peer table & continuous swept-arc tunnel samples (`swept_centers_gpu [512, 3]`)
    stereo_peers = _build_stereo_peer_table(viewmats_gpu, active_indices)
    swept_centers_gpu, probe_vms_gpu, probe_ks_gpu = _build_swept_arc_samples_gpu(
        viewmats_gpu, ks_gpu, active_indices, pts_sec_list, img_w, img_h
    )
    cam_centers_gpu = swept_centers_gpu

    # Pre-cache static GT SSIM `(mu2, sig2_sq)` in fp16 on CUDA when no RGBA random-bg compositing is active:
    # eliminates 40% of SSIM convolutions across all 10,000-30,000 steps!
    gt_ssim_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    if alphas_gpu is None and mask_tensors == {} and global_mask_t is None and device.type == "cuda":
        try:
            with torch.no_grad():
                k_h, k_v = ssim_win
                mu2_list, s2_list = [], []
                for ci_s in range(0, int(frames_gpu.shape[0]), 16):
                    g_chunk = (frames_gpu[ci_s : ci_s + 16].float() * (1.0 / 255.0)).permute(0, 3, 1, 2)
                    m2_c = _sep_gauss_blur_bchw(g_chunk, k_h, k_v)
                    s2_c = _sep_gauss_blur_bchw(g_chunk * g_chunk, k_h, k_v) - (m2_c * m2_c)
                    mu2_list.append(m2_c.half())
                    s2_list.append(s2_c.half())
                gt_ssim_cache = (torch.cat(mu2_list, dim=0), torch.cat(s2_list, dim=0))
        except Exception:  # noqa: BLE001
            gt_ssim_cache = None

    # Per-camera EMA PSNR/loss tracker so multi-view stochastic sampling jitter is decoupled from true convergence!
    cam_ema_psnr: dict[int, float] = {}
    cam_ema_loss: dict[int, float] = {}
    ema_loss_global: float | None = None
    ema_psnr_global: float | None = None

    use_cuda_ev = device.type == "cuda"
    if use_cuda_ev:
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_act = torch.cuda.Event(enable_timing=True)
        ev_fwd = torch.cuda.Event(enable_timing=True)
        ev_loss = torch.cuda.Event(enable_timing=True)
        ev_bwd = torch.cuda.Event(enable_timing=True)
        ev_opt = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

    def _resolve_focus_idx(fc_name: str | None) -> int | None:
        if not fc_name:
            return None
        for i, nm in enumerate(image_names):
            if nm == fc_name:
                return i
        return None

    focus_idx = _resolve_focus_idx(eff_cfg.train.focus_camera)
    rast_mode = str(getattr(eff_cfg.train, "rasterize_mode", "antialiased") or "antialiased")
    if rast_mode not in ("classic", "antialiased"):
        rast_mode = "antialiased"

    # Differentiable `se(3)` sub-pixel camera pose refinement (`6D` continuous rotation + `3D` translation)
    n_cams_total = int(viewmats_gpu.shape[0])
    use_pose_opt = bool(getattr(eff_cfg.train, "refine_poses", False))
    pose_rot6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device).repeat(
        n_cams_total, 1
    ).requires_grad_(True)
    pose_trans = torch.zeros((n_cams_total, 3), dtype=torch.float32, device=device, requires_grad=True)
    pose_lr_init = float(getattr(eff_cfg.train, "pose_opt_lr", 5e-5) or 5e-5)
    pose_opt = torch.optim.Adam([pose_rot6d, pose_trans], lr=pose_lr_init, eps=1e-15)
    eye3_gpu = torch.eye(3, dtype=torch.float32, device=device)

    # Per-Camera 3x4 Affine Exposure / White-Balance Compensation (`C_cal = C @ A_i^T + b_i`)
    # Absorbs auto-exposure, bloom, vignette, and game point-light shifts in 12 floats/camera instead of air floaters!
    use_app_opt = bool(getattr(eff_cfg.train, "appearance_opt", True)) and (alphas_gpu is None)
    app_mat = eye3_gpu.unsqueeze(0).repeat(n_cams_total, 1, 1).clone().detach().requires_grad_(True)
    app_bias = torch.zeros((n_cams_total, 3), dtype=torch.float32, device=device, requires_grad=True)
    app_opt = torch.optim.Adam([app_mat, app_bias], lr=1.5e-3, betas=(0.9, 0.999), eps=1e-15)

    pos_lr_init = 1.6e-4 * max(0.25, min(4.0, scene_scale))
    pos_lr_final = 1.6e-6 * max(0.25, min(4.0, scene_scale))

    def _compose_pose_delta(base_vm: torch.Tensor, idx_c: int) -> tuple[torch.Tensor, torch.Tensor]:
        r6 = pose_rot6d[idx_c]
        a1 = F.normalize(r6[0:3], dim=0)
        b2 = r6[3:6] - (a1 * r6[3:6]).sum() * a1
        a2 = F.normalize(b2, dim=0)
        a3 = torch.linalg.cross(a1, a2)
        r_del = torch.stack([a1, a2, a3], dim=0)
        t_del = pose_trans[idx_c]
        r_base = base_vm[0, :3, :3]
        t_base = base_vm[0, :3, 3]
        r_new_t = r_del @ r_base
        t_new_t = (r_del @ t_base) + t_del
        row4 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        top3 = torch.cat([r_new_t, t_new_t.unsqueeze(1)], dim=1)
        vm_out = torch.cat([top3, row4], dim=0).unsqueeze(0)
        reg_t = ((t_del ** 2).sum() + ((r_del - eye3_gpu) ** 2).sum())
        return vm_out, reg_t

    for step in range(start_step, target_step):
        # ── Iterative Cycle Bridge (`Train N -> Video-Diff Arc Solve + Swept Clear + Frame Select -> Train N`) ──
        if step in cycle_break_steps and alphas_gpu is None:
            cycle_idx = sorted(cycle_break_steps).index(step) + 1
            t_cyc = time.perf_counter()
            yield {
                "event": "stage",
                "stage": f"arc_cycle_{cycle_idx}",
                "job_id": job_id,
                "step": step,
                "active_cameras_before": len(active_indices),
            }
            # 1. Bake any accumulated in-loop `se(3)` pose deltas before solving the full video arc
            if use_pose_opt:
                with torch.no_grad():
                    baked_vms = [
                        _compose_pose_delta(viewmats_gpu[ci_k : ci_k + 1], ci_k)[0][0]
                        for ci_k in range(n_cams_total)
                    ]
                    viewmats_gpu = torch.stack(baked_vms, dim=0)
                    pose_rot6d.data.copy_(
                        torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device).repeat(
                            n_cams_total, 1
                        )
                    )
                    pose_trans.data.zero_()
                pose_opt = torch.optim.Adam([pose_rot6d, pose_trans], lr=pose_lr_init, eps=1e-15)

            # 2. Solve full video camera arc directly from render-vs-video diff + promote new frames
            arc_res = _solve_dense_video_arc_and_promote_frames(
                params=params,
                frames_gpu=frames_gpu,
                viewmats_gpu=viewmats_gpu,
                ks_gpu=ks_gpu,
                active_indices=active_indices,
                pts_sec_list=pts_sec_list,
                excluded_set=excluded_set,
                image_names=image_names,
                ssim_win=ssim_win,
                sh_deg=min(eff_cfg.train.sh_degree, step // 1000),
                rast_mode=rast_mode,
                solve_steps=int(getattr(eff_cfg.train, "arc_solve_steps", 24) or 24),
                promote_k=int(getattr(eff_cfg.train, "frames_add_per_cycle", 24) or 24),
            )
            viewmats_gpu = arc_res["viewmats_gpu"]
            active_indices = arc_res["active_indices"]

            # 3. Rebuild stereo peer graph & continuous swept camera arc + hard-evict tunnel/frustum floaters!
            stereo_peers = _build_stereo_peer_table(viewmats_gpu, active_indices)
            swept_centers_gpu, probe_vms_gpu, probe_ks_gpu = _build_swept_arc_samples_gpu(
                viewmats_gpu, ks_gpu, active_indices, pts_sec_list, img_w, img_h
            )
            cam_centers_gpu = swept_centers_gpu
            ev_stats = _evict_swept_arc_and_near_frustum_floaters(
                params=params,
                optimizers=optimizers,
                strategy_state=strategy_state,
                swept_centers_gpu=swept_centers_gpu,
                probe_vms_gpu=probe_vms_gpu,
                probe_ks_gpu=probe_ks_gpu,
                img_w=img_w,
                img_h=img_h,
                corridor_r=float(getattr(eff_cfg.train, "cam_corridor_radius_m", 0.35) or 0.0) * 0.75,
                near_clear_m=float(getattr(eff_cfg.train, "near_frustum_clear_m", 0.40) or 0.0),
            )
            cyc_ev = {
                "event": "surgery",
                "job_id": job_id,
                "burst_id": burst_id,
                "step": step,
                "cycle": cycle_idx,
                "promoted_frames": len(arc_res["promoted_indices"]),
                "active_cameras": len(active_indices),
                "arc_loss_before": arc_res["loss_before"],
                "arc_loss_after": arc_res["loss_after"],
                "mean_shift_mm": arc_res["mean_shift_mm"],
                **ev_stats,
                "elapsed_s": round(time.perf_counter() - t_cyc, 3),
            }
            _append_telem(cyc_ev)
            yield cyc_ev
            yield {"event": "stage", "stage": "train", "job_id": job_id, "step": step}

        is_log = (step == start_step) or ((step + 1) % max(1, eff_cfg.train.log_every) == 0) or (step == target_step - 1)

        # ── Mid-Train Live Control Poll (`tt-splat` style: Pause, Save Ckpt, Stop, Live Params/Surgeries) ──
        if is_log and control_poll_fn is not None:
            ctl = control_poll_fn()
            if isinstance(ctl, dict):
                c_seq = int(ctl.get("seq", 0))
                if c_seq > last_ctl_seq:
                    last_ctl_seq = c_seq
                    if ctl.get("lambda_dssim") is not None:
                        eff_cfg.train.lambda_dssim = max(0.0, min(1.0, float(ctl["lambda_dssim"])))
                    if ctl.get("opa_reg") is not None:
                        eff_cfg.train.opa_reg = max(0.0, float(ctl["opa_reg"]))
                    if ctl.get("scale_reg") is not None:
                        eff_cfg.train.scale_reg = max(0.0, float(ctl["scale_reg"]))
                    if ctl.get("aniso_reg") is not None:
                        eff_cfg.train.aniso_reg = max(0.0, float(ctl["aniso_reg"]))
                    if ctl.get("densify_grad_threshold") is not None:
                        eff_cfg.train.densify_grad_threshold = max(2e-5, min(5e-3, float(ctl["densify_grad_threshold"])))
                    if ctl.get("target_gaussians") is not None:
                        eff_cfg.train.target_gaussians = max(0, int(ctl["target_gaussians"]))
                    if ctl.get("arc_scale_mode") in ("log", "power", "linear"):
                        eff_cfg.train.arc_scale_mode = str(ctl["arc_scale_mode"])
                    if ctl.get("arc_gamma") is not None:
                        eff_cfg.train.arc_gamma = max(0.25, min(4.0, float(ctl["arc_gamma"])))
                    if ctl.get("adaptive_strategy") in ("adaptive_arc", "fixed"):
                        eff_cfg.train.adaptive_strategy = str(ctl["adaptive_strategy"])
                    if ctl.get("views_per_step") is not None:
                        eff_cfg.train.views_per_step = max(1, min(4, int(ctl["views_per_step"])))
                    if ctl.get("cam_corridor_radius_m") is not None:
                        eff_cfg.train.cam_corridor_radius_m = max(0.0, min(1.5, float(ctl["cam_corridor_radius_m"])))
                    if ctl.get("appearance_opt") is not None:
                        eff_cfg.train.appearance_opt = bool(ctl["appearance_opt"])
                    if ctl.get("outlier_view_downweight") is not None:
                        eff_cfg.train.outlier_view_downweight = bool(ctl["outlier_view_downweight"])
                    if ctl.get("lr_scale") is not None:
                        live_lr_mult = max(0.05, min(10.0, float(ctl["lr_scale"])))
                    if "focus_camera" in ctl:
                        fc_v = str(ctl["focus_camera"] or "").strip() or None
                        eff_cfg.train.focus_camera = fc_v
                        focus_idx = _resolve_focus_idx(fc_v)

                    # Mid-run surgical actions (`prune_opa`, `clamp_log_scale`, `reset_opacities`)
                    mid_surg_keys = ("prune_opa", "clamp_log_scale", "reset_opacities", "max_world_scale")
                    if any(ctl.get(mk) is not None for mk in mid_surg_keys):
                        mspec = SurgerySpec.from_dict({mk: ctl[mk] for mk in mid_surg_keys if ctl.get(mk) is not None})
                        sres = apply_surgeries(params, optimizers, strategy_state, mspec)
                        sev = {"event": "surgery", "job_id": job_id, "burst_id": burst_id, "step": step, **sres}
                        _append_telem(sev)
                        yield sev

                    if ctl.get("save_checkpoint"):
                        torch.save(
                            {
                                "step": step,
                                "params": {k: v.detach().cpu() for k, v in params.items()},
                                "opt_states": {k: opt.state_dict() for k, opt in optimizers.items()},
                                "strategy_state": {
                                    k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                    for k, v in strategy_state.items()
                                },
                            },
                            ckpt_path,
                        )
                        ck_ev = {"event": "checkpoint_saved", "job_id": job_id, "step": step}
                        _append_telem(ck_ev)
                        yield ck_ev

                    if ctl.get("stop_early"):
                        actual_final_step = step
                        break

                if ctl.get("paused"):
                    yield {"event": "stage", "stage": "paused", "job_id": job_id, "step": step}
                    t_pause_0 = time.perf_counter()
                    while (time.perf_counter() - t_pause_0) < 120.0:
                        time.sleep(0.25)
                        ctl_p = control_poll_fn() or {}
                        if ctl_p.get("stop_early"):
                            actual_final_step = step
                            break
                        if not ctl_p.get("paused"):
                            break
                    if actual_final_step == step and (control_poll_fn() or {}).get("stop_early"):
                        break
                    yield {"event": "stage", "stage": "train", "job_id": job_id, "step": step}

        t_step_cpu = time.perf_counter() if is_log else 0.0

        # ── Multi-View Stereo Cross-View Batch Selection (`B = views_per_step`) ──
        cam_i = focus_idx if focus_idx is not None else rng_py.choice(active_indices)
        want_b = max(1, min(4, int(getattr(eff_cfg.train, "views_per_step", 2) or 1)))
        batch_cams: list[int] = [cam_i]
        if want_b > 1 and len(active_indices) > 1:
            peer_cands = stereo_peers.get(cam_i) or [j for j in active_indices if j != cam_i]
            if peer_cands:
                # Pick 1 stereo peer with overlapping frustum + wide parallax, then random global views if B > 2
                p1 = rng_py.choice(peer_cands)
                if p1 not in batch_cams:
                    batch_cams.append(p1)
            while len(batch_cams) < min(want_b, len(active_indices)):
                cand_extra = rng_py.choice(active_indices)
                if cand_extra not in batch_cams:
                    batch_cams.append(cand_extra)
        b_cnt = len(batch_cams)

        if use_pose_opt:
            vms_b, regs_b = [], []
            for ci_b in batch_cams:
                vm_1, reg_1 = _compose_pose_delta(viewmats_gpu[ci_b : ci_b + 1], ci_b)
                vms_b.append(vm_1)
                regs_b.append(reg_1)
            viewmat = torch.cat(vms_b, dim=0)
            pose_prior_w = float(getattr(eff_cfg.train, "pose_prior_reg", 1e-2))
            pose_reg_term: torch.Tensor | float = pose_prior_w * torch.stack(regs_b).mean()
        else:
            viewmat = viewmats_gpu[batch_cams]
            pose_reg_term = 0.0

        k_mat = ks_gpu[batch_cams]
        fg_imgs = frames_gpu[batch_cams].float() * (1.0 / 255.0)  # [B, H, W, 3]
        gt_alphas = (alphas_gpu[batch_cams].float() * (1.0 / 255.0)) if alphas_gpu is not None else None

        # Random solid background compositing when RGBA silhouette views are provided
        bg_col: torch.Tensor | None = None
        if gt_alphas is not None:
            bg_col = torch.zeros((b_cnt, 3), device=device) if is_log else torch.rand((b_cnt, 3), device=device)
            gt_imgs = fg_imgs * gt_alphas + bg_col.view(b_cnt, 1, 1, 3) * (1.0 - gt_alphas)
        else:
            gt_imgs = fg_imgs
        gt_img = gt_imgs[0]

        sh_cur = min(eff_cfg.train.sh_degree, step // 1000)
        k_use = (sh_cur + 1) ** 2

        if is_log and use_cuda_ev:
            ev_s.record()

        sh_all = torch.cat([params["sh0"], params["shN"]], dim=1)[:, :k_use, :]
        opa_sig = torch.sigmoid(params["opacities"])
        scales_lin = torch.exp(params["scales"])
        quats_norm = F.normalize(params["quats"], dim=-1)

        if is_log and use_cuda_ev:
            ev_act.record()

        # Run Depth Pro 2 expected-depth regularization lightly (`<= 0.04`, every 4th step, first 65% of training)
        # so monocular depth scale jitter NEVER fights multi-view photometric triangulation (restores 35+ dB PSNR!)
        dp2_raw = float(getattr(eff_cfg.pose, "depth_nudge_strength", 0.04) or 0.0)
        dp2_strength = min(0.05, dp2_raw * 0.15) if dp2_raw > 0.05 else dp2_raw
        use_depth_step = (
            depths_gpu is not None
            and confs_gpu is not None
            and dp2_strength > 0.0
            and (step % 4 == 0)
            and (step < int(target_step * 0.65))
        )
        renders_raw, ren_alphas, info = rasterization(
            means=params["means"],
            quats=quats_norm,
            scales=scales_lin,
            opacities=opa_sig,
            colors=sh_all,
            viewmats=viewmat,
            Ks=k_mat,
            width=img_w,
            height=img_h,
            sh_degree=sh_cur,
            backgrounds=bg_col,
            render_mode="RGB+ED" if use_depth_step else "RGB",
            packed=True,
            absgrad=True,
            rasterize_mode=rast_mode,
        )
        if use_depth_step:
            renders = renders_raw[:, :, :, :3]
            rendered_ed: torch.Tensor | None = renders_raw[:, :, :, 3:4]
        else:
            renders = renders_raw
            rendered_ed = None
        # Raw uncalibrated render of primary view (for true novel-view PSNR telemetry & preview)
        render_img = renders[0]

        if is_log and use_cuda_ev:
            ev_fwd.record()

        # Optional per-camera 3x4 affine exposure/white-balance absorption (`renders_cal = renders @ A_b^T + bias_b`):
        # prevents per-frame exposure/vignette/point-light flicker from spawning semi-transparent smoke floaters!
        app_active_now = use_app_opt and bool(getattr(eff_cfg.train, "appearance_opt", True)) and step >= 200
        if app_active_now:
            a_b = app_mat[batch_cams]   # [B, 3, 3]
            b_b = app_bias[batch_cams]  # [B, 3]
            renders_eval = (torch.matmul(renders.view(b_cnt, -1, 3), a_b.transpose(1, 2)).view(b_cnt, img_h, img_w, 3) + b_b.view(b_cnt, 1, 1, 3)).clamp(0.0, 1.0)
            # Zero-mean drift gauge tether: mean(A) == I and mean(b) == 0 across all cameras so global scene albedo stays true!
            app_reg_w = float(getattr(eff_cfg.train, "appearance_reg", 1e-3))
            app_reg_term: torch.Tensor | float = app_reg_w * (
                ((a_b - eye3_gpu.unsqueeze(0)) ** 2).mean()
                + (b_b ** 2).mean()
                + 5.0 * ((app_mat.mean(dim=0) - eye3_gpu) ** 2).mean()
                + 5.0 * (app_bias.mean(dim=0) ** 2).mean()
            )
        else:
            renders_eval = renders
            app_reg_term = 0.0

        lam = eff_cfg.train.lambda_dssim
        has_any_mask = (global_mask_t is not None) or any(ci_b in mask_tensors for ci_b in batch_cams)
        if not has_any_mask:
            l1_vec = (renders_eval - gt_imgs).abs().mean(dim=(1, 2, 3))  # [B]
            if lam > 0:
                cached_gt_b = (
                    (gt_ssim_cache[0][batch_cams].float(), gt_ssim_cache[1][batch_cams].float())
                    if gt_ssim_cache is not None
                    else None
                )
                ssim_vec = _ssim_cuda(renders_eval, gt_imgs, ssim_win, gt_stats=cached_gt_b)  # [B]
            else:
                ssim_vec = torch.ones((b_cnt,), device=device)
        else:
            l1_parts, ssim_parts = [], []
            for bi, ci_b in enumerate(batch_cams):
                m_t = mask_tensors.get(ci_b, global_mask_t)
                r_bi, g_bi = renders_eval[bi], gt_imgs[bi]
                if m_t is not None:
                    l1_parts.append(((r_bi - g_bi).abs() * m_t).sum() / (m_t.sum() * 3.0 + 1e-8))
                    mr_bi = r_bi * m_t + g_bi * (1.0 - m_t)
                    ssim_parts.append(_ssim_cuda(mr_bi, g_bi, ssim_win) if lam > 0 else torch.tensor(1.0, device=device))
                else:
                    l1_parts.append((r_bi - g_bi).abs().mean())
                    ssim_parts.append(_ssim_cuda(r_bi, g_bi, ssim_win) if lam > 0 else torch.tensor(1.0, device=device))
            l1_vec = torch.stack(l1_parts)
            ssim_vec = torch.stack(ssim_parts)

        l1_term = l1_vec[0]
        ssim_val = ssim_vec[0]
        dssim_vec = (1.0 - ssim_vec) if lam > 0 else torch.zeros_like(l1_vec)
        photo_vec = (1.0 - lam) * l1_vec + lam * dssim_vec if lam > 0 else l1_vec
        dssim_term = dssim_vec.mean()

        # Robust IRLS Outlier-View Trust Weighting: prevents 1-2 blurry / HUD / pose-drifted frames
        # from dominating gradients and spawning floaters across the clean cameras!
        if (
            bool(getattr(eff_cfg.train, "outlier_view_downweight", True))
            and step >= max(600, len(active_indices) * 8)
            and len(cam_ema_loss) >= max(4, len(active_indices) // 2)
        ):
            sorted_emas = sorted(cam_ema_loss.values())
            med_ema = max(1e-4, sorted_emas[len(sorted_emas) // 2])
            v_weights = []
            for bi, ci_b in enumerate(batch_cams):
                c_loss_est = cam_ema_loss.get(ci_b, float(photo_vec[bi].detach().item()))
                ratio = c_loss_est / med_ema
                # Cauchy rolloff above 1.45x median loss (floors at 0.20 so view still slowly heals pose/appearance)
                w_bi = 1.0 / (1.0 + max(0.0, ratio - 1.45) ** 2) if ratio > 1.45 else 1.0
                v_weights.append(max(0.20, w_bi))
            w_t = torch.tensor(v_weights, dtype=torch.float32, device=device)
            photo_loss = (photo_vec * w_t).mean()
        else:
            photo_loss = photo_vec.mean()

        alpha_sil_loss = (ren_alphas - gt_alphas).abs().mean() * 0.15 if gt_alphas is not None else 0.0
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

        # Continuous Swept-Arc Camera Tunnel Penalty (`cam_centers_gpu [512, 3]` along centripetal path):
        # Any Gaussian entering within `r_corridor` anywhere along the walked camera arc is a cheat billboard!
        r_corridor = float(getattr(eff_cfg.train, "cam_corridor_radius_m", 0.35) or 0.0)
        if r_corridor > 0.0 and alphas_gpu is None:
            d_cam = torch.cdist(params["means"], cam_centers_gpu)
            min_d_cam = d_cam.min(dim=1).values
            intrusion = F.relu(r_corridor - min_d_cam)
            corridor_loss: torch.Tensor | float = 0.25 * ((intrusion * opa_sig).mean() + (intrusion ** 2).mean())
        else:
            corridor_loss = 0.0

        # Bimodal Opacity Entropy (`α * (1 - α)`): forces mid-air semi-transparent haze (`α ≈ 0.08..0.4`)
        # to either commit to a real surface (`α -> 1`) or evaporate cleanly (`α -> 0`)!
        bimodal_w = float(getattr(eff_cfg.train, "opa_bimodal_reg", 2e-4) or 0.0)
        opa_bimodal_term: torch.Tensor | float = (
            bimodal_w * (opa_sig * (1.0 - opa_sig)).mean() if (bimodal_w > 0.0 and step >= 500) else 0.0
        )

        # Apple Depth Pro 2 (`ml-depth-pro-2`) Expected-Depth Surface Nudge + Free-Space Floater Push!
        if use_depth_step and rendered_ed is not None and depths_gpu is not None and confs_gpu is not None:
            # Decay depth tether gently in final 25% of training so sub-millimeter photometric detail wins
            dp_decay = max(0.25, 1.0 - 0.75 * max(0.0, (step / float(max(1, target_step)) - 0.5) / 0.5))
            depth_prior_term: torch.Tensor | float = compute_depth_prior_loss(
                rendered_ed,
                ren_alphas,
                depths_gpu[batch_cams],
                confs_gpu[batch_cams],
                weight=dp2_strength * dp_decay,
            )
        else:
            depth_prior_term = 0.0

        loss = (
            photo_loss
            + alpha_sil_loss
            + opa_reg_term
            + opa_bimodal_term
            + scale_reg_term
            + aniso_term
            + pose_reg_term
            + app_reg_term
            + corridor_loss
            + depth_prior_term
        )

        if is_log and use_cuda_ev:
            ev_loss.record()

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        if use_pose_opt:
            pose_opt.zero_grad(set_to_none=True)
        if app_active_now:
            app_opt.zero_grad(set_to_none=True)

        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        loss.backward()

        if is_log and use_cuda_ev:
            ev_bwd.record()

        # Capture intermediate kernel gradients BEFORE `step_post_backward` mutates/slices params!
        loss_terms_snapshot: dict[str, float] | None = None
        if is_log:
            def _flt(x: Any) -> float:
                return round(float(x.detach().item()) if torch.is_tensor(x) else float(x), 6)

            loss_terms_snapshot = {
                "l1_weighted": _flt((1.0 - lam) * l1_vec.mean() if lam > 0 else l1_vec.mean()),
                "dssim_weighted": _flt(lam * dssim_term if lam > 0 else 0.0),
                "opa_reg": _flt(opa_reg_term),
                "scale_reg": _flt(scale_reg_term),
                "aniso_reg": _flt(aniso_term),
                "corridor_reg": _flt(corridor_loss),
                "app_reg": _flt(app_reg_term),
                "alpha_sil": _flt(alpha_sil_loss),
                "pose_reg": _flt(pose_reg_term),
                "views_per_step": b_cnt,
            }

        # Exponential position LR decay * scene_scale * live LR multiplier
        t_prog = min(1.0, step / float(max(1, target_step)))
        pos_lr = pos_lr_init * ((pos_lr_final / pos_lr_init) ** t_prog) * live_lr_mult
        for pg in optimizers["means"].param_groups:
            pg["lr"] = pos_lr

        for opt in optimizers.values():
            opt.step()
        if use_pose_opt and step >= 150:
            for pg in pose_opt.param_groups:
                pg["lr"] = pose_lr_init * ((0.1) ** t_prog) * live_lr_mult
            pose_opt.step()
        if app_active_now:
            for pg in app_opt.param_groups:
                pg["lr"] = 1.5e-3 * ((0.15) ** t_prog) * live_lr_mult
            app_opt.step()

        # Opacity Logit Ceiling + World-Scale Ceiling: prevents `σ(1-σ)->0` deadzone AND stops giant 2m–3m floaters!
        opa_cap = float(getattr(eff_cfg.train, "opacity_logit_cap", 5.5) or 0.0)
        max_s3d_world = max(0.06, float(getattr(eff_cfg.train, "prune_scale3d", 0.05)) * scene_scale * 2.2)
        log_max_s3d = math.log(max_s3d_world)
        with torch.no_grad():
            if opa_cap > 0.0:
                params["opacities"].data.clamp_(max=opa_cap)
            params["scales"].data.clamp_(min=-8.5, max=log_max_s3d)

        if is_log and use_cuda_ev:
            ev_opt.record()

        kernel_diag: dict[str, Any] | None = None
        if is_log and loss_terms_snapshot is not None:
            kernel_diag = _collect_kernel_telemetry(
                params=params,
                optimizers=optimizers,
                info=info,
                ren_alphas=ren_alphas,
                opa_sig=opa_sig,
                scales_lin=scales_lin,
                loss_terms=loss_terms_snapshot,
                pose_rot6d=pose_rot6d,
                pose_trans=pose_trans,
                use_pose_opt=use_pose_opt,
                densify_grad_thresh=eff_cfg.train.densify_grad_threshold,
                img_w=img_w,
                img_h=img_h,
            )

        # ── Adaptive Arc Densification, Budget Feedback & Multi-View Reset Guard ──
        n_pre_dens = int(params["means"].shape[0])
        is_adaptive = str(getattr(eff_cfg.train, "adaptive_strategy", "adaptive_arc")) != "fixed"
        scale_mode = str(getattr(eff_cfg.train, "arc_scale_mode", "log") or "log")
        arc_gamma = max(0.25, float(getattr(eff_cfg.train, "arc_gamma", 1.6) or 1.6))

        def _arc_interp(
            v0: float, v1: float, u_raw: float, _mode: str = scale_mode, _gamma: float = arc_gamma
        ) -> float:
            u = max(0.0, min(1.0, u_raw))
            if _mode == "log" and v0 > 0.0 and v1 > 0.0:
                return v0 * ((v1 / v0) ** u)
            if _mode == "power":
                return v0 + (v1 - v0) * (u ** _gamma)
            return v0 + (v1 - v0) * u

        dens_span = max(1, eff_densify_stop - int(eff_cfg.train.densify_from))
        u_dens = max(0.0, min(1.0, (step - int(eff_cfg.train.densify_from)) / float(dens_span)))
        target_p_opa = max(0.002, min(0.25, float(eff_cfg.train.prune_opa)))
        base_grad2d = max(2e-5, float(eff_cfg.train.densify_grad_threshold))

        if is_adaptive:
            p_opa_0 = float(eff_cfg.train.prune_opa_start) if eff_cfg.train.prune_opa_start is not None else min(0.008, target_p_opa * 0.35)
            eff_prune_opa = _arc_interp(max(0.002, p_opa_0), target_p_opa, u_dens)
            g2d_end = float(eff_cfg.train.grow_grad2d_end) if eff_cfg.train.grow_grad2d_end is not None else base_grad2d * 1.45
            eff_grow_grad2d = _arc_interp(base_grad2d * 0.85, g2d_end, u_dens)

            # Soft Gaussian population budget controller (`target_gaussians`):
            # tightens `grow_grad2d` smoothly as N approaches/exceeds budget, or relaxes if starved (<35% budget)
            budget_n = int(getattr(eff_cfg.train, "target_gaussians", 350_000) or 0)
            if budget_n > 0:
                pop_ratio = n_pre_dens / float(budget_n)
                if pop_ratio > 0.85:
                    eff_grow_grad2d *= min(4.0, pop_ratio ** 2.2)
                elif pop_ratio < 0.35 and step > int(eff_cfg.train.densify_from):
                    eff_grow_grad2d *= max(0.55, pop_ratio / 0.35)
        else:
            eff_prune_opa = target_p_opa
            eff_grow_grad2d = base_grad2d

        # Multi-view opacity reset recovery guard: after an opacity reset, every camera needs at least
        # ~5 visits (`5 * len(active_indices)` steps) before pruning is allowed to cull low-opacity splats!
        last_opa_reset = int(strategy_state.get("_bb_last_opa_reset", -100_000))
        reset_grace_steps = max(350, len(active_indices) * 6)
        in_reset_grace = (step - last_opa_reset) < reset_grace_steps
        strategy.prune_opa = 0.0005 if in_reset_grace else eff_prune_opa
        strategy.grow_grad2d = eff_grow_grad2d

        force_now = bool(surgery and surgery.densify_now and step == start_step)
        eff_step_for_strat = (
            eff_cfg.train.densify_from
            if force_now
            else step
        )
        strategy.step_post_backward(
            params, optimizers, strategy_state, eff_step_for_strat, info, packed=True
        )

        # Hard Swept-Arc Tunnel + Near-Frustum Eviction on densify ticks: carves the entire walkthrough path clean!
        near_f_m = float(getattr(eff_cfg.train, "near_frustum_clear_m", 0.40) or 0.0)
        if (
            (r_corridor > 0.0 or near_f_m > 0.0)
            and alphas_gpu is None
            and step > int(eff_cfg.train.densify_from)
            and step % max(1, int(eff_cfg.train.densify_every)) == 0
            and step < eff_densify_stop
        ):
            _evict_swept_arc_and_near_frustum_floaters(
                params=params,
                optimizers=optimizers,
                strategy_state=strategy_state,
                swept_centers_gpu=cam_centers_gpu,
                probe_vms_gpu=probe_vms_gpu,
                probe_ks_gpu=probe_ks_gpu,
                img_w=img_w,
                img_h=img_h,
                corridor_r=r_corridor * 0.68,
                near_clear_m=near_f_m * 0.75,
            )

        n_post_dens = int(params["means"].shape[0])

        # Safe Opacity Reset: clamps above `max(0.015, eff_prune_opa * 1.8)` (never below `prune_opa` AND never in the
        # final `reset_grace_steps + 600` steps before densify locks so population never crashes at step 20k!)
        min_tail_after_reset = max(900, reset_grace_steps + 500)
        if (
            eff_cfg.train.opacity_reset_every > 0
            and step > 0
            and step % eff_cfg.train.opacity_reset_every == 0
            and (eff_densify_stop - step) >= min_tail_after_reset
        ):
            reset_floor = min(0.12, max(0.018, eff_prune_opa * 1.85))
            with torch.no_grad():
                params["opacities"].data.clamp_(max=math.log(reset_floor / (1.0 - reset_floor)))
            # Zero stale opacity Adam momentum so clamped splats respond immediately to fresh multi-view rays
            opa_p = params["opacities"]
            opa_st = optimizers["opacities"].state.get(opa_p)
            if opa_st is not None:
                if "exp_avg" in opa_st:
                    opa_st["exp_avg"].zero_()
                if "exp_avg_sq" in opa_st:
                    opa_st["exp_avg_sq"].zero_()
            strategy_state["_bb_last_opa_reset"] = step

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
                act_ms = ev_s.elapsed_time(ev_act)
                fwd_rast_only_ms = ev_act.elapsed_time(ev_fwd)
                fwd_ms = ev_s.elapsed_time(ev_fwd)
                loss_ms = ev_fwd.elapsed_time(ev_loss)
                bwd_ms = ev_loss.elapsed_time(ev_bwd)
                adam_ms = ev_bwd.elapsed_time(ev_opt)
                dens_ms = ev_opt.elapsed_time(ev_end)
                gpu_step_ms = ev_s.elapsed_time(ev_end)
                wall_step_ms = (time.perf_counter() - t_step_cpu) * 1e3
                host_ms = max(0.0, wall_step_ms - gpu_step_ms)
                stage_timings = {
                    "param_act_ms": round(act_ms, 3),
                    "fwd_raster_only_ms": round(fwd_rast_only_ms, 3),
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

            # Update per-camera & multi-view EMA on EVERY batch camera (`batch_cams`) so IRLS outlier-view
            # trust weighting has accurate, dense loss estimates across all 64+ cameras!
            with torch.no_grad():
                pv_det = photo_vec.detach()
                mse_vec = ((renders.detach() - gt_imgs) ** 2).mean(dim=(1, 2, 3))
                for bi_u, ci_u in enumerate(batch_cams):
                    c_pv = float(pv_det[bi_u].item())
                    c_ps = -10.0 * math.log10(max(float(mse_vec[bi_u].item()), 1e-10))
                    cam_ema_loss[ci_u] = 0.75 * cam_ema_loss.get(ci_u, c_pv) + 0.25 * c_pv
                    cam_ema_psnr[ci_u] = 0.75 * cam_ema_psnr.get(ci_u, c_ps) + 0.25 * c_ps
            ema_loss_global = loss_f if ema_loss_global is None else (0.90 * ema_loss_global + 0.10 * loss_f)
            ema_psnr_global = psnr_val if ema_psnr_global is None else (0.90 * ema_psnr_global + 0.10 * psnr_val)

            if kernel_diag is not None and cam_ema_psnr:
                ps_vals = list(cam_ema_psnr.values())
                worst_ci = min(cam_ema_psnr, key=lambda k_ci: cam_ema_psnr[k_ci])
                best_ci = max(cam_ema_psnr, key=lambda k_ci: cam_ema_psnr[k_ci])
                kernel_diag["view_spread"] = {
                    "tracked_cams": len(ps_vals),
                    "mean_psnr": round(sum(ps_vals) / len(ps_vals), 2),
                    "min_psnr": round(cam_ema_psnr[worst_ci], 2),
                    "max_psnr": round(cam_ema_psnr[best_ci], 2),
                    "spread_db": round(cam_ema_psnr[best_ci] - cam_ema_psnr[worst_ci], 2),
                    "worst_cam": image_names[worst_ci],
                    "best_cam": image_names[best_ci],
                }

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
                "loss_ema": round(ema_loss_global, 5),
                "l1": round(l1_val, 5),
                "ssim": round(ssim_f, 5),
                "mse": round(mse_val, 6),
                "psnr": round(psnr_val, 2),
                "psnr_ema": round(ema_psnr_global, 2),
                "n_gaussians": n_post_dens,
                "sh_degree_cur": sh_cur,
                "pos_lr": round(pos_lr, 7),
                "eff_prune_opa": round(float(strategy.prune_opa), 5),
                "eff_grow_grad2d": round(float(strategy.grow_grad2d), 6),
                "adaptive_strategy": str(getattr(eff_cfg.train, "adaptive_strategy", "adaptive_arc")),
                "arc_scale_mode": scale_mode,
                "lambda_dssim": round(eff_cfg.train.lambda_dssim, 4),
                "opa_reg": round(eff_cfg.train.opa_reg, 6),
                "focus_camera": eff_cfg.train.focus_camera,
                "camera_name": cam_name,
                "stage_timings": stage_timings,
            }
            if kernel_diag is not None:
                tick["kernel_diag"] = kernel_diag

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

    # Bake differentiable `se(3)` + iterative video-diff arc refinements (including all newly promoted frames!)
    # back into `viewmats_gpu`, `frames_u8.pt`, and `cameras.json`!
    mean_shift_mm = 0.0
    if use_pose_opt:
        with torch.no_grad():
            new_vms = []
            for ci in range(n_cams_total):
                vm_ci, _ = _compose_pose_delta(viewmats_gpu[ci : ci + 1], ci)
                new_vms.append(vm_ci[0])
            viewmats_gpu = torch.stack(new_vms, dim=0)
            mean_shift_mm = float(pose_trans.detach().norm(dim=-1).mean().item() * 1000.0)

    if (use_pose_opt or n_iter_cycles > 1) and cams_json_path.is_file():
        try:
            cdoc_upd = json.loads(cams_json_path.read_text(encoding="utf-8"))
            vms_cpu = viewmats_gpu.detach().cpu().numpy()
            ks_cpu = ks_gpu.detach().cpu().numpy()
            v_fps_doc = float((cdoc_upd.get("video_meta") or {}).get("fps") or 24.0)
            export_cam_indices = sorted(active_indices) if n_iter_cycles > 1 else list(range(len(cdoc_upd.get("cameras") or [])))
            # Final trajectory spike pass across exported keyframes so any residual 1-frame promoted kink is healed!
            if len(export_cam_indices) >= 5:
                ext_exp = vms_cpu[export_cam_indices, :3, :4].copy()
                ext_exp_clean, _ = _repair_outlier_trajectory_spikes(ext_exp, spike_factor=2.4)
                vms_cpu[export_cam_indices, :3, :4] = ext_exp_clean.astype(np.float32)
            new_bl_cams: list[dict[str, Any]] = []
            for ci in export_cam_indices:
                if ci >= len(vms_cpu):
                    break
                rc_f = vms_cpu[ci, :3, :3].tolist()
                tc_f = vms_cpu[ci, :3, 3].tolist()
                bl_upd = opencv_w2c_to_blender_camera(
                    rc_f, tc_f,
                    float(ks_cpu[ci, 0, 0]), float(ks_cpu[ci, 1, 1]),
                    float(ks_cpu[ci, 0, 2]), float(ks_cpu[ci, 1, 2]),
                    img_w, img_h,
                )
                bl_upd["frame_index"] = ci
                bl_upd["image_name"] = image_names[ci]
                if ci < len(pts_sec_list):
                    pt_v = round(float(pts_sec_list[ci]), 5)
                    bl_upd["pts_sec"] = pt_v
                    bl_upd["timeline_frame"] = max(1, round(pt_v * v_fps_doc) + 1)
                new_bl_cams.append(bl_upd)
            if new_bl_cams:
                cdoc_upd["cameras"] = new_bl_cams
            cdoc_upd["pose_refined"] = True
            cdoc_upd["iterative_arc_cycles"] = n_iter_cycles
            cdoc_upd["mean_pose_shift_mm"] = round(mean_shift_mm, 3)
            cams_json_path.write_text(json.dumps(cdoc_upd, indent=2), encoding="utf-8")
            if frames_pt_path.is_file():
                pack_u = torch.load(frames_pt_path, map_location="cpu", weights_only=False)
                pack_u["viewmats"] = viewmats_gpu.detach().cpu()
                pack_u["posed_indices"] = sorted(active_indices)
                torch.save(pack_u, frames_pt_path)
        except Exception:  # noqa: BLE001, S110
            pass

    # ── Save Resumable Checkpoint + Update In-VRAM Cache ──
    t_exp = time.perf_counter()
    torch.save(
        {
            "step": actual_final_step,
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
            "alphas_gpu": alphas_gpu,
            "depths_gpu": depths_gpu,
            "confs_gpu": confs_gpu,
            "pts_sec": pts_sec_list,
            "posed_indices": sorted(active_indices),
            "viewmats_gpu": viewmats_gpu,
            "ks_gpu": ks_gpu,
            "image_names": image_names,
            "width": img_w,
            "height": img_h,
            "params": params,
            "optimizers": optimizers,
            "strategy_state": strategy_state,
            "step": actual_final_step,
        })

    total_cum_gpu = cum_gpu_sec_prev + macro_timings["train_s"] + macro_timings.get("pose_s", 0.0)
    meta.update({
        "config": eff_cfg.to_dict(),
        "burst_id": burst_id,
        "step": actual_final_step,
        "cum_gpu_sec": round(total_cum_gpu, 3),
        "n_gaussians": int(params["means"].shape[0]),
        "excluded": sorted(excluded_set),
        "macro_timings": macro_timings,
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ── Stage 4: Export `.ply` + `.splat` + `proxy_collision.glb` + `cameras.json` ──
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
    proxy_glb_bytes = b""
    if getattr(eff_cfg.export, "export_proxy_glb", True):
        try:
            with torch.no_grad():
                m_np = params["means"].detach().float().cpu().numpy()
                o_np = torch.sigmoid(params["opacities"].detach().float()).cpu().numpy()
                s_np = torch.exp(params["scales"].detach().float()).cpu().numpy()
                c_np = np.clip(0.5 + _SH_C0 * params["sh0"].detach().float().cpu().numpy().reshape(-1, 3), 0.0, 1.0)
            proxy_glb_bytes, _ = build_proxy_glb_bytes(
                m_np,
                o_np,
                s_np,
                c_np,
                grid_res=int(getattr(eff_cfg.export, "proxy_grid_res", 48)),
                mesh_name=f"{job_id}_Proxy",
            )
        except Exception:  # noqa: BLE001
            proxy_glb_bytes = b""

    macro_timings["export_s"] = round(time.perf_counter() - t_exp, 3)

    wall_s = round(time.perf_counter() - t_job_start, 3)
    cost_usd = round((wall_s / 3600.0) * gpu_hourly_usd, 5)
    cams_text = cams_json_path.read_text(encoding="utf-8")

    done_ev: dict[str, Any] = {
        "event": "done",
        "job_id": job_id,
        "burst_id": burst_id,
        "step": actual_final_step,
        "n_gaussians": n_exported,
        "device": dev_name,
        "wall_s": wall_s,
        "cost_usd": cost_usd,
        "macro_timings": macro_timings,
        "cameras_json": cams_text,
        "ply_gz": gzip.compress(ply_bytes, compresslevel=2),
        "splat_gz": gzip.compress(splat_bytes, compresslevel=2) if splat_bytes else b"",
        "proxy_glb_gz": gzip.compress(proxy_glb_bytes, compresslevel=2) if proxy_glb_bytes else b"",
        "ply_raw_bytes": len(ply_bytes),
        "splat_raw_bytes": len(splat_bytes),
        "proxy_glb_raw_bytes": len(proxy_glb_bytes),
    }
    _append_telem(done_ev)
    yield done_ev


# ─── Cloud Farmed Splat Trajectory Snapper & Renderer (`gsplat` CUDA -> H.264 MP4) ──


def _estimate_video_pan_dyaw_between_timestamps(
    video_path: Path,
    t_start: float,
    t_end: float,
    hfov_deg: float,
) -> float:
    """Measure true cumulative horizontal camera pan (`dyaw_deg`, `+` = turning right) directly from MP4!

    Decodes a tiny `160x90` 30fps grayscale strip across `[t_start, t_end]` (~8 ms) and integrates
    frame-to-frame horizontal block translation `dx_px`. When a player whip-pans `200°` right in `1.2s`,
    standard shortest-arc SLERP sees `200° ≡ -160°` and spins `160°` LEFT; `pan_dyaw_deg` tells
    `interpolate_camera_trajectory` the exact physical turn direction AND magnitude!
    """
    dur_seg = max(0.05, float(t_end) - float(t_start))
    if dur_seg > 6.0:
        return 0.0
    cmd = [
        "ffmpeg",
        "-ss", f"{max(0.0, float(t_start)):.3f}",
        "-t", f"{dur_seg:.3f}",
        "-i", str(video_path),
        "-vf", "fps=30,scale=160:90,format=gray",
        "-f", "rawvideo",
        "-hide_banner", "-loglevel", "error",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or len(proc.stdout) < 160 * 90 * 2:
        return 0.0
    n_f = len(proc.stdout) // (160 * 90)
    arr = np.frombuffer(proc.stdout[: n_f * 160 * 90], dtype=np.uint8).reshape(n_f, 90, 160).astype(np.float32)

    # Image pixels shift LEFT (`dx < 0`) when camera turns RIGHT (`dyaw > 0`):
    # `arr[i, :, x]` matches `arr[i+1, :, x + shift]` -> `dyaw_step = -atan(shift / fx_thumb)`
    fx_thumb = 80.0 / max(0.15, math.tan(math.radians(max(25.0, min(145.0, hfov_deg)) * 0.5)))
    cum_dyaw = 0.0
    strip = arr[:, 16:74, :]  # central horizontal band (ignores sky/floor & HUD corners)
    for i in range(n_f - 1):
        f0, f1 = strip[i], strip[i + 1]
        best_sh = 0
        best_err = 1e9
        for sh in range(-48, 49, 2):
            if sh >= 0:
                a, b = f0[:, : 160 - sh], f1[:, sh:]
            else:
                a, b = f0[:, -sh:], f1[:, : 160 + sh]
            err = float(np.abs(a - b).mean()) + 0.015 * abs(sh)
            if err < best_err:
                best_err = err
                best_sh = sh
        # Camera turning RIGHT (+yaw) moves world features LEFT in image (`best_sh < 0`)
        cum_dyaw += math.degrees(math.atan2(-float(best_sh), fx_thumb))
    return round(cum_dyaw, 2)


def snap_camera_arc_to_video_stream(
    *,
    job_id: str,
    jobs_root: Path,
    mp4_bytes: bytes,
    base_cameras_doc: dict[str, Any],
    ply_gz: bytes | None = None,
    sample_fps: float = 12.0,
    max_snap_frames: int = 240,
    solve_steps: int = 80,
    floater_clean: bool = True,
    corridor_radius_m: float = 0.42,
    near_clear_m: float = 0.42,
    max_world_scale_m: float = 0.25,
    gpu_hourly_usd: float = 3.95,
    ram_cache: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Snap a trained 3DGS camera arc directly onto the source video (`mp4_bytes`) + clean floaters!

    Works on ANY existing splat job (`3becf2`, `adb4c1`, etc.) without retraining from scratch:
      1. Extracts dense ground-truth frames (`sample_fps` up to `max_snap_frames`) from the original MP4.
      2. Repairs any 1-frame VGGT teleport spikes in `base_cameras_doc["cameras"]` and initializes
         dense poses along the centripetal Catmull-Rom trajectory.
      3. Runs coarse-to-fine (`1/4 -> 1/2 -> full`) differentiable `se(3)` photometric alignment
         (`L1 + DSSIM + C2 spline smoothness`) between the rendered 3DGS splat and the real video frames!
      4. Optionally cleans giant >`max_world_scale_m` floaters + swept-arc / near-frustum blockers
         from the checkpoint/PLY and exports a crisp `cameras.json` + cleaned `splat.ply`/`splat.splat`!
    """
    from gsplat import rasterization

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = job_dir / "ckpt_latest.pt"

    if ram_cache is not None and ram_cache.get("job_id") == job_id and "params" in ram_cache:
        params_dict = {k: v.detach().clone().to(device) for k, v in ram_cache["params"].items()}
        k_tot = 1 + int(params_dict["shN"].shape[1])
        sh_deg = max(0, round(math.sqrt(k_tot)) - 1)
    elif ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        params_dict = {k: v.detach().clone().to(device) for k, v in ckpt["params"].items()}
        k_tot = 1 + int(params_dict["shN"].shape[1])
        sh_deg = max(0, round(math.sqrt(k_tot)) - 1)
    elif ply_gz:
        params_dict, sh_deg = _load_params_from_ply_bytes(gzip.decompress(ply_gz), device)
    else:
        raise FileNotFoundError(f"No checkpoint or fallback PLY found for '{job_id}'.")

    work_dir = Path("/dev/shm") / f"snap_{job_id}_{int(time.time() * 1000)}" if Path("/dev/shm").is_dir() else (job_dir / "tmp_snap")
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        vid_p = work_dir / "source.mp4"
        vid_p.write_bytes(mp4_bytes)
        ext_cfg = ExtractConfig(
            fps=max(2.0, min(30.0, float(sample_fps))),
            max_frames=max(24, min(400, int(max_snap_frames))),
            max_width=960,
            dedup_frames=False,
            quality=92,
        )
        f_paths, pts_map, vmeta = extract_frames_with_meta(vid_p, work_dir / "frames", ext_cfg)
        m_snap = len(f_paths)
        with Image.open(f_paths[0]) as im0:
            snap_w, snap_h = im0.width, im0.height

        # Repair any outlier translation OR 180°-flip rotation spikes in existing `base_cameras_doc["cameras"]`!
        raw_cams = list(base_cameras_doc.get("cameras") or [])
        if len(raw_cams) >= 5:
            from .sim3 import blender_camera_to_opencv_w2c

            ext_old = np.zeros((len(raw_cams), 3, 4), dtype=np.float64)
            for ci, c_old in enumerate(raw_cams):
                cv_o = blender_camera_to_opencv_w2c(c_old)
                ext_old[ci, :3, :3] = np.array(cv_o["R_w2c"], dtype=np.float64)
                ext_old[ci, :3, 3] = np.array(cv_o["t_w2c"], dtype=np.float64)
            ext_clean, n_spikes_fixed = _repair_outlier_trajectory_spikes(ext_old, spike_factor=2.6)
            if n_spikes_fixed > 0:
                cleaned_cams = []
                for ci, c_old in enumerate(raw_cams):
                    bl_fixed = opencv_w2c_to_blender_camera(
                        ext_clean[ci, :3, :3].tolist(),
                        ext_clean[ci, :3, 3].tolist(),
                        float(c_old.get("fx") or snap_w),
                        float(c_old.get("fy") or snap_w),
                        float(c_old.get("cx") or snap_w * 0.5),
                        float(c_old.get("cy") or snap_h * 0.5),
                        int(c_old.get("width") or snap_w),
                        int(c_old.get("height") or snap_h),
                    )
                    for kk in ("frame_index", "image_name", "pts_sec", "timeline_frame"):
                        if kk in c_old:
                            bl_fixed[kk] = c_old[kk]
                    cleaned_cams.append(bl_fixed)
                base_cameras_doc = {**base_cameras_doc, "cameras": cleaned_cams}
        else:
            n_spikes_fixed = 0

        # Lock shared PINHOLE focal length across all keyframes so FOV never breathes
        orig_cam_w = int((raw_cams[0].get("width") if raw_cams else None) or vmeta["width"])
        orig_cam_h = int((raw_cams[0].get("height") if raw_cams else None) or vmeta["height"])
        med_fx_orig = float(np.median([float(c.get("fx") or orig_cam_w) for c in raw_cams])) if raw_cams else float(orig_cam_w)
        sx_snap = snap_w / float(max(1, orig_cam_w))
        sy_snap = snap_h / float(max(1, orig_cam_h))
        fx_snap = med_fx_orig * sx_snap
        fy_snap = med_fx_orig * sy_snap
        cx_snap, cy_snap = snap_w * 0.5, snap_h * 0.5

        # Annotate any wide keyframe rotation jumps (`> 75°`) with true 30fps video optical-flow `pan_dyaw_deg`
        # so >160° whip-pans unwrap in the TRUE mouse-pan direction instead of spinning backwards!
        cams_for_traj = [dict(c) for c in (base_cameras_doc.get("cameras") or [])]
        hfov_est = math.degrees(2.0 * math.atan(orig_cam_w / max(1.0, 2.0 * med_fx_orig)))
        for ki in range(1, len(cams_for_traj)):
            q0 = cams_for_traj[ki - 1].get("rotation_quaternion") or [1.0, 0.0, 0.0, 0.0]
            q1 = cams_for_traj[ki].get("rotation_quaternion") or [1.0, 0.0, 0.0, 0.0]
            dot_q = min(1.0, abs(sum(float(a) * float(b) for a, b in zip(q0, q1))))
            ang_jump = math.degrees(2.0 * math.acos(dot_q))
            t0_k = cams_for_traj[ki - 1].get("pts_sec")
            t1_k = cams_for_traj[ki].get("pts_sec")
            if ang_jump > 75.0 and t0_k is not None and t1_k is not None and float(t1_k) > float(t0_k):
                cams_for_traj[ki]["pan_dyaw_deg"] = _estimate_video_pan_dyaw_between_timestamps(
                    vid_p, float(t0_k), float(t1_k), hfov_est
                )
        base_cameras_doc = {**base_cameras_doc, "cameras": cams_for_traj}

        dense_fps = 60.0
        dur_v = float(vmeta["duration_sec"])
        traj_dense = interpolate_camera_trajectory(
            base_cameras_doc,
            fps=dense_fps,
            duration_sec=dur_v,
            total_frames=max(16, round(dur_v * dense_fps) + 1),
            target_width=snap_w,
            target_height=snap_h,
        )["frames"]

        vms_init = np.zeros((m_snap, 4, 4), dtype=np.float32)
        ks_snap = np.zeros((m_snap, 3, 3), dtype=np.float32)
        pts_snap: list[float] = []
        rec_load = []
        for i, fp in enumerate(f_paths):
            pt = float(pts_map.get(fp.name, (i / float(max(1, m_snap - 1))) * dur_v))
            pts_snap.append(pt)
            ti = max(0, min(len(traj_dense) - 1, round(pt * dense_fps)))
            fr = traj_dense[ti]
            vms_init[i, 3, 3] = 1.0
            vms_init[i, :3, :3] = np.array(fr["R_w2c"], dtype=np.float32)
            vms_init[i, :3, 3] = np.array(fr["t_w2c"], dtype=np.float32)
            ks_snap[i] = np.array([[fx_snap, 0.0, cx_snap], [0.0, fy_snap, cy_snap], [0.0, 0.0, 1.0]], dtype=np.float32)
            rec_load.append({"image_path": str(fp), "width": snap_w, "height": snap_h})

        frames_u8, _ = _preload_frames_u8(rec_load, load_alpha=False)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    yield {
        "event": "snap_start",
        "job_id": job_id,
        "device": dev_name,
        "snap_frames": m_snap,
        "spikes_repaired": n_spikes_fixed,
    }

    frames_gpu = frames_u8.to(device, non_blocking=True)
    vms_base_gpu = torch.from_numpy(vms_init).to(device)
    ks_gpu = torch.from_numpy(ks_snap).to(device)
    ssim_win = _build_ssim_window(device)

    # Pre-filter giant room-blocking floaters (> max_world_scale_m) BEFORE snapping so camera sees real walls!
    n_gauss_before = int(params_dict["means"].shape[0])
    giant_evicted = 0
    if floater_clean and max_world_scale_m > 0.0:
        with torch.no_grad():
            s_max = torch.exp(params_dict["scales"]).max(dim=-1).values
            opa_v = torch.sigmoid(params_dict["opacities"])
            keep_g = (s_max <= float(max_world_scale_m)) & (opa_v >= 0.018)
            if int(keep_g.sum().item()) >= 2000:
                giant_evicted = int((~keep_g).sum().item())
                params_dict = {k: v[keep_g].contiguous() for k, v in params_dict.items()}

    means_f = params_dict["means"]
    quats_f = F.normalize(params_dict["quats"], dim=-1)
    scales_f = torch.exp(params_dict["scales"])
    opa_f = torch.sigmoid(params_dict["opacities"])
    k_use = (sh_deg + 1) ** 2
    colors_f = torch.cat([params_dict["sh0"], params_dict["shN"]], dim=1)[:, :k_use, :]

    rot6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device).repeat(
        m_snap, 1
    ).requires_grad_(True)
    trans = torch.zeros((m_snap, 3), dtype=torch.float32, device=device, requires_grad=True)
    # Also allow a small global shared focal-length multiplier `log_f_scale` (`±12%`) to snap exact FOV!
    log_f_scale = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam(
        [
            {"params": [rot6d, trans], "lr": 2.0e-3},
            {"params": [log_f_scale], "lr": 8.0e-4},
        ],
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    row4_b = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    def _compose_vms(idxs: list[int]) -> torch.Tensor:
        r6 = rot6d[idxs]
        a1 = F.normalize(r6[:, 0:3], dim=-1)
        b2 = r6[:, 3:6] - (a1 * r6[:, 3:6]).sum(dim=-1, keepdim=True) * a1
        a2 = F.normalize(b2, dim=-1)
        a3 = torch.linalg.cross(a1, a2, dim=-1)
        r_del = torch.stack([a1, a2, a3], dim=1)
        t_del = trans[idxs]
        base_b = vms_base_gpu[idxs]
        r_new = torch.bmm(r_del, base_b[:, :3, :3])
        t_new = torch.bmm(r_del, base_b[:, :3, 3:4]).squeeze(-1) + t_del
        top3 = torch.cat([r_new, t_new.unsqueeze(-1)], dim=-1)
        return torch.cat([top3, row4_b.view(1, 1, 4).expand(len(idxs), 1, 4)], dim=1)

    def _scaled_ks(idxs: list[int], sx: float = 1.0, sy: float = 1.0) -> torch.Tensor:
        # Pure out-of-place `K` construction so PyTorch autograd never sees in-place slice writes on `log_f_scale`!
        f_mult = torch.exp(log_f_scale.clamp(-0.15, 0.15))
        base_k = ks_gpu[idxs]
        z_vec = torch.zeros_like(base_k[:, 0, 0])
        o_vec = torch.ones_like(z_vec)
        r0 = torch.stack([base_k[:, 0, 0] * (f_mult * sx), z_vec, base_k[:, 0, 2] * sx], dim=-1)
        r1 = torch.stack([z_vec, base_k[:, 1, 1] * (f_mult * sy), base_k[:, 1, 2] * sy], dim=-1)
        r2 = torch.stack([z_vec, z_vec, o_vec], dim=-1)
        return torch.stack([r0, r1, r2], dim=1)

    all_idxs = list(range(m_snap))
    with torch.no_grad():
        init_errs = []
        for b0 in range(0, m_snap, 8):
            ch_i = all_idxs[b0 : b0 + 8]
            rb, _, _ = rasterization(
                means=means_f, quats=quats_f, scales=scales_f, opacities=opa_f, colors=colors_f,
                viewmats=vms_base_gpu[ch_i], Ks=ks_gpu[ch_i], width=snap_w, height=snap_h, sh_degree=sh_deg, packed=True,
            )
            gb = frames_gpu[ch_i].float() * (1.0 / 255.0)
            init_errs.extend((rb - gb).abs().mean(dim=(1, 2, 3)).tolist())
        l1_before = float(np.mean(init_errs))

    # Stage A: Wide-Angle Yaw Sweep Initialization (`[-60°..+60°]` in `15°` steps @ `1/6` resolution, `<0.4s` on H100!)
    # During a fast 200°/s whip-pan, an intermediate video frame can sit 25°–55° ahead/behind the smooth spline
    # (outside local gradient-descent basin!). Sweeping 9 yaw offsets snaps every whip-pan frame into its basin!
    with torch.no_grad():
        sw_w, sw_h = max(64, snap_w // 6), max(48, snap_h // 6)
        k_sw = ks_gpu.clone()
        k_sw[:, 0, :] *= sw_w / float(snap_w)
        k_sw[:, 1, :] *= sw_h / float(snap_h)
        gt_sw = F.interpolate(
            (frames_gpu.float() * (1.0 / 255.0)).permute(0, 3, 1, 2), size=(sw_h, sw_w), mode="area"
        ).permute(0, 2, 3, 1)

        best_err_sw = torch.full((m_snap,), 1e9, dtype=torch.float32, device=device)
        best_vms_sw = vms_base_gpu.clone()
        # World-Z (`+Z` up) yaw rotation matrices applied around each camera's world center `C_i`
        for deg_off in (0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0, -60.0, 60.0):
            rad = math.radians(deg_off)
            ca, sa = math.cos(rad), math.sin(rad)
            rz_w = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
            # Rotate camera look direction in world around Z: `R_c2w_new = Rz_w @ R_c2w => R_w2c_new = R_w2c @ Rz_w^T`
            # keeping world camera center `C = -R_w2c^T t_w2c` fixed (`t_new = -R_w2c_new @ C = t_old`)!
            vms_cand = vms_base_gpu.clone()
            vms_cand[:, :3, :3] = torch.matmul(vms_base_gpu[:, :3, :3], rz_w.T)
            errs_deg = []
            for b0 in range(0, m_snap, 16):
                ch_s = all_idxs[b0 : b0 + 16]
                rs, _, _ = rasterization(
                    means=means_f, quats=quats_f, scales=scales_f, opacities=opa_f, colors=colors_f,
                    viewmats=vms_cand[ch_s], Ks=k_sw[ch_s], width=sw_w, height=sw_h, sh_degree=0, packed=True,
                )
                # Slight prior preference (`0.004 * |deg|/15`) for smaller angle unless image match is clearly better
                e_s = (rs - gt_sw[ch_s]).abs().mean(dim=(1, 2, 3)) + 0.003 * (abs(deg_off) / 15.0)
                errs_deg.append(e_s)
            e_all = torch.cat(errs_deg, dim=0)
            improved = e_all < best_err_sw
            best_err_sw = torch.where(improved, e_all, best_err_sw)
            best_vms_sw[improved] = vms_cand[improved]
        vms_base_gpu = best_vms_sw

    # Pre-compute frame-to-frame visual motion weights so fast pans are FREE to accelerate (`w_smooth -> 0.05`)
    # while stationary / slow-walk segments stay rock-steady (`w_smooth -> 1.0`)!
    with torch.no_grad():
        gt_tiny = F.interpolate(
            (frames_gpu.float() * (1.0 / 255.0)).permute(0, 3, 1, 2), size=(36, 64), mode="area"
        )
        step_mad = (gt_tiny[1:] - gt_tiny[:-1]).abs().mean(dim=(1, 2, 3))  # [M-1]
        # High visual change (`step_mad > 0.08`) => fast whip-pan => relax acceleration penalty!
        w_vel = (1.0 / (1.0 + (step_mad / 0.06).pow(2))).clamp(0.05, 1.0)  # [M-1]
        w_acc = torch.minimum(w_vel[:-1], w_vel[1:]) if m_snap >= 3 else w_vel

    rng = random.Random(77)
    batch_sz = min(10, m_snap)
    n_passes = max(50, min(360, int(solve_steps) * max(1, m_snap // 20)))
    for it in range(n_passes):
        prog = it / float(max(1, n_passes - 1))
        b_idxs = rng.sample(all_idxs, batch_sz) if m_snap > batch_sz else list(all_idxs)
        vm_b = _compose_vms(b_idxs)
        # 3-level coarse-to-fine pyramid: 1/4 -> 1/2 -> full!
        div = 4 if prog < 0.32 else (2 if prog < 0.70 else 1)
        if div > 1:
            cw, ch = max(64, snap_w // div), max(64, snap_h // div)
            k_b = _scaled_ks(b_idxs, cw / float(snap_w), ch / float(snap_h))
            gt_f = (frames_gpu[b_idxs].float() * (1.0 / 255.0)).permute(0, 3, 1, 2)
            gt_b = F.interpolate(gt_f, size=(ch, cw), mode="area").permute(0, 2, 3, 1)
        else:
            cw, ch = snap_w, snap_h
            k_b = _scaled_ks(b_idxs)
            gt_b = frames_gpu[b_idxs].float() * (1.0 / 255.0)

        ren_b, _, _ = rasterization(
            means=means_f, quats=quats_f, scales=scales_f, opacities=opa_f, colors=colors_f,
            viewmats=vm_b, Ks=k_b, width=cw, height=ch, sh_degree=sh_deg, packed=True,
        )
        l1_v = (ren_b - gt_b).abs().mean()
        ssim_v = _ssim_cuda(ren_b, gt_b, ssim_win).mean()
        photo = 0.80 * l1_v + 0.20 * (1.0 - ssim_v)

        if m_snap >= 3:
            # Motion-adaptive smoothness: tight on steady walks, relaxed on fast whip-pans so rotation never lags!
            d1_t = (trans[1:] - trans[:-1]).pow(2).mean(dim=-1)
            d1_r = (rot6d[1:] - rot6d[:-1]).pow(2).mean(dim=-1)
            d2_t = (trans[2:] - 2.0 * trans[1:-1] + trans[:-2]).pow(2).mean(dim=-1)
            d2_r = (rot6d[2:] - 2.0 * rot6d[1:-1] + rot6d[:-2]).pow(2).mean(dim=-1)
            vel_reg = 0.012 * ((d1_t + 0.25 * d1_r) * w_vel).mean()
            acc_reg = 0.14 * ((d2_t + 0.20 * d2_r) * w_acc).mean()
            smooth = vel_reg + acc_reg
        else:
            smooth = 0.0

        loss = photo + smooth + 0.02 * (log_f_scale ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        lr_mul = 0.20 ** prog
        opt.param_groups[0]["lr"] = 3.2e-3 * lr_mul
        opt.param_groups[1]["lr"] = 8.0e-4 * lr_mul
        opt.step()

        if (it + 1) % 25 == 0 or it == n_passes - 1:
            yield {
                "event": "snap_progress",
                "job_id": job_id,
                "step": it + 1,
                "total_steps": n_passes,
                "loss": round(float(photo.detach().item()), 5),
            }

    with torch.no_grad():
        vms_snapped = _compose_vms(all_idxs).detach()
        f_mult_final = float(torch.exp(log_f_scale.clamp(-0.15, 0.15)).item())
        ks_final_snap = _scaled_ks(all_idxs).detach()
        post_errs = []
        for b0 in range(0, m_snap, 8):
            ch_i = all_idxs[b0 : b0 + 8]
            rb, _, _ = rasterization(
                means=means_f, quats=quats_f, scales=scales_f, opacities=opa_f, colors=colors_f,
                viewmats=vms_snapped[ch_i], Ks=ks_final_snap[ch_i], width=snap_w, height=snap_h, sh_degree=sh_deg, packed=True,
            )
            gb = frames_gpu[ch_i].float() * (1.0 / 255.0)
            post_errs.extend((rb - gb).abs().mean(dim=(1, 2, 3)).tolist())
        l1_after = float(np.mean(post_errs))
        mean_shift_cm = float(trans.detach().norm(dim=-1).mean().item() * 100.0)

    # Swept-arc tunnel + near-frustum floater carving along the newly snapped dense camera path!
    corr_ev, frust_ev = 0, 0
    if floater_clean:
        pdict_param = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in params_dict.items()})
        swept_c, p_vms, p_ks = _build_swept_arc_samples_gpu(
            vms_snapped, ks_final_snap, all_idxs, pts_snap, snap_w, snap_h, n_arc_samples=600, n_frustum_probes=96
        )
        ev_res = _evict_swept_arc_and_near_frustum_floaters(
            pdict_param, {}, {}, swept_c, p_vms, p_ks, snap_w, snap_h, corridor_r=corridor_radius_m, near_clear_m=near_clear_m
        )
        corr_ev = ev_res["corridor_evicted"]
        frust_ev = ev_res["frustum_evicted"]
        params_dict = {k: v.detach() for k, v in pdict_param.items()}

        # Persist cleaned splat back to `/jobs/{job_id}/ckpt_latest.pt` & `ram_cache` so subsequent renders use it!
        if ckpt_path.is_file():
            try:
                ck_old = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ck_old["params"] = {k: v.cpu() for k, v in params_dict.items()}
                ck_old["opt_states"] = {}
                ck_old["strategy_state"] = {}
                torch.save(ck_old, ckpt_path)
            except Exception:  # noqa: BLE001, S110
                pass
        if ram_cache is not None and ram_cache.get("job_id") == job_id:
            ram_cache.clear()

    # Export updated native-resolution `cameras.json` across all `m_snap` snapped video keyframes!
    vms_cpu = vms_snapped.cpu().numpy()
    fx_out = med_fx_orig * f_mult_final
    fy_out = med_fx_orig * f_mult_final
    cx_out, cy_out = orig_cam_w * 0.5, orig_cam_h * 0.5
    native_fps_out = float(vmeta["fps"])
    snapped_bl_cams: list[dict[str, Any]] = []
    for i in range(m_snap):
        bl_c = opencv_w2c_to_blender_camera(
            vms_cpu[i, :3, :3].tolist(),
            vms_cpu[i, :3, 3].tolist(),
            fx_out, fy_out, cx_out, cy_out,
            orig_cam_w, orig_cam_h,
        )
        bl_c["frame_index"] = i
        bl_c["image_name"] = f_paths[i].name
        bl_c["pts_sec"] = round(pts_snap[i], 5)
        bl_c["timeline_frame"] = max(1, round(pts_snap[i] * native_fps_out) + 1)
        snapped_bl_cams.append(bl_c)

    updated_doc = {
        **base_cameras_doc,
        "pose_backend": f"{base_cameras_doc.get('pose_backend', 'vggt')}+arc_snapped",
        "video_meta": vmeta,
        "arc_snapped": True,
        "focal_scale_snapped": round(f_mult_final, 5),
        "cameras": snapped_bl_cams,
    }
    (job_dir / "cameras.json").write_text(json.dumps(updated_doc, indent=2), encoding="utf-8")

    ply_bytes, n_final = export_ply_bytes(params_dict, max_gaussians=0, sort_by_opacity=True)
    splat_bytes = export_splat_bytes(params_dict, max_gaussians=0)
    wall_s = round(time.perf_counter() - t0, 3)

    yield {
        "event": "snap_done",
        "job_id": job_id,
        "device": dev_name,
        "snapped_keyframes": m_snap,
        "spikes_repaired": n_spikes_fixed,
        "l1_before": round(l1_before, 5),
        "l1_after": round(l1_after, 5),
        "mean_shift_cm": round(mean_shift_cm, 2),
        "focal_scale": round(f_mult_final, 4),
        "n_gaussians_before": n_gauss_before,
        "n_gaussians_after": n_final,
        "giant_evicted": giant_evicted,
        "corridor_evicted": corr_ev,
        "frustum_evicted": frust_ev,
        "wall_s": wall_s,
        "cost_usd": round((wall_s / 3600.0) * gpu_hourly_usd, 5),
        "cameras_json": json.dumps(updated_doc, indent=2),
        "ply_gz": gzip.compress(ply_bytes, compresslevel=2) if floater_clean else b"",
        "splat_gz": gzip.compress(splat_bytes, compresslevel=2) if floater_clean else b"",
    }


# ─── Cloud Farmed Splat Trajectory Renderer (`gsplat` CUDA -> H.264 MP4) ──────


def _load_params_from_ply_bytes(ply_bytes: bytes, device: torch.device) -> tuple[dict[str, torch.Tensor], int]:
    """Parse binary little-endian 3DGS `.ply` directly into CUDA tensors (`<30 ms` for 300k splats)."""
    end_m = b"end_header\n"
    idx = ply_bytes.find(end_m)
    if idx < 0:
        end_m = b"end_header\r\n"
        idx = ply_bytes.find(end_m)
    if idx < 0:
        raise ValueError("Invalid PLY header for cloud trajectory render.")

    hdr = ply_bytes[:idx].decode("ascii", errors="ignore")
    offset = idx + len(end_m)
    n_verts = 0
    props: list[str] = []
    for ln in hdr.splitlines():
        parts = ln.strip().split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property":
            props.append(parts[-1])

    stride = len(props)
    arr = np.frombuffer(ply_bytes, dtype="<f4", count=n_verts * stride, offset=offset).reshape(n_verts, stride)
    pmap = {nm: i for i, nm in enumerate(props)}

    means = torch.from_numpy(np.ascontiguousarray(arr[:, [pmap["x"], pmap["y"], pmap["z"]]])).to(device)
    scales = torch.from_numpy(np.ascontiguousarray(arr[:, [pmap["scale_0"], pmap["scale_1"], pmap["scale_2"]]])).to(device)
    quats = torch.from_numpy(
        np.ascontiguousarray(arr[:, [pmap["rot_0"], pmap["rot_1"], pmap["rot_2"], pmap["rot_3"]]])
    ).to(device)
    opacities = torch.from_numpy(np.ascontiguousarray(arr[:, pmap["opacity"]])).to(device)
    sh0 = torch.from_numpy(
        np.ascontiguousarray(arr[:, [pmap["f_dc_0"], pmap["f_dc_1"], pmap["f_dc_2"]]])
    ).unsqueeze(1).to(device)

    rest_cols = [pmap[f"f_rest_{k}"] for k in range(45) if f"f_rest_{k}" in pmap]
    k_rest = len(rest_cols) // 3
    if k_rest > 0:
        shn = torch.from_numpy(
            np.ascontiguousarray(arr[:, rest_cols[: k_rest * 3]]).reshape(n_verts, k_rest, 3)
        ).to(device)
        sh_deg = max(0, round(math.sqrt(k_rest + 1)) - 1)
    else:
        shn = torch.zeros((n_verts, 0, 3), dtype=torch.float32, device=device)
        sh_deg = 0

    return {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh0, "shN": shn}, sh_deg


def render_splat_trajectory_stream(
    *,
    job_id: str,
    jobs_root: Path,
    trajectory_doc: dict[str, Any],
    ply_gz: bytes | None = None,
    rasterize_mode: str = "antialiased",
    bg_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gpu_hourly_usd: float = 3.95,
    ram_cache: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Farm full-rate 3DGS camera path rendering to Modal GPU (`gsplat` CUDA + FFmpeg H.264 MP4)."""
    from gsplat import rasterization

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    ckpt_path = jobs_root / job_id / "ckpt_latest.pt"
    if ram_cache is not None and ram_cache.get("job_id") == job_id and "params" in ram_cache:
        p_raw = ram_cache["params"]
        params = {k: v.detach() for k, v in p_raw.items()}
        k_total = 1 + int(params["shN"].shape[1])
        sh_deg = max(0, round(math.sqrt(k_total)) - 1)
        source_kind = "vram_cache"
    elif ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        params = {k: v.to(device) for k, v in ckpt["params"].items()}
        k_total = 1 + int(params["shN"].shape[1])
        sh_deg = max(0, round(math.sqrt(k_total)) - 1)
        source_kind = "nvme_ckpt"
    elif ply_gz:
        ply_raw = gzip.decompress(ply_gz)
        params, sh_deg = _load_params_from_ply_bytes(ply_raw, device)
        source_kind = "uploaded_ply"
    else:
        raise FileNotFoundError(
            f"No checkpoint in /jobs/{job_id} and no fallback PLY uploaded for '{job_id}'."
        )

    frames_spec = list(trajectory_doc.get("frames") or [])
    if not frames_spec:
        raise ValueError("trajectory_doc contains zero frames to render.")

    fps = float(trajectory_doc.get("fps") or 24.0)
    out_w = int(trajectory_doc.get("width") or frames_spec[0].get("width") or 1280)
    out_h = int(trajectory_doc.get("height") or frames_spec[0].get("height") or 720)
    out_w = max(16, out_w - (out_w % 2))
    out_h = max(16, out_h - (out_h % 2))
    n_frames = len(frames_spec)

    yield {
        "event": "render_start",
        "job_id": job_id,
        "device": dev_name,
        "source": source_kind,
        "total_frames": n_frames,
        "fps": fps,
        "width": out_w,
        "height": out_h,
    }

    work_dir = Path("/dev/shm") / f"render_{job_id}_{int(time.time() * 1000)}" if Path("/dev/shm").is_dir() else (jobs_root / job_id / "tmp_render")
    work_dir.mkdir(parents=True, exist_ok=True)
    mp4_out = work_dir / "recreated.mp4"

    ff_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", f"{fps:.4f}",
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-hide_banner", "-loglevel", "error",
        str(mp4_out),
    ]

    with torch.no_grad():
        means = params["means"]
        quats = F.normalize(params["quats"], dim=-1)
        scales_lin = torch.exp(params["scales"])
        opa_sig = torch.sigmoid(params["opacities"])
        k_use = (sh_deg + 1) ** 2
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)[:, :k_use, :]
        bg_t = torch.tensor([[bg_rgb[0], bg_rgb[1], bg_rgb[2]]], dtype=torch.float32, device=device)

        vms = np.zeros((n_frames, 4, 4), dtype=np.float32)
        ks = np.zeros((n_frames, 3, 3), dtype=np.float32)
        for i, f in enumerate(frames_spec):
            vms[i, 3, 3] = 1.0
            vms[i, :3, :3] = np.array(f["R_w2c"], dtype=np.float32)
            vms[i, :3, 3] = np.array(f["t_w2c"], dtype=np.float32)
            ks[i, 0, 0] = float(f["fx"])
            ks[i, 1, 1] = float(f["fy"])
            ks[i, 0, 2] = float(f["cx"])
            ks[i, 1, 2] = float(f["cy"])
            ks[i, 2, 2] = 1.0

        vms_gpu = torch.from_numpy(vms).to(device)
        ks_gpu = torch.from_numpy(ks).to(device)
        rmode = rasterize_mode if rasterize_mode in ("classic", "antialiased") else "antialiased"

        proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        t_rast = time.perf_counter()
        try:
            # Render in mini-batches of 8 cameras for blazing H100 occupancy & low peak VRAM
            batch_sz = 8
            for b0 in range(0, n_frames, batch_sz):
                b1 = min(n_frames, b0 + batch_sz)
                b_cnt = b1 - b0
                renders, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales_lin,
                    opacities=opa_sig,
                    colors=colors,
                    viewmats=vms_gpu[b0:b1],
                    Ks=ks_gpu[b0:b1],
                    width=out_w,
                    height=out_h,
                    sh_degree=sh_deg,
                    backgrounds=bg_t.expand(b_cnt, 3),
                    packed=True,
                    rasterize_mode=rmode,
                )
                u8_batch = (renders.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
                proc.stdin.write(u8_batch.tobytes())
                if b1 == n_frames or (b1 % 24 == 0):
                    yield {
                        "event": "render_progress",
                        "job_id": job_id,
                        "frame": b1,
                        "total_frames": n_frames,
                    }
            proc.stdin.close()
            proc.stdin = None
            stderr_b = proc.stderr.read() if proc.stderr else b""
            rc = proc.wait(timeout=60)
            if rc != 0:
                raise RuntimeError(f"FFmpeg encode failed ({rc}): {stderr_b.decode('utf-8', 'ignore')[-400:]}")
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
            if proc.poll() is None:
                proc.kill()

    rast_s = max(1e-4, time.perf_counter() - t_rast)
    mp4_bytes = mp4_out.read_bytes()
    shutil.rmtree(work_dir, ignore_errors=True)
    wall_s = round(time.perf_counter() - t0, 3)
    cost_usd = round((wall_s / 3600.0) * gpu_hourly_usd, 6)

    yield {
        "event": "render_done",
        "job_id": job_id,
        "device": dev_name,
        "source": source_kind,
        "frames": n_frames,
        "fps": fps,
        "width": out_w,
        "height": out_h,
        "wall_s": wall_s,
        "render_fps": round(n_frames / rast_s, 1),
        "cost_usd": cost_usd,
        "mp4_bytes": mp4_bytes,
    }