"""Noise-floor, kernel stage-timing, and parameter/gradient telemetry analyzer for 3DGS jobs.

Analyzes both:
  1. Existing completed `telemetry.jsonl` + `splat.ply` + `cameras.json` runs (retroactive diagnosis
     for jobs run before `kernel_diag` existed).
  2. Live/new runs carrying per-step `kernel_diag` (`grads`, `raster`, `params`, `adam`, `loss_terms`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .splat_edit import parse_ply_gaussians

_SH_C0 = 0.28209479177387814

STAGE_TIMING_KEYS: tuple[tuple[str, str, str], ...] = (
    ("param_act_ms", "Param Activations (sigmoid/exp/norm)", "#a855f7"),
    ("fwd_raster_only_ms", "gsplat Forward (Proj + Radix + Tile Raster)", "#38bdf8"),
    ("loss_ssim_ms", "Photometric L1 + 11×11 Conv SSIM + Reg", "#eab308"),
    ("bwd_ms", "gsplat Reverse-Order Tile Backward + Atomic Grad", "#f97316"),
    ("adam_ms", "6× Per-Tensor Adam Update (fp32, eps=1e-15)", "#22c55e"),
    ("densify_ms", "DefaultStrategy Clone / Split / Prune", "#ec4899"),
    ("host_ms", "Python / PyTorch Autograd Dispatch Overhead", "#64748b"),
)


def summarize_splat_telemetry(
    job_dir: Path,
    ticks: list[dict[str, Any]] | None = None,
    stages: list[dict[str, Any]] | None = None,
    densify_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a structured diagnostic summary of stage timings, view-sampling floor, and PLY distributions."""
    telem_p = job_dir / "telemetry.jsonl"
    if ticks is None or stages is None or densify_events is None:
        t_list: list[dict[str, Any]] = []
        s_list: list[dict[str, Any]] = []
        d_list: list[dict[str, Any]] = []
        if telem_p.is_file():
            for ln in telem_p.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except ValueError:
                    continue
                ev = row.get("event")
                if ev == "telemetry":
                    t_list.append(row)
                elif ev == "densify":
                    d_list.append(row)
                elif ev in ("stage", "done", "checkpoint_saved"):
                    s_list.append(row)
        ticks = t_list if ticks is None else ticks
        stages = s_list if stages is None else stages
        densify_events = d_list if densify_events is None else densify_events

    # 1. Compute smoothed EMA series + per-camera convergence table
    ema_loss = None
    ema_psnr = None
    by_cam: dict[str, list[dict[str, Any]]] = {}
    for t in ticks:
        lv = float(t.get("loss") or 0.0)
        pv = float(t.get("psnr") or 0.0)
        ema_loss = lv if ema_loss is None else (0.90 * ema_loss + 0.10 * lv)
        ema_psnr = pv if ema_psnr is None else (0.90 * ema_psnr + 0.10 * pv)
        if "loss_ema" not in t:
            t["loss_ema"] = round(ema_loss, 5)
        if "psnr_ema" not in t:
            t["psnr_ema"] = round(ema_psnr, 2)
        cname = str(t.get("camera_name") or "")
        if cname:
            by_cam.setdefault(cname, []).append(t)

    # 2. Per-iteration GPU stage timing breakdown (excluding step 1 cold CUDA context init spike)
    steady_ticks = [t for t in ticks if int(t.get("step") or 0) > 1 and isinstance(t.get("stage_timings"), dict)]
    if not steady_ticks and ticks:
        steady_ticks = [t for t in ticks if isinstance(t.get("stage_timings"), dict)]

    timing_summary: dict[str, Any] = {"samples": len(steady_ticks), "stages": [], "step_ms_mean": 0.0, "step_ms_p95": 0.0}
    if steady_ticks:
        has_split_fwd = any("fwd_raster_only_ms" in t["stage_timings"] for t in steady_ticks)
        step_arr = np.array([float(t["stage_timings"].get("step_ms") or 0.0) for t in steady_ticks], dtype=np.float64)
        mean_step = float(np.mean(step_arr)) if len(step_arr) else 0.0
        p95_step = float(np.percentile(step_arr, 95)) if len(step_arr) else 0.0
        timing_summary["step_ms_mean"] = round(mean_step, 3)
        timing_summary["step_ms_p95"] = round(p95_step, 3)
        timing_summary["it_per_sec_steady"] = round(1000.0 / max(mean_step, 1e-3), 1)

        stage_rows: list[dict[str, Any]] = []
        total_accounted = 0.0
        for key, label, color in STAGE_TIMING_KEYS:
            if key == "param_act_ms" and not has_split_fwd:
                continue
            eff_key = "fwd_raster_ms" if (key == "fwd_raster_only_ms" and not has_split_fwd) else key
            vals = np.array(
                [float(t["stage_timings"].get(eff_key) or 0.0) for t in steady_ticks],
                dtype=np.float64,
            )
            m_v = float(np.mean(vals))
            p50_v = float(np.median(vals))
            p95_v = float(np.percentile(vals, 95))
            mx_v = float(np.max(vals))
            total_accounted += m_v
            stage_rows.append({
                "key": eff_key,
                "label": label,
                "color": color,
                "mean_ms": round(m_v, 3),
                "p50_ms": round(p50_v, 3),
                "p95_ms": round(p95_v, 3),
                "max_ms": round(mx_v, 3),
                "pct": 0.0,
            })
        denom = max(total_accounted, 1e-6)
        for r in stage_rows:
            r["pct"] = round(100.0 * r["mean_ms"] / denom, 1)
        timing_summary["stages"] = stage_rows

    # 3. Macro Pipeline Stage Breakdown (`extract`, `pose`, `align`, `train`, `export`)
    macro: dict[str, float] = {}
    for s in stages:
        if s.get("event") == "done" and isinstance(s.get("macro_timings"), dict):
            macro = {k: round(float(v), 3) for k, v in s["macro_timings"].items()}
            break
    if not macro:
        for s in stages:
            if s.get("event") == "stage" and s.get("stage") and s.get("elapsed_s") is not None:
                macro[f"{s['stage']}_s"] = round(float(s["elapsed_s"]), 3)

    # 4. Per-Camera PSNR Variance vs Within-Camera Trajectory (Diagnoses single-camera SGD sawtooth!)
    cam_table: list[dict[str, Any]] = []
    for cname, cticks in by_cam.items():
        first_p = float(cticks[0].get("psnr") or 0.0)
        last_p = float(cticks[-1].get("psnr") or 0.0)
        bwd_vals = [
            float((ct.get("stage_timings") or {}).get("bwd_ms") or 0.0)
            for ct in cticks
            if isinstance(ct.get("stage_timings"), dict)
        ]
        cam_table.append({
            "camera": cname,
            "samples": len(cticks),
            "first_psnr": round(first_p, 2),
            "last_psnr": round(last_p, 2),
            "gain_db": round(last_p - first_p, 2),
            "last_loss": round(float(cticks[-1].get("loss") or 0.0), 5),
            "mean_bwd_ms": round(float(np.mean(bwd_vals)) if bwd_vals else 0.0, 2),
        })
    cam_table.sort(key=lambda r: r["last_psnr"])

    # Late-window statistics (last 25% of steps)
    late_cut = int(ticks[-1]["step"] * 0.75) if ticks else 0
    late_ticks = [t for t in ticks if int(t.get("step") or 0) >= late_cut]
    raw_psnr_std = float(np.std([float(t.get("psnr") or 0.0) for t in late_ticks])) if len(late_ticks) >= 2 else 0.0
    ema_psnr_std = float(np.std([float(t.get("psnr_ema") or 0.0) for t in late_ticks])) if len(late_ticks) >= 2 else 0.0

    # Within-camera step-to-step residual jitter on the same view in the late window
    same_cam_deltas: list[float] = []
    for cticks in by_cam.values():
        c_late = [float(t.get("psnr") or 0.0) for t in cticks if int(t.get("step") or 0) >= late_cut]
        for i in range(1, len(c_late)):
            same_cam_deltas.append(abs(c_late[i] - c_late[i - 1]))
    within_cam_jitter_db = float(np.median(same_cam_deltas)) if same_cam_deltas else 0.0

    # 5. Trained `.ply` Physical Parameter Distribution & Saturation Audit
    ply_audit: dict[str, Any] = {}
    ply_p = job_dir / "splat.ply"
    if ply_p.is_file():
        try:
            parsed = parse_ply_gaussians(ply_p.read_bytes())
            d = parsed["data"]
            pmap = {nm: idx for idx, nm in enumerate(parsed["props"])}
            n_g = len(d)
            if n_g > 0 and "opacity" in pmap and "scale_0" in pmap:
                opa_raw = d[:, pmap["opacity"]].astype(np.float64)
                opa = 1.0 / (1.0 + np.exp(-np.clip(opa_raw, -40.0, 40.0)))
                opa_jac = opa * (1.0 - opa)
                sc = np.exp(np.clip(d[:, [pmap["scale_0"], pmap["scale_1"], pmap["scale_2"]]].astype(np.float64), -20.0, 10.0))
                sc_sorted = np.sort(sc, axis=1)
                sc_max = sc_sorted[:, 2]
                aniso = sc_max / (sc_sorted[:, 1] + 1e-8)
                sh0 = d[:, [pmap["f_dc_0"], pmap["f_dc_1"], pmap["f_dc_2"]]].astype(np.float64)
                rgb = 0.5 + _SH_C0 * sh0
                oot_gamut = np.any((rgb < 0.0) | (rgb > 1.0), axis=1)

                ply_audit = {
                    "n_gaussians": int(n_g),
                    "opa_quantiles": [round(float(x), 4) for x in np.percentile(opa, [5, 25, 50, 75, 95])],
                    "opa_dead_low_frac": round(float(np.mean(opa < 0.02)), 4),
                    "opa_ghost_band_frac": round(float(np.mean((opa >= 0.02) & (opa < 0.08))), 4),
                    "opa_dead_high_frac": round(float(np.mean(opa > 0.99)), 4),
                    "opa_jacobian_mean": round(float(np.mean(opa_jac)), 5),
                    "scale_m_quantiles": [round(float(x), 5) for x in np.percentile(sc_max, [10, 50, 90, 99])],
                    "scale_m_max": round(float(np.max(sc_max)), 4),
                    "giant_billboard_count": int(np.sum(sc_max > 0.25)),
                    "aniso_p50": round(float(np.percentile(aniso, 50)), 2),
                    "aniso_p95": round(float(np.percentile(aniso, 95)), 2),
                    "aniso_max": round(float(np.max(aniso)), 2),
                    "sh0_oot_gamut_frac": round(float(np.mean(oot_gamut)), 4),
                }
        except Exception:  # noqa: BLE001
            ply_audit = {}

    # 6. Camera / Pose Alignment Audit (`cameras.json`)
    pose_audit: dict[str, Any] = {}
    cam_p = job_dir / "cameras.json"
    if cam_p.is_file():
        try:
            cdoc = json.loads(cam_p.read_text(encoding="utf-8"))
            vmeta = cdoc.get("video_meta") or {}
            sim3 = cdoc.get("sim3") or {}
            cams_l = cdoc.get("cameras") or []
            locs = np.array([c["location"] for c in cams_l], dtype=np.float64) if cams_l else np.zeros((0, 3))
            path_len = float(np.linalg.norm(np.diff(locs, axis=0), axis=1).sum()) if len(locs) >= 2 else 0.0
            pose_audit = {
                "pose_backend": cdoc.get("pose_backend", "vggt"),
                "pose_refined": bool(cdoc.get("pose_refined", False)),
                "mean_pose_shift_mm": cdoc.get("mean_pose_shift_mm"),
                "registered_cameras": len(cams_l),
                "source_duration_s": vmeta.get("duration_sec"),
                "source_fps": vmeta.get("fps"),
                "train_resolution": f"{cams_l[0].get('width')}x{cams_l[0].get('height')}" if cams_l else None,
                "floor_inlier_ratio": sim3.get("floor_inlier_ratio"),
                "baseline_ratio": sim3.get("baseline_ratio"),
                "sim3_scale": round(float(sim3.get("scale") or 1.0), 4),
                "camera_path_length_m": round(path_len, 2),
            }
        except Exception:  # noqa: BLE001
            pose_audit = {}

    # 7. Automated Root-Cause Evidence Findings ("Where is the noise floor?")
    findings: list[dict[str, str]] = []
    if raw_psnr_std > 1.5 and (cam_table and (cam_table[-1]["last_psnr"] - cam_table[0]["last_psnr"]) > 6.0):
        findings.append({
            "severity": "high",
            "subsystem": "Stochastic Batch=1 View Sampling (Chart Sawtooth vs True Floor)",
            "evidence": (
                f"Raw step-to-step PSNR swings ±{raw_psnr_std:.2f} dB ({cam_table[0]['camera']} = "
                f"{cam_table[0]['last_psnr']} dB vs {cam_table[-1]['camera']} = {cam_table[-1]['last_psnr']} dB), "
                f"whereas same-camera step delta is {within_cam_jitter_db:.2f} dB and multi-view EMA std is {ema_psnr_std:.2f} dB."
            ),
            "detail": (
                "Each training step rasterizes 1 random camera out of M views. Plotting raw single-frame loss/PSNR "
                "creates a dense sawtooth band proportional to inter-frame difficulty rather than kernel numerical noise."
            ),
        })

    if pose_audit:
        n_reg = int(pose_audit.get("registered_cameras") or 0)
        path_m = float(pose_audit.get("camera_path_length_m") or 0.0)
        refined = bool(pose_audit.get("pose_refined"))
        floor_ir = pose_audit.get("floor_inlier_ratio")
        shift_mm = float(pose_audit.get("mean_pose_shift_mm") or 0.0)
        if path_m > 20.0 and (not refined or pose_audit.get("pose_backend") == "vggt"):
            findings.append({
                "severity": "high" if not refined else "medium",
                "subsystem": "Feed-Forward Pose / Multi-Room Trajectory Consistency Floor",
                "evidence": (
                    f"{pose_audit.get('pose_backend')} solved {n_reg} views across a {path_m:.1f}m trajectory "
                    f"(se(3) refined={refined}{f', mean nudge {shift_mm:.1f}mm' if shift_mm else ''}, floor_inlier_ratio={floor_ir})."
                ),
                "detail": (
                    "Feed-forward VGGT-1B runs at 518px attention resolution (capped to 64 frames). On long interior "
                    "walkthroughs (>20m travel) or games with dynamic HUD/viewmodel/torch lighting, 0.5–2° loop-closure "
                    "drift (>8mm local se(3) tether) caps worst-view PSNR around 18–20 dB while clean views reach 28–29 dB."
                ),
            })

    if steady_ticks and timing_summary.get("stages"):
        bwd_row = next((r for r in timing_summary["stages"] if r["key"] == "bwd_ms"), None)
        if bwd_row and bwd_row["pct"] >= 55.0:
            findings.append({
                "severity": "medium",
                "subsystem": "Tile Raster Backward Atomic Contention & Overdraw",
                "evidence": (
                    f"`bwd_ms` consumes {bwd_row['pct']}% of step time (mean {bwd_row['mean_ms']} ms, "
                    f"p95 {bwd_row['p95_ms']} ms vs forward {timing_summary['stages'][0]['mean_ms']} ms)."
                ),
                "detail": (
                    "Interior room captures place cameras *inside* the Gaussian shell, so large wall/ceiling splats "
                    "overlap hundreds of 16×16 tiles. In `gsplat` backward, every pixel thread walks tile splats back-to-front "
                    "and issues `gpuAtomicAdd` onto packed gradient buffers (`v_means2d`, `v_colors`, `v_opacities`)."
                ),
            })

    if ply_audit:
        ghost_f = float(ply_audit.get("opa_ghost_band_frac") or 0.0)
        dead_lo = float(ply_audit.get("opa_dead_low_frac") or 0.0)
        giant_cnt = int(ply_audit.get("giant_billboard_count") or 0)
        if (ghost_f + dead_lo) > 0.20 or giant_cnt > 50:
            findings.append({
                "severity": "medium",
                "subsystem": "Sigmoid Opacity Dead-Zone & Unpruned Post-Densify Floaters",
                "evidence": (
                    f"{100.0 * (ghost_f + dead_lo):.1f}% of Gaussians sit below opacity 0.08 "
                    f"(sigmoid Jacobian σ(1−σ) < 0.074), and {giant_cnt} splats exceed 0.25m world radius "
                    f"(max={ply_audit.get('scale_m_max')}m)."
                ),
                "detail": (
                    "Opacity resets (`clamp <= 0.01`) and `opa_reg` push unused splats into low sigmoid slope where "
                    "gradients attenuate by >13×; once `step > densify_until` (72% of run), pruning stops while L1+SSIM "
                    "continues nudging semi-transparent haze."
                ),
            })

    latest_kdiag = next((t["kernel_diag"] for t in reversed(ticks) if isinstance(t.get("kernel_diag"), dict)), None)

    return {
        "timing_summary": timing_summary,
        "macro_timings": macro,
        "camera_convergence": {
            "total_cameras": len(cam_table),
            "late_raw_psnr_std_db": round(raw_psnr_std, 2),
            "late_ema_psnr_std_db": round(ema_psnr_std, 2),
            "within_cam_step_jitter_db": round(within_cam_jitter_db, 2),
            "worst_cameras": cam_table[:6],
            "best_cameras": list(reversed(cam_table[-6:])),
        },
        "ply_audit": ply_audit,
        "pose_audit": pose_audit,
        "latest_kernel_diag": latest_kdiag,
        "findings": findings,
        "kernel_architecture": {
            "forward_kernels": [
                "PyTorch Pointwise Activations (`sigmoid(opa)`, `exp(scales)`, `normalize(quats)`) — unfused HBM tensors",
                "`gsplat::fully_fused_projection_fwd_kernel` (`packed=True`) — frustum cull + EWA 3D→2D covariance + conic inverse",
                "`cub::DeviceRadixSort::SortPairs` — 64-bit `(tile_id << 32 | depth_f32_bits)` front-to-back key sort",
                "`gsplat::rasterize_to_pixels_fwd_kernel` — 16×16 pixel threadblock, shared-mem splat batches, `T < 1e-4` early stop",
            ],
            "loss_kernels": [
                "PyTorch ATen L1 + `F.conv2d(..., groups=3)` 11×11 separable-window SSIM (5 conv launches, unfused) + opa/scale/aniso penalties",
            ],
            "backward_kernels": [
                "`gsplat::rasterize_to_pixels_bwd_kernel` (`absgrad=True`) — reverse back-to-front transmittance recovery `T_prev = T_cur / (1 - α)`, warp/block reductions, `gpuAtomicAdd` to packed `M` visible splats",
                "`gsplat::fully_fused_projection_bwd_kernel` — chain rule from 2D conic/means2d/RGB back to 3D `means`, `quats`, `scales`, `sh0/shN` (+ sparse scatter `M -> N`)",
            ],
            "optimizer_kernels": [
                "6 separate `torch.optim.Adam` instances (`means`, `scales`, `quats`, `opacities`, `sh0`, `shN`), `betas=(0.9, 0.999)`, `eps=1e-15`, `fused=True` multi-tensor CUDA kernel per param group",
            ],
        },
    }