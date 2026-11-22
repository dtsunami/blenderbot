"""Tests for 3DGS kernel intermediate telemetry, retroactive & live noise-floor/stage-timing
analyzer (`summarize_splat_telemetry`), `/splat diag` CLI, and Web Studio `/api/splats/telemetry/{job_id}`
+ SVG hardware cartoon GUI wiring (runs on lightweight Windows host with zero local PyTorch needed).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import aiohttp
import numpy as np
import pytest

from blenderbot.server import start_web_server
from blenderbot.session.bootstrap import bootstrap_session
from blenderbot.splats.diagnostics import summarize_splat_telemetry
from blenderbot.ui.commands import handle_command


def test_core_worker_telemetry_helpers_syntax_and_contract() -> None:
    src_path = Path("src/blenderbot/splats/core_worker.py")
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(src_path))
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_tensor_grad_stats" in fn_names
    assert "_collect_kernel_telemetry" in fn_names
    assert '"kernel_diag"' in src
    assert '"param_act_ms"' in src
    assert '"fwd_raster_only_ms"' in src
    assert '"loss_ema"' in src
    assert '"psnr_ema"' in src
    assert "fused=True" in src


def test_core_worker_torch_tensor_helpers_when_cuda_or_torch_installed() -> None:
    torch = pytest.importorskip("torch", reason="Local Windows .venv is lightweight (no local torch)")
    from blenderbot.splats.core_worker import (
        _collect_kernel_telemetry,
        _init_gaussians_cuda,
        _make_optimizers,
        _tensor_grad_stats,
    )

    device = torch.device("cpu")
    rng = np.random.default_rng(123)
    xyz = rng.uniform(-2.0, 2.0, size=(64, 3))
    rgb = rng.integers(0, 255, size=(64, 3), dtype=np.uint8)
    params = _init_gaussians_cuda(xyz, rgb, sh_degree=1, device=device)
    opts = _make_optimizers(params, scene_scale=3.0)
    opa_sig = torch.sigmoid(params["opacities"])
    scales_lin = torch.exp(params["scales"])
    loss = (params["means"] ** 2).mean() + 0.05 * opa_sig.mean() + 0.01 * scales_lin.mean()
    loss.backward()
    for opt in opts.values():
        opt.step()
    assert _tensor_grad_stats(params["means"])["rms"] > 0.0
    kdiag = _collect_kernel_telemetry(
        params=params,
        optimizers=opts,
        info={"radii": torch.ones((32,)), "flatten_ids": torch.arange(512)},
        ren_alphas=torch.ones((1, 32, 32, 1)),
        opa_sig=opa_sig,
        scales_lin=scales_lin,
        loss_terms={"l1_weighted": 0.02, "dssim_weighted": 0.01, "opa_reg": 0.0, "scale_reg": 0.0, "aniso_reg": 0.0, "alpha_sil": 0.0, "pose_reg": 0.0},
    )
    assert kdiag["grads"]["means"]["rms"] > 0.0
    assert kdiag["adam"]["means"]["snr_p50"] > 0.0


def test_real_interior_capture_diagnostics_if_present() -> None:
    real_job = Path("assets/splats/2026-11-20_15-31-10_153534_676269")
    if not (real_job / "telemetry.jsonl").is_file():
        pytest.skip("Workspace interior capture telemetry not present")

    diag = summarize_splat_telemetry(real_job)
    tsum = diag["timing_summary"]
    assert tsum["samples"] >= 390
    bwd = next(r for r in tsum["stages"] if r["key"] == "bwd_ms")
    assert bwd["pct"] > 65.0
    assert diag["camera_convergence"]["total_cameras"] == 64
    assert diag["camera_convergence"]["late_raw_psnr_std_db"] > diag["camera_convergence"]["late_ema_psnr_std_db"]
    assert len(diag["findings"]) >= 3


@pytest.mark.asyncio
async def test_web_telemetry_diagnostics_and_splat_diag_cli(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    home = tmp_path / "home"
    jid = "interior_base_test_01"
    jdir = ws / "assets" / "splats" / jid
    jdir.mkdir(parents=True)

    # Write synthetic PLY + cameras.json + multi-camera telemetry.jsonl (including live `kernel_diag`)
    n_pts = 200
    xyz = np.zeros((n_pts, 3), dtype=np.float32)
    normals = np.zeros((n_pts, 3), dtype=np.float32)
    sh0 = np.zeros((n_pts, 3), dtype=np.float32)
    opa = np.concatenate([np.full((80, 1), -5.0, dtype=np.float32), np.full((120, 1), 1.5, dtype=np.float32)], axis=0)
    scales = np.full((n_pts, 3), math.log(0.04), dtype=np.float32)
    scales[0] = math.log(0.85)  # 1 giant wall billboard
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n_pts, 1))
    packed = np.concatenate([xyz, normals, sh0, opa, scales, quats], axis=1).astype("<f4")
    props = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    hdr = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_pts}\n"
        + "\n".join(f"property float {p}" for p in props)
        + "\nend_header\n"
    ).encode("ascii")
    (jdir / "splat.ply").write_bytes(hdr + packed.tobytes())

    (jdir / "cameras.json").write_text(
        json.dumps({
            "job_id": jid,
            "pose_backend": "vggt",
            "sim3": {"scale": 3.2, "floor_inlier_ratio": 0.0, "baseline_ratio": 2.8},
            "cameras": [
                {"image_name": "frame_00001.jpg", "location": [0.0, 0.0, 1.6], "width": 1280, "height": 720},
                {"image_name": "frame_00002.jpg", "location": [15.0, 12.0, 1.6], "width": 1280, "height": 720},
                {"image_name": "frame_00003.jpg", "location": [25.0, 20.0, 1.6], "width": 1280, "height": 720},
            ],
        }),
        encoding="utf-8",
    )

    rows = []
    for s in range(25, 525, 25):
        cam = "frame_00001.jpg" if (s // 25) % 2 == 0 else "frame_00002.jpg"
        psnr = 28.5 if cam == "frame_00001.jpg" else 18.2
        rows.append(json.dumps({
            "event": "telemetry",
            "job_id": jid,
            "step": s,
            "total_steps": 500,
            "loss": 0.03 if psnr > 25 else 0.11,
            "l1": 0.02,
            "ssim": 0.88,
            "psnr": psnr,
            "n_gaussians": n_pts,
            "camera_name": cam,
            "stage_timings": {
                "param_act_ms": 0.12,
                "fwd_raster_only_ms": 2.1,
                "fwd_raster_ms": 2.22,
                "loss_ssim_ms": 1.65,
                "bwd_ms": 18.4,
                "adam_ms": 0.88,
                "densify_ms": 0.02,
                "host_ms": 1.1,
                "step_ms": 24.27,
            },
            "kernel_diag": {
                "loss_terms": {"l1_weighted": 0.016, "dssim_weighted": 0.014, "opa_reg": 0.0005, "scale_reg": 0.0002, "aniso_reg": 0.0},
                "grads": {
                    "means": {"rms": 2.4e-5, "p50": 8.1e-6, "p99": 1.9e-4, "max": 8.2e-4, "nz_frac": 0.42, "nan_inf": 0},
                    "opacities": {"rms": 1.1e-4, "p50": 3.0e-5, "p99": 9.5e-4, "max": 3.1e-3, "nz_frac": 0.42, "nan_inf": 0},
                    "means2d_absgrad": {"p50": 4.2e-5, "p95": 3.8e-4, "p99": 1.1e-3, "max": 4.5e-3, "above_densify_frac": 0.09},
                },
                "raster": {"visible_gaussians": 84, "visible_frac": 0.42, "tile_isects": 92000, "avg_splats_per_tile": 25.5, "radius_px_p50": 9.4, "radius_px_max": 310.0, "T_floor_sat_frac": 0.34},
                "params": {"opa_jacobian_mean": 0.112},
                "adam": {"means": {"lr": 4e-5, "fused": True, "snr_p50": 0.38, "step_rms": 1.5e-5, "eps_dom_frac": 0.0}},
            },
        }))
    rows.append(json.dumps({
        "event": "done",
        "job_id": jid,
        "step": 500,
        "n_gaussians": n_pts,
        "macro_timings": {"extract_s": 1.5, "pose_s": 12.2, "align_s": 0.4, "train_s": 40.0, "export_s": 0.3},
    }))
    (jdir / "telemetry.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    sess = await bootstrap_session(
        cwd=ws,
        home=home,
        overrides={"BLENDERBOT_PROVIDER": "gemini", "BLENDERBOT_GEMINI_API_KEY": "dummy"},
    )
    pushed: list[dict] = []
    cmd_res = await handle_command(f"/splat diag {jid}", session=sess, push=pushed.append, exit_app=lambda: None)
    assert cmd_res["handled"] is True
    assert any("3DGS Kernel & Noise-Floor Report" in (m.get("text") or "") for m in pushed)

    # Also write a persisted local render + Seedream output to verify `scan_media_jobs` surfaces them
    (ws / "renders").mkdir(parents=True, exist_ok=True)
    (ws / "renders" / "shot_01.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    (ws / "assets" / "seedream").mkdir(parents=True, exist_ok=True)
    (ws / "assets" / "seedream" / "plate_01.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    srv = await start_web_server(session=sess, host="127.0.0.1", port=0)
    headers = {"Authorization": f"Bearer {srv.token}"}
    try:
        async with aiohttp.ClientSession(headers=headers) as http:
            r_html = await http.get(f"{srv.url}/")
            html_txt = await r_html.text()
            assert "svg-kdiag-cartoon" in html_txt
            assert "btn-view-kernel" in html_txt
            assert "btn-view-launch" in html_txt
            assert "btn-view-how" in html_txt
            assert "theater-splat-launch-wrap" in html_txt
            assert "btn-queue-splat-hero" in html_txt
            assert "theater-how-it-works-wrap" in html_txt
            assert "sec-active-splat-resume" in html_txt
            assert "btn-resume-splat-train" in html_txt
            assert "theater-job-wrap" in html_txt
            assert "param-tooltip" in html_txt
            assert 'data-filter="jobs"' in html_txt
            assert "svg-strategy-arc" in html_txt
            assert "btn-arc-lock" in html_txt

            r_js = await http.get(f"{srv.url}/assets/app.js")
            assert r_js.status == 200
            js_txt = await r_js.text()
            assert "openSplatLaunchDialog" in js_txt
            assert "openKernelTelemetryForJob" in js_txt
            assert "openHowItWorksInCenter" in js_txt
            assert "META_PARAM_DOCS" in js_txt
            assert "wireMetaParamTooltips" in js_txt
            assert "wireAdaptiveStrategyArc" in js_txt
            assert "evalArcScale" in js_txt

            # Verify nested ES modules serve cleanly under `/assets/modules/*`
            for mod_name in (
                "state.js",
                "sessions.js",
                "workbench.js",
                "explorer.js",
                "splat_launcher.js",
                "splat_telemetry.js",
                "composers.js",
                "fps_engine.js",
                "param_docs.js",
            ):
                r_mod = await http.get(f"{srv.url}/assets/modules/{mod_name}")
                assert r_mod.status == 200, f"Missing module {mod_name}"

            r_state = await http.get(f"{srv.url}/api/state")
            assert r_state.status == 200
            st_data = await r_state.json()
            assert any(j["job_id"] == jid for j in st_data["splat_jobs"])
            assert any(j["local_id"].startswith("bl-local:") for j in st_data["jobs"])
            assert any(j["local_id"].startswith("sr-disk:") for j in st_data["jobs"])

            r_tel = await http.get(f"{srv.url}/api/splats/telemetry/{jid}")
            assert r_tel.status == 200
            payload = await r_tel.json()
            diag = payload["diagnostics"]
            assert diag["timing_summary"]["samples"] == 20
            assert diag["latest_kernel_diag"]["grads"]["means"]["rms"] == pytest.approx(2.4e-5)
            assert len(diag["findings"]) >= 2
            assert diag["ply_audit"]["giant_billboard_count"] == 1
            assert "modal_status" in st_data
            assert len(st_data["modal_status"]["current_hash"]) == 48
    finally:
        await srv.close()
        sess.web_server = None
        await sess.close()


@pytest.mark.asyncio
async def test_modal_version_hash_and_autodeploy_trigger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    import blenderbot.splats.modal_deploy as md

    # 1. LF-normalized SHA-256 must match even when files have CRLF vs LF line endings
    d1 = tmp_path / "lf"
    d2 = tmp_path / "crlf"
    d1.mkdir()
    d2.mkdir()
    for fname in ("modal_app.py", "core_worker.py", "config.py"):
        (d1 / fname).write_bytes(b"def f():\n    return 1\n")
        (d2 / fname).write_bytes(b"def f():\r\n    return 1\r\n")
    h1, pf1 = md.compute_local_modal_hash(d1)
    h2, pf2 = md.compute_local_modal_hash(d2)
    assert h1 == h2 and len(h1) == 48
    assert pf1 == pf2

    # 2. When deployed hash != current hash, `ensure_modal_deployed` auto-deploys & stamps tags;
    #    on subsequent call with matching hash, it fast-paths without redeploying!
    monkeypatch.setattr(md, "_cached_verified_hash", None)
    monkeypatch.setattr(md, "_cached_status", None)
    monkeypatch.setenv("BLENDERBOT_MODAL_AUTODEPLOY", "1")

    cur_hash, _ = md.compute_local_modal_hash()
    stored_tags: dict[str, str] = {"source_sha256": "0" * 48, "source_short": "000000000000"}
    deploy_calls: list[dict] = []

    fake_remote_app = MagicMock()
    fake_remote_app.app_id = "ap-test123"
    fake_remote_app.get_tags.side_effect = lambda: dict(stored_tags)

    def _fake_set_tags(new_tags: dict[str, str]) -> None:
        stored_tags.update(new_tags)

    fake_remote_app.set_tags.side_effect = _fake_set_tags

    def _fake_deploy(**kwargs):
        deploy_calls.append(kwargs)
        return fake_remote_app

    monkeypatch.setattr("modal.App.lookup", lambda *_a, **_kw: fake_remote_app)
    from blenderbot.splats import modal_app

    monkeypatch.setattr(modal_app.app, "deploy", _fake_deploy)

    events: list[dict] = []
    r1 = await md.ensure_modal_deployed(on_event=events.append)
    assert r1["deployed"] is True
    assert r1["current_hash"] == cur_hash
    assert stored_tags["source_sha256"] == cur_hash
    assert len(deploy_calls) == 1
    assert [e["stage"] for e in events] == ["deploy", "deployed"]

    # Second call sees matching hash and does zero deploys!
    r2 = await md.ensure_modal_deployed()
    assert r2["deployed"] is False
    assert r2["up_to_date"] is True
    assert len(deploy_calls) == 1


def test_stereo_peer_table_and_multiview_config() -> None:
    from blenderbot.splats.config import SplatConfig
    from blenderbot.splats.sim3 import build_stereo_peer_table as _build_stereo_peer_table

    # 1. Verify `SplatConfig` routes all multi-view walkthrough consistency fields cleanly
    cfg = SplatConfig.from_dict(
        {
            "views_per_step": 3,
            "cam_corridor_radius_m": 0.4,
            "appearance_opt": True,
            "outlier_view_downweight": True,
            "adaptive_strategy": "adaptive_arc",
            "arc_scale_mode": "power",
            "arc_gamma": 1.8,
            "target_gaussians": 280000,
        }
    )
    assert cfg.train.views_per_step == 3
    assert cfg.train.cam_corridor_radius_m == pytest.approx(0.4)
    assert cfg.train.appearance_opt is True
    assert cfg.train.outlier_view_downweight is True
    assert cfg.train.arc_scale_mode == "power"
    assert cfg.train.arc_gamma == pytest.approx(1.8)
    assert cfg.train.target_gaussians == 280000

    # 2. Verify `_build_stereo_peer_table` pairs cameras with wide-baseline overlapping frustum peers
    vms = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 6, axis=0)
    for idx in range(6):
        ang = idx * 0.45  # ~25 deg steps around origin looking inward
        c, s = math.cos(ang), math.sin(ang)
        vms[idx, 0, 0], vms[idx, 0, 2] = c, -s
        vms[idx, 2, 0], vms[idx, 2, 2] = s, c
        vms[idx, 2, 3] = 3.0
    peers = _build_stereo_peer_table(vms, list(range(6)))
    assert len(peers) == 6
    assert all(len(v) >= 1 and k not in v for k, v in peers.items())
    # Nearest neighbor along arc (`idx=1` or `2` at ~25-50 deg) should outrank opposite-side cameras (`idx=5`)
    assert peers[0][0] in (1, 2)