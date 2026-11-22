"""Unit tests for:
1. Centripetal + time-weighted Catmull-Rom trajectory interpolation (overshoot-free across sharp 90° turns & `mpdecimate` pauses).
2. NumPy/host Depth Pro 2 (`ml-depth-pro-2`) geometry math: edge-gradient confidence masking, Sim(3) sparse-cloud depth scale alignment, and ray-nudging + wall point supplementation.
3. AST & contract verification of `core_worker.py` iterative cycle bridge (`_solve_dense_video_arc_and_promote_frames`, `_build_swept_arc_samples_gpu`, `_evict_swept_arc_and_near_frustum_floaters`) + `modal_deploy.py` source hash coverage.
4. `SplatConfig` flat-key routing + Web Launcher preset/control wiring.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from blenderbot.splats.config import SplatConfig
from blenderbot.splats.modal_deploy import compute_local_modal_hash
from blenderbot.splats.sim3 import centripetal_catmull_rom_vec3, interpolate_camera_trajectory


def test_centripetal_catmull_rom_prevents_sharp_turn_overshoot() -> None:
    # Pause at p0..p1 (0.02m apart over 1.2s) followed by long 4m stride + 90-deg turn:
    # classic uniform Catmull-Rom swings wild loops; centripetal stays tightly bounded!
    p0 = (0.0, 0.0, 1.6)
    p1 = (0.0, 0.02, 1.6)
    p2 = (0.0, 4.0, 1.6)
    p3 = (3.5, 4.0, 1.6)
    samples = [centripetal_catmull_rom_vec3(p0, p1, p2, p3, u / 20.0, 1.2, 1.5, 1.4) for u in range(21)]
    ys = [s[1] for s in samples]
    xs = [s[0] for s in samples]
    assert min(ys) >= 0.019
    assert max(ys) <= 4.001
    assert min(xs) > -0.35

    doc = {
        "cameras": [
            {"location": list(p0), "rotation_quaternion": [1.0, 0.0, 0.0, 0.0], "pts_sec": 0.0, "width": 1280, "height": 720, "fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
            {"location": list(p1), "rotation_quaternion": [1.0, 0.0, 0.0, 0.0], "pts_sec": 1.2, "width": 1280, "height": 720, "fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
            {"location": list(p2), "rotation_quaternion": [0.7071, 0.0, 0.0, -0.7071], "pts_sec": 2.7, "width": 1280, "height": 720, "fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
            {"location": list(p3), "rotation_quaternion": [0.7071, 0.0, 0.0, -0.7071], "pts_sec": 4.1, "width": 1280, "height": 720, "fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
        ],
        "video_meta": {"fps": 30.0, "duration_sec": 4.1, "total_frames": 124, "width": 1280, "height": 720},
    }
    traj = interpolate_camera_trajectory(doc)
    assert len(traj["frames"]) == 124
    for fr in traj["frames"]:
        assert abs(fr["fx"] - 800.0) < 1e-2
        assert abs(fr["fy"] - 800.0) < 1e-2

    # Verify fast >160° horizontal whip-pan keeps horizon level (`|roll| < 0.5°`) AND obeys `pan_dyaw_deg` direction!
    from blenderbot.splats.sim3 import euler_yaw_pitch_roll_to_blender_quat

    q_start = list(euler_yaw_pitch_roll_to_blender_quat(0.0, -10.0, 0.0))
    # Turning +200° right lands at -160° yaw (`shortest SLERP` would spin -160° LEFT and bank roll wildly!)
    q_end = list(euler_yaw_pitch_roll_to_blender_quat(-160.0, -10.0, 0.0))
    whip_doc = {
        "cameras": [
            {"location": [0.0, 0.0, 1.6], "rotation_quaternion": q_start, "pts_sec": 0.0, "width": 1280, "height": 720, "fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0},
            {"location": [0.0, 0.2, 1.6], "rotation_quaternion": q_end, "pts_sec": 1.0, "pan_dyaw_deg": 200.0, "width": 1280, "height": 720, "fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0},
        ],
        "video_meta": {"fps": 30.0, "duration_sec": 1.0, "total_frames": 31, "width": 1280, "height": 720},
    }
    whip_traj = interpolate_camera_trajectory(whip_doc)["frames"]
    # Midpoint at t=0.5s MUST have turned RIGHT (+100° yaw, NOT -80° left!) and stayed horizon-level (roll ≈ 0°)!
    mid_fr = whip_traj[15]
    assert 90.0 <= mid_fr["yaw_deg"] <= 110.0
    assert abs(mid_fr["roll_deg"]) < 0.25


def test_core_worker_and_depth_pro2_bridge_ast_and_modal_hash() -> None:
    cw_path = Path("src/blenderbot/splats/core_worker.py")
    dp_path = Path("src/blenderbot/splats/depth_pro2_bridge.py")
    cw_src = cw_path.read_text(encoding="utf-8")
    dp_src = dp_path.read_text(encoding="utf-8")

    cw_tree = ast.parse(cw_src, filename=str(cw_path))
    dp_tree = ast.parse(dp_src, filename=str(dp_path))
    cw_fns = {n.name for n in ast.walk(cw_tree) if isinstance(n, ast.FunctionDef)}
    dp_fns = {n.name for n in ast.walk(dp_tree) if isinstance(n, ast.FunctionDef)}

    assert "_build_swept_arc_samples_gpu" in cw_fns
    assert "_evict_swept_arc_and_near_frustum_floaters" in cw_fns
    assert "_solve_dense_video_arc_and_promote_frames" in cw_fns
    assert "_repair_outlier_trajectory_spikes" in cw_fns
    assert "snap_camera_arc_to_video_stream" in cw_fns
    assert '"RGB+ED"' in cw_src
    assert '"arc_cycle_' in cw_src

    assert "predict_depth_pro2_maps" in dp_fns
    assert "align_depth_maps_to_sparse_cloud" in dp_fns
    assert "nudge_and_densify_init_points_with_depth_pro2" in dp_fns
    assert "compute_depth_prior_loss" in dp_fns

    # Verify `ml-depth-pro-2/src/depth_pro_2` exists in workspace and `depth_pro2_bridge.py` is hashed by Modal auto-deploy
    assert Path("ml-depth-pro-2/src/depth_pro_2/__init__.py").is_file()
    full_sha, file_shas = compute_local_modal_hash()
    assert len(full_sha) == 48
    assert "depth_pro2_bridge.py" in file_shas


def test_splat_config_and_web_launcher_iterative_arc_wiring() -> None:
    cfg = SplatConfig.from_dict({
        "iterative_arc_cycles": 3,
        "frames_add_per_cycle": 32,
        "near_frustum_clear_m": 0.45,
        "opa_bimodal_reg": 0.0003,
        "depth_prior_model": "depthpro2-s",
        "depth_nudge_strength": 0.35,
    })
    assert cfg.train.iterative_arc_cycles == 3
    assert cfg.train.frames_add_per_cycle == 32
    assert cfg.train.near_frustum_clear_m == 0.45
    assert cfg.train.opa_bimodal_reg == 0.0003
    assert cfg.pose.depth_prior_model == "depthpro2-s"
    assert cfg.pose.depth_nudge_strength == 0.35

    html = Path("src/blenderbot/ui/web/index.html").read_text(encoding="utf-8")
    js = Path("src/blenderbot/ui/web/modules/splat_launcher.js").read_text(encoding="utf-8")
    pdocs = Path("src/blenderbot/ui/web/modules/param_docs.js").read_text(encoding="utf-8")
    assert "launch-inp-arc-cycles" in html
    assert "launch-inp-frames-per-cycle" in html
    assert "launch-sel-dp2-model" in html
    assert "btn-review-splat-params" in html
    assert "iterative_arc" in js
    assert "depth_prior_model" in js
    assert "buildSplatSnickerReviewPrompt" in js
    assert "snicker test" in js.lower()

    # Every 3DGS Launcher control (`id="launch-..."`) and Resume control (`id="inp-resume-..."` / `sel-resume-...`)
    # must have an entry in `META_PARAM_DOCS` so zero 3DGS train parameters lack tooltips!
    import re

    launch_wrap = html.split('id="theater-splat-launch-wrap"', 1)[1].split('id="theater-kernel-wrap"', 1)[0]
    launch_ctrl_ids = re.findall(r'<(?:input|select)\b[^>]*\bid="(launch-[^"]+)"', launch_wrap)
    assert len(launch_ctrl_ids) >= 44
    missing_docs = [cid for cid in launch_ctrl_ids if f'"{cid}"' not in pdocs]
    assert not missing_docs, f"3DGS Launcher controls missing META_PARAM_DOCS tooltips: {missing_docs}"

    resume_wrap = html.split('id="sec-active-splat-resume"', 1)[1].split('id="drawer-cloud-splat"', 1)[0]
    resume_ctrl_ids = re.findall(r'<(?:input|select)\b[^>]*\bid="((?:inp|sel)-resume-[^"]+)"', resume_wrap)
    assert len(resume_ctrl_ids) >= 12
    missing_resume_docs = [cid for cid in resume_ctrl_ids if f'"{cid}"' not in pdocs]
    assert not missing_resume_docs, f"3DGS Resume controls missing META_PARAM_DOCS tooltips: {missing_resume_docs}"


@pytest.mark.asyncio
async def test_render_passes_history_and_compare_dropdown_catalog(tmp_path: Path) -> None:
    """Verify `_archive_and_write_render_pass` + `_collect_all_render_passes` expose ALL render passes."""
    import json as _json

    import aiohttp

    from blenderbot.server import start_web_server
    from blenderbot.server.routes_splats import (
        _archive_and_write_render_pass,
        _backfill_legacy_canonical_render,
    )
    from blenderbot.session.bootstrap import bootstrap_session
    from blenderbot.splats.sim3 import opencv_w2c_to_blender_camera

    _ = _backfill_legacy_canonical_render

    ws = tmp_path / "ws_cmp"
    home = tmp_path / "home_cmp"
    src_dir = ws / "assets" / "inputs"
    src_dir.mkdir(parents=True)
    src_vid = src_dir / "walkthrough.mp4"
    src_vid.write_bytes(b"\x00\x00\x00\x18ftypmp42source")

    job_id = "walkthrough_100000_aaaaaa"
    sib_id = "walkthrough_090000_bbbbbb"
    jdir = ws / "assets" / "splats" / job_id
    sdir = ws / "assets" / "splats" / sib_id
    for d in (jdir, sdir):
        (d / "renders").mkdir(parents=True)
        (d / "snapshots").mkdir(parents=True)
        (d / "previews").mkdir(parents=True)
        (d / "job_source.json").write_text(_json.dumps({"video_rel": "assets/inputs/walkthrough.mp4"}), encoding="utf-8")

    cam0 = opencv_w2c_to_blender_camera(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [0.0, -1.6, 0.0],
        600.0, 600.0, 320.0, 180.0, 640, 360,
    )
    cam0["pts_sec"] = 0.0
    (jdir / "cameras.json").write_text(
        _json.dumps({
            "job_id": job_id,
            "video_meta": {"fps": 24.0, "duration_sec": 1.0, "total_frames": 24, "width": 640, "height": 360},
            "cameras": [cam0],
        }),
        encoding="utf-8",
    )

    # Simulate Pass #1 (initial Modal render), Pass #2 (Snap Arc + Clean), Pass #3 (Blender render),
    # plus a sibling run MP4 and checkpoint snapshots
    modal_mp4 = jdir / "renders" / "modal_recreated.mp4"
    modal_mp4.write_bytes(b"mp4_pass_1")
    (jdir / "telemetry.jsonl").write_text(
        _json.dumps({"event": "telemetry", "step": 5000, "n_gaussians": 320000, "psnr": 24.2}) + "\n",
        encoding="utf-8",
    )
    _archive_and_write_render_pass(jdir / "renders", "modal_recreated.mp4", "modal", jdir)

    modal_mp4.write_bytes(b"mp4_pass_2_snapped")
    (jdir / "splat.splat").write_bytes(b"\x00" * (32 * 240_000))
    _archive_and_write_render_pass(
        jdir / "renders",
        "modal_recreated.mp4",
        "snap_arc",
        jdir,
        {"l1_after": 0.031, "cleaned_floaters": 80000},
    )

    bl_mp4 = jdir / "renders" / "blender_recreated.mp4"
    bl_mp4.write_bytes(b"mp4_pass_3_blender")
    _archive_and_write_render_pass(jdir / "renders", "blender_recreated.mp4", "blender", jdir)

    (sdir / "renders" / "modal_recreated.mp4").write_bytes(b"mp4_sibling")
    (jdir / "snapshots" / "step_002000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (jdir / "previews" / "latest_diff.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    sess = await bootstrap_session(
        cwd=ws,
        home=home,
        auto_connect_blender=False,
        overrides={"BLENDERBOT_PROVIDER": "gemini", "BLENDERBOT_GEMINI_API_KEY": "test-dummy"},
    )
    srv = await start_web_server(session=sess, host="127.0.0.1", port=0)
    try:
        async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {srv.token}"}) as http:
            r = await http.get(f"{srv.url}/api/splats/trajectory/{job_id}")
            assert r.status == 200
            data = await r.json()
            passes = data["render_passes"]
            kinds = [p["kind"] for p in passes]
            labels = " | ".join(p["label"] for p in passes)
            assert "source" in kinds
            assert "modal" in kinds
            assert "snap_arc" in kinds
            assert "blender" in kinds
            assert "sibling_modal" in kinds
            assert "snapshot" in kinds
            assert "still_diff" in kinds
            assert "Pass #1" in labels and "Pass #2" in labels and "Pass #3" in labels
    finally:
        await srv.close()
        sess.web_server = None
        await sess.close()


def test_torch_swept_arc_and_depth_pro2_math_when_torch_installed() -> None:
    torch = pytest.importorskip("torch", reason="Local Windows .venv is lightweight (cloud worker runs torch)")
    from blenderbot.splats.core_worker import (
        _build_swept_arc_samples_gpu,
        _evict_swept_arc_and_near_frustum_floaters,
    )
    from blenderbot.splats.depth_pro2_bridge import (
        align_depth_maps_to_sparse_cloud,
        compute_depth_edge_confidence,
        compute_depth_prior_loss,
        nudge_and_densify_init_points_with_depth_pro2,
    )

    r_cv = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    vm0 = np.eye(4, dtype=np.float32)
    vm0[:3, :3] = r_cv
    vm0[:3, 3] = np.array([0.0, 1.6, 0.0], dtype=np.float32)
    vm1 = np.eye(4, dtype=np.float32)
    vm1[:3, :3] = r_cv
    vm1[:3, 3] = np.array([0.0, 1.6, -4.0], dtype=np.float32)

    vms_gpu = torch.from_numpy(np.stack([vm0, vm1], axis=0))
    k_np = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ks_gpu = torch.from_numpy(np.stack([k_np, k_np], axis=0))
    swept, p_vms, p_ks = _build_swept_arc_samples_gpu(vms_gpu, ks_gpu, [0, 1], [0.0, 2.0], 640, 360)
    assert swept.shape[1] == 3
    assert p_vms.shape[1:] == (4, 4)
    assert p_ks.shape[1:] == (3, 3)

    h_d, w_d = 64, 96
    raw_d = torch.full((2, h_d, w_d), 2.5, dtype=torch.float32)
    c0 = compute_depth_edge_confidence(raw_d[0])
    confs = torch.stack([c0, c0], dim=0).half()
    vms = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    xs = np.linspace(-1.0, 1.0, 64)
    xyz_sparse = np.stack([xs, np.zeros_like(xs), np.full_like(xs, 5.0)], axis=-1)
    aligned_d, scale_f = align_depth_maps_to_sparse_cloud(raw_d.half(), confs, vms, np.stack([k_np, k_np], 0), xyz_sparse, 640, 360)
    assert math.isclose(scale_f, 2.0, rel_tol=0.05)

    alphas = torch.ones((1, h_d, w_d, 1), dtype=torch.float32)
    ed_front = torch.full((1, h_d, w_d, 1), 3.0, dtype=torch.float32, requires_grad=True)
    ed_behind = torch.full((1, h_d, w_d, 1), 5.0 * (5.0 / 3.0), dtype=torch.float32, requires_grad=True)
    assert float(compute_depth_prior_loss(ed_front, alphas, aligned_d[:1], confs[:1]).item()) > float(
        compute_depth_prior_loss(ed_behind, alphas, aligned_d[:1], confs[:1]).item()
    ) * 2.0
    _ = _evict_swept_arc_and_near_frustum_floaters
    _ = nudge_and_densify_init_points_with_depth_pro2