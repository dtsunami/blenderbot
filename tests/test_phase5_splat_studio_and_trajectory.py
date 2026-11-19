"""Phase 5 tests:
1. `SplatConfig` flat/nested override routing & non-clobbering `merge_overrides` on `splat.resume`.
2. Exact inverse `blender_camera_to_opencv_w2c` <-> `opencv_w2c_to_blender_camera` + full-rate
   `interpolate_camera_trajectory` (Catmull-Rom + hemisphere SLERP + continuous unwrapped Yaw/Pitch/Roll).
3. Pure-NumPy Surface-Nets `proxy_collision.glb` extraction + headless Blender 5.3 `splat.import(import_proxy_mesh=True)`,
   `splat.crop_box` (`spawn` + `read`), and `splat.export_camera_path` (`seedance_camera_prompt`).
4. FastAPI Web Studio Phase 5 endpoints (`/api/splats/telemetry/{id}`, `/api/splats/trajectory/{id}`,
   `/api/splats/proxy/{id}`, `/api/splats/crop_box`, `/api/splats/camera_prompt`, and `/api/splats/render` `mode="blender"`).
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import aiohttp
import numpy as np
import pytest
import pytest_asyncio

from blenderbot.config.blender_paths import discover_blender
from blenderbot.server import start_web_server
from blenderbot.session.bootstrap import bootstrap_session
from blenderbot.splats.config import SplatConfig
from blenderbot.splats.proxy_mesh import extract_proxy_glb_from_ply
from blenderbot.splats.sim3 import (
    blender_camera_to_opencv_w2c,
    interpolate_camera_trajectory,
    opencv_w2c_to_blender_camera,
)
from blenderbot.transport import HeadlessTransport


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def blender_worker():
    inst = discover_blender()
    assert inst.ready
    worker = HeadlessTransport()
    await worker.start()
    try:
        yield worker
    finally:
        await worker.close()


def _write_synthetic_3dgs_sphere_ply(ply_path: Path, n_pts: int = 1200, radius: float = 0.8) -> None:
    rng = np.random.default_rng(7)
    vec = rng.normal(size=(n_pts, 3)).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    r = radius * (0.85 + 0.15 * rng.random((n_pts, 1), dtype=np.float32))
    xyz = vec * r + np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    normals = np.zeros((n_pts, 3), dtype=np.float32)
    sh0 = np.full((n_pts, 3), 0.6, dtype=np.float32)
    sh_rest = np.zeros((n_pts, 9), dtype=np.float32)
    opa = np.full((n_pts, 1), 2.5, dtype=np.float32)  # sigmoid ~ 0.92
    scales = np.full((n_pts, 3), math.log(0.05), dtype=np.float32)
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n_pts, 1))

    packed = np.concatenate([xyz, normals, sh0, sh_rest, opa, scales, quats], axis=1).astype("<f4")
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{k}" for k in range(9)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_pts}\n"
        + "\n".join(f"property float {p}" for p in props)
        + "\nend_header\n"
    ).encode("ascii")
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    ply_path.write_bytes(hdr + packed.tobytes())


def test_splat_config_flat_and_merge_overrides() -> None:
    cfg = SplatConfig.from_dict({
        "gpu": "l40s",
        "iterations": 8000,
        "sh_degree": 2,
        "lambda_dssim": 0.35,
        "opa_reg": 0.0012,
        "refine_poses": True,
        "pose_backend": "colmap_prior",
        "focus_camera": "frame_00012.jpg",
    })
    assert cfg.gpu == "L40S"
    assert cfg.train.iterations == 8000
    assert cfg.train.sh_degree == 2
    assert cfg.train.lambda_dssim == pytest.approx(0.35)
    assert cfg.train.opa_reg == pytest.approx(0.0012)
    assert cfg.train.refine_poses is True
    assert cfg.pose.backend == "colmap_prior"
    assert cfg.train.focus_camera == "frame_00012.jpg"

    # Resume override must preserve `sh_degree=2`, `lambda_dssim=0.35`, `refine_poses=True`!
    resumed = cfg.merge_overrides({"gpu": "H100", "opa_reg": 0.0008})
    assert resumed.gpu == "H100"
    assert resumed.train.sh_degree == 2
    assert resumed.train.lambda_dssim == pytest.approx(0.35)
    assert resumed.train.refine_poses is True
    assert resumed.train.opa_reg == pytest.approx(0.0008)


def test_camera_w2c_inverse_and_fullrate_trajectory_interpolation() -> None:
    # Orbiting camera crossing the +/-180 deg yaw seam to prove continuous unwrapping
    k_cams = []
    n_k = 6
    for i in range(n_k):
        ang = math.radians(150.0 + i * 16.0)  # crosses +180 -> -180!
        cx = 3.5 * math.sin(ang)
        cy = -3.5 * math.cos(ang)
        cz = 1.6
        # Look toward (0, 0, 1.0)
        fx, fy, fz = -cx, -cy, 1.0 - cz
        fl = math.sqrt(fx * fx + fy * fy + fz * fz)
        fwd = (fx / fl, fy / fl, fz / fl)
        rx, ry = fwd[1], -fwd[0]
        rl = math.hypot(rx, ry)
        right = (rx / rl, ry / rl, 0.0)
        up = (
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        )
        # OpenCV w2c rows: +X=right, +Y=-up, +Z=fwd
        r_w2c = [list(right), [-up[0], -up[1], -up[2]], list(fwd)]
        t_w2c = [
            -(r_w2c[0][0] * cx + r_w2c[0][1] * cy + r_w2c[0][2] * cz),
            -(r_w2c[1][0] * cx + r_w2c[1][1] * cy + r_w2c[1][2] * cz),
            -(r_w2c[2][0] * cx + r_w2c[2][1] * cy + r_w2c[2][2] * cz),
        ]
        bl = opencv_w2c_to_blender_camera(r_w2c, t_w2c, 1100.0, 1100.0, 640.0, 360.0, 1280, 720)
        rt = blender_camera_to_opencv_w2c(bl)
        assert np.allclose(rt["R_w2c"], r_w2c, atol=1e-5)
        assert np.allclose(rt["t_w2c"], t_w2c, atol=1e-5)
        assert rt["fx"] == pytest.approx(1100.0, rel=1e-5)

        bl["frame_index"] = i
        bl["pts_sec"] = round(i * 0.4, 4)
        bl["image_name"] = f"frame_{i + 1:05d}.jpg"
        k_cams.append(bl)

    doc = {
        "job_id": "orbit_test",
        "video_meta": {"fps": 25.0, "duration_sec": 2.0, "total_frames": 51, "width": 1280, "height": 720},
        "cameras": k_cams,
    }
    traj = interpolate_camera_trajectory(doc)
    assert traj["total_frames"] == 51
    assert traj["fps"] == 25.0
    assert len(traj["keyframe_markers"]) == n_k

    # Verify yaw unwraps smoothly across the 180 deg boundary (no >30 deg step jump!)
    yaws = [f["yaw_deg"] for f in traj["frames"]]
    max_step_dyaw = max(abs(yaws[i] - yaws[i - 1]) for i in range(1, len(yaws)))
    assert max_step_dyaw < 8.0


@pytest.mark.asyncio(loop_scope="module")
async def test_phase5_proxy_glb_cropbox_export_cam_and_web_comparator(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
) -> None:
    ws_dir = tmp_path / "p5_ws"
    home_dir = tmp_path / "p5_home"
    job_id = "seedance_20261118_120000_120500_a1b2c3"
    job_dir = ws_dir / "assets" / "splats" / job_id
    job_dir.mkdir(parents=True)

    # 1. Create synthetic 3DGS sphere `.ply` + extract `proxy_collision.glb`
    ply_p = job_dir / "splat.ply"
    _write_synthetic_3dgs_sphere_ply(ply_p, n_pts=1400, radius=0.75)
    glb_p = job_dir / "proxy_collision.glb"
    stats = extract_proxy_glb_from_ply(ply_p, glb_p, grid_res=32)
    assert glb_p.is_file() and glb_p.stat().st_size > 500
    magic, ver, _length = struct.unpack_from("<III", glb_p.read_bytes(), 0)
    assert magic == 0x46546C67 and ver == 2
    assert stats["vertices"] > 20 and stats["faces"] > 20

    # Write `cameras.json` + `telemetry.jsonl` + `snapshots/step_000500.jpg`
    cams = []
    for k in range(4):
        bl_c = opencv_w2c_to_blender_camera(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [-0.2 * k, -1.0, 3.2],
            520.0, 520.0, 160.0, 120.0, 320, 240,
        )
        bl_c["frame_index"] = k
        bl_c["pts_sec"] = round(k * 0.25, 4)
        bl_c["timeline_frame"] = k * 3 + 1
        bl_c["image_name"] = f"frame_{k + 1:05d}.jpg"
        cams.append(bl_c)

    (job_dir / "cameras.json").write_text(
        json.dumps({
            "job_id": job_id,
            "pose_backend": "vggt",
            "video_meta": {"fps": 12.0, "duration_sec": 1.0, "total_frames": 10, "width": 320, "height": 240},
            "cameras": cams,
        }),
        encoding="utf-8",
    )
    (job_dir / "telemetry.jsonl").write_text(
        "\n".join([
            json.dumps({
                "event": "telemetry", "job_id": job_id, "step": 250, "total_steps": 500,
                "loss": 0.042, "l1": 0.028, "ssim": 0.89, "psnr": 24.5, "n_gaussians": 1400,
                "stage_timings": {"fwd_raster_ms": 1.1, "loss_ssim_ms": 0.5, "bwd_ms": 1.8, "adam_ms": 0.9, "densify_ms": 0.1, "host_ms": 0.2, "step_ms": 4.6},
            }),
            json.dumps({"event": "densify", "job_id": job_id, "step": 300, "n_before": 1400, "n_after": 1650, "delta": 250}),
            json.dumps({
                "event": "telemetry", "job_id": job_id, "step": 500, "total_steps": 500,
                "loss": 0.019, "l1": 0.012, "ssim": 0.95, "psnr": 31.2, "n_gaussians": 1650,
                "stage_timings": {"fwd_raster_ms": 1.0, "loss_ssim_ms": 0.5, "bwd_ms": 1.7, "adam_ms": 0.9, "densify_ms": 0.1, "host_ms": 0.2, "step_ms": 4.4},
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    snap_dir = job_dir / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "step_000500.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    # 2. Verify Blender 5.3 `splat.import(import_proxy_mesh=True)`, `splat.crop_box`, & `splat.export_camera_path`
    await blender_worker.call("scene.clear")
    imp = await blender_worker.call(
        "splat.import",
        {
            "filepath": str(ply_p),
            "cameras_json": str(job_dir / "cameras.json"),
            "name": "SphereSplat",
            "render_mode": "pointcloud",
            "import_proxy_mesh": True,
            "shadow_catcher_ground": False,
        },
    )
    assert imp["proxy_mesh"] == "SphereSplat_ProxyMesh"
    assert imp["cameras"]["frames"] == 10
    assert imp["cameras"]["fps"] == 12.0

    # Verify proxy mesh imported at exact Z=1.0 sphere center (`Rx(-90)` glTF <-> Blender Z-up alignment!)
    insp = await blender_worker.call("scene.inspect", {"name": "SphereSplat_ProxyMesh"})
    aabb_p = insp["objects"][0]["world_aabb"]
    bmin_p = aabb_p["min"]
    bmax_p = aabb_p["max"]
    cz_proxy = 0.5 * (bmin_p[2] + bmax_p[2])
    assert abs(cz_proxy - 1.0) < 0.25

    cbox = await blender_worker.call(
        "splat.crop_box",
        {"action": "spawn", "location": [0.0, 0.0, 1.0], "dimensions": [1.6, 1.6, 1.6]},
    )
    assert cbox["object"] == "SplatCropBox"
    assert cbox["bounds_min"][2] == pytest.approx(0.2, abs=0.05)
    assert cbox["bounds_max"][2] == pytest.approx(1.8, abs=0.05)

    exp_cam = await blender_worker.call("splat.export_camera_path", {"camera": "SeedanceSourceCam"})
    assert exp_cam["frames_exported"] == 10
    assert "prime lens" in exp_cam["seedance_camera_prompt"]

    # 3. Verify FastAPI Phase 5 endpoints (`/telemetry`, `/trajectory`, `/crop_box`, `/camera_prompt`, `/render` blender)
    sess = await bootstrap_session(
        cwd=ws_dir,
        home=home_dir,
        overrides={"BLENDERBOT_PROVIDER": "gemini", "BLENDERBOT_GEMINI_API_KEY": "test-dummy"},
        injected_transport=blender_worker,
    )
    srv = await start_web_server(session=sess, host="127.0.0.1", port=0)
    headers = {"Authorization": f"Bearer {srv.token}"}
    try:
        async with aiohttp.ClientSession(headers=headers) as http:
            r_tel = await http.get(f"{srv.url}/api/splats/telemetry/{job_id}")
            assert r_tel.status == 200
            d_tel = await r_tel.json()
            assert len(d_tel["ticks"]) == 2
            assert len(d_tel["densify_events"]) == 1
            assert len(d_tel["snapshots"]) == 1
            assert d_tel["has_proxy_glb"] is True

            r_trj = await http.get(f"{srv.url}/api/splats/trajectory/{job_id}")
            assert r_trj.status == 200
            d_trj = await r_trj.json()
            assert d_trj["trajectory"]["total_frames"] == 10
            assert d_trj["trajectory"]["fps"] == 12.0

            r_cp = await http.post(f"{srv.url}/api/splats/camera_prompt", json={})
            assert r_cp.status == 200
            d_cp = await r_cp.json()
            assert "prime lens" in d_cp["seedance_camera_prompt"]

            r_ren = await http.post(
                f"{srv.url}/api/splats/render",
                json={"job_id": job_id, "mode": "blender", "engine": "BLENDER_WORKBENCH"},
            )
            assert r_ren.status == 200
            d_ren = await r_ren.json()
            assert d_ren["frames"] == 10
            assert (job_dir / "renders" / "blender_recreated.mp4").is_file()

            # 4. Verify FPS Camera Rail + Subject Track + Voxel Blockout + Spatial Ping -> `shot.usda` & Blender sync!
            tiny_png_b64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            )
            r_shot = await http.post(
                f"{srv.url}/api/shots/export",
                json={
                    "name": "rust_autorun_peek",
                    "splat_job_id": job_id,
                    "action": "sync_blender",
                    "duration_sec": 4.0,
                    "fps": 12.0,
                    "grounded_walk": True,
                    "track_subject": True,
                    "cam_nodes": [
                        {"position": [-2.0, -3.5, 1.65], "forward": [0.4, 0.9, -0.1], "lens_mm": 35},
                        {"position": [2.0, -3.5, 0.95], "forward": [-0.4, 0.9, 0.0], "lens_mm": 35},
                    ],
                    "subject_nodes": [
                        {"position": [-1.2, 0.5, 0.0]},
                        {"position": [1.4, 0.8, 0.0]},
                    ],
                    "voxels": [
                        {"center": [0.0, 1.5, 0.5], "size": [1.0, 1.0, 1.0]},
                    ],
                    "pings": [
                        {
                            "position": [0.0, 1.5, 1.0],
                            "camera_eye": [-2.0, -3.5, 1.65],
                            "note": "Build weathered blast door on this monument wall",
                            "image_b64": tiny_png_b64,
                        }
                    ],
                    "first_frame_b64": tiny_png_b64,
                    "last_frame_b64": tiny_png_b64,
                },
            )
            assert r_shot.status == 200
            d_shot = await r_shot.json()
            shot_id = d_shot["shot_id"]
            shot_dir = ws_dir / "assets" / "shots" / shot_id
            assert (shot_dir / "shot.usda").is_file()
            assert (shot_dir / "blockout.glb").is_file()
            assert (shot_dir / "brief.md").is_file()
            assert "subject moves" in d_shot["brief"]["optical_prompt"].lower()
            assert d_shot["blender_synced"]["camera"]["camera"] == "DirectorRailCam"

            # Prove `shot.usda` also imports cleanly via native Blender 5.3 OpenUSD importer!
            usd_imp = await blender_worker.call(
                "asset.import",
                {"filepath": str(shot_dir / "shot.usda"), "collection": "USD_Shot_Verify"},
            )
            assert usd_imp["count"] >= 2
            assert any("DirectorRailCam" in nm for nm in usd_imp["imported_objects"])
    finally:
        await srv.close()
        sess.web_server = None
        sess.transport = None
        await sess.close()