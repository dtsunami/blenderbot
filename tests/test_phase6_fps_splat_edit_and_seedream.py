"""Phase 6 tests:
1. Pure-NumPy 3DGS `.ply` <-> `.splat` 1:1 alignment, Default Cube & Courtyard benchmark splat
   generator, and surgical Gaussian brush culling, 3D AABB cropping, smart floater filtering,
   SH-preserving tint/scale painting, Undo/Redo index mask commit, and `.orig.ply` restoration.
2. OpenRouter Seedream 5.5 Image API (`POST /api/v1/images`, `GET /api/v1/images/models`,
   `bytedance-seed/seedream-5-5`): payload validation (`1K`/`1.5K`/`2K`, aspect ratios,
   `transparent` -> `png`, `input_references`), `429` backoff retry, `gen.seedream` `bl_ops`
   execution + vision attachment, `/seedream` CLI command, and `/api/seedream/generate` GUI job flow.
3. Dedicated Full-Screen `/fps` Camera & 3DGS Splat Studio endpoints:
   - `GET /fps`, `/assets/fps.js`, `/assets/fps.css`, `/api/fps/default_splat`
   - Live Blender 5.3 Default Cube spawn, `scene.viewport_triangles` mesh streaming, and
     exact quaternion Camera Pose Push/Pull (`/api/fps/camera_sync` <-> `/api/fps/camera_pull`)
   - `.ply`/`.splat` drag-drop upload (`/api/splats/upload`), surgical edit + live Blender 5.3
     `PointCloud` sync (`/api/splats/edit/apply`), and `/api/splats/edit/restore/{id}`.
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pytest
import pytest_asyncio
from aiohttp import web
from PIL import Image

from blenderbot.agent.executor import execute_batch
from blenderbot.agent.parser import OpCall
from blenderbot.config.blender_paths import discover_blender
from blenderbot.media_gen.seedream import (
    DEFAULT_SEEDREAM_MODEL,
    SeedreamError,
    generate_seedream_image,
    prepare_seedream_request,
)
from blenderbot.server import start_web_server
from blenderbot.session.bootstrap import bootstrap_session
from blenderbot.splats.splat_edit import (
    apply_splat_edits,
    build_default_cube_splat,
    ensure_aligned_splat_file,
    parse_ply_gaussians,
    restore_original_ply,
    splat_bytes_to_ply_bytes,
)
from blenderbot.transport import HeadlessTransport
from blenderbot.ui.commands import handle_command


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


def _make_test_png_bytes(width: int = 96, height: int = 54, color: tuple[int, int, int, int] = (255, 123, 41, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def test_default_cube_splat_and_surgical_edits_roundtrip(tmp_path: Path) -> None:
    ply_b, splat_b, meta = build_default_cube_splat("courtyard")
    n0 = meta["n_gaussians"]
    assert n0 > 2000
    assert meta["floaters_seeded"] == 160
    assert len(splat_b) == n0 * 32

    jdir = tmp_path / "splat_job"
    jdir.mkdir()
    ply_p = jdir / "splat.ply"
    splat_p = jdir / "splat.splat"
    ply_p.write_bytes(ply_b)
    splat_p.write_bytes(splat_b)

    aligned = ensure_aligned_splat_file(ply_p, splat_p)
    assert aligned["n_ply"] == n0
    assert aligned["n_splat"] == n0
    assert aligned["rebuilt"] is False

    # Verify `.splat` -> `.ply` inverse conversion preserves count & coordinates
    rt_ply, rt_n = splat_bytes_to_ply_bytes(splat_b)
    assert rt_n == n0
    parsed_rt = parse_ply_gaussians(rt_ply)
    parsed_0 = parse_ply_gaussians(ply_b)
    assert np.allclose(parsed_rt["data"][:, :3], parsed_0["data"][:, :3], atol=1e-5)

    # 1. Smart Floater Filter + Sphere Brush Cull around the seeded floater cluster at (2.3, -1.7, 1.85)
    res1 = apply_splat_edits(
        ply_p,
        splat_p,
        {
            "ops": [
                {"op": "filter_prune", "min_opacity": 0.25, "max_anisotropy": 12.0},
                {"op": "brush_cull", "points": [[2.3, -1.7, 1.85]], "radius": 2.0},
                {"op": "brush_paint", "points": [[0.0, -1.0, 1.0]], "radius": 0.6, "color_rgb": [0.1, 0.9, 0.4], "blend": 0.8},
            ]
        },
        backup_orig=True,
    )
    assert res1["removed"] >= 160
    assert res1["n_after"] == n0 - res1["removed"]
    assert res1["has_orig"] is True
    assert (jdir / "splat.orig.ply").is_file()
    assert len(splat_p.read_bytes()) == res1["n_after"] * 32

    # 2. Crop Box (`aabb_keep`) keeping ONLY the centerpiece Default Cube `[-1.1..1.1, -1.1..1.1, -0.05..2.1]`
    res2 = apply_splat_edits(
        ply_p,
        splat_p,
        {
            "ops": [
                {"op": "aabb_keep", "bounds_min": [-1.1, -1.1, -0.05], "bounds_max": [1.1, 1.1, 2.1]},
            ]
        },
    )
    assert 400 < res2["n_after"] < res1["n_after"]

    # 3. Restore original unedited `.orig.ply` backup
    rest = restore_original_ply(jdir)
    assert rest["restored_gaussians"] == n0
    assert len(splat_p.read_bytes()) == n0 * 32


def test_prepare_seedream_request_validation(tmp_path: Path) -> None:
    ref_p = tmp_path / "ref_frame.png"
    ref_p.write_bytes(_make_test_png_bytes(64, 64))

    payload, summary = prepare_seedream_request(
        prompt="Editorial snowy storefront with warm neon",
        mode="cutout_prop",
        resolution="2K",
        aspect_ratio="16:9",
        output_format="jpeg",
        background="transparent",  # must auto-coerce output_format -> png!
        reference_images=[ref_p],
        seed=777,
    )
    assert payload["model"] == DEFAULT_SEEDREAM_MODEL
    assert payload["n"] == 1
    assert payload["resolution"] == "2K"
    assert payload["aspect_ratio"] == "16:9"
    assert payload["background"] == "transparent"
    assert payload["output_format"] == "png"
    assert payload["seed"] == 777
    assert len(payload["input_references"]) == 1
    assert payload["input_references"][0]["type"] == "image_url"
    assert payload["input_references"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert summary["estimated_cost_usd"] == pytest.approx(0.04)

    with pytest.raises(SeedreamError):
        prepare_seedream_request(prompt="   ")
    with pytest.raises(SeedreamError):
        prepare_seedream_request(prompt="test", resolution="8K")


@pytest.mark.asyncio(loop_scope="module")
async def test_phase6_fps_studio_splat_edit_and_seedream_end_to_end(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws_dir = tmp_path / "p6_ws"
    home_dir = tmp_path / "p6_home"
    ws_dir.mkdir(parents=True)

    # 1. Spin up mock OpenRouter `/api/v1/images` + `/api/v1/images/models` server (with 429 retry check)
    out_png_bytes = _make_test_png_bytes(160, 90)
    out_png_b64 = base64.b64encode(out_png_bytes).decode("ascii")
    image_posts: list[dict[str, Any]] = []

    async def handle_get_image_models(_req: web.Request) -> web.Response:
        return web.json_response({
            "data": [
                {
                    "id": "bytedance-seed/seedream-5-5",
                    "name": "ByteDance Seed: Seedream 5.5",
                    "pricing": {"image": "0.02"},
                }
            ]
        })

    async def handle_post_images(req: web.Request) -> web.Response:
        assert req.headers.get("Authorization") == "Bearer sk-or-v1-seedreamtestkey999"
        body = await req.json()
        image_posts.append(body)
        # First call returns 429 to verify automatic backoff retry!
        if len(image_posts) == 1:
            return web.json_response({"error": {"code": 429, "message": "rate limited"}}, status=429)
        return web.json_response({
            "created": 1748372400,
            "data": [{"b64_json": out_png_b64, "media_type": "image/png"}],
            "usage": {"prompt_tokens": 18, "completion_tokens": 4175, "total_tokens": 4193, "cost": 0.04},
        })

    mock_app = web.Application()
    mock_app.router.add_get("/api/v1/images/models", handle_get_image_models)
    mock_app.router.add_post("/api/v1/images", handle_post_images)
    runner = web.AppRunner(mock_app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    mock_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    mock_base = f"http://127.0.0.1:{mock_port}/api/v1"
    monkeypatch.setenv("BLENDERBOT_IMAGE_BASE_URL", mock_base)

    try:
        # 2. Direct `generate_seedream_image` (exercises 429 -> 200 retry!)
        direct_out = ws_dir / "assets" / "seedream" / "direct_test.png"
        sr_res = await generate_seedream_image(
            api_key="sk-or-v1-seedreamtestkey999",
            output_path=direct_out,
            prompt="Architectural twilight storefront",
            resolution="2K",
            aspect_ratio="16:9",
            base_url=mock_base,
        )
        assert len(image_posts) == 2
        assert direct_out.is_file()
        assert sr_res["width"] == 160 and sr_res["height"] == 90
        assert sr_res["cost_usd"] == pytest.approx(0.04)

        # 3. Bootstrap Session + Web Sidecar
        await blender_worker.call("scene.clear")
        sess = await bootstrap_session(
            cwd=ws_dir,
            home=home_dir,
            overrides={
                "BLENDERBOT_PROVIDER": "gemini",
                "BLENDERBOT_GEMINI_API_KEY": "test-dummy",
                "BLENDERBOT_OPENROUTER_API_KEY": "sk-or-v1-seedreamtestkey999",
            },
            injected_transport=blender_worker,
        )
        srv = await start_web_server(session=sess, host="127.0.0.1", port=0)
        headers = {"Authorization": f"Bearer {srv.token}"}

        try:
            # 3a. Verify `gen.seedream` agent `bl_ops` execution + vision attachment
            outcome = await execute_batch(
                [
                    OpCall(
                        index=0,
                        op="gen.seedream",
                        params={
                            "prompt": "Weathered sci-fi bulkhead texture",
                            "mode": "pbr_texture",
                            "resolution": "1K",
                            "aspect_ratio": "1:1",
                            "reference_images": ["assets/seedream/direct_test.png"],
                            "output_path": "assets/seedream/agent_tex.png",
                        },
                    )
                ],
                sandbox=sess.sandbox,
                journal=sess.journal,
                transport=blender_worker,
                cfg=sess.cfg,
            )
            assert outcome.status == "ok"
            assert (ws_dir / "assets" / "seedream" / "agent_tex.png").is_file()
            assert len(outcome.attached_media) == 1
            assert outcome.attached_media[0]["kind"] == "image"

            # 3b. Verify `/seedream` CLI command & `/fps` CLI command
            cli_events: list[dict[str, Any]] = []
            cmd_res = await handle_command(
                "/seedream Studio lit cybernetic helmet --res 1.5K --aspect 1:1 --out assets/seedream/cli_helmet.png",
                session=sess,
                push=cli_events.append,
                exit_app=lambda: None,
            )
            assert cmd_res["handled"] is True
            assert (ws_dir / "assets" / "seedream" / "cli_helmet.png").is_file()

            # 4. Verify Dedicated `/fps` Page, Default Cube Splat, Live Blender Sync & Splat Editing HTTP API
            async with aiohttp.ClientSession(headers=headers) as http:
                # Unified 3-Pane Studio HTML (`/` and `/fps`) + static JS/CSS + 3D Snapshot Capture
                r_fps_html = await http.get(f"{srv.url}/fps")
                assert r_fps_html.status == 200
                html_txt = await r_fps_html.text()
                assert "FPS Camera &amp; Splat Editor" in html_txt
                assert "hierarchy-tree" in html_txt
                assert "vsc-tree" in html_txt
                assert "hierarchy-tooltip" in html_txt
                assert "sash-left" in html_txt
                assert "sash-right" in html_txt
                assert "sash-agent" in html_txt
                assert "theater-session-wrap" in html_txt
                assert "panel-session-inspector" in html_txt
                assert "insp-section" in html_txt
                assert "insp-more" in html_txt
                assert "panel-seedance-composer" in html_txt
                assert "panel-seedream-composer" in html_txt
                assert "panel-fps-splat" in html_txt
                assert (await http.get(f"{srv.url}/assets/app.js")).status == 200
                assert (await http.get(f"{srv.url}/assets/app.css")).status == 200

                r_snap = await http.post(
                    f"{srv.url}/api/workspace/save_snapshot",
                    json={"data_url": f"data:image/png;base64,{out_png_b64}", "prefix": "view_first"},
                )
                assert r_snap.status == 200
                snap_item = (await r_snap.json())["item"]
                assert snap_item["kind"] == "image"
                assert (ws_dir / snap_item["path"]).is_file()

                # Default Cube Courtyard `.splat` stream
                r_def_sp = await http.get(f"{srv.url}/api/fps/default_splat?variant=courtyard&fmt=splat")
                assert r_def_sp.status == 200
                sp_raw = await r_def_sp.read()
                assert len(sp_raw) > 32 * 1000 and len(sp_raw) % 32 == 0

                # Spawn Default Cube in Blender 5.3 -> stream live viewport triangles -> Push/Pull FPS Camera!
                r_spawn = await http.post(f"{srv.url}/api/fps/spawn_default_cube_blender", json={})
                assert r_spawn.status == 200

                r_mesh = await http.get(f"{srv.url}/api/fps/blender_scene")
                assert r_mesh.status == 200
                d_mesh = await r_mesh.json()
                assert d_mesh["triangles"] >= 12
                assert "DefaultCube" in d_mesh["objects"]

                r_csync = await http.post(
                    f"{srv.url}/api/fps/camera_sync",
                    json={
                        "eye": [2.4, -4.8, 1.75],
                        "forward": [-0.42, 0.88, -0.15],
                        "roll_deg": 4.5,
                        "lens_mm": 28.0,
                        "camera_name": "FPSDirectorCam",
                    },
                )
                assert r_csync.status == 200

                r_cpull = await http.get(f"{srv.url}/api/fps/camera_pull?camera=FPSDirectorCam")
                assert r_cpull.status == 200
                d_cpull = await r_cpull.json()
                assert np.allclose(d_cpull["eye"], [2.4, -4.8, 1.75], atol=1e-3)
                assert d_cpull["lens_mm"] == pytest.approx(28.0, abs=0.1)

                # Upload `.splat` -> Cull Gaussians + Sync to Blender 5.3 -> Restore Original!
                r_up = await http.post(
                    f"{srv.url}/api/splats/upload?filename=test_yard.splat",
                    data=sp_raw,
                    headers={"X-Filename": "test_yard.splat"},
                )
                assert r_up.status == 200
                up_job = (await r_up.json())["job"]
                up_id = up_job["job_id"]
                orig_n = up_job["n_gaussians"]

                r_edit = await http.post(
                    f"{srv.url}/api/splats/edit/apply",
                    json={
                        "job_id": up_id,
                        "ops": [{"op": "aabb_cull", "bounds_min": [1.0, -3.0, 0.5], "bounds_max": [3.6, -0.5, 3.0]}],
                        "drop_indices": [0, 1, 2, 3, 4],
                        "authoritative_mask": False,
                        "sync_blender": True,
                        "render_mode": "pointcloud",
                    },
                )
                assert r_edit.status == 200
                d_edit = await r_edit.json()
                assert d_edit["n_after"] < orig_n
                assert d_edit["has_orig"] is True
                assert d_edit["blender_imported"]["points_imported"] == d_edit["n_after"]

                r_rest = await http.post(f"{srv.url}/api/splats/edit/restore/{up_id}", json={})
                assert r_rest.status == 200
                assert (await r_rest.json())["restored_gaussians"] == orig_n

                # Web GUI `/api/seedream/generate` async job completion
                r_sr_web = await http.post(
                    f"{srv.url}/api/seedream/generate",
                    json={
                        "prompt": "Golden hour courtyard flower shop",
                        "resolution": "1K",
                        "aspect_ratio": "16:9",
                        "reference_images": ["assets/seedream/direct_test.png"],
                    },
                )
                assert r_sr_web.status == 202
                sr_jid = (await r_sr_web.json())["job"]["local_id"]
                assert sr_jid.startswith("sr-")

                for _ in range(40):
                    st_now = await (await http.get(f"{srv.url}/api/state")).json()
                    matched = next((j for j in st_now["jobs"] if j["local_id"] == sr_jid), None)
                    if matched and matched["stage"] == "completed":
                        assert matched["cost_usd"] == pytest.approx(0.04)
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("Seedream web job did not complete in time")
        finally:
            await srv.close()
            sess.web_server = None
            sess.transport = None
            await sess.close()
    finally:
        await runner.cleanup()