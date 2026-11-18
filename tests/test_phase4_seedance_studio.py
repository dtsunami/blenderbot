"""Phase 4 (Option A) tests: OpenRouter Seedance 2.5 Video API (`POST /api/v1/videos`),
local video/image FFmpeg conditioning decomposition, `gen.seedance` execution + cost accounting,
and the FastAPI Seedance 2.5 & 3D Studio GUI Sidecar (`blenderbot.server`).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from blenderbot.agent.executor import execute_batch
from blenderbot.agent.parser import OpCall
from blenderbot.config.blender_paths import discover_blender
from blenderbot.media_gen.seedance import prepare_seedance_request, snap_seedance_size
from blenderbot.server import start_web_server
from blenderbot.session.bootstrap import bootstrap_session
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


def test_snap_seedance_size() -> None:
    assert snap_seedance_size(1920, 1080) == "1920x1080"
    assert snap_seedance_size(1280, 720) == "1280x720"
    assert snap_seedance_size(1080, 1920) == "1080x1920"
    assert snap_seedance_size(512, 512) == "640x640"


@pytest.mark.asyncio(loop_scope="module")
async def test_seedance_video_conditioning_and_gui_sidecar_end_to_end(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify:
    1. Blender playblast `.mp4` -> `prepare_seedance_request` decomposes `first_frame`, `last_frame`,
       and interior `input_references` `data:image/...;base64,...` URLs matching OpenRouter `/api/v1/videos`.
    2. `gen.seedance` + FastAPI GUI Sidecar (`/api/workspace/upload`, `/api/blender/capture`,
       `/api/seedance/generate`, `/api/events`, `/api/media/file`) run end-to-end against a spec-accurate
       async `202 -> pending -> in_progress -> completed -> /content` OpenRouter mock server.
    """
    work_dir = tmp_path / "studio_ws"
    home_dir = tmp_path / "studio_home"
    work_dir.mkdir(parents=True)

    # 1. Build a quick animated Blender scene and render a real source MP4 playblast!
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "scene.setup",
        {"fps": 12, "frame_start": 1, "frame_end": 12, "resolution": [640, 360]},
    )
    await blender_worker.call(
        "mesh.primitive",
        {"name": "HeroCube", "shape": "cube", "size": 1.0, "location": [0, 0, 0.5], "color": [0.9, 0.4, 0.1]},
    )
    await blender_worker.call(
        "anim.keyframe",
        {
            "object": "HeroCube",
            "property": "rotation_euler",
            "index": 2,
            "keyframes": [[1, 0.0], [12, 1.57]],
        },
    )
    await blender_worker.call(
        "light.rig", {"preset": "three_point", "target": "HeroCube"}
    )
    await blender_worker.call(
        "camera.upsert",
        {"name": "HeroCam", "location": [2.8, -2.8, 1.8], "look_at": "HeroCube", "lens_mm": 50},
    )
    src_mp4 = work_dir / "renders" / "blockout.mp4"
    await blender_worker.call(
        "render.playblast", {"output_path": str(src_mp4), "engine": "BLENDER_WORKBENCH"}
    )
    assert src_mp4.is_file() and src_mp4.stat().st_size > 1000
    sample_mp4_bytes = src_mp4.read_bytes()

    # 2. Verify `prepare_seedance_request` decomposes local MP4 into `frame_images` + `input_references`
    req_payload, summary = prepare_seedance_request(
        prompt="Photoreal brushed titanium crate in misty hangar",
        mode="orbit_plate",
        video_path=src_mp4,
        video_mid_frames=2,
    )
    assert req_payload["model"] == "bytedance/seedance-2.5"
    assert "multi-view parallax" in req_payload["prompt"]
    assert req_payload["duration"] >= 4
    assert len(req_payload["frame_images"]) == 2
    assert req_payload["frame_images"][0]["frame_type"] == "first_frame"
    assert req_payload["frame_images"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert req_payload["frame_images"][1]["frame_type"] == "last_frame"
    assert len(req_payload["input_references"]) == 2
    assert summary["first_frame"] is True and summary["last_frame"] is True

    # 3. Spin up spec-accurate mock OpenRouter `/api/v1/videos` server
    submitted_bodies: list[dict[str, Any]] = []
    poll_counts: dict[str, int] = {}

    async def handle_post_videos(request: web.Request) -> web.Response:
        assert request.headers.get("Authorization") == "Bearer sk-or-v1-testseedancekey123456"
        body = await request.json()
        submitted_bodies.append(body)
        jid = f"job-test-{len(submitted_bodies)}"
        poll_counts[jid] = 0
        return web.json_response(
            {
                "id": jid,
                "generation_id": f"gen-{len(submitted_bodies)}",
                "polling_url": f"/api/v1/videos/{jid}",
                "status": "pending",
            },
            status=202,
        )

    async def handle_get_job(request: web.Request) -> web.Response:
        assert request.headers.get("Authorization") == "Bearer sk-or-v1-testseedancekey123456"
        jid = request.match_info["jid"]
        poll_counts[jid] = poll_counts.get(jid, 0) + 1
        if poll_counts[jid] == 1:
            return web.json_response({"id": jid, "status": "in_progress"})
        return web.json_response(
            {
                "id": jid,
                "status": "completed",
                "unsigned_urls": [f"http://127.0.0.1:{mock_port}/cdn/{jid}.mp4"],
                "usage": {"cost": 0.514},
            }
        )

    async def handle_get_content(request: web.Request) -> web.Response:
        assert request.headers.get("Authorization") == "Bearer sk-or-v1-testseedancekey123456"
        return web.Response(body=sample_mp4_bytes, content_type="video/mp4")

    mock_app = web.Application()
    mock_app.router.add_post("/api/v1/videos", handle_post_videos)
    mock_app.router.add_get("/api/v1/videos/{jid}", handle_get_job)
    mock_app.router.add_get("/api/v1/videos/{jid}/content", handle_get_content)

    runner = web.AppRunner(mock_app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    mock_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    mock_base = f"http://127.0.0.1:{mock_port}/api/v1"

    monkeypatch.setenv("BLENDERBOT_VIDEO_BASE_URL", mock_base)
    monkeypatch.setenv("BLENDERBOT_VIDEO_POLL_SEC", "0.05")

    session = await bootstrap_session(
        cwd=work_dir,
        home=home_dir,
        overrides={
            "BLENDERBOT_OPENROUTER_API_KEY": "sk-or-v1-testseedancekey123456",
            "BLENDERBOT_GEMINI_API_KEY": "fake-gemini-key",
            "BLENDERBOT_PROVIDER": "gemini",
        },
        injected_transport=blender_worker,
    )
    sidecar = await start_web_server(session=session, host="127.0.0.1", port=0)

    try:
        # 4. Test `gen.seedance` via `execute_batch` (agent op path)
        batch_out = await execute_batch(
            [
                OpCall(
                    index=0,
                    op="gen.seedance",
                    params={
                        "prompt": "Hangar hero crate reveal",
                        "video_path": "renders/blockout.mp4",
                        "output_path": "assets/seedance/agent_shot.mp4",
                        "duration": 5,
                    },
                )
            ],
            sandbox=session.sandbox,
            journal=session.journal,
            transport=blender_worker,
            cfg=session.cfg,
        )
        assert batch_out.status == "ok", batch_out.errors
        r0 = batch_out.results[0]["result"]
        assert r0["cost_usd"] == 0.514
        assert (work_dir / "assets" / "seedance" / "agent_shot.mp4").is_file()

        # 5. Test FastAPI GUI Sidecar HTTP endpoints (auth, upload, Blender capture, Seedance job, media serving)
        async with aiohttp.ClientSession() as http:
            # Index HTML loads without auth, API requires Bearer token
            r_idx = await http.get(f"{sidecar.url}/")
            assert r_idx.status == 200
            html_txt = await r_idx.text()
            assert "Seedance 2.5 Composer" in html_txt

            r_unauth = await http.get(f"{sidecar.url}/api/state")
            assert r_unauth.status == 401

            auth_hdr = {"Authorization": f"Bearer {sidecar.token}"}
            r_state = await http.get(f"{sidecar.url}/api/state", headers=auth_hdr)
            assert r_state.status == 200
            st_json = await r_state.json()
            assert st_json["openrouter"]["configured"] is True
            assert st_json["blender"]["connected"] is True
            assert "ledger" in st_json and "prompt_history" in st_json

            # Capture a still PNG directly from Blender 5.3 via GUI endpoint
            r_cap = await http.post(
                f"{sidecar.url}/api/blender/capture",
                headers=auth_hdr,
                json={"kind": "still", "engine": "BLENDER_WORKBENCH"},
            )
            assert r_cap.status == 200
            cap_item = (await r_cap.json())["item"]
            assert cap_item["kind"] == "image"

            # Upload a video file via raw binary upload endpoint
            r_up = await http.post(
                f"{sidecar.url}/api/workspace/upload?filename=user_clip.mp4",
                headers=auth_hdr,
                data=sample_mp4_bytes,
            )
            assert r_up.status == 200
            up_item = (await r_up.json())["item"]
            assert up_item["path"] == "assets/inputs/user_clip.mp4"

            completed_bus_events: list[dict[str, Any]] = []
            session.bus.subscribe(
                lambda ev: completed_bus_events.append(ev["data"])
                if ev["type"] == "seedance-job" and (ev["data"] or {}).get("stage") == "completed"
                else None
            )

            # Submit Seedance 2.5 job from GUI sidecar with uploaded video + captured Blender frame!
            r_gen = await http.post(
                f"{sidecar.url}/api/seedance/generate",
                headers=auth_hdr,
                json={
                    "prompt": "Cyberpunk rain reflections on metallic crate",
                    "mode": "orbit_plate",
                    "video_path": up_item["path"],
                    "video_path_2": "renders/blockout.mp4",
                    "dual_video_mode": "bridge_v1_to_v2",
                    "duration": 5,
                    "size": "1280x720",
                    "output_path": "assets/seedance/gui_shot.mp4",
                    "attach_to_context": False,
                },
            )
            assert r_gen.status == 202
            job_id = (await r_gen.json())["job"]["local_id"]

            # Wait for background job to finish polling & downloading
            for _ in range(60):
                st_poll = await (
                    await http.get(f"{sidecar.url}/api/state", headers=auth_hdr)
                ).json()
                matching = [j for j in st_poll["jobs"] if j["local_id"] == job_id]
                if matching and matching[0]["stage"] == "completed":
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("GUI Seedance job did not complete in time")

            assert (work_dir / "assets" / "seedance" / "gui_shot.mp4").is_file()
            assert session.agent.stats["cost"] >= 0.514
            # Every `completed` SSE event must already carry a populated `media_url`
            assert len(completed_bus_events) == 1
            assert completed_bus_events[0]["media_url"] is not None
            st_after = await (await http.get(f"{sidecar.url}/api/state", headers=auth_hdr)).json()
            assert "Cyberpunk rain reflections on metallic crate" in st_after["prompt_history"]

            # Verify `/api/media/file` streams inline (not attachment) & `/api/media/poster` serves JPEG
            r_vid = await http.get(
                f"{sidecar.url}/api/media/file?path=assets/seedance/gui_shot.mp4&token={sidecar.token}",
                headers={"Connection": "close"},
            )
            assert r_vid.status == 200
            assert "attachment" not in (r_vid.headers.get("Content-Disposition") or "")
            assert r_vid.headers.get("Accept-Ranges") == "bytes"
            assert len(await r_vid.read()) == len(sample_mp4_bytes)

            r_poster = await http.get(
                f"{sidecar.url}/api/media/poster?path=assets/seedance/gui_shot.mp4&token={sidecar.token}",
                headers={"Connection": "close"},
            )
            assert r_poster.status == 200
            assert r_poster.headers.get("Content-Type") == "image/jpeg"

            # Verify 3DGS Splat Studio endpoints (`/api/splats/jobs`, `/api/splats/import`)
            import json as _json
            import struct as _struct

            sp_dir = work_dir / "assets" / "splats" / "gui_splat_01"
            sp_dir.mkdir(parents=True, exist_ok=True)
            ply_hdr = (
                "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                "property float opacity\n"
                "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
                "end_header\n"
            ).encode("ascii")
            ply_body = b"".join(
                _struct.pack("<14f", float(k), 0.0, 0.3, 0.2, 0.4, 0.1, 2.5, -4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 0.0)
                for k in range(2)
            )
            (sp_dir / "splat.ply").write_bytes(ply_hdr + ply_body)
            (sp_dir / "splat.splat").write_bytes(b"\x00" * 64)
            (sp_dir / "telemetry.jsonl").write_text(
                _json.dumps({"event": "telemetry", "step": 2000, "psnr": 28.9, "n_gaussians": 2}) + "\n"
                + _json.dumps({"event": "done", "step": 2000, "n_gaussians": 2, "cost_usd": 0.025, "wall_s": 14.2}) + "\n",
                encoding="utf-8",
            )

            r_sp = await http.get(f"{sidecar.url}/api/splats/jobs", headers=auth_hdr)
            assert r_sp.status == 200
            sp_jobs = (await r_sp.json())["jobs"]
            assert any(j["job_id"] == "gui_splat_01" and j["psnr"] == 28.9 for j in sp_jobs)

            r_imp = await http.post(
                f"{sidecar.url}/api/splats/import",
                headers=auth_hdr,
                json={"job_id": "gui_splat_01", "render_mode": "ellipsoids"},
            )
            assert r_imp.status == 200
            imp_data = (await r_imp.json())["imported"]
            assert imp_data["points_imported"] == 2
    finally:
        await sidecar.close()
        session.web_server = None
        session.transport = None
        await session.close()
        await runner.cleanup()