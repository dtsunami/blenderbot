"""Tests for fal.ai `meshy/v7.1/image-to-3d` integration & 3D Studio pane:
1. `FAL_KEY` / `FAL_API_KEY` env loading & `/keys fal` state without leaking secret values.
2. Exact input schema validation (`prepare_meshy_request`), pricing calculation, `meshy-n1`
   cross-field exclusions, rigging/animation constraints, and trusted fal origin URL guards.
3. Queue submit -> poll (`IN_QUEUE` -> `IN_PROGRESS` -> `COMPLETED`) -> artifact download,
   non-retry safety on ambiguous submit responses, and HTTP 403 `"Exhausted balance"` /
   `"User is locked"` remedies.
4. Pure-NumPy Binary glTF 2.0 (`.glb`) -> Blender Z-up WebGL2 triangle extractor, Web Studio
   `/api/meshy/*` endpoints, `panel-meshy-composer` 3D pane, `gen.meshy` `bl_ops`, and `/meshy` CLI.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from PIL import Image

from blenderbot.agent.executor import execute_batch
from blenderbot.agent.parser import OpCall
from blenderbot.config.env import load_config
from blenderbot.media_gen.meshy import (
    DEFAULT_MESHY_MODEL,
    FAL_CREDITS_URL,
    MeshyError,
    estimate_meshy_cost_usd,
    extract_glb_viewport_triangles,
    generate_meshy_3d,
    is_trusted_fal_url,
    prepare_meshy_request,
)
from blenderbot.server import start_web_server
from blenderbot.session.bootstrap import bootstrap_session
from blenderbot.splats.proxy_mesh import pack_glb_bytes
from blenderbot.ui.commands import handle_command


def _make_png_bytes(w: int = 64, h: int = 64, rgba: tuple[int, int, int, int] = (80, 180, 240, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), rgba).save(buf, format="PNG")
    return buf.getvalue()


def _make_sample_cube_glb_bytes() -> bytes:
    verts = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0],
            [-0.5, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (8, 1))
    colors = np.tile(np.array([[0.3, 0.75, 0.95, 1.0]], dtype=np.float32), (8, 1))
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
        ],
        dtype=np.uint32,
    )
    return pack_glb_bytes(verts, normals, colors, faces, mesh_name="MeshyTestCube")


def test_fal_key_loading_and_redaction(tmp_path: Path) -> None:
    home = tmp_path / "bb_home"
    home.mkdir()
    (home / ".env").write_text("FAL_API_KEY=fal_dummy_key_1234567890abcdef\nGEMINI_API_KEY=gem_123\n", encoding="utf-8")

    cfg = load_config(cwd=tmp_path, home=home, environ={})
    assert cfg.keys["fal"] == "fal_dummy_key_1234567890abcdef"
    assert cfg.key_origins["fal"]["name"] == "FAL_API_KEY"
    assert cfg.meshy_model == DEFAULT_MESHY_MODEL
    pub = cfg.to_public_dict()
    assert pub["keys"]["fal"] == "set"
    assert "fal_dummy_key" not in str(pub)


def test_meshy_schema_validation_pricing_and_origin_guard(tmp_path: Path) -> None:
    img_p = tmp_path / "prop.png"
    img_p.write_bytes(_make_png_bytes())

    assert estimate_meshy_cost_usd(should_texture=False) == pytest.approx(0.80)
    assert estimate_meshy_cost_usd(should_texture=True) == pytest.approx(1.20)
    assert estimate_meshy_cost_usd(should_texture=True, enable_rigging=True) == pytest.approx(1.40)
    assert estimate_meshy_cost_usd(should_texture=True, enable_rigging=True, enable_animation=True) == pytest.approx(1.52)

    payload, summary = prepare_meshy_request(
        image=img_p,
        topology="quad",
        target_polycount=45000,
        symmetry_mode="on",
        should_remesh=True,
        should_texture=True,
        texture_model="meshy-7",
        enable_pbr=True,
        pose_mode="a-pose",
        texture_prompt="Brushed titanium armor plates",
        enable_rigging=True,
        rigging_height_meters=1.82,
        skeleton_template="blender_rigify_basic_human",
        include_fingers=True,
        enable_animation=True,
        animation_action_id=92,
        geometry_resolution="4k",
    )
    assert payload["image_url"].startswith("data:image/png;base64,")
    assert payload["model_type"] == "standard"
    assert payload["topology"] == "quad"
    assert payload["target_polycount"] == 45000
    assert payload["texture_model"] == "meshy-7"
    assert payload["enable_pbr"] is True
    assert payload["pose_mode"] == "a-pose"
    assert payload["skeleton_template"] == "blender_rigify_basic_human"
    assert payload["animation_action_id"] == 92
    assert payload["geometry_resolution"] == "4k"
    assert summary["estimated_cost_usd"] == pytest.approx(1.52)

    # meshy-n1 cross-field exclusions from schema
    with pytest.raises(MeshyError, match="meshy-n1"):
        prepare_meshy_request(image=img_p, texture_model="meshy-n1", texture_prompt="not allowed")
    with pytest.raises(MeshyError, match="meshy-n1"):
        prepare_meshy_request(image=img_p, texture_model="meshy-n1", enable_pbr=True)
    with pytest.raises(MeshyError, match="meshy-n1"):
        prepare_meshy_request(image=img_p, texture_model="meshy-n1", topology="quad", should_remesh=True)

    # enable_animation requires enable_rigging
    with pytest.raises(MeshyError, match="enable_rigging"):
        prepare_meshy_request(image=img_p, enable_rigging=False, enable_animation=True)
    with pytest.raises(MeshyError, match="animation_action_id"):
        prepare_meshy_request(image=img_p, enable_rigging=True, enable_animation=True, animation_action_id=999)

    # Trusted fal origin checks
    assert is_trusted_fal_url("https://queue.fal.run/meshy/requests/abc/status")
    assert is_trusted_fal_url("https://v3b.fal.media/files/b/zebra/model.glb")
    assert not is_trusted_fal_url("https://evil.example.com/fal.run/status")
    assert not is_trusted_fal_url("http://queue.fal.run/meshy/requests/abc")


def test_glb_viewport_triangle_extractor_z_up_grounding(tmp_path: Path) -> None:
    glb_p = tmp_path / "sample.glb"
    glb_p.write_bytes(_make_sample_cube_glb_bytes())

    extracted = extract_glb_viewport_triangles(glb_p, place_on_ground=True)
    assert extracted["total_triangles"] == 8
    assert extracted["triangles"] == 8
    assert len(extracted["vertex_floats"]) == 8 * 3 * 10
    ab = extracted["world_aabb"]
    assert ab["min"][2] == pytest.approx(0.0, abs=1e-4)
    assert ab["max"][2] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.asyncio
async def test_meshy_fal_queue_billing_403_and_web_studio_pane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws_dir = tmp_path / "meshy_ws"
    home_dir = tmp_path / "meshy_home"
    ws_dir.mkdir()
    (ws_dir / "assets" / "inputs").mkdir(parents=True)
    inp_png = ws_dir / "assets" / "inputs" / "hero_knight.png"
    inp_png.write_bytes(_make_png_bytes(96, 96))

    glb_bytes = _make_sample_cube_glb_bytes()
    thumb_bytes = _make_png_bytes(80, 80, (250, 140, 60, 255))

    submit_calls = 0
    poll_calls = 0
    mode_flag = {"state": "ok"}

    async def handle_submit(req: web.Request) -> web.Response:
        nonlocal submit_calls
        submit_calls += 1
        assert req.headers.get("Authorization") == "Key test_fal_secret_999"
        body = await req.json()
        assert body["image_url"].startswith("data:image/png;base64,")
        if mode_flag["state"] == "exhausted":
            return web.json_response({"detail": "Exhausted balance"}, status=403)
        if mode_flag["state"] == "locked":
            return web.json_response({"detail": "User is locked: monthly budget limit reached"}, status=403)
        if mode_flag["state"] == "ambiguous":
            return web.Response(text="<html>bad gatewaynonjson</html>", status=200)

        base = f"http://127.0.0.1:{req.url.port}"
        rid = f"req-meshy-{submit_calls:03d}"
        return web.json_response(
            {
                "status": "IN_QUEUE",
                "request_id": rid,
                "status_url": f"{base}/meshy/requests/{rid}/status",
                "response_url": f"{base}/meshy/requests/{rid}",
                "cancel_url": f"{base}/meshy/requests/{rid}/cancel",
            }
        )

    async def handle_status(req: web.Request) -> web.Response:
        nonlocal poll_calls
        poll_calls += 1
        rid = req.match_info["rid"]
        if poll_calls == 1:
            return web.json_response(
                {
                    "status": "IN_PROGRESS",
                    "request_id": rid,
                    "logs": [{"message": "Remeshing & baking 2K texture..."}],
                }
            )
        return web.json_response({"status": "COMPLETED", "request_id": rid, "logs": []})

    async def handle_result(req: web.Request) -> web.Response:
        base = f"http://127.0.0.1:{req.url.port}"
        return web.json_response(
            {
                "model_glb": {
                    "content_type": "model/gltf-binary",
                    "file_size": len(glb_bytes),
                    "url": f"{base}/files/model.glb",
                    "file_name": "model.glb",
                },
                "thumbnail": {
                    "content_type": "image/png",
                    "file_size": len(thumb_bytes),
                    "url": f"{base}/files/preview.png",
                    "file_name": "preview.png",
                },
                "model_urls": {
                    "glb": {
                        "content_type": "model/gltf-binary",
                        "file_size": len(glb_bytes),
                        "url": f"{base}/files/model.glb",
                        "file_name": "model.glb",
                    }
                },
                "seed": 424242,
            }
        )

    async def handle_file(req: web.Request) -> web.Response:
        name = req.match_info["name"]
        if name == "model.glb":
            return web.Response(body=glb_bytes, content_type="model/gltf-binary")
        return web.Response(body=thumb_bytes, content_type="image/png")

    app = web.Application()
    app.router.add_post("/meshy/v7.1/image-to-3d", handle_submit)
    app.router.add_get("/meshy/requests/{rid}/status", handle_status)
    app.router.add_get("/meshy/requests/{rid}", handle_result)
    app.router.add_get("/files/{name}", handle_file)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    mock_fal_base = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("BLENDERBOT_FAL_QUEUE_URL", mock_fal_base)

    try:
        # 1. Verify 403 Exhausted balance & User locked remedies (zero automatic retries!)
        mode_flag["state"] = "exhausted"
        with pytest.raises(MeshyError) as exc_exh:
            await generate_meshy_3d(api_key="test_fal_secret_999", image=inp_png, poll_interval_sec=0.02)
        assert exc_exh.value.exhausted_balance is True
        assert FAL_CREDITS_URL in str(exc_exh.value)
        assert submit_calls == 1

        mode_flag["state"] = "locked"
        with pytest.raises(MeshyError) as exc_lck:
            await generate_meshy_3d(api_key="test_fal_secret_999", image=inp_png, poll_interval_sec=0.02)
        assert exc_lck.value.user_locked is True
        assert "monthly budget limit reached" in str(exc_lck.value)
        assert submit_calls == 2

        # Ambiguous submit response must NOT resubmit a second billable job
        mode_flag["state"] = "ambiguous"
        with pytest.raises(MeshyError, match="did NOT resubmit"):
            await generate_meshy_3d(api_key="test_fal_secret_999", image=inp_png, poll_interval_sec=0.02)
        assert submit_calls == 3

        # 2. Full Session + Web Studio Sidecar + `gen.meshy` + `/meshy` + `/api/meshy/*`
        mode_flag["state"] = "ok"
        sess = await bootstrap_session(
            cwd=ws_dir,
            home=home_dir,
            overrides={
                "BLENDERBOT_PROVIDER": "gemini",
                "BLENDERBOT_GEMINI_API_KEY": "test-gem",
                "FAL_KEY": "test_fal_secret_999",
            },
        )
        srv = await start_web_server(session=sess, host="127.0.0.1", port=0)
        headers = {"Authorization": f"Bearer {srv.token}"}

        try:
            # 2a. `gen.meshy` agent op
            outcome = await execute_batch(
                [
                    OpCall(
                        index=0,
                        op="gen.meshy",
                        params={
                            "image": "assets/inputs/hero_knight.png",
                            "topology": "triangle",
                            "target_polycount": 20000,
                            "enable_pbr": True,
                            "output_path": "assets/meshy/agent_knight/model.glb",
                        },
                    )
                ],
                sandbox=sess.sandbox,
                journal=sess.journal,
                cfg=sess.cfg,
            )
            assert outcome.status == "ok"
            assert (ws_dir / "assets" / "meshy" / "agent_knight" / "model.glb").is_file()
            assert (ws_dir / "assets" / "meshy" / "agent_knight" / "preview.png").is_file()
            assert len(outcome.attached_media) == 1

            # 2b. `/meshy list` CLI command
            cli_out: list[dict[str, Any]] = []
            c_res = await handle_command("/meshy list", session=sess, push=cli_out.append, exit_app=lambda: None)
            assert c_res["handled"] is True
            assert any("agent_knight" in str(ev.get("text", "")) for ev in cli_out)

            # 2c. Web Studio HTML pane + `/api/state` + `/api/meshy/generate` + `/api/meshy/triangles/{id}`
            async with aiohttp.ClientSession(headers=headers) as http:
                html_txt = await (await http.get(f"{srv.url}/")).text()
                assert "tab-mode-meshy" in html_txt
                assert "panel-meshy-composer" in html_txt

                st_init = await (await http.get(f"{srv.url}/api/state")).json()
                assert st_init["fal"]["configured"] is True
                assert "test_fal_secret_999" not in str(st_init)
                assert st_init["meshy_spec"]["model"] == DEFAULT_MESHY_MODEL

                r_gen = await http.post(
                    f"{srv.url}/api/meshy/generate",
                    json={
                        "image": "assets/inputs/hero_knight.png",
                        "topology": "quad",
                        "target_polycount": 25000,
                        "should_texture": True,
                        "auto_import": False,
                    },
                )
                assert r_gen.status == 202
                web_job_id = (await r_gen.json())["job"]["job_id"]

                for _ in range(40):
                    st_poll = await (await http.get(f"{srv.url}/api/state")).json()
                    found = next((m for m in st_poll["meshy_jobs"] if m["job_id"] == web_job_id), None)
                    if found and found["stage"] == "completed":
                        assert found["request_id"].startswith("req-meshy-")
                        assert found["cost_usd"] == pytest.approx(1.20)
                        break
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("Meshy web job did not complete in time")

                r_tris = await http.get(f"{srv.url}/api/meshy/triangles/{web_job_id}")
                assert r_tris.status == 200
                d_tris = await r_tris.json()
                assert d_tris["triangles"] == 8
                assert len(d_tris["vertex_floats"]) == 8 * 3 * 10
        finally:
            await srv.close()
            sess.web_server = None
            await sess.close()
    finally:
        await runner.cleanup()