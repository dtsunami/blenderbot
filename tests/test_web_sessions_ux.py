"""Tests for the Session-First Web Studio UX & Agent Session APIs:
1. `/api/sessions` (list active + past sessions with `.blend` & `first_prompt` metadata),
   `/api/sessions/{id}` (full transcript event parsing & Journal state),
   `/api/sessions/new` (fresh session in-place), and
   `/api/sessions/{id}/resume` (continue past conversation + restore Ledger & Journal undo stack).
2. `/api/session/settings` (live role/effort switch) and `/api/context/{id}/pin`.
3. `/api/workspace/media` `.blend` discovery + session cross-referencing.
4. HTML/JS structural integrity check (no duplicate DOM IDs, Session-first layout & Context-driven Inspector).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import aiohttp
import pytest

from blenderbot.server import start_web_server
from blenderbot.session.bootstrap import bootstrap_session
from blenderbot.session.log import SessionLog


def test_index_html_has_unique_element_ids() -> None:
    html = Path("src/blenderbot/ui/web/index.html").read_text(encoding="utf-8")
    ids = re.findall(r'\bid="([^"]+)"', html)
    dupes = {k: v for k, v in Counter(ids).items() if v > 1}
    assert not dupes, f"Duplicate element IDs in index.html: {dupes}"
    assert "theater-session-wrap" in ids
    assert "theater-blend-wrap" in ids
    assert "theater-gen-wrap" in ids
    assert "panel-session-inspector" in ids
    assert "panel-media-inspector" in ids
    assert "panel-fps-splat" in ids
    assert "panel-copilot-jobs" in ids
    # Clean 6-pill interactive status bar + popover + collapsed deep telemetry
    for pill_id in ("pill-session", "pill-blender", "pill-modal-ver", "btn-or-key", "btn-fal-key", "pill-server", "status-popover", "details-kdiag-deep"):
        assert pill_id in ids
    assert "splat-live-preview-wrap" not in ids  # duplicate sidebar train status card removed!


@pytest.mark.asyncio
async def test_web_sessions_list_resume_new_and_blend_discovery(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    home = tmp_path / "home"
    ws.mkdir()
    home.mkdir()

    # Create a dummy workspace .blend file
    blend_file = ws / "hero_robot.blend"
    blend_file.write_bytes(b"BLENDER-v503REND" + b"\x00" * 64)

    # Seed a past session in `home/sessions/` associated with `hero_robot.blend`
    past_log = SessionLog(
        home=home,
        cwd=ws,
        provider="gemini",
        model_id="gemini-2.5-pro",
    )
    past_id = past_log.id
    past_log.update_blend_association(blend_filepath=str(blend_file.resolve()), scene_name="RobotScene")
    past_log.record("user", {"text": "Build a articulated mech leg at the origin"})
    past_log.record("assistant", {"text": "Created MechHip and MechKnee.", "turn": 1})
    past_log.record("ops-result", {"status": "ok", "results": [{"op": "mesh.primitive"}], "errors": []})
    past_log.update_totals({"turns": 1, "ops": 1, "cost": 0.0125})

    sess = await bootstrap_session(
        cwd=ws,
        home=home,
        auto_connect_blender=False,
        overrides={
            "BLENDERBOT_PROVIDER": "gemini",
            "BLENDERBOT_GEMINI_API_KEY": "test-gem-key",
        },
    )
    srv = await start_web_server(session=sess, host="127.0.0.1", port=0)
    headers = {"Authorization": f"Bearer {srv.token}"}

    try:
        async with aiohttp.ClientSession(headers=headers) as http:
            # 1. List sessions: both active session and past session appear
            r_list = await http.get(f"{srv.url}/api/sessions")
            assert r_list.status == 200
            d_list = await r_list.json()
            assert d_list["active_id"] == sess.log.id
            ids = [s["id"] for s in d_list["sessions"]]
            assert sess.log.id in ids
            assert past_id in ids
            past_summary = next(s for s in d_list["sessions"] if s["id"] == past_id)
            assert past_summary["first_prompt"] == "Build a articulated mech leg at the origin"
            assert past_summary["blend_name"] == "hero_robot.blend"
            assert past_summary["has_blend_state"] is True

            # 2. Workspace media endpoint returns `.blend` files cross-referenced to `past_id`
            r_media = await http.get(f"{srv.url}/api/workspace/media")
            assert r_media.status == 200
            d_media = await r_media.json()
            assert len(d_media["blends"]) == 1
            assert d_media["blends"][0]["name"] == "hero_robot.blend"
            assert past_id in d_media["blends"][0]["session_ids"]

            # 3. Inspect past session transcript events
            r_det = await http.get(f"{srv.url}/api/sessions/{past_id}")
            assert r_det.status == 200
            d_det = await r_det.json()
            ev_types = [e["type"] for e in d_det["events"]]
            assert ev_types == ["user", "assistant", "ops"]

            # 4. Continue / resume past session in-place -> restores Ledger & accumulated stats
            r_res = await http.post(f"{srv.url}/api/sessions/{past_id}/resume", json={"open_blend": False})
            assert r_res.status == 200
            d_res = await r_res.json()
            resumed_active_id = d_res["session"]["id"]
            assert resumed_active_id == sess.log.id
            assert d_res["session"]["resumed_from"] == past_id
            assert d_res["session"]["turns"] == 1
            assert len(sess.ledger.segments) >= 2

            # 5. Update role via `/api/session/settings`
            r_set = await http.post(f"{srv.url}/api/session/settings", json={"role": "viewer", "effort": "low"})
            assert r_set.status == 200
            d_set = await r_set.json()
            assert d_set["role"] == "viewer"
            assert sess.cfg.role == "viewer"

            # 6. Start a brand-new session in-place -> clears Ledger & resets stats
            r_new = await http.post(f"{srv.url}/api/sessions/new", json={"reset_blender": False})
            assert r_new.status == 200
            d_new = await r_new.json()
            assert d_new["session"]["id"] != resumed_active_id
            assert d_new["session"]["turns"] == 0
            assert len(sess.ledger.segments) == 0

            # 7. Active disk-backed CLI/Agent 3DGS run (recent telemetry.jsonl without `done`) is detected as `train` by polling
            import json as _json

            run_dir = ws / "assets" / "splats" / "live_cli_train_01"
            run_dir.mkdir(parents=True)
            (run_dir / "job_source.json").write_text(_json.dumps({"config": {"train": {"iterations": 5000}}}), encoding="utf-8")
            (run_dir / "telemetry.jsonl").write_text(
                _json.dumps({"event": "telemetry", "job_id": "live_cli_train_01", "step": 1250, "total_steps": 5000, "psnr": 26.4, "loss": 0.041, "n_gaussians": 182000}) + "\n",
                encoding="utf-8",
            )
            r_sp = await http.get(f"{srv.url}/api/splats/jobs")
            assert r_sp.status == 200
            sp_jobs = (await r_sp.json())["jobs"]
            live_j = next(j for j in sp_jobs if j["job_id"] == "live_cli_train_01")
            assert live_j["stage"] == "train"
            assert live_j["step"] == 1250
            assert live_j["total_steps"] == 5000

            r_tel_poll = await http.get(f"{srv.url}/api/splats/telemetry/live_cli_train_01")
            assert r_tel_poll.status == 200
            d_tel_poll = await r_tel_poll.json()
            assert d_tel_poll["job"]["stage"] == "train"
            assert d_tel_poll["job"]["step"] == 1250
    finally:
        await srv.close()
        sess.web_server = None
        await sess.close()