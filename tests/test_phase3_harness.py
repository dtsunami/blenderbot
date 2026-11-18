"""Phase 3 tests: Multi-ID Slotted Actions ('Project Baklava'), NLA animation layers,
`bl_ops` parser, `.blend` + file undo/redo journal, agent loop (with multimodal vision
attachment + superseded failed-op compression), stream watchdog, flight recorder,
slash commands, and CLI.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from blenderbot.agent.loop import EVENTS
from blenderbot.agent.parser import extract_turn
from blenderbot.cli import UsageError, main, parse_args
from blenderbot.config.blender_paths import discover_blender
from blenderbot.providers.base import ProviderError, StreamEvent
from blenderbot.providers.gemini import LEAKED_THOUGHT_PREFIX_RE, GeminiThoughtSanitizer
from blenderbot.providers.watchdog import iterate_with_idle_timeout
from blenderbot.session.bootstrap import bootstrap_session
from blenderbot.session.flight import read_breadcrumbs
from blenderbot.transport import HeadlessTransport
from blenderbot.ui.commands import command_list, handle_command


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def blender_worker():
    inst = discover_blender()
    assert inst.ready, f"Expected Blender 5.3+ installed, got: {inst}"
    worker = HeadlessTransport()
    await worker.start()
    try:
        yield worker
    finally:
        await worker.close()


def test_parser_bl_ops_and_verbatim_payloads() -> None:
    turn_txt = (
        "Building custom bmesh geo.\n\n"
        "```bl_ops\n"
        '{"ops": [\n'
        '  {"op": "bpy.tx", "description": "test", "code": "@geo"},\n'
        '  {"op": "fs.write", "path": "notes.txt", "text": "@note"}\n'
        "]}\n"
        "--blenderbot:geo--\n"
        "x = 1 + 2\n"
        "result = {'sum': x}\n"
        "--blenderbot:end--\n"
        "--blenderbot:note--\n"
        "hello world\n"
        "--blenderbot:end--\n"
        "```\n\n"
        "```python\n"
        "print('artifact')\n"
        "```"
    )
    ext = extract_turn(turn_txt)
    assert len(ext.ops) == 2
    assert ext.ops[0].op == "bpy.tx"
    assert "result = {'sum': x}" in ext.ops[0].params["code"]
    assert ext.ops[1].params["text"] == "hello world"
    assert len(ext.artifacts) == 1
    assert ext.artifacts[0].lang == "python"


def test_gemini_leaked_thought_sanitizer_and_prefix_regex() -> None:
    assert LEAKED_THOUGHT_PREFIX_RE.sub("", "sthought\n```bl_ops\n", count=1) == "```bl_ops\n"
    assert LEAKED_THOUGHT_PREFIX_RE.sub("", " düşünceler\nHello", count=1) == "Hello"

    # 1. Exact Turn 4 smoke-test pattern: split `sthought` chunk + leaked monologue + stripped `<ctrl95>` glue
    san = GeminiThoughtSanitizer()
    evs: list[StreamEvent] = []
    for ch in (
        "s",
        "thought",
        "\nThe crate looks clean, grounded, well-framed. Ready to report concisely.",
        "Built, audited, and rendered the grounded sci-fi crate scene:\n- Done.",
    ):
        evs.extend(san.feed(ch, is_thought=False))
    evs.extend(san.flush())

    assert any(e.t == "diagnostic" and e.reason == "thought_leak" for e in evs)
    think_txt = "".join(e.delta for e in evs if e.t == "thinking")
    visible_txt = "".join(e.delta for e in evs if e.t == "text")
    assert "Ready to report concisely." in think_txt
    assert "sthought" not in visible_txt
    assert visible_txt.startswith("Built, audited, and rendered")

    # 2. Split `sthought` followed immediately by ```bl_ops fence
    san2 = GeminiThoughtSanitizer()
    evs2 = [
        *san2.feed("sthought", is_thought=False),
        *san2.feed('\n```bl_ops\n{"ops": []}\n```', is_thought=False),
        *san2.flush(),
    ]
    assert any(e.t == "diagnostic" and e.reason == "thought_leak" for e in evs2)
    assert "".join(e.delta for e in evs2 if e.t == "text").lstrip().startswith("```bl_ops")

    # 3. Empty thought (` düşünceler\nHello`) flushes visible text cleanly at stream end
    san3 = GeminiThoughtSanitizer()
    evs3 = [*san3.feed(" düşünceler\nHello", is_thought=False), *san3.flush()]
    assert "".join(e.delta for e in evs3 if e.t == "text") == "Hello"


@pytest.mark.asyncio
async def test_watchdog_idle_timeout_and_instant_cancel() -> None:
    async def _stalled_gen() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(5.0)
        yield 2

    seen: list[int] = []
    with pytest.raises(ProviderError) as exc_info:
        async for item in iterate_with_idle_timeout(
            _stalled_gen(), provider="gemini", first_chunk_sec=1.0, idle_sec=0.15
        ):
            seen.append(item)
    assert seen == [1]
    assert exc_info.value.status == 408
    assert exc_info.value.retryable is True

    # Instant cancel race wakes immediately without waiting 5s
    cancel_ev = asyncio.Event()

    async def _trigger_cancel() -> None:
        await asyncio.sleep(0.05)
        cancel_ev.set()

    asyncio.create_task(_trigger_cancel())
    seen_cancel: list[int] = []
    async for item in iterate_with_idle_timeout(
        _stalled_gen(),
        provider="gemini",
        cancel_event=cancel_ev,
        first_chunk_sec=5.0,
        idle_sec=5.0,
    ):
        seen_cancel.append(item)
    assert seen_cancel == [1]


@pytest.mark.asyncio(loop_scope="module")
async def test_baklava_multi_id_slotted_actions_and_nla_layering(
    blender_worker: HeadlessTransport,
) -> None:
    """Verify Blender 5.3 Multi-ID shared Action Slots ('Project Baklava') + `anim.nla_layer`."""
    await blender_worker.call("scene.clear")
    await blender_worker.call("scene.setup", {"fps": 24, "frame_start": 1, "frame_end": 24})
    await blender_worker.call(
        "mesh.primitive", {"name": "BoxA", "shape": "cube", "size": 1.0, "location": [0, 0, 0.5]}
    )
    await blender_worker.call(
        "mesh.primitive", {"name": "BoxB", "shape": "cube", "size": 1.0, "location": [2, 0, 0.5]}
    )
    await blender_worker.call(
        "camera.upsert", {"name": "ShotCam", "location": [0, -6, 2], "lens_mm": 35}
    )

    # Author BoxA, BoxB, and ShotCam focal length into ONE shared Slotted Action `Shot_01`!
    kf_res = await blender_worker.call(
        "anim.keyframe",
        {
            "action": "Shot_01",
            "tracks": [
                {
                    "object": "BoxA",
                    "property": "location",
                    "index": 2,
                    "keyframes": [[1, 0.5], [25, 2.5]],
                    "interpolation": "LINEAR",
                },
                {
                    "object": "BoxB",
                    "property": "location",
                    "index": 2,
                    "keyframes": [[1, 0.5], [25, 4.5]],
                    "interpolation": "LINEAR",
                },
                {
                    "object": "ShotCam",
                    "property": "lens",
                    "keyframes": [[1, 35.0], [25, 85.0]],
                    "interpolation": "LINEAR",
                },
            ],
        },
    )
    assert kf_res["count"] == 3

    # Verify via `bpy.tx` that `Shot_01` holds 3 distinct Action Slots (2 OBJECT + 1 CAMERA)
    verify = await blender_worker.call(
        "bpy.tx",
        {
            "description": "Verify Multi-ID Action Slots",
            "code": (
                "act = bpy.data.actions['Shot_01']\n"
                "scene.frame_set(13)\n"
                "bpy.context.view_layer.update()\n"
                "result = {\n"
                "  'slot_count': len(act.slots),\n"
                "  'box_a_z': round(bpy.data.objects['BoxA'].location.z, 3),\n"
                "  'box_b_z': round(bpy.data.objects['BoxB'].location.z, 3),\n"
                "  'cam_lens': round(bpy.data.objects['ShotCam'].data.lens, 3),\n"
                "}\n"
            ),
        },
    )
    r = verify["result"]
    assert r["slot_count"] == 3
    assert abs(r["box_a_z"] - 1.5) < 0.05
    assert abs(r["box_b_z"] - 2.5) < 0.05
    assert abs(r["cam_lens"] - 60.0) < 0.2

    # Push BoxA's base animation onto NLA Track 1 (REPLACE), then author an ADD override layer on Track 2!
    nla_base = await blender_worker.call(
        "anim.nla_layer",
        {"object": "BoxA", "track": "BaseLayer", "blend_type": "REPLACE"},
    )
    assert nla_base["blend_type"] == "REPLACE"

    await blender_worker.call(
        "anim.keyframe",
        {
            "object": "BoxA",
            "action": "BoostOffset",
            "property": "location",
            "index": 2,
            "keyframes": [[1, 0.0], [25, 1.0]],
            "interpolation": "LINEAR",
        },
    )
    nla_add = await blender_worker.call(
        "anim.nla_layer",
        {
            "object": "BoxA",
            "track": "AdditiveAdjust",
            "blend_type": "ADD",
            "influence": 0.5,
        },
    )
    assert nla_add["blend_type"] == "ADD"
    assert nla_add["total_nla_tracks"] == 2

    # At frame 25: Base = 2.5 + 0.5 * Additive(1.0) = 3.0!
    eval_nla = await blender_worker.call(
        "bpy.tx",
        {
            "code": (
                "scene.frame_set(25)\n"
                "deps = bpy.context.evaluated_depsgraph_get()\n"
                "ev = bpy.data.objects['BoxA'].evaluated_get(deps)\n"
                "result = {'z25': round(float(ev.matrix_world.translation.z), 3)}\n"
            )
        },
    )
    assert abs(eval_nla["result"]["z25"] - 3.0) < 0.05


@pytest.mark.asyncio(loop_scope="module")
async def test_full_agent_loop_multimodal_undo_compression_and_commands(
    blender_worker: HeadlessTransport, tmp_path: Path
) -> None:
    """End-to-end test of `bootstrap_session`, `Agent.run`, `.blend` `/undo`+`/redo`,
    vision render auto-attachment, superseded failed-op compression, and `/flight`.
    """
    home_dir = tmp_path / "bb_home"
    work_dir = tmp_path / "workspace"
    work_dir.mkdir(parents=True)
    await blender_worker.call("scene.clear")

    scripted_turns = [
        # Turn 1: intentional bad op to verify `plan-failed` + superseded compression!
        '```bl_ops\n{"ops": [{"op": "nonexistent.op"}]}\n```',
        # Turn 2: real modeling + camera + viewport render (triggers checkpoint + vision attachment)
        (
            "```bl_ops\n"
            '{"ops": [\n'
            '  {"op": "mesh.primitive", "name": "Crate", "shape": "cube", "size": 1.2, "location": [0, 0, 0.6]},\n'
            '  {"op": "camera.upsert", "name": "CheckCam", "location": [3, -3, 2.2], "target": [0, 0, 0.6]},\n'
            '  {"op": "viewport.render_still", "output_path": "renders/crate_check.png"}\n'
            "]}\n"
            "```"
        ),
        # Turn 3: final conclusion after inspecting vision render
        "Built `Crate` (1.2m cube resting on Z=0) and verified framing in `renders/crate_check.png`.",
    ]
    call_idx = 0

    class FakeVisionProvider:
        name = "gemini"
        model = "gemini-4-carbon"

        async def send(
            self, *, system: str, messages: list[dict[str, Any]], cancel_event: Any = None
        ) -> AsyncIterator[StreamEvent]:
            nonlocal call_idx
            _ = (system, cancel_event)
            # On Turn 2+, `[Live Scene State]` must be ephemerally attached on messages[-1]
            last_parts = messages[-1]["content"]
            assert any(
                p.get("type") == "text" and "[Live Scene State" in p.get("text", "")
                for p in last_parts
            )
            reply = scripted_turns[call_idx]
            call_idx += 1
            yield StreamEvent(t="thinking", delta="Planning scene step...\n")
            yield StreamEvent(t="text", delta=reply)
            yield StreamEvent(
                t="usage", in_fresh=500, out=120, cache_read=1000, cache_write=0, context_in=1500
            )
            yield StreamEvent(t="stop", reason="end_turn")

        async def count_tokens(self, *, system: str, messages: list[dict[str, Any]]) -> int | None:
            _ = (system, messages)
            return 1500

    session = await bootstrap_session(
        cwd=work_dir,
        home=home_dir,
        overrides={"BLENDERBOT_GEMINI_API_KEY": "fake-key", "BLENDERBOT_PROVIDER": "gemini"},
        make_provider_fn=lambda cfg, effort_override=None: FakeVisionProvider(),
        injected_transport=blender_worker,
    )
    try:
        # Subscribe to every event in EVENTS so unretired_events stays 0
        for ev_name in EVENTS:
            session.agent.on(ev_name, lambda _p: None)

        out = await session.agent.run("Build a 1.2m crate and render a check still")
        assert out["status"] == "done"
        assert "Crate" in out["text"]
        assert (work_dir / "renders" / "crate_check.png").is_file()

        # Verify vision media segment was automatically attached into Ledger
        media_segs = [s for s in session.ledger.segments if s.type == "media"]
        assert len(media_segs) == 1
        assert "crate_check.png" in media_segs[0].label

        # Verify Turn 1's rejected batch was compressed into a 1-line superseded stub!
        superseded = [s for s in session.ledger.segments if "superseded" in s.label]
        assert len(superseded) == 1
        assert "superseded bl_ops attempt omitted" in superseded[0].part["text"]

        # Verify `.blend` /undo removes `Crate`, and `/redo` restores `Crate`!
        pushed: list[dict[str, Any]] = []
        await handle_command("/undo", session=session, push=pushed.append, exit_app=lambda: None)
        st_after_undo = await blender_worker.call("scene.passport")
        assert "Crate" not in {o["name"] for o in st_after_undo["objects"]}

        await handle_command("/redo", session=session, push=pushed.append, exit_app=lambda: None)
        st_after_redo = await blender_worker.call("scene.passport")
        assert "Crate" in {o["name"] for o in st_after_redo["objects"]}

        # Verify `/flight` breadcrumbs recorded phase transitions cleanly
        crumbs = read_breadcrumbs(session.log.dir)
        phases = {c.get("phase") for c in crumbs if c.get("kind") == "phase"}
        assert {"llm:await-first-byte", "llm:streaming", "ops:execute", "idle"}.issubset(phases)
    finally:
        # Don't close shared module-scoped `blender_worker` inside session.close()
        session.transport = None
        await session.close()


def test_cli_parse_args_and_command_catalog() -> None:
    cmds = command_list()
    names = {c["name"] for c in cmds}
    assert {
        "sh",
        "/blender",
        "/render",
        "/sheet",
        "/turntable",
        "/audit",
        "/undo",
        "/redo",
        "/role",
        "/flight",
        "/tps",
    }.issubset(names)

    parsed = parse_args(["-p", "--role", "builder", "--model", "gemini-4-carbon", "make a cube"])
    assert parsed.print_mode is True
    assert parsed.overrides["BLENDERBOT_ROLE"] == "builder"
    assert parsed.input_text == "make a cube"

    with pytest.raises(UsageError):
        parse_args(["--unknown-flag"])

    assert main(["--version"]) == 0


@pytest.mark.asyncio(loop_scope="module")
async def test_audit_bbox_frustum_margin_camera_target_alias_and_tx_fcurve_helpers(
    blender_worker: HeadlessTransport,
) -> None:
    """Verify `camera.upsert` `target` alias, `bpy.tx` `ensure_channelbag_fcurves` injection,
    and `scene.audit` full-AABB `frustum_margin` / `bbox_in_frame_pct` detection.
    """
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "mesh.primitive", {"name": "Crate", "shape": "cube", "size": 1.2, "location": [0, 0, 0.6]}
    )

    # `bpy.tx` exposes both `ensure_channelbag_fcurves` and `iter_action_fcurves`
    tx_res = await blender_worker.call(
        "bpy.tx",
        {
            "description": "Verify documented channelbag helper aliases in bpy.tx",
            "code": (
                "obj = bpy.data.objects['Crate']\n"
                "fcs = ensure_channelbag_fcurves(obj, action_name='CrateTestAct')\n"
                "obj.keyframe_insert(data_path='location', frame=1)\n"
                "count = sum(1 for _ in iter_action_fcurves(obj))\n"
                "result = {'fcurves': count}\n"
            ),
        },
    )
    assert tx_res["result"]["fcurves"] >= 3

    # Aim camera too close using `target` alias -> center in-frame, but AABB crowds/clips frame!
    await blender_worker.call(
        "camera.upsert",
        {"name": "TightCam", "location": [1.1, -1.1, 0.85], "target": [0, 0, 0.35], "lens_mm": 55},
    )
    tight_audit = await blender_worker.call(
        "scene.audit",
        {"subject": "Crate", "camera": "TightCam", "frustum_margin": 0.05},
    )
    assert tight_audit["passed"] is False
    assert tight_audit["framing"]["min_edge_margin_pct"] < 5.0
    assert any("AABB clips or crowds camera border" in i for i in tight_audit["issues"])

    # Pull camera back -> full 3D AABB clears the 5% safe margin!
    await blender_worker.call(
        "camera.upsert",
        {"name": "TightCam", "location": [3.8, -3.8, 2.3], "look_at": [0, 0, 0.6], "lens_mm": 50},
    )
    wide_audit = await blender_worker.call(
        "scene.audit",
        {"subject": "Crate", "camera": "TightCam", "frustum_margin": 0.05},
    )
    assert wide_audit["passed"] is True
    assert wide_audit["framing"]["bbox_in_frame_pct"] == 100.0
    assert wide_audit["framing"]["min_edge_margin_pct"] >= 5.0
    assert wide_audit["framing"]["subject_screen_height_pct"]["max"] > 10.0


@pytest.mark.asyncio(loop_scope="module")
async def test_gemini_degradation_health_tracking_and_proactive_compaction(
    blender_worker: HeadlessTransport, tmp_path: Path
) -> None:
    """Verify long/video context friction (`thought_leak` + `malformed` + `op_failures`) is recorded
    in `transcript.jsonl`, `breadcrumbs.jsonl`, and `meta.json`, and triggers proactive compaction
    even below 50% of a 1M Gemini window.
    """
    import json

    home_dir = tmp_path / "bb_health_home"
    work_dir = tmp_path / "health_ws"
    work_dir.mkdir(parents=True)
    await blender_worker.call("scene.clear")

    turn_n = 0

    class DegradingGeminiProvider:
        name = "gemini"
        model = "gemini-4-carbon"

        async def send(
            self, *, system: str, messages: list[dict[str, Any]], cancel_event: Any = None
        ) -> AsyncIterator[StreamEvent]:
            nonlocal turn_n
            _ = (system, messages, cancel_event)
            turn_n += 1
            if turn_n == 1:
                # Leak thought + emit malformed JSON bl_ops
                yield StreamEvent(
                    t="diagnostic",
                    reason="thought_leak",
                    detail="Gemini leaked unflagged sthought/düşünceler prefix",
                )
                yield StreamEvent(t="text", delta="```bl_ops\n{bad json\n```")
            elif turn_n == 2:
                # Emit failing op -> crosses degradation threshold with video in context -> triggers auto-compact!
                yield StreamEvent(
                    t="text", delta='```bl_ops\n{"ops": [{"op": "nonexistent.op"}]}\n```'
                )
            else:
                yield StreamEvent(t="text", delta="Recovered cleanly after context compaction.")
            yield StreamEvent(
                t="usage",
                in_fresh=2000,
                out=80,
                cache_read=20_000,
                cache_write=0,
                context_in=22_000,
            )
            yield StreamEvent(t="stop", reason="end_turn")

        async def count_tokens(self, *, system: str, messages: list[dict[str, Any]]) -> int | None:
            _ = (system, messages)
            return 22_000

    session = await bootstrap_session(
        cwd=work_dir,
        home=home_dir,
        overrides={"BLENDERBOT_GEMINI_API_KEY": "fake-key", "BLENDERBOT_PROVIDER": "gemini"},
        make_provider_fn=lambda cfg, effort_override=None: DegradingGeminiProvider(),
        injected_transport=blender_worker,
    )
    try:
        compactions_fired: list[dict[str, Any]] = []
        for ev_name in EVENTS:
            if ev_name == "compact-start":
                session.agent.on(ev_name, compactions_fired.append)
            else:
                session.agent.on(ev_name, lambda _p: None)

        # Seed Ledger with an older video segment + several turns so compressible segments exist (<5% of 1M limit!)
        session.ledger.add(
            seg_type="media",
            label="video ref.mp4 (files_api)",
            role="user",
            video={"uri": "https://example.com/files/vid1", "media_type": "video/mp4", "tokens": 18_000},
        )
        for i in range(5):
            session.ledger.add(
                seg_type="op-result",
                label=f"old batch #{i}",
                role="user",
                text="x" * 1600,
            )

        out = await session.agent.run("Check scene after heavy video session")
        assert out["status"] == "done"

        # Proactive health-degraded compaction fired despite total tokens << 85% of 1M!
        assert any(c.get("reason") == "health-degraded" for c in compactions_fired)
        assert session.agent.stats["thought_leaks"] == 1
        assert session.agent.stats["malformed"] == 1
        assert session.agent.stats["op_failures"] == 1
        assert session.agent.stats["compactions"] >= 1

        # Verify `transcript.jsonl`, `breadcrumbs.jsonl`, and `meta.json` all recorded health telemetry
        tr_lines = [
            json.loads(ln)
            for ln in session.log.transcript_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        health_events = [e for e in tr_lines if e.get("type") == "health"]
        assert len(health_events) >= 3
        signals = {e["data"]["signal"] for e in health_events}
        assert {"thought_leak", "malformed_bl_ops", "plan-failed"}.issubset(signals)

        crumbs = read_breadcrumbs(session.log.dir, limit=80)
        assert any(c.get("event") == "model-health" for c in crumbs)

        meta_disk = json.loads(session.log.meta_path.read_text(encoding="utf-8"))
        assert meta_disk["totals"]["thought_leaks"] == 1
        assert meta_disk["totals"]["compactions"] >= 1
        assert "health" in meta_disk
    finally:
        session.transport = None
        await session.close()