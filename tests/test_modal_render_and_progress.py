"""Regression tests for:
1. Compaction preserving full active & historical user instructions (no 48-char truncation).
2. Local render progress streaming (`render_pre` / `render_post` / `render_stats`) & timeout suspension.
3. `scene.pack_for_cloud` + `render.modal` (and `modal: true` delegation) planning, roles & execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio

from blenderbot.agent.executor import execute_batch, plan_batch
from blenderbot.agent.journal import Journal
from blenderbot.agent.parser import OpCall
from blenderbot.agent.prompt import build_system_prompt, builtin_prompts
from blenderbot.agent.roles import is_op_allowed_for_role
from blenderbot.agent.sandbox import Sandbox
from blenderbot.config.blender_paths import discover_blender
from blenderbot.context.compact import compact_ledger
from blenderbot.context.ledger import Ledger
from blenderbot.splats.blender_render import ModalBlenderRenderResult
from blenderbot.transport import HeadlessTransport


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


@pytest.mark.asyncio
async def test_compaction_preserves_active_and_historical_user_prompts() -> None:
    """Verify Pass 3 compaction never condenses the active user prompt and never clips older prompts at 48 chars."""
    ledger = Ledger(limit=2_000, compact_at=0.50)
    ledger.add(seg_type="grounding", label="grounding", role="user", text="Grounding", pinned=True)

    older_prompt = (
        "[Blender Live: mode=gui | v=5.3.0]\n\n"
        "heya bb! Can you create a realistic backrooms scene with fluorescent lights and damp carpet"
    )
    ledger.add(seg_type="user", label=older_prompt[:48], role="user", text=older_prompt)
    ledger.add(seg_type="assistant", label="turn #1", role="assistant", text="Built scene " * 60)
    ledger.add(seg_type="op-result", label="batch 1", role="user", text="ok " * 60)

    active_prompt = "ok, can you add a camera arc of an fps walkthrough and push the render to modal?"
    ledger.add(
        seg_type="user",
        label=active_prompt[:200],
        role="user",
        text=f"[Blender Live: mode=gui | v=5.3.0]\n\n{active_prompt}",
    )

    # Simulate 12 autonomous tool-loop turns after the user spoke (>6 segments behind tail!)
    for step in range(12):
        ledger.add(
            seg_type="assistant",
            label=f"turn #{step + 2}",
            role="assistant",
            text=f"Step {step} reasoning and ops " * 40,
        )
        ledger.add(
            seg_type="op-result",
            label=f"batch #{step + 2}",
            role="user",
            text=f"Result {step} " * 40,
        )

    await compact_ledger(ledger, target_fraction=0.10, force=True)

    all_text = "\n".join(str(s.part.get("text", "")) for s in ledger.segments)
    # 1. Active user prompt must remain verbatim as an un-condensed `user` segment!
    user_segs = [s for s in ledger.segments if s.type == "user"]
    assert any("and push the render to modal?" in str(s.part.get("text", "")) for s in user_segs)
    # 2. Summarized older turn must retain full clean user text rather than 48-char truncation!
    assert "with fluorescent lights and damp carpet" in all_text


@pytest.mark.asyncio(loop_scope="module")
async def test_local_render_streams_progress_and_suspends_short_timeout(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
) -> None:
    """Verify local `render.animation` streams frame-by-frame progress and suspends a tiny RPC timeout."""
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "scene.setup",
        {"fps": 24, "frame_start": 1, "frame_end": 8, "resolution": [320, 180], "engine": "BLENDER_WORKBENCH"},
    )
    await blender_worker.call(
        "mesh.primitive", {"name": "CubeA", "shape": "cube", "size": 1.0, "location": [0, 0, 0.5]}
    )
    await blender_worker.call(
        "camera.upsert", {"name": "CamA", "location": [0, -4, 1.5], "look_at": [0, 0, 0.5], "lens_mm": 35}
    )

    out_mp4 = tmp_path / "local_prog.mp4"
    progress_events: list[dict[str, Any]] = []

    # Pass an intentionally tiny timeout (0.02s = 20ms) with `suspend_timeout=True`:
    # Without timeout suspension, an 8-frame MP4 render + FFmpeg mux (~200-400ms) would immediately fail!
    res = await blender_worker.call(
        "render.animation",
        {
            "output_path": str(out_mp4),
            "engine": "BLENDER_WORKBENCH",
            "format": "MP4",
            "frame_start": 1,
            "frame_end": 8,
            "camera": "CamA",
        },
        timeout=0.02,
        on_progress=progress_events.append,
        suspend_timeout=True,
    )
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0
    assert res["frames"] == 8

    done_evs = [e for e in progress_events if e.get("phase") == "frame_done"]
    assert len(done_evs) == 8
    assert done_evs[-1]["done_frames"] == 8
    assert done_evs[-1]["total_frames"] == 8
    assert done_evs[-1]["pct"] == 100.0


@pytest.mark.asyncio(loop_scope="module")
async def test_pack_for_cloud_and_render_modal_routing(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
) -> None:
    """Verify `render.modal` role permissions, prompt documentation, `scene.pack_for_cloud`, and batch execution."""
    assert is_op_allowed_for_role("viewer", "render.modal") is False
    assert is_op_allowed_for_role("builder", "render.modal") is True
    assert is_op_allowed_for_role("master", "render.modal") is True

    sys_prompt = build_system_prompt(prompts=builtin_prompts(), grounding="G", role="builder")
    assert "render.modal" in sys_prompt

    sandbox = Sandbox(cwd=tmp_path)
    journal = Journal(tmp_path / ".bb_journal")

    # Verify plan_batch accepts `render.modal` under `builder`
    errs = plan_batch(
        [OpCall(index=0, op="render.modal", params={"output_path": "renders/cloud.mp4", "gpu": "H100"})],
        sandbox=sandbox,
        has_transport=True,
        role="builder",
    )
    assert errs == []

    packed_sizes: list[int] = []

    async def _fake_modal_render(**kwargs: Any) -> ModalBlenderRenderResult:
        bp: Path = kwargs["blend_path"]
        assert bp.is_file() and bp.stat().st_size > 1000
        packed_sizes.append(bp.stat().st_size)
        out_p: Path = kwargs["output_path"]
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(b"\x00\x00\x00\x18ftypmp42fake_modal_mp4")
        # 1x1 valid PNG preview
        png_1x1 = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return ModalBlenderRenderResult(
            output_path=out_p,
            mode="animation",
            engine="CYCLES",
            backend="OPTIX",
            device="NVIDIA H100 80GB HBM3",
            camera="CamA",
            frames=8,
            fps=24.0,
            width=1920,
            height=1080,
            wall_s=4.2,
            blender_render_s=3.1,
            render_fps=2.58,
            ms_per_frame=387.5,
            cost_usd=0.00461,
            preview_png_bytes=png_1x1,
        )

    with patch(
        "blenderbot.splats.blender_render.run_modal_blender_render",
        side_effect=_fake_modal_render,
    ):
        outcome = await execute_batch(
            [
                OpCall(
                    index=0,
                    op="render.modal",
                    params={
                        "output_path": "renders/cloud.mp4",
                        "engine": "CYCLES",
                        "gpu": "H100",
                    },
                )
            ],
            sandbox=sandbox,
            journal=journal,
            transport=blender_worker,
            role="builder",
        )

    assert outcome.status == "ok", outcome.errors
    assert len(packed_sizes) == 1
    assert (tmp_path / "renders" / "cloud.mp4").is_file()
    assert outcome.results[0]["result"]["backend"] == "OPTIX"
    assert outcome.results[0]["result"]["device"] == "NVIDIA H100 80GB HBM3"
    assert any(m["kind"] == "image" for m in outcome.attached_media)