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


def test_modal_worker_timeout_env_and_self_readme_guard(tmp_path: Path) -> None:
    """Verify `MODAL_WORKER_TIMEOUT` config/worker resolution and repo `blenderbot.md` exclusion."""
    from blenderbot.agent.agents_md import discover_agents_files
    from blenderbot.config.env import load_config, resolve_modal_worker_timeout
    from blenderbot.splats.modal_app import _resolve_modal_worker_timeout

    home = tmp_path / "home"
    home.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()

    cfg_default = load_config(cwd=ws, home=home, environ={})
    assert cfg_default.modal_worker_timeout == 900
    assert cfg_default.to_public_dict()["modal_worker_timeout"] == 900

    cfg_env = load_config(cwd=ws, home=home, environ={"MODAL_WORKER_TIMEOUT": "2400"})
    assert cfg_env.modal_worker_timeout == 2400
    assert resolve_modal_worker_timeout({"MODAL_WORKER_TIMEOUT": "2400"}) == 2400

    (home / ".env").write_text("MODAL_WORKER_TIMEOUT=1500\n", encoding="utf-8")
    cfg_dotenv = load_config(cwd=ws, home=home, environ={})
    assert cfg_dotenv.modal_worker_timeout == 1500
    assert resolve_modal_worker_timeout({}, cwd=ws, home=home) == 1500

    with pytest.raises(ValueError, match="MODAL_WORKER_TIMEOUT"):
        load_config(cwd=ws, home=home, environ={"MODAL_WORKER_TIMEOUT": "0"})

    with patch.dict("os.environ", {"MODAL_WORKER_TIMEOUT": "2100"}, clear=False):
        assert _resolve_modal_worker_timeout(900) == 2100
        assert _resolve_modal_worker_timeout(1800) == 2100

    # Repo root `blenderbot.md` (package readme) must NOT be auto-injected as `BLENDERBOT.md`
    repo_root = Path(__file__).resolve().parents[1]
    found_repo = discover_agents_files(cwd=repo_root)
    assert not any(Path(e["path"]).name == "blenderbot.md" for e in found_repo)

    # Genuine project `AGENTS.md` in a normal workspace still loads cleanly
    (ws / "AGENTS.md").write_text("# Studio Rules\nUse meters.\n", encoding="utf-8")
    found_ws = discover_agents_files(cwd=ws)
    assert len(found_ws) == 1 and Path(found_ws[0]["path"]).name == "AGENTS.md"


@pytest.mark.asyncio
async def test_splat_from_images_bundle_exif_timeout_and_execution(tmp_path: Path) -> None:
    """Verify `splat.from_images`, EXIF transpose + downscale, directory hint on `from_video`, and `with_options(timeout=...)`."""
    import gzip
    import json

    from PIL import Image

    from blenderbot.config.env import load_config
    from blenderbot.splats.client import (
        discover_image_files,
        pack_images_tar_gz,
        run_modal_splat_job,
        unpack_images_bundle,
    )
    from blenderbot.splats.config import ExtractConfig

    assert is_op_allowed_for_role("viewer", "splat.from_images") is False
    assert is_op_allowed_for_role("builder", "splat.from_images") is True
    assert is_op_allowed_for_role("master", "splat.from_images") is True

    sys_prompt = build_system_prompt(prompts=builtin_prompts(), grounding="G", role="builder")
    assert "splat.from_images" in sys_prompt
    assert "Zero Self-Source Inspection" in sys_prompt

    img_dir = tmp_path / "assets" / "inputs" / "pelle_test"
    img_dir.mkdir(parents=True)

    # Create 4 landscape 1600x1200 JPEGs tagged with EXIF Orientation=6 (90 deg CW -> upright 1200x1600!)
    # Named `img_1`, `img_2`, `img_10`, `img_20` to verify natural sort order.
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation = Rotate 90 CW
    for idx in (10, 2, 20, 1):
        im = Image.new("RGB", (1600, 1200), color=(20 * idx % 255, 100, 180))
        im.save(img_dir / f"img_{idx}.jpeg", format="JPEG", quality=92, exif=exif.tobytes())

    discovered = discover_image_files(images_path=img_dir)
    assert [p.name for p in discovered] == [
        "img_1.jpeg",
        "img_2.jpeg",
        "img_10.jpeg",
        "img_20.jpeg",
    ]

    ext_cfg = ExtractConfig(max_frames=10, max_width=800, quality=92, fps=8.0)
    tar_bytes = pack_images_tar_gz(discovered, ext_cfg)
    assert len(tar_bytes) > 200

    unpack_dir = tmp_path / "unpacked" / "frames"
    frames, pts_map, meta, has_alpha = unpack_images_bundle(tar_bytes, unpack_dir, ext_cfg)
    assert len(frames) == 4
    assert has_alpha is False
    # Raw 1600x1200 + EXIF 6 -> 1200x1600 -> scaled to max_width=800 on long edge -> 600x800!
    assert (meta["width"], meta["height"]) == (600, 800)
    assert [f.name for f in frames] == ["img_1.jpg", "img_2.jpg", "img_10.jpg", "img_20.jpg"]
    assert len(pts_map) == 4

    sandbox = Sandbox(cwd=tmp_path)
    journal = Journal(tmp_path / ".bb_journal")

    # Passing a folder to `splat.from_video` directs the agent straight to `splat.from_images`
    vid_dir_errs = plan_batch(
        [OpCall(index=0, op="splat.from_video", params={"video_path": "assets/inputs/pelle_test"})],
        sandbox=sandbox,
        has_transport=False,
        role="builder",
    )
    assert len(vid_dir_errs) == 1
    assert "splat.from_images" in vid_dir_errs[0]["message"]

    # `splat.from_images` passes planning without needing Blender connected (unless auto_import)
    img_errs = plan_batch(
        [
            OpCall(
                index=0,
                op="splat.from_images",
                params={"images_path": "assets/inputs/pelle_test", "job_id": "pelle_imgs"},
            )
        ],
        sandbox=sandbox,
        has_transport=False,
        role="builder",
    )
    assert img_errs == []

    # Verify `run_modal_splat_job` binds `with_options(timeout=...)` from `Config.modal_worker_timeout`
    # and streams `images_tar_gz` cleanly through `execute_batch`.
    seen_timeouts: list[int] = []
    seen_kwargs: list[dict[str, Any]] = []

    class _FakeBoundCls:
        def __call__(self) -> Any:
            return self

        @property
        def run_stream(self) -> Any:
            return self

        def remote_gen(self, **kwargs: Any):
            seen_kwargs.append(kwargs)
            yield {"event": "stage", "stage": "extract", "source": "images", "frames": 4}
            yield {
                "event": "done",
                "burst_id": 1,
                "step": 2500,
                "n_gaussians": 42000,
                "device": "NVIDIA H100 80GB HBM3",
                "cost_usd": 0.042,
                "macro_timings": {"extract_s": 0.2, "pose_s": 1.8, "train_s": 22.0},
                "ply_gz": gzip.compress(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"),
                "splat_gz": gzip.compress(b"\x00" * 32),
                "cameras_json": json.dumps({"job_id": kwargs["job_id"], "cameras": []}),
            }

    class _FakeModalCls:
        def with_options(self, **opts: Any) -> _FakeBoundCls:
            seen_timeouts.append(int(opts["timeout"]))
            return _FakeBoundCls()

    home_dir = tmp_path / "bb_home"
    home_dir.mkdir()
    cfg = load_config(cwd=tmp_path, home=home_dir, environ={"MODAL_WORKER_TIMEOUT": "1850"})

    with patch("modal.Cls.from_name", return_value=_FakeModalCls()):
        outcome = await execute_batch(
            [
                OpCall(
                    index=0,
                    op="splat.from_images",
                    params={
                        "images_path": "assets/inputs/pelle_test",
                        "job_id": "pelle_imgs",
                        "iterations": 2500,
                        "max_width": 800,
                    },
                )
            ],
            sandbox=sandbox,
            journal=journal,
            transport=None,
            cfg=cfg,
            role="builder",
        )

    assert outcome.status == "ok", outcome.errors
    assert seen_timeouts == [1850]
    assert len(seen_kwargs) == 1
    assert seen_kwargs[0]["mp4_bytes"] is None
    assert isinstance(seen_kwargs[0]["images_tar_gz"], bytes) and len(seen_kwargs[0]["images_tar_gz"]) > 200
    assert (tmp_path / "assets" / "splats" / "pelle_imgs" / "splat.ply").is_file()

    # Also verify timeout error message surfaces `MODAL_WORKER_TIMEOUT`
    class _TimeoutCls:
        def with_options(self, **opts: Any) -> Any:
            return self

        def __call__(self) -> Any:
            return self

        @property
        def run_stream(self) -> Any:
            return self

        def remote_gen(self, **_kwargs: Any):
            raise TimeoutError("Task's current input in-01ABC hit its timeout of 1200s")
            yield  # pragma: no cover

    with (
        patch("modal.Cls.from_name", return_value=_TimeoutCls()),
        pytest.raises(RuntimeError, match="MODAL_WORKER_TIMEOUT=1200s"),
    ):
        await run_modal_splat_job(
            output_dir=tmp_path / "assets" / "splats" / "t_err",
            job_id="t_err",
            images_path=img_dir,
            timeout_sec=1200,
        )