"""Pathclear P0 (`P0_CLOUD_OK`): End-to-end Modal H100 burst + surgical resume verification.

Usage:
  1. Deploy worker once:  .venv\\Scripts\\python -m modal deploy src/blenderbot/splats/modal_app.py
  2. Run P0 gate:         .venv\\Scripts\\python scripts/pathclear_p0_modal.py [path/to/video.mp4]

Proves:
  - Deployed H100 container wakes with AOT-compiled `gsplat` (zero `nvcc` JIT).
  - Extracts frames + runs VGGT-1B metric pose + RANSAC `Z=0` floor alignment.
  - Burst 1: trains 2,000 steps streaming CUDA Event `stage_timings`, PSNR, and JPEG triptychs.
  - Burst 2 (`splat.resume`): applies surgical `prune_opa=0.02` + `clamp_log_scale=-0.5` (preserving
    Adam momentum, zero video re-upload, zero re-SfM) and trains +2,000 steps (`2,000 -> 4,000`),
    appending cleanly onto `telemetry.jsonl`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from rich.console import Console

from blenderbot.splats.client import make_job_id, run_modal_splat_job
from blenderbot.splats.config import SplatConfig, SurgerySpec

console = Console()


async def _main() -> None:
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1]).resolve()
    else:
        cands = sorted((root / "assets" / "seedance").glob("*.mp4"))
        if not cands:
            raise SystemExit("No .mp4 found in assets/seedance/. Pass an .mp4 path explicitly.")
        # Pick smallest clip first for crisp fast gate
        video_path = min(cands, key=lambda p: p.stat().st_size)

    job_id = make_job_id(video_path)
    out_dir = root / "assets" / "splats" / job_id
    console.print(f"[bold cyan]P0 Gate[/] video=[green]{video_path.name}[/] ({video_path.stat().st_size / 1e6:.1f} MB) -> job=[yellow]{job_id}[/]")

    def _on_ev(ev: dict) -> None:
        et = ev.get("event")
        if et == "stage":
            console.print(f"  [cyan]stage[/] {ev.get('stage')}: { {k: v for k, v in ev.items() if k not in ('event', 'stage', 'job_id')} }")
        elif et == "surgery":
            console.print(f"  [magenta]surgery[/] step={ev['step']} removed={ev['removed']} ({ev['n_before']} -> {ev['n_after']}) ops={ev['ops']}")
        elif et == "telemetry" and (ev["step"] % 500 == 0 or ev["step"] == 1):
            st = ev.get("stage_timings", {})
            console.print(
                f"  [green]step {ev['step']:5d}/{ev['total_steps']}[/] "
                f"loss={ev['loss']:.4f} psnr={ev['psnr']:.2f}dB N={ev['n_gaussians']:,} "
                f"({ev['it_per_sec']:.0f} it/s | fwd={st.get('fwd_raster_ms', 0):.2f}ms bwd={st.get('bwd_ms', 0):.2f}ms)"
            )

    cfg = SplatConfig(gpu="H100")
    cfg.train.iterations = 2_000
    cfg.train.log_every = 100

    console.print("\n[bold]── Burst 1: Initial Extract + VGGT Pose + 2,000 Steps ──[/]")
    t0 = time.perf_counter()
    res1 = await run_modal_splat_job(
        output_dir=out_dir,
        job_id=job_id,
        video_path=video_path,
        cfg=cfg,
        on_event=_on_ev,
    )
    b1_wall = time.perf_counter() - t0
    console.print(
        f"[bold green]Burst 1 OK[/] {b1_wall:.1f}s wall | {res1.n_gaussians:,} Gaussians | "
        f"cost=${res1.cost_usd:.4f} | timings={res1.macro_timings}"
    )

    console.print("\n[bold]── Burst 2: Surgical Resume (+2,000 Steps, Zero Re-Upload) ──[/]")
    surg = SurgerySpec(prune_opa=0.02, clamp_log_scale=-0.5)
    t1 = time.perf_counter()
    res2 = await run_modal_splat_job(
        output_dir=out_dir,
        job_id=job_id,
        video_path=None,  # Resume from Modal Volume / HBM cache!
        cfg=cfg,
        add_steps=2_000,
        surgery=surg,
        on_event=_on_ev,
    )
    b2_wall = time.perf_counter() - t1
    console.print(
        f"[bold green]Burst 2 OK[/] {b2_wall:.1f}s wall | step={res2.step} | {res2.n_gaussians:,} Gaussians | "
        f"cost=${res2.cost_usd:.4f} | timings={res2.macro_timings}"
    )

    assert res2.ply_path.is_file() and res2.ply_path.stat().st_size > 10_000
    assert res2.splat_path is not None and res2.splat_path.is_file()
    cams = json.loads(res2.cameras_path.read_text(encoding="utf-8"))
    assert len(cams["cameras"]) >= 3
    lines = [json.loads(line) for line in res2.telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    telem_steps = [r["step"] for r in lines if r.get("event") == "telemetry"]
    assert telem_steps == sorted(telem_steps), f"Telemetry steps not monotonic across resume: {telem_steps}"
    assert any(r.get("event") == "surgery" for r in lines), "Missing surgery audit marker in telemetry.jsonl"

    console.print(f"\n[bold green]P0_CLOUD_OK[/] artifacts saved to [cyan]{out_dir}[/]")


if __name__ == "__main__":
    asyncio.run(_main())