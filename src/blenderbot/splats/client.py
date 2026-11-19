"""Local Windows host client for dispatching, controlling, resuming & rendering cloud 3DGS jobs.

Streams live telemetry into `assets/splats/{job_id}/telemetry.jsonl` + `snapshots/`, touches
`FlightRecorder` on every tick, emits SSE events on `SessionBus`, unpacks `splat.ply`,
`splat.splat`, `proxy_collision.glb`, and `cameras.json`, pushes live mid-train parameter/pause
updates via `modal.Dict`, and farms full-rate trajectory MP4 renders to Modal H100/L40S.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import gzip
import hashlib
import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SplatConfig, SurgerySpec

CTL_DICT_NAME = "blenderbot-splat-ctl"


@dataclass
class SplatJobResult:
    job_id: str
    burst_id: int
    step: int
    n_gaussians: int
    output_dir: Path
    ply_path: Path
    splat_path: Path | None
    proxy_glb_path: Path | None
    cameras_path: Path
    telemetry_path: Path
    device: str
    wall_s: float
    cost_usd: float
    macro_timings: dict[str, float]


@dataclass
class SplatRenderResult:
    job_id: str
    output_mp4: Path
    frames: int
    fps: float
    width: int
    height: int
    device: str
    wall_s: float
    render_fps: float
    cost_usd: float


def make_job_id(video_path: Path) -> str:
    stem = "".join(c if (c.isalnum() or c in "_-") else "_" for c in video_path.stem)[:28].strip("_")
    stamp = datetime.datetime.now().strftime("%H%M%S")
    digest = hashlib.sha1(f"{video_path.resolve()}:{time.time()}".encode()).hexdigest()[:6]
    return f"{stem or 'splat'}_{stamp}_{digest}"


def _write_job_outputs(
    out_dir: Path,
    done_ev: dict[str, Any],
) -> tuple[Path, Path | None, Path | None, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "splat.ply"
    ply_raw = gzip.decompress(done_ev["ply_gz"])
    ply_path.write_bytes(ply_raw)

    splat_path: Path | None = None
    if done_ev.get("splat_gz"):
        splat_path = out_dir / "splat.splat"
        splat_path.write_bytes(gzip.decompress(done_ev["splat_gz"]))

    proxy_path: Path | None = None
    if done_ev.get("proxy_glb_gz"):
        proxy_path = out_dir / "proxy_collision.glb"
        proxy_path.write_bytes(gzip.decompress(done_ev["proxy_glb_gz"]))

    cameras_path = out_dir / "cameras.json"
    cameras_path.write_text(done_ev["cameras_json"], encoding="utf-8")
    return ply_path, splat_path, proxy_path, cameras_path


def _persist_local_telemetry(out_dir: Path, ev: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    telem_file = out_dir / "telemetry.jsonl"
    scalar_row = {
        k: v
        for k, v in ev.items()
        if not k.endswith("_b64")
        and k not in ("ply_gz", "splat_gz", "proxy_glb_gz", "cameras_json", "mp4_bytes")
    }
    with open(telem_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(scalar_row) + "\n")

    if ev.get("event") == "telemetry" and ev.get("render_b64"):
        latest_dir = out_dir / "previews"
        latest_dir.mkdir(parents=True, exist_ok=True)
        (latest_dir / "latest_render.jpg").write_bytes(base64.b64decode(ev["render_b64"]))
        if ev.get("diff_b64"):
            (latest_dir / "latest_diff.jpg").write_bytes(base64.b64decode(ev["diff_b64"]))
        if ev.get("gt_b64"):
            cam_stem = Path(ev.get("camera_name", "cam")).stem
            (latest_dir / f"gt_{cam_stem}.jpg").write_bytes(base64.b64decode(ev["gt_b64"]))
        if ev.get("is_snapshot"):
            snap_dir = out_dir / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            step_n = int(ev.get("step", 0))
            (snap_dir / f"step_{step_n:06d}.jpg").write_bytes(base64.b64decode(ev["render_b64"]))


async def push_modal_splat_control(job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Push a live mid-training control dict (`paused`, `stop_early`, `save_checkpoint`, `lambda_dssim`, etc.)."""

    def _sync_push() -> dict[str, Any]:
        import modal

        d = modal.Dict.from_name(CTL_DICT_NAME, create_if_missing=True)
        cur = d.get(job_id)
        merged = dict(cur) if isinstance(cur, dict) else {}
        merged.update(updates)
        merged["seq"] = int(merged.get("seq", 0)) + 1
        merged["ts"] = round(time.time(), 3)
        d[job_id] = merged
        return merged

    return await asyncio.to_thread(_sync_push)


async def run_modal_splat_job(
    *,
    output_dir: Path,
    job_id: str,
    video_path: Path | None = None,
    views_bundle_path: Path | None = None,
    known_cameras_path: Path | None = None,
    known_cameras_json: str | None = None,
    cfg: SplatConfig | None = None,
    cfg_overrides: dict[str, Any] | None = None,
    add_steps: int | None = None,
    surgery: SurgerySpec | None = None,
    masks_paths: dict[str, Path] | None = None,
    exclude_frames: list[str] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> SplatJobResult:
    """Dispatch `from_video`, `from_blender` (`views_bundle_path`), or `resume` to Modal."""
    import modal

    is_resume = (video_path is None) and (views_bundle_path is None)
    eff_gpu = str(
        (cfg_overrides or {}).get("gpu")
        or (cfg.gpu if cfg is not None else "H100")
    ).upper()
    cls_name = "SplatWorkerL40S" if eff_gpu == "L40S" else "SplatWorker"

    mp4_bytes = video_path.read_bytes() if video_path is not None else None
    views_tar_gz = views_bundle_path.read_bytes() if views_bundle_path is not None else None
    eff_known_json = known_cameras_json
    if eff_known_json is None and known_cameras_path is not None and known_cameras_path.is_file():
        eff_known_json = known_cameras_path.read_text(encoding="utf-8")

    # On initial run pass full `cfg_dict`; on `resume` pass `cfg_dict=None` + `cfg_overrides`
    # so `core_worker.py` merges overrides onto `meta.json["config"]` without clobbering saved settings!
    if is_resume:
        send_cfg_dict = cfg.to_dict() if cfg is not None else None
        send_overrides = cfg_overrides
    else:
        base_cfg = cfg if cfg is not None else SplatConfig()
        if cfg_overrides:
            base_cfg = base_cfg.merge_overrides(cfg_overrides)
        send_cfg_dict = base_cfg.to_dict()
        send_overrides = None

    masks_b64: dict[str, str] | None = None
    if masks_paths:
        masks_b64 = {k: base64.b64encode(p.read_bytes()).decode("ascii") for k, p in masks_paths.items()}

    q: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _producer() -> None:
        try:
            worker_cls = modal.Cls.from_name("blenderbot-splat", cls_name)
            worker = worker_cls()
            gen = worker.run_stream.remote_gen(
                job_id=job_id,
                mp4_bytes=mp4_bytes,
                views_tar_gz=views_tar_gz,
                known_cameras_json=eff_known_json,
                cfg_dict=send_cfg_dict,
                cfg_overrides=send_overrides,
                add_steps=add_steps,
                surgery_dict=surgery.to_dict() if surgery else None,
                masks_b64=masks_b64,
                exclude_frames=exclude_frames,
            )
            for item in gen:
                q.put(("ev", item))
            q.put(("eof", None))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

    t_client_start = time.perf_counter()
    thr = threading.Thread(target=_producer, daemon=True)
    thr.start()

    done_ev: dict[str, Any] | None = None
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        if kind == "err":
            raise RuntimeError(f"Modal splat worker failed: {payload}") from payload
        if kind == "eof":
            break
        if kind == "ev":
            _persist_local_telemetry(output_dir, payload)
            if on_event is not None:
                on_event(payload)
            if payload.get("event") == "done":
                done_ev = payload

    if done_ev is None:
        raise RuntimeError("Modal splat stream ended without a 'done' artifact event.")

    ply_path, splat_path, proxy_path, cameras_path = _write_job_outputs(output_dir, done_ev)
    total_client_wall = round(time.perf_counter() - t_client_start, 3)

    return SplatJobResult(
        job_id=job_id,
        burst_id=int(done_ev["burst_id"]),
        step=int(done_ev["step"]),
        n_gaussians=int(done_ev["n_gaussians"]),
        output_dir=output_dir,
        ply_path=ply_path,
        splat_path=splat_path,
        proxy_glb_path=proxy_path,
        cameras_path=cameras_path,
        telemetry_path=output_dir / "telemetry.jsonl",
        device=str(done_ev.get("device", eff_gpu)),
        wall_s=total_client_wall,
        cost_usd=float(done_ev.get("cost_usd", 0.0)),
        macro_timings=dict(done_ev.get("macro_timings") or {}),
    )


async def run_modal_splat_render(
    *,
    job_id: str,
    output_mp4: Path,
    trajectory_doc: dict[str, Any],
    ply_fallback_path: Path | None = None,
    gpu: str = "H100",
    rasterize_mode: str = "antialiased",
    bg_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> SplatRenderResult:
    """Farm full-rate 3DGS camera trajectory MP4 rendering to Modal H100/L40S (`gsplat` CUDA)."""
    import modal

    eff_gpu = gpu.upper()
    cls_name = "SplatWorkerL40S" if eff_gpu == "L40S" else "SplatWorker"
    ply_gz = (
        gzip.compress(ply_fallback_path.read_bytes(), compresslevel=1)
        if ply_fallback_path is not None and ply_fallback_path.is_file()
        else None
    )

    q: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _producer() -> None:
        try:
            worker_cls = modal.Cls.from_name("blenderbot-splat", cls_name)
            worker = worker_cls()
            gen = worker.render_trajectory_stream.remote_gen(
                job_id=job_id,
                trajectory_doc=trajectory_doc,
                ply_gz=ply_gz,
                gpu=eff_gpu,
                rasterize_mode=rasterize_mode,
                bg_rgb=bg_rgb,
            )
            for item in gen:
                q.put(("ev", item))
            q.put(("eof", None))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

    t_start = time.perf_counter()
    thr = threading.Thread(target=_producer, daemon=True)
    thr.start()

    done_ev: dict[str, Any] | None = None
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.04)
            continue

        if kind == "err":
            raise RuntimeError(f"Modal trajectory render failed: {payload}") from payload
        if kind == "eof":
            break
        if kind == "ev":
            if on_event is not None:
                on_event(payload)
            if payload.get("event") == "render_done":
                done_ev = payload

    if done_ev is None or not done_ev.get("mp4_bytes"):
        raise RuntimeError("Modal trajectory render ended without MP4 bytes.")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    output_mp4.write_bytes(done_ev["mp4_bytes"])
    wall_s = round(time.perf_counter() - t_start, 3)

    return SplatRenderResult(
        job_id=job_id,
        output_mp4=output_mp4,
        frames=int(done_ev["frames"]),
        fps=float(done_ev["fps"]),
        width=int(done_ev["width"]),
        height=int(done_ev["height"]),
        device=str(done_ev.get("device", eff_gpu)),
        wall_s=wall_s,
        render_fps=float(done_ev.get("render_fps", 0.0)),
        cost_usd=float(done_ev.get("cost_usd", 0.0)),
    )