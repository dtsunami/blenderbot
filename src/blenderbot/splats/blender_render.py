"""Modal Serverless H100 / L40S Headless Blender 5.3 GPU Renderer (`render.modal`).

Runs official Blender 5.3 Linux x64 (`CYCLES` with hardware OptiX / CUDA raytracing + GPU
OptiX/OIDN denoising + `use_persistent_data=True`, or headless EGL `BLENDER_EEVEE` /
`BLENDER_WORKBENCH`) inside Modal H100 / L40S containers, streaming live frame/sample
telemetry back to the local Windows session and writing `.mp4` / `.png` / `.exr` outputs.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModalBlenderRenderResult:
    output_path: Path
    mode: str
    engine: str
    backend: str
    device: str
    camera: str
    frames: int
    fps: float
    width: int
    height: int
    wall_s: float
    blender_render_s: float
    render_fps: float
    ms_per_frame: float
    cost_usd: float
    shards: int = 1
    preview_png_bytes: bytes | None = None


_BLENDER_MODAL_DRIVER = r'''
import json
import os
import sys
import time
from pathlib import Path

import bpy

PREFIX = "BLBOT_EV:"


def emit(payload):
    sys.stdout.write(PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


cfg_path = Path(os.environ["BLBOT_RENDER_CFG"])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

sc = bpy.context.scene
if cfg.get("scene") and cfg["scene"] in bpy.data.scenes:
    sc = bpy.data.scenes[cfg["scene"]]
    if hasattr(bpy.context, "window") and bpy.context.window is not None:
        bpy.context.window.scene = sc

r = sc.render

cam_name = cfg.get("camera")
cam = bpy.data.objects.get(cam_name) if cam_name else sc.camera
if cam is None:
    cam = next((o for o in sc.objects if o.type == "CAMERA"), None)
if cam is None or cam.type != "CAMERA":
    raise RuntimeError("No active CAMERA found in .blend scene for Modal render.")
sc.camera = cam

target_engine = str(cfg.get("engine") or r.engine or "CYCLES").strip().upper()
r.engine = target_engine

backend_used = target_engine
device_label = "GPU"

if target_engine == "CYCLES" and hasattr(sc, "cycles"):
    prefs = bpy.context.preferences
    if "cycles" in prefs.addons:
        cprefs = prefs.addons["cycles"].preferences
        chosen_type = None
        chosen_names = []
        # Probe CUDA device name first: Hopper/Ampere compute GPUs (H100/H200/A100) have zero
        # hardware RT cores and ship precompiled `kernel_sm_90.cubin` / `kernel_sm_80.cubin` in
        # Blender 5.3 (0.0s JIT vs ~240s OptiX PTX JIT!), whereas L40S/RTX GPUs have hardware RT cores.
        prefer_order = ("OPTIX", "CUDA")
        try:
            cprefs.compute_device_type = "CUDA"
            if hasattr(cprefs, "refresh_devices"):
                cprefs.refresh_devices()
            else:
                cprefs.get_devices()
            cuda_names = " ".join(str(d.name).upper() for d in cprefs.devices if d.type != "CPU")
            if any(tag in cuda_names for tag in ("H100", "H200", "A100", "A800", "B100", "B200", "V100")):
                prefer_order = ("CUDA", "OPTIX")
        except Exception:
            pass
        if cfg.get("cycles_backend"):
            explicit_b = str(cfg["cycles_backend"]).strip().upper()
            if explicit_b in ("CUDA", "OPTIX"):
                prefer_order = (explicit_b,)

        for dev_type in prefer_order:
            try:
                cprefs.compute_device_type = dev_type
                if hasattr(cprefs, "refresh_devices"):
                    cprefs.refresh_devices()
                else:
                    cprefs.get_devices()
                gpu_devs = [d for d in cprefs.devices if d.type == dev_type]
                if not gpu_devs:
                    gpu_devs = [d for d in cprefs.devices if d.type != "CPU"]
                if gpu_devs:
                    for d in cprefs.devices:
                        d.use = (d.type != "CPU")
                    sc.cycles.device = "GPU"
                    chosen_type = dev_type
                    chosen_names = [str(d.name) for d in gpu_devs]
                    break
            except Exception as exc:
                emit({"event": "note", "message": f"Cycles {dev_type} probe note: {exc}"})
        if chosen_type:
            backend_used = chosen_type
            device_label = ", ".join(dict.fromkeys(chosen_names)) or chosen_type
        else:
            sc.cycles.device = "CPU"
            backend_used = "CPU"
            device_label = "CPU"

    if cfg.get("samples") is not None:
        sc.cycles.samples = max(1, int(cfg["samples"]))

    want_denoise = bool(cfg.get("denoise", True))
    sc.cycles.use_denoising = want_denoise
    denoiser_used = "NONE"
    if want_denoise:
        # Use OPTIX denoiser only when host driver mounts `/usr/share/nvidia/nvoptix.bin`;
        # otherwise use Blender 5.3's bundled GPU-accelerated OpenImageDenoise (`denoising_use_gpu=True`)!
        nvoptix_weights = Path("/usr/share/nvidia/nvoptix.bin")
        has_nvoptix_weights = nvoptix_weights.is_file() and nvoptix_weights.stat().st_size > 1024
        if backend_used == "OPTIX" and has_nvoptix_weights:
            try:
                sc.cycles.denoiser = "OPTIX"
                denoiser_used = "OPTIX"
            except Exception:
                pass
        if denoiser_used == "NONE":
            try:
                sc.cycles.denoiser = "OPENIMAGEDENOISE"
                if hasattr(sc.cycles, "denoising_use_gpu"):
                    sc.cycles.denoising_use_gpu = True
                if hasattr(sc.cycles, "denoising_prefilter"):
                    sc.cycles.denoising_prefilter = str(cfg.get("denoising_prefilter") or "FAST").upper()
                denoiser_used = "OPENIMAGEDENOISE"
            except Exception as d_exc:
                sc.cycles.use_denoising = False
                denoiser_used = f"DISABLED({d_exc})"
        for vl in sc.view_layers:
            if hasattr(vl, "cycles") and hasattr(vl.cycles, "denoiser") and denoiser_used in ("OPTIX", "OPENIMAGEDENOISE"):
                try:
                    vl.cycles.denoiser = denoiser_used
                except Exception:
                    pass

    if hasattr(r, "use_persistent_data"):
        r.use_persistent_data = bool(cfg.get("use_persistent_data", True))

elif target_engine == "BLENDER_EEVEE" and hasattr(sc, "eevee"):
    if cfg.get("samples") is not None and hasattr(sc.eevee, "taa_render_samples"):
        sc.eevee.taa_render_samples = max(1, int(cfg["samples"]))

if cfg.get("resolution"):
    r.resolution_x = max(16, int(cfg["resolution"][0]))
    r.resolution_y = max(16, int(cfg["resolution"][1]))
    r.resolution_percentage = 100

if cfg.get("film_transparent") is not None:
    r.film_transparent = bool(cfg["film_transparent"])

r.use_stamp = False

fps = float(r.fps) / float(r.fps_base or 1.0)
mode = str(cfg.get("mode") or "animation").lower()
out_dir = Path(cfg["out_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

if mode == "still":
    f_num = int(cfg["frame"]) if cfg.get("frame") is not None else int(sc.frame_current)
    sc.frame_set(f_num)
    f_start = f_end = f_num
    f_step = 1
    total_frames = 1
else:
    if cfg.get("frame_start") is not None:
        sc.frame_start = int(cfg["frame_start"])
    if cfg.get("frame_end") is not None:
        sc.frame_end = int(cfg["frame_end"])
    if cfg.get("frame_step") is not None:
        sc.frame_step = max(1, int(cfg["frame_step"]))
    f_start = int(sc.frame_start)
    f_end = max(f_start, int(sc.frame_end))
    f_step = max(1, int(sc.frame_step))
    total_frames = len(range(f_start, f_end + 1, f_step))

emit({
    "event": "stage",
    "stage": "blender_init",
    "blender_version": bpy.app.version_string,
    "engine": target_engine,
    "backend": backend_used,
    "denoiser": denoiser_used if target_engine == "CYCLES" else "N/A",
    "device": device_label,
    "camera": cam.name,
    "mode": mode,
    "frame_start": f_start,
    "frame_end": f_end,
    "total_frames": total_frames,
    "fps": round(fps, 4),
    "width": int(r.resolution_x * r.resolution_percentage / 100),
    "height": int(r.resolution_y * r.resolution_percentage / 100),
})

t_render0 = time.perf_counter()
state = {"cur_frame": f_start, "done": 0, "f_t0": t_render0, "last_stats": 0.0}


def _on_pre(sc_arg, *_rest):
    now = time.perf_counter()
    fr = int(getattr(sc_arg, "frame_current", state["cur_frame"]))
    state["cur_frame"] = fr
    state["f_t0"] = now


def _on_stats(stats_arg, *_rest):
    now = time.perf_counter()
    if (now - state["last_stats"]) < 1.0:
        return
    state["last_stats"] = now
    done_n = state["done"]
    emit({
        "event": "progress",
        "phase": "render_stats",
        "op": "render.modal",
        "engine": target_engine,
        "backend": backend_used,
        "device": device_label,
        "frame": state["cur_frame"],
        "done_frames": done_n,
        "total_frames": total_frames,
        "pct": round(100.0 * done_n / max(1, total_frames), 1),
        "stats": str(stats_arg or "")[:140],
        "elapsed_s": round(now - t_render0, 2),
    })


def _on_post(sc_arg, *_rest):
    now = time.perf_counter()
    fr = int(getattr(sc_arg, "frame_current", state["cur_frame"]))
    done_n = state["done"] + 1
    state["done"] = done_n
    elapsed_s = now - t_render0
    frame_ms = round((now - state["f_t0"]) * 1000.0, 1)
    avg_s = elapsed_s / max(1, done_n)
    rem_n = max(0, total_frames - done_n)
    emit({
        "event": "progress",
        "phase": "frame_done",
        "op": "render.modal",
        "engine": target_engine,
        "backend": backend_used,
        "device": device_label,
        "frame": fr,
        "done_frames": done_n,
        "total_frames": total_frames,
        "pct": round(100.0 * done_n / max(1, total_frames), 1),
        "frame_ms": frame_ms,
        "ms_per_frame": round(avg_s * 1000.0, 1),
        "eta_s": round(rem_n * avg_s, 1),
        "elapsed_s": round(elapsed_s, 2),
    })


bpy.app.handlers.render_pre.append(_on_pre)
bpy.app.handlers.render_stats.append(_on_stats)
bpy.app.handlers.render_post.append(_on_post)

fmt = str(cfg.get("format") or ("PNG" if mode == "still" else "MP4")).upper()
if hasattr(r.image_settings, "media_type"):
    r.image_settings.media_type = "IMAGE"

if mode == "still":
    if fmt == "EXR":
        r.image_settings.file_format = "OPEN_EXR"
        r.image_settings.color_mode = "RGBA"
        out_file = out_dir / "still.exr"
    elif fmt in ("JPG", "JPEG"):
        r.image_settings.file_format = "JPEG"
        r.image_settings.color_mode = "RGB"
        out_file = out_dir / "still.jpg"
    else:
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGBA" if r.film_transparent else "RGB"
        out_file = out_dir / "still.png"
    r.filepath = str(out_file)
    bpy.ops.render.render(write_still=True)
else:
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA" if r.film_transparent else "RGB"
    if hasattr(r.image_settings, "compression"):
        r.image_settings.compression = 15
    r.use_file_extension = True
    r.filepath = str(out_dir / "frame_")
    bpy.ops.render.render(animation=True)

render_elapsed = round(time.perf_counter() - t_render0, 3)
actual_w = int(r.resolution_x * r.resolution_percentage / 100)
actual_h = int(r.resolution_y * r.resolution_percentage / 100)

emit({
    "event": "blender_finished",
    "render_s": render_elapsed,
    "done_frames": max(1, state["done"]),
    "fps": round(fps, 4),
    "width": actual_w,
    "height": actual_h,
    "camera": cam.name,
    "engine": target_engine,
    "backend": backend_used,
    "device": device_label,
})
'''


def run_blender_modal_render_stream(
    *,
    blend_gz: bytes,
    render_spec: dict[str, Any],
    gpu_hourly_usd: float = 3.95,
) -> Iterator[dict[str, Any]]:
    """Execute headless Blender 5.3 GPU rendering inside a Modal H100/L40S container."""
    t0 = time.perf_counter()
    run_id = uuid.uuid4().hex[:10]
    shm_base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
    work_dir = shm_base / f"blbot_render_{run_id}"
    out_dir = work_dir / "out"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cache_sub in ("/blender_cache/optix", "/blender_cache/cuda"):
        try:
            Path(cache_sub).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    try:
        blend_path = work_dir / "scene.blend"
        blend_raw = gzip.decompress(blend_gz) if blend_gz[:2] == b"\x1f\x8b" else blend_gz
        blend_path.write_bytes(blend_raw)

        mode = str(render_spec.get("mode") or "animation").lower()
        fmt = str(render_spec.get("format") or ("PNG" if mode == "still" else "MP4")).upper()
        cfg_doc = {**render_spec, "mode": mode, "format": fmt, "out_dir": str(out_dir)}
        cfg_file = work_dir / "render_cfg.json"
        cfg_file.write_text(json.dumps(cfg_doc), encoding="utf-8")

        driver_file = work_dir / "driver.py"
        driver_file.write_text(_BLENDER_MODAL_DRIVER, encoding="utf-8")

        blender_bin = "/opt/blender/blender" if Path("/opt/blender/blender").is_file() else "blender"
        env = os.environ.copy()
        env["BLBOT_RENDER_CFG"] = str(cfg_file)
        env.setdefault("OPTIX_CACHE_PATH", "/blender_cache/optix")
        env.setdefault("CUDA_CACHE_PATH", "/blender_cache/cuda")

        proc = subprocess.Popen(
            [blender_bin, "-b", "-noaudio", str(blend_path), "--python", str(driver_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        tail_logs: list[str] = []
        fin_meta: dict[str, Any] = {}
        assert proc.stdout is not None
        for raw_ln in proc.stdout:
            ln = raw_ln.rstrip("\r\n")
            if ln.startswith("BLBOT_EV:"):
                try:
                    ev = json.loads(ln[len("BLBOT_EV:") :])
                except ValueError:
                    continue
                if ev.get("event") == "blender_finished":
                    fin_meta = ev
                else:
                    yield ev
            elif ln.strip():
                tail_logs.append(ln)
                if len(tail_logs) > 60:
                    tail_logs.pop(0)

        rc = proc.wait()
        if rc != 0 or not fin_meta:
            err_tail = "\n".join(tail_logs[-25:])
            raise RuntimeError(f"Modal Blender process exited with code {rc}:\n{err_tail}")

        fps = float(fin_meta.get("fps") or render_spec.get("fps") or 24.0)
        width = int(fin_meta.get("width") or 1920)
        height = int(fin_meta.get("height") or 1080)
        frames_rendered = int(fin_meta.get("done_frames") or 1)
        blender_render_s = float(fin_meta.get("render_s") or 0.001)
        preview_b64: str | None = None

        if mode == "still":
            candidates = sorted(out_dir.glob("still.*"))
            if not candidates:
                raise RuntimeError("Modal Blender still render finished without output image.")
            final_bytes = candidates[0].read_bytes()
            ext = candidates[0].suffix.lower()
        else:
            frame_files = sorted(out_dir.glob("frame_*.png"))
            if not frame_files:
                raise RuntimeError("Modal Blender animation render produced zero PNG frames.")
            frames_rendered = len(frame_files)
            mid_frame = frame_files[len(frame_files) // 2]
            preview_b64 = base64.b64encode(mid_frame.read_bytes()).decode("ascii")

            qual = str(render_spec.get("quality") or "HIGH").upper()
            crf = "15" if qual == "PERC_LOSSLESS" else ("22" if qual == "MEDIUM" else "18")
            mp4_out = work_dir / "encoded.mp4"
            # Pad odd dimensions safely if needed for yuv420p
            vf_pad = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
            ff_cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                f"{fps:.4f}",
                "-pattern_type",
                "glob",
                "-i",
                str(out_dir / "frame_*.png"),
                "-vf",
                vf_pad,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                crf,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4_out),
            ]
            ff_res = subprocess.run(ff_cmd, capture_output=True, text=True, check=False)
            if ff_res.returncode != 0 or not mp4_out.is_file():
                raise RuntimeError(f"Modal FFmpeg MP4 encode failed: {ff_res.stderr.strip()}")
            final_bytes = mp4_out.read_bytes()
            ext = ".mp4"

        wall_s = round(time.perf_counter() - t0, 3)
        cost_usd = round((wall_s / 3600.0) * gpu_hourly_usd, 5)
        render_fps = round(frames_rendered / max(0.001, blender_render_s), 2)
        ms_per_frame = round((blender_render_s * 1000.0) / max(1, frames_rendered), 2)

        yield {
            "event": "render_done",
            "mode": mode,
            "ext": ext,
            "output_gz": gzip.compress(final_bytes, compresslevel=1),
            "preview_png_b64": preview_b64,
            "engine": str(fin_meta.get("engine", "CYCLES")),
            "backend": str(fin_meta.get("backend", "OPTIX")),
            "device": str(fin_meta.get("device", "GPU")),
            "camera": str(fin_meta.get("camera", "")),
            "frames": frames_rendered,
            "fps": fps,
            "width": width,
            "height": height,
            "blender_render_s": blender_render_s,
            "wall_s": wall_s,
            "render_fps": render_fps,
            "ms_per_frame": ms_per_frame,
            "cost_usd": cost_usd,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _partition_frame_ranges(
    f_start: int, f_end: int, f_step: int, max_shards: int
) -> list[tuple[int, int]]:
    all_frames = list(range(f_start, f_end + 1, max(1, f_step)))
    total = len(all_frames)
    if total <= 1 or max_shards <= 1:
        return [(f_start, f_end)]
    n_s = min(max_shards, total)
    out: list[tuple[int, int]] = []
    for i in range(n_s):
        lo_i = (i * total) // n_s
        hi_i = ((i + 1) * total) // n_s - 1
        if lo_i <= hi_i:
            out.append((all_frames[lo_i], all_frames[hi_i]))
    return out or [(f_start, f_end)]


async def _run_single_modal_shard(
    *,
    cls_name: str,
    eff_gpu: str,
    blend_payload: bytes,
    shard_spec: dict[str, Any],
    timeout_sec: int | None = None,
    on_shard_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    import modal

    from blenderbot.config.env import resolve_modal_worker_timeout

    eff_timeout = (
        max(1, int(timeout_sec))
        if timeout_sec is not None and timeout_sec > 0
        else resolve_modal_worker_timeout(default=1800)
    )

    q: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _producer() -> None:
        try:
            worker_cls = modal.Cls.from_name("blenderbot-splat", cls_name)
            with_opts = getattr(worker_cls, "with_options", None)
            if callable(with_opts):
                worker_cls = with_opts(timeout=eff_timeout)
            worker = worker_cls()
            gen = worker.render_stream.remote_gen(
                blend_gz=blend_payload,
                render_spec=shard_spec,
                gpu=eff_gpu,
            )
            for item in gen:
                q.put(("ev", item))
            q.put(("eof", None))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

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
            raise RuntimeError(f"Modal Blender GPU render failed: {payload}") from payload
        if kind == "eof":
            break
        if kind == "ev":
            if on_shard_event is not None:
                on_shard_event(payload)
            if payload.get("event") == "render_done":
                done_ev = payload

    if done_ev is None or not done_ev.get("output_gz"):
        raise RuntimeError("Modal Blender GPU render shard ended without output artifact.")
    return done_ev


async def run_modal_blender_render(
    *,
    blend_path: Path,
    output_path: Path,
    mode: str = "animation",
    engine: str = "CYCLES",
    fmt: str | None = None,
    gpu: str = "H100",
    samples: int | None = None,
    denoise: bool = True,
    use_persistent_data: bool = True,
    film_transparent: bool | None = None,
    resolution: list[int] | None = None,
    frame: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_step: int | None = None,
    camera: str | None = None,
    scene: str | None = None,
    quality: str = "HIGH",
    shards: int | None = None,
    timeout_sec: float | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> ModalBlenderRenderResult:
    """Upload `.blend` to Modal H100/L40S (parallelizing multi-frame animations across GPU shards!)."""
    eff_gpu = str(gpu or "H100").strip().upper()
    cls_name = "BlenderRenderWorkerL40S" if eff_gpu == "L40S" else "BlenderRenderWorker"

    raw_blend = blend_path.read_bytes()
    blend_payload = (
        gzip.compress(raw_blend, compresslevel=1)
        if raw_blend[:7] == b"BLENDER"
        else raw_blend
    )

    eff_mode = str(mode or "animation").strip().lower()
    if fmt is None:
        ext_l = output_path.suffix.lower()
        if eff_mode == "still":
            fmt = "EXR" if ext_l == ".exr" else ("JPEG" if ext_l in (".jpg", ".jpeg") else "PNG")
        else:
            fmt = "MP4"

    base_spec: dict[str, Any] = {
        "mode": eff_mode,
        "engine": str(engine or "CYCLES").strip().upper(),
        "format": str(fmt).strip().upper(),
        "samples": int(samples) if samples is not None else None,
        "denoise": bool(denoise),
        "use_persistent_data": bool(use_persistent_data),
        "film_transparent": film_transparent,
        "resolution": [int(resolution[0]), int(resolution[1])] if resolution else None,
        "frame": int(frame) if frame is not None else None,
        "frame_start": int(frame_start) if frame_start is not None else None,
        "frame_end": int(frame_end) if frame_end is not None else None,
        "frame_step": int(frame_step) if frame_step is not None else None,
        "camera": camera,
        "scene": scene,
        "quality": str(quality or "HIGH").upper(),
    }

    t_client0 = time.perf_counter()
    step_val = max(1, int(frame_step or 1))
    can_shard = (
        eff_mode == "animation"
        and frame_start is not None
        and frame_end is not None
        and int(frame_end) > int(frame_start)
        and shutil.which("ffmpeg") is not None
    )
    if can_shard:
        total_expected = len(range(int(frame_start), int(frame_end) + 1, step_val))
        eff_shards = (
            max(1, min(16, int(shards)))
            if shards is not None
            else (min(8, max(1, math.ceil(total_expected / 24))) if total_expected >= 16 else 1)
        )
        ranges = _partition_frame_ranges(int(frame_start), int(frame_end), step_val, eff_shards)
    else:
        ranges = [(int(frame_start or 1), int(frame_end or 1))]
        total_expected = 1
        eff_shards = 1

    eff_timeout_int = (
        max(1, round(float(timeout_sec)))
        if timeout_sec is not None and float(timeout_sec) > 0
        else None
    )

    from .modal_deploy import ensure_modal_deployed

    await ensure_modal_deployed(on_event=on_event)

    if len(ranges) == 1:
        done_ev = await _run_single_modal_shard(
            cls_name=cls_name,
            eff_gpu=eff_gpu,
            blend_payload=blend_payload,
            shard_spec=base_spec,
            timeout_sec=eff_timeout_int,
            on_shard_event=on_event,
        )
        out_bytes = gzip.decompress(done_ev["output_gz"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(out_bytes)
        preview_bytes = (
            base64.b64decode(done_ev["preview_png_b64"])
            if done_ev.get("preview_png_b64")
            else None
        )
        client_wall_s = round(time.perf_counter() - t_client0, 3)
        return ModalBlenderRenderResult(
            output_path=output_path,
            mode=str(done_ev.get("mode", eff_mode)),
            engine=str(done_ev.get("engine", base_spec["engine"])),
            backend=str(done_ev.get("backend", "CUDA")),
            device=str(done_ev.get("device", eff_gpu)),
            camera=str(done_ev.get("camera", camera or "")),
            frames=int(done_ev.get("frames", 1)),
            fps=float(done_ev.get("fps", 24.0)),
            width=int(done_ev.get("width", 1920)),
            height=int(done_ev.get("height", 1080)),
            wall_s=client_wall_s,
            blender_render_s=float(done_ev.get("blender_render_s", client_wall_s)),
            render_fps=float(done_ev.get("render_fps", 0.0)),
            ms_per_frame=float(done_ev.get("ms_per_frame", 0.0)),
            cost_usd=float(done_ev.get("cost_usd", 0.0)),
            shards=1,
            preview_png_bytes=preview_bytes,
        )

    # ── Multi-Container Parallel Shard Fan-Out (`eff_shards` concurrent Modal GPUs!) ──
    n_shards = len(ranges)
    done_per_shard = [0] * n_shards
    stage_emitted = False

    def _make_shard_cb(s_idx: int) -> Callable[[dict[str, Any]], None]:
        def _cb(ev: dict[str, Any]) -> None:
            nonlocal stage_emitted
            if on_event is None:
                return
            if ev.get("event") == "stage" and not stage_emitted:
                stage_emitted = True
                merged_stage = dict(ev)
                merged_stage["frame_start"] = int(frame_start)  # type: ignore[arg-type]
                merged_stage["frame_end"] = int(frame_end)  # type: ignore[arg-type]
                merged_stage["total_frames"] = total_expected
                merged_stage["shards"] = n_shards
                merged_stage["device"] = f"{n_shards}x {ev.get('device', eff_gpu)}"
                on_event(merged_stage)
            elif ev.get("event") == "progress" and ev.get("phase") == "frame_done":
                done_per_shard[s_idx] = int(ev.get("done_frames") or 0)
                agg_done = sum(done_per_shard)
                elapsed = max(0.01, time.perf_counter() - t_client0)
                avg_s = elapsed / max(1, agg_done)
                rem_n = max(0, total_expected - agg_done)
                on_event(
                    {
                        "event": "progress",
                        "phase": "frame_done",
                        "op": "render.modal",
                        "engine": ev.get("engine"),
                        "backend": ev.get("backend"),
                        "device": f"{n_shards}x {ev.get('device', eff_gpu)}",
                        "shards": n_shards,
                        "frame": ev.get("frame"),
                        "done_frames": agg_done,
                        "total_frames": total_expected,
                        "pct": round(100.0 * agg_done / max(1, total_expected), 1),
                        "ms_per_frame": round(avg_s * 1000.0, 1),
                        "eta_s": round(rem_n * avg_s, 1),
                        "elapsed_s": round(elapsed, 2),
                    }
                )

        return _cb

    tasks = []
    for idx, (r_lo, r_hi) in enumerate(ranges):
        s_spec = dict(base_spec)
        s_spec["frame_start"] = r_lo
        s_spec["frame_end"] = r_hi
        tasks.append(
            _run_single_modal_shard(
                cls_name=cls_name,
                eff_gpu=eff_gpu,
                blend_payload=blend_payload,
                shard_spec=s_spec,
                timeout_sec=eff_timeout_int,
                on_shard_event=_make_shard_cb(idx),
            )
        )

    shard_dones = await asyncio.gather(*tasks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bb_modal_concat_") as cat_dir_s:
        cat_dir = Path(cat_dir_s)
        list_lines: list[str] = []
        for idx, d_ev in enumerate(shard_dones):
            seg_p = cat_dir / f"seg_{idx:03d}.mp4"
            seg_p.write_bytes(gzip.decompress(d_ev["output_gz"]))
            safe_posix = seg_p.as_posix().replace("'", "'\\''")
            list_lines.append(f"file '{safe_posix}'")
        concat_txt = cat_dir / "concat.txt"
        concat_txt.write_text("\n".join(list_lines) + "\n", encoding="utf-8")

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        proc = await asyncio.to_thread(
            subprocess.run,
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not output_path.is_file():
            raise RuntimeError(f"FFmpeg shard concat failed: {proc.stderr.strip()}")

    client_wall_s = round(time.perf_counter() - t_client0, 3)
    tot_frames = sum(int(d.get("frames", 0)) for d in shard_dones)
    tot_cost = round(sum(float(d.get("cost_usd", 0.0)) for d in shard_dones), 5)
    max_bl_s = round(max(float(d.get("blender_render_s", 0.0)) for d in shard_dones), 3)
    mid_shard = shard_dones[len(shard_dones) // 2]
    preview_bytes = (
        base64.b64decode(mid_shard["preview_png_b64"])
        if mid_shard.get("preview_png_b64")
        else None
    )
    dev_base = str(shard_dones[0].get("device", eff_gpu))

    return ModalBlenderRenderResult(
        output_path=output_path,
        mode="animation",
        engine=str(shard_dones[0].get("engine", base_spec["engine"])),
        backend=str(shard_dones[0].get("backend", "CUDA")),
        device=f"{n_shards}x {dev_base}",
        camera=str(shard_dones[0].get("camera", camera or "")),
        frames=tot_frames,
        fps=float(shard_dones[0].get("fps", 24.0)),
        width=int(shard_dones[0].get("width", 1920)),
        height=int(shard_dones[0].get("height", 1080)),
        wall_s=client_wall_s,
        blender_render_s=max_bl_s,
        render_fps=round(tot_frames / max(0.01, client_wall_s), 2),
        ms_per_frame=round((client_wall_s * 1000.0) / max(1, tot_frames), 2),
        cost_usd=tot_cost,
        shards=n_shards,
        preview_png_bytes=preview_bytes,
    )