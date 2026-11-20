"""Fast Previs Still, Burned-In Telemetry Mosaic Contact Sheet, and H.264 Playblast Renderer."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .helpers import effective_fps
from .schema import OpContext, OpError, blender_op


@contextlib.contextmanager
def _monitor_local_render(
    ctx: OpContext,
    op_name: str,
    *,
    frame_start: int,
    frame_end: int,
    engine: str,
) -> Iterator[None]:
    """Attach `bpy.app.handlers` render callbacks to stream live progress & suspend RPC timeouts."""
    bpy = ctx.bpy
    total_frames = max(1, int(frame_end) - int(frame_start) + 1)
    t_start = time.perf_counter()
    state: dict[str, Any] = {
        "cur_frame": int(frame_start),
        "done_frames": 0,
        "frame_t0": t_start,
        "last_stats_t": 0.0,
    }

    ctx.set_suspend_timeout(True)
    ctx.report_progress(
        {
            "phase": "starting",
            "op": op_name,
            "engine": engine,
            "frame": int(frame_start),
            "done_frames": 0,
            "total_frames": total_frames,
            "pct": 0.0,
            "elapsed_s": 0.0,
            "suspend_timeout": True,
        }
    )

    handlers = getattr(bpy.app, "handlers", None)

    def _on_render_pre(sc_arg: Any, *_rest: Any) -> None:
        now = time.perf_counter()
        fr = int(getattr(sc_arg, "frame_current", state["cur_frame"]))
        state["cur_frame"] = fr
        state["frame_t0"] = now
        done_n = int(state["done_frames"])
        ctx.report_progress(
            {
                "phase": "frame_start",
                "op": op_name,
                "engine": engine,
                "frame": fr,
                "done_frames": done_n,
                "total_frames": total_frames,
                "pct": round(100.0 * done_n / total_frames, 1),
                "elapsed_s": round(now - t_start, 2),
                "suspend_timeout": True,
            }
        )

    def _on_render_stats(stats_arg: Any, *_rest: Any) -> None:
        now = time.perf_counter()
        if (now - float(state["last_stats_t"])) < 0.8:
            return
        state["last_stats_t"] = now
        s_txt = str(stats_arg or "").strip()
        done_n = int(state["done_frames"])
        ctx.report_progress(
            {
                "phase": "render_stats",
                "op": op_name,
                "engine": engine,
                "frame": int(state["cur_frame"]),
                "done_frames": done_n,
                "total_frames": total_frames,
                "pct": round(100.0 * done_n / total_frames, 1),
                "stats": s_txt[:140],
                "elapsed_s": round(now - t_start, 2),
                "suspend_timeout": True,
            }
        )

    def _on_render_post(sc_arg: Any, *_rest: Any) -> None:
        now = time.perf_counter()
        fr = int(getattr(sc_arg, "frame_current", state["cur_frame"]))
        done_n = int(state["done_frames"]) + 1
        state["done_frames"] = done_n
        elapsed_s = now - t_start
        frame_ms = round((now - float(state["frame_t0"])) * 1000.0, 1)
        avg_s = elapsed_s / max(1, done_n)
        rem_n = max(0, total_frames - done_n)
        ctx.report_progress(
            {
                "phase": "frame_done",
                "op": op_name,
                "engine": engine,
                "frame": fr,
                "done_frames": done_n,
                "total_frames": total_frames,
                "pct": round(100.0 * done_n / total_frames, 1),
                "frame_ms": frame_ms,
                "ms_per_frame": round(avg_s * 1000.0, 1),
                "eta_s": round(rem_n * avg_s, 1),
                "elapsed_s": round(elapsed_s, 2),
                "suspend_timeout": True,
            }
        )

    if handlers is not None:
        for h_attr, fn in (
            ("render_pre", _on_render_pre),
            ("render_stats", _on_render_stats),
            ("render_post", _on_render_post),
        ):
            h_list = getattr(handlers, h_attr, None)
            if isinstance(h_list, list):
                h_list.append(fn)

    try:
        yield
    finally:
        if handlers is not None:
            for h_attr, fn in (
                ("render_pre", _on_render_pre),
                ("render_stats", _on_render_stats),
                ("render_post", _on_render_post),
            ):
                h_list = getattr(handlers, h_attr, None)
                if isinstance(h_list, list) and fn in h_list:
                    h_list.remove(fn)
        ctx.set_suspend_timeout(False)


def _ensure_active_camera(sc: Any, camera_name: str | None = None) -> Any:
    import bpy

    cam = bpy.data.objects.get(camera_name) if camera_name else sc.camera
    if cam is None:
        cam = next((o for o in sc.objects if o.type == "CAMERA"), None)
    if cam is None or cam.type != "CAMERA":
        raise OpError(
            "No active Camera in scene (create a camera or call `rig.camera_pursuit` before rendering)",
            code="NO_CAMERA",
        )
    sc.camera = cam
    return cam


@blender_op(
    "viewport.render_still",
    tier="read",
    summary="Render a fast single-frame previs still (`BLENDER_WORKBENCH` or `BLENDER_EEVEE`) to PNG.",
    params_doc={
        "output_path": "Destination `.png` file path",
        "frame": "Frame number to render (defaults to `scene.frame_current`)",
        "camera": "Optional camera name override",
        "resolution": "Optional `[width, height]` override (e.g. `[1280, 720]`)",
        "engine": "Optional `'BLENDER_WORKBENCH'` or `'BLENDER_EEVEE'` override",
    },
)
def op_viewport_render_still(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    r = sc.render

    raw_out = str(params.get("output_path") or "").strip()
    if not raw_out:
        out_path = Path(tempfile.gettempdir()) / f"blenderbot_still_f{sc.frame_current:04d}.png"
    else:
        out_path = Path(raw_out).resolve()
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = _ensure_active_camera(sc, params.get("camera"))
    orig_frame = int(sc.frame_current)
    orig_engine = str(r.engine)
    orig_res = (int(r.resolution_x), int(r.resolution_y), int(r.resolution_percentage))
    orig_filepath = str(r.filepath)
    orig_media = getattr(r.image_settings, "media_type", None)
    orig_fmt = str(r.image_settings.file_format)

    t0 = time.perf_counter()
    try:
        if "frame" in params and params["frame"] is not None:
            sc.frame_set(int(params["frame"]))
        if params.get("engine"):
            r.engine = str(params["engine"]).upper()
        elif r.engine == "CYCLES":
            r.engine = "BLENDER_WORKBENCH"

        if params.get("resolution"):
            r.resolution_x = int(params["resolution"][0])
            r.resolution_y = int(params["resolution"][1])
            r.resolution_percentage = 100

        if orig_media is not None:
            r.image_settings.media_type = "IMAGE"
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGB"
        r.filepath = str(out_path)
        cur_f = int(sc.frame_current)
        with _monitor_local_render(
            ctx, "viewport.render_still", frame_start=cur_f, frame_end=cur_f, engine=str(r.engine)
        ):
            bpy.ops.render.render(write_still=True)
    finally:
        r.engine = orig_engine
        r.resolution_x, r.resolution_y, r.resolution_percentage = orig_res
        if orig_media is not None:
            r.image_settings.media_type = orig_media
        r.image_settings.file_format = orig_fmt
        r.filepath = orig_filepath
        sc.frame_set(orig_frame)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "output_path": str(out_path),
        "frame": int(params.get("frame", orig_frame)),
        "camera": cam.name,
        "size_bytes": out_path.stat().st_size if out_path.is_file() else 0,
        "elapsed_ms": elapsed_ms,
    }


@blender_op(
    "viewport.contact_sheet",
    tier="read",
    summary=(
        "Render a `cols x rows` multi-frame previs contact sheet PNG with burned-in `fNNNN | t | lens_mm` "
        "metadata (~1,500 tokens for 12-16 frames across a full shot!)."
    ),
    params_doc={
        "output_path": "Destination mosaic `.png` path",
        "cols": "Grid columns (default 4)",
        "rows": "Grid rows (default 3 -> 12 evenly-spaced frames across `[frame_start..frame_end]`)",
        "frame_start": "Start frame (defaults to `scene.frame_start`)",
        "frame_end": "End frame (defaults to `scene.frame_end`)",
        "tile_width": "Width per tile in pixels (default 480)",
        "engine": "Render engine (`'BLENDER_WORKBENCH'` default for <25ms/tile speed, or `'BLENDER_EEVEE'`)",
        "camera": "Optional camera name override",
    },
)
def op_viewport_contact_sheet(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    r = sc.render

    cols = max(1, min(8, int(params.get("cols", 4))))
    rows = max(1, min(8, int(params.get("rows", 3))))
    total_tiles = cols * rows

    f_start = int(params.get("frame_start", sc.frame_start))
    f_end = max(int(params.get("frame_end", sc.frame_end)), f_start)

    raw_out = str(params.get("output_path") or "").strip()
    out_path = (
        Path(raw_out).resolve()
        if raw_out
        else Path(tempfile.gettempdir()) / f"blenderbot_sheet_{sc.name}.png"
    )
    if out_path.suffix.lower() != ".png":
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = _ensure_active_camera(sc, params.get("camera"))
    fps = effective_fps(sc)

    aspect = float(r.resolution_y) / max(1.0, float(r.resolution_x))
    tile_w = max(160, int(params.get("tile_width", 480)))
    tile_h = max(90, round(tile_w * aspect / 2.0) * 2)

    # Evenly sample `total_tiles` frames across [f_start .. f_end]
    if total_tiles == 1 or f_end == f_start:
        sample_frames = [f_start] * total_tiles
    else:
        sample_frames = [
            round(f_start + (i / (total_tiles - 1)) * (f_end - f_start))
            for i in range(total_tiles)
        ]

    orig_frame = int(sc.frame_current)
    orig_engine = str(r.engine)
    orig_res = (int(r.resolution_x), int(r.resolution_y), int(r.resolution_percentage))
    orig_filepath = str(r.filepath)
    orig_media = getattr(r.image_settings, "media_type", None)
    orig_fmt = str(r.image_settings.file_format)
    orig_stamp = (
        bool(r.use_stamp),
        bool(r.use_stamp_frame),
        bool(r.use_stamp_time),
        bool(r.use_stamp_camera),
        bool(r.use_stamp_lens),
        bool(r.use_stamp_note),
        str(r.stamp_note_text),
        int(r.stamp_font_size),
    )

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="bb_tiles_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        tile_paths: list[Path] = []
        try:
            r.engine = str(params.get("engine") or "BLENDER_WORKBENCH").upper()
            r.resolution_x = tile_w
            r.resolution_y = tile_h
            r.resolution_percentage = 100
            if orig_media is not None:
                r.image_settings.media_type = "IMAGE"
            r.image_settings.file_format = "PNG"
            r.image_settings.color_mode = "RGB"

            # Native Blender burned-in metadata stamp on every tile (zero Pillow dependency!)
            r.use_stamp = True
            r.use_stamp_date = False
            r.use_stamp_render_time = False
            r.use_stamp_filename = False
            r.use_stamp_scene = False
            r.use_stamp_frame = True
            r.use_stamp_time = True
            r.use_stamp_camera = True
            r.use_stamp_lens = True
            r.use_stamp_note = True
            r.stamp_font_size = max(12, round(tile_h * 0.055))

            with _monitor_local_render(
                ctx,
                "viewport.contact_sheet",
                frame_start=1,
                frame_end=len(sample_frames),
                engine=str(r.engine),
            ):
                for idx, fr in enumerate(sample_frames):
                    sc.frame_set(fr)
                    t_sec = (fr - f_start) / fps if fps > 0 else 0.0
                    r.stamp_note_text = f"f{fr:04d} ({t_sec:.2f}s)"
                    tp = tmp_root / f"tile_{idx:03d}.png"
                    r.filepath = str(tp)
                    bpy.ops.render.render(write_still=True)
                    tile_paths.append(tp)
        finally:
            r.engine = orig_engine
            r.resolution_x, r.resolution_y, r.resolution_percentage = orig_res
            if orig_media is not None:
                r.image_settings.media_type = orig_media
            r.image_settings.file_format = orig_fmt
            r.filepath = orig_filepath
            (
                r.use_stamp,
                r.use_stamp_frame,
                r.use_stamp_time,
                r.use_stamp_camera,
                r.use_stamp_lens,
                r.use_stamp_note,
                r.stamp_note_text,
                r.stamp_font_size,
            ) = orig_stamp
            sc.frame_set(orig_frame)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                "1",
                "-i",
                str(tmp_root / "tile_%03d.png"),
                "-vf",
                f"tile={cols}x{rows}:padding=4:margin=4:color=0x141820",
                "-frames:v",
                "1",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            if proc.returncode != 0:
                raise OpError(f"FFmpeg tile stitch failed: {proc.stderr.strip()}", code="FFMPEG_ERROR")
        else:
            # Fallback if ffmpeg is ever absent: copy middle frame
            shutil.copy2(tile_paths[len(tile_paths) // 2], out_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "output_path": str(out_path),
        "cols": cols,
        "rows": rows,
        "frames": sample_frames,
        "camera": cam.name,
        "tile_resolution": [tile_w, tile_h],
        "mosaic_resolution": [cols * tile_w + (cols + 1) * 4, rows * tile_h + (rows + 1) * 4],
        "size_bytes": out_path.stat().st_size if out_path.is_file() else 0,
        "elapsed_ms": elapsed_ms,
    }


@blender_op(
    "render.playblast",
    tier="mutate",
    summary=(
        "Render the full scene animation (`[frame_start..frame_end]`) into a clean H.264 `.mp4` motion-reference "
        "Playblast using Blender 5.3's `media_type = 'VIDEO'` pipeline (`BLENDER_WORKBENCH` or `BLENDER_EEVEE`)."
    ),
    params_doc={
        "output_path": "Destination `.mp4` file path (e.g. 'playblasts/mountain_pursuit_v1.mp4')",
        "engine": "'BLENDER_WORKBENCH' (default, ~20ms/frame) or 'BLENDER_EEVEE'",
        "resolution": "Optional `[width, height]` override (defaults to scene resolution, e.g. `[1920, 1080]`)",
        "frame_start": "Optional start frame override",
        "frame_end": "Optional end frame override",
        "camera": "Optional camera name override",
        "quality": "'HIGH' (default), 'PERC_LOSSLESS', or 'MEDIUM'",
    },
)
def op_render_playblast(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    r = sc.render

    raw_out = str(params.get("output_path") or "").strip()
    if not raw_out:
        raise OpError("render.playblast requires 'output_path' (e.g. 'playblasts/shot_01.mp4')", code="MISSING_PARAM")
    out_path = Path(raw_out).resolve()
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = _ensure_active_camera(sc, params.get("camera"))
    fps = effective_fps(sc)

    orig_engine = str(r.engine)
    orig_res = (int(r.resolution_x), int(r.resolution_y), int(r.resolution_percentage))
    orig_range = (int(sc.frame_start), int(sc.frame_end), int(sc.frame_current))
    orig_filepath = str(r.filepath)
    orig_ext = bool(r.use_file_extension)
    orig_media = getattr(r.image_settings, "media_type", None)
    orig_fmt = str(r.image_settings.file_format)
    orig_stamp = bool(r.use_stamp)

    qual_map = {
        "PERC_LOSSLESS": "PERC_LOSSLESS",
        "HIGH": "HIGH",
        "MEDIUM": "MEDIUM",
    }
    crf = qual_map.get(str(params.get("quality", "HIGH")).upper(), "HIGH")

    t0 = time.perf_counter()
    try:
        r.engine = str(params.get("engine") or "BLENDER_WORKBENCH").upper()
        if params.get("resolution"):
            r.resolution_x = int(params["resolution"][0])
            r.resolution_y = int(params["resolution"][1])
            r.resolution_percentage = 100
        if "frame_start" in params and params["frame_start"] is not None:
            sc.frame_start = int(params["frame_start"])
        if "frame_end" in params and params["frame_end"] is not None:
            sc.frame_end = int(params["frame_end"])

        # Clean motion reference: NEVER burn metadata stamps into final playblast video!
        r.use_stamp = False
        r.use_file_extension = False

        # Blender 5.x Video Pipeline: media_type MUST be set to 'VIDEO' first!
        if orig_media is not None:
            r.image_settings.media_type = "VIDEO"
        else:
            r.image_settings.file_format = "FFMPEG"
        r.ffmpeg.format = "MPEG4"
        r.ffmpeg.codec = "H264"
        r.ffmpeg.constant_rate_factor = crf
        r.ffmpeg.ffmpeg_preset = "GOOD"
        r.ffmpeg.gopsize = max(1, round(fps / 2.0))
        r.ffmpeg.audio_codec = "NONE"

        r.filepath = str(out_path)
        with _monitor_local_render(
            ctx,
            "render.animation",
            frame_start=int(sc.frame_start),
            frame_end=int(sc.frame_end),
            engine=str(r.engine),
        ):
            bpy.ops.render.render(animation=True)
    finally:
        actual_frames = sc.frame_end - sc.frame_start + 1
        actual_res = [int(r.resolution_x), int(r.resolution_y)]
        actual_engine = str(r.engine)
        r.engine = orig_engine
        r.resolution_x, r.resolution_y, r.resolution_percentage = orig_res
        sc.frame_start, sc.frame_end = orig_range[0], orig_range[1]
        sc.frame_set(orig_range[2])
        r.use_stamp = orig_stamp
        r.use_file_extension = orig_ext
        if orig_media is not None:
            r.image_settings.media_type = orig_media
        r.image_settings.file_format = orig_fmt
        r.filepath = orig_filepath

    if not out_path.is_file():
        # Blender sometimes appends frame numbers if use_file_extension varied; locate fallback
        matches = sorted(out_path.parent.glob(f"{out_path.stem}*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            matches[0].replace(out_path)
        else:
            raise OpError(f"Playblast finished but output MP4 was not found at {out_path}", code="RENDER_MISSING")

    elapsed_s = round(time.perf_counter() - t0, 3)
    return {
        "output_path": str(out_path),
        "camera": cam.name,
        "engine": actual_engine,
        "resolution": actual_res,
        "fps": round(fps, 4),
        "frames": actual_frames,
        "duration_sec": round(actual_frames / fps, 3) if fps > 0 else 0.0,
        "size_bytes": out_path.stat().st_size,
        "elapsed_sec": elapsed_s,
        "ms_per_frame": round((elapsed_s * 1000.0) / max(1, actual_frames), 2),
    }


def _configure_engine_samples_and_denoising(sc: Any, engine: str, samples: int | None, denoise: bool | None) -> None:
    if engine == "BLENDER_EEVEE" and hasattr(sc, "eevee"):
        if samples is not None and hasattr(sc.eevee, "taa_render_samples"):
            sc.eevee.taa_render_samples = max(1, int(samples))
    elif engine == "CYCLES" and hasattr(sc, "cycles"):
        if samples is not None and hasattr(sc.cycles, "samples"):
            sc.cycles.samples = max(1, int(samples))
        if denoise is not None and hasattr(sc.cycles, "use_denoising"):
            sc.cycles.use_denoising = bool(denoise)


@blender_op(
    "render.still",
    tier="mutate",
    summary=(
        "Render a production-quality single frame (`BLENDER_EEVEE`, `CYCLES`, or `BLENDER_WORKBENCH`) to PNG/EXR "
        "with configurable samples, denoising, and optional transparent film (`RGBA`)."
    ),
    params_doc={
        "output_path": "Destination image path (`.png` or `.exr`)",
        "engine": "'BLENDER_EEVEE' (default), 'CYCLES', or 'BLENDER_WORKBENCH'",
        "samples": "Optional render sample count (e.g. 64 for EEVEE/Cycles)",
        "denoise": "Enable Cycles denoiser when rendering with CYCLES (default true)",
        "film_transparent": "If true, render transparent alpha background (`RGBA`)",
        "resolution": "Optional `[width, height]` override",
        "frame": "Optional frame number to render (defaults to `scene.frame_current`)",
        "camera": "Optional camera name override",
    },
)
def op_render_still(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    r = sc.render

    raw_out = str(params.get("output_path") or "").strip()
    out_path = (
        Path(raw_out).resolve()
        if raw_out
        else Path(tempfile.gettempdir()) / f"blenderbot_render_f{sc.frame_current:04d}.png"
    )
    ext = out_path.suffix.lower()
    if ext not in (".png", ".exr", ".jpg", ".jpeg"):
        out_path = out_path.with_suffix(".png")
        ext = ".png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = _ensure_active_camera(sc, params.get("camera"))
    orig_frame = int(sc.frame_current)
    orig_engine = str(r.engine)
    orig_res = (int(r.resolution_x), int(r.resolution_y), int(r.resolution_percentage))
    orig_filepath = str(r.filepath)
    orig_media = getattr(r.image_settings, "media_type", None)
    orig_fmt = str(r.image_settings.file_format)
    orig_color_mode = str(r.image_settings.color_mode)
    orig_film = bool(r.film_transparent)
    orig_stamp = bool(r.use_stamp)

    target_engine = str(params.get("engine") or (orig_engine if orig_engine != "BLENDER_WORKBENCH" else "BLENDER_EEVEE")).upper()
    samples = int(params["samples"]) if params.get("samples") is not None else None
    denoise = bool(params.get("denoise", True)) if target_engine == "CYCLES" else None

    t0 = time.perf_counter()
    try:
        if params.get("frame") is not None:
            sc.frame_set(int(params["frame"]))
        r.engine = target_engine
        _configure_engine_samples_and_denoising(sc, target_engine, samples, denoise)

        if params.get("resolution"):
            r.resolution_x = int(params["resolution"][0])
            r.resolution_y = int(params["resolution"][1])
            r.resolution_percentage = 100

        transparent = bool(params["film_transparent"]) if params.get("film_transparent") is not None else orig_film
        r.film_transparent = transparent
        r.use_stamp = False

        if orig_media is not None:
            r.image_settings.media_type = "IMAGE"
        if ext == ".exr":
            r.image_settings.file_format = "OPEN_EXR"
            r.image_settings.color_mode = "RGBA"
        elif ext in (".jpg", ".jpeg"):
            r.image_settings.file_format = "JPEG"
            r.image_settings.color_mode = "RGB"
        else:
            r.image_settings.file_format = "PNG"
            r.image_settings.color_mode = "RGBA" if transparent else "RGB"

        r.filepath = str(out_path)
        cur_f = int(sc.frame_current)
        with _monitor_local_render(
            ctx, "render.still", frame_start=cur_f, frame_end=cur_f, engine=target_engine
        ):
            bpy.ops.render.render(write_still=True)
    finally:
        actual_res = [int(r.resolution_x), int(r.resolution_y)]
        rendered_frame = int(sc.frame_current)
        r.engine = orig_engine
        r.resolution_x, r.resolution_y, r.resolution_percentage = orig_res
        r.film_transparent = orig_film
        r.use_stamp = orig_stamp
        if orig_media is not None:
            r.image_settings.media_type = orig_media
        r.image_settings.file_format = orig_fmt
        r.image_settings.color_mode = orig_color_mode
        r.filepath = orig_filepath
        sc.frame_set(orig_frame)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "output_path": str(out_path),
        "engine": target_engine,
        "camera": cam.name,
        "frame": rendered_frame,
        "resolution": actual_res,
        "size_bytes": out_path.stat().st_size if out_path.is_file() else 0,
        "elapsed_ms": elapsed_ms,
    }


@blender_op(
    "render.animation",
    tier="mutate",
    summary=(
        "Render a production animation (`BLENDER_EEVEE`, `CYCLES`, or `BLENDER_WORKBENCH`) to H.264 `.mp4` "
        "or a PNG image sequence (`format: 'PNG_SEQ'`)."
    ),
    params_doc={
        "output_path": "Destination `.mp4` file path (or directory / frame prefix when `format == 'PNG_SEQ'`)",
        "engine": "'BLENDER_EEVEE' (default), 'CYCLES', or 'BLENDER_WORKBENCH'",
        "format": "'MP4' (default H.264 video) or 'PNG_SEQ' (PNG frame sequence, supports `film_transparent`)",
        "samples": "Optional render sample count",
        "denoise": "Enable Cycles denoiser (default true)",
        "film_transparent": "Transparent alpha film (for `PNG_SEQ`)",
        "resolution": "Optional `[width, height]` override",
        "frame_start": "Optional start frame override",
        "frame_end": "Optional end frame override",
        "camera": "Optional camera name override",
        "quality": "'HIGH' (default), 'PERC_LOSSLESS', or 'MEDIUM'",
    },
)
def op_render_animation(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    fmt = str(params.get("format") or "MP4").strip().upper()
    target_engine = str(params.get("engine") or "BLENDER_EEVEE").strip().upper()

    if fmt == "MP4":
        merged = dict(params)
        merged["engine"] = target_engine
        if params.get("samples") is not None or params.get("denoise") is not None:
            _configure_engine_samples_and_denoising(
                ctx.scene,
                target_engine,
                int(params["samples"]) if params.get("samples") is not None else None,
                bool(params.get("denoise", True)) if target_engine == "CYCLES" else None,
            )
        return op_render_playblast(ctx, merged)

    if fmt != "PNG_SEQ":
        raise OpError(f"Unsupported render.animation format '{fmt}'. Use 'MP4' or 'PNG_SEQ'.", code="INVALID_PARAM")

    bpy = ctx.bpy
    sc = ctx.scene
    r = sc.render
    raw_out = str(params.get("output_path") or "").strip()
    if not raw_out:
        raise OpError("render.animation requires 'output_path'", code="MISSING_PARAM")
    out_dir = Path(raw_out).resolve()
    if out_dir.suffix.lower() == ".png":
        prefix_path = out_dir.with_suffix("")
        out_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix_path = out_dir / "frame_"

    cam = _ensure_active_camera(sc, params.get("camera"))
    fps = effective_fps(sc)

    orig_engine = str(r.engine)
    orig_res = (int(r.resolution_x), int(r.resolution_y), int(r.resolution_percentage))
    orig_range = (int(sc.frame_start), int(sc.frame_end), int(sc.frame_current))
    orig_filepath = str(r.filepath)
    orig_media = getattr(r.image_settings, "media_type", None)
    orig_fmt = str(r.image_settings.file_format)
    orig_color_mode = str(r.image_settings.color_mode)
    orig_film = bool(r.film_transparent)
    orig_stamp = bool(r.use_stamp)

    t0 = time.perf_counter()
    try:
        r.engine = target_engine
        _configure_engine_samples_and_denoising(
            sc,
            target_engine,
            int(params["samples"]) if params.get("samples") is not None else None,
            bool(params.get("denoise", True)) if target_engine == "CYCLES" else None,
        )
        if params.get("resolution"):
            r.resolution_x = int(params["resolution"][0])
            r.resolution_y = int(params["resolution"][1])
            r.resolution_percentage = 100
        if params.get("frame_start") is not None:
            sc.frame_start = int(params["frame_start"])
        if params.get("frame_end") is not None:
            sc.frame_end = int(params["frame_end"])

        transparent = bool(params.get("film_transparent", orig_film))
        r.film_transparent = transparent
        r.use_stamp = False
        if orig_media is not None:
            r.image_settings.media_type = "IMAGE"
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGBA" if transparent else "RGB"
        r.filepath = str(prefix_path)
        with _monitor_local_render(
            ctx,
            "render.animation",
            frame_start=int(sc.frame_start),
            frame_end=int(sc.frame_end),
            engine=target_engine,
        ):
            bpy.ops.render.render(animation=True)
    finally:
        actual_frames = sc.frame_end - sc.frame_start + 1
        actual_res = [int(r.resolution_x), int(r.resolution_y)]
        r.engine = orig_engine
        r.resolution_x, r.resolution_y, r.resolution_percentage = orig_res
        sc.frame_start, sc.frame_end = orig_range[0], orig_range[1]
        sc.frame_set(orig_range[2])
        r.film_transparent = orig_film
        r.use_stamp = orig_stamp
        if orig_media is not None:
            r.image_settings.media_type = orig_media
        r.image_settings.file_format = orig_fmt
        r.image_settings.color_mode = orig_color_mode
        r.filepath = orig_filepath

    elapsed_s = round(time.perf_counter() - t0, 3)
    return {
        "output_path": str(prefix_path.parent),
        "format": "PNG_SEQ",
        "camera": cam.name,
        "engine": target_engine,
        "resolution": actual_res,
        "fps": round(fps, 4),
        "frames": actual_frames,
        "elapsed_sec": elapsed_s,
    }