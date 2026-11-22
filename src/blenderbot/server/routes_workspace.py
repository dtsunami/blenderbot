"""FastAPI routes for workspace media I/O, FFmpeg posters, viewport snapshots, and Blender 5.3 bridge actions."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from blenderbot.media_gen.seedance import SEEDANCE_MIMES, probe_media_info

from .routes_common import SAFE_FILENAME_RE, get_web_state
from .state import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS

router = APIRouter()


@router.get("/workspace/media")
async def api_workspace_media(req: Request) -> dict[str, Any]:
    st = get_web_state(req)
    items, blends = await asyncio.gather(
        asyncio.to_thread(st.scan_workspace_media),
        asyncio.to_thread(st.scan_workspace_blends),
    )
    return {"items": items, "blends": blends, "count": len(items)}


@router.post("/blender/open_blend")
async def api_blender_open_blend(req: Request) -> JSONResponse:
    """Open a workspace or session `.blend` file into Blender 5.3 (checkpointing first)."""
    st = get_web_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    body = await req.json()
    raw_path = str(body.get("path") or body.get("filepath") or "").strip()
    if not raw_path:
        return JSONResponse({"error": "Missing .blend path"}, status_code=400)

    try:
        cand = Path(raw_path)
        sessions_root = (sess.cfg.home / "sessions").resolve()
        if cand.is_absolute() and cand.is_file() and cand.suffix.lower() == ".blend":
            try:
                cand.resolve().relative_to(sessions_root)
                blend_p = cand.resolve()
            except ValueError:
                blend_p = sess.sandbox.resolve_for_read(raw_path)
        else:
            blend_p = sess.sandbox.resolve_for_read(raw_path)

        cp = await sess.journal.create_blend_checkpoint(sess.transport)
        bl_st = await sess.transport.call(
            "scene.open", {"filepath": str(blend_p)}, timeout=60.0
        )
        sess.journal.record_batch_action(
            actor="operator",
            action_type="scene.open",
            summary=f"Opened {blend_p.name}",
            checkpoint=cp,
        )
        sc_obj = bl_st.get("scene") if isinstance(bl_st.get("scene"), dict) else {}
        sess.log.update_blend_association(
            blend_filepath=str(bl_st.get("filepath") or blend_p),
            scene_name=str(sc_obj.get("name") or ""),
        )
        sess.bus.publish("blender-changed", {"connected": True, "mode": sess.transport.mode, "status": bl_st})
        return JSONResponse({"ok": True, "blender": bl_st, "filepath": str(blend_p)})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/blender/save_blend")
async def api_blender_save_blend(req: Request) -> JSONResponse:
    """Save the active Blender 5.3 scene to `.blend` and associate it with the current session."""
    st = get_web_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    raw_path = str(body.get("path") or body.get("filepath") or "").strip()
    try:
        sv_params: dict[str, Any] = {}
        if raw_path:
            dst = sess.sandbox.resolve_for_write(raw_path)
            if dst.suffix.lower() != ".blend":
                dst = dst.with_suffix(".blend")
            sv_params["filepath"] = str(dst)
        res = await sess.transport.call("scene.save", sv_params, timeout=60.0)
        saved_fp = str(res.get("filepath") or "")
        sess.log.update_blend_association(blend_filepath=saved_fp)
        sess.bus.publish("blender-changed", {"connected": True, "mode": sess.transport.mode, "saved": saved_fp})
        return JSONResponse({"ok": True, "result": res, "blends": await asyncio.to_thread(st.scan_workspace_blends)})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.get("/workspace/probe")
async def api_workspace_probe(req: Request, path: str) -> JSONResponse:
    st = get_web_state(req)
    try:
        resolved = st.session.sandbox.resolve_for_read(path)
        info = await asyncio.to_thread(probe_media_info, resolved)
        info["rel_path"] = st.session.sandbox.rel(resolved).replace("\\", "/")
        return JSONResponse(info)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.get("/media/file")
async def api_serve_media_file(req: Request, path: str, download: int = 0) -> Response:
    st = get_web_state(req)
    try:
        target = st.session.sandbox.resolve_for_read(path)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "file not found"}, status_code=404)
    mime = SEEDANCE_MIMES.get(target.suffix.lower()) or "application/octet-stream"
    if download:
        return FileResponse(
            target,
            media_type=mime,
            filename=target.name,
            content_disposition_type="attachment",
        )
    return FileResponse(target, media_type=mime, content_disposition_type="inline")


def _build_video_poster_sync(video_path: Path, cache_dir: Path) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{video_path}:{video_path.stat().st_mtime_ns}".encode()
    ).hexdigest()[:16]
    out_jpg = cache_dir / f"poster_{key}.jpg"
    if out_jpg.is_file() and out_jpg.stat().st_size > 0:
        return out_jpg
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        "0.05",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale='min(480,iw)':-2",
        "-q:v",
        "4",
        str(out_jpg),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=15, check=False)
    if proc.returncode == 0 and out_jpg.is_file() and out_jpg.stat().st_size > 0:
        return out_jpg
    return None


@router.get("/media/poster")
async def api_serve_video_poster(req: Request, path: str) -> Response:
    """Extract and cache a JPEG poster frame via FFmpeg so `.MOV` (HEVC) & `.MP4` thumbnails always render."""
    st = get_web_state(req)
    try:
        target = st.session.sandbox.resolve_for_read(path)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=403)
    if not target.is_file():
        return JSONResponse({"error": "file not found"}, status_code=404)
    thumbs_dir = st.session.log.dir / "thumbs"
    jpg = await asyncio.to_thread(_build_video_poster_sync, target, thumbs_dir)
    if jpg is None:
        return JSONResponse({"error": "poster unavailable"}, status_code=404)
    return FileResponse(jpg, media_type="image/jpeg", content_disposition_type="inline")


@router.post("/workspace/upload")
async def api_upload_media(req: Request, filename: str = "") -> JSONResponse:
    """Zero-dependency raw binary file upload into `assets/inputs/<filename>`."""
    st = get_web_state(req)
    raw_name = filename.strip() or str(req.headers.get("X-Filename") or "upload.bin").strip()
    clean_name = SAFE_FILENAME_RE.sub("_", Path(raw_name).name)
    ext = Path(clean_name).suffix.lower()
    if ext not in (IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS):
        return JSONResponse(
            {
                "error": f"Unsupported media extension '{ext}'. Allowed: images (.png/.jpg/.webp), videos (.mp4/.mov/.webm), audio (.mp3/.wav/.m4a)."
            },
            status_code=400,
        )

    data = await req.body()
    if not data:
        return JSONResponse({"error": "Empty upload body"}, status_code=400)
    if len(data) > 250 * 1024 * 1024:
        return JSONResponse({"error": "File exceeds 250 MB upload limit"}, status_code=413)

    in_dir = st.session.cfg.cwd / "assets" / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    dst = in_dir / clean_name
    if dst.exists():
        stem = dst.stem
        ts = datetime.datetime.now().strftime("%H%M%S")
        dst = in_dir / f"{stem}_{ts}{ext}"

    dst.write_bytes(data)
    rel = st.session.sandbox.rel(dst).replace("\\", "/")
    kind = "image" if ext in IMAGE_EXTS else "video" if ext in VIDEO_EXTS else "audio"
    info = await asyncio.to_thread(probe_media_info, dst)
    item = {
        "path": rel,
        "name": dst.name,
        "kind": kind,
        "ext": ext,
        "size_bytes": len(data),
        "mtime": int(dst.stat().st_mtime),
        "url": st.media_url_for(dst),
        "probe": info,
    }
    st.session.bus.publish("workspace-media-added", item)
    return JSONResponse({"ok": True, "item": item})


@router.post("/workspace/save_snapshot")
async def api_save_viewport_snapshot(req: Request) -> JSONResponse:
    """Save a clean base64 PNG captured from the Center WebGL2 3D/FPS Viewport into `renders/`."""
    import base64 as _b64

    st = get_web_state(req)
    sess = st.session
    body = await req.json()
    raw_url = str(body.get("data_url") or "").strip()
    if not raw_url:
        return JSONResponse({"error": "Missing data_url"}, status_code=400)
    b64_part = raw_url.split(",", 1)[1] if "," in raw_url and raw_url.startswith("data:") else raw_url
    try:
        png_bytes = _b64.b64decode(b64_part)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": f"Invalid base64 PNG: {err}"}, status_code=400)

    prefix = SAFE_FILENAME_RE.sub("_", str(body.get("prefix") or "fps_view").strip())[:24] or "fps_view"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    renders_dir = sess.cfg.cwd / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    dst = renders_dir / f"{prefix}_{ts}.png"
    await asyncio.to_thread(dst.write_bytes, png_bytes)

    rel = sess.sandbox.rel(dst).replace("\\", "/")
    item = {
        "path": rel,
        "folder": "renders",
        "name": dst.name,
        "kind": "image",
        "ext": ".png",
        "size_bytes": len(png_bytes),
        "mtime": int(dst.stat().st_mtime),
        "url": st.media_url_for(dst),
    }
    sess.bus.publish("workspace-media-added", item)
    return JSONResponse({"ok": True, "item": item})


@router.post("/blender/capture")
async def api_blender_capture(req: Request) -> JSONResponse:
    """Render a still PNG or MP4 playblast directly from the active Blender 5.3 session for conditioning."""
    st = get_web_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    body = await req.json()
    kind = str(body.get("kind") or "still").strip().lower()
    engine = str(body.get("engine") or "BLENDER_EEVEE").strip().upper()
    frame = body.get("frame")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    renders_dir = sess.cfg.cwd / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    try:
        if kind == "playblast":
            out_p = renders_dir / f"capture_{ts}.mp4"
            res = await sess.transport.call(
                "render.playblast",
                {"output_path": str(out_p), "engine": engine},
                timeout=300.0,
            )
            m_kind = "video"
        else:
            out_p = renders_dir / f"capture_{ts}.png"
            params_s: dict[str, Any] = {
                "output_path": str(out_p),
                "engine": engine,
                "samples": 32,
            }
            if frame is not None:
                params_s["frame"] = int(frame)
            res = await sess.transport.call("render.still", params_s, timeout=180.0)
            m_kind = "image"

        rel = sess.sandbox.rel(out_p).replace("\\", "/")
        item = {
            "path": rel,
            "name": out_p.name,
            "kind": m_kind,
            "ext": out_p.suffix.lower(),
            "size_bytes": out_p.stat().st_size,
            "mtime": int(out_p.stat().st_mtime),
            "url": st.media_url_for(out_p),
            "render": res,
        }
        sess.bus.publish("workspace-media-added", item)
        return JSONResponse({"ok": True, "item": item})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=500)


@router.post("/blender/connect")
async def api_blender_connect(req: Request) -> JSONResponse:
    """Connect or hot-swap onto the live headed Blender 5.3 GUI Bridge (or headless fallback)."""
    from blenderbot.config.blender_paths import discover_blender

    st = get_web_state(req)
    sess = st.session
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    mode = str(body.get("mode") or "auto").strip().lower()
    try:
        info = await sess.reconnect_blender(mode)
        notice_msg = f"Blender 5.3 connected [{info.get('mode')}] (PID {info.get('pid')})"
        gui_hint: str | None = None
        if mode == "auto" and info.get("mode") == "headless":
            inst = await asyncio.to_thread(discover_blender)
            if inst.blender_running:
                gui_hint = (
                    "GUI Blender is open without an active bridge — restart Blender or enable "
                    "'blenderbot Bridge' in Edit -> Preferences -> Add-ons, then click again."
                )
                notice_msg = f"{notice_msg} — {gui_hint}"
        sess.bus.publish("blender-changed", info)
        sess.bus.publish(
            "transcript",
            {
                "type": "notice",
                "origin": "web",
                "message": notice_msg,
            },
        )
        return JSONResponse({"ok": True, "blender": info, "hint": gui_hint})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)