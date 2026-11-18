"""FastAPI routes for Seedance 2.5 video generation, workspace media picking, Blender frame capture, and SSE."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from blenderbot.context.media import load_video_part
from blenderbot.media_gen.seedance import (
    ESTIMATED_COST_PER_SEC_USD,
    SEEDANCE_ASPECT_RATIOS,
    SEEDANCE_MIMES,
    SEEDANCE_RESOLUTIONS,
    SEEDANCE_SIZES,
    SeedanceError,
    default_seedance_output_path,
    fetch_openrouter_video_models,
    generate_seedance_video,
    probe_media_info,
)
from blenderbot.providers.index import mask_key
from blenderbot.session.keys_ctl import set_provider_key

from .state import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS, WebState

router = APIRouter(prefix="/api")

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _get_state(req: Request) -> WebState:
    st: WebState = req.app.state.blenderbot
    return st


@router.get("/state")
async def api_get_state(req: Request) -> dict[str, Any]:
    st = _get_state(req)
    sess = st.session
    cfg = sess.cfg

    bl_info: dict[str, Any] = {"connected": sess.transport is not None}
    if sess.transport is not None:
        bl_info["mode"] = sess.transport.mode
        try:
            pas = await sess.transport.call("scene.passport", {"max_objects": 24}, timeout=4.0)
            bl_info["scene"] = pas.get("scene")
            bl_info["active_camera"] = pas.get("active_camera")
            bl_info["cameras"] = [c.get("name") for c in (pas.get("cameras") or [])]
            bl_info["objects"] = [
                {"name": o.get("name"), "type": o.get("type")} for o in (pas.get("objects") or [])
            ]
        except Exception as err:  # noqa: BLE001
            bl_info["error"] = str(err)

    or_key = cfg.keys.get("openrouter")
    return {
        "cwd": str(cfg.cwd),
        "provider": cfg.provider,
        "model": cfg.model.id,
        "video_capable_agent": cfg.model.video,
        "vision_capable_agent": cfg.model.vision,
        "role": cfg.role,
        "video_model": cfg.video_model,
        "openrouter": {
            "configured": bool(or_key),
            "masked": mask_key(or_key),
        },
        "seedance_spec": {
            "sizes": list(SEEDANCE_SIZES),
            "resolutions": list(SEEDANCE_RESOLUTIONS),
            "aspect_ratios": list(SEEDANCE_ASPECT_RATIOS),
            "cost_per_sec_usd": ESTIMATED_COST_PER_SEC_USD,
        },
        "agent_pricing": {
            "input_per_1m_usd": cfg.model.input_price,
            "output_per_1m_usd": cfg.model.output_price,
        },
        "ledger": {
            "total_tokens": sess.ledger.total_tokens,
            "system_tokens": sess.ledger.system_tokens,
            "limit": sess.ledger.limit,
            "pct": round(sess.ledger.usage_fraction * 100.0, 2),
            "segments": [
                {
                    "id": seg.id,
                    "type": seg.type,
                    "label": seg.label,
                    "tokens": seg.tokens,
                    "pinned": seg.pinned,
                }
                for seg in sess.ledger.segments
            ],
        },
        "prompt_history": sess.history.strings()[-80:],
        "blender": bl_info,
        "stats": dict(sess.agent.stats),
        "health": sess.agent.context_health(),
        "busy": sess.agent.busy,
        "jobs": [st.jobs[jid].to_dict() for jid in st.job_order[:30] if jid in st.jobs],
        "splat_jobs": await asyncio.to_thread(st.scan_splat_jobs),
    }


@router.get("/seedance/models")
async def api_seedance_models(req: Request) -> dict[str, Any]:
    st = _get_state(req)
    or_key = st.session.cfg.keys.get("openrouter")
    models = await fetch_openrouter_video_models(api_key=or_key)
    return {"models": models, "default": st.session.cfg.video_model}


@router.post("/keys/openrouter")
async def api_set_openrouter_key(req: Request) -> JSONResponse:
    st = _get_state(req)
    body = await req.json()
    raw_key = str(body.get("key") or "").strip()
    persist = bool(body.get("save", True))
    if not raw_key:
        return JSONResponse({"error": "API key cannot be empty"}, status_code=400)
    set_provider_key(st.session, "openrouter", raw_key, persist=persist)
    return JSONResponse(
        {
            "ok": True,
            "configured": True,
            "masked": mask_key(st.session.cfg.keys.get("openrouter")),
            "saved": persist,
        }
    )


@router.get("/workspace/media")
async def api_workspace_media(req: Request) -> dict[str, Any]:
    st = _get_state(req)
    items = await asyncio.to_thread(st.scan_workspace_media)
    return {"items": items, "count": len(items)}


@router.get("/workspace/probe")
async def api_workspace_probe(req: Request, path: str) -> JSONResponse:
    st = _get_state(req)
    try:
        resolved = st.session.sandbox.resolve_for_read(path)
        info = await asyncio.to_thread(probe_media_info, resolved)
        info["rel_path"] = st.session.sandbox.rel(resolved).replace("\\", "/")
        return JSONResponse(info)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.get("/media/file")
async def api_serve_media_file(req: Request, path: str, download: int = 0) -> Response:
    st = _get_state(req)
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
    st = _get_state(req)
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
    st = _get_state(req)
    raw_name = filename.strip() or str(req.headers.get("X-Filename") or "upload.bin").strip()
    clean_name = _SAFE_FILENAME_RE.sub("_", Path(raw_name).name)
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


@router.post("/blender/capture")
async def api_blender_capture(req: Request) -> JSONResponse:
    """Render a still PNG or MP4 playblast directly from the active Blender 5.3 session for conditioning."""
    st = _get_state(req)
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


@router.post("/seedance/generate")
async def api_seedance_generate(req: Request) -> JSONResponse:
    st = _get_state(req)
    sess = st.session
    cfg = sess.cfg

    or_key = cfg.keys.get("openrouter")
    if not or_key:
        return JSONResponse(
            {
                "error": "OPENROUTER_API_KEY is not configured. Click 'Set OpenRouter Key' in the header or set OPENROUTER_API_KEY in ~/.blenderbot/.env."
            },
            status_code=401,
        )

    body = await req.json()
    prompt = str(body.get("prompt") or "").strip()
    model = str(body.get("model") or cfg.video_model).strip() or cfg.video_model
    mode = str(body.get("mode") or "standard").strip()
    duration = body.get("duration")
    size = str(body.get("size") or "").strip() or None
    resolution = str(body.get("resolution") or "").strip() or None
    aspect_ratio = str(body.get("aspect_ratio") or "").strip() or None
    generate_audio = body.get("generate_audio")
    seed_raw = body.get("seed")
    seed_val = int(seed_raw) if seed_raw not in (None, "") else None
    attach_ctx = bool(body.get("attach_to_context", False))

    try:
        first_p = (
            sess.sandbox.resolve_for_read(str(body["first_frame"]))
            if body.get("first_frame")
            else None
        )
        last_p = (
            sess.sandbox.resolve_for_read(str(body["last_frame"]))
            if body.get("last_frame")
            else None
        )
        video_p = (
            sess.sandbox.resolve_for_read(str(body["video_path"]))
            if body.get("video_path")
            else None
        )
        video_p2 = (
            sess.sandbox.resolve_for_read(str(body["video_path_2"]))
            if body.get("video_path_2")
            else None
        )
        audio_p = (
            sess.sandbox.resolve_for_read(str(body["audio_path"]))
            if body.get("audio_path")
            else None
        )
        ref_imgs = [
            sess.sandbox.resolve_for_read(str(r))
            for r in (body.get("reference_images") or [])
            if str(r).strip()
        ]
        ref_vids = [
            sess.sandbox.resolve_for_read(str(r))
            for r in (body.get("reference_videos") or [])
            if str(r).strip()
        ]

        custom_out = str(body.get("output_path") or "").strip()
        if custom_out:
            if not custom_out.lower().endswith(".mp4"):
                custom_out += ".mp4"
            out_path = sess.sandbox.resolve_for_write(custom_out)
        else:
            out_path = default_seedance_output_path(cfg.cwd)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    if not prompt and not first_p and not last_p and not video_p and not video_p2:
        return JSONResponse(
            {"error": "Provide a text prompt and/or select a conditioning video or frame image."},
            status_code=400,
        )

    if prompt:
        sess.history.add(prompt)
    out_rel = sess.sandbox.rel(out_path).replace("\\", "/")
    job_rec = st.new_job(prompt=prompt, model=model, mode=mode, output_rel=out_rel)

    def _on_prog(p: dict[str, Any]) -> None:
        incoming_stage = str(p.get("stage") or job_rec.stage)
        # Defer `"completed"` broadcast to `_runner` so `media_url` & `cost_usd` are populated first!
        if incoming_stage == "completed":
            return
        job_rec.stage = incoming_stage
        if p.get("job_id"):
            job_rec.remote_job_id = str(p["job_id"])
        if p.get("elapsed_sec") is not None:
            job_rec.elapsed_sec = float(p["elapsed_sec"])
        if p.get("conditioning"):
            job_rec.conditioning = dict(p["conditioning"])
        sess.log.flight.touch()
        sess.bus.publish("seedance-job", job_rec.to_dict())

    async def _runner() -> None:
        sess.log.flight.note("seedance:start", local_id=job_rec.local_id, model=model)
        try:
            res = await generate_seedance_video(
                api_key=or_key,
                output_path=out_path,
                model=model,
                prompt=prompt or None,
                mode=mode,
                duration=int(duration) if duration not in (None, "") else None,
                size=size,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                first_frame=first_p,
                last_frame=last_p,
                video_path=video_p,
                video_path_2=video_p2,
                dual_video_mode=str(body.get("dual_video_mode") or "bridge_v1_to_v2"),
                video_lock_first=bool(body.get("video_lock_first", True)),
                video_lock_last=bool(body.get("video_lock_last", True)),
                video_mid_frames=int(body.get("video_mid_frames", 2)),
                extract_video_audio=bool(body.get("extract_video_audio", False)),
                reference_images=ref_imgs,
                reference_videos=ref_vids,
                audio_path=audio_p,
                generate_audio=bool(generate_audio) if generate_audio is not None else None,
                seed=seed_val,
                on_progress=_on_prog,
                cancel_event=job_rec.cancel_ev,
            )
            cost = float(res.get("cost_usd") or 0.0)
            job_rec.stage = "completed"
            job_rec.cost_usd = cost
            job_rec.elapsed_sec = float(res.get("elapsed_sec") or job_rec.elapsed_sec)
            job_rec.width = res.get("width")
            job_rec.height = res.get("height")
            job_rec.duration_sec = res.get("duration_sec")
            job_rec.media_url = st.media_url_for(out_path)

            sess.agent.run_cost += cost
            sess.agent.stats["cost"] += cost
            sess.log.record("seedance", {**job_rec.to_dict(), "output_path": str(out_path)})
            sess.log.update_totals(
                sess.agent.stats,
                provider=cfg.provider,
                model_id=cfg.model.id,
                health=sess.agent.context_health(),
            )

            if attach_ctx and cfg.model.video:
                vid_part = load_video_part(out_path)
                sess.ledger.add(
                    seg_type="media",
                    label=f"seedance video ({out_path.name}, {vid_part['delivery']})",
                    role="user",
                    video=vid_part,
                )

            sess.bus.publish("seedance-job", job_rec.to_dict())
            sess.bus.publish(
                "transcript",
                {
                    "type": "notice",
                    "origin": "web",
                    "message": (
                        f"Seedance 2.5 generated {out_rel} "
                        f"({res.get('width')}x{res.get('height')}, {res.get('duration_sec')}s, ${cost:.4f})"
                    ),
                },
            )
        except asyncio.CancelledError:
            job_rec.stage = "cancelled"
            job_rec.error = "Cancelled by operator"
            sess.bus.publish("seedance-job", job_rec.to_dict())
        except SeedanceError as s_err:
            job_rec.stage = "failed"
            job_rec.error = str(s_err)
            sess.bus.publish("seedance-job", job_rec.to_dict())
            sess.bus.publish(
                "transcript",
                {"type": "warning", "origin": "web", "message": f"Seedance error: {s_err}"},
            )
        except Exception as err:  # noqa: BLE001
            job_rec.stage = "failed"
            job_rec.error = f"{type(err).__name__}: {err}"
            sess.bus.publish("seedance-job", job_rec.to_dict())

    job_rec.task = asyncio.create_task(_runner())
    return JSONResponse({"ok": True, "job": job_rec.to_dict()}, status_code=202)


@router.post("/seedance/cancel/{local_id}")
async def api_seedance_cancel(req: Request, local_id: str) -> JSONResponse:
    st = _get_state(req)
    rec = st.jobs.get(local_id)
    if rec is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    rec.cancel_ev.set()
    if rec.task and not rec.task.done():
        rec.task.cancel()
    return JSONResponse({"ok": True, "local_id": local_id})


@router.post("/blender/connect")
async def api_blender_connect(req: Request) -> JSONResponse:
    """Connect or hot-swap onto the live headed Blender 5.3 GUI Bridge (or headless fallback)."""
    st = _get_state(req)
    sess = st.session
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    mode = str(body.get("mode") or "auto").strip().lower()
    try:
        info = await sess.reconnect_blender(mode)
        sess.bus.publish("blender-changed", info)
        sess.bus.publish(
            "transcript",
            {
                "type": "notice",
                "origin": "web",
                "message": f"Blender 5.3 connected [{info.get('mode')}] (PID {info.get('pid')})",
            },
        )
        return JSONResponse({"ok": True, "blender": info})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.get("/splats/jobs")
async def api_splats_jobs(req: Request) -> dict[str, Any]:
    st = _get_state(req)
    jobs = await asyncio.to_thread(st.scan_splat_jobs)
    return {"jobs": jobs, "count": len(jobs)}


@router.post("/splats/train")
async def api_splats_train(req: Request) -> JSONResponse:
    """Launch a Modal H100 / L40S 3DGS job from a workspace video or live Blender 5.3 scene."""
    from blenderbot.splats.client import make_job_id, run_modal_splat_job
    from blenderbot.splats.config import SplatConfig

    st = _get_state(req)
    sess = st.session
    cfg = sess.cfg
    body = await req.json()

    source_mode = str(body.get("source_mode") or "video").strip().lower()
    gpu = str(body.get("gpu") or "H100").strip().upper()
    iterations = max(200, min(30_000, int(body.get("iterations") or 2000)))
    sh_degree = max(0, min(3, int(body.get("sh_degree", 1))))
    auto_import = bool(body.get("auto_import", False))
    render_mode = str(body.get("render_mode") or "ellipsoids").strip().lower()

    video_p: Path | None = None
    source_label = "Blender Scene"
    if source_mode == "video":
        raw_vp = str(body.get("video_path") or "").strip()
        if not raw_vp:
            return JSONResponse({"error": "Select a source video first."}, status_code=400)
        try:
            video_p = sess.sandbox.resolve_for_read(raw_vp)
        except Exception as err:  # noqa: BLE001
            return JSONResponse({"error": str(err)}, status_code=400)
        if not video_p.is_file():
            return JSONResponse({"error": f"Video not found: {raw_vp}"}, status_code=404)
        source_label = sess.sandbox.rel(video_p).replace("\\", "/")
        job_id = make_job_id(video_p)
    else:
        if sess.transport is None:
            ok = await sess.maybe_auto_connect_blender()
            if not ok or sess.transport is None:
                return JSONResponse(
                    {"error": "Blender 5.3 is not connected. Start Blender Bridge or click Blender status pill."},
                    status_code=400,
                )
        raw_targets = str(body.get("targets") or "").strip()
        source_label = f"Blender ({raw_targets or 'active scene'})"
        job_id = make_job_id(Path("blender_scene"))

    out_dir = cfg.cwd / "assets" / "splats" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = st.new_splat_job(
        job_id=job_id,
        source_mode=source_mode,
        source_label=source_label,
        gpu=gpu,
        iterations=iterations,
    )
    sess.bus.publish("splat-job", rec.to_dict())

    splat_cfg = SplatConfig()
    splat_cfg.gpu = gpu
    splat_cfg.train.iterations = iterations
    splat_cfg.train.sh_degree = sh_degree
    if body.get("opa_reg") is not None:
        splat_cfg.train.opa_reg = max(0.0, float(body["opa_reg"]))
    if body.get("prune_opa") is not None:
        splat_cfg.train.prune_opa = max(0.001, min(0.5, float(body["prune_opa"])))
    if body.get("prune_scale3d") is not None:
        splat_cfg.train.prune_scale3d = max(0.005, min(1.0, float(body["prune_scale3d"])))
    if body.get("opacity_reset_every") is not None:
        splat_cfg.train.opacity_reset_every = max(0, int(body["opacity_reset_every"]))
    if body.get("densify_until") is not None and int(body["densify_until"]) > 0:
        splat_cfg.train.densify_until = int(body["densify_until"])
    else:
        splat_cfg.train.densify_until = max(600, int(iterations * 0.72))
    if body.get("max_anisotropy") is not None:
        splat_cfg.train.max_anisotropy = max(2.0, float(body["max_anisotropy"]))
    if body.get("conf_percentile") is not None:
        splat_cfg.pose.conf_percentile = max(0.0, min(90.0, float(body["conf_percentile"])))
    if body.get("pose_backend") in ("vggt", "colmap"):
        splat_cfg.pose.backend = str(body["pose_backend"])

    def _on_splat_ev(ev: dict[str, Any]) -> None:
        sess.log.flight.touch()
        ev_kind = ev.get("event")
        rec.elapsed_sec = time.time() - rec.created_at
        if ev_kind == "stage":
            rec.stage = str(ev.get("stage") or rec.stage)
            if ev.get("init_points"):
                rec.n_gaussians = int(ev["init_points"])
        elif ev_kind == "telemetry":
            rec.stage = "train"
            rec.step = int(ev.get("step") or rec.step)
            rec.total_steps = int(ev.get("total_steps") or rec.total_steps)
            rec.psnr = float(ev.get("psnr") or rec.psnr or 0.0)
            rec.loss = float(ev.get("loss") or rec.loss or 0.0)
            rec.it_per_sec = float(ev.get("it_per_sec") or rec.it_per_sec or 0.0)
            rec.n_gaussians = int(ev.get("n_gaussians") or rec.n_gaussians)
            prev_f = out_dir / "previews" / "latest_render.jpg"
            if prev_f.is_file():
                rec.preview_url = f"{st.media_url_for(prev_f)}&s={rec.step}"
            diff_f = out_dir / "previews" / "latest_diff.jpg"
            if diff_f.is_file():
                rec.diff_url = f"{st.media_url_for(diff_f)}&s={rec.step}"
        sess.bus.publish("splat-job", rec.to_dict())

    async def _runner() -> None:
        sess.log.flight.note("splat:start", job_id=job_id, source=source_mode, gpu=gpu)
        try:
            bundle_p: Path | None = None
            if source_mode == "blender":
                rec.stage = "capture_views"
                sess.bus.publish("splat-job", rec.to_dict())
                assert sess.transport is not None
                bundle_p = out_dir / "views_bundle.tar.gz"
                num_views = max(8, min(96, int(body.get("num_views") or 28)))
                raw_targets = str(body.get("targets") or "").strip()
                targets_list = [t.strip() for t in raw_targets.split(",") if t.strip()] if raw_targets else None
                cap_params: dict[str, Any] = {
                    "output_path": str(bundle_p),
                    "num_views": num_views,
                    "resolution": [640, 640],
                    "engine": "BLENDER_EEVEE",
                    "transparent_bg": bool(body.get("transparent_bg", bool(targets_list))),
                }
                if targets_list:
                    cap_params["targets"] = targets_list
                await sess.transport.call("splat.capture_views", cap_params, timeout=300.0)

            rec.stage = "starting_gpu"
            sess.bus.publish("splat-job", rec.to_dict())

            res = await run_modal_splat_job(
                output_dir=out_dir,
                job_id=job_id,
                video_path=video_p,
                views_bundle_path=bundle_p,
                cfg=splat_cfg,
                on_event=_on_splat_ev,
            )

            rec.stage = "completed"
            rec.step = res.step
            rec.n_gaussians = res.n_gaussians
            rec.cost_usd = res.cost_usd
            rec.elapsed_sec = res.wall_s
            rec.ply_rel = sess.sandbox.rel(res.ply_path).replace("\\", "/")
            rec.cameras_rel = sess.sandbox.rel(res.cameras_path).replace("\\", "/")
            if res.splat_path and res.splat_path.is_file():
                rec.splat_url = st.media_url_for(res.splat_path)
            prev_f = out_dir / "previews" / "latest_render.jpg"
            if prev_f.is_file():
                rec.preview_url = st.media_url_for(prev_f)

            sess.agent.run_cost += res.cost_usd
            sess.agent.stats["cost"] += res.cost_usd
            sess.log.record("splat", rec.to_dict())
            sess.log.update_totals(
                sess.agent.stats,
                provider=cfg.provider,
                model_id=cfg.model.id,
                health=sess.agent.context_health(),
            )

            if auto_import and sess.transport is not None:
                imp = await sess.transport.call(
                    "splat.import",
                    {
                        "filepath": str(res.ply_path),
                        "cameras_json": str(res.cameras_path),
                        "name": f"Splat_{job_id[:18]}",
                        "render_mode": render_mode,
                        "max_points": 350_000,
                        "shadow_catcher_ground": False,
                    },
                    timeout=180.0,
                )
                rec.imported_object = imp.get("object")
                sess.bus.publish("blender-changed", imp)

            sess.bus.publish("splat-job", rec.to_dict())
            sess.bus.publish(
                "transcript",
                {
                    "type": "notice",
                    "origin": "web",
                    "message": (
                        f"3DGS Splat completed: {job_id} "
                        f"({res.n_gaussians:,} Gaussians, PSNR {rec.psnr:.1f}dB, ${res.cost_usd:.4f})"
                    ),
                },
            )
        except asyncio.CancelledError:
            rec.stage = "cancelled"
            rec.error = "Cancelled by operator"
            sess.bus.publish("splat-job", rec.to_dict())
        except Exception as err:  # noqa: BLE001
            rec.stage = "failed"
            rec.error = f"{type(err).__name__}: {err}"
            sess.bus.publish("splat-job", rec.to_dict())
            sess.bus.publish(
                "transcript",
                {"type": "warning", "origin": "web", "message": f"Splat job failed: {err}"},
            )

    rec.task = asyncio.create_task(_runner())
    return JSONResponse({"ok": True, "job": rec.to_dict()}, status_code=202)


@router.post("/splats/import")
async def api_splats_import(req: Request) -> JSONResponse:
    """Import a completed `.ply` (+ `cameras.json`) directly into the active Blender 5.3 session."""
    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    body = await req.json()
    job_id = str(body.get("job_id") or "").strip()
    raw_ply = str(body.get("ply_path") or "").strip()
    render_mode = str(body.get("render_mode") or "ellipsoids").strip().lower()
    max_points = max(10_000, min(1_000_000, int(body.get("max_points") or 350_000)))
    import_cameras = bool(body.get("import_cameras", True))

    if raw_ply:
        ply_p = sess.sandbox.resolve_for_read(raw_ply)
        cam_p = ply_p.parent / "cameras.json"
    elif job_id:
        job_dir = sess.sandbox.resolve_for_read(f"assets/splats/{job_id}")
        ply_p = job_dir / "splat.ply"
        cam_p = job_dir / "cameras.json"
    else:
        return JSONResponse({"error": "Provide job_id or ply_path"}, status_code=400)

    if not ply_p.is_file():
        return JSONResponse({"error": f"PLY not found: {ply_p}"}, status_code=404)

    params: dict[str, Any] = {
        "filepath": str(ply_p),
        "name": str(body.get("name") or f"Splat_{(job_id or ply_p.stem)[:20]}"),
        "render_mode": render_mode,
        "max_points": max_points,
        "min_opacity": float(body.get("min_opacity", 0.03)),
        "max_world_scale": float(body.get("max_world_scale", 0.25)),
        "max_anisotropy": float(body.get("max_anisotropy", 10.0)),
        "shadow_catcher_ground": bool(body.get("shadow_catcher_ground", False)),
    }
    if import_cameras and cam_p.is_file():
        params["cameras_json"] = str(cam_p)

    try:
        res = await sess.transport.call("splat.import", params, timeout=180.0)
        sess.bus.publish("blender-changed", res)
        sess.bus.publish(
            "transcript",
            {
                "type": "notice",
                "origin": "web",
                "message": (
                    f"Imported splat '{res.get('object')}' into Blender 5.3 "
                    f"({res.get('points_imported', 0):,} {render_mode} in {res.get('elapsed_s')}s)"
                ),
            },
        )
        return JSONResponse({"ok": True, "imported": res})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/splats/import_cameras")
async def api_splats_import_cameras(req: Request) -> JSONResponse:
    """Import ONLY the solved Seedance camera trajectory + 3D pose frustums + GT/Error overlay into Blender 5.3."""
    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    body = await req.json()
    job_id = str(body.get("job_id") or "").strip()
    overlay_kind = str(body.get("overlay") or "gt").strip().lower()
    if not job_id:
        return JSONResponse({"error": "Missing job_id"}, status_code=400)

    job_dir = sess.sandbox.resolve_for_read(f"assets/splats/{job_id}")
    cam_p = job_dir / "cameras.json"
    if not cam_p.is_file():
        return JSONResponse({"error": f"cameras.json not found in {job_id}"}, status_code=404)

    overlay_p: Path | None = None
    if overlay_kind == "diff":
        cand = job_dir / "previews" / "latest_diff.jpg"
        if cand.is_file():
            overlay_p = cand
    elif overlay_kind == "gt":
        # Prefer matching Seedance .mp4 if present so scrubbing Blender's timeline scrubs the video live!
        stem_parts = job_id.split("_")
        if len(stem_parts) >= 3 and stem_parts[0] == "seedance":
            vid_cand = sess.cfg.cwd / "assets" / "seedance" / f"{'_'.join(stem_parts[:3])}.mp4"
            if vid_cand.is_file():
                overlay_p = vid_cand
        if overlay_p is None:
            gts = sorted((job_dir / "previews").glob("gt_*.jpg"))
            if gts:
                overlay_p = gts[0]

    params: dict[str, Any] = {
        "cameras_json": str(cam_p),
        "spawn_frustum_rig": True,
    }
    if overlay_p is not None:
        params["overlay_path"] = str(overlay_p)

    try:
        res = await sess.transport.call("splat.import_cameras", params, timeout=60.0)
        sess.bus.publish("blender-changed", res)
        return JSONResponse({"ok": True, "cameras": res})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/splats/resume")
async def api_splats_resume(req: Request) -> JSONResponse:
    """Resume & surgically clean an existing Modal 3DGS job (`prune_opa`, `max_world_scale`, `reset_opacities`)."""
    from blenderbot.splats.client import run_modal_splat_job
    from blenderbot.splats.config import SplatConfig, SurgerySpec

    st = _get_state(req)
    sess = st.session
    cfg = sess.cfg
    body = await req.json()

    job_id = str(body.get("job_id") or "").strip()
    if not job_id:
        return JSONResponse({"error": "Missing job_id"}, status_code=400)

    add_steps = max(200, min(20_000, int(body.get("add_steps") or 2000)))
    gpu = str(body.get("gpu") or "H100").strip().upper()
    out_dir = cfg.cwd / "assets" / "splats" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    surg = SurgerySpec.from_dict(
        {
            "prune_opa": body.get("prune_opa", 0.03),
            "max_world_scale": body.get("max_world_scale", 0.25),
            "clamp_log_scale": body.get("clamp_log_scale"),
            "keep_radius": body.get("keep_radius"),
            "reset_opacities": body.get("reset_opacities"),
        }
    )

    splat_cfg = SplatConfig()
    splat_cfg.gpu = gpu
    if body.get("opa_reg") is not None:
        splat_cfg.train.opa_reg = float(body["opa_reg"])

    rec = st.new_splat_job(
        job_id=job_id,
        source_mode="resume",
        source_label=f"Resume {job_id} (+{add_steps})",
        gpu=gpu,
        iterations=add_steps,
    )
    rec.stage = "resuming"
    sess.bus.publish("splat-job", rec.to_dict())

    def _on_ev(ev: dict[str, Any]) -> None:
        sess.log.flight.touch()
        rec.elapsed_sec = time.time() - rec.created_at
        if ev.get("event") == "telemetry":
            rec.stage = "train"
            rec.step = int(ev.get("step") or rec.step)
            rec.total_steps = int(ev.get("total_steps") or rec.total_steps)
            rec.psnr = float(ev.get("psnr") or rec.psnr)
            rec.n_gaussians = int(ev.get("n_gaussians") or rec.n_gaussians)
            rec.it_per_sec = float(ev.get("it_per_sec") or rec.it_per_sec)
            prev_f = out_dir / "previews" / "latest_render.jpg"
            if prev_f.is_file():
                rec.preview_url = f"{st.media_url_for(prev_f)}&s={rec.step}"
        sess.bus.publish("splat-job", rec.to_dict())

    async def _runner() -> None:
        try:
            res = await run_modal_splat_job(
                output_dir=out_dir,
                job_id=job_id,
                cfg=splat_cfg,
                add_steps=add_steps,
                surgery=surg,
                on_event=_on_ev,
            )
            rec.stage = "completed"
            rec.step = res.step
            rec.total_steps = res.step
            rec.n_gaussians = res.n_gaussians
            rec.cost_usd = res.cost_usd
            rec.elapsed_sec = res.wall_s
            rec.ply_rel = sess.sandbox.rel(res.ply_path).replace("\\", "/")
            if res.splat_path and res.splat_path.is_file():
                rec.splat_url = st.media_url_for(res.splat_path)
            prev_f = out_dir / "previews" / "latest_render.jpg"
            if prev_f.is_file():
                rec.preview_url = st.media_url_for(prev_f)
            sess.agent.run_cost += res.cost_usd
            sess.agent.stats["cost"] += res.cost_usd
            sess.bus.publish("splat-job", rec.to_dict())
        except Exception as err:  # noqa: BLE001
            rec.stage = "failed"
            rec.error = f"{type(err).__name__}: {err}"
            sess.bus.publish("splat-job", rec.to_dict())

    rec.task = asyncio.create_task(_runner())
    return JSONResponse({"ok": True, "job": rec.to_dict()}, status_code=202)


@router.post("/chat/submit")
async def api_chat_submit(req: Request) -> JSONResponse:
    st = _get_state(req)
    body = await req.json()
    text = str(body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty prompt"}, status_code=400)
    asyncio.create_task(st.submit_fn(text, "web"))
    return JSONResponse({"ok": True})


@router.post("/chat/cancel")
async def api_chat_cancel(req: Request) -> JSONResponse:
    st = _get_state(req)
    st.session.agent.cancel()
    return JSONResponse({"ok": True})


@router.delete("/context/{seg_id}")
async def api_drop_context_segment(req: Request, seg_id: int) -> JSONResponse:
    st = _get_state(req)
    ok = st.session.ledger.drop(seg_id)
    if not ok:
        return JSONResponse(
            {"error": f"Segment #{seg_id} could not be dropped (pinned/protected or missing)"},
            status_code=400,
        )
    return JSONResponse({"ok": True, "dropped": seg_id})


@router.get("/events")
async def api_sse_events(req: Request, since: int = 0) -> StreamingResponse:
    st = _get_state(req)
    hdr_id = req.headers.get("Last-Event-ID", "").strip()
    last_seq = int(hdr_id) if hdr_id.isdigit() else int(since or 0)

    q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=512)
    st.sse_queues.add(q)

    def _on_bus(ev: dict[str, Any]) -> None:
        if not q.full():
            q.put_nowait(ev)

    unsub = st.session.bus.subscribe(_on_bus)

    async def _gen() -> AsyncIterator[str]:
        try:
            missed, _gap = st.session.bus.since(last_seq)
            for ev in missed:
                yield f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            while not st.shutdown_ev.is_set():
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if ev is None:
                    break
                yield f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            unsub()
            st.sse_queues.discard(q)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )