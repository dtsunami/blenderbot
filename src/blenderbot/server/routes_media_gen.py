"""FastAPI routes for OpenRouter Seedream 5.5 image generation and Seedance 2.5 video generation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from blenderbot.context.media import load_image_part, load_video_part
from blenderbot.media_gen.seedance import (
    SeedanceError,
    default_seedance_output_path,
    fetch_openrouter_video_models,
    generate_seedance_video,
)
from blenderbot.media_gen.seedream import (
    SeedreamError,
    default_seedream_output_path,
    fetch_openrouter_image_models,
    generate_seedream_image,
)

from .routes_common import get_web_state, refresh_env_keys

router = APIRouter()


@router.get("/seedream/models")
async def api_seedream_models(req: Request) -> dict[str, Any]:
    st = get_web_state(req)
    or_key = st.session.cfg.keys.get("openrouter")
    models = await fetch_openrouter_image_models(api_key=or_key)
    return {"models": models, "default": st.session.cfg.image_model}


@router.post("/seedream/generate")
async def api_seedream_generate(req: Request) -> JSONResponse:
    """Generate a Seedream 5.5 image (`POST /api/v1/images`) with optional multi-reference conditioning."""
    st = get_web_state(req)
    sess = st.session
    cfg = sess.cfg

    refresh_env_keys(cfg)
    or_key = cfg.keys.get("openrouter")
    if not or_key:
        return JSONResponse(
            {
                "error": "OPENROUTER_API_KEY is not configured. Click 'OpenRouter' in the header or set OPENROUTER_API_KEY in ~/.blenderbot/.env."
            },
            status_code=401,
        )

    body = await req.json()
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "Seedream 5.5 requires a non-empty text prompt."}, status_code=400)

    model = str(body.get("model") or cfg.image_model).strip() or cfg.image_model
    mode = str(body.get("mode") or "standard").strip()
    resolution = str(body.get("resolution") or "").strip() or None
    aspect_ratio = str(body.get("aspect_ratio") or "").strip() or None
    output_format = str(body.get("output_format") or "").strip() or None
    background = str(body.get("background") or "").strip() or None
    seed_raw = body.get("seed")
    seed_val = int(seed_raw) if seed_raw not in (None, "") else None
    attach_ctx = bool(body.get("attach_to_context", True))

    try:
        refs_resolved: list[str | Path] = []
        for r_raw in (body.get("reference_images") or [])[:10]:
            s_r = str(r_raw or "").strip()
            if not s_r:
                continue
            if s_r.startswith(("https://", "http://", "data:")):
                refs_resolved.append(s_r)
            else:
                refs_resolved.append(sess.sandbox.resolve_for_read(s_r))

        ext_hint = ".jpg" if (output_format or "").lower() in ("jpg", "jpeg") else ".png"
        if background and background.lower() == "transparent":
            ext_hint = ".png"
        custom_out = str(body.get("output_path") or "").strip()
        if custom_out:
            if Path(custom_out).suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                custom_out += ext_hint
            out_path = sess.sandbox.resolve_for_write(custom_out)
        else:
            out_path = default_seedream_output_path(cfg.cwd, ext=ext_hint)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    sess.history.add(prompt)
    out_rel = sess.sandbox.rel(out_path).replace("\\", "/")
    job_rec = st.new_job(prompt=prompt, model=model, mode=mode, output_rel=out_rel)
    job_rec.local_id = job_rec.local_id.replace("sd-", "sr-", 1)
    st.jobs.pop(st.job_order[0], None)
    st.job_order[0] = job_rec.local_id
    st.jobs[job_rec.local_id] = job_rec

    async def _runner() -> None:
        sess.log.flight.note("seedream:start", local_id=job_rec.local_id, model=model)
        job_rec.stage = "generating"
        sess.bus.publish("seedance-job", job_rec.to_dict())
        try:
            res = await generate_seedream_image(
                api_key=or_key,
                output_path=out_path,
                model=model,
                prompt=prompt,
                mode=mode,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                output_format=output_format,
                background=background,
                reference_images=refs_resolved,
                seed=seed_val,
                on_progress=lambda _p: sess.log.flight.touch(),
                cancel_event=job_rec.cancel_ev,
            )
            final_p = Path(str(res["output_path"]))
            final_rel = sess.sandbox.rel(final_p).replace("\\", "/")
            cost = float(res.get("cost_usd") or 0.0)
            job_rec.output_rel = final_rel
            job_rec.stage = "completed"
            job_rec.cost_usd = cost
            job_rec.elapsed_sec = float(res.get("elapsed_sec") or 0.0)
            job_rec.width = res.get("width")
            job_rec.height = res.get("height")
            job_rec.conditioning = dict(res.get("conditioning") or {})
            job_rec.media_url = st.media_url_for(final_p)

            sess.agent.run_cost += cost
            sess.agent.stats["cost"] += cost
            sess.log.record("seedream", {**job_rec.to_dict(), "output_path": str(final_p)})
            sess.log.update_totals(
                sess.agent.stats,
                provider=cfg.provider,
                model_id=cfg.model.id,
                health=sess.agent.context_health(),
            )

            if attach_ctx and cfg.model.vision:
                img_part = load_image_part(final_p)
                sess.ledger.add(
                    seg_type="media",
                    label=f"seedream image ({final_p.name}, {res.get('width')}x{res.get('height')})",
                    role="user",
                    image=img_part,
                )

            sess.bus.publish("workspace-media-added", {"path": final_rel, "kind": "image"})
            sess.bus.publish("seedance-job", job_rec.to_dict())
            sess.bus.publish(
                "transcript",
                {
                    "type": "notice",
                    "origin": "web",
                    "message": (
                        f"Seedream 5.5 generated {final_rel} "
                        f"({res.get('width')}x{res.get('height')}, ${cost:.4f})"
                    ),
                },
            )
        except asyncio.CancelledError:
            job_rec.stage = "cancelled"
            job_rec.error = "Cancelled by operator"
            sess.bus.publish("seedance-job", job_rec.to_dict())
        except SeedreamError as sr_err:
            job_rec.stage = "failed"
            job_rec.error = str(sr_err)
            sess.bus.publish("seedance-job", job_rec.to_dict())
            sess.bus.publish(
                "transcript",
                {"type": "warning", "origin": "web", "message": f"Seedream error: {sr_err}"},
            )
        except Exception as err:  # noqa: BLE001
            job_rec.stage = "failed"
            job_rec.error = f"{type(err).__name__}: {err}"
            sess.bus.publish("seedance-job", job_rec.to_dict())

    job_rec.task = asyncio.create_task(_runner())
    return JSONResponse({"ok": True, "job": job_rec.to_dict()}, status_code=202)


@router.get("/seedance/models")
async def api_seedance_models(req: Request) -> dict[str, Any]:
    st = get_web_state(req)
    or_key = st.session.cfg.keys.get("openrouter")
    models = await fetch_openrouter_video_models(api_key=or_key)
    return {"models": models, "default": st.session.cfg.video_model}


@router.post("/seedance/generate")
async def api_seedance_generate(req: Request) -> JSONResponse:
    st = get_web_state(req)
    sess = st.session
    cfg = sess.cfg

    refresh_env_keys(cfg)
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
    st = get_web_state(req)
    rec = st.jobs.get(local_id)
    if rec is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    rec.cancel_ev.set()
    if rec.task and not rec.task.done():
        rec.task.cancel()
    return JSONResponse({"ok": True, "local_id": local_id})