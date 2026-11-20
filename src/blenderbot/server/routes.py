"""FastAPI routes for Seedance 2.5 video generation, workspace media picking, Blender frame capture, and SSE."""

from __future__ import annotations

import asyncio
import contextlib
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

from blenderbot.context.media import load_image_part, load_video_part
from blenderbot.media_gen.meshy import (
    DEFAULT_MESHY_MODEL,
    FAL_CREDITS_URL,
    MESHY_ANIMATION_COST_USD,
    MESHY_BASE_COST_USD,
    MESHY_GEOMETRY_RESOLUTIONS,
    MESHY_POSE_MODES,
    MESHY_RIGGING_COST_USD,
    MESHY_SKELETON_TEMPLATES,
    MESHY_SYMMETRY_MODES,
    MESHY_TEXTURE_COST_USD,
    MESHY_TEXTURE_MODELS,
    MESHY_TOPOLOGIES,
    MeshyError,
    default_meshy_job_dir,
    extract_glb_viewport_triangles,
    generate_meshy_3d,
)
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
from blenderbot.media_gen.seedream import (
    ESTIMATED_COST_BY_RES_USD,
    ESTIMATED_COST_PER_IMAGE_USD,
    SEEDREAM_ASPECT_RATIOS,
    SEEDREAM_BACKGROUNDS,
    SEEDREAM_OUTPUT_FORMATS,
    SEEDREAM_RESOLUTIONS,
    SeedreamError,
    default_seedream_output_path,
    fetch_openrouter_image_models,
    generate_seedream_image,
)
from blenderbot.providers.index import mask_key
from blenderbot.session.keys_ctl import set_provider_key

from .state import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS, WebState

router = APIRouter(prefix="/api")

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _get_state(req: Request) -> WebState:
    st: WebState = req.app.state.blenderbot
    return st


def _refresh_env_keys(cfg: Any) -> None:
    """Hot-harvest newly added `.env` keys so running Studio sessions pick up `.blenderbot/.env` edits live."""
    if cfg.keys.get("fal") and cfg.keys.get("openrouter"):
        return
    from blenderbot.config.env import KEY_VARS
    from blenderbot.config.envfile import read_env

    for env_p in (cfg.home / ".env", cfg.cwd / ".env"):
        if not env_p.is_file():
            continue
        env_map = read_env(env_p)
        for k_prov in ("fal", "openrouter"):
            if cfg.keys.get(k_prov):
                continue
            for cand_var in KEY_VARS.get(k_prov, ()):
                val_c = env_map.get(cand_var, "").strip()
                if val_c:
                    cfg.keys[k_prov] = val_c
                    cfg.key_origins[k_prov] = {"name": cand_var, "from": str(env_p)}
                    break


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

    _refresh_env_keys(cfg)
    or_key = cfg.keys.get("openrouter")
    fal_key = cfg.keys.get("fal")
    return {
        "cwd": str(cfg.cwd),
        "provider": cfg.provider,
        "model": cfg.model.id,
        "video_capable_agent": cfg.model.video,
        "vision_capable_agent": cfg.model.vision,
        "role": cfg.role,
        "image_model": cfg.image_model,
        "video_model": cfg.video_model,
        "meshy_model": cfg.meshy_model,
        "openrouter": {
            "configured": bool(or_key),
            "masked": mask_key(or_key),
        },
        "fal": {
            "configured": bool(fal_key),
            "masked": mask_key(fal_key),
            "credits_url": FAL_CREDITS_URL,
        },
        "meshy_spec": {
            "model": DEFAULT_MESHY_MODEL,
            "topologies": list(MESHY_TOPOLOGIES),
            "symmetry_modes": list(MESHY_SYMMETRY_MODES),
            "texture_models": list(MESHY_TEXTURE_MODELS),
            "pose_modes": list(MESHY_POSE_MODES),
            "geometry_resolutions": list(MESHY_GEOMETRY_RESOLUTIONS),
            "skeleton_templates": list(MESHY_SKELETON_TEMPLATES),
            "pricing_usd": {
                "base": MESHY_BASE_COST_USD,
                "texture": MESHY_TEXTURE_COST_USD,
                "rigging": MESHY_RIGGING_COST_USD,
                "animation": MESHY_ANIMATION_COST_USD,
            },
        },
        "seedream_spec": {
            "resolutions": list(SEEDREAM_RESOLUTIONS),
            "aspect_ratios": list(SEEDREAM_ASPECT_RATIOS),
            "output_formats": list(SEEDREAM_OUTPUT_FORMATS),
            "backgrounds": list(SEEDREAM_BACKGROUNDS),
            "cost_per_image_usd": ESTIMATED_COST_PER_IMAGE_USD,
            "cost_by_res_usd": dict(ESTIMATED_COST_BY_RES_USD),
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
        "meshy_jobs": await asyncio.to_thread(st.scan_meshy_jobs),
    }


@router.get("/seedream/models")
async def api_seedream_models(req: Request) -> dict[str, Any]:
    st = _get_state(req)
    or_key = st.session.cfg.keys.get("openrouter")
    models = await fetch_openrouter_image_models(api_key=or_key)
    return {"models": models, "default": st.session.cfg.image_model}


@router.post("/seedream/generate")
async def api_seedream_generate(req: Request) -> JSONResponse:
    """Generate a Seedream 5.5 image (`POST /api/v1/images`) with optional multi-reference conditioning."""
    st = _get_state(req)
    sess = st.session
    cfg = sess.cfg

    _refresh_env_keys(cfg)
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


@router.post("/keys/fal")
async def api_set_fal_key(req: Request) -> JSONResponse:
    st = _get_state(req)
    body = await req.json()
    raw_key = str(body.get("key") or "").strip()
    persist = bool(body.get("save", True))
    if not raw_key:
        return JSONResponse({"error": "FAL_KEY cannot be empty"}, status_code=400)
    set_provider_key(st.session, "fal", raw_key, persist=persist)
    return JSONResponse(
        {
            "ok": True,
            "configured": True,
            "masked": mask_key(st.session.cfg.keys.get("fal")),
            "saved": persist,
        }
    )


@router.get("/meshy/jobs")
async def api_meshy_jobs(req: Request) -> dict[str, Any]:
    st = _get_state(req)
    jobs = await asyncio.to_thread(st.scan_meshy_jobs)
    return {"jobs": jobs, "count": len(jobs)}


@router.post("/meshy/generate")
async def api_meshy_generate(req: Request) -> JSONResponse:
    """Submit a fal `meshy/v7.1/image-to-3d` queue job, track status/logs over SSE, and save `.glb`."""
    st = _get_state(req)
    sess = st.session
    cfg = sess.cfg

    _refresh_env_keys(cfg)
    fal_key = cfg.keys.get("fal")
    if not fal_key:
        return JSONResponse(
            {
                "error": "FAL_KEY is not configured. Click 'fal: Set Key' in the header or set FAL_KEY in ~/.blenderbot/.env."
            },
            status_code=401,
        )

    body = await req.json()
    raw_image = str(body.get("image") or body.get("image_url") or "").strip()
    if not raw_image:
        return JSONResponse(
            {"error": "Meshy 7.1 requires an input image ('image' workspace path, https:// URL, or data: URI)."},
            status_code=400,
        )

    model = str(body.get("model") or cfg.meshy_model or DEFAULT_MESHY_MODEL).strip() or DEFAULT_MESHY_MODEL
    topology = str(body.get("topology") or "triangle").strip()
    target_polycount = int(body.get("target_polycount") or 30000)
    symmetry_mode = str(body.get("symmetry_mode") or "auto").strip()
    should_remesh = bool(body.get("should_remesh", True))
    should_texture = bool(body.get("should_texture", True))
    texture_model = str(body.get("texture_model") or "").strip() or None
    enable_pbr = bool(body.get("enable_pbr", False))
    pose_mode = str(body.get("pose_mode") or "").strip()
    texture_prompt = str(body.get("texture_prompt") or "").strip() or None
    raw_tex_image = str(body.get("texture_image") or body.get("texture_image_url") or "").strip()
    enable_rigging = bool(body.get("enable_rigging", False))
    rigging_height_meters = float(body.get("rigging_height_meters") or 1.7)
    skeleton_template = str(body.get("skeleton_template") or "").strip() or None
    include_fingers = bool(body.get("include_fingers", False))
    enable_animation = bool(body.get("enable_animation", False))
    animation_action_id = int(body.get("animation_action_id") if body.get("animation_action_id") not in (None, "") else 92)
    enable_safety_checker = bool(body.get("enable_safety_checker", True))
    geometry_resolution = str(body.get("geometry_resolution") or "").strip() or None
    auto_import = bool(body.get("auto_import", False))
    attach_ctx = bool(body.get("attach_to_context", True))

    try:
        if raw_image.startswith(("https://", "http://", "data:")):
            resolved_img: str | Path = raw_image
            img_label = "remote_image"
            stem_hint = "meshy"
        else:
            p_img = sess.sandbox.resolve_for_read(raw_image)
            resolved_img = p_img
            img_label = sess.sandbox.rel(p_img).replace("\\", "/")
            stem_hint = p_img.stem

        resolved_tex_img: str | Path | None = None
        if raw_tex_image:
            resolved_tex_img = (
                raw_tex_image
                if raw_tex_image.startswith(("https://", "http://", "data:"))
                else sess.sandbox.resolve_for_read(raw_tex_image)
            )

        custom_out = str(body.get("output_path") or "").strip()
        if custom_out:
            if not custom_out.lower().endswith(".glb"):
                custom_out += ".glb"
            glb_out = sess.sandbox.resolve_for_write(custom_out)
            job_dir = glb_out.parent
            job_dir.mkdir(parents=True, exist_ok=True)
            job_id = job_dir.name if job_dir != cfg.cwd else glb_out.stem
        else:
            job_id, job_dir = default_meshy_job_dir(cfg.cwd, stem=stem_hint)
            glb_out = job_dir / "model.glb"

        # Pre-validate schema synchronously so bad inputs return 400 before queueing
        from blenderbot.media_gen.meshy import prepare_meshy_request

        _, pre_summary = prepare_meshy_request(
            image=resolved_img,
            model=model,
            topology=topology,
            target_polycount=target_polycount,
            symmetry_mode=symmetry_mode,
            should_remesh=should_remesh,
            should_texture=should_texture,
            texture_model=texture_model,
            enable_pbr=enable_pbr,
            pose_mode=pose_mode,
            texture_prompt=texture_prompt,
            texture_image=resolved_tex_img,
            enable_rigging=enable_rigging,
            rigging_height_meters=rigging_height_meters,
            skeleton_template=skeleton_template,
            include_fingers=include_fingers,
            enable_animation=enable_animation,
            animation_action_id=animation_action_id,
            enable_safety_checker=enable_safety_checker,
            geometry_resolution=geometry_resolution,
        )
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    if texture_prompt:
        sess.history.add(texture_prompt)

    out_rel = sess.sandbox.rel(glb_out).replace("\\", "/")
    job_rec = st.new_meshy_job(
        job_id=job_id,
        prompt=texture_prompt or img_label,
        model=model,
        image_rel=img_label,
        output_rel=out_rel,
    )
    job_rec.conditioning = pre_summary

    def _on_prog(ev: dict[str, Any]) -> None:
        sess.log.flight.touch()
        job_rec.stage = str(ev.get("stage") or job_rec.stage)
        if ev.get("request_id"):
            job_rec.request_id = str(ev["request_id"])
        if "queue_position" in ev:
            job_rec.queue_position = ev.get("queue_position")
        if ev.get("logs"):
            for ln in ev["logs"]:
                job_rec.logs.append(str(ln))
        if ev.get("elapsed_sec") is not None:
            job_rec.elapsed_sec = float(ev["elapsed_sec"])
        sess.bus.publish("meshy-job", job_rec.to_dict())

    async def _runner() -> None:
        sess.log.flight.note("meshy:start", job_id=job_rec.job_id, model=model)
        job_rec.stage = "submitting"
        sess.bus.publish("meshy-job", job_rec.to_dict())
        try:
            res = await generate_meshy_3d(
                api_key=fal_key,
                output_dir=job_dir,
                output_path=glb_out,
                image=resolved_img,
                model=model,
                topology=topology,
                target_polycount=target_polycount,
                symmetry_mode=symmetry_mode,
                should_remesh=should_remesh,
                should_texture=should_texture,
                texture_model=texture_model,
                enable_pbr=enable_pbr,
                pose_mode=pose_mode,
                texture_prompt=texture_prompt,
                texture_image=resolved_tex_img,
                enable_rigging=enable_rigging,
                rigging_height_meters=rigging_height_meters,
                skeleton_template=skeleton_template,
                include_fingers=include_fingers,
                enable_animation=enable_animation,
                animation_action_id=animation_action_id,
                enable_safety_checker=enable_safety_checker,
                geometry_resolution=geometry_resolution,
                on_progress=_on_prog,
                cancel_event=job_rec.cancel_ev,
            )
            final_glb = Path(str(res["glb_path"]))
            final_rel = sess.sandbox.rel(final_glb).replace("\\", "/")
            cost = float(res.get("cost_usd") or 0.0)
            job_rec.output_rel = final_rel
            job_rec.stage = "completed"
            job_rec.request_id = res.get("request_id")
            job_rec.cost_usd = cost
            job_rec.elapsed_sec = float(res.get("elapsed_sec") or 0.0)
            job_rec.glb_url = st.media_url_for(final_glb)
            job_rec.triangles_total = int(res.get("triangles_total") or 0)
            job_rec.world_aabb = res.get("world_aabb")

            if res.get("thumbnail_path"):
                tp = Path(str(res["thumbnail_path"]))
                if tp.is_file():
                    job_rec.thumbnail_rel = sess.sandbox.rel(tp).replace("\\", "/")
                    job_rec.thumbnail_url = st.media_url_for(tp)
                    sess.bus.publish("workspace-media-added", {"path": job_rec.thumbnail_rel, "kind": "image"})
                    if attach_ctx and cfg.model.vision:
                        img_part = load_image_part(tp)
                        sess.ledger.add(
                            seg_type="media",
                            label=f"meshy 3D preview ({job_rec.job_id})",
                            role="user",
                            image=img_part,
                        )

            if res.get("rigged_glb_path"):
                rp = Path(str(res["rigged_glb_path"]))
                if rp.is_file():
                    job_rec.rigged_glb_rel = sess.sandbox.rel(rp).replace("\\", "/")
            if res.get("animation_glb_path"):
                ap = Path(str(res["animation_glb_path"]))
                if ap.is_file():
                    job_rec.animation_glb_rel = sess.sandbox.rel(ap).replace("\\", "/")

            if auto_import:
                if sess.transport is None:
                    await sess.maybe_auto_connect_blender()
                if sess.transport is not None:
                    target_imp = (
                        Path(str(res["animation_glb_path"]))
                        if res.get("animation_glb_path")
                        else Path(str(res["rigged_glb_path"]))
                        if res.get("rigged_glb_path")
                        else final_glb
                    )
                    with contextlib.suppress(Exception):
                        imp_res = await sess.transport.call(
                            "asset.import",
                            {
                                "filepath": str(target_imp),
                                "collection": "MeshyAssets",
                                "place_on_ground": True,
                            },
                            timeout=120.0,
                        )
                        job_rec.imported_objects = list(imp_res.get("imported_objects") or [])
                        sess.bus.publish("blender-changed", imp_res)

            sess.agent.run_cost += cost
            sess.agent.stats["cost"] += cost
            sess.log.record("meshy", {**job_rec.to_dict(), "glb_path": str(final_glb)})
            sess.log.update_totals(
                sess.agent.stats,
                provider=cfg.provider,
                model_id=cfg.model.id,
                health=sess.agent.context_health(),
            )

            sess.bus.publish("meshy-job", job_rec.to_dict())
            sess.bus.publish(
                "transcript",
                {
                    "type": "notice",
                    "origin": "web",
                    "message": (
                        f"Meshy 7.1 generated {final_rel} "
                        f"({job_rec.triangles_total:,} tris, req={job_rec.request_id}, ~${cost:.2f})"
                    ),
                },
            )
        except asyncio.CancelledError:
            job_rec.stage = "cancelled"
            job_rec.error = "Cancelled by operator"
            sess.bus.publish("meshy-job", job_rec.to_dict())
        except MeshyError as m_err:
            job_rec.stage = "failed"
            job_rec.error = str(m_err)
            if m_err.request_id:
                job_rec.request_id = m_err.request_id
            sess.bus.publish("meshy-job", job_rec.to_dict())
            sess.bus.publish(
                "transcript",
                {"type": "warning", "origin": "web", "message": f"Meshy 7.1 error: {m_err}"},
            )
        except Exception as err:  # noqa: BLE001
            job_rec.stage = "failed"
            job_rec.error = f"{type(err).__name__}: {err}"
            sess.bus.publish("meshy-job", job_rec.to_dict())

    job_rec.task = asyncio.create_task(_runner())
    return JSONResponse({"ok": True, "job": job_rec.to_dict()}, status_code=202)


@router.post("/meshy/cancel/{job_id}")
async def api_meshy_cancel(req: Request, job_id: str) -> JSONResponse:
    st = _get_state(req)
    rec = st.meshy_jobs.get(job_id)
    if rec is None:
        return JSONResponse({"error": "meshy job not found"}, status_code=404)
    rec.cancel_ev.set()
    if rec.task and not rec.task.done():
        rec.task.cancel()
    return JSONResponse({"ok": True, "job_id": job_id})


@router.get("/meshy/triangles/{job_id}")
async def api_meshy_triangles(req: Request, job_id: str, max_triangles: int = 65000, variant: str = "model") -> JSONResponse:
    """Extract Z-up WebGL2 triangles + sampled baseColor textures from `assets/meshy/<job_id>/*.glb`."""
    st = _get_state(req)
    sess = st.session
    try:
        jdir = sess.sandbox.resolve_for_read(f"assets/meshy/{job_id}")
        fname = (
            "animation.glb"
            if variant == "animation" and (jdir / "animation.glb").is_file()
            else "rigged_character.glb"
            if variant == "rigged" and (jdir / "rigged_character.glb").is_file()
            else "model.glb"
        )
        glb_p = jdir / fname
        if not glb_p.is_file():
            cands = sorted(jdir.glob("*.glb"))
            if not cands:
                return JSONResponse({"error": f"No .glb file found for {job_id}"}, status_code=404)
            glb_p = cands[0]
        res = await asyncio.to_thread(
            extract_glb_viewport_triangles,
            glb_p,
            max_triangles=max_triangles,
            place_on_ground=True,
        )
        return JSONResponse({"ok": True, "job_id": job_id, "glb_rel": sess.sandbox.rel(glb_p).replace("\\", "/"), **res})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/meshy/import")
async def api_meshy_import_blender(req: Request) -> JSONResponse:
    """Import a Meshy `.glb` (`model`, `rigged`, or `animation`) directly into Blender 5.3 on `Z=0`."""
    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    body = await req.json()
    job_id = str(body.get("job_id") or "").strip()
    variant = str(body.get("variant") or "auto").strip().lower()
    target_size = body.get("target_size")
    place_on_ground = bool(body.get("place_on_ground", True))

    try:
        jdir = sess.sandbox.resolve_for_read(f"assets/meshy/{job_id}")
        if variant == "animation" and (jdir / "animation.glb").is_file():
            glb_p = jdir / "animation.glb"
        elif variant in ("rigged", "auto") and (jdir / "rigged_character.glb").is_file() and variant == "rigged":
            glb_p = jdir / "rigged_character.glb"
        elif variant == "auto" and (jdir / "animation.glb").is_file():
            glb_p = jdir / "animation.glb"
        elif variant == "auto" and (jdir / "rigged_character.glb").is_file():
            glb_p = jdir / "rigged_character.glb"
        else:
            glb_p = jdir / "model.glb"
        if not glb_p.is_file():
            return JSONResponse({"error": f"GLB not found: {glb_p.name}"}, status_code=404)

        params_imp: dict[str, Any] = {
            "filepath": str(glb_p),
            "collection": str(body.get("collection") or "MeshyAssets"),
            "place_on_ground": place_on_ground,
        }
        if target_size is not None and str(target_size).strip() != "":
            params_imp["target_size"] = float(target_size)
        if body.get("location") is not None:
            params_imp["location"] = body["location"]

        imp = await sess.transport.call("asset.import", params_imp, timeout=120.0)
        rec = st.meshy_jobs.get(job_id)
        if rec is not None:
            rec.imported_objects = list(imp.get("imported_objects") or [])
            sess.bus.publish("meshy-job", rec.to_dict())
        sess.bus.publish("blender-changed", imp)
        return JSONResponse({"ok": True, "job_id": job_id, "imported": imp})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/meshy/upload_glb")
async def api_meshy_upload_glb(req: Request, filename: str = "") -> JSONResponse:
    """Upload a local `.glb` mesh into `assets/meshy/<job_id>/model.glb` for 3D Stage & Blender import."""
    st = _get_state(req)
    sess = st.session
    raw_name = filename.strip() or str(req.headers.get("X-Filename") or "uploaded.glb").strip()
    clean_name = _SAFE_FILENAME_RE.sub("_", Path(raw_name).name)
    if Path(clean_name).suffix.lower() != ".glb":
        return JSONResponse({"error": "Only .glb binary glTF files are supported here."}, status_code=400)

    data = await req.body()
    if not data:
        return JSONResponse({"error": "Empty .glb body"}, status_code=400)

    job_id, job_dir = default_meshy_job_dir(sess.cfg.cwd, stem=Path(clean_name).stem)
    glb_dst = job_dir / "model.glb"
    await asyncio.to_thread(glb_dst.write_bytes, data)
    stats = await asyncio.to_thread(extract_glb_viewport_triangles, glb_dst, max_triangles=1000)
    out_rel = sess.sandbox.rel(glb_dst).replace("\\", "/")
    rec = st.new_meshy_job(
        job_id=job_id,
        prompt=clean_name,
        model="local/glb",
        image_rel="",
        output_rel=out_rel,
    )
    rec.stage = "completed"
    rec.glb_url = st.media_url_for(glb_dst)
    rec.triangles_total = int(stats.get("total_triangles") or 0)
    rec.world_aabb = stats.get("world_aabb")
    sess.bus.publish("meshy-job", rec.to_dict())
    return JSONResponse({"ok": True, "job": rec.to_dict()})


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


@router.post("/workspace/save_snapshot")
async def api_save_viewport_snapshot(req: Request) -> JSONResponse:
    """Save a clean base64 PNG captured from the Center WebGL2 3D/FPS Viewport into `renders/`."""
    import base64 as _b64

    st = _get_state(req)
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

    prefix = _SAFE_FILENAME_RE.sub("_", str(body.get("prefix") or "fps_view").strip())[:24] or "fps_view"
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

    _refresh_env_keys(cfg)
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
    """Launch a Modal H100 / L40S 3DGS job from a workspace video, image folder, or live Blender 5.3 scene."""
    from blenderbot.splats.client import (
        discover_image_files,
        make_job_id,
        run_modal_splat_job,
    )
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
    images_dir_p: Path | None = None
    images_list_p: list[Path] | None = None
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
    elif source_mode == "images":
        raw_ip = str(body.get("images_path") or body.get("folder") or "").strip()
        raw_ilist = body.get("images")
        if not raw_ip and not raw_ilist:
            return JSONResponse({"error": "Select or enter an image folder first."}, status_code=400)
        try:
            if raw_ip:
                cand_ip = sess.sandbox.resolve_for_read(raw_ip)
                if cand_ip.is_dir():
                    images_dir_p = cand_ip
                elif cand_ip.is_file():
                    images_list_p = [cand_ip]
                else:
                    return JSONResponse({"error": f"Image folder not found: {raw_ip}"}, status_code=404)
            if isinstance(raw_ilist, list) and raw_ilist:
                images_list_p = list(images_list_p or []) + [
                    sess.sandbox.resolve_for_read(str(x)) for x in raw_ilist
                ]
            found_imgs = discover_image_files(images_path=images_dir_p, image_paths=images_list_p)
        except Exception as err:  # noqa: BLE001
            return JSONResponse({"error": str(err)}, status_code=400)
        stem_p = images_dir_p if images_dir_p is not None else found_imgs[0].parent
        rel_lbl = sess.sandbox.rel(stem_p).replace("\\", "/")
        source_label = f"{rel_lbl} ({len(found_imgs)} imgs)"
        job_id = make_job_id(stem_p)
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
    if video_p is not None:
        (out_dir / "job_source.json").write_text(
            json.dumps({"video_rel": source_label, "created_at": time.time()}, indent=2),
            encoding="utf-8",
        )
    elif source_mode == "images":
        (out_dir / "job_source.json").write_text(
            json.dumps({"images_rel": source_label, "created_at": time.time()}, indent=2),
            encoding="utf-8",
        )
    rec = st.new_splat_job(
        job_id=job_id,
        source_mode=source_mode,
        source_label=source_label,
        gpu=gpu,
        iterations=iterations,
    )
    sess.bus.publish("splat-job", rec.to_dict())

    splat_cfg = SplatConfig.from_dict(body)
    splat_cfg.gpu = gpu
    splat_cfg.train.iterations = iterations
    splat_cfg.train.sh_degree = sh_degree
    if body.get("densify_until") is None or int(body.get("densify_until") or 0) <= 0:
        splat_cfg.train.densify_until = max(600, int(iterations * 0.72))
    if body.get("pose_backend") in ("vggt", "colmap", "colmap_prior", "known_cameras"):
        splat_cfg.pose.backend = str(body["pose_backend"])

    known_cams_job = str(body.get("known_cameras_job") or "").strip()

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
            rec.l1 = float(ev.get("l1") or rec.l1 or 0.0)
            rec.ssim = float(ev.get("ssim") or rec.ssim or 0.0)
            rec.it_per_sec = float(ev.get("it_per_sec") or rec.it_per_sec or 0.0)
            rec.n_gaussians = int(ev.get("n_gaussians") or rec.n_gaussians)
            if isinstance(ev.get("stage_timings"), dict):
                rec.stage_timings = dict(ev["stage_timings"])
            prev_f = out_dir / "previews" / "latest_render.jpg"
            if prev_f.is_file():
                rec.preview_url = f"{st.media_url_for(prev_f)}&s={rec.step}"
            diff_f = out_dir / "previews" / "latest_diff.jpg"
            if diff_f.is_file():
                rec.diff_url = f"{st.media_url_for(diff_f)}&s={rec.step}"
        sess.bus.publish("splat-job", rec.to_dict())
        if ev_kind in ("telemetry", "densify", "surgery", "checkpoint_saved"):
            scalar_ev = {
                k: v
                for k, v in ev.items()
                if not k.endswith("_b64") and k not in ("ply_gz", "splat_gz", "proxy_glb_gz")
            }
            sess.bus.publish("splat-telemetry", scalar_ev)

    async def _runner() -> None:
        sess.log.flight.note("splat:start", job_id=job_id, source=source_mode, gpu=gpu)
        try:
            bundle_p: Path | None = None
            known_json_str: str | None = None

            if source_mode in ("video", "images") and known_cams_job:
                if known_cams_job == "__blender_active__":
                    if sess.transport is None:
                        await sess.maybe_auto_connect_blender()
                    if sess.transport is not None:
                        exp_c = await sess.transport.call("splat.export_camera_path", {}, timeout=60.0)
                        known_json_str = json.dumps(exp_c["cameras_doc"])
                elif known_cams_job.startswith("shot:"):
                    sh_id = known_cams_job.split(":", 1)[1]
                    kc_p = sess.sandbox.resolve_for_read(f"assets/shots/{sh_id}/cameras.json")
                    if kc_p.is_file():
                        known_json_str = kc_p.read_text(encoding="utf-8")
                else:
                    kc_p = sess.sandbox.resolve_for_read(f"assets/splats/{known_cams_job}/cameras.json")
                    if kc_p.is_file():
                        known_json_str = kc_p.read_text(encoding="utf-8")

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
                images_path=images_dir_p,
                image_paths=images_list_p,
                views_bundle_path=bundle_p,
                known_cameras_json=known_json_str,
                cfg=splat_cfg,
                timeout_sec=float(body["timeout_sec"]) if body.get("timeout_sec") is not None else cfg.modal_worker_timeout,
                on_event=_on_splat_ev,
            )

            rec.stage = "completed"
            rec.step = res.step
            rec.n_gaussians = res.n_gaussians
            rec.cost_usd = res.cost_usd
            rec.elapsed_sec = res.wall_s
            rec.ply_rel = sess.sandbox.rel(res.ply_path).replace("\\", "/")
            rec.cameras_rel = sess.sandbox.rel(res.cameras_path).replace("\\", "/")
            rec.cameras_url = st.media_url_for(res.cameras_path)
            if res.splat_path and res.splat_path.is_file():
                rec.splat_url = st.media_url_for(res.splat_path)
            if res.proxy_glb_path and res.proxy_glb_path.is_file():
                rec.proxy_glb_rel = sess.sandbox.rel(res.proxy_glb_path).replace("\\", "/")
                rec.proxy_glb_url = st.media_url_for(res.proxy_glb_path)
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
        "import_proxy_mesh": bool(body.get("import_proxy_mesh", False)),
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
    """Resume & surgically clean an existing Modal 3DGS job (merging overrides onto saved config!)."""
    from blenderbot.splats.client import run_modal_splat_job
    from blenderbot.splats.config import SurgerySpec

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

    surg_dict: dict[str, Any] = {}
    for sk in (
        "prune_opa",
        "max_world_scale",
        "clamp_log_scale",
        "keep_aabb",
        "drop_aabb",
        "keep_radius",
        "reset_opacities",
        "densify_now",
        "lr_scale",
    ):
        if body.get(sk) is not None:
            surg_dict[sk] = body[sk]
    surg = SurgerySpec.from_dict(surg_dict) if surg_dict else None

    # Only pass explicitly specified overrides so saved `meta.json` settings (e.g. `sh_degree`) survive!
    cfg_overrides: dict[str, Any] = {"gpu": gpu}
    for ck in ("opa_reg", "lambda_dssim", "scale_reg", "aniso_reg", "focus_camera", "rasterize_mode", "refine_poses"):
        if body.get(ck) is not None:
            cfg_overrides[ck] = body[ck]

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
        ev_kind = ev.get("event")
        rec.elapsed_sec = time.time() - rec.created_at
        if ev_kind == "stage":
            rec.stage = str(ev.get("stage") or rec.stage)
        elif ev_kind == "telemetry":
            rec.stage = "train"
            rec.step = int(ev.get("step") or rec.step)
            rec.total_steps = int(ev.get("total_steps") or rec.total_steps)
            rec.psnr = float(ev.get("psnr") or rec.psnr)
            rec.loss = float(ev.get("loss") or rec.loss)
            rec.l1 = float(ev.get("l1") or rec.l1)
            rec.ssim = float(ev.get("ssim") or rec.ssim)
            rec.n_gaussians = int(ev.get("n_gaussians") or rec.n_gaussians)
            rec.it_per_sec = float(ev.get("it_per_sec") or rec.it_per_sec)
            if isinstance(ev.get("stage_timings"), dict):
                rec.stage_timings = dict(ev["stage_timings"])
            prev_f = out_dir / "previews" / "latest_render.jpg"
            if prev_f.is_file():
                rec.preview_url = f"{st.media_url_for(prev_f)}&s={rec.step}"
            diff_f = out_dir / "previews" / "latest_diff.jpg"
            if diff_f.is_file():
                rec.diff_url = f"{st.media_url_for(diff_f)}&s={rec.step}"
        sess.bus.publish("splat-job", rec.to_dict())
        if ev_kind in ("telemetry", "densify", "surgery", "checkpoint_saved"):
            scalar_ev = {
                k: v
                for k, v in ev.items()
                if not k.endswith("_b64") and k not in ("ply_gz", "splat_gz", "proxy_glb_gz")
            }
            sess.bus.publish("splat-telemetry", scalar_ev)

    async def _runner() -> None:
        try:
            res = await run_modal_splat_job(
                output_dir=out_dir,
                job_id=job_id,
                cfg=None,
                cfg_overrides=cfg_overrides,
                add_steps=add_steps,
                surgery=surg,
                timeout_sec=float(body["timeout_sec"]) if body.get("timeout_sec") is not None else cfg.modal_worker_timeout,
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
            if res.proxy_glb_path and res.proxy_glb_path.is_file():
                rec.proxy_glb_rel = sess.sandbox.rel(res.proxy_glb_path).replace("\\", "/")
                rec.proxy_glb_url = st.media_url_for(res.proxy_glb_path)
            prev_f = out_dir / "previews" / "latest_render.jpg"
            if prev_f.is_file():
                rec.preview_url = st.media_url_for(prev_f)
            sess.agent.run_cost += res.cost_usd
            sess.agent.stats["cost"] += res.cost_usd
            sess.bus.publish("splat-job", rec.to_dict())
        except asyncio.CancelledError:
            rec.stage = "cancelled"
            rec.error = "Cancelled by operator"
            sess.bus.publish("splat-job", rec.to_dict())
        except Exception as err:  # noqa: BLE001
            rec.stage = "failed"
            rec.error = f"{type(err).__name__}: {err}"
            sess.bus.publish("splat-job", rec.to_dict())

    rec.task = asyncio.create_task(_runner())
    return JSONResponse({"ok": True, "job": rec.to_dict()}, status_code=202)


@router.post("/splats/cancel/{job_id}")
async def api_splats_cancel(req: Request, job_id: str) -> JSONResponse:
    """Stop a running Modal 3DGS job cleanly (`stop_early` exports `.ply`) or cancel immediately."""
    from blenderbot.splats.client import push_modal_splat_control

    st = _get_state(req)
    body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    hard = bool((body or {}).get("hard", False))
    rec = st.splat_jobs.get(job_id)

    try:
        await push_modal_splat_control(job_id, {"stop_early": True, "paused": False})
    except Exception:  # noqa: BLE001, S110
        pass

    if rec is not None and (hard or rec.stage in ("queued", "capture_views", "starting_gpu")):
        if rec.task and not rec.task.done():
            rec.task.cancel()
        rec.stage = "cancelled"
        st.session.bus.publish("splat-job", rec.to_dict())
    return JSONResponse({"ok": True, "job_id": job_id})


@router.post("/splats/control/{job_id}")
async def api_splats_control(req: Request, job_id: str) -> JSONResponse:
    """Push mid-training live `tt-splat` parameter & surgery updates to Modal (`blenderbot-splat-ctl`)."""
    from blenderbot.splats.client import push_modal_splat_control

    st = _get_state(req)
    body = await req.json()
    allowed = {
        "paused",
        "stop_early",
        "save_checkpoint",
        "lambda_dssim",
        "opa_reg",
        "scale_reg",
        "aniso_reg",
        "lr_scale",
        "focus_camera",
        "prune_opa",
        "clamp_log_scale",
        "reset_opacities",
        "max_world_scale",
    }
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        return JSONResponse({"error": "No valid control fields provided"}, status_code=400)

    merged = await push_modal_splat_control(job_id, updates)
    rec = st.splat_jobs.get(job_id)
    if rec is not None and "paused" in updates:
        rec.stage = "paused" if updates["paused"] else "train"
        st.session.bus.publish("splat-job", rec.to_dict())
    return JSONResponse({"ok": True, "control": merged})


@router.get("/splats/telemetry/{job_id}")
async def api_splats_telemetry(req: Request, job_id: str) -> JSONResponse:
    """Hydrate complete multi-burst `tt-splat` telemetry series, densify/surgery markers & snapshots."""
    st = _get_state(req)
    sess = st.session
    try:
        jdir = sess.sandbox.resolve_for_read(f"assets/splats/{job_id}")
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)
    if not jdir.is_dir():
        return JSONResponse({"error": f"Splat job not found: {job_id}"}, status_code=404)

    telem_p = jdir / "telemetry.jsonl"
    ticks: list[dict[str, Any]] = []
    densify_events: list[dict[str, Any]] = []
    surgery_events: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []

    if telem_p.is_file():
        for ln in telem_p.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            ev_t = row.get("event")
            if ev_t == "telemetry":
                ticks.append(row)
            elif ev_t == "densify":
                densify_events.append(row)
            elif ev_t == "surgery":
                surgery_events.append(row)
            elif ev_t in ("stage", "done", "checkpoint_saved"):
                stages.append(row)

    snap_dir = jdir / "snapshots"
    snapshots: list[dict[str, Any]] = []
    if snap_dir.is_dir():
        for sp in sorted(snap_dir.glob("step_*.jpg")):
            digits = "".join(c for c in sp.stem if c.isdigit())
            s_num = int(digits) if digits else 0
            snapshots.append({"step": s_num, "name": sp.name, "url": st.media_url_for(sp)})

    cam_names: list[str] = []
    cam_p = jdir / "cameras.json"
    if cam_p.is_file():
        try:
            cdoc = json.loads(cam_p.read_text(encoding="utf-8"))
            cam_names = [str(c.get("image_name") or "") for c in (cdoc.get("cameras") or []) if c.get("image_name")]
        except Exception:  # noqa: BLE001, S110
            pass

    proxy_p = jdir / "proxy_collision.glb"
    return JSONResponse({
        "job_id": job_id,
        "ticks": ticks,
        "densify_events": densify_events,
        "surgery_events": surgery_events,
        "stages": stages,
        "snapshots": snapshots,
        "cameras": cam_names,
        "has_proxy_glb": proxy_p.is_file(),
        "proxy_glb_url": st.media_url_for(proxy_p) if proxy_p.is_file() else None,
    })


@router.post("/splats/crop_box")
async def api_splats_crop_box(req: Request) -> JSONResponse:
    """Spawn or read `SplatCropBox` in Blender 5.3, or trigger 1-click surgical box cull/keep + heal."""
    from blenderbot.splats.client import run_modal_splat_job
    from blenderbot.splats.config import SurgerySpec

    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    body = await req.json()
    action = str(body.get("action") or "spawn").strip().lower()
    box_name = str(body.get("name") or "SplatCropBox")

    if action in ("spawn", "read"):
        try:
            res = await sess.transport.call(
                "splat.crop_box",
                {
                    "action": action,
                    "name": box_name,
                    "location": body.get("location"),
                    "dimensions": body.get("dimensions"),
                },
                timeout=30.0,
            )
            sess.bus.publish("blender-changed", res)
            return JSONResponse({"ok": True, "crop_box": res})
        except Exception as err:  # noqa: BLE001
            return JSONResponse({"error": str(err)}, status_code=400)

    # `cull_heal` (`drop_aabb`) or `keep_heal` (`keep_aabb`)
    job_id = str(body.get("job_id") or "").strip()
    if not job_id:
        return JSONResponse({"error": "Select a splat job_id to surgically heal."}, status_code=400)

    try:
        box_info = await sess.transport.call(
            "splat.crop_box", {"action": "read", "name": box_name}, timeout=30.0
        )
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": f"Read SplatCropBox failed: {err}"}, status_code=400)

    bmin, bmax = box_info["bounds_min"], box_info["bounds_max"]
    add_steps = max(200, min(10_000, int(body.get("add_steps") or 1500)))
    gpu = str(body.get("gpu") or "H100").upper()
    out_dir = sess.cfg.cwd / "assets" / "splats" / job_id

    surg_key = "keep_aabb" if action == "keep_heal" else "drop_aabb"
    surg = SurgerySpec.from_dict({surg_key: [bmin, bmax], "prune_opa": body.get("prune_opa", 0.025)})

    rec = st.new_splat_job(
        job_id=job_id,
        source_mode="resume",
        source_label=f"CropBox {surg_key} {job_id} (+{add_steps})",
        gpu=gpu,
        iterations=add_steps,
    )
    rec.stage = "resuming"
    sess.bus.publish("splat-job", rec.to_dict())

    async def _box_runner() -> None:
        try:
            res = await run_modal_splat_job(
                output_dir=out_dir,
                job_id=job_id,
                cfg=None,
                cfg_overrides={"gpu": gpu},
                add_steps=add_steps,
                surgery=surg,
                timeout_sec=sess.cfg.modal_worker_timeout,
                on_event=lambda ev: (
                    sess.log.flight.touch(),
                    setattr(rec, "step", int(ev.get("step") or rec.step)) if ev.get("event") == "telemetry" else None,
                    setattr(rec, "psnr", float(ev.get("psnr") or rec.psnr)) if ev.get("event") == "telemetry" else None,
                    setattr(rec, "n_gaussians", int(ev.get("n_gaussians") or rec.n_gaussians)) if ev.get("event") == "telemetry" else None,
                    sess.bus.publish("splat-job", rec.to_dict()),
                ),
            )
            rec.stage = "completed"
            rec.step = res.step
            rec.total_steps = res.step
            rec.n_gaussians = res.n_gaussians
            rec.cost_usd = res.cost_usd
            rec.elapsed_sec = res.wall_s
            if res.splat_path and res.splat_path.is_file():
                rec.splat_url = st.media_url_for(res.splat_path)
            sess.agent.run_cost += res.cost_usd
            sess.agent.stats["cost"] += res.cost_usd
            sess.bus.publish("splat-job", rec.to_dict())
        except Exception as err:  # noqa: BLE001
            rec.stage = "failed"
            rec.error = str(err)
            sess.bus.publish("splat-job", rec.to_dict())

    rec.task = asyncio.create_task(_box_runner())
    return JSONResponse({"ok": True, "crop_box": box_info, "job": rec.to_dict()}, status_code=202)


@router.post("/splats/proxy/{job_id}")
async def api_splats_proxy(req: Request, job_id: str) -> JSONResponse:
    """Build `proxy_collision.glb` (Surface-Nets iso-mesh) from `splat.ply` and optionally import to Blender."""
    from blenderbot.splats.proxy_mesh import extract_proxy_glb_from_ply

    st = _get_state(req)
    sess = st.session
    body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    jdir = sess.sandbox.resolve_for_read(f"assets/splats/{job_id}")
    ply_p = jdir / "splat.ply"
    if not ply_p.is_file():
        return JSONResponse({"error": f"splat.ply not found in {job_id}"}, status_code=404)

    glb_p = jdir / "proxy_collision.glb"
    grid_res = int((body or {}).get("grid_res") or 48)
    stats: dict[str, Any] = {}
    if not glb_p.is_file() or bool((body or {}).get("rebuild", False)):
        stats = await asyncio.to_thread(
            extract_proxy_glb_from_ply, ply_p, glb_p, grid_res=grid_res
        )

    imported = None
    if bool((body or {}).get("import_blender", True)):
        if sess.transport is None:
            await sess.maybe_auto_connect_blender()
        if sess.transport is not None:
            imported = await sess.transport.call(
                "asset.import",
                {"filepath": str(glb_p), "collection": "SplatProxies"},
                timeout=60.0,
            )
            sess.bus.publish("blender-changed", imported)

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "proxy_glb_rel": sess.sandbox.rel(glb_p).replace("\\", "/"),
        "proxy_glb_url": st.media_url_for(glb_p),
        "stats": stats,
        "imported": imported,
    })


@router.post("/splats/camera_prompt")
async def api_splats_camera_prompt(req: Request) -> JSONResponse:
    """Read active Blender 5.3 camera animation and compile an optical Seedance 2.5 prompt directive."""
    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)
    body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    try:
        res = await sess.transport.call(
            "splat.export_camera_path",
            {"camera": (body or {}).get("camera")},
            timeout=45.0,
        )
        res.pop("cameras_doc", None)
        return JSONResponse({"ok": True, **res})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


def _build_job_trajectory_sync(st: WebState, job_id: str) -> tuple[dict[str, Any], Path | None]:
    from blenderbot.media_gen.seedance import probe_media_info
    from blenderbot.server.state import resolve_source_video_for_job
    from blenderbot.splats.sim3 import interpolate_camera_trajectory

    cwd = st.session.cfg.cwd
    jdir = st.session.sandbox.resolve_for_read(f"assets/splats/{job_id}")
    cam_p = jdir / "cameras.json"
    if not cam_p.is_file():
        raise FileNotFoundError(f"cameras.json not found for {job_id}")

    cdoc = json.loads(cam_p.read_text(encoding="utf-8"))
    src_vid = resolve_source_video_for_job(cwd, job_id)
    fps_ov: float | None = None
    dur_ov: float | None = None
    w_ov: int | None = None
    h_ov: int | None = None

    if src_vid is not None and not cdoc.get("video_meta"):
        vinfo = probe_media_info(src_vid)
        fps_ov = float(vinfo["fps"]) if vinfo.get("fps") else 24.0
        dur_ov = float(vinfo["duration_sec"]) if vinfo.get("duration_sec") else None
        w_ov = int(vinfo["width"]) if vinfo.get("width") else None
        h_ov = int(vinfo["height"]) if vinfo.get("height") else None

    traj = interpolate_camera_trajectory(
        cdoc,
        fps=fps_ov,
        duration_sec=dur_ov,
        target_width=w_ov,
        target_height=h_ov,
    )
    (jdir / "poses_timeline.json").write_text(json.dumps(traj), encoding="utf-8")
    return traj, src_vid


@router.get("/splats/trajectory/{job_id}")
async def api_splats_trajectory(req: Request, job_id: str) -> JSONResponse:
    """Return full-rate interpolated camera path, Yaw/Pitch/Roll/Speed telemetry, markers & synced clip URLs."""
    st = _get_state(req)
    sess = st.session
    try:
        traj, src_vid = await asyncio.to_thread(_build_job_trajectory_sync, st, job_id)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    jdir = sess.cfg.cwd / "assets" / "splats" / job_id
    modal_mp4 = jdir / "renders" / "modal_recreated.mp4"
    blender_mp4 = jdir / "renders" / "blender_recreated.mp4"

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "trajectory": traj,
        "source_video_url": st.media_url_for(src_vid) if src_vid and src_vid.is_file() else None,
        "source_video_rel": sess.sandbox.rel(src_vid).replace("\\", "/") if src_vid and src_vid.is_file() else None,
        "modal_render_url": st.media_url_for(modal_mp4) if modal_mp4.is_file() else None,
        "blender_render_url": st.media_url_for(blender_mp4) if blender_mp4.is_file() else None,
    })


@router.post("/splats/render")
async def api_splats_render(req: Request) -> JSONResponse:
    """Render recreated video camera path (either farmed to Modal H100 `gsplat` OR via Blender 5.3 EEVEE)."""
    from blenderbot.splats.client import run_modal_splat_render
    from blenderbot.splats.sim3 import interpolate_camera_trajectory

    st = _get_state(req)
    sess = st.session
    body = await req.json()
    job_id = str(body.get("job_id") or "").strip()
    if not job_id:
        return JSONResponse({"error": "Missing job_id"}, status_code=400)

    mode = str(body.get("mode") or "modal").strip().lower()
    gpu = str(body.get("gpu") or "H100").strip().upper()
    bl_cam_override = str(body.get("blender_camera") or "").strip()
    jdir = sess.sandbox.resolve_for_read(f"assets/splats/{job_id}")
    ply_p = jdir / "splat.ply"
    renders_dir = jdir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    try:
        if bl_cam_override:
            if sess.transport is None:
                await sess.maybe_auto_connect_blender()
            if sess.transport is None:
                return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)
            exp_c = await sess.transport.call(
                "splat.export_camera_path", {"camera": bl_cam_override}, timeout=60.0
            )
            traj = interpolate_camera_trajectory(exp_c["cameras_doc"])
        else:
            traj, _ = await asyncio.to_thread(_build_job_trajectory_sync, st, job_id)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    if mode == "modal":
        out_mp4 = renders_dir / "modal_recreated.mp4"
        sess.bus.publish(
            "splat-render",
            {"job_id": job_id, "mode": "modal", "stage": "starting", "total_frames": traj["total_frames"]},
        )
        try:
            r_res = await run_modal_splat_render(
                job_id=job_id,
                output_mp4=out_mp4,
                trajectory_doc=traj,
                ply_fallback_path=ply_p if ply_p.is_file() else None,
                gpu=gpu,
                timeout_sec=sess.cfg.modal_worker_timeout,
                on_event=lambda ev: (
                    sess.log.flight.touch(),
                    sess.bus.publish("splat-render", {"job_id": job_id, "mode": "modal", **ev}),
                ),
            )
            sess.agent.run_cost += r_res.cost_usd
            sess.agent.stats["cost"] += r_res.cost_usd
            url = st.media_url_for(out_mp4)
            sess.bus.publish("workspace-media-added", {"path": sess.sandbox.rel(out_mp4)})
            return JSONResponse({
                "ok": True,
                "mode": "modal",
                "job_id": job_id,
                "url": url,
                "frames": r_res.frames,
                "fps": r_res.fps,
                "render_fps": r_res.render_fps,
                "wall_s": r_res.wall_s,
                "cost_usd": r_res.cost_usd,
            })
        except Exception as err:  # noqa: BLE001
            return JSONResponse({"error": str(err)}, status_code=400)

    # `mode == "blender"`: Render matching camera path clip directly inside Blender 5.3 (`BLENDER_EEVEE`)
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    out_mp4 = renders_dir / "blender_recreated.mp4"
    try:
        # Write full-rate interpolated cameras doc so Blender's SeedanceSourceCam matches every single frame 1:1
        synced_cam_json = renders_dir / "cameras_fullrate.json"
        synced_cam_json.write_text(
            json.dumps({
                "job_id": job_id,
                "coordinate_system": "BLENDER_RIGHT_HANDED_Z_UP",
                "pose_backend": traj.get("pose_backend", "vggt"),
                "video_meta": {
                    "fps": traj["fps"],
                    "duration_sec": traj["duration_sec"],
                    "total_frames": traj["total_frames"],
                    "width": traj["width"],
                    "height": traj["height"],
                },
                "cameras": [
                    {
                        "frame_index": i,
                        "timeline_frame": f["frame"],
                        "pts_sec": f["pts_sec"],
                        "location": f["location"],
                        "rotation_quaternion": f["rotation_quaternion"],
                        "lens_mm": f["lens_mm"],
                        "sensor_width_mm": f["sensor_width_mm"],
                        "width": f["width"],
                        "height": f["height"],
                        "fx": f["fx"],
                        "fy": f["fy"],
                        "cx": f["cx"],
                        "cy": f["cy"],
                    }
                    for i, f in enumerate(traj["frames"])
                ],
            }),
            encoding="utf-8",
        )

        if bool(body.get("ensure_imported", True)) and ply_p.is_file():
            await sess.transport.call(
                "splat.import",
                {
                    "filepath": str(ply_p),
                    "cameras_json": str(synced_cam_json),
                    "name": f"Splat_{job_id[:18]}",
                    "render_mode": str(body.get("render_mode") or "ellipsoids"),
                    "max_points": int(body.get("max_points") or 300_000),
                    "shadow_catcher_ground": False,
                },
                timeout=180.0,
            )
        elif not bl_cam_override:
            await sess.transport.call(
                "splat.import_cameras",
                {"cameras_json": str(synced_cam_json), "spawn_frustum_rig": False},
                timeout=60.0,
            )

        t_bl_0 = time.perf_counter()
        pb = await sess.transport.call(
            "render.playblast",
            {
                "output_path": str(out_mp4),
                "camera": bl_cam_override or "SeedanceSourceCam",
                "engine": str(body.get("engine") or "BLENDER_EEVEE"),
                "frame_start": 1,
                "frame_end": int(traj["total_frames"]),
                "resolution": [int(traj["width"]), int(traj["height"])],
            },
            timeout=600.0,
        )
        wall_bl = round(time.perf_counter() - t_bl_0, 2)
        url = st.media_url_for(out_mp4)
        sess.bus.publish("workspace-media-added", {"path": sess.sandbox.rel(out_mp4)})
        return JSONResponse({
            "ok": True,
            "mode": "blender",
            "job_id": job_id,
            "url": url,
            "frames": int(traj["total_frames"]),
            "fps": traj["fps"],
            "wall_s": wall_bl,
            "playblast": pb,
        })
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.get("/shots/list")
async def api_shots_list(req: Request) -> JSONResponse:
    """List exported FPS Camera Rail & Spatial Brief packs (`assets/shots/*`)."""
    st = _get_state(req)
    shots_root = st.session.cfg.cwd / "assets" / "shots"
    items: list[dict[str, Any]] = []
    if shots_root.is_dir():
        for d in sorted(shots_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:40]:
            bj = d / "brief.json"
            if d.is_dir() and bj.is_file():
                try:
                    info = json.loads(bj.read_text(encoding="utf-8"))
                    info["brief_rel"] = f"assets/shots/{d.name}/brief.md"
                    info["usda_rel"] = f"assets/shots/{d.name}/shot.usda"
                    items.append(info)
                except Exception:  # noqa: BLE001, S110
                    pass
    return JSONResponse({"shots": items})


@router.post("/shots/export")
async def api_shots_export(req: Request) -> JSONResponse:
    """Export an FPS / Minecraft Camera Rail + Subject Track + Voxel Blockout + Spatial Brief Pack."""
    from blenderbot.splats.client import run_modal_splat_render
    from blenderbot.splats.shot_pack import export_shot_collateral_pack, make_shot_id

    st = _get_state(req)
    sess = st.session
    body = await req.json()

    shot_id = str(body.get("shot_id") or "").strip() or make_shot_id(str(body.get("name") or "rail"))
    shot_dir = sess.cfg.cwd / "assets" / "shots" / shot_id
    splat_job_id = str(body.get("splat_job_id") or "").strip() or None

    try:
        brief_info = await asyncio.to_thread(
            export_shot_collateral_pack,
            shot_dir=shot_dir,
            shot_id=shot_id,
            cam_nodes=list(body.get("cam_nodes") or []),
            subject_nodes=list(body.get("subject_nodes") or []),
            track_subject=bool(body.get("track_subject", False)),
            voxels=list(body.get("voxels") or []),
            pings=list(body.get("pings") or []),
            splat_job_id=splat_job_id,
            task=str(body.get("task") or ""),
            duration_sec=float(body.get("duration_sec") or 5.0),
            fps=float(body.get("fps") or 24.0),
            width=int(body.get("width") or 1280),
            height=int(body.get("height") or 720),
            grounded_walk=bool(body.get("grounded_walk", False)),
            first_frame_b64=body.get("first_frame_b64"),
            last_frame_b64=body.get("last_frame_b64"),
        )
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    first_p = shot_dir / "first_frame.png"
    last_p = shot_dir / "last_frame.png"
    usda_p = shot_dir / "shot.usda"
    cams_p = shot_dir / "cameras.json"
    glb_p = shot_dir / "blockout.glb"
    brief_p = shot_dir / "brief.md"

    out_payload: dict[str, Any] = {
        "ok": True,
        "shot_id": shot_id,
        "shot_dir_rel": f"assets/shots/{shot_id}",
        "brief": brief_info,
        "brief_rel": f"assets/shots/{shot_id}/brief.md",
        "usda_rel": f"assets/shots/{shot_id}/shot.usda",
        "usda_url": st.media_url_for(usda_p),
        "cameras_rel": f"assets/shots/{shot_id}/cameras.json",
        "first_frame_rel": f"assets/shots/{shot_id}/first_frame.png" if first_p.is_file() else None,
        "first_frame_url": st.media_url_for(first_p) if first_p.is_file() else None,
        "last_frame_rel": f"assets/shots/{shot_id}/last_frame.png" if last_p.is_file() else None,
        "last_frame_url": st.media_url_for(last_p) if last_p.is_file() else None,
    }
    sess.bus.publish("workspace-media-added", {"path": f"assets/shots/{shot_id}"})

    # Optional immediate action: `"sync_blender"` or `"render_modal"`
    action = str(body.get("action") or "").strip().lower()
    if action == "sync_blender":
        if sess.transport is None:
            await sess.maybe_auto_connect_blender()
        if sess.transport is None:
            return JSONResponse({"error": "Shot exported, but Blender 5.3 is not connected."}, status_code=400)
        try:
            imp_cam = await sess.transport.call(
                "splat.import_cameras",
                {"cameras_json": str(cams_p), "name": "DirectorRailCam", "spawn_frustum_rig": True},
                timeout=60.0,
            )
            imp_glb = None
            if glb_p.is_file():
                imp_glb = await sess.transport.call(
                    "asset.import",
                    {"filepath": str(glb_p), "collection": f"Blockout_{shot_id[:14]}"},
                    timeout=60.0,
                )
            sess.bus.publish("blender-changed", imp_cam)
            out_payload["blender_synced"] = {"camera": imp_cam, "blockout": imp_glb}
            if bool(body.get("send_brief_to_agent", False)):
                prompt_msg = (
                    f"Read the spatial shot brief @{sess.sandbox.rel(brief_p)} "
                    f"(camera `DirectorRailCam` and voxel blockout are already imported into Blender 5.3) "
                    "and execute the requested 3D scene tasks."
                )
                asyncio.create_task(st.submit_fn(prompt_msg, "web"))
        except Exception as err:  # noqa: BLE001
            return JSONResponse({"error": f"Blender sync failed: {err}", **out_payload}, status_code=400)

    elif action == "render_modal" and splat_job_id:
        traj = json.loads((shot_dir / "poses_timeline.json").read_text(encoding="utf-8"))
        ply_fb = sess.cfg.cwd / "assets" / "splats" / splat_job_id / "splat.ply"
        out_mp4 = shot_dir / "modal_rail_render.mp4"
        try:
            r_res = await run_modal_splat_render(
                job_id=splat_job_id,
                output_mp4=out_mp4,
                trajectory_doc=traj,
                ply_fallback_path=ply_fb if ply_fb.is_file() else None,
                gpu=str(body.get("gpu") or "H100"),
                timeout_sec=sess.cfg.modal_worker_timeout,
            )
            sess.agent.run_cost += r_res.cost_usd
            sess.agent.stats["cost"] += r_res.cost_usd
            out_payload["modal_render_url"] = st.media_url_for(out_mp4)
            out_payload["modal_render_rel"] = f"assets/shots/{shot_id}/modal_rail_render.mp4"
            out_payload["render_fps"] = r_res.render_fps
            out_payload["wall_s"] = r_res.wall_s
            out_payload["cost_usd"] = r_res.cost_usd
            sess.bus.publish("workspace-media-added", {"path": out_payload["modal_render_rel"]})
        except Exception as err:  # noqa: BLE001
            return JSONResponse({"error": f"Modal rail render failed: {err}", **out_payload}, status_code=400)

    return JSONResponse(out_payload)


@router.get("/fps/default_splat")
async def api_fps_default_splat(
    variant: str = "courtyard",
    fmt: str = "splat",
) -> Response:
    """Serve metric-calibrated Default Cube / Courtyard 3DGS `.splat` or `.ply` benchmark."""
    from blenderbot.splats.splat_edit import build_default_cube_splat

    v = "cube" if variant.strip().lower() == "cube" else "courtyard"
    ply_b, splat_b, _meta = await asyncio.to_thread(build_default_cube_splat, v)
    if fmt.strip().lower() == "ply":
        return Response(
            content=ply_b,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'inline; filename="default_{v}.ply"'},
        )
    return Response(
        content=splat_b,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'inline; filename="default_{v}.splat"'},
    )


@router.post("/splats/ensure_splat/{job_id}")
async def api_splats_ensure_splat(req: Request, job_id: str) -> JSONResponse:
    """Ensure `splat.splat` exists and is 1:1 vertex-index aligned with `splat.ply` for editing."""
    from blenderbot.splats.splat_edit import ensure_aligned_splat_file

    st = _get_state(req)
    sess = st.session
    try:
        jdir = sess.sandbox.resolve_for_read(f"assets/splats/{job_id}")
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)
    ply_p = jdir / "splat.ply"
    if not ply_p.is_file():
        return JSONResponse({"error": f"splat.ply not found in {job_id}"}, status_code=404)
    splat_p = jdir / "splat.splat"
    try:
        info = await asyncio.to_thread(ensure_aligned_splat_file, ply_p, splat_p)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "splat_url": st.media_url_for(splat_p),
        "ply_url": st.media_url_for(ply_p),
        "ply_rel": sess.sandbox.rel(ply_p).replace("\\", "/"),
        **info,
    })


@router.post("/splats/upload")
async def api_splats_upload(req: Request, filename: str = "") -> JSONResponse:
    """Upload a local `.ply` or `.splat` file into `assets/splats/imported_<stem>_<ts>/` for FPS editing."""
    from blenderbot.splats.splat_edit import (
        parse_ply_gaussians,
        ply_data_to_splat_bytes,
        splat_bytes_to_ply_bytes,
    )

    st = _get_state(req)
    sess = st.session
    raw_name = filename.strip() or str(req.headers.get("X-Filename") or "uploaded.ply").strip()
    clean_name = _SAFE_FILENAME_RE.sub("_", Path(raw_name).name)
    ext = Path(clean_name).suffix.lower()
    if ext not in (".ply", ".splat"):
        return JSONResponse(
            {"error": f"Unsupported splat file '{ext}'. Drop a binary 3DGS .ply or .splat file."},
            status_code=400,
        )

    data = await req.body()
    if not data:
        return JSONResponse({"error": "Empty upload body"}, status_code=400)
    if len(data) > 350 * 1024 * 1024:
        return JSONResponse({"error": "Splat file exceeds 350 MB limit"}, status_code=413)

    stem = _SAFE_FILENAME_RE.sub("_", Path(clean_name).stem)[:28].strip("_") or "splat"
    stamp = datetime.datetime.now().strftime("%H%M%S")
    h4 = hashlib.sha1(f"{stem}:{time.time()}".encode()).hexdigest()[:4]
    job_id = f"imported_{stem}_{stamp}_{h4}"
    out_dir = sess.cfg.cwd / "assets" / "splats" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_p = out_dir / "splat.ply"
    splat_p = out_dir / "splat.splat"

    try:
        if ext == ".ply":
            parsed = await asyncio.to_thread(parse_ply_gaussians, data)
            n_g = len(parsed["data"])
            splat_b = await asyncio.to_thread(
                ply_data_to_splat_bytes, parsed["props"], parsed["data"], max_gaussians=350_000
            )
            ply_p.write_bytes(data)
            splat_p.write_bytes(splat_b)
        else:
            ply_b, n_g = await asyncio.to_thread(splat_bytes_to_ply_bytes, data)
            ply_p.write_bytes(ply_b)
            splat_p.write_bytes(data[: 350_000 * 32])
    except Exception as err:  # noqa: BLE001
        shutil.rmtree(out_dir, ignore_errors=True)
        return JSONResponse({"error": f"Failed to parse {clean_name}: {err}"}, status_code=400)

    rec = st.new_splat_job(
        job_id=job_id,
        source_mode="upload",
        source_label=clean_name,
        gpu="LOCAL",
        iterations=0,
    )
    rec.stage = "completed"
    rec.n_gaussians = n_g
    rec.ply_rel = sess.sandbox.rel(ply_p).replace("\\", "/")
    rec.splat_url = st.media_url_for(splat_p)
    sess.bus.publish("splat-job", rec.to_dict())
    return JSONResponse({"ok": True, "job": rec.to_dict()})


@router.post("/splats/edit/apply")
async def api_splats_edit_apply(req: Request) -> JSONResponse:
    """Persist FPS Splat culling, AABB crop, smart-filter & paint edits to `.ply` + `.splat` (+ Blender)."""
    from blenderbot.splats.splat_edit import (
        apply_splat_edits,
        build_default_cube_splat,
    )

    st = _get_state(req)
    sess = st.session
    body = await req.json()

    raw_job_id = str(body.get("job_id") or "").strip()
    save_as_name = str(body.get("save_as_name") or "").strip()
    sync_bl = bool(body.get("sync_blender", False))
    render_mode = str(body.get("render_mode") or "ellipsoids_viewport").strip().lower()

    if save_as_name or not raw_job_id or raw_job_id.startswith("__default_"):
        stem = _SAFE_FILENAME_RE.sub("_", save_as_name or raw_job_id.strip("_") or "edited_splat")[:26].strip("_")
        stamp = datetime.datetime.now().strftime("%H%M%S")
        h4 = hashlib.sha1(f"{stem}:{time.time()}".encode()).hexdigest()[:4]
        target_job_id = f"{stem}_{stamp}_{h4}"
        target_dir = sess.cfg.cwd / "assets" / "splats" / target_job_id
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_job_id = raw_job_id
        target_dir = sess.cfg.cwd / "assets" / "splats" / target_job_id

    if not raw_job_id or raw_job_id.startswith("__default_"):
        variant = "cube" if "cube" in raw_job_id and "courtyard" not in raw_job_id else "courtyard"
        ply_b, splat_b, _ = await asyncio.to_thread(build_default_cube_splat, variant)
        src_ply = target_dir / "splat.ply"
        src_splat = target_dir / "splat.splat"
        src_ply.write_bytes(ply_b)
        src_splat.write_bytes(splat_b)
    else:
        src_dir = sess.sandbox.resolve_for_read(f"assets/splats/{raw_job_id}")
        src_ply = src_dir / "splat.ply"
        src_splat = src_dir / "splat.splat"
        if not src_ply.is_file():
            return JSONResponse({"error": f"splat.ply not found for {raw_job_id}"}, status_code=404)
        if target_dir != src_dir:
            cam_src = src_dir / "cameras.json"
            if cam_src.is_file():
                shutil.copy2(cam_src, target_dir / "cameras.json")

    dst_ply = target_dir / "splat.ply"
    dst_splat = target_dir / "splat.splat"

    try:
        res = await asyncio.to_thread(
            apply_splat_edits,
            src_ply,
            src_splat,
            body,
            out_ply_path=dst_ply,
            out_splat_path=dst_splat,
            backup_orig=(dst_ply == src_ply),
        )
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    rec = st.splat_jobs.get(target_job_id)
    if rec is None:
        rec = st.new_splat_job(
            job_id=target_job_id,
            source_mode="edited",
            source_label=f"Edited {raw_job_id or 'Default Cube'}",
            gpu="LOCAL",
            iterations=0,
        )
        rec.stage = "completed"
    rec.n_gaussians = int(res["n_after"])
    rec.ply_rel = sess.sandbox.rel(dst_ply).replace("\\", "/")
    rec.splat_url = st.media_url_for(dst_splat)
    sess.bus.publish("splat-job", rec.to_dict())

    bl_imported = None
    if sync_bl:
        if sess.transport is None:
            await sess.maybe_auto_connect_blender()
        if sess.transport is not None:
            cam_p = target_dir / "cameras.json"
            imp_params: dict[str, Any] = {
                "filepath": str(dst_ply),
                "name": f"Splat_{target_job_id[:20]}",
                "render_mode": render_mode,
                "max_points": 350_000,
                "shadow_catcher_ground": False,
            }
            if cam_p.is_file():
                imp_params["cameras_json"] = str(cam_p)
            bl_imported = await sess.transport.call("splat.import", imp_params, timeout=180.0)
            sess.bus.publish("blender-changed", bl_imported)

    return JSONResponse({
        "ok": True,
        "job_id": target_job_id,
        "ply_rel": rec.ply_rel,
        "ply_url": st.media_url_for(dst_ply),
        "splat_url": rec.splat_url,
        "blender_imported": bl_imported,
        **res,
    })


@router.post("/splats/edit/restore/{job_id}")
async def api_splats_edit_restore(req: Request, job_id: str) -> JSONResponse:
    """Restore unedited `splat.orig.ply` backup for `job_id`."""
    from blenderbot.splats.splat_edit import restore_original_ply

    st = _get_state(req)
    sess = st.session
    try:
        jdir = sess.sandbox.resolve_for_read(f"assets/splats/{job_id}")
        res = await asyncio.to_thread(restore_original_ply, jdir)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    splat_p = jdir / "splat.splat"
    rec = st.splat_jobs.get(job_id)
    if rec is not None:
        rec.n_gaussians = int(res["restored_gaussians"])
        rec.splat_url = st.media_url_for(splat_p)
        sess.bus.publish("splat-job", rec.to_dict())

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "splat_url": st.media_url_for(splat_p),
        **res,
    })


@router.post("/fps/camera_sync")
async def api_fps_camera_sync(req: Request) -> JSONResponse:
    """Push current FPS camera pose (`eye`, `forward`, `roll_deg`, `lens_mm`) directly into Blender 5.3."""
    from blenderbot.splats.shot_pack import _look_to_opencv_w2c
    from blenderbot.splats.sim3 import opencv_w2c_to_blender_camera

    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    body = await req.json()
    eye_raw = body.get("eye") or [0.0, -4.5, 1.65]
    fwd_raw = body.get("forward") or [0.0, 1.0, 0.0]
    eye = (float(eye_raw[0]), float(eye_raw[1]), float(eye_raw[2]))
    fwd = (float(fwd_raw[0]), float(fwd_raw[1]), float(fwd_raw[2]))
    roll_deg = float(body.get("roll_deg") or 0.0)
    lens_mm = max(10.0, min(300.0, float(body.get("lens_mm") or 35.0)))
    cam_name = str(body.get("camera_name") or "FPSDirectorCam").strip() or "FPSDirectorCam"
    set_active = bool(body.get("set_active", True))

    r_w2c, t_w2c = _look_to_opencv_w2c(eye, fwd, roll_deg=roll_deg)
    fx = lens_mm * 1280.0 / 36.0
    bl = opencv_w2c_to_blender_camera(r_w2c, t_w2c, fx, fx, 640.0, 360.0, 1280, 720)

    try:
        res = await sess.transport.call(
            "camera.upsert",
            {
                "name": cam_name,
                "location": bl["location"],
                "rotation_quaternion": bl["rotation_quaternion"],
                "lens_mm": lens_mm,
                "set_active": set_active,
            },
            timeout=15.0,
        )
        return JSONResponse({"ok": True, "camera": res})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.get("/fps/camera_pull")
async def api_fps_camera_pull(req: Request, camera: str = "") -> JSONResponse:
    """Pull active (or named) camera pose from Blender 5.3 into the WebGL2 FPS pawn."""
    import math as _math

    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    try:
        pas = await sess.transport.call("scene.passport", {"max_objects": 8}, timeout=10.0)
        f_cur = int(pas.get("frame_current") or 1)
        exp = await sess.transport.call(
            "splat.export_camera_path",
            {"camera": camera.strip() or None, "frame_start": f_cur, "frame_end": f_cur},
            timeout=15.0,
        )
        doc = exp.get("cameras_doc") or {}
        cams = doc.get("cameras") or []
        if not cams:
            return JSONResponse({"error": "No camera frames returned from Blender."}, status_code=404)
        c0 = cams[0]
        r_w2c = c0["R_w2c"]
        # Row 2 of OpenCV R_w2c is the camera optical forward ray in Blender world space
        fwd = [float(r_w2c[2][0]), float(r_w2c[2][1]), float(r_w2c[2][2])]
        yaw_rad = _math.atan2(fwd[0], fwd[1])
        pitch_rad = _math.asin(max(-1.0, min(1.0, fwd[2])))
        return JSONResponse({
            "ok": True,
            "camera": exp.get("camera"),
            "eye": c0["location"],
            "forward": fwd,
            "yaw_rad": round(yaw_rad, 6),
            "pitch_rad": round(pitch_rad, 6),
            "lens_mm": c0["lens_mm"],
        })
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.get("/fps/blender_scene")
async def api_fps_blender_scene(req: Request, max_triangles: int = 45000) -> JSONResponse:
    """Pull live evaluated world-space triangles + colors from Blender 5.3 for FPS navigation."""
    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    try:
        res = await sess.transport.call(
            "scene.viewport_triangles",
            {"max_triangles": max_triangles},
            timeout=30.0,
        )
        return JSONResponse({"ok": True, **res})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/fps/spawn_default_cube_blender")
async def api_fps_spawn_default_cube_blender(req: Request) -> JSONResponse:
    """Ensure a metric `2m x 2m x 2m` Default Cube on `Z=0` + 3-point lighting exists in Blender 5.3."""
    st = _get_state(req)
    sess = st.session
    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    try:
        cube = await sess.transport.call(
            "mesh.primitive",
            {
                "name": "DefaultCube",
                "shape": "cube",
                "size": 2.0,
                "location": [0.0, 0.0, 1.0],
                "color": [0.76, 0.79, 0.84],
            },
            timeout=15.0,
        )
        await sess.transport.call(
            "light.rig",
            {"preset": "three_point", "target": "DefaultCube"},
            timeout=15.0,
        )
        cam = await sess.transport.call(
            "camera.upsert",
            {
                "name": "FPSDirectorCam",
                "location": [0.0, -5.2, 1.65],
                "look_at": [0.0, 0.0, 1.0],
                "lens_mm": 35.0,
                "set_active": True,
            },
            timeout=15.0,
        )
        sess.bus.publish("blender-changed", cube)
        return JSONResponse({"ok": True, "cube": cube, "camera": cam})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


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