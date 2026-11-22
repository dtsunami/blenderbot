"""Session state, API key management, Agent Co-Pilot chat, context ledger, and SSE stream routes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
)
from blenderbot.media_gen.seedance import (
    ESTIMATED_COST_PER_SEC_USD,
    SEEDANCE_ASPECT_RATIOS,
    SEEDANCE_RESOLUTIONS,
    SEEDANCE_SIZES,
)
from blenderbot.media_gen.seedream import (
    ESTIMATED_COST_BY_RES_USD,
    ESTIMATED_COST_PER_IMAGE_USD,
    SEEDREAM_ASPECT_RATIOS,
    SEEDREAM_BACKGROUNDS,
    SEEDREAM_OUTPUT_FORMATS,
    SEEDREAM_RESOLUTIONS,
)
from blenderbot.providers.index import mask_key
from blenderbot.session.keys_ctl import set_provider_key

from .routes_common import get_web_state, refresh_env_keys

router = APIRouter()


def _safe_modal_version_status() -> dict[str, Any]:
    from blenderbot.splats.modal_deploy import check_modal_version_sync

    return check_modal_version_sync(use_cache_sec=25.0).to_dict()


@router.get("/state")
async def api_get_state(req: Request) -> dict[str, Any]:
    st = get_web_state(req)
    sess = st.session
    cfg = sess.cfg

    if sess.transport is not None and sess.transport.mode == "headless":
        await sess.maybe_upgrade_to_gui_bridge()
    bl_info: dict[str, Any] = {"connected": sess.transport is not None}
    if sess.transport is not None:
        bl_info["mode"] = sess.transport.mode
        try:
            st_bl = await sess.transport.call("blender.status", {}, timeout=3.0)
            bl_info["filepath"] = st_bl.get("filepath") or ""
            bl_info["is_dirty"] = bool(st_bl.get("is_dirty", False))
            bl_info["counts"] = st_bl.get("counts") or {}
            pas = await sess.transport.call("scene.passport", {"max_objects": 24}, timeout=4.0)
            bl_info["scene"] = pas.get("scene")
            bl_info["active_camera"] = pas.get("active_camera")
            bl_info["cameras"] = [c.get("name") for c in (pas.get("cameras") or [])]
            bl_info["objects"] = [
                {"name": o.get("name"), "type": o.get("type")} for o in (pas.get("objects") or [])
            ]
            sc_name = (
                pas["scene"].get("name")
                if isinstance(pas.get("scene"), dict)
                else str(pas.get("scene") or "")
            )
            if bl_info["filepath"] or sc_name:
                sess.log.update_blend_association(
                    blend_filepath=bl_info["filepath"] if bl_info["filepath"] else None,
                    scene_name=sc_name or None,
                )
        except Exception as err:  # noqa: BLE001
            bl_info["error"] = str(err)

    refresh_env_keys(cfg)
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
        "modal_status": (await asyncio.to_thread(_safe_modal_version_status)),
        "jobs": await asyncio.to_thread(st.scan_media_jobs),
        "splat_jobs": await asyncio.to_thread(st.scan_splat_jobs),
        "meshy_jobs": await asyncio.to_thread(st.scan_meshy_jobs),
    }


@router.post("/keys/openrouter")
async def api_set_openrouter_key(req: Request) -> JSONResponse:
    st = get_web_state(req)
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
    st = get_web_state(req)
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


@router.post("/chat/submit")
async def api_chat_submit(req: Request) -> JSONResponse:
    st = get_web_state(req)
    body = await req.json()
    text = str(body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty prompt"}, status_code=400)
    asyncio.create_task(st.submit_fn(text, "web"))
    return JSONResponse({"ok": True})


@router.post("/chat/cancel")
async def api_chat_cancel(req: Request) -> JSONResponse:
    st = get_web_state(req)
    st.session.agent.cancel()
    return JSONResponse({"ok": True})


@router.delete("/context/{seg_id}")
async def api_drop_context_segment(req: Request, seg_id: int) -> JSONResponse:
    st = get_web_state(req)
    ok = st.session.ledger.drop(seg_id)
    if not ok:
        return JSONResponse(
            {"error": f"Segment #{seg_id} could not be dropped (pinned/protected or missing)"},
            status_code=400,
        )
    return JSONResponse({"ok": True, "dropped": seg_id})


@router.get("/events")
async def api_sse_events(req: Request, since: int = 0) -> StreamingResponse:
    st = get_web_state(req)
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