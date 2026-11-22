"""FastAPI routes for Meshy 7.1 Image-to-3D (`fal.ai`) generation, `.glb` upload, and Blender import."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from blenderbot.context.media import load_image_part
from blenderbot.media_gen.meshy import (
    DEFAULT_MESHY_MODEL,
    MeshyError,
    default_meshy_job_dir,
    extract_glb_viewport_triangles,
    generate_meshy_3d,
)

from .routes_common import SAFE_FILENAME_RE, get_web_state, refresh_env_keys

router = APIRouter()


@router.get("/meshy/jobs")
async def api_meshy_jobs(req: Request) -> dict[str, Any]:
    st = get_web_state(req)
    jobs = await asyncio.to_thread(st.scan_meshy_jobs)
    return {"jobs": jobs, "count": len(jobs)}


@router.post("/meshy/generate")
async def api_meshy_generate(req: Request) -> JSONResponse:
    """Submit a fal `meshy/v7.1/image-to-3d` queue job, track status/logs over SSE, and save `.glb`."""
    st = get_web_state(req)
    sess = st.session
    cfg = sess.cfg

    refresh_env_keys(cfg)
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
    st = get_web_state(req)
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
    st = get_web_state(req)
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
    st = get_web_state(req)
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
    st = get_web_state(req)
    sess = st.session
    raw_name = filename.strip() or str(req.headers.get("X-Filename") or "uploaded.glb").strip()
    clean_name = SAFE_FILENAME_RE.sub("_", Path(raw_name).name)
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