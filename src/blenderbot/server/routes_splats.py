"""FastAPI routes for Cloud 3DGS Splat training, telemetry, surgical resume, Blender import, and trajectory rendering."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .routes_common import get_web_state
from .state import WebState

router = APIRouter()


@router.get("/splats/modal_status")
async def api_splats_modal_status() -> dict[str, Any]:
    from blenderbot.splats.modal_deploy import check_modal_version_sync

    st = await asyncio.to_thread(check_modal_version_sync, use_cache_sec=0.0)
    return st.to_dict()


@router.post("/splats/modal_deploy")
async def api_splats_modal_deploy(req: Request) -> JSONResponse:
    from blenderbot.splats.modal_deploy import ensure_modal_deployed

    st = get_web_state(req)
    sess = st.session
    body = await req.json() if req.headers.get("content-type", "").startswith("application/json") else {}
    force = bool(body.get("force", False))

    def _on_ev(ev: dict[str, Any]) -> None:
        sess.log.flight.touch()
        sess.bus.publish("modal-deploy", ev)

    try:
        res = await ensure_modal_deployed(force=force, on_event=_on_ev)
        sess.bus.publish("modal-deploy", {"event": "modal_status", **res})
        return JSONResponse({"ok": True, **res})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.get("/splats/jobs")
async def api_splats_jobs(req: Request) -> dict[str, Any]:
    st = get_web_state(req)
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

    st = get_web_state(req)
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

    splat_cfg = SplatConfig.from_dict(body)
    splat_cfg.gpu = gpu
    splat_cfg.train.iterations = iterations
    splat_cfg.train.sh_degree = sh_degree
    if body.get("densify_until") is None or int(body.get("densify_until") or 0) <= 0:
        splat_cfg.train.densify_until = max(600, int(iterations * 0.72))
    if body.get("pose_backend") in ("vggt", "colmap", "colmap_prior", "known_cameras"):
        splat_cfg.pose.backend = str(body["pose_backend"])

    known_cams_job = str(body.get("known_cameras_job") or "").strip()
    cfg_dict = splat_cfg.to_dict()
    src_doc: dict[str, Any] = {
        "source_mode": source_mode,
        "source_label": source_label,
        "known_cameras_job": known_cams_job or None,
        "auto_import": auto_import,
        "render_mode": render_mode,
        "config": cfg_dict,
        "created_at": time.time(),
    }
    if video_p is not None:
        src_doc["video_rel"] = source_label
    elif source_mode == "images":
        src_doc["images_rel"] = source_label
    (out_dir / "job_source.json").write_text(json.dumps(src_doc, indent=2), encoding="utf-8")

    rec = st.new_splat_job(
        job_id=job_id,
        source_mode=source_mode,
        source_label=source_label,
        gpu=gpu,
        iterations=iterations,
    )
    rec.config = cfg_dict
    sess.bus.publish("splat-job", rec.to_dict())

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
    st = get_web_state(req)
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
    st = get_web_state(req)
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
    from blenderbot.splats.config import SplatConfig, SurgerySpec

    st = get_web_state(req)
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

    cfg_overrides: dict[str, Any] = {"gpu": gpu}
    for ck in (
        "sh_degree",
        "opa_reg",
        "lambda_dssim",
        "scale_reg",
        "aniso_reg",
        "max_anisotropy",
        "prune_scale3d",
        "opacity_reset_every",
        "opacity_logit_cap",
        "views_per_step",
        "cam_corridor_radius_m",
        "appearance_opt",
        "outlier_view_downweight",
        "focus_camera",
        "rasterize_mode",
        "refine_poses",
        "pose_opt_lr",
        "pose_prior_reg",
        "adaptive_strategy",
        "arc_scale_mode",
        "arc_gamma",
        "target_gaussians",
        "densify_grad_threshold",
        "prune_opa",
        "iterative_arc_cycles",
        "arc_solve_steps",
        "frames_add_per_cycle",
        "near_frustum_clear_m",
        "opa_bimodal_reg",
        "depth_prior_model",
        "depth_nudge_strength",
    ):
        if body.get(ck) is not None:
            cfg_overrides[ck] = body[ck]

    saved_cfg_doc: dict[str, Any] = {}
    src_meta_p = out_dir / "job_source.json"
    if src_meta_p.is_file():
        try:
            src_meta_doc = json.loads(src_meta_p.read_text(encoding="utf-8"))
            if isinstance(src_meta_doc.get("config"), dict):
                merged_c = SplatConfig.from_dict(src_meta_doc["config"]).merge_overrides(cfg_overrides)
                saved_cfg_doc = merged_c.to_dict()
                src_meta_doc["config"] = saved_cfg_doc
                src_meta_p.write_text(json.dumps(src_meta_doc, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001, S110
            pass

    rec = st.new_splat_job(
        job_id=job_id,
        source_mode="resume",
        source_label=f"Resume {job_id} (+{add_steps})",
        gpu=gpu,
        iterations=add_steps,
    )
    if saved_cfg_doc:
        rec.config = saved_cfg_doc
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

    st = get_web_state(req)
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

    st = get_web_state(req)
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
        "densify_grad_threshold",
        "target_gaussians",
        "adaptive_strategy",
        "arc_scale_mode",
        "arc_gamma",
        "views_per_step",
        "cam_corridor_radius_m",
        "appearance_opt",
        "outlier_view_downweight",
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
    st = get_web_state(req)
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

    from blenderbot.splats.diagnostics import summarize_splat_telemetry

    diagnostics = await asyncio.to_thread(
        summarize_splat_telemetry,
        jdir,
        ticks,
        stages,
        densify_events,
    )

    saved_cfg: dict[str, Any] = {}
    src_meta_p = jdir / "job_source.json"
    if src_meta_p.is_file():
        try:
            s_doc = json.loads(src_meta_p.read_text(encoding="utf-8"))
            if isinstance(s_doc.get("config"), dict):
                saved_cfg = dict(s_doc["config"])
        except Exception:  # noqa: BLE001, S110
            pass

    prev_p = jdir / "previews" / "latest_render.jpg"
    diff_p = jdir / "previews" / "latest_diff.jpg"
    gt_files = sorted((jdir / "previews").glob("gt_*.jpg")) if (jdir / "previews").is_dir() else []
    gt_p = gt_files[0] if gt_files else None
    proxy_p = jdir / "proxy_collision.glb"
    all_jobs = await asyncio.to_thread(st.scan_splat_jobs)
    job_summary = next((j for j in all_jobs if j.get("job_id") == job_id), None)
    return JSONResponse({
        "job_id": job_id,
        "job": job_summary,
        "config": saved_cfg,
        "preview_url": f"{st.media_url_for(prev_p)}&v={int(prev_p.stat().st_mtime_ns)}" if prev_p.is_file() else None,
        "diff_url": f"{st.media_url_for(diff_p)}&v={int(diff_p.stat().st_mtime_ns)}" if diff_p.is_file() else None,
        "gt_url": st.media_url_for(gt_p) if gt_p is not None else None,
        "ticks": ticks,
        "densify_events": densify_events,
        "surgery_events": surgery_events,
        "stages": stages,
        "snapshots": snapshots,
        "cameras": cam_names,
        "diagnostics": diagnostics,
        "has_proxy_glb": proxy_p.is_file(),
        "proxy_glb_url": st.media_url_for(proxy_p) if proxy_p.is_file() else None,
    })


@router.post("/splats/crop_box")
async def api_splats_crop_box(req: Request) -> JSONResponse:
    """Spawn or read `SplatCropBox` in Blender 5.3, or trigger 1-click surgical box cull/keep + heal."""
    from blenderbot.splats.client import run_modal_splat_job
    from blenderbot.splats.config import SurgerySpec

    st = get_web_state(req)
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

    st = get_web_state(req)
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
    st = get_web_state(req)
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


def _backfill_legacy_canonical_render(renders_dir: Path, canonical_name: str) -> None:
    """Before overwriting `modal_recreated.mp4` / `blender_recreated.mp4`, preserve any legacy
    un-manifested render from older sessions as `Pass #1` so prior renders are never lost!
    """
    import shutil

    canon_p = renders_dir / canonical_name
    manifest_p = renders_dir / "passes_manifest.json"
    if not canon_p.is_file() or manifest_p.is_file():
        return
    legacy_kind = "blender" if "blender" in canonical_name else "modal"
    bak_name = f"pass_01_{legacy_kind}_initial.mp4"
    bak_p = renders_dir / bak_name
    shutil.copy2(canon_p, bak_p)
    manifest_p.write_text(
        json.dumps(
            {
                "passes": [
                    {
                        "index": 1,
                        "filename": bak_name,
                        "kind": legacy_kind,
                        "label": f"Pass #1 · Initial {'Blender 5.3' if legacy_kind == 'blender' else 'Modal 3DGS'} Render",
                        "created_at": canon_p.stat().st_mtime,
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _archive_and_write_render_pass(
    renders_dir: Path,
    canonical_name: str,
    pass_kind: str,
    job_dir: Path,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Archive the newly rendered pass into `renders/pass_XX_...mp4` + `passes_manifest.json` so
    every training, culling, and arc-snapping pass remains selectable in the Compare dropdown!
    """
    import shutil

    renders_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = renders_dir / "passes_manifest.json"
    entries: list[dict[str, Any]] = []
    if manifest_p.is_file():
        try:
            doc = json.loads(manifest_p.read_text(encoding="utf-8"))
            if isinstance(doc.get("passes"), list):
                entries = list(doc["passes"])
        except Exception:  # noqa: BLE001
            entries = []

    canon_p = renders_dir / canonical_name

    # Read current step / Gaussian count from `splat.ply` or `telemetry.jsonl` for clear pass labeling
    step_cur = 0
    n_gauss = 0
    psnr_cur = 0.0
    telem_p = job_dir / "telemetry.jsonl"
    if telem_p.is_file():
        try:
            for ln in reversed(telem_p.read_text(encoding="utf-8", errors="ignore").splitlines()[-25:]):
                if not ln.strip():
                    continue
                row = json.loads(ln)
                if row.get("step") and not step_cur:
                    step_cur = int(row["step"])
                if row.get("n_gaussians") and not n_gauss:
                    n_gauss = int(row["n_gaussians"])
                if row.get("psnr") and not psnr_cur:
                    psnr_cur = float(row["psnr"])
                if step_cur and n_gauss and psnr_cur:
                    break
        except Exception:  # noqa: BLE001, S110
            pass
    splat_f = job_dir / "splat.splat"
    if splat_f.is_file():
        # `splat.splat` reflects interactive FPS culling/saving immediately (32 bytes / splat)
        n_gauss = splat_f.stat().st_size // 32

    next_idx = len(entries) + 1
    step_slug = f"_s{step_cur}" if step_cur else ""
    pts_slug = f"_{n_gauss // 1000}k" if n_gauss >= 1000 else ""
    pass_filename = f"pass_{next_idx:02d}_{pass_kind}{step_slug}{pts_slug}.mp4"
    pass_p = renders_dir / pass_filename
    if canon_p.is_file():
        shutil.copy2(canon_p, pass_p)

    meta = extra_meta or {}
    tag_map = {
        "modal": "Modal 3DGS",
        "snap_arc": "Snap Arc + Clean",
        "blender": "Blender 5.3",
    }
    kind_title = tag_map.get(pass_kind, pass_kind)
    parts = [f"Pass #{next_idx} · {kind_title}"]
    if step_cur:
        parts.append(f"step {step_cur:,}")
    if n_gauss:
        parts.append(f"{n_gauss:,} pts")
    if meta.get("l1_after") is not None:
        parts.append(f"L1 {meta['l1_after']}")
    elif psnr_cur > 0:
        parts.append(f"{psnr_cur:.1f}dB")
    if meta.get("cleaned_floaters"):
        parts.append(f"-{int(meta['cleaned_floaters']):,} floaters")

    entries.append({
        "index": next_idx,
        "filename": pass_filename,
        "kind": pass_kind,
        "step": step_cur,
        "n_gaussians": n_gauss,
        "psnr": round(psnr_cur, 2) if psnr_cur else None,
        "label": " · ".join(parts),
        "created_at": time.time(),
        **{k: v for k, v in meta.items() if k not in ("label",)},
    })
    manifest_p.write_text(json.dumps({"passes": entries}, indent=2), encoding="utf-8")
    return pass_p


def _collect_all_render_passes(st: WebState, job_dir: Path, src_vid: Path | None) -> list[dict[str, Any]]:
    """Collect Original Video + ALL historical & latest trajectory render passes + training snapshots."""
    sess = st.session
    cwd = sess.cfg.cwd
    job_id = job_dir.name
    renders_dir = job_dir / "renders"
    passes: list[dict[str, Any]] = []
    seen_rels: set[str] = set()

    def _add_clip(path: Path, label: str, kind: str, meta: dict[str, Any] | None = None) -> None:
        if not path.is_file():
            return
        rel = sess.sandbox.rel(path).replace("\\", "/")
        if rel in seen_rels:
            return
        seen_rels.add(rel)
        st_inf = path.stat()
        passes.append({
            "label": label,
            "kind": kind,
            "media_type": "image" if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") else "video",
            "rel": rel,
            "url": f"{st.media_url_for(path)}&t={int(st_inf.st_mtime_ns)}",
            "mtime": st_inf.st_mtime,
            "size_bytes": st_inf.st_size,
            **(meta or {}),
        })

    if src_vid is not None and src_vid.is_file():
        src_rel = sess.sandbox.rel(src_vid).replace("\\", "/")
        _add_clip(src_vid, f"🎬 Original Source ({Path(src_rel).name})", "source")

    # 1. Numbered historical passes from `renders/passes_manifest.json` (chronological, newest first in UI or ordered)
    manifest_entries: list[dict[str, Any]] = []
    manifest_p = renders_dir / "passes_manifest.json"
    if manifest_p.is_file():
        try:
            mdoc = json.loads(manifest_p.read_text(encoding="utf-8"))
            if isinstance(mdoc.get("passes"), list):
                manifest_entries = list(mdoc["passes"])
        except Exception:  # noqa: BLE001
            manifest_entries = []

    for entry in reversed(manifest_entries):
        fn = str(entry.get("filename") or "")
        if not fn:
            continue
        fp = renders_dir / fn
        if fp.is_file():
            _add_clip(
                fp,
                f"🎞 {entry.get('label') or fn}",
                str(entry.get("kind") or "pass"),
                {
                    "pass_index": entry.get("index"),
                    "step": entry.get("step"),
                    "n_gaussians": entry.get("n_gaussians"),
                },
            )

    # 2. Canonical latest Modal / Blender renders + any extra MP4s in `renders/` (if not already in manifest)
    modal_mp4 = renders_dir / "modal_recreated.mp4"
    blender_mp4 = renders_dir / "blender_recreated.mp4"
    if modal_mp4.is_file() and not any(p["kind"] in ("modal", "snap_arc") for p in passes):
        _add_clip(modal_mp4, "☁️ Modal H100 3DGS Render (Latest)", "modal")
    if blender_mp4.is_file() and not any(p["kind"] == "blender" for p in passes):
        _add_clip(blender_mp4, "🧊 Blender 5.3 EEVEE Render (Latest)", "blender")

    if renders_dir.is_dir():
        for extra_vid in sorted(renders_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            if extra_vid.name in ("modal_recreated.mp4", "blender_recreated.mp4"):
                continue
            _add_clip(extra_vid, f"🎞 Render Pass ({extra_vid.stem})", "pass")

    # 3. Exported FPS Camera Rail renders (`assets/shots/*/modal_rail_render.mp4`) linked to this splat job
    shots_root = cwd / "assets" / "shots"
    if shots_root.is_dir():
        for sh_dir in sorted(shots_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:25]:
            bj = sh_dir / "brief.json"
            rail_mp4 = sh_dir / "modal_rail_render.mp4"
            if sh_dir.is_dir() and bj.is_file() and rail_mp4.is_file():
                try:
                    bdoc = json.loads(bj.read_text(encoding="utf-8"))
                    if bdoc.get("splat_job_id") == job_id:
                        _add_clip(rail_mp4, f"🎮 FPS Rail Render ({sh_dir.name})", "fps_rail")
                except Exception:  # noqa: BLE001, S110
                    pass

    # 4. Sibling Splat Jobs trained from the SAME source video (compare across runs / presets / culling!)
    if src_vid is not None and src_vid.is_file():
        from blenderbot.server.state import resolve_source_video_for_job

        splats_root = cwd / "assets" / "splats"
        if splats_root.is_dir():
            src_resolved = src_vid.resolve()
            for sib in sorted(splats_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
                if not sib.is_dir() or sib.name == job_id:
                    continue
                sib_vid = resolve_source_video_for_job(cwd, sib.name)
                if sib_vid is None or sib_vid.resolve() != src_resolved:
                    continue
                sib_renders = sib / "renders"
                sib_modal = sib_renders / "modal_recreated.mp4"
                sib_bl = sib_renders / "blender_recreated.mp4"
                short_id = sib.name[-13:] if len(sib.name) > 13 else sib.name
                sib_pts = (sib / "splat.splat").stat().st_size // 32 if (sib / "splat.splat").is_file() else 0
                pts_lbl = f" · {sib_pts // 1000}k pts" if sib_pts >= 1000 else ""
                if sib_modal.is_file():
                    _add_clip(sib_modal, f"🔀 Sibling Run {short_id} (Modal{pts_lbl})", "sibling_modal")
                if sib_bl.is_file():
                    _add_clip(sib_bl, f"🔀 Sibling Run {short_id} (Blender{pts_lbl})", "sibling_blender")

    # 5. Still Training Checkpoints (`snapshots/step_*.jpg`, `latest_render.jpg`, `latest_diff.jpg`)
    prev_dir = job_dir / "previews"
    if (prev_dir / "latest_render.jpg").is_file():
        _add_clip(prev_dir / "latest_render.jpg", "🖼 Latest Training Render (Still)", "still_render")
    if (prev_dir / "latest_diff.jpg").is_file():
        _add_clip(prev_dir / "latest_diff.jpg", "🔥 Latest 3× Photometric Error (Still)", "still_diff")
    snap_dir = job_dir / "snapshots"
    if snap_dir.is_dir():
        for sp in sorted(snap_dir.glob("step_*.jpg")):
            digits = "".join(c for c in sp.stem if c.isdigit())
            s_num = int(digits) if digits else 0
            _add_clip(sp, f"📸 Checkpoint Snapshot · step {s_num:,} (Still)", "snapshot", {"step": s_num})

    return passes


@router.get("/splats/trajectory/{job_id}")
async def api_splats_trajectory(req: Request, job_id: str) -> JSONResponse:
    """Return full-rate interpolated camera path, Yaw/Pitch/Roll/Speed telemetry, markers & ALL render passes."""
    st = get_web_state(req)
    sess = st.session
    try:
        traj, src_vid = await asyncio.to_thread(_build_job_trajectory_sync, st, job_id)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)

    jdir = sess.cfg.cwd / "assets" / "splats" / job_id
    modal_mp4 = jdir / "renders" / "modal_recreated.mp4"
    blender_mp4 = jdir / "renders" / "blender_recreated.mp4"
    render_passes = await asyncio.to_thread(_collect_all_render_passes, st, jdir, src_vid)

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "trajectory": traj,
        "source_video_url": st.media_url_for(src_vid) if src_vid and src_vid.is_file() else None,
        "source_video_rel": sess.sandbox.rel(src_vid).replace("\\", "/") if src_vid and src_vid.is_file() else None,
        "modal_render_url": st.media_url_for(modal_mp4) if modal_mp4.is_file() else None,
        "blender_render_url": st.media_url_for(blender_mp4) if blender_mp4.is_file() else None,
        "render_passes": render_passes,
    })


@router.post("/splats/snap_arc")
async def api_splats_snap_arc(req: Request) -> JSONResponse:
    """Snap an existing splat job's camera arc directly onto its source video + clean floaters + optionally re-render MP4."""
    from blenderbot.server.state import resolve_source_video_for_job
    from blenderbot.splats.client import run_modal_snap_camera_arc, run_modal_splat_render

    st = get_web_state(req)
    sess = st.session
    body = await req.json()
    job_id = str(body.get("job_id") or "").strip()
    if not job_id:
        return JSONResponse({"error": "Missing job_id"}, status_code=400)

    jdir = sess.sandbox.resolve_for_read(f"assets/splats/{job_id}")
    cam_p = jdir / "cameras.json"
    ply_p = jdir / "splat.ply"
    if not cam_p.is_file():
        return JSONResponse({"error": f"cameras.json not found in {job_id}"}, status_code=404)

    src_vid = resolve_source_video_for_job(sess.cfg.cwd, job_id)
    if src_vid is None or not src_vid.is_file():
        return JSONResponse({"error": f"Source video not found for {job_id}"}, status_code=404)

    gpu = str(body.get("gpu") or "H100").strip().upper()
    floater_clean = bool(body.get("floater_clean", True))
    auto_render = bool(body.get("auto_render", True))
    base_doc = json.loads(cam_p.read_text(encoding="utf-8"))

    sess.bus.publish("splat-render", {"job_id": job_id, "mode": "snap", "stage": "snapping_arc"})
    try:
        snap_stats = await run_modal_snap_camera_arc(
            job_id=job_id,
            output_dir=jdir,
            video_path=src_vid,
            base_cameras_doc=base_doc,
            ply_fallback_path=ply_p if ply_p.is_file() else None,
            sample_fps=float(body.get("sample_fps") or 12.0),
            max_snap_frames=int(body.get("max_snap_frames") or 240),
            solve_steps=int(body.get("solve_steps") or 80),
            floater_clean=floater_clean,
            corridor_radius_m=float(body.get("corridor_radius_m") or 0.42),
            near_clear_m=float(body.get("near_clear_m") or 0.42),
            max_world_scale_m=float(body.get("max_world_scale_m") or 0.25),
            gpu=gpu,
            timeout_sec=sess.cfg.modal_worker_timeout,
            on_event=lambda ev: (
                sess.log.flight.touch(),
                sess.bus.publish("splat-render", {"job_id": job_id, "mode": "snap", **ev}),
            ),
        )
        sess.agent.run_cost += float(snap_stats.get("cost_usd") or 0.0)
        sess.agent.stats["cost"] += float(snap_stats.get("cost_usd") or 0.0)

        # Rebuild dense interpolated trajectory from newly snapped keyframes
        traj, _ = await asyncio.to_thread(_build_job_trajectory_sync, st, job_id)
        render_info: dict[str, Any] | None = None
        if auto_render:
            renders_dir = jdir / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(_backfill_legacy_canonical_render, renders_dir, "modal_recreated.mp4")
            out_mp4 = renders_dir / "modal_recreated.mp4"
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
            ev_tot = (
                int(snap_stats.get("giant_evicted") or 0)
                + int(snap_stats.get("corridor_evicted") or 0)
                + int(snap_stats.get("frustum_evicted") or 0)
            )
            pass_p = await asyncio.to_thread(
                _archive_and_write_render_pass,
                renders_dir,
                "modal_recreated.mp4",
                "snap_arc",
                jdir,
                {
                    "l1_before": snap_stats.get("l1_before"),
                    "l1_after": snap_stats.get("l1_after"),
                    "cleaned_floaters": ev_tot,
                    "snapped_keyframes": snap_stats.get("snapped_keyframes"),
                },
            )
            sess.bus.publish("workspace-media-added", {"path": sess.sandbox.rel(out_mp4)})
            render_info = {
                "url": st.media_url_for(pass_p if pass_p.is_file() else out_mp4),
                "frames": r_res.frames,
                "render_fps": r_res.render_fps,
                "wall_s": r_res.wall_s,
            }

        return JSONResponse({"ok": True, "snap": snap_stats, "render": render_info})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/splats/render")
async def api_splats_render(req: Request) -> JSONResponse:
    """Render recreated video camera path (either farmed to Modal H100 `gsplat` OR via Blender 5.3 EEVEE)."""
    from blenderbot.splats.client import run_modal_splat_render
    from blenderbot.splats.sim3 import interpolate_camera_trajectory

    st = get_web_state(req)
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
        await asyncio.to_thread(_backfill_legacy_canonical_render, renders_dir, "modal_recreated.mp4")
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
            pass_p = await asyncio.to_thread(
                _archive_and_write_render_pass,
                renders_dir,
                "modal_recreated.mp4",
                "modal",
                jdir,
                {"camera": bl_cam_override or "trajectory", "render_fps": r_res.render_fps},
            )
            url = st.media_url_for(pass_p if pass_p.is_file() else out_mp4)
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

    if sess.transport is None:
        ok = await sess.maybe_auto_connect_blender()
        if not ok or sess.transport is None:
            return JSONResponse({"error": "Blender 5.3 is not connected."}, status_code=400)

    await asyncio.to_thread(_backfill_legacy_canonical_render, renders_dir, "blender_recreated.mp4")
    out_mp4 = renders_dir / "blender_recreated.mp4"
    try:
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
        pass_p = await asyncio.to_thread(
            _archive_and_write_render_pass,
            renders_dir,
            "blender_recreated.mp4",
            "blender",
            jdir,
            {"engine": str(body.get("engine") or "BLENDER_EEVEE"), "camera": bl_cam_override or "SeedanceSourceCam"},
        )
        url = st.media_url_for(pass_p if pass_p.is_file() else out_mp4)
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