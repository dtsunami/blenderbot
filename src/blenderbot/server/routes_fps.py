"""FastAPI routes for FPS Camera Rail / Shot Packs, Splat Interactive Editing, and Live Blender Camera Sync."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from .routes_common import SAFE_FILENAME_RE, get_web_state

router = APIRouter()


@router.get("/shots/list")
async def api_shots_list(req: Request) -> JSONResponse:
    """List exported FPS Camera Rail & Spatial Brief packs (`assets/shots/*`)."""
    st = get_web_state(req)
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

    st = get_web_state(req)
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

    st = get_web_state(req)
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

    st = get_web_state(req)
    sess = st.session
    raw_name = filename.strip() or str(req.headers.get("X-Filename") or "uploaded.ply").strip()
    clean_name = SAFE_FILENAME_RE.sub("_", Path(raw_name).name)
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

    stem = SAFE_FILENAME_RE.sub("_", Path(clean_name).stem)[:28].strip("_") or "splat"
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

    st = get_web_state(req)
    sess = st.session
    body = await req.json()

    raw_job_id = str(body.get("job_id") or "").strip()
    save_as_name = str(body.get("save_as_name") or "").strip()
    sync_bl = bool(body.get("sync_blender", False))
    render_mode = str(body.get("render_mode") or "ellipsoids_viewport").strip().lower()

    if save_as_name or not raw_job_id or raw_job_id.startswith("__default_"):
        stem = SAFE_FILENAME_RE.sub("_", save_as_name or raw_job_id.strip("_") or "edited_splat")[:26].strip("_")
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

    st = get_web_state(req)
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

    st = get_web_state(req)
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

    st = get_web_state(req)
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
    st = get_web_state(req)
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
    st = get_web_state(req)
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