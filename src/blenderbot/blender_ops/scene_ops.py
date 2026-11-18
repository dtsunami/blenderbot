"""Core Blender 5.x scene inspection, Scene Passport grounding, setup, save/open & snapshots."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .helpers import (
    effective_fps,
    ensure_collection,
    iter_action_fcurves,
    restore_snapshot,
    save_temp_snapshot,
    summarize_fcurves,
    world_aabb,
)
from .schema import OpContext, OpError, blender_op


@blender_op(
    "system.ping",
    tier="read",
    summary="Lightweight health check returning Blender version, mode, and active scene name.",
)
def op_system_ping(ctx: OpContext, _params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    return {
        "ok": True,
        "mode": ctx.mode,
        "pid": os.getpid(),
        "blender_version": bpy.app.version_string,
        "python_version": sys.version.split()[0],
        "scene": bpy.context.scene.name if bpy.context.scene else None,
        "filepath": bpy.data.filepath or "",
    }


@blender_op(
    "blender.status",
    tier="read",
    summary="Return detailed Blender session, scene settings, datablock counts, and camera status.",
)
def op_blender_status(ctx: OpContext, _params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    r = sc.render
    fps = effective_fps(sc)
    frames = sc.frame_end - sc.frame_start + 1
    return {
        "mode": ctx.mode,
        "pid": os.getpid(),
        "blender_version": bpy.app.version_string,
        "background": bool(bpy.app.background),
        "filepath": bpy.data.filepath or "",
        "is_dirty": bool(bpy.data.is_dirty),
        "scene": {
            "name": sc.name,
            "engine": r.engine,
            "fps": round(fps, 4),
            "frame_start": int(sc.frame_start),
            "frame_end": int(sc.frame_end),
            "frame_current": int(sc.frame_current),
            "total_frames": int(frames),
            "duration_sec": round(frames / fps, 3) if fps > 0 else 0.0,
            "resolution": [int(r.resolution_x), int(r.resolution_y)],
            "resolution_percentage": int(r.resolution_percentage),
            "active_camera": sc.camera.name if sc.camera else None,
        },
        "counts": {
            "collections": len(bpy.data.collections),
            "objects": len(bpy.data.objects),
            "meshes": len(bpy.data.meshes),
            "curves": len(bpy.data.curves),
            "materials": len(bpy.data.materials),
            "actions": len(bpy.data.actions),
            "cameras": len(bpy.data.cameras),
            "lights": len(bpy.data.lights),
            "armatures": len(bpy.data.armatures),
        },
    }


def _collection_tree(col: Any, depth: int = 0, max_depth: int = 4) -> dict[str, Any]:
    node: dict[str, Any] = {
        "name": col.name,
        "objects": len(col.objects),
    }
    if depth < max_depth and col.children:
        node["children"] = [_collection_tree(c, depth + 1, max_depth) for c in col.children]
    return node


@blender_op(
    "scene.passport",
    tier="read",
    summary=(
        "Compact Scene Passport for LLM grounding: FPS, engine, world, collections, cameras, lights, "
        "armatures, and objects with world AABBs, materials, modifiers & animation spans."
    ),
    params_doc={
        "max_objects": "Max detailed object rows before grouping instances (default 60)",
    },
)
def op_scene_passport(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    max_objs = max(5, int(params.get("max_objects", 60)))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    fps = effective_fps(sc)
    r = sc.render
    total_frames = sc.frame_end - sc.frame_start + 1

    cameras: list[dict[str, Any]] = []
    lights: list[dict[str, Any]] = []
    armatures: list[dict[str, Any]] = []
    objects_out: list[dict[str, Any]] = []
    hidden_count = 0

    type_priority = {"CAMERA": 0, "ARMATURE": 1, "LIGHT": 2, "MESH": 3, "CURVE": 4}
    all_objs = sorted(sc.objects, key=lambda o: (type_priority.get(o.type, 9), o.name))
    for idx, obj in enumerate(all_objs):
        if obj.hide_render and obj.hide_get():
            hidden_count += 1

        constraints = [
            {
                "type": c.type,
                "name": c.name,
                "target": getattr(c, "target", None).name if getattr(c, "target", None) else None,
            }
            for c in obj.constraints
        ]
        fcurves_info = summarize_fcurves(obj)
        if obj.data is not None:
            fcurves_info.extend(
                {**fc, "data_path": f"data.{fc['data_path']}"} for fc in summarize_fcurves(obj.data)
            )

        if obj.type == "CAMERA" and obj.data is not None:
            cam_d = obj.data
            cameras.append(
                {
                    "name": obj.name,
                    "active": sc.camera == obj,
                    "lens_mm": round(float(cam_d.lens), 2),
                    "sensor_width_mm": round(float(cam_d.sensor_width), 2),
                    "clip_start": round(float(cam_d.clip_start), 4),
                    "clip_end": round(float(cam_d.clip_end), 2),
                    "dof_enabled": bool(cam_d.dof.use_dof),
                    "location": [round(float(v), 3) for v in obj.matrix_world.translation],
                    "parent": obj.parent.name if obj.parent else None,
                    "constraints": constraints,
                    "animation": fcurves_info,
                }
            )
        elif obj.type == "LIGHT" and obj.data is not None:
            ld = obj.data
            lights.append(
                {
                    "name": obj.name,
                    "light_type": ld.type,
                    "energy": round(float(ld.energy), 3),
                    "color": [round(float(c), 3) for c in ld.color],
                    "location": [round(float(v), 3) for v in obj.matrix_world.translation],
                }
            )
        elif obj.type == "ARMATURE" and obj.data is not None:
            ad = obj.data
            armatures.append(
                {
                    "name": obj.name,
                    "bones": len(ad.bones),
                    "root_bones": [b.name for b in ad.bones if b.parent is None][:12],
                    "animation": fcurves_info,
                }
            )

        if idx < max_objs:
            aabb = world_aabb(obj, depsgraph)
            cols = [c.name for c in obj.users_collection]
            row: dict[str, Any] = {
                "name": obj.name,
                "type": obj.type,
                "collections": cols,
                "parent": obj.parent.name if obj.parent else None,
                "location": [round(float(v), 3) for v in obj.matrix_world.translation],
                "size": aabb["size"] if aabb else [0.0, 0.0, 0.0],
            }
            if obj.type == "MESH" and obj.data is not None:
                row["verts"] = len(obj.data.vertices)
                row["polys"] = len(obj.data.polygons)
                mats = [m.name for m in obj.data.materials if m is not None]
                if mats:
                    row["materials"] = mats
            elif obj.type == "CURVE" and obj.data is not None:
                row["splines"] = len(obj.data.splines)
                row["path_duration"] = int(obj.data.path_duration)
            elif obj.type == "ARMATURE" and obj.data is not None:
                row["bones"] = len(obj.data.bones)
            mods = [m.type for m in obj.modifiers]
            if mods:
                row["modifiers"] = mods
            if constraints:
                row["constraints"] = constraints
            if fcurves_info:
                row["animation"] = fcurves_info
            objects_out.append(row)

    world_info: dict[str, Any] | None = None
    if sc.world is not None:
        world_info = {
            "name": sc.world.name,
            "use_nodes": bool(sc.world.use_nodes),
            "color": [round(float(c), 3) for c in sc.world.color],
        }

    return {
        "filepath": bpy.data.filepath or "(unsaved)",
        "scene": sc.name,
        "engine": r.engine,
        "shading_color": sc.display.shading.color_type,
        "fps": round(fps, 4),
        "frame_range": [int(sc.frame_start), int(sc.frame_end)],
        "frame_current": int(sc.frame_current),
        "total_frames": int(total_frames),
        "duration_sec": round(total_frames / fps, 3) if fps > 0 else 0.0,
        "resolution": [int(r.resolution_x), int(r.resolution_y)],
        "film_transparent": bool(r.film_transparent),
        "active_camera": sc.camera.name if sc.camera else None,
        "world": world_info,
        "collections": [_collection_tree(c) for c in sc.collection.children],
        "cameras": cameras,
        "lights": lights,
        "armatures": armatures,
        "objects": objects_out,
        "total_scene_objects": len(sc.objects),
        "omitted_objects": max(0, len(all_objs) - max_objs),
        "hidden_objects": hidden_count,
    }


@blender_op(
    "scene.inspect",
    tier="read",
    summary=(
        "Deep inspection of one or more Blender objects: world/local transforms, AABB, mesh stats & UV/vertex "
        "groups, modifier settings, material slots, constraints, armature bone hierarchy, and F-curves."
    ),
    params_doc={
        "name": "Single object name to inspect",
        "names": "Optional list of object names to inspect in one call",
    },
)
def op_scene_inspect(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    import math

    bpy = ctx.bpy
    raw_names: list[str] = []
    if params.get("name"):
        raw_names.append(str(params["name"]).strip())
    if isinstance(params.get("names"), list):
        raw_names.extend(str(n).strip() for n in params["names"] if str(n).strip())
    if not raw_names:
        raise OpError("scene.inspect requires 'name' or 'names'", code="MISSING_PARAM")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    inspected: list[dict[str, Any]] = []

    for name in raw_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise OpError(f"Object not found: '{name}'", code="NOT_FOUND")

        aabb = world_aabb(obj, depsgraph)
        mw = obj.matrix_world
        w_loc, w_quat, w_scale = mw.decompose()
        w_eul = w_quat.to_euler("XYZ")

        info: dict[str, Any] = {
            "name": obj.name,
            "type": obj.type,
            "collections": [c.name for c in obj.users_collection],
            "parent": obj.parent.name if obj.parent else None,
            "parent_bone": obj.parent_bone or None,
            "visible_viewport": not obj.hide_get(),
            "visible_render": not obj.hide_render,
            "local_transform": {
                "location": [round(float(v), 4) for v in obj.location],
                "rotation_mode": obj.rotation_mode,
                "rotation_deg": [round(math.degrees(float(v)), 3) for v in obj.rotation_euler],
                "scale": [round(float(v), 4) for v in obj.scale],
            },
            "world_transform": {
                "location": [round(float(v), 4) for v in w_loc],
                "rotation_deg": [round(math.degrees(float(v)), 3) for v in w_eul],
                "scale": [round(float(v), 4) for v in w_scale],
            },
            "world_aabb": aabb,
            "modifiers": [
                {
                    "name": m.name,
                    "type": m.type,
                    "show_viewport": bool(m.show_viewport),
                    "show_render": bool(m.show_render),
                }
                for m in obj.modifiers
            ],
            "constraints": [
                {
                    "name": c.name,
                    "type": c.type,
                    "influence": round(float(c.influence), 3),
                    "target": getattr(c, "target", None).name if getattr(c, "target", None) else None,
                    "subtarget": getattr(c, "subtarget", "") or None,
                }
                for c in obj.constraints
            ],
            "animation": summarize_fcurves(obj),
        }

        if obj.type == "MESH" and obj.data is not None:
            md = obj.data
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            try:
                eval_verts = len(eval_mesh.vertices)
                eval_polys = len(eval_mesh.polygons)
            finally:
                eval_obj.to_mesh_clear()

            info["mesh"] = {
                "data_name": md.name,
                "vertices": len(md.vertices),
                "edges": len(md.edges),
                "polygons": len(md.polygons),
                "evaluated_vertices": eval_verts,
                "evaluated_polygons": eval_polys,
                "uv_layers": [uv.name for uv in md.uv_layers],
                "vertex_groups": [vg.name for vg in obj.vertex_groups],
                "materials": [m.name if m else None for m in md.materials],
            }
        elif obj.type == "ARMATURE" and obj.data is not None:
            ad = obj.data
            bones_out = []
            for b in ad.bones:
                pb = obj.pose.bones.get(b.name) if obj.pose else None
                bones_out.append(
                    {
                        "name": b.name,
                        "parent": b.parent.name if b.parent else None,
                        "head_local": [round(float(v), 4) for v in b.head_local],
                        "tail_local": [round(float(v), 4) for v in b.tail_local],
                        "length": round(float(b.length), 4),
                        "pose_location": [round(float(v), 4) for v in pb.location] if pb else None,
                        "pose_rotation_deg": (
                            [round(math.degrees(float(v)), 3) for v in pb.rotation_quaternion.to_euler("XYZ")]
                            if pb and pb.rotation_mode == "QUATERNION"
                            else ([round(math.degrees(float(v)), 3) for v in pb.rotation_euler] if pb else None)
                        ),
                    }
                )
            info["armature"] = {"bones_count": len(ad.bones), "bones": bones_out}
        elif obj.type == "LIGHT" and obj.data is not None:
            ld = obj.data
            info["light"] = {
                "light_type": ld.type,
                "energy": round(float(ld.energy), 3),
                "color": [round(float(c), 4) for c in ld.color],
                "use_shadow": bool(ld.use_shadow),
            }
        elif obj.type == "CAMERA" and obj.data is not None:
            cd = obj.data
            info["camera"] = {
                "lens_mm": round(float(cd.lens), 2),
                "sensor_width_mm": round(float(cd.sensor_width), 2),
                "clip": [round(float(cd.clip_start), 4), round(float(cd.clip_end), 2)],
                "dof_enabled": bool(cd.dof.use_dof),
                "focus_object": cd.dof.focus_object.name if cd.dof.focus_object else None,
                "focus_distance_m": round(float(cd.dof.focus_distance), 3),
                "aperture_fstop": round(float(cd.dof.aperture_fstop), 2),
            }

        inspected.append(info)

    return {"objects": inspected, "count": len(inspected)}


@blender_op(
    "scene.setup",
    tier="mutate",
    snapshot_blend=True,
    summary="Configure FPS, frame range, resolution, render engine ('BLENDER_EEVEE'|'CYCLES'|'BLENDER_WORKBENCH'), shading & world.",
    params_doc={
        "fps": "Frames per second (e.g. 24, 30, 60)",
        "frame_start": "Start frame (default 1)",
        "frame_end": "End frame (e.g. 120 for 5s @ 24fps)",
        "resolution": "[width, height] (default [1920, 1080])",
        "engine": "'BLENDER_EEVEE', 'CYCLES', or 'BLENDER_WORKBENCH'",
        "samples": "Optional render sample count for EEVEE/Cycles",
        "film_transparent": "If true, render transparent alpha background",
        "clear_default_objects": "If true, delete startup Cube/Light/Camera before building",
        "studio_shading": "If true (default), enable Workbench cavity ridges, outlines & shadows",
        "mist_range": "[mist_start_m, mist_depth_m] to configure World Mist / Depth pass",
        "sky_color": "[r, g, b] world background color",
    },
)
def op_scene_setup(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    if "film_transparent" in params and params["film_transparent"] is not None:
        ctx.scene.render.film_transparent = bool(params["film_transparent"])
    if params.get("samples") is not None:
        n_samp = max(1, int(params["samples"]))
        if hasattr(ctx.scene, "eevee") and hasattr(ctx.scene.eevee, "taa_render_samples"):
            ctx.scene.eevee.taa_render_samples = n_samp
        if hasattr(ctx.scene, "cycles") and hasattr(ctx.scene.cycles, "samples"):
            ctx.scene.cycles.samples = n_samp
    bpy = ctx.bpy
    sc = ctx.scene
    r = sc.render

    if params.get("clear_default_objects"):
        for name in ("Cube", "Light", "Camera"):
            obj = bpy.data.objects.get(name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)

    if params.get("fps") is not None:
        fps_val = float(params["fps"])
        if abs(fps_val - round(fps_val)) < 1e-4:
            r.fps = round(fps_val)
            r.fps_base = 1.0
        else:
            r.fps = round(fps_val * 1000)
            r.fps_base = 1000.0

    if params.get("frame_start") is not None:
        sc.frame_start = int(params["frame_start"])
    if params.get("frame_end") is not None:
        sc.frame_end = int(params["frame_end"])
    if params.get("frame_current") is not None:
        sc.frame_set(int(params["frame_current"]))

    if params.get("resolution"):
        res = params["resolution"]
        if not isinstance(res, (list, tuple)) or len(res) != 2:
            raise OpError("'resolution' must be [width, height]", code="INVALID_PARAM")
        r.resolution_x = max(64, int(res[0]))
        r.resolution_y = max(64, int(res[1]))
        r.resolution_percentage = 100

    if params.get("engine"):
        eng = str(params["engine"]).strip().upper()
        if eng not in ("BLENDER_WORKBENCH", "BLENDER_EEVEE", "CYCLES"):
            raise OpError(f"Unsupported engine '{eng}'", code="INVALID_ENGINE")
        r.engine = eng

    if params.get("studio_shading", True):
        sh = sc.display.shading
        sh.light = "STUDIO"
        sh.color_type = "MATERIAL"
        sh.show_cavity = True
        sh.cavity_type = "BOTH"
        sh.show_object_outline = True
        sh.show_shadows = True
        sh.shadow_intensity = 0.65

    if params.get("sky_color"):
        rgb = [float(x) for x in params["sky_color"][:3]]
        sc.display.shading.background_type = "WORLD"
        if sc.world is None:
            sc.world = bpy.data.worlds.new("PrevisWorld")
        sc.world.color = (rgb[0], rgb[1], rgb[2])
        if sc.world.use_nodes and sc.world.node_tree:
            bg = next((n for n in sc.world.node_tree.nodes if n.type == "BACKGROUND"), None)
            if bg and "Color" in bg.inputs:
                bg.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)

    if params.get("mist_range"):
        m_start, m_depth = float(params["mist_range"][0]), float(params["mist_range"][1])
        ctx.view_layer.use_pass_mist = True
        ctx.view_layer.use_pass_z = True
        if sc.world is None:
            sc.world = bpy.data.worlds.new("PrevisWorld")
        sc.world.mist_settings.start = m_start
        sc.world.mist_settings.depth = m_depth
        sc.world.mist_settings.falloff = "LINEAR"

    fps = effective_fps(sc)
    total = sc.frame_end - sc.frame_start + 1
    return {
        "scene": sc.name,
        "engine": r.engine,
        "fps": round(fps, 4),
        "frame_range": [int(sc.frame_start), int(sc.frame_end)],
        "duration_sec": round(total / fps, 3) if fps > 0 else 0.0,
        "resolution": [int(r.resolution_x), int(r.resolution_y)],
    }


@blender_op(
    "collection.ensure",
    tier="mutate",
    summary="Ensure one or more named Collections exist under the scene root (or `parent`).",
    params_doc={
        "names": "List of collection names (e.g. ['ENV', 'TRACK', 'VEHICLE', 'CAMERA']) or single 'name'",
        "parent": "Optional parent collection name",
    },
)
def op_collection_ensure(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    _ = ctx.scene
    names = params.get("names")
    if isinstance(names, str):
        names = [names]
    elif not names and params.get("name"):
        names = [str(params["name"])]
    if not isinstance(names, list) or not names:
        raise OpError("collection.ensure requires 'names' list or 'name'", code="MISSING_PARAM")

    parent_name = str(params["parent"]) if params.get("parent") else None
    created = [ensure_collection(str(n).strip(), parent_name=parent_name).name for n in names if str(n).strip()]
    return {"collections": created, "parent": parent_name}


@blender_op(
    "scene.save",
    tier="mutate",
    summary="Save the current Blender scene to `filepath` (or current `bpy.data.filepath`).",
    params_doc={
        "filepath": "Destination `.blend` file path",
        "copy": "If true, save a copy without changing current `bpy.data.filepath` (default false)",
    },
)
def op_scene_save(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    raw_path = str(params.get("filepath") or bpy.data.filepath or "").strip()
    if not raw_path:
        raise OpError("scene.save requires 'filepath' when the scene has not been saved yet", code="MISSING_PARAM")
    dest = Path(raw_path).resolve()
    if dest.suffix.lower() != ".blend":
        dest = dest.with_suffix(".blend")
    dest.parent.mkdir(parents=True, exist_ok=True)
    as_copy = bool(params.get("copy", False))
    bpy.ops.wm.save_as_mainfile(filepath=str(dest), copy=as_copy, check_existing=False)
    return {"filepath": str(dest), "size_bytes": dest.stat().st_size, "copy": as_copy}


@blender_op(
    "scene.open",
    tier="destructive",
    snapshot_blend=True,
    summary="Open an existing `.blend` file into the current Blender session.",
    params_doc={"filepath": "Path to `.blend` file"},
)
def op_scene_open(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    raw_path = str(params.get("filepath") or "").strip()
    if not raw_path:
        raise OpError("scene.open requires 'filepath'", code="MISSING_PARAM")
    src = Path(raw_path).resolve()
    if not src.is_file():
        raise OpError(f".blend file not found: {src}", code="NOT_FOUND")
    bpy.ops.wm.open_mainfile(filepath=str(src), load_ui=False)
    return op_blender_status(ctx, {})


@blender_op(
    "scene.clear",
    tier="destructive",
    snapshot_blend=True,
    summary="Reset Blender to a clean empty scene (removes all objects, collections, orphan meshes & actions).",
)
def op_scene_clear(ctx: OpContext, _params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return op_blender_status(ctx, {})


@blender_op(
    "scene.snapshot_save",
    tier="read",
    summary="Save an internal rollback `.blend` checkpoint for the undo/transaction journal.",
    params_doc={"dir": "Snapshot directory", "prefix": "Filename prefix"},
)
def op_scene_snapshot_save(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    snap = save_temp_snapshot(
        ctx.bpy,
        params.get("dir") or ctx.snaps_dir,
        prefix=str(params.get("prefix") or "snap"),
    )
    return {"snapshot_path": str(snap), "original_filepath": ctx.bpy.data.filepath or ""}


@blender_op(
    "scene.snapshot_restore",
    tier="mutate",
    summary="Restore a `.blend` checkpoint produced by `scene.snapshot_save`.",
    params_doc={"snapshot_path": "Path to `.blend` snapshot"},
)
def op_scene_snapshot_restore(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    snap_path = str(params.get("snapshot_path") or "").strip()
    if not snap_path:
        raise OpError("scene.snapshot_restore requires 'snapshot_path'", code="MISSING_PARAM")
    restore_snapshot(
        ctx.bpy,
        snap_path,
        original_filepath=str(params.get("original_filepath") or ""),
    )
    return op_blender_status(ctx, {})


def _unused_import_guard() -> Any:
    return iter_action_fcurves