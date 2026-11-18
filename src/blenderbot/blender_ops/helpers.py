"""Blender 5.3+ stdlib/bpy helpers: slotted Action F-curves, world AABBs, materials & snapshots."""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

from .schema import OpError


def iter_action_fcurves(id_block: Any) -> list[Any]:
    """Return all FCurves driving `id_block` across Blender 5.x slotted Actions (and 4.x fallback).

    In Blender 5.0+, `Action.fcurves` was removed in favor of layered/slotted Channelbags:
    `bpy_extras.anim_utils.action_get_channelbag_for_slot(action, anim_data.action_slot)`.
    """
    anim = getattr(id_block, "animation_data", None)
    if anim is None or getattr(anim, "action", None) is None:
        return []
    action = anim.action

    # Legacy Blender <=4.x fallback if ever encountered
    if hasattr(action, "fcurves"):
        return list(action.fcurves)

    try:
        from bpy_extras import anim_utils

        slot = getattr(anim, "action_slot", None)
        if slot is not None:
            cb = anim_utils.action_get_channelbag_for_slot(action, slot)
            if cb is not None and hasattr(cb, "fcurves"):
                return list(cb.fcurves)
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Fallback: walk layers -> strips -> channelbags
    curves: list[Any] = []
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for cb in getattr(strip, "channelbags", ()):
                curves.extend(getattr(cb, "fcurves", ()))
    return curves


def ensure_channelbag_fcurves(id_block: Any, *, action_name: str | None = None) -> Any:
    """Ensure `id_block` has a Blender 5.3+ Slotted Action + Channelbag and return `.fcurves`.

    When `action_name` is provided, reuses an existing `bpy.data.actions[action_name]` if present
    (or creates it) and binds/reuses a matching `(id_type, id_block.name)` Action Slot inside it —
    enabling Blender 5.x Multi-ID Slotted Actions ('Project Baklava') where a single Action drives
    multiple objects, armatures, cameras, lights, shape keys, and materials simultaneously.
    """
    import bpy

    anim = id_block.animation_data_create()
    if action_name:
        target_act = bpy.data.actions.get(action_name)
        if target_act is None:
            target_act = bpy.data.actions.new(name=action_name)
        if anim.action != target_act:
            anim.action = target_act
            if hasattr(anim, "action_slot"):
                anim.action_slot = None
    elif anim.action is None:
        anim.action = bpy.data.actions.new(name=f"{id_block.name}_Action")
    act = anim.action

    if hasattr(act, "fcurves"):
        return act.fcurves

    from bpy_extras import anim_utils

    if anim.action_slot is None:
        id_type = getattr(id_block, "id_type", "OBJECT")
        existing_slot = next(
            (
                s
                for s in act.slots
                if getattr(s, "target_id_type", None) == id_type
                and (s.name_display == id_block.name or s.identifier.endswith(id_block.name))
            ),
            None,
        )
        anim.action_slot = (
            existing_slot
            if existing_slot is not None
            else act.slots.new(id_type=id_type, name=id_block.name)
        )

    cb = anim_utils.action_ensure_channelbag_for_slot(act, anim.action_slot)
    return cb.fcurves


def summarize_fcurves(id_block: Any) -> list[dict[str, Any]]:
    """Compact summary of animated data_paths and keyframe frame spans on `id_block`."""
    grouped: dict[str, dict[str, Any]] = {}
    for fc in iter_action_fcurves(id_block):
        dp = str(fc.data_path)
        kps = fc.keyframe_points
        n_keys = len(kps)
        f_min = round(float(kps[0].co[0]), 2) if n_keys else None
        f_max = round(float(kps[-1].co[0]), 2) if n_keys else None
        entry = grouped.setdefault(
            dp,
            {"data_path": dp, "channels": 0, "keys": 0, "frame_range": [f_min, f_max]},
        )
        entry["channels"] += 1
        entry["keys"] += n_keys
        if f_min is not None:
            cur_min = entry["frame_range"][0]
            entry["frame_range"][0] = min(cur_min, f_min) if cur_min is not None else f_min
        if f_max is not None:
            cur_max = entry["frame_range"][1]
            entry["frame_range"][1] = max(cur_max, f_max) if cur_max is not None else f_max
    return list(grouped.values())


def world_aabb(obj: Any, depsgraph: Any | None = None) -> dict[str, list[float]] | None:
    """Compute world-space axis-aligned bounding box (`min`, `max`, `size`, `center`) for `obj`."""
    from mathutils import Vector

    target = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
    bb = getattr(target, "bound_box", None)
    if not bb:
        loc = target.matrix_world.translation
        pt = [round(float(loc.x), 4), round(float(loc.y), 4), round(float(loc.z), 4)]
        return {"min": pt, "max": pt, "size": [0.0, 0.0, 0.0], "center": pt}

    mw = target.matrix_world
    pts = [mw @ Vector(corner) for corner in bb]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    mn = [min(xs), min(ys), min(zs)]
    mx = [max(xs), max(ys), max(zs)]
    sz = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]]
    ct = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5]
    return {
        "min": [round(v, 4) for v in mn],
        "max": [round(v, 4) for v in mx],
        "size": [round(v, 4) for v in sz],
        "center": [round(v, 4) for v in ct],
    }


def ensure_collection(name: str, parent_name: str | None = None) -> Any:
    """Get or create a Blender Collection and link it into the scene (or `parent_name`)."""
    import bpy

    scene_root = bpy.context.scene.collection
    parent = scene_root
    if parent_name:
        parent = bpy.data.collections.get(parent_name)
        if parent is None:
            parent = bpy.data.collections.new(parent_name)
            scene_root.children.link(parent)

    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        parent.children.link(col)
    else:
        if col.name not in parent.children and col != parent:
            # Unlink from scene_root if moving under a specific parent
            if parent != scene_root and col.name in scene_root.children:
                scene_root.children.unlink(col)
            if col.name not in parent.children:
                parent.children.link(col)
    return col


def link_exclusively(obj: Any, collection: Any) -> None:
    """Link `obj` into `collection` and unlink it from all other collections."""
    for c in list(obj.users_collection):
        if c != collection:
            c.objects.unlink(obj)
    if obj.name not in collection.objects:
        collection.objects.link(obj)


def assign_solid_material(
    obj: Any,
    name: str,
    rgba: tuple[float, float, float, float] | list[float],
    *,
    roughness: float = 0.45,
    metallic: float = 0.0,
) -> Any:
    """Create/update a clean Principled BSDF + viewport-colored previs material and assign to `obj`."""
    import bpy

    if len(rgba) == 3:
        color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
    else:
        color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))

    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.diffuse_color = color
    mat.roughness = float(roughness)
    mat.metallic = float(metallic)

    if mat.node_tree:
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = color
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = float(roughness)
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = float(metallic)

    data = getattr(obj, "data", None)
    if data is not None and hasattr(data, "materials"):
        if len(data.materials) == 0:
            data.materials.append(mat)
        else:
            data.materials[0] = mat
    obj.color = color
    return mat


def save_temp_snapshot(bpy_mod: Any, snaps_dir: str | Path | None = None, *, prefix: str = "tx") -> Path:
    """Save a copy-only `.blend` checkpoint without altering `bpy.data.filepath`."""
    base = Path(snaps_dir) if snaps_dir else Path(tempfile.gettempdir()) / "blenderbot_snaps"
    base.mkdir(parents=True, exist_ok=True)
    stamp = f"{prefix}_{int(time.time() * 1000)}.blend"
    snap_path = (base / stamp).resolve()
    bpy_mod.ops.wm.save_as_mainfile(filepath=str(snap_path), copy=True, check_existing=False)
    return snap_path


def restore_snapshot(bpy_mod: Any, snap_path: str | Path, *, original_filepath: str = "") -> None:
    """Reload `snap_path` cleanly into Blender so failed transactions leave zero orphan datablocks."""
    p = Path(snap_path).resolve()
    if not p.is_file():
        raise OpError(f"Snapshot file not found for rollback: {p}", code="MISSING_SNAPSHOT")
    bpy_mod.ops.wm.open_mainfile(filepath=str(p), load_ui=False)
    # Note: if original file was unsaved (""), keep working in memory; if it had a saved path,
    # save_as_mainfile(copy=False) is only done when the user explicitly saves.
    _ = original_filepath


def effective_fps(scene: Any) -> float:
    base = float(getattr(scene.render, "fps_base", 1.0) or 1.0)
    return float(scene.render.fps) / base


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return default
    return float(val)