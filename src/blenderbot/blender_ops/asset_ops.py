"""Multi-Format 3D Asset Import, Export & `.blend` Library Appending (`asset_ops.py`).

Supports `GLB` / `GLTF`, `FBX`, `OBJ`, `USD` / `USDA` / `USDC` / `USDZ`, `STL`, `PLY`, plus native
`.blend` datablock appending/linking via `bpy.data.libraries.load`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .helpers import ensure_collection, link_exclusively, world_aabb
from .schema import OpContext, OpError, blender_op


def _combined_world_aabb(objs: list[Any], depsgraph: Any) -> dict[str, list[float]] | None:
    mins: list[list[float]] = []
    maxs: list[list[float]] = []
    for o in objs:
        if o.type in ("MESH", "CURVE", "ARMATURE", "FONT", "SURFACE", "META"):
            ab = world_aabb(o, depsgraph)
            if ab is not None:
                mins.append(ab["min"])
                maxs.append(ab["max"])
    if not mins:
        return None
    mn = [min(m[i] for m in mins) for i in range(3)]
    mx = [max(m[i] for m in maxs) for i in range(3)]
    sz = [mx[i] - mn[i] for i in range(3)]
    ct = [(mn[i] + mx[i]) * 0.5 for i in range(3)]
    return {
        "min": [round(v, 4) for v in mn],
        "max": [round(v, 4) for v in mx],
        "size": [round(v, 4) for v in sz],
        "center": [round(v, 4) for v in ct],
    }


@blender_op(
    "asset.import",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Import an external 3D asset (`GLB`/`GLTF`, `FBX`, `OBJ`, `USD`/`USDA`/`USDC`/`USDZ`, `STL`, `PLY`) "
        "into the scene, optionally grouping into a collection, normalizing size, and placing on `z=0`."
    ),
    params_doc={
        "filepath": "Path to 3D file (`.glb`, `.gltf`, `.fbx`, `.obj`, `.usd`, `.usdc`, `.usda`, `.usdz`, `.stl`, `.ply`)",
        "collection": "Optional collection name to place imported objects into",
        "location": "Optional `[x, y, z]` world placement offset",
        "rotation_deg": "Optional `[rx, ry, rz]` Euler XYZ rotation in degrees applied to root imported objects",
        "target_size": "Optional max dimension in meters (uniformly rescales imported hierarchy so largest side == `target_size`)",
        "place_on_ground": "If true, shift imported hierarchy vertically so combined AABB bottom sits cleanly on `z = 0` (or `location[2]`)",
    },
)
def op_asset_import(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    raw_path = str(params.get("filepath") or "").strip()
    if not raw_path:
        raise OpError("asset.import requires 'filepath'", code="MISSING_PARAM")
    src = Path(raw_path).resolve()
    if not src.is_file():
        raise OpError(f"Asset file not found: {src}", code="NOT_FOUND")

    ext = src.suffix.lower()
    before_objs = set(bpy.data.objects)

    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(src))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(src))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(src))
    elif ext in (".usd", ".usda", ".usdc", ".usdz"):
        bpy.ops.wm.usd_import(filepath=str(src))
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=str(src))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(src))
    else:
        raise OpError(
            f"Unsupported import format '{ext}'. Supported: .glb, .gltf, .fbx, .obj, .usd/.usda/.usdc/.usdz, .stl, .ply",
            code="UNSUPPORTED_FORMAT",
        )

    imported = [o for o in bpy.data.objects if o not in before_objs]
    if not imported:
        raise OpError(f"Import succeeded for '{src.name}', but zero objects were added to the scene", code="EMPTY_IMPORT")

    col_name = str(params.get("collection") or "").strip()
    if col_name:
        col = ensure_collection(col_name)
        for o in imported:
            link_exclusively(o, col)

    imported_set = set(imported)
    roots = [o for o in imported if o.parent not in imported_set]

    if params.get("rotation_deg") is not None:
        rot = params["rotation_deg"]
        for r_obj in roots:
            if r_obj.rotation_mode == "QUATERNION":
                r_obj.rotation_euler = r_obj.rotation_quaternion.to_euler("XYZ")
                r_obj.rotation_mode = "XYZ"
            r_obj.rotation_euler = (
                r_obj.rotation_euler.x + math.radians(float(rot[0])),
                r_obj.rotation_euler.y + math.radians(float(rot[1])),
                r_obj.rotation_euler.z + math.radians(float(rot[2])),
            )

    ctx.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bbox = _combined_world_aabb(imported, depsgraph)

    if params.get("target_size") is not None and bbox is not None:
        max_dim = max(bbox["size"])
        if max_dim > 1e-6:
            factor = float(params["target_size"]) / max_dim
            for r_obj in roots:
                r_obj.scale = (r_obj.scale.x * factor, r_obj.scale.y * factor, r_obj.scale.z * factor)
            ctx.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            bbox = _combined_world_aabb(imported, depsgraph)

    target_loc = params.get("location")
    place_ground = bool(params.get("place_on_ground", False))
    if (target_loc is not None or place_ground) and bbox is not None:
        tx = float(target_loc[0]) if target_loc is not None else bbox["center"][0]
        ty = float(target_loc[1]) if target_loc is not None else bbox["center"][1]
        tz = float(target_loc[2]) if target_loc is not None else (0.0 if place_ground else bbox["center"][2])

        dx = tx - bbox["center"][0] if target_loc is not None else 0.0
        dy = ty - bbox["center"][1] if target_loc is not None else 0.0
        if place_ground:
            dz = tz - bbox["min"][2]
        elif target_loc is not None:
            dz = tz - bbox["center"][2]
        else:
            dz = 0.0

        for r_obj in roots:
            r_obj.location = (r_obj.location.x + dx, r_obj.location.y + dy, r_obj.location.z + dz)

        ctx.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bbox = _combined_world_aabb(imported, depsgraph)

    return {
        "filepath": str(src),
        "format": ext.lstrip("."),
        "imported_objects": [o.name for o in imported],
        "root_objects": [o.name for o in roots],
        "count": len(imported),
        "world_aabb": bbox,
    }


@blender_op(
    "asset.export",
    tier="mutate",
    summary=(
        "Export the scene (or specific `objects` / `collection`) to `GLB`, `GLTF`, `FBX`, `OBJ`, "
        "`USD`/`USDA`/`USDC`, `STL`, or `PLY`."
    ),
    params_doc={
        "filepath": "Destination file path (extension selects format: `.glb`, `.gltf`, `.fbx`, `.obj`, `.usdc`, `.stl`, `.ply`)",
        "objects": "Optional list of object names to export (exports entire scene if omitted)",
        "collection": "Optional collection name to export",
        "export_animations": "Include animations in GLB/FBX/USD export (default true)",
        "apply_modifiers": "Apply mesh modifiers on export (default true)",
    },
)
def op_asset_export(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    raw_path = str(params.get("filepath") or "").strip()
    if not raw_path:
        raise OpError("asset.export requires 'filepath'", code="MISSING_PARAM")

    dest = Path(raw_path).resolve()
    ext = dest.suffix.lower()
    if not ext:
        dest = dest.with_suffix(".glb")
        ext = ".glb"
    dest.parent.mkdir(parents=True, exist_ok=True)

    use_sel = False
    target_objs: list[Any] = []
    if isinstance(params.get("objects"), list) and params["objects"]:
        use_sel = True
        for nm in params["objects"]:
            o = bpy.data.objects.get(str(nm).strip())
            if o is None:
                raise OpError(f"Object not found for export: '{nm}'", code="NOT_FOUND")
            target_objs.append(o)
    elif params.get("collection"):
        use_sel = True
        col = bpy.data.collections.get(str(params["collection"]).strip())
        if col is None:
            raise OpError(f"Collection not found for export: '{params['collection']}'", code="NOT_FOUND")
        target_objs = list(col.all_objects)

    prev_sel = [o for o in ctx.view_layer.objects if o.select_get()]
    prev_active = ctx.view_layer.objects.active
    export_anim = bool(params.get("export_animations", True))
    apply_mods = bool(params.get("apply_modifiers", True))

    try:
        if use_sel:
            for o in ctx.view_layer.objects:
                o.select_set(False)
            for o in target_objs:
                if o.name in ctx.view_layer.objects:
                    o.select_set(True)
            if target_objs and target_objs[0].name in ctx.view_layer.objects:
                ctx.view_layer.objects.active = target_objs[0]

        if ext in (".glb", ".gltf"):
            gltf_fmt = "GLB" if ext == ".glb" else "GLTF_SEPARATE"
            bpy.ops.export_scene.gltf(
                filepath=str(dest),
                export_format=gltf_fmt,
                use_selection=use_sel,
                export_apply=apply_mods,
                export_animations=export_anim,
            )
        elif ext == ".fbx":
            bpy.ops.export_scene.fbx(
                filepath=str(dest),
                use_selection=use_sel,
                use_mesh_modifiers=apply_mods,
                bake_anim=export_anim,
            )
        elif ext == ".obj":
            bpy.ops.wm.obj_export(
                filepath=str(dest),
                export_selected_objects=use_sel,
                apply_modifiers=apply_mods,
            )
        elif ext in (".usd", ".usda", ".usdc"):
            bpy.ops.wm.usd_export(
                filepath=str(dest),
                selected_objects_only=use_sel,
                export_animation=export_anim,
            )
        elif ext == ".stl":
            bpy.ops.wm.stl_export(
                filepath=str(dest),
                export_selected_objects=use_sel,
                apply_modifiers=apply_mods,
            )
        elif ext == ".ply":
            bpy.ops.wm.ply_export(
                filepath=str(dest),
                export_selected_objects=use_sel,
                apply_modifiers=apply_mods,
            )
        else:
            raise OpError(
                f"Unsupported export format '{ext}'. Supported: .glb, .gltf, .fbx, .obj, .usd/.usda/.usdc, .stl, .ply",
                code="UNSUPPORTED_FORMAT",
            )
    finally:
        for o in ctx.view_layer.objects:
            o.select_set(o in prev_sel)
        ctx.view_layer.objects.active = prev_active

    if not dest.is_file():
        raise OpError(f"Export completed without creating '{dest}'", code="EXPORT_MISSING")

    return {
        "filepath": str(dest),
        "format": ext.lstrip("."),
        "size_bytes": dest.stat().st_size,
        "selected_only": use_sel,
    }


@blender_op(
    "asset.blend_append",
    tier="mutate",
    snapshot_blend=True,
    summary="Append or link objects, collections, or materials from another `.blend` file into the active scene.",
    params_doc={
        "filepath": "Source `.blend` file path",
        "objects": "Optional list of object names to append (or `'*'` for all objects)",
        "collections": "Optional list of collection names to append",
        "materials": "Optional list of material names to append",
        "link": "If true, library-link instead of appending local copies (default false)",
        "target_collection": "Optional collection in the active scene to link appended objects into",
    },
)
def op_asset_blend_append(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    raw_path = str(params.get("filepath") or "").strip()
    if not raw_path:
        raise OpError("asset.blend_append requires 'filepath'", code="MISSING_PARAM")
    src = Path(raw_path).resolve()
    if not src.is_file():
        raise OpError(f"Source .blend file not found: {src}", code="NOT_FOUND")

    do_link = bool(params.get("link", False))
    req_objs = params.get("objects")
    req_cols = params.get("collections")
    req_mats = params.get("materials")

    with bpy.data.libraries.load(str(src), link=do_link) as (data_from, data_to):
        if req_objs == "*":
            data_to.objects = list(data_from.objects)
        elif isinstance(req_objs, list):
            missing = [n for n in req_objs if n not in data_from.objects]
            if missing:
                raise OpError(f"Objects {missing} not found in '{src.name}' (available: {list(data_from.objects)})", code="NOT_FOUND")
            data_to.objects = list(req_objs)

        if req_cols == "*":
            data_to.collections = list(data_from.collections)
        elif isinstance(req_cols, list):
            missing_c = [n for n in req_cols if n not in data_from.collections]
            if missing_c:
                raise OpError(
                    f"Collections {missing_c} not found in '{src.name}' (available: {list(data_from.collections)})",
                    code="NOT_FOUND",
                )
            data_to.collections = list(req_cols)

        if req_mats == "*":
            data_to.materials = list(data_from.materials)
        elif isinstance(req_mats, list):
            missing_m = [n for n in req_mats if n not in data_from.materials]
            if missing_m:
                raise OpError(
                    f"Materials {missing_m} not found in '{src.name}' (available: {list(data_from.materials)})",
                    code="NOT_FOUND",
                )
            data_to.materials = list(req_mats)

    dest_col = (
        ensure_collection(str(params["target_collection"]).strip())
        if params.get("target_collection")
        else ctx.scene.collection
    )

    appended_objs: list[str] = []
    for obj in data_to.objects:
        if obj is not None:
            if obj.name not in dest_col.objects:
                dest_col.objects.link(obj)
            appended_objs.append(obj.name)

    appended_cols: list[str] = []
    for col in data_to.collections:
        if col is not None:
            if col.name not in ctx.scene.collection.children:
                ctx.scene.collection.children.link(col)
            appended_cols.append(col.name)

    appended_mats = [m.name for m in data_to.materials if m is not None]
    ctx.view_layer.update()

    return {
        "filepath": str(src),
        "linked": do_link,
        "objects": appended_objs,
        "collections": appended_cols,
        "materials": appended_mats,
    }