"""General 3D Modeling, Object Transforms, Parenting, Origins, Shading & Modifiers (`modeling_ops.py`).

All operations use direct `bpy.data` / `bmesh` / `evaluated_get` APIs rather than UI-context-sensitive
`bpy.ops.object.*` calls, so they run identically in live GUI `bpy.app.timers` and headless workers.
"""

from __future__ import annotations

import math
from typing import Any

from .helpers import (
    assign_solid_material,
    ensure_collection,
    link_exclusively,
    world_aabb,
)
from .schema import OpContext, OpError, blender_op

PRIMITIVE_SHAPES = (
    "cube",
    "uv_sphere",
    "ico_sphere",
    "cylinder",
    "cone",
    "plane",
    "grid",
    "torus",
    "monkey",
)


def _apply_smooth_shading(mesh: Any, smooth: bool) -> None:
    if hasattr(mesh, " shade_smooth") and False:  # pragma: no cover
        pass
    for poly in mesh.polygons:
        poly.use_smooth = bool(smooth)


def _build_torus_bmesh(
    bm: Any,
    *,
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    major_segments: int = 48,
    minor_segments: int = 16,
) -> None:
    rings: list[list[Any]] = []
    for i in range(major_segments):
        theta = (i / major_segments) * math.tau
        ct, st = math.cos(theta), math.sin(theta)
        ring: list[Any] = []
        for j in range(minor_segments):
            phi = (j / minor_segments) * math.tau
            cp, sp = math.cos(phi), math.sin(phi)
            r = major_radius + minor_radius * cp
            x = r * ct
            y = r * st
            z = minor_radius * sp
            ring.append(bm.verts.new((x, y, z)))
        rings.append(ring)
    bm.verts.ensure_lookup_table()

    for i in range(major_segments):
        i_next = (i + 1) % major_segments
        for j in range(minor_segments):
            j_next = (j + 1) % minor_segments
            v0 = rings[i][j]
            v1 = rings[i_next][j]
            v2 = rings[i_next][j_next]
            v3 = rings[i][j_next]
            bm.faces.new((v0, v1, v2, v3))


@blender_op(
    "mesh.primitive",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Create or replace a 3D mesh primitive (`cube`, `uv_sphere`, `ico_sphere`, `cylinder`, `cone`, "
        "`plane`, `grid`, `torus`, `monkey`) with transform, collection, shading & optional material."
    ),
    params_doc={
        "name": "Object name (e.g. 'Pedestal', 'HeroSphere')",
        "shape": "One of: cube, uv_sphere, ico_sphere, cylinder, cone, plane, grid, torus, monkey",
        "size": "Side length / diameter for cube, plane, grid (default 2.0)",
        "dimensions": "Optional exact `[dx, dy, dz]` dimensions in local space (overrides uniform size)",
        "radius": "Radius for uv_sphere, ico_sphere, cylinder, cone (default 1.0)",
        "radius_top": "Top radius for cone / tapered cylinder (default 0.0 for cone)",
        "depth": "Height/depth along Z for cylinder or cone (default 2.0)",
        "major_radius": "Torus major radius (default 1.0)",
        "minor_radius": "Torus minor tube radius (default 0.25)",
        "segments": "Radial segments for sphere/cylinder/cone/torus (default 32)",
        "rings": "Latitude rings for uv_sphere or minor segments for torus (default 16)",
        "subdivisions": "Subdivisions for ico_sphere (default 2) or segments per axis for grid (default 16)",
        "location": "`[x, y, z]` world location (default `[0, 0, 0]`)",
        "rotation_deg": "`[rx, ry, rz]` Euler XYZ degrees (default `[0, 0, 0]`)",
        "scale": "`[sx, sy, sz]` scale (default `[1, 1, 1]`)",
        "collection": "Target collection name (optional; links to scene root if omitted)",
        "material": "Existing material name to assign, OR created when `color` is also supplied",
        "color": "Optional `[r, g, b]` or `[r, g, b, a]` Principled BSDF base color",
        "roughness": "Material roughness when `color` is supplied (default 0.45)",
        "metallic": "Material metallic when `color` is supplied (default 0.0)",
        "smooth_shade": "If true, enable smooth polygon shading",
        "replace_existing": "Replace existing object of same name if present (default true)",
    },
)
def op_mesh_primitive(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    import bmesh
    from mathutils import Matrix

    bpy = ctx.bpy
    shape = str(params.get("shape") or "cube").strip().lower()
    if shape not in PRIMITIVE_SHAPES:
        raise OpError(
            f"Unsupported mesh.primitive shape '{shape}'. Expected one of: {', '.join(PRIMITIVE_SHAPES)}",
            code="INVALID_PARAM",
        )

    raw_name = str(params.get("name") or shape.capitalize()).strip()
    if bool(params.get("replace_existing", True)):
        existing = bpy.data.objects.get(raw_name)
        if existing is not None:
            old_data = existing.data if existing.type == "MESH" else None
            bpy.data.objects.remove(existing, do_unlink=True)
            if old_data is not None and old_data.users == 0:
                bpy.data.meshes.remove(old_data)

    size = float(params.get("size", 2.0))
    radius = float(params.get("radius", 1.0))
    depth = float(params.get("depth", 2.0))
    segments = max(3, int(params.get("segments", 32)))
    rings = max(3, int(params.get("rings", 16)))

    bm = bmesh.new()
    try:
        if shape == "cube":
            bmesh.ops.create_cube(bm, size=size, calc_uvs=True)
        elif shape == "plane":
            bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size * 0.5, calc_uvs=True)
        elif shape == "grid":
            sub = max(1, int(params.get("subdivisions", 16)))
            bmesh.ops.create_grid(bm, x_segments=sub, y_segments=sub, size=size * 0.5, calc_uvs=True)
        elif shape == "uv_sphere":
            bmesh.ops.create_uvsphere(
                bm, u_segments=segments, v_segments=rings, radius=radius, calc_uvs=True
            )
        elif shape == "ico_sphere":
            sub = max(1, min(6, int(params.get("subdivisions", 2))))
            bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=radius, calc_uvs=True)
        elif shape == "cylinder":
            r_top = float(params.get("radius_top", radius))
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=segments,
                radius1=radius,
                radius2=r_top,
                depth=depth,
                calc_uvs=True,
            )
        elif shape == "cone":
            r_top = float(params.get("radius_top", 0.0))
            bmesh.ops.create_cone(
                bm,
                cap_ends=True,
                cap_tris=False,
                segments=segments,
                radius1=radius,
                radius2=r_top,
                depth=depth,
                calc_uvs=True,
            )
        elif shape == "torus":
            maj_r = float(params.get("major_radius", 1.0))
            min_r = float(params.get("minor_radius", 0.25))
            _build_torus_bmesh(
                bm,
                major_radius=maj_r,
                minor_radius=min_r,
                major_segments=segments,
                minor_segments=rings,
            )
        elif shape == "monkey":
            bmesh.ops.create_monkey(bm)
            if abs(size - 2.0) > 1e-6:
                s_fac = size / 2.0
                bmesh.ops.scale(bm, vec=(s_fac, s_fac, s_fac), verts=bm.verts)

        dims = params.get("dimensions")
        if isinstance(dims, (list, tuple)) and len(dims) == 3 and bm.verts:
            xs = [v.co.x for v in bm.verts]
            ys = [v.co.y for v in bm.verts]
            zs = [v.co.z for v in bm.verts]
            cur_dx = max(xs) - min(xs)
            cur_dy = max(ys) - min(ys)
            cur_dz = max(zs) - min(zs)
            sx = float(dims[0]) / cur_dx if cur_dx > 1e-6 else 1.0
            sy = float(dims[1]) / cur_dy if cur_dy > 1e-6 else 1.0
            sz = float(dims[2]) / cur_dz if cur_dz > 1e-6 else 1.0
            bmesh.ops.transform(bm, matrix=Matrix.Diagonal((sx, sy, sz, 1.0)), verts=bm.verts)

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        mesh = bpy.data.meshes.new(f"{raw_name}_Mesh")
        bm.to_mesh(mesh)
    finally:
        bm.free()

    smooth_default = shape in ("uv_sphere", "torus", "monkey")
    smooth = bool(params.get("smooth_shade", smooth_default))
    _apply_smooth_shading(mesh, smooth)

    obj = bpy.data.objects.new(raw_name, mesh)
    col_name = str(params.get("collection") or "").strip()
    if col_name:
        col = ensure_collection(col_name)
        link_exclusively(obj, col)
    else:
        ctx.scene.collection.objects.link(obj)

    loc = params.get("location") or [0.0, 0.0, 0.0]
    rot_deg = params.get("rotation_deg") or [0.0, 0.0, 0.0]
    scl = params.get("scale") or [1.0, 1.0, 1.0]
    obj.rotation_mode = "XYZ"
    obj.location = (float(loc[0]), float(loc[1]), float(loc[2]))
    obj.rotation_euler = tuple(math.radians(float(a)) for a in rot_deg[:3])
    obj.scale = (float(scl[0]), float(scl[1]), float(scl[2]))

    mat_name = str(params.get("material") or "").strip()
    color = params.get("color")
    if color is not None:
        assign_solid_material(
            obj,
            mat_name or f"Mat_{obj.name}",
            [float(c) for c in color],
            roughness=float(params.get("roughness", 0.45)),
            metallic=float(params.get("metallic", 0.0)),
        )
    elif mat_name:
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            raise OpError(f"Material not found: '{mat_name}'", code="NOT_FOUND")
        mesh.materials.append(mat)

    ctx.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    aabb = world_aabb(obj, depsgraph)
    return {
        "name": obj.name,
        "shape": shape,
        "collection": col_name or ctx.scene.collection.name,
        "vertices": len(mesh.vertices),
        "polygons": len(mesh.polygons),
        "smooth_shade": smooth,
        "world_aabb": aabb,
    }


@blender_op(
    "object.transform",
    tier="mutate",
    snapshot_blend=True,
    summary="Set or offset an object's location, rotation (`rotation_deg`), or scale, and optionally bake/apply transforms into mesh vertices.",
    params_doc={
        "name": "Target object name",
        "location": "Absolute `[x, y, z]` location",
        "delta_location": "Additive `[dx, dy, dz]` offset",
        "rotation_deg": "Absolute `[rx, ry, rz]` Euler XYZ degrees",
        "delta_rotation_deg": "Additive `[drx, dry, drz]` degrees",
        "scale": "Absolute `[sx, sy, sz]` or uniform float scale",
        "apply_scale": "If true on a MESH object, bake scale into vertices and reset `obj.scale = [1, 1, 1]`",
        "apply_rotation": "If true on a MESH object, bake rotation into vertices and reset `obj.rotation_euler = [0, 0, 0]`",
        "apply_location": "If true on a MESH object, bake location into vertices and reset `obj.location = [0, 0, 0]`",
    },
)
def op_object_transform(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    from mathutils import Euler, Matrix

    bpy = ctx.bpy
    name = str(params.get("name") or "").strip()
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise OpError(f"Object not found: '{name}'", code="NOT_FOUND")

    if obj.rotation_mode == "QUATERNION" and (
        params.get("rotation_deg") is not None or params.get("delta_rotation_deg") is not None
    ):
        obj.rotation_euler = obj.rotation_quaternion.to_euler("XYZ")
        obj.rotation_mode = "XYZ"

    if params.get("location") is not None:
        loc = params["location"]
        obj.location = (float(loc[0]), float(loc[1]), float(loc[2]))
    if params.get("delta_location") is not None:
        dl = params["delta_location"]
        obj.location = (obj.location.x + float(dl[0]), obj.location.y + float(dl[1]), obj.location.z + float(dl[2]))

    if params.get("rotation_deg") is not None:
        rot = params["rotation_deg"]
        obj.rotation_euler = tuple(math.radians(float(a)) for a in rot[:3])
    if params.get("delta_rotation_deg") is not None:
        dr = params["delta_rotation_deg"]
        obj.rotation_euler = (
            obj.rotation_euler.x + math.radians(float(dr[0])),
            obj.rotation_euler.y + math.radians(float(dr[1])),
            obj.rotation_euler.z + math.radians(float(dr[2])),
        )

    if params.get("scale") is not None:
        sc_val = params["scale"]
        if isinstance(sc_val, (int, float)):
            s = float(sc_val)
            obj.scale = (s, s, s)
        else:
            obj.scale = (float(sc_val[0]), float(sc_val[1]), float(sc_val[2]))

    ap_loc = bool(params.get("apply_location", False))
    ap_rot = bool(params.get("apply_rotation", False))
    ap_scl = bool(params.get("apply_scale", False))
    if (ap_loc or ap_rot or ap_scl) and obj.type == "MESH" and obj.data is not None:
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        mat = Matrix.Identity(4)
        if ap_loc:
            mat = mat @ Matrix.Translation(obj.location)
        if ap_rot:
            eul = Euler(obj.rotation_euler, obj.rotation_mode if obj.rotation_mode != "QUATERNION" else "XYZ")
            mat = mat @ eul.to_matrix().to_4x4()
        if ap_scl:
            mat = mat @ Matrix.Diagonal((obj.scale.x, obj.scale.y, obj.scale.z, 1.0))

        obj.data.transform(mat)
        if obj.scale.x * obj.scale.y * obj.scale.z < 0 and ap_scl:
            obj.data.flip_normals()
        obj.data.update()

        if ap_loc:
            obj.location = (0.0, 0.0, 0.0)
        if ap_rot:
            obj.rotation_euler = (0.0, 0.0, 0.0)
        if ap_scl:
            obj.scale = (1.0, 1.0, 1.0)

    ctx.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return {
        "name": obj.name,
        "location": [round(float(v), 4) for v in obj.location],
        "rotation_deg": [round(math.degrees(float(v)), 3) for v in obj.rotation_euler],
        "scale": [round(float(v), 4) for v in obj.scale],
        "world_aabb": world_aabb(obj, depsgraph),
    }


@blender_op(
    "object.duplicate",
    tier="mutate",
    snapshot_blend=True,
    summary="Duplicate an object (deep mesh copy or linked instance) with optional new name, transform & collection.",
    params_doc={
        "source": "Source object name to duplicate",
        "name": "New object name (optional)",
        "linked": "If true, share datablock (`obj.data`) as an instance; if false (default), copy datablock",
        "location": "Optional `[x, y, z]` location override",
        "rotation_deg": "Optional `[rx, ry, rz]` Euler degrees override",
        "scale": "Optional `[sx, sy, sz]` or float scale override",
        "collection": "Optional destination collection name",
    },
)
def op_object_duplicate(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    src_name = str(params.get("source") or "").strip()
    src = bpy.data.objects.get(src_name)
    if src is None:
        raise OpError(f"Source object not found: '{src_name}'", code="NOT_FOUND")

    dup = src.copy()
    linked = bool(params.get("linked", False))
    if not linked and src.data is not None:
        dup.data = src.data.copy()
    if params.get("name"):
        dup.name = str(params["name"]).strip()

    col_name = str(params.get("collection") or "").strip()
    if col_name:
        col = ensure_collection(col_name)
        col.objects.link(dup)
    elif src.users_collection:
        src.users_collection[0].objects.link(dup)
    else:
        ctx.scene.collection.objects.link(dup)

    if params.get("location") is not None:
        loc = params["location"]
        dup.location = (float(loc[0]), float(loc[1]), float(loc[2]))
    if params.get("rotation_deg") is not None:
        rot = params["rotation_deg"]
        dup.rotation_mode = "XYZ"
        dup.rotation_euler = tuple(math.radians(float(a)) for a in rot[:3])
    if params.get("scale") is not None:
        sc_val = params["scale"]
        if isinstance(sc_val, (int, float)):
            s = float(sc_val)
            dup.scale = (s, s, s)
        else:
            dup.scale = (float(sc_val[0]), float(sc_val[1]), float(sc_val[2]))

    ctx.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return {
        "name": dup.name,
        "source": src.name,
        "linked": linked,
        "world_aabb": world_aabb(dup, depsgraph),
    }


@blender_op(
    "object.delete",
    tier="mutate",
    snapshot_blend=True,
    summary="Delete one or more objects (and optionally their child hierarchy and zero-user datablocks).",
    params_doc={
        "name": "Object name to delete",
        "names": "List of object names to delete",
        "hierarchy": "If true, recursively delete child objects too (default false)",
    },
)
def op_object_delete(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    targets: list[str] = []
    if params.get("name"):
        targets.append(str(params["name"]).strip())
    if isinstance(params.get("names"), list):
        targets.extend(str(n).strip() for n in params["names"] if str(n).strip())
    if not targets:
        raise OpError("object.delete requires 'name' or 'names'", code="MISSING_PARAM")

    include_children = bool(params.get("hierarchy", False))
    to_remove: list[Any] = []
    seen: set[str] = set()

    def _collect(o: Any) -> None:
        if o.name in seen:
            return
        seen.add(o.name)
        if include_children:
            for ch in list(o.children):
                _collect(ch)
        to_remove.append(o)

    for nm in targets:
        obj = bpy.data.objects.get(nm)
        if obj is None:
            raise OpError(f"Object not found: '{nm}'", code="NOT_FOUND")
        _collect(obj)

    removed_names: list[str] = []
    for o in to_remove:
        removed_names.append(o.name)
        data = o.data
        o_type = o.type
        bpy.data.objects.remove(o, do_unlink=True)
        if data is not None and getattr(data, "users", 1) == 0:
            if o_type == "MESH":
                bpy.data.meshes.remove(data)
            elif o_type == "CURVE":
                bpy.data.curves.remove(data)
            elif o_type == "LIGHT":
                bpy.data.lights.remove(data)
            elif o_type == "CAMERA":
                bpy.data.cameras.remove(data)
            elif o_type == "ARMATURE":
                bpy.data.armatures.remove(data)

    ctx.view_layer.update()
    return {"deleted": removed_names, "count": len(removed_names)}


@blender_op(
    "object.parent",
    tier="mutate",
    snapshot_blend=True,
    summary="Parent `child` (or `children`) to `parent` (or armature `bone`), optionally preserving world transforms. Omit `parent` to unparent.",
    params_doc={
        "child": "Child object name",
        "children": "List of child object names",
        "parent": "Parent object name (or null/empty string to clear parent)",
        "bone": "Optional bone name when `parent` is an ARMATURE",
        "keep_transform": "Preserve current world-space transform when parenting/unparenting (default true)",
    },
)
def op_object_parent(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    child_names: list[str] = []
    if params.get("child"):
        child_names.append(str(params["child"]).strip())
    if isinstance(params.get("children"), list):
        child_names.extend(str(c).strip() for c in params["children"] if str(c).strip())
    if not child_names:
        raise OpError("object.parent requires 'child' or 'children'", code="MISSING_PARAM")

    parent_name = str(params.get("parent") or "").strip()
    parent_obj = bpy.data.objects.get(parent_name) if parent_name else None
    if parent_name and parent_obj is None:
        raise OpError(f"Parent object not found: '{parent_name}'", code="NOT_FOUND")

    bone_name = str(params.get("bone") or "").strip()
    if bone_name and (
        parent_obj is None
        or parent_obj.type != "ARMATURE"
        or bone_name not in parent_obj.data.bones
    ):
        raise OpError(f"Bone '{bone_name}' not found on armature '{parent_name}'", code="NOT_FOUND")

    keep_tf = bool(params.get("keep_transform", True))
    ctx.view_layer.update()

    updated: list[str] = []
    for cn in child_names:
        ch = bpy.data.objects.get(cn)
        if ch is None:
            raise OpError(f"Child object not found: '{cn}'", code="NOT_FOUND")
        orig_mw = ch.matrix_world.copy()
        if parent_obj is None:
            ch.parent = None
            ch.parent_type = "OBJECT"
            ch.parent_bone = ""
            if keep_tf:
                ch.matrix_world = orig_mw
        elif bone_name:
            ch.parent = parent_obj
            ch.parent_type = "BONE"
            ch.parent_bone = bone_name
            if keep_tf:
                pb = parent_obj.pose.bones[bone_name]
                # Blender bone parent attaches relative to the bone TAIL in armature world space
                from mathutils import Matrix

                bone_world_tail = parent_obj.matrix_world @ pb.matrix @ Matrix.Translation((0.0, pb.bone.length, 0.0))
                ch.matrix_parent_inverse = bone_world_tail.inverted()
                ch.matrix_world = orig_mw
        else:
            ch.parent = parent_obj
            ch.parent_type = "OBJECT"
            ch.parent_bone = ""
            if keep_tf:
                ch.matrix_parent_inverse = parent_obj.matrix_world.inverted()
                ch.matrix_world = orig_mw
        updated.append(ch.name)

    ctx.view_layer.update()
    return {"children": updated, "parent": parent_obj.name if parent_obj else None, "bone": bone_name or None}


@blender_op(
    "object.origin",
    tier="mutate",
    snapshot_blend=True,
    summary="Move a mesh object's pivot/origin (`GEOMETRY_CENTER`, `BOTTOM_CENTER`, `WORLD_ORIGIN`, or `[x,y,z]`) without shifting its world geometry.",
    params_doc={
        "name": "Mesh object name",
        "mode": "'GEOMETRY_CENTER', 'BOTTOM_CENTER' (great for placing props/characters on z=0!), 'WORLD_ORIGIN', or 'POINT'",
        "point": "World `[x, y, z]` pivot coordinates when `mode == 'POINT'`",
    },
)
def op_object_origin(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    from mathutils import Matrix, Vector

    bpy = ctx.bpy
    name = str(params.get("name") or "").strip()
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH" or obj.data is None:
        raise OpError(f"Mesh object not found: '{name}'", code="NOT_FOUND")
    if obj.data.users > 1:
        obj.data = obj.data.copy()

    ctx.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    aabb = world_aabb(obj, depsgraph)
    assert aabb is not None

    mode = str(params.get("mode") or "GEOMETRY_CENTER").strip().upper()
    if mode == "GEOMETRY_CENTER":
        target_world = Vector(aabb["center"])
    elif mode == "BOTTOM_CENTER":
        target_world = Vector((aabb["center"][0], aabb["center"][1], aabb["min"][2]))
    elif mode == "WORLD_ORIGIN":
        target_world = Vector((0.0, 0.0, 0.0))
    elif mode == "POINT":
        pt = params.get("point")
        if not isinstance(pt, (list, tuple)) or len(pt) != 3:
            raise OpError("object.origin with mode='POINT' requires 3D 'point'", code="INVALID_PARAM")
        target_world = Vector((float(pt[0]), float(pt[1]), float(pt[2])))
    else:
        raise OpError(
            f"Unsupported origin mode '{mode}'. Use GEOMETRY_CENTER, BOTTOM_CENTER, WORLD_ORIGIN, or POINT.",
            code="INVALID_PARAM",
        )

    mw = obj.matrix_world.copy()
    local_pivot = mw.inverted() @ target_world
    obj.data.transform(Matrix.Translation(-local_pivot))
    obj.data.update()

    new_mw = mw.copy()
    new_mw.translation = target_world
    obj.matrix_world = new_mw
    ctx.view_layer.update()

    return {
        "name": obj.name,
        "mode": mode,
        "origin_world": [round(float(v), 4) for v in obj.matrix_world.translation],
        "world_aabb": world_aabb(obj, bpy.context.evaluated_depsgraph_get()),
    }


@blender_op(
    "object.shade",
    tier="mutate",
    snapshot_blend=True,
    summary="Set smooth, flat, or angle-based sharp-edge auto-smooth shading on a Mesh object.",
    params_doc={
        "name": "Mesh object name",
        "smooth": "True for smooth shading, false for flat shading (default true)",
        "auto_smooth_deg": "Optional angle threshold in degrees (e.g. 30.0) — marks edges sharper than this angle as sharp",
    },
)
def op_object_shade(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    import bmesh

    bpy = ctx.bpy
    name = str(params.get("name") or "").strip()
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "MESH" or obj.data is None:
        raise OpError(f"Mesh object not found: '{name}'", code="NOT_FOUND")

    smooth = bool(params.get("smooth", True))
    mesh = obj.data
    _apply_smooth_shading(mesh, smooth)

    sharp_edges = 0
    if params.get("auto_smooth_deg") is not None and smooth:
        thresh_rad = math.radians(float(params["auto_smooth_deg"]))
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            for edge in bm.edges:
                if len(edge.link_faces) == 2:
                    angle = edge.calc_face_angle(0.0)
                    is_sharp = angle > thresh_rad
                    edge.smooth = not is_sharp
                    if is_sharp:
                        sharp_edges += 1
                else:
                    edge.smooth = True
            bm.to_mesh(mesh)
        finally:
            bm.free()
    else:
        for edge in mesh.edges:
            edge.use_edge_sharp = False

    mesh.update()
    return {"name": obj.name, "smooth": smooth, "sharp_edges": sharp_edges}


@blender_op(
    "modifier.configure",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Add or update a non-destructive modifier (`SUBSURF`, `BEVEL`, `MIRROR`, `ARRAY`, `SOLIDIFY`, "
        "`BOOLEAN`, `DISPLACE`, `SIMPLE_DEFORM`, `WIREFRAME`, `DECIMATE`, `SHRINKWRAP`) on an object."
    ),
    params_doc={
        "object": "Target object name",
        "type": "Modifier type (e.g. 'SUBSURF', 'BEVEL', 'MIRROR', 'ARRAY', 'SOLIDIFY', 'BOOLEAN', 'SIMPLE_DEFORM', 'WIREFRAME', 'DECIMATE')",
        "name": "Optional modifier name (reuses existing modifier if name matches)",
        "settings": "Dict of modifier properties (e.g. `{'levels': 2, 'render_levels': 2}` or `{'width': 0.03, 'segments': 3}`)",
    },
)
def op_modifier_configure(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    obj_name = str(params.get("object") or "").strip()
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise OpError(f"Object not found: '{obj_name}'", code="NOT_FOUND")

    mod_type = str(params.get("type") or "").strip().upper()
    if not mod_type:
        raise OpError("modifier.configure requires 'type'", code="MISSING_PARAM")

    mod_name = str(params.get("name") or mod_type.capitalize()).strip()
    mod = obj.modifiers.get(mod_name)
    if mod is not None and mod.type != mod_type:
        obj.modifiers.remove(mod)
        mod = None
    if mod is None:
        try:
            mod = obj.modifiers.new(name=mod_name, type=mod_type)
        except TypeError as err:
            raise OpError(f"Unsupported Blender modifier type '{mod_type}': {err}", code="INVALID_PARAM") from err

    # Merge top-level convenience keys with explicit `settings` dict
    settings: dict[str, Any] = {}
    for k, v in params.items():
        if k not in ("object", "type", "name", "settings"):
            settings[k] = v
    if isinstance(params.get("settings"), dict):
        settings.update(params["settings"])

    applied_keys: list[str] = []
    for k, v in settings.items():
        # Friendly aliases & object resolution
        if k == "angle_deg" and hasattr(mod, "angle"):
            mod.angle = math.radians(float(v))
            applied_keys.append("angle")
            continue
        if k == "angle_limit_deg" and hasattr(mod, "angle_limit"):
            mod.angle_limit = math.radians(float(v))
            if hasattr(mod, "limit_method"):
                mod.limit_method = "ANGLE"
            applied_keys.append("angle_limit")
            continue
        if k in ("mirror_object", "target", "offset_object") or (k == "object" and mod_type == "BOOLEAN"):
            ref_obj = bpy.data.objects.get(str(v)) if v else None
            if v and ref_obj is None:
                raise OpError(f"Referenced modifier target object not found: '{v}'", code="NOT_FOUND")
            setattr(mod, k, ref_obj)
            applied_keys.append(k)
            continue
        if k == "target_object" and mod_type == "BOOLEAN":
            ref_obj = bpy.data.objects.get(str(v))
            if ref_obj is None:
                raise OpError(f"Boolean target object not found: '{v}'", code="NOT_FOUND")
            mod.object = ref_obj
            applied_keys.append("object")
            continue
        if hasattr(mod, k):
            cur = getattr(mod, k)
            if hasattr(cur, "__len__") and not isinstance(cur, str) and isinstance(v, (list, tuple)):
                for idx, item in enumerate(v):
                    cur[idx] = item
            else:
                setattr(mod, k, v)
            applied_keys.append(k)
        else:
            raise OpError(
                f"Modifier '{mod.name}' ({mod_type}) has no attribute '{k}'",
                code="INVALID_PARAM",
            )

    ctx.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    try:
        ev_verts = len(eval_mesh.vertices)
        ev_polys = len(eval_mesh.polygons)
    finally:
        eval_obj.to_mesh_clear()

    return {
        "object": obj.name,
        "modifier": mod.name,
        "type": mod.type,
        "applied_settings": applied_keys,
        "evaluated_vertices": ev_verts,
        "evaluated_polygons": ev_polys,
    }


@blender_op(
    "modifier.apply",
    tier="mutate",
    snapshot_blend=True,
    summary="Bake/apply one modifier (or all modifiers) into a Mesh object's real geometry.",
    params_doc={
        "object": "Target Mesh object name",
        "modifier": "Modifier name to apply (or omit when `all: true`)",
        "all": "If true, bake the entire modifier stack into the mesh",
    },
)
def op_modifier_apply(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    obj_name = str(params.get("object") or "").strip()
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "MESH" or obj.data is None:
        raise OpError(f"Mesh object not found: '{obj_name}'", code="NOT_FOUND")

    apply_all = bool(params.get("all", False))
    mod_name = str(params.get("modifier") or "").strip()
    if not apply_all and not mod_name:
        raise OpError("modifier.apply requires 'modifier' name or `all: true`", code="MISSING_PARAM")

    if not apply_all:
        target_mod = obj.modifiers.get(mod_name)
        if target_mod is None:
            raise OpError(f"Modifier '{mod_name}' not found on '{obj.name}'", code="NOT_FOUND")
        # Temporarily mute modifiers that should NOT be baked yet
        saved_vis: list[tuple[Any, bool]] = []
        for m in obj.modifiers:
            if m != target_mod:
                saved_vis.append((m, bool(m.show_viewport)))
                m.show_viewport = False
        try:
            ctx.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            baked_mesh = bpy.data.meshes.new_from_object(eval_obj, preserve_all_data_layers=True, depsgraph=depsgraph)
        finally:
            for m, vis in saved_vis:
                m.show_viewport = vis
        old_mesh = obj.data
        baked_mesh.name = f"{obj.name}_Mesh"
        obj.data = baked_mesh
        obj.modifiers.remove(target_mod)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
        applied = [mod_name]
    else:
        applied = [m.name for m in obj.modifiers]
        ctx.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        baked_mesh = bpy.data.meshes.new_from_object(eval_obj, preserve_all_data_layers=True, depsgraph=depsgraph)
        old_mesh = obj.data
        baked_mesh.name = f"{obj.name}_Mesh"
        obj.data = baked_mesh
        obj.modifiers.clear()
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)

    ctx.view_layer.update()
    return {
        "object": obj.name,
        "applied_modifiers": applied,
        "remaining_modifiers": [m.name for m in obj.modifiers],
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
    }