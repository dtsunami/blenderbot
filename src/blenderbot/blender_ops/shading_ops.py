"""Materials (Principled BSDF + PBR Textures), Lights, Studio Light Rigs, World & Cameras (`shading_ops.py`)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .helpers import (
    ensure_channelbag_fcurves,
    ensure_collection,
    link_exclusively,
    world_aabb,
)
from .schema import OpContext, OpError, blender_op


def _set_bsdf_input(bsdf: Any, candidates: tuple[str, ...], value: Any) -> str | None:
    for sock_name in candidates:
        if sock_name in bsdf.inputs:
            sock = bsdf.inputs[sock_name]
            if hasattr(sock.default_value, "__len__") and isinstance(value, (list, tuple)):
                if len(sock.default_value) == 4 and len(value) == 3:
                    sock.default_value = (float(value[0]), float(value[1]), float(value[2]), 1.0)
                else:
                    for i in range(min(len(sock.default_value), len(value))):
                        sock.default_value[i] = float(value[i])
            else:
                sock.default_value = float(value)
            return sock_name
    return None


def _attach_image_texture(
    bpy: Any,
    nt: Any,
    bsdf: Any,
    filepath: str,
    target_sockets: tuple[str, ...],
    *,
    non_color: bool = False,
    is_normal_map: bool = False,
    normal_strength: float = 1.0,
    loc_y: float = 0.0,
) -> None:
    img_path = Path(filepath).resolve()
    if not img_path.is_file():
        raise OpError(f"Texture image file not found: {img_path}", code="NOT_FOUND")
    img = bpy.data.images.load(str(img_path), check_existing=True)
    if non_color and hasattr(img, "colorspace_settings"):
        img.colorspace_settings.name = "Non-Color"

    tex_node = nt.nodes.new(type="ShaderNodeTexImage")
    tex_node.image = img
    tex_node.location = (-520.0, loc_y)

    dest_sock = next((bsdf.inputs[s] for s in target_sockets if s in bsdf.inputs), None)
    if dest_sock is None:
        return

    if is_normal_map:
        norm_node = nt.nodes.new(type="ShaderNodeNormalMap")
        norm_node.location = (-240.0, loc_y)
        norm_node.inputs["Strength"].default_value = float(normal_strength)
        nt.links.new(tex_node.outputs["Color"], norm_node.inputs["Color"])
        nt.links.new(norm_node.outputs["Normal"], dest_sock)
    else:
        nt.links.new(tex_node.outputs["Color"], dest_sock)


@blender_op(
    "material.upsert",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Create or update a Principled BSDF PBR material (base color, metallic, roughness, IOR, transmission, "
        "emission, coat, alpha, image textures) and optionally assign it to one or more objects."
    ),
    params_doc={
        "name": "Material name (e.g. 'Mat_BrushedAluminum', 'Mat_Glass')",
        "base_color": "`[r, g, b]` or `[r, g, b, a]` linear RGBA (0.0..1.0)",
        "metallic": "Metallic weight 0.0..1.0 (default 0.0)",
        "roughness": "Surface roughness 0.0..1.0 (default 0.4)",
        "ior": "Index of refraction (default 1.45; e.g. 1.52 glass, 1.333 water)",
        "alpha": "Opacity 0.0..1.0 (default 1.0)",
        "transmission": "Refractive transmission weight 0.0..1.0 for glass/liquid",
        "coat": "Clearcoat weight 0.0..1.0 (e.g. car paint / lacquer)",
        "coat_roughness": "Clearcoat roughness 0.0..1.0",
        "subsurface": "Subsurface scattering weight 0.0..1.0",
        "emission_color": "`[r, g, b]` emission color",
        "emission_strength": "Emission strength (e.g. 5.0 for neon/LEDs)",
        "base_color_texture": "Optional path to albedo/diffuse image file",
        "roughness_texture": "Optional path to non-color roughness map image file",
        "metallic_texture": "Optional path to non-color metallic map image file",
        "normal_texture": "Optional path to tangent-space normal map image file",
        "normal_strength": "Normal map strength (default 1.0)",
        "assign_to": "Object name or list of object names to assign this material to",
        "slot": "Material slot index when assigning (default 0; use -1 to append a new slot)",
    },
)
def op_material_upsert(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    mat_name = str(params.get("name") or "").strip()
    if not mat_name:
        raise OpError("material.upsert requires 'name'", code="MISSING_PARAM")

    mat = bpy.data.materials.get(mat_name)
    created = False
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        created = True

    mat.use_nodes = True
    nt = mat.node_tree
    assert nt is not None

    out_node = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out_node is None:
        out_node = nt.nodes.new(type="ShaderNodeOutputMaterial")
        out_node.location = (300.0, 0.0)

    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nt.nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0.0, 0.0)
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    if params.get("base_color") is not None:
        bc = [float(x) for x in params["base_color"]]
        rgba = (bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 1.0)
        _set_bsdf_input(bsdf, ("Base Color",), rgba)
        mat.diffuse_color = rgba

    if params.get("metallic") is not None:
        m_val = float(params["metallic"])
        _set_bsdf_input(bsdf, ("Metallic",), m_val)
        mat.metallic = m_val

    if params.get("roughness") is not None:
        r_val = float(params["roughness"])
        _set_bsdf_input(bsdf, ("Roughness",), r_val)
        mat.roughness = r_val

    if params.get("ior") is not None:
        _set_bsdf_input(bsdf, ("IOR",), float(params["ior"]))

    if params.get("alpha") is not None:
        a_val = float(params["alpha"])
        _set_bsdf_input(bsdf, ("Alpha",), a_val)
        if a_val < 0.999 and hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"

    if params.get("transmission") is not None:
        t_val = float(params["transmission"])
        _set_bsdf_input(bsdf, ("Transmission Weight", "Transmission"), t_val)
        if t_val > 0.01 and hasattr(mat, "use_raytrace_refraction"):
            mat.use_raytrace_refraction = True

    if params.get("coat") is not None:
        _set_bsdf_input(bsdf, ("Coat Weight", "Clearcoat"), float(params["coat"]))
    if params.get("coat_roughness") is not None:
        _set_bsdf_input(bsdf, ("Coat Roughness", "Clearcoat Roughness"), float(params["coat_roughness"]))
    if params.get("subsurface") is not None:
        _set_bsdf_input(bsdf, ("Subsurface Weight", "Subsurface"), float(params["subsurface"]))

    if params.get("emission_color") is not None:
        ec = [float(x) for x in params["emission_color"]]
        _set_bsdf_input(bsdf, ("Emission Color", "Emission"), (ec[0], ec[1], ec[2], 1.0))
        if params.get("emission_strength") is None:
            _set_bsdf_input(bsdf, ("Emission Strength",), 1.0)
    if params.get("emission_strength") is not None:
        _set_bsdf_input(bsdf, ("Emission Strength",), float(params["emission_strength"]))

    # Optional image textures
    if params.get("base_color_texture"):
        _attach_image_texture(bpy, nt, bsdf, str(params["base_color_texture"]), ("Base Color",), loc_y=260.0)
    if params.get("metallic_texture"):
        _attach_image_texture(
            bpy, nt, bsdf, str(params["metallic_texture"]), ("Metallic",), non_color=True, loc_y=40.0
        )
    if params.get("roughness_texture"):
        _attach_image_texture(
            bpy, nt, bsdf, str(params["roughness_texture"]), ("Roughness",), non_color=True, loc_y=-180.0
        )
    if params.get("normal_texture"):
        _attach_image_texture(
            bpy,
            nt,
            bsdf,
            str(params["normal_texture"]),
            ("Normal",),
            non_color=True,
            is_normal_map=True,
            normal_strength=float(params.get("normal_strength", 1.0)),
            loc_y=-420.0,
        )

    assigned_objs: list[str] = []
    raw_targets = params.get("assign_to")
    target_names: list[str] = []
    if isinstance(raw_targets, str) and raw_targets.strip():
        target_names = [raw_targets.strip()]
    elif isinstance(raw_targets, list):
        target_names = [str(t).strip() for t in raw_targets if str(t).strip()]

    slot_idx = int(params.get("slot", 0))
    for tn in target_names:
        obj = bpy.data.objects.get(tn)
        if obj is None:
            raise OpError(f"Object not found for material assignment: '{tn}'", code="NOT_FOUND")
        data = getattr(obj, "data", None)
        if data is None or not hasattr(data, "materials"):
            raise OpError(f"Object '{tn}' ({obj.type}) does not support materials", code="INVALID_TARGET")
        if slot_idx < 0 or len(data.materials) == 0:
            data.materials.append(mat)
        else:
            while len(data.materials) <= slot_idx:
                data.materials.append(mat)
            data.materials[slot_idx] = mat
        obj.color = tuple(mat.diffuse_color)
        assigned_objs.append(obj.name)

    return {
        "name": mat.name,
        "created": created,
        "assigned_to": assigned_objs,
        "users": int(mat.users),
    }


@blender_op(
    "material.list",
    tier="read",
    summary="List all materials in `bpy.data.materials` with Principled BSDF properties and assigned objects.",
)
def op_material_list(ctx: OpContext, _params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    usage: dict[str, list[str]] = {}
    for obj in ctx.scene.objects:
        data = getattr(obj, "data", None)
        if data is not None and hasattr(data, "materials"):
            for m in data.materials:
                if m is not None:
                    usage.setdefault(m.name, []).append(obj.name)

    mats_out: list[dict[str, Any]] = []
    for mat in bpy.data.materials:
        entry: dict[str, Any] = {
            "name": mat.name,
            "users": int(mat.users),
            "use_nodes": bool(mat.use_nodes),
            "assigned_objects": usage.get(mat.name, []),
        }
        if mat.use_nodes and mat.node_tree:
            bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None:
                if "Base Color" in bsdf.inputs:
                    entry["base_color"] = [round(float(c), 4) for c in bsdf.inputs["Base Color"].default_value]
                if "Metallic" in bsdf.inputs:
                    entry["metallic"] = round(float(bsdf.inputs["Metallic"].default_value), 4)
                if "Roughness" in bsdf.inputs:
                    entry["roughness"] = round(float(bsdf.inputs["Roughness"].default_value), 4)
        mats_out.append(entry)

    return {"materials": mats_out, "count": len(mats_out)}


def _aim_object_negative_z_at(obj: Any, target_xyz: tuple[float, float, float]) -> None:
    """Orient `obj` (Camera or Light) so its local `-Z` axis points directly at `target_xyz` with `+Y` up."""
    from mathutils import Vector

    origin = Vector(obj.location)
    aim = Vector(target_xyz)
    direction = aim - origin
    if direction.length < 1e-6:
        return
    quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = quat.to_euler("XYZ")


def _resolve_target_point(bpy: Any, depsgraph: Any, spec: Any) -> tuple[float, float, float]:
    if isinstance(spec, str):
        t_obj = bpy.data.objects.get(spec.strip())
        if t_obj is None:
            raise OpError(f"Target object not found: '{spec}'", code="NOT_FOUND")
        aabb = world_aabb(t_obj, depsgraph)
        if aabb:
            return (float(aabb["center"][0]), float(aabb["center"][1]), float(aabb["center"][2]))
        loc = t_obj.matrix_world.translation
        return (float(loc.x), float(loc.y), float(loc.z))
    if isinstance(spec, (list, tuple)) and len(spec) == 3:
        return (float(spec[0]), float(spec[1]), float(spec[2]))
    raise OpError("'look_at' must be `[x, y, z]` or an existing object name", code="INVALID_PARAM")


@blender_op(
    "light.upsert",
    tier="mutate",
    snapshot_blend=True,
    summary="Create or update a Blender Light (`AREA`, `SUN`, `POINT`, `SPOT`) with energy, color, size & aim target.",
    params_doc={
        "name": "Light object name (e.g. 'KeyLight', 'SunLight')",
        "light_type": "'AREA' (default), 'SUN', 'POINT', or 'SPOT'",
        "energy": "Power in Watts (AREA/POINT/SPOT, e.g. 250.0) or W/m^2 (SUN, e.g. 3.5)",
        "color": "`[r, g, b]` light color (default `[1.0, 1.0, 1.0]`)",
        "location": "`[x, y, z]` world location",
        "rotation_deg": "`[rx, ry, rz]` Euler degrees (optional if `look_at` is passed)",
        "look_at": "Optional `[x, y, z]` coordinate or object name to aim the light at",
        "size": "Area light width or Point/Spot shadow soft radius in meters (default 1.5 for AREA)",
        "size_y": "Optional Area light height when `shape` is 'RECTANGLE' or 'ELLIPSE'",
        "shape": "Area light shape: 'SQUARE', 'RECTANGLE', 'DISK', 'ELLIPSE'",
        "spot_size_deg": "Spot cone angle in degrees (default 45.0)",
        "spot_blend": "Spot cone edge softness 0.0..1.0 (default 0.25)",
        "angle_deg": "Sun angular diameter in degrees for soft sun shadows (default 1.2)",
        "use_shadow": "Enable shadow casting (default true)",
        "collection": "Collection name (default 'LIGHTS')",
    },
)
def op_light_upsert(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    name = str(params.get("name") or "Light").strip()
    ltype = str(params.get("light_type") or params.get("type") or "AREA").strip().upper()
    if ltype not in ("AREA", "SUN", "POINT", "SPOT"):
        raise OpError(f"Unsupported light_type '{ltype}'. Expected AREA, SUN, POINT, or SPOT.", code="INVALID_PARAM")

    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type != "LIGHT":
        bpy.data.objects.remove(obj, do_unlink=True)
        obj = None

    if obj is None:
        ldata = bpy.data.lights.new(name=f"{name}_Data", type=ltype)
        obj = bpy.data.objects.new(name, ldata)
    else:
        ldata = obj.data
        ldata.type = ltype

    col = ensure_collection(str(params.get("collection") or "LIGHTS"))
    link_exclusively(obj, col)

    default_energy = 3.5 if ltype == "SUN" else 200.0
    ldata.energy = float(params.get("energy", default_energy))
    if params.get("color") is not None:
        c = params["color"]
        ldata.color = (float(c[0]), float(c[1]), float(c[2]))
    ldata.use_shadow = bool(params.get("use_shadow", True))

    if ltype == "AREA":
        if params.get("shape"):
            ldata.shape = str(params["shape"]).upper()
        ldata.size = float(params.get("size", 1.5))
        if params.get("size_y") is not None and ldata.shape in ("RECTANGLE", "ELLIPSE"):
            ldata.size_y = float(params["size_y"])
    elif ltype in ("POINT", "SPOT"):
        if params.get("size") is not None:
            ldata.shadow_soft_size = float(params["size"])
        if ltype == "SPOT":
            ldata.spot_size = math.radians(float(params.get("spot_size_deg", 45.0)))
            ldata.spot_blend = float(params.get("spot_blend", 0.25))
    elif ltype == "SUN" and params.get("angle_deg") is not None:
        ldata.angle = math.radians(float(params["angle_deg"]))

    if params.get("location") is not None:
        loc = params["location"]
        obj.location = (float(loc[0]), float(loc[1]), float(loc[2]))

    aim_spec = params.get("look_at") if params.get("look_at") is not None else params.get("target")
    if params.get("rotation_deg") is not None:
        rot = params["rotation_deg"]
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = tuple(math.radians(float(a)) for a in rot[:3])
    elif aim_spec is not None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        aim_pt = _resolve_target_point(bpy, depsgraph, aim_spec)
        _aim_object_negative_z_at(obj, aim_pt)

    ctx.view_layer.update()
    return {
        "name": obj.name,
        "light_type": ldata.type,
        "energy": round(float(ldata.energy), 3),
        "location": [round(float(v), 4) for v in obj.location],
        "rotation_deg": [round(math.degrees(float(v)), 2) for v in obj.rotation_euler],
    }


@blender_op(
    "light.rig",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Build a complete proportional lighting setup (`three_point`, `studio_softbox`, `outdoor_sun`, "
        "`dramatic_rim`) automatically scaled to `target` object AABB or explicit `center` + `radius`."
    ),
    params_doc={
        "preset": "'three_point' (default), 'studio_softbox', 'outdoor_sun', or 'dramatic_rim'",
        "target": "Optional hero object name to center and scale the light rig around",
        "center": "Optional `[x, y, z]` focus point (default `[0, 0, 0.8]` if no `target`)",
        "radius": "Optional subject scale radius in meters (auto-derived from `target` AABB when omitted)",
        "intensity": "Global energy multiplier (default 1.0)",
        "collection": "Collection name (default 'LIGHTS')",
        "clear_existing": "Remove prior lights in `collection` before building (default true)",
    },
)
def op_light_rig(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    preset = str(params.get("preset") or "three_point").strip().lower()
    if preset not in ("three_point", "studio_softbox", "outdoor_sun", "dramatic_rim"):
        raise OpError(
            f"Unsupported light.rig preset '{preset}'. Use three_point, studio_softbox, outdoor_sun, or dramatic_rim.",
            code="INVALID_PARAM",
        )

    col_name = str(params.get("collection") or "LIGHTS")
    col = ensure_collection(col_name)
    if bool(params.get("clear_existing", True)):
        for o in list(col.objects):
            if o.type == "LIGHT":
                ld = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                if ld and ld.users == 0:
                    bpy.data.lights.remove(ld)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    center = [0.0, 0.0, 0.8]
    subj_r = 1.5
    raw_target = params.get("target")
    if isinstance(raw_target, (list, tuple)) and len(raw_target) >= 3:
        center = [float(x) for x in raw_target[:3]]
    elif raw_target:
        t_obj = bpy.data.objects.get(str(raw_target).strip())
        if t_obj is None:
            raise OpError(f"Target object not found: '{raw_target}'", code="NOT_FOUND")
        aabb = world_aabb(t_obj, depsgraph)
        if aabb:
            center = list(aabb["center"])
            subj_r = max(0.5, max(aabb["size"]) * 0.65)
    if params.get("center") is not None:
        center = [float(x) for x in params["center"][:3]]
    if params.get("radius") is not None:
        subj_r = max(0.25, float(params["radius"]))

    mult = max(0.05, float(params.get("intensity", 1.0)))
    dist = max(0.5, float(params["distance"])) if params.get("distance") is not None else subj_r * 3.0
    # Area light power scales quadratically with distance to preserve exposure regardless of model size!
    base_watts = 35.0 * (dist * dist) * mult
    cx, cy, cz = center

    specs: list[dict[str, Any]] = []
    if preset == "three_point":
        specs = [
            {
                "name": "LGT_Key",
                "light_type": "AREA",
                "energy": base_watts * 1.0,
                "color": [1.0, 0.96, 0.92],
                "size": subj_r * 1.4,
                "location": [cx - dist * 0.85, cy - dist * 0.95, cz + dist * 0.85],
            },
            {
                "name": "LGT_Fill",
                "light_type": "AREA",
                "energy": base_watts * 0.38,
                "color": [0.86, 0.92, 1.0],
                "size": subj_r * 2.2,
                "location": [cx + dist * 1.05, cy - dist * 0.75, cz + dist * 0.45],
            },
            {
                "name": "LGT_Rim",
                "light_type": "AREA",
                "energy": base_watts * 0.85,
                "color": [0.95, 0.98, 1.0],
                "size": subj_r * 1.1,
                "location": [cx + dist * 0.35, cy + dist * 1.05, cz + dist * 0.9],
            },
        ]
    elif preset == "studio_softbox":
        specs = [
            {
                "name": "LGT_TopSoftbox",
                "light_type": "AREA",
                "shape": "RECTANGLE",
                "size": subj_r * 2.6,
                "size_y": subj_r * 1.8,
                "energy": base_watts * 0.95,
                "color": [1.0, 0.99, 0.97],
                "location": [cx, cy - dist * 0.15, cz + dist * 1.25],
            },
            {
                "name": "LGT_LeftStrip",
                "light_type": "AREA",
                "shape": "RECTANGLE",
                "size": subj_r * 0.8,
                "size_y": subj_r * 2.4,
                "energy": base_watts * 0.65,
                "color": [1.0, 0.97, 0.94],
                "location": [cx - dist * 1.1, cy - dist * 0.55, cz + dist * 0.35],
            },
            {
                "name": "LGT_RightStrip",
                "light_type": "AREA",
                "shape": "RECTANGLE",
                "size": subj_r * 0.8,
                "size_y": subj_r * 2.4,
                "energy": base_watts * 0.55,
                "color": [0.9, 0.95, 1.0],
                "location": [cx + dist * 1.1, cy - dist * 0.55, cz + dist * 0.35],
            },
        ]
    elif preset == "outdoor_sun":
        specs = [
            {
                "name": "LGT_Sun",
                "light_type": "SUN",
                "energy": 4.2 * mult,
                "color": [1.0, 0.95, 0.86],
                "angle_deg": 1.1,
                "location": [cx - dist, cy - dist * 0.7, cz + dist * 1.4],
            },
            {
                "name": "LGT_SkyFill",
                "light_type": "AREA",
                "size": subj_r * 4.0,
                "energy": base_watts * 0.25,
                "color": [0.68, 0.82, 1.0],
                "location": [cx + dist * 0.6, cy + dist * 0.4, cz + dist * 1.5],
            },
        ]
    elif preset == "dramatic_rim":
        specs = [
            {
                "name": "LGT_BackRimL",
                "light_type": "SPOT",
                "spot_size_deg": 50.0,
                "spot_blend": 0.4,
                "size": subj_r * 0.4,
                "energy": base_watts * 1.35,
                "color": [0.35, 0.68, 1.0],
                "location": [cx - dist * 0.9, cy + dist * 0.95, cz + dist * 0.7],
            },
            {
                "name": "LGT_BackRimR",
                "light_type": "SPOT",
                "spot_size_deg": 50.0,
                "spot_blend": 0.4,
                "size": subj_r * 0.4,
                "energy": base_watts * 1.25,
                "color": [1.0, 0.55, 0.28],
                "location": [cx + dist * 0.9, cy + dist * 0.95, cz + dist * 0.7],
            },
            {
                "name": "LGT_SoftFrontKey",
                "light_type": "AREA",
                "size": subj_r * 1.8,
                "energy": base_watts * 0.32,
                "color": [0.95, 0.96, 1.0],
                "location": [cx - dist * 0.3, cy - dist * 1.1, cz + dist * 0.4],
            },
        ]

    created_lights: list[dict[str, Any]] = []
    for sp in specs:
        sp["look_at"] = center
        sp["collection"] = col_name
        created_lights.append(op_light_upsert(ctx, sp))

    return {
        "preset": preset,
        "collection": col_name,
        "center": [round(v, 3) for v in center],
        "subject_radius_m": round(subj_r, 3),
        "lights": created_lights,
    }


@blender_op(
    "world.setup",
    tier="mutate",
    snapshot_blend=True,
    summary="Configure Blender World environment lighting (`solid`, procedural `sky`, `gradient`, or `hdri` image).",
    params_doc={
        "mode": "'solid' (default), 'sky' (physical sun/sky model), 'gradient', or 'hdri'",
        "color": "`[r, g, b]` background color for 'solid' or horizon color for 'gradient'",
        "zenith_color": "`[r, g, b]` zenith sky color for 'gradient'",
        "strength": "Environment strength multiplier (default 1.0)",
        "hdri_path": "Path to `.hdr` / `.exr` image when `mode == 'hdri'`",
        "rotation_z_deg": "Environment Z rotation in degrees for 'hdri' or 'sky' (default 0.0)",
        "sun_elevation_deg": "Sun elevation in degrees for `mode == 'sky'` (default 35.0)",
    },
)
def op_world_setup(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    world = sc.world
    world.use_nodes = True
    nt = world.node_tree
    assert nt is not None
    nt.nodes.clear()

    mode = str(params.get("mode") or "solid").strip().lower()
    strength = float(params.get("strength", 1.0))
    rot_z = math.radians(float(params.get("rotation_z_deg", 0.0)))

    out_node = nt.nodes.new(type="ShaderNodeOutputWorld")
    out_node.location = (320.0, 0.0)
    bg_node = nt.nodes.new(type="ShaderNodeBackground")
    bg_node.location = (80.0, 0.0)
    bg_node.inputs["Strength"].default_value = strength
    nt.links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    if mode == "solid":
        col = [float(x) for x in (params.get("color") or [0.05, 0.055, 0.065])[:3]]
        bg_node.inputs["Color"].default_value = (col[0], col[1], col[2], 1.0)
        world.color = (col[0], col[1], col[2])
    elif mode == "sky":
        sky = nt.nodes.new(type="ShaderNodeTexSky")
        sky.location = (-220.0, 0.0)
        # Blender 5.x uses SINGLE_SCATTERING / MULTIPLE_SCATTERING (or NISHITA in 4.x)
        for sky_enum in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA", "PREETHAM"):
            try:
                sky.sky_type = sky_enum
                break
            except TypeError:
                continue
        if hasattr(sky, "sun_elevation"):
            sky.sun_elevation = math.radians(float(params.get("sun_elevation_deg", 35.0)))
        if hasattr(sky, "sun_rotation"):
            sky.sun_rotation = rot_z
        nt.links.new(sky.outputs["Color"], bg_node.inputs["Color"])
    elif mode == "gradient":
        horizon = [float(x) for x in (params.get("color") or [0.18, 0.22, 0.28])[:3]]
        zenith = [float(x) for x in (params.get("zenith_color") or [0.03, 0.05, 0.10])[:3]]
        tex_coord = nt.nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-660.0, 0.0)
        sep = nt.nodes.new(type="ShaderNodeSeparateXYZ")
        sep.location = (-450.0, 0.0)
        ramp = nt.nodes.new(type="ShaderNodeValToRGB")
        ramp.location = (-220.0, 0.0)
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (horizon[0], horizon[1], horizon[2], 1.0)
        ramp.color_ramp.elements[1].position = 0.65
        ramp.color_ramp.elements[1].color = (zenith[0], zenith[1], zenith[2], 1.0)
        nt.links.new(tex_coord.outputs["Generated"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bg_node.inputs["Color"])
    elif mode == "hdri":
        raw_hdri = str(params.get("hdri_path") or "").strip()
        if not raw_hdri:
            raise OpError("world.setup with mode='hdri' requires 'hdri_path'", code="MISSING_PARAM")
        hp = Path(raw_hdri).resolve()
        if not hp.is_file():
            raise OpError(f"HDRI file not found: {hp}", code="NOT_FOUND")
        tex_coord = nt.nodes.new(type="ShaderNodeTexCoord")
        tex_coord.location = (-680.0, 0.0)
        mapping = nt.nodes.new(type="ShaderNodeMapping")
        mapping.location = (-470.0, 0.0)
        mapping.inputs["Rotation"].default_value[2] = rot_z
        env_tex = nt.nodes.new(type="ShaderNodeTexEnvironment")
        env_tex.location = (-220.0, 0.0)
        env_tex.image = bpy.data.images.load(str(hp), check_existing=True)
        nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
        nt.links.new(env_tex.outputs["Color"], bg_node.inputs["Color"])
    else:
        raise OpError(f"Unsupported world.setup mode '{mode}'. Use solid, sky, gradient, or hdri.", code="INVALID_PARAM")

    sc.display.shading.background_type = "WORLD"
    return {"world": world.name, "mode": mode, "strength": strength}


@blender_op(
    "camera.upsert",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Create or update a Camera (focal length, sensor, DOF, stationary `look_at`, live `track_target` "
        "constraint, or animated 360-degree `turntable` orbit) and optionally make it active."
    ),
    params_doc={
        "name": "Camera object name (default 'Camera')",
        "location": "`[x, y, z]` world location",
        "rotation_deg": "`[rx, ry, rz]` Euler degrees (optional if `look_at` or `track_target` is set)",
        "look_at": "One-shot `[x, y, z]` point or object name to aim the camera at (alias: `target`)",
        "track_target": "Object name to keep aiming at dynamically via a `TRACK_TO` constraint",
        "lens_mm": "Focal length in mm (default 50.0; e.g. 24 wide, 50 standard, 85 portrait)",
        "sensor_width_mm": "Camera sensor width in mm (default 36.0 full-frame)",
        "projection": "'PERSP' (default) or 'ORTHO'",
        "ortho_scale": "Orthographic width in meters when `projection == 'ORTHO'`",
        "clip_start": "Near clip plane in meters (default 0.1)",
        "clip_end": "Far clip plane in meters (default 1000.0)",
        "dof_fstop": "Optional aperture f-stop (enables Depth of Field when set, e.g. 2.8)",
        "dof_focus_object": "Optional object name for autofocus distance",
        "dof_focus_distance": "Manual focus distance in meters",
        "turntable": (
            "Optional dict `{'target': 'HeroObj', 'radius': 4.0, 'height': 1.8, 'start_angle_deg': -90, "
            "'revolutions': 1.0, 'frame_start': 1, 'frame_end': 120}` to animate a smooth orbit"
        ),
        "collection": "Collection name (default 'CAMERA')",
        "set_active": "Make this the active scene camera (default true)",
    },
)
def op_camera_upsert(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    name = str(params.get("name") or "Camera").strip()

    cam_obj = bpy.data.objects.get(name)
    if cam_obj is not None and cam_obj.type != "CAMERA":
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        cam_obj = None

    if cam_obj is None:
        cam_data = bpy.data.cameras.new(name=f"{name}_Data")
        cam_obj = bpy.data.objects.new(name, cam_data)
    else:
        cam_data = cam_obj.data

    col = ensure_collection(str(params.get("collection") or "CAMERA"))
    link_exclusively(cam_obj, col)

    if params.get("projection"):
        cam_data.type = str(params["projection"]).upper()
    if params.get("ortho_scale") is not None:
        cam_data.ortho_scale = float(params["ortho_scale"])
    if params.get("lens_mm") is not None:
        cam_data.lens = float(params["lens_mm"])
    if params.get("sensor_width_mm") is not None:
        cam_data.sensor_width = float(params["sensor_width_mm"])
    if params.get("clip_start") is not None:
        cam_data.clip_start = float(params["clip_start"])
    if params.get("clip_end") is not None:
        cam_data.clip_end = float(params["clip_end"])

    # Depth of Field
    if any(k in params for k in ("dof_fstop", "dof_focus_object", "dof_focus_distance")):
        cam_data.dof.use_dof = True
        if params.get("dof_fstop") is not None:
            cam_data.dof.aperture_fstop = float(params["dof_fstop"])
        if params.get("dof_focus_object"):
            fo = bpy.data.objects.get(str(params["dof_focus_object"]).strip())
            if fo is None:
                raise OpError(f"DOF focus object not found: '{params['dof_focus_object']}'", code="NOT_FOUND")
            cam_data.dof.focus_object = fo
        if params.get("dof_focus_distance") is not None:
            cam_data.dof.focus_distance = float(params["dof_focus_distance"])

    if params.get("location") is not None:
        loc = params["location"]
        cam_obj.location = (float(loc[0]), float(loc[1]), float(loc[2]))

    cam_aim = params.get("look_at") if params.get("look_at") is not None else params.get("target")
    if params.get("rotation_deg") is not None:
        rot = params["rotation_deg"]
        cam_obj.rotation_mode = "XYZ"
        cam_obj.rotation_euler = tuple(math.radians(float(a)) for a in rot[:3])
    elif cam_aim is not None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        aim_pt = _resolve_target_point(bpy, depsgraph, cam_aim)
        _aim_object_negative_z_at(cam_obj, aim_pt)

    if params.get("track_target"):
        t_obj = bpy.data.objects.get(str(params["track_target"]).strip())
        if t_obj is None:
            raise OpError(f"track_target object not found: '{params['track_target']}'", code="NOT_FOUND")
        for c in list(cam_obj.constraints):
            if c.type == "TRACK_TO":
                cam_obj.constraints.remove(c)
        tr = cam_obj.constraints.new(type="TRACK_TO")
        tr.target = t_obj
        tr.track_axis = "TRACK_NEGATIVE_Z"
        tr.up_axis = "UP_Y"

    turntable_baked = 0
    if isinstance(params.get("turntable"), dict):
        tt = params["turntable"]
        depsgraph = bpy.context.evaluated_depsgraph_get()
        center = [0.0, 0.0, 0.5]
        auto_r = 4.0
        if tt.get("target"):
            t_obj = bpy.data.objects.get(str(tt["target"]).strip())
            if t_obj is None:
                raise OpError(f"Turntable target object not found: '{tt['target']}'", code="NOT_FOUND")
            aabb = world_aabb(t_obj, depsgraph)
            if aabb:
                center = list(aabb["center"])
                auto_r = max(1.5, max(aabb["size"]) * 2.1)
        if tt.get("center") is not None:
            center = [float(x) for x in tt["center"][:3]]

        radius = float(tt.get("radius", auto_r))
        height = float(tt.get("height", center[2] + radius * 0.32))
        start_deg = float(tt.get("start_angle_deg", -90.0))
        revs = float(tt.get("revolutions", 1.0))
        f_start = int(tt.get("frame_start", sc.frame_start))
        f_end = int(tt.get("frame_end", sc.frame_end))
        n_frames = max(2, f_end - f_start + 1)

        # Create/reuse pivot Empty at `center` and animate Z rotation linearly for perfectly circular orbit
        rig_name = f"{name}_TurntablePivot"
        pivot = bpy.data.objects.get(rig_name)
        if pivot is None:
            pivot = bpy.data.objects.new(rig_name, None)
            pivot.empty_display_size = 0.4
        link_exclusively(pivot, col)
        pivot.location = (center[0], center[1], center[2])
        pivot.rotation_mode = "XYZ"
        pivot.rotation_euler = (0.0, 0.0, math.radians(start_deg))

        cam_obj.parent = pivot
        cam_obj.matrix_parent_inverse.identity()
        cam_obj.location = (radius, 0.0, height - center[2])
        _aim_object_negative_z_at(cam_obj, (0.0, 0.0, 0.0))

        fcs = ensure_channelbag_fcurves(pivot, action_name=f"{rig_name}_Action")
        for fc in list(fcs):
            if fc.data_path == "rotation_euler":
                fcs.remove(fc)
        rot_z_fc = fcs.new(data_path="rotation_euler", index=2)
        rot_z_fc.keyframe_points.add(2)
        rot_z_fc.keyframe_points[0].co = (float(f_start), math.radians(start_deg))
        rot_z_fc.keyframe_points[0].interpolation = "LINEAR"
        rot_z_fc.keyframe_points[1].co = (float(f_end), math.radians(start_deg + 360.0 * revs))
        rot_z_fc.keyframe_points[1].interpolation = "LINEAR"
        rot_z_fc.update()
        turntable_baked = n_frames

    if bool(params.get("set_active", True)):
        sc.camera = cam_obj

    ctx.view_layer.update()
    return {
        "name": cam_obj.name,
        "active": sc.camera == cam_obj,
        "lens_mm": round(float(cam_data.lens), 2),
        "location": [round(float(v), 4) for v in cam_obj.matrix_world.translation],
        "dof_enabled": bool(cam_data.dof.use_dof),
        "turntable_frames": turntable_baked,
    }