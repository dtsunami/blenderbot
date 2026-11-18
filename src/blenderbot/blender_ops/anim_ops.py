"""General Blender 5.3+ Slotted-Action Keyframing, F-Curve Modifiers, Clearing & Baking (`anim_ops.py`)."""

from __future__ import annotations

import math
from typing import Any

from .helpers import ensure_channelbag_fcurves, iter_action_fcurves
from .schema import OpContext, OpError, blender_op

VALID_INTERPOLATIONS = {
    "BEZIER",
    "LINEAR",
    "CONSTANT",
    "SINE",
    "QUAD",
    "CUBIC",
    "QUART",
    "QUINT",
    "EXPO",
    "CIRC",
    "BACK",
    "BOUNCE",
    "ELASTIC",
}


def _resolve_anim_id_and_path(
    bpy: Any,
    obj: Any,
    raw_path: str,
    bone_name: str | None = None,
) -> tuple[Any, str, bool]:
    """Resolve `(id_block, data_path, convert_deg_to_rad)` for friendly or canonical Blender data paths."""
    prop = raw_path.strip()
    is_deg = False

    if prop == "rotation_deg":
        prop = "rotation_euler"
        is_deg = True
    elif prop.endswith(".rotation_deg"):
        prop = prop[: -len(".rotation_deg")] + ".rotation_euler"
        is_deg = True

    if bone_name:
        if obj.type != "ARMATURE" or obj.pose is None or bone_name not in obj.pose.bones:
            raise OpError(f"Pose bone '{bone_name}' not found on '{obj.name}'", code="NOT_FOUND")
        pb = obj.pose.bones[bone_name]
        if prop in ("rotation_euler", "location", "scale", "rotation_quaternion"):
            if prop == "rotation_euler" and pb.rotation_mode == "QUATERNION":
                pb.rotation_mode = "XYZ"
            return obj, f'pose.bones["{bone_name}"].{prop}', is_deg
        return obj, f'pose.bones["{bone_name}"].{prop}', is_deg

    if prop in ("lens", "lens_mm") and obj.type == "CAMERA" and obj.data is not None:
        return obj.data, "lens", False
    if prop == "ortho_scale" and obj.type == "CAMERA" and obj.data is not None:
        return obj.data, "ortho_scale", False
    if prop in ("energy", "color") and obj.type == "LIGHT" and obj.data is not None:
        return obj.data, prop, False
    if prop.startswith("data.") and obj.data is not None:
        return obj.data, prop[5:], is_deg

    if prop == "rotation_euler" and obj.rotation_mode == "QUATERNION":
        obj.rotation_euler = obj.rotation_quaternion.to_euler("XYZ")
        obj.rotation_mode = "XYZ"

    return obj, prop, is_deg


def _infer_channel_count(id_block: Any, data_path: str, first_val: Any) -> int:
    if isinstance(first_val, (list, tuple)):
        return len(first_val)
    try:
        resolved = id_block.path_resolve(data_path)
        if hasattr(resolved, "__len__") and not isinstance(resolved, str):
            return len(resolved)
    except ValueError:
        pass
    return 1


@blender_op(
    "anim.keyframe",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Author keyframes in Blender 5.3's slotted Action Channelbag system across any object, pose bone, "
        "camera `lens`, light `energy`/`color`, or modifier property with full easing/interpolation support."
    ),
    params_doc={
        "object": "Target object name",
        "bone": "Optional pose bone name when animating an Armature bone",
        "property": (
            "Property / data_path to animate: e.g. `'location'`, `'rotation_deg'` (auto-converted to radians!), "
            "`'scale'`, `'lens'`, `'energy'`, `'color'`, or any Blender RNA `data_path`"
        ),
        "index": "Optional single array channel index (e.g. `2` for Z-only location/rotation)",
        "degrees": "If true and `property` ends with `rotation_euler`, convert keyframe values from degrees to radians",
        "action": (
            "Optional shared Multi-ID Action name (e.g. `'Shot_01'`). Multiple objects, cameras, lights, "
            "and armatures given the same `action` share one Blender 5.3 Slotted Action via separate slots!"
        ),
        "interpolation": "'BEZIER' (default), 'LINEAR', 'CONSTANT', 'BACK', 'ELASTIC', 'BOUNCE', 'SINE', 'CUBIC', etc.",
        "keyframes": "List of `{'frame': int, 'value': float | [x,y,z], 'interpolation'?: str}` or `[frame, value]` pairs",
        "tracks": "Optional list of track dicts (`{'object'?, 'bone'?, 'property', 'keyframes', ...}`) to batch multiple channels",
    },
)
def op_anim_keyframe(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    tracks = params.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        tracks = [params]

    summary_tracks: list[dict[str, Any]] = []
    for tr in tracks:
        obj_name = str(tr.get("object") or params.get("object") or "").strip()
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise OpError(f"Object not found: '{obj_name}'", code="NOT_FOUND")

        raw_prop = str(tr.get("property") or tr.get("data_path") or "").strip()
        if not raw_prop:
            raise OpError("anim.keyframe requires 'property' (or 'data_path')", code="MISSING_PARAM")

        kfs = tr.get("keyframes")
        if not isinstance(kfs, list) or not kfs:
            raise OpError(f"anim.keyframe track '{raw_prop}' requires non-empty 'keyframes' list", code="MISSING_PARAM")

        bone_name = str(tr.get("bone") or params.get("bone") or "").strip() or None
        act_name = str(tr.get("action") or params.get("action") or "").strip() or None
        id_block, data_path, auto_deg = _resolve_anim_id_and_path(bpy, obj, raw_prop, bone_name=bone_name)
        use_deg = bool(tr.get("degrees", params.get("degrees", auto_deg)))
        default_interp = str(tr.get("interpolation") or params.get("interpolation") or "BEZIER").upper()
        if default_interp not in VALID_INTERPOLATIONS:
            raise OpError(f"Unsupported interpolation '{default_interp}'", code="INVALID_PARAM")

        # Parse normalized `(frame, val, interp, easing)` tuples
        parsed_keys: list[tuple[float, Any, str, str | None]] = []
        for item in kfs:
            if isinstance(item, dict):
                fr = float(item["frame"])
                val = item["value"]
                interp = str(item.get("interpolation") or default_interp).upper()
                easing = str(item["easing"]).upper() if item.get("easing") else None
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                fr = float(item[0])
                val = item[1]
                interp = str(item[2]).upper() if len(item) >= 3 else default_interp
                easing = None
            else:
                raise OpError(f"Invalid keyframe entry: {item!r}", code="INVALID_PARAM")
            parsed_keys.append((fr, val, interp, easing))

        parsed_keys.sort(key=lambda k: k[0])

        explicit_idx = tr.get("index")
        fcurves = ensure_channelbag_fcurves(id_block, action_name=act_name)

        if explicit_idx is not None:
            indices = [int(explicit_idx)]
        else:
            n_ch = _infer_channel_count(id_block, data_path, parsed_keys[0][1])
            indices = list(range(n_ch))

        touched_fcs: list[Any] = []
        for ch_pos, arr_idx in enumerate(indices):
            fc = next(
                (c for c in fcurves if c.data_path == data_path and (c.array_index == arr_idx or len(indices) == 1 and explicit_idx is None and c.array_index == 0)),
                None,
            )
            if fc is None:
                fc = fcurves.new(data_path=data_path, index=arr_idx if len(indices) > 1 or explicit_idx is not None else 0)

            for fr, raw_val, interp, easing in parsed_keys:
                if isinstance(raw_val, (list, tuple)):
                    scalar = float(raw_val[ch_pos])
                else:
                    scalar = float(raw_val)
                if use_deg and data_path.endswith("rotation_euler"):
                    scalar = math.radians(scalar)

                kp = fc.keyframe_points.insert(fr, scalar, options={"FAST"})
                kp.interpolation = interp
                if interp == "BEZIER":
                    kp.handle_left_type = "AUTO_CLAMPED"
                    kp.handle_right_type = "AUTO_CLAMPED"
                if easing:
                    kp.easing = easing

            fc.update()
            touched_fcs.append(fc)

        summary_tracks.append(
            {
                "object": obj.name,
                "bone": bone_name,
                "data_path": data_path,
                "channels": len(touched_fcs),
                "keyframes": len(parsed_keys),
                "frame_span": [parsed_keys[0][0], parsed_keys[-1][0]],
            }
        )

    # Refresh current frame evaluation so object transforms reflect authored keys immediately
    ctx.scene.frame_set(ctx.scene.frame_current)
    ctx.view_layer.update()
    return {"tracks": summary_tracks, "count": len(summary_tracks)}


@blender_op(
    "anim.fcurve_modifier",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Add or update procedural F-Curve modifiers (`CYCLES` seamless loops, `NOISE` secondary motion/camera "
        "shake, `LIMITS`, `STEPPED`) on an animated object's channels."
    ),
    params_doc={
        "object": "Target object name",
        "property": "Optional property/data_path filter (e.g. 'location', 'rotation_euler'); applies to all F-curves if omitted",
        "index": "Optional channel index filter (0, 1, 2)",
        "type": "'CYCLES', 'NOISE', 'LIMITS', or 'STEPPED'",
        "mode_before": "For CYCLES: 'REPEAT' (default), 'REPEAT_OFFSET', 'MIRROR', 'NONE'",
        "mode_after": "For CYCLES: 'REPEAT' (default), 'REPEAT_OFFSET', 'MIRROR', 'NONE'",
        "scale": "For NOISE: time frequency scale (default 10.0)",
        "strength": "For NOISE: amplitude strength (default 0.2)",
        "phase": "For NOISE: phase offset (auto-staggered per channel when omitted!)",
        "frame_step": "For STEPPED: hold step in frames (e.g. 2 for animate-on-twos stylized look)",
    },
)
def op_anim_fcurve_modifier(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    obj_name = str(params.get("object") or "").strip()
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise OpError(f"Object not found: '{obj_name}'", code="NOT_FOUND")

    mod_type = str(params.get("type") or "CYCLES").strip().upper()
    if mod_type not in ("CYCLES", "NOISE", "LIMITS", "STEPPED"):
        raise OpError(f"Unsupported F-Curve modifier '{mod_type}'. Use CYCLES, NOISE, LIMITS, or STEPPED.", code="INVALID_PARAM")

    prop_filter = str(params.get("property") or params.get("data_path") or "").strip()
    if prop_filter == "rotation_deg":
        prop_filter = "rotation_euler"
    idx_filter = int(params["index"]) if params.get("index") is not None else None

    all_fcs = list(iter_action_fcurves(obj))
    if obj.data is not None:
        all_fcs.extend(iter_action_fcurves(obj.data))
    if not all_fcs:
        raise OpError(f"Object '{obj.name}' has no F-Curves yet (call `anim.keyframe` first)", code="NO_ANIMATION")

    matched = 0
    for ch_i, fc in enumerate(all_fcs):
        if prop_filter and prop_filter not in fc.data_path:
            continue
        if idx_filter is not None and fc.array_index != idx_filter:
            continue

        # Replace existing modifier of same type to stay idempotent
        for existing in list(fc.modifiers):
            if existing.type == mod_type:
                fc.modifiers.remove(existing)

        fmod = fc.modifiers.new(type=mod_type)
        if mod_type == "CYCLES":
            fmod.mode_before = str(params.get("mode_before") or "REPEAT").upper()
            fmod.mode_after = str(params.get("mode_after") or "REPEAT").upper()
        elif mod_type == "NOISE":
            fmod.scale = float(params.get("scale", 10.0))
            fmod.strength = float(params.get("strength", 0.2))
            fmod.phase = float(params["phase"]) if params.get("phase") is not None else float(ch_i * 17.3)
        elif mod_type == "STEPPED":
            fmod.frame_step = max(1.0, float(params.get("frame_step", 2.0)))
        elif mod_type == "LIMITS":
            if params.get("min_y") is not None:
                fmod.use_min_y = True
                fmod.min_y = float(params["min_y"])
            if params.get("max_y") is not None:
                fmod.use_max_y = True
                fmod.max_y = float(params["max_y"])
        fc.update()
        matched += 1

    ctx.scene.frame_set(ctx.scene.frame_current)
    ctx.view_layer.update()
    return {"object": obj.name, "modifier": mod_type, "channels_modified": matched}


@blender_op(
    "anim.clear",
    tier="mutate",
    snapshot_blend=True,
    summary="Clear all animation or specific F-Curve properties/frame ranges on an object (and its datablock).",
    params_doc={
        "object": "Target object name",
        "property": "Optional property substring filter (clears all animation on `object` if omitted)",
        "frame_range": "Optional `[start_frame, end_frame]` to delete only keyframes inside that window",
    },
)
def op_anim_clear(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    obj_name = str(params.get("object") or "").strip()
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise OpError(f"Object not found: '{obj_name}'", code="NOT_FOUND")

    prop_filter = str(params.get("property") or "").strip()
    if prop_filter == "rotation_deg":
        prop_filter = "rotation_euler"
    fr_range = params.get("frame_range")

    removed_curves = 0
    removed_keys = 0

    for id_block in (obj, obj.data):
        if id_block is None or getattr(id_block, "animation_data", None) is None:
            continue
        if not prop_filter and not fr_range:
            id_block.animation_data_clear()
            removed_curves += 1
            continue

        fcs = ensure_channelbag_fcurves(id_block)
        for fc in list(fcs):
            if prop_filter and prop_filter not in fc.data_path:
                continue
            if isinstance(fr_range, (list, tuple)) and len(fr_range) == 2:
                f0, f1 = float(fr_range[0]), float(fr_range[1])
                for kp in reversed(list(fc.keyframe_points)):
                    if f0 <= float(kp.co[0]) <= f1:
                        fc.keyframe_points.remove(kp)
                        removed_keys += 1
                if len(fc.keyframe_points) == 0:
                    fcs.remove(fc)
                    removed_curves += 1
                else:
                    fc.update()
            else:
                fcs.remove(fc)
                removed_curves += 1

    ctx.scene.frame_set(ctx.scene.frame_current)
    ctx.view_layer.update()
    return {"object": obj.name, "removed_curves": removed_curves, "removed_keyframes": removed_keys}


@blender_op(
    "anim.bake",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Bake evaluated world-space motion (constraints, parenting, drivers) into clean Blender 5.3 "
        "keyframes across `[frame_start..frame_end]` and optionally remove constraints/parenting."
    ),
    params_doc={
        "object": "Target object name",
        "frame_start": "Start frame (defaults to `scene.frame_start`)",
        "frame_end": "End frame (defaults to `scene.frame_end`)",
        "step": "Frame sampling step (default 1)",
        "clear_constraints": "Remove object constraints after baking (default true)",
        "clear_parent": "Unparent object after baking world transforms (default false)",
    },
)
def op_anim_bake(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    sc = ctx.scene
    obj_name = str(params.get("object") or "").strip()
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise OpError(f"Object not found: '{obj_name}'", code="NOT_FOUND")

    f_start = int(params.get("frame_start", sc.frame_start))
    f_end = int(params.get("frame_end", sc.frame_end))
    step = max(1, int(params.get("step", 1)))
    frames = list(range(f_start, f_end + 1, step))
    if frames[-1] != f_end:
        frames.append(f_end)

    orig_frame = int(sc.frame_current)
    samples: list[tuple[int, Any, Any, Any]] = []
    prev_eul = None
    try:
        for fr in frames:
            sc.frame_set(fr)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            ev = obj.evaluated_get(depsgraph)
            mw = ev.matrix_world.copy()
            if obj.parent is not None and not bool(params.get("clear_parent", False)):
                p_ev = obj.parent.evaluated_get(depsgraph)
                local_m = p_ev.matrix_world.inverted() @ mw
            else:
                local_m = mw
            loc, quat, scl = local_m.decompose()
            eul = quat.to_euler("XYZ", prev_eul) if prev_eul is not None else quat.to_euler("XYZ")
            prev_eul = eul
            samples.append((fr, loc, eul, scl))
    finally:
        sc.frame_set(orig_frame)

    if bool(params.get("clear_parent", False)) and obj.parent is not None:
        obj.parent = None
    if bool(params.get("clear_constraints", True)):
        obj.constraints.clear()

    obj.rotation_mode = "XYZ"
    fcurves = ensure_channelbag_fcurves(obj)
    for fc in list(fcurves):
        if fc.data_path in ("location", "rotation_euler", "scale"):
            fcurves.remove(fc)

    loc_fcs = [fcurves.new(data_path="location", index=ax) for ax in range(3)]
    rot_fcs = [fcurves.new(data_path="rotation_euler", index=ax) for ax in range(3)]
    scl_fcs = [fcurves.new(data_path="scale", index=ax) for ax in range(3)]
    for fc in [*loc_fcs, *rot_fcs, *scl_fcs]:
        fc.keyframe_points.add(len(samples))

    for i, (fr, loc, eul, scl) in enumerate(samples):
        for ax in range(3):
            kl = loc_fcs[ax].keyframe_points[i]
            kl.co = (float(fr), float(loc[ax]))
            kl.interpolation = "LINEAR"

            kr = rot_fcs[ax].keyframe_points[i]
            kr.co = (float(fr), float(eul[ax]))
            kr.interpolation = "LINEAR"

            ks = scl_fcs[ax].keyframe_points[i]
            ks.co = (float(fr), float(scl[ax]))
            ks.interpolation = "LINEAR"

    for fc in [*loc_fcs, *rot_fcs, *scl_fcs]:
        fc.update()

    sc.frame_set(orig_frame)
    ctx.view_layer.update()
    return {
        "object": obj.name,
        "frames_baked": len(samples),
        "frame_range": [f_start, f_end],
        "constraints_cleared": bool(params.get("clear_constraints", True)),
    }


@blender_op(
    "anim.nla_layer",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Non-destructive animation layering in Blender 5.3: push an object's active (or named) Slotted Action "
        "onto an NLA track with `REPLACE`, `ADD`, `SUBTRACT`, `MULTIPLY`, or `COMBINE` blending, `influence`, "
        "`scale`, `repeat`, and `mute` — enabling layered override/additive animation without third-party add-ons."
    ),
    params_doc={
        "object": "Target object name",
        "track": "NLA track name to create or update (default `'AnimLayer'`)",
        "action": "Action name to place on the strip (defaults to `object`'s current active Action, detaching it from the active slot so you can author a new layer on top!)",
        "start_frame": "Timeline frame where the NLA strip starts (defaults to the Action's first keyframe)",
        "blend_type": "`'REPLACE'`, `'ADD'` (default for override layers!), `'SUBTRACT'`, `'MULTIPLY'`, or `'COMBINE'`",
        "influence": "Layer influence `0.0 .. 1.0` (default `1.0`)",
        "scale": "Time playback scale factor (default `1.0`)",
        "repeat": "Number of loop repetitions (default `1.0`)",
        "extrapolation": "`'NOTHING'`, `'HOLD'`, or `'HOLD_FORWARD'`",
        "mute": "Mute this animation layer (default `false`)",
        "keep_active_action": "If false (default when pushing current Action), clears `anim_data.action` after pushing to NLA so subsequent `anim.keyframe` calls author a clean override layer",
    },
)
def op_anim_nla_layer(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    bpy = ctx.bpy
    obj_name = str(params.get("object") or "").strip()
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise OpError(f"Object not found: '{obj_name}'", code="NOT_FOUND")

    anim = obj.animation_data_create()
    explicit_act_name = str(params.get("action") or "").strip()
    pushed_active = False
    if explicit_act_name:
        act = bpy.data.actions.get(explicit_act_name)
        if act is None:
            raise OpError(f"Action not found: '{explicit_act_name}'", code="NOT_FOUND")
    else:
        act = anim.action
        if act is None:
            raise OpError(
                f"Object '{obj.name}' has no active Action to push onto NLA (pass 'action' or keyframe first)",
                code="NO_ANIMATION",
            )
        pushed_active = True

    slot_to_bind = getattr(anim, "action_slot", None) if anim.action == act else None
    if slot_to_bind is None and hasattr(act, "slots") and len(act.slots) > 0:
        id_type = getattr(obj, "id_type", "OBJECT")
        slot_to_bind = next(
            (
                s
                for s in act.slots
                if getattr(s, "target_id_type", None) == id_type
                and (s.name_display == obj.name or s.identifier.endswith(obj.name))
            ),
            act.slots[0],
        )

    track_name = str(params.get("track") or f"Layer_{len(anim.nla_tracks) + 1}").strip()
    nla_track = next((t for t in anim.nla_tracks if t.name == track_name), None)
    if nla_track is None:
        nla_track = anim.nla_tracks.new()
        nla_track.name = track_name
    nla_track.mute = bool(params.get("mute", False))

    act_start = round(float(act.frame_range[0])) if hasattr(act, "frame_range") else ctx.scene.frame_start
    start_fr = int(params.get("start_frame", act_start))

    # Remove overlapping strip with same action on this track for idempotence
    for st_existing in list(nla_track.strips):
        if st_existing.action == act and round(float(st_existing.frame_start)) == start_fr:
            nla_track.strips.remove(st_existing)

    strip = nla_track.strips.new(name=act.name, start=start_fr, action=act)
    if slot_to_bind is not None and hasattr(strip, "action_slot"):
        strip.action_slot = slot_to_bind

    default_blend = "REPLACE" if len(anim.nla_tracks) == 1 else "ADD"
    blend_type = str(params.get("blend_type") or default_blend).strip().upper()
    strip.blend_type = blend_type
    if params.get("extrapolation"):
        strip.extrapolation = str(params["extrapolation"]).strip().upper()
    elif blend_type == "ADD":
        strip.extrapolation = "NOTHING"

    if params.get("influence") is not None:
        inf = max(0.0, min(1.0, float(params["influence"])))
        strip.use_animated_influence = True
        strip.influence = inf
        strip.keyframe_insert(data_path="influence", frame=float(start_fr))
    if params.get("scale") is not None:
        strip.scale = max(0.01, float(params["scale"]))
    if params.get("repeat") is not None:
        strip.repeat = max(0.1, float(params["repeat"]))

    if (pushed_active or anim.action == act) and not bool(params.get("keep_active_action", False)):
        anim.action = None

    ctx.scene.frame_set(ctx.scene.frame_current)
    ctx.view_layer.update()
    return {
        "object": obj.name,
        "track": nla_track.name,
        "strip": strip.name,
        "action": act.name,
        "blend_type": strip.blend_type,
        "influence": round(float(strip.influence), 4),
        "frame_span": [round(float(strip.frame_start), 2), round(float(strip.frame_end), 2)],
        "total_nla_tracks": len(anim.nla_tracks),
    }