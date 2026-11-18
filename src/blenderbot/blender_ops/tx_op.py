"""Transactional `bpy.tx` procedural code execution + gated `bpy.eval` escape hatch.

Why `bpy.tx` is a game-changer for Blender previs:
- Procedural meshes, terrain displacement, shader node trees, and multi-object constraints are
  far cleaner and 10x more expressive in 30 lines of `bpy`/`bmesh`/`mathutils` than 400 rigid RPCs.
- Raw `Execute Python` (as in basic MCP plugins) leaves half-built orphan meshes, broken constraints,
  and dirty state whenever a script raises mid-way.
- `bpy.tx` automatically saves a `.blend` snapshot BEFORE executing `code`, provides Blender 5.x
  slotted-action & previs helpers (`ensure_collection`, `link_exclusively`, `assign_solid_material`,
  `ensure_channelbag_fcurves`, `iter_action_fcurves`, `world_aabb`, `BVHTree`), and if ANY exception
  or failed `assert` occurs, immediately reloads the snapshot so `bpy.data` rolls back atomically!
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import traceback
from typing import Any

from .helpers import (
    assign_solid_material,
    ensure_channelbag_fcurves,
    ensure_collection,
    iter_action_fcurves,
    link_exclusively,
    restore_snapshot,
    save_temp_snapshot,
    world_aabb,
)
from .schema import OpContext, OpError, blender_op


def _sanitize_json_value(val: Any, depth: int = 0) -> Any:
    if depth > 6:
        return repr(val)
    if val is None or isinstance(val, (str, int, bool)):
        return val
    if isinstance(val, float):
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(val, (list, tuple)):
        return [_sanitize_json_value(v, depth + 1) for v in val[:500]]
    if isinstance(val, dict):
        return {str(k): _sanitize_json_value(v, depth + 1) for k, v in list(val.items())[:200]}
    # mathutils Vector / Euler / Color
    if hasattr(val, "to_tuple"):
        with contextlib.suppress(TypeError, ValueError, AttributeError):
            return [round(float(x), 5) for x in val.to_tuple()]
    return repr(val)


def _build_tx_globals(ctx: OpContext) -> dict[str, Any]:
    import bmesh
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Color, Euler, Matrix, Quaternion, Vector
    from mathutils.bvhtree import BVHTree

    return {
        "__builtins__": __builtins__,
        "bpy": bpy,
        "bmesh": bmesh,
        "mathutils": mathutils,
        "math": math,
        "json": json,
        "Vector": Vector,
        "Matrix": Matrix,
        "Euler": Euler,
        "Quaternion": Quaternion,
        "Color": Color,
        "BVHTree": BVHTree,
        "world_to_camera_view": world_to_camera_view,
        "scene": ctx.scene,
        "view_layer": ctx.view_layer,
        # Blender 5.x safe helpers pre-injected for concise procedural scripts:
        "ensure_collection": ensure_collection,
        "link_exclusively": link_exclusively,
        "assign_solid_material": assign_solid_material,
        "ensure_fcurves": ensure_channelbag_fcurves,
        "ensure_channelbag_fcurves": ensure_channelbag_fcurves,
        "iter_fcurves": iter_action_fcurves,
        "iter_action_fcurves": iter_action_fcurves,
        "world_aabb": world_aabb,
        "result": None,
    }


@blender_op(
    "bpy.tx",
    tier="mutate",
    snapshot_blend=True,
    summary=(
        "Execute procedural `bpy`/`bmesh`/`mathutils` Python atomically inside Blender with automatic "
        "`.blend` snapshot rollback on any exception or assertion failure."
    ),
    params_doc={
        "description": "Concise description of what this procedural transaction builds or modifies",
        "code": "Python script (ALWAYS pass via `@NAME` payload block!). Assign summary dict to `result`.",
    },
)
def op_bpy_tx(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    code = str(params.get("code") or "")
    if not code.strip():
        raise OpError("bpy.tx requires non-empty 'code'", code="MISSING_PARAM")

    bpy = ctx.bpy
    orig_filepath = bpy.data.filepath or ""
    snap_path = save_temp_snapshot(bpy, ctx.snaps_dir, prefix="bpy_tx")

    env_globals = _build_tx_globals(ctx)
    stdout_buf = io.StringIO()

    try:
        compiled = compile(code, "<bpy.tx>", "exec")
        with contextlib.redirect_stdout(stdout_buf):
            exec(compiled, env_globals, env_globals)  # noqa: S102 — transactional Blender procedural mode
        bpy.context.view_layer.update()
    except Exception as err:
        tb_short = "".join(traceback.format_exception_only(type(err), err)).strip()
        lineno = getattr(err, "lineno", None)
        if lineno is None and err.__traceback__ is not None:
            tb_frames = traceback.extract_tb(err.__traceback__)
            for fr in reversed(tb_frames):
                if fr.filename == "<bpy.tx>":
                    lineno = fr.lineno
                    break
        # Atomic rollback!
        rollback_ok = True
        try:
            restore_snapshot(bpy, snap_path, original_filepath=orig_filepath)
        except Exception:  # noqa: BLE001
            rollback_ok = False
        finally:
            with contextlib.suppress(OSError):
                snap_path.unlink()

        loc_str = f" (line {lineno})" if lineno else ""
        out_str = stdout_buf.getvalue().strip()
        extra = f"\nstdout before failure:\n{out_str}" if out_str else ""
        rb_note = "Scene rolled back cleanly to pre-tx state." if rollback_ok else "WARNING: snapshot rollback failed!"
        raise OpError(
            f"bpy.tx aborted{loc_str}: {tb_short}. {rb_note}{extra}",
            code="TX_ABORTED",
        ) from err

    # Clean up temporary per-op snapshot once committed cleanly
    with contextlib.suppress(OSError):
        snap_path.unlink()

    return {
        "committed": True,
        "description": str(params.get("description") or ""),
        "stdout": stdout_buf.getvalue(),
        "result": _sanitize_json_value(env_globals.get("result")),
    }


@blender_op(
    "bpy.eval",
    tier="eval",
    snapshot_blend=True,
    summary="Gated raw Python evaluation inside Blender (requires `master` role + operator approval).",
    params_doc={
        "description": "Explanation shown in operator approval prompt",
        "code": "Python code (pass via `@NAME` payload block)",
    },
)
def op_bpy_eval(ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
    return op_bpy_tx(ctx, params)