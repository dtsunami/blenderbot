"""Two-phase `bl_ops` batch planner & executor (`blenderbot.agent.executor`).

Supports:
- All Blender 5.3+ domain ops (`blenderbot.blender_ops`) including transactional `bpy.tx`
- Automatic `.blend` pre-snapshot checkpointing (`scene.snapshot_save`) before mutating batches
  so failed batches roll back cleanly and `/undo` + `/redo` work across the session
- Automatic multimodal context attachment when `viewport.render_still`, `viewport.contact_sheet`,
  `render.still` (image), or `render.playblast` / `render.animation` (`attach_to_context: true`, video) run
- Workspace filesystem ops (`fs.read`, `fs.write`, `fs.replace`, `fs.list`)
- Session undo/redo ops (`journal.history`, `journal.undo`, `journal.redo`)
- Segregated approval gate for `bpy.eval`, `shell`, and `destructive` ops (`scene.clear`, `scene.open`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blenderbot.blender_ops import get_op
from blenderbot.config.env import Config
from blenderbot.transport.base import BlenderTransport

from .journal import BlendCheckpoint, FileEntry, Journal
from .parser import OpCall
from .sandbox import Sandbox, SandboxError

FS_OPS = ("fs.read", "fs.write", "fs.replace", "fs.list")
JOURNAL_OPS = ("journal.history", "journal.undo", "journal.redo")
GATED_TIERS = ("eval", "destructive")

# Render/export ops whose `output_path` or `filepath` writes to the workspace
WRITE_PATH_PARAM_OPS: dict[str, str] = {
    "scene.save": "filepath",
    "asset.export": "filepath",
    "viewport.render_still": "output_path",
    "viewport.contact_sheet": "output_path",
    "render.playblast": "output_path",
    "render.still": "output_path",
    "render.animation": "output_path",
    "splat.capture_views": "output_path",
}

READ_PATH_PARAM_OPS: dict[str, str] = {
    "scene.open": "filepath",
    "asset.import": "filepath",
    "asset.blend_append": "filepath",
    "splat.import": "filepath",
    "splat.project_video": "video_path",
}


@dataclass
class ApprovalDecision:
    allow: bool
    reason: str | None = None
    checkpoint_first: bool = False


ApproveHook = Callable[[OpCall], Awaitable[ApprovalDecision]]


@dataclass
class BatchOutcome:
    status: str  # "ok" | "plan-failed" | "apply-failed" | "denied"
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: BlendCheckpoint | None = None
    attached_media: list[dict[str, Any]] = field(default_factory=list)


def _plan_fs_op(call: OpCall, sandbox: Sandbox) -> str | None:
    p = call.params
    if call.op == "fs.read":
        if not p.get("path"):
            return "fs.read requires 'path'"
        target = sandbox.resolve_for_read(str(p["path"]))
        if not target.is_file():
            return f"{sandbox.rel(target)} does not exist"
    elif call.op == "fs.list":
        target = sandbox.resolve_for_read(str(p.get("path", ".")))
        if not target.is_dir():
            return f"{sandbox.rel(target)} is not a directory"
    elif call.op == "fs.write":
        if not p.get("path") or "text" not in p:
            return "fs.write requires 'path' and 'text'"
        target = sandbox.resolve_for_write(str(p["path"]))
        if target.exists() and not bool(p.get("overwrite", False)):
            return f"{sandbox.rel(target)} already exists (pass overwrite: true or use fs.replace)"
    elif call.op == "fs.replace":
        if not p.get("path") or "old" not in p or "new" not in p:
            return "fs.replace requires 'path', 'old', and 'new'"
        target = sandbox.resolve_for_write(str(p["path"]))
        if not target.is_file():
            return f"{sandbox.rel(target)} does not exist"
        if str(p["old"]) == "":
            return "fs.replace 'old' cannot be empty"
        content = target.read_text(encoding="utf-8")
        count = content.count(str(p["old"]))
        if count == 0:
            return f"'old' text not found in {sandbox.rel(target)}"
        if count > 1 and not bool(p.get("all", False)):
            return (
                f"'old' text occurs {count} times in {sandbox.rel(target)} "
                "(include surrounding context or set all: true)"
            )
    return None


def plan_batch(
    ops: list[OpCall],
    *,
    sandbox: Sandbox,
    has_transport: bool,
    role: str = "master",
) -> list[dict[str, Any]]:
    from .roles import is_op_allowed_for_role

    errors: list[dict[str, Any]] = []

    for call in ops:
        spec = get_op(call.op)
        if not is_op_allowed_for_role(role, call.op, spec.tier if spec else None):
            errors.append(
                {
                    "index": call.index,
                    "op": call.op,
                    "message": (
                        f'op "{call.op}" is not permitted under active role "{role}" '
                        "(switch via `/role builder` or `/role master`)"
                    ),
                }
            )
            continue

        try:
            if call.op == "shell":
                if not str(call.params.get("cmd", "")).strip():
                    errors.append(
                        {"index": call.index, "op": call.op, "message": "shell requires 'cmd'"}
                    )
                elif call.params.get("cwd"):
                    sh_dir = sandbox.resolve_for_read(str(call.params["cwd"]))
                    if not sh_dir.is_dir():
                        errors.append(
                            {
                                "index": call.index,
                                "op": call.op,
                                "message": f"{sandbox.rel(sh_dir)} is not a directory",
                            }
                        )
            elif call.op in JOURNAL_OPS:
                pass
            elif call.op == "gen.seedance":
                p = call.params
                has_cond = any(
                    p.get(k) for k in ("prompt", "first_frame", "last_frame", "video_path")
                )
                if not has_cond:
                    errors.append(
                        {
                            "index": call.index,
                            "op": call.op,
                            "message": "gen.seedance requires 'prompt' and/or 'first_frame'/'last_frame'/'video_path'",
                        }
                    )
                if p.get("output_path"):
                    sandbox.resolve_for_write(str(p["output_path"]))
                for rkey in ("first_frame", "last_frame", "video_path", "audio_path"):
                    rval = p.get(rkey)
                    if rval and not str(rval).startswith(("https://", "http://", "data:")):
                        rf = sandbox.resolve_for_read(str(rval))
                        if not rf.is_file():
                            errors.append(
                                {
                                    "index": call.index,
                                    "op": call.op,
                                    "message": f"{sandbox.rel(rf)} does not exist",
                                }
                            )
                for ref_img in p.get("reference_images") or []:
                    if ref_img and not str(ref_img).startswith(("https://", "http://", "data:")):
                        rf = sandbox.resolve_for_read(str(ref_img))
                        if not rf.is_file():
                            errors.append(
                                {
                                    "index": call.index,
                                    "op": call.op,
                                    "message": f"{sandbox.rel(rf)} does not exist",
                                }
                            )
            elif call.op == "splat.from_video":
                p = call.params
                if not p.get("video_path"):
                    errors.append(
                        {"index": call.index, "op": call.op, "message": "splat.from_video requires 'video_path'"}
                    )
                else:
                    vf = sandbox.resolve_for_read(str(p["video_path"]))
                    if not vf.is_file():
                        errors.append(
                            {"index": call.index, "op": call.op, "message": f"{sandbox.rel(vf)} does not exist"}
                        )
                if p.get("output_dir"):
                    sandbox.resolve_for_write(str(p["output_dir"]))
            elif call.op == "splat.from_blender":
                p = call.params
                if p.get("views_bundle_path"):
                    vb = sandbox.resolve_for_read(str(p["views_bundle_path"]))
                    if not vb.is_file():
                        errors.append(
                            {"index": call.index, "op": call.op, "message": f"{sandbox.rel(vb)} does not exist"}
                        )
                elif not has_transport:
                    errors.append(
                        {
                            "index": call.index,
                            "op": call.op,
                            "message": "splat.from_blender requires Blender connected (or pass 'views_bundle_path')",
                        }
                    )
                if p.get("output_dir"):
                    sandbox.resolve_for_write(str(p["output_dir"]))
            elif call.op == "splat.resume":
                p = call.params
                if not p.get("job_id"):
                    errors.append(
                        {"index": call.index, "op": call.op, "message": "splat.resume requires 'job_id'"}
                    )
                for mpath in (p.get("masks") or {}).values():
                    if mpath:
                        mf = sandbox.resolve_for_read(str(mpath))
                        if not mf.is_file():
                            errors.append(
                                {"index": call.index, "op": call.op, "message": f"{sandbox.rel(mf)} does not exist"}
                            )
                if p.get("output_dir"):
                    sandbox.resolve_for_write(str(p["output_dir"]))
            elif call.op in FS_OPS:
                err = _plan_fs_op(call, sandbox)
                if err:
                    errors.append({"index": call.index, "op": call.op, "message": err})
            else:
                if spec is None:
                    errors.append(
                        {"index": call.index, "op": call.op, "message": f'unknown op "{call.op}"'}
                    )
                elif not has_transport:
                    errors.append(
                        {
                            "index": call.index,
                            "op": call.op,
                            "message": "Blender is not connected (run `/blender` to connect)",
                        }
                    )
                else:
                    if call.op in WRITE_PATH_PARAM_OPS:
                        key = WRITE_PATH_PARAM_OPS[call.op]
                        raw_val = call.params.get(key)
                        if raw_val:
                            sandbox.resolve_for_write(str(raw_val))
                    if call.op in READ_PATH_PARAM_OPS:
                        key = READ_PATH_PARAM_OPS[call.op]
                        raw_val = call.params.get(key)
                        if not raw_val:
                            errors.append(
                                {
                                    "index": call.index,
                                    "op": call.op,
                                    "message": f"{call.op} requires '{key}'",
                                }
                            )
                        else:
                            sandbox.resolve_for_read(str(raw_val))
        except SandboxError as s_err:
            errors.append({"index": call.index, "op": call.op, "message": str(s_err)})
    return errors


def _exec_fs_op(
    call: OpCall, sandbox: Sandbox, journal: Journal
) -> tuple[FileEntry | None, FileEntry | None, dict[str, Any]]:
    p = call.params
    if call.op == "fs.read":
        target = sandbox.resolve_for_read(str(p["path"]))
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        line_range = p.get("lines")
        if isinstance(line_range, str) and "-" in line_range:
            parts = line_range.split("-", 1)
            if parts[0].strip().isdigit() and parts[1].strip().isdigit():
                line_range = [int(parts[0].strip()), int(parts[1].strip())]
        if isinstance(line_range, list) and len(line_range) == 2:
            lo = max(1, int(line_range[0]))
            hi = min(len(lines), int(line_range[1]))
        else:
            lo, hi = 1, min(len(lines), 400)
        sliced = lines[lo - 1 : hi]
        return (
            None,
            None,
            {
                "path": sandbox.rel(target),
                "lines": [lo, hi],
                "total_lines": len(lines),
                "truncated": hi < len(lines) and line_range is None,
                "text": "\n".join(sliced),
            },
        )

    if call.op == "fs.list":
        target = sandbox.resolve_for_read(str(p.get("path", ".")))
        max_items = int(p.get("max", 200))
        entries = []
        for child in sorted(target.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if sandbox.is_ignored(child):
                continue
            entries.append(
                f"{child.name}/" if child.is_dir() else f"{child.name} ({child.stat().st_size}b)"
            )
            if len(entries) >= max_items:
                break
        return None, None, {"path": sandbox.rel(target), "entries": entries}

    if call.op == "fs.write":
        target = sandbox.resolve_for_write(str(p["path"]))
        before = journal.snapshot_file(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.blenderbot-tmp")
        text = str(p["text"])
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        after = journal.snapshot_file(target, log_event=False)
        return (
            before,
            after,
            {
                "path": sandbox.rel(target),
                "action": "overwritten" if before.existed else "created",
                "bytes": len(text.encode("utf-8")),
            },
        )

    if call.op == "fs.replace":
        target = sandbox.resolve_for_write(str(p["path"]))
        before = journal.snapshot_file(target)
        raw = target.read_text(encoding="utf-8")
        old_s = str(p["old"])
        new_s = str(p["new"])
        replaced = (
            raw.replace(old_s, new_s)
            if bool(p.get("all", False))
            else raw.replace(old_s, new_s, 1)
        )
        tmp = target.with_name(f"{target.name}.blenderbot-tmp")
        tmp.write_text(replaced, encoding="utf-8")
        os.replace(tmp, target)
        after = journal.snapshot_file(target, log_event=False)
        return before, after, {"path": sandbox.rel(target), "action": "replaced"}

    raise ValueError(f"Unhandled fs op: {call.op}")


def _exec_journal_op_sync_history(call: OpCall, journal: Journal) -> dict[str, Any]:
    state = journal.list_state()
    limit = int(call.params.get("limit", 25))
    actor_filter = str(call.params.get("actor", "")).strip().lower()
    rows = state["actions"]
    if actor_filter:
        rows = [r for r in rows if str(r.get("actor", "")).lower() == actor_filter]
    return {
        "cursor": state["cursor"],
        "can_undo": state["can_undo"],
        "can_redo": state["can_redo"],
        "actions": [
            {
                "id": r["id"],
                "seq": r["seq"],
                "actor": r["actor"],
                "type": r["action_type"],
                "applied": r["applied"],
                "superseded": r["superseded"],
                "summary": r["summary"],
            }
            for r in rows[-limit:]
        ],
    }


async def _run_shell_cmd(cmd: str, cwd: Path, timeout_sec: float) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def _blocking() -> dict[str, Any]:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(1.0, float(timeout_sec)),
                check=False,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-16000:],
                "stderr": proc.stderr[-8000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": 124,
                "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
                "stderr": f"Timed out after {timeout_sec}s",
            }

    return await loop.run_in_executor(None, _blocking)


def _needs_blend_checkpoint(calls: list[OpCall]) -> bool:
    """Return True if any op in `calls` mutates the Blender scene state (excluding pure render/save)."""
    non_scene_mutators = frozenset(
        {"scene.save", "asset.export", "render.playblast", "render.still", "render.animation"}
    )
    for c in calls:
        if c.op in non_scene_mutators:
            continue
        spec = get_op(c.op)
        if spec and (spec.snapshot_blend or spec.tier in ("mutate", "destructive", "eval")):
            return True
    return False


def _is_gated_op(call: OpCall) -> bool:
    if call.op == "shell":
        return True
    spec = get_op(call.op)
    return bool(spec and spec.tier in GATED_TIERS)


def _normalize_blender_params(call: OpCall, sandbox: Sandbox) -> dict[str, Any]:
    params = dict(call.params)
    # Strip Harness-only `attach_to_context` before sending to Blender if desired (safe either way)
    params.pop("attach_to_context", None)

    if call.op in WRITE_PATH_PARAM_OPS:
        key = WRITE_PATH_PARAM_OPS[call.op]
        if params.get(key):
            params[key] = str(sandbox.resolve_for_write(str(params[key])))
        elif call.op in ("viewport.render_still", "viewport.contact_sheet"):
            renders_dir = sandbox.cwd / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)
            default_name = (
                "viewport_still.png"
                if call.op == "viewport.render_still"
                else "contact_sheet.png"
            )
            params[key] = str(renders_dir / default_name)

    if call.op in READ_PATH_PARAM_OPS:
        key = READ_PATH_PARAM_OPS[call.op]
        if params.get(key):
            params[key] = str(sandbox.resolve_for_read(str(params[key])))
    if call.op == "splat.import" and params.get("cameras_json"):
        params["cameras_json"] = str(sandbox.resolve_for_read(str(params["cameras_json"])))

    return params


def _op_timeout_for(op_name: str, params: dict[str, Any]) -> float:
    if op_name in ("render.animation", "render.playblast", "splat.capture_views"):
        return float(params.get("timeout_sec", 300.0))
    if op_name in ("render.still", "viewport.contact_sheet", "bpy.tx", "bpy.eval", "anim.bake", "splat.import"):
        return float(params.get("timeout_sec", 180.0))
    return 90.0


def _collect_media_attachments(
    call: OpCall, res: dict[str, Any], sandbox: Sandbox
) -> list[dict[str, Any]]:
    out_path_raw = res.get("output_path")
    if not out_path_raw or not isinstance(out_path_raw, str):
        return []
    out_p = Path(out_path_raw)
    if not out_p.is_file():
        return []

    sandbox.allow_read_file(out_p)
    ext = out_p.suffix.lower()

    # Default `attach_to_context` is True for viewport stills, contact sheets, and PNG/JPG `render.still`
    default_attach = call.op in ("viewport.render_still", "viewport.contact_sheet") or (
        call.op == "render.still" and ext in (".png", ".jpg", ".jpeg", ".webp")
    )
    want_attach = bool(call.params.get("attach_to_context", default_attach))
    if not want_attach:
        return []

    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        return [{"kind": "image", "path": str(out_p), "op": call.op}]
    if ext in (".mp4", ".mov", ".webm"):
        return [{"kind": "video", "path": str(out_p), "op": call.op}]
    return []


async def execute_batch(
    ops: list[OpCall],
    *,
    sandbox: Sandbox,
    journal: Journal,
    transport: BlenderTransport | None = None,
    cfg: Config | None = None,
    role: str | None = None,
    yolo: bool = False,
    shell_timeout: float = 120.0,
    auto_checkpoint: bool = True,
    approve: ApproveHook | None = None,
    actor: str = "agent",
) -> BatchOutcome:
    eff_role = role or (cfg.role if cfg else "master")
    plan_errors = plan_batch(
        ops,
        sandbox=sandbox,
        has_transport=transport is not None,
        role=eff_role,
    )
    if plan_errors:
        return BatchOutcome(status="plan-failed", errors=plan_errors)

    safe_queue = [c for c in ops if not _is_gated_op(c)]
    gated_queue = [c for c in ops if _is_gated_op(c)]

    checkpoint: BlendCheckpoint | None = None
    if auto_checkpoint and transport is not None and _needs_blend_checkpoint(safe_queue):
        with contextlib.suppress(Exception):
            checkpoint = await journal.create_blend_checkpoint(transport)

    results: list[dict[str, Any]] = []
    before_files: list[FileEntry] = []
    after_files: list[FileEntry] = []
    mutated_ops_names: list[str] = []
    attached_media: list[dict[str, Any]] = []

    # --- Pass 1: Safe / non-gated ops (transactional rollback on failure) ---
    for call in safe_queue:
        try:
            if call.op in FS_OPS:
                bf, af, res = _exec_fs_op(call, sandbox, journal)
                if bf is not None:
                    before_files.append(bf)
                    mutated_ops_names.append(f"{call.op}({call.params.get('path', '')})")
                if af is not None:
                    after_files.append(af)
            elif call.op == "journal.history":
                res = _exec_journal_op_sync_history(call, journal)
            elif call.op == "journal.undo":
                if call.params.get("id"):
                    res = await journal.undo_entry(str(call.params["id"]), transport)
                else:
                    res = await journal.undo_last(
                        transport, steps=int(call.params.get("steps", 1))
                    )
            elif call.op == "journal.redo":
                if call.params.get("id"):
                    res = await journal.redo_entry(str(call.params["id"]), transport)
                else:
                    res = await journal.redo_next(
                        transport, steps=int(call.params.get("steps", 1))
                    )
            elif call.op == "gen.seedance":
                from blenderbot.media_gen.seedance import (
                    default_seedance_output_path,
                    generate_seedance_video,
                )

                or_key = (cfg.keys.get("openrouter") if cfg else None) or os.environ.get(
                    "OPENROUTER_API_KEY", ""
                )
                p = call.params
                if p.get("output_path"):
                    out_sd = sandbox.resolve_for_write(str(p["output_path"]))
                else:
                    out_sd = default_seedance_output_path(sandbox.cwd)

                def _res_opt_read(val: Any) -> str | Path | None:
                    if not val:
                        return None
                    s = str(val).strip()
                    if s.startswith(("https://", "http://", "data:")):
                        return s
                    return sandbox.resolve_for_read(s)

                bf_sd = journal.snapshot_file(out_sd)
                res = await generate_seedance_video(
                    api_key=or_key,
                    output_path=out_sd,
                    model=str(p.get("model") or (cfg.video_model if cfg else "bytedance/seedance-2.5")),
                    prompt=str(p["prompt"]) if p.get("prompt") else None,
                    mode=str(p.get("mode") or "standard"),
                    duration=p.get("duration"),
                    size=str(p["size"]) if p.get("size") else None,
                    resolution=str(p["resolution"]) if p.get("resolution") else None,
                    aspect_ratio=str(p["aspect_ratio"]) if p.get("aspect_ratio") else None,
                    first_frame=_res_opt_read(p.get("first_frame")),
                    last_frame=_res_opt_read(p.get("last_frame")),
                    video_path=_res_opt_read(p.get("video_path")),
                    video_lock_first=bool(p.get("video_lock_first", True)),
                    video_lock_last=bool(p.get("video_lock_last", True)),
                    video_mid_frames=int(p.get("video_mid_frames", 2)),
                    extract_video_audio=bool(p.get("extract_video_audio", False)),
                    reference_images=[
                        r_item
                        for r_item in (_res_opt_read(x) for x in (p.get("reference_images") or []))
                        if r_item is not None
                    ],
                    audio_path=_res_opt_read(p.get("audio_path")),
                    generate_audio=p.get("generate_audio"),
                    seed=int(p["seed"]) if p.get("seed") is not None else None,
                    timeout_sec=float(p.get("timeout_sec", 600.0)),
                )
                before_files.append(bf_sd)
                after_files.append(journal.snapshot_file(out_sd))
                mutated_ops_names.append(f"gen.seedance({sandbox.rel(out_sd)})")
                attached_media.extend(_collect_media_attachments(call, res, sandbox))
            elif call.op in ("splat.from_video", "splat.from_blender", "splat.resume"):
                from blenderbot.splats.client import make_job_id, run_modal_splat_job
                from blenderbot.splats.config import SplatConfig, SurgerySpec

                p = call.params
                splat_cfg = SplatConfig.from_dict(p.get("config") or p)
                if p.get("gpu"):
                    splat_cfg.gpu = str(p["gpu"]).upper()
                if p.get("iterations") is not None:
                    splat_cfg.train.iterations = int(p["iterations"])
                if p.get("sh_degree") is not None:
                    splat_cfg.train.sh_degree = int(p["sh_degree"])
                if p.get("pose_backend"):
                    splat_cfg.pose.backend = str(p["pose_backend"])

                video_p: Path | None = None
                bundle_p: Path | None = None
                if call.op == "splat.from_video":
                    video_p = sandbox.resolve_for_read(str(p["video_path"]))
                    job_id = str(p.get("job_id") or make_job_id(video_p))
                elif call.op == "splat.from_blender":
                    job_id = str(p.get("job_id") or make_job_id(Path("blender_scene")))
                else:
                    job_id = str(p["job_id"])

                out_dir = (
                    sandbox.resolve_for_write(str(p["output_dir"]))
                    if p.get("output_dir")
                    else (sandbox.cwd / "assets" / "splats" / job_id)
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                if call.op == "splat.from_blender":
                    if p.get("views_bundle_path"):
                        bundle_p = sandbox.resolve_for_read(str(p["views_bundle_path"]))
                    else:
                        assert transport is not None
                        bundle_p = out_dir / "views_bundle.tar.gz"
                        cap_params: dict[str, Any] = {"output_path": str(bundle_p)}
                        for ck in (
                            "targets",
                            "collection",
                            "cameras",
                            "num_views",
                            "elevations_deg",
                            "distance_scale",
                            "lens_mm",
                            "resolution",
                            "engine",
                            "transparent_bg",
                            "max_init_points",
                        ):
                            if ck in p and p[ck] is not None:
                                cap_params[ck] = p[ck]
                        await transport.call(
                            "splat.capture_views",
                            cap_params,
                            timeout=float(p.get("capture_timeout_sec", 300.0)),
                        )

                masks_map: dict[str, Path] | None = None
                if p.get("masks"):
                    masks_map = {
                        str(k): sandbox.resolve_for_read(str(v))
                        for k, v in p["masks"].items()
                        if v
                    }
                surg_spec = SurgerySpec.from_dict(p["surgery"]) if p.get("surgery") else None

                splat_res = await run_modal_splat_job(
                    output_dir=out_dir,
                    job_id=job_id,
                    video_path=video_p,
                    views_bundle_path=bundle_p,
                    cfg=splat_cfg,
                    add_steps=int(p["add_steps"]) if p.get("add_steps") is not None else None,
                    surgery=surg_spec,
                    masks_paths=masks_map,
                    exclude_frames=[str(x) for x in p["exclude_frames"]] if p.get("exclude_frames") else None,
                )

                imported_info: dict[str, Any] | None = None
                if bool(p.get("auto_import", False)) and transport is not None:
                    imp_params: dict[str, Any] = {
                        "filepath": str(splat_res.ply_path),
                        "cameras_json": str(splat_res.cameras_path),
                        "render_mode": str(p.get("render_mode") or "pointcloud"),
                        "max_points": int(p.get("max_points", 250_000)),
                    }
                    if p.get("name"):
                        imp_params["name"] = str(p["name"])
                    imported_info = await transport.call("splat.import", imp_params, timeout=180.0)

                preview_img = out_dir / "previews" / "latest_render.jpg"
                if preview_img.is_file() and bool(p.get("attach_to_context", True)):
                    sandbox.allow_read_file(preview_img)
                    attached_media.append({"kind": "image", "path": str(preview_img), "op": call.op})

                res = {
                    "job_id": splat_res.job_id,
                    "burst_id": splat_res.burst_id,
                    "step": splat_res.step,
                    "n_gaussians": splat_res.n_gaussians,
                    "device": splat_res.device,
                    "wall_s": splat_res.wall_s,
                    "cost_usd": splat_res.cost_usd,
                    "macro_timings": splat_res.macro_timings,
                    "ply_path": sandbox.rel(splat_res.ply_path),
                    "splat_path": sandbox.rel(splat_res.splat_path) if splat_res.splat_path else None,
                    "cameras_path": sandbox.rel(splat_res.cameras_path),
                    "views_bundle_path": sandbox.rel(bundle_p) if bundle_p else None,
                    "imported": imported_info,
                }
                mutated_ops_names.append(f"{call.op}({splat_res.job_id})")
            else:
                assert transport is not None
                call_params = _normalize_blender_params(call, sandbox)
                res = await transport.call(
                    call.op, call_params, timeout=_op_timeout_for(call.op, call.params)
                )
                spec_c = get_op(call.op)
                if spec_c and spec_c.tier != "read":
                    mutated_ops_names.append(call.op)
                attached_media.extend(_collect_media_attachments(call, res, sandbox))

            results.append({"index": call.index, "op": call.op, "ok": True, "result": res})
        except Exception as err:  # noqa: BLE001
            journal.rollback_files(before_files)
            if checkpoint is not None and transport is not None:
                with contextlib.suppress(Exception):
                    await journal.restore_blend_checkpoint(checkpoint, transport)
                checkpoint = None
            results.append({"index": call.index, "op": call.op, "ok": False, "error": str(err)})
            return BatchOutcome(
                status="apply-failed",
                results=results,
                errors=[{"index": call.index, "op": call.op, "message": str(err)}],
            )

    # --- Pass 2: Segregated approval-gated ops (`scene.clear`, `scene.open`, `bpy.eval`, `shell`) ---
    for call in gated_queue:
        decision = ApprovalDecision(allow=True)
        if not yolo:
            if approve is None:
                decision = ApprovalDecision(
                    allow=False,
                    reason="gated operation requires interactive approval (or --yolo)",
                )
            else:
                decision = await approve(call)

        if not decision.allow:
            msg = decision.reason or "denied by operator"
            results.append({"index": call.index, "op": call.op, "ok": False, "error": msg})
            return BatchOutcome(
                status="denied",
                results=results,
                errors=[{"index": call.index, "op": call.op, "message": msg}],
                checkpoint=checkpoint,
                attached_media=attached_media,
            )

        if (
            (decision.checkpoint_first or (auto_checkpoint and _needs_blend_checkpoint([call])))
            and checkpoint is None
            and transport is not None
        ):
            with contextlib.suppress(Exception):
                checkpoint = await journal.create_blend_checkpoint(transport)

        try:
            if call.op == "shell":
                sh_cwd = (
                    sandbox.resolve_for_read(str(call.params["cwd"]))
                    if call.params.get("cwd")
                    else sandbox.cwd
                )
                res = await _run_shell_cmd(str(call.params["cmd"]), sh_cwd, shell_timeout)
                ok = res["exit_code"] == 0
                results.append({"index": call.index, "op": call.op, "ok": ok, "result": res})
                if not ok:
                    return BatchOutcome(
                        status="apply-failed",
                        results=results,
                        errors=[
                            {
                                "index": call.index,
                                "op": "shell",
                                "message": f"command exited {res['exit_code']}",
                            }
                        ],
                        checkpoint=checkpoint,
                        attached_media=attached_media,
                    )
            else:
                assert transport is not None
                call_params = _normalize_blender_params(call, sandbox)
                res = await transport.call(
                    call.op, call_params, timeout=_op_timeout_for(call.op, call.params)
                )
                mutated_ops_names.append(call.op)
                attached_media.extend(_collect_media_attachments(call, res, sandbox))
                results.append({"index": call.index, "op": call.op, "ok": True, "result": res})
        except Exception as err:  # noqa: BLE001
            results.append({"index": call.index, "op": call.op, "ok": False, "error": str(err)})
            return BatchOutcome(
                status="apply-failed",
                results=results,
                errors=[{"index": call.index, "op": call.op, "message": str(err)}],
                checkpoint=checkpoint,
                attached_media=attached_media,
            )

    results.sort(key=lambda r: int(r["index"]))
    if checkpoint is not None or before_files:
        op_list_str = ", ".join(mutated_ops_names[:3]) + (
            f" (+{len(mutated_ops_names) - 3})" if len(mutated_ops_names) > 3 else ""
        )
        prefix = "Operator" if actor == "operator" else "Agent batch"
        summary_str = f"{prefix}: {op_list_str or 'edit'}"
        act_type = mutated_ops_names[0].split("(")[0] if len(mutated_ops_names) == 1 else "batch"
        journal.record_batch_action(
            actor=actor,
            action_type=act_type,
            summary=summary_str,
            checkpoint=checkpoint,
            before_files=before_files,
            after_files=after_files,
            details={"ops": mutated_ops_names},
        )

    return BatchOutcome(
        status="ok",
        results=results,
        checkpoint=checkpoint,
        attached_media=attached_media,
    )


def format_batch_results(outcome: BatchOutcome) -> str:
    lines: list[str] = ["```bl_results"]
    if outcome.status == "plan-failed":
        lines.append("BATCH REJECTED — nothing was applied. Fix and resubmit the whole batch.")
        for err in outcome.errors:
            lines.append(f"  [{err['index']}] {err['op']}: {err['message']}")
        lines.append("```")
        return "\n".join(lines)

    if outcome.checkpoint is not None:
        snap_name = Path(outcome.checkpoint.snapshot_path).name
        lines.append(f"[checkpoint] saved .blend snapshot '{snap_name}' (use /undo to restore)")

    for item in outcome.results:
        idx = item["index"]
        op = item["op"]
        if item.get("ok"):
            body = json.dumps(item.get("result"), ensure_ascii=False)
            lines.append(f"[{idx}] {op}: {body}")
        else:
            lines.append(f"[{idx}] {op}: ERROR {item.get('error')}")

    if outcome.status != "ok":
        for err in outcome.errors:
            lines.append(f"[{err['index']}] {err['op']}: FAILED — {err['message']}")

    lines.append("```")
    return "\n".join(lines)