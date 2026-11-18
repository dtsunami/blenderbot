"""Activity-ranked workspace grounding + upfront & ephemeral Blender 5.3+ Scene Grounding.

1. `snapshot_blender_grounding`: Captures initial Blender version, mode (`gui` vs `headless`),
   render engine, resolution, FPS, frame range, collections, cameras, lights, armatures, and
   scene objects into the byte-stable cached system prompt.
2. `format_live_scene_state`: Ephemerally attaches a compact `[Live Scene State]` summary
   ONLY to `messages[-1]` on outgoing turns so prompt-prefix caching stays 100% intact while
   the model always sees current object names, world locations/sizes, materials, modifiers,
   active camera, and animation ranges without wasting turns on `scene.passport`!
"""

from __future__ import annotations

import datetime
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from blenderbot.config.env import Config
from blenderbot.transport.base import BlenderTransport

from .agents_md import render_agents, resolve_agents
from .prompt import build_system_prompt
from .sandbox import Sandbox

MAX_TOP_ENTRIES = 40
MAX_PREVIEW_DIRS = 6
MAX_PREVIEW_FILES = 6
MAX_GROUNDING_OBJECTS = 45


def _ago(mtime_sec: float) -> str:
    s = max(0.0, time.time() - mtime_sec)
    if s < 90:
        return "just now"
    if s < 5400:
        return f"{round(s / 60)}m ago"
    if s < 129600:
        return f"{round(s / 3600)}h ago"
    return f"{round(s / 86400)}d ago"


def _activity_score(dir_path: Path, mtime_sec: float) -> float:
    try:
        n = sum(1 for _ in dir_path.iterdir())
    except OSError:
        n = 0
    age_days = max(0.0, (time.time() - mtime_sec) / 86400.0)
    return math.log1p(n) * 2.0 + 10.0 / (1.0 + age_days)


def _preview_dir(dir_path: Path, sandbox: Sandbox) -> list[str]:
    kept: list[tuple[str, float]] = []
    try:
        for child in dir_path.iterdir():
            if sandbox.is_ignored(child):
                continue
            try:
                mt = child.stat().st_mtime
            except OSError:
                continue
            suffix = "/" if child.is_dir() else ""
            kept.append((f"{child.name}{suffix}", mt))
    except OSError:
        return []
    kept.sort(key=lambda x: x[1], reverse=True)
    out = [name for name, _ in kept[:MAX_PREVIEW_FILES]]
    if len(kept) > MAX_PREVIEW_FILES:
        out.append(f"… {len(kept) - MAX_PREVIEW_FILES} more")
    return out


def _activity_listing(root: Path, sandbox: Sandbox) -> str:
    dirs: list[tuple[str, Path, float, float]] = []
    files: list[tuple[str, float]] = []
    try:
        for child in root.iterdir():
            if sandbox.is_ignored(child):
                continue
            try:
                mt = child.stat().st_mtime
            except OSError:
                continue
            if child.is_dir():
                dirs.append((child.name, child, mt, _activity_score(child, mt)))
            else:
                files.append((child.name, mt))
    except OSError as err:
        return f"  (cannot read: {err})"

    dirs.sort(key=lambda d: d[3], reverse=True)
    files.sort(key=lambda f: f[1], reverse=True)

    out: list[str] = []
    for name, mt in files[:MAX_TOP_ENTRIES]:
        out.append(f"  {name}  ({_ago(mt)})")
    if len(files) > MAX_TOP_ENTRIES:
        out.append(f"  … {len(files) - MAX_TOP_ENTRIES} more files")

    for name, _p, mt, _sc in dirs:
        out.append(f"  {name}/  ({_ago(mt)})")

    for name, p, _mt, _sc in dirs[:MAX_PREVIEW_DIRS]:
        prev = _preview_dir(p, sandbox)
        if prev:
            out.append("")
            out.append(f"  {name}/")
            for ln in prev:
                out.append(f"    {ln}")
    return "\n".join(out)


def _git_info(root: Path) -> dict[str, Any] | None:
    def _run(args: list[str]) -> str:
        res = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return res.stdout.strip()

    try:
        branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
        status = _run(["status", "--porcelain"])
        dirty = len([ln for ln in status.splitlines() if ln.strip()])
        try:
            recent = [
                ln
                for ln in _run(["log", "-3", "--oneline", "--no-decorate"]).splitlines()
                if ln
            ]
        except (OSError, subprocess.SubprocessError):
            recent = []
        return {"branch": branch, "dirty": dirty, "recent": recent}
    except (OSError, subprocess.SubprocessError):
        return None


def _format_object_line(obj: dict[str, Any]) -> str:
    loc = obj.get("location") or [0, 0, 0]
    sz = obj.get("size") or [0, 0, 0]
    parts = [
        f'{obj["name"]} ({obj["type"]})',
        f"loc={loc}",
        f"size={sz}",
    ]
    if obj.get("verts") is not None:
        parts.append(f"v={obj['verts']}/p={obj.get('polys', 0)}")
    if obj.get("bones") is not None:
        parts.append(f"bones={obj['bones']}")
    if obj.get("materials"):
        parts.append(f"mat={obj['materials']}")
    if obj.get("modifiers"):
        parts.append(f"mod={obj['modifiers']}")
    if obj.get("parent"):
        parts.append(f"parent={obj['parent']}")
    if obj.get("animation"):
        spans = [
            f"{fc['data_path']}[{fc['frame_range'][0]}..{fc['frame_range'][1]}]"
            for fc in obj["animation"][:4]
        ]
        parts.append(f"anim={','.join(spans)}")
    return " | ".join(parts)


async def snapshot_blender_grounding(transport: BlenderTransport | None) -> str:
    """Gather a compact initial snapshot of the connected Blender 5.x session for Grounding."""
    if transport is None:
        return "## Blender Session\n  Status: disconnected (run `/blender` to connect)"

    try:
        st = await transport.call("blender.status", timeout=4.0)
        pp = await transport.call(
            "scene.passport", {"max_objects": MAX_GROUNDING_OBJECTS}, timeout=4.0
        )
    except Exception as err:  # noqa: BLE001
        return f"## Blender Session\n  Status: unreachable ({err})"

    res_xy = pp.get("resolution") or [1920, 1080]
    fr = pp.get("frame_range") or [1, 250]
    lines = [
        "## Blender Session (Snapshot at Session Start)",
        (
            f"  Blender: {st.get('blender_version')} | transport={st.get('mode')} | "
            f"file={pp.get('filepath')}"
        ),
        (
            f"  Scene: \"{pp.get('scene')}\" | engine={pp.get('engine')} | "
            f"{res_xy[0]}x{res_xy[1]} @ {pp.get('fps')} fps | "
            f"frames={fr[0]}..{fr[1]} (cur={pp.get('frame_current')}, {pp.get('duration_sec')}s) | "
            f"active_camera={pp.get('active_camera') or 'none'}"
        ),
    ]

    cols = [f"{c['name']}({c['objects']})" for c in (pp.get("collections") or [])]
    if cols:
        lines.append(f"  Collections: {', '.join(cols)}")

    objs = pp.get("objects") or []
    lines.append(f"  Scene Objects ({pp.get('total_scene_objects', len(objs))} total):")
    for o in objs:
        lines.append(f"    - {_format_object_line(o)}")
    if pp.get("omitted_objects"):
        lines.append(f"    … +{pp['omitted_objects']} more objects (use `scene.passport`)")

    return "\n".join(lines)


def build_grounding_text(
    *,
    cfg: Config,
    sandbox: Sandbox,
    blender_section: str = "",
) -> tuple[str, list[dict[str, Any]], list[str]]:
    root = sandbox.cwd
    shell_name = os.environ.get("COMSPEC", "cmd.exe") if os.name == "nt" else "/bin/sh"
    lines: list[str] = [
        "# Project grounding",
        "",
        f"Working directory: {root}",
        f"Platform: {platform.system()} ({platform.machine()}), shell: {shell_name}",
        f"Date: {datetime.date.today().isoformat()}",
    ]

    if cfg.ro_dirs:
        lines.append("")
        lines.append("Readable outside working directory (BLENDERBOT_RO_DIRS), NOT writable:")
        for d in cfg.ro_dirs:
            lines.append(f"  {d}")

    git = _git_info(root)
    if git:
        lines.append("")
        lines.append(f"Git: branch {git['branch']}, {git['dirty']} uncommitted change(s)")
        if git["recent"]:
            lines.append("Recent commits:")
            for c in git["recent"]:
                lines.append(f"  {c}")

    if blender_section:
        lines.extend(["", blender_section])

    lines.extend(["", "## Workspace Directory", _activity_listing(root, sandbox)])

    sections, notices = resolve_agents(
        home=cfg.home,
        cwd=root,
        explicit_files=getattr(cfg, "agents_files", None),
        agents_budget=getattr(cfg, "agents_budget", None),
        context_limit=cfg.context_limit,
    )
    lines.extend(render_agents(sections))
    return "\n".join(lines), sections, notices


def build_static_system_prompt(
    *,
    prompts: dict[str, str],
    home: Path,
    cwd: Path,
    cfg: Config | None = None,
    sandbox: Sandbox | None = None,
    blender_section: str = "",
) -> str:
    role = cfg.role if cfg is not None else "master"
    if cfg is not None and sandbox is not None:
        g_text, _, _ = build_grounding_text(
            cfg=cfg, sandbox=sandbox, blender_section=blender_section
        )
    else:
        sb = sandbox or Sandbox(cwd)
        sections, _ = resolve_agents(home=home, cwd=cwd)
        parts = [
            "# Project grounding",
            f"Working directory: {cwd}",
            f"Platform: {platform.system()} ({platform.machine()})",
            f"Date: {datetime.date.today().isoformat()}",
        ]
        if blender_section:
            parts.extend(["", blender_section])
        parts.extend(
            ["", "## Workspace Directory", _activity_listing(cwd, sb), *render_agents(sections)]
        )
        g_text = "\n".join(parts)

    return build_system_prompt(prompts=prompts, grounding=g_text, role=role)


async def poll_live_blender_banner(transport: BlenderTransport | None) -> str | None:
    """Lightweight poll of Blender session summary prepended to user turns ONLY when changed."""
    if transport is None:
        return "[Blender State: disconnected — run `/blender` to connect]"
    try:
        st = await transport.call("blender.status", timeout=2.5)
    except Exception:  # noqa: BLE001
        return "[Blender State: unreachable]"

    sc = st.get("scene") or {}
    cnt = st.get("counts") or {}
    res_xy = sc.get("resolution") or [1920, 1080]
    return (
        f"[Blender Live: mode={st.get('mode')} | v={st.get('blender_version')} | "
        f"file=\"{st.get('filepath') or 'unsaved'}\" | scene=\"{sc.get('name')}\" | "
        f"engine={sc.get('engine')} | {res_xy[0]}x{res_xy[1]}@{sc.get('fps')}fps | "
        f"frames={sc.get('frame_start')}..{sc.get('frame_end')} (cur={sc.get('frame_current')}) | "
        f"cam={sc.get('active_camera') or 'none'} | objs={cnt.get('objects', 0)}]"
    )


async def format_live_scene_state(transport: BlenderTransport | None) -> str | None:
    """Build a compact (~250-500 tok) ephemeral summary of the live Blender scene attached ONLY to
    `messages[-1]` on outgoing turns.

    Preserves 100% prompt-prefix caching on historical Ledger segments while giving the model
    authoritative current object names, world bounds, materials, modifiers, cameras, lights, and
    animation ranges on every step!
    """
    if transport is None:
        return None
    try:
        pp = await transport.call("scene.passport", {"max_objects": 40}, timeout=2.5)
    except Exception:  # noqa: BLE001
        return None

    res_xy = pp.get("resolution") or [1920, 1080]
    fr = pp.get("frame_range") or [1, 250]
    lines = [
        "[Live Scene State — authoritative current Blender objects, world AABB sizes & animation spans; do NOT call `scene.passport` or `blender.status` unless >40 objects are omitted]",
        (
            f"Scene \"{pp.get('scene')}\" ({pp.get('filepath')}) | engine={pp.get('engine')} | "
            f"{res_xy[0]}x{res_xy[1]}@{pp.get('fps')}fps | frames={fr[0]}..{fr[1]} (cur={pp.get('frame_current')}) | "
            f"active_camera={pp.get('active_camera') or 'none'} | film_transparent={pp.get('film_transparent')}"
        ),
    ]
    objs = pp.get("objects") or []
    if not objs:
        lines.append("  Objects: (empty scene)")
    else:
        for o in objs:
            lines.append(f"  • {_format_object_line(o)}")
        if pp.get("omitted_objects"):
            lines.append(f"  • … +{pp['omitted_objects']} more objects")
    return "\n".join(lines)