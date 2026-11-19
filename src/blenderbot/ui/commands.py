"""Single-source-of-truth REPL & Web slash commands (`blenderbot.ui.commands`).

Parsed directly from `HELP_TEXT` -> `command_list()` so TUI tab completion and Web
command completion never drift.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from blenderbot.context.compact import compact_ledger
from blenderbot.context.media import load_image_part, load_video_part, youtube_ref
from blenderbot.session.bootstrap import Session
from blenderbot.session.flight import read_breadcrumbs
from blenderbot.session.keys_ctl import key_state, set_provider_key
from blenderbot.session.log import list_sessions
from blenderbot.session.models_ctl import apply_model_change
from blenderbot.session.profiles_ctl import profile_state, switch_profile

HELP_TEXT = """Commands
  sh <cmd>                    run a local shell command; output is added to context
  /blender [mode]             show Blender 5.3+ status or reconnect (`auto` | `bridge` | `headless`)
  /seedream [prompt] [flags]  open Seedream 5.5 Image Studio (or generate `--ref`/`--res`/`--aspect` PNG/JPG)
  /seedance [prompt] [flags]  open Seedance 2.5 GUI Studio (or generate `--video`/`--first`/`--last` MP4)
  /splat [list|import <id>]   open 3DGS Splat Studio GUI, list splats, or import `<job_id>` into Blender
  /fps [scene|job_id]         open dedicated full-screen Minecraft FPS Camera & 3DGS Splat Editor (`/fps`)
  /web [open]                 start/show the Seedance 2.5 & 3D Studio Web GUI sidecar
  /render [frame] [engine]    render still (`BLENDER_EEVEE` | `CYCLES` | `BLENDER_WORKBENCH`) & attach to context
  /sheet [cols] [rows]          render burned-in `fNNNN | t | lens` mosaic contact sheet & attach to context
  /turntable [target] [frames]  create an orbiting turntable camera + studio 3-point rig & render MP4 playblast
  /audit [obj1 obj2 ...]        run spatial & mesh topology/collision/clearance audit on the active scene
  /import <path> [collection] import GLB/GLTF/FBX/OBJ/USD/STL/PLY asset into the Blender scene
  /export <path> [objects...] export scene or named objects to GLB/GLTF/FBX/OBJ/USD/STL/PLY
  /save [filepath]            save the active Blender scene to a `.blend` file
  /open <filepath>            open a `.blend` file into the active Blender session (checkpoints first)
  /undo [id|steps]            undo last `.blend` / file action (or a specific `act-XXXX` entry)
  /redo [id|steps]            redo next undone action (or a specific `act-XXXX` entry)
  /actions [n]                list session Action History with undo/redo cursor status
  /role [name]                view or switch capability surface (`viewer` | `builder` | `master`)
  /model [id] [--save]        switch model or provider (`/model gemini-4-carbon`, `/model opus`, `/model gpt`)
  /effort <level>             set reasoning effort: low | medium | high | xhigh | max
  /image <path>               attach a workspace image to context
  /video <path>               attach a workspace video to context (Gemini inline <=18MB / Files API >18MB)
  /youtube <url> [minutes]    attach a YouTube video by reference (Gemini only)
  /context [drop|pin <id>]    show context segment breakdown, drop a segment, or toggle pin
  /tps                        show tok/s throughput and TTFT per turn
  /flight [n]                 show recent flight-recorder phase & stall breadcrumbs
  /artifacts [save <id> <p>]  list emitted code/script artifacts or save one
  /compact                    compact context ledger now
  /clear                      clear conversation history while keeping system grounding
  /prompts [sub]              inspect/edit prompts & protocol: export | diff | accept | reset | apply
  /keys [prov] [key]          inspect configured API keys or set one (`--save` persists to ~/.blenderbot/.env)
  /env [block|off]            list `#---#` .env provider blocks or switch active block
  /config                     display resolved configuration
  /sessions                   list recent sessions on disk
  /history [n]                show recent prompts across sessions
  /replay <n>                 re-run prompt #n from `/history` (e.g. `/replay -1`)
  /yolo [on|off]              toggle auto-approval for gated ops in this session
  /events                     show harness events emitted vs retired this session
  /help                       show this command reference
  /exit                       quit blenderbot

Keys
  esc / ctrl+c                cancel running turn (ctrl+c when idle exits)
  tab                         complete `/commands` and `@paths`"""

MODEL_SHORTCUTS: dict[str, tuple[str, str]] = {
    "opus": ("anthropic", "claude-opus-5-5"),
    "opus5": ("anthropic", "claude-opus-5"),
    "sonnet": ("anthropic", "claude-sonnet-5-5"),
    "haiku": ("anthropic", "claude-haiku-5-5"),
    "gemini": ("gemini", "gemini-4-carbon"),
    "carbon": ("gemini", "gemini-4-carbon"),
    "flash": ("gemini", "gemini-3.8-flash"),
    "argon": ("gemini", "gemini-4-argon"),
    "gpt": ("openai", "gpt-5.1"),
    "mini": ("openai", "gpt-5-mini"),
    "grok": ("grok", "grok-4"),
}


def command_list() -> list[dict[str, str]]:
    """Parse `HELP_TEXT` Commands section so REPL and Web completions never drift."""
    out: list[dict[str, str]] = []
    for line in HELP_TEXT.splitlines():
        if line and not line.startswith(" ") and out:
            break
        m = re.match(r"^  (\S.*?)\s{2,}(\S.*)$", line)
        if not m:
            continue
        usage = m.group(1).strip()
        name = usage.split()[0]
        if not name.startswith("/") and name != "sh":
            continue
        out.append({"name": name, "usage": usage, "desc": m.group(2).strip()})
    return out


def _infer_provider_for_model(model_id: str, current_provider: str) -> str:
    low = model_id.lower()
    if "/" in model_id:
        return "openrouter"
    if low.startswith("claude-"):
        return "anthropic"
    if low.startswith("gemini-"):
        return "gemini"
    if low.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if low.startswith("grok-"):
        return "grok"
    return current_provider


async def handle_command(
    raw_text: str,
    *,
    session: Session,
    push: Callable[[dict[str, Any]], None],
    exit_app: Callable[[], None],
) -> dict[str, Any]:
    """Execute a local REPL/Web command. Returns `{"handled": bool, "replay_input"?: str}`."""
    text = raw_text.strip()

    # --- `sh <cmd>` ---
    if text == "sh" or text.startswith("sh "):
        cmd_str = text[2:].strip()
        if not cmd_str:
            push({"type": "notice", "message": "usage: sh <command>"})
            return {"handled": True}
        push({"type": "user", "text": text})
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd_str,
            shell=True,
            cwd=str(session.cfg.cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=session.cfg.shell_timeout,
            check=False,
        )
        combined = ((proc.stdout or "") + (proc.stderr or "")).strip()
        session.ledger.add(
            seg_type="shell",
            label=f"$ {cmd_str[:40]} (exit {proc.returncode})",
            role="user",
            text=f"$ {cmd_str}\n(exit {proc.returncode})\n{combined[:12_000]}",
        )
        push(
            {
                "type": "shell",
                "cmd": cmd_str,
                "exit_code": proc.returncode,
                "output": combined[:8000],
            }
        )
        return {"handled": True}

    if not text.startswith("/"):
        return {"handled": False}

    parts = text.split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("/help", "/?"):
        push({"type": "info", "text": HELP_TEXT})
        return {"handled": True}

    if cmd == "/web":
        from blenderbot.server import ensure_web_sidecar

        want_open = not args or args[0].lower() not in ("status", "url")
        srv = await ensure_web_sidecar(session, open_browser=want_open)
        push(
            {
                "type": "notice",
                "message": f"Seedance 2.5 & 3D Studio running at {srv.auth_url}",
            }
        )
        return {"handled": True}

    if cmd == "/fps":
        import webbrowser as _wb
        from urllib.parse import quote as _quote

        from blenderbot.server import ensure_web_sidecar

        srv = await ensure_web_sidecar(session, open_browser=False)
        scene_q = f"&scene={_quote(args[0], safe='')}" if args else ""
        fps_url = f"{srv.url}/fps?token={_quote(srv.token, safe='')}{scene_q}"
        try:
            _wb.open(fps_url)
        except Exception:  # noqa: BLE001, S110
            pass
        push(
            {
                "type": "notice",
                "message": f"Opened Full-Screen FPS Camera & 3DGS Splat Editor -> {fps_url}",
            }
        )
        return {"handled": True}

    if cmd == "/splat":
        if not args or args[0].lower() in ("gui", "open", "studio"):
            from blenderbot.server import ensure_web_sidecar

            srv = await ensure_web_sidecar(session, open_browser=True)
            push(
                {
                    "type": "notice",
                    "message": f"Opened 3DGS Splat & Seedance Studio -> {srv.auth_url}#splat",
                }
            )
            return {"handled": True}

        sub = args[0].lower()
        splats_root = session.cfg.cwd / "assets" / "splats"
        if sub == "list":
            dirs = sorted(
                (d for d in splats_root.glob("*") if d.is_dir() and (d / "splat.ply").is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not dirs:
                push({"type": "info", "text": "No splat jobs in assets/splats/ yet."})
            else:
                lines = ["Saved 3DGS Splat Jobs (`assets/splats/`):"]
                for d in dirs[:20]:
                    sz_mb = (d / "splat.ply").stat().st_size / (1024 * 1024)
                    lines.append(f"  {d.name}  ({sz_mb:.1f} MB .ply)")
                push({"type": "info", "text": "\n".join(lines)})
            return {"handled": True}

        if sub == "import" and len(args) >= 2:
            if session.transport is None:
                await session.maybe_auto_connect_blender()
            if session.transport is None:
                push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
                return {"handled": True}
            jid = args[1]
            mode_r = args[2].lower() if len(args) >= 3 else "ellipsoids"
            jdir = session.sandbox.resolve_for_read(f"assets/splats/{jid}")
            ply_p = jdir / "splat.ply"
            cam_p = jdir / "cameras.json"
            imp_res = await session.transport.call(
                "splat.import",
                {
                    "filepath": str(ply_p),
                    "cameras_json": str(cam_p) if cam_p.is_file() else None,
                    "name": f"Splat_{jid[:20]}",
                    "render_mode": mode_r,
                    "max_points": 350_000,
                    "shadow_catcher_ground": False,
                },
                timeout=180.0,
            )
            push(
                {
                    "type": "notice",
                    "message": f"Imported {imp_res.get('object')} ({imp_res.get('points_imported', 0):,} {mode_r}) in {imp_res.get('elapsed_s')}s",
                }
            )
            return {"handled": True}

        push(
            {
                "type": "info",
                "text": "usage: /splat | /splat list | /splat import <job_id> [ellipsoids|pointcloud]",
            }
        )
        return {"handled": True}

    if cmd == "/seedream":
        if not args or (
            len(args) == 1 and args[0].lower() in ("gui", "open", "--gui", "--open", "studio")
        ):
            from blenderbot.server import ensure_web_sidecar

            srv = await ensure_web_sidecar(session, open_browser=False)
            import webbrowser as _wb_sr

            sr_url = f"{srv.auth_url}#seedream"
            with contextlib.suppress(Exception):
                _wb_sr.open(sr_url)
            push({"type": "notice", "message": f"Opened Seedream 5.5 Image Studio -> {sr_url}"})
            return {"handled": True}

        from blenderbot.media_gen.seedream import (
            default_seedream_output_path,
            generate_seedream_image,
        )

        or_key = session.cfg.keys.get("openrouter")
        if not or_key:
            push(
                {
                    "type": "warning",
                    "message": "OPENROUTER_API_KEY is not configured. Run `/keys openrouter <key> --save` or `/seedream` to set it in the GUI.",
                }
            )
            return {"handled": True}

        refs_sr: list[Path | str] = []
        res_sr: str | None = None
        ar_sr: str | None = None
        fmt_sr: str | None = None
        bg_sr: str | None = None
        mode_sr = "standard"
        seed_sr: int | None = None
        out_sr: str | None = None
        words_sr: list[str] = []
        i_sr = 0
        while i_sr < len(args):
            t_sr = args[i_sr]
            if t_sr in ("--ref", "-r") and i_sr + 1 < len(args):
                rv = args[i_sr + 1]
                refs_sr.append(
                    rv if rv.startswith(("https://", "http://", "data:")) else session.sandbox.resolve_for_read(rv)
                )
                i_sr += 2
            elif t_sr in ("--res", "--resolution") and i_sr + 1 < len(args):
                res_sr = args[i_sr + 1]
                i_sr += 2
            elif t_sr in ("--aspect", "--ar", "-a") and i_sr + 1 < len(args):
                ar_sr = args[i_sr + 1]
                i_sr += 2
            elif t_sr in ("--format", "--fmt") and i_sr + 1 < len(args):
                fmt_sr = args[i_sr + 1]
                i_sr += 2
            elif t_sr in ("--bg", "--background") and i_sr + 1 < len(args):
                bg_sr = args[i_sr + 1]
                i_sr += 2
            elif t_sr == "--mode" and i_sr + 1 < len(args):
                mode_sr = args[i_sr + 1]
                i_sr += 2
            elif t_sr == "--seed" and i_sr + 1 < len(args):
                seed_sr = int(args[i_sr + 1])
                i_sr += 2
            elif t_sr in ("--out", "-o") and i_sr + 1 < len(args):
                out_sr = args[i_sr + 1]
                i_sr += 2
            else:
                words_sr.append(t_sr)
                i_sr += 1

        prompt_sr = " ".join(words_sr).strip()
        try:
            ext_hint = ".jpg" if (fmt_sr or "").lower() in ("jpg", "jpeg") else ".png"
            out_p_sr = (
                session.sandbox.resolve_for_write(out_sr)
                if out_sr
                else default_seedream_output_path(session.cfg.cwd, ext=ext_hint)
            )
            push(
                {
                    "type": "notice",
                    "message": f"Generating Seedream 5.5 image ({session.cfg.image_model}) -> {session.sandbox.rel(out_p_sr)}…",
                }
            )
            res_img = await generate_seedream_image(
                api_key=or_key,
                output_path=out_p_sr,
                model=session.cfg.image_model,
                prompt=prompt_sr,
                mode=mode_sr,
                resolution=res_sr,
                aspect_ratio=ar_sr,
                output_format=fmt_sr,
                background=bg_sr,
                reference_images=refs_sr,
                seed=seed_sr,
                on_progress=lambda _p: session.log.flight.touch(),
            )
            final_p = Path(str(res_img["output_path"]))
            cost_sr = float(res_img.get("cost_usd") or 0.0)
            session.agent.run_cost += cost_sr
            session.agent.stats["cost"] += cost_sr
            session.log.record("seedream", res_img)
            if session.cfg.model.vision:
                img_part = load_image_part(final_p)
                session.ledger.add(
                    seg_type="media",
                    label=f"seedream {final_p.name} ({res_img.get('width')}x{res_img.get('height')})",
                    role="user",
                    image=img_part,
                )
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Seedream 5.5 completed in {res_img['elapsed_sec']}s (${cost_sr:.4f}, "
                        f"{res_img.get('width')}x{res_img.get('height')}) -> {session.sandbox.rel(final_p)}"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/seedream failed: {err}"})
        return {"handled": True}

    if cmd == "/seedance":
        # With no args (or `--gui` / `gui` / `open`), launch & open the interactive GUI Studio!
        if not args or args[0].lower() in ("gui", "open", "--gui", "--open", "studio"):
            from blenderbot.server import ensure_web_sidecar

            srv = await ensure_web_sidecar(session, open_browser=True)
            push(
                {
                    "type": "notice",
                    "message": f"Opened Seedance 2.5 GUI Studio -> {srv.auth_url}",
                }
            )
            return {"handled": True}

        # Direct CLI flag form: `/seedance <prompt...> [--video p.mp4] [--first f1.png] [--last f2.png] [--dur 5] [--size 1280x720] [--mode orbit_plate]`
        from blenderbot.media_gen.seedance import (
            default_seedance_output_path,
            generate_seedance_video,
        )

        or_key = session.cfg.keys.get("openrouter")
        if not or_key:
            push(
                {
                    "type": "warning",
                    "message": "OPENROUTER_API_KEY is not configured. Run `/keys openrouter <key> --save` or `/seedance` to set it in the GUI.",
                }
            )
            return {"handled": True}

        vid_arg: str | None = None
        first_arg: str | None = None
        last_arg: str | None = None
        audio_arg: str | None = None
        dur_arg: int | None = None
        size_arg: str | None = None
        mode_arg = "standard"
        out_arg: str | None = None
        prompt_words: list[str] = []
        idx_a = 0
        while idx_a < len(args):
            tok_a = args[idx_a]
            if tok_a in ("--video", "-v") and idx_a + 1 < len(args):
                vid_arg = args[idx_a + 1]
                idx_a += 2
            elif tok_a in ("--first", "-f") and idx_a + 1 < len(args):
                first_arg = args[idx_a + 1]
                idx_a += 2
            elif tok_a in ("--last", "-l") and idx_a + 1 < len(args):
                last_arg = args[idx_a + 1]
                idx_a += 2
            elif tok_a in ("--audio", "-a") and idx_a + 1 < len(args):
                audio_arg = args[idx_a + 1]
                idx_a += 2
            elif tok_a in ("--dur", "--duration", "-d") and idx_a + 1 < len(args):
                dur_arg = int(args[idx_a + 1])
                idx_a += 2
            elif tok_a in ("--size", "-s") and idx_a + 1 < len(args):
                size_arg = args[idx_a + 1]
                idx_a += 2
            elif tok_a == "--mode" and idx_a + 1 < len(args):
                mode_arg = args[idx_a + 1]
                idx_a += 2
            elif tok_a in ("--out", "-o") and idx_a + 1 < len(args):
                out_arg = args[idx_a + 1]
                idx_a += 2
            else:
                prompt_words.append(tok_a)
                idx_a += 1

        prompt_s = " ".join(prompt_words).strip()
        try:
            out_p = (
                session.sandbox.resolve_for_write(out_arg)
                if out_arg
                else default_seedance_output_path(session.cfg.cwd)
            )
            first_p = session.sandbox.resolve_for_read(first_arg) if first_arg else None
            last_p = session.sandbox.resolve_for_read(last_arg) if last_arg else None
            vid_p = session.sandbox.resolve_for_read(vid_arg) if vid_arg else None
            aud_p = session.sandbox.resolve_for_read(audio_arg) if audio_arg else None

            push(
                {
                    "type": "notice",
                    "message": f"Submitting Seedance 2.5 job ({session.cfg.video_model}) -> {session.sandbox.rel(out_p)}…",
                }
            )
            res_sd = await generate_seedance_video(
                api_key=or_key,
                output_path=out_p,
                model=session.cfg.video_model,
                prompt=prompt_s or None,
                mode=mode_arg,
                duration=dur_arg,
                size=size_arg,
                first_frame=first_p,
                last_frame=last_p,
                video_path=vid_p,
                audio_path=aud_p,
                on_progress=lambda _p: session.log.flight.touch(),
            )
            cost_sd = float(res_sd.get("cost_usd") or 0.0)
            session.agent.run_cost += cost_sd
            session.agent.stats["cost"] += cost_sd
            session.log.record("seedance", res_sd)
            if session.cfg.model.video:
                vp_part = load_video_part(out_p)
                session.ledger.add(
                    seg_type="media",
                    label=f"seedance {out_p.name} ({vp_part['delivery']})",
                    role="user",
                    video=vp_part,
                )
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Seedance 2.5 completed in {res_sd['elapsed_sec']}s (${cost_sd:.4f}) -> "
                        f"{session.sandbox.rel(out_p)}"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/seedance failed: {err}"})
        return {"handled": True}

    if cmd in ("/exit", "/quit", "/q"):
        exit_app()
        return {"handled": True}

    if cmd == "/clear":
        dropped = session.ledger.clear_conversation()
        session.bus.publish("cleared", {"dropped": dropped})
        push(
            {
                "type": "notice",
                "message": f"Cleared {dropped} conversation segments (grounding kept).",
            }
        )
        return {"handled": True}

    if cmd == "/yolo":
        if args and args[0].lower() in ("on", "true", "1"):
            want = True
        elif args and args[0].lower() in ("off", "false", "0"):
            want = False
        else:
            want = not session.cfg.yolo
        if want and session.cfg.web_enabled:
            from blenderbot.cli import UsageError, assert_safe_bind

            try:
                assert_safe_bind(host=session.cfg.web_host, yolo=True)
            except UsageError as err:
                push({"type": "warning", "message": str(err)})
                return {"handled": True}
        session.cfg.yolo = want
        session.bus.publish("yolo-changed", {"yolo": want})
        push(
            {
                "type": "notice",
                "message": f"yolo mode {'ENABLED' if want else 'disabled'} for this session.",
            }
        )
        return {"handled": True}

    if cmd == "/effort":
        if not args:
            push(
                {
                    "type": "info",
                    "text": f"current effort: {session.cfg.effort} (low | medium | high | xhigh | max)",
                }
            )
            return {"handled": True}
        try:
            res = apply_model_change(session, effort=args[0])
            push(
                {
                    "type": "notice",
                    "message": f"effort -> {res['effort']} ({res['provider']}/{res['model']['id']})",
                }
            )
        except ValueError as err:
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd == "/model":
        persist = "--save" in args
        clean_args = [a for a in args if a != "--save"]
        if not clean_args:
            m = session.cfg.model
            push(
                {
                    "type": "info",
                    "text": (
                        f"current: {session.cfg.provider} / {m.id} (effort={session.cfg.effort}, "
                        f"ctx={m.context:,}, vision={'yes' if m.vision else 'no'}, video={'yes' if m.video else 'no'})\n"
                        "shortcuts: /model carbon | /model flash | /model opus | /model sonnet | /model gpt | /model <provider>/<slug>"
                    ),
                }
            )
            return {"handled": True}
        target = clean_args[0]
        if target.lower() in MODEL_SHORTCUTS:
            prov, mid = MODEL_SHORTCUTS[target.lower()]
        elif ":" in target and not target.startswith("http"):
            prov, mid = target.split(":", 1)
        else:
            prov = _infer_provider_for_model(target, session.cfg.provider)
            mid = target
        try:
            res = apply_model_change(session, provider=prov, model_id=mid, persist=persist)
            msg = (
                f"switched -> {res['provider']} / {res['model']['id']} "
                f"(vision={'yes' if res['model']['vision'] else 'no'}, video={'yes' if res['model']['video'] else 'no'})"
            )
            if res.get("note"):
                msg += f"\n{res['note']}"
            push({"type": "notice", "message": msg})
        except ValueError as err:
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd == "/role":
        from blenderbot.agent.roles import ROLE_SPECS, ROLES
        from blenderbot.session.promptedit import apply_staged_prompts

        if not args:
            lines = ["Capability Roles (`/role <name>`):"]
            for r_key in ROLES:
                rs = ROLE_SPECS[r_key]
                cur = "  <-- active" if session.cfg.role == r_key else ""
                lines.append(f"  • {r_key:<10} — {rs.title}: {rs.summary}{cur}")
            push({"type": "info", "text": "\n".join(lines)})
            return {"handled": True}

        target_role = args[0].strip().lower()
        if target_role not in ROLES:
            push(
                {
                    "type": "warning",
                    "message": f"Unknown role '{target_role}'. Choose one of: {', '.join(ROLES)}",
                }
            )
            return {"handled": True}
        session.cfg.role = target_role
        reb = apply_staged_prompts(session)
        session.bus.publish("config-changed", {"role": target_role})
        rs = ROLE_SPECS[target_role]
        push(
            {
                "type": "notice",
                "message": (
                    f"Switched active role -> {target_role} ({rs.title}; "
                    f"system prompt {reb['after_tokens']:,} tok)"
                ),
            }
        )
        return {"handled": True}

    # --- Blender 5.3+ Domain Slash Commands ---
    if cmd == "/blender":
        mode_arg = args[0].lower() if args else None
        try:
            if mode_arg or session.transport is None:
                st = await session.reconnect_blender(mode_arg)
            else:
                st = await session.transport.call("blender.status", timeout=4.0)
            sc = st.get("scene") or {}
            cnt = st.get("counts") or {}
            res_xy = sc.get("resolution") or [1920, 1080]
            push(
                {
                    "type": "info",
                    "text": (
                        f"Connected [{st.get('mode')}] -> Blender {st.get('blender_version')} (PID {st.get('pid')})\n"
                        f"  file:     {st.get('filepath') or '(unsaved)'}\n"
                        f"  scene:    {sc.get('name')} | engine={sc.get('engine')} | {res_xy[0]}x{res_xy[1]} @ {sc.get('fps')} fps\n"
                        f"  frames:   {sc.get('frame_start')}..{sc.get('frame_end')} (current={sc.get('frame_current')})\n"
                        f"  camera:   {sc.get('active_camera') or '(none)'}\n"
                        f"  objects:  {cnt.get('objects', 0)} ({cnt.get('meshes', 0)} meshes, {cnt.get('lights', 0)} lights, {cnt.get('armatures', 0)} armatures)"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd == "/render":
        if session.transport is None:
            push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
            return {"handled": True}
        try:
            renders_dir = session.cfg.cwd / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)
            params: dict[str, Any] = {"output_path": str(renders_dir / "still_preview.png")}
            for a in args:
                if a.lstrip("-").isdigit():
                    params["frame"] = int(a)
                else:
                    params["engine"] = a.upper()
            op_name = (
                "viewport.render_still"
                if params.get("engine") == "BLENDER_WORKBENCH"
                else "render.still"
            )
            res = await session.transport.call(op_name, params, timeout=180.0)
            out_p = Path(str(res["output_path"]))
            session.sandbox.allow_read_file(out_p)
            tok_note = ""
            if session.cfg.model.vision and out_p.is_file():
                img = load_image_part(out_p)
                session.ledger.add(
                    seg_type="media",
                    label=f"render still f{res.get('frame')} ({img['bytes'] // 1024}kB)",
                    role="user",
                    image=img,
                )
                tok_note = f" & attached (~{img['tokens']} tokens)"
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Rendered frame {res.get('frame')} ({res.get('elapsed_ms')}ms) -> "
                        f"{session.sandbox.rel(out_p)}{tok_note}"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/render failed: {err}"})
        return {"handled": True}

    if cmd == "/sheet":
        if session.transport is None:
            push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
            return {"handled": True}
        try:
            cols = int(args[0]) if len(args) >= 1 and args[0].isdigit() else 4
            rows = int(args[1]) if len(args) >= 2 and args[1].isdigit() else 3
            renders_dir = session.cfg.cwd / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)
            out_p = renders_dir / f"contact_sheet_{cols}x{rows}.png"
            res = await session.transport.call(
                "viewport.contact_sheet",
                {"output_path": str(out_p), "cols": cols, "rows": rows},
                timeout=180.0,
            )
            session.sandbox.allow_read_file(out_p)
            tok_note = ""
            if session.cfg.model.vision and out_p.is_file():
                img = load_image_part(out_p)
                session.ledger.add(
                    seg_type="media",
                    label=f"contact sheet {cols}x{rows} ({img['bytes'] // 1024}kB)",
                    role="user",
                    image=img,
                )
                tok_note = f" & attached (~{img['tokens']} tokens)"
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Rendered {cols}x{rows} contact sheet ({res.get('elapsed_ms')}ms) -> "
                        f"{session.sandbox.rel(out_p)}{tok_note}"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/sheet failed: {err}"})
        return {"handled": True}

    if cmd == "/turntable":
        if session.transport is None:
            push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
            return {"handled": True}
        try:
            target_obj = args[0] if len(args) >= 1 and not args[0].isdigit() else None
            frames = (
                int(args[1])
                if len(args) >= 2 and args[1].isdigit()
                else (int(args[0]) if len(args) == 1 and args[0].isdigit() else 72)
            )
            cp = await session.journal.create_blend_checkpoint(session.transport)
            await session.transport.call("scene.setup", {"frame_start": 1, "frame_end": frames})
            await session.transport.call("light.rig", {"preset": "three_point", "distance": 6.0})
            cam_params: dict[str, Any] = {
                "name": "TurntableCam",
                "mode": "turntable",
                "radius": 5.5,
                "elevation": 2.2,
                "frame_start": 1,
                "frame_end": frames,
                "make_active": True,
            }
            if target_obj:
                cam_params["target"] = target_obj
            else:
                cam_params["target"] = [0.0, 0.0, 0.5]
            await session.transport.call("camera.upsert", cam_params)
            renders_dir = session.cfg.cwd / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)
            mp4_path = renders_dir / "turntable.mp4"
            pb = await session.transport.call(
                "render.playblast",
                {"output_path": str(mp4_path), "engine": "BLENDER_EEVEE"},
                timeout=300.0,
            )
            session.journal.record_batch_action(
                actor="operator",
                action_type="turntable",
                summary=f"Operator: /turntable ({frames} frames)",
                checkpoint=cp,
            )
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Rendered {frames}-frame EEVEE turntable ({pb.get('elapsed_sec')}s) -> "
                        f"{session.sandbox.rel(mp4_path)}"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/turntable failed: {err}"})
        return {"handled": True}

    if cmd == "/audit":
        if session.transport is None:
            push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
            return {"handled": True}
        try:
            params_a: dict[str, Any] = {"meshes": args if args else "*"}
            if len(args) >= 2:
                params_a["check_collisions"] = args
            rep = await session.transport.call("scene.audit", params_a, timeout=60.0)
            session.ledger.add(
                seg_type="op-result",
                label=f"scene.audit (passed={rep.get('passed')})",
                role="user",
                text=f"[Operator /audit report]\n{json.dumps(rep, indent=2)}",
            )
            push({"type": "info", "text": json.dumps(rep, indent=2)})
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/audit failed: {err}"})
        return {"handled": True}

    if cmd == "/import":
        if not args:
            push({"type": "notice", "message": "usage: /import <path> [collection]"})
            return {"handled": True}
        if session.transport is None:
            push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
            return {"handled": True}
        try:
            src_p = session.sandbox.resolve_for_read(args[0])
            cp = await session.journal.create_blend_checkpoint(session.transport)
            imp_params: dict[str, Any] = {"filepath": str(src_p)}
            if len(args) >= 2:
                imp_params["collection"] = args[1]
            res = await session.transport.call("asset.import", imp_params, timeout=120.0)
            session.journal.record_batch_action(
                actor="operator",
                action_type="asset.import",
                summary=f"Operator: /import {src_p.name}",
                checkpoint=cp,
            )
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Imported {src_p.name} ({len(res.get('imported_objects') or [])} objects)"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/import failed: {err}"})
        return {"handled": True}

    if cmd == "/export":
        if not args:
            push({"type": "notice", "message": "usage: /export <path> [obj1 obj2 ...]"})
            return {"handled": True}
        if session.transport is None:
            push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
            return {"handled": True}
        try:
            dst_p = session.sandbox.resolve_for_write(args[0])
            exp_params: dict[str, Any] = {"filepath": str(dst_p)}
            if len(args) >= 2:
                exp_params["objects"] = args[1:]
            res = await session.transport.call("asset.export", exp_params, timeout=120.0)
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Exported {res.get('format')} ({res.get('size_bytes', 0) // 1024} KB) -> "
                        f"{session.sandbox.rel(dst_p)}"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/export failed: {err}"})
        return {"handled": True}

    if cmd == "/save":
        if session.transport is None:
            push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
            return {"handled": True}
        try:
            sv_params: dict[str, Any] = {}
            if args:
                sv_params["filepath"] = str(session.sandbox.resolve_for_write(args[0]))
            res = await session.transport.call("scene.save", sv_params, timeout=60.0)
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Saved .blend ({res.get('size_bytes', 0) // 1024} KB) -> {res.get('filepath')}"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/save failed: {err}"})
        return {"handled": True}

    if cmd == "/open":
        if not args:
            push({"type": "notice", "message": "usage: /open <filepath.blend>"})
            return {"handled": True}
        if session.transport is None:
            push({"type": "warning", "message": "Blender is not connected (`/blender` first)."})
            return {"handled": True}
        try:
            blend_p = session.sandbox.resolve_for_read(args[0])
            cp = await session.journal.create_blend_checkpoint(session.transport)
            st = await session.transport.call(
                "scene.open", {"filepath": str(blend_p)}, timeout=60.0
            )
            session.journal.record_batch_action(
                actor="operator",
                action_type="scene.open",
                summary=f"Operator: /open {blend_p.name}",
                checkpoint=cp,
            )
            sc = st.get("scene") or {}
            cnt = st.get("counts") or {}
            push(
                {
                    "type": "notice",
                    "message": (
                        f"Opened {blend_p.name} -> scene '{sc.get('name')}' ({cnt.get('objects', 0)} objects)"
                    ),
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": f"/open failed: {err}"})
        return {"handled": True}

    # --- Journal Undo / Redo / History ---
    if cmd == "/undo":
        try:
            arg = args[0].strip() if args else ""
            if arg.lower().startswith("act-"):
                res = await session.journal.undo_entry(arg, session.transport)
            else:
                steps_n = int(arg) if arg.isdigit() else 1
                res = await session.journal.undo_last(session.transport, steps=steps_n)
            u = res.get("undone") or {}
            msg = f"Undid [{u.get('actor', 'session')}] {u.get('id', '')}: {u.get('summary', 'action')}"
            push({"type": "notice", "message": msg})
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd == "/redo":
        try:
            arg = args[0].strip() if args else ""
            if arg.lower().startswith("act-"):
                res = await session.journal.redo_entry(arg, session.transport)
            else:
                steps_n = int(arg) if arg.isdigit() else 1
                res = await session.journal.redo_next(session.transport, steps=steps_n)
            r_it = res.get("redone") or {}
            push(
                {
                    "type": "notice",
                    "message": f"Redid [{r_it.get('actor', 'session')}] {r_it.get('id', '')}: {r_it.get('summary', 'action')}",
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd == "/actions":
        st_j = session.journal.list_state()
        acts = st_j["actions"]
        if not acts:
            push(
                {
                    "type": "info",
                    "text": "No recorded operator or agent actions in this session yet.",
                }
            )
            return {"handled": True}
        lim = int(args[0]) if args and args[0].isdigit() else 30
        lines = [
            f"Action Timeline (cursor={st_j['cursor']}/{st_j['max_seq']}, applied={st_j['applied_count']}, undone={st_j['undone_count']}):"
        ]
        for a in acts[-lim:]:
            state_lbl = "SUP" if a.get("superseded") else ("ON " if a.get("applied") else "OFF")
            who = "OPR" if a.get("actor") == "operator" else "AGT"
            lines.append(f"  [{state_lbl}] {a['id']} ({who}) {a['summary']}")
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    # --- Media Attachments & Context Inspection ---
    if cmd == "/image":
        if not args:
            push({"type": "notice", "message": "usage: /image <path>"})
            return {"handled": True}
        try:
            img = load_image_part(args[0], session.sandbox)
            session.ledger.add(
                seg_type="media",
                label=f"image {img['name']}",
                role="user",
                image=img,
            )
            push({"type": "notice", "message": f"attached {img['name']} (~{img['tokens']} tokens)"})
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd == "/video":
        if not args:
            push({"type": "notice", "message": "usage: /video <path>"})
            return {"handled": True}
        if not session.cfg.model.video:
            push(
                {
                    "type": "warning",
                    "message": f"{session.cfg.model.id} does not support video — switch via `/model carbon` or `/model flash`.",
                }
            )
            return {"handled": True}
        try:
            vid = load_video_part(args[0], session.sandbox)
            session.ledger.add(
                seg_type="media",
                label=f"video {vid['name']} ({vid['delivery']})",
                role="user",
                video=vid,
            )
            push(
                {
                    "type": "notice",
                    "message": f"attached {vid['name']} via {vid['delivery']} (~{vid['tokens']:,} tokens)",
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd == "/youtube":
        if not args:
            push({"type": "notice", "message": "usage: /youtube <url> [minutes]"})
            return {"handled": True}
        if not session.cfg.model.video:
            push(
                {
                    "type": "warning",
                    "message": f"{session.cfg.model.id} does not support YouTube video — switch via `/model carbon`.",
                }
            )
            return {"handled": True}
        try:
            mins = float(args[1]) if len(args) >= 2 else None
            yt = youtube_ref(args[0], minutes=mins)
            session.ledger.add(
                seg_type="media", label=f"youtube {yt['uri']}", role="user", video=yt
            )
            push(
                {
                    "type": "notice",
                    "message": f"attached {yt['uri']} (~{yt['tokens']:,} tokens, assuming {yt['minutes']} min)",
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd == "/context":
        if len(args) >= 2 and args[0] == "drop" and args[1].isdigit():
            ok = session.ledger.drop(int(args[1]))
            push(
                {
                    "type": "notice" if ok else "warning",
                    "message": f"Segment #{args[1]} {'dropped' if ok else 'could not be dropped (pinned/protected or missing)'}.",
                }
            )
            return {"handled": True}
        if len(args) >= 2 and args[0] == "pin" and args[1].isdigit():
            pinned_now = session.ledger.toggle_pin(int(args[1]))
            push(
                {
                    "type": "notice",
                    "message": f"Segment #{args[1]} {'pinned' if pinned_now else 'unpinned'}.",
                }
            )
            return {"handled": True}

        lg = session.ledger
        pct = round(lg.usage_fraction * 100, 1)
        h = session.agent.context_health()
        lines = [
            f"Context: {lg.total_tokens:,} / {lg.limit:,} tok ({pct}% · compact_at={round(lg.compact_at * 100)}%)",
            (
                f"  [health] {h['status']} (score={h['score']}, action={h['recommended_action']}, "
                f"media={h['media_segments']}/{h['video_segments']}vid, "
                f"slips={h['totals']['malformed']}m/{h['totals']['op_failures']}f/{h['totals']['thought_leaks']}tl)"
            ),
            f"  [sys] system prompt + grounding: {lg.system_tokens:,} tok",
        ]
        for s in lg.segments:
            pin_m = "📌" if s.pinned else "  "
            lines.append(f"  {pin_m} #{s.id:<3} [{s.type:<9}] {s.tokens:>7,} tok  {s.label}")
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    if cmd == "/compact":
        try:
            res = await compact_ledger(session.ledger, force=True)
            session.agent.stats["compactions"] += 1
            session.agent._reset_rolling_health()
            push(
                {
                    "type": "notice",
                    "message": f"Compacted {res['dropped_count']} segments (freed ~{res['freed_tokens']:,} tokens).",
                }
            )
        except Exception as err:  # noqa: BLE001
            push({"type": "warning", "message": str(err)})
        return {"handled": True}

    if cmd in ("/tps", "/perf", "/speed"):
        snap = session.perf.snapshot()
        turns_list = snap.get("turns") or []
        if not turns_list:
            push({"type": "info", "text": "No completed turns recorded yet."})
            return {"handled": True}
        lines = [
            f"Throughput ({len(turns_list)} turns · avg {snap['avg_tps']:.1f} tok/s · peak {snap['peak_tps']:.1f} tok/s):"
        ]
        for t in turns_list[-15:]:
            ttft_s = f"{t['ttft']:.2f}s" if t.get("ttft") is not None else "—"
            lines.append(
                f"  #{t['turn']:<3} {t['model']:<22} {t['tps']:>6.1f} tok/s  ttft={ttft_s:<6}  out={t['out_tokens']} tok  ({t['elapsed']:.1f}s)"
            )
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    if cmd == "/flight":
        lim = int(args[0]) if args and args[0].isdigit() else 25
        rows_f = read_breadcrumbs(session.log.dir, limit=lim)
        if not rows_f:
            push({"type": "info", "text": "No flight breadcrumbs recorded yet."})
            return {"handled": True}
        lines = [f"Flight Breadcrumbs ({session.log.dir.name}, last {len(rows_f)}):"]
        for b in rows_f:
            kind = b.get("kind", "")
            ph = b.get("phase", "")
            ts_short = str(b.get("ts", ""))[11:23]
            if kind == "stall-heartbeat":
                lines.append(
                    f"  {ts_short}  [STALL] {ph}  idle={b.get('idle_ms')}ms  elapsed={b.get('elapsed_ms')}ms"
                )
            elif kind == "phase":
                prev_s = f" (from {b.get('prev')} in {b.get('prev_ms')}ms)" if b.get("prev") else ""
                lines.append(f"  {ts_short}  [PHASE] {ph}{prev_s}")
            else:
                lines.append(f"  {ts_short}  [{kind.upper()}] {ph} {b.get('event', '')}")
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    if cmd == "/artifacts":
        if len(args) >= 3 and args[0] == "save" and args[1].isdigit():
            try:
                art = session.artifacts.get(int(args[1]))
                if art is None:
                    push({"type": "warning", "message": f"Artifact #{args[1]} not found."})
                    return {"handled": True}
                dst = session.sandbox.resolve_for_write(args[2])
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(art.body, encoding="utf-8")
                push(
                    {
                        "type": "notice",
                        "message": f"Saved artifact #{art.id} ({art.lines} lines) -> {session.sandbox.rel(dst)}",
                    }
                )
            except Exception as err:  # noqa: BLE001
                push({"type": "warning", "message": str(err)})
            return {"handled": True}

        items = session.artifacts.list_meta()
        if not items:
            push({"type": "info", "text": "No code/script artifacts captured yet."})
            return {"handled": True}
        lines = ["Artifacts (`/artifacts save <id> <path>`):"]
        for it in items:
            lines.append(
                f"  #{it['id']:<3} [{it['lang'] or 'text':<8}] {it['lines']:>4} lines  (turn #{it['turn']})  {it['preview']}"
            )
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    if cmd == "/keys":
        persist = "--save" in args
        clean_k = [a for a in args if a != "--save"]
        if len(clean_k) >= 2:
            try:
                st_k = set_provider_key(session, clean_k[0], clean_k[1], persist=persist)
                where = "saved to ~/.blenderbot/.env" if persist else "set for this session"
                push({"type": "notice", "message": f"API key for {clean_k[0]} {where}."})
                _ = st_k
            except Exception as err:  # noqa: BLE001
                push({"type": "warning", "message": str(err)})
            return {"handled": True}
        st_k = key_state(session)
        lines = ["API Keys (`/keys <provider> <key> [--save]`):"]
        for row in st_k["providers"]:
            mark = "✓" if row["configured"] else "—"
            src = f" ({row['var']} from {row['source']})" if row["configured"] else ""
            lines.append(f"  {mark} {row['provider']:<12}{src}")
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    if cmd == "/env":
        if args:
            target_blk = None if args[0].lower() in ("off", "none") else args[0]
            try:
                res_p = switch_profile(session, target_blk)
                push(
                    {
                        "type": "notice",
                        "message": f"Active .env block -> {res_p['active'] or 'none'} ({res_p['provider']}/{res_p['model']})",
                    }
                )
            except Exception as err:  # noqa: BLE001
                push({"type": "warning", "message": str(err)})
            return {"handled": True}

        st = profile_state(session)
        if not st["blocks"]:
            push({"type": "info", "text": f"No `#---#` provider blocks in {st['file']}"})
            return {"handled": True}
        rows = [f"Provider blocks in {st['file']} (`/env <name>` or `/env off`):"]
        for b in st["blocks"]:
            mark = "●" if b["enabled"] else "○"
            rows.append(f"  {mark} {b['name']} {b['note']}")
        push({"type": "info", "text": "\n".join(rows)})
        return {"handled": True}

    if cmd == "/prompts":
        from blenderbot.agent.prompt import PROMPT_SPECS, builtin_prompts, prompt_dirs
        from blenderbot.agent.promptfork import (
            check_forks,
            hash_text,
            snapshot_prompt_file,
            stamp_fork,
        )
        from blenderbot.session.promptedit import export_prompts_to_project, stage_prompt_apply

        dirs = prompt_dirs(session.cfg.home, session.cfg.cwd)
        forks = check_forks(sources=session.prompt_sources, layers=session.prompt_layers)
        sub = args[0].lower() if args else ""
        flags = args[1:]

        if not sub:
            def _drift_of(k: str) -> str:
                if any(s["key"] == k for s in forks["stale"]):
                    return "STALE"
                if any(u["key"] == k for u in forks["unstamped"]):
                    return "unstamped"
                return "ok"

            lines = ["Prompts (highest wins: built-in -> home -> project):"]
            for s in PROMPT_SPECS:
                lyr = session.prompt_layers.get(s.key, "default")
                where = "built-in" if lyr == "default" else session.prompt_sources.get(s.key, "")
                lines.append(f"  {s.key:<12} {lyr:<9} {_drift_of(s.key):<10} {where}")
            lines.extend(
                [
                    "",
                    f"  home     {dirs['home']}",
                    f"  project  {dirs['project']}",
                    "",
                    "  /prompts export [--force]   write resolved prompts (incl. protocol.md) to ./.prompts/",
                    "  /prompts apply              re-read disk prompts & rebuild system prefix",
                    "  /prompts diff <key>         compare your override against current built-in",
                    "  /prompts accept <key|all>   re-baseline override against current built-in",
                    "  /prompts reset <key>        remove override and revert to built-in",
                ]
            )
            push({"type": "info", "text": "\n".join(lines)})
            return {"handled": True}

        if sub == "export":
            force = "--force" in flags
            try:
                exp = export_prompts_to_project(session, force=force)
                push(
                    {
                        "type": "notice",
                        "message": f"Exported {exp['count']} prompts to {exp['dir']}/ — edit any `.md` file then run `/prompts apply`.",
                    }
                )
            except Exception as err:  # noqa: BLE001
                push({"type": "warning", "message": str(err)})
            return {"handled": True}

        if sub == "apply":
            try:
                res = stage_prompt_apply(session)
                if res.get("applied"):
                    push(
                        {
                            "type": "notice",
                            "message": f"Rebuilt system prompt ({res['before_tokens']:,} -> {res['after_tokens']:,} tok; prompt cache reset).",
                        }
                    )
                else:
                    push({"type": "notice", "message": "Staged prompt rebuild for next step boundary."})
            except Exception as err:  # noqa: BLE001
                push({"type": "warning", "message": str(err)})
            return {"handled": True}

        if sub in ("diff", "accept", "reset"):
            known = [s.key for s in PROMPT_SPECS]
            key = flags[0] if flags else ""
            if sub == "accept" and key == "all":
                drifted = [*forks["stale"], *forks["unstamped"]]
                if not drifted:
                    push({"type": "notice", "message": "Every override is already current."})
                    return {"handled": True}
                by_dir: dict[str, list[str]] = {}
                for f in drifted:
                    by_dir.setdefault(f["dir"], []).append(f["key"])
                for d_str, ks in by_dir.items():
                    stamp_fork(Path(d_str), ks)
                push({"type": "notice", "message": f"Re-baselined {', '.join(f['key'] for f in drifted)}."})
                return {"handled": True}

            if key not in known:
                push({"type": "warning", "message": f"usage: /prompts {sub} <{'|'.join(known)}>"})
                return {"handled": True}
            if session.prompt_layers.get(key, "default") == "default":
                push({"type": "notice", "message": f"{key} is using the built-in default (not overridden)."})
                return {"handled": True}

            file_str = session.prompt_sources[key]
            defaults = builtin_prompts()
            if sub == "diff":
                push(
                    {
                        "type": "info",
                        "text": (
                            f"{key} ({file_str}, built-in hash {hash_text(defaults[key])})\n\n"
                            f"--- current built-in ---\n{defaults[key]}\n\n"
                            f"--- yours ---\n{session.prompts.get(key, '')}"
                        ),
                    }
                )
            elif sub == "accept":
                stamp_fork(Path(file_str).parent, [key])
                push({"type": "notice", "message": f"{key} re-baselined against current built-in."})
            elif sub == "reset":
                snap = snapshot_prompt_file(Path(file_str))
                Path(file_str).unlink(missing_ok=True)
                stage_prompt_apply(session)
                push({"type": "notice", "message": f"Reset {key} to built-in (backup: {snap})."})
            return {"handled": True}

        push({"type": "warning", "message": "usage: /prompts [export [--force] | apply | diff <key> | accept <key|all> | reset <key>]"})
        return {"handled": True}

    if cmd == "/config":
        push({"type": "info", "text": json.dumps(session.cfg.to_public_dict(), indent=2)})
        return {"handled": True}

    if cmd == "/sessions":
        rows_meta = list_sessions(Path(session.cfg.home), limit=15)
        if not rows_meta:
            push({"type": "info", "text": "No past sessions yet."})
            return {"handled": True}
        lines = ["Recent Sessions (`blenderbot --resume <id>`):"]
        for s in rows_meta:
            t = s.get("totals") or {}
            h_status = (s.get("health") or {}).get("status", "healthy")
            lines.append(
                f"  {s.get('id')}  {s.get('started_at')}  {s.get('model')}  "
                f"{t.get('turns', 0)}t  ${float(t.get('cost', 0.0)):.4f}  [{h_status}]"
            )
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    if cmd == "/history":
        limit_n = int(args[0]) if args and args[0].isdigit() else 15
        entries = session.history.recent(limit_n)
        if not entries:
            push({"type": "info", "text": "No prompt history yet."})
            return {"handled": True}
        lines = ["Recent Prompts (`/replay <index>`):"]
        base_idx = len(session.history.entries) - len(entries)
        for offset, e in enumerate(entries):
            lines.append(f"  [{base_idx + offset}] {e.get('text', '')[:90]}")
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    if cmd == "/replay":
        if not args:
            push({"type": "notice", "message": "usage: /replay <index> (e.g. /replay -1)"})
            return {"handled": True}
        try:
            idx = int(args[0])
            prompt_str = session.history.entries[idx]["text"]
            return {"handled": False, "replay_input": str(prompt_str)}
        except (ValueError, IndexError):
            push({"type": "warning", "message": f"Invalid history index: {args[0]}"})
            return {"handled": True}

    if cmd == "/events":
        lines = ["Harness Events (emitted / unretired):"]
        for ev_name, count in sorted(session.agent.event_counts.items()):
            unret = session.agent.unretired_events.get(ev_name, 0)
            lines.append(f"  {ev_name:<18} emitted={count:<4} unretired={unret}")
        push({"type": "info", "text": "\n".join(lines)})
        return {"handled": True}

    push({"type": "warning", "message": f"Unknown command: {cmd} (type /help for commands)"})
    return {"handled": True}