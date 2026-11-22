"""blenderbot CLI entry point (`blenderbot.cli`).

Modeled on `davai/bin/davai.js` and `davag/src/davag/cli.py`:
- `parse_args(argv)` raises `UsageError` on bad input (exits 2 in `run()`).
- `--host` or `--port` implies `--web`; binding off-loopback with `--yolo` is refused.
- Supports interactive Rich/prompt_toolkit TUI REPL, `-p`/`--print`, `--json` NDJSON,
  piped stdin, `--resume`, `--sessions`, `--models`, `--doctor`, `--install-addon`, and `--config`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .config import (
    Config,
    discover_blender,
    load_config,
    load_settings,
    resolve_home,
    settings_to_defaults,
)

HELP = f"""blenderbot {__version__} — General Blender 5.3+ 3D Creation, Modeling, Shading, Rigging & Animation Agent

USAGE
  blenderbot                         start the interactive TUI REPL
  blenderbot "model a sci-fi crate"  start REPL (or one-shot with -p) with initial prompt
  echo "audit scene" | blenderbot    read the request from stdin (headless)
  blenderbot -p "turntable render"   one-shot plain-text headless run
  blenderbot --resume [id]           reopen a past session (default: most recent)

OPTIONS
  -p, --print                one-shot, stream plain text to stdout (no TUI)
      --json                 one-shot, emit newline-delimited JSON events
  -y, --yes                  auto-approve gated ops (headless only)
      --yolo                 auto-approve gated ops in the REPL too (/yolo toggles)
  -m, --model <id>           override the model for this run
      --provider <name>      anthropic | gemini | openai | grok | openrouter
      --effort <level>       low | medium | high | xhigh | max
      --role <role>          viewer | builder | master
      --transport <mode>     auto | bridge | headless
  -C, --cwd <dir>            run against a different working directory
      --web                  enable Web Studio config (Phase 4 sidecar; default 127.0.0.1:7880)
      --open                 open tokenized Web URL in browser on launch (implies --web)
      --port <n>             port for --web (implies --web; 0 picks a free port)
      --host <addr>          bind address for --web (implies --web; refuses --yolo off-loopback)
      --doctor               inspect Blender 5.3+, FFmpeg, addon status, live bridge, and API keys
      --install-addon        install blenderbot_bridge into Blender 5.x user extensions
      --models [provider]    list models the provider serves
      --sessions             list recent sessions
      --resume [id]          rebuild context from a past session; id may be a prefix
      --config               show resolved configuration (keys redacted)
      --accept-prompts       start even if a prompt override was forked from an older blenderbot
  -h, --help                 show help
  -v, --version              show version

CONFIG
  Reads $BLENDERBOT_HOME/settings.json, ~/.davai/.env & ~/.davag/.env (automatic key fallback),
  $BLENDERBOT_HOME/.env, ./.env, then OS environment. BLENDERBOT_HOME defaults to ~/.blenderbot
"""


class UsageError(Exception):
    """Raised on invalid CLI invocation; maps to exit code 2."""


@dataclass
class ParsedArgs:
    help: bool = False
    version: bool = False
    print_mode: bool = False
    json_mode: bool = False
    yes: bool = False
    accept_stale_prompts: bool = False
    cwd: str | None = None
    web: bool = False
    port: int | None = None
    host: str | None = None
    doctor: bool = False
    install_addon: bool = False
    list_models: bool = False
    list_sessions: bool = False
    resume: str | bool = False
    show_config: bool = False
    overrides: dict[str, str] = field(default_factory=dict)
    words: list[str] = field(default_factory=list)
    input_text: str = ""


def is_loopback_host(host: str) -> bool:
    h = host.strip().lower()
    if h == "localhost":
        return True
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def assert_safe_bind(*, host: str, yolo: bool) -> None:
    if yolo and not is_loopback_host(host):
        raise UsageError(
            f'Refusing to combine --host "{host}" with --yolo: binding off-loopback while '
            "auto-approving shell/eval ops exposes unauthenticated remote execution."
        )


def parse_args(argv: list[str]) -> ParsedArgs:
    out = ParsedArgs()
    i = 0
    n = len(argv)

    def next_val(flag: str) -> str:
        nonlocal i
        i += 1
        if i >= n:
            raise UsageError(f"{flag} requires a value")
        return argv[i]

    while i < n:
        a = argv[i]
        if a in ("-h", "--help"):
            out.help = True
        elif a in ("-v", "--version"):
            out.version = True
        elif a in ("-p", "--print"):
            out.print_mode = True
        elif a == "--json":
            out.json_mode = True
            out.print_mode = True
        elif a in ("-y", "--yes"):
            out.yes = True
        elif a == "--accept-prompts":
            out.accept_stale_prompts = True
        elif a == "--yolo":
            out.overrides["BLENDERBOT_YOLO"] = "true"
            out.yes = True
        elif a in ("-m", "--model"):
            out.overrides["BLENDERBOT_MODEL"] = next_val(a)
        elif a == "--provider":
            out.overrides["BLENDERBOT_PROVIDER"] = next_val(a)
        elif a == "--effort":
            out.overrides["BLENDERBOT_EFFORT"] = next_val(a)
        elif a == "--role":
            out.overrides["BLENDERBOT_ROLE"] = next_val(a)
        elif a == "--transport":
            out.overrides["BLENDERBOT_TRANSPORT"] = next_val(a)
        elif a in ("-C", "--cwd"):
            out.cwd = next_val(a)
        elif a == "--web":
            if i + 1 < n and argv[i + 1].isdigit():
                raise UsageError(f"--web takes no port argument. Use: --web --port {argv[i + 1]}")
            out.web = True
            out.overrides["BLENDERBOT_WEB"] = "true"
        elif a == "--open":
            out.web = True
            out.overrides["BLENDERBOT_WEB"] = "true"
            out.overrides["BLENDERBOT_WEB_OPEN"] = "true"
        elif a == "--port":
            v = next_val(a)
            if not v.isdigit() or int(v) > 65535:
                raise UsageError(f'--port must be a number between 0 and 65535 (got "{v}")')
            out.port = int(v)
            out.web = True
            out.overrides["BLENDERBOT_WEB"] = "true"
            out.overrides["BLENDERBOT_WEB_PORT"] = v
        elif a == "--host":
            h = next_val(a)
            out.host = h
            out.web = True
            out.overrides["BLENDERBOT_WEB"] = "true"
            out.overrides["BLENDERBOT_WEB_HOST"] = h
        elif a == "--web-host":
            raise UsageError("--web-host is --host")
        elif a in ("--doctor",):
            out.doctor = True
        elif a in ("--install-addon", "--install-bridge"):
            out.install_addon = True
        elif a == "--models":
            out.list_models = True
            if i + 1 < n and not argv[i + 1].startswith("-"):
                out.overrides["BLENDERBOT_PROVIDER"] = next_val(a)
        elif a == "--sessions":
            out.list_sessions = True
        elif a == "--resume":
            if i + 1 < n and not argv[i + 1].startswith("-"):
                out.resume = next_val(a)
            else:
                out.resume = True
        elif a == "--config":
            out.show_config = True
        elif a.startswith("-"):
            raise UsageError(f"unknown option: {a}\n\n{HELP}")
        else:
            out.words.append(a)
        i += 1

    out.input_text = " ".join(out.words)
    return out


def _build_config(args: ParsedArgs) -> Config:
    home = resolve_home()
    defaults = settings_to_defaults(load_settings(home))
    cfg = load_config(cwd=args.cwd, defaults=defaults, overrides=args.overrides, home=home)
    if cfg.web_enabled:
        assert_safe_bind(host=cfg.web_host, yolo=cfg.yolo)
    return cfg


async def _probe_live_bridge() -> dict[str, object]:
    from .bridge.protocol import read_discovery
    from .transport.bridge_client import BridgeTransport

    disc = read_discovery(resolve_home())
    if not disc:
        return {"connected": False, "reason": "no ~/.blenderbot/bridge.json"}
    bt = BridgeTransport(host=disc.host, port=disc.port, token=disc.token)
    try:
        st = await bt.call("blender.status", timeout=3.0)
        return {"connected": True, "status": st}
    except Exception as err:  # noqa: BLE001
        return {"connected": False, "reason": str(err)}
    finally:
        await bt.close()


def _print_doctor(cfg: Config) -> int:
    from .blender_ops import list_ops
    from .bridge.installer import STARTUP_SCRIPT_NAME

    inst = discover_blender()
    addon_manifest = Path(inst.user_extensions_dir) / "blenderbot_bridge" / "blender_manifest.toml"
    startup_hook = Path(inst.user_scripts_startup_dir) / STARTUP_SCRIPT_NAME
    live = asyncio.run(_probe_live_bridge())
    ops = list_ops()

    lines = [
        f"blenderbot {__version__} — Blender 5.3+ environment & provider check",
        "",
        "Host & Media Tools",
        f"  python              {inst.python_version} ({inst.python_arch_bits}-bit) on {inst.platform}",
        f"  ffmpeg              {inst.ffmpeg_path or 'missing'}",
        f"  ffprobe             {inst.ffprobe_path or 'missing'}",
        "",
        "Blender 5.x",
        f"  blender executable  {inst.blender_exe_path or 'NOT FOUND'}",
        f"  blender version     {inst.blender_version or 'unknown'}",
        f"  gui process         {'RUNNING' if inst.blender_running else 'not running (HeadlessTransport ready)'}",
        f"  user extension      {'installed (' + str(addon_manifest.parent) + ')' if addon_manifest.is_file() else 'not installed (run `blenderbot --install-addon`)'}",
        f"  startup auto-bridge {'installed (' + str(startup_hook) + ')' if startup_hook.is_file() else 'not installed (run `blenderbot --install-addon`)'}",
        f"  registered bl_ops   {len(ops)} stdlib/bpy operations",
    ]
    if live.get("connected"):
        st = live["status"]  # type: ignore[index]
        assert isinstance(st, dict)
        sc = st.get("scene") or {}
        lines.append(
            f"  live gui bridge     CONNECTED -> Blender {st.get('blender_version')} | scene '{sc.get('name')}' ({sc.get('fps')} fps)"
        )
    elif inst.blender_running:
        lines.append(
            f"  live gui bridge     standby ({live.get('reason')}) — open Blender was launched before bridge loaded; restart Blender or enable 'blenderbot Bridge' in Preferences -> Add-ons"
        )
    else:
        lines.append(
            f"  live gui bridge     standby ({live.get('reason')}) — will auto-spawn warm HeadlessTransport"
        )

    modal_toml = Path.home() / ".modal.toml"
    has_modal_auth = bool(os.environ.get("MODAL_TOKEN_ID") or modal_toml.is_file())
    modal_status = "configured (~/.modal.toml / env)" if has_modal_auth else "not configured (run `modal setup`)"

    lines.extend(
        [
            "",
            "Providers & Cloud Compute",
            f"  active              {cfg.provider} / {cfg.model.id} (effort={cfg.effort}, role={cfg.role}, vision={'yes' if cfg.model.vision else 'no'}, video={'yes' if cfg.model.video else 'no'})",
            f"  image generator     {cfg.image_model} (OpenRouter /api/v1/images)",
            f"  video generator     {cfg.video_model} (OpenRouter /api/v1/videos)",
            f"  3d mesh generator   {cfg.meshy_model} (fal Queue API)",
            f"  modal 3dgs cloud    {modal_status} (app: blenderbot-splat)",
        ]
    )
    for prov, key_val in cfg.keys.items():
        if key_val:
            origin = cfg.key_origins.get(prov, {})
            src = origin.get("from") or "unknown"
            var = origin.get("name") or ""
            lines.append(f"  {prov:<18}  configured ({var} from {src})")
        else:
            lines.append(f"  {prov:<18}  —")
    if cfg.warnings:
        lines.append("")
        lines.append("Warnings")
        for w in cfg.warnings:
            lines.append(f"  ! {w}")

    sys.stdout.write("\n".join(lines) + "\n")
    return 0 if inst.ready else 1


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.help:
        sys.stdout.write(HELP)
        return 0
    if args.version:
        sys.stdout.write(f"{__version__}\n")
        return 0

    if args.install_addon:
        from .bridge.installer import STARTUP_SCRIPT_NAME, install_blender_addon

        inst = discover_blender()
        path = install_blender_addon()
        startup_path = Path(inst.user_scripts_startup_dir) / STARTUP_SCRIPT_NAME
        sys.stdout.write(
            f"Installed blenderbot Blender 5.x extension & GUI auto-start hook:\n"
            f"  extension: {path}\n"
            f"  startup:   {startup_path}\n\n"
            "Blender 5.3+ will now auto-start the GUI bridge whenever opened.\n"
            "If Blender is already open right now, either restart it or tick 'blenderbot Bridge' in Edit -> Preferences -> Add-ons.\n"
        )
        return 0

    cfg = _build_config(args)

    if args.show_config:
        sys.stdout.write(json.dumps(cfg.to_public_dict(), indent=2) + "\n")
        return 0

    if args.doctor:
        return _print_doctor(cfg)

    if args.list_models:
        from .config.models import MODELS
        from .providers import list_remote_models

        try:
            ids = asyncio.run(list_remote_models(cfg))
        except Exception:  # noqa: BLE001
            ids = [m.id for m in MODELS if m.provider == cfg.provider]
        sys.stdout.write("\n".join(ids) + ("\n" if ids else ""))
        return 0

    if args.list_sessions:
        from .session.log import list_sessions

        rows = list_sessions(Path(cfg.home))
        if not rows:
            sys.stdout.write("no sessions yet\n")
            return 0
        for s in rows:
            t = s.get("totals") or {}
            sys.stdout.write(
                f"{s.get('id')}  {s.get('started_at')}  {s.get('model')}  "
                f"{t.get('turns', 0)} turns  ${float(t.get('cost', 0.0)):.4f}  {s.get('cwd')}\n"
            )
        return 0

    from .headless import read_piped_stdin, run_headless_async

    piped = read_piped_stdin() if not args.input_text else ""
    combined_input = "\n".join(x for x in (args.input_text, piped) if x)

    interactive = (
        not args.print_mode
        and sys.stdout is not None
        and sys.stdin is not None
        and sys.stdout.isatty()
        and sys.stdin.isatty()
    )

    if not interactive:
        if not combined_input:
            sys.stderr.write("nothing to do: pass a request or pipe one in\n")
            return 2
        return asyncio.run(
            run_headless_async(
                combined_input,
                json_mode=args.json_mode,
                yes=args.yes,
                cwd=args.cwd,
                overrides=args.overrides,
                resume=args.resume,
                accept_stale_prompts=args.accept_stale_prompts,
            )
        )

    from .ui.start import start_tui

    return start_tui(
        initial_input=combined_input,
        cwd=args.cwd,
        overrides=args.overrides,
        resume=args.resume,
        accept_stale_prompts=args.accept_stale_prompts,
    )


def run() -> None:
    try:
        code = main()
        raise SystemExit(code)
    except UsageError as err:
        sys.stderr.write(f"{err}\n")
        raise SystemExit(2) from None
    except Exception as err:  # noqa: BLE001
        sys.stderr.write(f"\n{err}\n")
        if os.environ.get("BLENDERBOT_DEBUG"):
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    run()