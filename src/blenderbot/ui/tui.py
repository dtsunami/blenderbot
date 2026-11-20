"""Interactive Terminal TUI REPL (`prompt_toolkit` + `rich`) for `blenderbot`.

Uses `prompt_toolkit.shortcuts.print_formatted_text(ANSI(...))` rather than `patch_stdout`
so Rich panels, colors, and streamed lines render natively through prompt_toolkit's Win32/VT
renderer with zero raw-escape artifacts, sanitized single-line previews, and non-blocking
async exception handling.
"""

from __future__ import annotations

import asyncio
import html
import io
import json
import re
import shutil
import textwrap
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import print_formatted_text
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from blenderbot import __version__
from blenderbot.context.utf8 import sanitize_utf8_str
from blenderbot.session.bootstrap import Session
from blenderbot.session.dispatch import create_dispatcher
from blenderbot.ui.commands import command_list

_ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_APPROVAL_FOOTER = (
    "Press [bold green]y[/] allow · [bold cyan]c[/] checkpoint .blend & allow · "
    "[bold red]n[/] deny  [dim](single key — no Enter needed)[/]"
)


def _sanitize_line(s: str, max_cols: int = 400) -> str:
    clean = _ANSI_STRIP_RE.sub("", sanitize_utf8_str(s)).replace("\r", "").replace("\t", "  ")
    if len(clean) > max_cols:
        return clean[: max_cols - 1] + "…"
    return clean


def _format_approval_body(op_dict: dict[str, Any]) -> Group:
    """Format gated ops cleanly — line-numbered Python for `bpy.eval`, bash for `shell`, JSON for `scene.*`."""
    op_name = str(op_dict.get("op", ""))
    parts: list[Any] = []

    if op_name == "bpy.eval":
        header = Text()
        header.append("bpy.eval", style="bold bright_cyan")
        desc = str(op_dict.get("description") or "").strip()
        if desc:
            header.append(f" — {desc}", style="bold bright_white")
        parts.append(header)

        raw_code = str(op_dict.get("code") or "")
        clean_code = textwrap.dedent(raw_code).strip("\n") or "# (empty)"
        parts.append(
            Panel(
                Syntax(
                    clean_code,
                    "python",
                    theme="monokai",
                    background_color="default",
                    line_numbers=True,
                    word_wrap=True,
                    indent_guides=True,
                    padding=(0, 1),
                ),
                title="[dim cyan]python · live blender 5.3 session[/]",
                title_align="left",
                border_style="bright_black",
                padding=(0, 0),
            )
        )
        parts.append(Text.from_markup(_APPROVAL_FOOTER))
        return Group(*parts)

    if op_name == "shell":
        header = Text()
        header.append("shell", style="bold bright_cyan")
        if op_dict.get("cwd"):
            header.append(f"  (cwd: {op_dict['cwd']})", style="dim")
        parts.append(header)
        clean_cmd = textwrap.dedent(str(op_dict.get("cmd") or "")).strip("\n")
        parts.append(
            Panel(
                Syntax(
                    clean_cmd,
                    "bash",
                    theme="monokai",
                    background_color="default",
                    line_numbers=True,
                    word_wrap=True,
                    padding=(0, 1),
                ),
                title="[dim cyan]command[/]",
                title_align="left",
                border_style="bright_black",
                padding=(0, 0),
            )
        )
        parts.append(Text.from_markup(_APPROVAL_FOOTER))
        return Group(*parts)

    parts.append(
        Syntax(
            json.dumps(op_dict, indent=2, ensure_ascii=False),
            "json",
            theme="monokai",
            background_color="default",
            word_wrap=True,
        )
    )
    parts.append(Text(""))
    parts.append(Text.from_markup(_APPROVAL_FOOTER))
    return Group(*parts)


class BlenderbotCompleter(Completer):
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self._cmds = command_list()

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        _ = complete_event
        text = document.text_before_cursor

        if text.startswith("/"):
            first = text.split()[0] if text.strip() else "/"
            if " " not in text:
                for c in self._cmds:
                    if c["name"].startswith(first.lower()):
                        yield Completion(
                            c["name"],
                            start_position=-len(first),
                            display_meta=c["desc"][:52],
                        )
            return

        at_idx = text.rfind("@")
        if at_idx >= 0 and (at_idx == 0 or text[at_idx - 1].isspace()):
            frag = text[at_idx + 1 :]
            if " " not in frag:
                parent_rel = Path(frag).parent if "/" in frag or "\\" in frag else Path(".")
                prefix = Path(frag).name.lower()
                search_dir = (self.cwd / parent_rel).resolve()
                if search_dir.is_dir():
                    try:
                        for child in sorted(
                            search_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
                        )[:40]:
                            if child.name.startswith("."):
                                continue
                            if child.name.lower().startswith(prefix):
                                rel_str = str(child.relative_to(self.cwd)).replace("\\", "/")
                                if child.is_dir():
                                    rel_str += "/"
                                yield Completion(rel_str, start_position=-len(frag))
                    except OSError:
                        return


class BlenderbotPtHistory(History):
    """Bridges `~/.blenderbot/history.jsonl` into prompt_toolkit's Up/Down arrow history."""

    def __init__(self, store: Any) -> None:
        super().__init__()
        self._store = store

    def load_history_strings(self) -> Iterable[str]:
        return list(reversed(self._store.strings()))

    def store_string(self, string: str) -> None:
        self._store.add(string)


def _context_bar(fraction: float, width: int = 8) -> str:
    filled = min(width, max(0, round(fraction * width)))
    return "█" * filled + "░" * (width - filled)


def _pt_print(renderable: Any = "") -> None:
    cols = max(60, shutil.get_terminal_size((100, 24)).columns - 1)
    buf = io.StringIO()
    c = Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
        width=cols,
        legacy_windows=False,
    )
    c.print(renderable)
    out = buf.getvalue().removesuffix("\n")
    print_formatted_text(ANSI(out))


def _is_benign_socket_disconnect(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
        return True
    winerr = getattr(exc, "winerror", None)
    return winerr in (64, 995, 10053, 10054)


async def run_tui_loop(
    *,
    session: Session,
    initial_input: str = "",
) -> int:
    spinner_frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    should_exit = False
    pending_approvals: list[dict[str, Any]] = []
    in_thinking = False
    think_buf = ""
    text_buf = ""
    activity_label = ""
    activity_started = 0.0
    slash_busy = False
    pt_history = BlenderbotPtHistory(session.history)

    def _set_activity(label: str) -> None:
        nonlocal activity_label, activity_started
        activity_label = label
        if label:
            activity_started = time.monotonic()

    def _spin_glyph() -> str:
        idx = int(time.monotonic() * 10) % len(spinner_frames)
        return spinner_frames[idx]

    def _request_exit() -> None:
        nonlocal should_exit
        should_exit = True
        session.agent.cancel()

    def _flush_think_buf(*, force: bool = False) -> None:
        nonlocal think_buf
        cols = max(60, shutil.get_terminal_size((100, 24)).columns * 2)
        while "\n" in think_buf:
            line, think_buf = think_buf.split("\n", 1)
            clean = _sanitize_line(line, cols)
            _pt_print(Text(f"  💭 {clean}" if clean else "", style="dim italic"))
        if force and think_buf:
            clean = _sanitize_line(think_buf, cols)
            _pt_print(Text(f"  💭 {clean}", style="dim italic"))
            think_buf = ""

    def _flush_text_buf(*, force: bool = False) -> None:
        nonlocal text_buf
        cols = max(80, shutil.get_terminal_size((100, 24)).columns * 3)
        while "\n" in text_buf:
            line, text_buf = text_buf.split("\n", 1)
            _pt_print(Text(_sanitize_line(line, cols)))
        if force and text_buf:
            _pt_print(Text(_sanitize_line(text_buf, cols)))
            text_buf = ""

    def _render_entry(entry: dict[str, Any], *, from_web: bool = False) -> None:
        etype = entry.get("type", "info")
        if etype == "user":
            txt = str(entry.get("text", ""))
            if from_web:
                _pt_print(f"\n[bold bright_white]❯ [web][/] {_sanitize_line(txt, 500)}")
            elif txt.startswith("[steer]"):
                _pt_print(f"[magenta]  ↳ {_sanitize_line(txt, 300)}[/]")
        elif etype in ("notice", "info"):
            msg = str(entry.get("message") or entry.get("text") or "")
            _pt_print(Text(f"  · {msg}", style="cyan"))
        elif etype == "warning":
            _pt_print(Text(f"  ! {entry.get('message', '')}", style="yellow"))
        elif etype == "error":
            _pt_print(Text(f"  ✗ {entry.get('message', '')}", style="bold red"))
        elif etype == "shell":
            code = entry.get("exit_code", 0)
            color = "green" if code == 0 else "red"
            _pt_print(f"[{color}]  $ {entry.get('cmd')} (exit {code})[/]")
            if entry.get("output"):
                _pt_print(Text(str(entry["output"]), style="dim"))

    submit = create_dispatcher(
        session=session,
        local_push=(
            lambda entry: _render_entry(entry, from_web=False)
            if entry.get("origin") != "web"
            else None
        ),
        on_exit=_request_exit,
    )

    def _on_bus_event(ev: dict[str, Any]) -> None:
        if (
            ev["type"] == "transcript"
            and isinstance(ev["data"], dict)
            and ev["data"].get("origin") == "web"
        ):
            _render_entry(ev["data"], from_web=True)
            if ev["data"].get("type") == "user" and ev["data"].get("text"):
                pt_history.append_string(str(ev["data"]["text"]))
        elif ev["type"] == "approval-resolved":
            ap_id = ev["data"].get("id")
            before = len(pending_approvals)
            pending_approvals[:] = [p for p in pending_approvals if p.get("id") != ap_id]
            if len(pending_approvals) < before:
                dec = ev["data"].get("decision") or {}
                allow = bool(dec.get("allow"))
                verdict = "allowed" if allow else "denied"
                _pt_print(f"[dim]  ✓ approval {ap_id} {verdict} (web)[/]")
                _set_activity(
                    "waiting for approval [y/c/n]…"
                    if pending_approvals
                    else ("running approved op…" if allow else "processing denial…")
                )

    session.bus.subscribe(_on_bus_event)

    def _on_turn_start(_p: Any) -> None:
        nonlocal in_thinking, think_buf, text_buf
        in_thinking = False
        think_buf = ""
        text_buf = ""
        _set_activity(f"thinking ({session.cfg.model.id})…")

    def _on_thinking(delta: str) -> None:
        nonlocal in_thinking, activity_label, think_buf
        if activity_label != "thinking…":
            activity_label = "thinking…"
        if not session.cfg.thinking_visible or not delta:
            return
        in_thinking = True
        think_buf += delta
        if "\n" in think_buf:
            _flush_think_buf(force=False)

    def _on_text(delta: str) -> None:
        nonlocal in_thinking, activity_label, text_buf
        activity_label = "streaming response…"
        if in_thinking:
            _flush_think_buf(force=True)
            in_thinking = False
        text_buf += delta
        if "\n" in text_buf:
            _flush_text_buf(force=False)

    def _on_turn_end(p: dict[str, Any]) -> None:
        nonlocal in_thinking
        if in_thinking:
            _flush_think_buf(force=True)
            in_thinking = False
        _flush_text_buf(force=True)
        _set_activity("processing turn…")
        tps = session.perf.live_tps
        ttft = p.get("ttft")
        ttft_s = f" · ttft {ttft:.2f}s" if ttft is not None else ""
        if tps > 0:
            _pt_print(f"[dim]{tps:.1f} tok/s{ttft_s}[/]")

    def _on_ops_parsed(ops: list[dict[str, Any]]) -> None:
        names = ", ".join(o["op"] for o in ops[:2])
        if len(ops) > 2:
            names += f" (+{len(ops) - 2})"
        _set_activity(f"running {names}…")
        for o in ops:
            _pt_print(f"[bright_cyan]  · {o['op']}[/]")

    def _on_ops_result(res: dict[str, Any]) -> None:
        _set_activity("preparing next turn…")
        if res.get("checkpoint"):
            _pt_print(f"[dim]  ✓ checkpoint: {res['checkpoint']} (/undo to restore)[/]")
        for m_att in res.get("attached_media") or []:
            _pt_print(
                f"[dim cyan]  👁 attached {m_att['kind']} to vision context: {Path(m_att['path']).name}[/]"
            )
        for err in res.get("errors") or []:
            _pt_print(
                Text(f"  ✗ [{err.get('index')}] {err.get('op')}: {err.get('message')}", style="red")
            )

    def _on_artifact(art: dict[str, Any]) -> None:
        _pt_print(
            f"[dim]  📄 artifact #{art['id']} [{art['lang']}] ({art['lines']} lines) — /artifacts[/]"
        )

    def _on_approval_request(req: dict[str, Any]) -> None:
        pending_approvals.append(req)
        _set_activity("waiting for approval [y/c/n]…")
        _pt_print(
            Panel(
                _format_approval_body(req["op"]),
                title=f"▲ Approval Needed ({req.get('id', 'ap')})",
                border_style="yellow",
            )
        )

    def _on_op_progress(p: dict[str, Any]) -> None:
        nonlocal activity_label
        if not isinstance(p, dict):
            return
        op_lbl = str(p.get("op") or p.get("stage") or "render")
        if p.get("done_frames") is not None and p.get("total_frames"):
            done_f = int(p["done_frames"])
            tot_f = int(p["total_frames"])
            pct = p.get("pct", 0.0)
            ms_f = f" · {p['ms_per_frame']}ms/f" if p.get("ms_per_frame") else ""
            eta = f" · ETA {p['eta_s']}s" if p.get("eta_s") else ""
            dev = f" [{p['device']}]" if p.get("device") else ""
            activity_label = f"{op_lbl}{dev} frame {done_f}/{tot_f} ({pct}%){ms_f}{eta}…"
        elif p.get("phase") == "render_stats" and p.get("stats"):
            activity_label = f"{op_lbl} f{p.get('frame', '?')} · {str(p['stats'])[:64]}…"
        elif p.get("step") is not None and p.get("total_steps"):
            activity_label = (
                f"Modal 3DGS step {p['step']}/{p['total_steps']} "
                f"(PSNR {p.get('psnr', 0)} dB)…"
            )

    session.agent.on("turn-start", _on_turn_start)
    session.agent.on("thinking", _on_thinking)
    session.agent.on("text", _on_text)
    session.agent.on("turn-end", _on_turn_end)
    session.agent.on("ops-parsed", _on_ops_parsed)
    session.agent.on("op-progress", _on_op_progress)
    session.agent.on("ops-result", _on_ops_result)
    session.agent.on("artifact", _on_artifact)
    session.agent.on("approval-request", _on_approval_request)
    session.agent.on("usage", lambda _u: None)
    session.agent.on(
        "compact-start",
        lambda _p: (
            _set_activity("compacting context…"),
            _pt_print("[dim]  ⟳ compacting context…[/]"),
        ),
    )
    session.agent.on(
        "compact-done",
        lambda p: _pt_print(f"[dim]  ✓ compacted (freed {p.get('freed_tokens', 0):,} tok)[/]"),
    )
    session.agent.on("nudge", lambda _p: _pt_print("[dim]  ↻ nudging model for bl_ops…[/]"))
    session.agent.on(
        "retry",
        lambda p: (
            _set_activity(f"retrying provider (#{p.get('attempt')})…"),
            _pt_print(f"[yellow]  ↻ retrying provider (#{p.get('attempt')})…[/]"),
        ),
    )
    session.agent.on(
        "steer-queued",
        lambda p: _pt_print(f"[magenta]  ↳ steer queued ({p.get('queued')})[/]"),
    )
    session.agent.on("steered", lambda _p: _pt_print("[magenta]  ↳ steer applied[/]"))
    session.agent.on(
        "prompts-applied",
        lambda p: _pt_print(
            f"[cyan]  ✓ prompts rebuilt ({p.get('after_tokens', 0):,} tok; cache reset)[/]"
        ),
    )
    session.agent.on(
        "repair",
        lambda p: _pt_print(f"[yellow]  ⚙ repairing bl_ops syntax (#{p.get('attempt')})…[/]"),
    )
    session.agent.on(
        "warning",
        lambda p: _pt_print(Text(f"  ! {p.get('message', p) if isinstance(p, dict) else p}", style="yellow")),
    )
    session.agent.on(
        "cancelled",
        lambda _p: (_set_activity(""), _pt_print("[yellow]  ■ turn cancelled[/]")),
    )
    session.agent.on(
        "error",
        lambda p: (
            _set_activity(""),
            _pt_print(Text(f"  ✗ {p.get('message', p) if isinstance(p, dict) else p}", style="bold red")),
        ),
    )
    session.agent.on("done", lambda _p: _set_activity(""))

    bl_badge = (
        f"[green]connected ({session.transport.mode})[/]"
        if session.transport is not None
        else f"[yellow]disconnected ({session.blender_connect_note or 'run /blender'})[/]"
    )
    ro_note = (
        f" [dim](ro: {', '.join(str(p) for p in session.cfg.ro_dirs)})[/]"
        if session.cfg.ro_dirs
        else ""
    )
    web_line = ""
    if session.cfg.web_enabled:
        from blenderbot.server import ensure_web_sidecar

        srv = await ensure_web_sidecar(
            session, submit_fn=submit, open_browser=session.cfg.web_open
        )
        web_line = f"\nSeedance & 3D Studio: [bold bright_cyan]{srv.auth_url}[/]"

    resumed_from = session.log.meta.get("resumed_from")
    banner_lines = [
        f"[bold]blenderbot {__version__}[/] · [cyan]{session.cfg.provider}/{session.cfg.model.id}[/] (effort={session.cfg.effort}, role={session.cfg.role})",
        f"Workspace: [bold]{session.cfg.cwd}[/]{ro_note}"
        + (f" · [dim]resumed {resumed_from}[/]" if resumed_from else ""),
        f"Blender 5.3+: {bl_badge}{web_line}",
        "[dim]Type `/seedance` to open the Seedance 2.5 Studio GUI, `/help` for commands, `@file` to attach media, `Esc` to cancel.[/]",
    ]
    _pt_print(Panel("\n".join(banner_lines), border_style="bright_black", expand=False))

    kb = KeyBindings()

    @kb.add("escape")
    def _esc_cancel(_event: Any) -> None:
        if session.agent.busy:
            session.agent.cancel()

    def _is_working() -> bool:
        return bool(session.agent.busy or slash_busy or activity_label)

    def _toolbar() -> HTML:
        cfg = session.cfg
        frac = session.ledger.usage_fraction
        pct = round(frac * 100)
        bar = _context_bar(frac)
        st = session.agent.stats
        bl_mode = session.transport.mode if session.transport else "offline"
        spark = session.perf.sparkline(5)
        tps = session.perf.live_tps
        tps_part = f" · {tps:.0f} tok/s {spark}" if tps > 0 else ""
        yolo_part = ' · <style bg="ansired" fg="ansiwhite"> YOLO </style>' if cfg.yolo else ""
        busy_part = ""
        if _is_working():
            elapsed = max(0.0, time.monotonic() - activity_started) if activity_started else 0.0
            lbl = html.escape(activity_label or "working…")
            busy_part = (
                f' · <style fg="ansiyellow"><b>{_spin_glyph()} {lbl} ({elapsed:.1f}s)</b></style>'
                ' <style fg="ansibrightblack">[type to steer · Esc cancel]</style>'
            )
        return HTML(
            f" <b>{html.escape(cfg.model.id)}</b>/{cfg.effort} · [{bl_mode}:{cfg.role}] · {bar} {pct}% "
            f"· {st['turns']}t {st['ops']}ops · ${st['cost']:.3f}{tps_part}{yolo_part}{busy_part}"
        )

    def _prompt_message() -> HTML:
        cols = max(40, shutil.get_terminal_size((100, 24)).columns - 12)
        preview = ""
        if text_buf:
            tail = html.escape(_sanitize_line(text_buf[-cols:], cols))
            preview = f"{tail}\n"
        elif think_buf and session.cfg.thinking_visible:
            tail = html.escape(_sanitize_line(think_buf[-cols:], cols))
            preview = f'<style fg="ansibrightblack"><i>  💭 {tail}</i></style>\n'

        if pending_approvals:
            return HTML(f'{preview}<style fg="ansiyellow"><b>▲ approve [y/c/n] ❯ </b></style>')
        if _is_working():
            return HTML(f'{preview}<style fg="ansicyan"><b>{_spin_glyph()}</b></style> <b>❯ </b>')
        return HTML("<b>❯ </b>")

    prompt_session: PromptSession[str] = PromptSession(
        history=pt_history,
        completer=BlenderbotCompleter(session.cfg.cwd),
        key_bindings=kb,
        bottom_toolbar=_toolbar,
        refresh_interval=0.12,
    )

    def _resolve_next_approval(choice: str) -> None:
        if not pending_approvals:
            return
        ap = pending_approvals.pop(0)
        ap_id = ap.get("id")
        label = ap_id or "ap"
        allow = choice in ("y", "yes", "c", "checkpoint")
        cp_first = choice in ("c", "checkpoint")
        if cp_first:
            _pt_print(f"[cyan]  ✓ approval {label}: checkpointing .blend & allowing[/]")
        elif allow:
            _pt_print(f"[green]  ✓ approval {label}: allowed[/]")
        else:
            _pt_print(f"[red]  ✗ approval {label}: denied[/]")
        _set_activity(
            "waiting for approval [y/c/n]…"
            if pending_approvals
            else ("running approved op…" if allow else "processing denial…")
        )
        if ap_id:
            session.bus.resolve_approval(ap_id, {"allow": allow, "checkpoint_first": cp_first})
        else:
            ap["respond"]({"allow": allow, "checkpoint_first": cp_first})

    _approval_hotkey = Condition(
        lambda: bool(pending_approvals) and not prompt_session.default_buffer.text
    )
    for _key_ch in ("y", "Y", "c", "C", "n", "N"):

        @kb.add(_key_ch, filter=_approval_hotkey)
        def _on_instant_approval(event: Any, ch: str = _key_ch.lower()) -> None:
            _resolve_next_approval(ch)
            event.app.invalidate()

    def _nonblocking_loop_exception_handler(
        _loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        exc = context.get("exception")
        if _is_benign_socket_disconnect(exc):
            return
        msg = str(exc or context.get("message") or "event loop error")
        _pt_print(f"[dim yellow]  ⚠ async loop warning: {html.escape(msg)}[/]")

    prompt_session.app._handle_exception = _nonblocking_loop_exception_handler  # type: ignore[method-assign]

    active_tasks: set[asyncio.Task[Any]] = set()

    async def _safe_submit(line: str) -> None:
        nonlocal slash_busy
        is_cmd = line.strip().startswith(("/", "sh ")) or line.strip() == "sh"
        if is_cmd:
            slash_busy = True
            _set_activity(f"running {line.strip().split()[0]}…")
        elif not session.agent.busy:
            _set_activity("sending prompt…")
        try:
            await submit(line, "tui")
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            _pt_print(Text(f"  ✗ {type(err).__name__}: {err}", style="bold red"))
        finally:
            if is_cmd:
                slash_busy = False
                if not session.agent.busy:
                    _set_activity("")

    def _spawn_submit(line: str) -> None:
        t = asyncio.create_task(_safe_submit(line))
        active_tasks.add(t)
        t.add_done_callback(active_tasks.discard)

    if initial_input.strip():
        _pt_print(f"\n[bold bright_white]❯[/] {_sanitize_line(initial_input.strip(), 500)}")
        _spawn_submit(initial_input.strip())

    while not should_exit:
        try:
            line = sanitize_utf8_str(await prompt_session.prompt_async(_prompt_message))
        except KeyboardInterrupt:
            if session.agent.busy:
                session.agent.cancel()
                continue
            break
        except (EOFError, asyncio.CancelledError):
            break

        trimmed = line.strip()
        if not trimmed:
            continue

        if pending_approvals and trimmed.lower() in ("y", "yes", "n", "no", "c", "checkpoint"):
            _resolve_next_approval(trimmed.lower())
            continue

        if line.count("\n") >= 4:
            token_str, item = session.pastes.add(line)
            _pt_print(f"[dim]  folded {item.lines}-line paste as {token_str}[/]")
            _spawn_submit(token_str)
        else:
            _spawn_submit(trimmed)

    if active_tasks:
        for t in list(active_tasks):
            t.cancel()
        await asyncio.gather(*active_tasks, return_exceptions=True)

    return 0