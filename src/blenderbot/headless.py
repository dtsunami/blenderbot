"""Headless runner (`blenderbot -p "..."`, `blenderbot --json "..."`, and piped stdin).

Consumes every declared event in `blenderbot.agent.loop.EVENTS` without importing any TUI modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .session.bootstrap import bootstrap_session


def read_piped_stdin() -> str:
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    try:
        return sys.stdin.read().strip()
    except OSError:
        return ""


async def run_headless_async(
    prompt: str,
    *,
    json_mode: bool = False,
    yes: bool = False,
    cwd: str | Path | None = None,
    overrides: dict[str, str] | None = None,
    resume: str | bool = False,
    accept_stale_prompts: bool = False,
    home: str | Path | None = None,
) -> int:
    eff_overrides = dict(overrides or {})
    if yes:
        eff_overrides["BLENDERBOT_YOLO"] = "true"

    session = await bootstrap_session(
        cwd=cwd,
        overrides=eff_overrides,
        resume=resume,
        accept_stale_prompts=accept_stale_prompts,
        home=home,
    )
    exit_code = 0

    def _emit_ndjson(ev_type: str, payload: Any) -> None:
        sys.stdout.write(json.dumps({"event": ev_type, "data": payload}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    if json_mode:
        for ev_name in (
            "turn-start",
            "thinking",
            "text",
            "turn-end",
            "usage",
            "ops-parsed",
            "op-progress",
            "ops-result",
            "artifact",
            "compact-start",
            "compact-done",
            "nudge",
            "retry",
            "steer-queued",
            "steered",
            "prompts-applied",
            "repair",
            "warning",
            "cancelled",
            "done",
        ):
            session.agent.on(ev_name, lambda p, n=ev_name: _emit_ndjson(n, p))

        def _on_err_json(p: Any) -> None:
            nonlocal exit_code
            exit_code = 1
            _emit_ndjson("error", p)

        session.agent.on("error", _on_err_json)
    else:
        stdout_needs_nl = False

        def _ensure_stdout_line_break() -> None:
            nonlocal stdout_needs_nl
            if stdout_needs_nl:
                sys.stdout.write("\n")
                sys.stdout.flush()
                stdout_needs_nl = False

        def _on_text_plain(delta: Any) -> None:
            nonlocal stdout_needs_nl
            s = str(delta)
            if not s:
                return
            sys.stdout.write(s)
            sys.stdout.flush()
            stdout_needs_nl = not s.endswith("\n")

        def _on_ops_plain(ops: list[dict[str, Any]]) -> None:
            _ensure_stdout_line_break()
            sys.stderr.write("  · " + ", ".join(str(o["op"]) for o in ops) + "\n")
            sys.stderr.flush()

        def _on_warn_plain(w: Any) -> None:
            _ensure_stdout_line_break()
            msg = w.get("message", w) if isinstance(w, dict) else w
            sys.stderr.write(f"  ! {msg}\n")
            sys.stderr.flush()

        def _on_retry_plain(r: dict[str, Any]) -> None:
            _ensure_stdout_line_break()
            sys.stderr.write(f"  ~ retry #{r.get('attempt')} in {r.get('delay')}s\n")
            sys.stderr.flush()

        session.agent.on("text", _on_text_plain)
        session.agent.on("turn-end", lambda _p: _ensure_stdout_line_break())
        session.agent.on("ops-parsed", _on_ops_plain)
        session.agent.on("warning", _on_warn_plain)
        session.agent.on("retry", _on_retry_plain)
        for silent_ev in (
            "turn-start",
            "thinking",
            "usage",
            "op-progress",
            "ops-result",
            "artifact",
            "compact-start",
            "compact-done",
            "nudge",
            "steer-queued",
            "steered",
            "prompts-applied",
            "repair",
            "cancelled",
            "done",
        ):
            session.agent.on(silent_ev, lambda _p: None)

        def _on_err_plain(p: Any) -> None:
            nonlocal exit_code
            exit_code = 1
            _ensure_stdout_line_break()
            msg = p.get("message", p) if isinstance(p, dict) else p
            sys.stderr.write(f"error: {msg}\n")
            sys.stderr.flush()

        session.agent.on("error", _on_err_plain)

    def _on_approval(req: dict[str, Any]) -> None:
        if session.cfg.yolo:
            req["respond"]({"allow": True})
        else:
            req["respond"](
                {
                    "allow": False,
                    "reason": "headless mode denied gated op (pass -y / --yolo to allow)",
                }
            )

    session.agent.on("approval-request", _on_approval)

    session.history.add(prompt)
    try:
        out = await session.agent.run(prompt)
        if not json_mode and out.get("status") == "done" and stdout_needs_nl:
            sys.stdout.write("\n")
            sys.stdout.flush()
    finally:
        await session.close()

    return exit_code