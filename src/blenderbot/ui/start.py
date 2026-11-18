"""Bootstraps `Session`, prepares Windows console mode, and runs `run_tui_loop`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from blenderbot.session.bootstrap import bootstrap_session

from .clipboard import prepare_windows_console
from .tui import run_tui_loop


async def _start_async(
    *,
    initial_input: str = "",
    cwd: str | Path | None = None,
    overrides: dict[str, str] | None = None,
    resume: str | bool = False,
    accept_stale_prompts: bool = False,
    home: str | Path | None = None,
) -> int:
    prepare_windows_console()
    session = await bootstrap_session(
        cwd=cwd,
        overrides=overrides,
        resume=resume,
        accept_stale_prompts=accept_stale_prompts,
        home=home,
    )
    try:
        return await run_tui_loop(
            session=session,
            initial_input=initial_input,
        )
    finally:
        await session.close()


def start_tui(
    *,
    initial_input: str = "",
    cwd: str | Path | None = None,
    overrides: dict[str, str] | None = None,
    resume: str | bool = False,
    accept_stale_prompts: bool = False,
    home: str | Path | None = None,
) -> int:
    try:
        return asyncio.run(
            _start_async(
                initial_input=initial_input,
                cwd=cwd,
                overrides=overrides,
                resume=resume,
                accept_stale_prompts=accept_stale_prompts,
                home=home,
            )
        )
    except KeyboardInterrupt:
        return 130