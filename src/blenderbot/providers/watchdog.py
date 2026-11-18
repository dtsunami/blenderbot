"""Stream idle-stall watchdog + instant cancellation race (`blenderbot.providers.watchdog`).

Ported from `davai/src/providers/watchdog.js`:
1. Half-open SSE / HTTP/2 sockets behind NAT, OpenRouter, or Gemini gateways can suspend
   `async for chunk in stream` indefinitely with no FIN/RST.
2. When the operator presses `Esc` (`cancel_event.set()`), waiting for the next network
   chunk to notice cancellation freezes the REPL on `cancelling…`.

`iterate_with_idle_timeout` races every `anext()` against BOTH:
- `cancel_event.wait()` -> exits immediately and closes the underlying stream in the background.
- `first_chunk_sec` (default 180s so deep reasoning models can finish thinking) and
  `idle_sec` (default 90s between chunks mid-stream) -> raises a retryable `ProviderError(status=408)`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from .base import ProviderError

T = TypeVar("T")

DEFAULT_FIRST_CHUNK_TIMEOUT_SEC = 180.0
DEFAULT_IDLE_CHUNK_TIMEOUT_SEC = 90.0


def resolve_stream_timeouts() -> tuple[float, float]:
    first_raw = (
        os.environ.get("BLENDERBOT_STREAM_FIRST_BYTE_TIMEOUT")
        or os.environ.get("BLENDERBOT_STREAM_IDLE_TIMEOUT")
        or os.environ.get("DAVAG_STREAM_IDLE_TIMEOUT")
    )
    idle_raw = (
        os.environ.get("BLENDERBOT_STREAM_IDLE_TIMEOUT")
        or os.environ.get("DAVAG_STREAM_IDLE_TIMEOUT")
    )
    first_sec = float(first_raw) if first_raw else DEFAULT_FIRST_CHUNK_TIMEOUT_SEC
    idle_sec = float(idle_raw) if idle_raw else DEFAULT_IDLE_CHUNK_TIMEOUT_SEC
    return first_sec, idle_sec


async def _close_async_iter(aiter: Any, stream: Any) -> None:
    for obj, method in ((stream, "close"), (stream, "aclose"), (aiter, "aclose")):
        fn = getattr(obj, method, None)
        if callable(fn):
            with contextlib.suppress(Exception):
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
            return


async def iterate_with_idle_timeout(
    stream: Any,
    *,
    provider: str,
    cancel_event: asyncio.Event | None = None,
    first_chunk_sec: float | None = None,
    idle_sec: float | None = None,
) -> AsyncIterator[Any]:
    """Yield items from `stream` with per-chunk idle timeout and instant `cancel_event` wakeup."""
    if cancel_event is not None and cancel_event.is_set():
        return

    eff_first, eff_idle = resolve_stream_timeouts()
    if first_chunk_sec is not None:
        eff_first = first_chunk_sec
    if idle_sec is not None:
        eff_idle = idle_sec

    aiter = stream.__aiter__()
    saw_chunk = False

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return

            budget = eff_idle if saw_chunk else eff_first
            next_task: asyncio.Task[Any] = asyncio.create_task(aiter.__anext__())
            cancel_task: asyncio.Task[bool] | None = (
                asyncio.create_task(cancel_event.wait())
                if cancel_event is not None
                else None
            )

            wait_set: set[asyncio.Task[Any]] = {next_task}
            if cancel_task is not None:
                wait_set.add(cancel_task)

            try:
                done, pending = await asyncio.wait(
                    wait_set,
                    timeout=budget if budget > 0 else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                next_task.cancel()
                if cancel_task is not None:
                    cancel_task.cancel()
                raise

            for p in pending:
                p.cancel()
                # Suppress late exceptions from cancelled tasks tearing down sockets
                p.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

            if cancel_task is not None and cancel_task in done:
                return

            if not done:
                where = (
                    f"stream stalled with no data for {budget:.0f}s mid-response"
                    if saw_chunk
                    else f"no response bytes received after {budget:.0f}s"
                )
                raise ProviderError(
                    f"[{provider}] stream timeout: {where}",
                    provider=provider,
                    status=408,
                    retryable=True,
                )

            try:
                item = next_task.result()
            except StopAsyncIteration:
                break

            saw_chunk = True
            yield item
    finally:
        await _close_async_iter(aiter, stream)