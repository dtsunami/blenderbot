"""Async TCP JSON-RPC client talking to `blenderbot.bridge.server` inside Blender."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

from blenderbot.blender_ops import OpError
from blenderbot.bridge.protocol import (
    DEFAULT_BRIDGE_HOST,
    DEFAULT_BRIDGE_PORT,
    MAX_LINE_BYTES,
    decode_frame,
    encode_frame,
)

from .base import BlenderTransport, TransportConnectionError


class BridgeTransport(BlenderTransport):
    """Connects over localhost TCP to the stdlib bridge inside GUI or headless Blender."""

    mode = "gui"

    def __init__(
        self,
        *,
        host: str = DEFAULT_BRIDGE_HOST,
        port: int = DEFAULT_BRIDGE_PORT,
        token: str,
        mode_label: str = "gui",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.token = token
        self.mode = mode_label
        self._seq = 0
        self._lock = asyncio.Lock()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def _ensure_open(self, timeout: float) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if self._reader is not None and self._writer is not None and not self._writer.is_closing():
            return self._reader, self._writer
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port, limit=MAX_LINE_BYTES),
                timeout=min(timeout, 5.0),
            )
        except OSError as err:
            raise TransportConnectionError(
                f"Cannot connect to blenderbot Bridge at {self.host}:{self.port} ({err})"
            ) from err
        self._reader = reader
        self._writer = writer
        return reader, writer

    async def _reset_socket(self) -> None:
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def call(
        self,
        op_name: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 120.0,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        suspend_timeout: bool = False,
    ) -> dict[str, Any]:
        async with self._lock:
            self._seq += 1
            req_id = self._seq
            # When `suspend_timeout=True` (local render ops), advertise a 24h ceiling so even an
            # un-restarted GUI Blender bridge loaded prior to streaming progress never aborts at 300s.
            wire_timeout = 86400.0 if suspend_timeout else float(timeout)
            frame = encode_frame(
                {
                    "id": req_id,
                    "token": self.token,
                    "op": op_name,
                    "params": dict(params or {}),
                    "timeout": wire_timeout,
                    "stream_progress": True,
                    "suspend_timeout": bool(suspend_timeout),
                }
            )
            reader, writer = await self._ensure_open(timeout)
            t0 = time.monotonic()
            active_suspend = bool(suspend_timeout)
            last_progress_rx = t0

            async def _local_liveness_ticker() -> None:
                while True:
                    await asyncio.sleep(2.5)
                    now = time.monotonic()
                    if active_suspend and on_progress is not None and (now - last_progress_rx) >= 2.2:
                        with contextlib.suppress(Exception):
                            on_progress(
                                {
                                    "phase": "heartbeat",
                                    "op": op_name,
                                    "elapsed_s": round(now - t0, 1),
                                    "suspend_timeout": True,
                                }
                            )

            ticker_task = (
                asyncio.create_task(_local_liveness_ticker())
                if (suspend_timeout and on_progress is not None)
                else None
            )
            try:
                try:
                    writer.write(frame)
                    await asyncio.wait_for(writer.drain(), timeout=max(5.0, float(timeout)))
                    while True:
                        if active_suspend:
                            raw = await reader.readline()
                        else:
                            raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
                        if not raw:
                            await self._reset_socket()
                            raise TransportConnectionError(
                                "Blender Bridge closed the connection unexpectedly"
                            )
                        resp = decode_frame(raw)
                        if resp.get("event") == "progress":
                            last_progress_rx = time.monotonic()
                            pdata = resp.get("data")
                            if isinstance(pdata, dict):
                                if pdata.get("suspend_timeout"):
                                    active_suspend = True
                                    if ticker_task is None and on_progress is not None:
                                        ticker_task = asyncio.create_task(_local_liveness_ticker())
                                if on_progress is not None:
                                    with contextlib.suppress(Exception):
                                        on_progress(pdata)
                            continue
                        break
                except (TimeoutError, OSError) as err:
                    await self._reset_socket()
                    raise TransportConnectionError(f'Bridge RPC "{op_name}" failed: {err}') from err
            finally:
                if ticker_task is not None:
                    ticker_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ticker_task

            if not resp.get("ok"):
                err_obj = resp.get("error") or {}
                code = str(err_obj.get("code", "OP_ERROR"))
                msg = str(err_obj.get("message", "Unknown bridge error"))
                if code == "UNAUTHORIZED":
                    await self._reset_socket()
                    raise TransportConnectionError(f"Bridge rejected token ({msg})")
                raise OpError(msg, code=code)

            result = resp.get("result")
            return result if isinstance(result, dict) else {"value": result}

    async def close(self) -> None:
        async with self._lock:
            await self._reset_socket()