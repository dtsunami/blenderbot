"""Persistent Warm Headless Blender 5.x Worker (`HeadlessTransport`).

Spawns `blender.exe -b --factory-startup --python-expr ...` running `run_headless_daemon` on an
ephemeral localhost port (`127.0.0.1:0`) with a per-session secret token. Because the process stays
alive for the entire session:
- Scene state (`bpy.data`) persists across turns just like GUI mode.
- Subsequent `BLENDER_WORKBENCH` renders run at ~19ms/frame with zero startup tax.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from blenderbot.bridge.protocol import generate_bridge_token
from blenderbot.config.blender_paths import discover_blender

from .base import BlenderTransport, TransportConnectionError
from .bridge_client import BridgeTransport


class HeadlessTransport(BlenderTransport):
    """Manages a dedicated warm `blender -b` background daemon and delegates RPCs over localhost."""

    mode = "headless"

    def __init__(
        self,
        *,
        blender_exe: str | Path | None = None,
        snaps_dir: str | Path | None = None,
        startup_timeout: float = 20.0,
    ) -> None:
        self._explicit_exe = str(blender_exe) if blender_exe else None
        self._snaps_dir = str(Path(snaps_dir).resolve()) if snaps_dir else None
        self._startup_timeout = startup_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._client: BridgeTransport | None = None
        self.pid: int | None = None
        self.blender_version: str = ""

    async def start(self) -> dict[str, Any]:
        if self._client is not None and self._proc is not None and self._proc.poll() is None:
            return await self._client.call("system.ping", timeout=5.0)

        exe = self._explicit_exe or discover_blender().blender_exe_path
        if not exe or not Path(exe).is_file():
            raise TransportConnectionError(
                "Blender 5.x executable not found (install Blender 5.3+ or set BLENDERBOT_BLENDER_EXE)"
            )

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="blenderbot_worker_")
        tmp_root = Path(self._tmp_dir.name)
        ready_file = tmp_root / "ready.json"
        snaps = self._snaps_dir or str(tmp_root / "snaps")
        token = generate_bridge_token()

        src_root = str(Path(__file__).resolve().parents[2])
        boot_expr = (
            f"import sys; sys.path.insert(0, {src_root!r}); "
            "from blenderbot.bridge.server import run_headless_daemon; "
            f"run_headless_daemon(port=0, token={token!r}, ready_file={str(ready_file)!r}, snaps_dir={snaps!r})"
        )

        self._proc = await asyncio.to_thread(
            subprocess.Popen,
            [str(exe), "-b", "--factory-startup", "--python-expr", boot_expr],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.pid = self._proc.pid

        deadline = time.monotonic() + self._startup_timeout
        info: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                stderr_out = self._proc.stderr.read() if self._proc.stderr else ""
                raise TransportConnectionError(
                    f"Headless Blender exited prematurely (code {self._proc.returncode}): {stderr_out.strip()}"
                )
            if ready_file.is_file():
                try:
                    info = json.loads(ready_file.read_text(encoding="utf-8"))
                    break
                except (OSError, ValueError):
                    pass
            await asyncio.sleep(0.05)

        if not info:
            await self.close()
            raise TransportConnectionError(
                f"Headless Blender worker timed out after {self._startup_timeout:.1f}s waiting for bridge socket"
            )

        self.blender_version = str(info.get("blender_version", ""))
        self._client = BridgeTransport(
            host=str(info["host"]),
            port=int(info["port"]),
            token=token,
            mode_label="headless",
        )
        return await self._client.call("system.ping", timeout=5.0)

    async def call(
        self,
        op_name: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 120.0,
        on_progress: Any = None,
        suspend_timeout: bool = False,
    ) -> dict[str, Any]:
        if self._client is None or self._proc is None or self._proc.poll() is not None:
            await self.start()
        assert self._client is not None
        return await self._client.call(
            op_name,
            params,
            timeout=timeout,
            on_progress=on_progress,
            suspend_timeout=suspend_timeout,
        )

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.call("bridge.shutdown", timeout=2.0)
            await client.close()

        proc = self._proc
        self._proc = None
        if proc is not None:
            if proc.poll() is None:
                try:
                    await asyncio.to_thread(proc.wait, 3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(proc.wait, 2.0)
            if proc.stderr is not None:
                with contextlib.suppress(OSError):
                    proc.stderr.close()

        if self._tmp_dir is not None:
            with contextlib.suppress(OSError):
                self._tmp_dir.cleanup()
            self._tmp_dir = None