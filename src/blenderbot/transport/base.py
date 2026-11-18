"""Abstract `BlenderTransport` interface and `auto` connection negotiation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from blenderbot.bridge.protocol import DEFAULT_BRIDGE_HOST, DEFAULT_BRIDGE_PORT, read_discovery
from blenderbot.config.blender_paths import discover_blender


class TransportConnectionError(Exception):
    """Raised when neither GUI Bridge nor Headless Blender worker can be established."""


class BlenderTransport(ABC):
    """Unified async RPC interface over GUI `BridgeTransport` and background `HeadlessTransport`."""

    mode: str

    @abstractmethod
    async def call(
        self, op_name: str, params: dict[str, Any] | None = None, *, timeout: float = 120.0
    ) -> dict[str, Any]:
        """Execute a registered `blender_ops` operation and return its JSON-serializable dict."""

    @abstractmethod
    async def close(self) -> None:
        """Release underlying socket or subprocess resources."""


async def connect_blender(
    *,
    mode: str = "auto",
    home: str | Path | None = None,
    bridge_host: str = DEFAULT_BRIDGE_HOST,
    bridge_port: int = DEFAULT_BRIDGE_PORT,
    bridge_token: str | None = None,
    snaps_dir: str | Path | None = None,
) -> BlenderTransport:
    """Connect to Blender using `"auto"`, `"bridge"`, or `"headless"` mode.

    In `"auto"` mode:
    1. If `~/.blenderbot/bridge.json` exists (or `bridge_token` is supplied), probe live GUI
       `BridgeTransport` first. If alive, return it immediately so edits appear live in viewport!
    2. Otherwise spawn a persistent warm `HeadlessTransport` (`blender.exe -b --factory-startup`)
       so the agent works out-of-the-box with zero manual GUI setup!
    """
    from .bridge_client import BridgeTransport
    from .headless_worker import HeadlessTransport

    want = mode.strip().lower()
    errors: list[str] = []

    if want in ("auto", "bridge", "gui"):
        disc = read_discovery(home)
        eff_host = disc.host if disc else bridge_host
        eff_port = disc.port if disc else bridge_port
        eff_token = bridge_token or (disc.token if disc else None)

        if eff_token:
            bt = BridgeTransport(host=eff_host, port=eff_port, token=eff_token)
            try:
                await bt.call("system.ping", timeout=2.5)
                return bt
            except Exception as err:  # noqa: BLE001
                await bt.close()
                errors.append(f"gui-bridge ({eff_host}:{eff_port}): {err}")
        elif want in ("bridge", "gui"):
            errors.append(
                "gui-bridge: no active ~/.blenderbot/bridge.json found "
                "(run `blenderbot --install-addon`, enable 'blenderbot Bridge' in Blender Preferences -> Add-ons)"
            )

    if want in ("auto", "headless"):
        ht = HeadlessTransport(snaps_dir=snaps_dir)
        try:
            await ht.start()
            return ht
        except Exception as err:  # noqa: BLE001
            await ht.close()
            errors.append(f"headless: {err}")

    inst = discover_blender()
    exe_note = inst.blender_exe_path or "Blender 5.3+ not found on PATH or C:\\Program Files\\Blender Foundation"
    raise TransportConnectionError(
        f"Could not connect to Blender ({'; '.join(errors) or 'no transport'}).\n"
        f"  • Detected Blender binary: {exe_note}\n"
        "  • Set BLENDERBOT_BLENDER_EXE if Blender is installed in a custom path."
    )