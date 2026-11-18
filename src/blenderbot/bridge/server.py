"""In-Blender TCP JSON-RPC server + main-thread queue dispatcher (stdlib only).

Supports BOTH execution modes with the exact same wire protocol and op handlers:
1. **GUI Mode (`run_in_blender_gui`):**
   - Background `ThreadingTCPServer` accepts socket frames and enqueues `_PendingRPC` items.
   - `bpy.app.timers.register(drain_timer, persistent=True)` drains RPCs every ~16ms strictly
     on Blender's main UI thread (`persistent=True` survives `File -> Open` / `wm.open_mainfile`!).
2. **Headless Worker Mode (`run_headless_daemon`):**
   - Runs inside `blender.exe -b --factory-startup --python ...` so `bpy.data` stays warm in RAM
     across an entire session (`~19ms/frame` Workbench renders, zero cold-boot overhead per op!).
"""

from __future__ import annotations

import os
import queue
import socketserver
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blenderbot.blender_ops import OpContext, OpError, dispatch_op

from .protocol import (
    DEFAULT_BRIDGE_HOST,
    DEFAULT_BRIDGE_PORT,
    MAX_LINE_BYTES,
    BridgeDiscovery,
    decode_frame,
    encode_frame,
    generate_bridge_token,
    remove_discovery_if_owned,
    verify_bridge_token,
    write_discovery,
)


@dataclass
class _PendingRPC:
    op: str
    params: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] = field(default_factory=dict)


class _BridgeTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[socketserver.BaseRequestHandler],
        bridge: BlenderBridgeServer,
    ) -> None:
        self.bridge = bridge
        super().__init__(server_address, handler_cls)


class _BridgeConnectionHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        bridge: BlenderBridgeServer = self.server.bridge  # type: ignore[attr-defined]
        while not bridge.stopped:
            try:
                raw_line = self.rfile.readline(MAX_LINE_BYTES + 1)
            except OSError:
                break
            if not raw_line:
                break

            req_id: Any = None
            try:
                msg = decode_frame(raw_line)
                req_id = msg.get("id")
                if not verify_bridge_token(bridge.token, msg.get("token")):
                    self._send(
                        {
                            "id": req_id,
                            "ok": False,
                            "error": {
                                "code": "UNAUTHORIZED",
                                "message": "Invalid or missing bridge token",
                            },
                        }
                    )
                    continue

                op_name = str(msg.get("op", "")).strip()
                if op_name == "bridge.shutdown":
                    self._send({"id": req_id, "ok": True, "result": {"shutting_down": True}})
                    bridge.request_shutdown()
                    break

                params = msg.get("params") or {}
                timeout = float(msg.get("timeout", 120.0))
                resp = bridge.enqueue_and_wait(op_name, params, timeout=timeout)
                self._send({"id": req_id, **resp})
            except Exception as err:  # noqa: BLE001
                self._send(
                    {
                        "id": req_id,
                        "ok": False,
                        "error": {"code": "PROTOCOL_ERROR", "message": str(err)},
                    }
                )

    def _send(self, payload: dict[str, Any]) -> None:
        try:
            self.wfile.write(encode_frame(payload))
            self.wfile.flush()
        except OSError:
            pass


class BlenderBridgeServer:
    """Thread-safe localhost TCP server bridging external async clients to Blender's main thread."""

    def __init__(
        self,
        *,
        bpy_mod: Any,
        mode: str = "gui",
        host: str = DEFAULT_BRIDGE_HOST,
        port: int = DEFAULT_BRIDGE_PORT,
        token: str | None = None,
        home: str | Path | None = None,
        discovery_filename: str | None = "bridge.json",
        snaps_dir: str | None = None,
    ) -> None:
        self.bpy = bpy_mod
        self.mode = mode
        self.host = host
        self.port = int(port)
        self.token = token or generate_bridge_token()
        self.home = Path(home).resolve() if home else None
        self.discovery_filename = discovery_filename
        self.snaps_dir = snaps_dir

        self._queue: queue.Queue[_PendingRPC] = queue.Queue()
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._tcp: _BridgeTCPServer | None = None
        self._thread: threading.Thread | None = None
        self.ops_executed = 0

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    @property
    def bound_port(self) -> int:
        if self._tcp is not None:
            return int(self._tcp.server_address[1])
        return self.port

    def start_background_listener(self) -> BridgeDiscovery:
        self._tcp = _BridgeTCPServer((self.host, self.port), _BridgeConnectionHandler, self)
        self.port = int(self._tcp.server_address[1])
        self._thread = threading.Thread(
            target=self._tcp.serve_forever,
            name="blenderbot-bridge-tcp",
            daemon=True,
        )
        self._thread.start()

        info = BridgeDiscovery(
            host=self.host,
            port=self.port,
            token=self.token,
            pid=os.getpid(),
            started_at=time.time(),
            mode=self.mode,
            blender_version=str(getattr(self.bpy.app, "version_string", "")),
        )
        if self.discovery_filename:
            write_discovery(info, self.home, filename=self.discovery_filename)
        return info

    def request_shutdown(self) -> None:
        self._stopped.set()
        self._wake.set()

    def enqueue_and_wait(
        self, op_name: str, params: dict[str, Any], *, timeout: float = 120.0
    ) -> dict[str, Any]:
        if self.stopped:
            return {
                "ok": False,
                "error": {"code": "STOPPED", "message": "Bridge server is shutting down"},
            }
        rpc = _PendingRPC(op=op_name, params=params)
        self._queue.put(rpc)
        self._wake.set()
        if not rpc.done.wait(timeout=max(1.0, timeout)):
            return {
                "ok": False,
                "error": {
                    "code": "TIMEOUT",
                    "message": f'Main-thread execution of "{op_name}" timed out after {timeout:.1f}s',
                },
            }
        return rpc.response

    def drain_pending(self, *, max_items: int = 32) -> int:
        """Execute queued RPCs strictly on Blender's main thread."""
        processed = 0
        ctx = OpContext(bpy=self.bpy, mode=self.mode, snaps_dir=self.snaps_dir)
        while processed < max_items:
            try:
                rpc = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                res = dispatch_op(ctx, rpc.op, rpc.params)
                rpc.response = {"ok": True, "result": res}
                self.ops_executed += 1
            except OpError as op_err:
                rpc.response = {
                    "ok": False,
                    "error": {"code": op_err.code, "message": str(op_err)},
                }
            except Exception as exc:  # noqa: BLE001
                rpc.response = {
                    "ok": False,
                    "error": {"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"},
                }
            finally:
                rpc.done.set()
                processed += 1
        return processed

    def close(self) -> None:
        self._stopped.set()
        self._wake.set()
        if self._tcp is not None:
            try:
                self._tcp.shutdown()
                self._tcp.server_close()
            except OSError:
                pass
            self._tcp = None
        if self.discovery_filename:
            remove_discovery_if_owned(os.getpid(), self.home, filename=self.discovery_filename)


_ACTIVE_GUI_SERVER: BlenderBridgeServer | None = None


def start_in_blender_gui(
    *,
    host: str = DEFAULT_BRIDGE_HOST,
    port: int = DEFAULT_BRIDGE_PORT,
    home: str | Path | None = None,
) -> BridgeDiscovery:
    """Start `BlenderBridgeServer` inside interactive GUI Blender via `bpy.app.timers`."""
    import bpy

    global _ACTIVE_GUI_SERVER
    if _ACTIVE_GUI_SERVER is not None and not _ACTIVE_GUI_SERVER.stopped:
        stop_in_blender_gui()

    srv = BlenderBridgeServer(
        bpy_mod=bpy,
        mode="gui",
        host=host,
        port=port,
        home=home,
        discovery_filename="bridge.json",
    )
    info = srv.start_background_listener()
    _ACTIVE_GUI_SERVER = srv

    def _timer_callback() -> float | None:
        if srv.stopped:
            srv.close()
            return None
        srv.drain_pending()
        return 0.016  # ~60 Hz main-thread poll

    bpy.app.timers.register(_timer_callback, first_interval=0.01, persistent=True)
    return info


def stop_in_blender_gui() -> None:
    global _ACTIVE_GUI_SERVER
    if _ACTIVE_GUI_SERVER is not None:
        _ACTIVE_GUI_SERVER.close()
        _ACTIVE_GUI_SERVER = None


def get_gui_bridge_status() -> dict[str, Any]:
    srv = _ACTIVE_GUI_SERVER
    if srv is None or srv.stopped:
        return {"running": False, "port": None, "ops_executed": 0}
    return {
        "running": True,
        "host": srv.host,
        "port": srv.bound_port,
        "ops_executed": srv.ops_executed,
    }


def run_headless_daemon(
    *,
    host: str = DEFAULT_BRIDGE_HOST,
    port: int = 0,
    token: str | None = None,
    ready_file: str | Path | None = None,
    snaps_dir: str | None = None,
) -> None:
    """Blocking main-thread RPC pump used by `HeadlessTransport` inside `blender.exe -b`."""
    import json

    import bpy

    srv = BlenderBridgeServer(
        bpy_mod=bpy,
        mode="headless",
        host=host,
        port=port,
        token=token,
        discovery_filename=None,  # Headless workers report directly via `ready_file`
        snaps_dir=snaps_dir,
    )
    info = srv.start_background_listener()
    if ready_file:
        rp = Path(ready_file).resolve()
        rp.parent.mkdir(parents=True, exist_ok=True)
        tmp = rp.with_name(f"{rp.name}.tmp")
        tmp.write_text(json.dumps(info.to_dict()) + "\n", encoding="utf-8")
        os.replace(tmp, rp)

    try:
        while not srv.stopped:
            srv._wake.wait(timeout=0.05)
            srv._wake.clear()
            srv.drain_pending()
    finally:
        srv.close()