"""FastAPI + Uvicorn Web Studio & Seedance 2.5 GUI Sidecar (`blenderbot.server.http`)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import sys
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from blenderbot.context.utf8 import sanitize_utf8, sanitize_utf8_str
from blenderbot.session.bootstrap import Session
from blenderbot.session.dispatch import create_dispatcher

from .auth import assert_safe_bind, make_web_token, read_request_token, token_matches
from .routes import router as api_router
from .state import WebState

WEB_STATIC_DIR = Path(__file__).resolve().parents[1] / "ui" / "web"
_PROACTOR_PATCHED = False
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


def is_benign_socket_disconnect(exc: BaseException | None) -> bool:
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return True
    return bool(
        isinstance(exc, OSError)
        and getattr(exc, "winerror", None) in (64, 121, 995, 10038, 10053, 10054)
    )


def install_win32_proactor_reset_guard() -> None:
    """Prevent CPython Win32 Proactor `WinError 10054` when browsers scrub `<video>` streams."""
    global _PROACTOR_PATCHED
    if _PROACTOR_PATCHED or sys.platform != "win32":
        return
    _PROACTOR_PATCHED = True
    with contextlib.suppress(Exception):
        from asyncio import proactor_events

        cls = proactor_events._ProactorBasePipeTransport

        def _safe_call_connection_lost(self: Any, exc: Exception | None) -> None:
            if self._called_connection_lost:
                return
            try:
                self._protocol.connection_lost(exc)
            finally:
                if hasattr(self._sock, "shutdown") and self._sock.fileno() != -1:
                    with contextlib.suppress(OSError):
                        self._sock.shutdown(socket.SHUT_RDWR)
                if self._sock is not None:
                    with contextlib.suppress(OSError):
                        self._sock.close()
                    self._sock = None
                server = self._server
                if server is not None:
                    # Python 3.13+ `Server._detach()` takes 0 args; Python <=3.12 took `(self)`.
                    try:
                        server._detach()
                    except TypeError:
                        server._detach(self)
                    self._server = None
                self._called_connection_lost = True

        cls._call_connection_lost = _safe_call_connection_lost  # type: ignore[method-assign]


@dataclass
class RunningWebServer:
    host: str
    port: int
    token: str
    state: WebState
    server: uvicorn.Server
    serve_task: asyncio.Task[None]
    sock: socket.socket
    url: str
    auth_url: str

    async def close(self) -> None:
        self.state.shutdown_ev.set()
        for q in list(self.state.sse_queues):
            q.put_nowait(None)
        for rec in list(self.state.jobs.values()):
            rec.cancel_ev.set()
            if rec.task and not rec.task.done():
                rec.task.cancel()
        self.server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(self.serve_task, timeout=3.0)
        with contextlib.suppress(OSError):
            self.sock.close()


class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        text = json.dumps(
            sanitize_utf8(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return sanitize_utf8_str(text).encode("utf-8", "replace")


def create_fastapi_app(state: WebState) -> FastAPI:
    app = FastAPI(
        title="blenderbot Seedance & 3D Studio",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        default_response_class=SafeJSONResponse,
    )
    app.state.blenderbot = state

    @app.middleware("http")
    async def bearer_auth_middleware(
        req: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        path = req.url.path
        if (
            path in ("/", "/fps")
            or path.startswith("/assets/")
            or path in ("/favicon.svg", "/favicon.ico")
        ):
            res_open: Response = await call_next(req)
            return res_open

        allow_q = path in (
            "/api/events",
            "/api/media/file",
            "/api/media/poster",
            "/api/fps/default_splat",
        )
        given = read_request_token(req, allow_query=allow_q)
        if not token_matches(state.secret, given):
            return JSONResponse({"error": "bad or missing token"}, status_code=401)
        res_authed: Response = await call_next(req)
        return res_authed

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(WEB_STATIC_DIR / "index.html", headers=_NO_STORE_HEADERS)

    @app.get("/fps", include_in_schema=False)
    async def serve_fps_studio() -> FileResponse:
        return FileResponse(WEB_STATIC_DIR / "index.html", headers=_NO_STORE_HEADERS)

    @app.get("/favicon.svg", include_in_schema=False)
    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon() -> FileResponse:
        return FileResponse(WEB_STATIC_DIR / "favicon.svg")

    @app.get("/assets/{name}", include_in_schema=False)
    async def serve_asset(name: str) -> Response:
        target = (WEB_STATIC_DIR / Path(name).name).resolve()
        if target.parent != WEB_STATIC_DIR or not target.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(target, headers=_NO_STORE_HEADERS)

    app.include_router(api_router, default_response_class=SafeJSONResponse)
    return app


async def start_web_server(
    *,
    session: Session,
    submit_fn: Callable[[str, str], Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 7880,
    token: str | None = None,
) -> RunningWebServer:
    assert_safe_bind(host=host, yolo=session.cfg.yolo)
    install_win32_proactor_reset_guard()
    loop = asyncio.get_running_loop()
    prev_eh = loop.get_exception_handler()

    def _loop_eh(lp: asyncio.AbstractEventLoop, ctx: dict[str, Any]) -> None:
        if is_benign_socket_disconnect(ctx.get("exception")):
            return
        if prev_eh is not None:
            prev_eh(lp, ctx)
        else:
            lp.default_exception_handler(ctx)

    loop.set_exception_handler(_loop_eh)
    eff_submit = submit_fn or create_dispatcher(session=session)
    secret = token or session.cfg.web_token or make_web_token()
    state = WebState(session=session, submit_fn=eff_submit, secret=secret)

    app = create_fastapi_app(state)
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        # If default port 7880 is already occupied, bind an ephemeral port cleanly
        sock.bind((host, 0))
    sock.setblocking(False)
    actual_port = int(sock.getsockname()[1])

    uv_cfg = uvicorn.Config(
        app, host=host, port=actual_port, log_level="critical", access_log=False
    )
    server = uvicorn.Server(uv_cfg)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    serve_task = asyncio.create_task(server.serve(sockets=[sock]))

    for _ in range(100):
        if server.started or serve_task.done():
            break
        await asyncio.sleep(0.01)

    disp_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{disp_host}:{actual_port}"
    q_tok = quote(secret, safe="")
    auth_url = f"{url}/?token={q_tok}"

    running = RunningWebServer(
        host=host,
        port=actual_port,
        token=secret,
        state=state,
        server=server,
        serve_task=serve_task,
        sock=sock,
        url=url,
        auth_url=auth_url,
    )
    session.web_server = running
    return running


async def ensure_web_sidecar(
    session: Session,
    *,
    submit_fn: Callable[[str, str], Any] | None = None,
    open_browser: bool = False,
) -> RunningWebServer:
    """Start the GUI sidecar if not already running on `session`, and optionally open browser."""
    existing: RunningWebServer | None = getattr(session, "web_server", None)
    if existing is not None:
        if open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(existing.auth_url)
        return existing

    srv = await start_web_server(
        session=session,
        submit_fn=submit_fn,
        host=session.cfg.web_host,
        port=session.cfg.web_port,
        token=session.cfg.web_token,
    )
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(srv.auth_url)
    return srv