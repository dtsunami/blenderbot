"""FastAPI + Uvicorn Web Studio & Seedance 2.5 GUI Sidecar (`blenderbot.server`)."""

from .http import RunningWebServer, ensure_web_sidecar, start_web_server

__all__ = ["RunningWebServer", "ensure_web_sidecar", "start_web_server"]