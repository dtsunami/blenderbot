"""Aggregated FastAPI router composing focused Web Studio sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from .routes_fps import router as fps_router
from .routes_media_gen import router as media_gen_router
from .routes_meshy import router as meshy_router
from .routes_sessions import router as sessions_router
from .routes_splats import router as splats_router
from .routes_state import router as state_router
from .routes_workspace import router as workspace_router

router = APIRouter(prefix="/api")
router.include_router(state_router)
router.include_router(sessions_router)
router.include_router(media_gen_router)
router.include_router(meshy_router)
router.include_router(workspace_router)
router.include_router(splats_router)
router.include_router(fps_router)

__all__ = ["router"]