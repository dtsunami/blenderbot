"""Shared FastAPI route helpers for the blenderbot Web Studio sidecar."""

from __future__ import annotations

import re
from typing import Any

from fastapi import Request

from .state import WebState

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def get_web_state(req: Request) -> WebState:
    st: WebState = req.app.state.blenderbot
    return st


def refresh_env_keys(cfg: Any) -> None:
    """Hot-harvest newly added `.env` keys so running Studio sessions pick up `.blenderbot/.env` edits live."""
    if cfg.keys.get("fal") and cfg.keys.get("openrouter"):
        return
    from blenderbot.config.env import KEY_VARS
    from blenderbot.config.envfile import read_env

    for env_p in (cfg.home / ".env", cfg.cwd / ".env"):
        if not env_p.is_file():
            continue
        env_map = read_env(env_p)
        for k_prov in ("fal", "openrouter"):
            if cfg.keys.get(k_prov):
                continue
            for cand_var in KEY_VARS.get(k_prov, ()):
                val_c = env_map.get(cand_var, "").strip()
                if val_c:
                    cfg.keys[k_prov] = val_c
                    cfg.key_origins[k_prov] = {"name": cand_var, "from": str(env_p)}
                    break