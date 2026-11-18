"""Remembered preferences persisted to `$BLENDERBOT_HOME/settings.json`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIELDS = (
    "provider",
    "model",
    "effort",
    "max_tokens",
    "temperature",
    "thinking_visible",
    "compact_at",
    "transport",
    "role",
)


def settings_path(home: str | Path) -> Path:
    return Path(home) / "settings.json"


def load_settings(home: str | Path) -> dict[str, Any]:
    try:
        raw = settings_path(home).read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        return {k: v for k, v in parsed.items() if k in FIELDS}
    except (OSError, ValueError):
        return {}


def save_settings(home: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    home_path = Path(home)
    merged = dict(load_settings(home_path))
    for k, v in updates.items():
        if k not in FIELDS:
            continue
        if v is None:
            merged.pop(k, None)
        else:
            merged[k] = v
    home_path.mkdir(parents=True, exist_ok=True)
    settings_path(home_path).write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return merged


def settings_to_defaults(settings: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    if settings.get("provider"):
        out["BLENDERBOT_PROVIDER"] = str(settings["provider"])
    if settings.get("model"):
        out["BLENDERBOT_MODEL"] = str(settings["model"])
    if settings.get("effort"):
        out["BLENDERBOT_EFFORT"] = str(settings["effort"])
    if settings.get("max_tokens") is not None:
        out["BLENDERBOT_MAX_TOKENS"] = str(settings["max_tokens"])
    if settings.get("temperature") is not None:
        out["BLENDERBOT_TEMPERATURE"] = str(settings["temperature"])
    if settings.get("compact_at") is not None:
        out["BLENDERBOT_COMPACT_AT"] = str(settings["compact_at"])
    if settings.get("thinking_visible") is not None:
        out["BLENDERBOT_THINKING"] = "true" if settings["thinking_visible"] else "false"
    if settings.get("transport"):
        out["BLENDERBOT_TRANSPORT"] = str(settings["transport"])
    if settings.get("role"):
        out["BLENDERBOT_ROLE"] = str(settings["role"])
    return out