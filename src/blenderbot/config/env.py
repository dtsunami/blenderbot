"""`BLENDERBOT_HOME` resolution, `.env` loading (with `~/.davai/.env` & `~/.davag/.env` key fallback), and validated `Config`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .blender_paths import resolve_home
from .envfile import parse_dotenv, read_env, read_env_text
from .models import ModelSpec, resolve_model
from .profiles import parse_blocks

PROVIDERS = ("anthropic", "gemini", "openai", "grok", "openrouter")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
TRANSPORTS = ("auto", "headless", "bridge", "gui")
BASE_URL_PROVIDERS = ("openai", "grok", "openrouter")
HOME_ONLY_VARS = ("BLENDERBOT_BASE_URL", "DAVAG_BASE_URL", "DAVAI_BASE_URL", "OPENAI_BASE_URL")
DEFAULT_MAX_TOKENS = 32_000
DEFAULT_WEB_PORT = 7880
DEFAULT_BRIDGE_PORT = 18766

# Lookup order per provider: BLENDERBOT_* first, standard SDK names, then DAVAG_*/DAVAI_*
# so existing `~/.davai/.env` or `~/.davag/.env` keys work out-of-the-box with zero re-entry!
KEY_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": (
        "BLENDERBOT_ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY",
        "DAVAG_ANTHROPIC_API_KEY",
        "DAVAI_ANTHROPIC_API_KEY",
    ),
    "gemini": (
        "BLENDERBOT_GEMINI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "DAVAG_GEMINI_API_KEY",
        "DAVAI_GEMINI_API_KEY",
    ),
    "openai": (
        "BLENDERBOT_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "DAVAG_OPENAI_API_KEY",
        "DAVAI_OPENAI_API_KEY",
    ),
    "grok": (
        "BLENDERBOT_GROK_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "DAVAG_GROK_API_KEY",
        "DAVAI_GROK_API_KEY",
    ),
    "openrouter": (
        "BLENDERBOT_OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY",
        "DAVAG_OPENROUTER_API_KEY",
        "DAVAI_OPENROUTER_API_KEY",
    ),
}

FALLBACK_KEYS: frozenset[str] = frozenset(var for group in KEY_VARS.values() for var in group)


def key_var_for(provider: str) -> str:
    names = KEY_VARS.get(provider)
    if not names:
        raise ValueError(f'Unknown provider "{provider}"')
    return names[0]


def _resolve_sibling_homes(env: dict[str, str]) -> list[Path]:
    homes: list[Path] = []
    for env_var, folder in (("DAVAI_HOME", ".davai"), ("DAVAG_HOME", ".davag")):
        raw = env.get(env_var, "").strip()
        p = Path(raw).expanduser().resolve() if raw else (Path.home() / folder).resolve()
        if p not in homes:
            homes.append(p)
    return homes


def _parse_bool(v: str | None, default: bool) -> bool:
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _parse_positive_int(v: str | None, name: str) -> int | None:
    if v is None or str(v).strip() == "":
        return None
    cleaned = str(v).strip().replace("_", "")
    try:
        n = int(cleaned)
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer (got "{v}").') from exc
    if n <= 0:
        raise ValueError(f'{name} must be a positive integer (got "{v}").')
    return n


def _parse_port(v: str | None, name: str) -> int | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        n = int(str(v).strip())
    except ValueError as exc:
        raise ValueError(f'{name} must be a port between 0 and 65535 (got "{v}").') from exc
    if n < 0 or n > 65535:
        raise ValueError(f'{name} must be a port between 0 and 65535 (got "{v}").')
    return n


def _parse_float(v: str | None, default: float) -> float:
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(str(v).strip())
    except ValueError:
        return default


def resolve_limits(
    model: ModelSpec,
    *,
    max_tokens: int | None = None,
    context_limit: int | None = None,
) -> tuple[int, int, list[str]]:
    clamped: list[str] = []

    def cap(want: int, limit: int, name: str) -> int:
        if model.unverified or want <= limit:
            return want
        clamped.append(f"{name}={want} exceeds {model.id}'s limit of {limit}; using {limit}")
        return limit

    eff_max = (
        cap(max_tokens, model.max_output, "BLENDERBOT_MAX_TOKENS")
        if max_tokens is not None
        else min(model.max_output, DEFAULT_MAX_TOKENS)
    )
    eff_ctx = (
        cap(context_limit, model.context, "BLENDERBOT_CONTEXT_LIMIT")
        if context_limit is not None
        else model.context
    )
    return eff_max, eff_ctx, clamped


def _resolve_base_url(
    provider: str, merged: dict[str, str], origins: dict[str, str]
) -> tuple[str | None, dict[str, str | None] | None]:
    bb_url = merged.get("BLENDERBOT_BASE_URL", "").strip()
    openai_url = merged.get("OPENAI_BASE_URL", "").strip() if provider == "openai" else ""
    raw = bb_url or openai_url
    if not raw:
        return None, None
    var_name = "BLENDERBOT_BASE_URL" if bb_url else "OPENAI_BASE_URL"
    if provider not in BASE_URL_PROVIDERS:
        supported = ", ".join(BASE_URL_PROVIDERS)
        raise ValueError(
            f'{var_name} is set, but provider "{provider}" does not support a custom base URL '
            f"(supported: {supported})."
        )
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f'{var_name}="{raw}" must be an http or https URL.')
    return raw.rstrip("/"), {"name": var_name, "from": origins.get(var_name)}


def _drop_project_only(
    merged: dict[str, str],
    *,
    home_vars: dict[str, str],
    project_vars: dict[str, str],
    home_env_file: Path,
    cwd_env_file: Path,
    real_env: dict[str, str],
    overrides: dict[str, str],
    origins: dict[str, str],
) -> list[str]:
    warnings: list[str] = []
    for name in HOME_ONLY_VARS:
        val = project_vars.get(name, "").strip()
        if not val:
            continue
        if real_env.get(name) or name in overrides:
            continue
        fallback = home_vars.get(name, "").strip()
        if fallback:
            merged[name] = fallback
            origins[name] = str(home_env_file)
        else:
            merged.pop(name, None)
            origins.pop(name, None)
        warnings.append(
            f"{cwd_env_file} sets {name}={val}, which was ignored: a project .env cannot "
            "choose where your API key is sent."
        )
    return warnings


@dataclass
class Config:
    home: Path
    cwd: Path
    provider: str
    model: ModelSpec
    effort: str
    max_tokens: int
    context_limit: int
    limits_requested: dict[str, int | None]
    temperature: float | None
    compact_at: float
    max_cost: float
    thinking_visible: bool
    yolo: bool
    openai_surface: str
    base_url: str | None
    base_url_origin: dict[str, str | None] | None
    transport: str  # "auto" | "headless" | "bridge" | "gui"
    bridge_host: str
    bridge_port: int
    web_enabled: bool
    web_host: str
    web_port: int
    web_token: str | None
    web_open: bool
    shell_timeout: float
    ro_dirs: list[Path] = field(default_factory=list)
    agents_files: list[str] = field(default_factory=list)
    agents_budget: int | None = None
    role: str = "master"
    image_model: str = "black-forest-labs/flux.2-pro"
    video_model: str = "bytedance/seedance-2.5"
    keys: dict[str, str | None] = field(default_factory=dict)
    key_origins: dict[str, dict[str, str | None]] = field(default_factory=dict)
    var_origins: dict[str, str] = field(default_factory=dict)
    env_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "home": str(self.home),
            "cwd": str(self.cwd),
            "provider": self.provider,
            "model": self.model.id,
            "effort": self.effort,
            "role": self.role,
            "max_tokens": self.max_tokens,
            "context_limit": self.context_limit,
            "temperature": self.temperature,
            "compact_at": self.compact_at,
            "max_cost": self.max_cost,
            "thinking_visible": self.thinking_visible,
            "yolo": self.yolo,
            "openai_surface": self.openai_surface,
            "base_url": self.base_url,
            "transport": self.transport,
            "bridge": {"host": self.bridge_host, "port": self.bridge_port},
            "video_model": self.video_model,
            "web": {
                "enabled": self.web_enabled,
                "host": self.web_host,
                "port": self.web_port,
                "open_on_launch": self.web_open,
                "token": "set" if self.web_token else "ephemeral",
            },
            "keys": {p: ("set" if v else "—") for p, v in self.keys.items()},
            "key_origins": self.key_origins,
            "env_files": self.env_files,
            "warnings": self.warnings,
        }


def _harvest_fallback_keys(
    env_file: Path,
    merged: dict[str, str],
    origins: dict[str, str],
    env_files: list[str],
) -> None:
    raw = read_env_text(env_file)
    if not raw:
        return
    parsed_vars = parse_dotenv(raw)
    adopted: dict[str, str] = {
        k: v for k, v in parsed_vars.items() if k in FALLBACK_KEYS and v
    }
    for blk in parse_blocks(raw)["blocks"]:
        blk_var_names = {v["name"] for v in blk["vars"] if not v["shadowed"]}
        has_custom_base = bool(
            blk_var_names & {"DAVAI_BASE_URL", "OPENAI_BASE_URL", "DAVAG_BASE_URL", "BLENDERBOT_BASE_URL"}
        )
        for v in blk["vars"]:
            if v["shadowed"] or v["name"] not in FALLBACK_KEYS:
                continue
            if has_custom_base and v["name"] in ("OPENAI_API_KEY", "DAVAI_OPENAI_API_KEY", "DAVAG_OPENAI_API_KEY"):
                continue
            if v["name"] not in adopted:
                val_parsed = parse_dotenv(f"{v['name']}={v['value']}").get(v["name"], "")
                if val_parsed:
                    adopted[v["name"]] = val_parsed
    if adopted:
        env_files.append(str(env_file))
        for k, v in adopted.items():
            if k not in merged:
                merged[k] = v
                origins[k] = str(env_file)


def load_config(
    *,
    cwd: str | Path | None = None,
    defaults: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
    environ: dict[str, str] | None = None,
    home: str | Path | None = None,
) -> Config:
    real_env = dict(os.environ if environ is None else environ)
    eff_cwd = Path(cwd or os.getcwd()).resolve()
    eff_home = Path(home).resolve() if home else resolve_home(real_env)

    home_env_file = eff_home / ".env"
    cwd_env_file = eff_cwd / ".env"

    merged: dict[str, str] = {}
    origins: dict[str, str] = {}
    env_files: list[str] = []

    for k, v in (defaults or {}).items():
        if v is not None and v != "":
            merged[k] = str(v)
            origins[k] = "settings.json"

    for sib_home in _resolve_sibling_homes(real_env):
        sib_env = sib_home / ".env"
        if sib_env != home_env_file:
            _harvest_fallback_keys(sib_env, merged, origins, env_files)

    home_vars = read_env(home_env_file)
    if home_vars:
        env_files.append(str(home_env_file))
        for k, v in home_vars.items():
            merged[k] = v
            origins[k] = str(home_env_file)

    project_vars: dict[str, str] = {}
    if cwd_env_file != home_env_file:
        project_vars = read_env(cwd_env_file)
        if project_vars:
            env_files.append(str(cwd_env_file))
            for k, v in project_vars.items():
                merged[k] = v
                origins[k] = str(cwd_env_file)

    for k, v in real_env.items():
        if v is not None and v != "":
            merged[k] = v
            origins[k] = "environment"

    eff_overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    for k, v in eff_overrides.items():
        merged[k] = str(v)
        origins[k] = "command-line flag"

    warnings = (
        []
        if cwd_env_file == home_env_file
        else _drop_project_only(
            merged,
            home_vars=home_vars,
            project_vars=project_vars,
            home_env_file=home_env_file,
            cwd_env_file=cwd_env_file,
            real_env=real_env,
            overrides=eff_overrides,
            origins=origins,
        )
    )

    keys: dict[str, str | None] = {}
    key_origins: dict[str, dict[str, str | None]] = {}
    for prov, candidates in KEY_VARS.items():
        found_val: str | None = None
        for var in candidates:
            val = merged.get(var, "").strip()
            if val:
                found_val = val
                key_origins[prov] = {"name": var, "from": origins.get(var)}
                break
        keys[prov] = found_val

    # Smart default provider selection if not explicitly specified: prefer configured key
    explicit_prov = merged.get("BLENDERBOT_PROVIDER", "").strip().lower()
    if explicit_prov:
        provider = explicit_prov
    else:
        provider = next(
            (p for p in ("gemini", "anthropic", "openai", "openrouter", "grok") if keys.get(p)),
            "gemini",
        )
    if provider not in PROVIDERS:
        raise ValueError(
            f'Unknown BLENDERBOT_PROVIDER="{provider}" (expected one of: {", ".join(PROVIDERS)}).'
        )

    model = resolve_model(provider, merged.get("BLENDERBOT_MODEL"))

    effort = merged.get("BLENDERBOT_EFFORT", "high").strip().lower()
    if effort not in EFFORTS:
        raise ValueError(
            f'Invalid BLENDERBOT_EFFORT="{effort}" (expected one of: {", ".join(EFFORTS)}).'
        )

    req_max = _parse_positive_int(merged.get("BLENDERBOT_MAX_TOKENS"), "BLENDERBOT_MAX_TOKENS")
    req_ctx = _parse_positive_int(merged.get("BLENDERBOT_CONTEXT_LIMIT"), "BLENDERBOT_CONTEXT_LIMIT")
    max_tokens, context_limit, clamp_notes = resolve_limits(
        model, max_tokens=req_max, context_limit=req_ctx
    )
    warnings.extend(clamp_notes)

    temp_raw = merged.get("BLENDERBOT_TEMPERATURE", "").strip()
    temperature: float | None = None
    if temp_raw:
        try:
            temperature = float(temp_raw)
        except ValueError as exc:
            raise ValueError(f'BLENDERBOT_TEMPERATURE must be a number (got "{temp_raw}").') from exc

    compact_at = _parse_float(merged.get("BLENDERBOT_COMPACT_AT"), 0.75)
    if not (0.2 <= compact_at <= 0.98):
        raise ValueError(f"BLENDERBOT_COMPACT_AT must be between 0.2 and 0.98 (got {compact_at}).")

    max_cost = _parse_float(merged.get("BLENDERBOT_MAX_COST"), 0.0)
    thinking_visible = _parse_bool(merged.get("BLENDERBOT_THINKING"), True)
    yolo = _parse_bool(merged.get("BLENDERBOT_YOLO"), False)

    base_url, base_url_origin = _resolve_base_url(provider, merged, origins)

    surface_raw = merged.get("BLENDERBOT_OPENAI_SURFACE", "").strip().lower()
    if surface_raw:
        if surface_raw not in ("responses", "chat"):
            raise ValueError(
                f'BLENDERBOT_OPENAI_SURFACE must be "responses" or "chat" (got "{surface_raw}").'
            )
        openai_surface = surface_raw
    else:
        openai_surface = "chat" if base_url else "responses"

    transport = merged.get("BLENDERBOT_TRANSPORT", "auto").strip().lower()
    if transport not in TRANSPORTS:
        raise ValueError(
            f'Invalid BLENDERBOT_TRANSPORT="{transport}" (expected one of: {", ".join(TRANSPORTS)}).'
        )

    bridge_host = merged.get("BLENDERBOT_BRIDGE_HOST", "127.0.0.1").strip() or "127.0.0.1"
    bridge_port = _parse_port(merged.get("BLENDERBOT_BRIDGE_PORT"), "BLENDERBOT_BRIDGE_PORT")
    if bridge_port is None:
        bridge_port = DEFAULT_BRIDGE_PORT

    web_open = _parse_bool(merged.get("BLENDERBOT_WEB_OPEN"), False)
    web_enabled = _parse_bool(merged.get("BLENDERBOT_WEB"), False) or web_open
    web_host = merged.get("BLENDERBOT_WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    web_port = _parse_port(merged.get("BLENDERBOT_WEB_PORT"), "BLENDERBOT_WEB_PORT")
    if web_port is None:
        web_port = DEFAULT_WEB_PORT
    web_token = merged.get("BLENDERBOT_WEB_TOKEN", "").strip() or None

    shell_timeout = _parse_float(merged.get("BLENDERBOT_SHELL_TIMEOUT"), 120.0)

    ro_raw = merged.get("BLENDERBOT_RO_DIRS") or merged.get("DAVAI_RO_DIRS") or ""
    ro_dirs = [Path(p.strip()).resolve() for p in ro_raw.split(os.pathsep) if p.strip()]

    agents_md_raw = merged.get("BLENDERBOT_AGENTS_MD", "").strip()
    agents_files = [p.strip() for p in agents_md_raw.split(os.pathsep) if p.strip()]
    agents_budget = _parse_positive_int(merged.get("BLENDERBOT_AGENTS_BUDGET"), "BLENDERBOT_AGENTS_BUDGET")

    role = merged.get("BLENDERBOT_ROLE", "master").strip().lower() or "master"
    from blenderbot.agent.roles import ROLES

    if role not in ROLES:
        raise ValueError(f'Unknown BLENDERBOT_ROLE="{role}" (expected one of: {", ".join(ROLES)}).')

    image_model = (
        merged.get("BLENDERBOT_IMAGE_MODEL", "black-forest-labs/flux.2-pro").strip()
        or "black-forest-labs/flux.2-pro"
    )
    video_model = (
        merged.get("BLENDERBOT_VIDEO_MODEL", "bytedance/seedance-2.5").strip()
        or "bytedance/seedance-2.5"
    )

    return Config(
        home=eff_home,
        cwd=eff_cwd,
        provider=provider,
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        context_limit=context_limit,
        limits_requested={"max_tokens": req_max, "context_limit": req_ctx},
        temperature=temperature,
        compact_at=compact_at,
        max_cost=max_cost,
        thinking_visible=thinking_visible,
        yolo=yolo,
        openai_surface=openai_surface,
        base_url=base_url,
        base_url_origin=base_url_origin,
        transport=transport,
        bridge_host=bridge_host,
        bridge_port=bridge_port,
        web_enabled=web_enabled,
        web_host=web_host,
        web_port=web_port,
        web_token=web_token,
        web_open=web_open,
        shell_timeout=shell_timeout,
        ro_dirs=ro_dirs,
        agents_files=agents_files,
        agents_budget=agents_budget,
        role=role,
        image_model=image_model,
        video_model=video_model,
        keys=keys,
        key_origins=key_origins,
        var_origins=origins,
        env_files=env_files,
        warnings=warnings,
    )