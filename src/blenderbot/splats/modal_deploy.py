"""Versioned SHA-256 source hashing, remote deployment inspection, and auto-deploy for Modal (`blenderbot-splat`).

Computes a deterministic SHA-256 digest (`current_hash`) over all Python modules packaged into the
Modal worker image (`src/blenderbot/splats/*.py`), compares it against the live `source_sha256` tag
stored on the deployed Modal App (`modal.App.lookup("blenderbot-splat").get_tags()`), and automatically
triggers `app.deploy(tag=short_hash)` + `app.set_tags({"source_sha256": full_hash, ...})` whenever
local worker code has drifted ahead of the deployed cloud app!
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

APP_NAME = "blenderbot-splat"
TAG_SOURCE_SHA = "source_sha256"
TAG_SOURCE_SHORT = "source_short"
TAG_DEPLOYED_AT = "deployed_at_utc"

# Bundled worker source files whose changes alter remote Modal execution behavior
_HASHED_SPLAT_FILES: tuple[str, ...] = (
    "modal_app.py",
    "core_worker.py",
    "depth_pro2_bridge.py",
    "config.py",
    "sim3.py",
    "proxy_mesh.py",
    "blender_render.py",
)

_deploy_lock: asyncio.Lock | None = None
_cached_verified_hash: str | None = None
_cached_status: dict[str, Any] | None = None
_cached_status_ts: float = 0.0


def _get_deploy_lock() -> asyncio.Lock:
    global _deploy_lock
    if _deploy_lock is None:
        _deploy_lock = asyncio.Lock()
    return _deploy_lock


def compute_local_modal_hash(splats_dir: Path | None = None) -> tuple[str, dict[str, str]]:
    """Return `(full_sha256, per_file_short_sha)` normalized to LF line endings across OSes."""
    base = splats_dir or Path(__file__).resolve().parent
    h_all = hashlib.sha256()
    per_file: dict[str, str] = {}
    for fname in _HASHED_SPLAT_FILES:
        fpath = base / fname
        if not fpath.is_file():
            continue
        # Normalize CRLF -> LF so Windows and Linux compute identical source digests
        norm_bytes = fpath.read_bytes().replace(b"\r\n", b"\n")
        f_hex = hashlib.sha256(norm_bytes).hexdigest()
        per_file[fname] = f_hex[:12]
        h_all.update(f"{fname}:{f_hex}\n".encode("ascii"))
    # Truncate to 48 hex chars (192 bits — zero collision risk & fits Modal's <=63-char tag rule!)
    digest_48 = h_all.hexdigest()[:48]
    return digest_48, per_file


@dataclass
class ModalVersionStatus:
    app_name: str
    current_hash: str
    current_short: str
    deployed_hash: str | None
    deployed_short: str | None
    app_id: str | None
    deployed_at_utc: str | None
    up_to_date: bool
    configured: bool
    file_hashes: dict[str, str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_modal_version_sync(*, use_cache_sec: float = 0.0) -> ModalVersionStatus:
    """Query Modal for the deployed `blenderbot-splat` app tags and compare against `current_hash`."""
    global _cached_status, _cached_status_ts

    cur_hash, per_file = compute_local_modal_hash()
    cur_short = cur_hash[:12]

    now = time.monotonic()
    if (
        use_cache_sec > 0.0
        and _cached_status is not None
        and (now - _cached_status_ts) < use_cache_sec
        and _cached_status.get("current_hash") == cur_hash
    ):
        return ModalVersionStatus(**_cached_status)

    try:
        import modal
    except Exception as exc:  # noqa: BLE001
        st = ModalVersionStatus(
            app_name=APP_NAME,
            current_hash=cur_hash,
            current_short=cur_short,
            deployed_hash=None,
            deployed_short=None,
            app_id=None,
            deployed_at_utc=None,
            up_to_date=False,
            configured=False,
            file_hashes=per_file,
            error=f"modal package unavailable: {exc}",
        )
        return st

    try:
        remote_app = modal.App.lookup(APP_NAME, create_if_missing=False)
        tags = dict(remote_app.get_tags() or {})
        dep_hash = tags.get(TAG_SOURCE_SHA) or None
        dep_short = dep_hash[:12] if dep_hash else (tags.get(TAG_SOURCE_SHORT) or "untagged")
        up_to_date = dep_hash == cur_hash
        st = ModalVersionStatus(
            app_name=APP_NAME,
            current_hash=cur_hash,
            current_short=cur_short,
            deployed_hash=dep_hash,
            deployed_short=dep_short,
            app_id=getattr(remote_app, "app_id", None),
            deployed_at_utc=tags.get(TAG_DEPLOYED_AT),
            up_to_date=up_to_date,
            configured=True,
            file_hashes=per_file,
            error=None,
        )
    except modal.exception.NotFoundError:
        st = ModalVersionStatus(
            app_name=APP_NAME,
            current_hash=cur_hash,
            current_short=cur_short,
            deployed_hash=None,
            deployed_short=None,
            app_id=None,
            deployed_at_utc=None,
            up_to_date=False,
            configured=True,
            file_hashes=per_file,
            error="App not yet deployed",
        )
    except modal.exception.AuthError as exc:
        st = ModalVersionStatus(
            app_name=APP_NAME,
            current_hash=cur_hash,
            current_short=cur_short,
            deployed_hash=None,
            deployed_short=None,
            app_id=None,
            deployed_at_utc=None,
            up_to_date=False,
            configured=False,
            file_hashes=per_file,
            error=f"Modal auth not configured ({exc})",
        )
    except Exception as exc:  # noqa: BLE001
        st = ModalVersionStatus(
            app_name=APP_NAME,
            current_hash=cur_hash,
            current_short=cur_short,
            deployed_hash=None,
            deployed_short=None,
            app_id=None,
            deployed_at_utc=None,
            up_to_date=False,
            configured=False,
            file_hashes=per_file,
            error=str(exc),
        )

    _cached_status = st.to_dict()
    _cached_status_ts = now
    return st


def deploy_modal_app_sync(*, force: bool = False) -> dict[str, Any]:
    """Deploy `blenderbot.splats.modal_app:app` via Modal SDK and stamp `source_sha256` on the app."""
    global _cached_verified_hash, _cached_status, _cached_status_ts

    t0 = time.perf_counter()
    before = check_modal_version_sync(use_cache_sec=0.0)
    if before.up_to_date and not force:
        _cached_verified_hash = before.current_hash
        return {
            "deployed": False,
            "reason": "already_up_to_date",
            "elapsed_s": round(time.perf_counter() - t0, 3),
            **before.to_dict(),
        }

    from blenderbot.splats import modal_app

    cur_hash = before.current_hash
    cur_short = before.current_short
    # Modal tag values allow `[A-Za-z0-9._-]` (no colons) up to 63 chars
    utc_now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")

    deployed_app = modal_app.app.deploy(
        name=APP_NAME,
        tag=f"sha-{cur_short}",
    )
    deployed_app.set_tags(
        {
            TAG_SOURCE_SHA: cur_hash,
            TAG_SOURCE_SHORT: cur_short,
            TAG_DEPLOYED_AT: utc_now,
        }
    )

    _cached_verified_hash = cur_hash
    after = ModalVersionStatus(
        app_name=APP_NAME,
        current_hash=cur_hash,
        current_short=cur_short,
        deployed_hash=cur_hash,
        deployed_short=cur_short,
        app_id=getattr(deployed_app, "app_id", before.app_id),
        deployed_at_utc=utc_now,
        up_to_date=True,
        configured=True,
        file_hashes=before.file_hashes,
        error=None,
    )
    _cached_status = after.to_dict()
    _cached_status_ts = time.monotonic()

    return {
        "deployed": True,
        "previous_hash": before.deployed_hash,
        "previous_short": before.deployed_short,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        **after.to_dict(),
    }


async def ensure_modal_deployed(
    *,
    force: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Ensure the remote Modal `blenderbot-splat` app matches local `current_hash`, auto-deploying if drifted.

    Fast-paths in 0ms once `current_hash` has been verified in this process (unless local files are edited!),
    and skips gracefully during unit tests when `modal.Cls.from_name` is mocked or `BLENDERBOT_MODAL_AUTODEPLOY=0`.
    """
    global _cached_verified_hash

    if os.environ.get("BLENDERBOT_MODAL_AUTODEPLOY", "1").strip().lower() in ("0", "false", "no", "off"):
        cur_hash, per_file = compute_local_modal_hash()
        return {"deployed": False, "skipped": "disabled_by_env", "current_hash": cur_hash, "current_short": cur_hash[:12], "file_hashes": per_file}

    # Skip network check/deploy when `modal.Cls.from_name` is patched in offline unit tests
    try:
        import modal

        if getattr(modal.Cls.from_name, "__module__", "").startswith("unittest.mock"):
            cur_hash, _ = compute_local_modal_hash()
            return {"deployed": False, "skipped": "mocked_cls", "current_hash": cur_hash, "current_short": cur_hash[:12]}
    except Exception:  # noqa: BLE001, S110
        pass

    cur_hash, _ = compute_local_modal_hash()
    if not force and _cached_verified_hash == cur_hash:
        return {"deployed": False, "up_to_date": True, "cached": True, "current_hash": cur_hash, "current_short": cur_hash[:12]}

    async with _get_deploy_lock():
        if not force and _cached_verified_hash == cur_hash:
            return {"deployed": False, "up_to_date": True, "cached": True, "current_hash": cur_hash, "current_short": cur_hash[:12]}

        status = await asyncio.to_thread(check_modal_version_sync)
        if status.up_to_date and not force:
            _cached_verified_hash = cur_hash
            return {"deployed": False, **status.to_dict()}

        if not status.configured and not force:
            # Let downstream Modal call surface any auth error cleanly
            return {"deployed": False, **status.to_dict()}

        if on_event is not None:
            on_event(
                {
                    "event": "stage",
                    "stage": "deploy",
                    "app_name": APP_NAME,
                    "previous_short": status.deployed_short or "none",
                    "current_short": status.current_short,
                    "detail": f"Auto-deploying Modal worker ({status.deployed_short or 'untagged'} → {status.current_short})",
                }
            )

        res = await asyncio.to_thread(deploy_modal_app_sync, force=force)

        if on_event is not None:
            on_event(
                {
                    "event": "stage",
                    "stage": "deployed",
                    "app_name": APP_NAME,
                    "current_short": res["current_short"],
                    "elapsed_s": res["elapsed_s"],
                }
            )
        return res