"""OpenRouter Seedream 5.5 Image API client (`bytedance-seed/seedream-5-5`).

Implements the confirmed OpenRouter Image API contract (`https://openrouter.ai/bytedance-seed/seedream-5-5`):
  - Discovery: `GET https://openrouter.ai/api/v1/images/models`
  - Generate:  `POST https://openrouter.ai/api/v1/images`
  - Billing:   `usage.cost` (USD) returned for session ledger & `BLENDERBOT_MAX_COST` tracking.

Supports text-to-image & multi-reference image-to-image (`input_references` up to 10 images):
  - `prompt` (required)
  - `resolution`: `"1K" | "1.5K" | "2K"`
  - `aspect_ratio`: `"1:1" | "16:9" | "9:16" | "4:3" | "3:4" | "3:2" | "2:3" | "21:9" | ... | "auto"`
  - `output_format`: `"png" | "jpeg"`
  - `background`: `"auto" | "transparent" | "opaque"` (`"transparent"` automatically uses `"png"`)
  - `input_references`: up to 10 local file paths, `https://` URLs, or `data:image/...` URLs
  - `seed`: optional deterministic integer seed.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import io
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp

from blenderbot.providers.index import OPENROUTER_BASE_URL

from .seedance import encode_file_data_url

DEFAULT_SEEDREAM_MODEL = "bytedance-seed/seedream-5-5"
ESTIMATED_COST_PER_IMAGE_USD = 0.02
ESTIMATED_COST_BY_RES_USD: dict[str, float] = {
    "1K": 0.02,
    "1.5K": 0.03,
    "2K": 0.04,
}

SEEDREAM_RESOLUTIONS: tuple[str, ...] = ("1K", "1.5K", "2K")
SEEDREAM_ASPECT_RATIOS: tuple[str, ...] = (
    "auto",
    "16:9",
    "1:1",
    "4:3",
    "3:4",
    "9:16",
    "3:2",
    "2:3",
    "4:5",
    "5:4",
    "21:9",
    "9:21",
    "2:1",
    "1:2",
    "19.5:9",
    "9:19.5",
    "20:9",
    "9:20",
)
SEEDREAM_OUTPUT_FORMATS: tuple[str, ...] = ("png", "jpeg")
SEEDREAM_BACKGROUNDS: tuple[str, ...] = ("auto", "transparent", "opaque")

SEEDREAM_MODES: dict[str, str] = {
    "standard": "",
    "concept_plate": (
        " Crisp 35mm deep-focus production concept plate, clean architectural/spatial geometry, "
        "coherent physical lighting, zero lens distortion, suitable for 3D projection & Seedance conditioning."
    ),
    "pbr_texture": (
        " Seamless orthographic flat-lit surface texture, even diffuse illumination, zero cast shadows, "
        "zero perspective vignetting, high micro-surface detail."
    ),
    "cutout_prop": (
        " Isolated clean hero 3D prop asset, crisp silhouette edges, studio three-point rim & key lighting, "
        "centered framing with clean alpha separation."
    ),
}


class SeedreamError(Exception):
    """Raised when Seedream payload validation or OpenRouter `/api/v1/images` generation fails."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _parse_api_error(status: int, raw_text: str) -> str:
    try:
        obj = json.loads(raw_text)
        err = obj.get("error")
        if isinstance(err, dict):
            code = err.get("code") or status
            return f"[{code}] {err.get('message') or raw_text[:300]}"
        if isinstance(err, str):
            return f"[{status}] {err}"
    except ValueError:
        pass
    return f"[{status}] {raw_text[:300] or 'HTTP error'}"


def prepare_seedream_request(
    *,
    model: str = DEFAULT_SEEDREAM_MODEL,
    prompt: str,
    mode: str = "standard",
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    output_format: str | None = None,
    background: str | None = None,
    reference_images: list[str | Path] | None = None,
    seed: int | None = None,
    provider: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and construct a spec-compliant `POST /api/v1/images` JSON payload for Seedream 5.5."""
    clean_prompt = str(prompt or "").strip()
    if not clean_prompt:
        raise SeedreamError("Seedream 5.5 requires a non-empty 'prompt'.")

    mode_key = (mode or "standard").strip().lower()
    suffix = SEEDREAM_MODES.get(mode_key, "")
    compiled_prompt = f"{clean_prompt}{suffix}".strip()

    eff_model = (model or DEFAULT_SEEDREAM_MODEL).strip() or DEFAULT_SEEDREAM_MODEL
    payload: dict[str, Any] = {
        "model": eff_model,
        "prompt": compiled_prompt,
        "n": 1,
    }

    eff_res: str | None = None
    if resolution:
        r_up = resolution.strip().upper()
        if r_up not in SEEDREAM_RESOLUTIONS:
            raise SeedreamError(
                f"Unsupported resolution '{resolution}'. Expected one of: {', '.join(SEEDREAM_RESOLUTIONS)}"
            )
        eff_res = r_up
        payload["resolution"] = eff_res

    eff_ar: str | None = None
    if aspect_ratio:
        ar_clean = aspect_ratio.strip().lower()
        if ar_clean not in SEEDREAM_ASPECT_RATIOS:
            raise SeedreamError(
                f"Unsupported aspect_ratio '{aspect_ratio}'. Expected one of: {', '.join(SEEDREAM_ASPECT_RATIOS)}"
            )
        eff_ar = ar_clean
        payload["aspect_ratio"] = eff_ar

    eff_bg: str | None = None
    if background:
        bg_clean = background.strip().lower()
        if bg_clean not in SEEDREAM_BACKGROUNDS:
            raise SeedreamError(
                f"Unsupported background '{background}'. Expected one of: {', '.join(SEEDREAM_BACKGROUNDS)}"
            )
        eff_bg = bg_clean
        payload["background"] = eff_bg

    eff_fmt: str | None = None
    if output_format:
        fmt_clean = output_format.strip().lower()
        if fmt_clean == "jpg":
            fmt_clean = "jpeg"
        if fmt_clean not in SEEDREAM_OUTPUT_FORMATS:
            raise SeedreamError(
                f"Unsupported output_format '{output_format}'. Expected 'png' or 'jpeg'."
            )
        eff_fmt = fmt_clean

    # Per OpenRouter Seedream 5.5 contract: transparent background requires png/webp output
    if eff_bg == "transparent":
        eff_fmt = "png"

    if eff_fmt:
        payload["output_format"] = eff_fmt

    refs_list: list[dict[str, Any]] = []
    ref_labels: list[str] = []
    for raw_ref in (reference_images or [])[:10]:
        if not raw_ref:
            continue
        data_url = encode_file_data_url(raw_ref)
        refs_list.append({"type": "image_url", "image_url": {"url": data_url}})
        s_ref = str(raw_ref)
        ref_labels.append(
            Path(s_ref).name if not s_ref.startswith(("http://", "https://", "data:")) else s_ref[:48]
        )

    if refs_list:
        payload["input_references"] = refs_list

    if seed is not None:
        payload["seed"] = int(seed)
    if isinstance(provider, dict) and provider:
        payload["provider"] = provider

    est_cost = ESTIMATED_COST_BY_RES_USD.get(eff_res or "1K", ESTIMATED_COST_PER_IMAGE_USD)
    summary = {
        "model": eff_model,
        "mode": mode_key,
        "resolution": eff_res or "1K (default)",
        "aspect_ratio": eff_ar or "auto",
        "output_format": eff_fmt or "png",
        "background": eff_bg or "auto",
        "reference_count": len(refs_list),
        "references": ref_labels,
        "estimated_cost_usd": round(est_cost, 4),
    }
    return payload, summary


def _extract_image_cost(item: dict[str, Any]) -> float:
    pricing = item.get("pricing") or {}
    if isinstance(pricing, dict):
        for k in ("image", "per_image", "request"):
            v = pricing.get(k)
            if v is not None:
                with contextlib.suppress(ValueError, TypeError):
                    fv = float(v)
                    if fv > 0:
                        return fv
    return ESTIMATED_COST_PER_IMAGE_USD


async def fetch_openrouter_image_models(
    *, api_key: str | None = None, base_url: str = OPENROUTER_BASE_URL
) -> list[dict[str, Any]]:
    """Query `GET /api/v1/images/models` for available OpenRouter image models & pricing."""
    default_item = {
        "id": DEFAULT_SEEDREAM_MODEL,
        "name": "ByteDance Seed: Seedream 5.5",
        "cost_per_image_usd": ESTIMATED_COST_PER_IMAGE_USD,
    }
    url = f"{base_url.rstrip('/')}/images/models"
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with contextlib.suppress(Exception):
        async with (
            aiohttp.ClientSession() as http,
            http.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=12)) as resp,
        ):
            if resp.status != 200:
                return [default_item]
            data = await resp.json()
            rows = data.get("data") if isinstance(data, dict) else data
            if isinstance(rows, list) and rows:
                out: list[dict[str, Any]] = []
                has_seedream = False
                for m in rows:
                    if isinstance(m, dict) and m.get("id"):
                        mid = str(m["id"])
                        if mid == DEFAULT_SEEDREAM_MODEL:
                            has_seedream = True
                        out.append(
                            {
                                "id": mid,
                                "name": str(m.get("name") or mid),
                                "cost_per_image_usd": _extract_image_cost(m),
                            }
                        )
                if not has_seedream:
                    out.insert(0, default_item)
                else:
                    # Put Seedream 5.5 first in the dropdown!
                    out.sort(key=lambda x: (0 if x["id"] == DEFAULT_SEEDREAM_MODEL else 1, x["id"]))
                if out:
                    return out
    return [default_item]


def _probe_image_dimensions(img_bytes: bytes) -> tuple[int | None, int | None]:
    with contextlib.suppress(Exception):
        from PIL import Image

        with Image.open(io.BytesIO(img_bytes)) as im:
            return int(im.width), int(im.height)
    return None, None


def default_seedream_output_path(
    cwd: Path,
    *,
    prefix: str = "seedream",
    ext: str = ".png",
) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cwd / "assets" / "seedream"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    cand = out_dir / f"{prefix}_{ts}{clean_ext}"
    if cand.exists():
        cand = out_dir / f"{prefix}_{ts}_{int(time.time() * 1000) % 1000:03d}{clean_ext}"
    return cand


async def generate_seedream_image(
    *,
    api_key: str,
    output_path: str | Path,
    model: str = DEFAULT_SEEDREAM_MODEL,
    prompt: str,
    mode: str = "standard",
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    output_format: str | None = None,
    background: str | None = None,
    reference_images: list[str | Path] | None = None,
    seed: int | None = None,
    provider: dict[str, Any] | None = None,
    base_url: str | None = None,
    timeout_sec: float = 180.0,
    max_retries_429: int = 2,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: asyncio.Event | None = None,
    http_session: aiohttp.ClientSession | None = None,
) -> dict[str, Any]:
    """Generate an image with Seedream 5.5 via `POST https://openrouter.ai/api/v1/images`."""
    if not api_key or not api_key.strip():
        raise SeedreamError(
            "OPENROUTER_API_KEY is not configured. Set OPENROUTER_API_KEY in ~/.blenderbot/.env "
            "(or via `/keys openrouter <key> --save`).",
            status=401,
        )

    t0 = time.monotonic()
    eff_base = (base_url or os.environ.get("BLENDERBOT_IMAGE_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
    submit_url = f"{eff_base}/images"

    dst_initial = Path(output_path)
    inferred_fmt = output_format
    if inferred_fmt is None and dst_initial.suffix.lower() in (".jpg", ".jpeg"):
        inferred_fmt = "jpeg"
    elif inferred_fmt is None and dst_initial.suffix.lower() == ".png":
        inferred_fmt = "png"

    payload, summary = prepare_seedream_request(
        model=model,
        prompt=prompt,
        mode=mode,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        output_format=inferred_fmt,
        background=background,
        reference_images=reference_images,
        seed=seed,
        provider=provider,
    )

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/blenderbot",
        "X-Title": "blenderbot",
    }

    own_session = http_session is None
    session = http_session or aiohttp.ClientSession()
    resp_json: dict[str, Any] = {}

    try:
        if on_progress:
            on_progress({"stage": "generating", "elapsed_sec": 0.0, "conditioning": summary})

        for attempt in range(max_retries_429 + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("Seedream image generation cancelled by operator")

            async with session.post(
                submit_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=max(15.0, float(timeout_sec))),
            ) as resp:
                raw_text = await resp.text()
                if resp.status == 429 and attempt < max_retries_429:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                if resp.status not in (200, 201):
                    raise SeedreamError(
                        f"Seedream generation failed {_parse_api_error(resp.status, raw_text)}",
                        status=resp.status,
                    )
                resp_json = json.loads(raw_text)
                break

        data_list = resp_json.get("data") or []
        if not isinstance(data_list, list) or not data_list:
            raise SeedreamError(f"OpenRouter /images returned empty data array: {str(resp_json)[:240]}")

        first_item = data_list[0] if isinstance(data_list[0], dict) else {}
        media_type = str(first_item.get("media_type") or "image/png").lower()
        b64_str = first_item.get("b64_json")

        img_bytes: bytes = b""
        if b64_str and isinstance(b64_str, str):
            clean_b64 = b64_str.split(",", 1)[1] if b64_str.startswith("data:") and "," in b64_str else b64_str
            img_bytes = base64.b64decode(clean_b64)
        elif first_item.get("url"):
            async with session.get(str(first_item["url"]), timeout=aiohttp.ClientTimeout(total=60)) as dl_r:
                if dl_r.status != 200:
                    raise SeedreamError(f"Failed downloading image URL (HTTP {dl_r.status})", status=dl_r.status)
                img_bytes = await dl_r.read()

        if not img_bytes:
            raise SeedreamError("OpenRouter /images returned empty image bytes.")

    finally:
        if own_session:
            await session.close()

    # Align file extension with returned `media_type` if caller passed default path or mismatched ext
    ext_from_mime = ".jpg" if "jpeg" in media_type or "jpg" in media_type else ".webp" if "webp" in media_type else ".png"
    dst = dst_initial
    if not dst.suffix:
        dst = dst.with_suffix(ext_from_mime)
    elif payload.get("background") == "transparent" and dst.suffix.lower() in (".jpg", ".jpeg"):
        dst = dst.with_suffix(".png")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(dst.suffix + ".part")
    await asyncio.to_thread(tmp_dst.write_bytes, img_bytes)
    tmp_dst.replace(dst)

    width, height = await asyncio.to_thread(_probe_image_dimensions, img_bytes)
    usage = resp_json.get("usage") or {}
    actual_cost = round(float(usage.get("cost") or 0.0), 6)
    elapsed_sec = round(time.monotonic() - t0, 2)

    result = {
        "created": resp_json.get("created"),
        "model": payload["model"],
        "output_path": str(dst),
        "media_type": media_type,
        "size_bytes": len(img_bytes),
        "width": width,
        "height": height,
        "cost_usd": actual_cost,
        "usage": usage,
        "elapsed_sec": elapsed_sec,
        "conditioning": summary,
    }
    if on_progress:
        on_progress({"stage": "completed", **result})
    return result