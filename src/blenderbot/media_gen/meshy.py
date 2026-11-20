"""fal.ai Meshy 7.1 Image-to-3D Queue API client & Binary glTF 2.0 (`.glb`) parser.

Implements the official fal Queue API contract for `meshy/v7.1/image-to-3d`
(`https://fal.run/meshy/v7.1/image-to-3d` / `https://queue.fal.run/meshy/v7.1/image-to-3d`):
  1. Submit: `POST https://queue.fal.run/meshy/v7.1/image-to-3d` (`Authorization: Key <FAL_KEY>`)
     - Keeps `request_id` immediately without duplicate billable retries on ambiguous responses.
     - Validates `status_url`, `response_url`, and `cancel_url` use the trusted fal origin.
  2. Poll:   `GET <status_url>?logs=1` (`IN_QUEUE` | `IN_PROGRESS` -> `COMPLETED`)
  3. Result: `GET <response_url>` -> downloads `model_glb`, optional `thumbnail`, `rigged_character_glb`,
     `animation_glb`, and writes `meshy_meta.json` for workspace persistence.
  4. Billing: Exact pricing model ($0.80 base, +$0.40 texture, +$0.20 rigging, +$0.12 animation)
     plus explicit HTTP 403 `"Exhausted balance"` vs `"User is locked"` remedy reporting.

Also includes a pure-NumPy + stdlib Binary glTF 2.0 (`.glb`) triangle/normal/base-color extractor
so generated Meshy `.glb` assets stream directly into the WebGL2 3D & Minecraft FPS viewport
with zero external mesh dependencies and zero Blender requirement.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import io
import json
import mimetypes
import os
import secrets
import struct
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import numpy as np

DEFAULT_MESHY_MODEL = "meshy/v7.1/image-to-3d"
FAL_QUEUE_BASE_URL = "https://queue.fal.run"
FAL_CREDITS_URL = "https://fal.ai/dashboard/usage-billing/credits"

MESHY_BASE_COST_USD = 0.80
MESHY_TEXTURE_COST_USD = 0.40
MESHY_RIGGING_COST_USD = 0.20
MESHY_ANIMATION_COST_USD = 0.12

MESHY_MODEL_TYPES: tuple[str, ...] = ("standard",)
MESHY_TOPOLOGIES: tuple[str, ...] = ("triangle", "quad")
MESHY_SYMMETRY_MODES: tuple[str, ...] = ("off", "auto", "on")
MESHY_TEXTURE_MODELS: tuple[str, ...] = ("meshy-7", "meshy-n1", "meshy-6-lite")
MESHY_POSE_MODES: tuple[str, ...] = ("", "a-pose", "t-pose")
MESHY_GEOMETRY_RESOLUTIONS: tuple[str, ...] = ("standard", "2k", "4k")
MESHY_SKELETON_TEMPLATES: tuple[str, ...] = (
    "mixamo",
    "ue5_mannequin",
    "unity_humanoid",
    "maya_human_ik",
    "blender_rigify_basic_human",
    "max_biped",
    "cc5_hd_neutral_base",
    "daz_genesis9",
    "meshyrig",
)

MESHY_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".avif", ".heif", ".heic"})
TRUSTED_FAL_HOSTS: frozenset[str] = frozenset({"fal.run", "fal.ai", "fal.media"})


class MeshyError(Exception):
    """Raised when Meshy 7.1 input validation, fal queue submission, polling, or download fails."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
        exhausted_balance: bool = False,
        user_locked: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.request_id = request_id
        self.exhausted_balance = exhausted_balance
        self.user_locked = user_locked


def is_trusted_fal_url(raw_url: str, *, custom_base_url: str | None = None) -> bool:
    """Verify queue status/result/cancel URLs belong to the trusted fal origin (or test override)."""
    parsed = urlparse(str(raw_url or "").strip())
    if parsed.scheme == "https":
        host = (parsed.hostname or "").lower()
        for root in TRUSTED_FAL_HOSTS:
            if host == root or host.endswith(f".{root}"):
                return True
    if custom_base_url:
        base_p = urlparse(custom_base_url.strip())
        if (
            parsed.scheme in ("http", "https")
            and parsed.scheme == base_p.scheme
            and (parsed.hostname or "").lower() == (base_p.hostname or "").lower()
            and parsed.port == base_p.port
        ):
            return True
    return False


def _encode_meshy_image_url(source: str | Path) -> tuple[str, str]:
    """Encode a local image (`.jpg`, `.jpeg`, `.png`, `.avif`, `.heif`, or `.webp` -> `.png`) or pass URL."""
    s = str(source or "").strip()
    if not s:
        raise MeshyError("Meshy 7.1 requires 'image_url' (an image file path, https:// URL, or data: URI).")
    if s.startswith("data:"):
        return s, "data_uri"
    if s.startswith(("https://", "http://")):
        return s, s[:64]

    p = Path(s)
    if not p.is_file():
        raise MeshyError(f"Input image file does not exist: {p}")
    ext = p.suffix.lower()
    if ext not in MESHY_IMAGE_EXTS:
        raise MeshyError(
            f"Unsupported image extension '{ext}'. Meshy 7.1 supports .png, .jpg, .jpeg, .avif, .heif (and .webp auto-converted to .png)."
        )

    raw = p.read_bytes()
    if ext == ".webp":
        from PIL import Image

        buf = io.BytesIO()
        with Image.open(io.BytesIO(raw)) as im:
            im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}", p.name

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".avif": "image/avif",
        ".heif": "image/heif",
        ".heic": "image/heic",
    }
    mime = mime_map.get(ext) or mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", p.name


def estimate_meshy_cost_usd(
    *,
    should_texture: bool = True,
    enable_rigging: bool = False,
    enable_animation: bool = False,
) -> float:
    cost = MESHY_BASE_COST_USD
    if should_texture:
        cost += MESHY_TEXTURE_COST_USD
    if enable_rigging:
        cost += MESHY_RIGGING_COST_USD
        if enable_animation:
            cost += MESHY_ANIMATION_COST_USD
    return round(cost, 4)


def _parse_fal_error(status: int, raw_text: str, *, request_id: str | None = None) -> MeshyError:
    detail = ""
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            raw_detail = obj.get("detail") or obj.get("error") or obj.get("message")
            if isinstance(raw_detail, str):
                detail = raw_detail.strip()
            elif isinstance(raw_detail, (list, dict)):
                detail = json.dumps(raw_detail, ensure_ascii=False)
    except ValueError:
        pass
    msg_body = detail or (raw_text.strip()[:400] or f"HTTP {status}")

    if status == 403:
        low = msg_body.lower()
        if "exhausted balance" in low:
            return MeshyError(
                f"fal API returned 403 Exhausted balance: {msg_body} — Add credits at {FAL_CREDITS_URL}",
                status=403,
                request_id=request_id,
                exhausted_balance=True,
            )
        if "user is locked" in low:
            return MeshyError(
                f"fal API returned 403: {msg_body}",
                status=403,
                request_id=request_id,
                user_locked=True,
            )
        return MeshyError(
            f"fal API returned 403: {msg_body}",
            status=403,
            request_id=request_id,
        )

    return MeshyError(
        f"Meshy 7.1 fal API error [{status}]: {msg_body}",
        status=status,
        request_id=request_id,
    )


def prepare_meshy_request(
    *,
    image: str | Path,
    model: str = DEFAULT_MESHY_MODEL,
    model_type: str = "standard",
    topology: str = "triangle",
    target_polycount: int = 30000,
    symmetry_mode: str = "auto",
    should_remesh: bool = True,
    should_texture: bool = True,
    texture_model: str | None = None,
    enable_pbr: bool = False,
    pose_mode: str = "",
    texture_prompt: str | None = None,
    texture_image: str | Path | None = None,
    enable_rigging: bool = False,
    rigging_height_meters: float = 1.7,
    skeleton_template: str | None = None,
    include_fingers: bool = False,
    enable_animation: bool = False,
    animation_action_id: int = 92,
    enable_safety_checker: bool = True,
    geometry_resolution: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and build an exact `meshy/v7.1/image-to-3d` input payload per verified schema."""
    eff_model = (model or DEFAULT_MESHY_MODEL).strip() or DEFAULT_MESHY_MODEL
    if eff_model != DEFAULT_MESHY_MODEL:
        raise MeshyError(
            f"Endpoint '{eff_model}' is not approved; configured endpoint is '{DEFAULT_MESHY_MODEL}'."
        )

    image_url, image_label = _encode_meshy_image_url(image)

    mt = (model_type or "standard").strip().lower()
    if mt not in MESHY_MODEL_TYPES:
        raise MeshyError(f"Unsupported model_type '{model_type}'. Only 'standard' is supported.")

    topo = (topology or "triangle").strip().lower()
    if topo not in MESHY_TOPOLOGIES:
        raise MeshyError(
            f"Unsupported topology '{topology}'. Expected one of: {', '.join(MESHY_TOPOLOGIES)}"
        )

    poly = int(target_polycount)
    if not (100 <= poly <= 300_000):
        raise MeshyError(f"target_polycount must be between 100 and 300000 (got {poly}).")

    sym = (symmetry_mode or "auto").strip().lower()
    if sym not in MESHY_SYMMETRY_MODES:
        raise MeshyError(
            f"Unsupported symmetry_mode '{symmetry_mode}'. Expected one of: {', '.join(MESHY_SYMMETRY_MODES)}"
        )

    pose = (pose_mode or "").strip().lower()
    if pose not in MESHY_POSE_MODES:
        raise MeshyError("Unsupported pose_mode. Expected '', 'a-pose', or 't-pose'.")

    remesh_b = bool(should_remesh)
    tex_b = bool(should_texture)
    pbr_b = bool(enable_pbr)

    tex_model_clean: str | None = None
    if texture_model and str(texture_model).strip():
        tm = str(texture_model).strip().lower()
        if tm not in MESHY_TEXTURE_MODELS:
            raise MeshyError(
                f"Unsupported texture_model '{texture_model}'. Expected one of: {', '.join(MESHY_TEXTURE_MODELS)}"
            )
        tex_model_clean = tm

    tex_prompt_clean = str(texture_prompt or "").strip()
    if len(tex_prompt_clean) > 800:
        raise MeshyError(f"texture_prompt exceeds 800-character limit ({len(tex_prompt_clean)} chars).")

    # Schema constraint: meshy-n1 rejects texture_prompt, enable_pbr, and quad topology with remeshing
    if tex_b and tex_model_clean == "meshy-n1":
        if tex_prompt_clean:
            raise MeshyError("texture_model 'meshy-n1' rejects 'texture_prompt' (uses reference images only).")
        if pbr_b:
            raise MeshyError("texture_model 'meshy-n1' rejects 'enable_pbr'.")
        if topo == "quad" and remesh_b:
            raise MeshyError("texture_model 'meshy-n1' rejects 'quad' topology when 'should_remesh' is true.")

    tex_img_url: str | None = None
    tex_img_label: str | None = None
    if texture_image and str(texture_image).strip():
        tex_img_url, tex_img_label = _encode_meshy_image_url(texture_image)

    rig_b = bool(enable_rigging)
    anim_b = bool(enable_animation)
    if anim_b and not rig_b:
        raise MeshyError("'enable_animation' requires 'enable_rigging' to be true.")

    skel_clean: str | None = None
    if skeleton_template and str(skeleton_template).strip():
        sk = str(skeleton_template).strip().lower()
        if sk not in MESHY_SKELETON_TEMPLATES:
            raise MeshyError(
                f"Unsupported skeleton_template '{skeleton_template}'. Expected one of: {', '.join(MESHY_SKELETON_TEMPLATES)}"
            )
        skel_clean = sk

    act_id = int(animation_action_id)
    if anim_b and not (0 <= act_id <= 696):
        raise MeshyError(f"animation_action_id must be in [0, 696] when enable_animation is true (got {act_id}).")

    geom_res_clean: str | None = None
    if geometry_resolution and str(geometry_resolution).strip():
        gr = str(geometry_resolution).strip().lower()
        if gr not in MESHY_GEOMETRY_RESOLUTIONS:
            raise MeshyError(
                f"Unsupported geometry_resolution '{geometry_resolution}'. Expected one of: {', '.join(MESHY_GEOMETRY_RESOLUTIONS)}"
            )
        geom_res_clean = gr

    payload: dict[str, Any] = {
        "image_url": image_url,
        "model_type": mt,
        "topology": topo,
        "target_polycount": poly,
        "symmetry_mode": sym,
        "should_remesh": remesh_b,
        "should_texture": tex_b,
        "enable_pbr": pbr_b,
        "pose_mode": pose,
        "enable_rigging": rig_b,
        "enable_animation": anim_b,
        "enable_safety_checker": bool(enable_safety_checker),
    }

    if tex_b and tex_model_clean:
        payload["texture_model"] = tex_model_clean
    if tex_b and tex_prompt_clean:
        payload["texture_prompt"] = tex_prompt_clean
    if tex_b and tex_img_url:
        payload["texture_image_url"] = tex_img_url
    if geom_res_clean:
        payload["geometry_resolution"] = geom_res_clean

    if rig_b:
        payload["rigging_height_meters"] = round(float(rigging_height_meters), 3)
        payload["include_fingers"] = bool(include_fingers)
        if skel_clean:
            payload["skeleton_template"] = skel_clean
    if anim_b:
        payload["animation_action_id"] = act_id

    est_cost = estimate_meshy_cost_usd(
        should_texture=tex_b,
        enable_rigging=rig_b,
        enable_animation=anim_b,
    )
    summary = {
        "model": eff_model,
        "image": image_label,
        "topology": topo,
        "target_polycount": poly,
        "symmetry_mode": sym,
        "should_remesh": remesh_b,
        "should_texture": tex_b,
        "texture_model": tex_model_clean or "default",
        "enable_pbr": pbr_b,
        "pose_mode": pose or "none",
        "texture_image": tex_img_label,
        "enable_rigging": rig_b,
        "skeleton_template": skel_clean or "meshyrig" if rig_b else None,
        "enable_animation": anim_b,
        "animation_action_id": act_id if anim_b else None,
        "geometry_resolution": geom_res_clean or "standard",
        "estimated_cost_usd": est_cost,
    }
    return payload, summary


def default_meshy_job_dir(cwd: Path, *, stem: str = "meshy") -> tuple[str, Path]:
    safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem).strip("_")[:24] or "meshy"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{safe_stem}_{ts}_{secrets.token_hex(2)}"
    out_dir = cwd / "assets" / "meshy" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return job_id, out_dir


async def _download_fal_artifact(
    session: aiohttp.ClientSession,
    url: str,
    dst: Path,
    *,
    custom_base_url: str | None = None,
    timeout_sec: float = 180.0,
) -> int:
    if not is_trusted_fal_url(url, custom_base_url=custom_base_url):
        raise MeshyError(f"Refusing to download artifact from untrusted host: {url[:120]}")
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
        if resp.status != 200:
            raise MeshyError(f"Failed downloading Meshy artifact (HTTP {resp.status})", status=resp.status)
        data = await resp.read()
    if not data:
        raise MeshyError(f"Downloaded empty artifact from {url[:80]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    await asyncio.to_thread(tmp.write_bytes, data)
    tmp.replace(dst)
    return len(data)


async def generate_meshy_3d(
    *,
    api_key: str,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    image: str | Path,
    model: str = DEFAULT_MESHY_MODEL,
    model_type: str = "standard",
    topology: str = "triangle",
    target_polycount: int = 30000,
    symmetry_mode: str = "auto",
    should_remesh: bool = True,
    should_texture: bool = True,
    texture_model: str | None = None,
    enable_pbr: bool = False,
    pose_mode: str = "",
    texture_prompt: str | None = None,
    texture_image: str | Path | None = None,
    enable_rigging: bool = False,
    rigging_height_meters: float = 1.7,
    skeleton_template: str | None = None,
    include_fingers: bool = False,
    enable_animation: bool = False,
    animation_action_id: int = 92,
    enable_safety_checker: bool = True,
    geometry_resolution: str | None = None,
    base_url: str | None = None,
    poll_interval_sec: float = 2.5,
    timeout_sec: float = 900.0,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: asyncio.Event | None = None,
    http_session: aiohttp.ClientSession | None = None,
) -> dict[str, Any]:
    """Submit, poll, and download a `meshy/v7.1/image-to-3d` job via fal's Queue API."""
    if not api_key or not api_key.strip():
        raise MeshyError(
            "FAL_KEY is not configured. Set FAL_KEY in ~/.blenderbot/.env (or via `/keys fal <key> --save`).",
            status=401,
        )

    payload, summary = prepare_meshy_request(
        image=image,
        model=model,
        model_type=model_type,
        topology=topology,
        target_polycount=target_polycount,
        symmetry_mode=symmetry_mode,
        should_remesh=should_remesh,
        should_texture=should_texture,
        texture_model=texture_model,
        enable_pbr=enable_pbr,
        pose_mode=pose_mode,
        texture_prompt=texture_prompt,
        texture_image=texture_image,
        enable_rigging=enable_rigging,
        rigging_height_meters=rigging_height_meters,
        skeleton_template=skeleton_template,
        include_fingers=include_fingers,
        enable_animation=enable_animation,
        animation_action_id=animation_action_id,
        enable_safety_checker=enable_safety_checker,
        geometry_resolution=geometry_resolution,
    )

    custom_base = (base_url or os.environ.get("BLENDERBOT_FAL_QUEUE_URL") or "").strip().rstrip("/") or None
    eff_base = custom_base or FAL_QUEUE_BASE_URL
    submit_url = f"{eff_base}/{DEFAULT_MESHY_MODEL}"

    if output_path is not None:
        glb_dst = Path(output_path)
        if glb_dst.suffix.lower() != ".glb":
            glb_dst = glb_dst.with_suffix(".glb")
        target_dir = glb_dst.parent
        target_dir.mkdir(parents=True, exist_ok=True)
    elif output_dir is not None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        glb_dst = target_dir / "model.glb"
    else:
        _, target_dir = default_meshy_job_dir(Path.cwd())
        glb_dst = target_dir / "model.glb"

    headers = {
        "Authorization": f"Key {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    t0 = time.monotonic()
    own_session = http_session is None
    session = http_session or aiohttp.ClientSession(trust_env=True)
    request_id: str | None = None
    status_url: str | None = None
    response_url: str | None = None
    cancel_url: str | None = None
    seen_logs: set[str] = set()

    try:
        if on_progress:
            on_progress({"stage": "submitting", "elapsed_sec": 0.0, "conditioning": summary})

        # IMPORTANT per fal queue rules: NEVER auto-retry a submit on ambiguous error (a second submit is a new billable job).
        async with session.post(
            submit_url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60.0),
        ) as sub_resp:
            sub_raw = await sub_resp.text()
            if sub_resp.status not in (200, 201, 202):
                raise _parse_fal_error(sub_resp.status, sub_raw)
            try:
                sub_json = json.loads(sub_raw)
            except ValueError as exc:
                raise MeshyError(
                    f"Ambiguous non-JSON queue response from fal (did NOT resubmit to avoid duplicate billing): {sub_raw[:240]}"
                ) from exc

        # Direct synchronous response fallback (if caller pointed base_url at fal.run)
        if isinstance(sub_json, dict) and isinstance(sub_json.get("model_glb"), dict) and not sub_json.get("request_id"):
            result_json = sub_json
            request_id = str(sub_resp.headers.get("x-fal-request-id") or f"sync-{secrets.token_hex(4)}")
        else:
            request_id = str(sub_json.get("request_id") or "").strip()
            if not request_id:
                raise MeshyError(
                    f"fal queue response missing 'request_id' (did NOT resubmit): {str(sub_json)[:240]}"
                )
            status_url = str(sub_json.get("status_url") or f"{eff_base}/meshy/requests/{request_id}/status").strip()
            response_url = str(sub_json.get("response_url") or f"{eff_base}/meshy/requests/{request_id}").strip()
            cancel_url = str(sub_json.get("cancel_url") or f"{eff_base}/meshy/requests/{request_id}/cancel").strip()

            for u_name, u_val in (("status_url", status_url), ("response_url", response_url), ("cancel_url", cancel_url)):
                if u_val and not is_trusted_fal_url(u_val, custom_base_url=custom_base):
                    raise MeshyError(
                        f"Refusing untrusted fal queue {u_name}: {u_val[:120]}",
                        request_id=request_id,
                    )

            if on_progress:
                on_progress(
                    {
                        "stage": "in_queue",
                        "request_id": request_id,
                        "elapsed_sec": round(time.monotonic() - t0, 1),
                        "conditioning": summary,
                    }
                )

            deadline = t0 + max(30.0, float(timeout_sec))
            result_json = {}
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    if cancel_url:
                        with contextlib.suppress(Exception):
                            async with session.put(
                                cancel_url,
                                headers=headers,
                                timeout=aiohttp.ClientTimeout(total=10.0),
                            ) as _cr:
                                await _cr.read()
                    raise asyncio.CancelledError("Meshy 7.1 job cancelled by operator")

                if time.monotonic() > deadline:
                    raise MeshyError(
                        f"Timed out after {timeout_sec:.0f}s waiting for Meshy 7.1 request_id={request_id}",
                        request_id=request_id,
                    )

                poll_u = f"{status_url}{'&' if '?' in status_url else '?'}logs=1"
                async with session.get(
                    poll_u,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as st_resp:
                    st_raw = await st_resp.text()
                    if st_resp.status == 429 or st_resp.status >= 500:
                        await asyncio.sleep(poll_interval_sec * 1.5)
                        continue
                    if st_resp.status not in (200, 202):
                        raise _parse_fal_error(st_resp.status, st_raw, request_id=request_id)
                    st_json = json.loads(st_raw)

                q_status = str(st_json.get("status") or "").upper()
                new_logs: list[str] = []
                for lg in st_json.get("logs") or []:
                    msg = str(lg.get("message") if isinstance(lg, dict) else lg or "").strip()
                    if msg and msg not in seen_logs:
                        seen_logs.add(msg)
                        new_logs.append(msg)

                if on_progress:
                    on_progress(
                        {
                            "stage": q_status.lower() or "in_progress",
                            "request_id": request_id,
                            "queue_position": st_json.get("queue_position"),
                            "logs": new_logs,
                            "elapsed_sec": round(time.monotonic() - t0, 1),
                        }
                    )

                if q_status == "COMPLETED":
                    break
                if q_status in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
                    raise MeshyError(
                        f"Meshy 7.1 job {q_status}: {st_json.get('error') or str(st_json)[:260]}",
                        request_id=request_id,
                    )

                await asyncio.sleep(poll_interval_sec)

            # Fetch final response payload
            async with session.get(
                response_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60.0),
            ) as res_resp:
                res_raw = await res_resp.text()
                if res_resp.status != 200:
                    raise _parse_fal_error(res_resp.status, res_raw, request_id=request_id)
                loaded = json.loads(res_raw)
                result_json = (
                    loaded["response"]
                    if isinstance(loaded, dict) and isinstance(loaded.get("response"), dict) and "model_glb" not in loaded
                    else loaded
                )

        model_glb_obj = result_json.get("model_glb") or (result_json.get("model_urls") or {}).get("glb")
        if not isinstance(model_glb_obj, dict) or not model_glb_obj.get("url"):
            raise MeshyError(
                f"Meshy 7.1 response missing 'model_glb.url': {str(result_json)[:260]}",
                request_id=request_id,
            )

        if on_progress:
            on_progress(
                {
                    "stage": "downloading",
                    "request_id": request_id,
                    "elapsed_sec": round(time.monotonic() - t0, 1),
                }
            )

        glb_bytes = await _download_fal_artifact(
            session,
            str(model_glb_obj["url"]),
            glb_dst,
            custom_base_url=custom_base,
        )

        thumb_dst: Path | None = None
        thumb_obj = result_json.get("thumbnail")
        if isinstance(thumb_obj, dict) and thumb_obj.get("url"):
            cand_thumb = target_dir / "preview.png"
            with contextlib.suppress(Exception):
                await _download_fal_artifact(
                    session,
                    str(thumb_obj["url"]),
                    cand_thumb,
                    custom_base_url=custom_base,
                    timeout_sec=60.0,
                )
                thumb_dst = cand_thumb

        rigged_dst: Path | None = None
        rig_obj = result_json.get("rigged_character_glb")
        if isinstance(rig_obj, dict) and rig_obj.get("url"):
            cand_rig = target_dir / "rigged_character.glb"
            with contextlib.suppress(Exception):
                await _download_fal_artifact(
                    session,
                    str(rig_obj["url"]),
                    cand_rig,
                    custom_base_url=custom_base,
                )
                rigged_dst = cand_rig

        anim_dst: Path | None = None
        anim_obj = result_json.get("animation_glb")
        if isinstance(anim_obj, dict) and anim_obj.get("url"):
            cand_anim = target_dir / "animation.glb"
            with contextlib.suppress(Exception):
                await _download_fal_artifact(
                    session,
                    str(anim_obj["url"]),
                    cand_anim,
                    custom_base_url=custom_base,
                )
                anim_dst = cand_anim

    finally:
        if own_session:
            await session.close()

    elapsed_sec = round(time.monotonic() - t0, 2)
    cost_usd = summary["estimated_cost_usd"]
    mesh_stats = await asyncio.to_thread(extract_glb_viewport_triangles, glb_dst, max_triangles=1000)

    out_meta = {
        "job_id": target_dir.name,
        "request_id": request_id,
        "model": DEFAULT_MESHY_MODEL,
        "created_at": time.time(),
        "elapsed_sec": elapsed_sec,
        "cost_usd": cost_usd,
        "seed": result_json.get("seed"),
        "rig_task_id": result_json.get("rig_task_id"),
        "glb_path": str(glb_dst),
        "glb_size_bytes": glb_bytes,
        "thumbnail_path": str(thumb_dst) if thumb_dst else None,
        "rigged_glb_path": str(rigged_dst) if rigged_dst else None,
        "animation_glb_path": str(anim_dst) if anim_dst else None,
        "model_urls": result_json.get("model_urls") or {},
        "texture_urls": result_json.get("texture_urls") or [],
        "basic_animations": result_json.get("basic_animations"),
        "triangles_total": mesh_stats.get("total_triangles", 0),
        "world_aabb": mesh_stats.get("world_aabb"),
        "conditioning": summary,
    }
    with contextlib.suppress(OSError):
        (target_dir / "meshy_meta.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    if on_progress:
        on_progress({"stage": "completed", **out_meta})
    return out_meta


# ==============================================================================
# Pure-NumPy + Pillow Binary glTF 2.0 (`.glb`) -> Blender Z-Up WebGL2 Triangles
# ==============================================================================

_COMP_DTYPES: dict[int, tuple[str, int]] = {
    5120: ("<i1", 1),
    5121: ("<u1", 1),
    5122: ("<i2", 2),
    5123: ("<u2", 2),
    5125: ("<u4", 4),
    5126: ("<f4", 4),
}
_TYPE_COUNTS: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _read_gltf_accessor(gltf: dict[str, Any], bin_chunk: bytes, acc_idx: int) -> np.ndarray:
    acc = gltf["accessors"][acc_idx]
    comp_type = int(acc["componentType"])
    dt_str, comp_bytes = _COMP_DTYPES[comp_type]
    n_comp = _TYPE_COUNTS[str(acc["type"])]
    count = int(acc["count"])
    if count <= 0:
        return np.zeros((0, n_comp) if n_comp > 1 else (0,), dtype=np.float32)

    bv = gltf["bufferViews"][int(acc["bufferView"])]
    bv_offset = int(bv.get("byteOffset") or 0)
    acc_offset = int(acc.get("byteOffset") or 0)
    start = bv_offset + acc_offset
    elem_bytes = comp_bytes * n_comp
    stride = int(bv.get("byteStride") or elem_bytes)

    if stride == elem_bytes:
        arr = np.frombuffer(bin_chunk, dtype=dt_str, count=count * n_comp, offset=start)
        arr = arr.reshape((count, n_comp)) if n_comp > 1 else arr
    else:
        raw_slice = memoryview(bin_chunk)[start : start + stride * (count - 1) + elem_bytes]
        arr = np.ndarray(
            shape=(count, n_comp) if n_comp > 1 else (count,),
            dtype=dt_str,
            buffer=raw_slice,
            strides=(stride, comp_bytes) if n_comp > 1 else (stride,),
        ).copy()

    if acc.get("normalized"):
        if comp_type == 5121:
            arr = arr.astype(np.float32) / 255.0
        elif comp_type == 5123:
            arr = arr.astype(np.float32) / 65535.0
    return arr


def _quat_to_mat4(q_xyzw: list[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[0, 1] = 2.0 * (x * y - z * w)
    m[0, 2] = 2.0 * (x * z + y * w)
    m[1, 0] = 2.0 * (x * y + z * w)
    m[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[1, 2] = 2.0 * (y * z - x * w)
    m[2, 0] = 2.0 * (x * z - y * w)
    m[2, 1] = 2.0 * (y * z + x * w)
    m[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def _node_local_mat4(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        # glTF stores 4x4 matrices column-major
        return np.asarray(node["matrix"], dtype=np.float32).reshape((4, 4), order="F")
    t = node.get("translation") or [0.0, 0.0, 0.0]
    r = node.get("rotation") or [0.0, 0.0, 0.0, 1.0]
    s = node.get("scale") or [1.0, 1.0, 1.0]
    mt = np.eye(4, dtype=np.float32)
    mt[:3, 3] = [float(t[0]), float(t[1]), float(t[2])]
    mr = _quat_to_mat4(r)
    ms = np.diag([float(s[0]), float(s[1]), float(s[2]), 1.0]).astype(np.float32)
    return mt @ mr @ ms


def _decode_gltf_base_texture(gltf: dict[str, Any], bin_chunk: bytes, mat_idx: int | None) -> np.ndarray | None:
    if mat_idx is None:
        return None
    mats = gltf.get("materials") or []
    if not (0 <= mat_idx < len(mats)):
        return None
    pbr = mats[mat_idx].get("pbrMetallicRoughness") or {}
    tex_info = pbr.get("baseColorTexture")
    if not isinstance(tex_info, dict) or "index" not in tex_info:
        return None
    textures = gltf.get("textures") or []
    t_idx = int(tex_info["index"])
    if not (0 <= t_idx < len(textures)):
        return None
    src_idx = textures[t_idx].get("source")
    images = gltf.get("images") or []
    if src_idx is None or not (0 <= int(src_idx) < len(images)):
        return None
    img_def = images[int(src_idx)]
    raw_img: bytes | None = None
    if "bufferView" in img_def:
        bv = gltf["bufferViews"][int(img_def["bufferView"])]
        off = int(bv.get("byteOffset") or 0)
        ln = int(bv["byteLength"])
        raw_img = bin_chunk[off : off + ln]
    elif str(img_def.get("uri") or "").startswith("data:"):
        uri = str(img_def["uri"])
        if "," in uri:
            raw_img = base64.b64decode(uri.split(",", 1)[1])
    if not raw_img:
        return None
    with contextlib.suppress(Exception):
        from PIL import Image

        with Image.open(io.BytesIO(raw_img)) as im:
            rgba = im.convert("RGBA")
            if max(rgba.width, rgba.height) > 512:
                rgba.thumbnail((512, 512))
            return np.asarray(rgba, dtype=np.float32) / 255.0
    return None


def extract_glb_viewport_triangles(
    glb_path: str | Path,
    *,
    max_triangles: int = 65_000,
    place_on_ground: bool = True,
) -> dict[str, Any]:
    """Parse a Binary glTF 2.0 (`.glb`) into Blender Z-up `[x,y,z, nx,ny,nz, r,g,b,a]` WebGL2 floats.

    Applies `Rx(+90°)` (`(x, y, z)_gltf -> (x, -z, y)_bl`, `det=+1`) so glTF `+Y` up maps cleanly
    to Blender `+Z` up (`+Y` forward), baked with UVs -> `baseColorTexture` or `COLOR_0` vertex colors.
    """
    p = Path(glb_path)
    data = p.read_bytes()
    if len(data) < 20:
        raise MeshyError(f"Invalid GLB file (too short): {p.name}")
    magic, _version, _total_len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise MeshyError(f"Invalid GLB magic header in {p.name}")

    offset = 12
    gltf: dict[str, Any] = {}
    bin_chunk = b""
    while offset + 8 <= len(data):
        c_len, c_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + c_len]
        offset += c_len
        if c_type == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif c_type == 0x004E4942:
            bin_chunk = chunk

    if not gltf:
        raise MeshyError(f"GLB missing JSON chunk: {p.name}")

    nodes = gltf.get("nodes") or []
    meshes = gltf.get("meshes") or []
    tex_cache: dict[int | None, np.ndarray | None] = {}

    tri_positions: list[np.ndarray] = []
    tri_colors: list[np.ndarray] = []
    total_tris = 0

    def _visit_node(n_idx: int, parent_m: np.ndarray) -> None:
        nonlocal total_tris
        if not (0 <= n_idx < len(nodes)):
            return
        nd = nodes[n_idx]
        world_m = parent_m @ _node_local_mat4(nd)
        if "mesh" in nd and 0 <= int(nd["mesh"]) < len(meshes):
            mesh_def = meshes[int(nd["mesh"])]
            for prim in mesh_def.get("primitives") or []:
                if int(prim.get("mode", 4)) != 4:
                    continue
                attrs = prim.get("attributes") or {}
                if "POSITION" not in attrs:
                    continue
                pos = _read_gltf_accessor(gltf, bin_chunk, int(attrs["POSITION"])).astype(np.float32)
                if len(pos) < 3:
                    continue
                ones = np.ones((len(pos), 1), dtype=np.float32)
                pos_w = (np.hstack([pos, ones]) @ world_m.T)[:, :3]
                # glTF (+X right, +Y up, +Z toward viewer) -> Blender (+X right, +Y forward, +Z up)
                pos_bl = np.stack([pos_w[:, 0], -pos_w[:, 2], pos_w[:, 1]], axis=-1)

                if "indices" in prim:
                    idx = _read_gltf_accessor(gltf, bin_chunk, int(prim["indices"])).astype(np.int64).reshape(-1)
                else:
                    idx = np.arange(len(pos_bl), dtype=np.int64)
                n_tri = len(idx) // 3
                if n_tri <= 0:
                    continue
                total_tris += n_tri
                idx_tri = idx[: n_tri * 3].reshape((n_tri, 3))

                mat_idx = prim.get("material")
                mats = gltf.get("materials") or []
                base_fac = [0.82, 0.84, 0.88, 1.0]
                if mat_idx is not None and 0 <= int(mat_idx) < len(mats):
                    pbr = mats[int(mat_idx)].get("pbrMetallicRoughness") or {}
                    bf = pbr.get("baseColorFactor")
                    if isinstance(bf, list) and len(bf) >= 3:
                        base_fac = [float(bf[0]), float(bf[1]), float(bf[2]), float(bf[3]) if len(bf) > 3 else 1.0]

                v_col = np.broadcast_to(np.asarray(base_fac, dtype=np.float32), (len(pos_bl), 4)).copy()
                if "COLOR_0" in attrs:
                    c0 = _read_gltf_accessor(gltf, bin_chunk, int(attrs["COLOR_0"])).astype(np.float32)
                    if c0.ndim == 2 and c0.shape[0] == len(pos_bl):
                        v_col[:, : min(4, c0.shape[1])] *= c0[:, : min(4, c0.shape[1])]

                if "TEXCOORD_0" in attrs and mat_idx is not None:
                    m_key = int(mat_idx)
                    if m_key not in tex_cache:
                        tex_cache[m_key] = _decode_gltf_base_texture(gltf, bin_chunk, m_key)
                    tex_arr = tex_cache[m_key]
                    if tex_arr is not None:
                        uv = _read_gltf_accessor(gltf, bin_chunk, int(attrs["TEXCOORD_0"])).astype(np.float32)
                        if uv.ndim == 2 and uv.shape[0] == len(pos_bl):
                            th, tw = tex_arr.shape[:2]
                            u_px = np.clip((np.mod(uv[:, 0], 1.0) * (tw - 1)).astype(np.int32), 0, tw - 1)
                            v_px = np.clip((np.mod(uv[:, 1], 1.0) * (th - 1)).astype(np.int32), 0, th - 1)
                            sampled = tex_arr[v_px, u_px]
                            v_col[:, :3] = sampled[:, :3] * np.asarray(base_fac[:3], dtype=np.float32)

                tri_positions.append(pos_bl[idx_tri])  # (T, 3, 3)
                tri_colors.append(v_col[idx_tri])  # (T, 3, 4)

        for ch in nd.get("children") or []:
            _visit_node(int(ch), world_m)

    scenes = gltf.get("scenes") or []
    sc_idx = int(gltf.get("scene") or 0)
    root_nodes = (
        scenes[sc_idx].get("nodes")
        if 0 <= sc_idx < len(scenes)
        else list(range(len(nodes)))
    ) or []
    ident = np.eye(4, dtype=np.float32)
    for rn in root_nodes:
        _visit_node(int(rn), ident)

    if not tri_positions:
        return {
            "triangles": 0,
            "total_triangles": 0,
            "vertex_floats": [],
            "world_aabb": None,
        }

    all_pos = np.concatenate(tri_positions, axis=0)  # (N_tri, 3, 3)
    all_col = np.concatenate(tri_colors, axis=0)  # (N_tri, 3, 4)

    mn = all_pos.reshape((-1, 3)).min(axis=0)
    mx = all_pos.reshape((-1, 3)).max(axis=0)
    if place_on_ground:
        cx = 0.5 * float(mn[0] + mx[0])
        cy = 0.5 * float(mn[1] + mx[1])
        z_floor = float(mn[2])
        offset_vec = np.asarray([-cx, -cy, -z_floor], dtype=np.float32)
        all_pos = all_pos + offset_vec
        mn = mn + offset_vec
        mx = mx + offset_vec

    stride = max(1, len(all_pos) // max(500, int(max_triangles)))
    sub_pos = all_pos[::stride]
    sub_col = np.clip(all_col[::stride], 0.0, 1.0)

    e1 = sub_pos[:, 1] - sub_pos[:, 0]
    e2 = sub_pos[:, 2] - sub_pos[:, 0]
    nrm = np.cross(e1, e2)
    n_len = np.linalg.norm(nrm, axis=-1, keepdims=True)
    valid = (n_len[:, 0] > 1e-10)
    sub_pos = sub_pos[valid]
    sub_col = sub_col[valid]
    nrm = nrm[valid] / np.maximum(n_len[valid], 1e-10)
    nrm_rep = np.repeat(nrm[:, None, :], 3, axis=1)

    packed = np.concatenate([sub_pos, nrm_rep, sub_col], axis=-1).astype(np.float32)  # (K, 3, 10)
    sz = mx - mn
    ct = 0.5 * (mn + mx)
    return {
        "triangles": len(sub_pos),
        "total_triangles": total_tris,
        "vertex_floats": np.round(packed.reshape(-1), 5).tolist(),
        "world_aabb": {
            "min": [round(float(x), 4) for x in mn],
            "max": [round(float(x), 4) for x in mx],
            "size": [round(float(x), 4) for x in sz],
            "center": [round(float(x), 4) for x in ct],
        },
    }