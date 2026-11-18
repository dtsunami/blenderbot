"""OpenRouter Seedance 2.5 Async Video API client & FFmpeg video-conditioning decomposer.

Implements the confirmed OpenRouter Video API contract (`https://openrouter.ai/bytedance/seedance-2.5`):
  1. Submit (`202 Accepted`): `POST https://openrouter.ai/api/v1/videos`
  2. Poll: `GET https://openrouter.ai/api/v1/videos/{id}` (`pending` | `in_progress` -> `completed`)
  3. Download: `GET https://openrouter.ai/api/v1/videos/{id}/content` (fallback `unsigned_urls[0]`)
  4. Billing: `usage.cost` (USD) returned for session ledger & `BLENDERBOT_MAX_COST` tracking.

Supports conditioning on:
  - `prompt` (with optional splat-friendly `orbit_plate` / `subject_4d` camera hints)
  - `first_frame` / `last_frame` images (`frame_images` with `frame_type`)
  - `reference_images` & `audio_path` (`input_references` with `image_url` / `audio_url`)
  - `video_path` (local `.mp4`/`.mov`/`.webm` automatically decomposed via `ffprobe` + `ffmpeg`
    into `first_frame`, optional `last_frame`, interior `reference_images`, and optional MP3 `audio_url`).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import json
import math
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp

from blenderbot.providers.index import OPENROUTER_BASE_URL

DEFAULT_SEEDANCE_MODEL = "bytedance/seedance-2.5"
ESTIMATED_COST_PER_SEC_USD = 0.1028

SEEDANCE_SIZES: tuple[str, ...] = (
    "854x480",
    "752x560",
    "640x640",
    "560x752",
    "480x854",
    "992x432",
    "1280x720",
    "1112x834",
    "960x960",
    "834x1112",
    "720x1280",
    "1470x630",
    "1920x1080",
    "1440x1080",
    "1080x1080",
    "1080x1440",
    "1080x1920",
    "2520x1080",
)

SEEDANCE_RESOLUTIONS: tuple[str, ...] = ("480p", "720p", "1080p")
SEEDANCE_ASPECT_RATIOS: tuple[str, ...] = ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9")
SEEDANCE_MODES: dict[str, str] = {
    "standard": "",
    "orbit_plate": (
        " Smooth steady camera arc with clear multi-view parallax, sharp 35mm deep-focus "
        "optics, consistent scene lighting and exposure, zero motion blur, no jump cuts."
    ),
    "subject_4d": (
        " Coherent continuous subject motion, crisp silhouette separation, steady camera "
        "framing, consistent lighting, no abrupt cuts."
    ),
}

SEEDANCE_MIMES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}


class SeedanceError(Exception):
    """Raised when Seedance payload validation, submission, polling, or download fails."""

    def __init__(self, message: str, *, status: int | None = None, job_id: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.job_id = job_id


def snap_seedance_size(width: int, height: int) -> str:
    """Snap arbitrary `(width, height)` to the closest supported Seedance `WIDTHxHEIGHT` enum."""
    if width <= 0 or height <= 0:
        return "1280x720"
    target_ar = width / height
    target_area = width * height
    best = "1280x720"
    best_score = float("inf")
    for cand in SEEDANCE_SIZES:
        cw_s, ch_s = cand.split("x", 1)
        cw, ch = int(cw_s), int(ch_s)
        cand_ar = cw / ch
        ar_diff = abs(math.log(target_ar / cand_ar))
        area_diff = abs(math.log(max(1, target_area) / (cw * ch)))
        score = ar_diff * 3.0 + area_diff * 0.35
        if score < best_score:
            best_score = score
            best = cand
    return best


def clamp_seedance_duration(duration_sec: float | None) -> int | None:
    if duration_sec is None:
        return None
    val = round(float(duration_sec))
    if val == -1:
        return -1
    return max(4, min(30, val))


def encode_file_data_url(path: str | Path, mime_override: str | None = None) -> str:
    s = str(path).strip()
    if s.startswith(("https://", "http://", "data:")):
        return s
    p = Path(s)
    if not p.is_file():
        raise SeedanceError(f"Conditioning file does not exist: {p}")
    ext = p.suffix.lower()
    mime = mime_override or SEEDANCE_MIMES.get(ext) or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def probe_media_info(path: str | Path) -> dict[str, Any]:
    """Inspect duration, dimensions, fps, and audio presence via `ffprobe`."""
    p = Path(path)
    if not p.is_file():
        raise SeedanceError(f"Media file not found: {p}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"path": str(p), "size_bytes": p.stat().st_size}

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"path": str(p), "size_bytes": p.stat().st_size}

    try:
        info = json.loads(proc.stdout)
    except ValueError:
        return {"path": str(p), "size_bytes": p.stat().st_size}

    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    width = 0
    height = 0
    has_audio = False
    fps = 0.0
    for st in streams:
        stype = st.get("codec_type")
        if stype == "video" and width == 0:
            width = int(st.get("width") or 0)
            height = int(st.get("height") or 0)
            r_fr = str(st.get("r_frame_rate") or "0/1")
            if "/" in r_fr:
                num_s, den_s = r_fr.split("/", 1)
                if float(den_s or 0) > 0:
                    fps = round(float(num_s) / float(den_s), 3)
        elif stype == "audio":
            has_audio = True

    dur = 0.0
    if fmt.get("duration"):
        try:
            dur = float(fmt["duration"])
        except (TypeError, ValueError):
            dur = 0.0

    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "duration_sec": round(dur, 3) if dur > 0 else None,
        "width": width or None,
        "height": height or None,
        "fps": fps or None,
        "has_audio": has_audio,
    }


def _extract_frame_data_url(
    ffmpeg: str, video_path: Path, timestamp_sec: float, out_file: Path, *, max_dim: int = 1920
) -> str | None:
    scale_expr = f"scale='min({max_dim},iw)':-2"
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(0.0, timestamp_sec):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        scale_expr,
    ]
    if out_file.suffix.lower() in (".jpg", ".jpeg"):
        cmd.extend(["-q:v", "3"])
    cmd.append(str(out_file))
    proc = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    if proc.returncode != 0 or not out_file.is_file() or out_file.stat().st_size == 0:
        return None
    return encode_file_data_url(out_file)


def _transcode_audio_mp3_data_url(ffmpeg: str, media_path: Path, out_mp3: Path) -> str | None:
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-b:a",
        "64k",
        "-t",
        "30",
        str(out_mp3),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    if proc.returncode != 0 or not out_mp3.is_file() or out_mp3.stat().st_size == 0:
        return None
    return encode_file_data_url(out_mp3, mime_override="audio/mpeg")


def decompose_video_for_seedance(
    video_path: str | Path,
    *,
    lock_first_frame: bool = True,
    lock_last_frame: bool = True,
    mid_reference_count: int = 2,
    extract_audio: bool = False,
) -> dict[str, Any]:
    """Decompose a local video into Seedance 2.5 `frame_images` + `input_references` data URLs."""
    vp = Path(video_path)
    if not vp.is_file():
        raise SeedanceError(f"Video file not found: {vp}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SeedanceError("FFmpeg is required to extract conditioning keyframes from video files, but was not found on PATH.")

    meta = probe_media_info(vp)
    dur = float(meta.get("duration_sec") or 4.0)
    width = int(meta.get("width") or 1280)
    height = int(meta.get("height") or 720)

    first_url: str | None = None
    last_url: str | None = None
    mid_urls: list[str] = []
    audio_url: str | None = None

    with tempfile.TemporaryDirectory(prefix="bb_seedance_") as tmp_s:
        tmp_dir = Path(tmp_s)
        if lock_first_frame:
            first_url = _extract_frame_data_url(ffmpeg, vp, 0.0, tmp_dir / "first.png", max_dim=1920)
        if lock_last_frame and dur > 0.25:
            last_t = max(0.0, dur - 0.10)
            last_url = _extract_frame_data_url(ffmpeg, vp, last_t, tmp_dir / "last.png", max_dim=1920)
            if last_url is None and dur > 0.5:
                last_url = _extract_frame_data_url(ffmpeg, vp, max(0.0, dur - 0.35), tmp_dir / "last_fb.png", max_dim=1920)

        n_mid = max(0, min(4, int(mid_reference_count)))
        if n_mid > 0 and dur > 0.4:
            for idx in range(n_mid):
                frac = (idx + 1) / (n_mid + 1)
                u = _extract_frame_data_url(
                    ffmpeg, vp, dur * frac, tmp_dir / f"mid_{idx}.jpg", max_dim=1280
                )
                if u:
                    mid_urls.append(u)

        if extract_audio and meta.get("has_audio"):
            audio_url = _transcode_audio_mp3_data_url(ffmpeg, vp, tmp_dir / "track.mp3")

    return {
        "source_video": vp.name,
        "source_duration_sec": round(dur, 2),
        "recommended_duration": clamp_seedance_duration(dur) or 5,
        "recommended_size": snap_seedance_size(width, height),
        "first_frame_url": first_url,
        "last_frame_url": last_url,
        "mid_reference_urls": mid_urls,
        "audio_url": audio_url,
    }


def prepare_seedance_request(
    *,
    model: str = DEFAULT_SEEDANCE_MODEL,
    prompt: str | None = None,
    mode: str = "standard",
    duration: float | None = None,
    size: str | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    first_frame: str | Path | None = None,
    last_frame: str | Path | None = None,
    video_path: str | Path | None = None,
    video_path_2: str | Path | None = None,
    dual_video_mode: str = "bridge_v1_to_v2",
    video_lock_first: bool = True,
    video_lock_last: bool = True,
    video_mid_frames: int = 2,
    extract_video_audio: bool = False,
    reference_images: list[str | Path] | None = None,
    reference_videos: list[str | Path] | None = None,
    audio_path: str | Path | None = None,
    generate_audio: bool | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the validated JSON request body for `POST /api/v1/videos` + a human-readable conditioning summary."""
    frame_images: list[dict[str, Any]] = []
    input_references: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "model": model or DEFAULT_SEEDANCE_MODEL,
        "mode": mode,
        "first_frame": False,
        "last_frame": False,
        "reference_images": 0,
        "audio_reference": False,
        "source_video": None,
    }

    has_v1 = video_path is not None and bool(str(video_path).strip())
    has_v2 = video_path_2 is not None and bool(str(video_path_2).strip())

    vid_decomp: dict[str, Any] | None = None
    vid2_decomp: dict[str, Any] | None = None
    if has_v1:
        assert video_path is not None
        need_v1_first = (
            (dual_video_mode != "bridge_v1_to_v2") if has_v2 else video_lock_first
        ) and not first_frame
        need_v1_last = (
            (dual_video_mode in ("bridge_v1_to_v2", "v1_primary_v2_ref"))
            if has_v2
            else video_lock_last
        ) and not (last_frame and not (has_v2 and dual_video_mode == "bridge_v1_to_v2"))
        vid_decomp = decompose_video_for_seedance(
            video_path,
            lock_first_frame=need_v1_first,
            lock_last_frame=need_v1_last,
            mid_reference_count=video_mid_frames,
            extract_audio=extract_video_audio and not audio_path,
        )
        summary["source_video"] = vid_decomp["source_video"]

    if has_v2:
        assert video_path_2 is not None
        vid2_decomp = decompose_video_for_seedance(
            video_path_2,
            lock_first_frame=(dual_video_mode == "bridge_v1_to_v2") and not last_frame,
            lock_last_frame=(dual_video_mode == "span_v1_to_v2" or not has_v1) and not last_frame,
            mid_reference_count=max(1, video_mid_frames),
            extract_audio=extract_video_audio and not has_v1 and not audio_path,
        )
        summary["source_video_2"] = vid2_decomp["source_video"]
        summary["dual_video_mode"] = dual_video_mode

    # 1. First & Last Frame Conditioning (`frame_images`)
    eff_first_url: str | None = None
    if first_frame is not None and str(first_frame).strip():
        eff_first_url = encode_file_data_url(first_frame)
    elif has_v1 and has_v2 and dual_video_mode == "bridge_v1_to_v2":
        # Seamless transition: start from Video 1's closing frame!
        eff_first_url = (vid_decomp or {}).get("last_frame_url")
    elif vid_decomp and vid_decomp.get("first_frame_url"):
        eff_first_url = vid_decomp["first_frame_url"]
    elif not has_v1 and vid2_decomp and vid2_decomp.get("first_frame_url"):
        eff_first_url = vid2_decomp["first_frame_url"]

    if eff_first_url:
        frame_images.append(
            {
                "type": "image_url",
                "image_url": {"url": eff_first_url},
                "frame_type": "first_frame",
            }
        )
        summary["first_frame"] = True

    eff_last_url: str | None = None
    if last_frame is not None and str(last_frame).strip():
        eff_last_url = encode_file_data_url(last_frame)
    elif has_v1 and has_v2 and dual_video_mode == "bridge_v1_to_v2":
        # Seamless transition: land on Video 2's opening frame!
        eff_last_url = (vid2_decomp or {}).get("first_frame_url")
    elif has_v1 and has_v2 and dual_video_mode == "span_v1_to_v2":
        # Span arc: start at Video 1's opening frame, land on Video 2's closing frame!
        eff_last_url = (vid2_decomp or {}).get("last_frame_url")
    elif vid_decomp and vid_decomp.get("last_frame_url") and video_lock_last:
        eff_last_url = vid_decomp["last_frame_url"]
    elif not has_v1 and vid2_decomp and vid2_decomp.get("last_frame_url"):
        eff_last_url = vid2_decomp["last_frame_url"]

    if eff_last_url:
        frame_images.append(
            {
                "type": "image_url",
                "image_url": {"url": eff_last_url},
                "frame_type": "last_frame",
            }
        )
        summary["last_frame"] = True

    # 2. Reference Images & Audio (`input_references`)
    for dec_obj in (vid_decomp, vid2_decomp):
        if dec_obj:
            for mid_u in dec_obj.get("mid_reference_urls") or []:
                input_references.append({"type": "image_url", "image_url": {"url": mid_u}})
                summary["reference_images"] += 1

    summary["reference_videos"] = []
    for ref_vid in reference_videos or []:
        if ref_vid is not None and str(ref_vid).strip():
            rv_dec = decompose_video_for_seedance(
                ref_vid,
                lock_first_frame=False,
                lock_last_frame=False,
                mid_reference_count=2,
                extract_audio=False,
            )
            summary["reference_videos"].append(rv_dec["source_video"])
            for mid_u in rv_dec.get("mid_reference_urls") or []:
                input_references.append({"type": "image_url", "image_url": {"url": mid_u}})
                summary["reference_images"] += 1

    for ref_img in reference_images or []:
        if ref_img is not None and str(ref_img).strip():
            input_references.append(
                {"type": "image_url", "image_url": {"url": encode_file_data_url(ref_img)}}
            )
            summary["reference_images"] += 1

    eff_audio_url: str | None = None
    if audio_path is not None and str(audio_path).strip():
        ap_str = str(audio_path).strip()
        if ap_str.startswith(("https://", "http://", "data:")):
            eff_audio_url = ap_str
        else:
            ap = Path(ap_str)
            ffmpeg = shutil.which("ffmpeg")
            if ap.suffix.lower() not in (".mp3", ".wav", ".ogg", ".m4a") and ffmpeg:
                with tempfile.TemporaryDirectory(prefix="bb_aud_") as td:
                    eff_audio_url = _transcode_audio_mp3_data_url(ffmpeg, ap, Path(td) / "ref.mp3")
            else:
                eff_audio_url = encode_file_data_url(ap)
    elif vid_decomp and vid_decomp.get("audio_url"):
        eff_audio_url = vid_decomp["audio_url"]

    if eff_audio_url:
        input_references.append({"type": "audio_url", "audio_url": {"url": eff_audio_url}})
        summary["audio_reference"] = True

    # 3. Prompt & Mode Hints
    clean_prompt = (prompt or "").strip()
    mode_suffix = SEEDANCE_MODES.get(mode, "")
    if clean_prompt and mode_suffix:
        clean_prompt = f"{clean_prompt}{mode_suffix}"

    if not clean_prompt and not frame_images:
        raise SeedanceError("Seedance 2.5 requires 'prompt' unless 'first_frame' / 'last_frame' (or 'video_path') is provided.")

    payload: dict[str, Any] = {"model": model or DEFAULT_SEEDANCE_MODEL}
    if clean_prompt:
        payload["prompt"] = clean_prompt

    # 4. Duration & Dimensions
    eff_dur = clamp_seedance_duration(duration)
    if eff_dur is None and vid_decomp:
        eff_dur = vid_decomp["recommended_duration"]
    if eff_dur is not None:
        payload["duration"] = eff_dur
        summary["duration"] = eff_dur

    if size:
        s_clean = size.strip().lower()
        if s_clean not in SEEDANCE_SIZES:
            if "x" in s_clean:
                w_s, h_s = s_clean.split("x", 1)
                if w_s.isdigit() and h_s.isdigit():
                    s_clean = snap_seedance_size(int(w_s), int(h_s))
                else:
                    raise SeedanceError(f"Unsupported Seedance size '{size}'. Allowed: {', '.join(SEEDANCE_SIZES)}")
            else:
                raise SeedanceError(f"Unsupported Seedance size '{size}'. Allowed: {', '.join(SEEDANCE_SIZES)}")
        payload["size"] = s_clean
        summary["size"] = s_clean
    elif resolution or aspect_ratio:
        if resolution:
            if resolution not in SEEDANCE_RESOLUTIONS:
                raise SeedanceError(f"Unsupported resolution '{resolution}' (expected 480p, 720p, or 1080p).")
            payload["resolution"] = resolution
            summary["resolution"] = resolution
        if aspect_ratio:
            if aspect_ratio not in SEEDANCE_ASPECT_RATIOS:
                raise SeedanceError(f"Unsupported aspect_ratio '{aspect_ratio}'.")
            payload["aspect_ratio"] = aspect_ratio
            summary["aspect_ratio"] = aspect_ratio
    elif vid_decomp and vid_decomp.get("recommended_size"):
        payload["size"] = vid_decomp["recommended_size"]
        summary["size"] = vid_decomp["recommended_size"]

    if frame_images:
        payload["frame_images"] = frame_images
    if input_references:
        payload["input_references"] = input_references
    if generate_audio is not None:
        payload["generate_audio"] = bool(generate_audio)
    if seed is not None:
        payload["seed"] = int(seed)

    est_sec = eff_dur if (eff_dur and eff_dur > 0) else 5
    summary["estimated_cost_usd"] = round(est_sec * ESTIMATED_COST_PER_SEC_USD, 4)
    return payload, summary


def _parse_api_error(status: int, raw_text: str) -> str:
    try:
        obj = json.loads(raw_text)
        err = obj.get("error")
        if isinstance(err, dict):
            return f"[{status}] {err.get('message') or raw_text[:300]}"
        if isinstance(err, str):
            return f"[{status}] {err}"
    except ValueError:
        pass
    return f"[{status}] {raw_text[:300] or 'HTTP error'}"


def _extract_video_cost_per_sec(item: dict[str, Any], model_id: str) -> float:
    pricing = item.get("pricing") or {}
    if isinstance(pricing, dict):
        for k in ("video_second", "per_second", "second", "video"):
            v = pricing.get(k)
            if v is not None:
                with contextlib.suppress(ValueError, TypeError):
                    fv = float(v)
                    if fv > 0:
                        return fv
    if "seedance-2.5" in model_id.lower():
        return ESTIMATED_COST_PER_SEC_USD
    return ESTIMATED_COST_PER_SEC_USD


async def fetch_openrouter_video_models(
    *, api_key: str | None = None, base_url: str = OPENROUTER_BASE_URL
) -> list[dict[str, Any]]:
    """Query `GET /api/v1/videos/models` for available OpenRouter video models & pricing."""
    default_item = {
        "id": DEFAULT_SEEDANCE_MODEL,
        "name": "ByteDance: Seedance 2.5",
        "cost_per_sec_usd": ESTIMATED_COST_PER_SEC_USD,
    }
    url = f"{base_url.rstrip('/')}/videos/models"
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
                out = []
                for m in rows:
                    if isinstance(m, dict) and m.get("id"):
                        mid = str(m["id"])
                        out.append(
                            {
                                "id": mid,
                                "name": str(m.get("name") or mid),
                                "cost_per_sec_usd": _extract_video_cost_per_sec(m, mid),
                            }
                        )
                if out:
                    return out
    return [default_item]


async def generate_seedance_video(
    *,
    api_key: str,
    output_path: str | Path,
    model: str = DEFAULT_SEEDANCE_MODEL,
    prompt: str | None = None,
    mode: str = "standard",
    duration: float | None = None,
    size: str | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    first_frame: str | Path | None = None,
    last_frame: str | Path | None = None,
    video_path: str | Path | None = None,
    video_path_2: str | Path | None = None,
    dual_video_mode: str = "bridge_v1_to_v2",
    video_lock_first: bool = True,
    video_lock_last: bool = True,
    video_mid_frames: int = 2,
    extract_video_audio: bool = False,
    reference_images: list[str | Path] | None = None,
    reference_videos: list[str | Path] | None = None,
    audio_path: str | Path | None = None,
    generate_audio: bool | None = None,
    seed: int | None = None,
    base_url: str | None = None,
    poll_interval_sec: float | None = None,
    timeout_sec: float = 600.0,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: asyncio.Event | None = None,
    http_session: aiohttp.ClientSession | None = None,
) -> dict[str, Any]:
    """Submit, poll, and download a Seedance 2.5 video clip via OpenRouter `/api/v1/videos`."""
    if not api_key or not api_key.strip():
        raise SeedanceError(
            "OPENROUTER_API_KEY is not configured. Set OPENROUTER_API_KEY in ~/.blenderbot/.env "
            "(or via `/keys openrouter <key> --save`).",
            status=401,
        )

    eff_base = (base_url or os.environ.get("BLENDERBOT_VIDEO_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
    eff_poll = (
        poll_interval_sec
        if poll_interval_sec is not None
        else float(os.environ.get("BLENDERBOT_VIDEO_POLL_SEC") or 2.5)
    )

    t0 = time.monotonic()
    if on_progress:
        on_progress({"stage": "preparing", "elapsed_sec": 0.0})

    payload, cond_summary = await asyncio.to_thread(
        prepare_seedance_request,
        model=model,
        prompt=prompt,
        mode=mode,
        duration=duration,
        size=size,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        first_frame=first_frame,
        last_frame=last_frame,
        video_path=video_path,
        video_path_2=video_path_2,
        dual_video_mode=dual_video_mode,
        video_lock_first=video_lock_first,
        video_lock_last=video_lock_last,
        video_mid_frames=video_mid_frames,
        extract_video_audio=extract_video_audio,
        reference_images=reference_images,
        reference_videos=reference_videos,
        audio_path=audio_path,
        generate_audio=generate_audio,
        seed=seed,
    )

    root_api = eff_base
    poll_interval_sec = eff_poll
    submit_url = f"{root_api}/videos"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/blenderbot/blenderbot",
        "X-Title": "blenderbot",
    }

    own_session = http_session is None
    session = http_session or aiohttp.ClientSession()
    try:
        if on_progress:
            on_progress(
                {
                    "stage": "submitting",
                    "elapsed_sec": round(time.monotonic() - t0, 1),
                    "conditioning": cond_summary,
                }
            )

        async with session.post(
            submit_url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as sub_resp:
            sub_text = await sub_resp.text()
            if sub_resp.status not in (200, 201, 202):
                raise SeedanceError(
                    f"Seedance submit failed {_parse_api_error(sub_resp.status, sub_text)}",
                    status=sub_resp.status,
                )
            sub_json = json.loads(sub_text)

        job_id = str(sub_json.get("id") or "").strip()
        if not job_id:
            raise SeedanceError(f"OpenRouter /videos returned no job id: {sub_text[:240]}")
        gen_id = sub_json.get("generation_id")
        raw_poll_url = str(sub_json.get("polling_url") or f"/api/v1/videos/{job_id}")
        if raw_poll_url.startswith("http"):
            poll_url = raw_poll_url
        elif raw_poll_url.startswith("/api/v1/"):
            poll_url = f"{root_api}/{raw_poll_url.removeprefix('/api/v1/')}"
        else:
            poll_url = f"{root_api}/videos/{job_id}"

        status_str = str(sub_json.get("status") or "pending")
        if on_progress:
            on_progress(
                {
                    "stage": status_str,
                    "job_id": job_id,
                    "generation_id": gen_id,
                    "elapsed_sec": round(time.monotonic() - t0, 1),
                }
            )

        # Poll until completed / failed / cancelled / expired
        job_json: dict[str, Any] = sub_json
        deadline = t0 + max(10.0, float(timeout_sec))
        while status_str in ("pending", "in_progress", "queued", "processing"):
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("Seedance video job cancelled by operator")
            if time.monotonic() >= deadline:
                raise SeedanceError(
                    f"Seedance job {job_id} timed out after {timeout_sec:.0f}s (last status: {status_str})",
                    status=408,
                    job_id=job_id,
                )
            await asyncio.sleep(poll_interval_sec)
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("Seedance video job cancelled by operator")

            async with session.get(
                poll_url,
                headers={"Authorization": headers["Authorization"], "Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as poll_resp:
                poll_text = await poll_resp.text()
                if poll_resp.status != 200:
                    raise SeedanceError(
                        f"Seedance poll failed for {job_id} {_parse_api_error(poll_resp.status, poll_text)}",
                        status=poll_resp.status,
                        job_id=job_id,
                    )
                job_json = json.loads(poll_text)

            status_str = str(job_json.get("status") or "unknown").lower()
            if on_progress:
                on_progress(
                    {
                        "stage": status_str,
                        "job_id": job_id,
                        "generation_id": gen_id,
                        "elapsed_sec": round(time.monotonic() - t0, 1),
                    }
                )

        if status_str != "completed":
            err_detail = job_json.get("error") or job_json.get("message") or status_str
            if isinstance(err_detail, dict):
                err_detail = err_detail.get("message") or json.dumps(err_detail)
            raise SeedanceError(
                f"Seedance job {job_id} {status_str}: {err_detail}",
                job_id=job_id,
            )

        # Download MP4 bytes from `/videos/{id}/content` (fallback to `unsigned_urls[0]`)
        content_url = f"{root_api}/videos/{job_id}/content"
        unsigned_urls = job_json.get("unsigned_urls") or []
        if on_progress:
            on_progress(
                {
                    "stage": "downloading",
                    "job_id": job_id,
                    "elapsed_sec": round(time.monotonic() - t0, 1),
                }
            )

        dst = Path(output_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_dst = dst.with_suffix(dst.suffix + ".part")

        downloaded = False
        async with session.get(
            content_url,
            headers={"Authorization": headers["Authorization"]},
            timeout=aiohttp.ClientTimeout(total=180),
        ) as dl_resp:
            if dl_resp.status == 200:
                raw_vid = await dl_resp.read()
                if raw_vid:
                    await asyncio.to_thread(tmp_dst.write_bytes, raw_vid)
                    downloaded = True

        if not downloaded and unsigned_urls:
            async with session.get(
                str(unsigned_urls[0]),
                timeout=aiohttp.ClientTimeout(total=180),
            ) as fb_resp:
                if fb_resp.status != 200:
                    raise SeedanceError(
                        f"Failed downloading completed video {job_id} (HTTP {fb_resp.status})",
                        status=fb_resp.status,
                        job_id=job_id,
                    )
                raw_vid = await fb_resp.read()
                if raw_vid:
                    await asyncio.to_thread(tmp_dst.write_bytes, raw_vid)
                    downloaded = True

        if not downloaded:
            raise SeedanceError(f"Seedance job {job_id} completed, but video download returned empty content.", job_id=job_id)

        tmp_dst.replace(dst)
    finally:
        if own_session:
            await session.close()

    usage = job_json.get("usage") or {}
    actual_cost = round(float(usage.get("cost") or 0.0), 6)
    out_meta = await asyncio.to_thread(probe_media_info, dst)
    elapsed_total = round(time.monotonic() - t0, 2)

    result = {
        "job_id": job_id,
        "generation_id": gen_id,
        "model": payload["model"],
        "output_path": str(dst),
        "size_bytes": dst.stat().st_size,
        "duration_sec": out_meta.get("duration_sec"),
        "width": out_meta.get("width"),
        "height": out_meta.get("height"),
        "fps": out_meta.get("fps"),
        "has_audio": out_meta.get("has_audio"),
        "cost_usd": actual_cost,
        "elapsed_sec": elapsed_total,
        "conditioning": cond_summary,
    }
    if on_progress:
        on_progress({"stage": "completed", **result})
    return result


def default_seedance_output_path(cwd: Path, prefix: str = "seedance") -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cwd / "assets" / "seedance"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{prefix}_{ts}.mp4"