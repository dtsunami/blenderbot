"""Local Windows host client for dispatching, controlling, resuming & rendering cloud 3DGS jobs.

Streams live telemetry into `assets/splats/{job_id}/telemetry.jsonl` + `snapshots/`, touches
`FlightRecorder` on every tick, emits SSE events on `SessionBus`, unpacks `splat.ply`,
`splat.splat`, `proxy_collision.glb`, and `cameras.json`, pushes live mid-train parameter/pause
updates via `modal.Dict`, and farms full-rate trajectory MP4 renders to Modal H100/L40S.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import gzip
import hashlib
import io
import json
import queue
import re
import tarfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from blenderbot.config.env import DEFAULT_MODAL_WORKER_TIMEOUT, resolve_modal_worker_timeout

from .config import ExtractConfig, SplatConfig, SurgerySpec

CTL_DICT_NAME = "blenderbot-splat-ctl"
SUPPORTED_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)


def _natural_sort_key(name: str) -> list[int | str]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def discover_image_files(
    images_path: Path | None = None,
    image_paths: Sequence[Path] | None = None,
) -> list[Path]:
    """Discover and natural-sort multi-view images from a directory or explicit path list."""
    found: list[Path] = []
    if image_paths:
        for p in image_paths:
            rp = Path(p)
            if rp.is_dir():
                found.extend(
                    sorted(
                        (
                            c
                            for c in rp.iterdir()
                            if c.is_file()
                            and not c.name.startswith((".", "._"))
                            and c.suffix.lower() in SUPPORTED_IMAGE_EXTS
                        ),
                        key=lambda c: _natural_sort_key(c.name),
                    )
                )
            elif rp.is_file() and rp.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                found.append(rp)
    elif images_path is not None:
        p_dir = Path(images_path)
        if not p_dir.is_dir():
            raise FileNotFoundError(f"Images directory does not exist: {p_dir}")
        direct = [
            c
            for c in p_dir.iterdir()
            if c.is_file()
            and not c.name.startswith((".", "._"))
            and c.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
        if len(direct) >= 3:
            found = sorted(direct, key=lambda c: _natural_sort_key(c.name))
        else:
            nested = [
                c
                for c in p_dir.rglob("*")
                if c.is_file()
                and not c.name.startswith((".", "._"))
                and c.suffix.lower() in SUPPORTED_IMAGE_EXTS
            ]
            found = sorted(nested, key=lambda c: _natural_sort_key(c.relative_to(p_dir).as_posix()))

    # Deduplicate preserving order
    deduped: list[Path] = []
    seen_res: set[Path] = set()
    for p in found:
        rp = p.resolve()
        if rp not in seen_res:
            seen_res.add(rp)
            deduped.append(p)
    if len(deduped) < 3:
        raise ValueError(
            f"Need at least 3 valid images ({', '.join(sorted(SUPPORTED_IMAGE_EXTS))}) for 3DGS; found {len(deduped)}."
        )
    return deduped


def pack_images_tar_gz(
    image_files: Sequence[Path],
    cfg: ExtractConfig | None = None,
) -> bytes:
    """EXIF-orient, optionally downscale (`max_width`), and pack multi-view photos into `.tar.gz` bytes.

    Pre-scaling 24MP/48MP camera JPEGs locally reduces upload size ~15x (e.g. 155 MB -> 8 MB)
    while avoiding lossy H.264 MP4 re-encoding and preserving RGBA silhouettes when present.
    """
    eff_ext = cfg or ExtractConfig()
    files = list(image_files)
    if len(files) < 3:
        raise ValueError(f"Need at least 3 images to pack; got {len(files)}.")
    if eff_ext.max_frames > 0 and len(files) > eff_ext.max_frames:
        idxs = np.linspace(0, len(files) - 1, eff_ext.max_frames, dtype=int)
        files = [files[int(i)] for i in idxs]

    jpg_q = max(75, min(100, int(eff_ext.quality or 95)))
    used_names: set[str] = set()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=1) as tf:
        for idx, src_p in enumerate(files):
            with Image.open(src_p) as im_raw:
                im = ImageOps.exif_transpose(im_raw)
                has_alpha = False
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    im_rgba = im.convert("RGBA")
                    extrema = im_rgba.getchannel("A").getextrema()
                    if extrema and extrema[0] < 250:
                        has_alpha = True
                    im_work = im_rgba if has_alpha else im_rgba.convert("RGB")
                else:
                    im_work = im.convert("RGB")

                if eff_ext.max_width > 0 and max(im_work.width, im_work.height) > eff_ext.max_width:
                    scale = eff_ext.max_width / float(max(im_work.width, im_work.height))
                    nw = max(2, round(im_work.width * scale / 2.0) * 2)
                    nh = max(2, round(im_work.height * scale / 2.0) * 2)
                    im_work = im_work.resize((nw, nh), Image.LANCZOS)
                else:
                    nw = max(2, (im_work.width // 2) * 2)
                    nh = max(2, (im_work.height // 2) * 2)
                    if (nw, nh) != (im_work.width, im_work.height):
                        im_work = im_work.resize((nw, nh), Image.LANCZOS)

                clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", src_p.stem).strip("._") or f"view_{idx:04d}"
                ext_out = ".png" if has_alpha else ".jpg"
                arc_name = f"{clean_stem}{ext_out}"
                if arc_name.lower() in used_names:
                    arc_name = f"{idx:04d}_{clean_stem}{ext_out}"
                used_names.add(arc_name.lower())

                img_io = io.BytesIO()
                if has_alpha:
                    im_work.save(img_io, format="PNG", compress_level=3)
                else:
                    im_work.save(
                        img_io,
                        format="JPEG",
                        quality=jpg_q,
                        subsampling=0 if jpg_q >= 92 else 2,
                    )
                im_work.close()
                raw_bytes = img_io.getvalue()

            ti = tarfile.TarInfo(name=arc_name)
            ti.size = len(raw_bytes)
            ti.mtime = int(time.time())
            tf.addfile(ti, io.BytesIO(raw_bytes))

    return buf.getvalue()


def unpack_images_bundle(
    images_tar_gz: bytes,
    frames_dir: Path,
    cfg: ExtractConfig | None = None,
) -> tuple[list[Path], dict[str, float], dict[str, Any], bool]:
    """Unpack a `.tar.gz` of multi-view photos, normalize EXIF orientation/scale, and build PTS metadata."""
    import shutil

    eff_cfg = cfg or ExtractConfig()
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = frames_dir.parent / "raw_images_unpack"
    if raw_dir.exists():
        shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=io.BytesIO(images_tar_gz), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            rel_parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in rel_parts:
                continue
            leaf = rel_parts[-1] if rel_parts else ""
            if not leaf or leaf.startswith((".", "._")):
                continue
            if Path(leaf).suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue
            tf.extract(member, path=raw_dir)

    candidates = [
        p
        for p in raw_dir.rglob("*")
        if p.is_file()
        and not p.name.startswith((".", "._"))
        and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
    candidates.sort(key=lambda p: _natural_sort_key(p.relative_to(raw_dir).as_posix()))

    if eff_cfg.max_frames > 0 and len(candidates) > eff_cfg.max_frames:
        idxs = np.linspace(0, len(candidates) - 1, eff_cfg.max_frames, dtype=int)
        candidates = [candidates[int(i)] for i in idxs]

    if len(candidates) < 3:
        shutil.rmtree(raw_dir, ignore_errors=True)
        raise RuntimeError(
            f"Too few valid images in bundle ({len(candidates)}). Provide at least 3 overlapping views."
        )

    decoded_items: list[tuple[str, Image.Image, bool]] = []
    any_alpha = False
    used_stems: set[str] = set()
    for idx, src_p in enumerate(candidates):
        with Image.open(src_p) as im_raw:
            im = ImageOps.exif_transpose(im_raw)
            has_a = False
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im_rgba = im.convert("RGBA")
                extrema = im_rgba.getchannel("A").getextrema()
                if extrema and extrema[0] < 250:
                    has_a = True
                    any_alpha = True
                im_work = im_rgba if has_a else im_rgba.convert("RGB")
            else:
                im_work = im.convert("RGB")

            if eff_cfg.max_width > 0 and max(im_work.width, im_work.height) > eff_cfg.max_width:
                scale = eff_cfg.max_width / float(max(im_work.width, im_work.height))
                nw = max(2, round(im_work.width * scale / 2.0) * 2)
                nh = max(2, round(im_work.height * scale / 2.0) * 2)
                im_work = im_work.resize((nw, nh), Image.LANCZOS)
            else:
                nw = max(2, (im_work.width // 2) * 2)
                nh = max(2, (im_work.height // 2) * 2)
                if (nw, nh) != (im_work.width, im_work.height):
                    im_work = im_work.resize((nw, nh), Image.LANCZOS)

            clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", src_p.stem).strip("._") or f"view_{idx:04d}"
            if clean_stem.lower() in used_stems:
                clean_stem = f"{idx:04d}_{clean_stem}"
            used_stems.add(clean_stem.lower())
            decoded_items.append((clean_stem, im_work, has_a))

    shutil.rmtree(raw_dir, ignore_errors=True)

    w0, h0 = decoded_items[0][1].size
    out_paths: list[Path] = []
    jpg_q = max(75, min(100, int(eff_cfg.quality or 95)))
    for clean_stem, im_work, _has_a in decoded_items:
        if any_alpha:
            out_f = frames_dir / f"{clean_stem}.png"
            im_work.convert("RGBA").save(out_f, format="PNG", compress_level=3)
        else:
            out_f = frames_dir / f"{clean_stem}.jpg"
            im_work.convert("RGB").save(
                out_f,
                format="JPEG",
                quality=jpg_q,
                subsampling=0 if jpg_q >= 92 else 2,
            )
        im_work.close()
        out_paths.append(out_f)

    eff_fps = max(1.0, float(eff_cfg.fps or 8.0))
    dur_sec = round(len(out_paths) / eff_fps, 4)
    pts_by_name = {fp.name: round(i / eff_fps, 5) for i, fp in enumerate(out_paths)}
    image_meta: dict[str, Any] = {
        "source": "images",
        "duration_sec": dur_sec,
        "fps": round(eff_fps, 4),
        "total_frames": len(out_paths),
        "width": int(w0),
        "height": int(h0),
    }
    return out_paths, pts_by_name, image_meta, any_alpha


def _resolve_client_timeout(
    timeout_sec: float | None,
    *,
    default: int = DEFAULT_MODAL_WORKER_TIMEOUT,
) -> int:
    if timeout_sec is not None and float(timeout_sec) > 0:
        return max(1, round(float(timeout_sec)))
    return resolve_modal_worker_timeout(default=default)


def _bind_modal_worker_cls(worker_cls: Any, timeout_sec: int) -> Any:
    with_opts = getattr(worker_cls, "with_options", None)
    if callable(with_opts):
        return with_opts(timeout=int(timeout_sec))
    return worker_cls


@dataclass
class SplatJobResult:
    job_id: str
    burst_id: int
    step: int
    n_gaussians: int
    output_dir: Path
    ply_path: Path
    splat_path: Path | None
    proxy_glb_path: Path | None
    cameras_path: Path
    telemetry_path: Path
    device: str
    wall_s: float
    cost_usd: float
    macro_timings: dict[str, float]


@dataclass
class SplatRenderResult:
    job_id: str
    output_mp4: Path
    frames: int
    fps: float
    width: int
    height: int
    device: str
    wall_s: float
    render_fps: float
    cost_usd: float


def make_job_id(video_path: Path) -> str:
    stem = "".join(c if (c.isalnum() or c in "_-") else "_" for c in video_path.stem)[:28].strip("_")
    stamp = datetime.datetime.now().strftime("%H%M%S")
    digest = hashlib.sha1(f"{video_path.resolve()}:{time.time()}".encode()).hexdigest()[:6]
    return f"{stem or 'splat'}_{stamp}_{digest}"


def _write_job_outputs(
    out_dir: Path,
    done_ev: dict[str, Any],
) -> tuple[Path, Path | None, Path | None, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "splat.ply"
    ply_raw = gzip.decompress(done_ev["ply_gz"])
    ply_path.write_bytes(ply_raw)

    splat_path: Path | None = None
    if done_ev.get("splat_gz"):
        splat_path = out_dir / "splat.splat"
        splat_path.write_bytes(gzip.decompress(done_ev["splat_gz"]))

    proxy_path: Path | None = None
    if done_ev.get("proxy_glb_gz"):
        proxy_path = out_dir / "proxy_collision.glb"
        proxy_path.write_bytes(gzip.decompress(done_ev["proxy_glb_gz"]))

    cameras_path = out_dir / "cameras.json"
    cameras_path.write_text(done_ev["cameras_json"], encoding="utf-8")
    return ply_path, splat_path, proxy_path, cameras_path


def _persist_local_telemetry(out_dir: Path, ev: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    telem_file = out_dir / "telemetry.jsonl"
    scalar_row = {
        k: v
        for k, v in ev.items()
        if not k.endswith("_b64")
        and k not in ("ply_gz", "splat_gz", "proxy_glb_gz", "cameras_json", "mp4_bytes")
    }
    with open(telem_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(scalar_row) + "\n")

    if ev.get("event") == "telemetry" and ev.get("render_b64"):
        latest_dir = out_dir / "previews"
        latest_dir.mkdir(parents=True, exist_ok=True)
        (latest_dir / "latest_render.jpg").write_bytes(base64.b64decode(ev["render_b64"]))
        if ev.get("diff_b64"):
            (latest_dir / "latest_diff.jpg").write_bytes(base64.b64decode(ev["diff_b64"]))
        if ev.get("gt_b64"):
            cam_stem = Path(ev.get("camera_name", "cam")).stem
            (latest_dir / f"gt_{cam_stem}.jpg").write_bytes(base64.b64decode(ev["gt_b64"]))
        if ev.get("is_snapshot"):
            snap_dir = out_dir / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            step_n = int(ev.get("step", 0))
            (snap_dir / f"step_{step_n:06d}.jpg").write_bytes(base64.b64decode(ev["render_b64"]))


async def push_modal_splat_control(job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Push a live mid-training control dict (`paused`, `stop_early`, `save_checkpoint`, `lambda_dssim`, etc.)."""

    def _sync_push() -> dict[str, Any]:
        import modal

        d = modal.Dict.from_name(CTL_DICT_NAME, create_if_missing=True)
        cur = d.get(job_id)
        merged = dict(cur) if isinstance(cur, dict) else {}
        merged.update(updates)
        merged["seq"] = int(merged.get("seq", 0)) + 1
        merged["ts"] = round(time.time(), 3)
        d[job_id] = merged
        return merged

    return await asyncio.to_thread(_sync_push)


async def run_modal_splat_job(
    *,
    output_dir: Path,
    job_id: str,
    video_path: Path | None = None,
    images_path: Path | None = None,
    image_paths: Sequence[Path] | None = None,
    images_tar_gz: bytes | None = None,
    views_bundle_path: Path | None = None,
    known_cameras_path: Path | None = None,
    known_cameras_json: str | None = None,
    cfg: SplatConfig | None = None,
    cfg_overrides: dict[str, Any] | None = None,
    add_steps: int | None = None,
    surgery: SurgerySpec | None = None,
    masks_paths: dict[str, Path] | None = None,
    exclude_frames: list[str] | None = None,
    timeout_sec: float | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> SplatJobResult:
    """Dispatch `from_video`, `from_images`, `from_blender` (`views_bundle_path`), or `resume` to Modal."""
    import modal

    has_images_src = (
        images_tar_gz is not None
        or images_path is not None
        or (image_paths is not None and len(image_paths) > 0)
    )
    is_resume = (video_path is None) and (views_bundle_path is None) and (not has_images_src)
    eff_gpu = str(
        (cfg_overrides or {}).get("gpu")
        or (cfg.gpu if cfg is not None else "H100")
    ).upper()
    cls_name = "SplatWorkerL40S" if eff_gpu == "L40S" else "SplatWorker"
    eff_timeout = _resolve_client_timeout(timeout_sec, default=DEFAULT_MODAL_WORKER_TIMEOUT)

    # On initial run pass full `cfg_dict`; on `resume` pass `cfg_dict=None` + `cfg_overrides`
    # so `core_worker.py` merges overrides onto `meta.json["config"]` without clobbering saved settings!
    base_cfg: SplatConfig | None = None
    if is_resume:
        send_cfg_dict = cfg.to_dict() if cfg is not None else None
        send_overrides = cfg_overrides
    else:
        base_cfg = cfg if cfg is not None else SplatConfig()
        if cfg_overrides:
            base_cfg = base_cfg.merge_overrides(cfg_overrides)
        send_cfg_dict = base_cfg.to_dict()
        send_overrides = None

    mp4_bytes = video_path.read_bytes() if video_path is not None else None
    views_tar_gz = views_bundle_path.read_bytes() if views_bundle_path is not None else None
    eff_images_tar_gz = images_tar_gz
    if eff_images_tar_gz is None and (images_path is not None or image_paths):
        discovered = await asyncio.to_thread(discover_image_files, images_path, image_paths)
        eff_images_tar_gz = await asyncio.to_thread(
            pack_images_tar_gz,
            discovered,
            base_cfg.extract if base_cfg is not None else None,
        )

    eff_known_json = known_cameras_json
    if eff_known_json is None and known_cameras_path is not None and known_cameras_path.is_file():
        eff_known_json = known_cameras_path.read_text(encoding="utf-8")

    masks_b64: dict[str, str] | None = None
    if masks_paths:
        masks_b64 = {k: base64.b64encode(p.read_bytes()).decode("ascii") for k, p in masks_paths.items()}

    q: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _producer() -> None:
        try:
            worker_cls = _bind_modal_worker_cls(
                modal.Cls.from_name("blenderbot-splat", cls_name),
                eff_timeout,
            )
            worker = worker_cls()
            call_kwargs: dict[str, Any] = {
                "job_id": job_id,
                "mp4_bytes": mp4_bytes,
                "views_tar_gz": views_tar_gz,
                "known_cameras_json": eff_known_json,
                "cfg_dict": send_cfg_dict,
                "cfg_overrides": send_overrides,
                "add_steps": add_steps,
                "surgery_dict": surgery.to_dict() if surgery else None,
                "masks_b64": masks_b64,
                "exclude_frames": exclude_frames,
            }
            if eff_images_tar_gz is not None:
                call_kwargs["images_tar_gz"] = eff_images_tar_gz
            gen = worker.run_stream.remote_gen(**call_kwargs)
            for item in gen:
                q.put(("ev", item))
            q.put(("eof", None))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

    t_client_start = time.perf_counter()
    thr = threading.Thread(target=_producer, daemon=True)
    thr.start()

    done_ev: dict[str, Any] | None = None
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue

        if kind == "err":
            err_msg = str(payload)
            if isinstance(payload, TypeError) and "images_tar_gz" in err_msg:
                raise RuntimeError(
                    "Deployed Modal worker predates `images_tar_gz` support. Redeploy once via: "
                    "`modal deploy src/blenderbot/splats/modal_app.py`"
                ) from payload
            if "hit its timeout" in err_msg:
                err_msg = (
                    f"{err_msg} (MODAL_WORKER_TIMEOUT={eff_timeout}s; set MODAL_WORKER_TIMEOUT "
                    "in ~/.blenderbot/.env or pass timeout_sec to raise)"
                )
            raise RuntimeError(f"Modal splat worker failed: {err_msg}") from payload
        if kind == "eof":
            break
        if kind == "ev":
            _persist_local_telemetry(output_dir, payload)
            if on_event is not None:
                on_event(payload)
            if payload.get("event") == "done":
                done_ev = payload

    if done_ev is None:
        raise RuntimeError("Modal splat stream ended without a 'done' artifact event.")

    ply_path, splat_path, proxy_path, cameras_path = _write_job_outputs(output_dir, done_ev)
    total_client_wall = round(time.perf_counter() - t_client_start, 3)

    return SplatJobResult(
        job_id=job_id,
        burst_id=int(done_ev["burst_id"]),
        step=int(done_ev["step"]),
        n_gaussians=int(done_ev["n_gaussians"]),
        output_dir=output_dir,
        ply_path=ply_path,
        splat_path=splat_path,
        proxy_glb_path=proxy_path,
        cameras_path=cameras_path,
        telemetry_path=output_dir / "telemetry.jsonl",
        device=str(done_ev.get("device", eff_gpu)),
        wall_s=total_client_wall,
        cost_usd=float(done_ev.get("cost_usd", 0.0)),
        macro_timings=dict(done_ev.get("macro_timings") or {}),
    )


async def run_modal_splat_render(
    *,
    job_id: str,
    output_mp4: Path,
    trajectory_doc: dict[str, Any],
    ply_fallback_path: Path | None = None,
    gpu: str = "H100",
    rasterize_mode: str = "antialiased",
    bg_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    timeout_sec: float | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> SplatRenderResult:
    """Farm full-rate 3DGS camera trajectory MP4 rendering to Modal H100/L40S (`gsplat` CUDA)."""
    import modal

    eff_gpu = gpu.upper()
    cls_name = "SplatWorkerL40S" if eff_gpu == "L40S" else "SplatWorker"
    eff_timeout = _resolve_client_timeout(timeout_sec, default=DEFAULT_MODAL_WORKER_TIMEOUT)
    ply_gz = (
        gzip.compress(ply_fallback_path.read_bytes(), compresslevel=1)
        if ply_fallback_path is not None and ply_fallback_path.is_file()
        else None
    )

    q: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _producer() -> None:
        try:
            worker_cls = _bind_modal_worker_cls(
                modal.Cls.from_name("blenderbot-splat", cls_name),
                eff_timeout,
            )
            worker = worker_cls()
            gen = worker.render_trajectory_stream.remote_gen(
                job_id=job_id,
                trajectory_doc=trajectory_doc,
                ply_gz=ply_gz,
                gpu=eff_gpu,
                rasterize_mode=rasterize_mode,
                bg_rgb=bg_rgb,
            )
            for item in gen:
                q.put(("ev", item))
            q.put(("eof", None))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

    t_start = time.perf_counter()
    thr = threading.Thread(target=_producer, daemon=True)
    thr.start()

    done_ev: dict[str, Any] | None = None
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.04)
            continue

        if kind == "err":
            raise RuntimeError(f"Modal trajectory render failed: {payload}") from payload
        if kind == "eof":
            break
        if kind == "ev":
            if on_event is not None:
                on_event(payload)
            if payload.get("event") == "render_done":
                done_ev = payload

    if done_ev is None or not done_ev.get("mp4_bytes"):
        raise RuntimeError("Modal trajectory render ended without MP4 bytes.")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    output_mp4.write_bytes(done_ev["mp4_bytes"])
    wall_s = round(time.perf_counter() - t_start, 3)

    return SplatRenderResult(
        job_id=job_id,
        output_mp4=output_mp4,
        frames=int(done_ev["frames"]),
        fps=float(done_ev["fps"]),
        width=int(done_ev["width"]),
        height=int(done_ev["height"]),
        device=str(done_ev.get("device", eff_gpu)),
        wall_s=wall_s,
        render_fps=float(done_ev.get("render_fps", 0.0)),
        cost_usd=float(done_ev.get("cost_usd", 0.0)),
    )