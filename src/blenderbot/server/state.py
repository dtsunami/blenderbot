"""Shared mutable state for the Web Studio & Seedance 2.5 GUI Sidecar."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from blenderbot.session.bootstrap import Session

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".mkv"})
AUDIO_EXTS = frozenset({".mp3", ".wav", ".m4a", ".ogg"})
SKIP_SCAN_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "davai",
        "davag",
        "splats",
        "tt-splat",
        "arcgs",
        "p2_gate",
        "p3_gate",
    }
)


@dataclass
class SeedanceJobRecord:
    local_id: str
    prompt: str
    model: str
    mode: str
    output_rel: str
    created_at: float
    stage: str = "queued"
    remote_job_id: str | None = None
    elapsed_sec: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    media_url: str | None = None
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None
    conditioning: dict[str, Any] = field(default_factory=dict)
    cancel_ev: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    task: asyncio.Task[Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_id": self.local_id,
            "remote_job_id": self.remote_job_id,
            "prompt": self.prompt,
            "model": self.model,
            "mode": self.mode,
            "output_rel": self.output_rel,
            "created_at": self.created_at,
            "stage": self.stage,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "cost_usd": self.cost_usd,
            "error": self.error,
            "media_url": self.media_url,
            "width": self.width,
            "height": self.height,
            "duration_sec": self.duration_sec,
            "conditioning": self.conditioning,
        }


def resolve_source_video_for_job(cwd: Path, job_id: str, source_label: str = "") -> Path | None:
    """Locate the original source video corresponding to `job_id` (`job_source.json` or stem match)."""
    jdir = cwd / "assets" / "splats" / job_id
    src_meta = jdir / "job_source.json"
    if src_meta.is_file():
        try:
            meta = json.loads(src_meta.read_text(encoding="utf-8"))
            rel_v = str(meta.get("video_rel") or "")
            if rel_v:
                cand = (cwd / rel_v).resolve()
                if cand.is_file():
                    return cand
        except Exception:  # noqa: BLE001, S110
            pass

    if source_label and not source_label.startswith(("Blender", "Resume")):
        cand_lbl = (cwd / source_label).resolve()
        if cand_lbl.is_file():
            return cand_lbl

    # `make_job_id` builds `{video_stem}_{HHMMSS}_{6hex}`: strip trailing `_HHMMSS_hex6`
    parts = job_id.split("_")
    if len(parts) >= 3 and len(parts[-1]) == 6 and len(parts[-2]) == 6 and parts[-2].isdigit():
        stem = "_".join(parts[:-2])
        for folder in (cwd / "assets" / "seedance", cwd / "assets" / "inputs", cwd / "renders", cwd):
            if not folder.is_dir():
                continue
            for ext in (".mp4", ".mov", ".webm", ".mkv", ".MP4", ".MOV"):
                cand = folder / f"{stem}{ext}"
                if cand.is_file():
                    return cand
    return None


@dataclass
class SplatJobRecord:
    job_id: str
    source_mode: str
    source_label: str
    gpu: str
    iterations: int
    created_at: float
    stage: str = "queued"
    step: int = 0
    total_steps: int = 0
    psnr: float = 0.0
    loss: float = 0.0
    l1: float = 0.0
    ssim: float = 0.0
    it_per_sec: float = 0.0
    n_gaussians: int = 0
    elapsed_sec: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    ply_rel: str | None = None
    splat_url: str | None = None
    proxy_glb_rel: str | None = None
    proxy_glb_url: str | None = None
    preview_url: str | None = None
    diff_url: str | None = None
    gt_url: str | None = None
    cameras_rel: str | None = None
    cameras_url: str | None = None
    source_video_rel: str | None = None
    source_video_url: str | None = None
    modal_render_url: str | None = None
    blender_render_url: str | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    imported_object: str | None = None
    task: asyncio.Task[Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_mode": self.source_mode,
            "source_label": self.source_label,
            "gpu": self.gpu,
            "iterations": self.iterations,
            "created_at": self.created_at,
            "stage": self.stage,
            "step": self.step,
            "total_steps": self.total_steps or self.iterations,
            "psnr": round(self.psnr, 2),
            "loss": round(self.loss, 4),
            "l1": round(self.l1, 4),
            "ssim": round(self.ssim, 4),
            "it_per_sec": round(self.it_per_sec, 1),
            "n_gaussians": self.n_gaussians,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "cost_usd": round(self.cost_usd, 5),
            "error": self.error,
            "ply_rel": self.ply_rel,
            "splat_url": self.splat_url,
            "proxy_glb_rel": self.proxy_glb_rel,
            "proxy_glb_url": self.proxy_glb_url,
            "preview_url": self.preview_url,
            "diff_url": self.diff_url,
            "gt_url": self.gt_url,
            "cameras_rel": self.cameras_rel,
            "cameras_url": self.cameras_url,
            "source_video_rel": self.source_video_rel,
            "source_video_url": self.source_video_url,
            "modal_render_url": self.modal_render_url,
            "blender_render_url": self.blender_render_url,
            "stage_timings": self.stage_timings,
            "imported_object": self.imported_object,
        }


@dataclass
class WebState:
    session: Session
    submit_fn: Callable[[str, str], Any]
    secret: str
    lan_auth_url: str | None = None
    shutdown_ev: asyncio.Event = field(default_factory=asyncio.Event)
    sse_queues: set[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=set)
    jobs: dict[str, SeedanceJobRecord] = field(default_factory=dict)
    job_order: list[str] = field(default_factory=list)
    splat_jobs: dict[str, SplatJobRecord] = field(default_factory=dict)
    splat_job_order: list[str] = field(default_factory=list)

    def media_url_for(self, path: Path) -> str:
        rel = self.session.sandbox.rel(path).replace("\\", "/")
        return f"/api/media/file?path={quote(rel, safe='/')}&v={int(path.stat().st_mtime)}"

    def new_job(self, *, prompt: str, model: str, mode: str, output_rel: str) -> SeedanceJobRecord:
        jid = f"sd-{int(time.time()) % 100000}-{secrets.token_hex(2)}"
        rec = SeedanceJobRecord(
            local_id=jid,
            prompt=prompt,
            model=model,
            mode=mode,
            output_rel=output_rel,
            created_at=time.time(),
        )
        self.jobs[jid] = rec
        self.job_order.insert(0, jid)
        return rec

    def new_splat_job(
        self,
        *,
        job_id: str,
        source_mode: str,
        source_label: str,
        gpu: str,
        iterations: int,
    ) -> SplatJobRecord:
        rec = SplatJobRecord(
            job_id=job_id,
            source_mode=source_mode,
            source_label=source_label,
            gpu=gpu,
            iterations=iterations,
            total_steps=iterations,
            created_at=time.time(),
        )
        src_v = resolve_source_video_for_job(self.session.cfg.cwd, job_id, source_label)
        if src_v is not None:
            rec.source_video_rel = self.session.sandbox.rel(src_v).replace("\\", "/")
            rec.source_video_url = self.media_url_for(src_v)
        self.splat_jobs[job_id] = rec
        if job_id not in self.splat_job_order:
            self.splat_job_order.insert(0, job_id)
        return rec

    def scan_splat_jobs(self, max_items: int = 40) -> list[dict[str, Any]]:
        """Merge active in-memory splat jobs with persisted `assets/splats/*` artifacts on disk."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        cwd = self.session.cfg.cwd

        for jid in self.splat_job_order:
            rec = self.splat_jobs.get(jid)
            if rec is not None:
                d_live = cwd / "assets" / "splats" / jid
                m_vid = d_live / "renders" / "modal_recreated.mp4"
                b_vid = d_live / "renders" / "blender_recreated.mp4"
                p_glb = d_live / "proxy_collision.glb"
                if m_vid.is_file():
                    rec.modal_render_url = self.media_url_for(m_vid)
                if b_vid.is_file():
                    rec.blender_render_url = self.media_url_for(b_vid)
                if p_glb.is_file():
                    rec.proxy_glb_rel = self.session.sandbox.rel(p_glb).replace("\\", "/")
                    rec.proxy_glb_url = self.media_url_for(p_glb)
                out.append(rec.to_dict())
                seen.add(jid)

        splats_root = cwd / "assets" / "splats"
        if not splats_root.is_dir():
            return out[:max_items]

        disk_dirs = sorted(
            (d for d in splats_root.iterdir() if d.is_dir() and d.name not in seen),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in disk_dirs[:max_items]:
            ply_p = d / "splat.ply"
            if not ply_p.is_file():
                continue
            splat_p = d / "splat.splat"
            proxy_p = d / "proxy_collision.glb"
            cam_p = d / "cameras.json"
            prev_p = d / "previews" / "latest_render.jpg"
            diff_p = d / "previews" / "latest_diff.jpg"
            gt_files = sorted((d / "previews").glob("gt_*.jpg")) if (d / "previews").is_dir() else []
            gt_p = gt_files[0] if gt_files else None
            modal_mp4 = d / "renders" / "modal_recreated.mp4"
            blender_mp4 = d / "renders" / "blender_recreated.mp4"
            src_vid = resolve_source_video_for_job(cwd, d.name)
            telem_p = d / "telemetry.jsonl"

            step = 0
            n_gauss = 0
            psnr = 0.0
            loss_v = 0.0
            l1_v = 0.0
            ssim_v = 0.0
            cost_usd = 0.0
            wall_s = 0.0
            gpu_lbl = "H100"
            src_mode = "blender" if d.name.startswith("blender_") else "video"
            st_timings: dict[str, float] = {}

            if telem_p.is_file():
                try:
                    lines = telem_p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                    for ln in reversed(lines[-25:]):
                        row = json.loads(ln)
                        if row.get("event") == "done" and step == 0:
                            step = int(row.get("step") or 0)
                            n_gauss = int(row.get("n_gaussians") or 0)
                            cost_usd = float(row.get("cost_usd") or 0.0)
                            wall_s = float(row.get("wall_s") or 0.0)
                            dev = str(row.get("device") or "")
                            gpu_lbl = "L40S" if "L40S" in dev else "H100"
                        elif row.get("event") == "telemetry" and psnr == 0.0:
                            psnr = float(row.get("psnr") or 0.0)
                            loss_v = float(row.get("loss") or 0.0)
                            l1_v = float(row.get("l1") or 0.0)
                            ssim_v = float(row.get("ssim") or 0.0)
                            st_timings = dict(row.get("stage_timings") or {})
                            if step == 0:
                                step = int(row.get("step") or 0)
                            if n_gauss == 0:
                                n_gauss = int(row.get("n_gaussians") or 0)
                except Exception:  # noqa: BLE001, S110
                    pass

            out.append(
                {
                    "job_id": d.name,
                    "source_mode": src_mode,
                    "source_label": self.session.sandbox.rel(src_vid).replace("\\", "/") if src_vid else d.name,
                    "gpu": gpu_lbl,
                    "iterations": step,
                    "created_at": ply_p.stat().st_mtime,
                    "stage": "completed",
                    "step": step,
                    "total_steps": step,
                    "psnr": round(psnr, 2),
                    "loss": round(loss_v, 4),
                    "l1": round(l1_v, 4),
                    "ssim": round(ssim_v, 4),
                    "it_per_sec": 0.0,
                    "n_gaussians": n_gauss,
                    "elapsed_sec": round(wall_s, 1),
                    "cost_usd": round(cost_usd, 5),
                    "error": None,
                    "ply_rel": self.session.sandbox.rel(ply_p).replace("\\", "/"),
                    "splat_url": self.media_url_for(splat_p) if splat_p.is_file() else None,
                    "proxy_glb_rel": self.session.sandbox.rel(proxy_p).replace("\\", "/") if proxy_p.is_file() else None,
                    "proxy_glb_url": self.media_url_for(proxy_p) if proxy_p.is_file() else None,
                    "preview_url": self.media_url_for(prev_p) if prev_p.is_file() else None,
                    "diff_url": self.media_url_for(diff_p) if diff_p.is_file() else None,
                    "gt_url": self.media_url_for(gt_p) if gt_p is not None else None,
                    "cameras_url": self.media_url_for(cam_p) if cam_p.is_file() else None,
                    "cameras_rel": self.session.sandbox.rel(cam_p).replace("\\", "/") if cam_p.is_file() else None,
                    "source_video_rel": self.session.sandbox.rel(src_vid).replace("\\", "/") if src_vid else None,
                    "source_video_url": self.media_url_for(src_vid) if src_vid else None,
                    "modal_render_url": self.media_url_for(modal_mp4) if modal_mp4.is_file() else None,
                    "blender_render_url": self.media_url_for(blender_mp4) if blender_mp4.is_file() else None,
                    "stage_timings": st_timings,
                    "imported_object": None,
                }
            )
        return out[:max_items]

    def scan_workspace_media(self, max_items: int = 160) -> list[dict[str, Any]]:
        cwd = self.session.cfg.cwd
        found: list[tuple[float, Path, str]] = []

        def _walk(folder: Path, depth: int) -> None:
            if depth > 4:
                return
            try:
                entries = list(folder.iterdir())
            except OSError:
                return
            for entry in entries:
                name = entry.name
                if name.startswith(".") and name not in (".",):
                    continue
                if entry.is_dir():
                    if name in SKIP_SCAN_DIRS:
                        continue
                    _walk(entry, depth + 1)
                elif entry.is_file():
                    ext = entry.suffix.lower()
                    kind = (
                        "image"
                        if ext in IMAGE_EXTS
                        else "video"
                        if ext in VIDEO_EXTS
                        else "audio"
                        if ext in AUDIO_EXTS
                        else None
                    )
                    if kind:
                        try:
                            st = entry.stat()
                            if st.st_size > 0:
                                found.append((st.st_mtime, entry, kind))
                        except OSError:
                            continue

        _walk(cwd, 0)
        found.sort(key=lambda t: t[0], reverse=True)

        cost_by_rel: dict[str, float] = {
            r.output_rel: r.cost_usd
            for r in self.jobs.values()
            if r.cost_usd > 0 and r.output_rel
        }
        items: list[dict[str, Any]] = []
        for mtime, p, kind in found[:max_items]:
            rel = self.session.sandbox.rel(p).replace("\\", "/")
            items.append(
                {
                    "path": rel,
                    "name": p.name,
                    "kind": kind,
                    "ext": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                    "mtime": int(mtime),
                    "is_seedance": "seedance" in rel.lower(),
                    "cost_usd": cost_by_rel.get(rel),
                    "url": self.media_url_for(p),
                }
            )
        return items