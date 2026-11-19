"""Modal Serverless H100 / L40S 3DGS Worker (`modal deploy src/blenderbot/splats/modal_app.py`).

Compiles `gsplat.csrc` directly into `site-packages` (`--no-binary gsplat` on 32 vCPUs, so runtime
never invokes `cpp_extension.load` / `ninja`), pre-caches `facebook/VGGT-1B` in `blenderbot-splat-models`,
stores resumable burst checkpoints on `blenderbot-splat-jobs`, polls live mid-train controls via
`blenderbot-splat-ctl`, and farms full-rate trajectory MP4 renders (`render_trajectory_stream`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import modal

APP_NAME = "blenderbot-splat"
MODELS_VOL_NAME = "blenderbot-splat-models"
JOBS_VOL_NAME = "blenderbot-splat-jobs"
CTL_DICT_NAME = "blenderbot-splat-ctl"

GPU_HOURLY_USD: dict[str, float] = {
    "H100": 3.95,
    "L40S": 1.95,
    "A100": 2.50,
}

models_vol = modal.Volume.from_name(MODELS_VOL_NAME, create_if_missing=True)
jobs_vol = modal.Volume.from_name(JOBS_VOL_NAME, create_if_missing=True)
ctl_dict = modal.Dict.from_name(CTL_DICT_NAME, create_if_missing=True)


def _bake_csrc() -> None:
    """32-vCPU build step: compile `gsplat.csrc` directly into site-packages (no runtime JIT)."""
    env = os.environ.copy()
    env["CC"] = "gcc"
    env["CXX"] = "g++"
    env["MAX_JOBS"] = "32"
    env["TORCH_CUDA_ARCH_LIST"] = "8.9;9.0"
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["FORCE_CUDA"] = "1"
    subprocess.run(
        [
            "git",
            "clone",
            "--recursive",
            "--depth",
            "1",
            "--branch",
            "v1.4.0",
            "https://github.com/nerfstudio-project/gsplat.git",
            "/tmp/gsplat_src",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-build-isolation",
            "--no-deps",
            "/tmp/gsplat_src",
        ],
        env=env,
        check=True,
    )
    import shutil

    shutil.rmtree("/tmp/gsplat_src", ignore_errors=True)
    from gsplat import csrc
    from gsplat.cuda._backend import _C
    from vggt.models.vggt import VGGT  # noqa: F401

    assert _C is csrc, f"Expected static gsplat.csrc, got {_C}"
    print("STATIC_CSRC_OK:", csrc.__file__)


splat_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        "build-essential",
        "ninja-build",
        "colmap",
        "libgl1",
        "libglib2.0-0",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.9;9.0",
        "MAX_JOBS": "16",
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HOME": "/models/hf",
        "TORCH_HOME": "/models/torch",
    })
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",
        "pillow>=10.2.0",
        "scipy>=1.12.0",
        "einops>=0.7.0",
        "timm>=1.0.0",
        "huggingface_hub>=0.24.0",
        "safetensors>=0.4.3",
        "opencv-python-headless>=4.9.0",
        "jaxtyping>=0.2.28",
        "rich>=13.0.0",
        "ninja>=1.11.0",
        "setuptools>=68.0",
        "wheel",
    )
    .run_commands(
        "pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/vggt.git",
    )
    .run_function(_bake_csrc, cpu=32.0)
    .add_local_python_source("blenderbot")
)

app = modal.App(APP_NAME)


class _BaseSplatWorker:
    """Shared stateful 3DGS worker implementation (90s HBM warm cache + persistent NVMe `/jobs` Volume)."""

    @modal.enter()
    def _setup(self) -> None:
        self._ram_cache: dict[str, Any] = {}

    @modal.method()
    def run_stream(
        self,
        *,
        job_id: str,
        mp4_bytes: bytes | None = None,
        views_tar_gz: bytes | None = None,
        known_cameras_json: str | None = None,
        cfg_dict: dict[str, Any] | None = None,
        cfg_overrides: dict[str, Any] | None = None,
        add_steps: int | None = None,
        surgery_dict: dict[str, Any] | None = None,
        masks_b64: dict[str, str] | None = None,
        exclude_frames: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        from blenderbot.splats.config import SplatConfig, SurgerySpec
        from blenderbot.splats.core_worker import run_splat_job_stream

        jobs_vol.reload()
        # Clear any stale live-control entry from a prior run on this job_id
        try:
            ctl_dict.pop(job_id, None)
        except Exception:  # noqa: BLE001, S110
            pass

        def _poll_ctl() -> dict[str, Any] | None:
            try:
                val = ctl_dict.get(job_id)
                return val if isinstance(val, dict) else None
            except Exception:  # noqa: BLE001
                return None

        cfg = SplatConfig.from_dict(cfg_dict) if cfg_dict else None
        surgery = SurgerySpec.from_dict(surgery_dict) if surgery_dict else None
        gpu_name = str(
            (cfg_overrides or {}).get("gpu")
            or (cfg.gpu if cfg else "H100")
        ).upper()
        rate = GPU_HOURLY_USD.get(gpu_name, 3.95)

        for ev in run_splat_job_stream(
            job_id=job_id,
            jobs_root=Path("/jobs"),
            models_dir=Path("/models"),
            mp4_bytes=mp4_bytes,
            views_tar_gz=views_tar_gz,
            known_cameras_json=known_cameras_json,
            cfg=cfg,
            cfg_overrides=cfg_overrides,
            add_steps=add_steps,
            surgery=surgery,
            masks_b64=masks_b64,
            exclude_frames=exclude_frames,
            control_poll_fn=_poll_ctl,
            gpu_hourly_usd=rate,
            ram_cache=self._ram_cache,
        ):
            if ev.get("event") in ("done", "checkpoint_saved"):
                jobs_vol.commit()
            yield ev

    @modal.method()
    def render_trajectory_stream(
        self,
        *,
        job_id: str,
        trajectory_doc: dict[str, Any],
        ply_gz: bytes | None = None,
        gpu: str = "H100",
        rasterize_mode: str = "antialiased",
        bg_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Iterator[dict[str, Any]]:
        from blenderbot.splats.core_worker import render_splat_trajectory_stream

        jobs_vol.reload()
        rate = GPU_HOURLY_USD.get(gpu.upper(), 3.95)
        yield from render_splat_trajectory_stream(
            job_id=job_id,
            jobs_root=Path("/jobs"),
            trajectory_doc=trajectory_doc,
            ply_gz=ply_gz,
            rasterize_mode=rasterize_mode,
            bg_rgb=bg_rgb,
            gpu_hourly_usd=rate,
            ram_cache=self._ram_cache,
        )


@app.cls(
    image=splat_image,
    gpu="H100",
    cpu=8.0,
    volumes={"/models": models_vol, "/jobs": jobs_vol},
    scaledown_window=90,
    timeout=900,
)
class SplatWorker(_BaseSplatWorker):
    """Primary H100 (80 GB HBM3) 3DGS worker."""


@app.cls(
    image=splat_image,
    gpu="L40S",
    cpu=8.0,
    volumes={"/models": models_vol, "/jobs": jobs_vol},
    scaledown_window=90,
    timeout=900,
)
class SplatWorkerL40S(_BaseSplatWorker):
    """Cost-optimized L40S (48 GB Ada, $1.95/hr) peer worker."""