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
BLENDER_CACHE_VOL_NAME = "blenderbot-blender-cache"
CTL_DICT_NAME = "blenderbot-splat-ctl"

GPU_HOURLY_USD: dict[str, float] = {
    "H100": 3.95,
    "L40S": 1.95,
    "A100": 2.50,
}

models_vol = modal.Volume.from_name(MODELS_VOL_NAME, create_if_missing=True)
jobs_vol = modal.Volume.from_name(JOBS_VOL_NAME, create_if_missing=True)
blender_cache_vol = modal.Volume.from_name(BLENDER_CACHE_VOL_NAME, create_if_missing=True)
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


def _install_blender_linux() -> None:
    """Download & verify official Blender 5.3 Linux x64 release into `/opt/blender`."""
    import re
    import shutil
    import urllib.request

    candidate_urls = [
        "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz",
        "https://mirrors.ocf.berkeley.edu/blender/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz",
        "https://mirrors.dotsrc.org/blender/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz",
    ]
    # Also dynamically check release directory index in case a newer 5.3.x patch exists
    for idx_url in (
        "https://download.blender.org/release/Blender5.3/",
        "https://download.blender.org/release/Blender5.2/",
        "https://download.blender.org/release/Blender5.1/",
    ):
        try:
            req = urllib.request.Request(idx_url, headers={"User-Agent": "blenderbot-modal/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            matches = sorted(
                set(re.findall(r'blender-\d+\.\d+\.\d+-linux-x64\.tar\.xz', html)),
                reverse=True,
            )
            for m in matches:
                u = idx_url + m
                if u not in candidate_urls:
                    candidate_urls.append(u)
        except Exception:  # noqa: BLE001, S112
            continue

    tar_dst = Path("/tmp/blender-linux-x64.tar.xz")
    downloaded_from = None
    for url in candidate_urls:
        try:
            print(f"Trying Blender Linux x64 tarball: {url}")
            res = subprocess.run(
                ["curl", "-fSL", "--retry", "3", "--connect-timeout", "15", "-o", str(tar_dst), url],
                check=False,
            )
            if res.returncode == 0 and tar_dst.is_file() and tar_dst.stat().st_size > 100_000_000:
                downloaded_from = url
                break
        except Exception:  # noqa: BLE001, S112
            continue

    if not downloaded_from:
        raise RuntimeError(f"Could not download Blender 5.x Linux x64 from any mirror: {candidate_urls[:4]}")

    opt_bl = Path("/opt/blender")
    shutil.rmtree(opt_bl, ignore_errors=True)
    opt_bl.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-xJf", str(tar_dst), "-C", str(opt_bl), "--strip-components=1"],
        check=True,
    )
    tar_dst.unlink(missing_ok=True)

    usr_bin = Path("/usr/local/bin/blender")
    usr_bin.unlink(missing_ok=True)
    usr_bin.symlink_to(opt_bl / "blender")

    ver_out = subprocess.run(
        [str(opt_bl / "blender"), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    print(f"BLENDER_LINUX_OK ({downloaded_from}):\n{ver_out.stdout.splitlines()[0]}")


blender_render_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11")
    .env({
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "OPTIX_CACHE_PATH": "/blender_cache/optix",
        "CUDA_CACHE_PATH": "/blender_cache/cuda",
    })
    .apt_install(
        "curl",
        "xz-utils",
        "ca-certificates",
        "ffmpeg",
        "libx11-6",
        "libxi6",
        "libxxf86vm1",
        "libxfixes3",
        "libxrender1",
        "libxkbcommon0",
        "libxkbcommon-x11-0",
        "libsm6",
        "libice6",
        "libgl1",
        "libegl1",
        "libopengl0",
        "libglvnd0",
        "libgomp1",
        "libdbus-1-3",
        "fontconfig",
    )
    .run_function(_install_blender_linux, cpu=8.0)
    .add_local_python_source("blenderbot")
)


class _BaseBlenderRenderWorker:
    """Headless Blender 5.3 GPU renderer (`CYCLES` OptiX/CUDA + `BLENDER_EEVEE` EGL)."""

    @modal.method()
    def render_stream(
        self,
        *,
        blend_gz: bytes,
        render_spec: dict[str, Any],
        gpu: str = "H100",
    ) -> Iterator[dict[str, Any]]:
        from blenderbot.splats.blender_render import run_blender_modal_render_stream

        blender_cache_vol.reload()
        rate = GPU_HOURLY_USD.get(gpu.upper(), 3.95)
        for ev in run_blender_modal_render_stream(
            blend_gz=blend_gz,
            render_spec=render_spec,
            gpu_hourly_usd=rate,
        ):
            if ev.get("event") == "render_done" and ev.get("backend") == "OPTIX":
                try:
                    blender_cache_vol.commit()
                except Exception:  # noqa: BLE001, S110
                    pass
            yield ev


@app.cls(
    image=blender_render_image,
    gpu="H100",
    cpu=8.0,
    volumes={"/blender_cache": blender_cache_vol},
    scaledown_window=90,
    timeout=1800,
)
class BlenderRenderWorker(_BaseBlenderRenderWorker):
    """Primary H100 (80 GB HBM3) Blender 5.3 Cycles OptiX / EEVEE cloud renderer."""


@app.cls(
    image=blender_render_image,
    gpu="L40S",
    cpu=8.0,
    volumes={"/blender_cache": blender_cache_vol},
    scaledown_window=90,
    timeout=1800,
)
class BlenderRenderWorkerL40S(_BaseBlenderRenderWorker):
    """Cost-optimized L40S (48 GB Ada RT-Core, $1.95/hr) Blender 5.3 cloud renderer."""