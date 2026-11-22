"""Dataclass configurations for the cloud 3DGS pipeline (`arcgs` style, pure Python stdlib)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExtractConfig:
    fps: float = 8.0
    max_frames: int = 96
    max_width: int = 1280
    dedup_frames: bool = True
    quality: int = 95


@dataclass
class PoseConfig:
    backend: str = "vggt"          # "vggt" | "colmap" | "colmap_prior" (Blender K + Sim3 nudge) | "known_cameras" (hard lock)
    matcher: str = "auto"          # "auto" (exhaustive for <=80 photos, sequential for video) | "exhaustive" | "sequential"
    vggt_max_frames: int = 64      # VGGT attention batch; remaining frames registered via iterative video-diff arc cycles
    conf_percentile: float = 35.0  # drop lowest N% confidence VGGT points on init
    max_init_points: int = 180_000
    lock_pinhole_intrinsics: bool = True  # Use distortion-free shared PINHOLE model (`fx = fy`)
    align_floor_z0: bool = True
    target_cam_height_m: float = 1.60
    # Apple Depth Pro 2 (`ml-depth-pro-2`) temporally-consistent metric depth prior & point nudging
    depth_prior_model: str = "auto"       # "auto" (use depthpro2-b if installed) | "depthpro2-s" | "depthpro2-b" | "depthpro2-l" | "depthpro2-h" | "off"
    depth_nudge_strength: float = 0.04    # Gentle expected-depth free-space floater push (`0` = off)


@dataclass
class TrainConfig:
    iterations: int = 5_000
    sh_degree: int = 1             # SH1 avoids fitting AI-video temporal flicker & shrinks PLY 2.7x
    rasterize_mode: str = "antialiased"  # "antialiased" (Mip-Splatting 2D EWA filter) | "classic"
    refine_poses: bool = False     # Differentiable se(3) 6D+3D sub-pixel camera pose optimization inside gsplat
    pose_opt_lr: float = 5e-5
    pose_prior_reg: float = 1e-2   # Soft L2 tether keeping refined poses anchored to Blender/VGGT/COLMAP frame
    lambda_dssim: float = 0.2
    opa_reg: float = 5e-4          # L1 opacity penalty to evaporate mid-air AI-video floaters
    scale_reg: float = 1e-3        # Penalizes huge billboard/needle splats
    aniso_reg: float = 1e-2        # Penalizes extreme needle aspect ratios (s_max / s_mid > max_anisotropy)
    max_anisotropy: float = 6.0
    densify_from: int = 400
    densify_until: int = 3_600     # Auto-capped to 72% of iterations so final 28% polishes & prunes
    densify_every: int = 100
    densify_grad_threshold: float = 0.0002
    prune_opa: float = 0.02
    prune_scale3d: float = 0.05    # Relative to scene_scale (~12cm max splat in a 2.5m scene)
    opacity_reset_every: int = 2_000
    adaptive_strategy: str = "adaptive_arc"  # "adaptive_arc" | "fixed"
    arc_scale_mode: str = "log"    # "log" | "power" | "linear"
    arc_gamma: float = 1.6         # exponent when arc_scale_mode == "power"
    target_gaussians: int = 350_000  # soft budget controller (`0` = uncapped)
    prune_opa_start: float | None = None  # optional arc start override (auto = min(0.008, prune_opa * 0.35))
    grow_grad2d_end: float | None = None  # optional arc end override
    # Multi-view anti-floater consistency, swept-arc frustum carving & iterative arc registration
    views_per_step: int = 2        # Stereo cross-view mini-batch (`1` = single view, `2..4` = stereo anti-floater)
    appearance_opt: bool = True    # Per-camera 3x4 affine exposure/white-balance absorption (`A_i * C + b_i`)
    appearance_reg: float = 1e-3   # L2 tether keeping mean color transform anchored to Identity
    cam_corridor_radius_m: float = 0.35  # Evicts cheat billboards inside the continuous swept camera arc (`0` = off)
    near_frustum_clear_m: float = 0.40   # Evicts near-lens floaters inside `0 < z_cam < near_frustum_clear_m` across the arc
    opa_bimodal_reg: float = 2e-4        # Binary-entropy `α(1-α)` penalty evaporating semi-transparent room fog
    outlier_view_downweight: bool = True  # IRLS Cauchy downweighting of blurry/HUD/glare outlier frames
    opacity_logit_cap: float = 5.5  # Caps max opacity at `sigmoid(5.5) = 0.9959` so mistakes never freeze in `σ(1-σ)->0`
    # Iterative Cycle Mode: SfM -> Train -> Video-Diff Arc Solve + Swept Clear + Active Frame Promotion -> Train...
    iterative_arc_cycles: int = 1        # `1` = single pass; `2..5` = alternating Train <-> Arc Solve <-> Frame Select cycles
    arc_solve_steps: int = 24            # Photometric video-diff `se(3)` trajectory registration steps per cycle
    frames_add_per_cycle: int = 24       # Highest-error / gap candidate video frames promoted into active set per cycle
    log_every: int = 25            # scalar + CUDA Event stage timing interval
    preview_hz: float = 4.0        # wall-clock cap for live JPEG triptych pushes
    snapshot_every: int = 1_000    # save timeline scrubber snapshot every N steps (0 = off)
    focus_camera: str | None = None


@dataclass
class ExportConfig:
    export_ply: bool = True
    export_splat: bool = True      # 32-byte antimatter15 .splat for WebGL Studio preview
    export_proxy_glb: bool = True  # Coarse Surface-Nets 3D proxy mesh (`proxy_collision.glb`)
    proxy_grid_res: int = 48
    sort_by_opacity: bool = True
    max_gaussians: int = 0         # 0 = keep all trained Gaussians in cloud export
    compress_gzip: bool = True


@dataclass
class SurgerySpec:
    """Ordered tensor surgery applied before a `splat.resume` training burst."""

    prune_opa: float | None = None
    max_world_scale: float | None = None
    clamp_log_scale: float | None = None
    keep_aabb: tuple[list[float], list[float]] | None = None  # ([xmin,ymin,zmin], [xmax,ymax,zmax])
    drop_aabb: tuple[list[float], list[float]] | None = None
    keep_radius: float | None = None
    reset_opacities: float | None = None
    densify_now: bool = False
    lr_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v is not False and v != 1.0}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurgerySpec:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for box_key in ("keep_aabb", "drop_aabb"):
            if valid.get(box_key) is not None:
                lo, hi = valid[box_key]
                valid[box_key] = ([float(x) for x in lo], [float(x) for x in hi])
        return cls(**valid)


# Flat-key alias table so top-level agent/API params (`opa_reg`, `lambda_dssim`, `pose_backend`, etc.)
# automatically route into the right sub-config without requiring nested dicts.
_FLAT_KEY_ALIASES: dict[str, tuple[str, str]] = {
    "pose_backend": ("pose", "backend"),
    "colmap_matcher": ("pose", "matcher"),
}


@dataclass
class SplatConfig:
    gpu: str = "H100"              # "H100" | "L40S" | "A100"
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_overrides(self, data: dict[str, Any] | None) -> SplatConfig:
        """Mutate & return `self` applying only the nested or flat keys explicitly present in `data`."""
        if not data:
            return self
        if data.get("gpu"):
            self.gpu = str(data["gpu"]).upper()

        sub_objs = {
            "extract": self.extract,
            "pose": self.pose,
            "train": self.train,
            "export": self.export,
        }
        # 1. Apply nested dicts (`{"train": {...}}`)
        for sub_name, sub_obj in sub_objs.items():
            raw_sub = data.get(sub_name)
            if isinstance(raw_sub, dict):
                for fk in sub_obj.__dataclass_fields__:
                    if fk in raw_sub and raw_sub[fk] is not None:
                        setattr(sub_obj, fk, raw_sub[fk])

        # 2. Apply flat top-level keys (`{"opa_reg": 0.001, "lambda_dssim": 0.25, "focus_camera": ...}`)
        for k, v in data.items():
            if k in ("gpu", "extract", "pose", "train", "export") or v is None:
                continue
            if k in _FLAT_KEY_ALIASES:
                s_name, f_name = _FLAT_KEY_ALIASES[k]
                setattr(sub_objs[s_name], f_name, v)
                continue
            # Route unambiguous field names to their matching sub-dataclass
            for sub_obj in sub_objs.values():
                if k in sub_obj.__dataclass_fields__:
                    setattr(sub_obj, k, v)
                    break
        if self.train.rasterize_mode not in ("classic", "antialiased"):
            self.train.rasterize_mode = "antialiased"
        return self

    def merge_overrides(self, overrides: dict[str, Any] | None) -> SplatConfig:
        """Return a copy of `self` with `overrides` merged on top (preserving all untouched saved settings)."""
        cloned = SplatConfig.from_dict(self.to_dict())
        return cloned.apply_overrides(overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SplatConfig:
        inst = cls()
        return inst.apply_overrides(data)