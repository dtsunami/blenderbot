"""Pure-stdlib Sim(3) coordinate, quaternion, camera basis, and RANSAC `Z=0` floor alignment math.

Zero third-party imports (`math` + `random` only) so identical reference math runs on the Windows
host, inside unit tests (`P1` gate), and on the cloud worker.

Conventions:
  - Quaternions are `(w, x, y, z)` unit-normalized (matches `gsplat` & INRIA 3DGS `rot_0..3`).
  - Scales in 3DGS `.ply` are stored in log-space (`scale_lin = exp(log_scale)`).
  - `gsplat` / COLMAP / OpenCV camera space: `+X right, +Y down, +Z forward` (`x_cam = R_c x_w + t_c`).
  - Blender 5.3 world space: right-handed `+X right, +Y forward, +Z up`, dominant floor at `Z = 0`.
  - Blender 5.3 camera local space: `+X right, +Y up, -Z forward`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Mat3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]


@dataclass(frozen=True)
class Sim3Transform:
    """Proper similarity transform `x_new = scale * (R @ x_old) + t` with `det(R) == +1`, `scale > 0`."""

    scale: float
    R: Mat3
    t: Vec3
    floor_inlier_ratio: float = 0.0
    baseline_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.scale <= 0.0 or not math.isfinite(self.scale):
            raise ValueError(f"Sim3 scale must be finite and > 0 (got {self.scale}).")
        det = mat3_det(self.R)
        if abs(det - 1.0) > 1e-4:
            raise ValueError(
                f"Sim3 rotation R must be a proper SO(3) matrix with det=+1 (got det={det:.6f})."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "R": [list(row) for row in self.R],
            "t": list(self.t),
            "floor_inlier_ratio": round(self.floor_inlier_ratio, 4),
            "baseline_ratio": round(self.baseline_ratio, 5),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Sim3Transform:
        r = d["R"]
        t = d["t"]
        return cls(
            scale=float(d["scale"]),
            R=(
                (float(r[0][0]), float(r[0][1]), float(r[0][2])),
                (float(r[1][0]), float(r[1][1]), float(r[1][2])),
                (float(r[2][0]), float(r[2][1]), float(r[2][2])),
            ),
            t=(float(t[0]), float(t[1]), float(t[2])),
            floor_inlier_ratio=float(d.get("floor_inlier_ratio", 0.0)),
            baseline_ratio=float(d.get("baseline_ratio", 0.0)),
        )

    @classmethod
    def identity(cls) -> Sim3Transform:
        return cls(
            scale=1.0,
            R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            t=(0.0, 0.0, 0.0),
        )


# ─── Linear algebra primitives ────────────────────────────────────────────────


def vec3_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_scale(a: Sequence[float], s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec3_cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec3_norm(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vec3_normalize(a: Sequence[float], fallback: Vec3 = (0.0, 0.0, 1.0)) -> Vec3:
    n = vec3_norm(a)
    if n < 1e-12:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def mat3_det(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def mat3_transpose(m: Sequence[Sequence[float]]) -> Mat3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def mat3_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Mat3:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def mat3_vec(m: Sequence[Sequence[float]], v: Sequence[float]) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


# ─── Quaternion `(w, x, y, z)` <-> SO(3) ──────────────────────────────────────


def quat_normalize(q: Sequence[float]) -> QuatWXYZ:
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def quat_mul(q1: Sequence[float], q2: Sequence[float]) -> QuatWXYZ:
    """Hamilton product `q1 * q2` for `(w, x, y, z)` quaternions."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return quat_normalize((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def quat_to_rotmat(q: Sequence[float]) -> Mat3:
    """Convert `(w, x, y, z)` quaternion to 3x3 rotation matrix (matches `gsplat` & COLMAP)."""
    w, x, y, z = quat_normalize(q)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
        (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
        (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    )


def rotmat_to_quat(r: Sequence[Sequence[float]]) -> QuatWXYZ:
    """Shepperd's method: convert proper 3x3 rotation matrix `R` (`det=+1`) to `(w, x, y, z)`."""
    m00, m01, m02 = r[0][0], r[0][1], r[0][2]
    m10, m11, m12 = r[1][0], r[1][1], r[1][2]
    m20, m21, m22 = r[2][0], r[2][1], r[2][2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = quat_normalize((w, x, y, z))
    return q if q[0] >= 0.0 else (-q[0], -q[1], -q[2], -q[3])


def rotation_between_vectors(a: Sequence[float], b: Sequence[float]) -> Mat3:
    """Minimal proper rotation `R` (`det=+1`) sending unit vector `a` to unit vector `b`."""
    ua = vec3_normalize(a)
    ub = vec3_normalize(b)
    c = max(-1.0, min(1.0, vec3_dot(ua, ub)))
    if c > 1.0 - 1e-10:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if c < -1.0 + 1e-10:
        # 180 deg rotation around an axis orthogonal to `ua`
        ortho = (1.0, 0.0, 0.0) if abs(ua[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = vec3_normalize(vec3_cross(ua, ortho))
        return quat_to_rotmat((0.0, axis[0], axis[1], axis[2]))
    v = vec3_cross(ua, ub)
    s = math.sqrt((1.0 + c) * 2.0)
    q = quat_normalize((0.5 * s, v[0] / s, v[1] / s, v[2] / s))
    return quat_to_rotmat(q)


def gaussian_covariance_3d(quat_wxyz: Sequence[float], log_scale: Sequence[float]) -> Mat3:
    """Compute 3D covariance `Sigma = R(q) * diag(exp(2 * log_scale)) * R(q)^T`."""
    r = quat_to_rotmat(quat_wxyz)
    s2 = (
        math.exp(2.0 * log_scale[0]),
        math.exp(2.0 * log_scale[1]),
        math.exp(2.0 * log_scale[2]),
    )
    rs2 = (
        (r[0][0] * s2[0], r[0][1] * s2[1], r[0][2] * s2[2]),
        (r[1][0] * s2[0], r[1][1] * s2[1], r[1][2] * s2[2]),
        (r[2][0] * s2[0], r[2][1] * s2[1], r[2][2] * s2[2]),
    )
    return mat3_mul(rs2, mat3_transpose(r))


# ─── Sim(3) transforms on Gaussians & Cameras ─────────────────────────────────


def apply_sim3_to_point(sim3: Sim3Transform, xyz: Sequence[float]) -> Vec3:
    rx = mat3_vec(sim3.R, xyz)
    return (
        sim3.scale * rx[0] + sim3.t[0],
        sim3.scale * rx[1] + sim3.t[1],
        sim3.scale * rx[2] + sim3.t[2],
    )


def apply_sim3_to_gaussians_py(
    sim3: Sim3Transform,
    means: Sequence[Sequence[float]],
    log_scales: Sequence[Sequence[float]],
    quats_wxyz: Sequence[Sequence[float]],
) -> tuple[list[Vec3], list[Vec3], list[QuatWXYZ]]:
    """Transform 3DGS `(means, log_scales, quats_wxyz)` by `sim3` without distorting anisotropy."""
    log_s = math.log(sim3.scale)
    q_rot = rotmat_to_quat(sim3.R)
    out_means: list[Vec3] = []
    out_scales: list[Vec3] = []
    out_quats: list[QuatWXYZ] = []
    for m, ls, q in zip(means, log_scales, quats_wxyz):
        out_means.append(apply_sim3_to_point(sim3, m))
        out_scales.append((ls[0] + log_s, ls[1] + log_s, ls[2] + log_s))
        out_quats.append(quat_mul(q_rot, q))
    return out_means, out_scales, out_quats


def apply_sim3_to_camera_w2c(
    sim3: Sim3Transform,
    r_w2c: Sequence[Sequence[float]],
    t_w2c: Sequence[float],
) -> tuple[Mat3, Vec3]:
    """Transform OpenCV world-to-camera `(R_c, t_c)` under `x_new = s * R_w * x_old + t_w`.

    Preserves orthonormal `R_c_new` (`det=+1`) and exact pixel rays:
      `R_c_new = R_c @ R_w^T`
      `t_c_new = s * t_c - R_c_new @ t_w`
      `x_cam_new = s * x_cam_old`
    """
    r_new = mat3_mul(r_w2c, mat3_transpose(sim3.R))
    rt_tw = mat3_vec(r_new, sim3.t)
    t_new = (
        sim3.scale * t_w2c[0] - rt_tw[0],
        sim3.scale * t_w2c[1] - rt_tw[1],
        sim3.scale * t_w2c[2] - rt_tw[2],
    )
    return r_new, t_new


def opencv_w2c_to_blender_camera(
    r_w2c: Sequence[Sequence[float]],
    t_w2c: Sequence[float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    sensor_width_mm: float = 36.0,
) -> dict[str, Any]:
    """Convert OpenCV world-to-camera `[R|t]` + `K` to Blender 5.3 Camera parameters.

    OpenCV camera axes: `+X right, +Y down, +Z forward`.
    Blender camera axes: `+X right, +Y up, -Z forward` (`B = diag(1, -1, -1)`, `det(B) = +1`).
    """
    r_c2w = mat3_transpose(r_w2c)
    neg_rt_t = mat3_vec(r_c2w, (-t_w2c[0], -t_w2c[1], -t_w2c[2]))

    # R_bl_c2w = R_c2w @ diag(1, -1, -1): negate columns 1 and 2
    r_bl: Mat3 = (
        (r_c2w[0][0], -r_c2w[0][1], -r_c2w[0][2]),
        (r_c2w[1][0], -r_c2w[1][1], -r_c2w[1][2]),
        (r_c2w[2][0], -r_c2w[2][1], -r_c2w[2][2]),
    )
    quat_bl = rotmat_to_quat(r_bl)

    max_dim = float(max(width, height))
    if width >= height:
        sensor_fit = "HORIZONTAL"
        lens_mm = fx * sensor_width_mm / float(width)
    else:
        sensor_fit = "VERTICAL"
        # When fit='VERTICAL', Blender interprets sensor_height (default 24mm or sensor_width)
        lens_mm = fy * sensor_width_mm / float(height)

    # Blender optical-center shift conventions (normalized by max(W, H)):
    # shift_x > 0 shifts principal point left in image -> (width/2 - cx) / max_dim?
    # Standard Blender projection matrix: M[0,2] = -2 * shift_x * (max_dim / width)
    # Matching OpenCV u = fx * X / (-Z) + cx gives shift_x = -(cx - width * 0.5) / max_dim
    # and shift_y = (cy - height * 0.5) / max_dim (since +Y is down in OpenCV, up in Blender).
    shift_x = -(cx - 0.5 * width) / max_dim
    shift_y = (cy - 0.5 * height) / max_dim

    matrix_world = [
        [r_bl[0][0], r_bl[0][1], r_bl[0][2], neg_rt_t[0]],
        [r_bl[1][0], r_bl[1][1], r_bl[1][2], neg_rt_t[1]],
        [r_bl[2][0], r_bl[2][1], r_bl[2][2], neg_rt_t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "location": [neg_rt_t[0], neg_rt_t[1], neg_rt_t[2]],
        "rotation_quaternion": [quat_bl[0], quat_bl[1], quat_bl[2], quat_bl[3]],
        "matrix_world": matrix_world,
        "lens_mm": round(lens_mm, 5),
        "sensor_width_mm": sensor_width_mm,
        "sensor_fit": sensor_fit,
        "shift_x": round(shift_x, 6),
        "shift_y": round(shift_y, 6),
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }


# ─── Gravity + RANSAC `Z = 0` Floor Aligner ───────────────────────────────────


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if (n % 2 == 1) else 0.5 * (s[mid - 1] + s[mid])


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, round((pct / 100.0) * (len(s) - 1))))
    return s[idx]


def estimate_floor_aligned_sim3(
    cam_r_w2c_list: Sequence[Sequence[Sequence[float]]],
    cam_t_w2c_list: Sequence[Sequence[float]],
    points_xyz: Sequence[Sequence[float]],
    *,
    align_floor: bool = True,
    target_cam_height_m: float = 1.60,
    ransac_iters: int = 240,
    seed: int = 42,
) -> Sim3Transform:
    """Compute a proper Sim(3) mapping arbitrary OpenCV/SfM space into Blender `+Z` up, `Z=0` floor.

    1. Uses average camera `-Y` (OpenCV up) + first camera `+Z` (forward) to build a gravity-level
       right-handed basis `R0` (`+X right, +Y forward, +Z up`, `det=+1`).
    2. Runs deterministic RANSAC plane fitting on the lower 35% of nearby scene points whose plane
       normal is within 28° of `+Z` to lock the dominant floor to exact `+Z` normal.
    3. Scales the world so median camera elevation above the floor equals `target_cam_height_m`
       (falls back to median camera-to-scene depth = `4.0m` if no clear floor exists) and shifts
       translation so the floor sits at `Z = 0` and scene look-center sits near `(X=0, Y=0)`.
    """
    if not cam_r_w2c_list:
        return Sim3Transform.identity()

    # 1. Camera centers and average OpenCV up (`-Y_cam`) & initial forward (`+Z_cam`)
    cam_centers: list[Vec3] = []
    up_sum = (0.0, 0.0, 0.0)
    fwd_0 = (0.0, 1.0, 0.0)

    for idx, (r_c, t_c) in enumerate(zip(cam_r_w2c_list, cam_t_w2c_list)):
        r_c2w = mat3_transpose(r_c)
        c_w = mat3_vec(r_c2w, (-t_c[0], -t_c[1], -t_c[2]))
        cam_centers.append(c_w)
        # OpenCV -Y axis in world is -row_1(R_c)
        cam_up = (-r_c[1][0], -r_c[1][1], -r_c[1][2])
        up_sum = vec3_add(up_sum, vec3_normalize(cam_up))
        if idx == 0:
            fwd_0 = vec3_normalize((r_c[2][0], r_c[2][1], r_c[2][2]), fallback=(0.0, 1.0, 0.0))

    z_axis = vec3_normalize(up_sum, fallback=(0.0, 0.0, 1.0))
    # Project fwd_0 onto horizontal plane orthogonal to z_axis -> Blender +Y
    fwd_proj = vec3_sub(fwd_0, vec3_scale(z_axis, vec3_dot(fwd_0, z_axis)))
    if vec3_norm(fwd_proj) < 1e-6:
        alt = (1.0, 0.0, 0.0) if abs(z_axis[0]) < 0.9 else (0.0, 1.0, 0.0)
        fwd_proj = vec3_sub(alt, vec3_scale(z_axis, vec3_dot(alt, z_axis)))
    y_axis = vec3_normalize(fwd_proj)
    # Right-handed Blender basis: X = Y x Z (since i = j x k in right-handed XYZ!)
    x_axis = vec3_normalize(vec3_cross(y_axis, z_axis))
    y_axis = vec3_normalize(vec3_cross(z_axis, x_axis))

    # Rows of R0 map old world vectors onto (x_axis, y_axis, z_axis)
    r0: Mat3 = (x_axis, y_axis, z_axis)

    rng = random.Random(seed)
    pts_sample = list(points_xyz)
    if len(pts_sample) > 12_000:
        pts_sample = rng.sample(pts_sample, 12_000)

    pts_r0 = [mat3_vec(r0, p) for p in pts_sample]
    cams_r0 = [mat3_vec(r0, c) for c in cam_centers]

    cam_cx = _median([c[0] for c in cams_r0])
    cam_cy = _median([c[1] for c in cams_r0])
    cam_cz = _median([c[2] for c in cams_r0])

    # Baseline ratio diagnostic (max camera pair distance / median point distance to cam centroid)
    max_cam_span = 0.0
    for i in range(len(cams_r0)):
        for j in range(i + 1, len(cams_r0)):
            d = vec3_norm(vec3_sub(cams_r0[i], cams_r0[j]))
            max_cam_span = max(max_cam_span, d)

    if not pts_r0:
        t_only = (-cam_cx, -cam_cy, target_cam_height_m - cam_cz)
        return Sim3Transform(scale=1.0, R=r0, t=t_only)

    dists_to_cam = [vec3_norm(vec3_sub(p, (cam_cx, cam_cy, cam_cz))) for p in pts_r0]
    med_depth = max(_median(dists_to_cam), 1e-4)
    baseline_ratio = max_cam_span / med_depth

    # Filter out extreme sky/outlier shell beyond 85th percentile distance
    max_r = max(_percentile(dists_to_cam, 85.0), med_depth * 1.5)
    near_pts = [p for p, d in zip(pts_r0, dists_to_cam) if d <= max_r]
    if len(near_pts) < 24:
        near_pts = pts_r0

    r_world = r0
    floor_inlier_ratio = 0.0

    # 2. RANSAC floor plane fit on bottom 40% Z candidates
    if align_floor and len(near_pts) >= 24:
        z_cut = _percentile([p[2] for p in near_pts], 40.0)
        floor_cands = [p for p in near_pts if p[2] <= z_cut]
        if len(floor_cands) >= 12:
            inlier_tol = max(0.025 * med_depth, 1e-4)
            cos_max_tilt = math.cos(math.radians(28.0))
            best_normal: Vec3 | None = None
            best_inliers = 0

            n_cands = len(floor_cands)
            for _ in range(ransac_iters):
                i1, i2, i3 = rng.sample(range(n_cands), 3)
                p1, p2, p3 = floor_cands[i1], floor_cands[i2], floor_cands[i3]
                n_raw = vec3_cross(vec3_sub(p2, p1), vec3_sub(p3, p1))
                n_len = vec3_norm(n_raw)
                if n_len < 1e-10:
                    continue
                n_unit = (n_raw[0] / n_len, n_raw[1] / n_len, n_raw[2] / n_len)
                if n_unit[2] < 0.0:
                    n_unit = (-n_unit[0], -n_unit[1], -n_unit[2])
                if n_unit[2] < cos_max_tilt:
                    continue
                d0 = vec3_dot(n_unit, p1)
                cnt = sum(1 for q in floor_cands if abs(vec3_dot(n_unit, q) - d0) <= inlier_tol)
                if cnt > best_inliers:
                    best_inliers = cnt
                    best_normal = n_unit

            if best_normal is not None and best_inliers >= max(8, int(0.18 * n_cands)):
                floor_inlier_ratio = best_inliers / float(n_cands)
                r_tilt = rotation_between_vectors(best_normal, (0.0, 0.0, 1.0))
                r_world = mat3_mul(r_tilt, r0)

    # 3. Recompute rotated coordinates to determine floor Z=0 and metric scale
    near_rot = [mat3_vec(r_world, p) for p in points_xyz[: len(pts_sample)]]
    if len(near_rot) > len(near_pts):
        # Apply same outlier filter
        near_rot = [
            mat3_vec(r_world, p)
            for p, d in zip(pts_sample, dists_to_cam)
            if d <= max_r
        ] or near_rot
    cams_rot = [mat3_vec(r_world, c) for c in cam_centers]

    if floor_inlier_ratio > 0.0:
        # Re-select inliers near bottom floor band for crisp Z=0 median
        z_vals = [p[2] for p in near_rot]
        z_low = _percentile(z_vals, 20.0)
        band = max(0.03 * med_depth, 1e-4)
        floor_zs = [z for z in z_vals if abs(z - z_low) <= band * 2.0]
        floor_z = _median(floor_zs) if floor_zs else z_low
    else:
        floor_z = _percentile([p[2] for p in near_rot], 5.0)

    med_cam_z = _median([c[2] for c in cams_rot])
    raw_cam_height = med_cam_z - floor_z

    if target_cam_height_m > 0.0 and raw_cam_height > 0.15 * med_depth:
        scale = target_cam_height_m / raw_cam_height
    else:
        scale = 4.0 / med_depth
    scale = max(1e-3, min(1e3, scale))

    center_x = _median([p[0] for p in near_rot])
    center_y = _median([p[1] for p in near_rot])
    t_world: Vec3 = (
        -scale * center_x,
        -scale * center_y,
        -scale * floor_z,
    )

    return Sim3Transform(
        scale=scale,
        R=r_world,
        t=t_world,
        floor_inlier_ratio=floor_inlier_ratio,
        baseline_ratio=baseline_ratio,
    )


# ─── Camera Inverse Mapping, SLERP/Spline Trajectory & Pose Telemetry ─────────


def blender_camera_to_opencv_w2c(
    cam: dict[str, Any],
    target_width: int | None = None,
    target_height: int | None = None,
) -> dict[str, Any]:
    """Exact inverse of `opencv_w2c_to_blender_camera`.

    Converts a `cameras.json` camera dict (`matrix_world` or `location` + `rotation_quaternion`)
    into OpenCV `(R_w2c, t_w2c, fx, fy, cx, cy, width, height)` for `gsplat` rasterization or
    known-pose video training.
    """
    if cam.get("matrix_world"):
        mw = cam["matrix_world"]
        r_bl: Mat3 = (
            (float(mw[0][0]), float(mw[0][1]), float(mw[0][2])),
            (float(mw[1][0]), float(mw[1][1]), float(mw[1][2])),
            (float(mw[2][0]), float(mw[2][1]), float(mw[2][2])),
        )
        loc: Vec3 = (float(mw[0][3]), float(mw[1][3]), float(mw[2][3]))
    else:
        q = cam.get("rotation_quaternion") or [1.0, 0.0, 0.0, 0.0]
        r_bl = quat_to_rotmat([float(x) for x in q])
        raw_loc = cam.get("location") or [0.0, 0.0, 0.0]
        loc = (float(raw_loc[0]), float(raw_loc[1]), float(raw_loc[2]))

    # r_c2w = r_bl @ diag(1, -1, -1); r_w2c = r_c2w^T
    r_w2c: Mat3 = (
        (r_bl[0][0], r_bl[1][0], r_bl[2][0]),
        (-r_bl[0][1], -r_bl[1][1], -r_bl[2][1]),
        (-r_bl[0][2], -r_bl[1][2], -r_bl[2][2]),
    )
    rt_loc = mat3_vec(r_w2c, loc)
    t_w2c: Vec3 = (-rt_loc[0], -rt_loc[1], -rt_loc[2])

    src_w = int(cam.get("width") or target_width or 1280)
    src_h = int(cam.get("height") or target_height or 720)
    out_w = int(target_width or src_w)
    out_h = int(target_height or src_h)

    if cam.get("fx") and cam.get("fy"):
        sx = out_w / float(max(1, src_w))
        sy = out_h / float(max(1, src_h))
        fx = float(cam["fx"]) * sx
        fy = float(cam["fy"]) * sy
        cx = float(cam.get("cx", src_w * 0.5)) * sx
        cy = float(cam.get("cy", src_h * 0.5)) * sy
    else:
        lens_mm = float(cam.get("lens_mm", 50.0))
        sensor_w = float(cam.get("sensor_width_mm", 36.0))
        sfit = str(cam.get("sensor_fit", "HORIZONTAL" if out_w >= out_h else "VERTICAL"))
        if sfit == "VERTICAL":
            fy = lens_mm * float(out_h) / max(1e-4, sensor_w)
            fx = fy
        else:
            fx = lens_mm * float(out_w) / max(1e-4, sensor_w)
            fy = fx
        max_dim = float(max(out_w, out_h))
        shift_x = float(cam.get("shift_x", 0.0))
        shift_y = float(cam.get("shift_y", 0.0))
        cx = 0.5 * out_w - shift_x * max_dim
        cy = 0.5 * out_h + shift_y * max_dim

    return {
        "R_w2c": [list(row) for row in r_w2c],
        "t_w2c": list(t_w2c),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": out_w,
        "height": out_h,
    }


def quat_slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> QuatWXYZ:
    """Shortest-arc spherical linear interpolation on unit `(w, x, y, z)` quaternions."""
    qa = quat_normalize(q0)
    qb = quat_normalize(q1)
    dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]
    if dot < 0.0:
        qb = (-qb[0], -qb[1], -qb[2], -qb[3])
        dot = -dot
    u = max(0.0, min(1.0, float(t)))
    if dot > 0.9995:
        return quat_normalize((
            qa[0] + u * (qb[0] - qa[0]),
            qa[1] + u * (qb[1] - qa[1]),
            qa[2] + u * (qb[2] - qa[2]),
            qa[3] + u * (qb[3] - qa[3]),
        ))
    theta_0 = math.acos(min(1.0, dot))
    sin_0 = math.sin(theta_0)
    theta = theta_0 * u
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_0
    s1 = math.sin(theta) / sin_0
    return quat_normalize((
        s0 * qa[0] + s1 * qb[0],
        s0 * qa[1] + s1 * qb[1],
        s0 * qa[2] + s1 * qb[2],
        s0 * qa[3] + s1 * qb[3],
    ))


def catmull_rom_vec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    """C1-smooth cubic Catmull-Rom spline between `p1` (`t=0`) and `p2` (`t=1`)."""
    u = max(0.0, min(1.0, float(t)))
    u2 = u * u
    u3 = u2 * u
    out = []
    for k in range(3):
        v = 0.5 * (
            (2.0 * p1[k])
            + (-p0[k] + p2[k]) * u
            + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * u2
            + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * u3
        )
        out.append(v)
    return (out[0], out[1], out[2])


def centripetal_catmull_rom_vec3(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    u: float,
    dt01: float = 1.0,
    dt12: float = 1.0,
    dt23: float = 1.0,
) -> Vec3:
    """Barry-Goldman centripetal (`α=0.5`) + time-chord non-uniform Catmull-Rom spline.

    Guaranteed cusp-free and overshoot-free across sharp 90° indoor walkthrough turns and
    non-uniform `mpdecimate` pauses (`dt01 != dt12 != dt23`).
    """
    u_c = max(0.0, min(1.0, float(u)))

    def _knot_step(pa: Vec3, pb: Vec3, dt_s: float) -> float:
        chord = math.sqrt(vec3_norm(vec3_sub(pb, pa)))
        return max(1e-4, 0.7 * chord + 0.3 * math.sqrt(max(1e-4, dt_s)))

    s0 = 0.0
    s1 = s0 + _knot_step(p0, p1, dt01)
    s2 = s1 + _knot_step(p1, p2, dt12)
    s3 = s2 + _knot_step(p2, p3, dt23)
    s = s1 + u_c * (s2 - s1)

    def _lerp(va: Vec3, vb: Vec3, ta: float, tb: float, tc: float) -> Vec3:
        denom = max(1e-8, tb - ta)
        w = (tc - ta) / denom
        return (
            va[0] + w * (vb[0] - va[0]),
            va[1] + w * (vb[1] - va[1]),
            va[2] + w * (vb[2] - va[2]),
        )

    a1 = _lerp(p0, p1, s0, s1, s)
    a2 = _lerp(p1, p2, s1, s2, s)
    a3 = _lerp(p2, p3, s2, s3, s)
    b1 = _lerp(a1, a2, s0, s2, s)
    b2 = _lerp(a2, a3, s1, s3, s)
    return _lerp(b1, b2, s1, s2, s)


def camera_orientation_angles_deg(quat_bl_wxyz: Sequence[float]) -> dict[str, Any]:
    """Compute intuitive world-space camera basis + Yaw / Pitch / Roll (degrees) in Blender Z-up space.

    - `forward`: optical look ray (`-Z_cam` in world)
    - `yaw_deg`: horizontal heading relative to `+Y` forward (`+90°` = looking right `+X`)
    - `pitch_deg`: elevation angle (`+` looking up `+Z`, `-` looking down)
    - `roll_deg`: bank/tilt around optical axis relative to world horizon (`0°` = level)
    """
    r_bl = quat_to_rotmat(quat_bl_wxyz)
    right: Vec3 = (r_bl[0][0], r_bl[1][0], r_bl[2][0])
    up: Vec3 = (r_bl[0][1], r_bl[1][1], r_bl[2][1])
    fwd: Vec3 = vec3_normalize((-r_bl[0][2], -r_bl[1][2], -r_bl[2][2]), fallback=(0.0, 1.0, 0.0))

    yaw = math.degrees(math.atan2(fwd[0], fwd[1]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))

    world_up: Vec3 = (0.0, 0.0, 1.0)
    horiz_right_raw = vec3_cross(fwd, world_up)
    if vec3_norm(horiz_right_raw) < 1e-5:
        roll = 0.0
    else:
        h_right = vec3_normalize(horiz_right_raw)
        h_up = vec3_normalize(vec3_cross(h_right, fwd))
        roll = math.degrees(math.atan2(vec3_dot(right, h_up), vec3_dot(right, h_right)))

    return {
        "forward": [round(fwd[0], 5), round(fwd[1], 5), round(fwd[2], 5)],
        "right": [round(right[0], 5), round(right[1], 5), round(right[2], 5)],
        "up": [round(up[0], 5), round(up[1], 5), round(up[2], 5)],
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
    }


def euler_yaw_pitch_roll_to_blender_quat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> QuatWXYZ:
    """Exact inverse of `camera_orientation_angles_deg`: construct a gravity-level Blender `(w,x,y,z)` quaternion.

    Interpolating unwrapped `(yaw_deg, pitch_deg, roll_deg)` keeps the horizon locked (`roll ≈ 0°`)
    across fast 150°–360° whip-pans instead of quaternion SLERP banking diagonally across the ceiling
    or flipping 180° the wrong way!
    """
    yr = math.radians(float(yaw_deg))
    pr = math.radians(max(-89.5, min(89.5, float(pitch_deg))))
    rr = math.radians(float(roll_deg))

    cp = math.cos(pr)
    fwd: Vec3 = (math.sin(yr) * cp, math.cos(yr) * cp, math.sin(pr))
    world_up: Vec3 = (0.0, 0.0, 1.0)
    h_right_raw = vec3_cross(fwd, world_up)
    if vec3_norm(h_right_raw) < 1e-6:
        h_right: Vec3 = (math.cos(yr), -math.sin(yr), 0.0)
    else:
        h_right = vec3_normalize(h_right_raw)
    h_up = vec3_normalize(vec3_cross(h_right, fwd))

    cr, sr = math.cos(rr), math.sin(rr)
    right = (
        cr * h_right[0] + sr * h_up[0],
        cr * h_right[1] + sr * h_up[1],
        cr * h_right[2] + sr * h_up[2],
    )
    up = (
        -sr * h_right[0] + cr * h_up[0],
        -sr * h_right[1] + cr * h_up[1],
        -sr * h_right[2] + cr * h_up[2],
    )
    # Blender camera looks down `-Z_cam`, so column 2 of `R_bl` is `-fwd`
    r_bl: Mat3 = (
        (right[0], up[0], -fwd[0]),
        (right[1], up[1], -fwd[1]),
        (right[2], up[2], -fwd[2]),
    )
    return rotmat_to_quat(r_bl)


def pchip_monotone_scalar(
    v0: float,
    v1: float,
    v2: float,
    v3: float,
    u: float,
    dt01: float = 1.0,
    dt12: float = 1.0,
    dt23: float = 1.0,
) -> float:
    """Fritsch–Carlson monotone cubic Hermite (PCHIP) interpolation on `[v1, v2]`.

    Unlike linear SLERP (constant angular speed that misses fast whip-pan S-curves) and unlike
    unconstrained Catmull-Rom (which overshoots/reverses direction after a sudden stop!), PCHIP
    is C1-smooth, captures fast pan acceleration, and NEVER overshoots or reverses pan direction!
    """
    uc = max(0.0, min(1.0, float(u)))
    h0 = max(1e-5, float(dt01))
    h1 = max(1e-5, float(dt12))
    h2 = max(1e-5, float(dt23))

    d0 = (v1 - v0) / h0
    d1 = (v2 - v1) / h1
    d2 = (v3 - v2) / h2

    def _slope(da: float, db: float, ha: float, hb: float) -> float:
        if da * db <= 0.0:
            return 0.0
        w1 = 2.0 * hb + ha
        w2 = hb + 2.0 * ha
        return (w1 + w2) / (w1 / da + w2 / db)

    m1 = _slope(d0, d1, h0, h1)
    m2 = _slope(d1, d2, h1, h2)

    u2 = uc * uc
    u3 = u2 * uc
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + uc
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return h00 * v1 + h10 * (h1 * m1) + h01 * v2 + h11 * (h1 * m2)


def interpolate_camera_trajectory(
    cameras_doc: dict[str, Any],
    *,
    fps: float | None = None,
    duration_sec: float | None = None,
    total_frames: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
) -> dict[str, Any]:
    """Build a full-rate frame-accurate camera trajectory + orientation/velocity telemetry.

    Recreates exact original video timing (`fps`, `duration_sec`, `total_frames`) from solved
    keyframes (`pts_sec` or `frame_index`) using C1 Catmull-Rom positions and hemisphere-aligned
    quaternion SLERP, marking exact solved keyframes (`is_keyframe: True`).
    """
    raw_cams = list(cameras_doc.get("cameras") or [])
    if not raw_cams:
        raise ValueError("cameras_doc contains no cameras.")

    vmeta = cameras_doc.get("video_meta") or {}
    eff_fps = float(fps or vmeta.get("fps") or 24.0)
    eff_fps = max(1.0, min(240.0, eff_fps))

    c0 = raw_cams[0]
    out_w = int(target_width or vmeta.get("width") or c0.get("width") or 1280)
    out_h = int(target_height or vmeta.get("height") or c0.get("height") or 720)
    # Ensure even dimensions for H.264 yuv420p encoding
    out_w = max(16, out_w - (out_w % 2))
    out_h = max(16, out_h - (out_h % 2))

    has_pts = any(c.get("pts_sec") is not None for c in raw_cams)
    if duration_sec is not None:
        eff_dur = max(0.0, float(duration_sec))
    elif vmeta.get("duration_sec") is not None:
        eff_dur = max(0.0, float(vmeta["duration_sec"]))
    elif has_pts:
        max_pts = max(float(c.get("pts_sec") or 0.0) for c in raw_cams)
        eff_dur = max(max_pts + 1.0 / eff_fps, len(raw_cams) / eff_fps)
    else:
        eff_dur = max(0.0, (len(raw_cams) - 1) / eff_fps)

    if total_frames is not None:
        n_out = max(1, int(total_frames))
    elif vmeta.get("total_frames"):
        n_out = max(1, int(vmeta["total_frames"]))
    elif has_pts or duration_sec is not None or vmeta.get("duration_sec") is not None:
        n_out = max(1, round(eff_dur * eff_fps))
    else:
        n_out = len(raw_cams)

    # Normalize & sort keyframes by timestamp `t_sec`
    k_times: list[float] = []
    k_locs: list[Vec3] = []
    k_quats: list[QuatWXYZ] = []
    k_intr: list[dict[str, float]] = []
    k_names: list[str] = []

    span_dur = max(1e-6, (n_out - 1) / eff_fps)
    n_k = len(raw_cams)
    prev_q: QuatWXYZ | None = None

    for idx, c in enumerate(raw_cams):
        if c.get("pts_sec") is not None:
            tk = float(c["pts_sec"])
        elif n_k > 1:
            tk = (idx / float(n_k - 1)) * span_dur
        else:
            tk = 0.0
        # Ensure strictly non-decreasing timestamps
        if k_times and tk <= k_times[-1]:
            tk = k_times[-1] + 1e-4
        k_times.append(tk)

        loc_c = c.get("location")
        if loc_c is None and c.get("matrix_world"):
            mw = c["matrix_world"]
            loc_c = [mw[0][3], mw[1][3], mw[2][3]]
        loc_v: Vec3 = (float(loc_c[0]), float(loc_c[1]), float(loc_c[2])) if loc_c else (0.0, 0.0, 0.0)
        k_locs.append(loc_v)

        if c.get("rotation_quaternion"):
            rq = quat_normalize([float(x) for x in c["rotation_quaternion"]])
        elif c.get("matrix_world"):
            mw = c["matrix_world"]
            rq = rotmat_to_quat((
                (float(mw[0][0]), float(mw[0][1]), float(mw[0][2])),
                (float(mw[1][0]), float(mw[1][1]), float(mw[1][2])),
                (float(mw[2][0]), float(mw[2][1]), float(mw[2][2])),
            ))
        else:
            rq = (1.0, 0.0, 0.0, 0.0)

        if prev_q is not None and (prev_q[0] * rq[0] + prev_q[1] * rq[1] + prev_q[2] * rq[2] + prev_q[3] * rq[3]) < 0.0:
            rq = (-rq[0], -rq[1], -rq[2], -rq[3])
        prev_q = rq
        k_quats.append(rq)

        cv_info = blender_camera_to_opencv_w2c(c, target_width=out_w, target_height=out_h)
        k_intr.append({
            "fx": cv_info["fx"],
            "fy": cv_info["fy"],
            "cx": cv_info["cx"],
            "cy": cv_info["cy"],
            "lens_mm": float(c.get("lens_mm", 50.0)),
            "sensor_width_mm": float(c.get("sensor_width_mm", 36.0)),
        })
        k_names.append(str(c.get("image_name") or f"key_{idx:04d}"))

    # Unwrap keyframe `(yaw_deg, pitch_deg, roll_deg)` continuously (+ optional optical-flow pan hints `pan_dyaw_deg`)
    # so fast >150° FPS mouse pans turn in the TRUE pan direction and keep the horizon level (`roll ≈ 0°`)!
    k_yaws: list[float] = []
    k_pitches: list[float] = []
    k_rolls: list[float] = []
    for ki, qk in enumerate(k_quats):
        ang_k = camera_orientation_angles_deg(qk)
        yk = float(ang_k["yaw_deg"])
        pk = float(ang_k["pitch_deg"])
        rk = float(ang_k["roll_deg"])
        if ki > 0:
            # Optional video optical-flow delta-yaw hint resolves >160° pan direction ambiguity (`+190°` vs `-170°`)!
            hint_dyaw = raw_cams[ki].get("pan_dyaw_deg")
            target_y = (k_yaws[-1] + float(hint_dyaw)) if hint_dyaw is not None else k_yaws[-1]
            while yk - target_y > 180.0:
                yk -= 360.0
            while yk - target_y < -180.0:
                yk += 360.0
            while rk - k_rolls[-1] > 180.0:
                rk -= 360.0
            while rk - k_rolls[-1] < -180.0:
                rk += 360.0
        k_yaws.append(yk)
        k_pitches.append(pk)
        k_rolls.append(rk)

    # Use horizon-preserving PCHIP yaw/pitch/roll interpolation whenever camera isn't looking straight at zenith/nadir
    use_horizon_interp = max((abs(p) for p in k_pitches), default=0.0) < 80.0

    # Map each solved keyframe to its closest output frame index for clean timeline markers
    keyframe_At_out: dict[int, dict[str, Any]] = {}
    for ki, tk in enumerate(k_times):
        out_fi = max(0, min(n_out - 1, round(tk * eff_fps)))
        if out_fi not in keyframe_At_out:
            keyframe_At_out[out_fi] = {"key_index": ki, "image_name": k_names[ki], "pts_sec": round(tk, 4)}

    frames_out: list[dict[str, Any]] = []
    seg_i = 0
    prev_yaw: float | None = None
    prev_loc: Vec3 | None = None
    prev_q_out: QuatWXYZ | None = None
    dt_frame = 1.0 / eff_fps

    for fi in range(n_out):
        t_cur = fi * dt_frame if n_out > 1 else 0.0
        if n_k == 1 or t_cur <= k_times[0]:
            loc_f = k_locs[0]
            q_f = k_quats[0]
            intr_f = dict(k_intr[0])
        elif t_cur >= k_times[-1]:
            loc_f = k_locs[-1]
            q_f = k_quats[-1]
            intr_f = dict(k_intr[-1])
        else:
            while seg_i + 1 < n_k - 1 and k_times[seg_i + 1] < t_cur:
                seg_i += 1
            t0, t1 = k_times[seg_i], k_times[seg_i + 1]
            u = (t_cur - t0) / max(1e-8, t1 - t0)
            i_m1 = max(0, seg_i - 1)
            i_p2 = min(n_k - 1, seg_i + 2)
            p0 = k_locs[i_m1]
            p1 = k_locs[seg_i]
            p2 = k_locs[seg_i + 1]
            p3 = k_locs[i_p2]
            dt01 = k_times[seg_i] - k_times[i_m1] if seg_i > 0 else (t1 - t0)
            dt12 = t1 - t0
            dt23 = k_times[i_p2] - k_times[seg_i + 1] if (seg_i + 1) < (n_k - 1) else (t1 - t0)
            loc_f = centripetal_catmull_rom_vec3(p0, p1, p2, p3, u, dt01, dt12, dt23)
            if use_horizon_interp:
                yaw_i = pchip_monotone_scalar(k_yaws[i_m1], k_yaws[seg_i], k_yaws[seg_i + 1], k_yaws[i_p2], u, dt01, dt12, dt23)
                pit_i = pchip_monotone_scalar(k_pitches[i_m1], k_pitches[seg_i], k_pitches[seg_i + 1], k_pitches[i_p2], u, dt01, dt12, dt23)
                rol_i = pchip_monotone_scalar(k_rolls[i_m1], k_rolls[seg_i], k_rolls[seg_i + 1], k_rolls[i_p2], u, dt01, dt12, dt23)
                q_f = euler_yaw_pitch_roll_to_blender_quat(yaw_i, pit_i, rol_i)
            else:
                q_f = quat_slerp(k_quats[seg_i], k_quats[seg_i + 1], u)
            i0, i1 = k_intr[seg_i], k_intr[seg_i + 1]
            intr_f = {
                k: i0[k] + u * (i1[k] - i0[k])
                for k in ("fx", "fy", "cx", "cy", "lens_mm", "sensor_width_mm")
            }

        if prev_q_out is not None and (
            prev_q_out[0] * q_f[0] + prev_q_out[1] * q_f[1] + prev_q_out[2] * q_f[2] + prev_q_out[3] * q_f[3]
        ) < 0.0:
            q_f = (-q_f[0], -q_f[1], -q_f[2], -q_f[3])

        angles = camera_orientation_angles_deg(q_f)
        yaw = angles["yaw_deg"]
        if prev_yaw is not None:
            while yaw - prev_yaw > 180.0:
                yaw -= 360.0
            while yaw - prev_yaw < -180.0:
                yaw += 360.0
        prev_yaw = yaw

        if prev_loc is not None:
            speed = vec3_norm(vec3_sub(loc_f, prev_loc)) / dt_frame
        else:
            speed = 0.0
        if prev_q_out is not None:
            dq_dot = min(1.0, abs(
                prev_q_out[0] * q_f[0] + prev_q_out[1] * q_f[1] + prev_q_out[2] * q_f[2] + prev_q_out[3] * q_f[3]
            ))
            ang_vel = math.degrees(2.0 * math.acos(dq_dot)) / dt_frame
        else:
            ang_vel = 0.0

        prev_loc = loc_f
        prev_q_out = q_f

        bl_cam_tmp = {
            "location": list(loc_f),
            "rotation_quaternion": list(q_f),
            "fx": intr_f["fx"],
            "fy": intr_f["fy"],
            "cx": intr_f["cx"],
            "cy": intr_f["cy"],
            "width": out_w,
            "height": out_h,
        }
        cv_w2c = blender_camera_to_opencv_w2c(bl_cam_tmp, target_width=out_w, target_height=out_h)
        kf_meta = keyframe_At_out.get(fi)

        frames_out.append({
            "frame": fi + 1,
            "pts_sec": round(t_cur, 4),
            "is_keyframe": kf_meta is not None,
            "keyframe_index": kf_meta["key_index"] if kf_meta else None,
            "image_name": kf_meta["image_name"] if kf_meta else None,
            "location": [round(loc_f[0], 5), round(loc_f[1], 5), round(loc_f[2], 5)],
            "rotation_quaternion": [round(q_f[0], 6), round(q_f[1], 6), round(q_f[2], 6), round(q_f[3], 6)],
            "forward": angles["forward"],
            "right": angles["right"],
            "up": angles["up"],
            "yaw_deg": round(yaw, 2),
            "pitch_deg": round(angles["pitch_deg"], 2),
            "roll_deg": round(angles["roll_deg"], 2),
            "speed_m_s": round(speed, 4),
            "ang_vel_deg_s": round(ang_vel, 2),
            "lens_mm": round(intr_f["lens_mm"], 3),
            "sensor_width_mm": round(intr_f["sensor_width_mm"], 2),
            "width": out_w,
            "height": out_h,
            "fx": round(cv_w2c["fx"], 4),
            "fy": round(cv_w2c["fy"], 4),
            "cx": round(cv_w2c["cx"], 4),
            "cy": round(cv_w2c["cy"], 4),
            "R_w2c": cv_w2c["R_w2c"],
            "t_w2c": cv_w2c["t_w2c"],
        })

    # Path summary bounds & total travel distance
    xs = [f["location"][0] for f in frames_out]
    ys = [f["location"][1] for f in frames_out]
    zs = [f["location"][2] for f in frames_out]
    total_dist = sum(
        vec3_norm(vec3_sub(frames_out[i]["location"], frames_out[i - 1]["location"]))
        for i in range(1, len(frames_out))
    )

    return {
        "job_id": cameras_doc.get("job_id", ""),
        "fps": round(eff_fps, 4),
        "duration_sec": round((n_out - 1) / eff_fps if n_out > 1 else eff_dur, 4),
        "total_frames": n_out,
        "solved_keyframes": n_k,
        "width": out_w,
        "height": out_h,
        "pose_backend": cameras_doc.get("pose_backend", "vggt"),
        "path_length_m": round(total_dist, 4),
        "bounds_min": [round(min(xs), 4), round(min(ys), 4), round(min(zs), 4)],
        "bounds_max": [round(max(xs), 4), round(max(ys), 4), round(max(zs), 4)],
        "keyframe_markers": [
            {"frame": fi + 1, **meta}
            for fi, meta in sorted(keyframe_At_out.items())
        ],
        "frames": frames_out,
    }


def build_stereo_peer_table(viewmats_w2c: Any, active_indices: list[int]) -> dict[int, list[int]]:
    """Pair each OpenCV W2C camera (`[M, 4, 4]`) with overlapping wide-baseline stereo peers (`12°..85°`).

    Used by `core_worker` multi-view mini-batching so any near-plane floater spawned in front of
    Camera A immediately projects across Camera B's frustum on the very same backward pass.
    """
    import numpy as _np

    vm = _np.asarray(viewmats_w2c, dtype=_np.float64)
    n_c = len(vm)
    centers = _np.zeros((n_c, 3), dtype=_np.float64)
    fwds = _np.zeros((n_c, 3), dtype=_np.float64)
    for i in range(n_c):
        r_i = vm[i, :3, :3]
        t_i = vm[i, :3, 3]
        centers[i] = -r_i.T @ t_i
        f_i = r_i[2, :]
        fwds[i] = f_i / max(1e-8, float(_np.linalg.norm(f_i)))

    active_set = set(active_indices)
    look_pts = centers + fwds * 2.2
    med_span = max(0.5, float(_np.percentile(_np.linalg.norm(centers - centers.mean(axis=0), axis=1), 75)))

    peers_map: dict[int, list[int]] = {}
    for i in active_indices:
        scored: list[tuple[float, int]] = []
        for j in active_indices:
            if j == i or j not in active_set:
                continue
            baseline = float(_np.linalg.norm(centers[j] - centers[i]))
            look_dist = float(_np.linalg.norm(look_pts[j] - look_pts[i]))
            cos_ang = float(_np.clip(_np.dot(fwds[i], fwds[j]), -1.0, 1.0))
            if cos_ang > 0.05 and baseline > 0.04 * med_span:
                ang_score = 1.0 - abs(cos_ang - 0.72)
                prox_score = 1.0 / (1.0 + look_dist / med_span)
                scored.append((ang_score * prox_score, j))
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored:
            peers_map[i] = [idx_j for _, idx_j in scored[:8]]
        else:
            peers_map[i] = [j for j in active_indices if j != i][:8]
    return peers_map