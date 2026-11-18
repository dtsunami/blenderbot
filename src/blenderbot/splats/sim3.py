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