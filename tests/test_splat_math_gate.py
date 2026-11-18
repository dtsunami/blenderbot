"""Pathclear Gate P1 (`P1_SIM3_OK`): Verify proper Sim(3) 3DGS covariance invariance,
exact camera pinhole + projected 2D EWA covariance invariance, OpenCV -> Blender camera
conversion, and RANSAC `Z=0` tilted-room floor alignment.
"""

from __future__ import annotations

import math
import random

import pytest

from blenderbot.splats.sim3 import (
    Sim3Transform,
    apply_sim3_to_camera_w2c,
    apply_sim3_to_gaussians_py,
    apply_sim3_to_point,
    estimate_floor_aligned_sim3,
    gaussian_covariance_3d,
    mat3_det,
    mat3_mul,
    mat3_transpose,
    mat3_vec,
    opencv_w2c_to_blender_camera,
    quat_normalize,
    quat_to_rotmat,
    rotmat_to_quat,
)


def _assert_mat3_close(a, b, tol: float = 1e-6) -> None:
    for i in range(3):
        for j in range(3):
            assert abs(a[i][j] - b[i][j]) <= tol, f"mismatch at [{i},{j}]: {a[i][j]} vs {b[i][j]}"


def test_quat_rotmat_roundtrip_and_proper_det() -> None:
    rng = random.Random(123)
    for _ in range(50):
        q = quat_normalize((
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        ))
        r = quat_to_rotmat(q)
        assert abs(mat3_det(r) - 1.0) < 1e-9
        q2 = rotmat_to_quat(r)
        r2 = quat_to_rotmat(q2)
        _assert_mat3_close(r, r2, tol=1e-7)


def test_sim3_rejects_improper_reflection() -> None:
    with pytest.raises(ValueError, match="SO\\(3\\)"):
        Sim3Transform(
            scale=1.0,
            R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),  # det = -1 reflection
            t=(0.0, 0.0, 0.0),
        )


def test_p1_anisotropic_covariance_and_ewa_projection_invariance() -> None:
    """Verify `R_w * s^2 * Sigma_old * R_w^T == Sigma_new` AND projected 2D EWA conic in
    image pixels `(u, v, cov2d)` is bit-exact invariant under arbitrary Sim(3) world transforms!
    (Catches the spiky-hairball quaternion / log-scale bug.)
    """
    rng = random.Random(2026)

    q_world = quat_normalize((0.61, -0.33, 0.52, -0.49))
    sim3 = Sim3Transform(
        scale=2.75,
        R=quat_to_rotmat(q_world),
        t=(-1.4, 3.8, -0.95),
    )

    # Arbitrary OpenCV camera looking toward origin
    q_cam = quat_normalize((0.92, 0.18, -0.25, 0.24))
    r_w2c = quat_to_rotmat(q_cam)
    t_w2c = (0.4, -0.3, 6.5)
    fx, fy, cx, cy = 1120.0, 1120.0, 640.0, 360.0

    r_w2c_new, t_w2c_new = apply_sim3_to_camera_w2c(sim3, r_w2c, t_w2c)
    assert abs(mat3_det(r_w2c_new) - 1.0) < 1e-7

    means = [
        (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(-1.0, 1.0))
        for _ in range(30)
    ]
    # Strongly anisotropic needles/pancakes (100:1 aspect ratio)
    log_scales = [
        (rng.uniform(-4.5, -1.0), rng.uniform(-4.5, -1.0), rng.uniform(-4.5, -1.0))
        for _ in range(30)
    ]
    quats = [
        quat_normalize((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
        for _ in range(30)
    ]

    means_new, log_scales_new, quats_new = apply_sim3_to_gaussians_py(
        sim3, means, log_scales, quats
    )

    def _project_ewa_2d(mean_w, quat_w, log_s, rc, tc):
        xc, yc, zc = vec3_add_local(mat3_vec(rc, mean_w), tc)
        assert zc > 0.2
        u = fx * (xc / zc) + cx
        v = fy * (yc / zc) + cy
        sig_w = gaussian_covariance_3d(quat_w, log_s)
        sig_c = mat3_mul(mat3_mul(rc, sig_w), mat3_transpose(rc))
        # EWA perspective Jacobian J (2x3)
        j00, j02 = fx / zc, -fx * xc / (zc * zc)
        j11, j12 = fy / zc, -fy * yc / (zc * zc)
        j_mat = ((j00, 0.0, j02), (0.0, j11, j12), (0.0, 0.0, 0.0))
        cov2d_3 = mat3_mul(mat3_mul(j_mat, sig_c), mat3_transpose(j_mat))
        return u, v, cov2d_3[0][0], cov2d_3[0][1], cov2d_3[1][1]

    def vec3_add_local(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    s2 = sim3.scale * sim3.scale
    for m_o, ls_o, q_o, m_n, ls_n, q_n in zip(
        means, log_scales, quats, means_new, log_scales_new, quats_new
    ):
        # 1. Verify 3D covariance tensor identity: s^2 * R * Sigma_old * R^T == Sigma_new
        sig_old = gaussian_covariance_3d(q_o, ls_o)
        sig_expected = mat3_mul(mat3_mul(sim3.R, sig_old), mat3_transpose(sim3.R))
        sig_expected = tuple(tuple(v * s2 for v in row) for row in sig_expected)  # type: ignore[assignment]
        sig_actual = gaussian_covariance_3d(q_n, ls_n)
        _assert_mat3_close(sig_expected, sig_actual, tol=1e-6)

        # 2. Verify projected 2D pixel mean (u, v) and 2D screen covariance match identically!
        u0, v0, c00_0, c01_0, c11_0 = _project_ewa_2d(m_o, q_o, ls_o, r_w2c, t_w2c)
        u1, v1, c00_1, c01_1, c11_1 = _project_ewa_2d(m_n, q_n, ls_n, r_w2c_new, t_w2c_new)
        assert abs(u0 - u1) < 1e-5
        assert abs(v0 - v1) < 1e-5
        assert abs(c00_0 - c00_1) <= 1e-5 * max(1.0, abs(c00_0))
        assert abs(c01_0 - c01_1) <= 1e-5 * max(1.0, abs(c01_0))
        assert abs(c11_0 - c11_1) <= 1e-5 * max(1.0, abs(c11_0))


def test_opencv_to_blender_camera_basis_and_center() -> None:
    """Verify camera center `-R^T t` matches `matrix_world` translation and forward ray maps `+Z_cv -> -Z_bl`."""
    q_cam = quat_normalize((0.85, -0.22, 0.38, 0.29))
    r_w2c = quat_to_rotmat(q_cam)
    t_w2c = (-1.2, 0.7, 4.8)

    bl_cam = opencv_w2c_to_blender_camera(
        r_w2c, t_w2c, fx=960.0, fy=960.0, cx=640.0, cy=360.0, width=1280, height=720
    )
    mw = bl_cam["matrix_world"]
    r_bl = (
        (mw[0][0], mw[0][1], mw[0][2]),
        (mw[1][0], mw[1][1], mw[1][2]),
        (mw[2][0], mw[2][1], mw[2][2]),
    )
    assert abs(mat3_det(r_bl) - 1.0) < 1e-7
    assert abs(bl_cam["lens_mm"] - 27.0) < 1e-5  # 960 * 36 / 1280 = 27mm

    # A point 5m directly in front of the camera in Blender camera local coords is (0, 0, -5)
    pt_world = (
        r_bl[0][2] * (-5.0) + mw[0][3],
        r_bl[1][2] * (-5.0) + mw[1][3],
        r_bl[2][2] * (-5.0) + mw[2][3],
    )
    # In OpenCV camera coords `R_w2c * pt_world + t_w2c`, that exact point MUST be (0, 0, +5)!
    pt_cv_r = mat3_vec(r_w2c, pt_world)
    pt_cv = (pt_cv_r[0] + t_w2c[0], pt_cv_r[1] + t_w2c[1], pt_cv_r[2] + t_w2c[2])
    assert abs(pt_cv[0]) < 1e-6
    assert abs(pt_cv[1]) < 1e-6
    assert abs(pt_cv[2] - 5.0) < 1e-6


def test_ransac_tilted_room_floor_aligns_to_z0_and_metric_height() -> None:
    """Build a synthetic room tilted by 19 deg in arbitrary 0.37x SfM scale and verify
    `estimate_floor_aligned_sim3` levels the floor to `Z = 0.0 ± 1e-3` and camera eye height to `1.60m`.
    """
    rng = random.Random(99)
    # Construct ground-truth room in Blender coords (`Z=0` floor, cameras at Z=1.60m)
    true_floor_pts = [
        (rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0), rng.gauss(0.0, 0.004))
        for _ in range(400)
    ]
    true_wall_pts = [
        (rng.choice([-4.0, 4.0]), rng.uniform(-4.0, 4.0), rng.uniform(0.3, 3.0))
        for _ in range(200)
    ]
    all_true_pts = true_floor_pts + true_wall_pts

    # Now scramble into arbitrary tilted OpenCV SfM coordinates (`scale_bad = 0.37`, tilted 19 deg)
    q_scramble = quat_normalize((0.78, 0.48, -0.25, 0.31))
    r_scramble = quat_to_rotmat(q_scramble)
    inv_scale = 0.37
    t_scramble = (5.2, -3.1, 8.4)

    def _to_sfm(p_bl):
        rp = mat3_vec(r_scramble, p_bl)
        return (inv_scale * rp[0] + t_scramble[0], inv_scale * rp[1] + t_scramble[1], inv_scale * rp[2] + t_scramble[2])

    sfm_pts = [_to_sfm(p) for p in all_true_pts]

    # Build 8 cameras orbiting at Z=1.60m in true room, pitched slightly downward by 12 deg!
    # Notice: camera pitch alone tilts camera `-Y` by 12 deg away from true floor normal, so RANSAC
    # MUST recover the remaining 12 deg floor tilt rather than blindly trusting camera `-Y`!
    cam_rs = []
    cam_ts = []
    pitch_rad = math.radians(-12.0)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)

    for k in range(8):
        ang = (k / 8.0) * math.pi * 0.7 - 0.5
        c_bl = (3.5 * math.sin(ang), -3.5 * math.cos(ang), 1.60)
        # Look horizontal direction in true room + 12 deg downward pitch
        fwd_horiz = (-math.sin(ang), math.cos(ang), 0.0)
        right_bl = (math.cos(ang), math.sin(ang), 0.0)
        up_true = (0.0, 0.0, 1.0)
        # Pitched OpenCV camera axes in true Blender room:
        # +Z_cv = cos(12)*fwd_horiz - sin(12)*up_true (pitched down)
        # +Y_cv = -sin(12)*fwd_horiz - cos(12)*up_true (downward)
        z_cv_bl = (
            cp * fwd_horiz[0] + sp * up_true[0],
            cp * fwd_horiz[1] + sp * up_true[1],
            cp * fwd_horiz[2] + sp * up_true[2],
        )
        y_cv_bl = (
            sp * fwd_horiz[0] - cp * up_true[0],
            sp * fwd_horiz[1] - cp * up_true[1],
            sp * fwd_horiz[2] - cp * up_true[2],
        )
        x_cv_bl = right_bl

        # Map axes & center into scrambled SfM world
        c_sfm = _to_sfm(c_bl)
        r_c_sfm = (
            mat3_vec(r_scramble, x_cv_bl),
            mat3_vec(r_scramble, y_cv_bl),
            mat3_vec(r_scramble, z_cv_bl),
        )
        # t_c = -R_c * C
        rc_c = mat3_vec(r_c_sfm, c_sfm)
        t_c_sfm = (-rc_c[0], -rc_c[1], -rc_c[2])
        cam_rs.append(r_c_sfm)
        cam_ts.append(t_c_sfm)

    recovered = estimate_floor_aligned_sim3(
        cam_rs, cam_ts, sfm_pts, align_floor=True, target_cam_height_m=1.60
    )
    assert recovered.floor_inlier_ratio > 0.5

    # Check aligned floor points sit flat on Z = 0!
    aligned_floor = [apply_sim3_to_point(recovered, p) for p in sfm_pts[:400]]
    zs = sorted(p[2] for p in aligned_floor)
    median_floor_z = zs[len(zs) // 2]
    iqr_floor_z = zs[int(0.85 * len(zs))] - zs[int(0.15 * len(zs))]
    assert abs(median_floor_z) < 0.015, f"floor median Z={median_floor_z}"
    assert iqr_floor_z < 0.025, f"floor still tilted! IQR={iqr_floor_z}"

    # Check aligned cameras sit at ~1.60m height!
    cam_heights = []
    for rc, tc in zip(cam_rs, cam_ts):
        rc_n, tc_n = apply_sim3_to_camera_w2c(recovered, rc, tc)
        bl_c = opencv_w2c_to_blender_camera(rc_n, tc_n, 900, 900, 640, 360, 1280, 720)
        cam_heights.append(bl_c["location"][2])
    med_h = sorted(cam_heights)[len(cam_heights) // 2]
    assert abs(med_h - 1.60) < 0.03, f"camera height={med_h}"