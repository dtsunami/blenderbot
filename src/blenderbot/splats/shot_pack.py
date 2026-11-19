"""FPS / Minecraft Camera Rail, Subject-Track, Voxel Blockout & Mule-Style Spatial Brief Exporter.

Converts gamer-authored 3D scene intent (`cam_nodes`, `subject_nodes`, `voxels`, `pings`,
`crop_box`, `first_frame_b64` / `last_frame_b64`) into a self-contained Shot Collateral Pack
(`assets/shots/<shot_id>/`):
  - `cameras.json` (Blender + OpenCV `gsplat` ready, with `video_meta` & `seedance_camera_prompt`)
  - `shot.usda` (ASCII OpenUSD 1.0 `Z`-up stage: animated `DirectorRailCam`, `SubjectActor` track,
    `Voxel_###` blockout cubes, and `Ping_##` spatial markers — imports natively into Blender 5.3 & UE5)
  - `blockout.glb` (Binary glTF 2.0 mesh of all snap-placed voxel blocks)
  - `first_frame.png`, `last_frame.png`, & `pings/ping_##.png` (clean perspective conditioning stills)
  - `brief.md` & `brief.json` (MuleCreate-inspired Spatial Agent Brief for `blenderbot` / `unrealbot`).
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .proxy_mesh import pack_glb_bytes
from .sim3 import (
    catmull_rom_vec3,
    interpolate_camera_trajectory,
    opencv_w2c_to_blender_camera,
    vec3_cross,
    vec3_norm,
    vec3_normalize,
    vec3_sub,
)


def make_shot_id(name: str = "rail") -> str:
    clean = "".join(c if (c.isalnum() or c in "_-") else "_" for c in (name or "rail"))[:24].strip("_")
    stamp = datetime.datetime.now().strftime("%H%M%S")
    h6 = hashlib.sha1(f"{clean}:{time.time()}".encode()).hexdigest()[:6]
    return f"{clean or 'rail'}_{stamp}_{h6}"


def _decode_data_png(raw_b64: str) -> bytes:
    s = raw_b64.strip()
    if "," in s and s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def _look_to_opencv_w2c(
    eye: tuple[float, float, float],
    forward: tuple[float, float, float],
    roll_deg: float = 0.0,
) -> tuple[list[list[float]], list[float]]:
    """Build orthonormal OpenCV `(R_w2c, t_w2c)` from Blender Z-up `eye`, `forward` look ray & `roll_deg`."""
    fwd = vec3_normalize(forward, fallback=(0.0, 1.0, 0.0))
    world_up = (0.0, 0.0, 1.0)
    r_raw = vec3_cross(fwd, world_up)
    if vec3_norm(r_raw) < 1e-5:
        r_raw = (1.0, 0.0, 0.0)
    right = vec3_normalize(r_raw)
    cam_up = vec3_normalize(vec3_cross(right, fwd))

    if abs(roll_deg) > 1e-4:
        rad = math.radians(roll_deg)
        cr, sr = math.cos(rad), math.sin(rad)
        r_rot = (
            right[0] * cr + cam_up[0] * sr,
            right[1] * cr + cam_up[1] * sr,
            right[2] * cr + cam_up[2] * sr,
        )
        u_rot = (
            -right[0] * sr + cam_up[0] * cr,
            -right[1] * sr + cam_up[1] * cr,
            -right[2] * sr + cam_up[2] * cr,
        )
        right = vec3_normalize(r_rot)
        cam_up = vec3_normalize(u_rot)

    # OpenCV camera axes: +X = right, +Y = -cam_up (down), +Z = fwd
    r_w2c = [
        [right[0], right[1], right[2]],
        [-cam_up[0], -cam_up[1], -cam_up[2]],
        [fwd[0], fwd[1], fwd[2]],
    ]
    t_w2c = [
        -(r_w2c[0][0] * eye[0] + r_w2c[0][1] * eye[1] + r_w2c[0][2] * eye[2]),
        -(r_w2c[1][0] * eye[0] + r_w2c[1][1] * eye[1] + r_w2c[1][2] * eye[2]),
        -(r_w2c[2][0] * eye[0] + r_w2c[2][1] * eye[1] + r_w2c[2][2] * eye[2]),
    ]
    return r_w2c, t_w2c


def _interpolate_subject_track(
    subject_nodes: list[dict[str, Any]],
    duration_sec: float,
    fps: float,
    total_frames: int,
) -> list[dict[str, Any]]:
    """Interpolate moving subject/player waypoints (`[x, y, z]`) across `1..total_frames`."""
    if not subject_nodes:
        return []
    n_s = len(subject_nodes)
    pts = [
        (float(s["position"][0]), float(s["position"][1]), float(s["position"][2]))
        for s in subject_nodes
    ]
    times = [
        float(s["pts_sec"])
        if s.get("pts_sec") is not None
        else (i / float(max(1, n_s - 1))) * duration_sec
        for i, s in enumerate(subject_nodes)
    ]
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + 1e-4

    out: list[dict[str, Any]] = []
    seg = 0
    dt = 1.0 / max(1.0, fps)
    for fi in range(total_frames):
        t = fi * dt if total_frames > 1 else 0.0
        if n_s == 1 or t <= times[0]:
            p = pts[0]
        elif t >= times[-1]:
            p = pts[-1]
        else:
            while seg + 1 < n_s - 1 and times[seg + 1] < t:
                seg += 1
            u = (t - times[seg]) / max(1e-6, times[seg + 1] - times[seg])
            p = catmull_rom_vec3(
                pts[max(0, seg - 1)],
                pts[seg],
                pts[seg + 1],
                pts[min(n_s - 1, seg + 2)],
                u,
            )
        out.append({"frame": fi + 1, "pts_sec": round(t, 4), "position": [round(p[0], 4), round(p[1], 4), round(p[2], 4)]})
    return out


def build_voxel_blockout_glb(voxels: list[dict[str, Any]], mesh_name: str = "VoxelBlockout") -> bytes:
    """Pack snap-placed Minecraft/Rust voxel blocks (`center`, `size`, `color`) into a single `.glb`."""
    unit_v = np.array([
        # +X
        [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5],
        # -X
        [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],
        # +Y
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
        # -Y
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5],
        # +Z
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        # -Z
        [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
    ], dtype=np.float32)

    unit_n = np.repeat(
        np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32),
        4,
        axis=0,
    )
    quad_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    unit_f = np.concatenate([quad_tris + 4 * f for f in range(6)], axis=0)

    verts_all = []
    nrms_all = []
    cols_all = []
    faces_all = []

    for idx, vx in enumerate(voxels):
        c = np.asarray(vx.get("center") or [0.0, 0.0, 0.5], dtype=np.float32)
        sz_raw = vx.get("size") or [1.0, 1.0, 1.0]
        sz = np.asarray(
            [sz_raw, sz_raw, sz_raw] if isinstance(sz_raw, (int, float)) else sz_raw,
            dtype=np.float32,
        )
        col_raw = vx.get("color") or [0.55, 0.64, 0.78, 1.0]
        rgba = np.array(
            [*col_raw[:3], col_raw[3] if len(col_raw) > 3 else 1.0],
            dtype=np.float32,
        )
        v_box = unit_v * sz[None, :] + c[None, :]
        verts_all.append(v_box)
        nrms_all.append(unit_n)
        cols_all.append(np.tile(rgba[None, :], (24, 1)))
        faces_all.append(unit_f + np.uint32(idx * 24))

    return pack_glb_bytes(
        np.concatenate(verts_all, axis=0),
        np.concatenate(nrms_all, axis=0),
        np.concatenate(cols_all, axis=0),
        np.concatenate(faces_all, axis=0),
        mesh_name=mesh_name,
    )


def _esc_usd(s: str) -> str:
    return str(s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def build_shot_usda_text(
    *,
    shot_id: str,
    traj: dict[str, Any],
    subject_frames: list[dict[str, Any]],
    voxels: list[dict[str, Any]],
    pings: list[dict[str, Any]],
    task: str = "",
) -> str:
    """Serialize an ASCII OpenUSD 1.0 stage (`shot.usda`) compatible with Blender 5.3 & Unreal 5.5+."""
    fps = float(traj["fps"])
    n_f = int(traj["total_frames"])
    w = int(traj["width"])
    h = int(traj["height"])
    sensor_w = 36.0
    sensor_h = sensor_w * (float(h) / float(max(1, w)))

    frames = traj["frames"]
    trans_ts = ", ".join(
        f"{f['frame']}: ({f['location'][0]:.5f}, {f['location'][1]:.5f}, {f['location'][2]:.5f})"
        for f in frames
    )
    orient_ts = ", ".join(
        f"{f['frame']}: ({f['rotation_quaternion'][0]:.6f}, {f['rotation_quaternion'][1]:.6f}, "
        f"{f['rotation_quaternion'][2]:.6f}, {f['rotation_quaternion'][3]:.6f})"
        for f in frames
    )
    focal_ts = ", ".join(f"{f['frame']}: {f['lens_mm']:.4f}" for f in frames)

    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Shot"',
        f'    doc = "{_esc_usd(task or shot_id)}"',
        "    startTimeCode = 1",
        f"    endTimeCode = {n_f}",
        f"    framesPerSecond = {fps:.4f}",
        f"    timeCodesPerSecond = {fps:.4f}",
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "Shot"',
        "{",
        '    def Scope "Cameras"',
        "    {",
        '        def Camera "DirectorRailCam"',
        "        {",
        f"            double3 xformOp:translate.timeSamples = {{{trans_ts}}}",
        f"            quatf xformOp:orient.timeSamples = {{{orient_ts}}}",
        '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
        f"            float focalLength.timeSamples = {{{focal_ts}}}",
        f"            float horizontalAperture = {sensor_w:.4f}",
        f"            float verticalAperture = {sensor_h:.4f}",
        "            float2 clippingRange = (0.02, 1000.0)",
        "        }",
        "    }",
    ]

    if subject_frames:
        sub_ts = ", ".join(
            f"{sf['frame']}: ({sf['position'][0]:.5f}, {sf['position'][1]:.5f}, {sf['position'][2]:.5f})"
            for sf in subject_frames
        )
        lines.extend([
            '    def Scope "SubjectTrack"',
            "    {",
            '        def Xform "SubjectActor"',
            "        {",
            f"            double3 xformOp:translate.timeSamples = {{{sub_ts}}}",
            '            uniform token[] xformOpOrder = ["xformOp:translate"]',
            "        }",
            "    }",
        ])

    if voxels:
        lines.append('    def Scope "Blockout"')
        lines.append("    {")
        for idx, vx in enumerate(voxels):
            c = vx.get("center") or [0.0, 0.0, 0.5]
            sz_raw = vx.get("size") or [1.0, 1.0, 1.0]
            sz = [sz_raw, sz_raw, sz_raw] if isinstance(sz_raw, (int, float)) else list(sz_raw)
            lines.extend([
                f'        def Cube "Voxel_{idx + 1:03d}"',
                "        {",
                "            double size = 1.0",
                f"            double3 xformOp:translate = ({float(c[0]):.4f}, {float(c[1]):.4f}, {float(c[2]):.4f})",
                f"            float3 xformOp:scale = ({float(sz[0]):.4f}, {float(sz[1]):.4f}, {float(sz[2]):.4f})",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                "        }",
            ])
        lines.append("    }")

    if pings:
        lines.append('    def Scope "Pings"')
        lines.append("    {")
        for idx, pg in enumerate(pings):
            pos = pg.get("position") or [0.0, 0.0, 0.0]
            note = _esc_usd(str(pg.get("note") or f"Ping #{idx + 1}"))
            lines.extend([
                f'        def Xform "Ping_{idx + 1:02d}"',
                "        {",
                f'            custom string brief:note = "{note}"',
                f"            double3 xformOp:translate = ({float(pos[0]):.4f}, {float(pos[1]):.4f}, {float(pos[2]):.4f})",
                '            uniform token[] xformOpOrder = ["xformOp:translate"]',
                "        }",
            ])
        lines.append("    }")

    lines.append("}")
    return "\n".join(lines) + "\n"


def export_shot_collateral_pack(
    *,
    shot_dir: Path,
    shot_id: str,
    cam_nodes: list[dict[str, Any]],
    subject_nodes: list[dict[str, Any]] | None = None,
    track_subject: bool = False,
    voxels: list[dict[str, Any]] | None = None,
    pings: list[dict[str, Any]] | None = None,
    splat_job_id: str | None = None,
    task: str = "",
    duration_sec: float = 5.0,
    fps: float = 24.0,
    width: int = 1280,
    height: int = 720,
    grounded_walk: bool = False,
    first_frame_b64: str | None = None,
    last_frame_b64: str | None = None,
) -> dict[str, Any]:
    """Write a complete FPS Camera Rail + Subject-Track + Voxel + Spatial Brief Pack to `shot_dir`."""
    if not cam_nodes:
        raise ValueError("Place at least 1 Camera Rail Node (`[1] + LMB`) before exporting a shot pack.")

    shot_dir.mkdir(parents=True, exist_ok=True)
    dur = max(0.5, min(60.0, float(duration_sec)))
    eff_fps = max(1.0, min(120.0, float(fps)))
    total_frames = max(2, round(dur * eff_fps)) if len(cam_nodes) > 1 else 1
    out_w = max(64, int(width) - (int(width) % 2))
    out_h = max(64, int(height) - (int(height) % 2))

    sub_nodes = list(subject_nodes or [])
    sub_frames = _interpolate_subject_track(sub_nodes, dur, eff_fps, total_frames)

    n_c = len(cam_nodes)
    bl_cams: list[dict[str, Any]] = []
    sensor_w = 36.0

    for i, cn in enumerate(cam_nodes):
        pts_s = (
            float(cn["pts_sec"])
            if cn.get("pts_sec") is not None
            else ((i / float(max(1, n_c - 1))) * dur if n_c > 1 else 0.0)
        )
        t_frame = max(1, min(total_frames, round(pts_s * eff_fps) + 1))
        eye = (float(cn["position"][0]), float(cn["position"][1]), float(cn["position"][2]))

        if track_subject and sub_frames:
            sub_pt = sub_frames[min(len(sub_frames) - 1, t_frame - 1)]["position"]
            # Aim at subject chest/head height (+1.1m above foot waypoint)
            aim_target = (sub_pt[0], sub_pt[1], sub_pt[2] + 1.1)
            fwd = vec3_normalize(vec3_sub(aim_target, eye), fallback=(0.0, 1.0, 0.0))
        elif cn.get("target"):
            tgt = (float(cn["target"][0]), float(cn["target"][1]), float(cn["target"][2]))
            fwd = vec3_normalize(vec3_sub(tgt, eye), fallback=(0.0, 1.0, 0.0))
        else:
            raw_f = cn.get("forward") or [0.0, 1.0, 0.0]
            fwd = vec3_normalize((float(raw_f[0]), float(raw_f[1]), float(raw_f[2])), fallback=(0.0, 1.0, 0.0))

        roll_d = float(cn.get("roll_deg", 0.0))
        lens_mm = max(10.0, min(300.0, float(cn.get("lens_mm", 35.0))))
        fx = lens_mm * float(out_w) / sensor_w if out_w >= out_h else lens_mm * float(out_h) / sensor_w
        fy = fx
        cx, cy = 0.5 * out_w, 0.5 * out_h

        r_w2c, t_w2c = _look_to_opencv_w2c(eye, fwd, roll_deg=roll_d)
        bl = opencv_w2c_to_blender_camera(r_w2c, t_w2c, fx, fy, cx, cy, out_w, out_h, sensor_width_mm=sensor_w)
        bl["frame_index"] = i
        bl["pts_sec"] = round(pts_s, 4)
        bl["timeline_frame"] = t_frame
        bl["image_name"] = f"rail_node_{i + 1:03d}.png"
        bl["R_w2c"] = r_w2c
        bl["t_w2c"] = t_w2c
        bl_cams.append(bl)

    # Compile Seedance 2.5 Optical + Landmark Subject-Prediction Prompt!
    temp_doc = {
        "job_id": shot_id,
        "coordinate_system": "BLENDER_RIGHT_HANDED_Z_UP",
        "pose_backend": "fps_rail",
        "video_meta": {"fps": eff_fps, "duration_sec": dur, "total_frames": total_frames, "width": out_w, "height": out_h},
        "cameras": bl_cams,
    }
    traj = interpolate_camera_trajectory(temp_doc)
    f0 = traj["frames"][0]
    f_last = traj["frames"][-1]
    yaw_arc = f_last["yaw_deg"] - f0["yaw_deg"]
    pitch_arc = f_last["pitch_deg"] - f0["pitch_deg"]
    dist_m = float(traj["path_length_m"])
    rig_label = "grounded 1.65m eye-level Steadicam walk" if grounded_walk else "smooth 3D crane/drone rail"
    lens_0, lens_1 = f0["lens_mm"], f_last["lens_mm"]
    lens_txt = (
        f"Locked {round(lens_0)}mm prime lens with zero optical zoom"
        if abs(lens_1 - lens_0) < 0.8
        else f"Lens pull from {round(lens_0)}mm to {round(lens_1)}mm"
    )
    orbit_dir = "clockwise" if yaw_arc > 2.0 else "counter-clockwise" if yaw_arc < -2.0 else "steady heading"
    cam_clause = (
        f"{lens_txt}, {rig_label} traveling {dist_m:.2f}m ({orbit_dir} {abs(yaw_arc):.1f}° yaw, "
        f"{pitch_arc:+.1f}° pitch)"
        if dist_m >= 0.05
        else f"{lens_txt}, stationary tripod pan ({yaw_arc:+.1f}° yaw, {pitch_arc:+.1f}° pitch)"
    )

    sub_clause = ""
    if len(sub_frames) >= 2:
        s0 = sub_frames[0]["position"]
        s1 = sub_frames[-1]["position"]
        s_dist = vec3_norm(vec3_sub(s1, s0))
        s_spd = s_dist / max(0.1, dur)
        track_lbl = ", camera actively tracking subject" if track_subject else ""
        sub_clause = (
            f" Primary subject moves {s_dist:.2f}m ({s_spd:.2f} m/s) from ({s0[0]:.1f}, {s0[1]:.1f}, {s0[2]:.1f})m "
            f"to ({s1[0]:.1f}, {s1[1]:.1f}, {s1[2]:.1f})m relative to scene landmarks{track_lbl}."
        )

    optical_prompt = (
        f"{cam_clause}, strong multi-view motion parallax, consistent lighting, zero motion blur.{sub_clause}"
    ).strip()

    temp_doc["seedance_camera_prompt"] = optical_prompt
    cams_path = shot_dir / "cameras.json"
    cams_path.write_text(json.dumps(temp_doc, indent=2), encoding="utf-8")
    (shot_dir / "poses_timeline.json").write_text(json.dumps(traj), encoding="utf-8")

    # Save clean perspective conditioning PNGs if captured from WebGL2 canvas
    first_png_path: Path | None = None
    last_png_path: Path | None = None
    if first_frame_b64:
        first_png_path = shot_dir / "first_frame.png"
        first_png_path.write_bytes(_decode_data_png(first_frame_b64))
    if last_frame_b64:
        last_png_path = shot_dir / "last_frame.png"
        last_png_path.write_bytes(_decode_data_png(last_frame_b64))

    vx_list = list(voxels or [])
    glb_path: Path | None = None
    if vx_list:
        glb_path = shot_dir / "blockout.glb"
        glb_path.write_bytes(build_voxel_blockout_glb(vx_list, mesh_name=f"{shot_id}_Blockout"))

    ping_list = list(pings or [])
    saved_pings: list[dict[str, Any]] = []
    if ping_list:
        pings_dir = shot_dir / "pings"
        pings_dir.mkdir(exist_ok=True)
        for idx, pg in enumerate(ping_list):
            p_entry = {
                "id": f"P{idx + 1}",
                "position": [round(float(x), 4) for x in (pg.get("position") or [0.0, 0.0, 0.0])],
                "camera_eye": [round(float(x), 4) for x in (pg.get("camera_eye") or [0.0, -3.0, 1.65])],
                "note": str(pg.get("note") or f"Marker #{idx + 1}"),
                "image": None,
            }
            if pg.get("image_b64"):
                img_rel = f"pings/ping_{idx + 1:02d}.png"
                (shot_dir / img_rel).write_bytes(_decode_data_png(str(pg["image_b64"])))
                p_entry["image"] = img_rel
            saved_pings.append(p_entry)

    # Write OpenUSD stage (`shot.usda`)
    usda_path = shot_dir / "shot.usda"
    usda_path.write_text(
        build_shot_usda_text(
            shot_id=shot_id,
            traj=traj,
            subject_frames=sub_frames,
            voxels=vx_list,
            pings=saved_pings,
            task=task,
        ),
        encoding="utf-8",
    )

    # Write MuleCreate-style Spatial Brief (`brief.md` + `brief.json`)
    md_lines = [
        f"# Spatial Shot Brief — `{shot_id}`",
        "",
        f"- **Task / Intent:** {task or 'Execute camera rail, blockout massing, and spatial markers in 3D scene.'}",
        "- **Coordinate System:** Blender Right-Handed `+Z` Up (`1.0 = 1 meter`, floor `Z = 0`)",
        f"- **Source Splat Scene:** `{splat_job_id or 'none (pure blockout)'}`",
        f"- **OpenUSD Stage:** `shot.usda` (`DirectorRailCam`, {total_frames} frames @ {eff_fps:g} fps, {dur:g}s)",
        f"- **Optical Camera Directive:** {optical_prompt}",
        "",
        f"## 1. Camera Rail (`cameras.json` — {len(cam_nodes)} placed nodes -> {total_frames} frames)",
    ]
    for i, bc in enumerate(bl_cams):
        loc = bc["location"]
        md_lines.append(
            f"- **Node {i + 1}** (`t={bc['pts_sec']}s`, `f={bc['timeline_frame']}`): "
            f"`XYZ=({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})m`, `{bc['lens_mm']:.0f}mm`"
        )

    if sub_nodes:
        md_lines.extend(["", f"## 2. Subject / Actor Landmark Trajectory ({len(sub_nodes)} waypoints)"])
        for i, sn in enumerate(sub_nodes):
            sp = sn["position"]
            md_lines.append(f"- **Waypoint {i + 1}**: `XYZ=({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})m`")

    if vx_list:
        md_lines.extend(["", f"## 3. Snap-Grid Voxel Massing (`blockout.glb` — {len(vx_list)} blocks)"])
        for i, vx in enumerate(vx_list[:24]):
            c = vx.get("center") or [0, 0, 0.5]
            sz = vx.get("size") or [1, 1, 1]
            md_lines.append(f"- `Voxel_{i + 1:03d}`: center `({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})m`, size `{sz}`")

    if saved_pings:
        md_lines.extend(["", "## 4. Raycast Spatial Agent Pings (Mule-Style Notes)"])
        for p_item in saved_pings:
            pos = p_item["position"]
            img_ref = f" (`{p_item['image']}`)" if p_item["image"] else ""
            md_lines.append(
                f"- **`[{p_item['id']}]` at `({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m`{img_ref}:** {p_item['note']}"
            )

    brief_md_path = shot_dir / "brief.md"
    brief_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (shot_dir / "optical_prompt.txt").write_text(optical_prompt + "\n", encoding="utf-8")

    brief_json = {
        "shot_id": shot_id,
        "task": task,
        "splat_job_id": splat_job_id,
        "fps": eff_fps,
        "duration_sec": dur,
        "total_frames": total_frames,
        "width": out_w,
        "height": out_h,
        "track_subject": track_subject,
        "grounded_walk": grounded_walk,
        "optical_prompt": optical_prompt,
        "cam_nodes_count": len(cam_nodes),
        "subject_nodes_count": len(sub_nodes),
        "voxels_count": len(vx_list),
        "pings": saved_pings,
        "files": {
            "cameras_json": "cameras.json",
            "usda": "shot.usda",
            "blockout_glb": "blockout.glb" if glb_path else None,
            "first_frame": "first_frame.png" if first_png_path else None,
            "last_frame": "last_frame.png" if last_png_path else None,
            "brief_md": "brief.md",
        },
    }
    (shot_dir / "brief.json").write_text(json.dumps(brief_json, indent=2), encoding="utf-8")
    return brief_json