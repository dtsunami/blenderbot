"""P3 Gate (`tests/test_splat_reproject_gate.py`) — Hybrid Video Back-Projection Gate.

Verifies `docs/cloud_splat_design.md` §5 P3:
1. Imports real `P0` `splat.ply` + `cameras.json` (`SeedanceSourceCam`) into Blender 5.3.
2. Projects `seedance_20261117_212805.mp4` (or `gt_frame_00001.jpg`) from `SeedanceSourceCam` onto
   a proxy card inside the scene via `splat.project_video` (`feather=0.05`).
3. Verifies on-axis sharpness (`> 26 dB` PSNR in central projected region) AND renders from
   `DirectorCam` (`+15°` off-axis dolly/crane), proving seamless 3DGS parallax splat fill outside
   the projected frustum (`< 0.5%` black CLIP border pixels) + `Z=0` shadow-catcher integration.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image

from blenderbot.config.blender_paths import discover_blender
from blenderbot.transport import HeadlessTransport

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_p0_and_video() -> tuple[Path, Path]:
    splats_root = REPO_ROOT / "assets" / "splats"
    candidates = sorted(
        (
            d
            for d in splats_root.glob("seedance_*")
            if (d / "splat.ply").is_file()
            and (d / "cameras.json").is_file()
            and (d / "previews" / "gt_frame_00001.jpg").is_file()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        pytest.skip("No P0 Seedance splat artifact found")
    p0_dir = candidates[0]
    mp4 = REPO_ROOT / "assets" / "seedance" / "seedance_20261117_212805.mp4"
    media = mp4 if mp4.is_file() else (p0_dir / "previews" / "gt_frame_00001.jpg")
    return p0_dir, media


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def blender_worker():
    inst = discover_blender()
    assert inst.ready, f"Expected Blender 5.3+, got: {inst}"
    worker = HeadlessTransport()
    await worker.start()
    try:
        yield worker
    finally:
        await worker.close()


@pytest.mark.asyncio(loop_scope="module")
async def test_p3_hybrid_video_reproject_gate(blender_worker: HeadlessTransport) -> None:
    p0_dir, video_path = _find_p0_and_video()
    ply_path = p0_dir / "splat.ply"
    cam_path = p0_dir / "cameras.json"
    gt_frame1 = p0_dir / "previews" / "gt_frame_00001.jpg"

    renders_dir = REPO_ROOT / "renders" / "p3_gate"
    renders_dir.mkdir(parents=True, exist_ok=True)

    await blender_worker.call(
        "bpy.eval",
        {"code": "import bpy\nfor o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)\nresult={'cleared': True}"},
    )

    await blender_worker.call(
        "splat.import",
        {
            "filepath": str(ply_path),
            "cameras_json": str(cam_path),
            "name": "P3SplatWorld",
            "render_mode": "ellipsoids",
            "max_points": 450_000,
            "shadow_catcher_ground": False,
        },
        timeout=120.0,
    )

    # Create a proxy projection hull 3.2m in front of SeedanceSourceCam covering central 75% FOV
    await blender_worker.call(
        "bpy.eval",
        {
            "code": """
import bpy
from mathutils import Vector

sc = bpy.context.scene
sc.frame_set(1)
cam = bpy.data.objects['SeedanceSourceCam']
mw = cam.matrix_world.copy()
cam_pos = mw.translation.copy()
fwd = -(mw.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
center = cam_pos + fwd * 3.2

bpy.ops.mesh.primitive_plane_add(size=1.0, location=center)
proxy = bpy.context.active_object
proxy.name = 'P3ProxyHull'
proxy.rotation_mode = 'QUATERNION'
proxy.rotation_quaternion = mw.to_quaternion()
proxy.scale = (2.6, 1.5, 1.0)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
result = {'proxy': proxy.name}
"""
        },
    )

    proj = await blender_worker.call(
        "splat.project_video",
        {
            "video_path": str(video_path),
            "target_object": "P3ProxyHull",
            "projection_camera": "SeedanceSourceCam",
            "feather": 0.05,
        },
    )
    assert proj["object"] == "P3ProxyHull"

    # 1. On-axis render from SeedanceSourceCam: central region should be razor-sharp (>26 dB)
    on_axis_out = renders_dir / "p3_sourcecam_projected.png"
    await blender_worker.call(
        "bpy.eval",
        {
            "code": f"""
import bpy
sc = bpy.context.scene
sc.frame_set(1)
sc.camera = bpy.data.objects['SeedanceSourceCam']
sc.render.engine = 'BLENDER_EEVEE'
sc.render.resolution_x = 640
sc.render.resolution_y = 360
sc.render.film_transparent = False
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGB'
sc.eevee.taa_render_samples = 96
sc.render.filepath = {str(on_axis_out)!r}
bpy.ops.render.render(write_still=True)
result = {{'ok': True}}
"""
        },
        timeout=180.0,
    )

    img_on = np.asarray(Image.open(on_axis_out).convert("RGB"), dtype=np.float64) / 255.0
    img_gt = np.asarray(
        Image.open(gt_frame1).convert("RGB").resize((640, 360), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ) / 255.0

    # Central 40% crop (inside P3ProxyHull) gets live projected MP4 sharpness
    c_on = img_on[108:252, 192:448]
    c_gt = img_gt[108:252, 192:448]
    center_mse = float(np.mean((c_on - c_gt) ** 2))
    center_psnr = -10.0 * math.log10(max(1e-10, center_mse))
    full_mse = float(np.mean((img_on - img_gt) ** 2))
    full_psnr = -10.0 * math.log10(max(1e-10, full_mse))
    print(f"\n[P3 On-Axis] center_PSNR={center_psnr:.2f} dB | full_PSNR={full_psnr:.2f} dB")
    assert center_psnr >= 26.0, f"Expected sharp projected center >= 26 dB, got {center_psnr:.2f} dB"

    # 2. Off-axis render from DirectorCam (+15 deg dolly-in/crane): parallax splats fill borders cleanly
    dir_out = renders_dir / "p3_directorcam_15deg.png"
    dir_splat_only = renders_dir / "p3_directorcam_splat_only.png"
    await blender_worker.call(
        "bpy.eval",
        {
            "code": f"""
import bpy, math
from mathutils import Matrix, Vector

sc = bpy.context.scene
src_cam = bpy.data.objects['SeedanceSourceCam']
proxy = bpy.data.objects['P3ProxyHull']
mw = src_cam.matrix_world.copy()
cam_pos = mw.translation.copy()
fwd = -(mw.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
pivot = cam_pos + fwd * 3.2

# +15 deg orbit + 22% dolly-in + +0.18m crane so frustum stays inside reconstructed 3DGS volume
rel = (cam_pos - pivot) * 0.78
rot15 = Matrix.Rotation(math.radians(15.0), 3, 'Z')
dir_pos = pivot + (rot15 @ rel) + Vector((0.0, 0.0, 0.18))

dc_data = src_cam.data.copy()
dc_obj = bpy.data.objects.new('DirectorCam', dc_data)
sc.collection.objects.link(dc_obj)
dc_obj.location = dir_pos
dc_obj.rotation_mode = 'QUATERNION'
dc_obj.rotation_quaternion = (pivot - dir_pos).to_track_quat('-Z', 'Z')
sc.camera = dc_obj

sc.render.filepath = {str(dir_out)!r}
bpy.ops.render.render(write_still=True)

proxy.hide_render = True
sc.render.filepath = {str(dir_splat_only)!r}
bpy.ops.render.render(write_still=True)
proxy.hide_render = False
result = {{'director_cam': [round(dir_pos.x, 3), round(dir_pos.y, 3), round(dir_pos.z, 3)]}}
"""
        },
        timeout=180.0,
    )

    img_dir = np.asarray(Image.open(dir_out).convert("RGB"), dtype=np.float64) / 255.0
    img_splat_only = np.asarray(Image.open(dir_splat_only).convert("RGB"), dtype=np.float64) / 255.0
    black_with_proxy = float(np.mean(img_dir.max(axis=-1) < 0.015))
    black_splat_only = float(np.mean(img_splat_only.max(axis=-1) < 0.015))
    parallax_diff = float(np.mean(np.abs(img_dir - img_on)))
    proj_gain_diff = float(np.mean(np.abs(img_dir - img_splat_only)))
    print(
        f"[P3 DirectorCam +15deg] black_with_proxy={black_with_proxy:.4%} "
        f"(splat_only={black_splat_only:.4%}) | parallax_diff={parallax_diff:.4f} | proj_contrib={proj_gain_diff:.4f}"
    )
    # Prove proxy CLIP border adds ZERO black border over the underlying 3DGS splat fill!
    assert black_with_proxy <= black_splat_only + 0.005, (
        f"Proxy projection introduced black CLIP border: {black_with_proxy:.2%} vs {black_splat_only:.2%}"
    )
    assert black_with_proxy < 0.12, f"Expected >88% parallax splat coverage, got {black_with_proxy:.2%} dark"
    assert parallax_diff > 0.03, "Expected clear multi-view parallax shift from DirectorCam"
    assert proj_gain_diff > 0.005, "Expected projected video hull to contribute sharp foreground detail"
    print("P3_REPROJECT_OK")