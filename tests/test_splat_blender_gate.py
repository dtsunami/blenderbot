"""P2 Gate (`tests/test_splat_blender_gate.py`) — Blender 5.3 Native Splat Render Gate.

Verifies `docs/cloud_splat_design.md` §5 P2:
1. Imports real `P0` `splat.ply` (710k Gaussians -> 500k sorted) + `cameras.json` into headless
   Blender 5.3 via vectorized `numpy` + `foreach_set` (`foreach_ms < 150 ms`).
2. Renders Frame 1 from `SeedanceSourceCam` in both `BLENDER_EEVEE` (`DITHERED` alpha) and
   `CYCLES` (`transparent_max_bounces=1024`) and checks PSNR / SSIM against Seedance Frame 1
   (`>= 22 dB`, zero black alpha-bounce blotches).
3. Places a 3D hero cube on `Z=0` over `ShadowGround` (`is_shadow_catcher=True`), renders a
   `30°` off-axis novel view + raytraced shadow composite (`P2_BLENDER_OK`).
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


def _find_p0_artifact_dir() -> Path:
    splats_root = REPO_ROOT / "assets" / "splats"
    candidates = sorted(
        (
            d
            for d in splats_root.glob("seedance_*")
            if (d / "splat.ply").is_file()
            and (d / "cameras.json").is_file()
            and (d / "previews" / "gt_frame_00001.jpg").is_file()
        ),
        key=lambda p: (not p.name.startswith("seedance_20261117_212805_"), -p.stat().st_mtime),
    )
    if not candidates:
        pytest.skip("No P0 Seedance splat artifact found in assets/splats/")
    return candidates[0]


def _psnr_and_ssim(render_path: Path, gt_path: Path) -> tuple[float, float, float]:
    """Return `(psnr_db, ssim_global, black_pixel_ratio)` comparing `render_path` to `gt_path`."""
    img_r = Image.open(render_path).convert("RGB")
    img_g = Image.open(gt_path).convert("RGB")
    if img_g.size != img_r.size:
        img_g = img_g.resize(img_r.size, Image.Resampling.BILINEAR)

    r = np.asarray(img_r, dtype=np.float64) / 255.0
    g = np.asarray(img_g, dtype=np.float64) / 255.0

    mse = float(np.mean((r - g) ** 2))
    psnr = 99.0 if mse <= 1e-10 else float(-10.0 * math.log10(mse))

    h, w, _ = r.shape
    bh, bw = (h // 8) * 8, (w // 8) * 8
    rb = r[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 64, 3)
    gb = g[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 64, 3)

    mu_x = rb.mean(axis=1)
    mu_y = gb.mean(axis=1)
    var_x = rb.var(axis=1)
    var_y = gb.var(axis=1)
    cov_xy = ((rb - mu_x[:, None, :]) * (gb - mu_y[:, None, :])).mean(axis=1)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    )
    ssim_val = float(ssim_map.mean())

    r_max = r.max(axis=-1)
    g_lum = g.mean(axis=-1)
    bright_gt = g_lum > 0.15
    black_blotch_ratio = float(np.mean(r_max[bright_gt] < 0.015)) if np.any(bright_gt) else 0.0
    return psnr, ssim_val, black_blotch_ratio


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
async def test_p2_blender_native_splat_render_gate(blender_worker: HeadlessTransport) -> None:
    p0_dir = _find_p0_artifact_dir()
    ply_path = p0_dir / "splat.ply"
    cam_path = p0_dir / "cameras.json"
    gt_frame1 = p0_dir / "previews" / "gt_frame_00001.jpg"

    renders_dir = REPO_ROOT / "renders" / "p2_gate"
    renders_dir.mkdir(parents=True, exist_ok=True)

    await blender_worker.call(
        "bpy.eval",
        {"code": "import bpy\nfor o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)\nresult={'cleared': True}"},
    )

    imp = await blender_worker.call(
        "splat.import",
        {
            "filepath": str(ply_path),
            "cameras_json": str(cam_path),
            "name": "P0SeedanceSplat",
            "render_mode": "ellipsoids",
            "max_points": 500_000,
            "min_opacity": 0.01,
            "ellipsoid_radius": 2.5,
            "alpha_mode": "DITHERED",
            "unlit": True,
            "shadow_catcher_ground": False,
        },
        timeout=120.0,
    )
    print(f"\n[P2 Import] {imp}")
    assert imp["foreach_ms"] < 150.0, f"Expected foreach_set < 150 ms, got {imp['foreach_ms']} ms"
    assert imp["cameras"]["imported"] is True

    # 1. Render Frame 1 in BLENDER_EEVEE (`DITHERED` stochastic OIT, 128 TAA samples)
    eevee_out = renders_dir / "p2_eevee_frame1.png"
    await blender_worker.call(
        "bpy.eval",
        {
            "code": f"""
import bpy
sc = bpy.context.scene
sc.frame_set(1)
sc.render.engine = 'BLENDER_EEVEE'
sc.render.resolution_x = 640
sc.render.resolution_y = 360
sc.render.resolution_percentage = 100
sc.render.film_transparent = False
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGB'
sc.eevee.taa_render_samples = 128
sc.render.filepath = {str(eevee_out)!r}
bpy.ops.render.render(write_still=True)
result = {{'rendered': str({str(eevee_out)!r})}}
"""
        },
        timeout=180.0,
    )
    eevee_psnr, eevee_ssim, eevee_blotch = _psnr_and_ssim(eevee_out, gt_frame1)
    print(
        f"[P2 EEVEE]   PSNR={eevee_psnr:.2f} dB | SSIM={eevee_ssim:.4f} | black_blotch={eevee_blotch:.4%}"
    )

    # 2. Render Frame 1 in CYCLES (`transparent_max_bounces=1024`, OPTIX GPU)
    cycles_out = renders_dir / "p2_cycles_frame1.png"
    cycles_info = await blender_worker.call(
        "bpy.eval",
        {
            "code": f"""
import bpy
sc = bpy.context.scene
sc.frame_set(1)
sc.render.engine = 'CYCLES'
sc.cycles.samples = 32
sc.cycles.use_denoising = False
sc.cycles.transparent_max_bounces = 1024
sc.cycles.max_bounces = 4
used_dev = 'CPU'
try:
    cprefs = bpy.context.preferences.addons['cycles'].preferences
    for btype in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
        try:
            cprefs.compute_device_type = btype
            cprefs.get_devices()
            gpus = [d for d in cprefs.devices if d.type != 'CPU']
            if gpus:
                for d in cprefs.devices:
                    d.use = (d.type != 'CPU')
                sc.cycles.device = 'GPU'
                used_dev = f'GPU:{{btype}}'
                break
        except Exception:
            pass
except Exception:
    pass
sc.render.filepath = {str(cycles_out)!r}
bpy.ops.render.render(write_still=True)
result = {{'device': used_dev, 'transparent_max_bounces': sc.cycles.transparent_max_bounces}}
"""
        },
        timeout=240.0,
    )
    cycles_psnr, cycles_ssim, cycles_blotch = _psnr_and_ssim(cycles_out, gt_frame1)
    print(
        f"[P2 CYCLES]  PSNR={cycles_psnr:.2f} dB | SSIM={cycles_ssim:.4f} | "
        f"black_blotch={cycles_blotch:.4%} | info={cycles_info}"
    )

    assert eevee_blotch < 0.01, f"EEVEE black alpha blotches detected: {eevee_blotch:.2%}"
    assert cycles_blotch < 0.01, f"Cycles black bounce blotches detected: {cycles_blotch:.2%}"
    assert eevee_psnr >= 22.0, f"Expected EEVEE PSNR >= 22.0 dB, got {eevee_psnr:.2f} dB"
    assert cycles_psnr >= 22.0, f"Expected Cycles PSNR >= 22.0 dB, got {cycles_psnr:.2f} dB"

    # 3. 30-degree off-axis novel view + Z=0 shadow-catcher hero cube composite
    novel_splat_bg = renders_dir / "p2_novel_30deg_splat_bg.png"
    novel_fg_rgba = renders_dir / "p2_novel_30deg_fg_rgba.png"
    novel_comp_out = renders_dir / "p2_novel_30deg_hero_shadow.png"

    novel_res = await blender_worker.call(
        "bpy.eval",
        {
            "code": f"""
import bpy, math
from mathutils import Matrix, Vector

sc = bpy.context.scene
splat_obj = bpy.data.objects['P0SeedanceSplat']
src_cam = bpy.data.objects['SeedanceSourceCam']
mw = src_cam.matrix_world.copy()
cam_pos = mw.translation.copy()
fwd = -(mw.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()

pivot = cam_pos + fwd * 3.5
pivot_floor = Vector((pivot.x, pivot.y, 0.0))

# Create novel camera orbited +30 deg around pivot Z-axis, looking from above Z=0
rel = cam_pos - pivot_floor
rot30 = Matrix.Rotation(math.radians(30.0), 3, 'Z')
novel_pos = pivot_floor + (rot30 @ rel)
novel_pos.z = max(novel_pos.z, 1.35)

nc_data = src_cam.data.copy()
nc_obj = bpy.data.objects.new('P2Novel30Cam', nc_data)
sc.collection.objects.link(nc_obj)
nc_obj.location = novel_pos
look_dir = (pivot_floor + Vector((0.0, 0.0, 0.25))) - novel_pos
nc_obj.rotation_mode = 'QUATERNION'
nc_obj.rotation_quaternion = look_dir.to_track_quat('-Z', 'Z')
sc.camera = nc_obj

# Pass A: 30-deg novel view of 3DGS splat plate
sc.render.engine = 'CYCLES'
sc.cycles.samples = 24
sc.render.film_transparent = False
sc.render.image_settings.color_mode = 'RGB'
sc.render.filepath = {str(novel_splat_bg)!r}
bpy.ops.render.render(write_still=True)

# Pass B: 3D Hero Cube + Z=0 Shadow Catcher ground (RGBA with raytraced Cycles shadow)
splat_obj.hide_render = True
bpy.ops.mesh.primitive_plane_add(size=8.0, location=(pivot_floor.x, pivot_floor.y, 0.0))
sg = bpy.context.active_object
sg.name = 'P2ShadowGround'
sg.is_shadow_catcher = True

cube_sz = 0.55
bpy.ops.mesh.primitive_cube_add(size=cube_sz, location=(pivot_floor.x, pivot_floor.y, cube_sz * 0.5))
hero = bpy.context.active_object
hero.name = 'P2HeroCube'
mat = bpy.data.materials.new('HeroCoralMat')
mat.use_nodes = True
bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
bsdf.inputs['Base Color'].default_value = (0.92, 0.24, 0.18, 1.0)
bsdf.inputs['Roughness'].default_value = 0.35
hero.data.materials.append(mat)

ld = bpy.data.lights.new('P2HeroSun', type='SUN')
ld.energy = 5.0
ld.angle = math.radians(4.0)
lo = bpy.data.objects.new('P2HeroSun', ld)
lo.rotation_euler = (math.radians(48.0), math.radians(20.0), math.radians(35.0))
sc.collection.objects.link(lo)

sc.render.film_transparent = True
sc.render.image_settings.color_mode = 'RGBA'
sc.cycles.samples = 32
sc.render.filepath = {str(novel_fg_rgba)!r}
bpy.ops.render.render(write_still=True)
splat_obj.hide_render = False

result = {{
    'pivot': [round(pivot_floor.x, 3), round(pivot_floor.y, 3), 0.0],
    'novel_cam': [round(novel_pos.x, 3), round(novel_pos.y, 3), round(novel_pos.z, 3)],
}}
"""
        },
        timeout=240.0,
    )

    bg_img = Image.open(novel_splat_bg).convert("RGBA")
    fg_img = Image.open(novel_fg_rgba).convert("RGBA")
    comp_img = Image.alpha_composite(bg_img, fg_img).convert("RGB")
    comp_img.save(novel_comp_out)

    fg_arr = np.asarray(fg_img, dtype=np.float64) / 255.0
    # Verify ShadowGround caught non-trivial semi-transparent shadow pixels (0.05 < alpha < 0.90)
    shadow_pixels = int(np.sum((fg_arr[:, :, 3] > 0.05) & (fg_arr[:, :, 3] < 0.90)))
    hero_pixels = int(np.sum(fg_arr[:, :, 3] >= 0.95))
    print(
        f"[P2 Novel 30deg] {novel_res} | hero_px={hero_pixels} | shadow_px={shadow_pixels} -> {novel_comp_out}"
    )
    assert hero_pixels > 200, f"Expected visible hero cube pixels, got {hero_pixels}"
    assert shadow_pixels > 100, f"Expected raytraced shadow-catcher pixels on Z=0, got {shadow_pixels}"
    print("P2_BLENDER_OK")