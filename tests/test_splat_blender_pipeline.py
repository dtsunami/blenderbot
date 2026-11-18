"""Tests for Blender 5.3 <-> Known-Pose Cloud 3DGS (`splat.capture_views`, `splat.import`, `splat.crop_clean`, harness wiring)."""

from __future__ import annotations

import json
import math
import struct
import tarfile
from pathlib import Path

import pytest
import pytest_asyncio

from blenderbot.agent.executor import plan_batch
from blenderbot.agent.parser import OpCall
from blenderbot.agent.roles import is_op_allowed_for_role
from blenderbot.agent.sandbox import Sandbox
from blenderbot.blender_ops import list_ops
from blenderbot.config.blender_paths import discover_blender
from blenderbot.splats.sim3 import opencv_w2c_to_blender_camera
from blenderbot.transport import HeadlessTransport


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


def test_splat_ops_registered_and_role_gated(tmp_path: Path) -> None:
    op_names = {o["name"] for o in list_ops()}
    assert {
        "splat.capture_views",
        "splat.import",
        "splat.crop_clean",
        "splat.project_video",
    }.issubset(op_names)

    assert is_op_allowed_for_role("viewer", "splat.capture_views", "read") is True
    assert is_op_allowed_for_role("viewer", "splat.import", "mutate") is False
    assert is_op_allowed_for_role("viewer", "splat.from_blender", None) is False
    assert is_op_allowed_for_role("builder", "splat.from_blender", None) is True
    assert is_op_allowed_for_role("builder", "splat.from_video", None) is True
    assert is_op_allowed_for_role("builder", "splat.resume", None) is True

    sb = Sandbox(tmp_path)
    errs = plan_batch(
        [OpCall(index=0, op="splat.from_blender", params={"num_views": 24})],
        sandbox=sb,
        has_transport=True,
        role="builder",
    )
    assert errs == []

    errs_no_bl = plan_batch(
        [OpCall(index=0, op="splat.from_blender", params={"num_views": 24})],
        sandbox=sb,
        has_transport=False,
        role="builder",
    )
    assert len(errs_no_bl) == 1


@pytest.mark.asyncio(loop_scope="module")
async def test_capture_views_pose_math_and_ply_roundtrip(
    blender_worker: HeadlessTransport, tmp_path: Path
) -> None:
    build_code = """
import bpy
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

def make_mat(name, rgba):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = rgba
    return m

bpy.ops.mesh.primitive_monkey_add(size=1.2, location=(0.0, 0.0, 0.7))
suz = bpy.context.active_object
suz.name = "HeroSuzanne"
suz.data.materials.append(make_mat("Gold", (0.95, 0.65, 0.15, 1.0)))

bpy.ops.mesh.primitive_cylinder_add(radius=0.7, depth=0.2, location=(0.0, 0.0, 0.1))
ped = bpy.context.active_object
ped.name = "Pedestal"
ped.data.materials.append(make_mat("Teal", (0.10, 0.60, 0.72, 1.0)))

ld = bpy.data.lights.new("SunKey", type='SUN')
ld.energy = 3.0
lo = bpy.data.objects.new("SunKey", ld)
lo.rotation_euler = (0.6, 0.2, 0.8)
bpy.context.scene.collection.objects.link(lo)
result = {"ok": True}
"""
    await blender_worker.call("bpy.eval", {"code": build_code})

    bundle_path = tmp_path / "hero_views.tar.gz"
    cap = await blender_worker.call(
        "splat.capture_views",
        {
            "output_path": str(bundle_path),
            "targets": ["HeroSuzanne", "Pedestal"],
            "num_views": 12,
            "resolution": [256, 256],
            "engine": "BLENDER_WORKBENCH",
            "transparent_bg": True,
            "max_init_points": 5000,
        },
        timeout=120.0,
    )
    assert cap["views"] == 12
    assert cap["transparent_bg"] is True
    assert cap["init_points"] == 5000
    assert bundle_path.is_file()

    with tarfile.open(bundle_path, "r:gz") as tf:
        names = set(tf.getnames())
        assert "manifest.json" in names
        assert "init_points.bin" in names
        assert "view_0000.png" in names
        assert "view_0011.png" in names

        manifest = json.loads(tf.extractfile("manifest.json").read().decode("utf-8"))  # type: ignore[union-attr]
        pts_raw = tf.extractfile("init_points.bin").read()  # type: ignore[union-attr]

    (n_pts,) = struct.unpack_from("<I", pts_raw, 0)
    assert n_pts == 5000
    zs = [struct.unpack_from("<3f3B", pts_raw, 4 + i * 15)[2] for i in range(n_pts)]
    assert min(zs) >= -0.01
    assert max(zs) <= 1.5

    cx, cy, cz = (float(c) for c in manifest["center"])
    for v in manifest["views"]:
        bl_cam = opencv_w2c_to_blender_camera(
            v["R_w2c"], v["t_w2c"], v["fx"], v["fy"], v["cx"], v["cy"], v["width"], v["height"]
        )
        lx, ly, lz = bl_cam["location"]
        dist = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2 + (lz - cz) ** 2)
        assert dist > 1.5

    ply_path = tmp_path / "test_splat.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    ).encode("ascii")
    verts_bin = b"".join(
        struct.pack(
            "<14f",
            float(i) * 0.5, 0.0, 0.2,
            0.5, 0.1, -0.2,
            2.0,
            -4.0, -4.0, -4.0,
            1.0, 0.0, 0.0, 0.0,
        )
        for i in range(4)
    )
    ply_path.write_bytes(header + verts_bin)

    imp = await blender_worker.call(
        "splat.import",
        {
            "filepath": str(ply_path),
            "name": "TestSplatEllipsoids",
            "render_mode": "ellipsoids",
            "shadow_catcher_ground": True,
        },
    )
    assert imp["points_imported"] == 4
    assert imp["render_mode"] == "ellipsoids"
    assert imp["shadow_catcher"] == "TestSplatEllipsoids_ShadowGround"

    crop = await blender_worker.call(
        "splat.crop_clean",
        {
            "object_name": "TestSplatEllipsoids",
            "bounds_min": [-1.0, -1.0, -1.0],
            "bounds_max": [1.1, 1.0, 1.0],
        },
    )
    assert crop["kept"] == 3
    assert crop["removed"] == 1