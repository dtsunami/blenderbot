"""End-to-end Phase 2 tests against live Blender 5.3 (`HeadlessTransport`).

Verifies modeling, transforms, modifiers, PBR materials, studio lighting, cameras, Blender 5.3
slotted-Action keyframing, F-curve modifiers, baking, armatures + skinning + posing, topology &
BVHTree collision audits, multi-format asset import/export (GLB/OBJ/STL + .blend append), and
headless EEVEE production rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from blenderbot.config.blender_paths import discover_blender
from blenderbot.transport import HeadlessTransport


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def blender_worker():
    inst = discover_blender()
    assert inst.ready, f"Expected Blender 5.3+ installed, got: {inst}"
    worker = HeadlessTransport()
    await worker.start()
    try:
        yield worker
    finally:
        await worker.close()


@pytest.mark.asyncio(loop_scope="module")
async def test_modeling_modifiers_inspect_and_topology_collision_audit(
    blender_worker: HeadlessTransport,
) -> None:
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "scene.setup",
        {
            "fps": 24,
            "frame_start": 1,
            "frame_end": 48,
            "resolution": [640, 360],
            "engine": "BLENDER_EEVEE",
        },
    )

    # Create pedestal cube, snap pivot to BOTTOM_CENTER, place cleanly on z=0
    ped = await blender_worker.call(
        "mesh.primitive",
        {
            "name": "Pedestal",
            "shape": "cylinder",
            "radius": 1.2,
            "depth": 0.4,
            "segments": 24,
            "location": [0.0, 0.0, 0.5],
            "collection": "SET",
            "color": [0.15, 0.16, 0.18],
            "roughness": 0.35,
            "smooth_shade": True,
        },
    )
    assert ped["name"] == "Pedestal"

    orig_res = await blender_worker.call(
        "object.origin",
        {"name": "Pedestal", "mode": "BOTTOM_CENTER"},
    )
    assert abs(orig_res["origin_world"][2] - 0.3) < 1e-3  # bottom of depth=0.4 centered at z=0.5 is z=0.3

    await blender_worker.call(
        "object.transform",
        {"name": "Pedestal", "location": [0.0, 0.0, 0.0]},
    )

    # Create hero torus sitting above pedestal (z = 0.4 .. 1.9), configure bevel/subsurf, apply one
    await blender_worker.call(
        "mesh.primitive",
        {
            "name": "HeroSculpt",
            "shape": "torus",
            "major_radius": 0.55,
            "minor_radius": 0.18,
            "location": [0.0, 0.0, 1.15],
            "rotation_deg": [90.0, 0.0, 25.0],
            "collection": "HERO",
        },
    )
    await blender_worker.call(
        "object.shade",
        {"name": "HeroSculpt", "smooth": True, "auto_smooth_deg": 35.0},
    )
    mod_res = await blender_worker.call(
        "modifier.configure",
        {
            "object": "HeroSculpt",
            "type": "SUBSURF",
            "name": "Subsurf",
            "settings": {"levels": 1, "render_levels": 1},
        },
    )
    assert mod_res["type"] == "SUBSURF"

    insp_before = await blender_worker.call("scene.inspect", {"names": ["Pedestal", "HeroSculpt"]})
    assert len(insp_before["objects"]) == 2
    hero_info = next(o for o in insp_before["objects"] if o["name"] == "HeroSculpt")
    base_polys = hero_info["mesh"]["polygons"]
    assert len(hero_info["modifiers"]) == 1

    apply_res = await blender_worker.call(
        "modifier.apply",
        {"object": "HeroSculpt", "modifier": "Subsurf"},
    )
    assert apply_res["polygons"] > base_polys
    assert apply_res["remaining_modifiers"] == []

    # Duplicate, parent, and test BVH collision detection before & after separating
    dup_res = await blender_worker.call(
        "object.duplicate",
        {"source": "Pedestal", "name": "ColliderBox", "location": [0.0, 0.0, 0.1]},
    )
    assert dup_res["name"] == "ColliderBox"

    col_audit = await blender_worker.call(
        "scene.audit",
        {"meshes": ["Pedestal", "ColliderBox"], "check_collisions": ["Pedestal", "ColliderBox"]},
    )
    assert col_audit["passed"] is False
    assert len(col_audit["collisions"]) == 1

    # Parent ColliderBox, then delete hierarchy child cleanly
    await blender_worker.call(
        "object.parent",
        {"child": "ColliderBox", "parent": "Pedestal", "keep_transform": True},
    )
    del_res = await blender_worker.call("object.delete", {"name": "ColliderBox"})
    assert "ColliderBox" in del_res["deleted"]

    clean_audit = await blender_worker.call(
        "scene.audit",
        {"meshes": ["Pedestal", "HeroSculpt"], "check_collisions": ["Pedestal", "HeroSculpt"]},
    )
    assert clean_audit["passed"] is True
    assert clean_audit["collisions"] == []


@pytest.mark.asyncio(loop_scope="module")
async def test_materials_lights_world_turntable_camera_and_eevee_renders(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
) -> None:
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "scene.setup",
        {
            "fps": 24,
            "frame_start": 1,
            "frame_end": 48,
            "resolution": [320, 180],
            "engine": "BLENDER_EEVEE",
        },
    )
    await blender_worker.call(
        "mesh.primitive",
        {
            "name": "HeroSculpt",
            "shape": "torus",
            "major_radius": 0.6,
            "minor_radius": 0.2,
            "location": [0.0, 0.0, 1.0],
            "smooth_shade": True,
        },
    )

    # Assign metallic gold material to HeroSculpt
    mat_res = await blender_worker.call(
        "material.upsert",
        {
            "name": "Mat_BrushedGold",
            "base_color": [0.94, 0.74, 0.31],
            "metallic": 0.92,
            "roughness": 0.22,
            "coat": 0.3,
            "assign_to": "HeroSculpt",
        },
    )
    assert mat_res["name"] == "Mat_BrushedGold"
    assert "HeroSculpt" in mat_res["assigned_to"]

    mat_list = await blender_worker.call("material.list")
    gold_entry = next(m for m in mat_list["materials"] if m["name"] == "Mat_BrushedGold")
    assert gold_entry["metallic"] == pytest.approx(0.92, abs=1e-2)

    # Studio softbox rig + gradient world + rim spot light
    rig_res = await blender_worker.call(
        "light.rig",
        {"preset": "studio_softbox", "target": "HeroSculpt", "intensity": 1.2},
    )
    assert len(rig_res["lights"]) >= 3

    spot_res = await blender_worker.call(
        "light.upsert",
        {
            "name": "AccentSpot",
            "light_type": "SPOT",
            "energy": 300.0,
            "color": [0.4, 0.7, 1.0],
            "location": [-2.0, 2.0, 3.0],
            "look_at": "HeroSculpt",
            "spot_size_deg": 40.0,
        },
    )
    assert spot_res["name"] == "AccentSpot"

    world_res = await blender_worker.call(
        "world.setup",
        {
            "mode": "gradient",
            "color": [0.04, 0.05, 0.07],
            "zenith_color": [0.12, 0.16, 0.24],
            "strength": 0.9,
        },
    )
    assert world_res["mode"] == "gradient"

    # Turntable camera with DOF focused on HeroSculpt
    cam_res = await blender_worker.call(
        "camera.upsert",
        {
            "name": "CAM_Turntable",
            "lens_mm": 50.0,
            "dof_fstop": 2.8,
            "dof_focus_object": "HeroSculpt",
            "turntable": {
                "target": "HeroSculpt",
                "radius": 3.4,
                "height": 1.6,
                "revolutions": 1.0,
                "frame_start": 1,
                "frame_end": 48,
            },
            "set_active": True,
        },
    )
    assert cam_res["dof_enabled"] is True
    assert cam_res["turntable_frames"] == 48

    # Production headless EEVEE still render (transparent RGBA PNG)
    still_path = tmp_path / "eevee_hero_rgba.png"
    still = await blender_worker.call(
        "render.still",
        {
            "output_path": str(still_path),
            "engine": "BLENDER_EEVEE",
            "samples": 8,
            "film_transparent": True,
            "resolution": [320, 180],
            "frame": 12,
        },
    )
    assert still_path.is_file() and still["size_bytes"] > 2_000
    assert still["engine"] == "BLENDER_EEVEE"

    # Fast PNG sequence animation render (3 frames)
    seq_dir = tmp_path / "png_seq"
    seq_res = await blender_worker.call(
        "render.animation",
        {
            "output_path": str(seq_dir),
            "engine": "BLENDER_WORKBENCH",
            "format": "PNG_SEQ",
            "resolution": [320, 180],
            "frame_start": 1,
            "frame_end": 3,
        },
    )
    assert seq_res["frames"] == 3
    assert len(list(seq_dir.glob("*.png"))) == 3


@pytest.mark.asyncio(loop_scope="module")
async def test_slotted_actions_fcurves_baking_and_armature_skinning(
    blender_worker: HeadlessTransport,
) -> None:
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "scene.setup",
        {"fps": 24, "frame_start": 1, "frame_end": 24, "engine": "BLENDER_WORKBENCH"},
    )

    # 1. Keyframe a bouncing/spinning cube + F-curve modifiers + bake + clear
    await blender_worker.call(
        "mesh.primitive",
        {"name": "Spinner", "shape": "cube", "size": 1.0, "location": [0.0, 0.0, 0.5]},
    )
    kf_res = await blender_worker.call(
        "anim.keyframe",
        {
            "object": "Spinner",
            "tracks": [
                {
                    "property": "location",
                    "index": 2,
                    "interpolation": "BEZIER",
                    "keyframes": [{"frame": 1, "value": 0.5}, {"frame": 12, "value": 2.0}, {"frame": 24, "value": 0.5}],
                },
                {
                    "property": "rotation_deg",
                    "interpolation": "LINEAR",
                    "keyframes": [[1, [0.0, 0.0, 0.0]], [24, [0.0, 0.0, 360.0]]],
                },
            ],
        },
    )
    assert len(kf_res["tracks"]) == 2

    fc_mod = await blender_worker.call(
        "anim.fcurve_modifier",
        {"object": "Spinner", "property": "location", "type": "CYCLES"},
    )
    assert fc_mod["channels_modified"] >= 1

    bake_res = await blender_worker.call(
        "anim.bake",
        {"object": "Spinner", "frame_start": 1, "frame_end": 12, "step": 2},
    )
    assert bake_res["frames_baked"] >= 6

    clr_res = await blender_worker.call("anim.clear", {"object": "Spinner"})
    assert clr_res["removed_curves"] >= 1

    # 2. Build articulated robotic arm mesh + 3-bone Armature with smooth skinning + pose animation
    await blender_worker.call(
        "mesh.primitive",
        {
            "name": "ArmColumn",
            "shape": "cylinder",
            "radius": 0.25,
            "depth": 2.4,
            "segments": 16,
            "location": [0.0, 0.0, 1.2],
        },
    )
    arm_res = await blender_worker.call(
        "rig.armature_create",
        {
            "name": "RobotRig",
            "bones": [
                {"name": "Base", "head": [0.0, 0.0, 0.0], "tail": [0.0, 0.0, 0.8]},
                {"name": "LowerArm", "head": [0.0, 0.0, 0.8], "tail": [0.0, 0.0, 1.6], "parent": "Base", "connected": True},
                {"name": "UpperArm", "head": [0.0, 0.0, 1.6], "tail": [0.0, 0.0, 2.4], "parent": "LowerArm", "connected": True},
            ],
            "bind_meshes": ["ArmColumn"],
        },
    )
    assert arm_res["bones_count"] == 3
    assert arm_res["bound_meshes"] == ["ArmColumn"]

    # Pose at frame 1 (rest) and frame 24 (bent 45 degrees each) and verify world tail shifts!
    await blender_worker.call(
        "rig.pose",
        {
            "armature": "RobotRig",
            "frame": 1,
            "poses": {
                "LowerArm": {"rotation_deg": [0.0, 0.0, 0.0]},
                "UpperArm": {"rotation_deg": [0.0, 0.0, 0.0]},
            },
        },
    )
    rest_insp = await blender_worker.call("rig.armature_inspect", {"name": "RobotRig"})
    upper_rest = next(b for b in rest_insp["bones"] if b["name"] == "UpperArm")
    assert abs(upper_rest["world_tail"][0]) < 1e-3
    assert upper_rest["world_tail"][2] == pytest.approx(2.4, abs=1e-2)

    pose_res = await blender_worker.call(
        "rig.pose",
        {
            "armature": "RobotRig",
            "frame": 24,
            "poses": {
                "LowerArm": {"rotation_deg": [45.0, 0.0, 0.0]},
                "UpperArm": {"rotation_deg": [35.0, 0.0, 0.0]},
            },
        },
    )
    assert pose_res["keyframed"] is True
    posed_insp = await blender_worker.call("rig.armature_inspect", {"name": "RobotRig"})
    upper_posed = next(b for b in posed_insp["bones"] if b["name"] == "UpperArm")
    # Bending vertical (+Z) bones around local X pitches them in YZ plane and lowers Z below 2.4
    assert abs(upper_posed["world_tail"][1]) > 0.4
    assert upper_posed["world_tail"][2] < 2.25


@pytest.mark.asyncio(loop_scope="module")
async def test_multi_format_asset_io_blend_append_and_snapshots(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
) -> None:
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "mesh.primitive",
        {
            "name": "CrateAsset",
            "shape": "cube",
            "dimensions": [1.4, 1.0, 0.8],
            "location": [0.0, 0.0, 0.4],
            "color": [0.7, 0.45, 0.2],
        },
    )

    # Save library .blend + export GLB, OBJ, and STL
    lib_blend = tmp_path / "crate_library.blend"
    save_res = await blender_worker.call("scene.save", {"filepath": str(lib_blend)})
    assert lib_blend.is_file() and save_res["size_bytes"] > 10_000

    glb_path = tmp_path / "crate.glb"
    obj_path = tmp_path / "crate.obj"
    stl_path = tmp_path / "crate.stl"
    for p in (glb_path, obj_path, stl_path):
        exp = await blender_worker.call("asset.export", {"filepath": str(p), "objects": ["CrateAsset"]})
        assert p.is_file() and exp["size_bytes"] > 50

    # Clear scene, take snapshot, import GLB normalized to target_size=2.8 on ground z=0
    await blender_worker.call("scene.clear")
    snap_empty = await blender_worker.call("scene.snapshot_save", {"dir": str(tmp_path), "prefix": "empty"})

    imp_glb = await blender_worker.call(
        "asset.import",
        {
            "filepath": str(glb_path),
            "collection": "IMPORTED_GLB",
            "target_size": 2.8,
            "location": [3.0, -1.0, 0.0],
            "place_on_ground": True,
        },
    )
    assert imp_glb["count"] >= 1
    assert max(imp_glb["world_aabb"]["size"]) == pytest.approx(2.8, abs=1e-2)
    assert imp_glb["world_aabb"]["min"][2] == pytest.approx(0.0, abs=1e-2)

    # Append CrateAsset from library .blend into same scene
    app_res = await blender_worker.call(
        "asset.blend_append",
        {
            "filepath": str(lib_blend),
            "objects": ["CrateAsset"],
            "target_collection": "APPENDED",
        },
    )
    assert len(app_res["objects"]) == 1

    # Restore empty snapshot and verify objects were rolled back cleanly
    await blender_worker.call("scene.snapshot_restore", {"snapshot_path": snap_empty["snapshot_path"]})
    status_after = await blender_worker.call("blender.status")
    assert status_after["counts"]["objects"] == 0