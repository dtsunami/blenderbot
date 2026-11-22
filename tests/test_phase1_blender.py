"""End-to-end Phase 1 tests against real Blender 5.3.0 (`HeadlessTransport`, `bpy.tx`, `rig.*`, `scene.audit`, `render.playblast`)."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

from blenderbot.blender_ops import OpError, list_ops
from blenderbot.bridge.installer import install_blender_addon
from blenderbot.bridge.protocol import (
    BridgeDiscovery,
    read_discovery,
    verify_bridge_token,
    write_discovery,
)
from blenderbot.config.blender_paths import discover_blender
from blenderbot.transport import BridgeTransport, HeadlessTransport, TransportConnectionError


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


def test_discover_blender_and_addon_installer(tmp_path: Path) -> None:
    inst = discover_blender()
    assert inst.ready
    assert inst.blender_version is not None and inst.blender_version.startswith("5.")
    assert inst.ffmpeg_path is not None

    ops = list_ops()
    op_names = {o["name"] for o in ops}
    assert {
        "system.ping",
        "blender.status",
        "scene.passport",
        "scene.setup",
        "bpy.tx",
        "rig.curve_path",
        "rig.follow_path",
        "rig.camera_pursuit",
        "terrain.scatter",
        "scene.audit",
        "viewport.contact_sheet",
        "render.playblast",
    }.issubset(op_names)

    startup_dir = tmp_path / "startup"
    addon_dir = install_blender_addon(target_extensions_dir=tmp_path, target_startup_dir=startup_dir)
    assert (addon_dir / "blender_manifest.toml").is_file()
    assert (addon_dir / "__init__.py").is_file()
    assert (startup_dir / "blenderbot_bridge_startup.py").is_file()


@pytest.mark.asyncio(loop_scope="module")
async def test_auto_upgrade_headless_to_live_gui_bridge(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
) -> None:
    from blenderbot.session.bootstrap import bootstrap_session

    sess = await bootstrap_session(
        cwd=tmp_path,
        home=tmp_path / "home",
        injected_transport=None,
        auto_connect_blender=False,
    )
    try:
        # Simulate starting with a headless transport in `auto` mode before GUI Blender opens
        assert blender_worker._client is not None
        fake_headless = BridgeTransport(
            host=blender_worker._client.host,
            port=blender_worker._client.port,
            token=blender_worker._client.token,
            mode_label="headless",
        )
        sess.transport = fake_headless
        sess.agent.transport = fake_headless

        # No bridge.json yet -> stays headless
        assert await sess.maybe_upgrade_to_gui_bridge() is False
        assert sess.transport.mode == "headless"

        # Now GUI Blender writes `~/.blenderbot/bridge.json` -> hot-upgrades to `gui`!
        disc = BridgeDiscovery(
            host=blender_worker._client.host,
            port=blender_worker._client.port,
            token=blender_worker._client.token,
            pid=12345,
            started_at=200.0,
            mode="gui",
            blender_version="5.3.0",
        )
        write_discovery(disc, home=sess.cfg.home)
        assert await sess.maybe_upgrade_to_gui_bridge() is True
        assert sess.transport is not None
        assert sess.transport.mode == "gui"
    finally:
        await sess.close()


def test_bridge_discovery_and_token_auth(tmp_path: Path) -> None:
    disc = BridgeDiscovery(
        host="127.0.0.1",
        port=18766,
        token="secret_tok_123",
        pid=9999,
        started_at=100.0,
        mode="gui",
        blender_version="5.3.0",
    )
    write_discovery(disc, home=tmp_path)
    loaded = read_discovery(home=tmp_path)
    assert loaded == disc
    assert verify_bridge_token("secret_tok_123", "secret_tok_123")
    assert not verify_bridge_token("secret_tok_123", "wrong_token")


@pytest.mark.asyncio(loop_scope="module")
async def test_unauthorized_token_rejected(blender_worker: HeadlessTransport) -> None:
    assert blender_worker._client is not None
    bad_client = BridgeTransport(
        host=blender_worker._client.host,
        port=blender_worker._client.port,
        token="forged_bad_token",
    )
    try:
        with pytest.raises(TransportConnectionError, match="rejected token"):
            await bad_client.call("system.ping")
    finally:
        await bad_client.close()


@pytest.mark.asyncio(loop_scope="module")
async def test_bpy_tx_atomic_rollback_leaves_zero_orphans(blender_worker: HeadlessTransport) -> None:
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "scene.setup",
        {"fps": 30, "frame_start": 1, "frame_end": 60, "resolution": [1280, 720], "engine": "BLENDER_WORKBENCH"},
    )

    before = await blender_worker.call("blender.status")
    assert before["counts"]["objects"] == 0

    # Execute a `bpy.tx` script that creates objects/meshes/materials and then raises mid-script!
    failing_tx = """
col = ensure_collection("LEAK_TEST")
mesh = bpy.data.meshes.new("OrphanMesh")
obj = bpy.data.objects.new("OrphanObject", mesh)
col.objects.link(obj)
assign_solid_material(obj, "OrphanMat", (1.0, 0.0, 0.0, 1.0))
assert len(bpy.data.objects) == 1
raise ValueError("Intentional mid-tx failure!")
"""
    with pytest.raises(OpError, match="bpy.tx aborted.*Scene rolled back cleanly"):
        await blender_worker.call("bpy.tx", {"description": "broken tx", "code": failing_tx})

    after = await blender_worker.call("blender.status")
    assert after["counts"]["objects"] == 0
    assert after["counts"]["meshes"] == 0
    assert after["counts"]["materials"] == 0


@pytest.mark.asyncio(loop_scope="module")
async def test_full_automotive_previs_audit_sheet_and_playblast(
    blender_worker: HeadlessTransport,
    tmp_path: Path,
) -> None:
    """Build a miniature 60-frame mountain pursuit previs, verify `scene.audit` catches sunken car
    penetration, fix height, pass audit, render burned-in contact sheet, and render H.264 playblast!
    """
    await blender_worker.call("scene.clear")
    await blender_worker.call(
        "scene.setup",
        {
            "fps": 30,
            "frame_start": 1,
            "frame_end": 60,
            "resolution": [640, 360],
            "engine": "BLENDER_WORKBENCH",
            "sky_color": [0.55, 0.65, 0.78],
            "mist_range": [10.0, 250.0],
        },
    )

    # 1. Build ground terrain + red sports car + 4 wheels via transactional `bpy.tx`
    build_scene_code = """
env_col = ensure_collection("ENV")
veh_col = ensure_collection("VEHICLE")

# Flat + gently banked road corridor terrain
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=24, y_segments=48, size=120.0)
for v in bm.verts:
    # Gentle valley outside the 12m road corridor
    corridor = max(0.0, abs(v.co.x) - 8.0)
    v.co.z = 0.04 * (corridor ** 1.35)
road_mesh = bpy.data.meshes.new("Terrain_Mesh")
bm.to_mesh(road_mesh)
bm.free()
terrain = bpy.data.objects.new("ENV_Terrain", road_mesh)
link_exclusively(terrain, env_col)
assign_solid_material(terrain, "Mat_Terrain", (0.18, 0.28, 0.16, 1.0), roughness=0.85)

# Car body: 4.5m long x 2.0m wide x 1.0m tall, bottom of chassis at z = 0.20m when origin is at z=0
bm_car = bmesh.new()
bmesh.ops.create_cube(bm_car, size=1.0)
for v in bm_car.verts:
    v.co.x *= 2.0
    v.co.y *= 4.5
    v.co.z = (v.co.z + 0.5) * 0.9 + 0.22
car_mesh = bpy.data.meshes.new("Car_Mesh")
bm_car.to_mesh(car_mesh)
bm_car.free()
car = bpy.data.objects.new("CAR_Body", car_mesh)
link_exclusively(car, veh_col)
assign_solid_material(car, "Mat_SupercarRed", (0.82, 0.08, 0.06, 1.0), roughness=0.25, metallic=0.2)

# 4 wheels parented to CAR_Body (radius = 0.34m, bottom touches z = 0.0m when CAR_Body.z = 0.0)
wheel_names = []
for idx, (wx, wy) in enumerate([(-0.95, 1.35), (0.95, 1.35), (-0.95, -1.35), (0.95, -1.35)]):
    bm_w = bmesh.new()
    bmesh.ops.create_cone(bm_w, cap_ends=True, segments=16, radius1=0.34, radius2=0.34, depth=0.28)
    bmesh.ops.rotate(bm_w, verts=bm_w.verts, matrix=Matrix.Rotation(math.pi / 2.0, 3, 'Y'))
    w_mesh = bpy.data.meshes.new(f"Wheel_Mesh_{idx}")
    bm_w.to_mesh(w_mesh)
    bm_w.free()
    w_obj = bpy.data.objects.new(f"CAR_Wheel_{idx}", w_mesh)
    w_obj.parent = car
    w_obj.location = (wx, wy, 0.34)
    link_exclusively(w_obj, veh_col)
    assign_solid_material(w_obj, "Mat_Tire", (0.08, 0.08, 0.09, 1.0), roughness=0.7)
    wheel_names.append(w_obj.name)

result = {"terrain": terrain.name, "car": car.name, "wheels": wheel_names}
"""
    tx_res = await blender_worker.call(
        "bpy.tx",
        {"description": "Create valley terrain and red previs car with 4 wheels", "code": build_scene_code},
    )
    assert tx_res["committed"] is True
    wheels = tx_res["result"]["wheels"]

    # 2. Build winding road spline & scatter trees outside road corridor
    curve_res = await blender_worker.call(
        "rig.curve_path",
        {
            "name": "TRACK_Centerline",
            "points": [[-2.0, -70.0, 0.0], [1.5, -20.0, 0.0], [-1.5, 30.0, 0.0], [2.0, 80.0, 0.0]],
            "collection": "RIGS",
        },
    )
    assert curve_res["arc_length_m"] > 140.0

    scatter_res = await blender_worker.call(
        "terrain.scatter",
        {
            "terrain": "ENV_Terrain",
            "exclude_curve": "TRACK_Centerline",
            "exclude_margin_m": 9.0,
            "count": 30,
        },
    )
    assert scatter_res["placed"] > 10

    # 3. Deliberately sink car by -0.25m first to prove `scene.audit` catches ground penetration!
    await blender_worker.call(
        "rig.follow_path",
        {
            "object": "CAR_Body",
            "curve": "TRACK_Centerline",
            "start_dist_m": 10.0,
            "end_dist_m": 90.0,
            "z_offset_m": -0.25,
            "wheels": wheels,
            "wheel_radius_m": 0.34,
        },
    )
    bad_audit = await blender_worker.call(
        "scene.audit",
        {"subject": "CAR_Body", "ground": "ENV_Terrain", "step": 5},
    )
    assert bad_audit["passed"] is False
    assert any("penetrates ground" in issue for issue in bad_audit["issues"])

    # 4. Fix `z_offset_m` so chassis/wheels ride cleanly & add pursuit camera progression
    follow_ok = await blender_worker.call(
        "rig.follow_path",
        {
            "object": "CAR_Body",
            "curve": "TRACK_Centerline",
            "start_dist_m": 10.0,
            "end_dist_m": 90.0,
            "z_offset_m": -0.21,  # CAR_Body min_z is +0.22m in local space -> +10mm road clearance!
            "bank_gain_deg": 4.0,
            "wheels": wheels,
            "wheel_radius_m": 0.34,
        },
    )
    assert follow_ok["wheels_animated"] == 4

    cam_res = await blender_worker.call(
        "rig.camera_pursuit",
        {
            "name": "CAM_Main",
            "target": "CAR_Body",
            "beats": [
                {"frame": 1, "offset": [-6.0, -28.0, 9.0], "look_ahead": [0.0, 4.0, 0.8], "lens_mm": 26.0},
                {"frame": 35, "offset": [2.2, -9.5, 2.2], "look_ahead": [0.0, 2.0, 0.7], "lens_mm": 35.0},
                {"frame": 60, "offset": [2.8, -1.5, 1.1], "look_ahead": [0.0, 0.5, 0.6], "lens_mm": 50.0},
            ],
        },
    )
    assert cam_res["active"] is True

    # 5. Re-run `scene.audit`: must PASS 100% with rich 5-beat camera telemetry!
    good_audit = await blender_worker.call(
        "scene.audit",
        {"subject": "CAR_Body", "ground": "ENV_Terrain", "camera": "CAM_Main", "step": 2},
    )
    assert good_audit["passed"] is True, f"Audit failed unexpectedly: {good_audit}"
    assert good_audit["framing"]["in_frame_pct"] == 100.0
    assert good_audit["framing"]["cam_distance_m"]["start"] > good_audit["framing"]["cam_distance_m"]["end"]
    assert len(good_audit["framing"]["beats"]) >= 4

    # 6. Verify `scene.passport` captures Blender 5.3 slotted Action fcurves cleanly
    passport = await blender_worker.call("scene.passport", {"max_objects": 20})
    assert passport["active_camera"] == "CAM_Main"
    assert len(passport["cameras"]) == 1
    assert len(passport["cameras"][0]["animation"]) >= 2  # location + data.lens

    # 7. Render burned-in 3x2 Mosaic Contact Sheet PNG
    sheet_path = tmp_path / "pursuit_sheet.png"
    sheet = await blender_worker.call(
        "viewport.contact_sheet",
        {"output_path": str(sheet_path), "cols": 3, "rows": 2, "tile_width": 320},
    )
    assert sheet_path.is_file() and sheet["size_bytes"] > 5_000
    assert len(sheet["frames"]) == 6

    # 8. Render 60-frame H.264 `.mp4` Playblast and verify with `ffprobe`!
    mp4_path = tmp_path / "pursuit_playblast.mp4"
    pb = await blender_worker.call(
        "render.playblast",
        {"output_path": str(mp4_path), "resolution": [640, 360]},
    )
    assert mp4_path.is_file() and pb["size_bytes"] > 5_000
    assert pb["frames"] == 60

    inst = discover_blender()
    assert inst.ffprobe_path is not None
    probe = await asyncio.to_thread(
        subprocess.run,
        [
            inst.ffprobe_path,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(mp4_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    assert streams[0]["codec_name"] == "h264"
    assert int(streams[0]["width"]) == 640
    assert int(streams[0]["height"]) == 360
    assert int(streams[0]["nb_frames"]) == 60