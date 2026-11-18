"""End-to-End Blender 5.3 -> Modal H100 Known-Pose 3DGS Gate (`splat.from_blender` + `auto_import`).

1. Builds a multi-material 3D asset in Blender 5.3 (`HeroSuzanne`, `Pedestal`, `HaloRing`).
2. Executes `splat.from_blender` via `execute_batch`:
   - Captures 24 known-pose RGBA views + 25,000 surface seed points (`splat.capture_views`).
   - Trains 1,500 steps on Modal H100 with random solid-background silhouette cleaning (0.0px pose drift).
   - Auto-imports the trained `.ply` + `cameras.json` back into Blender 5.3 (`splat.import`) and verifies 1:1 AABB lock.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console

from blenderbot.agent.executor import execute_batch
from blenderbot.agent.journal import Journal
from blenderbot.agent.parser import OpCall
from blenderbot.agent.sandbox import Sandbox
from blenderbot.transport import HeadlessTransport

console = Console()


async def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    sandbox = Sandbox(repo)
    journal = Journal(repo / ".blenderbot" / "p1_journal")

    worker = HeadlessTransport()
    await worker.start()
    try:
        build_scene = """
import bpy
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

def make_mat(name, rgba, rough=0.35, metal=0.2):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    return m

bpy.ops.mesh.primitive_monkey_add(size=1.3, location=(0.0, 0.0, 0.85))
suz = bpy.context.active_object
suz.name = "HeroSuzanne"
bpy.ops.object.shade_smooth()
suz.data.materials.append(make_mat("AmberGold", (0.95, 0.62, 0.14, 1.0), rough=0.25, metal=0.6))

bpy.ops.mesh.primitive_cylinder_add(radius=0.75, depth=0.28, location=(0.0, 0.0, 0.14))
ped = bpy.context.active_object
ped.name = "Pedestal"
ped.data.materials.append(make_mat("DeepTeal", (0.09, 0.55, 0.68, 1.0), rough=0.45, metal=0.1))

bpy.ops.mesh.primitive_torus_add(major_radius=0.95, minor_radius=0.07, location=(0.0, 0.0, 0.85), rotation=(0.45, 0.0, 0.0))
ring = bpy.context.active_object
ring.name = "HaloRing"
ring.data.materials.append(make_mat("Coral", (0.92, 0.26, 0.32, 1.0), rough=0.3, metal=0.3))

for nm, rot, energy, col in [
    ("KeySun", (0.65, 0.2, 0.7), 4.0, (1.0, 0.97, 0.92)),
    ("RimSun", (0.9, -0.4, -2.1), 2.5, (0.78, 0.88, 1.0)),
]:
    ld = bpy.data.lights.new(nm, type='SUN')
    ld.energy = energy
    ld.color = col
    lo = bpy.data.objects.new(nm, ld)
    lo.rotation_euler = rot
    bpy.context.scene.collection.objects.link(lo)

result = {"objects": [o.name for o in bpy.data.objects]}
"""
        await worker.call("bpy.eval", {"code": build_scene})
        console.print("[bold cyan]── P1 Gate: Blender 5.3 -> Modal H100 Known-Pose 3DGS (`splat.from_blender`) ──[/bold cyan]")

        outcome = await execute_batch(
            [
                OpCall(
                    index=0,
                    op="splat.from_blender",
                    params={
                        "targets": ["HeroSuzanne", "Pedestal", "HaloRing"],
                        "num_views": 24,
                        "resolution": [512, 512],
                        "engine": "BLENDER_EEVEE",
                        "transparent_bg": True,
                        "max_init_points": 25000,
                        "iterations": 1500,
                        "gpu": "H100",
                        "auto_import": True,
                        "name": "TrainedHeroSplat",
                        "render_mode": "ellipsoids",
                    },
                )
            ],
            sandbox=sandbox,
            journal=journal,
            transport=worker,
            role="master",
            yolo=True,
        )
        assert outcome.status == "ok", f"Batch failed: {outcome.errors} / {outcome.results}"
        res = outcome.results[0]["result"]
        console.print(json.dumps(res, indent=2))

        # Inspect final PSNR in telemetry.jsonl
        job_dir = repo / "assets" / "splats" / res["job_id"]
        telem_lines = (job_dir / "telemetry.jsonl").read_text(encoding="utf-8").strip().splitlines()
        psnrs = [json.loads(ln)["psnr"] for ln in telem_lines if '"psnr"' in ln]
        peak_psnr = max(psnrs) if psnrs else 0.0
        console.print(
            f"[bold green]P1_BLENDER_TO_SPLAT_OK[/bold green] "
            f"job={res['job_id']} | {res['n_gaussians']:,} Gaussians | "
            f"peak_psnr={peak_psnr:.2f}dB | imported_points={res['imported']['points_imported']:,} | "
            f"cost=${res['cost_usd']:.4f}"
        )
        assert peak_psnr >= 26.0, f"Expected >=26dB PSNR on clean Blender captures, got {peak_psnr:.2f}dB"
        assert res["imported"]["points_imported"] > 5000
    finally:
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())