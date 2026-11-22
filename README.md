# blenderbot

**Blender 5.3+ AI 3D Creation Harness, Seedream 5.5 × Seedance 2.5 × Meshy 7.1 ↔ Modal H100 3D Gaussian Splatting (`gsplat` + `VGGT-1B` + `COLMAP` + `Depth Pro 2`) & Full-Screen Minecraft FPS Splat Editor.**

`blenderbot` pairs a multi-provider terminal REPL + Session-First Web 3D Studio with a zero-dependency Blender 5.3+ bridge, OpenRouter **Seedream 5.5** (`bytedance-seed/seedream-5-5`) image generation, **Seedance 2.5** (`bytedance/seedance-2.5`) video conditioning, **fal.ai Meshy 7.1** (`meshy/v7.1/image-to-3d`) `.glb` authoring, serverless **Modal H100 / L40S** 3D Gaussian Splat reconstruction & sharded Cloud Cycles rendering (`gsplat` + `VGGT-1B` + `COLMAP` + `ml-depth-pro-2` + `render.modal`), and a **Dedicated Full-Screen Minecraft FPS Camera & 3DGS Splat Editor (`/fps`)** for flying around 3DGS / `.glb` scenes, surgically culling & painting Gaussians at 60fps, syncing live cameras with Blender 5.3, and authoring Shot Collateral Packs.

---

## Core Capabilities

- **General Blender 5.3+ 3D Artist & TD Agent (`bl_ops` + transactional `bpy.tx` + `render.modal`):**
  - **45+ pure-stdlib domain ops** across modeling (`mesh.primitive`, `modifier.configure`), PBR shading (`material.upsert`), studio lighting (`light.rig`, `world.setup`), cameras (`camera.upsert`), Blender 5.3 Slotted Actions & layered NLA (`anim.*`, `rig.*`), multi-format USD/GLB/FBX/OBJ/PLY I/O (`asset.*`), topology/BVH collision/framing audits (`scene.audit`), vision self-verification (`viewport.render_still`, `viewport.contact_sheet`), and **sharded cloud Cycles/EEVEE GPU rendering (`render.modal`)** across Modal H100/L40S GPUs.
  - Dual transport: hot-attaches to a live GUI **Blender 5.3+** session (`BridgeTransport` on `127.0.0.1:18766`) or auto-spawns a warm background `blender.exe -b` worker (`HeadlessTransport`), with full `.blend` + filesystem undo/redo (`journal.undo` / `journal.redo`) and resumable Web Agent Sessions (`/api/sessions`).
- **Seedance 2.5 × Meshy 7.1 ↔ Cloud 3D Gaussian Splatting Loop (`splats/`, `media_gen/`):**
  - **Video → Floor-Aligned 3DGS (`splat.from_video`):** Motion-adaptive FFmpeg keyframe + PTS extraction, metric poses via **VGGT-1B** (sliding 40-frame Sim(3) windows), **COLMAP `PINHOLE`**, **`colmap_prior`**, or **`known_cameras`**, **Apple Depth Pro 2 (`ml-depth-pro-2`)** RANSAC-scaled surface seed nudging + dense wall supplementation + `RGB+ED` asymmetric free-space floater carving, iterative `Train <-> Video-Diff Arc Solve <-> Frame Promotion` cycles, swept camera-tunnel & near-frustum carving, and 1-click **`🎯 Snap Arc & Clean Floaters`**.
  - **Blender 5.3 → 3DGS (`splat.from_blender`):** Captures multi-view RGBA silhouettes + exact camera poses + surface seed points (`0.0px` pose drift, `>42 dB` PSNR in ~15s on H100).
  - **Native Blender 5.3 Splat Import (`splat.import`):** Strided `PointCloud` + Geometry Nodes anisotropic 3D ellipsoids (`Hybrid` 60fps viewport mode) + exact Mahalanobis ray-integral shader (`BLENDER_EEVEE` & `CYCLES`) + `L=1` Spherical Harmonics + `SeedanceSourceCam` trajectory rig.
- **3DGS Launcher Tooltips, Agent *Snicker-Test* Review & Multi-Pass Comparator:**
  - Rich hover tooltips (`META_PARAM_DOCS`) explaining the algorithm role and expected usage of **every** 3DGS Launcher, Resume/Surgery, and Strategy Arc parameter, plus a 1-click **`🧐 Review Parameters`** button so the Agent audits your config + source media against the snicker test before GPU spend.
  - **🎬 Compare & Path (All Render Passes):** Automatically versions every Modal 3DGS, `Snap Arc + Clean`, and Blender 5.3 render pass (`renders/passes_manifest.json`), letting you compare Original Video against **any** historical training/culling/snapping pass, FPS Rail render, sibling splat run, or checkpoint snapshot (`step_*.jpg` + `3×` error heatmap) in lockstep alongside the 3D trajectory & `Yaw° / Pitch° / Roll° / Speed` curves.
- **Rust / ARK / Minecraft FPS Camera-Rail & Spatial Brief Sandbox (`🎮 FPS Rail Sandbox`):**
  - Pointer-lock `WASD` + **`T` Auto-Run** + **`Hold Alt` Over-the-Shoulder Free-Look** (decoupled body movement vs. gimbal pan) + **`C` Crouch (`0.95m`) / `Z` Prone (`0.32m`)** + **`Q`/`E` Corner Peek** + **`Hold RMB` Landmark Target-Lock** + **`Scroll` Prime Lens (`18–135mm`)** + **`MMB` / `V` Tactical Pings** + **Top-Right Tactical Radar**.
  - 4-slot hotbar (`[1] Cam Node`, `[2] Subject Track`, `[3] Snap Voxel Block`, `[4] Spatial Ping`) exporting self-contained **Shot Collateral Packs** (`assets/shots/<shot_id>/`: `shot.usda`, `cameras.json`, `blockout.glb`, clean `first_frame.png`/`last_frame.png`, `optical_prompt.txt`, and `brief.md`) directly into Seedance 2.5, Modal H100, or Blender 5.3.

---

## Quick Start

### 1. Requirements & Install
- **Python 3.11+**, **Blender 5.3+**, and **FFmpeg** on `PATH` (or auto-discovered on Windows).
- Provider API key in `~/.blenderbot/.env` (automatically falls back to `~/.davai/.env` / `~/.davag/.env`): `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `OPENROUTER_API_KEY` (for Seedance 2.5).

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"

# Verify Blender 5.3+, FFmpeg, bridge add-on, Modal auth & API keys
blenderbot --doctor

# Optional: install the live GUI bridge extension into Blender 5.x
blenderbot --install-addon
```

### 2. Deploy the Cloud 3DGS Worker (Modal H100 / L40S)
```bash
modal setup
modal deploy src/blenderbot/splats/modal_app.py
```

### 3. Launch TUI REPL & Web 3D Studio
```bash
# Interactive terminal REPL
blenderbot

# Launch with the Seedance 2.5 + 3DGS Splat + FPS Rail Web Studio (http://127.0.0.1:7880)
blenderbot --open

# One-shot headless prompt
blenderbot -p "Build a 2m default cube shrine on Z=0 with 3-point lighting and render renders/cube.png"
```

---

## Key Slash Commands (REPL & Web Co-Pilot)

| Command | Description |
|---|---|
| `/web` | Start & open the Seedream 5.5, Seedance 2.5 & 3DGS Splat Studio in browser |
| `/fps [scene\|job_id]` | Open Dedicated Full-Screen Minecraft FPS Camera & 3DGS Splat Editor (`/fps`) |
| `/blender [auto\|bridge\|headless]` | Show Blender 5.3+ connection status or switch transport |
| `/splat [list\|import <job_id>]` | Open 3DGS Splat Studio, list local splats, or import into Blender 5.3 |
| `/seedream [prompt] [flags]` | Open Seedream 5.5 Image Studio or generate conditioned PNG/JPEG from CLI |
| `/seedance [prompt] [flags]` | Open Seedance 2.5 Studio or generate conditioned MP4 from CLI |
| `/render [frame] [engine]` | Render still (`BLENDER_EEVEE` / `CYCLES`) and attach to vision context |
| `/sheet [cols] [rows]` | Render burned-in multi-frame animation contact sheet |
| `/audit [objects...]` | Run manifold, scale, collision, clearance & camera-framing audit |
| `/undo` / `/redo` / `/actions` | Step backward/forward across unified `.blend` + workspace checkpoints |
| `/model < carbon \| opus \| gpt >` | Switch active LLM provider/model mid-session |
| `/role < viewer \| builder \| master >` | Scope agent capability surface |

---

## Repository Layout

```text
src/blenderbot/
  agent/         bl_ops parser, two-phase executor, .blend+file journal, roles & prompts
  blender_ops/   Zero-dep stdlib/bpy ops (modeling, shading, rig/anim, audit, render, splat_ops, bpy.tx)
  bridge/        Blender 5.x GUI extension (bpy.app.timers main-thread JSON-RPC server)
  context/       Token ledger, multimodal image/video loaders, compaction & artifact store
  media_gen/     OpenRouter Seedance 2.5 async client & FFmpeg keyframe decomposer
  providers/     Anthropic, Gemini, OpenAI, Grok & OpenRouter streaming adapters + watchdogs
  server/        FastAPI SSE sidecar (/api/seedance/*, /api/splats/*, /api/shots/*)
  splats/        Modal H100/L40S gsplat worker, Sim(3) floor aligner, proxy_mesh.py & shot_pack.py
  transport/     BridgeTransport (live GUI) & HeadlessTransport (persistent blender -b worker)
  ui/            Rich/prompt_toolkit TUI REPL & zero-dependency WebGL2 Studio (ui/web/)
docs/
  cloud_splat_design.md         Modal 3DGS, Sim(3) math, VRAM trainer & surgical resume spec
  splat_seedance_workflow.md    End-to-end Seedance 2.5 <-> 3DGS <-> Blender workflow guide
blenderbot.md                   Architectural invariants, Phase 1–5 status & Unreal/FPS roadmap
```

---

## Running Tests & Lint

```bash
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m pytest -q
```