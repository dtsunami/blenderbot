# blenderbot

**Blender 5.3+ AI 3D Creation Harness, Seedream 5.5 × Seedance 2.5 ↔ Modal H100 3D Gaussian Splatting Pipeline & Full-Screen Minecraft FPS Splat Editor.**

`blenderbot` pairs a multi-provider terminal REPL + Web 3D Studio with a zero-dependency Blender 5.3+ bridge, OpenRouter **Seedream 5.5** (`bytedance-seed/seedream-5-5`) image generation & **Seedance 2.5** (`bytedance/seedance-2.5`) video conditioning, serverless **Modal H100 / L40S** 3D Gaussian Splat reconstruction & rendering (`gsplat` + `VGGT-1B` + `COLMAP`), and a **Dedicated Full-Screen Minecraft FPS Camera & 3DGS Splat Editor (`/fps`)** for flying around Default Cube / 3DGS scenes, surgically culling & painting Gaussians at 60fps, syncing live cameras with Blender 5.3, and authoring Shot Collateral Packs.

---

## Core Capabilities

- **General Blender 5.3+ 3D Artist & TD Agent (`bl_ops` + transactional `bpy.tx`):**
  - **45+ pure-stdlib domain ops** across modeling (`mesh.primitive`, `modifier.add`), PBR shading (`material.upsert`), studio lighting (`light.rig`, `world.setup`), cameras (`camera.upsert`), Blender 5.3 Slotted Actions & layered NLA (`anim.*`, `rig.*`), multi-format USD/GLB/FBX/OBJ/PLY I/O (`asset.*`), topology/BVH collision/framing audits (`scene.audit`), and automatic vision self-verification (`viewport.render_still`, `viewport.contact_sheet`).
  - Dual transport: hot-attaches to a live GUI **Blender 5.3+** session (`BridgeTransport` on `127.0.0.1:18766`) or auto-spawns a warm background `blender.exe -b` worker (`HeadlessTransport`), with full `.blend` + filesystem undo/redo (`journal.undo` / `journal.redo`).
- **Seedance 2.5 ↔ Cloud 3D Gaussian Splatting Loop (`splats/`, `media_gen/`):**
  - **Video → Floor-Aligned 3DGS (`splat.from_video`):** Extracts exact video PTS timestamps, solves metric poses via **VGGT-1B** (`mode="pad"`, portrait/ultrawide safe), **COLMAP `PINHOLE`**, **`colmap_prior`** (Blender intrinsic lock + closed-form Umeyama `det=+1` Sim(3)), or **`known_cameras`** (zero-drift hard lock), aligns the dominant floor to `Z = 0`, optimizes sub-pixel `se(3)` camera poses + Mip-Splatting antialiased Gaussians on Modal H100/L40S, and exports `splat.ply`, `splat.splat`, `proxy_collision.glb` (Surface-Nets 3D mesh), and `cameras.json`.
  - **Blender 5.3 → 3DGS (`splat.from_blender`):** Captures multi-view RGBA silhouettes + exact camera poses + surface seed points (`0.0px` pose drift, `>42 dB` PSNR in ~15s on H100).
  - **Native Blender 5.3 Splat Import (`splat.import`):** Strided `PointCloud` + Geometry Nodes anisotropic 3D ellipsoids + exact Mahalanobis ray-integral shader (`BLENDER_EEVEE` & `CYCLES`) + `L=1` Spherical Harmonics + `SeedanceSourceCam` trajectory rig.
- **`tt-splat` Telemetry Workbench, Modal-Farmed Trajectory Renderer & Synced Comparator:**
  - Live Loss / L1 / SSIM / PSNR & Gaussian-count `<canvas>` plots, stacked GPU stage-timing bar (`fwd_raster | loss_ssim | bwd | adam | densify | host`), `snapshots/` scrubber, live `modal.Dict` mid-run controls (`Pause`, `Save Ckpt`, `Stop & Export`, `Focus Camera`, live `λ_dssim` / `opa_reg` push), and 3D Blender `SplatCropBox` $\rightarrow$ Modal `drop_aabb` surgical Cull & Heal.
  - **🎬 Compare & Path:** Farms full-rate H.264 MP4 renders of the recreated original video trajectory (or any Blender camera) to Modal H100 (~1.5–2s) alongside matching Blender 5.3 EEVEE clips, with lockstep dual-video playback, orbitable 3D camera path + live moving frustum, and synchronized `Yaw° / Pitch° / Roll° / Speed` curves.
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