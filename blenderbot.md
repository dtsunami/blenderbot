# blenderbot — General Blender 5.3+ 3D & Animation Agent

**blenderbot** is a general-purpose agentic 3D creation, procedural modeling, shading/lighting, rigging/animation, spatial-verification, and rendering harness for **Blender 5.3+**, built on `@davag`'s single-owner `SessionBus` + multi-provider streaming architecture and `@davai`'s atomic `bl_ops` + transactional `bpy.tx` sandbox design.

---

## 1. Core Design Principles

1. **General 3D & Animation First (Previs as One Workflow Among Many):**
   - Author 3D scenes, hard-surface & procedural geometry, modifier stacks, Principled BSDF materials, studio/HDRI/sun lighting, keyframe/F-curve/NLA animation, armatures & bone poses, camera choreography, and multi-format asset pipelines (`GLB`/`GLTF`, `FBX`, `OBJ`, `USD`, `STL`, `.blend` libraries) directly in Blender 5.3+.
   - Previs, product turntables, motion graphics, game-ready GLB authoring, architectural layout, and technical visualization all compose cleanly out of general 3D + animation primitives rather than hard-coding narrow genre assumptions into the core.
2. **Programmatic + Multimodal Self-Verification Loop:**
   - **Deterministic Spatial & Mesh Audits (`scene.audit`, `scene.inspect`):** Inspect topology health (non-manifold edges, zero-area faces, unapplied/negative scale), world-space AABBs, BVHTree surface contact/clearance, pairwise mesh collisions, camera frustum framing, and animation continuity across frames.
   - **Multimodal Visual Grounding (`viewport.render_still`, `viewport.contact_sheet`, `render.*`):** Vision-capable models automatically receive rendered stills and burned-in frame/time/lens mosaic contact sheets (~1.5k tokens for 12–16 frames across an animation) to visually verify silhouette, shading, lighting, and motion arcs.
3. **Hybrid Structured `bl_ops` + Transactional `bpy.tx`:**
   - Structured `bl_ops` give fast, schema-validated, zero-boilerplate operations for primitives, transforms, modifiers, materials, lights, cameras, Blender 5.3 slotted-Action keyframing, armature posing, asset import/export, spatial audits, and EEVEE/Cycles/Workbench rendering.
   - Transactional `bpy.tx` executes procedural `bpy`/`bmesh`/`mathutils` Python (custom topology, Geometry Nodes, shader graphs, procedural rigs) with **automatic `.blend` pre-snapshot rollback** on any exception or assertion failure — leaving zero orphan datablocks on error.

---

## 2. Verified Blender 5.3 Technical Invariants (Probed Locally)

Live probes against `C:\Program Files\Blender Foundation\Blender 5.3\blender.exe` (`5.3.0`, embedded Python `3.13.13`, FFmpeg `8.1` on PATH) established five critical Blender 5.x API rules baked into `blender_ops` and system prompts:

1. **Slotted / Layered Actions (`Action.fcurves` Removed in Blender 5.x):**
   - `bpy.types.Action` no longer has `.fcurves`. Animation curves live inside `action.layers[...] -> strips[...] -> channelbags` and must be created/queried via `bpy_extras.anim_utils.action_ensure_channelbag_for_slot(action, anim_data.action_slot)` (wrapped by `ensure_channelbag_fcurves` / `iter_action_fcurves`).
2. **Render Image vs. Video `media_type` (`Blender 5.0+`):**
   - Setting `render.image_settings.file_format = 'FFMPEG'` directly fails unless `render.image_settings.media_type = 'VIDEO'` is set first (`IMAGE` | `MULTI_LAYER_IMAGE` | `VIDEO`), followed by `render.ffmpeg.format = 'MPEG4'` and `render.ffmpeg.codec = 'H264'`.
3. **Headless `BLENDER_WORKBENCH`, `BLENDER_EEVEE` & `CYCLES`:**
   - Warm `BLENDER_WORKBENCH` renders in `blender.exe -b` run at **~19.5 ms/frame** (`>50 fps`), while `BLENDER_EEVEE` and `CYCLES` render full PBR materials, lights, shadows, and reflections headlessly with zero GUI window required.
4. **Thread-Safe Main-Thread Execution (`bpy.app.timers`):**
   - `bpy` and `gpu` contexts are bound to Blender's main thread. Inside GUI Blender, `bridge/server.py` runs a stdlib `ThreadingTCPServer` on `127.0.0.1:18766` and drains `(op, params)` on the main thread via `bpy.app.timers.register(..., persistent=True)`.
5. **Blender 5.x Extensions & Persistent Warm Workers:**
   - User extensions live at `%APPDATA%\Blender Foundation\Blender\5.3\extensions\user_default\blenderbot_bridge\` (`blender_manifest.toml` + `__init__.py`), while headless mode spawns a persistent warm `blender.exe -b` daemon (`HeadlessTransport`) so `bpy.data` persists across turns.

---

## 3. Architectural Invariants

1. **Zero Third-Party Dependencies Inside Blender (`src/blenderbot/blender_ops/` & `src/blenderbot/bridge/`):**
   - Imports Python stdlib + Blender built-ins (`bpy`, `bmesh`, `mathutils`, `bpy_extras`) only. Identical code runs inside live GUI Blender (`BridgeTransport`) and persistent background `blender.exe -b` workers (`HeadlessTransport`).
2. **Single `bl_ops` Fenced Protocol + Verbatim Payloads:**
   - No brittle native tool-calling schema; at most one ```` ```bl_ops ```` block per turn (accepts `da_ops` / `dr_ops` aliases) with verbatim `--blenderbot:NAME-- ... --blenderbot:end--` (and `--davai:NAME--` / `--davag:NAME--`) multi-line payload blocks.
3. **Unified `.blend` + Workspace Undo/Redo `Journal`:**
   - Mutating `bl_ops` batches automatically record `.blend` checkpoints (`scene.snapshot_save` / `scene.snapshot_restore`) alongside workspace file snapshots (`fs.write`, `fs.replace`), enabling instant `/undo`, `/redo`, and `journal.*` rewinds across sessions (`--resume`).
4. **3-Tier Capability Roles (`--role`, `/role`):**
   - `viewer` — Read-only inspection & visual review (`scene.passport`, `scene.inspect`, `scene.audit`, `rig.armature_inspect`, `viewport.render_still`, `viewport.contact_sheet`, `fs.read`, `fs.list`, `journal.history`).
   - `builder` *(default for safe autonomous 3D & animation)* — Full non-destructive 3D modeling, modifiers, materials, lighting, cameras, keyframing/F-curves/armatures/NLA, asset import/export, renders/playblasts, transactional `bpy.tx`, `.blend` save, and `/undo` + `/redo`. Blocks `destructive` ops (`scene.clear`, `scene.open`), `bpy.eval`, and `shell`.
   - `master` — Full capability surface including approval-gated `destructive` ops, `bpy.eval`, and `shell`.

---

## 4. Phased Implementation Roadmap

### Phase 1 — Blender 5.3 Core: Dual Transport, Bridge Add-on, Transactional `bpy.tx` & Fast Viewport Ops *(COMPLETE)*
- [x] Workspace setup (`pyproject.toml`, `.gitignore`, `.daignore`).
- [x] **Blender Discovery (`config/blender_paths.py`):** Auto-locate Blender 5.3+, embedded Python 3.13, user extensions dir, and FFmpeg/FFprobe.
- [x] **Stdlib Localhost TCP Bridge (`bridge/protocol.py`, `bridge/server.py`, `bridge/installer.py`):** Token-authenticated JSON-RPC (`127.0.0.1:18766`) + Blender 5.x extension installer.
- [x] **Dual Transports (`transport/`):** `BridgeTransport` (live GUI) + `HeadlessTransport` (persistent warm `blender -b` daemon) + `connect_blender(mode="auto")`.
- [x] **Transactional `bpy.tx` (`blender_ops/tx_op.py`):** Atomic `.blend` snapshot rollback on exception + pre-injected Blender 5.3 helpers.

### Phase 2 — General 3D Modeling, Shading/Lighting, Multi-ID 'Baklava' Animation & Asset I/O `blender_ops` *(COMPLETE)*
- [x] **Scene & Deep Inspection (`scene_ops.py`):** Enriched `scene.passport` (lights, armatures, modifiers, materials, world) + `scene.inspect` (deep per-object mesh topology, modifiers, materials, constraints, bone hierarchy, and F-curves) + neutral `scene.setup` (`BLENDER_EEVEE` / `CYCLES` / `BLENDER_WORKBENCH`).
- [x] **Modeling, Transforms & Modifiers (`modeling_ops.py`):**
  - `mesh.primitive` (`cube`, `uv_sphere`, `ico_sphere`, `cylinder`, `cone`, `plane`, `grid`, `torus`, `monkey` with location/rotation/scale/material/collection/smooth shading).
  - `object.transform`, `object.duplicate`, `object.delete`, `object.parent`, `object.origin`, `object.shade`.
  - `modifier.configure` (`SUBSURF`, `BEVEL`, `MIRROR`, `ARRAY`, `SOLIDIFY`, `BOOLEAN`, `DISPLACE`, `SIMPLE_DEFORM`, `WIREFRAME`, `DECIMATE`, `SHRINKWRAP`) & `modifier.apply`.
- [x] **Materials, Lighting & Cameras (`shading_ops.py`):**
  - `material.upsert` (Principled BSDF base color, metallic, roughness, IOR, transmission, alpha, coat, emission, normal/roughness/albedo image textures) & `material.list`.
  - `light.upsert` (`AREA`, `SUN`, `POINT`, `SPOT`), `light.rig` (`three_point`, `studio_softbox`, `outdoor_sun`, `dramatic_rim`), and `world.setup` (solid, sky, gradient, or HDRI environment texture + strength).
  - `camera.upsert` (lens mm, sensor width, DOF focus target/f-stop, stationary / track-to target / turntable orbit animation).
- [x] **Multi-ID Slotted Actions ('Project Baklava'), NLA Layering, Armatures & Path/Scatter (`anim_ops.py`, `rig_ops.py`):**
  - `anim.keyframe` (Blender 5.3 Multi-ID shared `action` slots across objects, `pose.bones["..."]`, camera `lens`, light `energy`/`color`, modifiers, and materials with `BEZIER` / `LINEAR` / `CONSTANT` / `BACK` / `ELASTIC` / `BOUNCE` interpolation).
  - `anim.nla_layer` (non-destructive animation layering on NLA tracks with `REPLACE` / `ADD` / `COMBINE` blending, `influence`, `scale`, `repeat`, and `mute`).
  - `anim.fcurve_modifier` (`CYCLES` loops, `NOISE` secondary motion, `LIMITS`, `STEPPED`) and range retiming/clearing via `anim.clear`.
  - `anim.bake` (bake procedural constraints/drivers/parenting/NLA to clean keyframes across `[frame_start..frame_end]`).
  - `rig.armature_create` (build connected/hierarchical bone chains from joint coordinates + smooth mesh skinning), `rig.armature_inspect`, and `rig.pose` (pose and keyframe armature bones with shared Multi-ID `action` slot support).
  - Generalized `rig.curve_path`, `rig.follow_path`, `rig.camera_pursuit` (optional tracking rig), and `scatter.surface` (with `terrain.scatter` alias).
- [x] **Multi-Format Asset Import/Export & Production Rendering (`asset_ops.py`, `render_ops.py`, `audit_ops.py`):**
  - `asset.import` & `asset.export` (`GLB`/`GLTF`, `FBX`, `OBJ`, `USD`/`USDA`/`USDC`, `STL`, `PLY`) + `asset.blend_append` (append/link objects, collections, or materials from `.blend` files).
  - Generalized `scene.audit` (mesh topology checks: non-manifold, zero-area faces, negative/unapplied scale; pairwise AABB/BVH intersection checks; ground clearance; frustum framing; kinematic smoothness).
  - `render.still` & `render.animation` (production EEVEE / Cycles / Workbench renders with sample count, denoiser, transparent film, and MP4/PNG output) alongside fast `viewport.render_still`, `viewport.contact_sheet`, and `render.playblast`.

### Phase 3 — Multi-Provider Agent Harness, Stability Watchdogs, Multimodal Vision Loop & Interactive CLI/TUI *(COMPLETE)*
- [x] **Context, Providers & Stream Watchdog (`context/`, `providers/`):** `Ledger` (with `replace_text` superseded failed-op compression), `ArtifactStore`, `PasteStore`, auto-compaction, image/video multimodal loaders, `providers/watchdog.py` (180s first-byte + 90s per-chunk idle timeout + instant `Esc` cancellation race), Gemini `sthought`/`düşünceler` leak stripping, and `~/.davai/.env` + `~/.davag/.env` automatic key harvesting.
- [x] **Agent Loop, `bl_ops` Parser, Sandbox & `.blend` Journal (`agent/`):**
  - `parser.py`: `bl_ops` fenced block + `--blenderbot:NAME--` (and `--davai`/`--davag`) verbatim payload parser.
  - `journal.py`: Unified `.blend` snapshot (`scene.snapshot_save`/`restore`) + workspace file undo/redo journal.
  - `executor.py`: Two-phase batch validator + executor with automatic multimodal image attachment when `viewport.render_still`, `viewport.contact_sheet`, or `render.still` runs (`attach_to_context: true`).
  - `grounding.py` & `prompt.py`: Upfront Blender Scene Grounding + ephemeral `[Live Scene State]` on outgoing turns + General 3D & Animation Artist/TD Playbook + 3-tier roles (`viewer`, `builder`, `master`).
- [x] **Session, Flight Recorder, Headless Runner, Slash Commands & Rich TUI REPL (`session/`, `headless.py`, `ui/`, `cli.py`):**
  - `session/flight.py`: Synchronous `breadcrumbs.jsonl` phase + `idle_ms` stall-heartbeat recorder (`/flight`).
  - `ui/clipboard.py`: Automatic Win32 `conhost` QuickEdit disable (`prepare_windows_console`) so console clicks never freeze stdout.
  - Full `blenderbot` CLI + interactive `prompt_toolkit`/`rich` TUI REPL (`/blender`, `/render`, `/sheet`, `/turntable`, `/audit`, `/import`, `/export`, `/save`, `/open`, `/undo`, `/redo`, `/role`, `/model`, `/context`, `/tps`, `/flight`).

### Phase 4 — Seedance 2.5 ↔ Modal Cloud 3DGS Loop & Web 3D Studio *(COMPLETE)*
- [x] **OpenRouter Seedance 2.5 Video Generation (`media_gen/seedance.py`, `gen.seedance`, `/seedance`):**
  - Async `POST /api/v1/videos` client (`bytedance/seedance-2.5`) with single/dual source video decomposition (`bridge_v1_to_v2`, `span_v1_to_v2`, `v1_primary_v2_ref`), `first_frame`/`last_frame` PNG conditioning, interior `input_references` keyframes + audio extraction, `orbit_plate` / `subject_4d` motion hints, and automatic USD spend accounting.
- [x] **Modal Serverless H100 / L40S 3DGS Pipeline (`splats/`, `splat.from_video`, `splat.from_blender`, `splat.resume`, `/splat`):**
  - Static AOT-baked `gsplat==1.4.0` (`gsplat.csrc` compiled on 32 vCPUs for `sm_89;sm_90`, zero runtime JIT), `facebook/VGGT-1B` letterbox-safe (`mode="pad"`) metric pose solver + headless 16-thread `COLMAP` SIFT/mapper solver, proper `det(R)=+1` Sim(3) `Z = 0` RANSAC floor aligner (`splats/sim3.py`), anisotropy + opacity/scale floater regularizers, and stateful multi-burst surgical resume (`blenderbot-splat-jobs` Volume + 90s HBM warm cache).
  - **Direct Blender 5.3 $\rightarrow$ 3DGS Export (`splat.capture_views` + `splat.from_blender`):** Captures multi-view RGBA silhouettes + exact OpenCV `(R_w2c, t_w2c, K)` + surface seed points directly from Blender (`0.0px` pose drift, `>42 dB` PSNR in ~15s on H100).
- [x] **Native Blender 5.3 Splat Engine & Camera Re-Projection (`blender_ops/splat_ops.py`):**
  - Vectorized `numpy` + `PointCloud.resize(n)` + `foreach_set` `.ply` loader (`<10 ms` `foreach_set` for 500k splats) with automatic needle/billboard culling, 8-vertex `[-2.5σ, +2.5σ]^3` Geometry Nodes instancer + exact Mahalanobis ray-integral shader (`exp(-0.5 * ρ²)`) + live `L=1` Spherical Harmonics (`BLENDER_EEVEE` & `CYCLES`), optional `IsViewport` 60fps PointCloud toggle, hemisphere-continuous `SeedanceSourceCam` + `SeedanceSourceCam_Poses` 3D frustum trajectory rig + live GT video/frame camera overlay (`splat.import_cameras`), `splat.crop_clean`, and feathered `splat.project_video`.
  - All 4 pathclearing gates verified: `P0_CLOUD_OK` (`scripts/pathclear_p0_modal.py`), `P1_SIM3_OK` / `P1_BLENDER_TO_SPLAT_OK` (`tests/test_splat_math_gate.py`, `scripts/pathclear_p1_blender_to_modal.py`), `P2_BLENDER_OK` (`tests/test_splat_blender_gate.py`), and `P3_REPROJECT_OK` (`tests/test_splat_reproject_gate.py`).
- [x] **FastAPI SSE Web Studio & Embedded WebGL2 `.splat` 3D Viewer (`server/`, `ui/web/`, `--web --open`):**
  - Dual **Seedance 2.5 Composer** & **3DGS Splat Studio** (`127.0.0.1:7880`) with workspace media library, live headed Blender 5.3 GUI Bridge hot-connect (`blenderbot --install-addon`), editable training iterations + anti-floater/culling knobs, live SSE training telemetry + render/error triptychs, zero-dependency WebGL2 `.splat` interactive 3D orbit viewer, and 1-click `+2k Polish` & Blender import (`📥 Splat+Cams` / `🎥 Cams Only`).

### Phase 5 — Interactive Studio Refinements & Domain Recipes *(OPTIONAL EXTENSIONS)*
- [ ] **Advanced `arcgs` / `tt-splat` Studio Controls:** Interactive 2D dynamic-subject mask painter, live loss/PSNR chart timeline scrubber across `snapshots/`, and 3D Blender CropBox $\rightarrow$ Modal `drop_aabb` sync button.
- [ ] **4D Dynamic Splat Playback & Motion-Arc Overlays:** Frame-indexed `.ply` sequence handler + `anim.motion_paths` viewport overlays.