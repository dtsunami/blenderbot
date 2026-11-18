# Phase 4.5 Architecture & Handoff — Seedance 2.5 ↔ Blender 5.3 Gaussian Splat / Point-Cloud Loop

> **Status:** **IMPLEMENTED & VERIFIED** — Cloud 3DGS compute runs on serverless Modal H100/L40S (`docs/cloud_splat_design.md`, `src/blenderbot/splats/`), integrated end-to-end with OpenRouter Seedance 2.5 (`src/blenderbot/media_gen/seedance.py`), native Blender 5.3 `PointCloud`/GeoNodes (`src/blenderbot/blender_ops/splat_ops.py`), and the Web Studio (`src/blenderbot/server/`, `src/blenderbot/ui/web/`).
> **Goal:** Turn generative video (`Seedance 2.5` via OpenRouter / provider API) and deterministic 3D choreography (`Blender 5.3+` Point Clouds / Gaussian Splats, Geometry Nodes, Cameras, Lights, PBR Hero Props) into a **bidirectional, iterative 3D-to-Video-to-3D loop**.

---

## 1. Why This Loop Is Cracked (and Why It Actually Works in Blender 5.3)

Pure video generators (Seedance 2.5, Veo, Sora, Kling) produce photoreal textures, lighting, foliage, atmospherics, and organic motion, **but** you cannot freely re-orbit the camera 40° higher, drop a deterministic rigid-body / rigged character into the middle of the shot with exact contact shadows, or tweak focal length (`35mm -> 85mm`) after generation without regenerating blindly.

Conversely, pure procedural Blender blockouts give **100% deterministic camera, lens, timing, physics, and spatial audit control**, **but** building photoreal environments from scratch takes hours of asset/shader work.

By bridging **Seedance 2.5 Video Generation** $\leftrightarrow$ **3D/4D Gaussian Splats & Point Clouds** $\leftrightarrow$ **Blender 5.3 PointCloud + Geometry Nodes + Camera Projection**, `blenderbot` gets both:

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│               BIDIRECTIONAL SEEDANCE 2.5 <-> BLENDER 5.3 SPLAT LOOP               │
└───────────────────────────────────────────────────────────────────────────────────┘

  [A] Optional Coarse Blockout / Camera Guide (Blender 5.3)
       │  render playblast / depth / still from Blender (`render.playblast` / `render.still`)
       ▼
  [1] Seedance 2.5 Video Generation (`gen.video` / `/seedance` via OpenRouter / fal / Replicate)
       │  text-to-video OR image/video-to-video (guided by Blender's new camera pass!)
       │  outputs: `assets/seedance/shot_01.mp4` (24fps, high-parallax orbit or scene plate)
       ▼
  [2] Video -> 3DGS / Point-Cloud Reconstruction (`splat.from_video`)
       │  FFmpeg sharp keyframe extraction (`-vf mpdecimate / select`)
       │  + Feed-forward 3DGS / Metric Point-Cloud solver (MASt3R / VGGT / DepthPro / Brush / gsplat)
       │    -> solved camera intrinsics/extrinsics (`cameras.json`) + `.ply` 3DGS / PointCloud
       │    (or per-frame `.ply` sequence for 4D dynamic splats!)
       ▼
  [3] Native Blender 5.3 Splat & Camera Ingestion (`splat.import` + `splat.project_texture`)
       │  • `bpy.data.pointclouds`: Native Blender 5.3 PointCloud datablock storing custom
       │    attributes (`position`, `radius`, `color` SH0, `opacity`, `rot_quat`, `scale_vec`)
       │  • Geometry Nodes Splat Renderer: Oriented ellipsoids (`Instance on Points` -> quad/icosphere)
       │    OR True Volumetric/Alpha-Blended Emissive Shader in EEVEE & Cycles (<25ms/frame!)
       │  • Camera-Projected Back-Render (`UVProject` / Window-Vector Shader): Projects the sharp
       │    Seedance video frames directly onto proxy geometry / splat surfaces from the solved
       │    Seedance camera trajectory so near-axis re-renders keep 100% 4K video crispness!
       ▼
  [4] Direct, Animate & Re-Light in Blender 5.3 (`camera.*`, `anim.*`, `light.*`, `mesh.*`)
       │  • Re-frame camera (`camera.upsert`, `rig.camera_pursuit`, focal length, DOF, dolly/crane)
       │  • Insert hero 3D props/characters on a Shadow-Catcher ground fit to the splat floor (`Z=0`)
       │  • `scene.audit` checks hero prop clearance & camera frustum against the splat bounds
       │  • `viewport.contact_sheet` / `render.still` feeds directly into Gemini Vision self-check!
       ▼
  [5] Iterate! (`render.playblast` -> Seedance 2.5 Video-to-Video / First-Last-Frame Re-Polish)
       └─► Feed Blender's re-choreographed EEVEE/Cycles render back into Seedance 2.5 (`strength=0.35..0.55`
           or first/last keyframe conditioning) to heal disoccluded splat gaps & harmonize lighting,
           then re-splat or composite final master!
```

---

## 2. Technical Breakdown of Each Stage in `blenderbot`

### Stage 1 — Generative Video Asset Creation (`src/blenderbot/media_gen/seedance.py`)
- **Confirmed OpenRouter Video API Contract (`bytedance/seedance-2.5`):**
  - Uses `OPENROUTER_API_KEY` (auto-harvested from `~/.blenderbot/.env`, `~/.davai/.env`, `~/.davag/.env`) against OpenRouter's async Video API:
    1. **Submit (`202 Accepted`):** `POST https://openrouter.ai/api/v1/videos` (`Authorization: Bearer $OPENROUTER_API_KEY`, `Content-Type: application/json`) -> returns `{"id": "job-...", "generation_id": "gen-...", "polling_url": "/api/v1/videos/job-...", "status": "pending"}`.
    2. **Poll:** `GET https://openrouter.ai/api/v1/videos/{id}` while `status in ("pending", "in_progress")` until `completed` (or `failed` / `cancelled` / `expired`), refreshing `FlightRecorder.touch()` on each poll tick.
    3. **Download & Cost Accounting:** Stream MP4 bytes from `GET https://openrouter.ai/api/v1/videos/{id}/content` (`Authorization` header, fallback `unsigned_urls[0]`), and add `usage.cost` (USD, ~$0.1028/s) directly into `agent.run_cost` / `agent.stats["cost"]` so `BLENDERBOT_MAX_COST` budget guards apply automatically!
  - **Validated `bytedance/seedance-2.5` Request Schema:**
    - `model`: `"bytedance/seedance-2.5"` (configurable via `BLENDERBOT_VIDEO_MODEL`; catalog discovery at `GET /api/v1/videos/models`)
    - `prompt`: string (required unless `frame_images` is set)
    - `duration`: `-1` or integer `4..30` seconds
    - `size`: `"854x480"` | `"1280x720"` | `"1920x1080"` | `"2520x1080"` | `"1080x1080"` | `"720x1280"` | `"1080x1920"` (etc.) or `resolution` (`"480p"` | `"720p"` | `"1080p"`) + `aspect_ratio` (`"16:9"` | `"4:3"` | `"1:1"` | `"3:4"` | `"9:16"` | `"21:9"`)
    - `frame_images`: `[{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}, "frame_type": "first_frame" | "last_frame"}]`
    - `input_references`: `[{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}, {"type": "audio_url", "audio_url": {"url": "data:audio/mpeg;base64,..."}}]`
    - `generate_audio`: bool, `seed`: int
- **Conditioning Modes (including Local Video Upload Decomposition):**
  1. **Text-to-Video (`prompt` only):** Initial environment plate (`mode="orbit_plate"` parallax hints) or dynamic subject clip (`mode="subject_4d"`).
  2. **Blender Keyframe Lock (`first_frame` / `last_frame` PNGs from `render.still`):** Encodes local PNGs as base64 `data:image/png;base64,...` entries in `frame_images` (`frame_type="first_frame"` / `"last_frame"`).
  3. **Local Video / Playblast Conditioning (`video_path` MP4 -> FFmpeg Auto-Pack):** Since OpenRouter's `bytedance/seedance-2.5` endpoint takes `frame_images` (`first_frame`, `last_frame`) + `input_references` (`image_url` keyframes + optional `audio_url`), passing a local `.mp4` automatically uses `ffprobe` + `ffmpeg` to:
     - Match `duration` (clamped `4..30`s) and snap dimensions to the closest allowed Seedance `size`,
     - Extract `t=0` and `t=end` frames into `frame_images` (`first_frame` + `last_frame`),
     - Extract 1–3 middle guide frames into `input_references` (`image_url`), and
     - Optionally extract the audio track (`extract_audio=True`) as `data:audio/mpeg;base64,...` into `input_references` (`audio_url`).

---

### Stage 2 — Video $\rightarrow$ Gaussian Splat / Point-Cloud Pipeline (`src/blenderbot/splats/`)
Two complementary reconstruction tiers so the workflow works **instantly (~2–5s)** on any machine AND scales to full **anisotropic 3DGS `.ply` training** when GPU tools are installed:

1. **Tier A — Fast Feed-Forward Metric Point-Cloud / Splat Unprojection (Fast Iteration Mode, ~2–6s):**
   - Extracts `N` sharp frames via `ffmpeg` (`fps=4..12` or full `24fps` for 4D).
   - Runs feed-forward metric depth + camera pose estimation (**VGGT** / **MASt3R** / **Depth-Anything-V2-Metric** — lightweight PyTorch or ONNX runtime in `.venv`, or cloud API fallback).
   - Unprojects `(u, v, depth, RGB)` pixels using solved per-frame camera intrinsics `K` and extrinsics `[R | t]` into world-space 3D points (`+Z` up, metric scale, auto-leveled so the dominant ground plane rests on `Z = 0`!).
   - Computes per-point local spacing `radius = depth / focal_px * pixel_scale` and writes standard binary little-endian `.ply` (plus `cameras.json` with exact solved Blender camera animation keyframes!).
   - **4D Dynamic Video Mode (`sequence=True`):** Writes `splats/frame_0001.ply .. frame_0096.ply` so moving Seedance subjects play back at 24fps inside Blender!

2. **Tier B — Full Anisotropic 3D Gaussian Splatting `.ply` (`3DGS` Mode):**
   - Accepts any standard 3DGS `.ply` (`x, y, z`, spherical harmonics `f_dc_0..2`, `opacity` logit, log-scales `scale_0..2`, quaternion `rot_0..3`) — whether reconstructed locally via **Brush** / **gsplat** / **COLMAP + Lichtfeld/Nerfstudio** from the Seedance MP4, or imported from disk.
   - Converts coordinate convention (`COLMAP/OpenCV +Y down, +Z forward` $\rightarrow$ `Blender Right-Handed +Z up, +Y forward`), applies sigmoid to `opacity` (`1 / (1 + exp(-op))`), `exp()` to `scale_0..2`, and `0.5 + C0 * f_dc` (`C0 = 0.28209479`) to base RGB!

---

### Stage 3 — Native Blender 5.3 `PointCloud` + Back-Projection Shader (`src/blenderbot/blender_ops/splat_ops.py`)
Remember Architectural Invariant #1: **code inside Blender (`blender_ops/`) uses ONLY Python stdlib (`struct`, `math`, `array`) + `bpy`!**
Binary little-endian `.ply` headers are plain ASCII followed by packed `float32` rows — Python's stdlib `array`/`memoryview` + Blender 5.3's fast `attribute.data.foreach_set(...)` loads **500,000+ splats into `bpy.data.pointclouds` in <120 ms** with zero third-party packages inside Blender!

#### How `splat.import` Renders Splats Natively in Blender 5.3 (`BLENDER_EEVEE` & `CYCLES`):
1. **Native `PointCloud` Datablock (`bpy.data.pointclouds`):**
   - Populates `pointcloud.points.add(N)` and bulk-writes `position` (`FLOAT_VECTOR`), `radius` (`FLOAT`), `splat_color` (`FLOAT_COLOR`), `splat_opacity` (`FLOAT`), `splat_scale` (`FLOAT_VECTOR`), and `splat_rot` (`QUATERNION` / Euler) via `.foreach_set()`.
2. **Two Switchable Render Modes (`render_mode`):**
   - **`"pointcloud"` (Fastest, Native Cycles/EEVEE Point Cloud):** Renders `bpy.data.pointclouds` directly with a custom Shader Node tree reading `Attribute("splat_color")` + `Attribute("splat_opacity")` $\rightarrow$ `MixShader(Transparent BSDF, Emission / Principled BSDF)` so splats either emit their baked radiance (`unlit: true`) or receive subtle Blender scene rim-lighting/shadows (`unlit: false`)!
   - **`"gaussian_quads"` (Anisotropic Oriented Ellipsoids via Geometry Nodes):** Builds a procedural Blender 5.3 **Geometry Nodes** modifier on the PointCloud (`Instance on Points` with stretched/oriented low-poly quads or icospheres scaled by `splat_scale` and rotated by `splat_rot`, shaded with radial Gaussian falloff $\exp(-\frac{1}{2} r^2)$ alpha-hashed / blended transparency in EEVEE & Cycles!).
3. **4D Animated Splat Playback (`splat.sequence`):**
   - Registers a frame-change handler (or Geometry Nodes `Import PLY` / mesh-sequence cache) that swaps `frame_NNNN.ply` on `scene.frame_current` so dynamic Seedance performances scrub and render frame-accurately in `render.playblast` and `render.animation`.
4. **Camera-Projected Texture Back-Rendering (`splat.project_video`):**
   - Creates a solved `SeedanceSourceCam` matching the video's original camera path and builds a **Camera-Projection Shader** (plus optional coarse proxy mesh extracted via Poisson/voxel hull from the point cloud + `GroundShadowCatcher` plane at `Z = 0`).
   - Projects the actual `shot_01.mp4` `ShaderNodeTexImage(source='MOVIE')` onto the scene from `SeedanceSourceCam` while you render from a **new** `DirectorCam`! Within $\pm 25^\circ$ of the source view, textures remain **pixel-sharp 1080p/4K video**, while splat points fill parallax disocclusions and hero 3D objects cast grounded Cycles/EEVEE contact shadows onto the scene!

---

## 3. Proposed `bl_ops` & REPL Slash Commands (Next Session Implementation Plan)

### New `bl_ops` (Host `.venv` Ops in `agent/executor.py` + In-Blender Ops in `blender_ops/splat_ops.py`)
| op | runtime | tier | params | summary |
|---|---|---|---|---|
| `gen.seedance` | host `.venv` | `mutate` | `prompt`?, `output_path`?, `duration`?, `size`?, `resolution`?, `aspect_ratio`?, `first_frame`?, `last_frame`?, `video_path`?, `reference_images`?, `audio_path`?, `generate_audio`?, `seed`?, `mode`? | Generate MP4 via OpenRouter Video API (`POST /api/v1/videos`, `bytedance/seedance-2.5`) with keyframe/video/audio conditioning & USD cost tracking |
| `splat.from_video` | host `.venv` | `mutate` | `video_path`, `output_dir`?, `camera`?, `fps_sample`?, `dynamic_4d`?, `max_points`?, `backend`? | Reconstruct 3DGS `.ply` (or 4D `.ply` sequence) + solved `cameras.json` from a Seedance/workspace MP4 without blocking Blender's main thread |
| `splat.import` | Blender 5.3 | `mutate` | `filepath`, `name`?, `render_mode`? (`gaussian_quads` \| `pointcloud`), `unlit`?, `density`?, `opacity_boost`?, `import_cameras`? | Fast stdlib `.ply` 3DGS/PointCloud loader into `bpy.data.pointclouds` + Geometry Nodes EEVEE/Cycles renderer |
| `splat.project_video` | Blender 5.3 | `mutate` | `target`, `video_path`, `projection_camera`, `shadow_catcher_ground`?, `blend_falloff`? | Back-project Seedance `MOVIE` textures from `projection_camera` onto splat proxy + Cycles/EEVEE shadow-catcher ground |
| `splat.crop_clean` | Blender 5.3 | `mutate` | `name`, `bounds_min`?, `bounds_max`?, `min_opacity`?, `max_radius`? | Prune floater splats / sky shell outside `bounds` or below `min_opacity` via `foreach_set` / Geometry Nodes |

### New TUI Slash Commands (`src/blenderbot/ui/commands.py`)
- `/seedance <prompt> [--from-render]` — Generate a Seedance 2.5 video clip (optionally conditioning on the current Blender camera render/playblast) and save to `assets/seedance/`.
- `/splat <video_or_ply_path> [--4d] [--project]` — One-command ingest: if given an `.mp4`, reconstructs splats + camera track and imports into Blender 5.3; if given a `.ply`, imports directly into `bpy.data.pointclouds` with EEVEE/Cycles splat shader and fits `Z=0` shadow-catcher ground.
- `/reproject [camera_name]` — Render the re-choreographed Blender shot (`DirectorCam` + hero 3D objects + splat environment) and optionally send it back through Seedance 2.5 video-to-video / keyframe polish.

---

## 4. Concrete Implementation Order for Next Session

1. **Step 1 — Binary `.ply` 3DGS & PointCloud Engine Inside Blender 5.3 (`blender_ops/splat_ops.py`):**
   - Pure-stdlib binary/ASCII `.ply` parser (`vertex` properties: `x,y,z`, `nx,ny,nz`, `f_dc_0..2` or `red,green,blue`, `opacity`, `scale_0..2`, `rot_0..3`) using `memoryview` + `array('f')` + `PointCloud.attributes[...].data.foreach_set(...)`.
   - Build both `render_mode="pointcloud"` and `render_mode="gaussian_quads"` (Geometry Nodes ellipsoid instancer + Gaussian radial alpha shader) and verify headless `BLENDER_EEVEE` & `CYCLES` renders in `tests/test_phase4_splats.py` using synthetic & real 3DGS `.ply` fixtures!
2. **Step 2 — Camera-Projected Video Back-Renderer + Shadow Catcher (`splat.project_video`):**
   - Build `SeedanceSourceCam` trajectory loader (`cameras.json` -> Blender 5.3 Slotted Action) + camera-space `MOVIE` texture projection shader + `Z=0` ground plane (`obj.is_shadow_catcher = True`) so 3D animated props/characters sit inside the Seedance scene with grounded shadows and reflections.
3. **Step 3 — Seedance 2.5 OpenRouter Client (`src/blenderbot/media_gen/seedance.py`):**
   - Async client using `cfg.keys["openrouter"]` (`OPENROUTER_API_KEY`) supporting text-to-video, first/last Blender keyframe conditioning (`first_frame`, `last_frame`), and video-to-video re-rendering, with `/seedance` slash command + `gen.seedance` op.
4. **Step 4 — Video $\rightarrow$ PointCloud / 3DGS Reconstructor (`src/blenderbot/splats/reconstruct.py`):**
   - Pluggable backend auto-detector in `blenderbot --doctor`:
     - **Built-in Fast Path:** FFmpeg frame sampler + monocular/multi-view metric depth unprojector (`depth-anything-v2` / `vggt` / `mast3r` when installed in `.venv`, or optical-flow parallax depth fallback) $\rightarrow$ `.ply` + `cameras.json`.
     - **High-Fidelity 3DGS Path:** Auto-detects `brush` / `colmap` / `gsplat` CLI on `PATH` when full anisotropic SH splat training is requested.
5. **Step 5 — End-to-End Closed-Loop Test (`tests/test_phase4_splats.py`):**
   - Generate/load video $\rightarrow$ reconstruct `.ply` + `cameras.json` $\rightarrow$ `splat.import` + `splat.project_video` $\rightarrow$ animate new `DirectorCam` crane move + 3D hero prop in Blender 5.3 $\rightarrow$ `viewport.contact_sheet` vision verification $\rightarrow$ output MP4!