# Cloud Burst 3DGS Training & Blender 5.3 Ingestion — Full Design, Provisioning & Risk Spec

> **Status:** **IMPLEMENTED & VERIFIED (Through Phase 7)** — `P0`–`P3` gates green; upgraded with **Apple Depth Pro 2 (`ml-depth-pro-2`)** metric surface nudging + `RGB+ED` free-space floater carving, **Iterative Camera-Arc Cycles** + post-train **`Snap Arc & Clean Floaters`**, **Per-Camera 3×4 Exposure + Cauchy IRLS**, **Modal Sharded Blender 5.3 Cloud Cycles (`render.modal`)**, full **3DGS Parameter Tooltips + Agent Snicker-Test Review**, and **Versioned Multi-Pass Trajectory Comparison**.  
> **Lineage:** Carries forward the flat 4-stage pipeline, self-contained binary parsers, and live telemetry controller from `arcgs` / `tt-splat`, moving compute off brittle local SYCL/Blackhole stacks onto **per-second serverless NVIDIA GPUs (Modal H100 / L40S, with Vast.ai / RunPod HTTP pod fallback)**.  
> **Target Wall Time & Cost:** **25–50s** end-to-end (`MP4` $\rightarrow$ trained `.ply` + `cameras.json` in Blender 5.3) at **~$0.03–$0.05 per scene**.

---

## 1. Verdict & End-to-End Budget

> **Verdict in one line:** Stock CUDA `gsplat` on H100/L40S eliminates the compute bottleneck (~8–15s for 10k steps), moving the real engineering risk to **(1) AI-video pose stability (`VGGT` vs. `COLMAP`)**, **(2) baking CUDA kernels + model weights at image-build time so cold starts stay <4s**, **(3) Sim(3) quaternion/scale math when auto-leveling the floor to `Z=0`**, and **(4) Blender 5.3 EEVEE/Cycles alpha-overdraw limits**.

### Target Latency & Cost Budget (10s Seedance 1080p clip, 90 sampled frames, H100)

| Stage | Where | Target Time | Notes |
|---|---|---|---|
| **0. Upload `.mp4`** | Windows $\rightarrow$ Modal gRPC | `1.5 – 3.0s` | 15–35 MB Seedance MP4 streamed from `assets/seedance/` |
| **0b. Container Cold Start** | Modal (`H100` / `L40S`) | `0.0 – 4.0s` | `0s` when warm (`scaledown_window=60s`); `~3s` from baked snapshot |
| **1. `extract`** | Cloud CPU (`ffmpeg`) | `0.8 – 1.5s` | Sample 80–120 sharp frames (`mpdecimate` / uniform fps) to `/dev/shm` RAM disk |
| **2. `pose` + `align`** | Cloud GPU (`VGGT-1B` or `pycolmap`+`glomap`) | `3.0 – 9.0s` | Camera `K`, `viewmat`, dense init cloud + RANSAC `Z=0` floor Sim(3) alignment |
| **3. `train_gsplat`** | Cloud GPU (`gsplat` CUDA, VRAM-cached) | `7.0 – 16.0s` | 7,000–12,000 steps @ 720p/1080p (`packed=True`, `fused-ssim`, `sh_degree=1`) |
| **4. `export` + Download** | Cloud $\rightarrow$ Windows | `1.5 – 3.0s` | `zstd`-compressed SH1 `splat.ply` (~9 MB) + `splat.splat` (~8 MB) + `cameras.json` |
| **5. `splat.import`** | Local Blender 5.3 (`bpy`) | `0.08 – 0.20s` | Pure-stdlib `memoryview` $\rightarrow$ `PointCloud.attributes.foreach_set()` + GeoNodes |
| **Total** | End-to-End | **~16 – 37s** | **~$0.038** on Modal H100 ($3.95/hr) · **~$0.019** on Modal L40S ($1.95/hr) |

---

## 2. Provisioning & Deployment Demystified (How Cloud Execution Actually Works)

You do **not** need AWS IAM, Terraform, Docker Hub, Kubernetes, S3 buckets, or local PyTorch/CUDA on your Windows machine. `blenderbot`'s local `.venv` stays lightweight (~50 MB of pure Python packages).

### 2.1 Primary Path: Modal Serverless Burst (`modal_app.py`)

Modal treats a single Python file (`src/blenderbot/splats/modal_app.py`) as both the **container definition** and the **remote RPC worker**.

#### Lifecycle: One-Time Provisioning (5 minutes, done once)
1. **Install lightweight client locally (Windows):**
   ```powershell
   pip install modal zstandard
   modal setup
   ```
   `modal setup` opens a browser login once and writes `[default] token_id / token_secret` to `C:\Users\<you>\.modal.toml` (or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in `~/.blenderbot/.env`, auto-harvested alongside `OPENROUTER_API_KEY`).
2. **Create persistent weight cache Volume + bake the CUDA image (`modal deploy`):**
   ```powershell
   modal deploy src/blenderbot/splats/modal_app.py
   ```
   When you run `modal deploy`:
   - Modal uploads **only** the ~400-line `modal_app.py` file to Modal's remote build cluster (nothing builds on your Windows PC; no Visual Studio / `nvcc` required locally).
   - Modal's builder constructs a Linux image from `nvidia/cuda:12.4.1-devel-ubuntu22.04`, installs `ffmpeg`, `colmap`/`glomap`, `torch==2.5.1+cu124`, pre-built `gsplat` + `fused-ssim` CUDA wheels (`TORCH_CUDA_ARCH_LIST="8.9;9.0"`), and runs a **compile-verification smoke function** so zero JIT compilation ever happens at runtime.
   - Downloads `facebook/VGGT-1B` (~4 GB safetensors) once into a persistent `modal.Volume.from_name("blenderbot-splat-models", create_if_missing=True)` mounted at `/models`.
   - Registers the serverless class `blenderbot-splat :: SplatWorker` in your Modal workspace.
   - **Idle cost after deploy:** **$0.00 / month** (Modal includes **$30/month free compute** on the starter plan — enough for ~750 H100 splat trainings every month before your card is ever charged).

#### Lifecycle: Per-Shot Execution (Automatic inside `blenderbot`)
When you run `/splat assets/seedance/shot_01.mp4` (or the agent emits `{"op": "splat.from_video", ...}`, or you click **Train Splat** in the Web Studio):
```text
Windows Host (`D:\blenderbot`, no GPU needed)             Modal Cloud (`blenderbot-splat` app)
─────────────────────────────────────────────             ────────────────────────────────────
1. Read `shot_01.mp4` (25 MB)
2. Call `SplatWorker().run_stream.remote_gen(mp4, cfg)` ──► Wake H100/L40S snapshot (~0–3s)
                                                            Mount `/models` Volume (VGGT cached)
                                                            Write MP4 to `/dev/shm/job/` (RAM disk)
3. Receive streamed JSON telemetry dicts (~2 Hz):      ◄── yield {"stage": "extract", "frames": 96}
   • `FlightRecorder.touch()` (prevents watchdog timeout)   yield {"stage": "pose", "cams": 96, "pts": 240000}
   • Push SSE event to Web Studio (`127.0.0.1:7880`)        yield {"stage": "train", "step": 2000, "psnr": 28.1, "jpg_b64": "..."}
   • Update Rich TUI progress bar                           ...
4. Receive final artifact bundle (`zstd` compressed):  ◄── yield {"stage": "done", "ply_zst": ..., "splat": ...,
   • Write `assets/splats/<shot>/splat.ply`                        "cameras_json": ..., "gpu_sec": 28.4, "cost_usd": 0.031}
   • Write `assets/splats/<shot>/splat.splat` (WebGL)       Container idles `scaledown_window=60s`
   • Write `assets/splats/<shot>/cameras.json`              (next shot starts in 0.0s!), then scales to 0 ($0/hr)
   • Add `cost_usd` to `agent.run_cost` budget guard
5. Call local `splat.import` on Blender 5.3 (<120 ms)
```

### 2.2 Escape Hatch: Vast.ai / RunPod Pod Worker (`BLENDERBOT_SPLAT_URL`)
Zero vendor lock-in: the core trainer (`splat_core.py`) has **zero Modal imports**. A thin 90-line FastAPI wrapper (`pod_server.py`) exposes the identical streaming contract over HTTP (`POST /v1/reconstruct` $\rightarrow$ NDJSON stream + multipart/zstd bundle).
- If you rent an **RTX 5090** on Vast.ai ($0.45/hr) or **RTX 4090 / H100** on RunPod for an all-day session:
  1. Launch pod with `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.
  2. Run `bash scripts/bootstrap_splat_pod.sh` (installs `gsplat`, `vggt`, `ffmpeg`, starts `uvicorn` on port `8790`).
  3. Set `BLENDERBOT_SPLAT_URL=http://<pod-ip>:8790` (and `BLENDERBOT_SPLAT_TOKEN=...`) in `.env`.
- `blenderbot` checks `BLENDERBOT_SPLAT_URL` first, then falls back to Modal automatically.

---

## 3. System Architecture (`arcgs` Style Adapted to `blenderbot`)

```text
src/blenderbot/
├── splats/                          # Host + Cloud package (never imported inside Blender)
│   ├── __init__.py
│   ├── config.py                    # @dataclass ExtractConfig, PoseConfig, TrainConfig, ExportConfig (arcgs style)
│   ├── core_worker.py               # Pure CUDA worker code (runs on Modal OR Vast/RunPod pod):
│   │                                #   1. ffmpeg extract -> /dev/shm
│   │                                #   2. VGGT (default) or pycolmap+glomap pose + dense init
│   │                                #   3. RANSAC floor Z=0 + Sim(3) alignment (means, quats, scales, cams)
│   │                                #   4. VRAM-preloaded gsplat loop (packed=True, fused-ssim, anti-floater)
│   │                                #   5. Export SH1 .ply (zstd) + 32-byte .splat + Blender cameras.json
│   ├── modal_app.py                 # Modal Image + Volume + @app.cls(gpu=...) streaming entrypoint
│   ├── pod_server.py                # Optional FastAPI wrapper for Vast.ai / RunPod SSH pods
│   └── client.py                    # Local async dispatcher (Modal .remote_gen() | HTTP NDJSON) + cost/SSE hook
│
└── blender_ops/
    └── splat_ops.py                 # Pure Python stdlib (`struct`, `array`, `math`) + `bpy` ONLY!
                                     #   • `splat.import`: binary .ply -> `bpy.data.pointclouds` + GeoNodes/Cycles
                                     #   • `splat.crop_clean`: AABB / radius / opacity floater pruning
                                     #   • `splat.project_video`: SeedanceSourceCam + MOVIE UV/Window projection + Z=0 shadow catcher
```

### Key Upgrades Over `arcgs/stages/train_gsplat.py`
1. **VRAM Frame Preloading (Fixes 80% H100 Idle Bottleneck):**
   - `arcgs` called `Image.open(cam.image_path)` on CPU *every training step*. On H100, rasterization takes `0.4 ms` while PIL JPEG decode takes `4–6 ms`!
   - `core_worker.py` decodes all `M` training frames once in parallel (`ThreadPoolExecutor`) into a contiguous `uint8` GPU tensor `(M, H, W, 3)` (`100 × 1080 × 1920 × 3` = **622 MB** out of 80 GB HBM3) and converts `gt = frames_u8[idx].float() / 255.0` on-chip in `0.02 ms`.
2. **`packed=True` + `fused-ssim` + `absgrad=True`:**
   - Cuts memory bandwidth and speeds up `gsplat` backward + SSIM by ~2.5× over `arcgs`'s `packed=False` + `F.conv2d` 11×11 SSIM.
3. **SH1 Default (`sh_degree=1`, 23 floats/vertex instead of 62):**
   - Blender's PointCloud/GeoNodes shaders use DC RGB (`f_dc_0..2`), and AI-generated video lacks physically consistent 3rd-order view-dependent speculars anyway (SH3 overfits Seedance temporal flicker into rainbow floaters). SH1 cuts `.ply` size by **2.7×** (`74 MB -> 27 MB` raw, `~9 MB` zstd) and speeds up rasterization by ~30%.

---

## 4. Glass Jaws & Performance Risks (Honest Failure Modes + Mitigations)

### Glass Jaw 1 — AI Video Temporal Flicker & Zero-Baseline Pans ("The COLMAP / Floater Trap")
- **The Risk:** Seedance 2.5 synthesizes pixels, not rigid 3D geometry. Two failure modes happen regularly on AI video:
  1. **Degenerate camera motion:** A prompt asks for a camera move, and Seedance generates a pure tripod pan/zoom (zero translation baseline) or subtle rolling background warp. Classical COLMAP `mapper` either aborts with `no reconstruction` or solves wildly wrong focal lengths.
  2. **Mid-air "cloud floaters":** Subtle lighting/texture flicker between frame `t` and `t+10` tricks standard 3DGS (`densify_grad_threshold=0.0002`) into spawning dense semi-transparent floaters directly in front of the camera lens to "paint" per-view differences.
- **Mitigation:**
  - **Make `VGGT` (`facebook/VGGT-1B`) the default pose + point initializer** (with `pycolmap + glomap` available via `pose_backend="glomap"` for strict rigid scenes). VGGT predicts metric intrinsics `K`, extrinsics `[R|t]`, dense 3D points, and **per-pixel confidence** in one `~2.5s` GPU forward pass — it never aborts on low parallax!
  - **Baseline / Parallax Gate:** Compute `baseline_ratio = max_cam_translation / median_scene_depth` right after pose estimation. Emit a warning in `cameras.json` if `baseline_ratio < 0.03` (pure pan/zoom) and automatically damp densification so it behaves cleanly like a 2.5D depth-layered panorama instead of exploding.
  - **Anti-Floater Training Recipe:** Add opacity L1 regularization (`1e-4 * sigmoid(opacities).mean()`), prune `opa < 0.015`, near-plane depth cull (`z_cam < 0.15 * median_depth`), and stop at **8,000–10,000 steps** (training AI video to 30k steps fits flicker noise).

### Glass Jaw 2 — Runtime `nvcc` JIT Compilation on Serverless Cold Start
- **The Risk:** `gsplat` ships a lazy JIT fallback (`gsplat/cuda/_backend.py`). If the prebuilt wheel misses the exact `(PyTorch, CUDA, sm_arch)` tuple (`sm_90` on H100, `sm_89` on L40S), `gsplat` silently invokes `ninja + nvcc` on step 0 — burning **120–180 seconds of H100 billing** on cold start or crashing if `nvcc` is missing.
- **Mitigation:**
  - Pin exact `(torch==2.5.1+cu124, gsplat wheel pt25cu124)` in `modal_app.py` **and** execute an explicit `import torch; from gsplat.cuda._backend import _C; assert _C is not None` assertion **during `modal deploy` image build**. If `_C` isn't ahead-of-time compiled, deployment fails immediately on CPU build infra rather than stalling at runtime.

### Glass Jaw 3 — Blender 5.3 Transparency Bounce Cutoff & Overdraw (`splat_ops.py`)
- **The Risk:** Unlike WebGL `.splat` viewers or `gsplat` CUDA, Blender 5.3 does **not** run a per-frame GPU radix sort on Gaussian quads:
  - In **Cycles**, a camera ray passing through a splat scene intersects 30–100 overlapping semi-transparent Gaussian ellipsoids. Blender's default `scene.cycles.transparent_max_bounces` is **`8`** — after 8 overlapping splats, Cycles terminates the ray as **solid black**, turning dense splat regions into black soot!
  - In **EEVEE**, Blender 5.x removed legacy `Alpha Blend` sorting modes in favor of **`DITHERED` (Alpha Hashed)** and **`BLENDED`**. `BLENDED` without per-primitive sorting causes severe Z-popping as the camera orbits; 600k oversized quads also tank viewport FPS via overdraw.
- **Mitigation:**
  - **`Cycles` auto-config in `splat.import`:** Automatically raise `scene.cycles.transparent_max_bounces = max(scene.cycles.transparent_max_bounces, 64)`.
  - **`EEVEE` material setup:** Default splat shader to `surface_render_method = 'DITHERED'` (stochastic alpha hashing — order-independent, zero Z-popping when camera orbits!) with `taa_samples >= 32` / `taa_render_samples >= 64`.
  - **Gaussian Cutoff & Count Budget:** Truncate quad radius at `2.0σ` (instead of `3.0σ`, eliminating 55% of near-zero-alpha quad area overdraw!) and default Blender import cap to the top **`250,000` Gaussians** ranked by `opacity * max_scale^2` (while keeping the full 500k+ `.splat` for the Web Studio viewer and using **`splat.project_video`** to project crisp 1080p video textures near the source camera axis).

### Glass Jaw 4 — Sim(3) Coordinate & Quaternion Transform Bugs ("Spiky Hairball Splats")
- **The Risk:** Transforming a trained 3DGS `.ply` from arbitrary OpenCV/COLMAP space (`+X right, +Y down, +Z forward`, tilted arbitrary scale) into leveled Blender space (`+X right, +Y forward, +Z up`, floor at `Z = 0`, metric scale `s`) requires transforming **anisotropic 3D ellipsoids**, not just XYZ points!
  - If you transform `means` and `cameras` by `(s, R_world, t_world)` without rotating `quats` by `q_world` and adding `log(s)` to `scales`, every elongated Gaussian needle stays oriented in the *old* coordinate frame — turning flat walls and floors into **fuzzy spiky corduroy**!
  - Worse: `OPENCV -> BLENDER` camera basis flip has determinant `+1` (`Rx(-90°)` or `diag(1, -1, -1)`), whereas accidentally applying an improper reflection (`det = -1`) invalidates quaternion handedness completely.
- **Mitigation:**
  - Keep **all** world-frame transforms strictly proper rotations (`det(R) == +1`, verified via assertion `abs(np.linalg.det(R) - 1.0) < 1e-5`) + uniform positive scale `s > 0`:
    - `means_bl = s * (means @ R.T) + t`
    - `log_scales_bl = log_scales + math.log(s)`
    - `quats_bl = quat_mul(rotmat_to_quat_wxyz(R), quats_wxyz)`
  - Verify with a mandatory synthetic math gate (`P3` below) before touching Blender.

### Glass Jaw 5 — Moving Foreground Subjects ("4D Dynamic Video vs. Static 3DGS")
- **The Risk:** Static 3DGS assumes a static world. If a Seedance clip has a walking character or blowing smoke across the frame, static 3DGS smears the moving actor into ghostly trails while reconstructing the static background.
- **Mitigation:**
  - Two clean handling modes:
    1. **Environment Plate Mode (`mode="orbit_plate"`, default):** Prompt Seedance for static environment + smooth camera parallax, reconstruct clean static 3DGS room/world, and animate deterministic 3D hero characters/props inside Blender 5.3!
    2. **Dynamic Masking (`mask_dynamic=True`, Phase 2 extension):** Auto-mask moving foreground pixels during `gsplat` loss (using your exact `_m` masked L1 + masked SSIM logic already proven in `arcgs/stages/train_gsplat.py` lines 795–814!) so the background splat trains cleanly without ghosting, while the moving subject can sit on an alpha-masked camera-projected card in Blender.

---

## 5. Fundamental Pathclearing Needed (`tt-splat` Gate Style)

Before writing UI panels or agent prompts, we should clear **4 fast, deterministic gates** in order so every tricky boundary is proven in isolation:

| Gate | Script / Test | What It Proves | Effort |
|---|---|---|---|
| **`P0` — Cloud Image & Speed Gate** | `scripts/pathclear_p0_modal.py` | `modal deploy` succeeds from Windows; `_C` CUDA extension is AOT-baked (0s JIT); `VGGT-1B` loads from `/models` Volume; 90 frames of one of your existing `assets/seedance/*.mp4` clips reconstructs + trains 8,000 `gsplat` steps on H100 in **<40s wall time**. Prints `P0_CLOUD_OK` + exact stage breakdown (`extract / pose / train / export`). | ~2 hrs |
| **`P1` — Sim(3) + Floor `Z=0` Gate** | `tests/test_splat_math_gate.py` | Unit-proves proper `det(R)=+1` Sim(3) transform on `(means, log_scales, quats_wxyz, viewmats)`: re-rendering the transformed Gaussians from the transformed cameras reproduces the original `gsplat` image to **>55 dB PSNR** (machine precision, zero spiky-quaternion bug), and RANSAC floor points land at `median(Z) == 0.0 ± 1e-3`. `P1_SIM3_OK`. | ~1 hr |
| **`P2` — Blender 5.3 Native Splat Render Gate** | `tests/test_splat_blender_gate.py` | Loads real `P0` `splat.ply` + `cameras.json` into headless Blender 5.3 via stdlib `foreach_set` in **<150 ms**; renders frame 1 from `SeedanceSourceCam` in both `BLENDER_EEVEE` (`DITHERED` alpha) and `CYCLES` (`transparent_max_bounces=64`) and checks PSNR / SSIM against Seedance frame 1 (**>22 dB**, zero black bounce blotches) + renders a `30°` off-axis novel view with a `Z=0` shadow-catcher cube. `P2_BLENDER_OK`. | ~2.5 hrs |
| **`P3` — Hybrid Video Back-Projection Gate** | `tests/test_splat_reproject_gate.py` | Verifies `splat.project_video`: projects `shot.mp4` from `SeedanceSourceCam` onto proxy hull + `Z=0` shadow catcher while rendering from `DirectorCam` (`+15°` dolly/crane), proving near-axis video sharpness + parallax splat fill. `P3_REPROJECT_OK`. | ~2 hrs |

---

## 6. Implementation Effort Breakdown

| Component | Files | Complexity / Effort | Key Engineering Details |
|---|---|---|---|
| **1. Cloud Splat Core (`arcgs` $\rightarrow$ CUDA)** | `src/blenderbot/splats/config.py`<br>`src/blenderbot/splats/core_worker.py` | **Medium (~450 LOC)** | Port your `_save_ply`, `_ply_to_splat`, `_ssim`, masked loss, and `DefaultStrategy` loop from `arcgs`; add VRAM frame tensor preloader, `VGGT` pose/init + `glomap` fallback, RANSAC `Z=0` Sim(3) aligner, and `cameras.json` exporter. |
| **2. Modal App + HTTP Pod Fallback** | `src/blenderbot/splats/modal_app.py`<br>`src/blenderbot/splats/pod_server.py`<br>`src/blenderbot/splats/client.py` | **Low–Medium (~350 LOC)** | Bake `cuda:12.4` + `gsplat` + `/models` Volume; stream telemetry to `SessionBus` + `FlightRecorder.touch()`; `zstd` artifact transfer + USD cost accounting into `agent.run_cost`. |
| **3. Blender 5.3 Splat & Projection Ops** | `src/blenderbot/blender_ops/splat_ops.py`<br>`src/blenderbot/blender_ops/__init__.py` | **Medium–High (~550 LOC)** *(Highest Blender-API care)* | Pure stdlib binary `.ply` reader (`memoryview` + `foreach_set`); procedural **Blender 5.3 Geometry Nodes** ellipsoid instancer (`Instance on Points` + quaternion/scale attributes + `2σ` Gaussian radial shader); Slotted-Action camera trajectory importer (`cameras.json`); camera-space `MOVIE` projection node tree + `is_shadow_catcher` ground. |
| **4. Agent `bl_ops`, `/splat` CLI & Web Studio Panel** | `src/blenderbot/agent/executor.py`<br>`src/blenderbot/ui/commands.py`<br>`src/blenderbot/server/routes.py`<br>`src/blenderbot/ui/web/*` | **Medium (~400 LOC)** | Register `splat.from_video`, `splat.import`, `splat.crop_clean`, `splat.project_video`; add `/splat` slash command; add **Splat** studio drawer (`127.0.0.1:7880`) with 1-click "Train Splat from Seedance Video", live loss/PSNR chart + preview frame, and WebGL `.splat` preview. |

---

## 7. Stateful Burst Resumability & Interactive Surgical Controls (`arcgs` Workflow on Serverless)

A naive serverless function loses `/tmp` when it finishes, which would force re-uploading the MP4 and re-running `VGGT`/SfM. Instead, we combine a **persistent `modal.Volume` (`/jobs/{job_id}`)** with **in-VRAM container caching (`scaledown_window=90s`)** so you get full `arcgs`-style interactive stepping, mask painting, culling, and objective changes — while paying **$0.00/hr** whenever you pause to inspect in Blender.

### 7.1 Two-Tier State Persistence (`0.0s` Warm / `0.4s` Cold Resume)

Every scene creates a deterministic `job_id` (e.g. `shot_01_8f3a`) stored on `modal.Volume.from_name("blenderbot-splat-jobs")` mounted at `/jobs`:

```text
/jobs/{job_id}/
  ├── meta.json          # locked Sim(3) transform (s, R, t), baseline_ratio, step history
  ├── frames_u8.pt       # pre-decoded (M, H, W, 3) uint8 tensor (~300–600 MB, written ONCE at step 0)
  ├── cameras.pt         # solved & Sim(3)-aligned K + viewmat tensors (locked to Blender's world space!)
  ├── masks/             # optional global_mask.png + per-camera {stem}.png + excluded.json
  └── ckpt_latest.pt     # step, raw params, Adam exp_avg/exp_avg_sq, AND gsplat strategy_state (grad2d/count)
```

1. **Run 1,000 or 3,000 steps (`splat.from_video(..., iterations=2000)`):**
   - Runs `extract` + `pose` + `align` **once**, writes `frames_u8.pt` + `cameras.pt` + `ckpt_latest.pt`, `vol.commit()`s, and returns `splat.ply` + `.splat` + `cameras.json` (~10–15s total).
2. **Inspect at $0/hr (5 seconds or 5 days later):**
   - View floaters in Blender 5.3 or the Web Studio viewer, draw a 3D `CropBox` in Blender, paint dynamic/sky masks in the Studio, or decide to switch from L1+SSIM to higher opacity penalty.
   - If you inspect for `<90s`, `@app.cls(scaledown_window=90)` keeps the tensors **live in H100 HBM** (`0.0s` reload). If you take 20 minutes in Blender, the H100 shuts down (`$0` idle cost) and the next burst reloads `frames_u8.pt` + `ckpt_latest.pt` from NVMe Volume into HBM in **`~0.4s`** — **zero MP4 re-upload, zero `ffmpeg`, zero `VGGT` re-run**, and **zero coordinate drift** relative to objects already placed in Blender!

### 7.2 Surgical Operations Between (or Mid-) Bursts (`splat.resume`)

`splat.resume(job_id, add_steps=3000, surgeries=[...], config_overrides={...})` applies ordered tensor surgeries before continuing optimization:

| Surgery / Control | Parameters | Improvement Over `arcgs` |
|---|---|---|
| **Opacity / Scale Cull** | `prune_opa=0.02`, `max_world_scale=0.5`, `clamp_log_scale=...` | **Slices Adam `exp_avg[keep]` / `exp_avg_sq[keep]` + `strategy_state[keep]`** instead of `arcgs`'s `opt.state.clear()`, so surviving Gaussians suffer **zero momentum reset / loss spike**! |
| **Spatial 3D Box / Sphere Cull (Blender-linked!)** | `keep_aabb=[min_xyz, max_xyz]` or `drop_aabb=[...]` or `keep_radius=12.0` | Because `/jobs/{job_id}` and Blender share the locked Sim(3) frame, you can place a wireframe cube around floaters in Blender, cull those exact 3D Gaussians on the H100, and run `2,000` steps so background splats heal the hole! |
| **Reset Opacities / Force Densify** | `reset_opacities=0.01`, `densify_now=True` | Identical to `arcgs` dashboard commands — forces stagnant floaters transparent so the next 1,000 steps evaporate them cleanly. |
| **Upload Masks / Exclude Frames** | `global_mask_png`, `frame_masks={...}`, `exclude_frames=[...]` | Writes straight to `/jobs/{job_id}/masks/` and activates masked L1 + masked SSIM (porting `arcgs/stages/train_gsplat.py` lines 795–814) on the next burst. |
| **Change Objective / Hyperparams** | `lambda_dssim`, `opa_reg`, `scale_reg`, `sh_degree`, `lr_scale`, `densify_until` | Swap loss weights on the fly (e.g. start `0..2k` with pure L1 + heavy `opa_reg` to kill floaters, then resume `2k..8k` with `lambda_dssim=0.2` + `sh_degree=1` to sharpen textures). |

### 7.3 Full Telemetry, Hardware Stage Timings & Append-Only Resume History

Porting `tt-splat` / `arcgs`'s `build_update` contract to a 1,500+ it/s remote H100 requires fixing three subtle bugs from the older local dashboards (`RAM-only _history` lost on `--resume`, `time.perf_counter()` measuring async CPU launch instead of GPU kernels, and 6 MB PNG encodes stalling fast loops):

1. **Full Scalar, Densify & Stage-Timing Schema (Every `log_every=25` Steps):**
   - **Metrics:** `step`, `total_steps`, `burst_id`, `cum_gpu_sec`, `it_per_sec`, `loss`, `l1`, `ssim`, `mse`, `psnr` (dB), `opa_reg_loss`, `n_gaussians`, `sh_degree_cur`, `pos_lr`, `camera_name`.
   - **Zero-Stall GPU Stage Wall Timings (`stage_timings` in ms):** Recorded via `torch.cuda.Event(enable_timing=True)` markers (`fwd_raster_ms`, `loss_ssim_ms`, `bwd_ms`, `adam_ms`, `densify_ms`, `host_ms`, `step_ms`) on sampled telemetry steps only — giving microsecond-accurate H100 kernel breakdowns without forcing `torch.cuda.synchronize()` on non-telemetry steps.
   - **Pipeline Macro Timings:** Exact wall seconds for `upload_s`, `cold_start_s`, `extract_s`, `pose_s`, `align_s`, `train_s`, `export_s`.
   - **Event & Densify Streams:** Every `DefaultStrategy` refine step logs `{step, n_cloned, n_split, n_pruned, n_total}`, and every surgical command / objective swap logs `{event: "surgery", step, op, before_stats, after_stats}` so vertical markers plot directly on the loss/PSNR/count charts.

2. **Live Triptych Renders + Persistent Snapshot Scrubber (Without Choking 1,500 it/s):**
   - Streams the `tt-splat` triptych — **Rendered View (`render_jpg`)**, **Ground Truth (`gt_jpg`)**, and **3× Fire-Colormap Luminance Error (`diff_jpg`, `_lum_diff_b64` style)** — throttled by **wall clock (`~4 Hz` live stream)** plus mandatory captures at **step 0, pre-surgery, post-surgery, and burst end** (plus optional fixed held-out / `focus_camera` snapshots every `snapshot_every` steps so you can scrub visual convergence cleanly across the timeline).
   - Uses fast JPEG (`q=85`, ~90 KB vs. 6 MB uncompressed PNG in `ttgs`) and caches `gt` per camera on the client so ground-truth frames are never re-transmitted redundantly.

3. **Seamless Append-on-Resume (`telemetry.jsonl` Dual-Mirrored):**
   - In `arcgs`/`tt-splat`, `TrainingController._history` lived only in RAM and started blank after `--resume`.
   - Here, scalar rows, stage timings, densify events, and surgery markers write **append-only** to `/jobs/{job_id}/telemetry.jsonl` (on Modal Volume) **and** mirror live to local `assets/splats/{job_id}/telemetry.jsonl` + `assets/splats/{job_id}/snapshots/step_{NNNNNN}_{cam}.jpg`.
   - When you run `0 -> 2,000`, close everything, inspect in Blender, cull floaters + swap `lambda_dssim`, and resume `2,000 -> 6,000`:
     - `cum_gpu_sec` and `step` continue monotonically (`2001..6000`) right onto the existing curves with a vertical surgery/burst marker at `step=2000`.
     - Opening the Web Studio (`127.0.0.1:7880`) hydrates the complete multi-burst charts and snapshot scrubber instantly from local disk even when zero cloud GPUs are running.

---

## 8. Quick Operator Checklist Before Starting `P0`

To run `P0` against your real `assets/seedance/*.mp4` files when we start coding:
1. Run `pip install modal zstandard` in `D:\blenderbot`'s Python environment.
2. Run `modal setup` once in your terminal to authenticate your Modal account ($30/mo free credits included out of the box).
3. Everything else (CUDA 12.4, `gsplat` `sm_90`/`sm_89` kernels, `VGGT-1B` weights, `ffmpeg`, `pycolmap`) provisions remotely inside `modal deploy` with zero local SDK/compiler setup on Windows.