# `unbot` Handoff — Unreal Engine 5.5–5.7+ Agentic 3D & Real-Time Studio Harness

> **Source Reference Repo:** `D:\blenderbot` (sibling to `D:\davai` and `D:\davag`)  
> **Target Repo:** `D:\unbot` (`unbot` CLI + Web Studio + Unreal Editor Python Bridge)

---

## 1. Current State of `blenderbot` (`D:\blenderbot`)

All 6 phases of `blenderbot` are complete, lint-clean, and verified end-to-end:
- **Blender 5.3+ Core Harness (`src/blenderbot/agent/`, `blender_ops/`, `bridge/`, `transport/`):**
  - Single fenced `bl_ops` protocol + verbatim `--blenderbot:NAME--` payload blocks + transactional `bpy.tx` rollback + unified `.blend`/workspace `Journal` (`/undo`, `/redo`).
  - Dual transport: `BridgeTransport` (`127.0.0.1:18766`, `~/.blenderbot/bridge.json` token-authenticated NDJSON RPC drained on Blender's main thread) + `HeadlessTransport` (warm `blender.exe -b` daemon).
- **Cloud 3DGS + Seedance 2.5 + Seedream 5.5 Loop (`splats/`, `media_gen/`):**
  - Serverless Modal H100/L40S `blenderbot-splat` worker (`VGGT-1B`, `COLMAP PINHOLE`, `colmap_prior`, `known_cameras`, Sim(3) `Z=0` RANSAC floor aligner, `se(3)` pose refinement, Mip-Splatting, Surface-Nets `proxy_collision.glb` mesh extractor, and farmed H.264 trajectory rendering).
  - OpenRouter `bytedance/seedance-2.5` video client (`media_gen/seedance.py`) & `bytedance-seed/seedream-5-5` image client (`media_gen/seedream.py`), auto-harvesting keys from `~/.blenderbot/.env`, `~/.davai/.env`, and `~/.davag/.env`.
- **VS Code / Chrome DevTools 3-Pane Web Studio (`server/`, `ui/web/{index.html,app.css,app.js}`, `:7880`):**
  - VS Code nested Explorer Hierarchy Tree (with indent guides, inline micro-thumbs, role badges, hover quick-action buttons, search/filter tabs, and dynamic floating video/image hover tooltips).
  - Pointer-captured resize splitters (`#sash-left`, `#sash-right`, `#sash-agent`) with snap-to-hide, expand-down `<details>` inspector accordions (`insp-section` / `insp-more`), and an always-present, hidden-by-default expandable **Agent Co-Pilot** bottom drawer (`Ctrl+J` / ``Ctrl+` ``).
  - Multi-pass WebGL2 3DGS + Solid Mesh + Minecraft Creative Flight FPS Editor (`/fps`) + 8-slot hotbar + Shot Collateral Pack exporter (`assets/shots/<shot_id>/`).

### Remaining Optional Backlog on `blenderbot` (Non-Blocking)
1. **Shared `keys.cfg` Rust-Style Console Macro System:** `F1` console + composable `bind <key> "<cmd1>;<cmd2>"` (`+` press/release actions) shared between Web FPS and `unbot` (`EnhancedInput`).
2. **2D In-Browser Mask Brush, 4D Splat Sequence Playback & Vast/RunPod HTTP Pod Wrapper (`pod_server.py`).**

---

## 2. `unbot` Architecture Blueprint (Direct Carry-Over from `blenderbot`)

### 2.1 Reusable Modules (Copy / Share Directly from `D:\blenderbot\src\blenderbot\`)
1. **Multi-Provider Streaming, Context & Session Spine (100% engine-agnostic):**
   - `providers/` (Anthropic, Gemini w/ `sthought` leak stripping, OpenAI Responses/Chat, OpenRouter + `watchdog.py` 180s/90s timeouts + `Esc` cancel).
   - `context/` (`Ledger`, `ArtifactStore`, `PasteStore`, compaction, multimodal image/video loaders).
   - `session/` (`SessionBus`, `flight.py` breadcrumbs, `history.py`, `keys_ctl.py` harvesting `~/.unbot/.env` -> `~/.blenderbot/.env` -> `~/.davai/.env` -> `~/.davag/.env`).
   - `agent/parser.py` — Add `ue_ops` as primary fenced block tag (accepting `bl_ops` / `da_ops` / `dr_ops` aliases) and `--unbot:NAME-- ... --unbot:end--` payload blocks.
2. **Media Generation, Cloud 3DGS & Web Studio UI (`media_gen/`, `splats/`, `ui/web/`):**
   - Reuse `media_gen/seedance.py` + `media_gen/seedream.py` unchanged.
   - Point `splats/client.py` at the **already-deployed** Modal app `blenderbot-splat` (zero duplicate cloud deploy needed!).
   - Port the VS Code / Chrome DevTools 3-pane Web Studio (`ui/web/{index.html,app.css,app.js}`), swapping Blender bridge endpoints (`/api/blender/*` on `:7880`) for Unreal Editor bridge endpoints (`/api/unreal/*` on **`127.0.0.1:7881`** so `blenderbot` and `unbot` Web Studios can run side-by-side simultaneously!).

### 2.2 Unreal Engine 5.5–5.7+ Dual Transport & Main-Thread Execution
- **Bridge Discovery & Port Contract:**
  - `blenderbot` uses `127.0.0.1:18766` (`~/.blenderbot/bridge.json`), Web Studio `:7880`.
  - **`unbot` uses `127.0.0.1:18767` (`~/.unbot/bridge.json`), Web Studio `:7881`.**
- **Inside Live Unreal Editor (`PythonScriptPlugin` Startup Script / Plugin):**
  - `unreal` API calls **must** execute on the Game/Slate main thread.
  - Spawn a background stdlib `ThreadingTCPServer` on `127.0.0.1:18767` (same token-authenticated NDJSON framing as `blenderbot/bridge/protocol.py`) pushing `(req_id, op, params, future)` onto a `queue.Queue`, and drain the queue on the Game thread via **`unreal.register_slate_post_tick_callback(drain_fn)`**!
  - Optional supplement: UE Remote Control HTTP/WebSocket API (`:30010`) for zero-latency live camera pose streaming (`/api/fps/camera_sync`) when Live-Link is active.
- **Headless Worker (`HeadlessTransport`):**
  - Spawn `UnrealEditor-Cmd.exe <Project.uproject> -run=pythonscript -script=... -NullRHI` (or with GPU enabled for headless Movie Render Queue / Path Tracer / Lumen captures!).
- **Transactional `ue.tx` & Undo/Redo Journal:**
  - Wrap mutating `ue_ops` batches and `ue.tx` scripts in `with unreal.ScopedEditorTransaction("unbot batch"):` + `unreal.EditorLoadingAndSavingUtils` / Transaction Buffer undo (`GEditor->Trans` / `unreal.SystemLibrary.execute_console_command(world, "TRANSACTION UNDO")`) paired with workspace file snapshots.

### 2.3 Core `ue_ops` Surface to Build First (Phase 1–2 of `unbot`)
- **Level & Inspection:** `level.passport` (actors, lights, cameras, SkyAtmosphere, PostProcessVolume, Sequencer tracks, bounds), `level.inspect`, `level.audit` (ground snap/penetration via `World.line_trace_single_by_channel`, bounds, Nanite/collision status, camera framing).
- **Actors, Greybox & Assets:** `actor.spawn` (StaticMesh, Shapes, Blueprints, Lights, CineCameraActor), `actor.transform`, `actor.duplicate`, `actor.delete`, `asset.import` (UE **Interchange** pipeline for `.glb`/`.gltf`, `.usd`/`.usda` Shot Packs from `blenderbot`, `.fbx`), `pcg.scatter` / foliage scatter.
- **Lumen Lighting, Post-Process & Materials:** `light.env` (`DirectionalLight` + `SkyAtmosphere` + `SkyLight` + `ExponentialHeightFog` + `VolumetricCloud` time-of-day/sun-rig), `light.upsert` (`RectLight`, `SpotLight`, `PointLight`), `material.instance` (`MaterialEditingLibrary` MI scalar/vector/texture params), `postprocess.configure` (exposure, bloom, vignette, ACES tone curve).
- **CineCamera, Sequencer & Movie Render Queue (MRQ):**
  - `camera.cine_upsert` (`CineCameraActor` filmback `36×20.25mm` 16:9, `current_focal_length`, `current_aperture`, manual/actor focus tracking).
  - `sequencer.upsert` / `sequencer.keyframe` (`LevelSequence` camera cuts, transform/focal-length tracks, skeletal animation clips, sub-scenes).
  - `viewport.render_still` (fast high-res viewport screenshot / SceneCapture2D PNG attached automatically to Vision LLM context), `viewport.contact_sheet`, and `mrq.render` (Lumen / Path Tracer MP4/EXR sequence via `MoviePipelineQueueSubsystem`).
- **3DGS Splat & Shot Pack Ingestion (`splat.import_ue`, `shot.import_ue`):**
  - Ingest `splat.ply` (Niagara `UNiagaraSystem` / `LumaAI` / `XScene` / custom Niagara GPU ribbon-sprite orient-to-camera splat emitter) + `proxy_collision.glb` (invisible Chaos complex-as-simple collision + Lumen contact-shadow receiver!) + `shot.usda` / `cameras.json` -> `LevelSequence` `CineCameraActor`!

---

## 3. Critical Mathematical Seam: Blender ↔ Unreal Coordinate, Quaternion & SH Handedness Contract

**Do not improvise this transform** — an improper reflection (`det = -1`) without basis-conjugating rotations and SH1 will mirror scenes or turn anisotropic 3DGS ellipsoids into exploded spikes!

### 3.1 Basis & Unit Definitions
- **Blender (`BLENDER_RH_ZUP_M`):** Right-handed, `+X` East/Right, `+Y` North/Forward, `+Z` Up, **meters (`1.0 = 1m`)**.
- **Unreal (`UNREAL_LH_ZUP_CM`):** Left-handed, `+X` Forward, `+Y` Right, `+Z` Up, **centimeters (`100.0 = 1m`)**.

The axis permutation matrix $P$ mapping Blender basis vectors to Unreal basis vectors ($X_{\text{ue}} = Y_{\text{bl}},\; Y_{\text{ue}} = X_{\text{bl}},\; Z_{\text{ue}} = Z_{\text{bl}}$) is symmetric, orthogonal, and **improper ($\det(P) = -1$)**:
$$P = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad P^{-1} = P^T = P, \quad \det(P) = -1$$

### 3.2 Exact Conversion Rules (`blender_to_unreal` / `unreal_to_blender`)
1. **3D Positions (`means`):**
   $$\mathbf{p}_{\text{ue}} = 100 \cdot (P\,\mathbf{p}_{\text{bl}}) = (100\,y_{\text{bl}},\; 100\,x_{\text{bl}},\; 100\,z_{\text{bl}})$$
2. **3DGS Log-Scales (`scale_0, scale_1, scale_2`):**
   - If converting `.ply` to centimeter units: $\mathbf{s}_{\log,\text{ue}} = \mathbf{s}_{\log,\text{bl}} + \ln(100)$ (note: keep local ellipsoid axis indices $(0, 1, 2)$ matched to the right-handed rotation matrix $R_{\text{ue}}$ below, or swap $s_0 \leftrightarrow s_1$ if post-multiplying by $P$!).
3. **3DGS Ellipsoid Orientation Quaternion (`rot_0..3` = `(w, x, y, z)`):**
   - A Gaussian's world covariance is $\Sigma_{\text{bl}} = R_{\text{bl}} S^2 R_{\text{bl}}^T$. Under $\mathbf{p}' = s P \mathbf{p}$, the covariance transforms as $\Sigma_{\text{ue}} = s^2 (P R_{\text{bl}}) S^2 (P R_{\text{bl}})^T$.
   - Because $\det(P R_{\text{bl}}) = -1$ (improper!), you **cannot** convert $P R_{\text{bl}}$ directly to a quaternion! Instead, insert $P$ on the right as well using $P^2 = I$:
     $$\Sigma_{\text{ue}} = s^2 (P R_{\text{bl}} P)(P S^2 P)(P R_{\text{bl}} P)^T$$
   - Now **$R_{\text{ue}} = P R_{\text{bl}} P$ has $\det(R_{\text{ue}}) = (-1)(+1)(-1) = +1$** (proper $SO(3)$!), while $(P S^2 P)$ simply swaps local scale axes **`scale_0` $\leftrightarrow$ `scale_1`**!
   - In quaternion form, conjugating by $P$ ($X \leftrightarrow Y$ reflection) maps `q_bl = (w, x, y, z)` directly in $O(1)$ closed form to:
     $$\mathbf{q}_{\text{ue}} = (w,\; -y,\; -x,\; -z) \equiv (-w,\; y,\; x,\; z) \quad \text{paired with swapping } (\text{scale}_0 \leftrightarrow \text{scale}_1)!$$
   - *(Verify: if $\mathbf{q}_{\text{bl}}$ is a $+90^\circ$ yaw around $+Z$, $R_{\text{ue}} = P R_z(+90^\circ) P = R_z(-90^\circ)$, and $(w, -y, -x, -z) = (\cos 45^\circ, 0, 0, -\sin 45^\circ)$, with $\Sigma_{\text{ue}} = R_{\text{ue}} \operatorname{diag}(s_1^2, s_0^2, s_2^2) R_{\text{ue}}^T \equiv P \Sigma_{\text{bl}} P$ to machine precision!)*
4. **Spherical Harmonics (`f_rest_*` Degree 1):**
   - Standard INRIA 3DGS SH1 bases correspond to $(Y_1^{-1} \propto y,\; Y_1^0 \propto z,\; Y_1^1 \propto x)$. Swapping world $X \leftrightarrow Y$ swaps the $x$ and $y$ SH1 coefficients ($c_{Y_1^{-1}} \leftrightarrow c_{Y_1^1}$).
5. **Camera Pose (`eye`, `forward`, `up` $\rightarrow$ UE `FRotator(Pitch, Yaw, Roll)`):**
   - Transform eye & look vectors via $P$:
     - $\text{Loc}_{\text{ue}} = (100\,y_{\text{bl}},\; 100\,x_{\text{bl}},\; 100\,z_{\text{bl}})$
     - $\mathbf{f}_{\text{ue}} = (f_{y,\text{bl}},\; f_{x,\text{bl}},\; f_{z,\text{bl}})$
   - In Unreal (`+X` forward, `+Y` right, `+Z` up):
     - $\text{Yaw}_{\text{ue}} = \operatorname{atan2}(f_{y,\text{ue}},\, f_{x,\text{ue}}) \cdot \frac{180}{\pi}$ (`0°` = `+X` = Blender `+Y`, `+90°` = `+Y` = Blender `+X`)
     - $\text{Pitch}_{\text{ue}} = \operatorname{asin}(f_{z,\text{ue}}) \cdot \frac{180}{\pi}$
     - Focal length `lens_mm` (`35mm` on `36mm` sensor width $\implies \text{HFOV} = 2\arctan(\frac{18}{\text{lens\_mm}})$) carries over 1:1 to `CineCameraComponent` (`sensor_width = 36.0`, `sensor_height = 20.25` for 16:9)!

---

## 4. Suggested Phase Roadmap for `unbot`

1. **Phase 1 — Unreal 5.5–5.7 Discovery, `unbot_bridge` Editor Plugin (`:18767`), `HeadlessTransport` & `ue.tx`:**
   - Auto-discover `UnrealEditor.exe` / `UnrealEditor-Cmd.exe` across Epic Games Launcher `LauncherInstalled.dat` & registry, auto-enable `PythonScriptPlugin` + `EditorScriptingUtilities` + `InterchangeEditor` + `MovieRenderPipeline`, and verify main-thread RPC + `ScopedEditorTransaction` rollback.
2. **Phase 2 — Core `ue_ops` (Actors, Lumen Env, Interchange USD/GLB Import, CineCamera, Sequencer, MRQ & Audits):**
   - Implement deterministic spatial raycast audits (`level.audit`), viewport still/contact-sheet vision loop, and 1-click `.glb`/`.usd` ingestion from `D:\blenderbot\assets\shots\`.
3. **Phase 3 — VS Code / Chrome 3-Pane Unreal Web Studio (`:7881`), FPS Sandbox & Cloud 3DGS / Seedance / Seedream Loop:**
   - Port `ui/web/` + `media_gen/` + `splats/` with the exact `BLENDER_RH_ZUP_M <-> UNREAL_LH_ZUP_CM` Sim(3)+Reflection converter (`unbot/splats/coords_ue.py` gated by a unit test verifying $\Sigma_{\text{ue}} = 10^4 P \Sigma_{\text{bl}} P$!).
4. **Phase 4 — Cross-Harness Peer Delegation (`unbot` ↔ `blenderbot` ↔ `davag`):**
   - `peer.blenderbot` op inside `unbot` to request bespoke `bmesh` procedural geometry/UVs/boolean bevels exported to shared `.glb`/`.usd`, imported straight into the active Unreal level via Interchange.