// Meta-parameter algorithm & expected-usage tooltips across Launcher, Resume, Mid-Run Inspector, and Composers.

import { el } from "./state.js";

export const META_PARAM_DOCS = {
  // ─── 3DGS Launcher: Card 1 (Input Source & Frame Extraction) ───
  "launch-sel-video": {
    title: "Source Video Clip (`video_path`)",
    range: "Workspace MP4 / MOV / WEBM",
    algo: "Primary input sequence for FFmpeg motion-adaptive keyframe extraction + `showinfo` PTS timestamp capture.",
    usage: "Pick clips with steady camera translation (baseline > 3% of scene depth); pure tripod pans lack 3D parallax.",
  },
  "launch-sel-images-dir": {
    title: "Workspace Image Folder (`images_path`)",
    range: "Folder with ≥ 3 overlapping images",
    algo: "EXIF-orients, downscales locally to `max_width`, and packs multi-view stills/RGBA PNGs into a lossless `.tar.gz` upload.",
    usage: "Use for photogrammetry turntables or multi-camera studio captures.",
  },
  "launch-inp-images-path": {
    title: "Custom Image Folder Path",
    range: "Relative workspace directory",
    algo: "Overrides the dropdown folder path when pointing to nested photo sets.",
    usage: "Leave blank to use the dropdown selection, or type e.g. `assets/inputs/statue_views`.",
  },
  "launch-inp-bl-views": {
    title: "Blender Turntable / Hemisphere Views (`num_views`)",
    range: "8 – 96 views (default 28)",
    algo: "Renders multi-angle Fibonacci hemisphere views with exact known `(R_w2c, t_w2c, K)` matrices + surface mesh seed points.",
    usage: "24–36 views hits >42 dB PSNR in ~15s on H100; raise to 48–64 for intricate occluded interiors.",
  },
  "launch-inp-bl-targets": {
    title: "Target Blender Objects (`targets`)",
    range: "Comma-separated object names | blank = full scene",
    algo: "Isolates specific objects during `splat.capture_views` and samples surface points directly from their evaluated triangles.",
    usage: "Leave blank to capture the whole scene, or specify hero objects for isolated RGBA prop splats.",
  },
  "launch-chk-bl-rgba": {
    title: "RGBA Alpha Silhouette Supervision (`transparent_bg`)",
    range: "Boolean (default ON)",
    algo: "Composites random RGB backgrounds each step and adds `0.15 · |α_render − α_gt|` loss so empty sky/background gets zero Gaussians.",
    usage: "Keep ON for isolated props/characters; turn OFF only when capturing enclosed rooms with world HDRI.",
  },
  "launch-inp-ext-fps": {
    title: "Extraction Sample Rate (`fps`)",
    range: "0.5 – 60.0 fps (default 8.0)",
    algo: "Base FFmpeg decode rate before motion-adaptive arc-length subsampling (`0.35·Δt + 0.65·MAD`).",
    usage: "8–12 fps captures fast whip-pans without starving slow walks; raise to 15–24 fps on rapid <5s clips.",
  },
  "launch-inp-ext-maxframes": {
    title: "Maximum Extracted Frames (`max_frames`)",
    range: "8 – 320 frames (default 96)",
    algo: "Caps total candidate frames stored in `frames_u8.pt` via motion-weighted arc-length selection.",
    usage: "96–160 for 10–20s clips; raise to 192–320 when using Iterative Arc Cycles on 45–60s walkthroughs.",
  },
  "launch-sel-ext-width": {
    title: "Training Frame Width (`max_width`)",
    range: "960 px – 1920 px (default 1280 px)",
    algo: "Lanczos-downscales training frames once before VRAM preload. Tile raster/backward cost scales linearly with pixel count.",
    usage: "1280px (720p) gives the best speed/sharpness sweet spot; use 1600–1920px only when pose error is sub-pixel.",
  },
  "launch-inp-ext-quality": {
    title: "Extracted Frame JPEG Quality (`quality`)",
    range: "70 – 100 (default 95)",
    algo: "Controls FFmpeg `-q:v` / Pillow 4:4:4 chroma subsampling when staging frames to `/dev/shm`.",
    usage: "Keep at 92–96 so 8×8 JPEG macroblock ringing doesn't spawn high-frequency SH noise.",
  },
  "launch-chk-ext-dedup": {
    title: "Static Frame Deduplication (`dedup_frames`)",
    range: "Boolean (`mpdecimate`, default ON)",
    algo: "Drops near-identical frames when the camera pauses while preserving exact `showinfo` `pts_sec` timestamps.",
    usage: "Keep ON for gameplay/walkthroughs with stops so stationary segments don't overweight SGD sampling.",
  },

  // ─── 3DGS Launcher: Card 2 (SfM Camera Pose & World Alignment) ───
  "launch-sel-pose-backend": {
    title: "SfM Camera Pose Backend (`pose.backend`)",
    range: "VGGT-1B | COLMAP | COLMAP+Prior | Hard Lock",
    algo: "Selects feed-forward metric `VGGT-1B` GPU transformer (sliding 40-frame Sim(3) windows), classical `COLMAP` SIFT bundle adjustment, Umeyama prior-locked COLMAP, or zero-drift `known_cameras`.",
    usage: "Use `COLMAP` on rigid textured rooms/walkthroughs for tight loop closure; use `VGGT-1B` on short, low-texture, or AI-generated clips.",
  },
  "launch-sel-known-cams": {
    title: "Camera Prior Trajectory (`known_cameras_job`)",
    range: "Auto Solve | Active Blender Cam | FPS Shot | Prior Job",
    algo: "Supplies external `(R_w2c, t_w2c, K)` matrices to hard-lock poses (`known_cameras`) or anchor scale/floor (`colmap_prior`).",
    usage: "Leave on Auto Solve for raw videos; pick Active Blender Cam or FPS Rail when round-tripping conditioned Seedance plates.",
  },
  "launch-sel-matcher": {
    title: "COLMAP Feature Matcher (`pose.matcher`)",
    range: "auto | sequential | exhaustive",
    algo: "Controls SIFT pair matching: `sequential` matches temporal neighbors + loop-closure stride; `exhaustive` tests all `O(M²)` pairs.",
    usage: "Leave `auto` (exhaustive ≤80 views, sequential >80), or force `exhaustive` on unordered photo folders.",
  },
  "launch-inp-vggt-frames": {
    title: "VGGT Max Attention Views (`vggt_max_frames`)",
    range: "16 – 128 views (default 64)",
    algo: "Number of anchor keyframes solved via VGGT-1B multi-view attention (stitched in 40-frame windows with 10-frame Sim(3) overlap). Remaining frames register during Iterative Arc Cycles.",
    usage: "Keep at 48–96; pushing single-shot attention too wide dilutes angular precision.",
  },
  "launch-inp-conf-pct": {
    title: "VGGT Low-Confidence Drop % (`conf_percentile`)",
    range: "0% – 85% (default 35%)",
    algo: "Discards the lowest N% confidence VGGT depth unprojections before seeding initial 3D Gaussians.",
    usage: "35–45% strips sky/silhouette bleed while keeping solid wall and floor seeds.",
  },
  "launch-inp-init-pts": {
    title: "Max Initial Seed Points (`max_init_points`)",
    range: "10,000 – 500,000 (default 180,000)",
    algo: "Caps initial Gaussian count after combining SfM/VGGT seeds + Depth Pro 2 dense wall/floor supplementation.",
    usage: "150k–200k gives fast early convergence without starving blank indoor walls.",
  },
  "launch-inp-cam-height": {
    title: "Target Eye Height Above Floor (`target_cam_height_m`)",
    range: "0.20m – 10.0m (default 1.60m)",
    algo: "Sets world metric scale `s` in Sim(3) alignment so median camera elevation above the RANSAC `Z = 0` floor equals this height.",
    usage: "1.60–1.70m for standing human/FPS eye level; 0.95m for crouch/tabletop; 3–6m for drone shots.",
  },
  "launch-sel-dp2-model": {
    title: "Apple Depth Pro 2 Prior (`depth_prior_model`)",
    range: "auto (B 386MB) | S (97MB) | B | L (1.3GB) | off",
    algo: "Runs `ml-depth-pro-2` in 16-frame cross-attention windows locked to `fx_px`, RANSAC-scales metric depth onto the Sim(3) sparse cloud, ray-nudges noisy seeds + supplements blank walls, and supervises `gsplat` `RGB+ED` expected depth.",
    usage: "Keep `auto` (`depthpro2-b`) on indoor/monocular videos to anchor textureless walls and push floaters out of free space; set `off` only for pure turntable RGBA props.",
  },
  "launch-inp-dp2-strength": {
    title: "Depth Prior Nudge Weight (`depth_nudge_strength`)",
    range: "0.00 – 1.50 (default 0.04–0.25)",
    algo: "Scales the asymmetric log-Huber `RGB+ED` expected-depth loss (2.5× stronger push when rendered Gaussians float closer than `0.88 · D_prior`, annealed over the first 65% of steps).",
    usage: "0.04–0.15 gently carves mid-air floaters while letting multi-view photometry win fine edges; >0.40 can fight slight monocular depth tilt.",
  },
  "launch-inp-pose-lr": {
    title: "Sub-Pixel Pose Opt LR (`pose_opt_lr`)",
    range: "5e-6 – 1e-3 (default 5e-5)",
    algo: "Adam learning rate for per-camera differentiable `se(3)` (`6D` rotation + `3D` translation) deltas inside `gsplat`, decaying 10× over training.",
    usage: "Keep at `3e-5 – 8e-5` to fix 1–3px VGGT/COLMAP jitter; >`3e-4` risks camera-splat co-drift.",
  },
  "launch-inp-pose-reg": {
    title: "Pose Prior L2 Tether (`pose_prior_reg`)",
    range: "0.0 – 1.0 (default 0.01)",
    algo: "Soft L2 penalty anchoring optimized `se(3)` camera deltas to the initial floor-aligned Sim(3) trajectory.",
    usage: "0.01 allows millimeter/sub-degree alignment snaps without letting the scene drift off `Z = 0`.",
  },
  "launch-chk-pinhole": {
    title: "Lock Distortion-Free Shared `PINHOLE` (`lock_pinhole_intrinsics`)",
    range: "Boolean (default ON)",
    algo: "Forces `fx = fy`, `cx = W/2, cy = H/2`, `k1 = k2 = 0` across all frames instead of 8-parameter per-frame `OPENCV` radial distortion.",
    usage: "Keep ON for rendered, game, and AI videos (prevents focal-length ↔ Z-dolly collinearity blowups); turn OFF only for raw fisheye/wide GoPro lenses.",
  },
  "launch-chk-floor-z0": {
    title: "Auto-Level Floor to `Z = 0` (`align_floor_z0`)",
    range: "Boolean (default ON)",
    algo: "Fits a deterministic 240-iter RANSAC plane to the bottom 40% scene points within 28° of camera-up and applies a proper `det(R)=+1` Sim(3) transform to means, log-scales, quaternions, and cameras.",
    usage: "Keep ON so imported splats land level on Blender's `Z = 0` ground grid; turn OFF only when using pre-registered world coordinates.",
  },
  "launch-chk-refine-poses": {
    title: "Sub-Pixel `se(3)` Pose Refinement (`refine_poses`)",
    range: "Boolean (default ON)",
    algo: "Jointly backpropagates photometric loss into continuous `6D+3D` camera extrinsics during 3DGS training and writes refined poses back to `cameras.json`.",
    usage: "Keep ON whenever poses come from VGGT or COLMAP (heals 518px attention quantization jitter); turn OFF for exact synthetic Blender renders.",
  },

  // ─── 3DGS Launcher: Card 3 (Optimization Schedule & Radiance) ───
  "launch-sel-gpu": {
    title: "Cloud GPU Hardware Tier (`gpu`)",
    range: "H100 80GB HBM3 ($3.95/h) | L40S 48GB ($1.95/h)",
    algo: "Selects the Modal cloud accelerator. 3DGS tile backward (`rasterize_to_pixels_bwd_kernel`) is L2 atomic & HBM bandwidth bound (~3.35 TB/s on H100 vs 864 GB/s on L40S).",
    usage: "Choose H100 for 720p/1080p rooms (~2.4× faster per step, often lower total cost); L40S for quick 2k previews.",
  },
  "launch-inp-iters": {
    title: "Training Iterations (`iterations`)",
    range: "200 – 30,000 (default 5,000)",
    algo: "Total SGD steps across your sampled camera views. Controls exponential `pos_lr` decay (`1.6e-4·scale → 1.6e-6·scale`) and locks densification after 72% of steps.",
    usage: "Use 5,000–7,000 on single-pass interior walkthroughs (30k overfits pose jitter into floaters); use 9,000–12,000 when running 3 Iterative Arc Cycles.",
  },
  "launch-sel-sh": {
    title: "Spherical Harmonics Radiance Degree (`sh_degree`)",
    range: "Degree 0 (Diffuse) | Degree 1 | Degree 2",
    algo: "Degree 0 stores view-invariant RGB (`3` floats); Degree 1 adds `9` directional lobes; Degree 2 adds `24` specular lobes (unlocked progressively every 1,000 steps).",
    usage: "Use Degree 0 on dark/flickering game or AI clips so splats cannot hide mid-air floaters in view-dependent lobes; Degree 1 for standard rooms; Degree 2 for glossy studio turntables.",
  },
  "launch-sel-rast": {
    title: "Rasterizer Filter (`rasterize_mode`)",
    range: "antialiased (Mip-Splatting) | classic",
    algo: "Mip-Splatting convolves a screen-space 2D EWA dilation filter and compensates opacity so sub-pixel Gaussians don't alias or pop into spikes when zooming in.",
    usage: "Keep `antialiased` for FPS walkthroughs and novel-view camera moves in Blender.",
  },
  "launch-inp-dssim": {
    title: "DSSIM Structural Loss Weight (`lambda_dssim`)",
    range: "0.00 – 1.00 (default 0.20)",
    algo: "Blends `(1 − λ)·L1 + λ·(1 − SSIM)` using separable 11×11 Gaussian window convolutions.",
    usage: "0.20 is standard; raise to 0.25–0.35 to sharpen architectural edges, or lower to 0.10 if subtle lighting flicker causes haloing.",
  },
  "launch-inp-budget-k": {
    title: "Soft Gaussian Budget (`target_gaussians`, thousands)",
    range: "50k – 1,000k (default 350k)",
    algo: "Closed-loop controller that smoothly raises `grow_grad2d` as Gaussian count approaches budget instead of hard-clipping.",
    usage: "220k–380k keeps Blender `PointCloud`/GeoNodes and WebGL2 FPS smooth; >700k bloats `.ply` above 70MB and slows Blender viewport.",
  },
  "launch-inp-arc-cycles": {
    title: "Iterative Arc Registration Cycles (`iterative_arc_cycles`)",
    range: "1 – 6 cycles (default 1)",
    algo: "When >1, alternates: `Train 3DGS → Differentiable Video-Diff se(3) Arc Solve + Swept Tunnel Carve + Active Frame Promotion → Train 3DGS`.",
    usage: "Set to `3` on long (>15s) walkthroughs so unposed gap frames register against the improving 3D splat and get promoted cleanly.",
  },
  "launch-inp-frames-per-cycle": {
    title: "Promoted Frames Per Cycle (`frames_add_per_cycle`)",
    range: "0 – 96 frames (default 24)",
    algo: "Promotes up to K candidate video frames into the active training set at each cycle boundary, gated by `L1 ≤ 1.85 · median_active_L1` so blurry/misaligned frames never poison gradients.",
    usage: "24–32 frames/cycle steadily fills wide whip-pan gaps across 2–3 cycles.",
  },
  "launch-inp-near-clear": {
    title: "Near-Lens Frustum Clear Distance (`near_frustum_clear_m`)",
    range: "0.00m – 1.50m (default 0.40m)",
    algo: "Evicts Gaussians sitting inside `0 < z_cam < near_frustum_clear_m` of 96 probe frustums along the swept camera trajectory.",
    usage: "0.35–0.50m prevents camera-hugging 'windshield' floaters; lower to 0.15m only for macro close-ups.",
  },
  "launch-inp-dens-from": {
    title: "Densify Start Step (`densify_from`)",
    range: "100 – 5,000 (default 400)",
    algo: "Warmup steps before `DefaultStrategy` starts cloning/splitting Gaussians, letting initial seeds + exposure/pose settle first.",
    usage: "400–600 prevents early pose/color transients from triggering premature splat explosions.",
  },
  "launch-inp-dens-until": {
    title: "Densify Stop Step (`densify_until`, 0 = auto 72%)",
    range: "0 (auto) | 600 – 30,000",
    algo: "Freezes Gaussian topology after this step so the final ~28% of iterations purely polish positions, covariance, SH, and prune haze.",
    usage: "Leave at `0` (auto 72% of `iterations`) so every run finishes with a dedicated settle & prune phase.",
  },
  "launch-inp-dens-every": {
    title: "Densify & Prune Interval (`densify_every`)",
    range: "25 – 500 steps (default 100)",
    algo: "Step cadence for `DefaultStrategy` clone/split/prune and corridor eviction.",
    usage: "100 is standard; use 150–250 on noisy COLMAP clouds so 2D gradients average over more views before splitting.",
  },
  "launch-chk-adaptive": {
    title: "Warmup-Protected Adaptive Schedule (`adaptive_strategy`)",
    range: "Boolean (`adaptive_arc` vs `fixed`, default ON)",
    algo: "Ramps `prune_opa` gently from `0.35×` target at start up to full `prune_opa` along the Log/Power/Linear arc, with post-opacity-reset grace.",
    usage: "Keep ON — prevents step-500 or post-reset opacity wipeouts on cameras that haven't been sampled yet.",
  },

  // ─── 3DGS Launcher: Card 4 (Placement Penalties & Anti-Overfitting Guards) ───
  "launch-sel-views-step": {
    title: "Stereo Cross-View Mini-Batch (`views_per_step`)",
    range: "1 – 4 cameras / step (default 2)",
    algo: "Rasterizes B wide-baseline overlapping stereo peers simultaneously in one batched `gsplat` launch.",
    usage: "Use 3–4 Views on interior camera arcs so any cheat billboard placed in front of Camera A immediately projects as an error across Camera B.",
  },
  "launch-inp-corridor": {
    title: "Swept Camera-Corridor Radius (`cam_corridor_radius_m`)",
    range: "0.00m – 1.50m (default 0.35m)",
    algo: "Samples 512 points along the continuous centripetal Catmull-Rom camera path, applying `0.25·relu(r − d)` repulsion every step plus hard eviction on densify.",
    usage: "0.35–0.55m carves a clean walkable tunnel along the camera path; set 0 only if the camera physically passes through foliage/smoke.",
  },
  "launch-inp-opareg": {
    title: "Opacity L1 Penalty (`opa_reg`)",
    range: "0.0 – 0.02 (default 0.0005)",
    algo: "Adds `opa_reg · mean(σ(opacities))` (+ bimodal `α(1−α)` entropy penalty) every step, pushing redundant semi-transparent room mist below `prune_opa`.",
    usage: "Increase to `0.0010–0.0020` when mid-air haze appears; >`0.005` will erode thin railings/foliage and drop PSNR.",
  },
  "launch-inp-pruneopa": {
    title: "Prune Minimum Opacity (`prune_opa`)",
    range: "0.002 – 0.20 (default 0.020)",
    algo: "Target opacity cutoff during `DefaultStrategy` densification ticks (slices Adam `exp_avg`/`exp_avg_sq` cleanly without loss spikes).",
    usage: "Use `0.018–0.035` for crisp interiors; >`0.06` can punch holes in dimly lit surfaces.",
  },
  "launch-inp-gradthresh": {
    title: "Grow |∇2D| Split/Clone Threshold (`densify_grad_threshold`)",
    range: "0.00004 – 0.0020 (default 0.00020)",
    algo: "Accumulated screen-space `absgrad` threshold above which Gaussians clone (small) or split (large).",
    usage: "0.00014–0.00020 for sharp rigid scenes; raise to 0.00030–0.00040 on noisy/AI video so flicker isn't densified into floaters.",
  },
  "launch-inp-scalereg": {
    title: "World Scale Penalty (`scale_reg`)",
    range: "0.0 – 0.05 (default 0.0010)",
    algo: "Penalizes `scale_reg · mean(exp(scales))` so Gaussians cannot bloat into 1m+ wall-spanning billboards that block hallways and serialize `gpuAtomicAdd` in tile backward.",
    usage: "Keep at `0.001–0.004` on indoor rooms so surfaces are represented by compact, well-localized splats.",
  },
  "launch-inp-prunescale3d": {
    title: "Prune 3D Scale Ratio (`prune_scale3d`)",
    range: "0.005 – 0.25 (default 0.050)",
    algo: "Evicts oversized Gaussians whose world radius exceeds `prune_scale3d · scene_scale`.",
    usage: "0.030–0.050 (~8–15cm max splat in a room) stops giant ceiling/corridor blotches that ruin Blender imports.",
  },
  "launch-inp-maxaniso": {
    title: "Max Needle Anisotropy (`max_anisotropy`)",
    range: "2.0× – 30.0× (default 6.0×)",
    algo: "Caps ellipsoid aspect ratio `s_max / s_mid` via `aniso_reg · relu(ratio − max_anisotropy)` so grazing views cannot stretch spiky needles into free space.",
    usage: "Set to `4.5–6.0×` for clean off-axis FPS flight and artifact-free Blender GeoNodes ellipsoids.",
  },
  "launch-inp-anisoreg": {
    title: "Needle Anisotropy Penalty Weight (`aniso_reg`)",
    range: "0.0 – 0.20 (default 0.010)",
    algo: "Multiplier on the hinge penalty `relu(s_max / s_mid − max_anisotropy)`.",
    usage: "0.010–0.030 keeps splats disc/ellipsoid-shaped instead of razor spikes.",
  },
  "launch-inp-opareset": {
    title: "Opacity Reset Interval (`opacity_reset_every`)",
    range: "0 (off) | 1,000 – 15,000 steps (default 2,000)",
    algo: "Periodically clamps opacities down toward ~0.01 during the densify window so occluded/redundant floaters must re-prove multi-view consistency or be pruned.",
    usage: "2,000–3,200 steps (must be `< densify_until` to take effect!). Avoid resetting right before training ends.",
  },
  "launch-inp-opacap": {
    title: "Opacity Logit Cap (`opacity_logit_cap`)",
    range: "2.5 – 10.0 (default 5.5 → α ≤ 0.996)",
    algo: "Clamps raw opacity logits so `σ(1 − σ)` never collapses to 0 when a misplaced floater hits `α = 1.0`.",
    usage: "4.2–5.5 keeps every Gaussian's opacity and behind-splat gradients alive throughout training.",
  },
  "launch-inp-bimodal": {
    title: "Bimodal Opacity Entropy (`opa_bimodal_reg`)",
    range: "0.0 – 0.005 (default 0.0002)",
    algo: "Adds `w · mean(α · (1 − α))` after step 500, pushing mid-opacity ghost haze (`α ≈ 0.08..0.45`) to either commit to a solid multi-view surface (`α → 1`) or decay cleanly below `prune_opa` (`α → 0`).",
    usage: "Keep at `0.0002–0.0005` on indoor rooms so multi-layer semi-transparent fog cannot linger; set `0` only for volumetric smoke/clouds.",
  },
  "launch-inp-snapevery": {
    title: "Timeline Snapshot Interval (`snapshot_every`)",
    range: "0 (off) | 250 – 5,000 steps (default 1,000)",
    algo: "Writes periodic `snapshots/step_NNNNNN.jpg` preview frames for the Job Status & Compare progress scrubbers.",
    usage: "500–1,000 lets you scrub visual convergence step-by-step.",
  },
  "launch-chk-appopt": {
    title: "Per-Camera 3×4 Exposure Absorption (`appearance_opt`)",
    range: "Boolean (default ON)",
    algo: "Learns a zero-mean-tethered `3×3` affine color matrix + `3D` bias per frame (`A_i·C + b_i`) so auto-exposure, bloom, and vignette shifts don't spawn air smoke.",
    usage: "Keep ON for real/game/AI videos with lighting or exposure shifts; auto-disabled on RGBA synthetic captures.",
  },
  "launch-chk-outlier": {
    title: "Cauchy IRLS Outlier-View Downweight (`outlier_view_downweight`)",
    range: "Boolean (default ON)",
    algo: "Downweights frames whose EMA loss exceeds `1.45×` the median camera loss (`1 / (1 + Δ²)` down to `0.20`) so 1–2 motion-blurred or HUD-occluded frames can't corrupt the cloud.",
    usage: "Keep ON for handheld, gameplay, and AI video captures.",
  },

  // ─── 3DGS Launcher: Card 5 (Export & Blender 5.3 Post-Actions) ───
  "launch-chk-proxy-glb": {
    title: "Export Collision Proxy Mesh (`export_proxy_glb`)",
    range: "Boolean (`proxy_collision.glb`, default ON)",
    algo: "Extracts a coarse 3D Surface-Nets iso-surface mesh (`<45 ms`, Z-up aligned) from trained Gaussian density.",
    usage: "Keep ON for Cycles shadow-catching, `splat.project_video` re-projection, and rigid-body/FPS collisions in Blender.",
  },
  "launch-inp-proxy-res": {
    title: "Proxy Voxel Grid Resolution (`proxy_grid_res`)",
    range: "24 – 96³ grid (default 48)",
    algo: "3D voxel resolution used by Surface-Nets when contouring Gaussian opacity density into `proxy_collision.glb`.",
    usage: "48 gives clean room/prop hulls (~600KB); raise to 64–80 for tighter architectural collision meshes.",
  },
  "launch-chk-auto-import": {
    title: "Auto-Import Completed Splat into Blender 5.3 (`auto_import`)",
    range: "Boolean (default OFF)",
    algo: "Automatically runs `splat.import` (`PointCloud` + GeoNodes + `SeedanceSourceCam` trajectory) on the connected Blender session when training finishes.",
    usage: "Enable when live-linked to Blender 5.3 for hands-free scene ingestion.",
  },
  "launch-sel-bl-mode": {
    title: "Blender 5.3 Splat Render Mode (`render_mode`)",
    range: "Smooth Ellipsoids | Hybrid Viewport | PointCloud",
    algo: "`Smooth Ellipsoids` instances 3DGS Mahalanobis ray-integral ellipsoids; `Hybrid` uses 60fps PointCloud in viewport + full ellipsoids on F12 render; `PointCloud` uses native spheres.",
    usage: "Use `Hybrid Viewport` on >250k splats so Blender's viewport stays 60fps while EEVEE/Cycles renders smooth 3DGS!",
  },

  // ─── Right Inspector: Active Splat Resume (+Steps) & Pre-Resume Surgery ───
  "inp-resume-steps": {
    title: "Additional Training Steps (`add_steps`)",
    range: "200 – 20,000 steps (default +2,000)",
    algo: "Resumes directly from `ckpt_latest.pt` + `frames_u8.pt` on the Modal volume (0s warm VRAM hit, preserves Adam momentum) with densification active for the first 65% of the burst.",
    usage: "Run +2,000–4,000 steps after culling floaters or tightening regularizers.",
  },
  "sel-resume-gpu": {
    title: "Resume Cloud GPU (`gpu`)",
    range: "H100 80GB | L40S 48GB",
    algo: "Selects Modal GPU container for the continuation burst.",
    usage: "Keep matched to the original GPU within 90s of completion to hit the 0.0s in-VRAM `ram_cache`!",
  },
  "sel-resume-sh": {
    title: "Resume Spherical Harmonics Degree (`sh_degree`)",
    range: "Degree 0 | Degree 1 | Degree 2",
    algo: "Zero-pads `shN` + fresh Adam state if raising degree, or restricts active SH bands if lowering.",
    usage: "Drop to Degree 0 if SH1 overfitted view-dependent color blotches, or step up 0 → 1 for final polish.",
  },
  "sel-resume-views": {
    title: "Resume Stereo Batch (`views_per_step`)",
    range: "1 – 4 views / step",
    algo: "Multi-view stereo peer batch size during the continuation burst.",
    usage: "Raise to 3–4 views when healing areas where single-view floaters were culled.",
  },
  "inp-resume-opareg": {
    title: "Resume Opacity L1 Penalty (`opa_reg`)",
    range: "0.0 – 0.02",
    algo: "Overrides continuous L1 opacity penalty for the resumed burst without clobbering other saved settings.",
    usage: "Bump to `0.0010–0.0020` on a polish burst to dissolve lingering room mist.",
  },
  "inp-resume-pruneopa": {
    title: "Resume Prune Opacity (`prune_opa`)",
    range: "0.002 – 0.20",
    algo: "Opacity eviction threshold during the resumed burst's densify/prune ticks.",
    usage: "Raise to `0.025–0.040` to strip ghost splats cleanly.",
  },
  "inp-resume-corridor": {
    title: "Resume Swept Corridor Radius (`cam_corridor_radius_m`)",
    range: "0.00m – 1.50m",
    algo: "Camera-tunnel free-space repulsion & eviction radius during the resumed burst.",
    usage: "Widen to `0.45–0.60m` if floaters remain near the walked camera path.",
  },
  "inp-resume-scalereg": {
    title: "Resume Scale Penalty (`scale_reg`)",
    range: "0.0 – 0.05",
    algo: "World-space linear scale regularizer during the resumed burst.",
    usage: "Increase to `0.002–0.005` if large blurry wall pancakes formed.",
  },
  "inp-resume-maxaniso": {
    title: "Resume Max Needle Anisotropy (`max_anisotropy`)",
    range: "2.0× – 30.0×",
    algo: "Maximum `s_max / s_mid` aspect ratio before `aniso_reg` penalizes needle stretching.",
    usage: "Tighten to `4.5–5.5×` before importing into Blender.",
  },
  "inp-resume-budget": {
    title: "Resume Soft Gaussian Budget (`target_gaussians`, k)",
    range: "50k – 1,000k",
    algo: "Target Gaussian population cap for the resumed burst.",
    usage: "Keep ≤350k if the splat grew above 500k and feels heavy in Blender.",
  },
  "inp-resume-reset-opa": {
    title: "Pre-Resume Reset Opacities (`reset_opacities`)",
    range: "Blank (off) | 0.005 – 0.50",
    algo: "Surgically resets all Gaussian opacities to this value at step 0 of the resume burst so trapped `α≈1` floaters must re-earn opacity across views.",
    usage: "Set `0.015–0.03` with `+2,000` steps to flush stubborn mid-air clutter.",
  },
  "inp-resume-max-ws": {
    title: "Pre-Resume Max World Scale Cull (`max_world_scale`)",
    range: "Blank (off) | 0.05m – 2.0m",
    algo: "Immediately slices out all Gaussians with `max(exp(scale)) > max_world_scale` (preserving Adam momentum on survivors) before training resumes.",
    usage: "Set `0.20–0.28`m to surgically delete giant room-spanning billboards and heal the background.",
  },
  "inp-resume-lr-scale": {
    title: "Resume Learning-Rate Multiplier (`lr_scale`)",
    range: "0.1× – 5.0× (default 1.0×)",
    algo: "Scales position & Adam learning rates across the continuation burst.",
    usage: "Use `0.4–0.7×` for gentle fine-detail polish, or `1.5–2.0×` after major CropBox culling so holes fill fast.",
  },
  "sel-resume-focus-cam": {
    title: "Focus Camera Override (`focus_camera`)",
    range: "All Views | Specific keyframe",
    algo: "Pins primary batch slot 0 to a chosen camera while stereo peers still regularize 3D consistency.",
    usage: "Select a worst-PSNR camera for 500 steps to heal a specific corner, then return to All Views.",
  },

  // ─── Right Inspector: Mid-Run Strategy Arc & Live Knobs ───
  "inp-splat-pruneopa": {
    title: "Target Densify Prune Opacity (`prune_opa`)",
    range: "0.002 – 0.12",
    algo: "Target opacity threshold for `DefaultStrategy` pruning along the Adaptive Strategy Arc.",
    usage: "Keep around `0.018–0.035` for floater-free interiors.",
  },
  "inp-splat-opareg": {
    title: "Opacity L1 Regularizer (`opa_reg`)",
    range: "0.00005 – 0.005",
    algo: "Continuous L1 penalty on `σ(opacities)` to evaporate mid-air haze.",
    usage: "Push to `0.001–0.002` if corridor fog appears.",
  },
  "inp-splat-gradthresh": {
    title: "2D Screen-Space Clone/Split Threshold (`grow_grad2d`)",
    range: "0.00006 – 0.0010",
    algo: "Screen-space gradient threshold (`absgrad=True`) triggering Gaussian clone/split.",
    usage: "Higher (`0.00035+`) prevents overfitting noisy/dark frames.",
  },
  "inp-splat-budget": {
    title: "Soft Gaussian Population Budget (`target_gaussians`)",
    range: "80k – 900k",
    algo: "Closed-loop controller scaling `grow_grad2d` smoothly as N approaches budget.",
    usage: "200k–350k keeps FPS flight at 60fps and prevents splat bloat.",
  },
  "inp-splat-opareset": {
    title: "Opacity Reset Cadence (`opacity_reset_every`)",
    range: "1,500 – 10,000 steps",
    algo: "Periodic opacity clamp interval coupled to the Adaptive Strategy Arc.",
    usage: "2,000–3,500 steps flushes occluded floaters during densification.",
  },
  "chk-splat-adaptive": {
    title: "Warmup-Protected Adaptive Schedule (`adaptive_arc`)",
    range: "Boolean (default ON)",
    algo: "Prevents early-step and post-opacity-reset mass pruning before all views are visited.",
    usage: "Keep ON during live mid-run pushes.",
  },
  "sel-splat-rendermode": {
    title: "Blender Import Render Mode",
    range: "Smooth Ellipsoids | Hybrid Viewport | PointCloud",
    algo: "Controls Geometry Nodes ellipsoid instancing vs fast native `PointCloud` viewport switch.",
    usage: "Pick `Hybrid Viewport` for fast 60fps Blender navigation + full ellipsoid renders.",
  },
  "sel-splat-sh": {
    title: "Live / Inspector SH Degree",
    range: "Degree 0 | Degree 1 | Degree 2",
    algo: "Active Spherical Harmonics order for view-dependent color.",
    usage: "Use Degree 0–1 on indoor video arcs.",
  },
  "inp-splat-dssim": {
    title: "Live DSSIM Structural Weight (`lambda_dssim`)",
    range: "0.00 – 1.00 (default 0.20)",
    algo: "Pushes updated SSIM vs L1 balance to the running Modal worker via `blenderbot-splat-ctl`.",
    usage: "0.20 standard; 0.30 for sharper structural edges.",
  },
  "inp-splat-vggtconf": {
    title: "VGGT Confidence Cull % (`conf_percentile`)",
    range: "0% – 85% (default 45%)",
    algo: "Drops lowest-confidence feed-forward points at initialization.",
    usage: "35–45% removes noisy sky/boundary points.",
  },
  "inp-splat-maxscale": {
    title: "Max World Splat Size (`max_world_scale`, m)",
    range: "0.05m – 1.50m (default 0.25m)",
    algo: "Upper world-space radius cap when importing or surgically clamping giant splats.",
    usage: "0.20–0.28m filters huge room-blocking pancakes on Blender import.",
  },
  "inp-splat-maxaniso": {
    title: "Max Needle Ratio (`max_anisotropy`)",
    range: "2× – 30× (default 10×)",
    algo: "Filters or penalizes elongated spiky splats (`s_max / s_mid`).",
    usage: "5–10× eliminates spiky corduroy artifacts in Blender.",
  },
  "sel-splat-views-step": {
    title: "Live Stereo Views / Step (`views_per_step`)",
    range: "1 – 4 views / step",
    algo: "Updates multi-view stereo batch size mid-training.",
    usage: "Push to 3–4 views if single-view floaters start forming.",
  },
  "inp-splat-corridor": {
    title: "Live Swept Corridor Radius (`cam_corridor_radius_m`)",
    range: "0.00m – 1.20m (default 0.35m)",
    algo: "Updates camera-path tunnel repulsion radius mid-run.",
    usage: "0.35–0.55m keeps the camera walkthrough path clear.",
  },
  "sel-splat-pose-backend": {
    title: "Inspector Pose Solver (`pose.backend`)",
    range: "VGGT-1B | COLMAP | COLMAP+Prior | Hard Lock",
    algo: "Selects Stage 2 camera solver when launching from the compact inspector.",
    usage: "Prefer COLMAP on rigid multi-room loops; VGGT-1B on short/low-parallax clips.",
  },

  // ─── Viewport, FPS Brush & Generative Studio Controls ───
  "sel-lens": {
    title: "Rectilinear Prime Focal Length (36mm Full-Frame)",
    range: "14mm – 135mm (default 35mm)",
    algo: "Sets horizontal FOV via `2 · atan(36 / (2 · f_mm))` in WebGL2 and synced Blender cameras.",
    usage: "Use 18–24mm inside tight rooms; 35–50mm for natural Seedance plates.",
  },
  "sel-splat-vis": {
    title: "Real-Time WebGL2 Splat Diagnostic Shader",
    range: "Smooth | Centers | Shrunk 0.35× | Z-Heatmap | X-Ray",
    algo: "Switches instanced WebGL2 visualization without altering `.ply` data.",
    usage: "Use `Floater / Needle X-Ray` or `Shrunk 0.35×` to spot oversized billboards before culling.",
  },
  "rng-brush-rad": {
    title: "3D Spherical Brush Radius (Meters)",
    range: "0.05m – 4.50m (`[` / `]` keys)",
    algo: "World-space radius tested against all Gaussian centers at 60fps.",
    usage: "Use 0.15–0.30m near walls; 0.8–2.0m with Air-Reach lock (`R`) to vacuum corridor floaters.",
  },
  "sel-meshy-topology": {
    title: "Mesh Polygon Topology (`topology`)",
    range: "triangle | quad",
    algo: "Selects Meshy 7.1 remesher (`triangle` for crisp static detail, `quad` for rigging/subsurf).",
    usage: "Use `triangle` for props; `quad` for rigged characters.",
  },
  "sel-duration": {
    title: "Seedance 2.5 Clip Duration (Seconds)",
    range: "Auto | 4s – 30s (~$0.1028/s)",
    algo: "Sets temporal latent length between 1st & Last anchor frames.",
    usage: "Use 4–6s for tight camera arcs; 8–15s for extended walkthroughs.",
  },
  "sel-sr-res": {
    title: "Seedream 5.5 Spatial Resolution Tier",
    range: "1K ($0.02) | 1.5K ($0.03) | 2K ($0.04)",
    algo: "Controls native diffusion pixel budget in `bytedance-seed/seedream-5-5`.",
    usage: "Use `1K` for ideation; `2K` for hero plates and Meshy inputs.",
  },
};

export function wireMetaParamTooltips() {
  const tt = el("param-tooltip");
  if (!tt) return;

  const showParamTip = (anchor, doc) => {
    el("pt-title").textContent = doc.title;
    el("pt-range").textContent = doc.range || "";
    el("pt-algo").textContent = doc.algo;
    el("pt-usage").textContent = doc.usage;
    tt.classList.remove("hidden");

    const r = anchor.getBoundingClientRect();
    const ttW = tt.offsetWidth || 336;
    const ttH = tt.offsetHeight || 175;
    let leftX = r.left - ttW - 12;
    if (leftX < 12) leftX = Math.min(window.innerWidth - ttW - 12, r.right + 10);
    if (leftX + ttW > window.innerWidth - 12) leftX = Math.max(12, window.innerWidth - ttW - 12);
    const topY = Math.max(52, Math.min(window.innerHeight - ttH - 12, r.top - 8));
    tt.style.left = `${leftX}px`;
    tt.style.top = `${topY}px`;
  };

  const hideParamTip = () => tt.classList.add("hidden");

  Object.entries(META_PARAM_DOCS).forEach(([ctrlId, doc]) => {
    const ctrl = el(ctrlId);
    if (!ctrl) return;
    const hostLabel = ctrl.closest("label") || ctrl.parentElement;
    if (!hostLabel || hostLabel.dataset.paramHelpWired === "1") return;
    hostLabel.dataset.paramHelpWired = "1";
    hostLabel.classList.add("has-param-help");

    const lblSpan = hostLabel.querySelector(":scope > span:first-child");
    if (lblSpan && !lblSpan.querySelector(".param-help-badge")) {
      const qBadge = document.createElement("span");
      qBadge.className = "param-help-badge";
      qBadge.textContent = "?";
      lblSpan.appendChild(qBadge);
    } else if (!lblSpan && hostLabel.tagName === "LABEL" && !hostLabel.querySelector(".param-help-badge")) {
      const qBadge = document.createElement("span");
      qBadge.className = "param-help-badge";
      qBadge.textContent = "?";
      hostLabel.appendChild(qBadge);
    }

    hostLabel.addEventListener("mouseenter", () => showParamTip(hostLabel, doc));
    hostLabel.addEventListener("mouseleave", hideParamTip);
    ctrl.addEventListener("focus", () => showParamTip(hostLabel, doc));
    ctrl.addEventListener("blur", hideParamTip);
  });

  document.querySelectorAll(".composer-scroll, .theater-kernel-wrap").forEach((sc) =>
    sc.addEventListener("scroll", hideParamTip)
  );
}