// Center-Stage 3DGS Splat Job Launcher Dialog, Adaptive Strategy Arc, and Right-Inspector Splat Resume Panel.

import { actions, apiFetch, el, escapeHtml, formatBytes, showToast, state, withTokenUrl } from "./state.js";

export const arcStrategy = {
  u: 0.45,
  locked: true,
  scale: "log",
  gamma: 1.6,
  dragging: false,
};

const ARC_COUPLED_SPECS = [
  { rngId: "rng-arc-pruneopa", numId: "inp-splat-pruneopa", launchId: "launch-inp-pruneopa", v0: 0.005, v1: 0.060, decimals: 4 },
  { rngId: "rng-arc-opareg", numId: "inp-splat-opareg", launchId: "launch-inp-opareg", v0: 0.00008, v1: 0.0025, decimals: 5 },
  { rngId: "rng-arc-gradthresh", numId: "inp-splat-gradthresh", launchId: "launch-inp-gradthresh", v0: 0.00009, v1: 0.00060, decimals: 5 },
  { rngId: "rng-arc-budget", numId: "inp-splat-budget", launchId: "launch-inp-budget-k", v0: 650, v1: 140, decimals: 0 },
  { rngId: "rng-arc-opareset", numId: "inp-splat-opareset", launchId: "launch-inp-opareset", v0: 5000, v1: 2000, decimals: 0 },
];

const SPLAT_PRESETS = {
  iterative_arc: {
    iters: 12000,
    sh: 1,
    views_step: 4,
    corridor: 0.38,
    near_clear: 0.38,
    arc_cycles: 3,
    frames_cycle: 28,
    dp2_model: "auto",
    dp2_strength: 0.04,
    opareg: 0.00045,
    pruneopa: 0.018,
    gradthresh: 0.00020,
    scalereg: 0.0012,
    prunescale3d: 0.045,
    maxaniso: 6.0,
    anisoreg: 0.012,
    opareset: 3200,
    opacap: 5.5,
    budget_k: 360,
    max_frames: 192,
    vggt_frames: 96,
    conf_pct: 35,
    pose_backend: "vggt",
  },
  anti_floater: {
    iters: 6000,
    sh: 0,
    views_step: 4,
    corridor: 0.55,
    opareg: 0.0015,
    pruneopa: 0.035,
    gradthresh: 0.00035,
    scalereg: 0.0040,
    prunescale3d: 0.030,
    maxaniso: 4.5,
    anisoreg: 0.030,
    opareset: 2000,
    opacap: 4.2,
    budget_k: 220,
    max_frames: 150,
    vggt_frames: 64,
    conf_pct: 45,
    pose_backend: "vggt",
  },
  balanced: {
    iters: 5000,
    sh: 1,
    views_step: 2,
    corridor: 0.35,
    opareg: 0.0005,
    pruneopa: 0.020,
    gradthresh: 0.00020,
    scalereg: 0.0010,
    prunescale3d: 0.050,
    maxaniso: 6.0,
    anisoreg: 0.010,
    opareset: 2000,
    opacap: 5.5,
    budget_k: 350,
    max_frames: 96,
    vggt_frames: 64,
    conf_pct: 35,
    pose_backend: "vggt",
  },
  crisp_detail: {
    iters: 8000,
    sh: 1,
    views_step: 3,
    corridor: 0.30,
    opareg: 0.0004,
    pruneopa: 0.015,
    gradthresh: 0.00014,
    scalereg: 0.0010,
    prunescale3d: 0.045,
    maxaniso: 8.0,
    anisoreg: 0.010,
    opareset: 2500,
    opacap: 5.5,
    budget_k: 500,
    max_frames: 160,
    vggt_frames: 64,
    conf_pct: 25,
    pose_backend: "colmap",
  },
  fast_preview: {
    iters: 2000,
    sh: 0,
    views_step: 2,
    corridor: 0.40,
    opareg: 0.0008,
    pruneopa: 0.025,
    gradthresh: 0.00025,
    scalereg: 0.0015,
    prunescale3d: 0.045,
    maxaniso: 6.0,
    anisoreg: 0.015,
    opareset: 1500,
    opacap: 5.0,
    budget_k: 220,
    max_frames: 64,
    vggt_frames: 48,
    conf_pct: 40,
    pose_backend: "vggt",
  },
};

export function evalArcScale(v0, v1, uRaw, mode = arcStrategy.scale, gamma = arcStrategy.gamma) {
  const u = Math.max(0, Math.min(1, uRaw));
  if (mode === "log" && v0 > 0 && v1 > 0) return v0 * Math.pow(v1 / v0, u);
  if (mode === "power") return v0 + (v1 - v0) * Math.pow(u, Math.max(0.25, gamma));
  return v0 + (v1 - v0) * u;
}

export function invertArcScale(v0, v1, val, mode = arcStrategy.scale, gamma = arcStrategy.gamma) {
  const lo = Math.min(v0, v1), hi = Math.max(v0, v1);
  const clamped = Math.max(lo, Math.min(hi, val));
  if (Math.abs(v1 - v0) < 1e-12) return 0.5;
  if (mode === "log" && v0 > 0 && v1 > 0) {
    return Math.max(0, Math.min(1, Math.log(clamped / v0) / Math.log(v1 / v0)));
  }
  const lin = Math.max(0, Math.min(1, (clamped - v0) / (v1 - v0)));
  if (mode === "power") return Math.pow(lin, 1.0 / Math.max(0.25, gamma));
  return lin;
}

export function renderArcSvgAndControls(syncSlidersFromArc = true) {
  const u = Math.max(0, Math.min(1, arcStrategy.u));
  const x0 = 26, y0 = 88, cx = 160, cy = 12, x1 = 294, y1 = 88;
  const bx = (1 - u) * (1 - u) * x0 + 2 * (1 - u) * u * cx + u * u * x1;
  const by = (1 - u) * (1 - u) * y0 + 2 * (1 - u) * u * cy + u * u * y1;

  const handle = el("arc-handle"), glow = el("arc-handle-glow"), fillPath = el("arc-track-fill"), bgPath = el("arc-track-bg");
  if (handle) { handle.setAttribute("cx", bx.toFixed(1)); handle.setAttribute("cy", by.toFixed(1)); }
  if (glow) { glow.setAttribute("cx", bx.toFixed(1)); glow.setAttribute("cy", by.toFixed(1)); }
  if (fillPath && bgPath && bgPath.getTotalLength) {
    const totalLen = bgPath.getTotalLength();
    fillPath.style.strokeDasharray = `${(totalLen * u).toFixed(1)} ${totalLen.toFixed(1)}`;
  }

  const lawPath = el("arc-law-curve");
  if (lawPath) {
    const pts = [];
    for (let k = 0; k <= 24; k++) {
      const uk = k / 24;
      const px = x0 + (x1 - x0) * uk;
      const yNorm = evalArcScale(0.01, 1.0, uk, arcStrategy.scale, arcStrategy.gamma);
      const py = 88 - yNorm * 36;
      pts.push(`${k === 0 ? "M" : "L"} ${px.toFixed(1)} ${py.toFixed(1)}`);
    }
    lawPath.setAttribute("d", pts.join(" "));
  }

  const scaleTitle = arcStrategy.scale === "log" ? "Log" : arcStrategy.scale === "power" ? `Pow^${arcStrategy.gamma.toFixed(1)}` : "Linear";
  const lockLbl = arcStrategy.locked ? "Locked" : "Unlocked";
  if (el("lbl-arc-summary")) el("lbl-arc-summary").textContent = `${Math.round(u * 100)}% · ${scaleTitle}`;
  if (el("arc-center-readout")) el("arc-center-readout").textContent = `u = ${u.toFixed(2)} · ${scaleTitle} (${lockLbl})`;

  const lockBtn = el("btn-arc-lock");
  if (lockBtn) {
    lockBtn.textContent = arcStrategy.locked ? "🔒 Lock Arc" : "🔓 Free Sliders";
    lockBtn.classList.toggle("active-lock", arcStrategy.locked);
  }
  el("wrap-arc-gamma")?.classList.toggle("hidden", arcStrategy.scale !== "power");
  el("arc-scale-pills")?.querySelectorAll(".mode-pill").forEach((b) => {
    b.classList.toggle("active", b.dataset.arcscale === arcStrategy.scale);
  });

  if (syncSlidersFromArc) {
    ARC_COUPLED_SPECS.forEach((sp) => {
      const val = evalArcScale(sp.v0, sp.v1, u);
      const fmtVal = sp.decimals === 0 ? String(Math.round(val)) : val.toFixed(sp.decimals);
      const rng = el(sp.rngId), num = el(sp.numId), lInp = el(sp.launchId);
      if (rng) rng.value = String(Math.round(u * 1000));
      if (num) num.value = fmtVal;
      if (lInp) lInp.value = fmtVal;
    });
  }
}

export function populateSplatVideoDropdown() {
  const videos = state.mediaItems.filter((it) => it.kind === "video");
  ["sel-splat-video", "launch-sel-video"].forEach((selId) => {
    const sel = el(selId);
    if (!sel) return;
    const prev = state.launchVideoPath || sel.value;
    sel.innerHTML = `<option value="">— Select workspace video —</option>`;
    videos.forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.path;
      opt.textContent = `${it.path} (${formatBytes(it.size_bytes)})`;
      sel.appendChild(opt);
    });
    if (prev && videos.some((v) => v.path === prev)) sel.value = prev;
    else if (videos.length) sel.value = videos[0].path;
  });
  syncLaunchVideoPreview();
}

export function populateSplatImagesDropdown() {
  const counts = new Map();
  state.mediaItems
    .filter((it) => it.kind === "image" && typeof it.path === "string")
    .forEach((it) => {
      const norm = it.path.replace(/\\/g, "/");
      const slash = norm.lastIndexOf("/");
      const dir = slash > 0 ? norm.slice(0, slash) : ".";
      counts.set(dir, (counts.get(dir) || 0) + 1);
    });
  const folders = Array.from(counts.entries())
    .filter(([, n]) => n >= 3)
    .sort((a, b) => b[1] - a[1]);

  ["sel-splat-images-dir", "launch-sel-images-dir"].forEach((selId) => {
    const sel = el(selId);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = `<option value="">— Select workspace image folder (≥3 imgs) —</option>`;
    folders.forEach(([dir, n]) => {
      const opt = document.createElement("option");
      opt.value = dir;
      opt.textContent = `📁 ${dir} (${n} images)`;
      sel.appendChild(opt);
    });
    if (prev && folders.some(([d]) => d === prev)) sel.value = prev;
    else if (folders.length) sel.value = folders[0][0];
  });
}

export function populateKnownCamsDropdown() {
  ["sel-splat-known-cams", "launch-sel-known-cams"].forEach((selId) => {
    const sel = el(selId);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML =
      `<option value="">— Auto Solve Poses (VGGT / COLMAP) —</option>` +
      `<option value="__blender_active__">🎥 Active Blender Camera Prior</option>`;
    (state.exportedShots || []).forEach((sh) => {
      const o = document.createElement("option");
      o.value = `shot:${sh.shot_id}`;
      o.textContent = `🎮 FPS Rail: ${sh.shot_id}`;
      sel.appendChild(o);
    });
    Array.from(state.splatJobs.values()).forEach((j) => {
      if (j.cameras_rel) {
        const o = document.createElement("option");
        o.value = j.job_id;
        o.textContent = `📁 ${j.job_id} cameras.json`;
        sel.appendChild(o);
      }
    });
    if (prev) sel.value = prev;
  });
}

function syncLaunchVideoPreview() {
  const vPath = el("launch-sel-video")?.value || "";
  state.launchVideoPath = vPath;
  const vEl = el("launch-video-preview");
  const metaEl = el("launch-video-probe-meta");
  if (!vEl) return;

  const found = state.mediaItems.find((it) => it.kind === "video" && it.path === vPath);
  if (!found) {
    vEl.pause();
    vEl.removeAttribute("src");
    if (metaEl) metaEl.textContent = "Select a workspace video clip…";
    return;
  }
  vEl.poster = withTokenUrl(`/api/media/poster?path=${encodeURIComponent(found.path)}&v=${found.mtime || 0}`);
  vEl.src = withTokenUrl(found.url);
  if (metaEl) metaEl.textContent = `${found.path} · ${formatBytes(found.size_bytes)} (probing…)`;
  apiFetch(`/api/workspace/probe?path=${encodeURIComponent(found.path)}`)
    .then((info) => {
      if (state.launchVideoPath !== found.path || !metaEl) return;
      const dur = info.duration_sec ? `${Number(info.duration_sec).toFixed(1)}s` : "—";
      const fps = info.fps ? `${Number(info.fps).toFixed(1)} fps` : "—";
      const dim = info.width && info.height ? `${info.width}×${info.height}` : "";
      metaEl.textContent = `${dim} · ${dur} @ ${fps} · ${formatBytes(found.size_bytes)}`;
      if (el("launch-dlg-title")) el("launch-dlg-title").textContent = `Configure 3DGS Splat — ${found.name}`;
    })
    .catch(() => {});
}

export function setSplatSourceMode(mode) {
  state.splatSourceMode = mode === "blender" ? "blender" : mode === "images" ? "images" : "video";
  const isVid = state.splatSourceMode === "video";
  const isImg = state.splatSourceMode === "images";
  const isBl = state.splatSourceMode === "blender";

  el("launch-source-pills")?.querySelectorAll(".mode-pill").forEach((b) => {
    b.classList.toggle("active", b.dataset.lsrc === state.splatSourceMode);
  });
  el("launch-src-video-pane")?.classList.toggle("hidden", !isVid);
  el("launch-src-images-pane")?.classList.toggle("hidden", !isImg);
  el("launch-src-blender-pane")?.classList.toggle("hidden", !isBl);
  el("launch-extract-grid")?.classList.toggle("hidden", isBl);
  el("launch-extract-toggles")?.classList.toggle("hidden", !isVid);
  if (isImg) populateSplatImagesDropdown();
  if (isVid) syncLaunchVideoPreview();
  updateSplatCostEstimate();
}

export function updateSplatCostEstimate() {
  const iters = Math.max(200, Number(el("launch-inp-iters")?.value || el("inp-splat-iters")?.value || 5000));
  const gpu = el("launch-sel-gpu")?.value || el("sel-splat-gpu")?.value || "H100";
  const viewsStep = Number(el("launch-sel-views-step")?.value || 2);
  const rateHr = gpu === "L40S" ? 1.95 : 3.95;
  const poseSec = state.splatSourceMode === "blender" ? 2 : (el("launch-sel-pose-backend")?.value === "colmap" ? 28 : 12);
  const viewMult = 0.65 + 0.35 * viewsStep;
  const estSec = Math.round(poseSec + (iters / 2000) * (gpu === "L40S" ? 36 : 20) * viewMult);
  const costTxt = `~$${((estSec / 3600) * rateHr).toFixed(3)} (${gpu})`;

  if (el("launch-est-cost")) el("launch-est-cost").textContent = costTxt;
  if (el("launch-est-time")) el("launch-est-time").textContent = `~${estSec}s est`;
  if (el("splat-est-cost")) el("splat-est-cost").textContent = costTxt;
}

export function applySplatPreset(presetName) {
  const p = SPLAT_PRESETS[presetName];
  if (!p) return;
  document.querySelectorAll("[data-splat-preset]").forEach((b) => {
    b.classList.toggle("active", b.dataset.splatPreset === presetName);
  });
  const setV = (id, v) => { const n = el(id); if (n && v !== undefined) n.value = String(v); };
  setV("launch-inp-iters", p.iters);
  setV("launch-sel-sh", p.sh);
  setV("launch-sel-views-step", p.views_step);
  setV("launch-inp-corridor", p.corridor);
  setV("launch-inp-opareg", p.opareg);
  setV("launch-inp-pruneopa", p.pruneopa);
  setV("launch-inp-gradthresh", p.gradthresh);
  setV("launch-inp-scalereg", p.scalereg);
  setV("launch-inp-prunescale3d", p.prunescale3d);
  setV("launch-inp-maxaniso", p.maxaniso);
  setV("launch-inp-anisoreg", p.anisoreg);
  setV("launch-inp-opareset", p.opareset);
  setV("launch-inp-opacap", p.opacap);
  setV("launch-inp-budget-k", p.budget_k);
  setV("launch-inp-ext-maxframes", p.max_frames);
  setV("launch-inp-vggt-frames", p.vggt_frames);
  setV("launch-inp-conf-pct", p.conf_pct);
  setV("launch-sel-pose-backend", p.pose_backend);
  setV("launch-inp-arc-cycles", p.arc_cycles ?? 1);
  setV("launch-inp-frames-per-cycle", p.frames_cycle ?? 24);
  setV("launch-inp-near-clear", p.near_clear ?? 0.40);
  setV("launch-sel-dp2-model", p.dp2_model ?? "auto");
  setV("launch-inp-dp2-strength", p.dp2_strength ?? 0.25);
  setV("launch-inp-bimodal", p.bimodal ?? 0.0002);
  updateSplatCostEstimate();
}

export function openSplatLaunchDialog(sourceMode = "video", videoOrFolderPath = "") {
  actions.hideAllCenterViews?.();
  state.centerKind = "launch";

  populateSplatVideoDropdown();
  populateSplatImagesDropdown();
  populateKnownCamsDropdown();

  if (sourceMode === "video" && videoOrFolderPath) {
    state.launchVideoPath = videoOrFolderPath;
    if (el("launch-sel-video")) el("launch-sel-video").value = videoOrFolderPath;
  } else if (sourceMode === "images" && videoOrFolderPath) {
    if (el("launch-sel-images-dir")) el("launch-sel-images-dir").value = videoOrFolderPath;
    if (el("launch-inp-images-path")) el("launch-inp-images-path").value = videoOrFolderPath;
  }
  setSplatSourceMode(sourceMode || "video");

  const activeVid = el("launch-sel-video")?.value || "";
  el("stage-kind-badge").textContent = "3DGS LAUNCHER";
  el("stage-title").textContent =
    state.splatSourceMode === "video" && activeVid
      ? activeVid.split("/").pop()
      : "Configure & Queue Cloud 3D Gaussian Splat";
  el("theater-meta").textContent = "Launch Arguments";

  el("stage-3d-btns")?.classList.add("hidden");
  el("stage-media-btns")?.classList.add("hidden");
  el("theater-splat-launch-wrap")?.classList.remove("hidden");

  actions.syncInspectorContext?.();
  actions.syncViewModeButtons?.();
  actions.renderHierarchyTree?.();
}

export function collectSplatLaunchPayload() {
  const knownCamVal = el("launch-sel-known-cams")?.value || "";
  const budgetK = Number(el("launch-inp-budget-k")?.value || 350);
  const densUntilRaw = Number(el("launch-inp-dens-until")?.value || 0);

  const payload = {
    source_mode: state.splatSourceMode,
    gpu: el("launch-sel-gpu")?.value || "H100",
    iterations: Math.max(200, Math.min(30000, Number(el("launch-inp-iters")?.value || 5000))),
    sh_degree: Number(el("launch-sel-sh")?.value ?? 1),
    rasterize_mode: el("launch-sel-rast")?.value || "antialiased",
    lambda_dssim: Number(el("launch-inp-dssim")?.value ?? 0.2),
    target_gaussians: Math.max(50000, Math.round(budgetK * 1000)),
    densify_from: Number(el("launch-inp-dens-from")?.value || 400),
    densify_until: densUntilRaw > 0 ? densUntilRaw : undefined,
    densify_every: Number(el("launch-inp-dens-every")?.value || 100),
    densify_grad_threshold: Number(el("launch-inp-gradthresh")?.value || 0.0002),
    adaptive_strategy: (el("launch-chk-adaptive")?.checked ?? true) ? "adaptive_arc" : "fixed",
    arc_scale_mode: arcStrategy.scale,
    arc_gamma: arcStrategy.gamma,

    // Iterative Cycle Mode & Depth Pro 2 Surface Nudging
    iterative_arc_cycles: Number(el("launch-inp-arc-cycles")?.value || 1),
    frames_add_per_cycle: Number(el("launch-inp-frames-per-cycle")?.value ?? 24),
    near_frustum_clear_m: Number(el("launch-inp-near-clear")?.value ?? 0.40),
    depth_prior_model: el("launch-sel-dp2-model")?.value || "auto",
    depth_nudge_strength: Number(el("launch-inp-dp2-strength")?.value ?? 0.25),

    // Placement penalties & anti-overfitting guards
    views_per_step: Number(el("launch-sel-views-step")?.value || 2),
    cam_corridor_radius_m: Number(el("launch-inp-corridor")?.value ?? 0.35),
    opa_reg: Number(el("launch-inp-opareg")?.value ?? 0.0005),
    opa_bimodal_reg: Number(el("launch-inp-bimodal")?.value ?? 0.0002),
    prune_opa: Number(el("launch-inp-pruneopa")?.value ?? 0.02),
    scale_reg: Number(el("launch-inp-scalereg")?.value ?? 0.001),
    prune_scale3d: Number(el("launch-inp-prunescale3d")?.value ?? 0.05),
    max_anisotropy: Number(el("launch-inp-maxaniso")?.value ?? 6.0),
    aniso_reg: Number(el("launch-inp-anisoreg")?.value ?? 0.01),
    opacity_reset_every: Number(el("launch-inp-opareset")?.value ?? 2000),
    opacity_logit_cap: Number(el("launch-inp-opacap")?.value ?? 5.5),
    snapshot_every: Number(el("launch-inp-snapevery")?.value ?? 1000),
    appearance_opt: el("launch-chk-appopt")?.checked ?? true,
    outlier_view_downweight: el("launch-chk-outlier")?.checked ?? true,

    // Extract & SfM Pose options
    fps: Number(el("launch-inp-ext-fps")?.value || 8.0),
    max_frames: Number(el("launch-inp-ext-maxframes")?.value || 96),
    max_width: Number(el("launch-sel-ext-width")?.value || 1280),
    quality: Number(el("launch-inp-ext-quality")?.value || 95),
    dedup_frames: el("launch-chk-ext-dedup")?.checked ?? true,

    pose_backend: el("launch-sel-pose-backend")?.value || "vggt",
    colmap_matcher: el("launch-sel-matcher")?.value || "auto",
    vggt_max_frames: Number(el("launch-inp-vggt-frames")?.value || 64),
    conf_percentile: Number(el("launch-inp-conf-pct")?.value ?? 35),
    max_init_points: Number(el("launch-inp-init-pts")?.value || 180000),
    target_cam_height_m: Number(el("launch-inp-cam-height")?.value || 1.6),
    lock_pinhole_intrinsics: el("launch-chk-pinhole")?.checked ?? true,
    align_floor_z0: el("launch-chk-floor-z0")?.checked ?? true,
    refine_poses: el("launch-chk-refine-poses")?.checked ?? true,
    pose_opt_lr: Number(el("launch-inp-pose-lr")?.value || 0.00005),
    pose_prior_reg: Number(el("launch-inp-pose-reg")?.value || 0.01),

    // Export & Blender post-actions
    export_proxy_glb: el("launch-chk-proxy-glb")?.checked ?? true,
    proxy_grid_res: Number(el("launch-inp-proxy-res")?.value || 48),
    auto_import: el("launch-chk-auto-import")?.checked ?? false,
    render_mode: el("launch-sel-bl-mode")?.value || "ellipsoids_viewport",
  };

  if (state.splatSourceMode === "video") {
    payload.video_path = el("launch-sel-video")?.value || "";
    if (knownCamVal) payload.known_cameras_job = knownCamVal;
  } else if (state.splatSourceMode === "images") {
    const customFolder = (el("launch-inp-images-path")?.value || "").trim();
    const pickedFolder = el("launch-sel-images-dir")?.value || "";
    payload.images_path = customFolder || pickedFolder;
    if (knownCamVal) payload.known_cameras_job = knownCamVal;
  } else {
    payload.num_views = Number(el("launch-inp-bl-views")?.value || 28);
    payload.targets = (el("launch-inp-bl-targets")?.value || "").trim();
    payload.transparent_bg = el("launch-chk-bl-rgba")?.checked ?? true;
  }
  return payload;
}

export function buildSplatSnickerReviewPrompt(payload, probeMetaText = "") {
  const srcRef =
    payload.source_mode === "video" && payload.video_path
      ? `@${payload.video_path}`
      : payload.source_mode === "images" && payload.images_path
      ? `image folder \`${payload.images_path}\``
      : `live Blender 5.3 scene (${payload.num_views || 28} hemisphere views, RGBA=${payload.transparent_bg})`;

  return [
    `Run a strict **3DGS Parameter Snicker Test** on my proposed cloud splat launch configuration before I spend GPU time on ${srcRef}${probeMetaText ? ` (${probeMetaText})` : ""}:`,
    "```json",
    JSON.stringify(payload, null, 2),
    "```",
    "Audit every coupled subsystem (do NOT rubber-stamp — call out anything that fails the snicker test):",
    "1. **Frame Budget vs Clip Duration & Motion:** Does `(fps, max_frames, dedup_frames, vggt_max_frames, iterative_arc_cycles, frames_add_per_cycle)` actually cover the clip without blind whip-pan gaps or starving COLMAP/VGGT?",
    "2. **Registration, Floor `Z=0` & Depth Pro 2 (`ml-depth-pro-2`):** Is `pose_backend` (`vggt` vs `colmap` vs `colmap_prior`), `lock_pinhole_intrinsics`, `align_floor_z0`, `target_cam_height_m`, `refine_poses`, and `depth_prior_model` / `depth_nudge_strength` sensible for this capture so Blender registration & scale won't tilt or drift?",
    "3. **Schedule & Budget Coupling:** Check `iterations` vs `densify_from`, `densify_until` (auto 72% if omitted), `opacity_reset_every` (does a reset fire right before training stops?), and `target_gaussians` (will >400k bloat `.ply` / slow Blender 5.3 GeoNodes?).",
    "4. **Anti-Floater vs Over-Pruning Balance:** Check `views_per_step`, `sh_degree` (SH1/SH2 flicker overfitting risk?), `opa_reg`, `opa_bimodal_reg`, `prune_opa`, `scale_reg`, `prune_scale3d`, `max_anisotropy`, `cam_corridor_radius_m`, and `near_frustum_clear_m`.",
    "5. **Verdict (`PASS` / `TWEAK` / `FAIL`):** Give a crisp bullet list of any contradictions and exact recommended parameter overrides (do NOT launch the job automatically unless I confirm).",
  ].join("\n");
}

async function reviewSplatParametersWithAgent() {
  const payload = collectSplatLaunchPayload();
  if (payload.source_mode === "video" && !payload.video_path) {
    return showToast("Select a source video first so the Agent can inspect it.", true);
  }
  if (payload.source_mode === "images" && !payload.images_path) {
    return showToast("Select an image folder first.", true);
  }
  const probeMeta = (el("launch-video-probe-meta")?.textContent || "").trim();
  const promptText = buildSplatSnickerReviewPrompt(payload, probeMeta);

  const btns = [el("btn-review-splat-params"), el("btn-review-splat-bottom")].filter(Boolean);
  btns.forEach((b) => { b.disabled = true; });
  try {
    await actions.openSessionInCenter?.(state.activeSessionId || "active", false);
    await apiFetch("/api/chat/submit", { method: "POST", json: { text: promptText } });
    showToast("🧐 Sent 3DGS Snicker-Test review to Agent Session!");
  } catch (err) {
    showToast(`Parameter review failed: ${err.message}`, true);
  } finally {
    btns.forEach((b) => { b.disabled = false; });
  }
}

async function reviewResumeParametersWithAgent() {
  const jid = state.activeSplatJob?.job_id || actions.getActiveWbJobId?.();
  if (!jid) return showToast("Select a completed splat first.", true);
  const resumeSpec = {
    job_id: jid,
    add_steps: Math.max(200, Math.min(20000, Number(el("inp-resume-steps")?.value || 2000))),
    gpu: el("sel-resume-gpu")?.value || "H100",
    sh_degree: Number(el("sel-resume-sh")?.value ?? 1),
    views_per_step: Number(el("sel-resume-views")?.value || 2),
    opa_reg: Number(el("inp-resume-opareg")?.value ?? 0.0005),
    prune_opa: Number(el("inp-resume-pruneopa")?.value ?? 0.02),
    cam_corridor_radius_m: Number(el("inp-resume-corridor")?.value ?? 0.35),
    scale_reg: Number(el("inp-resume-scalereg")?.value ?? 0.001),
    max_anisotropy: Number(el("inp-resume-maxaniso")?.value ?? 6.0),
    target_gaussians: Math.max(50000, Math.round(Number(el("inp-resume-budget")?.value || 350) * 1000)),
    reset_opacities: (el("inp-resume-reset-opa")?.value || "").trim() || null,
    max_world_scale: (el("inp-resume-max-ws")?.value || "").trim() || null,
    lr_scale: Number(el("inp-resume-lr-scale")?.value || 1.0),
    focus_camera: el("sel-resume-focus-cam")?.value || null,
  };
  const promptText = [
    `Run a **3DGS Resume & Surgery Snicker Test** on existing splat job \`${jid}\` (\`assets/splats/${jid}/\`).`,
    "Inspect `job_source.json`, the tail of `telemetry.jsonl`, and `cameras.json` if needed, then audit these proposed continuation/surgery overrides:",
    "```json",
    JSON.stringify(resumeSpec, null, 2),
    "```",
    "Check whether more steps, culling (`max_world_scale` / `prune_opa` / `reset_opacities`), or `snap_arc` registration is the real bottleneck for usable Blender 5.3 quality, and give a `PASS / TWEAK / FAIL` recommendation with exact numbers.",
  ].join("\n");

  try {
    await actions.openSessionInCenter?.(state.activeSessionId || "active", false);
    await apiFetch("/api/chat/submit", { method: "POST", json: { text: promptText } });
    showToast(`🧐 Sent Resume Snicker-Test for '${jid}' to Agent Session!`);
  } catch (err) {
    showToast(err.message, true);
  }
}

async function submitSplatLaunchJob() {
  const payload = collectSplatLaunchPayload();
  if (state.splatSourceMode === "video" && !payload.video_path) {
    return showToast("Select a source video first.", true);
  }
  if (state.splatSourceMode === "images" && !payload.images_path) {
    return showToast("Select or enter an image folder first.", true);
  }

  const btns = [el("btn-queue-splat-hero"), el("btn-queue-splat-bottom"), el("btn-train-splat")].filter(Boolean);
  btns.forEach((b) => { b.disabled = true; });
  try {
    const res = await apiFetch("/api/splats/train", { method: "POST", json: payload });
    if (res.job) {
      const jid = res.job.job_id;
      state.splatJobs.set(jid, res.job);
      // Expand `jobs/cloud/modal-3dgs` in the Left Hierarchy Browser & select the newly launched job in Center Stage!
      if (state.filter !== "all" && state.filter !== "jobs") {
        state.filter = "all";
        document.querySelectorAll("#media-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.filter === "all"));
      }
      actions.expandTreePath?.("jobs/cloud/modal-3dgs");
      actions.handleSplatJobUpdate?.(res.job);
      await actions.openKernelTelemetryForJob?.(jid);
      actions.scrollActiveTreeRowIntoView?.();
    }
    showToast(`🚀 Queued Modal ${payload.gpu} 3DGS job: ${res.job?.job_id || ""}`);
  } catch (err) {
    showToast(err.message, true);
  } finally {
    btns.forEach((b) => { b.disabled = false; });
  }
}

export function syncInspectorActiveSplatPanel(splatJob) {
  const sec = el("sec-active-splat-resume");
  if (!sec) return;
  if (!splatJob) {
    sec.classList.add("hidden");
    return;
  }
  sec.classList.remove("hidden");
  sec.open = true;

  const stepsDone = splatJob.step || splatJob.total_steps || splatJob.iterations || 0;
  if (el("badge-active-splat-steps")) {
    el("badge-active-splat-steps").textContent = `${Number(stepsDone).toLocaleString()} steps`;
  }

  const cfg = splatJob.config || {};
  const tr = cfg.train || {};
  const po = cfg.pose || {};
  const setIf = (id, v) => { const n = el(id); if (n && v !== undefined && v !== null) n.value = String(v); };
  setIf("sel-resume-gpu", cfg.gpu || splatJob.gpu || "H100");
  if (tr.sh_degree !== undefined) setIf("sel-resume-sh", tr.sh_degree);
  if (tr.views_per_step !== undefined) setIf("sel-resume-views", tr.views_per_step);
  if (tr.opa_reg !== undefined) setIf("inp-resume-opareg", tr.opa_reg);
  if (tr.prune_opa !== undefined) setIf("inp-resume-pruneopa", tr.prune_opa);
  if (tr.cam_corridor_radius_m !== undefined) setIf("inp-resume-corridor", tr.cam_corridor_radius_m);
  if (tr.scale_reg !== undefined) setIf("inp-resume-scalereg", tr.scale_reg);
  if (tr.max_anisotropy !== undefined) setIf("inp-resume-maxaniso", tr.max_anisotropy);
  if (tr.target_gaussians !== undefined && tr.target_gaussians > 0) {
    setIf("inp-resume-budget", Math.round(tr.target_gaussians / 1000));
  }

  const sumBox = el("insp-splat-saved-summary");
  if (sumBox) {
    sumBox.innerHTML = `
      <div class="insp-row"><span class="insp-key">splat_id:</span><span class="insp-val ok">${escapeHtml(splatJob.job_id)}</span></div>
      <div class="insp-row"><span class="insp-key">source:</span><span class="insp-val">${escapeHtml(splatJob.source_video_rel || splatJob.source_label || "—")}</span></div>
      <div class="insp-row"><span class="insp-key">quality:</span><span class="insp-val">${Number(splatJob.n_gaussians || 0).toLocaleString()} pts · PSNR ${splatJob.psnr || "—"} dB · SSIM ${splatJob.ssim || "—"}</span></div>
      <div class="insp-row"><span class="insp-key">sfm_pose:</span><span class="insp-val">${escapeHtml(po.backend || "vggt")} · SH${tr.sh_degree ?? 1} · ${tr.views_per_step ?? 2} views/step</span></div>
    `;
  }
}

async function submitAdditionalSplatSteps() {
  const jid = state.activeSplatJob?.job_id || actions.getActiveWbJobId?.();
  if (!jid) return showToast("Select a completed splat first.", true);

  const addSteps = Math.max(200, Math.min(20000, Number(el("inp-resume-steps")?.value || 2000)));
  const gpu = el("sel-resume-gpu")?.value || "H100";
  const payload = {
    job_id: jid,
    add_steps: addSteps,
    gpu,
    sh_degree: Number(el("sel-resume-sh")?.value ?? 1),
    views_per_step: Number(el("sel-resume-views")?.value || 2),
    opa_reg: Number(el("inp-resume-opareg")?.value ?? 0.0005),
    prune_opa: Number(el("inp-resume-pruneopa")?.value ?? 0.02),
    cam_corridor_radius_m: Number(el("inp-resume-corridor")?.value ?? 0.35),
    scale_reg: Number(el("inp-resume-scalereg")?.value ?? 0.001),
    max_anisotropy: Number(el("inp-resume-maxaniso")?.value ?? 6.0),
    target_gaussians: Math.max(50000, Math.round(Number(el("inp-resume-budget")?.value || 350) * 1000)),
  };

  const resetOpaRaw = (el("inp-resume-reset-opa")?.value || "").trim();
  if (resetOpaRaw !== "") payload.reset_opacities = Number(resetOpaRaw);
  const maxWsRaw = (el("inp-resume-max-ws")?.value || "").trim();
  if (maxWsRaw !== "") payload.max_world_scale = Number(maxWsRaw);
  const lrScale = Number(el("inp-resume-lr-scale")?.value || 1.0);
  if (lrScale > 0 && Math.abs(lrScale - 1.0) > 1e-3) payload.lr_scale = lrScale;
  const focusCam = el("sel-resume-focus-cam")?.value || "";
  if (focusCam) payload.focus_camera = focusCam;

  const btn = el("btn-resume-splat-train");
  if (btn) btn.disabled = true;
  try {
    const res = await apiFetch("/api/splats/resume", { method: "POST", json: payload });
    if (res.job) {
      state.splatJobs.set(jid, res.job);
      actions.expandTreePath?.("jobs/cloud/modal-3dgs");
      actions.handleSplatJobUpdate?.(res.job);
      await actions.openKernelTelemetryForJob?.(jid);
      actions.scrollActiveTreeRowIntoView?.();
    }
    showToast(`➕ Resumed '${jid}' (+${addSteps} steps on ${gpu})`);
  } catch (err) {
    showToast(err.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

export async function importSplatToBlender(jobId) {
  showToast(`Importing '${jobId}' into Blender 5.3…`);
  try {
    const res = await apiFetch("/api/splats/import", {
      method: "POST",
      json: {
        job_id: jobId,
        render_mode: el("launch-sel-bl-mode")?.value || el("sel-splat-rendermode")?.value || "ellipsoids_viewport",
        max_points: 350000,
        import_cameras: true,
      },
    });
    showToast(`Imported '${res.imported?.object}' into Blender 5.3!`);
    actions.refreshState?.();
  } catch (err) {
    showToast(err.message, true);
  }
}

export async function importSplatCamerasOnlyToBlender(jobId) {
  showToast(`Importing solved cameras for '${jobId}' into Blender 5.3…`);
  try {
    const res = await apiFetch("/api/splats/import_cameras", {
      method: "POST",
      json: { job_id: jobId, overlay: "gt" },
    });
    showToast(`Imported camera rig '${res.cameras?.camera || "SeedanceSourceCam"}' into Blender 5.3!`);
    actions.refreshState?.();
  } catch (err) {
    showToast(err.message, true);
  }
}

export function wireAdaptiveStrategyArc() {
  const svg = el("svg-strategy-arc");
  if (!svg) return;

  const pointerToArcU = (clientX) => {
    const r = svg.getBoundingClientRect();
    const relX = ((clientX - r.left) / Math.max(1, r.width)) * 320;
    return Math.max(0, Math.min(1, (relX - 26) / (294 - 26)));
  };

  svg.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    arcStrategy.dragging = true;
    svg.setPointerCapture?.(e.pointerId);
    arcStrategy.u = pointerToArcU(e.clientX);
    renderArcSvgAndControls(true);
  });
  svg.addEventListener("pointermove", (e) => {
    if (!arcStrategy.dragging) return;
    arcStrategy.u = pointerToArcU(e.clientX);
    renderArcSvgAndControls(true);
  });
  const stopDrag = () => { arcStrategy.dragging = false; };
  svg.addEventListener("pointerup", stopDrag);
  svg.addEventListener("pointercancel", stopDrag);

  el("btn-arc-lock")?.addEventListener("click", () => {
    arcStrategy.locked = !arcStrategy.locked;
    renderArcSvgAndControls(false);
  });

  el("arc-scale-pills")?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-arcscale]");
    if (!b) return;
    arcStrategy.scale = b.dataset.arcscale;
    renderArcSvgAndControls(arcStrategy.locked);
  });

  el("inp-arc-gamma")?.addEventListener("input", (e) => {
    arcStrategy.gamma = Math.max(0.25, Math.min(4.0, Number(e.target.value || 1.6)));
    renderArcSvgAndControls(arcStrategy.locked);
  });

  ARC_COUPLED_SPECS.forEach((sp) => {
    const rng = el(sp.rngId), num = el(sp.numId);
    rng?.addEventListener("input", () => {
      const uLocal = Number(rng.value || 0) / 1000;
      if (arcStrategy.locked) {
        arcStrategy.u = uLocal;
        renderArcSvgAndControls(true);
      } else {
        const val = evalArcScale(sp.v0, sp.v1, uLocal);
        const fmt = sp.decimals === 0 ? String(Math.round(val)) : val.toFixed(sp.decimals);
        if (num) num.value = fmt;
        if (el(sp.launchId)) el(sp.launchId).value = fmt;
      }
    });
    num?.addEventListener("change", () => {
      const v = Number(num.value || sp.v0);
      const uInv = invertArcScale(sp.v0, sp.v1, v);
      if (arcStrategy.locked) {
        arcStrategy.u = uInv;
        renderArcSvgAndControls(true);
      } else if (rng) {
        rng.value = String(Math.round(uInv * 1000));
      }
    });
  });

  renderArcSvgAndControls(true);
}

export function initSplatLauncherAndResume() {
  wireAdaptiveStrategyArc();

  el("launch-source-pills")?.addEventListener("click", (e) => {
    const b = e.target.closest(".mode-pill");
    if (b?.dataset.lsrc) setSplatSourceMode(b.dataset.lsrc);
  });

  el("launch-sel-video")?.addEventListener("change", syncLaunchVideoPreview);

  document.querySelector(".splat-presets-bar")?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-splat-preset]");
    if (b?.dataset.splatPreset) applySplatPreset(b.dataset.splatPreset);
  });

  ["launch-inp-iters", "launch-sel-gpu", "launch-sel-views-step", "launch-sel-pose-backend"].forEach((id) => {
    el(id)?.addEventListener("input", updateSplatCostEstimate);
    el(id)?.addEventListener("change", updateSplatCostEstimate);
  });

  el("btn-queue-splat-hero")?.addEventListener("click", submitSplatLaunchJob);
  el("btn-queue-splat-bottom")?.addEventListener("click", submitSplatLaunchJob);
  el("btn-train-splat")?.addEventListener("click", submitSplatLaunchJob);
  el("btn-review-splat-params")?.addEventListener("click", reviewSplatParametersWithAgent);
  el("btn-review-splat-bottom")?.addEventListener("click", reviewSplatParametersWithAgent);
  el("btn-review-resume-params")?.addEventListener("click", reviewResumeParametersWithAgent);

  el("btn-open-center-launcher")?.addEventListener("click", () => openSplatLaunchDialog());
  el("btn-launch-how-it-works")?.addEventListener("click", () => actions.openHowItWorksInCenter?.());
  el("btn-open-how-from-insp")?.addEventListener("click", () => actions.openHowItWorksInCenter?.());

  // Active Splat Inspector Quick Bar & Resume (+Steps)
  el("btn-insp-splat-cmp")?.addEventListener("click", () => {
    if (state.activeSplatJob) actions.openComparatorForJob?.(state.activeSplatJob);
  });
  el("btn-insp-splat-status")?.addEventListener("click", () => {
    if (state.activeSplatJob) actions.openKernelTelemetryForJob?.(state.activeSplatJob.job_id);
  });
  el("btn-insp-import-bl")?.addEventListener("click", () => {
    if (state.activeSplatJob) importSplatToBlender(state.activeSplatJob.job_id);
  });
  el("btn-insp-cams-bl")?.addEventListener("click", () => {
    if (state.activeSplatJob) importSplatCamerasOnlyToBlender(state.activeSplatJob.job_id);
  });
  el("btn-resume-splat-train")?.addEventListener("click", submitAdditionalSplatSteps);

  actions.openSplatLaunchDialog = openSplatLaunchDialog;
  actions.collectSplatLaunchPayload = collectSplatLaunchPayload;
  actions.buildSplatSnickerReviewPrompt = buildSplatSnickerReviewPrompt;
  actions.jumpToSplatWithVideo = (videoPath) => openSplatLaunchDialog("video", videoPath);
  actions.populateSplatVideoDropdown = populateSplatVideoDropdown;
  actions.populateSplatImagesDropdown = populateSplatImagesDropdown;
  actions.populateKnownCamsDropdown = populateKnownCamsDropdown;
  actions.updateSplatCostEstimate = updateSplatCostEstimate;
  actions.syncInspectorActiveSplatPanel = syncInspectorActiveSplatPanel;
  actions.importSplatToBlender = importSplatToBlender;
}