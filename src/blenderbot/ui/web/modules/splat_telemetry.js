// Splat Job Status Stage (Render/Error Triptych + Telemetry), Dedicated "How Does It Work?" Stage, and Trajectory Comparator.

import { arcStrategy } from "./splat_launcher.js";
import { actions, apiFetch, el, escapeHtml, formatBytes, showToast, state, withTokenUrl } from "./state.js";

export const wbState = {
  jobId: null,
  paused: false,
  ticks: [],
  densify: [],
  surgeries: [],
  snapshots: [],
  stages: [],
  cameras: [],
  diagnostics: null,
  config: {},
  previewUrl: null,
  diffUrl: null,
  selectedCartoonStage: "bwd",
};

export function activeWbJobId() {
  return wbState.jobId || state.activeSplatJob?.job_id || Array.from(state.splatJobs.keys())[0] || null;
}

const KERNEL_STAGE_EXPLAINERS = {
  act: {
    title: "① Parameter Activations (`sigmoid`, `exp`, `F.normalize`)",
    fusion: "UNFUSED PyTorch Pointwise ATen Kernels (3 short HBM round-trips over N Gaussians before `gsplat`)",
    hw: "1D elementwise CUDA grid over N (~324k) Gaussians; memory-bandwidth bound (~0.10–0.20 ms on H100 HBM3).",
    floor: "Sigmoid Jacobian `dσ/dx = σ(1 − σ)` collapses toward 0 when `opa < 0.01` or `opa > 0.99`. `opacity_logit_cap` prevents `α → 1` lockup.",
  },
  proj: {
    title: "② `gsplat::fully_fused_projection_fwd_kernel` + `cub::DeviceRadixSort`",
    fusion: "FUSED Projection + EWA Covariance + SH Eval (`packed=True`), followed by CUB 64-bit Radix Sort",
    hw: "1 CUDA thread per Gaussian transforms `means` to camera space, computes 3D covariance `Σ = R S Sᵀ Rᵀ`, projects EWA 2D conic (+ Mip-Splatting 2D filter), evaluates SH color, and compacts `M ≪ N` visible splats.",
    floor: "Packed mode skips off-frustum splats completely (`M` visible vs `N` total), cutting HBM traffic.",
  },
  fwd_rast: {
    title: "③ `gsplat::rasterize_to_pixels_fwd_kernel` (16×16 Tile Volume Renderer)",
    fusion: "FUSED Cooperative Shared-Memory Tile Rasterizer (`16×16 = 256` threads/block = 8 warps)",
    hw: "Each CUDA threadblock owns one 16×16 screen tile, burst-loads sorted 2D splats into Shared Memory (SMEM), and marches front-to-back: `C += T·α·c`, `T *= (1 − α)`.",
    floor: "Hard transmittance floor: when `T < 1e-4`, blending terminates (`T_floor_sat_frac`), so occluded splats behind dense walls get zero gradient.",
  },
  loss: {
    title: "④ Photometric L1 + 11×11 Gaussian SSIM + Corridor/Opacity/Scale Penalties",
    fusion: "Separable 1D `1×11` + `11×1` Gaussian SSIM + Multi-View Stereo Batch (`views_per_step = 2..4`)",
    hw: "Evaluates L1 + separable SSIM across `B` stereo cameras simultaneously, plus `opa_reg`, `scale_reg`, `aniso_reg`, and `cam_corridor_radius_m` free-space repulsion.",
    floor: "Stereo batching (`B ≥ 2`) punishes near-lens cheat floaters immediately across wide-baseline peer views.",
  },
  bwd: {
    title: "⑤ `gsplat::rasterize_to_pixels_bwd_kernel` + `projection_bwd` (DOMINANT ~72% STEP TIME)",
    fusion: "FUSED Reverse-Order Tile Backward (`absgrad=True`) with Warp-Shuffle Reduction + Global `gpuAtomicAdd`",
    hw: "Walks tile intersections BACK-TO-FRONT, recovering `T_prev = T_cur / max(1 − α, 1e-6)` in registers, warp-reducing gradients, and issuing `gpuAtomicAdd` into global buffers.",
    floor: "Inside an interior room, giant 1m+ wall splats overlap dozens of 16×16 tiles and serialize `gpuAtomicAdd` (`scale_reg` & `prune_scale3d` keep splats compact!).",
  },
  adam: {
    title: "⑥ 6× Per-Parameter Fused CUDA Adam (`betas=(0.9, 0.999)`, `eps=1e-15`, `fused=True`)",
    fusion: "FUSED Per-Group Multi-Tensor CUDA Adam Kernel",
    hw: "6 independent fp32 Adam states (`means`, `scales`, `quats`, `opacities`, `sh0`, `shN`) with `eps=1e-15` and exponential `pos_lr` decay.",
    floor: "Surgical resume (`splat.resume`) slices `exp_avg` & `exp_avg_sq` in-place so pruning never spikes loss.",
  },
  densify: {
    title: "⑦ `DefaultStrategy` Densify (Clone / Split / Prune) & Corridor Eviction",
    fusion: "Periodic Graph Topology Mutation (`densify_every=100` up to `72%` of iterations)",
    hw: "Accumulates `|∇_2D|` (`absgrad=True`) to clone/split under-reconstructed regions while pruning low-opacity haze, huge billboards, and camera-corridor intrusions.",
    floor: "Warmup-protected multi-view reset grace prevents unobserved cameras from losing splats right after an opacity reset.",
  },
};

export function renderCartoonExplainer(stageKey) {
  wbState.selectedCartoonStage = stageKey || wbState.selectedCartoonStage || "bwd";
  const svg = el("svg-kdiag-cartoon");
  if (svg) {
    svg.querySelectorAll(".kstage-node").forEach((g) => {
      g.classList.toggle("active", g.getAttribute("data-kstage") === wbState.selectedCartoonStage);
    });
  }
  const info = KERNEL_STAGE_EXPLAINERS[wbState.selectedCartoonStage] || KERNEL_STAGE_EXPLAINERS.bwd;
  const box = el("kdiag-stage-explainer");
  if (!box) return;
  box.innerHTML = `
    <div class="kdiag-exp-col">
      <h4>${escapeHtml(info.title)}</h4>
      <p><strong>Fusion Config:</strong> ${escapeHtml(info.fusion)}</p>
    </div>
    <div class="kdiag-exp-col">
      <h4>GPU Thread / Memory Mapping</h4>
      <p>${escapeHtml(info.hw)}</p>
    </div>
    <div class="kdiag-exp-col">
      <h4>Noise-Floor &amp; Anti-Floater Impact</h4>
      <p>${escapeHtml(info.floor)}</p>
    </div>
  `;
}

function renderStageTimingBar(barEl, legendEl, timings) {
  if (!barEl || !timings) return;
  const defs = [
    ["param_act_ms", "Act", "#a855f7"],
    [timings.fwd_raster_only_ms != null ? "fwd_raster_only_ms" : "fwd_raster_ms", "Fwd", "#38bdf8"],
    ["loss_ssim_ms", "Loss+SSIM", "#eab308"],
    ["bwd_ms", "Bwd", "#f97316"],
    ["adam_ms", "Adam", "#22c55e"],
    ["densify_ms", "Dens", "#ec4899"],
    ["host_ms", "Host", "#64748b"],
  ];
  let tot = 0;
  const items = [];
  defs.forEach(([k, lbl, col]) => {
    const v = Number(timings[k] || 0);
    if (v > 0) { tot += v; items.push({ k, lbl, col, v }); }
  });
  if (tot <= 0) return;
  barEl.innerHTML = items
    .map((it) => `<div class="splat-stage-seg" style="width:${((it.v / tot) * 100).toFixed(1)}%;background:${it.col}" title="${it.lbl}: ${it.v.toFixed(2)} ms (${((it.v / tot) * 100).toFixed(1)}%)"></div>`)
    .join("");
  if (legendEl) {
    legendEl.innerHTML = items
      .map((it) => `<span><i class="kdiag-swatch" style="background:${it.col}"></i>${it.lbl} ${it.v.toFixed(2)}ms</span>`)
      .join("");
  }
}

function computeEmaSeries(arr, alpha = 0.1) {
  const out = [];
  let cur = null;
  for (let i = 0; i < arr.length; i++) {
    const v = Number(arr[i] || 0);
    cur = cur === null ? v : (1 - alpha) * cur + alpha * v;
    out.push(cur);
  }
  return out;
}

function renderMiniKernelDiag(latestTick) {
  const mini = el("splat-kdiag-mini");
  if (!mini) return;
  const st = latestTick?.stage_timings;
  if (st) {
    el("splat-stage-wrap")?.classList.remove("hidden");
    renderStageTimingBar(el("splat-stage-bar"), el("splat-stage-legend"), st);
  }
  const kd = latestTick?.kernel_diag;
  if (!kd) {
    if (wbState.diagnostics?.findings?.length) {
      mini.classList.remove("hidden");
      const f0 = wbState.diagnostics.findings[0];
      mini.innerHTML = `<div>🔎 <b>${escapeHtml(f0.subsystem)}:</b> ${escapeHtml(f0.evidence.slice(0, 125))}…</div>`;
    } else {
      mini.classList.add("hidden");
    }
    return;
  }
  mini.classList.remove("hidden");
  const gm = kd.grads?.means || {};
  const g2d = kd.grads?.means2d_absgrad || {};
  const rs = kd.raster || {};
  const ad = kd.adam?.means || {};
  const vs = kd.view_spread || {};
  mini.innerHTML = `
    <div>∇means rms <b>${Number(gm.rms || 0).toExponential(2)}</b> (nz ${(Number(gm.nz_frac || 0) * 100).toFixed(0)}%) · |∇2D| p95 <b>${Number(g2d.p95 || 0).toExponential(2)}</b></div>
    <div>Tile isects <b>${Number(rs.tile_isects || 0).toLocaleString()}</b> (${rs.avg_splats_per_tile || 0}/tile) · T-floor <b>${(Number(rs.T_floor_sat_frac || 0) * 100).toFixed(1)}%</b> · Adam SNR <b>${ad.snr_p50 ?? "—"}</b></div>
    ${vs.tracked_cams ? `<div>EMA PSNR <b>${vs.mean_psnr}dB</b> [worst ${escapeHtml(vs.worst_cam || "")}: ${vs.min_psnr}dB · best: ${vs.max_psnr}dB]</div>` : ""}
  `;
}

export function drawWbCharts() {
  const ticks = wbState.ticks;
  const c1 = el("canvas-splat-loss-psnr"), c2 = el("canvas-splat-count");
  if (!ticks.length || !c1 || !c2) return;
  el("splat-charts-wrap")?.classList.remove("hidden");
  const g1 = c1.getContext("2d"), W = c1.width, H = c1.height;
  g1.clearRect(0, 0, W, H);
  const rawL = ticks.map((t) => Number(t.loss || 0));
  const rawP = ticks.map((t) => Number(t.psnr || 0));
  const emaL = ticks.map((t, i) => (t.loss_ema != null ? Number(t.loss_ema) : computeEmaSeries(rawL, 0.1)[i]));
  const emaP = ticks.map((t, i) => (t.psnr_ema != null ? Number(t.psnr_ema) : computeEmaSeries(rawP, 0.1)[i]));

  const maxL = Math.max(0.01, ...rawL), minP = Math.min(...rawP, 12), maxP = Math.max(minP + 2, ...rawP);
  const xOf = (i) => 10 + (ticks.length > 1 ? (i / (ticks.length - 1)) * (W - 20) : 0);
  const yOf = (v, mn, mx, h) => 8 + (1 - (v - mn) / Math.max(1e-6, mx - mn)) * (h - 16);

  const maxStep = Math.max(1, Number(ticks[ticks.length - 1]?.step || 1));
  g1.strokeStyle = "rgba(236,72,153,0.18)";
  g1.lineWidth = 1;
  wbState.densify.forEach((d) => {
    if (Math.abs(Number(d.delta || 0)) > 20000) {
      const xd = 10 + (Number(d.step || 0) / maxStep) * (W - 20);
      g1.beginPath(); g1.moveTo(xd, 4); g1.lineTo(xd, H - 4); g1.stroke();
    }
  });

  rawP.forEach((v, i) => {
    g1.fillStyle = "rgba(74,222,128,0.22)";
    g1.fillRect(xOf(i) - 1, yOf(v, minP, maxP, H) - 1, 2, 2);
  });
  rawL.forEach((v, i) => {
    g1.fillStyle = "rgba(251,146,60,0.20)";
    g1.fillRect(xOf(i) - 1, yOf(v, 0, maxL, H) - 1, 2, 2);
  });

  const plotLine = (ctx, arr, mn, mx, col, lw = 2.0) => {
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
    arr.forEach((v, i) => {
      const x = xOf(i), y = yOf(v, mn, mx, ctx.canvas.height);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plotLine(g1, emaL, 0, maxL, "#fb923c", 2.0);
  plotLine(g1, emaP, minP, maxP, "#4ade80", 2.1);

  const lastIdx = ticks.length - 1;
  const hoverLbl = el("lbl-chart-hover");
  if (hoverLbl && lastIdx >= 0) {
    hoverLbl.textContent = `EMA ${emaP[lastIdx].toFixed(1)}dB (raw ${rawP[lastIdx].toFixed(1)}) · Loss ${emaL[lastIdx].toFixed(4)}`;
  }

  const g2 = c2.getContext("2d"); g2.clearRect(0, 0, c2.width, c2.height);
  const cnts = ticks.map((t) => Number(t.n_gaussians || 0));
  plotLine(g2, cnts, Math.min(...cnts) * 0.9, Math.max(100, ...cnts) * 1.05, "#a78bfa", 1.8);
  renderMiniKernelDiag(ticks[lastIdx]);
}

function updateCenterJobStatusHeaderAndPreviews(jobObj) {
  const stg = String(jobObj?.stage || (wbState.ticks.length ? "completed" : "ready")).toLowerCase();
  const isRunning = !["completed", "failed", "cancelled", "stopped", "ready"].includes(stg);
  if (el("kdiag-job-status-badge")) {
    el("kdiag-job-status-badge").textContent = stg.toUpperCase();
  }
  el("btn-kdiag-pause")?.classList.toggle("hidden", !isRunning);
  el("btn-kdiag-ckpt")?.classList.toggle("hidden", !isRunning);
  el("btn-kdiag-stop")?.classList.toggle("hidden", !isRunning);

  const hasPly = !!(jobObj?.ply_rel);
  el("btn-kdiag-open-3d")?.classList.toggle("hidden", !hasPly);
  el("btn-kdiag-import-bl")?.classList.toggle("hidden", !hasPly);
  el("btn-kdiag-open-cmp")?.classList.toggle("hidden", !(jobObj?.cameras_rel));

  const stepCur = jobObj?.step || wbState.ticks[wbState.ticks.length - 1]?.step || 0;
  const stepTot = jobObj?.total_steps || jobObj?.iterations || stepCur || 1;
  const pct = stg === "completed" ? 100 : Math.min(100, Math.max(4, Math.round((stepCur / Math.max(1, stepTot)) * 100)));
  if (el("kdiag-prog-fill")) el("kdiag-prog-fill").style.width = `${pct}%`;

  if (el("kdiag-live-head-title") && jobObj) {
    el("kdiag-live-head-title").textContent = `🖼️ ${jobObj.job_id} (${stepCur}/${stepTot} steps)`;
  }
  if (el("kdiag-cost-badge")) {
    el("kdiag-cost-badge").textContent = jobObj?.cost_usd ? `$${Number(jobObj.cost_usd).toFixed(4)} USD` : "";
  }

  const prevUrl = jobObj?.preview_url || wbState.previewUrl;
  const diffUrl = jobObj?.diff_url || wbState.diffUrl;
  const rImg = el("kdiag-render-img"), rEmp = el("kdiag-render-empty");
  const dImg = el("kdiag-diff-img"), dEmp = el("kdiag-diff-empty");
  if (rImg && rEmp) {
    if (prevUrl) {
      rImg.src = withTokenUrl(prevUrl);
      rImg.classList.remove("hidden");
      rEmp.classList.add("hidden");
    } else {
      rImg.classList.add("hidden");
      rEmp.classList.remove("hidden");
      rEmp.textContent = jobObj?.error ? `⚠ ${jobObj.error}` : `Stage: ${stg.toUpperCase()} — waiting for preview frame…`;
    }
  }
  if (dImg && dEmp) {
    if (diffUrl) {
      dImg.src = withTokenUrl(diffUrl);
      dImg.classList.remove("hidden");
      dEmp.classList.add("hidden");
    } else {
      dImg.classList.add("hidden");
      dEmp.classList.remove("hidden");
    }
  }

  // Job Summary Cards (Results + Saved Config)
  const sumGrid = el("kdiag-job-summary-grid");
  if (sumGrid) {
    const cfg = jobObj?.config || wbState.config || {};
    const tr = cfg.train || {};
    const po = cfg.pose || {};
    const cards = [
      ["Status / Steps", `${stg.toUpperCase()} · ${stepCur}/${stepTot}`, jobObj?.gpu ? `Modal ${jobObj.gpu}` : "Cloud GPU"],
      ["PSNR / SSIM", `${jobObj?.psnr || "—"} dB`, `SSIM ${jobObj?.ssim || "—"} · Loss ${jobObj?.loss || "—"}`],
      ["Gaussians", Number(jobObj?.n_gaussians || 0).toLocaleString(), jobObj?.ply_size_bytes ? formatBytes(jobObj.ply_size_bytes) : (tr.target_gaussians ? `budget ${(tr.target_gaussians / 1000).toFixed(0)}k` : "3DGS .ply")],
      ["Source & Pose", String(po.backend || wbState.diagnostics?.pose_audit?.pose_backend || "vggt").toUpperCase(), String(jobObj?.source_video_rel || jobObj?.source_label || "—")],
      ["Anti-Floater Config", `SH${tr.sh_degree ?? 1} · ${tr.views_per_step ?? 2} views/step`, `corridor ${tr.cam_corridor_radius_m ?? 0.35}m · opa_reg ${tr.opa_reg ?? "—"}`],
    ];
    sumGrid.innerHTML = cards
      .map(([k, v, sub]) => `<div class="kdiag-kv-card"><span class="kv-k">${escapeHtml(k)}</span><span class="kv-v">${escapeHtml(v)}</span><span class="kv-sub">${escapeHtml(sub)}</span></div>`)
      .join("");
  }
}

export function renderFullKernelTelemetryStage() {
  if (state.centerKind === "how") {
    renderCartoonExplainer(wbState.selectedCartoonStage);
    const stRowsH = wbState.diagnostics?.timing_summary?.stages || [];
    const findH = (k) => stRowsH.find((r) => r.key === k);
    const rActH = findH("param_act_ms"), rFwdH = findH("fwd_raster_only_ms") || findH("fwd_raster_ms");
    const rLossH = findH("loss_ssim_ms"), rBwdH = findH("bwd_ms"), rAdamH = findH("adam_ms");
    if (rActH && el("svg-ms-act")) el("svg-ms-act").textContent = `act: ${rActH.mean_ms.toFixed(2)} ms (${rActH.pct}%)`;
    if (rFwdH && el("svg-ms-fwd")) el("svg-ms-fwd").textContent = `fwd: ${rFwdH.mean_ms.toFixed(2)} ms (${rFwdH.pct}%)`;
    if (rLossH && el("svg-ms-loss")) el("svg-ms-loss").textContent = `loss: ${rLossH.mean_ms.toFixed(2)} ms (${rLossH.pct}%)`;
    if (rBwdH && el("svg-ms-bwd")) el("svg-ms-bwd").textContent = `bwd: ${rBwdH.mean_ms.toFixed(2)} ms (${rBwdH.pct}% · p95 ${rBwdH.p95_ms.toFixed(1)}ms)`;
    if (rAdamH && el("svg-ms-adam")) el("svg-ms-adam").textContent = `adam: ${rAdamH.mean_ms.toFixed(2)} ms (${rAdamH.pct}%)`;
    return;
  }
  if (state.centerKind !== "kernel") return;

  const sel = el("sel-kdiag-job");
  if (sel) {
    const jobs = Array.from(state.splatJobs.values());
    sel.innerHTML = jobs.length
      ? jobs.map((j) => `<option value="${escapeHtml(j.job_id)}"${j.job_id === wbState.jobId ? " selected" : ""}>${escapeHtml(j.job_id)} (${Number(j.n_gaussians || 0).toLocaleString()} pts)</option>`).join("")
      : `<option value="">— No Splat Jobs Yet —</option>`;
  }

  const jobObj = wbState.jobId ? state.splatJobs.get(wbState.jobId) || null : null;
  updateCenterJobStatusHeaderAndPreviews(jobObj);

  const diag = wbState.diagnostics || {};
  const tsum = diag.timing_summary || {};
  const macro = diag.macro_timings || {};
  const camConv = diag.camera_convergence || {};
  const plyA = diag.ply_audit || {};
  const poseA = diag.pose_audit || {};
  const ticks = wbState.ticks || [];
  const lastTick = ticks.length ? ticks[ticks.length - 1] : null;
  const kdiagLive = lastTick?.kernel_diag || diag.latest_kernel_diag || null;

  const mp = el("kdiag-macro-pills");
  if (mp) {
    const pills = [];
    if (macro.extract_s != null) pills.push(`extract ${macro.extract_s}s`);
    if (macro.pose_s != null) pills.push(`pose ${macro.pose_s}s (${poseA.pose_backend || "vggt"})`);
    if (macro.align_s != null) pills.push(`Sim3 ${macro.align_s}s`);
    if (macro.train_s != null) pills.push(`train ${macro.train_s}s`);
    if (macro.export_s != null) pills.push(`export ${macro.export_s}s`);
    if (poseA.train_resolution) pills.push(`${poseA.train_resolution} × ${poseA.registered_cameras || "?"} cams`);
    mp.innerHTML = pills.map((p) => `<span class="kdiag-macro-pill">${escapeHtml(p)}</span>`).join("");
  }

  const stRows = tsum.stages || [];
  if (el("kdiag-step-summary")) {
    el("kdiag-step-summary").textContent = tsum.step_ms_mean
      ? `${tsum.step_ms_mean.toFixed(2)} ms/it mean · p95 ${tsum.step_ms_p95.toFixed(1)} ms · ~${tsum.it_per_sec_steady || "—"} it/s`
      : "No timing ticks";
  }
  const mainBar = el("kdiag-main-stage-bar");
  if (mainBar) {
    mainBar.innerHTML = stRows
      .map((r) => `<div class="splat-stage-seg" style="width:${r.pct}%;background:${r.color}" title="${escapeHtml(r.label)}: ${r.mean_ms} ms (${r.pct}%)"></div>`)
      .join("");
  }
  const tb = el("kdiag-timing-table")?.querySelector("tbody");
  if (tb) {
    tb.innerHTML = stRows.length
      ? stRows
          .map(
            (r) => `<tr>
              <td><span class="kdiag-swatch" style="background:${r.color}"></span>${escapeHtml(r.label)}</td>
              <td><b>${r.pct.toFixed(1)}%</b></td>
              <td>${r.mean_ms.toFixed(3)} ms</td>
              <td>${r.p50_ms.toFixed(3)} ms</td>
              <td>${r.p95_ms.toFixed(3)} ms</td>
              <td>${r.max_ms.toFixed(2)} ms</td>
            </tr>`
          )
          .join("")
      : `<tr><td colspan="6" style="text-align:center;color:var(--muted)">Waiting for CUDA Event timing ticks…</td></tr>`;
  }

  const flist = el("kdiag-findings-list");
  const findings = diag.findings || [];
  if (el("kdiag-verdict-pill")) {
    el("kdiag-verdict-pill").textContent = findings.length ? `${findings.length} signal(s)` : "Clean convergence";
  }
  if (flist) {
    flist.innerHTML = findings.length
      ? findings
          .map(
            (f) => `<div class="kdiag-finding sev-${escapeHtml(f.severity || "info")}">
              <div class="kdiag-finding-title">${escapeHtml(f.subsystem)}</div>
              <div class="kdiag-finding-ev">${escapeHtml(f.evidence)}</div>
              <div class="kdiag-finding-det">${escapeHtml(f.detail)}</div>
            </div>`
          )
          .join("")
      : `<div class="wb-lbl">No noise-floor anomalies detected on this job.</div>`;
  }

  const cConv = el("canvas-kdiag-convergence");
  if (cConv && ticks.length) {
    const ctx = cConv.getContext("2d"), W = cConv.width, H = cConv.height;
    ctx.clearRect(0, 0, W, H);
    const rawL = ticks.map((t) => Number(t.loss || 0));
    const rawP = ticks.map((t) => Number(t.psnr || 0));
    const emaL = ticks.map((t, i) => (t.loss_ema != null ? Number(t.loss_ema) : computeEmaSeries(rawL, 0.1)[i]));
    const emaP = ticks.map((t, i) => (t.psnr_ema != null ? Number(t.psnr_ema) : computeEmaSeries(rawP, 0.1)[i]));
    const maxL = Math.max(0.02, ...rawL);
    const minP = Math.min(...rawP, 13), maxP = Math.max(minP + 3, ...rawP);
    const maxS = Math.max(1, Number(ticks[ticks.length - 1].step || 1));
    const xS = (s) => 12 + (s / maxS) * (W - 24);
    const yV = (v, mn, mx) => 10 + (1 - (v - mn) / Math.max(1e-6, mx - mn)) * (H - 22);

    wbState.densify.forEach((d) => {
      const isReset = Number(d.delta || 0) < -20000;
      ctx.strokeStyle = isReset ? "rgba(244,63,94,0.45)" : "rgba(236,72,153,0.12)";
      ctx.lineWidth = isReset ? 1.5 : 1;
      const x = xS(Number(d.step || 0));
      ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, H - 12); ctx.stroke();
    });

    ticks.forEach((t, i) => {
      const x = xS(Number(t.step || i));
      ctx.fillStyle = "rgba(74,222,128,0.28)";
      ctx.fillRect(x - 1.5, yV(rawP[i], minP, maxP) - 1.5, 3, 3);
      ctx.fillStyle = "rgba(251,146,60,0.24)";
      ctx.fillRect(x - 1.5, yV(rawL[i], 0, maxL) - 1.5, 3, 3);
    });

    const drawCurve = (arr, mn, mx, col) => {
      ctx.strokeStyle = col; ctx.lineWidth = 2.3; ctx.beginPath();
      ticks.forEach((t, i) => {
        const x = xS(Number(t.step || i)), y = yV(arr[i], mn, mx);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };
    drawCurve(emaL, 0, maxL, "#fb923c");
    drawCurve(emaP, minP, maxP, "#4ade80");

    if (el("kdiag-psnr-spread-lbl")) {
      el("kdiag-psnr-spread-lbl").textContent =
        `Raw σ=${camConv.late_raw_psnr_std_db ?? "—"}dB vs EMA σ=${camConv.late_ema_psnr_std_db ?? "—"}dB (same-cam Δ=${camConv.within_cam_step_jitter_db ?? "—"}dB)`;
    }
  }

  const cTim = el("canvas-kdiag-timings");
  const steady = ticks.filter((t) => Number(t.step || 0) > 1 && t.stage_timings);
  if (cTim && steady.length) {
    const ctx = cTim.getContext("2d"), W = cTim.width, H = cTim.height;
    ctx.clearRect(0, 0, W, H);
    const totals = steady.map((t) => Number(t.stage_timings.step_ms || 0));
    const maxMs = Math.max(5, ...totals) * 1.05;
    const barW = Math.max(1.5, (W - 24) / steady.length);
    const stackSpec = [
      ["bwd_ms", "#f97316"],
      ["fwd_raster_ms", "#38bdf8"],
      ["loss_ssim_ms", "#eab308"],
      ["adam_ms", "#22c55e"],
      ["host_ms", "#64748b"],
    ];
    steady.forEach((t, i) => {
      const x = 12 + i * ((W - 24) / steady.length);
      let yCur = H - 10;
      stackSpec.forEach(([k, col]) => {
        const v = Number(t.stage_timings[k] || 0);
        const hSeg = (v / maxMs) * (H - 22);
        if (hSeg > 0.2) {
          ctx.fillStyle = col;
          ctx.fillRect(x, yCur - hSeg, Math.max(1.2, barW - 0.4), hSeg);
          yCur -= hSeg;
        }
      });
    });
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px monospace";
    ctx.fillText(`peak ${Math.max(...totals).toFixed(1)} ms`, 14, 13);
  }

  const gradTicks = ticks.filter((t) => t.kernel_diag?.grads?.means);
  const cGrad = el("canvas-kdiag-grads");
  if (cGrad) {
    if (gradTicks.length >= 2) {
      cGrad.classList.remove("hidden");
      const ctx = cGrad.getContext("2d"), W = cGrad.width, H = cGrad.height;
      ctx.clearRect(0, 0, W, H);
      const series = [
        ["means", "#38bdf8", (t) => Math.log10(Math.max(1e-12, Number(t.kernel_diag.grads.means?.rms || 0)))],
        ["opacities", "#a855f7", (t) => Math.log10(Math.max(1e-12, Number(t.kernel_diag.grads.opacities?.rms || 0)))],
        ["scales", "#eab308", (t) => Math.log10(Math.max(1e-12, Number(t.kernel_diag.grads.scales?.rms || 0)))],
        ["sh0", "#22c55e", (t) => Math.log10(Math.max(1e-12, Number(t.kernel_diag.grads.sh0?.rms || 0)))],
      ];
      const allVals = series.flatMap(([, , fn]) => gradTicks.map(fn));
      const mnG = Math.min(...allVals) - 0.3, mxG = Math.max(...allVals) + 0.3;
      series.forEach(([, col, fn]) => {
        ctx.strokeStyle = col; ctx.lineWidth = 1.9; ctx.beginPath();
        gradTicks.forEach((t, i) => {
          const x = 12 + (i / (gradTicks.length - 1)) * (W - 24);
          const y = 10 + (1 - (fn(t) - mnG) / Math.max(1e-4, mxG - mnG)) * (H - 20);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
      ctx.fillStyle = "#94a3b8"; ctx.font = "10px monospace";
      ctx.fillText(`log10(∇ RMS): means(cyan) opa(purple) scales(yellow) sh0(green) [${mnG.toFixed(1)}..${mxG.toFixed(1)}]`, 14, 13);
    } else {
      cGrad.classList.add("hidden");
    }
  }

  const igrid = el("kdiag-interm-grid");
  if (igrid) {
    const cards = [];
    if (kdiagLive) {
      const gm = kdiagLive.grads?.means || {};
      const go = kdiagLive.grads?.opacities || {};
      const g2 = kdiagLive.grads?.means2d_absgrad || {};
      const rs = kdiagLive.raster || {};
      const pr = kdiagLive.params || {};
      const am = kdiagLive.adam?.means || {};
      const lt = kdiagLive.loss_terms || {};
      cards.push(
        ["∇means RMS / p99", `${Number(gm.rms || 0).toExponential(2)}`, `p99 ${Number(gm.p99 || 0).toExponential(2)} · nz ${(Number(gm.nz_frac || 0) * 100).toFixed(1)}%`],
        ["|∇2D| Screen Grad", `p95 ${Number(g2.p95 || 0).toExponential(2)}`, `${(Number(g2.above_densify_frac || 0) * 100).toFixed(2)}% ≥ densify`],
        ["∇opacities RMS", `${Number(go.rms || 0).toExponential(2)}`, `σ(1−σ) mean ${pr.opa_jacobian_mean ?? "—"}`],
        ["Visible / Tile Overdraw", `${Number(rs.visible_gaussians || 0).toLocaleString()} (${(Number(rs.visible_frac || 0) * 100).toFixed(0)}%)`, `${Number(rs.tile_isects || 0).toLocaleString()} isects (${rs.avg_splats_per_tile}/tile)`],
        ["T < 1e-4 Early-Stop", `${(Number(rs.T_floor_sat_frac || 0) * 100).toFixed(1)}% px`, `rad p50=${rs.radius_px_p50}px max=${rs.radius_px_max}px`],
        ["Adam SNR (`means`)", `SNR ${am.snr_p50 ?? "—"}`, `step_rms ${Number(am.step_rms || 0).toExponential(2)} · fused=${am.fused}`],
        ["Loss Term Split", `L1 ${lt.l1_weighted ?? "—"} + SSIM ${lt.dssim_weighted ?? "—"}`, `opa ${lt.opa_reg ?? 0} · scale ${lt.scale_reg ?? 0} · corridor ${lt.corridor_reg ?? 0}`]
      );
    }
    if (plyA.n_gaussians) {
      const oq = plyA.opa_quantiles || [];
      const sq = plyA.scale_m_quantiles || [];
      cards.push(
        ["PLY Opacity p5 / p50 / p95", `${oq[0] ?? "—"} / ${oq[2] ?? "—"} / ${oq[4] ?? "—"}`, `dead<0.02: ${(plyA.opa_dead_low_frac * 100).toFixed(1)}% · ghost: ${(plyA.opa_ghost_band_frac * 100).toFixed(1)}%`],
        ["PLY World Scale (m)", `p50 ${sq[1] ?? "—"}m · p99 ${sq[3] ?? "—"}m`, `max ${plyA.scale_m_max}m · >0.25m billboards: ${plyA.giant_billboard_count}`],
        ["PLY Anisotropy (Needle)", `p50 ${plyA.aniso_p50}× · p95 ${plyA.aniso_p95}×`, `max ${plyA.aniso_max}× · SH0 clip ${(plyA.sh0_oot_gamut_frac * 100).toFixed(1)}%`]
      );
    }
    if (poseA.registered_cameras) {
      cards.push([
        "Pose & Trajectory Floor",
        `${poseA.pose_backend} (${poseA.registered_cameras} cams)`,
        `path ${poseA.camera_path_length_m}m · floor_inlier ${poseA.floor_inlier_ratio} · se(3)=${poseA.pose_refined}`,
      ]);
    }
    igrid.innerHTML = cards
      .map(
        ([k, v, sub]) => `<div class="kdiag-kv-card"><span class="kv-k">${escapeHtml(k)}</span><span class="kv-v">${escapeHtml(v)}</span><span class="kv-sub">${escapeHtml(sub)}</span></div>`
      )
      .join("");
  }

  const camBody = el("kdiag-cam-audit-body");
  if (camBody) {
    const worst = camConv.worst_cameras || [];
    const best = camConv.best_cameras || [];
    if (el("kdiag-cam-audit-lbl")) {
      el("kdiag-cam-audit-lbl").textContent = worst.length
        ? `${camConv.total_cameras} views · spread ${(best[0]?.last_psnr - worst[0]?.last_psnr).toFixed(1)} dB`
        : "—";
    }
    const mkCamTbl = (title, rows, col) => `
      <div>
        <div class="wb-lbl" style="margin-bottom:4px;color:${col};font-weight:700">${title}</div>
        <table class="kdiag-table">
          <thead><tr><th>Camera</th><th>PSNR</th><th>ΔdB</th><th>Bwd ms</th></tr></thead>
          <tbody>
            ${rows.map((r) => `<tr><td>${escapeHtml(r.camera)}</td><td><b>${r.last_psnr.toFixed(1)}</b></td><td>${r.gain_db >= 0 ? "+" : ""}${r.gain_db.toFixed(1)}</td><td>${r.mean_bwd_ms.toFixed(1)}</td></tr>`).join("")}
          </tbody>
        </table>
      </div>`;
    camBody.innerHTML = worst.length
      ? mkCamTbl("⚠️ Lowest PSNR Views (Limiting Floor)", worst, "#fb923c") + mkCamTbl("✅ Highest PSNR Views", best, "#4ade80")
      : `<div class="wb-lbl">No per-camera telemetry recorded yet.</div>`;
  }

  const logBox = el("kdiag-iter-log");
  if (logBox) {
    const merged = [];
    (wbState.stages || []).forEach((s) =>
      merged.push({ sort: Number(s.step || 0) - 0.3, cls: "ev-stage", step: `stage`, txt: `${s.stage}${s.detail ? " — " + JSON.stringify(s.detail) : ""}` })
    );
    (wbState.surgeries || []).forEach((sg) =>
      merged.push({ sort: Number(sg.step || 0) - 0.2, cls: "ev-surgery", step: `step ${sg.step || 0}`, txt: `🩹 SURGERY: ${JSON.stringify(sg)}` })
    );
    (wbState.densify || []).forEach((dn) => {
      const dlt = Number(dn.delta || 0);
      if (Math.abs(dlt) >= 500) {
        merged.push({
          sort: Number(dn.step || 0) - 0.1,
          cls: "ev-densify",
          step: `step ${dn.step}`,
          txt: `🌱 DENSIFY ${Number(dn.n_before || 0).toLocaleString()} → ${Number(dn.n_after || 0).toLocaleString()} (${dlt >= 0 ? "+" : ""}${dlt.toLocaleString()})`,
        });
      }
    });
    ticks.forEach((t) => {
      const st = t.stage_timings || {};
      const stepMs = st.step_ms != null ? `${Number(st.step_ms).toFixed(1)}ms (bwd ${Number(st.bwd_ms || 0).toFixed(1)}ms)` : `${t.it_per_sec || "—"} it/s`;
      const emaStr = t.psnr_ema != null ? ` [EMA ${Number(t.psnr_ema).toFixed(2)}dB]` : "";
      merged.push({
        sort: Number(t.step || 0),
        cls: "ev-tick",
        step: `it ${t.step}`,
        txt: `loss=${Number(t.loss || 0).toFixed(4)} (L1=${Number(t.l1 || 0).toFixed(4)} SSIM=${Number(t.ssim || 0).toFixed(3)}) · PSNR=${Number(t.psnr || 0).toFixed(2)}dB${emaStr} · N=${Number(t.n_gaussians || 0).toLocaleString()} · ${stepMs} · cam=${t.camera_name || "—"}`,
      });
    });
    merged.sort((a, b) => a.sort - b.sort);
    const tail = merged.slice(-260);
    if (el("kdiag-log-count")) el("kdiag-log-count").textContent = `${merged.length} events`;
    logBox.innerHTML = tail.length
      ? tail.map((r) => `<div class="klog-row ${r.cls}"><span class="klog-step">${escapeHtml(r.step)}</span><span>${escapeHtml(r.txt)}</span></div>`).join("")
      : `<div class="wb-lbl">Waiting for first training telemetry tick…</div>`;
    logBox.scrollTop = logBox.scrollHeight;
  }
}

export async function hydrateWbTelemetry(jobId) {
  if (!jobId) return;
  try {
    const data = await apiFetch(`/api/splats/telemetry/${encodeURIComponent(jobId)}`);
    wbState.jobId = jobId;
    wbState.ticks = data.ticks || [];
    wbState.densify = data.densify_events || [];
    wbState.surgeries = data.surgery_events || [];
    wbState.stages = data.stages || [];
    wbState.snapshots = data.snapshots || [];
    wbState.cameras = data.cameras || [];
    wbState.diagnostics = data.diagnostics || null;
    wbState.config = data.config || {};
    wbState.previewUrl = data.preview_url || null;
    wbState.diffUrl = data.diff_url || null;

    if (data.job && data.job.job_id) {
      state.splatJobs.set(jobId, data.job);
      if (state.activeSplatJob?.job_id === jobId) state.activeSplatJob = data.job;
    } else if (data.config && Object.keys(data.config).length) {
      const existingJob = state.splatJobs.get(jobId);
      if (existingJob) existingJob.config = data.config;
    }
    const curJob = state.splatJobs.get(jobId);
    if (curJob && state.activeSplatJob?.job_id === jobId) {
      actions.syncInspectorActiveSplatPanel?.(curJob);
    }

    ["sel-wb-focus-cam", "sel-resume-focus-cam"].forEach((fcId) => {
      const fcSel = el(fcId);
      if (fcSel && wbState.cameras.length) {
        const curFc = fcSel.value;
        fcSel.innerHTML = `<option value="">🎯 All Views (${wbState.cameras.length})</option>` +
          wbState.cameras.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
        fcSel.value = curFc;
      }
    });

    const syncScrubber = (rowId, rngId, lblId) => {
      const snapRow = el(rowId), snapRng = el(rngId), snapLbl = el(lblId);
      if (snapRow && snapRng && wbState.snapshots.length) {
        snapRow.classList.remove("hidden");
        snapRng.max = String(wbState.snapshots.length - 1);
        snapRng.value = String(wbState.snapshots.length - 1);
        const sLast = wbState.snapshots[wbState.snapshots.length - 1];
        if (snapLbl) snapLbl.textContent = `step ${sLast.step}`;
      } else {
        snapRow?.classList.add("hidden");
      }
    };
    syncScrubber("splat-snap-row", "rng-splat-snap", "lbl-splat-snap");
    syncScrubber("kdiag-snap-row", "rng-kdiag-snap", "lbl-kdiag-snap");

    if (wbState.ticks.length) {
      el("splat-live-card")?.classList.remove("hidden");
      drawWbCharts();
    }
    renderFullKernelTelemetryStage();
  } catch (_err) {}
}

export async function openKernelTelemetryForJob(jobId) {
  actions.hideAllCenterViews?.();
  state.centerKind = "kernel";
  const effId = jobId || activeWbJobId();
  const jobObj = effId ? state.splatJobs.get(effId) || null : null;
  if (jobObj) {
    state.activeSplatJob = jobObj;
    state.activeJobKey = `splat:${effId}`;
    actions.syncInspectorActiveSplatPanel?.(jobObj);
  }

  el("stage-kind-badge").textContent = "SPLAT JOB STATUS";
  el("stage-title").textContent = effId || "3DGS Job Status & Telemetry";
  el("theater-meta").textContent = jobObj
    ? `${String(jobObj.stage || "").toUpperCase()} · ${Number(jobObj.n_gaussians || 0).toLocaleString()} pts · ${jobObj.psnr || "—"} dB`
    : "Telemetry";
  el("stage-3d-btns")?.classList.add("hidden");
  el("stage-media-btns")?.classList.add("hidden");
  el("theater-kernel-wrap")?.classList.remove("hidden");

  actions.syncInspectorContext?.();
  actions.syncViewModeButtons?.();
  actions.renderHierarchyTree?.();
  if (effId) {
    wbState.jobId = effId;
    updateCenterJobStatusHeaderAndPreviews(jobObj);
    await hydrateWbTelemetry(effId);
  } else {
    renderFullKernelTelemetryStage();
  }
}

export function openHowItWorksInCenter() {
  actions.hideAllCenterViews?.();
  state.centerKind = "how";

  el("stage-kind-badge").textContent = "HOW IT WORKS";
  el("stage-title").textContent = "3DGS SfM, CUDA Hardware Kernels & Anti-Floater Regularization";
  el("theater-meta").textContent = "Architecture & Math Reference";
  el("stage-3d-btns")?.classList.add("hidden");
  el("stage-media-btns")?.classList.add("hidden");
  el("theater-how-it-works-wrap")?.classList.remove("hidden");

  renderCartoonExplainer(wbState.selectedCartoonStage);
  renderFullKernelTelemetryStage();
  actions.syncInspectorContext?.();
  actions.syncViewModeButtons?.();
  actions.renderHierarchyTree?.();
}

const announcedSplatDone = new Set();
export function handleSplatJobUpdate(j, fromPoll = false) {
  if (!j || !j.job_id) return;
  const prev = state.splatJobs.get(j.job_id);
  if (prev?.config && (!j.config || !Object.keys(j.config).length)) {
    j.config = prev.config;
  }
  state.splatJobs.set(j.job_id, j);
  if (state.activeSplatJob?.job_id === j.job_id) {
    state.activeSplatJob = j;
  }

  const active = !["completed", "failed", "cancelled", "stopped"].includes(String(j.stage || "").toLowerCase());
  el("splat-live-card")?.classList.toggle("hidden", !active);

  // Keep Center Job Status Stage live-synced with incoming preview/diff & progress!
  if (state.centerKind === "kernel" && wbState.jobId === j.job_id) {
    updateCenterJobStatusHeaderAndPreviews(j);
    el("theater-meta").textContent = `${String(j.stage || "").toUpperCase()} · ${j.step || 0}/${j.total_steps || j.iterations || 0} · ${Number(j.n_gaussians || 0).toLocaleString()} pts · ${j.psnr || "—"} dB`;
  }

  actions.syncModalStatusPill?.();

  // When a job transitions to `completed`, automatically expand `assets/splats` in the Hierarchy Browser,
  // select the completed splat in the 3D Viewport, and open its Inspector settings + Resume panel!
  if (j.stage === "completed" && prev && prev.stage !== "completed" && !announcedSplatDone.has(j.job_id)) {
    announcedSplatDone.add(j.job_id);
    showToast(`✅ 3DGS Splat completed: ${j.job_id} (${Number(j.n_gaussians || 0).toLocaleString()} Gaussians)`);
    if (state.filter === "jobs") {
      state.filter = "all";
      document.querySelectorAll("#media-tabs .tab").forEach((t) => t.classList.toggle("active", t.dataset.filter === "all"));
    }
    actions.expandTreePath?.("assets/splats");
    actions.open3DSceneInCenter?.(j.job_id, false).then(() => {
      actions.scrollActiveTreeRowIntoView?.();
    });
    return;
  }

  if (!fromPoll) {
    actions.renderHierarchyTree?.();
    actions.renderJobs?.();
  }
}

// --- Side-by-Side Synced Trajectory Comparator ---
const cmpState = { active: false, job: null, traj: null, curFrameIdx: 0, orbitYaw: -0.75, orbitPitch: 0.48, dragging: false, lastX: 0, lastY: 0, rafId: 0 };

export async function openComparatorForJob(job) {
  actions.hideAllCenterViews?.();
  state.centerKind = "compare";
  state.activeSplatJob = job;
  cmpState.job = job;
  cmpState.active = true;

  el("stage-kind-badge").textContent = "COMPARE & PATH";
  el("stage-title").textContent = job.job_id;
  el("theater-compare-wrap")?.classList.remove("hidden");
  actions.syncInspectorActiveSplatPanel?.(job);
  actions.syncInspectorContext?.();
  actions.syncViewModeButtons?.();
  actions.renderHierarchyTree?.();

  try {
    const data = await apiFetch(`/api/splats/trajectory/${encodeURIComponent(job.job_id)}`);
    cmpState.traj = data.trajectory;
    cmpState.curFrameIdx = 0;

    let allPasses = Array.isArray(data.render_passes) ? data.render_passes.slice() : [];
    if (!allPasses.length) {
      if (data.source_video_url) {
        allPasses.push({ label: `🎬 Original (${data.source_video_rel || "source"})`, url: data.source_video_url, kind: "source", media_type: "video" });
      }
      if (data.modal_render_url) {
        allPasses.push({ label: "☁️ Modal H100 3DGS Render", url: data.modal_render_url, kind: "modal", media_type: "video" });
      }
      if (data.blender_render_url) {
        allPasses.push({ label: "🧊 Blender 5.3 EEVEE Render", url: data.blender_render_url, kind: "blender", media_type: "video" });
      }
    }

    const selL = el("sel-cmp-left"), selR = el("sel-cmp-right");
    const optsHtml = allPasses.length
      ? allPasses.map((o, i) => `<option value="${i}">${escapeHtml(o.label)}</option>`).join("")
      : `<option value="">— No passes yet —</option>`;
    selL.innerHTML = optsHtml;
    selR.innerHTML = optsHtml;
    selL._opts = allPasses;
    selR._opts = allPasses;

    const srcIdx = Math.max(0, allPasses.findIndex((p) => p.kind === "source"));
    const latestRenderIdx = allPasses.findIndex((p) => p.media_type !== "image" && p.kind !== "source");
    const fallbackStillIdx = allPasses.findIndex((p) => p.kind === "still_render" || p.kind === "snapshot");
    const rightDefaultIdx = latestRenderIdx >= 0 ? latestRenderIdx : fallbackStillIdx;

    selL.value = allPasses.length ? String(srcIdx) : "";
    if (rightDefaultIdx >= 0) selR.value = String(rightDefaultIdx);

    cmpState.applySlotMedia = (side, opt) => {
      const vEl = el(side === "left" ? "vid-cmp-left" : "vid-cmp-right");
      const iEl = el(side === "left" ? "img-cmp-left" : "img-cmp-right");
      const bEl = el(side === "left" ? "lbl-cmp-left" : "lbl-cmp-right");
      const phEl = side === "right" ? el("cmp-right-placeholder") : null;
      if (!opt?.url) {
        vEl?.pause();
        vEl?.removeAttribute("src");
        vEl?.classList.add("hidden");
        iEl?.classList.add("hidden");
        phEl?.classList.remove("hidden");
        if (bEl) bEl.textContent = "No render pass yet";
        return;
      }
      phEl?.classList.add("hidden");
      if (bEl) bEl.textContent = opt.label;
      if (opt.media_type === "image") {
        vEl?.pause();
        vEl?.removeAttribute("src");
        vEl?.classList.add("hidden");
        if (iEl) {
          iEl.src = withTokenUrl(opt.url);
          iEl.classList.remove("hidden");
        }
      } else {
        iEl?.classList.add("hidden");
        if (vEl) {
          vEl.classList.remove("hidden");
          vEl.src = withTokenUrl(opt.url);
          vEl.playbackRate = Number(el("sel-cmp-speed")?.value || 1);
          vEl.load();
        }
      }
    };

    cmpState.applySlotMedia("left", allPasses[srcIdx] || null);
    cmpState.applySlotMedia("right", rightDefaultIdx >= 0 ? allPasses[rightDefaultIdx] : null);

    const nF = cmpState.traj.total_frames;
    el("rng-cmp-scrub").max = String(Math.max(0, nF - 1));
    el("rng-cmp-scrub").value = "0";
    el("theater-meta").textContent = `${nF}f @ ${cmpState.traj.fps}fps · ${cmpState.traj.path_length_m}m`;
    drawComparatorCanvases();
    startComparatorLoop();
  } catch (err) {
    showToast(err.message, true);
  }
}

function seekCmpFrame(fi) {
  if (!cmpState.traj) return;
  const clamped = Math.max(0, Math.min(cmpState.traj.total_frames - 1, fi));
  cmpState.curFrameIdx = clamped;
  const tSec = clamped / Math.max(1, cmpState.traj.fps);
  [el("vid-cmp-left"), el("vid-cmp-right")].forEach((v) => {
    if (v?.src && Number.isFinite(v.duration)) v.currentTime = Math.min(v.duration - 0.001, tSec);
  });
  el("rng-cmp-scrub").value = String(clamped);
  drawComparatorCanvases();
}

function startComparatorLoop() {
  if (cmpState.rafId) cancelAnimationFrame(cmpState.rafId);
  function tick() {
    if (!cmpState.active || !cmpState.traj) return;
    const vL = el("vid-cmp-left");
    if (vL.src && !vL.paused) {
      const fi = Math.max(0, Math.min(cmpState.traj.total_frames - 1, Math.floor(vL.currentTime * (cmpState.traj.fps || 24))));
      if (fi !== cmpState.curFrameIdx) { cmpState.curFrameIdx = fi; el("rng-cmp-scrub").value = String(fi); drawComparatorCanvases(); }
    } else if (cmpState._virt) {
      cmpState.curFrameIdx = (cmpState.curFrameIdx + 1) % cmpState.traj.total_frames;
      el("rng-cmp-scrub").value = String(cmpState.curFrameIdx);
      drawComparatorCanvases();
    }
    cmpState.rafId = requestAnimationFrame(tick);
  }
  cmpState.rafId = requestAnimationFrame(tick);
}

function drawComparatorCanvases() {
  const traj = cmpState.traj;
  if (!traj?.frames?.length) return;
  const frames = traj.frames;
  const fi = Math.max(0, Math.min(frames.length - 1, cmpState.curFrameIdx));
  const cur = frames[fi];
  el("lbl-cmp-readout").textContent =
    `f ${cur.frame}/${frames.length} · t=${cur.pts_sec.toFixed(2)}s · XYZ=(${cur.location.map((v) => v.toFixed(2)).join(", ")})m · Yaw ${cur.yaw_deg.toFixed(1)}° Pitch ${cur.pitch_deg.toFixed(1)}°`;

  const c3 = el("canvas-cmp-path3d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w3 = Math.max(120, Math.floor(c3.clientWidth * dpr)), h3 = Math.max(90, Math.floor(c3.clientHeight * dpr));
  if (c3.width !== w3 || c3.height !== h3) { c3.width = w3; c3.height = h3; }
  const g3 = c3.getContext("2d");
  g3.clearRect(0, 0, w3, h3);

  const bmin = traj.bounds_min, bmax = traj.bounds_max;
  const cx = 0.5 * (bmin[0] + bmax[0]), cy = 0.5 * (bmin[1] + bmax[1]), cz = 0.5 * bmax[2];
  const span = Math.max(1.4, Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]));
  const eye = [
    cx + span * 2.0 * Math.cos(cmpState.orbitPitch) * Math.cos(cmpState.orbitYaw),
    cy + span * 2.0 * Math.cos(cmpState.orbitPitch) * Math.sin(cmpState.orbitYaw),
    cz + span * 2.0 * Math.sin(cmpState.orbitPitch),
  ];
  let fx = cx - eye[0], fy = cy - eye[1], fz = cz - eye[2];
  const fn = Math.hypot(fx, fy, fz) || 1; fx /= fn; fy /= fn; fz /= fn;
  let rx = fy, ry = -fx; const rn = Math.hypot(rx, ry) || 1; rx /= rn; ry /= rn;
  const ux = ry * fz, uy = -rx * fz, uz = rx * fy - ry * fx;
  const focal = Math.min(w3, h3) * 1.1;
  const proj = (pt) => {
    const dx = pt[0] - eye[0], dy = pt[1] - eye[1], dz = pt[2] - eye[2];
    const zc = dx * fx + dy * fy + dz * fz;
    if (zc < 0.08) return null;
    return [w3 * 0.5 + ((dx * rx + dy * ry) / zc) * focal, h3 * 0.5 - ((dx * ux + dy * uy + dz * uz) / zc) * focal];
  };
  g3.lineWidth = 2;
  for (let i = 1; i < frames.length; i++) {
    const p0 = proj(frames[i - 1].location), p1 = proj(frames[i].location);
    if (!p0 || !p1) continue;
    g3.strokeStyle = i <= fi ? "#38bdf8" : "rgba(148, 163, 184, 0.35)";
    g3.beginPath(); g3.moveTo(p0[0], p0[1]); g3.lineTo(p1[0], p1[1]); g3.stroke();
  }
  const pCur = proj(cur.location);
  if (pCur) { g3.fillStyle = "#ff7b29"; g3.beginPath(); g3.arc(pCur[0], pCur[1], 5, 0, Math.PI * 2); g3.fill(); }

  const cA = el("canvas-cmp-angles");
  const wA = Math.max(120, Math.floor(cA.clientWidth * dpr)), hA = Math.max(90, Math.floor(cA.clientHeight * dpr));
  if (cA.width !== wA || cA.height !== hA) { cA.width = wA; cA.height = hA; }
  const gA = cA.getContext("2d");
  gA.clearRect(0, 0, wA, hA);
  const yaws = frames.map((f) => f.yaw_deg), pitches = frames.map((f) => f.pitch_deg);
  const minAng = Math.min(...yaws, ...pitches, -5), maxAng = Math.max(...yaws, ...pitches, 5);
  const drawCurve = (arr, col) => {
    gA.strokeStyle = col; gA.lineWidth = 2; gA.beginPath();
    arr.forEach((v, i) => {
      const x = 14 + (frames.length > 1 ? (i / (frames.length - 1)) * (wA - 24) : 0);
      const y = 14 + (1 - (v - minAng) / Math.max(1e-5, maxAng - minAng)) * (hA - 28);
      if (i === 0) gA.moveTo(x, y); else gA.lineTo(x, y);
    });
    gA.stroke();
  };
  drawCurve(pitches, "#fb923c");
  drawCurve(yaws, "#38bdf8");
  const px = 14 + (frames.length > 1 ? (fi / (frames.length - 1)) * (wA - 24) : 0);
  gA.strokeStyle = "#fff"; gA.beginPath(); gA.moveTo(px, 8); gA.lineTo(px, hA - 8); gA.stroke();
}

export function initSplatTelemetryAndComparator() {
  window._bbHydrateSplatTelemetry = hydrateWbTelemetry;
  window._bbStopComparator = () => {
    cmpState.active = false;
    if (cmpState.rafId) cancelAnimationFrame(cmpState.rafId);
    el("vid-cmp-left")?.pause();
    el("vid-cmp-right")?.pause();
  };

  window._bbPushSplatTelemetry = (ev) => {
    if (!wbState.jobId || ev.step === 1) {
      wbState.jobId = ev.job_id;
      if (ev.step === 1) { wbState.ticks = []; wbState.densify = []; }
    }
    if (ev.job_id !== wbState.jobId) return;
    if (ev.event === "telemetry") {
      wbState.ticks.push(ev);
      drawWbCharts();
      if (state.centerKind === "kernel" || state.centerKind === "how") renderFullKernelTelemetryStage();
    } else if (ev.event === "densify") {
      wbState.densify.push(ev);
    }
  };

  el("svg-kdiag-cartoon")?.addEventListener("click", (e) => {
    const node = e.target.closest(".kstage-node");
    if (node) renderCartoonExplainer(node.getAttribute("data-kstage"));
  });

  el("btn-splat-show-render")?.addEventListener("click", () => {
    el("splat-live-img")?.classList.remove("hidden");
    el("splat-live-diff")?.classList.add("hidden");
  });
  el("btn-splat-show-diff")?.addEventListener("click", () => {
    el("splat-live-img")?.classList.add("hidden");
    el("splat-live-diff")?.classList.remove("hidden");
  });

  const bindSnapRange = (rngId, lblId, imgId, wrapId) => {
    el(rngId)?.addEventListener("input", (e) => {
      const idx = Number(e.target.value || 0);
      const snap = wbState.snapshots[idx];
      if (!snap) return;
      if (el(lblId)) el(lblId).textContent = `step ${snap.step}`;
      if (el(imgId)) {
        el(imgId).src = withTokenUrl(snap.url);
        el(imgId).classList.remove("hidden");
        if (wrapId) el(wrapId)?.classList.remove("hidden");
        if (imgId === "kdiag-render-img") el("kdiag-render-empty")?.classList.add("hidden");
      }
    });
  };
  bindSnapRange("rng-splat-snap", "lbl-splat-snap", "splat-live-img", "splat-live-preview-wrap");
  bindSnapRange("rng-kdiag-snap", "lbl-kdiag-snap", "kdiag-render-img", null);

  el("btn-open-full-kdiag")?.addEventListener("click", () => openKernelTelemetryForJob(activeWbJobId()));
  el("sel-kdiag-job")?.addEventListener("change", (e) => {
    const jid = e.target.value;
    if (jid) openKernelTelemetryForJob(jid);
  });
  el("btn-kdiag-refresh")?.addEventListener("click", () => {
    const jid = el("sel-kdiag-job")?.value || activeWbJobId();
    if (jid) hydrateWbTelemetry(jid);
  });

  // Status Header Quick Jumps & How-It-Works Navigation
  el("btn-kdiag-open-how")?.addEventListener("click", () => openHowItWorksInCenter());
  el("btn-how-back-launch")?.addEventListener("click", () => actions.openSplatLaunchDialog?.());
  el("btn-how-open-status")?.addEventListener("click", () => openKernelTelemetryForJob(activeWbJobId()));

  el("btn-kdiag-open-3d")?.addEventListener("click", () => {
    const jid = activeWbJobId();
    if (jid) actions.open3DSceneInCenter?.(jid, false);
  });
  el("btn-kdiag-open-cmp")?.addEventListener("click", () => {
    const jid = activeWbJobId();
    const j = jid ? state.splatJobs.get(jid) : null;
    if (j) openComparatorForJob(j);
  });
  el("btn-kdiag-import-bl")?.addEventListener("click", () => {
    const jid = activeWbJobId();
    if (jid) actions.importSplatToBlender?.(jid);
  });

  const togglePause = async () => {
    const jid = activeWbJobId(); if (!jid) return;
    wbState.paused = !wbState.paused;
    const lbl = wbState.paused ? "▶ Resume" : "⏸ Pause";
    if (el("btn-wb-pause")) el("btn-wb-pause").textContent = lbl;
    if (el("btn-kdiag-pause")) el("btn-kdiag-pause").textContent = lbl;
    await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, { method: "POST", json: { paused: wbState.paused } }).catch(() => {});
  };
  const triggerCkpt = async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, { method: "POST", json: { save_checkpoint: true } }).catch(() => {});
    showToast("Requested cloud checkpoint");
  };
  const triggerStop = async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch(`/api/splats/cancel/${encodeURIComponent(jid)}`, { method: "POST", json: {} }).catch(() => {});
  };

  el("btn-wb-pause")?.addEventListener("click", togglePause);
  el("btn-kdiag-pause")?.addEventListener("click", togglePause);
  el("btn-wb-saveckpt")?.addEventListener("click", triggerCkpt);
  el("btn-kdiag-ckpt")?.addEventListener("click", triggerCkpt);
  el("btn-wb-stop")?.addEventListener("click", triggerStop);
  el("btn-kdiag-stop")?.addEventListener("click", triggerStop);

  el("btn-wb-push-knobs")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    const budgetK = Number(el("inp-splat-budget")?.value || 350);
    await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, {
      method: "POST",
      json: {
        lambda_dssim: Number(el("inp-splat-dssim")?.value || 0.2),
        opa_reg: Number(el("inp-splat-opareg")?.value || 0.0005),
        densify_grad_threshold: Number(el("inp-splat-gradthresh")?.value || 0.0002),
        target_gaussians: Math.max(50000, Math.round(budgetK * 1000)),
        adaptive_strategy: (el("chk-splat-adaptive")?.checked ?? true) ? "adaptive_arc" : "fixed",
        arc_scale_mode: arcStrategy.scale,
        arc_gamma: arcStrategy.gamma,
        views_per_step: Number(el("sel-splat-views-step")?.value || 2),
        cam_corridor_radius_m: Number(el("inp-splat-corridor")?.value ?? 0.35),
        appearance_opt: el("chk-splat-appopt")?.checked ?? true,
        outlier_view_downweight: el("chk-splat-outlier")?.checked ?? true,
        focus_camera: el("sel-wb-focus-cam")?.value || "",
      },
    }).catch(() => {});
    showToast(`Pushed live strategy arc (${arcStrategy.scale}) & knobs`);
  });

  el("btn-surg-reset-opa")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch("/api/splats/resume", { method: "POST", json: { job_id: jid, add_steps: 1500, reset_opacities: 0.015 } }).catch((e) => showToast(e.message, true));
  });
  el("btn-surg-clamp-scale")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch("/api/splats/resume", { method: "POST", json: { job_id: jid, add_steps: 1000, clamp_log_scale: -1.6 } }).catch((e) => showToast(e.message, true));
  });
  el("btn-cropbox-spawn")?.addEventListener("click", async () => {
    await apiFetch("/api/splats/crop_box", { method: "POST", json: { action: "spawn" } }).then(() => showToast("Spawned SplatCropBox in Blender 5.3")).catch((e) => showToast(e.message, true));
  });
  el("btn-cropbox-cull")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch("/api/splats/crop_box", { method: "POST", json: { action: "cull_heal", job_id: jid } }).catch((e) => showToast(e.message, true));
  });
  el("btn-cropbox-keep")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch("/api/splats/crop_box", { method: "POST", json: { action: "keep_heal", job_id: jid } }).catch((e) => showToast(e.message, true));
  });
  el("btn-build-proxy")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch(`/api/splats/proxy/${encodeURIComponent(jid)}`, { method: "POST", json: { import_blender: true } })
      .then((r) => showToast(`Built ${r.proxy_glb_rel} & imported to Blender!`))
      .catch((e) => showToast(e.message, true));
  });

  // Comparator controls (Left/Right pass dropdowns, playback speed, transport)
  const syncCmpDropdownSwitch = (side) => {
    const sel = el(side === "left" ? "sel-cmp-left" : "sel-cmp-right");
    const opt = sel?._opts?.[Number(sel.value)];
    if (!cmpState.applySlotMedia) return;
    const refVid = el(side === "left" ? "vid-cmp-right" : "vid-cmp-left");
    const wasPlaying = !!(refVid?.src && !refVid.paused);
    const curT = refVid?.src && Number.isFinite(refVid.currentTime)
      ? refVid.currentTime
      : (cmpState.curFrameIdx / Math.max(1, cmpState.traj?.fps || 24));
    cmpState.applySlotMedia(side, opt || null);
    const targetVid = el(side === "left" ? "vid-cmp-left" : "vid-cmp-right");
    if (targetVid && opt && opt.media_type !== "image") {
      targetVid.addEventListener(
        "loadedmetadata",
        () => {
          if (Number.isFinite(targetVid.duration)) {
            targetVid.currentTime = Math.min(Math.max(0, targetVid.duration - 0.001), curT);
          }
          if (wasPlaying) targetVid.play().catch(() => {});
        },
        { once: true }
      );
    }
  };
  el("sel-cmp-left")?.addEventListener("change", () => syncCmpDropdownSwitch("left"));
  el("sel-cmp-right")?.addEventListener("change", () => syncCmpDropdownSwitch("right"));
  el("sel-cmp-speed")?.addEventListener("change", (e) => {
    const rate = Number(e.target.value || 1);
    [el("vid-cmp-left"), el("vid-cmp-right")].forEach((v) => {
      if (v) v.playbackRate = rate;
    });
  });

  el("btn-cmp-play")?.addEventListener("click", () => {
    const vL = el("vid-cmp-left"), vR = el("vid-cmp-right");
    const hasL = !!(vL?.src && !vL.classList.contains("hidden"));
    const hasR = !!(vR?.src && !vR.classList.contains("hidden"));
    const m = hasL ? vL : hasR ? vR : null;
    if (!m) {
      cmpState._virt = !cmpState._virt;
      el("btn-cmp-play").textContent = cmpState._virt ? "⏸ Pause" : "▶ Play";
      return;
    }
    const rate = Number(el("sel-cmp-speed")?.value || 1);
    if (m.paused) {
      if (hasL && hasR) vR.currentTime = vL.currentTime;
      if (hasL) { vL.playbackRate = rate; vL.play().catch(() => {}); }
      if (hasR) { vR.playbackRate = rate; vR.play().catch(() => {}); }
      el("btn-cmp-play").textContent = "⏸ Pause";
    } else {
      vL?.pause(); vR?.pause(); el("btn-cmp-play").textContent = "▶ Play";
    }
  });
  el("btn-cmp-prevf")?.addEventListener("click", () => seekCmpFrame(cmpState.curFrameIdx - 1));
  el("btn-cmp-nextf")?.addEventListener("click", () => seekCmpFrame(cmpState.curFrameIdx + 1));
  el("rng-cmp-scrub")?.addEventListener("input", (e) => seekCmpFrame(Number(e.target.value || 0)));

  el("btn-cmp-snap-arc")?.addEventListener("click", async () => {
    if (!cmpState.job) return;
    const btn = el("btn-cmp-snap-arc");
    const prevTxt = btn.textContent;
    btn.disabled = true;
    btn.textContent = "⏳ Snapping Arc…";
    try {
      const res = await apiFetch("/api/splats/snap_arc", {
        method: "POST",
        json: { job_id: cmpState.job.job_id, floater_clean: true, auto_render: true },
      });
      const sn = res.snap || {};
      const evTot = (sn.giant_evicted || 0) + (sn.corridor_evicted || 0) + (sn.frustum_evicted || 0);
      showToast(
        `🎯 Snapped ${sn.snapped_keyframes || 0} keyframes (fixed ${sn.spikes_repaired || 0} spikes, L1 ${sn.l1_before} → ${sn.l1_after}, cleaned ${evTot.toLocaleString()} floaters)!`
      );
      await actions.refreshSplats?.();
      await openComparatorForJob(state.splatJobs.get(cmpState.job.job_id) || cmpState.job);
    } catch (e) {
      showToast(e.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = prevTxt;
    }
  });
  el("btn-cmp-farm-modal")?.addEventListener("click", async () => {
    if (!cmpState.job) return;
    const btn = el("btn-cmp-farm-modal"); btn.disabled = true;
    try {
      const res = await apiFetch("/api/splats/render", { method: "POST", json: { job_id: cmpState.job.job_id, mode: "modal" } });
      showToast(`Rendered ${res.frames} frames @ ${res.render_fps} fps!`);
      await actions.refreshSplats?.();
      await openComparatorForJob(state.splatJobs.get(cmpState.job.job_id) || cmpState.job);
    } catch (e) { showToast(e.message, true); } finally { btn.disabled = false; }
  });
  el("btn-cmp-render-blender")?.addEventListener("click", async () => {
    if (!cmpState.job) return;
    const btn = el("btn-cmp-render-blender"); btn.disabled = true;
    try {
      await apiFetch("/api/splats/render", { method: "POST", json: { job_id: cmpState.job.job_id, mode: "blender" } });
      await actions.refreshSplats?.();
      await openComparatorForJob(state.splatJobs.get(cmpState.job.job_id) || cmpState.job);
    } catch (e) { showToast(e.message, true); } finally { btn.disabled = false; }
  });

  el("canvas-cmp-path3d")?.addEventListener("mousedown", (e) => { cmpState.dragging = true; cmpState.lastX = e.clientX; cmpState.lastY = e.clientY; });
  window.addEventListener("mouseup", () => { cmpState.dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!cmpState.dragging || !cmpState.active) return;
    cmpState.orbitYaw -= (e.clientX - cmpState.lastX) * 0.01;
    cmpState.orbitPitch = Math.max(-1.35, Math.min(1.42, cmpState.orbitPitch + (e.clientY - cmpState.lastY) * 0.01));
    cmpState.lastX = e.clientX; cmpState.lastY = e.clientY;
    drawComparatorCanvases();
  });

  actions.getActiveWbJobId = activeWbJobId;
  actions.openKernelTelemetryForJob = openKernelTelemetryForJob;
  actions.openHowItWorksInCenter = openHowItWorksInCenter;
  actions.openComparatorForJob = openComparatorForJob;
  actions.handleSplatJobUpdate = handleSplatJobUpdate;
}