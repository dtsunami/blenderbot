// Seedance 2.5, Seedream 5.5, Meshy 7.1, Uploads, Blender Capture, Generic Job Center, and SSE synchronization.

import {
  DURATION_CHOICES,
  MODE_SUFFIXES,
  SEEDREAM_MODE_SUFFIXES,
  SEEDREAM_RES_COST,
  SIZE_CHOICES,
  actions,
  apiFetch,
  el,
  escapeHtml,
  formatBytes,
  getToken,
  showToast,
  state,
  withTokenUrl,
} from "./state.js";
import {
  appendFeedLine,
  mergeHistoryStrings,
  pushLocalPromptHistory,
  renderAgentContextBar,
  setAgentBusyUi,
  switchInspectorTab,
} from "./workbench.js";

function activeDurationSec() {
  const durVal = el("sel-duration")?.value;
  if (durVal && Number(durVal) > 0) return Number(durVal);
  if (state.sourceVideoProbe && state.sourceVideoProbe.duration_sec) {
    return Math.max(4, Math.min(30, Math.round(state.sourceVideoProbe.duration_sec)));
  }
  return 5;
}

function activeRatePerSec() {
  const curModel = el("select-video-model")?.value;
  const found = state.videoModels.find((m) => m.id === curModel);
  return (found && found.cost_per_sec_usd) || state.costPerSec || 0.1028;
}

export function populatePricingDropdowns() {
  const rate = activeRatePerSec();
  const durSel = el("sel-duration");
  if (!durSel) return;
  const prevDur = durSel.dataset.initialized ? durSel.value : "5";
  durSel.innerHTML = "";
  DURATION_CHOICES.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.val;
    const effS = c.sec !== null ? c.sec : (state.sourceVideoProbe?.duration_sec ? Math.max(4, Math.min(30, Math.round(state.sourceVideoProbe.duration_sec))) : 5);
    opt.textContent = c.val === "" ? `Auto (${effS}s) · ~$${(effS * rate).toFixed(2)}` : `${c.label} · ~$${(effS * rate).toFixed(2)}`;
    if (c.val === prevDur) opt.selected = true;
    durSel.appendChild(opt);
  });
  durSel.dataset.initialized = "1";

  const sizeSel = el("sel-size");
  const prevSize = sizeSel.dataset.initialized ? sizeSel.value : "1280x720";
  sizeSel.innerHTML = "";
  SIZE_CHOICES.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s.val;
    opt.textContent = s.label;
    if (s.val === prevSize) opt.selected = true;
    sizeSel.appendChild(opt);
  });
  sizeSel.dataset.initialized = "1";

  const modSel = el("select-video-model");
  const curMod = modSel.value || "bytedance/seedance-2.5";
  modSel.innerHTML = "";
  state.videoModels.forEach((m) => {
    const r = m.cost_per_sec_usd || state.costPerSec || 0.1028;
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.id} ($${r.toFixed(3)}/s)`;
    if (m.id === curMod) opt.selected = true;
    modSel.appendChild(opt);
  });
}

export function updatePricingAndInspector() {
  populatePricingDropdowns();
  const sec = activeDurationSec();
  const rate = activeRatePerSec();
  const estUsd = sec * rate;

  if (el("est-cost-label")) el("est-cost-label").textContent = `~$${estUsd.toFixed(2)} USD (${sec}s)`;
  if (el("inspector-cost-pill")) el("inspector-cost-pill").textContent = `~$${estUsd.toFixed(3)}`;

  const modelId = el("select-video-model")?.value || "bytedance/seedance-2.5";
  const sizeVal = el("sel-size")?.value || "auto";
  const rawPrompt = (el("input-prompt")?.value || "").trim();
  const compiledPrompt = rawPrompt ? `${rawPrompt}${MODE_SUFFIXES[state.mode] || ""}` : "";

  let firstDesc = state.firstFrame ? state.firstFrame.path : state.sourceVideo ? `from ${state.sourceVideo.name}` : "none";
  let lastDesc = state.lastFrame ? state.lastFrame.path : state.sourceVideo ? `from ${state.sourceVideo.name}` : "none";

  if (el("inspector-body")) {
    el("inspector-body").innerHTML = `
      <div class="insp-row"><span class="insp-key">model:</span><span class="insp-val">${escapeHtml(modelId)} · ${sec}s · ${escapeHtml(sizeVal)}</span></div>
      <div class="insp-row"><span class="insp-key">first_frame:</span><span class="insp-val ${firstDesc !== "none" ? "ok" : ""}">${escapeHtml(firstDesc)}</span></div>
      <div class="insp-row"><span class="insp-key">last_frame:</span><span class="insp-val ${lastDesc !== "none" ? "ok" : ""}">${escapeHtml(lastDesc)}</span></div>
      <div class="insp-row"><span class="insp-key">prompt:</span><span class="insp-val">${escapeHtml(compiledPrompt ? compiledPrompt.slice(0, 120) : "(empty)")}</span></div>
    `;
  }
}

export function populateSourceVideoDropdown() {
  const videos = state.mediaItems.filter((it) => it.kind === "video");
  const sel1 = el("sel-source-video"), sel2 = el("sel-source-video-2");
  if (!sel1 || !sel2) return;
  const cur1 = state.sourceVideo ? state.sourceVideo.path : "";
  const cur2 = state.sourceVideo2 ? state.sourceVideo2.path : "";
  sel1.innerHTML = `<option value="">— None —</option>`;
  sel2.innerHTML = `<option value="">— Optional 2nd clip —</option>`;
  videos.forEach((it) => {
    const o1 = document.createElement("option");
    o1.value = it.path; o1.textContent = `${it.name} (${formatBytes(it.size_bytes)})`;
    if (it.path === cur1) o1.selected = true;
    sel1.appendChild(o1);

    const o2 = document.createElement("option");
    o2.value = it.path; o2.textContent = `${it.name} (${formatBytes(it.size_bytes)})`;
    if (it.path === cur2) o2.selected = true;
    sel2.appendChild(o2);
  });
}

export function setSourceVideo(item) {
  state.sourceVideo = item || null;
  state.sourceVideoProbe = null;
  if (item) {
    if (state.sourceVideo2 && state.sourceVideo2.path === item.path) state.sourceVideo2 = null;
    apiFetch(`/api/workspace/probe?path=${encodeURIComponent(item.path)}`)
      .then((info) => {
        if (state.sourceVideo && state.sourceVideo.path === item.path) {
          state.sourceVideoProbe = info;
          updatePricingAndInspector();
        }
      })
      .catch(() => {});
  }
  populateSourceVideoDropdown();
  renderConditioningSlots();
  actions.renderHierarchyTree?.();
  updatePricingAndInspector();
}

export function setSourceVideo2(item) {
  state.sourceVideo2 = item || null;
  state.sourceVideo2Probe = null;
  if (item && state.sourceVideo && state.sourceVideo.path === item.path) state.sourceVideo = null;
  populateSourceVideoDropdown();
  renderConditioningSlots();
  actions.renderHierarchyTree?.();
  updatePricingAndInspector();
}

export function setFirstFrame(item) {
  state.firstFrame = item && state.firstFrame && state.firstFrame.path === item.path ? null : item;
  renderConditioningSlots();
  actions.renderHierarchyTree?.();
  updatePricingAndInspector();
  actions.syncViewModeButtons?.();
}

export function setLastFrame(item) {
  state.lastFrame = item && state.lastFrame && state.lastFrame.path === item.path ? null : item;
  renderConditioningSlots();
  actions.renderHierarchyTree?.();
  updatePricingAndInspector();
  actions.syncViewModeButtons?.();
}

export function setRefAudio(item) {
  state.refAudio = item && state.refAudio && state.refAudio.path === item.path ? null : item;
  renderConditioningSlots();
  actions.renderHierarchyTree?.();
  updatePricingAndInspector();
}

export function renderConditioningSlots() {
  if (!el("slot-video-card")) return;
  const hasAnyVid = !!(state.sourceVideo || state.sourceVideo2);
  const hasBoth = !!(state.sourceVideo && state.sourceVideo2);
  el("slot-video-card").classList.toggle("active", hasAnyVid);
  el("btn-clear-video")?.classList.toggle("hidden", !state.sourceVideo);
  el("btn-view-vid1")?.classList.toggle("hidden", !state.sourceVideo);
  el("btn-clear-video-2")?.classList.toggle("hidden", !state.sourceVideo2);
  el("btn-view-vid2")?.classList.toggle("hidden", !state.sourceVideo2);
  el("btn-swap-videos")?.classList.toggle("hidden", !hasBoth);
  el("dual-video-mode-bar")?.classList.toggle("hidden", !hasBoth);
  el("slot-video-opts")?.classList.toggle("hidden", !hasAnyVid);

  const vidBadge = el("badge-sd-video-state");
  if (vidBadge) vidBadge.textContent = hasBoth ? "Vid1 + Vid2" : hasAnyVid ? "1 clip" : "Optional";
  if (hasAnyVid && el("sec-sd-video")) el("sec-sd-video").open = true;

  el("btn-clear-first")?.classList.toggle("hidden", !state.firstFrame);
  el("slot-first")?.classList.toggle("filled", !!state.firstFrame);
  el("slot-first-body").innerHTML = state.firstFrame
    ? `<img src="${withTokenUrl(state.firstFrame.url)}" alt="first" /><div class="slot-caption">${escapeHtml(state.firstFrame.name)}</div>`
    : `<span class="empty-placeholder">Click <b>1st</b> on an image or <b>📸 3D</b></span>`;

  el("btn-clear-last")?.classList.toggle("hidden", !state.lastFrame);
  el("slot-last")?.classList.toggle("filled", !!state.lastFrame);
  el("slot-last-body").innerHTML = state.lastFrame
    ? `<img src="${withTokenUrl(state.lastFrame.url)}" alt="last" /><div class="slot-caption">${escapeHtml(state.lastFrame.name)}</div>`
    : `<span class="empty-placeholder">Click <b>Last</b> on an image or <b>📸 3D</b></span>`;

  const extraCnt = state.refVideos.length + state.refImages.length + (state.refAudio ? 1 : 0);
  const refsBadge = el("badge-sd-refs-state");
  if (refsBadge) refsBadge.textContent = `${extraCnt} ref${extraCnt === 1 ? "" : "s"}`;
  if (extraCnt > 0 && el("sec-sd-refs")) el("sec-sd-refs").open = true;

  const chips = el("refs-chips");
  chips.innerHTML = "";
  if (!extraCnt) {
    chips.innerHTML = `<span class="empty-placeholder">Optional — click <b>Audio</b> or add extra refs</span>`;
  } else {
    const addChip = (icon, item, onRem) => {
      const c = document.createElement("span");
      c.className = "ref-chip";
      c.innerHTML = `${icon} ${escapeHtml(item.name)} <button type="button" title="Remove">✕</button>`;
      c.addEventListener("click", () => actions.openMediaInCenter?.(item));
      c.querySelector("button").addEventListener("click", (e) => { e.stopPropagation(); onRem(); });
      chips.appendChild(c);
    };
    state.refVideos.forEach((v, i) => addChip("🎞", v, () => { state.refVideos.splice(i, 1); renderConditioningSlots(); actions.renderHierarchyTree?.(); }));
    state.refImages.forEach((im, i) => addChip("🖼", im, () => { state.refImages.splice(i, 1); renderConditioningSlots(); actions.renderHierarchyTree?.(); }));
    if (state.refAudio) addChip("🔊", state.refAudio, () => setRefAudio(null));
  }
}

export function populateSeedreamModels() {
  const sel = el("select-image-model");
  if (!sel) return;
  const prev = sel.value || "bytedance-seed/seedream-5-5";
  sel.innerHTML = "";
  state.imageModels.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.id} (~$${(m.cost_per_image_usd || 0.02).toFixed(2)})`;
    if (m.id === prev) opt.selected = true;
    sel.appendChild(opt);
  });
}

export function toggleSeedreamRef(item) {
  if (!item) return;
  const idx = state.seedreamRefs.findIndex((r) => r.path === item.path);
  if (idx >= 0) state.seedreamRefs.splice(idx, 1);
  else if (state.seedreamRefs.length < 10) state.seedreamRefs.push(item);
  else return showToast("Seedream 5.5 accepts up to 10 reference images", true);
  renderSeedreamRefs();
  actions.renderHierarchyTree?.();
  updateSeedreamInspector();
}

export function renderSeedreamRefs() {
  const chips = el("sr-refs-chips");
  if (!chips) return;
  chips.innerHTML = "";
  if (!state.seedreamRefs.length) {
    chips.innerHTML = `<span class="empty-placeholder">0 refs (Text-to-Image) — click <b>🎨 SR</b> on images or <b>📸 Grab 3D</b></span>`;
    return;
  }
  state.seedreamRefs.forEach((img, idx) => {
    const chip = document.createElement("span");
    chip.className = "ref-chip";
    chip.title = "Click to view in Center Stage";
    chip.innerHTML = `🎨 #${idx + 1} ${escapeHtml(img.name)} <button type="button" title="Remove">✕</button>`;
    chip.addEventListener("click", () => actions.openMediaInCenter?.(img));
    chip.querySelector("button").addEventListener("click", (e) => {
      e.stopPropagation();
      state.seedreamRefs.splice(idx, 1);
      renderSeedreamRefs();
      actions.renderHierarchyTree?.();
      updateSeedreamInspector();
    });
    chips.appendChild(chip);
  });
}

export function updateSeedreamInspector() {
  const body = el("sr-inspector-body");
  if (!body) return;
  const modelId = el("select-image-model")?.value || "bytedance-seed/seedream-5-5";
  const resTier = el("sel-sr-res")?.value || "1K";
  const ar = el("sel-sr-aspect")?.value || "16:9";
  const bg = el("sel-sr-bg")?.value || "auto";
  let fmt = el("sel-sr-fmt")?.value || "png";
  if (bg === "transparent" && fmt === "jpeg") {
    fmt = "png";
    el("sel-sr-fmt").value = "png";
  }
  const estCost = SEEDREAM_RES_COST[resTier] || 0.02;
  el("sr-inspector-cost").textContent = `~$${estCost.toFixed(2)}`;
  el("sr-est-cost-label").textContent = `~$${estCost.toFixed(2)} USD (${resTier})`;

  const rawP = (el("input-sr-prompt")?.value || "").trim();
  const compiled = rawP ? `${rawP}${SEEDREAM_MODE_SUFFIXES[state.seedreamMode] || ""}` : "";
  body.innerHTML = `
    <div class="insp-row"><span class="insp-key">model:</span><span class="insp-val">${escapeHtml(modelId)} · ${resTier} · ${ar} · ${fmt} (${bg})</span></div>
    <div class="insp-row"><span class="insp-key">references:</span><span class="insp-val ${state.seedreamRefs.length ? "ok" : ""}">${state.seedreamRefs.length ? escapeHtml(state.seedreamRefs.map((r) => r.name).join(", ")) : "none"}</span></div>
    <div class="insp-row"><span class="insp-key">prompt:</span><span class="insp-val">${escapeHtml(compiled ? compiled.slice(0, 120) : "(enter prompt)")}</span></div>
  `;
}

export function setMeshyInputImage(item, forceSet = false) {
  if (!item) state.meshyInputImage = null;
  else if (!forceSet && state.meshyInputImage && state.meshyInputImage.path === item.path) state.meshyInputImage = null;
  else state.meshyInputImage = item;
  renderMeshySlots();
  actions.renderHierarchyTree?.();
  updateMeshyInspector();
}

export function setMeshyTextureImage(item) {
  state.meshyTextureImage = item && state.meshyTextureImage && state.meshyTextureImage.path === item.path ? null : item;
  renderMeshySlots();
  actions.renderHierarchyTree?.();
  updateMeshyInspector();
}

export function renderMeshySlots() {
  if (!el("slot-meshy-img")) return;
  el("btn-clear-meshy-img")?.classList.toggle("hidden", !state.meshyInputImage);
  el("slot-meshy-img")?.classList.toggle("filled", !!state.meshyInputImage);
  el("slot-meshy-img-body").innerHTML = state.meshyInputImage
    ? `<img src="${withTokenUrl(state.meshyInputImage.url)}" alt="meshy-input" /><div class="slot-caption">${escapeHtml(state.meshyInputImage.name)}</div>`
    : `<span class="empty-placeholder">Click <b>🧊 3D</b> on any image or <b>📸 3D</b></span>`;

  el("btn-clear-meshy-teximg")?.classList.toggle("hidden", !state.meshyTextureImage);
  el("slot-meshy-teximg")?.classList.toggle("filled", !!state.meshyTextureImage);
  el("slot-meshy-teximg-body").innerHTML = state.meshyTextureImage
    ? `<img src="${withTokenUrl(state.meshyTextureImage.url)}" alt="meshy-tex" /><div class="slot-caption">${escapeHtml(state.meshyTextureImage.name)}</div>`
    : `<span class="empty-placeholder">Optional front texture guide</span>`;
}

export function updateMeshyInspector() {
  const body = el("meshy-inspector-body");
  if (!body) return;
  const texOn = el("chk-meshy-texture")?.checked ?? true;
  const remeshOn = el("chk-meshy-remesh")?.checked ?? true;
  let pbrOn = el("chk-meshy-pbr")?.checked ?? false;
  const texModel = el("sel-meshy-texmodel")?.value || "";
  let topo = el("sel-meshy-topology")?.value || "triangle";

  if (texOn && texModel === "meshy-n1") {
    if (pbrOn) { pbrOn = false; if (el("chk-meshy-pbr")) el("chk-meshy-pbr").checked = false; }
    if (topo === "quad" && remeshOn) { topo = "triangle"; if (el("sel-meshy-topology")) el("sel-meshy-topology").value = "triangle"; }
  }

  const rigOn = el("chk-meshy-rig")?.checked ?? false;
  const animOn = el("chk-meshy-anim")?.checked ?? false;
  if (animOn && !rigOn && el("chk-meshy-rig")) el("chk-meshy-rig").checked = true;
  const effRig = el("chk-meshy-rig")?.checked ?? false;
  const effAnim = effRig && (el("chk-meshy-anim")?.checked ?? false);
  el("meshy-rig-opts")?.classList.toggle("hidden", !effRig);

  const estCost = 0.8 + (texOn ? 0.4 : 0.0) + (effRig ? 0.2 : 0.0) + (effAnim ? 0.12 : 0.0);
  if (el("meshy-est-cost-label")) el("meshy-est-cost-label").textContent = `$${estCost.toFixed(2)} USD`;
  if (el("meshy-inspector-cost")) el("meshy-inspector-cost").textContent = `$${estCost.toFixed(2)}`;

  const poly = Number(el("inp-meshy-polycount")?.value || 30000);
  const sym = el("sel-meshy-symmetry")?.value || "auto";
  const geomRes = el("sel-meshy-geomres")?.value || "standard";
  const pose = el("sel-meshy-pose")?.value || "";
  const skel = el("sel-meshy-skeleton")?.value || "meshyrig";
  const imgLbl = state.meshyInputImage ? state.meshyInputImage.name : "(select input image)";

  body.innerHTML = `
    <div class="insp-row"><span class="insp-key">endpoint:</span><span class="insp-val">meshy/v7.1/image-to-3d (fal Queue)</span></div>
    <div class="insp-row"><span class="insp-key">image_url:</span><span class="insp-val ${state.meshyInputImage ? "ok" : ""}">${escapeHtml(imgLbl)}</span></div>
    <div class="insp-row"><span class="insp-key">geometry:</span><span class="insp-val">${escapeHtml(topo)} · ${poly.toLocaleString()} polys · sym=${escapeHtml(sym)} · res=${escapeHtml(geomRes)}</span></div>
    <div class="insp-row"><span class="insp-key">shading:</span><span class="insp-val">${texOn ? `textured (${texModel || "default"}${pbrOn ? " + PBR" : ""})` : "untextured mesh ($0.80)"}</span></div>
    ${effRig ? `<div class="insp-row"><span class="insp-key">rigging:</span><span class="insp-val ok">${escapeHtml(skel)}${pose ? ` · ${escapeHtml(pose)}` : ""}${effAnim ? ` · anim #${el("inp-meshy-animid")?.value || 92}` : ""}</span></div>` : ""}
  `;
}

export async function importMeshyToBlender(jobId, variant = "auto") {
  showToast(`Importing Meshy '${jobId}' (.glb) into Blender 5.3…`);
  try {
    const res = await apiFetch("/api/meshy/import", {
      method: "POST",
      json: { job_id: jobId, variant, place_on_ground: true },
    });
    const names = (res.imported?.imported_objects || []).join(", ");
    showToast(`Imported into Blender 5.3: ${names || jobId}`);
    refreshState();
  } catch (err) {
    showToast(err.message, true);
  }
}

export function renderMeshyAssetsList() {
  const listEl = el("meshy-assets-list");
  if (!listEl) return;
  const arr = Array.from(state.meshyJobs.values()).sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  if (el("badge-meshy-cnt")) el("badge-meshy-cnt").textContent = String(arr.length);
  if (!arr.length) {
    listEl.innerHTML = `<div class="empty-placeholder">No Meshy 3D meshes yet — select an Input Image above or drop a <code>.glb</code> file.</div>`;
    return;
  }
  listEl.innerHTML = "";
  arr.forEach((m) => {
    const active = !["completed", "failed", "cancelled"].includes(m.stage);
    const cls = m.stage === "completed" ? "completed" : m.stage === "failed" ? "failed" : active ? "running" : "";
    const card = document.createElement("div");
    card.className = `job-card ${cls}`;
    const costBadge = m.cost_usd ? `<span class="job-cost-pill">$${Number(m.cost_usd).toFixed(2)}</span>` : "";
    const trisBadge = m.triangles_total ? ` · ${(m.triangles_total / 1000).toFixed(1)}k tris` : "";
    card.innerHTML = `
      <div class="job-top">
        <span>🧊 ${escapeHtml(m.job_id)} · <code>${escapeHtml(m.stage)}</code>${escapeHtml(trisBadge)}</span>
        <span>
          ${costBadge}
          ${m.stage === "completed" ? `<button type="button" class="btn btn-secondary btn-xs btn-imp-meshy" title="Import .glb into Blender 5.3">→ BL</button>` : ""}
          ${active ? `<button type="button" class="btn btn-ghost btn-xs btn-cancel-meshy">Cancel</button>` : ""}
        </span>
      </div>
      <div class="job-prompt">${escapeHtml(m.error ? "⚠ " + m.error : (m.request_id ? `req: ${m.request_id} · ` : "") + (m.prompt || m.output_rel))}</div>
    `;
    card.querySelector(".btn-imp-meshy")?.addEventListener("click", (e) => { e.stopPropagation(); importMeshyToBlender(m.job_id); });
    card.querySelector(".btn-cancel-meshy")?.addEventListener("click", (e) => {
      e.stopPropagation();
      apiFetch(`/api/meshy/cancel/${encodeURIComponent(m.job_id)}`, { method: "POST" }).catch(() => {});
    });
    card.addEventListener("click", () => {
      if (m.stage === "completed") actions.open3DSceneInCenter?.(`meshy:${m.job_id}`, false);
    });
    listEl.appendChild(card);
  });
}

function renderGenericJobCenter(jobKind, job, jobPath) {
  const statusPill = el("jobview-status-pill"), pathLbl = el("jobview-path-lbl"), costLbl = el("jobview-cost-lbl"), elapsedLbl = el("jobview-elapsed-lbl");
  const prevBox = el("jobview-preview-box"), statsGrid = el("jobview-stats-grid"), promptBox = el("jobview-prompt-box"), logsBox = el("jobview-logs-box"), acts = el("jobview-actions");
  if (!prevBox || !statsGrid || !logsBox) return;

  const stg = String(job.stage || "completed").toUpperCase();
  if (statusPill) statusPill.textContent = `${stg}${job.queue_position != null ? ` (Queue #${job.queue_position})` : ""}`;
  if (pathLbl) pathLbl.textContent = jobPath;
  if (costLbl) costLbl.textContent = job.cost_usd ? `$${Number(job.cost_usd).toFixed(jobKind === "meshy" ? 2 : 4)} USD` : "";
  if (elapsedLbl) elapsedLbl.textContent = job.elapsed_sec ? `${Number(job.elapsed_sec).toFixed(1)}s elapsed` : "";

  if (acts) {
    acts.innerHTML = "";
    const addAct = (lbl, cls, fn) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = `btn ${cls} btn-xs`;
      b.textContent = lbl;
      b.addEventListener("click", fn);
      acts.appendChild(b);
    };
    const isRunning = !["completed", "failed", "cancelled", "stopped"].includes(String(job.stage || "").toLowerCase());
    if (isRunning) {
      addAct("⏹ Cancel Job", "btn-secondary", () => {
        const u = jobKind === "meshy" ? `/api/meshy/cancel/${encodeURIComponent(job.job_id)}` : `/api/seedance/cancel/${encodeURIComponent(job.local_id)}`;
        apiFetch(u, { method: "POST" }).catch(() => {});
      });
    }
    if (jobKind === "meshy" && job.stage === "completed") {
      addAct("🧊 Open .GLB in 3D Stage", "btn-primary", () => actions.open3DSceneInCenter?.(`meshy:${job.job_id}`, false));
      addAct("📥 Import → Blender 5.3", "btn-secondary", () => importMeshyToBlender(job.job_id));
    } else if (job.media_url && job.stage === "completed") {
      const isImg = /\.(png|jpe?g|webp)$/i.test(job.output_rel || "") || jobKind === "seedream";
      addAct("👁 Open Full Media Stage", "btn-primary", () =>
        actions.openMediaInCenter?.({
          path: job.output_rel,
          name: (job.output_rel || "").split("/").pop() || job.local_id,
          kind: isImg ? "image" : "video",
          size_bytes: job.conditioning?.size_bytes || 0,
          width: job.width,
          height: job.height,
          cost_usd: job.cost_usd,
          url: job.media_url,
        })
      );
      if (!isImg && job.output_rel) {
        addAct("🌐 Configure 3DGS Splat", "btn-secondary", () => actions.openSplatLaunchDialog?.("video", job.output_rel));
      }
    }
  }

  prevBox.innerHTML = "";
  if (jobKind === "meshy" && job.thumbnail_url) {
    prevBox.innerHTML = `<img src="${withTokenUrl(job.thumbnail_url)}" alt="Meshy preview" />`;
  } else if (job.media_url) {
    const isImg = /\.(png|jpe?g|webp)$/i.test(job.output_rel || "") || jobKind === "seedream";
    prevBox.innerHTML = isImg
      ? `<img src="${withTokenUrl(job.media_url)}" alt="Output image" />`
      : `<video src="${withTokenUrl(job.media_url)}" controls autoplay loop muted playsinline></video>`;
  } else {
    prevBox.innerHTML = `<div class="wb-lbl" style="padding:24px;text-align:center">${job.error ? "⚠ " + escapeHtml(job.error) : `Job is ${escapeHtml(stg)} — preview appears automatically on completion.`}</div>`;
  }

  const cards = [
    ["Job ID", String(job.job_id || job.local_id || "—"), String(job.request_id || job.remote_job_id || "local/persisted")],
    ["Compute Backend", jobKind === "meshy" ? "fal.ai Cloud Queue" : jobKind === "local" ? "Local Blender 5.3" : "OpenRouter Cloud API", String(job.model || "—")],
    ["Status", stg, job.created_at ? new Date(job.created_at * 1000).toLocaleString() : "—"],
    ["Output Path", String(job.output_rel || "pending"), job.conditioning?.size_bytes ? formatBytes(job.conditioning.size_bytes) : "workspace sandbox"],
  ];
  if (jobKind === "meshy") {
    cards.push(["Triangles", Number(job.triangles_total || 0).toLocaleString(), job.rigged_glb_rel ? "Rigged + Mesh" : "Static GLB"]);
  } else if (job.width && job.height) {
    cards.push(["Dimensions", `${job.width} × ${job.height}`, job.duration_sec ? `${job.duration_sec}s video` : "still"]);
  }
  statsGrid.innerHTML = cards
    .map(([k, v, sub]) => `<div class="kdiag-kv-card"><span class="kv-k">${escapeHtml(k)}</span><span class="kv-v">${escapeHtml(v)}</span><span class="kv-sub">${escapeHtml(sub)}</span></div>`)
    .join("");

  if (promptBox) {
    if (job.prompt || job.error) {
      promptBox.classList.remove("hidden");
      promptBox.innerHTML = `<strong>${job.error ? "⚠ Error:" : "Prompt / Conditioning:"}</strong> ${escapeHtml(job.error || job.prompt || "")}`;
    } else promptBox.classList.add("hidden");
  }

  const logLines = [];
  if (Array.isArray(job.logs) && job.logs.length) job.logs.forEach((ln, idx) => logLines.push({ cls: "ev-tick", step: `log #${idx + 1}`, txt: String(ln) }));
  if (job.conditioning && Object.keys(job.conditioning).length) logLines.push({ cls: "ev-stage", step: "params", txt: JSON.stringify(job.conditioning) });
  if (job.error) logLines.push({ cls: "ev-err", step: "error", txt: String(job.error) });
  if (!logLines.length) logLines.push({ cls: "ev-done", step: stg.toLowerCase(), txt: `Artifact ready at ${job.output_rel || jobPath}` });
  if (el("jobview-log-meta")) el("jobview-log-meta").textContent = `${logLines.length} entries`;
  logsBox.innerHTML = logLines
    .map((r) => `<div class="klog-row ${r.cls}"><span class="klog-step">${escapeHtml(r.step)}</span><span>${escapeHtml(r.txt)}</span></div>`)
    .join("");
}

export function openJobInCenter(jobKind, job, jobKey, jobPath) {
  state.activeJobKey = jobKey;
  if (jobKind === "splat") {
    actions.openKernelTelemetryForJob?.(job.job_id);
    return;
  }
  actions.hideAllCenterViews?.();
  state.centerKind = "job";
  el("stage-kind-badge").textContent = jobKind === "local" ? "LOCAL JOB" : "CLOUD JOB";
  el("stage-title").textContent = jobPath || job.job_id || job.local_id || "Job Inspector";
  el("theater-meta").textContent = `${String(job.stage || "").toUpperCase()}${job.cost_usd ? ` · $${Number(job.cost_usd).toFixed(4)}` : ""}`;
  el("stage-3d-btns")?.classList.add("hidden");
  el("stage-media-btns")?.classList.add("hidden");
  el("theater-job-wrap")?.classList.remove("hidden");

  renderGenericJobCenter(jobKind, job, jobPath);
  actions.syncInspectorContext?.();
  actions.syncViewModeButtons?.();
  actions.renderHierarchyTree?.();
}

export function renderJobs() {
  const jobsList = el("jobs-list");
  if (!jobsList) return;
  const spArr = Array.from(state.splatJobs.values()).map((s) => Object.assign({ _jkind: "splat" }, s));
  const sdArr = Array.from(state.jobs.values()).map((j) => Object.assign({ _jkind: "media" }, j));
  const mArr = Array.from(state.meshyJobs.values()).map((m) => Object.assign({ _jkind: "meshy" }, m));
  const arr = [...spArr, ...sdArr, ...mArr].sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  const runningCnt = arr.filter((j) => !["completed", "failed", "cancelled", "stopped"].includes(j.stage)).length;
  const badge = el("badge-jobs-cnt");
  if (badge) {
    badge.textContent = String(runningCnt || arr.length);
    badge.classList.toggle("hidden", arr.length === 0);
  }
  if (!arr.length) {
    jobsList.innerHTML = `<div class="empty-placeholder">No jobs submitted yet.</div>`;
    return;
  }
  jobsList.innerHTML = "";
  arr.forEach((j) => {
    const active = !["completed", "failed", "cancelled", "stopped"].includes(j.stage);
    const cls = j.stage === "completed" ? "completed" : j.stage === "failed" ? "failed" : active ? "running" : "";
    const card = document.createElement("div");
    card.className = `job-card ${cls}`;
    const idLbl = j._jkind === "splat" ? `🌐 ${j.job_id}` : j._jkind === "meshy" ? `🧊 ${j.job_id}` : j.local_id;
    const costBadge = j.cost_usd ? `<span class="job-cost-pill">$${Number(j.cost_usd).toFixed(j._jkind === "meshy" ? 2 : 4)}</span>` : "";
    const subSummary = j._jkind === "splat"
      ? `PSNR ${j.psnr || "—"}dB · ${Number(j.n_gaussians || 0).toLocaleString()} pts · ${j.step || 0}/${j.total_steps || 0}`
      : j.error ? "⚠ " + j.error : j.prompt || j.output_rel || "";
    card.innerHTML = `
      <div class="job-top">
        <span>${escapeHtml(idLbl)} · <code>${escapeHtml(j.stage)}</code></span>
        <span>${costBadge} ${active ? `<button type="button" class="btn btn-ghost btn-xs btn-cancel-job">Cancel</button>` : ""}</span>
      </div>
      <div class="job-prompt">${escapeHtml(subSummary)}</div>
    `;
    card.querySelector(".btn-cancel-job")?.addEventListener("click", (e) => {
      e.stopPropagation();
      const cancelUrl =
        j._jkind === "splat"
          ? `/api/splats/cancel/${encodeURIComponent(j.job_id)}`
          : j._jkind === "meshy"
          ? `/api/meshy/cancel/${encodeURIComponent(j.job_id)}`
          : `/api/seedance/cancel/${j.local_id}`;
      apiFetch(cancelUrl, { method: "POST" }).catch(() => {});
    });
    card.addEventListener("click", () => {
      if (j._jkind === "splat") openJobInCenter("splat", j, `splat:${j.job_id}`, `jobs/cloud/modal-3dgs/${j.job_id}`);
      else if (j._jkind === "meshy") openJobInCenter("meshy", j, `meshy:${j.job_id}`, `jobs/cloud/fal-meshy/${j.job_id}`);
      else {
        const lid = String(j.local_id || "");
        const isLocal = j.mode === "local" || lid.startsWith("bl-local");
        const isSr = lid.startsWith("sr-") || String(j.model || "").includes("seedream");
        const kind = isLocal ? "local" : isSr ? "seedream" : "seedance";
        const folder = isLocal ? "jobs/local/blender" : isSr ? "jobs/cloud/seedream" : "jobs/cloud/seedance";
        openJobInCenter(kind, j, `media:${lid}`, `${folder}/${lid}`);
      }
    });
    jobsList.appendChild(card);
  });
}

const announcedMeshyDone = new Set();
export function handleMeshyJobUpdate(m, fromPoll = false) {
  if (!m || !m.job_id) return;
  const prev = state.meshyJobs.get(m.job_id);
  state.meshyJobs.set(m.job_id, m);
  if (!fromPoll) {
    renderMeshyAssetsList();
    renderJobs();
    actions.renderHierarchyTree?.();
    if (state.centerKind === "job" && state.activeJobKey === `meshy:${m.job_id}`) {
      renderGenericJobCenter("meshy", m, `jobs/cloud/fal-meshy/${m.job_id}`);
    }
  }

  const active = !["completed", "failed", "cancelled"].includes(m.stage);
  const liveCard = el("meshy-live-card");
  if (liveCard && (active || (prev && prev.stage !== "completed" && m.stage === "completed"))) {
    liveCard.classList.remove("hidden");
    const qPos = m.queue_position !== null && m.queue_position !== undefined ? ` (pos #${m.queue_position})` : "";
    el("meshy-live-title").textContent = `${m.job_id}${m.request_id ? ` · ${m.request_id.slice(0, 10)}` : ""}`;
    el("meshy-live-stage").textContent = `${String(m.stage).toUpperCase()}${qPos} · ${Number(m.elapsed_sec || 0).toFixed(0)}s`;
    if (Array.isArray(m.logs) && m.logs.length) el("meshy-live-logs").textContent = m.logs.slice(-4).join("\n");
  }

  if (m.stage === "completed" && prev && prev.stage !== "completed" && !announcedMeshyDone.has(m.job_id)) {
    announcedMeshyDone.add(m.job_id);
    refreshMedia();
    refreshState();
    actions.open3DSceneInCenter?.(`meshy:${m.job_id}`, false);
    showToast(`🧊 Meshy 7.1 3D Mesh ready: ${m.job_id} (${Number(m.triangles_total || 0).toLocaleString()} tris)`);
  } else if (m.stage === "failed" && m.error && (!prev || prev.stage !== "failed")) {
    showToast(m.error, true);
  }
}

const announcedCompletions = new Set();
export function handleJobUpdate(data, fromPoll = false) {
  if (!data || !data.local_id) return;
  const prev = state.jobs.get(data.local_id);
  state.jobs.set(data.local_id, data);
  if (!fromPoll) {
    renderJobs();
    actions.renderHierarchyTree?.();
    if (state.centerKind === "job" && state.activeJobKey === `media:${data.local_id}`) {
      const lid = String(data.local_id || "");
      const isLocal = data.mode === "local" || lid.startsWith("bl-local");
      const isSr = lid.startsWith("sr-") || String(data.model || "").includes("seedream");
      const kind = isLocal ? "local" : isSr ? "seedream" : "seedance";
      const folder = isLocal ? "jobs/local/blender" : isSr ? "jobs/cloud/seedream" : "jobs/cloud/seedance";
      renderGenericJobCenter(kind, data, `${folder}/${lid}`);
    }
  }

  if (data.stage === "completed" && data.media_url && !announcedCompletions.has(data.local_id)) {
    announcedCompletions.add(data.local_id);
    if (prev) {
      const isImg = /\.(png|jpe?g|webp)$/i.test(data.output_rel || "") || String(data.local_id).startsWith("sr-");
      refreshMedia();
      if (!fromPoll) refreshState();
      const thItem = {
        path: data.output_rel,
        name: data.output_rel.split("/").pop(),
        kind: isImg ? "image" : "video",
        width: data.width,
        height: data.height,
        size_bytes: 0,
        cost_usd: data.cost_usd,
        url: data.media_url,
      };
      actions.openMediaInCenter?.(thItem);
      if (isImg && el("chk-sr-to-seedance")?.checked) {
        state.firstFrame = thItem;
        renderConditioningSlots();
        updatePricingAndInspector();
      }
      showToast(`${isImg ? "Seedream 5.5" : "Seedance 2.5"} ready: ${data.output_rel}`);
    }
  } else if (data.stage === "failed" && data.error && (!prev || prev.stage !== "failed")) {
    showToast(data.error, true);
  }
}

let cachedModalStatus = null;

export function syncModalStatusPill() {
  const ms = cachedModalStatus || {};
  const dot = el("dot-modal");
  const val = el("val-modal");
  const sha = el("spop-modal-sha");
  const meta = el("spop-modal-meta");
  const badge = el("badge-splat-modal-sha");
  const runDot = el("badge-topbar-run");

  const activeJobs = Array.from(state.splatJobs.values()).filter(
    (j) => !["completed", "failed", "cancelled", "stopped"].includes(String(j.stage || "").toLowerCase())
  );
  const runningJob = activeJobs[0] || null;
  runDot?.classList.toggle("hidden", !runningJob);

  const curS = String(ms.current_short || "").slice(0, 7) || "—";
  const depS = String(ms.deployed_short || "none").slice(0, 7);

  if (runningJob) {
    if (dot) dot.className = "sys-dot busy";
    const stp = runningJob.total_steps ? `${runningJob.step || 0}/${runningJob.total_steps}` : runningJob.stage;
    if (val) val.textContent = `${stp}${runningJob.psnr ? ` · ${Number(runningJob.psnr).toFixed(1)}dB` : ""}`;
  } else if (!ms.configured && ms.error) {
    if (dot) dot.className = "sys-dot muted";
    if (val) val.textContent = "auth?";
  } else if (ms.up_to_date) {
    if (dot) dot.className = "sys-dot ok";
    if (val) val.textContent = `${curS}`;
  } else if (ms.current_short) {
    if (dot) dot.className = "sys-dot warn";
    if (val) val.textContent = `${depS}→${curS}`;
  }

  if (sha) sha.textContent = ms.up_to_date ? `${curS} ✓` : `${depS} → ${curS}`;
  if (badge) badge.textContent = ms.up_to_date ? `Modal ${curS} ✓` : `Auto-Sync → ${curS}`;
  if (meta) {
    meta.innerHTML = `
      <div class="insp-row"><span class="insp-key">worker_sha:</span><span class="insp-val ${ms.up_to_date ? "ok" : ""}">${escapeHtml(ms.current_short || "—")}${ms.up_to_date ? " (deployed)" : " (auto-sync on run)"}</span></div>
      <div class="insp-row"><span class="insp-key">active_3dgs:</span><span class="insp-val">${runningJob ? `${escapeHtml(runningJob.job_id)} (${runningJob.stage} · ${runningJob.step || 0}/${runningJob.total_steps || 0})` : "idle"}</span></div>
      <div class="insp-row"><span class="insp-key">total_splats:</span><span class="insp-val">${state.splatJobs.size} job(s) in workspace</span></div>
    `;
  }
}

function renderModalVersionUi(ms) {
  if (!ms) return;
  cachedModalStatus = ms;
  syncModalStatusPill();
}

function handleModalDeployEvent(ev) {
  if (ev.stage === "deploy") {
    if (el("dot-modal")) el("dot-modal").className = "sys-dot busy";
    if (el("val-modal")) el("val-modal").textContent = `deploy ${String(ev.current_short || "").slice(0, 6)}…`;
    showToast(`☁️ Auto-deploying Modal worker (${ev.previous_short} → ${ev.current_short})…`);
  } else if (ev.stage === "deployed" || ev.event === "modal_status") {
    if (ev.up_to_date || ev.stage === "deployed") {
      renderModalVersionUi(Object.assign({ configured: true, up_to_date: true, deployed_short: ev.current_short }, ev));
      if (ev.deployed || ev.stage === "deployed") showToast(`✅ Modal worker synced to ${ev.current_short} (${ev.elapsed_s}s)`);
    }
  }
}

export async function refreshState() {
  try {
    const s = await apiFetch("/api/state");
    if (s.seedance_spec?.cost_per_sec_usd) state.costPerSec = s.seedance_spec.cost_per_sec_usd;
    if (s.agent_pricing) state.agentPricing = s.agent_pricing;
    if (s.ledger) { state.ledger = s.ledger; renderAgentContextBar(); }
    if (Array.isArray(s.prompt_history)) mergeHistoryStrings(s.prompt_history);

    state.blenderInfo = s.blender || { connected: false };
    if (s.model) state.agentModelId = s.model;
    if (s.role) state.agentRole = s.role;

    if (el("dot-server")) el("dot-server").className = "sys-dot ok";
    if (s.cwd && el("spop-cwd")) el("spop-cwd").textContent = s.cwd;

    // 1. Blender Status Pill & Popover
    const blConn = !!s.blender?.connected;
    const scName = typeof s.blender?.scene === "string" ? s.blender.scene : (s.blender?.scene?.name || "Scene");
    const fpShort = s.blender?.filepath ? s.blender.filepath.replace(/\\/g, "/").split("/").pop() : scName;
    if (el("dot-blender")) el("dot-blender").className = `sys-dot ${blConn ? "ok" : "muted"}`;
    if (el("val-blender")) el("val-blender").textContent = blConn ? `${s.blender.mode}` : "standby";
    if (el("spop-bl-mode")) el("spop-bl-mode").textContent = blConn ? `5.3 ${s.blender.mode}` : "standby";
    if (el("spop-bl-meta")) {
      el("spop-bl-meta").innerHTML = blConn
        ? `<div class="insp-row"><span class="insp-key">scene:</span><span class="insp-val ok">${escapeHtml(fpShort)}${s.blender.is_dirty ? " *" : ""}</span></div>
           <div class="insp-row"><span class="insp-key">camera:</span><span class="insp-val">${escapeHtml(s.blender.active_camera || "none")} · ${(s.blender.objects || []).length} objs</span></div>`
        : `<div class="wb-lbl">Blender 5.3 is on standby. Click Connect or ask the Agent to create geometry.</div>`;
    }

    // 2. OpenRouter Status Pill & Popover
    const orOk = !!s.openrouter?.configured;
    if (el("dot-or")) el("dot-or").className = `sys-dot ${orOk ? "ok" : "warn"}`;
    if (el("val-or")) el("val-or").textContent = orOk ? "ready" : "set key";
    if (el("spop-or-masked")) el("spop-or-masked").textContent = s.openrouter?.masked || (orOk ? "configured" : "unset");

    // 3. fal.ai Status Pill & Popover
    const falOk = !!s.fal?.configured;
    if (el("dot-fal")) el("dot-fal").className = `sys-dot ${falOk ? "ok" : "muted"}`;
    if (el("val-fal")) el("val-fal").textContent = falOk ? "ready" : "set key";
    if (el("spop-fal-masked")) el("spop-fal-masked").textContent = s.fal?.masked || (falOk ? "configured" : "unset");

    // 4. Health & Server Spend Pill
    if (s.health && el("pill-health")) {
      const hs = s.health.status || "healthy";
      el("pill-health").textContent = hs;
      el("pill-health").className = hs === "healthy" ? "pill pill-ok" : hs === "watch" ? "pill pill-warn" : "pill pill-err";
    }
    if (typeof s.stats?.cost === "number") {
      if (el("pill-cost")) el("pill-cost").textContent = `$${s.stats.cost.toFixed(4)}`;
      if (el("val-server")) el("val-server").textContent = `$${s.stats.cost.toFixed(2)}`;
    }
    setAgentBusyUi(!!s.busy);

    if (Array.isArray(s.meshy_jobs)) {
      s.meshy_jobs.forEach((mj) => handleMeshyJobUpdate(mj, true));
      renderMeshyAssetsList();
    }
    if (Array.isArray(s.jobs)) s.jobs.forEach((j) => handleJobUpdate(j, true));
    renderJobs();
    if (Array.isArray(s.splat_jobs)) {
      s.splat_jobs.forEach((sj) => actions.handleSplatJobUpdate ? actions.handleSplatJobUpdate(sj, true) : state.splatJobs.set(sj.job_id, sj));
      actions.populateKnownCamsDropdown?.();
    }
    if (s.modal_status) renderModalVersionUi(s.modal_status);
    else syncModalStatusPill();

    actions.syncTopbarSessionPill?.();
    actions.syncInspectorContext?.();
    actions.renderHierarchyTree?.();
    updatePricingAndInspector();
  } catch (err) {
    if (el("dot-server")) el("dot-server").className = "sys-dot err";
    if (el("val-server")) el("val-server").textContent = String(err.message || "").includes("401") ? "auth!" : "offline";
  }
}

export async function refreshMedia() {
  try {
    const data = await apiFetch("/api/workspace/media");
    state.mediaItems = data.items || [];
    if (Array.isArray(data.blends)) state.blendItems = data.blends;
    populateSourceVideoDropdown();
    actions.populateSplatVideoDropdown?.();
    actions.populateSplatImagesDropdown?.();
    actions.renderHierarchyTree?.();
  } catch (_e) {}
}

export async function refreshSplats() {
  try {
    const res = await apiFetch("/api/splats/jobs");
    if (Array.isArray(res.jobs)) {
      res.jobs.forEach((j) => actions.handleSplatJobUpdate ? actions.handleSplatJobUpdate(j, true) : state.splatJobs.set(j.job_id, j));
      syncModalStatusPill();
      actions.populateKnownCamsDropdown?.();
      actions.renderHierarchyTree?.();
      actions.renderJobs?.();
    }
  } catch (_e) {}
}

export async function refreshModels() {
  apiFetch("/api/seedance/models").then((d) => {
    if (Array.isArray(d.models) && d.models.length) { state.videoModels = d.models; populatePricingDropdowns(); }
  }).catch(() => {});
  apiFetch("/api/seedream/models").then((d) => {
    if (Array.isArray(d.models) && d.models.length) { state.imageModels = d.models; populateSeedreamModels(); }
  }).catch(() => {});
}

export async function refreshShotsList() {
  apiFetch("/api/shots/list").then((r) => {
    if (Array.isArray(r.shots)) { state.exportedShots = r.shots; actions.populateKnownCamsDropdown?.(); }
  }).catch(() => {});
}

let lastEventSeq = 0;
export function connectSSE() {
  const es = new EventSource(withTokenUrl(`/api/events?since=${lastEventSeq}`));
  es.onmessage = (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch (_e) { return; }
    if (typeof ev.seq === "number" && ev.seq > lastEventSeq) lastEventSeq = ev.seq;
    const { type, data } = ev;
    if (type === "seedance-job" && data) handleJobUpdate(data, false);
    else if (type === "meshy-job" && data) handleMeshyJobUpdate(data, false);
    else if (type === "splat-job" && data) actions.handleSplatJobUpdate?.(data);
    else if (type === "modal-deploy" && data) handleModalDeployEvent(data);
    else if (type === "splat-telemetry" && data && window._bbPushSplatTelemetry) window._bbPushSplatTelemetry(data);
    else if (type === "workspace-media-added") refreshMedia();
    else if (type === "journal-updated" && data) {
      state.journalState = data;
      actions.renderJournalInspector?.(data);
    } else if (type === "session-changed" && data) {
      if (data.active_id) state.activeSessionId = data.active_id;
      actions.refreshSessions?.();
    } else if (type === "approval-request" && data) {
      if (!state.pendingApprovals.some((p) => p.id === data.id)) state.pendingApprovals.push(data);
      actions.renderApprovalsBanner?.();
      if (state.centerKind !== "session") {
        showToast("⚠ Agent needs approval — click 🤖 Session to review & approve.");
      }
    } else if (type === "approval-resolved" && data) {
      state.pendingApprovals = state.pendingApprovals.filter((p) => p.id !== data.id);
      actions.renderApprovalsBanner?.();
    } else if (type === "transcript" && data) {
      if (data.type === "user") actions.appendLiveSessionEvent?.("user", data.text);
      else if (data.message || data.text) actions.appendLiveSessionEvent?.(data.type === "warning" ? "warning" : "notice", data.message || data.text);
    } else if (type === "turn-start") {
      setAgentBusyUi(true, "Agent streaming response…");
      actions.appendLiveSessionEvent?.("turn-start", "");
    } else if (type === "text") {
      actions.appendLiveSessionEvent?.("text", data);
    } else if (type === "ops-parsed" && Array.isArray(data)) {
      actions.appendLiveSessionEvent?.("ops-parsed", data);
    } else if (["turn-end", "done", "cancelled", "error"].includes(type)) {
      actions.appendLiveSessionEvent?.("turn-end", "");
      setAgentBusyUi(false);
      refreshState();
      refreshMedia();
      actions.refreshSessions?.();
    } else if (type === "blender-changed") {
      refreshState();
      refreshMedia();
    }
  };
  es.onerror = () => { es.close(); setTimeout(connectSSE, 3000); };
}

export async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  const tok = getToken();
  for (const file of files) {
    const low = file.name.toLowerCase();
    if (low.endsWith(".ply") || low.endsWith(".splat")) {
      if (window._bbEngine) await window._bbEngine.importLocalSplatFile(file);
      continue;
    }
    if (low.endsWith(".glb")) {
      try {
        showToast(`Importing 3D mesh ${file.name}…`);
        const buf = await file.arrayBuffer();
        const rGlb = await fetch(`/api/meshy/upload_glb?filename=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: Object.assign({ "X-Filename": file.name }, tok ? { Authorization: `Bearer ${tok}` } : {}),
          body: buf,
        });
        const dGlb = await rGlb.json();
        if (!rGlb.ok) throw new Error(dGlb.error || `HTTP ${rGlb.status}`);
        state.meshyJobs.set(dGlb.job.job_id, dGlb.job);
        renderMeshyAssetsList();
        actions.renderHierarchyTree?.();
        await actions.open3DSceneInCenter?.(`meshy:${dGlb.job.job_id}`, false);
        showToast(`Loaded 3D mesh '${dGlb.job.job_id}' (${Number(dGlb.job.triangles_total || 0).toLocaleString()} tris)`);
      } catch (err) {
        showToast(`GLB upload failed: ${err.message}`, true);
      }
      continue;
    }
    try {
      showToast(`Uploading ${file.name} (${formatBytes(file.size)})…`);
      const buf = await file.arrayBuffer();
      const res = await fetch(`/api/workspace/upload?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: Object.assign({ "X-Filename": file.name }, tok ? { Authorization: `Bearer ${tok}` } : {}),
        body: buf,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      await refreshMedia();
      if (data.item?.kind === "video") {
        actions.openSplatLaunchDialog?.("video", data.item.path);
      } else {
        actions.openMediaInCenter?.(data.item);
      }
      showToast(`Uploaded ${data.item.name}`);
    } catch (err) {
      showToast(`Upload failed: ${err.message}`, true);
    }
  }
}

export function initComposersAndJobs() {
  el("slot-first")?.addEventListener("click", () => state.firstFrame && actions.openMediaInCenter?.(state.firstFrame));
  el("slot-last")?.addEventListener("click", () => state.lastFrame && actions.openMediaInCenter?.(state.lastFrame));
  el("btn-clear-first")?.addEventListener("click", (e) => { e.stopPropagation(); setFirstFrame(null); });
  el("btn-clear-last")?.addEventListener("click", (e) => { e.stopPropagation(); setLastFrame(null); });
  el("btn-view-vid1")?.addEventListener("click", () => state.sourceVideo && actions.openMediaInCenter?.(state.sourceVideo));
  el("btn-view-vid2")?.addEventListener("click", () => state.sourceVideo2 && actions.openMediaInCenter?.(state.sourceVideo2));

  el("btn-grab3d-first")?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (state.centerKind !== "3d") actions.open3DSceneInCenter?.(state.active3DId, false).then(() => actions.captureCenter3DIntoRole?.("first"));
    else actions.captureCenter3DIntoRole?.("first");
  });
  el("btn-grab3d-last")?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (state.centerKind !== "3d") actions.open3DSceneInCenter?.(state.active3DId, false).then(() => actions.captureCenter3DIntoRole?.("last"));
    else actions.captureCenter3DIntoRole?.("last");
  });
  el("btn-sr-grab3d")?.addEventListener("click", () => {
    if (state.centerKind !== "3d") actions.open3DSceneInCenter?.(state.active3DId, false).then(() => actions.captureCenter3DIntoRole?.("seedream"));
    else actions.captureCenter3DIntoRole?.("seedream");
  });
  el("btn-meshy-grab3d")?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (state.centerKind !== "3d") actions.open3DSceneInCenter?.(state.active3DId, false).then(() => actions.captureCenter3DIntoRole?.("meshy"));
    else actions.captureCenter3DIntoRole?.("meshy");
  });

  ["sel-duration", "sel-size", "select-video-model", "chk-vid-first", "chk-vid-last", "sel-vid-mids", "chk-vid-audio", "chk-gen-audio"].forEach((id) => {
    el(id)?.addEventListener("change", updatePricingAndInspector);
  });
  el("input-prompt")?.addEventListener("input", updatePricingAndInspector);

  el("sel-source-video")?.addEventListener("change", (e) => {
    const it = e.target.value ? state.mediaItems.find((x) => x.path === e.target.value) || null : null;
    setSourceVideo(it);
    if (it) actions.openMediaInCenter?.(it);
  });
  el("sel-source-video-2")?.addEventListener("change", (e) => {
    const it = e.target.value ? state.mediaItems.find((x) => x.path === e.target.value) || null : null;
    setSourceVideo2(it);
    if (it) actions.openMediaInCenter?.(it);
  });
  el("sel-dual-mode")?.addEventListener("change", () => {
    state.dualVideoMode = el("sel-dual-mode").value;
    renderConditioningSlots();
    updatePricingAndInspector();
  });
  el("btn-swap-videos")?.addEventListener("click", () => {
    const tv = state.sourceVideo, tp = state.sourceVideoProbe;
    state.sourceVideo = state.sourceVideo2; state.sourceVideoProbe = state.sourceVideo2Probe;
    state.sourceVideo2 = tv; state.sourceVideo2Probe = tp;
    populateSourceVideoDropdown();
    renderConditioningSlots();
    actions.renderHierarchyTree?.();
    updatePricingAndInspector();
  });

  el("btn-clear-video")?.addEventListener("click", () => setSourceVideo(null));
  el("btn-clear-video-2")?.addEventListener("click", () => setSourceVideo2(null));
  el("btn-clear-cond")?.addEventListener("click", () => {
    state.sourceVideo = null; state.sourceVideo2 = null; state.refVideos = [];
    state.firstFrame = null; state.lastFrame = null; state.refImages = []; state.refAudio = null;
    populateSourceVideoDropdown();
    renderConditioningSlots();
    actions.renderHierarchyTree?.();
    updatePricingAndInspector();
    actions.syncViewModeButtons?.();
  });

  el("mode-pills")?.addEventListener("click", (e) => {
    const p = e.target.closest(".mode-pill");
    if (!p || !p.dataset.mode) return;
    document.querySelectorAll("#mode-pills .mode-pill").forEach((b) => b.classList.remove("active"));
    p.classList.add("active");
    state.mode = p.dataset.mode;
    updatePricingAndInspector();
  });

  el("btn-pull-cam-prompt")?.addEventListener("click", async () => {
    try {
      const res = await apiFetch("/api/splats/camera_prompt", { method: "POST", json: {} });
      if (res.seedance_camera_prompt) {
        const inp = el("input-prompt");
        const cur = inp.value.trim();
        inp.value = cur ? `${cur} ${res.seedance_camera_prompt}` : res.seedance_camera_prompt;
        updatePricingAndInspector();
        showToast(`Pulled Blender camera '${res.camera}' into prompt`);
      }
    } catch (err) {
      showToast(err.message, true);
    }
  });

  const submitSeedance = async () => {
    const prompt = el("input-prompt").value.trim();
    const durVal = el("sel-duration").value;
    const sizeVal = el("sel-size").value;
    const seedVal = el("input-seed").value.trim();
    if (prompt) pushLocalPromptHistory(prompt);

    const btn = el("btn-generate");
    btn.disabled = true;
    try {
      const res = await apiFetch("/api/seedance/generate", {
        method: "POST",
        json: {
          model: el("select-video-model").value || "bytedance/seedance-2.5",
          prompt,
          mode: state.mode,
          duration: durVal !== "" ? Number(durVal) : null,
          size: sizeVal || null,
          seed: seedVal !== "" ? Number(seedVal) : null,
          generate_audio: el("chk-gen-audio").checked ? true : null,
          attach_to_context: el("chk-attach-ctx").checked,
          first_frame: state.firstFrame ? state.firstFrame.path : null,
          last_frame: state.lastFrame ? state.lastFrame.path : null,
          video_path: state.sourceVideo ? state.sourceVideo.path : null,
          video_path_2: state.sourceVideo2 ? state.sourceVideo2.path : null,
          dual_video_mode: el("sel-dual-mode").value || "bridge_v1_to_v2",
          video_lock_first: el("chk-vid-first").checked,
          video_lock_last: el("chk-vid-last").checked,
          video_mid_frames: Number(el("sel-vid-mids").value || 2),
          extract_video_audio: el("chk-vid-audio").checked,
          reference_images: state.refImages.map((r) => r.path),
          reference_videos: state.refVideos.map((v) => v.path),
          audio_path: state.refAudio ? state.refAudio.path : null,
        },
      });
      state.jobs.set(res.job.local_id, res.job);
      renderJobs();
      showToast(`Submitted Seedance 2.5 job ${res.job.local_id}…`);
    } catch (err) {
      showToast(err.message, true);
      if (err.message.includes("OPENROUTER_API_KEY")) actions.openStatusPopover?.("openrouter", el("btn-or-key"));
    } finally {
      btn.disabled = false;
    }
  };
  el("btn-generate")?.addEventListener("click", submitSeedance);
  el("input-prompt")?.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); submitSeedance(); }
  });

  // Seedream 5.5
  ["sel-sr-res", "sel-sr-aspect", "sel-sr-bg", "sel-sr-fmt", "select-image-model"].forEach((id) => {
    el(id)?.addEventListener("change", updateSeedreamInspector);
  });
  el("input-sr-prompt")?.addEventListener("input", updateSeedreamInspector);
  el("btn-sr-clear-refs")?.addEventListener("click", () => {
    state.seedreamRefs = [];
    renderSeedreamRefs();
    actions.renderHierarchyTree?.();
    updateSeedreamInspector();
  });
  el("sr-mode-pills")?.addEventListener("click", (e) => {
    const p = e.target.closest(".mode-pill");
    if (!p || !p.dataset.srmode) return;
    state.seedreamMode = p.dataset.srmode;
    el("sr-mode-pills").querySelectorAll(".mode-pill").forEach((b) => b.classList.toggle("active", b === p));
    if (state.seedreamMode === "cutout_prop") {
      el("sel-sr-bg").value = "transparent";
      el("sel-sr-fmt").value = "png";
    }
    updateSeedreamInspector();
  });

  const submitSeedream = async () => {
    const prompt = (el("input-sr-prompt")?.value || "").trim();
    if (!prompt) {
      showToast("Enter a Seedream 5.5 prompt first.", true);
      el("input-sr-prompt")?.focus();
      return;
    }
    pushLocalPromptHistory(prompt);
    const seedVal = (el("inp-sr-seed")?.value || "").trim();
    const btn = el("btn-generate-seedream");
    btn.disabled = true;
    try {
      const res = await apiFetch("/api/seedream/generate", {
        method: "POST",
        json: {
          model: el("select-image-model")?.value || "bytedance-seed/seedream-5-5",
          prompt,
          mode: state.seedreamMode,
          resolution: el("sel-sr-res")?.value || "1K",
          aspect_ratio: el("sel-sr-aspect")?.value || "16:9",
          background: el("sel-sr-bg")?.value || "auto",
          output_format: el("sel-sr-fmt")?.value || "png",
          seed: seedVal !== "" ? Number(seedVal) : null,
          reference_images: state.seedreamRefs.map((r) => r.path),
          attach_to_context: el("chk-sr-attach-ctx")?.checked ?? true,
        },
      });
      state.jobs.set(res.job.local_id, res.job);
      renderJobs();
      showToast(`Submitted Seedream 5.5 job ${res.job.local_id}…`);
    } catch (err) {
      showToast(err.message, true);
      if (err.message.includes("OPENROUTER_API_KEY")) actions.openStatusPopover?.("openrouter", el("btn-or-key"));
    } finally {
      btn.disabled = false;
    }
  };
  el("btn-generate-seedream")?.addEventListener("click", submitSeedream);
  el("input-sr-prompt")?.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); submitSeedream(); }
  });

  // Meshy 7.1
  el("slot-meshy-img")?.addEventListener("click", () => state.meshyInputImage && actions.openMediaInCenter?.(state.meshyInputImage));
  el("slot-meshy-teximg")?.addEventListener("click", () => {
    if (state.meshyTextureImage) actions.openMediaInCenter?.(state.meshyTextureImage);
    else if (state.stageItem && state.stageItem.kind === "image") setMeshyTextureImage(state.stageItem);
  });
  el("btn-clear-meshy-img")?.addEventListener("click", (e) => { e.stopPropagation(); setMeshyInputImage(null); });
  el("btn-clear-meshy-teximg")?.addEventListener("click", (e) => { e.stopPropagation(); setMeshyTextureImage(null); });

  [
    "sel-meshy-topology", "inp-meshy-polycount", "sel-meshy-symmetry", "sel-meshy-geomres",
    "chk-meshy-remesh", "chk-meshy-texture", "chk-meshy-pbr", "sel-meshy-texmodel",
    "sel-meshy-pose", "chk-meshy-rig", "chk-meshy-fingers", "chk-meshy-anim",
    "sel-meshy-skeleton", "inp-meshy-height", "inp-meshy-animid",
  ].forEach((id) => {
    el(id)?.addEventListener("change", updateMeshyInspector);
    el(id)?.addEventListener("input", updateMeshyInspector);
  });

  el("btn-generate-meshy")?.addEventListener("click", async () => {
    let inputImg = state.meshyInputImage;
    if (!inputImg && state.centerKind === "media" && state.stageItem && state.stageItem.kind === "image") {
      inputImg = state.stageItem;
      state.meshyInputImage = inputImg;
      renderMeshySlots();
    }
    if (!inputImg) return showToast("Select an Input Image first (click 🧊 3D on an image or 📸 3D).", true);
    const texPrompt = (el("inp-meshy-texprompt")?.value || "").trim();
    const texModel = el("sel-meshy-texmodel")?.value || "";
    if (texModel === "meshy-n1" && texPrompt) {
      return showToast("meshy-n1 uses reference images only — clear Texture Prompt or switch Texture Model.", true);
    }

    const btn = el("btn-generate-meshy");
    btn.disabled = true;
    try {
      const res = await apiFetch("/api/meshy/generate", {
        method: "POST",
        json: {
          image: inputImg.path,
          topology: el("sel-meshy-topology")?.value || "triangle",
          target_polycount: Number(el("inp-meshy-polycount")?.value || 30000),
          symmetry_mode: el("sel-meshy-symmetry")?.value || "auto",
          geometry_resolution: el("sel-meshy-geomres")?.value || null,
          should_remesh: el("chk-meshy-remesh")?.checked ?? true,
          should_texture: el("chk-meshy-texture")?.checked ?? true,
          texture_model: texModel || null,
          enable_pbr: el("chk-meshy-pbr")?.checked ?? false,
          pose_mode: el("sel-meshy-pose")?.value || "",
          texture_prompt: texPrompt || null,
          texture_image: state.meshyTextureImage ? state.meshyTextureImage.path : null,
          enable_rigging: el("chk-meshy-rig")?.checked ?? false,
          include_fingers: el("chk-meshy-fingers")?.checked ?? false,
          skeleton_template: el("sel-meshy-skeleton")?.value || null,
          rigging_height_meters: Number(el("inp-meshy-height")?.value || 1.7),
          enable_animation: el("chk-meshy-anim")?.checked ?? false,
          animation_action_id: Number(el("inp-meshy-animid")?.value || 92),
          enable_safety_checker: (el("sel-meshy-safety")?.value || "true") === "true",
          auto_import: el("chk-meshy-autoimport")?.checked ?? true,
        },
      });
      if (res.job) handleMeshyJobUpdate(res.job, false);
      showToast(`Submitted Meshy 7.1 job ${res.job?.job_id || ""} to fal queue…`);
    } catch (err) {
      showToast(err.message, true);
      if (err.message.includes("FAL_KEY")) actions.openStatusPopover?.("fal", el("btn-fal-key"));
    } finally {
      btn.disabled = false;
    }
  });

  // Uploads, Dropzones, Blender Capture & Modal Version Sync
  el("btn-upload-trigger")?.addEventListener("click", () => el("file-upload-input")?.click());
  el("file-upload-input")?.addEventListener("change", (e) => { uploadFiles(e.target.files); e.target.value = ""; });

  [el("dropzone-library"), el("theater-box")].forEach((zone) => {
    if (!zone) return;
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      if (zone.id === "theater-box") el("stage-drop-overlay")?.classList.remove("hidden");
      else zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", (e) => {
      if (e.target === zone) {
        el("stage-drop-overlay")?.classList.add("hidden");
        zone.classList.remove("drag-over");
      }
    });
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      el("stage-drop-overlay")?.classList.add("hidden");
      zone.classList.remove("drag-over");
      if (e.dataTransfer?.files) uploadFiles(e.dataTransfer.files);
    });
  });

  const captureFromBlender = async (kind) => {
    const btn = kind === "playblast" ? el("btn-cap-playblast") : el("btn-cap-still");
    if (btn) btn.disabled = true;
    try {
      showToast(`Rendering ${kind} from Blender 5.3…`);
      const res = await apiFetch("/api/blender/capture", { method: "POST", json: { kind } });
      await refreshMedia();
      actions.openMediaInCenter?.(res.item);
      if (res.item.kind === "video") setSourceVideo(res.item);
      else if (!state.firstFrame) setFirstFrame(res.item);
      showToast(`Captured ${res.item.name}`);
    } catch (err) {
      showToast(`Blender capture failed: ${err.message}`, true);
    } finally {
      if (btn) btn.disabled = false;
    }
  };
  el("btn-cap-still")?.addEventListener("click", () => captureFromBlender("still"));
  el("btn-cap-playblast")?.addEventListener("click", () => captureFromBlender("playblast"));
  actions.captureFromBlender = captureFromBlender;

  el("btn-spop-modal-deploy")?.addEventListener("click", async () => {
    if (el("val-modal")) el("val-modal").textContent = "syncing…";
    try {
      const res = await apiFetch("/api/splats/modal_deploy", { method: "POST", json: { force: false } });
      renderModalVersionUi(res);
      showToast(res.deployed ? `✅ Deployed Modal worker ${res.current_short} in ${res.elapsed_s}s` : `Modal worker already up to date (${res.current_short})`);
    } catch (err) {
      showToast(`Modal sync failed: ${err.message}`, true);
    }
  });

  // Reliable Live Background Poller: polls `/api/splats/jobs` + active `/api/splats/telemetry/{id}`
  // every 2s whenever any job is running or the Run/Job view is open, so train status ALWAYS updates!
  let pollInFlight = false;
  setInterval(async () => {
    if (pollInFlight || document.hidden) return;
    const anySplatRunning = Array.from(state.splatJobs.values()).some(
      (j) => !["completed", "failed", "cancelled", "stopped"].includes(String(j.stage || "").toLowerCase())
    );
    const anyMediaOrMeshyRunning =
      Array.from(state.jobs.values()).some((j) => !["completed", "failed", "cancelled", "stopped"].includes(j.stage)) ||
      Array.from(state.meshyJobs.values()).some((m) => !["completed", "failed", "cancelled"].includes(m.stage));
    const onRunView = state.centerKind === "kernel" || state.centerKind === "job";

    if (!anySplatRunning && !anyMediaOrMeshyRunning && !onRunView) return;
    pollInFlight = true;
    try {
      if (anySplatRunning || state.centerKind === "kernel") {
        await refreshSplats();
        const activeId = actions.getActiveWbJobId?.();
        if (activeId && (anySplatRunning || state.centerKind === "kernel")) {
          await window._bbHydrateSplatTelemetry?.(activeId);
        }
      }
      if (anyMediaOrMeshyRunning) {
        await refreshState();
      }
    } catch (_e) {
    } finally {
      pollInFlight = false;
    }
  }, 2000);

  actions.syncModalStatusPill = syncModalStatusPill;
  actions.updatePricingAndInspector = updatePricingAndInspector;
  actions.updateSeedreamInspector = updateSeedreamInspector;
  actions.updateMeshyInspector = updateMeshyInspector;
  actions.renderConditioningSlots = renderConditioningSlots;
  actions.renderSeedreamRefs = renderSeedreamRefs;
  actions.renderMeshySlots = renderMeshySlots;
  actions.setSourceVideo = setSourceVideo;
  actions.setSourceVideo2 = setSourceVideo2;
  actions.setFirstFrame = setFirstFrame;
  actions.setLastFrame = setLastFrame;
  actions.setRefAudio = setRefAudio;
  actions.toggleSeedreamRef = toggleSeedreamRef;
  actions.setMeshyInputImage = setMeshyInputImage;
  actions.importMeshyToBlender = importMeshyToBlender;
  actions.openJobInCenter = openJobInCenter;
  actions.renderJobs = renderJobs;
  actions.refreshState = refreshState;
  actions.refreshMedia = refreshMedia;
  actions.refreshSplats = refreshSplats;
  actions.refreshModels = refreshModels;
  actions.refreshShotsList = refreshShotsList;
}