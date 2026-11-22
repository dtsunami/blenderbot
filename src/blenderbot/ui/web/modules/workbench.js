// Resizable 3-Pane VS Code Workbench layout, splitters, Context-Driven Inspector, Generative Studio Switcher & Prompt History.

import { actions, apiFetch, el, escapeHtml, formatBytes, showToast, state, withTokenUrl } from "./state.js";

export function persistWorkbenchLayout() {
  try {
    localStorage.setItem(
      "bb_studio_layout_v3",
      JSON.stringify({
        leftPaneW: state.leftPaneW,
        rightPaneW: state.rightPaneW,
        hideLeft: state.hideLeft,
        hideRight: state.hideRight,
      })
    );
  } catch (_e) {}
}

export function loadSavedWorkbenchLayout() {
  try {
    const savedLayout = JSON.parse(
      localStorage.getItem("bb_studio_layout_v3") ||
        localStorage.getItem("bb_studio_layout_v2") ||
        "{}"
    );
    if (typeof savedLayout.leftPaneW === "number") state.leftPaneW = Math.max(185, Math.min(720, savedLayout.leftPaneW));
    if (typeof savedLayout.rightPaneW === "number") state.rightPaneW = Math.max(265, Math.min(820, savedLayout.rightPaneW));
    if (typeof savedLayout.hideLeft === "boolean") state.hideLeft = savedLayout.hideLeft;
    if (typeof savedLayout.hideRight === "boolean") state.hideRight = savedLayout.hideRight;
  } catch (_e) {}
}

export function applyWorkbenchLayout() {
  const root = document.documentElement;
  root.style.setProperty("--left-pane-w", `${Math.round(state.leftPaneW)}px`);
  root.style.setProperty("--right-pane-w", `${Math.round(state.rightPaneW)}px`);

  document.body.classList.toggle("hide-left", !!state.hideLeft);
  document.body.classList.toggle("hide-right", !!state.hideRight);

  el("btn-toggle-left")?.classList.toggle("active", !state.hideLeft);
  el("btn-toggle-right")?.classList.toggle("active", !state.hideRight);
  el("edge-restore-left")?.classList.toggle("hidden", !state.hideLeft);
  el("edge-restore-right")?.classList.toggle("hidden", !state.hideRight);
  actions.syncTopbarSessionPill?.();
}

export function setLeftHidden(hide) {
  state.hideLeft = !!hide;
  if (state.hideLeft && actions.hideHoverTooltip) actions.hideHoverTooltip();
  applyWorkbenchLayout();
  persistWorkbenchLayout();
}

export function setRightHidden(hide) {
  state.hideRight = !!hide;
  applyWorkbenchLayout();
  persistWorkbenchLayout();
}

export function setAgentDrawerOpen(open, focusInput = false) {
  if (open) {
    actions.openSessionInCenter?.(state.activeSessionId || "active", focusInput);
  }
}

export function toggleAgentDrawer(focusInput = true) {
  actions.toggleSessionInCenter?.();
  if (focusInput && state.centerKind === "session") {
    setTimeout(() => el("agent-input")?.focus(), 20);
  }
}

export function toggleZenMode() {
  const bothHidden = state.hideLeft && state.hideRight;
  state.hideLeft = !bothHidden;
  state.hideRight = !bothHidden;
  if (actions.hideHoverTooltip) actions.hideHoverTooltip();
  applyWorkbenchLayout();
  persistWorkbenchLayout();
}

export function openGenerativeStudioInCenter(studio = "seedream") {
  const clean = ["seedream", "seedance", "meshy"].includes(studio) ? studio : "seedream";
  state.activeGenStudio = clean;
  actions.hideAllCenterViews?.();
  state.centerKind = "gen";

  el("theater-gen-wrap")?.classList.remove("hidden");
  el("stage-3d-btns")?.classList.add("hidden");
  el("stage-media-btns")?.classList.add("hidden");

  const titles = {
    seedream: ["GEN IMAGE", "Seedream 5.5 Image Studio", "OpenRouter /api/v1/images (Experimental)"],
    seedance: ["GEN VIDEO", "Seedance 2.5 Video Studio", "OpenRouter /api/v1/videos (Experimental)"],
    meshy: ["GEN 3D GLB", "Meshy 7.1 Image-to-3D Studio", "fal.ai Queue API (Experimental)"],
  };
  const [badge, title, meta] = titles[clean];
  if (el("stage-kind-badge")) el("stage-kind-badge").textContent = badge;
  if (el("stage-title")) el("stage-title").textContent = title;
  if (el("theater-meta")) el("theater-meta").textContent = meta;

  el("panel-seedream-composer")?.classList.toggle("hidden", clean !== "seedream");
  el("panel-seedance-composer")?.classList.toggle("hidden", clean !== "seedance");
  el("panel-meshy-composer")?.classList.toggle("hidden", clean !== "meshy");

  document.querySelectorAll("#gen-studio-tabs [data-genstudio]").forEach((b) => {
    b.classList.toggle("active", b.dataset.genstudio === clean);
  });

  if (clean === "seedream") actions.updateSeedreamInspector?.();
  if (clean === "seedance") actions.updatePricingAndInspector?.();
  if (clean === "meshy") actions.updateMeshyInspector?.();

  syncInspectorContext();
  actions.syncViewModeButtons?.();
  actions.renderHierarchyTree?.();
}

function renderBlenderSceneInspector() {
  const bl = state.blenderInfo || {};
  const badge = el("badge-bl-insp-status");
  if (badge) {
    badge.textContent = bl.connected ? `${bl.mode || "connected"}` : "standby";
    badge.className = `sec-meta-pill ${bl.connected ? "ok" : ""}`;
  }
  const box = el("insp-blender-passport");
  if (!box) return;
  if (!bl.connected) {
    box.innerHTML = `<div class="empty-placeholder">Blender 5.3 is on standby. Click <b>Blender 5.3</b> in the top bar or ask the Agent to create geometry.</div>`;
    return;
  }
  const scName = typeof bl.scene === "string" ? bl.scene : bl.scene?.name || "Scene";
  const fpName = bl.filepath ? bl.filepath.replace(/\\/g, "/").split("/").pop() : "Unsaved Scene";
  const objs = Array.isArray(bl.objects) ? bl.objects : [];
  const objChips = objs.length
    ? objs
        .slice(0, 18)
        .map((o) => `<span class="ctx-obj-chip" title="${escapeHtml(o.type || "")}">${escapeHtml(o.name)}</span>`)
        .join(" ")
    : `<span class="wb-lbl">Empty scene</span>`;

  box.innerHTML = `
    <div class="insp-row"><span class="insp-key">blend_file:</span><span class="insp-val ${bl.filepath ? "ok" : ""}">${escapeHtml(fpName)}${bl.is_dirty ? " *" : ""}</span></div>
    <div class="insp-row"><span class="insp-key">scene / cam:</span><span class="insp-val">${escapeHtml(scName)} · cam: ${escapeHtml(bl.active_camera || "none")}</span></div>
    <div class="insp-obj-chips-wrap">${objChips}</div>
  `;
}

function renderMediaInspector() {
  const box = el("insp-media-details");
  const it = state.stageItem;
  if (!box) return;
  if (!it) {
    box.innerHTML = `<div class="empty-placeholder">Select an image, video, or audio file.</div>`;
    return;
  }
  const u = withTokenUrl(it.url);
  box.innerHTML = `
    <div class="insp-row"><span class="insp-key">name:</span><span class="insp-val ok">${escapeHtml(it.name || "")}</span></div>
    <div class="insp-row"><span class="insp-key">path:</span><span class="insp-val">${escapeHtml(it.path || "")}</span></div>
    <div class="insp-row"><span class="insp-key">kind / size:</span><span class="insp-val">${escapeHtml((it.kind || "").toUpperCase())} · ${formatBytes(it.size_bytes)}${it.width && it.height ? ` · ${it.width}×${it.height}` : ""}</span></div>
    ${it.cost_usd ? `<div class="insp-row"><span class="insp-key">cost:</span><span class="insp-val">$${Number(it.cost_usd).toFixed(4)} USD</span></div>` : ""}
    <div class="btn-grid-2" style="margin-top:6px">
      <button type="button" id="btn-minsp-at-agent" class="btn btn-primary btn-xs">🤖 @Attach to Session</button>
      <a href="${u}&download=1" download="${escapeHtml(it.name || "asset")}" class="btn btn-secondary btn-xs">⬇ Download</a>
      ${it.kind === "video" ? `<button type="button" id="btn-minsp-to-splat" class="btn btn-accent btn-xs">🌐 Train 3DGS Splat</button>` : ""}
      ${it.kind === "image" ? `<button type="button" id="btn-minsp-to-meshy" class="btn btn-secondary btn-xs">🧊 Image → 3D GLB</button>` : ""}
      ${it.kind === "image" ? `<button type="button" id="btn-minsp-to-sr" class="btn btn-ghost btn-xs">🎨 +Seedream Ref</button>` : ""}
      ${it.kind === "video" ? `<button type="button" id="btn-minsp-to-sd" class="btn btn-ghost btn-xs">🎬 +Seedance Vid</button>` : ""}
    </div>
  `;
  el("btn-minsp-at-agent")?.addEventListener("click", async () => {
    await actions.openSessionInCenter?.(state.activeSessionId || "active", true);
    const inp = el("agent-input");
    if (inp) {
      inp.value = `${inp.value ? inp.value.trim() + " " : ""}@${it.path} `;
      inp.focus();
    }
  });
  el("btn-minsp-to-splat")?.addEventListener("click", () => actions.openSplatLaunchDialog?.("video", it.path));
  el("btn-minsp-to-meshy")?.addEventListener("click", () => {
    actions.setMeshyInputImage?.(it, true);
    openGenerativeStudioInCenter("meshy");
  });
  el("btn-minsp-to-sr")?.addEventListener("click", () => {
    actions.toggleSeedreamRef?.(it);
    openGenerativeStudioInCenter("seedream");
  });
  el("btn-minsp-to-sd")?.addEventListener("click", () => {
    actions.setSourceVideo?.(it);
    openGenerativeStudioInCenter("seedance");
  });
}

export function openStatusPopover(name, anchorEl) {
  const pop = el("status-popover");
  if (!pop) return;
  const curOpen = !pop.classList.contains("hidden") && pop.dataset.active === name;
  document.querySelectorAll("#topbar-status-cluster .sys-pill").forEach((b) => b.classList.remove("open"));
  if (curOpen || !name) {
    pop.classList.add("hidden");
    pop.dataset.active = "";
    return;
  }
  pop.dataset.active = name;
  pop.querySelectorAll(".spop-pane").forEach((p) => {
    p.classList.toggle("hidden", p.dataset.spop !== name);
  });
  pop.classList.remove("hidden");
  anchorEl?.classList.add("open");

  if (anchorEl) {
    const r = anchorEl.getBoundingClientRect();
    const pw = pop.offsetWidth || 320;
    const left = Math.max(12, Math.min(window.innerWidth - pw - 12, r.right - pw));
    pop.style.top = `${Math.round(r.bottom + 6)}px`;
    pop.style.left = `${Math.round(left)}px`;
  }
  if (name === "openrouter") setTimeout(() => el("input-or-key")?.focus(), 20);
  if (name === "fal") setTimeout(() => el("input-fal-key")?.focus(), 20);
}

export function closeStatusPopover() {
  const pop = el("status-popover");
  if (!pop) return;
  pop.classList.add("hidden");
  pop.dataset.active = "";
  document.querySelectorAll("#topbar-status-cluster .sys-pill").forEach((b) => b.classList.remove("open"));
}

export function syncInspectorContext() {
  const ck = state.centerKind;
  let activePanel = "3d";
  let title = "3D & BLENDER INSPECTOR";
  let pill = "3D Stage";

  if (ck === "session") {
    activePanel = "session";
    title = "SESSION INSPECTOR";
    pill = state.viewedSessionId === state.activeSessionId ? "Live" : "History";
  } else if (ck === "media") {
    activePanel = "media";
    title = "ASSET INSPECTOR";
    pill = (state.stageItem?.kind || "media").toUpperCase();
  } else if (ck === "launch" || ck === "gen" || ck === "job") {
    activePanel = "jobs";
    title = ck === "launch" ? "CLOUD SPLAT QUEUE" : "CLOUD & LOCAL JOBS";
    pill = ck === "launch" ? "3DGS" : ck === "gen" ? state.activeGenStudio.toUpperCase() : "Job";
  } else if (ck === "kernel" || ck === "compare" || ck === "how") {
    activePanel = "3d";
    title = ck === "kernel" ? "RUN CONTROLS & RESUME" : "TRAJECTORY & SPLAT";
    pill = ck === "kernel" ? (state.activeSplatJob?.stage || "Run").toUpperCase() : "Path";
  } else if (ck === "blend") {
    activePanel = "3d";
    title = "BLENDER SCENE";
    pill = ".blend";
  } else {
    const isLiveBl = state.active3DId === "__blender_live__";
    const isSp = !!state.activeSplatJob || String(state.active3DId).includes("splat") || state.active3DId === "__courtyard__";
    activePanel = "3d";
    title = isLiveBl ? "BLENDER 5.3 SCENE" : isSp ? "3D / SPLAT INSPECTOR" : "3D VIEWPORT";
    pill = isLiveBl ? "Blender 5.3" : state.activeSplatJob ? "3DGS" : "Orbit/FPS";
  }

  state.inspectorTab = activePanel;
  if (el("insp-context-title")) el("insp-context-title").textContent = title;
  if (el("insp-context-pill")) el("insp-context-pill").textContent = pill;

  el("panel-session-inspector")?.classList.toggle("hidden", activePanel !== "session");
  el("panel-fps-splat")?.classList.toggle("hidden", activePanel !== "3d");
  el("panel-media-inspector")?.classList.toggle("hidden", activePanel !== "media");
  el("panel-copilot-jobs")?.classList.toggle("hidden", activePanel !== "jobs");

  // Context-focus individual sections inside `panel-fps-splat` so the Inspector only shows tools relevant to the active stage!
  const on3DCanvas = ck === "3d";
  const onSplatRunOrCmp = ck === "kernel" || ck === "compare" || ck === "how";
  const isPureMesh =
    String(state.active3DId).startsWith("meshy:") ||
    state.active3DId === "__default_cube_solid__" ||
    state.active3DId === "__blender_live__" ||
    ck === "blend";

  el("sec-fps-cam")?.classList.toggle("hidden", !on3DCanvas);
  el("sec-fps-rail")?.classList.toggle("hidden", !on3DCanvas);
  el("sec-fps-splat")?.classList.toggle("hidden", !on3DCanvas || isPureMesh);
  el("sec-bl-live-passport")?.classList.toggle("hidden", onSplatRunOrCmp);

  const cloudDrawer = el("drawer-cloud-splat");
  if (cloudDrawer) {
    const jobRunning = state.activeSplatJob && !["completed", "failed", "cancelled", "stopped"].includes(state.activeSplatJob.stage);
    cloudDrawer.classList.toggle("hidden", isPureMesh && on3DCanvas);
    if (ck === "kernel" && jobRunning) cloudDrawer.open = true;
  }

  renderBlenderSceneInspector();
  if (activePanel === "media") renderMediaInspector();
}

export function switchInspectorTab(tab) {
  if (["seedance", "seedream", "meshy"].includes(tab)) {
    openGenerativeStudioInCenter(tab);
    return;
  }
  if (tab === "copilot") {
    syncInspectorContext();
    return;
  }
  syncInspectorContext();
}

export function mergeHistoryStrings(serverList) {
  const seen = new Set();
  const combined = [];
  [...(serverList || []), ...state.promptHistory].forEach((p) => {
    const t = String(p || "").trim();
    if (t && !seen.has(t)) {
      seen.add(t);
      combined.push(t);
    }
  });
  state.promptHistory = combined.slice(-100);
  localStorage.setItem("bb_agent_prompts", JSON.stringify(state.promptHistory));
  renderPromptHistoryDropdown();
}

export function pushLocalPromptHistory(text) {
  const clean = String(text || "").trim();
  if (!clean) return;
  state.promptHistory = state.promptHistory.filter((x) => x !== clean);
  state.promptHistory.push(clean);
  if (state.promptHistory.length > 100) state.promptHistory.shift();
  localStorage.setItem("bb_agent_prompts", JSON.stringify(state.promptHistory));
  renderPromptHistoryDropdown();
}

export function renderPromptHistoryDropdown() {
  const sel = el("sel-prompt-history");
  if (!sel) return;
  sel.innerHTML = `<option value="">🕒 History (${state.promptHistory.length})…</option>`;
  [...state.promptHistory].reverse().forEach((p, idx) => {
    const opt = document.createElement("option");
    opt.value = String(state.promptHistory.length - 1 - idx);
    opt.textContent = p.length > 56 ? p.slice(0, 53) + "…" : p;
    sel.appendChild(opt);
  });
}

export function renderAgentContextBar() {
  const lg = state.ledger;
  if (!lg) return;
  const txt = `${lg.total_tokens.toLocaleString()} / ${(lg.limit / 1000).toFixed(0)}k tok (${lg.pct}%)`;
  if (el("agent-ctx-summary")) el("agent-ctx-summary").textContent = `Context: ${txt}`;
  if (el("badge-ctx-pct")) el("badge-ctx-pct").textContent = `${lg.pct}%`;
  const fill = el("insp-ctx-fill");
  if (fill) fill.style.width = `${Math.min(100, Math.max(2, lg.pct))}%`;

  const renderSegsInto = (container) => {
    if (!container) return;
    container.innerHTML = (lg.segments || [])
      .map(
        (s) =>
          `<div class="ctx-seg-row"><span>#${s.id} [${escapeHtml(s.type)}] ${escapeHtml(s.label)} <em>(${s.tokens}t)</em></span><span>` +
          `<button type="button" data-pin-seg="${s.id}" title="Pin / Unpin segment">${s.pinned ? "📌" : "📍"}</button>` +
          `${s.pinned ? "" : `<button type="button" data-drop-seg="${s.id}" title="Drop segment">✕</button>`}</span></div>`
      )
      .join("");
    container.querySelectorAll("button[data-drop-seg]").forEach((b) => {
      b.addEventListener("click", async (e) => {
        e.stopPropagation();
        await apiFetch(`/api/context/${b.dataset.dropSeg}`, { method: "DELETE" }).catch(() => {});
        actions.refreshState?.();
      });
    });
    container.querySelectorAll("button[data-pin-seg]").forEach((b) => {
      b.addEventListener("click", async (e) => {
        e.stopPropagation();
        await apiFetch(`/api/context/${b.dataset.pinSeg}/pin`, { method: "POST" }).catch(() => {});
        actions.refreshState?.();
      });
    });
  };
  renderSegsInto(el("agent-ctx-drawer"));
  renderSegsInto(el("insp-ctx-segments"));
}

export function setAgentBusyUi(busy, _previewText = "") {
  state.agentBusy = !!busy;
  el("btn-agent-cancel")?.classList.toggle("hidden", !busy);
  const pill = el("agent-status-pill");
  if (pill) {
    pill.textContent = busy ? "● Streaming / Running bl_ops…" : "Ready";
    pill.classList.toggle("busy", !!busy);
  }
  const sendBtn = el("btn-agent-send");
  if (sendBtn) sendBtn.textContent = busy ? "⚡ Steer" : "Send ↵";
  actions.syncTopbarSessionPill?.();
}

export function appendFeedLine(text, cls = "feed-agent") {
  const div = document.createElement("div");
  div.className = `feed-msg ${cls}`;
  div.textContent = text;
  const feed = el("agent-feed");
  if (feed) {
    feed.querySelector(".feed-empty-welcome")?.remove();
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
  }
  return div;
}

function bindSash(sashEl, isVertical, onMoveFn, onDblClickFn) {
  if (!sashEl) return;
  let dragging = false;
  sashEl.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    dragging = true;
    sashEl.setPointerCapture(e.pointerId);
    sashEl.classList.add("dragging");
    document.body.classList.add(isVertical ? "is-resizing-col" : "is-resizing-row");
  });
  sashEl.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    onMoveFn(e);
  });
  const stopDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    try { sashEl.releasePointerCapture(e.pointerId); } catch (_err) {}
    sashEl.classList.remove("dragging");
    document.body.classList.remove("is-resizing-col", "is-resizing-row");
    persistWorkbenchLayout();
  };
  sashEl.addEventListener("pointerup", stopDrag);
  sashEl.addEventListener("pointercancel", stopDrag);
  sashEl.addEventListener("dblclick", (e) => {
    e.preventDefault();
    onDblClickFn();
    persistWorkbenchLayout();
  });
}

function attachArrowHistory(inputEl, isTextarea) {
  if (!inputEl) return;
  let cursor = -1;
  let draft = "";
  inputEl.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const hist = state.promptHistory;
    if (!hist.length) return;
    if (e.key === "ArrowUp") {
      if (isTextarea && cursor === -1 && inputEl.value.slice(0, inputEl.selectionStart || 0).includes("\n")) return;
      e.preventDefault();
      if (cursor === -1) { draft = inputEl.value; cursor = hist.length - 1; }
      else if (cursor > 0) cursor -= 1;
      inputEl.value = hist[cursor];
    } else if (e.key === "ArrowDown") {
      if (cursor === -1) return;
      if (isTextarea && inputEl.value.slice(inputEl.selectionEnd || 0).includes("\n")) return;
      e.preventDefault();
      if (cursor < hist.length - 1) { cursor += 1; inputEl.value = hist[cursor]; }
      else { cursor = -1; inputEl.value = draft; }
    } else if (e.key.length === 1 || e.key === "Backspace") {
      cursor = -1;
    }
  });
}

export function initWorkbenchChrome() {
  loadSavedWorkbenchLayout();

  el("btn-toggle-left")?.addEventListener("click", () => setLeftHidden(!state.hideLeft));
  el("btn-close-left")?.addEventListener("click", () => setLeftHidden(true));
  el("edge-restore-left")?.addEventListener("click", () => setLeftHidden(false));

  el("btn-toggle-right")?.addEventListener("click", () => setRightHidden(!state.hideRight));
  el("btn-close-right")?.addEventListener("click", () => setRightHidden(true));
  el("edge-restore-right")?.addEventListener("click", () => setRightHidden(false));

  el("btn-expand-right")?.addEventListener("click", () => {
    state.hideRight = false;
    state.rightPaneW = state.rightPaneW >= 510 ? 390 : Math.min(620, Math.floor(window.innerWidth * 0.45));
    applyWorkbenchLayout();
    persistWorkbenchLayout();
  });

  el("btn-zen-mode")?.addEventListener("click", toggleZenMode);

  el("gen-studio-tabs")?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-genstudio]");
    if (b?.dataset.genstudio) openGenerativeStudioInCenter(b.dataset.genstudio);
  });

  bindSash(
    el("sash-left"),
    true,
    (e) => {
      const wbRect = el("studio-workbench").getBoundingClientRect();
      const rawW = e.clientX - wbRect.left;
      if (rawW < 105) {
        state.hideLeft = true;
        actions.hideHoverTooltip?.();
      } else {
        state.hideLeft = false;
        state.leftPaneW = Math.max(185, Math.min(Math.floor(wbRect.width * 0.55), rawW));
      }
      applyWorkbenchLayout();
    },
    () => {
      state.hideLeft = false;
      state.leftPaneW = 300;
      applyWorkbenchLayout();
    }
  );

  bindSash(
    el("sash-right"),
    true,
    (e) => {
      const wbRect = el("studio-workbench").getBoundingClientRect();
      const rawW = wbRect.right - e.clientX;
      if (rawW < 130) {
        state.hideRight = true;
      } else {
        state.hideRight = false;
        state.rightPaneW = Math.max(265, Math.min(Math.floor(wbRect.width * 0.58), rawW));
      }
      applyWorkbenchLayout();
    },
    () => {
      state.hideRight = false;
      state.rightPaneW = 390;
      applyWorkbenchLayout();
    }
  );

  if (typeof ResizeObserver !== "undefined" && el("pane-inspector")) {
    const inspEl = el("pane-inspector");
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect?.width || 390;
      inspEl.classList.toggle("wide-inspector", w >= 520);
      inspEl.classList.toggle("narrow-inspector", w > 0 && w < 310);
    });
    ro.observe(inspEl);
  }

  document.querySelectorAll("summary button, summary select, summary input").forEach((ctrl) => {
    ctrl.addEventListener("click", (e) => e.stopPropagation());
    ctrl.addEventListener("keydown", (e) => {
      if (e.key === " " || e.key === "Enter") e.stopPropagation();
    });
  });

  el("btn-toggle-all-sections")?.addEventListener("click", () => {
    const activePanel = document.querySelector("#pane-inspector > .composer-scroll:not(.hidden)");
    if (!activePanel) return;
    const drawers = Array.from(activePanel.querySelectorAll("details.insp-more, details.insp-section"));
    if (!drawers.length) return;
    const anyClosed = drawers.some((d) => !d.open);
    drawers.forEach((d) => { d.open = anyClosed; });
    el("btn-toggle-all-sections").textContent = anyClosed ? "⊟" : "⊞";
  });

  window.addEventListener("keydown", (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (!mod) return;
    if (e.code === "KeyB" && !e.shiftKey) {
      e.preventDefault();
      if (e.altKey) setRightHidden(!state.hideRight);
      else setLeftHidden(!state.hideLeft);
    } else if ((e.code === "KeyJ" && !e.shiftKey && !e.altKey) || e.code === "Backquote") {
      e.preventDefault();
      actions.toggleSessionInCenter?.();
    } else if (e.code === "KeyZ" && !e.altKey && state.centerKind !== "3d") {
      const tag = (document.activeElement?.tagName || "").toLowerCase();
      if (tag !== "input" && tag !== "textarea") {
        e.preventDefault();
        actions.openSessionInCenter?.(state.activeSessionId || "active", false);
      }
    }
  });

  el("sel-prompt-history")?.addEventListener("change", (e) => {
    if (e.target.value === "") return;
    const chosen = state.promptHistory[Number(e.target.value)];
    if (chosen) {
      const targetInp = state.centerKind === "session" ? el("agent-input") : el("input-prompt");
      if (targetInp) {
        targetInp.value = chosen;
        targetInp.focus();
      }
      actions.updatePricingAndInspector?.();
    }
    e.target.value = "";
  });

  attachArrowHistory(el("input-prompt"), true);
  attachArrowHistory(el("input-sr-prompt"), true);
  attachArrowHistory(el("agent-input"), true);

  // Center Session Composer Form (`Enter` submits, `Shift+Enter` newline)
  el("agent-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      el("agent-form")?.requestSubmit();
    }
  });

  el("agent-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const inp = el("agent-input");
    const txt = (inp?.value || "").trim();
    if (!txt) return;
    pushLocalPromptHistory(txt);
    inp.value = "";
    try {
      await apiFetch("/api/chat/submit", { method: "POST", json: { text: txt } });
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-agent-cancel")?.addEventListener("click", (e) => {
    e.stopPropagation();
    apiFetch("/api/chat/cancel", { method: "POST" }).catch(() => {});
  });

  // 6-Pill Interactive Status Bar Popover Delegation
  el("topbar-status-cluster")?.addEventListener("click", (e) => {
    const pill = e.target.closest(".sys-pill[data-popover]");
    if (!pill) return;
    e.stopPropagation();
    openStatusPopover(pill.dataset.popover, pill);
  });

  document.addEventListener("click", (e) => {
    const pop = el("status-popover");
    if (!pop || pop.classList.contains("hidden")) return;
    if (!e.target.closest("#status-popover") && !e.target.closest("#topbar-status-cluster")) {
      closeStatusPopover();
    }
  });

  el("btn-spop-open-session")?.addEventListener("click", () => {
    closeStatusPopover();
    actions.openSessionInCenter?.(state.activeSessionId || "active", true);
  });
  el("btn-spop-new-session")?.addEventListener("click", () => {
    closeStatusPopover();
    actions.startNewSession?.(false);
  });
  el("btn-spop-compact")?.addEventListener("click", () => {
    closeStatusPopover();
    el("btn-agent-compact")?.click();
  });

  const connectBlenderNow = async () => {
    if (el("val-blender")) el("val-blender").textContent = "connecting…";
    try {
      const res = await apiFetch("/api/blender/connect", { method: "POST", json: { mode: "auto" } });
      if (res.hint) showToast(res.hint, true);
      else showToast(`Connected Blender 5.3 [${res.blender?.mode || "auto"}]`);
      await actions.refreshState?.();
      await actions.refreshSessions?.();
    } catch (err) {
      showToast(err.message, true);
      actions.refreshState?.();
    }
  };
  el("btn-spop-bl-connect")?.addEventListener("click", connectBlenderNow);
  el("btn-spop-bl-3d")?.addEventListener("click", () => {
    closeStatusPopover();
    actions.open3DSceneInCenter?.("__blender_live__", false);
  });

  el("btn-spop-modal-run")?.addEventListener("click", () => {
    closeStatusPopover();
    const jid = state.activeSplatJob?.job_id || Array.from(state.splatJobs.keys())[0] || null;
    actions.openKernelTelemetryForJob?.(jid);
  });
  el("btn-spop-modal-launch")?.addEventListener("click", () => {
    closeStatusPopover();
    actions.openSplatLaunchDialog?.();
  });

  el("btn-spop-refresh-all")?.addEventListener("click", () => {
    actions.refreshState?.();
    actions.refreshSplats?.();
    actions.refreshSessions?.();
    showToast("Polled server & job status");
  });

  el("btn-bl-insp-still")?.addEventListener("click", () => actions.captureFromBlender?.("still"));
  el("btn-bl-insp-playblast")?.addEventListener("click", () => actions.captureFromBlender?.("playblast"));
  el("btn-bl-insp-stream")?.addEventListener("click", () => actions.open3DSceneInCenter?.("__blender_live__", false));

  el("btn-close-key")?.addEventListener("click", closeStatusPopover);
  el("btn-save-key")?.addEventListener("click", async () => {
    const k = el("input-or-key").value.trim();
    if (!k) return;
    try {
      await apiFetch("/api/keys/openrouter", { method: "POST", json: { key: k, save: true } });
      el("input-or-key").value = "";
      closeStatusPopover();
      showToast("Saved OPENROUTER_API_KEY");
      actions.refreshState?.();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-close-fal-key")?.addEventListener("click", closeStatusPopover);
  el("btn-save-fal-key")?.addEventListener("click", async () => {
    const k = (el("input-fal-key")?.value || "").trim();
    if (!k) return;
    try {
      await apiFetch("/api/keys/fal", { method: "POST", json: { key: k, save: true } });
      el("input-fal-key").value = "";
      closeStatusPopover();
      showToast("Saved FAL_KEY");
      actions.refreshState?.();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  applyWorkbenchLayout();
  actions.switchInspectorTab = switchInspectorTab;
  actions.syncInspectorContext = syncInspectorContext;
  actions.openStatusPopover = openStatusPopover;
  actions.closeStatusPopover = closeStatusPopover;
  actions.openGenerativeStudioInCenter = openGenerativeStudioInCenter;
  actions.setAgentDrawerOpen = setAgentDrawerOpen;
  actions.setRightHidden = setRightHidden;
  actions.toggleZenMode = toggleZenMode;
}