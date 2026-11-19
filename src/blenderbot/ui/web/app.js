(() => {
  // --- Auth & Boot URL Handling ---
  const urlParams = new URLSearchParams(window.location.search);
  const initialSceneParam = urlParams.get("scene") || "";
  const bootInFps =
    window.location.pathname === "/fps" ||
    window.location.hash === "#fps" ||
    !!initialSceneParam;

  const token = urlParams.get("token") || sessionStorage.getItem("bb_token") || "";
  if (urlParams.get("token")) {
    sessionStorage.setItem("bb_token", token);
    urlParams.delete("token");
    const cleanQ = urlParams.toString();
    window.history.replaceState({}, "", window.location.pathname + (cleanQ ? `?${cleanQ}` : ""));
  }

  function withTokenUrl(rawUrl) {
    if (!rawUrl || !token) return rawUrl;
    const sep = rawUrl.includes("?") ? "&" : "?";
    return `${rawUrl}${sep}token=${encodeURIComponent(token)}`;
  }

  async function apiFetch(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    if (token) headers["Authorization"] = `Bearer ${token}`;
    if (opts.json !== undefined) {
      headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.json);
    }
    const resp = await fetch(path, Object.assign({}, opts, { headers }));
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }

  const el = (id) => document.getElementById(id);

  const MODE_SUFFIXES = {
    standard: "",
    orbit_plate:
      " Smooth steady camera arc with clear multi-view parallax, sharp 35mm deep-focus optics, consistent scene lighting and exposure, zero motion blur, no jump cuts.",
    subject_4d:
      " Coherent continuous subject motion, crisp silhouette separation, steady camera framing, consistent lighting, no abrupt cuts.",
  };

  const SEEDREAM_MODE_SUFFIXES = {
    standard: "",
    concept_plate:
      " Crisp 35mm deep-focus production concept plate, clean architectural/spatial geometry, coherent physical lighting, zero lens distortion, suitable for 3D projection & Seedance conditioning.",
    pbr_texture:
      " Seamless orthographic flat-lit surface texture, even diffuse illumination, zero cast shadows, zero perspective vignetting, high micro-surface detail.",
    cutout_prop:
      " Isolated clean hero 3D prop asset, crisp silhouette edges, studio three-point rim & key lighting, centered framing with clean alpha separation.",
  };

  const SEEDREAM_RES_COST = { "1K": 0.02, "1.5K": 0.03, "2K": 0.04 };

  const DURATION_CHOICES = [
    { val: "", label: "Auto", sec: null },
    { val: "4", label: "4s", sec: 4 },
    { val: "5", label: "5s", sec: 5 },
    { val: "6", label: "6s", sec: 6 },
    { val: "8", label: "8s", sec: 8 },
    { val: "10", label: "10s", sec: 10 },
    { val: "12", label: "12s", sec: 12 },
    { val: "15", label: "15s", sec: 15 },
    { val: "20", label: "20s", sec: 20 },
    { val: "30", label: "30s", sec: 30 },
  ];

  const SIZE_CHOICES = [
    { val: "", label: "Auto (1280x720)" },
    { val: "1280x720", label: "1280x720 (16:9)" },
    { val: "1920x1080", label: "1920x1080 (1080p)" },
    { val: "854x480", label: "854x480 (480p)" },
    { val: "2520x1080", label: "2520x1080 (21:9)" },
    { val: "1080x1080", label: "1080x1080 (1:1)" },
    { val: "1080x1920", label: "1080x1920 (9:16)" },
    { val: "720x1280", label: "720x1280 (9:16)" },
    { val: "1440x1080", label: "1440x1080 (4:3)" },
    { val: "1080x1440", label: "1080x1440 (3:4)" },
  ];

  const BUILTIN_3D_STAGES = [
    {
      id: "__courtyard__",
      name: "Default Cube Courtyard",
      sub: "Solid 2m Cube + Stairs + Editable 3DGS",
      glyph: "🏛",
      desc: "Metric 2m×2m×2m Default Cube on Z=0 checkerboard with walkable stairs, parallax pillars, and editable 3DGS splats + test floaters.",
    },
    {
      id: "__default_cube_solid__",
      name: "Default Cube (Solid 2m)",
      sub: "2m × 2m × 2m Mesh on Z=0",
      glyph: "🧊",
      desc: "Classic Blender 2m×2m×2m Default Cube resting on Z=0 metric grid.",
    },
    {
      id: "__default_cube_splat__",
      name: "Default Cube Splat",
      sub: "Pure 3DGS + Test Floaters",
      glyph: "✨",
      desc: "Pure anisotropic 3D Gaussian Splat cube + seeded air floaters for testing 60fps brush culling.",
    },
    {
      id: "__blender_live__",
      name: "Live Blender 5.3 Scene",
      sub: "Stream Viewport Meshes & Cam",
      glyph: "🟠",
      desc: "Streams live evaluated mesh triangles & active camera directly from your connected Blender 5.3 session.",
    },
    {
      id: "__blank_grid__",
      name: "Blank Z=0 Voxel Stage",
      sub: "Empty Metric Grid",
      glyph: "📐",
      desc: "Clean Z=0 checkerboard stage for building Voxel blockouts and FPS Camera Rails.",
    },
  ];

  // --- Unified Studio State ---
  const state = {
    mediaItems: [],
    imageModels: [{ id: "bytedance-seed/seedream-5-5", name: "ByteDance Seed: Seedream 5.5", cost_per_image_usd: 0.02 }],
    videoModels: [{ id: "bytedance/seedance-2.5", name: "ByteDance: Seedance 2.5", cost_per_sec_usd: 0.1028 }],
    filter: "all",
    search: "",
    collapsedGroups: new Set(),

    // Center Stage State
    centerKind: "3d", // "3d" | "media" | "compare"
    active3DId: "__courtyard__",
    activeSplatJob: null,
    stageItem: null,

    // Right Inspector Tab: "seedance" | "seedream" | "fps" | "copilot"
    inspectorTab:
      bootInFps || window.location.hash === "#splat"
        ? "fps"
        : window.location.hash === "#seedream"
        ? "seedream"
        : "seedance",

    // Seedance & Seedream Conditioning
    mode: "standard",
    seedreamMode: "standard",
    seedreamRefs: [],
    costPerSec: 0.1028,
    sourceVideo: null,
    sourceVideoProbe: null,
    sourceVideo2: null,
    sourceVideo2Probe: null,
    dualVideoMode: "bridge_v1_to_v2",
    refVideos: [],
    firstFrame: null,
    lastFrame: null,
    refImages: [],
    refAudio: null,

    // Jobs, Splats & Co-Pilot
    splatSourceMode: "video",
    splatJobs: new Map(),
    jobs: new Map(),
    exportedShots: [],
    promptHistory: JSON.parse(localStorage.getItem("bb_seedance_prompts") || "[]"),
    agentPricing: { input_per_1m_usd: null, output_per_1m_usd: null },
    ledger: null,
    sessionCost: 0.0,
    liveAgentNode: null,

    // Resizable Workbench Layout & Hidden-by-Default Agent Chat Drawer
    leftPaneW: 300,
    rightPaneW: 410,
    hideLeft: false,
    hideRight: false,
    agentOpen: false,
    agentHeight: 260,
    agentMaximized: false,
    agentUnread: 0,
    agentBusy: false,
    allTreeBranchKeys: [],
  };

  try {
    const savedLayout = JSON.parse(localStorage.getItem("bb_studio_layout_v2") || "{}");
    if (typeof savedLayout.leftPaneW === "number") state.leftPaneW = Math.max(185, Math.min(720, savedLayout.leftPaneW));
    if (typeof savedLayout.rightPaneW === "number") state.rightPaneW = Math.max(265, Math.min(820, savedLayout.rightPaneW));
    if (typeof savedLayout.hideLeft === "boolean") state.hideLeft = savedLayout.hideLeft;
    if (typeof savedLayout.hideRight === "boolean") state.hideRight = savedLayout.hideRight;
    if (typeof savedLayout.agentHeight === "number") state.agentHeight = Math.max(135, Math.min(680, savedLayout.agentHeight));
  } catch (_e) {}

  function persistWorkbenchLayout() {
    try {
      localStorage.setItem(
        "bb_studio_layout_v2",
        JSON.stringify({
          leftPaneW: state.leftPaneW,
          rightPaneW: state.rightPaneW,
          hideLeft: state.hideLeft,
          hideRight: state.hideRight,
          agentHeight: state.agentHeight,
        })
      );
    } catch (_e) {}
  }

  function showToast(msg, isErr = false) {
    const t = el("composer-toast");
    if (!t) return;
    t.textContent = msg;
    t.classList.toggle("err", !!isErr);
    t.classList.remove("hidden");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => t.classList.add("hidden"), 4800);
  }

  function formatBytes(b) {
    if (!b) return "0 B";
    if (b < 1024) return `${b} B`;
    if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
    return `${(b / (1024 * 1024)).toFixed(1)} MB`;
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function hexToRgb01(hex) {
    const clean = String(hex || "#ff7b29").replace("#", "");
    const n = parseInt(clean, 16) || 0xff7b29;
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }

  // --- VS Code / Chrome DevTools Workbench Layout, Splitters & Agent Chat Drawer ---
  function applyWorkbenchLayout() {
    const root = document.documentElement;
    root.style.setProperty("--left-pane-w", `${Math.round(state.leftPaneW)}px`);
    root.style.setProperty("--right-pane-w", `${Math.round(state.rightPaneW)}px`);
    root.style.setProperty("--agent-drawer-h", `${Math.round(state.agentHeight)}px`);

    document.body.classList.toggle("hide-left", !!state.hideLeft);
    document.body.classList.toggle("hide-right", !!state.hideRight);

    el("btn-toggle-left")?.classList.toggle("active", !state.hideLeft);
    el("btn-toggle-right")?.classList.toggle("active", !state.hideRight);
    el("edge-restore-left")?.classList.toggle("hidden", !state.hideLeft);
    el("edge-restore-right")?.classList.toggle("hidden", !state.hideRight);

    const drawer = el("agent-drawer");
    if (drawer) {
      drawer.classList.toggle("collapsed", !state.agentOpen);
      drawer.classList.toggle("expanded", !!state.agentOpen);
      drawer.classList.toggle("maximized", !!(state.agentOpen && state.agentMaximized));
    }
    el("btn-topbar-agent")?.classList.toggle("active", !!state.agentOpen);
    const chev = el("agent-drawer-chevron");
    if (chev) chev.textContent = state.agentOpen ? "▾" : "▴";
    const tglBtn = el("btn-agent-toggle");
    if (tglBtn) tglBtn.textContent = state.agentOpen ? "▾ Hide" : "▴ Expand";
    const maxBtn = el("btn-agent-maximize");
    if (maxBtn) maxBtn.textContent = state.agentMaximized ? "⤡" : "⤢";

    if (state.agentOpen) {
      state.agentUnread = 0;
      el("agent-unread-badge")?.classList.add("hidden");
    } else if (state.agentUnread > 0) {
      const ub = el("agent-unread-badge");
      if (ub) {
        ub.textContent = String(state.agentUnread);
        ub.classList.remove("hidden");
      }
    }
  }

  function setLeftHidden(hide) {
    state.hideLeft = !!hide;
    if (state.hideLeft) hideHoverTooltip();
    applyWorkbenchLayout();
    persistWorkbenchLayout();
  }

  function setRightHidden(hide) {
    state.hideRight = !!hide;
    applyWorkbenchLayout();
    persistWorkbenchLayout();
  }

  function setAgentDrawerOpen(open, focusInput = false) {
    state.agentOpen = !!open;
    if (!state.agentOpen) state.agentMaximized = false;
    applyWorkbenchLayout();
    if (state.agentOpen) {
      const feed = el("agent-feed");
      if (feed) feed.scrollTop = feed.scrollHeight;
      if (focusInput) setTimeout(() => el("agent-input")?.focus(), 20);
    }
  }

  function toggleAgentDrawer(focusInput = true) {
    setAgentDrawerOpen(!state.agentOpen, focusInput && !state.agentOpen);
  }

  el("btn-toggle-left").addEventListener("click", () => setLeftHidden(!state.hideLeft));
  el("btn-close-left")?.addEventListener("click", () => setLeftHidden(true));
  el("edge-restore-left")?.addEventListener("click", () => setLeftHidden(false));

  el("btn-toggle-right").addEventListener("click", () => setRightHidden(!state.hideRight));
  el("btn-close-right")?.addEventListener("click", () => setRightHidden(true));
  el("edge-restore-right")?.addEventListener("click", () => setRightHidden(false));

  el("btn-expand-right")?.addEventListener("click", () => {
    state.hideRight = false;
    state.rightPaneW = state.rightPaneW >= 520 ? 410 : Math.min(620, Math.floor(window.innerWidth * 0.46));
    applyWorkbenchLayout();
    persistWorkbenchLayout();
  });

  function toggleZenMode() {
    const bothHidden = state.hideLeft && state.hideRight;
    state.hideLeft = !bothHidden;
    state.hideRight = !bothHidden;
    hideHoverTooltip();
    applyWorkbenchLayout();
    persistWorkbenchLayout();
  }
  el("btn-zen-mode").addEventListener("click", toggleZenMode);

  // Agent Chat Drawer Controls
  el("btn-topbar-agent")?.addEventListener("click", () => toggleAgentDrawer(true));
  el("btn-open-agent-from-jobs")?.addEventListener("click", () => setAgentDrawerOpen(true, true));
  el("btn-agent-toggle")?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleAgentDrawer(true);
  });
  el("btn-agent-maximize")?.addEventListener("click", (e) => {
    e.stopPropagation();
    state.agentOpen = true;
    state.agentMaximized = !state.agentMaximized;
    applyWorkbenchLayout();
  });
  el("btn-agent-clear")?.addEventListener("click", (e) => {
    e.stopPropagation();
    const feed = el("agent-feed");
    if (feed) feed.innerHTML = `<div class="feed-msg feed-op">Cleared local chat view.</div>`;
  });
  el("agent-drawer-header")?.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    toggleAgentDrawer(true);
  });

  // Drag-to-Resize Sash Splitters (Left, Right & Bottom Agent Drawer)
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

  bindSash(
    el("sash-left"),
    true,
    (e) => {
      const wbRect = el("studio-workbench").getBoundingClientRect();
      const rawW = e.clientX - wbRect.left;
      if (rawW < 105) {
        state.hideLeft = true;
        hideHoverTooltip();
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
      state.rightPaneW = 410;
      applyWorkbenchLayout();
    }
  );

  bindSash(
    el("sash-agent"),
    false,
    (e) => {
      const centerRect = el("pane-center-stage").getBoundingClientRect();
      const rawH = centerRect.bottom - e.clientY;
      if (rawH < 88) {
        state.agentOpen = false;
        state.agentMaximized = false;
      } else {
        state.agentOpen = true;
        state.agentMaximized = false;
        state.agentHeight = Math.max(135, Math.min(Math.floor(centerRect.height * 0.82), rawH));
      }
      applyWorkbenchLayout();
    },
    () => {
      state.agentOpen = true;
      state.agentMaximized = false;
      state.agentHeight = 260;
      applyWorkbenchLayout();
    }
  );

  // Responsive Inspector Breakpoint Observer
  if (typeof ResizeObserver !== "undefined" && el("pane-inspector")) {
    const inspEl = el("pane-inspector");
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect?.width || 410;
      inspEl.classList.toggle("wide-inspector", w >= 530);
      inspEl.classList.toggle("narrow-inspector", w > 0 && w < 315);
    });
    ro.observe(inspEl);
  }

  // Prevent interactive controls inside `<summary>` from toggling parent `<details>`
  document.querySelectorAll("summary button, summary select, summary input").forEach((ctrl) => {
    ctrl.addEventListener("click", (e) => e.stopPropagation());
    ctrl.addEventListener("keydown", (e) => {
      if (e.key === " " || e.key === "Enter") e.stopPropagation();
    });
  });

  // Expand / Collapse Additional Options in Active Inspector Tab
  el("btn-toggle-all-sections")?.addEventListener("click", () => {
    const panelMap = {
      seedance: "panel-seedance-composer",
      seedream: "panel-seedream-composer",
      fps: "panel-fps-splat",
      copilot: "panel-copilot-jobs",
    };
    const activePanel = el(panelMap[state.inspectorTab] || "panel-seedance-composer");
    if (!activePanel) return;
    const drawers = Array.from(activePanel.querySelectorAll("details.insp-more, details.insp-section"));
    if (!drawers.length) return;
    const anyMoreClosed = drawers.some((d) => !d.open);
    drawers.forEach((d) => { d.open = anyMoreClosed; });
    el("btn-toggle-all-sections").textContent = anyMoreClosed ? "⊟" : "⊞";
  });

  // Global VS Code Workbench Shortcuts (`Ctrl+B` Explorer, `Ctrl+Alt+B` Inspector, `Ctrl+J` / ``Ctrl+` `` Agent Chat)
  window.addEventListener("keydown", (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (!mod) return;
    if (e.code === "KeyB" && !e.shiftKey) {
      e.preventDefault();
      if (e.altKey) setRightHidden(!state.hideRight);
      else setLeftHidden(!state.hideLeft);
    } else if ((e.code === "KeyJ" && !e.shiftKey && !e.altKey) || e.code === "Backquote") {
      e.preventDefault();
      toggleAgentDrawer(true);
    }
  });

  applyWorkbenchLayout();

  // --- Right Inspector Tabs ---
  function switchInspectorTab(tab) {
    state.inspectorTab = ["seedance", "seedream", "fps", "copilot"].includes(tab) ? tab : "seedance";
    el("tab-mode-seedance").classList.toggle("active", state.inspectorTab === "seedance");
    el("tab-mode-seedream").classList.toggle("active", state.inspectorTab === "seedream");
    el("tab-mode-fps").classList.toggle("active", state.inspectorTab === "fps");
    el("tab-mode-copilot").classList.toggle("active", state.inspectorTab === "copilot");

    el("panel-seedance-composer").classList.toggle("hidden", state.inspectorTab !== "seedance");
    el("panel-seedream-composer").classList.toggle("hidden", state.inspectorTab !== "seedream");
    el("panel-fps-splat").classList.toggle("hidden", state.inspectorTab !== "fps");
    el("panel-copilot-jobs").classList.toggle("hidden", state.inspectorTab !== "copilot");

    if (state.inspectorTab === "seedream") updateSeedreamInspector();
    if (state.inspectorTab === "seedance") updatePricingAndInspector();
    renderHierarchyTree();
  }

  el("tab-mode-seedance").addEventListener("click", () => switchInspectorTab("seedance"));
  el("tab-mode-seedream").addEventListener("click", () => switchInspectorTab("seedream"));
  el("tab-mode-fps").addEventListener("click", () => switchInspectorTab("fps"));
  el("tab-mode-copilot").addEventListener("click", () => switchInspectorTab("copilot"));

  // --- Prompt History ---
  function mergeHistoryStrings(serverList) {
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
    localStorage.setItem("bb_seedance_prompts", JSON.stringify(state.promptHistory));
    renderPromptHistoryDropdown();
  }

  function pushLocalPromptHistory(text) {
    const clean = String(text || "").trim();
    if (!clean) return;
    state.promptHistory = state.promptHistory.filter((x) => x !== clean);
    state.promptHistory.push(clean);
    if (state.promptHistory.length > 100) state.promptHistory.shift();
    localStorage.setItem("bb_seedance_prompts", JSON.stringify(state.promptHistory));
    renderPromptHistoryDropdown();
  }

  function renderPromptHistoryDropdown() {
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

  el("sel-prompt-history").addEventListener("change", (e) => {
    if (e.target.value === "") return;
    const chosen = state.promptHistory[Number(e.target.value)];
    if (chosen) {
      el("input-prompt").value = chosen;
      updatePricingAndInspector();
      el("input-prompt").focus();
    }
    e.target.value = "";
  });

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
  attachArrowHistory(el("input-prompt"), true);
  attachArrowHistory(el("input-sr-prompt"), true);
  attachArrowHistory(el("agent-input"), false);

  // =========================================================================
  // LEFT HIERARCHY BROWSER + FLOATING RICH HOVER TOOLTIP
  // =========================================================================
  const tooltipEl = el("hierarchy-tooltip");

  function showHoverTooltip(anchorEl, cfg) {
    if (!tooltipEl || document.body.classList.contains("hide-left")) return;
    const prevBox = el("ht-preview-box");
    prevBox.innerHTML = "";
    if (cfg.videoUrl) {
      const v = document.createElement("video");
      v.src = cfg.videoUrl;
      if (cfg.posterUrl) v.poster = cfg.posterUrl;
      v.muted = true;
      v.loop = true;
      v.autoplay = true;
      v.playsInline = true;
      prevBox.appendChild(v);
      v.play().catch(() => {});
    } else if (cfg.imageUrl) {
      const im = document.createElement("img");
      im.src = cfg.imageUrl;
      im.alt = "";
      prevBox.appendChild(im);
    } else {
      prevBox.innerHTML = `<div class="ht-preview-glyph">${escapeHtml(cfg.glyph || "🌐")}</div>`;
    }

    el("ht-title").textContent = cfg.title || "";
    el("ht-path").textContent = cfg.path || "";
    el("ht-specs").innerHTML = (cfg.specs || [])
      .map(([k, v]) => `<span class="k">${escapeHtml(k)}</span><span class="v">${escapeHtml(v)}</span>`)
      .join("");

    tooltipEl.classList.remove("hidden");
    const rect = anchorEl.getBoundingClientRect();
    const libRect = el("dropzone-library")?.getBoundingClientRect();
    const anchorRight = libRect ? Math.max(libRect.right, rect.right) : rect.right;
    const ttH = tooltipEl.offsetHeight || 280;
    const ttW = tooltipEl.offsetWidth || 316;
    const topY = Math.max(54, Math.min(window.innerHeight - ttH - 12, rect.top - 16));
    const leftX = Math.max(12, Math.min(window.innerWidth - ttW - 12, anchorRight + 8));
    tooltipEl.style.top = `${topY}px`;
    tooltipEl.style.left = `${leftX}px`;
  }

  function hideHoverTooltip() {
    if (!tooltipEl) return;
    tooltipEl.classList.add("hidden");
    const vid = el("ht-preview-box")?.querySelector("video");
    if (vid) vid.pause();
  }

  el("hierarchy-tree").addEventListener("scroll", hideHoverTooltip);

  function rolesForMedia(it) {
    const roles = [];
    if (state.sourceVideo && state.sourceVideo.path === it.path) roles.push({ label: "VID1", full: "VID 1", primary: true });
    if (state.sourceVideo2 && state.sourceVideo2.path === it.path) roles.push({ label: "VID2", full: "VID 2", primary: true });
    if (state.firstFrame && state.firstFrame.path === it.path) roles.push({ label: "1ST", full: "1ST FRAME", primary: false });
    if (state.lastFrame && state.lastFrame.path === it.path) roles.push({ label: "LAST", full: "LAST FRAME", primary: false });
    if (state.seedreamRefs.some((r) => r.path === it.path)) roles.push({ label: "SR", full: "SEEDREAM REF", primary: state.inspectorTab === "seedream" });
    if (state.refImages.some((r) => r.path === it.path)) roles.push({ label: "+REF", full: "+REF", primary: false });
    if (state.refVideos.some((v) => v.path === it.path)) roles.push({ label: "+VID", full: "+REF VID", primary: false });
    if (state.refAudio && state.refAudio.path === it.path) roles.push({ label: "AUD", full: "AUDIO", primary: false });
    return roles;
  }

  function makeIndentGuidesHtml(depth) {
    if (depth <= 0) return "";
    let out = `<span class="vsc-indents">`;
    for (let i = 0; i < depth; i++) out += `<span class="vsc-indent-guide"></span>`;
    out += `</span>`;
    return out;
  }

  function folderSemanticIcon(segName, isOpen) {
    const low = segName.toLowerCase();
    if (low === "splats") return "🌐";
    if (low === "seedance") return "🎬";
    if (low === "seedream") return "🎨";
    if (low === "renders") return "🖼";
    if (low === "shots") return "🎥";
    return isOpen ? "📂" : "📁";
  }

  function folderSortRank(segName) {
    const low = segName.toLowerCase();
    if (low === "assets") return 1;
    if (low === "splats") return 2;
    if (low === "seedance") return 3;
    if (low === "seedream") return 4;
    if (low === "renders") return 5;
    if (low === "shots") return 6;
    if (low === "inputs") return 7;
    return 10;
  }

  function buildVscSection(branchKey, title, count, forceExpand, renderChildrenFn) {
    state.allTreeBranchKeys.push(branchKey);
    const isCollapsed = !forceExpand && state.collapsedGroups.has(branchKey);
    const wrap = document.createElement("div");
    wrap.className = `vsc-branch${isCollapsed ? " collapsed" : ""}`;
    wrap.dataset.branchKey = branchKey;

    const hdr = document.createElement("div");
    hdr.className = "vsc-tree-row vsc-section-row";
    hdr.tabIndex = 0;
    hdr.setAttribute("role", "treeitem");
    hdr.setAttribute("aria-expanded", String(!isCollapsed));
    hdr.dataset.branchKey = branchKey;
    hdr.innerHTML = `
      <span class="vsc-twistie">${isCollapsed ? "▸" : "▾"}</span>
      <span class="vsc-label-wrap"><span class="tree-name">${escapeHtml(title)}</span></span>
      <span class="tree-count">${count}</span>
    `;
    hdr.addEventListener("click", () => {
      if (state.collapsedGroups.has(branchKey)) state.collapsedGroups.delete(branchKey);
      else state.collapsedGroups.add(branchKey);
      renderHierarchyTree();
    });

    const body = document.createElement("div");
    body.className = "vsc-branch-children";
    body.setAttribute("role", "group");
    if (!isCollapsed) renderChildrenFn(body);

    wrap.appendChild(hdr);
    wrap.appendChild(body);
    return wrap;
  }

  function renderHierarchyTree() {
    const tree = el("hierarchy-tree");
    if (!tree) return;
    hideHoverTooltip();
    tree.innerHTML = "";
    state.allTreeBranchKeys = [];
    const q = state.search.trim().toLowerCase();
    const forceExpand = q.length > 0;
    const flt = state.filter;

    // --- Section 1: 3D Stages & Live Blender ---
    if (flt === "all" || flt === "splat") {
      const stages = BUILTIN_3D_STAGES.filter(
        (s) => !q || s.name.toLowerCase().includes(q) || s.sub.toLowerCase().includes(q)
      );
      if (stages.length) {
        tree.appendChild(
          buildVscSection("sec:builtin_3d", "3D Stages & Live Blender", stages.length, forceExpand, (body) => {
            stages.forEach((stItem) => {
              const row = document.createElement("div");
              const isAct = state.centerKind === "3d" && state.active3DId === stItem.id;
              row.className = `vsc-tree-row tree-item${isAct ? " active-stage" : ""}`;
              row.tabIndex = 0;
              row.setAttribute("role", "treeitem");
              row.innerHTML = `
                ${makeIndentGuidesHtml(1)}
                <span class="vsc-twistie spacer">·</span>
                <span class="vsc-node-icon">${stItem.glyph}</span>
                <span class="vsc-label-wrap">
                  <span class="tree-name" title="${escapeHtml(stItem.name)}">${escapeHtml(stItem.name)}</span>
                </span>
                <span class="tree-quick-btns"></span>
              `;
              const qBtns = row.querySelector(".tree-quick-btns");
              const fpsBtn = document.createElement("button");
              fpsBtn.type = "button";
              fpsBtn.textContent = "🎮 FPS";
              fpsBtn.title = "Enter FPS Mode in this 3D Stage";
              fpsBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                open3DSceneInCenter(stItem.id, true);
              });
              qBtns.appendChild(fpsBtn);

              row.addEventListener("click", () => open3DSceneInCenter(stItem.id, false));
              row.addEventListener("mouseenter", () =>
                showHoverTooltip(row, {
                  glyph: stItem.glyph,
                  title: stItem.name,
                  path: "Built-in Interactive 3D / FPS Stage",
                  specs: [
                    ["Mode", "Orbit + Minecraft Creative Fly (F)"],
                    ["Summary", stItem.sub],
                    ["Info", stItem.desc],
                  ],
                })
              );
              row.addEventListener("mouseleave", hideHoverTooltip);
              body.appendChild(row);
            });
          })
        );
      }
    }

    // --- Section 2: Nested Workspace Folder & Asset Hierarchy (VS Code Explorer Tree) ---
    const rootNode = { key: "dir:", name: "", dirs: new Map(), leaves: [], count: 0 };
    const insertLeafAtFolder = (folderPath, leafEntry) => {
      const cleanParts = String(folderPath || "")
        .replace(/\\/g, "/")
        .split("/")
        .map((p) => p.trim())
        .filter((p) => p && p !== ".");
      let cur = rootNode;
      cur.count += 1;
      let accum = "";
      for (const seg of cleanParts) {
        accum = accum ? `${accum}/${seg}` : seg;
        if (!cur.dirs.has(seg)) {
          cur.dirs.set(seg, { key: `dir:${accum}`, name: seg, path: accum, dirs: new Map(), leaves: [], count: 0 });
        }
        cur = cur.dirs.get(seg);
        cur.count += 1;
      }
      cur.leaves.push(leafEntry);
    };

    // Add 3D Gaussian Splats under `assets/splats` (or their actual parent dir)
    if (flt === "all" || flt === "splat") {
      const splats = Array.from(state.splatJobs.values())
        .filter((j) => !q || j.job_id.toLowerCase().includes(q) || String(j.source_label || "").toLowerCase().includes(q))
        .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

      splats.forEach((j) => {
        let folder = "assets/splats";
        if (j.ply_rel && j.ply_rel.includes("/")) {
          const parts = j.ply_rel.replace(/\\/g, "/").split("/");
          // e.g. `assets/splats/job_01/splat.ply` -> parent folder is `assets/splats`
          if (parts.length >= 3) folder = parts.slice(0, -2).join("/");
          else folder = parts.slice(0, -1).join("/");
        }
        insertLeafAtFolder(folder, { leafType: "splat", job: j });
      });
    }

    // Add Workspace Media (Videos, Images, Audio) into their nested folder paths
    if (flt !== "splat") {
      state.mediaItems.forEach((it) => {
        if (flt !== "all" && it.kind !== flt) return;
        if (q && !it.path.toLowerCase().includes(q) && !it.name.toLowerCase().includes(q)) return;
        const folder = it.path.includes("/") ? it.path.split("/").slice(0, -1).join("/") : "";
        insertLeafAtFolder(folder, { leafType: "media", item: it });
      });
    }

    const renderLeafRow = (entry, depth) => {
      if (entry.leafType === "splat") {
        const j = entry.job;
        const row = document.createElement("div");
        const isAct =
          (state.centerKind === "3d" || state.centerKind === "compare") &&
          state.activeSplatJob &&
          state.activeSplatJob.job_id === j.job_id;
        row.className = `vsc-tree-row tree-item${isAct ? " active-stage" : ""}`;
        row.tabIndex = 0;
        row.setAttribute("role", "treeitem");

        const subLbl = j.psnr
          ? `${Number(j.psnr).toFixed(1)}dB`
          : j.n_gaussians
          ? `${(j.n_gaussians / 1000).toFixed(0)}k`
          : j.stage;
        const iconHtml = j.preview_url
          ? `<span class="vsc-node-icon has-thumb"><img src="${withTokenUrl(j.preview_url)}" loading="lazy" alt="" /></span>`
          : `<span class="vsc-node-icon">🌐</span>`;

        row.innerHTML = `
          ${makeIndentGuidesHtml(depth)}
          <span class="vsc-twistie spacer">·</span>
          ${iconHtml}
          <span class="vsc-label-wrap">
            <span class="tree-name" title="${escapeHtml(j.job_id)}">${escapeHtml(j.job_id)}</span>
          </span>
          <span class="tree-sub">${escapeHtml(subLbl)}</span>
          <span class="tree-quick-btns"></span>
        `;

        const qBtns = row.querySelector(".tree-quick-btns");
        const mkBtn = (lbl, title, fn) => {
          const b = document.createElement("button");
          b.type = "button";
          b.textContent = lbl;
          if (title) b.title = title;
          b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
          qBtns.appendChild(b);
        };
        mkBtn("🎮 FPS", "Open in FPS Splat Editor", () => open3DSceneInCenter(j.job_id, true));
        if (j.cameras_rel) mkBtn("🎬 Path", "Compare Clip & 3D Camera Path", () => openComparatorForJob(j));
        if (j.ply_rel) mkBtn("→ BL", "Import Splat into Blender 5.3", () => importSplatToBlender(j.job_id));

        row.addEventListener("click", () => open3DSceneInCenter(j.job_id, false));
        row.addEventListener("mouseenter", () => {
          const specs = [
            ["Type", "3D Gaussian Splat (.ply + .splat)"],
            ["Gaussians", Number(j.n_gaussians || 0).toLocaleString()],
          ];
          if (j.psnr) specs.push(["PSNR / SSIM", `${j.psnr} dB / ${j.ssim || "—"}`]);
          if (j.ply_size_bytes) specs.push([".PLY Size", formatBytes(j.ply_size_bytes)]);
          if (j.cameras_rel) specs.push(["Cameras", "Solved cameras.json"]);
          if (j.cost_usd) specs.push(["Cloud Cost", `$${Number(j.cost_usd).toFixed(4)}`]);
          showHoverTooltip(row, {
            imageUrl: j.preview_url ? withTokenUrl(j.preview_url) : null,
            glyph: "🌐",
            title: j.job_id,
            path: j.ply_rel || `assets/splats/${j.job_id}`,
            specs,
          });
        });
        row.addEventListener("mouseleave", hideHoverTooltip);
        return row;
      }

      const it = entry.item;
      const roles = rolesForMedia(it);
      const isPrimary = roles.some((r) => r.primary);
      const isAct = state.centerKind === "media" && state.stageItem && state.stageItem.path === it.path;

      const row = document.createElement("div");
      row.className =
        `vsc-tree-row tree-item` +
        (isAct ? " active-stage" : "") +
        (isPrimary ? " has-primary-role" : roles.length ? " has-role" : "");
      row.tabIndex = 0;
      row.setAttribute("role", "treeitem");

      const u = withTokenUrl(it.url);
      let iconHtml = "";
      if (it.kind === "image") {
        iconHtml = `<span class="vsc-node-icon has-thumb"><img src="${u}" loading="lazy" alt="" /></span>`;
      } else if (it.kind === "video") {
        const posterU = withTokenUrl(`/api/media/poster?path=${encodeURIComponent(it.path)}&v=${it.mtime || 0}`);
        iconHtml = `<span class="vsc-node-icon has-thumb"><img src="${posterU}" loading="lazy" alt="" /></span>`;
      } else {
        iconHtml = `<span class="vsc-node-icon">♫</span>`;
      }

      const rolePillsHtml = roles.length
        ? `<span class="tree-roles-row">${roles
            .map((r) => `<span class="role-pill ${r.primary ? "" : "orange"}">${escapeHtml(r.label)}</span>`)
            .join("")}</span>`
        : "";

      row.innerHTML = `
        ${makeIndentGuidesHtml(depth)}
        <span class="vsc-twistie spacer">·</span>
        ${iconHtml}
        <span class="vsc-label-wrap">
          <span class="tree-name" title="${escapeHtml(it.path)}">${escapeHtml(it.name)}</span>
          ${rolePillsHtml}
        </span>
        <span class="tree-sub">${escapeHtml(formatBytes(it.size_bytes))}</span>
        <span class="tree-quick-btns"></span>
      `;

      const qBtns = row.querySelector(".tree-quick-btns");
      const addQ = (lbl, cls, title, fn) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = lbl;
        if (cls) b.className = cls;
        if (title) b.title = title;
        b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
        qBtns.appendChild(b);
      };

      if (it.kind === "image") {
        const is1 = state.firstFrame && state.firstFrame.path === it.path;
        const isL = state.lastFrame && state.lastFrame.path === it.path;
        const isSr = state.seedreamRefs.some((r) => r.path === it.path);
        addQ(is1 ? "✓ 1st" : "1st", is1 ? "active-orange" : "", "Set as Seedance First Frame", () => setFirstFrame(it));
        addQ(isL ? "✓ Last" : "Last", isL ? "active-orange" : "", "Set as Seedance Last Frame", () => setLastFrame(it));
        addQ(isSr ? "✓ SR" : "🎨 SR", isSr ? "active-green" : "", "Toggle Seedream Reference", () => toggleSeedreamRef(it));
      } else if (it.kind === "video") {
        const isV1 = state.sourceVideo && state.sourceVideo.path === it.path;
        const isV2 = state.sourceVideo2 && state.sourceVideo2.path === it.path;
        addQ(isV1 ? "✓ Vid1" : "Vid1", isV1 ? "active-green" : "", "Set as Seedance Video 1", () => setSourceVideo(isV1 ? null : it));
        addQ(isV2 ? "✓ Vid2" : "Vid2", isV2 ? "active-green" : "", "Set as Seedance Video 2", () => setSourceVideo2(isV2 ? null : it));
        addQ("🌐 Splat", "", "Train Cloud 3DGS Splat from Video", () => jumpToSplatWithVideo(it.path));
      } else if (it.kind === "audio") {
        const isAud = state.refAudio && state.refAudio.path === it.path;
        addQ(isAud ? "✓ Audio" : "Audio", isAud ? "active-orange" : "", "Set as Seedance Audio Reference", () => setRefAudio(it));
      }

      row.addEventListener("click", () => openMediaInCenter(it));
      row.addEventListener("mouseenter", () => {
        const specs = [["Kind", it.is_seedance ? "Seedance Video" : it.is_seedream ? "Seedream Image" : it.kind.toUpperCase()]];
        if (it.width && it.height) specs.push(["Dimensions", `${it.width} × ${it.height} px`]);
        specs.push(["File Size", formatBytes(it.size_bytes)]);
        if (it.cost_usd) specs.push(["Billed Cost", `$${Number(it.cost_usd).toFixed(4)} USD`]);
        if (roles.length) specs.push(["Assigned To", roles.map((r) => r.full).join(", ")]);
        showHoverTooltip(row, {
          imageUrl: it.kind === "image" ? u : null,
          videoUrl: it.kind === "video" ? u : null,
          posterUrl: it.kind === "video" ? withTokenUrl(`/api/media/poster?path=${encodeURIComponent(it.path)}`) : null,
          glyph: "♫",
          title: it.name,
          path: it.path,
          specs,
        });
      });
      row.addEventListener("mouseleave", hideHoverTooltip);
      return row;
    };

    const renderDirBranch = (dirNode, depth, parentEl) => {
      state.allTreeBranchKeys.push(dirNode.key);
      const isCollapsed = !forceExpand && state.collapsedGroups.has(dirNode.key);
      const branch = document.createElement("div");
      branch.className = `vsc-branch${isCollapsed ? " collapsed" : ""}`;
      branch.dataset.branchKey = dirNode.key;

      const row = document.createElement("div");
      row.className = "vsc-tree-row vsc-folder-row";
      row.tabIndex = 0;
      row.setAttribute("role", "treeitem");
      row.setAttribute("aria-expanded", String(!isCollapsed));
      row.dataset.branchKey = dirNode.key;
      row.innerHTML = `
        ${makeIndentGuidesHtml(depth)}
        <span class="vsc-twistie">${isCollapsed ? "▸" : "▾"}</span>
        <span class="vsc-node-icon">${folderSemanticIcon(dirNode.name, !isCollapsed)}</span>
        <span class="vsc-label-wrap">
          <span class="tree-name" title="${escapeHtml(dirNode.path)}">${escapeHtml(dirNode.name)}</span>
        </span>
        <span class="tree-count">${dirNode.count}</span>
      `;
      row.addEventListener("click", () => {
        if (state.collapsedGroups.has(dirNode.key)) state.collapsedGroups.delete(dirNode.key);
        else state.collapsedGroups.add(dirNode.key);
        renderHierarchyTree();
      });

      const childrenWrap = document.createElement("div");
      childrenWrap.className = "vsc-branch-children";
      childrenWrap.setAttribute("role", "group");

      if (!isCollapsed) {
        const subDirs = Array.from(dirNode.dirs.values()).sort(
          (a, b) => folderSortRank(a.name) - folderSortRank(b.name) || a.name.localeCompare(b.name)
        );
        subDirs.forEach((sd) => renderDirBranch(sd, depth + 1, childrenWrap));
        dirNode.leaves.forEach((lf) => childrenWrap.appendChild(renderLeafRow(lf, depth + 1)));
      }

      branch.appendChild(row);
      branch.appendChild(childrenWrap);
      parentEl.appendChild(branch);
    };

    if (rootNode.count > 0) {
      tree.appendChild(
        buildVscSection("sec:workspace", "Workspace Files & Splats", rootNode.count, forceExpand, (body) => {
          const topDirs = Array.from(rootNode.dirs.values()).sort(
            (a, b) => folderSortRank(a.name) - folderSortRank(b.name) || a.name.localeCompare(b.name)
          );
          topDirs.forEach((td) => renderDirBranch(td, 1, body));
          rootNode.leaves.forEach((lf) => body.appendChild(renderLeafRow(lf, 1)));
        })
      );
    }

    if (!tree.children.length) {
      tree.innerHTML = `<div class="empty-placeholder" style="padding:20px">No matching items. Drop files or clear search filter.</div>`;
    }
  }

  // Collapse / Expand All Folders in Explorer Tree
  el("btn-collapse-all-tree")?.addEventListener("click", () => {
    const dirKeys = state.allTreeBranchKeys.filter((k) => k.startsWith("dir:"));
    const targets = dirKeys.length ? dirKeys : state.allTreeBranchKeys;
    const anyExpanded = targets.some((k) => !state.collapsedGroups.has(k));
    if (anyExpanded) {
      targets.forEach((k) => state.collapsedGroups.add(k));
      el("btn-collapse-all-tree").textContent = "⊞";
    } else {
      state.collapsedGroups.clear();
      el("btn-collapse-all-tree").textContent = "⊟";
    }
    renderHierarchyTree();
  });

  // VS Code Keyboard Tree Navigation (`ArrowUp`, `ArrowDown`, `ArrowLeft`, `ArrowRight`, `Enter`)
  el("hierarchy-tree").addEventListener("keydown", (e) => {
    const curRow = e.target.closest(".vsc-tree-row");
    if (!curRow) return;
    const rows = Array.from(el("hierarchy-tree").querySelectorAll(".vsc-tree-row"));
    const idx = rows.indexOf(curRow);
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (idx >= 0 && idx < rows.length - 1) rows[idx + 1].focus();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (idx > 0) rows[idx - 1].focus();
    } else if (e.key === "Enter") {
      if (e.target === curRow) {
        e.preventDefault();
        curRow.click();
      }
    } else if (e.key === "ArrowLeft" && curRow.dataset.branchKey) {
      if (!state.collapsedGroups.has(curRow.dataset.branchKey)) {
        e.preventDefault();
        state.collapsedGroups.add(curRow.dataset.branchKey);
        renderHierarchyTree();
      }
    } else if (e.key === "ArrowRight" && curRow.dataset.branchKey) {
      if (state.collapsedGroups.has(curRow.dataset.branchKey)) {
        e.preventDefault();
        state.collapsedGroups.delete(curRow.dataset.branchKey);
        renderHierarchyTree();
      }
    }
  });

  el("media-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    document.querySelectorAll("#media-tabs .tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    renderHierarchyTree();
  });
  el("media-search").addEventListener("input", (e) => {
    state.search = e.target.value;
    renderHierarchyTree();
  });
  el("btn-refresh-media").addEventListener("click", () => {
    refreshMedia(false);
    refreshSplats();
  });

  // =========================================================================
  // CENTER STAGE CONTROLLER (MEDIA + 3D / FPS + COMPARATOR + ANCHOR CLICKS)
  // =========================================================================
  function syncViewModeButtons() {
    const isFpsActive = state.centerKind === "3d" && window._bbEngine && window._bbEngine.isFps();
    const isCmp = state.centerKind === "compare";
    el("btn-view-inspect").classList.toggle("active", !isFpsActive && !isCmp);
    el("btn-view-fps").classList.toggle("active", !!isFpsActive);
    el("btn-view-compare").classList.toggle("active", isCmp);
    el("btn-view-compare").classList.toggle("hidden", !(state.activeSplatJob && state.activeSplatJob.cameras_rel));

    // Highlight First/Last Frame cards if currently displayed in Center Stage!
    const viewingFirst = state.centerKind === "media" && state.stageItem && state.firstFrame && state.stageItem.path === state.firstFrame.path;
    const viewingLast = state.centerKind === "media" && state.stageItem && state.lastFrame && state.stageItem.path === state.lastFrame.path;
    el("slot-first")?.classList.toggle("viewing-center", !!viewingFirst);
    el("slot-last")?.classList.toggle("viewing-center", !!viewingLast);
  }

  function hideAllCenterViews() {
    el("theater-empty").classList.add("hidden");
    const vid = el("theater-video");
    const img = el("theater-image");
    const audWrap = el("theater-audio-wrap");
    const aud = el("theater-audio");
    vid.pause();
    aud.pause();
    vid.classList.add("hidden");
    img.classList.add("hidden");
    audWrap.classList.add("hidden");
    el("theater-splat-canvas").classList.add("hidden");
    el("fps-top-strip").classList.add("hidden");
    el("fps-crosshair-wrap").classList.add("hidden");
    el("fps-radar-canvas").classList.add("hidden");
    el("theater-splat-hud").classList.add("hidden");
    el("theater-compare-wrap").classList.add("hidden");
    if (window._bbStopComparator) window._bbStopComparator();
    if (window._bbEngine && document.pointerLockElement) document.exitPointerLock();
  }

  function openMediaInCenter(item) {
    if (!item) return;
    hideAllCenterViews();
    state.centerKind = "media";
    state.stageItem = item;

    el("stage-kind-badge").textContent = item.kind.toUpperCase();
    el("stage-title").textContent = item.name || item.path || "";
    const dims = item.width && item.height ? `${item.width}×${item.height} · ` : "";
    const cost = item.cost_usd ? ` · $${Number(item.cost_usd).toFixed(4)}` : "";
    el("theater-meta").textContent = `${dims}${formatBytes(item.size_bytes)}${cost}`;

    const u = withTokenUrl(item.url);
    if (item.kind === "video") {
      const vid = el("theater-video");
      vid.poster = withTokenUrl(`/api/media/poster?path=${encodeURIComponent(item.path || "")}`);
      vid.src = u;
      vid.classList.remove("hidden");
      vid.load();
      vid.play().catch(() => {});
    } else if (item.kind === "image") {
      const img = el("theater-image");
      img.src = u;
      img.classList.remove("hidden");
    } else if (item.kind === "audio") {
      el("theater-audio-title").textContent = item.name || item.path;
      const aud = el("theater-audio");
      aud.src = u;
      el("theater-audio-wrap").classList.remove("hidden");
      aud.load();
    }

    el("stage-3d-btns").classList.add("hidden");
    el("stage-media-btns").classList.remove("hidden");
    el("btn-stage-use-first").classList.toggle("hidden", item.kind !== "image");
    el("btn-stage-use-last").classList.toggle("hidden", item.kind !== "image");
    el("btn-stage-use-sr").classList.toggle("hidden", item.kind !== "image");
    el("btn-stage-use-vid").classList.toggle("hidden", item.kind !== "video");
    el("btn-stage-ref-vid").classList.toggle("hidden", item.kind !== "video");
    el("btn-stage-to-splat").classList.toggle("hidden", item.kind !== "video");
    const dl = el("link-stage-download");
    dl.href = `${u}&download=1`;
    dl.download = item.name || "asset";

    syncViewModeButtons();
    renderHierarchyTree();
  }

  async function open3DSceneInCenter(sceneId, enterFps = false) {
    hideAllCenterViews();
    state.centerKind = "3d";
    state.stageItem = null;
    state.active3DId = sceneId || "__courtyard__";

    const splatJob = state.splatJobs.get(state.active3DId) || null;
    state.activeSplatJob = splatJob;

    el("theater-splat-canvas").classList.remove("hidden");
    el("fps-top-strip").classList.remove("hidden");
    el("theater-splat-hud").classList.remove("hidden");
    el("stage-media-btns").classList.add("hidden");
    el("stage-3d-btns").classList.remove("hidden");

    const builtIn = BUILTIN_3D_STAGES.find((s) => s.id === state.active3DId);
    el("stage-kind-badge").textContent = splatJob ? "3DGS SPLAT" : "3D STAGE";
    el("stage-title").textContent = builtIn ? builtIn.name : state.active3DId;
    el("theater-meta").textContent = splatJob
      ? `${Number(splatJob.n_gaussians || 0).toLocaleString()} pts${splatJob.psnr ? ` · ${splatJob.psnr} dB` : ""}`
      : builtIn
      ? builtIn.sub
      : "";

    if (window._bbEngine) {
      await window._bbEngine.loadScene(state.active3DId);
      window._bbEngine.setFpsMode(!!enterFps, false);
    }
    if (enterFps && state.inspectorTab !== "fps") {
      switchInspectorTab("fps");
    }
    syncViewModeButtons();
    renderHierarchyTree();
  }

  // Top Header Mode Switcher Buttons
  el("btn-view-inspect").addEventListener("click", () => {
    if (state.centerKind === "3d" && window._bbEngine) {
      window._bbEngine.setFpsMode(false, false);
      syncViewModeButtons();
    } else if (state.centerKind === "compare" && state.activeSplatJob) {
      open3DSceneInCenter(state.activeSplatJob.job_id, false);
    }
  });

  el("btn-view-fps").addEventListener("click", () => {
    const target3D = state.activeSplatJob ? state.activeSplatJob.job_id : state.active3DId || "__courtyard__";
    open3DSceneInCenter(target3D, true);
  });

  el("btn-view-compare").addEventListener("click", () => {
    if (state.activeSplatJob) openComparatorForJob(state.activeSplatJob);
  });

  // Capture Current Center 3D View into Seedance 1st / Last Frame or Seedream Reference!
  async function captureCenter3DIntoRole(role) {
    if (!window._bbEngine) return;
    try {
      const dataUrl = window._bbEngine.captureCleanPng();
      const res = await apiFetch("/api/workspace/save_snapshot", {
        method: "POST",
        json: { data_url: dataUrl, prefix: `view_${role}` },
      });
      const item = res.item;
      await refreshMedia(false);
      if (role === "first") {
        state.firstFrame = item;
        renderConditioningSlots();
        updatePricingAndInspector();
        switchInspectorTab("seedance");
        showToast(`📸 Captured 3D View → Seedance 1st Frame (${item.name})`);
      } else if (role === "last") {
        state.lastFrame = item;
        renderConditioningSlots();
        updatePricingAndInspector();
        switchInspectorTab("seedance");
        showToast(`📸 Captured 3D View → Seedance Last Frame (${item.name})`);
      } else if (role === "seedream") {
        if (!state.seedreamRefs.some((r) => r.path === item.path)) {
          state.seedreamRefs.unshift(item);
          state.seedreamRefs = state.seedreamRefs.slice(0, 10);
        }
        renderSeedreamRefs();
        updateSeedreamInspector();
        switchInspectorTab("seedream");
        showToast(`🎨 Captured 3D View → Seedream Reference (${item.name})`);
      }
      renderHierarchyTree();
    } catch (err) {
      showToast(`3D capture failed: ${err.message}`, true);
    }
  }

  el("btn-3d-to-first").addEventListener("click", () => captureCenter3DIntoRole("first"));
  el("btn-3d-to-last").addEventListener("click", () => captureCenter3DIntoRole("last"));
  el("btn-3d-to-seedream").addEventListener("click", () => captureCenter3DIntoRole("seedream"));
  el("btn-grab3d-first").addEventListener("click", (e) => {
    e.stopPropagation();
    if (state.centerKind !== "3d") open3DSceneInCenter(state.active3DId, false).then(() => captureCenter3DIntoRole("first"));
    else captureCenter3DIntoRole("first");
  });
  el("btn-grab3d-last").addEventListener("click", (e) => {
    e.stopPropagation();
    if (state.centerKind !== "3d") open3DSceneInCenter(state.active3DId, false).then(() => captureCenter3DIntoRole("last"));
    else captureCenter3DIntoRole("last");
  });
  el("btn-sr-grab3d").addEventListener("click", () => {
    if (state.centerKind !== "3d") open3DSceneInCenter(state.active3DId, false).then(() => captureCenter3DIntoRole("seedream"));
    else captureCenter3DIntoRole("seedream");
  });

  // Center Media Action Buttons
  el("btn-stage-use-first").addEventListener("click", () => state.stageItem && setFirstFrame(state.stageItem));
  el("btn-stage-use-last").addEventListener("click", () => state.stageItem && setLastFrame(state.stageItem));
  el("btn-stage-use-sr").addEventListener("click", () => {
    if (state.stageItem) {
      toggleSeedreamRef(state.stageItem);
      switchInspectorTab("seedream");
    }
  });
  el("btn-stage-use-vid").addEventListener("click", () => state.stageItem && setSourceVideo(state.stageItem));
  el("btn-stage-ref-vid").addEventListener("click", () => state.stageItem && setSourceVideo2(state.stageItem));
  el("btn-stage-to-splat").addEventListener("click", () => state.stageItem && jumpToSplatWithVideo(state.stageItem.path));
  el("btn-stage-at-agent").addEventListener("click", () => {
    if (!state.stageItem) return;
    setAgentDrawerOpen(true, false);
    const inp = el("agent-input");
    inp.value = `${inp.value ? inp.value + " " : ""}@${state.stageItem.path} `;
    inp.focus();
  });

  el("btn-stage-save-splat").addEventListener("click", () => window._bbEngine && window._bbEngine.saveSplat(false));
  el("btn-stage-import-bl").addEventListener("click", () => {
    if (state.activeSplatJob) importSplatToBlender(state.activeSplatJob.job_id);
    else if (window._bbEngine) window._bbEngine.saveSplat(true);
  });

  // =========================================================================
  // SEEDANCE 2.5 & SEEDREAM 5.5 INSPECTORS + ANCHOR CLICK -> CENTER VIEW
  // =========================================================================
  // Clicking First Frame or Last Frame card in Right Inspector shows it in Center Stage!
  el("slot-first").addEventListener("click", () => {
    if (state.firstFrame) openMediaInCenter(state.firstFrame);
  });
  el("slot-last").addEventListener("click", () => {
    if (state.lastFrame) openMediaInCenter(state.lastFrame);
  });
  el("btn-clear-first").addEventListener("click", (e) => {
    e.stopPropagation();
    setFirstFrame(null);
  });
  el("btn-clear-last").addEventListener("click", (e) => {
    e.stopPropagation();
    setLastFrame(null);
  });
  el("btn-view-vid1").addEventListener("click", () => state.sourceVideo && openMediaInCenter(state.sourceVideo));
  el("btn-view-vid2").addEventListener("click", () => state.sourceVideo2 && openMediaInCenter(state.sourceVideo2));

  function activeDurationSec() {
    const durVal = el("sel-duration").value;
    if (durVal && Number(durVal) > 0) return Number(durVal);
    if (state.sourceVideoProbe && state.sourceVideoProbe.duration_sec) {
      return Math.max(4, Math.min(30, Math.round(state.sourceVideoProbe.duration_sec)));
    }
    return 5;
  }

  function activeRatePerSec() {
    const curModel = el("select-video-model").value;
    const found = state.videoModels.find((m) => m.id === curModel);
    return (found && found.cost_per_sec_usd) || state.costPerSec || 0.1028;
  }

  function populatePricingDropdowns() {
    const rate = activeRatePerSec();
    const durSel = el("sel-duration");
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

    const secNow = activeDurationSec();
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

  function updatePricingAndInspector() {
    populatePricingDropdowns();
    const sec = activeDurationSec();
    const rate = activeRatePerSec();
    const estUsd = sec * rate;

    el("est-cost-label").textContent = `~$${estUsd.toFixed(2)} USD (${sec}s)`;
    el("inspector-cost-pill").textContent = `~$${estUsd.toFixed(3)}`;

    const modelId = el("select-video-model").value || "bytedance/seedance-2.5";
    const sizeVal = el("sel-size").value || "auto";
    const rawPrompt = el("input-prompt").value.trim();
    const compiledPrompt = rawPrompt ? `${rawPrompt}${MODE_SUFFIXES[state.mode] || ""}` : "";

    let firstDesc = state.firstFrame ? state.firstFrame.path : state.sourceVideo ? `from ${state.sourceVideo.name}` : "none";
    let lastDesc = state.lastFrame ? state.lastFrame.path : state.sourceVideo ? `from ${state.sourceVideo.name}` : "none";

    el("inspector-body").innerHTML = `
      <div class="insp-row"><span class="insp-key">model:</span><span class="insp-val">${escapeHtml(modelId)} · ${sec}s · ${escapeHtml(sizeVal)}</span></div>
      <div class="insp-row"><span class="insp-key">first_frame:</span><span class="insp-val ${firstDesc !== "none" ? "ok" : ""}">${escapeHtml(firstDesc)}</span></div>
      <div class="insp-row"><span class="insp-key">last_frame:</span><span class="insp-val ${lastDesc !== "none" ? "ok" : ""}">${escapeHtml(lastDesc)}</span></div>
      <div class="insp-row"><span class="insp-key">prompt:</span><span class="insp-val">${escapeHtml(compiledPrompt ? compiledPrompt.slice(0, 120) : "(empty)")}</span></div>
    `;
  }

  ["sel-duration", "sel-size", "select-video-model", "chk-vid-first", "chk-vid-last", "sel-vid-mids", "chk-vid-audio", "chk-gen-audio"].forEach((id) => {
    el(id)?.addEventListener("change", updatePricingAndInspector);
  });
  el("input-prompt").addEventListener("input", updatePricingAndInspector);

  function populateSourceVideoDropdown() {
    const videos = state.mediaItems.filter((it) => it.kind === "video");
    const sel1 = el("sel-source-video");
    const sel2 = el("sel-source-video-2");
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

  el("sel-source-video").addEventListener("change", (e) => {
    const it = e.target.value ? state.mediaItems.find((x) => x.path === e.target.value) || null : null;
    setSourceVideo(it);
    if (it) openMediaInCenter(it);
  });
  el("sel-source-video-2").addEventListener("change", (e) => {
    const it = e.target.value ? state.mediaItems.find((x) => x.path === e.target.value) || null : null;
    setSourceVideo2(it);
    if (it) openMediaInCenter(it);
  });
  el("sel-dual-mode").addEventListener("change", () => {
    state.dualVideoMode = el("sel-dual-mode").value;
    renderConditioningSlots();
    updatePricingAndInspector();
  });
  el("btn-swap-videos").addEventListener("click", () => {
    const tv = state.sourceVideo, tp = state.sourceVideoProbe;
    state.sourceVideo = state.sourceVideo2; state.sourceVideoProbe = state.sourceVideo2Probe;
    state.sourceVideo2 = tv; state.sourceVideo2Probe = tp;
    populateSourceVideoDropdown();
    renderConditioningSlots();
    renderHierarchyTree();
    updatePricingAndInspector();
  });

  function setSourceVideo(item) {
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
    renderHierarchyTree();
    updatePricingAndInspector();
  }

  function setSourceVideo2(item) {
    state.sourceVideo2 = item || null;
    state.sourceVideo2Probe = null;
    if (item && state.sourceVideo && state.sourceVideo.path === item.path) state.sourceVideo = null;
    populateSourceVideoDropdown();
    renderConditioningSlots();
    renderHierarchyTree();
    updatePricingAndInspector();
  }

  function setFirstFrame(item) {
    state.firstFrame = item && state.firstFrame && state.firstFrame.path === item.path ? null : item;
    renderConditioningSlots();
    renderHierarchyTree();
    updatePricingAndInspector();
    syncViewModeButtons();
  }

  function setLastFrame(item) {
    state.lastFrame = item && state.lastFrame && state.lastFrame.path === item.path ? null : item;
    renderConditioningSlots();
    renderHierarchyTree();
    updatePricingAndInspector();
    syncViewModeButtons();
  }

  function setRefAudio(item) {
    state.refAudio = item && state.refAudio && state.refAudio.path === item.path ? null : item;
    renderConditioningSlots();
    renderHierarchyTree();
    updatePricingAndInspector();
  }

  function renderConditioningSlots() {
    const hasAnyVid = !!(state.sourceVideo || state.sourceVideo2);
    const hasBoth = !!(state.sourceVideo && state.sourceVideo2);
    el("slot-video-card").classList.toggle("active", hasAnyVid);
    el("btn-clear-video").classList.toggle("hidden", !state.sourceVideo);
    el("btn-view-vid1").classList.toggle("hidden", !state.sourceVideo);
    el("btn-clear-video-2").classList.toggle("hidden", !state.sourceVideo2);
    el("btn-view-vid2").classList.toggle("hidden", !state.sourceVideo2);
    el("btn-swap-videos").classList.toggle("hidden", !hasBoth);
    el("dual-video-mode-bar").classList.toggle("hidden", !hasBoth);
    el("slot-video-opts").classList.toggle("hidden", !hasAnyVid);

    const vidBadge = el("badge-sd-video-state");
    if (vidBadge) vidBadge.textContent = hasBoth ? "Vid1 + Vid2" : hasAnyVid ? "1 clip" : "Optional";
    if (hasAnyVid && el("sec-sd-video")) el("sec-sd-video").open = true;

    el("btn-clear-first").classList.toggle("hidden", !state.firstFrame);
    el("slot-first").classList.toggle("filled", !!state.firstFrame);
    el("slot-first-body").innerHTML = state.firstFrame
      ? `<img src="${withTokenUrl(state.firstFrame.url)}" alt="first" /><div class="slot-caption">${escapeHtml(state.firstFrame.name)}</div>`
      : `<span class="empty-placeholder">Click <b>1st</b> on an image or <b>📸 3D</b></span>`;

    el("btn-clear-last").classList.toggle("hidden", !state.lastFrame);
    el("slot-last").classList.toggle("filled", !!state.lastFrame);
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
        c.addEventListener("click", () => openMediaInCenter(item));
        c.querySelector("button").addEventListener("click", (e) => { e.stopPropagation(); onRem(); });
        chips.appendChild(c);
      };
      state.refVideos.forEach((v, i) => addChip("🎞", v, () => { state.refVideos.splice(i, 1); renderConditioningSlots(); renderHierarchyTree(); }));
      state.refImages.forEach((im, i) => addChip("🖼", im, () => { state.refImages.splice(i, 1); renderConditioningSlots(); renderHierarchyTree(); }));
      if (state.refAudio) addChip("🔊", state.refAudio, () => setRefAudio(null));
    }
  }

  el("btn-clear-video").addEventListener("click", () => setSourceVideo(null));
  el("btn-clear-video-2").addEventListener("click", () => setSourceVideo2(null));
  el("btn-clear-cond").addEventListener("click", () => {
    state.sourceVideo = null; state.sourceVideo2 = null; state.refVideos = [];
    state.firstFrame = null; state.lastFrame = null; state.refImages = []; state.refAudio = null;
    populateSourceVideoDropdown();
    renderConditioningSlots();
    renderHierarchyTree();
    updatePricingAndInspector();
    syncViewModeButtons();
  });

  el("mode-pills").addEventListener("click", (e) => {
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

  async function submitSeedance() {
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
      if (err.message.includes("OPENROUTER_API_KEY")) el("key-drawer").classList.remove("hidden");
    } finally {
      btn.disabled = false;
    }
  }
  el("btn-generate").addEventListener("click", submitSeedance);
  el("input-prompt").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); submitSeedance(); }
  });

  // --- Seedream 5.5 Inspector ---
  function populateSeedreamModels() {
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

  function toggleSeedreamRef(item) {
    if (!item) return;
    const idx = state.seedreamRefs.findIndex((r) => r.path === item.path);
    if (idx >= 0) state.seedreamRefs.splice(idx, 1);
    else if (state.seedreamRefs.length < 10) state.seedreamRefs.push(item);
    else return showToast("Seedream 5.5 accepts up to 10 reference images", true);
    renderSeedreamRefs();
    renderHierarchyTree();
    updateSeedreamInspector();
  }

  function renderSeedreamRefs() {
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
      chip.addEventListener("click", () => openMediaInCenter(img));
      chip.querySelector("button").addEventListener("click", (e) => {
        e.stopPropagation();
        state.seedreamRefs.splice(idx, 1);
        renderSeedreamRefs();
        renderHierarchyTree();
        updateSeedreamInspector();
      });
      chips.appendChild(chip);
    });
  }

  function updateSeedreamInspector() {
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

  ["sel-sr-res", "sel-sr-aspect", "sel-sr-bg", "sel-sr-fmt", "select-image-model"].forEach((id) => {
    el(id)?.addEventListener("change", updateSeedreamInspector);
  });
  el("input-sr-prompt")?.addEventListener("input", updateSeedreamInspector);
  el("btn-sr-clear-refs")?.addEventListener("click", () => {
    state.seedreamRefs = [];
    renderSeedreamRefs();
    renderHierarchyTree();
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

  async function submitSeedream() {
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
      if (err.message.includes("OPENROUTER_API_KEY")) el("key-drawer").classList.remove("hidden");
    } finally {
      btn.disabled = false;
    }
  }
  el("btn-generate-seedream")?.addEventListener("click", submitSeedream);
  el("input-sr-prompt")?.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); submitSeedream(); }
  });

  // =========================================================================
  // UNIFIED UPLOADS, BLENDER CAPTURE, JOBS, CO-PILOT & SSE
  // =========================================================================
  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    for (const file of files) {
      const low = file.name.toLowerCase();
      if (low.endsWith(".ply") || low.endsWith(".splat")) {
        if (window._bbEngine) await window._bbEngine.importLocalSplatFile(file);
        continue;
      }
      try {
        showToast(`Uploading ${file.name} (${formatBytes(file.size)})…`);
        const buf = await file.arrayBuffer();
        const res = await fetch(`/api/workspace/upload?filename=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: Object.assign({ "X-Filename": file.name }, token ? { Authorization: `Bearer ${token}` } : {}),
          body: buf,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        await refreshMedia(false);
        openMediaInCenter(data.item);
        showToast(`Uploaded ${data.item.name}`);
      } catch (err) {
        showToast(`Upload failed: ${err.message}`, true);
      }
    }
  }

  el("btn-upload-trigger").addEventListener("click", () => el("file-upload-input").click());
  el("file-upload-input").addEventListener("change", (e) => {
    uploadFiles(e.target.files);
    e.target.value = "";
  });

  [el("dropzone-library"), el("theater-box")].forEach((zone) => {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      if (zone.id === "theater-box") el("stage-drop-overlay").classList.remove("hidden");
      else zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", (e) => {
      if (e.target === zone) {
        el("stage-drop-overlay").classList.add("hidden");
        zone.classList.remove("drag-over");
      }
    });
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      el("stage-drop-overlay").classList.add("hidden");
      zone.classList.remove("drag-over");
      if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
    });
  });

  async function captureFromBlender(kind) {
    const btn = kind === "playblast" ? el("btn-cap-playblast") : el("btn-cap-still");
    btn.disabled = true;
    try {
      showToast(`Rendering ${kind} from Blender 5.3…`);
      const res = await apiFetch("/api/blender/capture", { method: "POST", json: { kind } });
      await refreshMedia(false);
      openMediaInCenter(res.item);
      if (res.item.kind === "video") setSourceVideo(res.item);
      else if (!state.firstFrame) setFirstFrame(res.item);
      showToast(`Captured ${res.item.name}`);
    } catch (err) {
      showToast(`Blender capture failed: ${err.message}`, true);
    } finally {
      btn.disabled = false;
    }
  }
  el("btn-cap-still").addEventListener("click", () => captureFromBlender("still"));
  el("btn-cap-playblast").addEventListener("click", () => captureFromBlender("playblast"));

  function renderJobs() {
    const jobsList = el("jobs-list");
    const arr = Array.from(state.jobs.values()).sort((a, b) => b.created_at - a.created_at);
    const runningCnt = arr.filter((j) => !["completed", "failed", "cancelled"].includes(j.stage)).length;
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
      const active = !["completed", "failed", "cancelled"].includes(j.stage);
      const cls = j.stage === "completed" ? "completed" : j.stage === "failed" ? "failed" : active ? "running" : "";
      const card = document.createElement("div");
      card.className = `job-card ${cls}`;
      const costBadge = j.cost_usd ? `<span class="job-cost-pill">$${Number(j.cost_usd).toFixed(4)}</span>` : "";
      card.innerHTML = `
        <div class="job-top">
          <span>${escapeHtml(j.local_id)} · <code>${escapeHtml(j.stage)}</code></span>
          <span>${costBadge} ${active ? `<button type="button" class="btn btn-ghost btn-xs btn-cancel-job">Cancel</button>` : ""}</span>
        </div>
        <div class="job-prompt">${escapeHtml(j.error ? "⚠ " + j.error : j.prompt || j.output_rel)}</div>
      `;
      card.querySelector(".btn-cancel-job")?.addEventListener("click", (e) => {
        e.stopPropagation();
        apiFetch(`/api/seedance/cancel/${j.local_id}`, { method: "POST" }).catch(() => {});
      });
      card.addEventListener("click", () => {
        if (j.stage === "completed" && j.media_url) {
          const isImg = /\.(png|jpe?g|webp)$/i.test(j.output_rel || "") || String(j.local_id).startsWith("sr-");
          openMediaInCenter({
            path: j.output_rel,
            name: j.output_rel.split("/").pop(),
            kind: isImg ? "image" : "video",
            size_bytes: 0,
            width: j.width,
            height: j.height,
            cost_usd: j.cost_usd,
            url: j.media_url,
          });
        }
      });
      jobsList.appendChild(card);
    });
  }

  function renderAgentContextBar() {
    const lg = state.ledger;
    if (!lg) return;
    el("agent-ctx-summary").textContent = `Context: ${lg.total_tokens.toLocaleString()} / ${(lg.limit / 1000).toFixed(0)}k tok (${lg.pct}%)`;
    const drawer = el("agent-ctx-drawer");
    drawer.innerHTML = (lg.segments || [])
      .map(
        (s) => `<div class="ctx-seg-row"><span>#${s.id} [${escapeHtml(s.type)}] ${escapeHtml(s.label)}</span>${s.pinned ? "<span>📌</span>" : `<button type="button" data-drop-seg="${s.id}">✕</button>`}</div>`
      )
      .join("");
    drawer.querySelectorAll("button[data-drop-seg]").forEach((b) => {
      b.addEventListener("click", async (e) => {
        e.stopPropagation();
        await apiFetch(`/api/context/${b.dataset.dropSeg}`, { method: "DELETE" }).catch(() => {});
        refreshState();
      });
    });
  }
  el("agent-ctx-bar").addEventListener("click", () => el("agent-ctx-drawer").classList.toggle("hidden"));

  function setAgentBusyUi(busy, previewText = "") {
    state.agentBusy = !!busy;
    el("btn-agent-cancel")?.classList.toggle("hidden", !busy);
    el("topbar-agent-dot")?.classList.toggle("hidden", !busy && state.agentUnread === 0);
    const pill = el("agent-status-pill");
    if (pill) {
      pill.textContent = busy ? "● Thinking…" : "Ready";
      pill.classList.toggle("busy", !!busy);
    }
    if (previewText) {
      const prev = el("agent-drawer-preview");
      if (prev) prev.textContent = previewText.replace(/\s+/g, " ").trim().slice(0, 120);
    }
  }

  function appendFeedLine(text, cls = "feed-agent") {
    const div = document.createElement("div");
    div.className = `feed-msg ${cls}`;
    div.textContent = text;
    const feed = el("agent-feed");
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
    if (!state.agentOpen && text) {
      state.agentUnread += 1;
      applyWorkbenchLayout();
      el("topbar-agent-dot")?.classList.remove("hidden");
      const prev = el("agent-drawer-preview");
      if (prev) prev.textContent = text.replace(/\s+/g, " ").trim().slice(0, 120);
    }
    return div;
  }

  el("agent-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const inp = el("agent-input");
    const txt = inp.value.trim();
    if (!txt) return;
    pushLocalPromptHistory(txt);
    inp.value = "";
    setAgentDrawerOpen(true, true);
    try {
      await apiFetch("/api/chat/submit", { method: "POST", json: { text: txt } });
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("agent-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !el("agent-input").value.trim()) {
      e.preventDefault();
      el("agent-input").blur();
      setAgentDrawerOpen(false, false);
    }
  });

  el("btn-agent-cancel").addEventListener("click", (e) => {
    e.stopPropagation();
    apiFetch("/api/chat/cancel", { method: "POST" }).catch(() => {});
  });

  el("pill-blender").addEventListener("click", async () => {
    el("pill-blender").textContent = "Blender: connecting…";
    try {
      const res = await apiFetch("/api/blender/connect", { method: "POST", json: { mode: "auto" } });
      showToast(`Connected Blender 5.3 [${res.blender?.mode || "auto"}]`);
      refreshState();
    } catch (err) {
      showToast(err.message, true);
      refreshState();
    }
  });

  el("btn-or-key").addEventListener("click", () => {
    el("key-drawer").classList.toggle("hidden");
    if (!el("key-drawer").classList.contains("hidden")) el("input-or-key").focus();
  });
  el("btn-close-key").addEventListener("click", () => el("key-drawer").classList.add("hidden"));
  el("btn-save-key").addEventListener("click", async () => {
    const k = el("input-or-key").value.trim();
    if (!k) return;
    try {
      await apiFetch("/api/keys/openrouter", { method: "POST", json: { key: k, save: true } });
      el("input-or-key").value = "";
      el("key-drawer").classList.add("hidden");
      showToast("Saved OPENROUTER_API_KEY");
      refreshState();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  const announcedCompletions = new Set();
  function handleJobUpdate(data, fromPoll = false) {
    if (!data || !data.local_id) return;
    const prev = state.jobs.get(data.local_id);
    state.jobs.set(data.local_id, data);
    if (!fromPoll) renderJobs();

    if (data.stage === "completed" && data.media_url && !announcedCompletions.has(data.local_id)) {
      announcedCompletions.add(data.local_id);
      if (prev) {
        const isImg = /\.(png|jpe?g|webp)$/i.test(data.output_rel || "") || String(data.local_id).startsWith("sr-");
        refreshMedia(false);
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
        openMediaInCenter(thItem);
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

  function applyTopState(s) {
    if (s.seedance_spec?.cost_per_sec_usd) state.costPerSec = s.seedance_spec.cost_per_sec_usd;
    if (s.agent_pricing) state.agentPricing = s.agent_pricing;
    if (s.ledger) { state.ledger = s.ledger; renderAgentContextBar(); }
    if (Array.isArray(s.prompt_history)) mergeHistoryStrings(s.prompt_history);

    const pb = el("pill-blender");
    if (s.blender?.connected) {
      const scName = typeof s.blender.scene === "string" ? s.blender.scene : (s.blender.scene?.name || "Scene");
      pb.textContent = `Blender 5.3: ${s.blender.mode} (${scName})`;
      pb.className = "pill pill-btn pill-ok";
    } else {
      pb.textContent = "Blender 5.3: standby";
      pb.className = "pill pill-btn pill-muted";
    }

    const okBtn = el("btn-or-key");
    if (s.openrouter?.configured) {
      okBtn.textContent = `OpenRouter: ready`;
      okBtn.className = "pill pill-btn pill-ok";
    } else {
      okBtn.textContent = "OpenRouter: Set Key";
      okBtn.className = "pill pill-btn pill-warn";
    }

    if (s.health) {
      const hs = s.health.status || "healthy";
      el("pill-health").textContent = `Health: ${hs}`;
      el("pill-health").className = hs === "healthy" ? "pill pill-ok" : hs === "watch" ? "pill pill-warn" : "pill pill-err";
    }
    if (typeof s.stats?.cost === "number") {
      el("pill-cost").textContent = `$${s.stats.cost.toFixed(4)}`;
    }
    setAgentBusyUi(!!s.busy);

    if (Array.isArray(s.jobs)) { s.jobs.forEach((j) => handleJobUpdate(j, true)); renderJobs(); }
    if (Array.isArray(s.splat_jobs)) {
      s.splat_jobs.forEach((sj) => state.splatJobs.set(sj.job_id, sj));
      populateKnownCamsDropdown();
      renderHierarchyTree();
    }
    updatePricingAndInspector();
  }

  async function refreshState() {
    try { applyTopState(await apiFetch("/api/state")); } catch (_e) {}
  }

  async function refreshMedia() {
    try {
      const data = await apiFetch("/api/workspace/media");
      state.mediaItems = data.items || [];
      populateSourceVideoDropdown();
      populateSplatVideoDropdown();
      renderHierarchyTree();
    } catch (_e) {}
  }

  async function refreshSplats() {
    try {
      const res = await apiFetch("/api/splats/jobs");
      if (Array.isArray(res.jobs)) {
        res.jobs.forEach((j) => state.splatJobs.set(j.job_id, j));
        populateKnownCamsDropdown();
        renderHierarchyTree();
      }
    } catch (_e) {}
  }

  async function refreshModels() {
    apiFetch("/api/seedance/models").then((d) => {
      if (Array.isArray(d.models) && d.models.length) { state.videoModels = d.models; populatePricingDropdowns(); }
    }).catch(() => {});
    apiFetch("/api/seedream/models").then((d) => {
      if (Array.isArray(d.models) && d.models.length) { state.imageModels = d.models; populateSeedreamModels(); }
    }).catch(() => {});
  }

  async function refreshShotsList() {
    apiFetch("/api/shots/list").then((r) => {
      if (Array.isArray(r.shots)) { state.exportedShots = r.shots; populateKnownCamsDropdown(); }
    }).catch(() => {});
  }

  let lastEventSeq = 0;
  function connectSSE() {
    const es = new EventSource(withTokenUrl(`/api/events?since=${lastEventSeq}`));
    es.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); } catch (_e) { return; }
      if (typeof ev.seq === "number" && ev.seq > lastEventSeq) lastEventSeq = ev.seq;
      const { type, data } = ev;
      if (type === "seedance-job" && data) handleJobUpdate(data, false);
      else if (type === "splat-job" && data) handleSplatJobUpdate(data);
      else if (type === "splat-telemetry" && data && window._bbPushSplatTelemetry) window._bbPushSplatTelemetry(data);
      else if (type === "splat-render" && data && window._bbOnSplatRenderProgress) window._bbOnSplatRenderProgress(data);
      else if (type === "workspace-media-added") refreshMedia();
      else if (type === "transcript" && data) {
        if (data.type === "user") appendFeedLine(`❯ ${data.text}`, "feed-user");
        else if (data.message || data.text) appendFeedLine(`· ${data.message || data.text}`, data.type === "warning" ? "feed-warn" : "feed-op");
      } else if (type === "turn-start") {
        state.liveAgentNode = appendFeedLine("", "feed-agent");
        setAgentBusyUi(true, "Agent streaming response…");
      } else if (type === "text") {
        if (!state.liveAgentNode) state.liveAgentNode = appendFeedLine("", "feed-agent");
        state.liveAgentNode.textContent += String(data || "");
        el("agent-feed").scrollTop = el("agent-feed").scrollHeight;
        if (!state.agentOpen) {
          const prev = el("agent-drawer-preview");
          if (prev) prev.textContent = state.liveAgentNode.textContent.replace(/\s+/g, " ").trim().slice(-120);
        }
      } else if (type === "ops-parsed" && Array.isArray(data)) {
        appendFeedLine(`⚙ ops: ${data.map((o) => o.op).join(", ")}`, "feed-op");
      } else if (["turn-end", "done", "cancelled", "error"].includes(type)) {
        const finalTxt = state.liveAgentNode?.textContent || "";
        state.liveAgentNode = null;
        setAgentBusyUi(false, finalTxt ? finalTxt.slice(-120) : "Ask Blender 5.3 Agent or press Ctrl+J to expand…");
        refreshState();
        refreshMedia();
      } else if (type === "blender-changed") {
        refreshState();
      }
    };
    es.onerror = () => { es.close(); setTimeout(connectSSE, 3000); };
  }

  // =========================================================================
  // CLOUD 3DGS TRAINER, `tt-splat` TELEMETRY & TRAJECTORY COMPARATOR
  // =========================================================================
  function jumpToSplatWithVideo(videoPath) {
    if (state.hideRight) setRightHidden(false);
    switchInspectorTab("fps");
    const drawer = el("drawer-cloud-splat");
    if (drawer) drawer.open = true;
    setSplatSourceMode("video");
    populateSplatVideoDropdown();
    if (videoPath) el("sel-splat-video").value = videoPath;
  }

  function setSplatSourceMode(mode) {
    state.splatSourceMode = mode === "blender" ? "blender" : "video";
    const isBl = state.splatSourceMode === "blender";
    el("splat-source-pills").querySelectorAll(".mode-pill").forEach((b) => {
      b.classList.toggle("active", b.dataset.src === state.splatSourceMode);
    });
    el("splat-src-video-box").classList.toggle("hidden", isBl);
    el("splat-src-blender-box").classList.toggle("hidden", !isBl);
    updateSplatCostEstimate();
  }
  el("splat-source-pills").addEventListener("click", (e) => {
    const b = e.target.closest(".mode-pill");
    if (b?.dataset.src) setSplatSourceMode(b.dataset.src);
  });

  function populateSplatVideoDropdown() {
    const sel = el("sel-splat-video");
    if (!sel) return;
    const prev = sel.value;
    const videos = state.mediaItems.filter((it) => it.kind === "video");
    sel.innerHTML = `<option value="">— Select workspace video —</option>`;
    videos.forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.path;
      opt.textContent = `${it.name} (${formatBytes(it.size_bytes)})`;
      sel.appendChild(opt);
    });
    if (prev && videos.some((v) => v.path === prev)) sel.value = prev;
    else if (videos.length) sel.value = videos[0].path;
  }

  function populateKnownCamsDropdown() {
    const sel = el("sel-splat-known-cams");
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
  }

  function updateSplatCostEstimate() {
    const iters = Math.max(200, Number(el("inp-splat-iters").value || 5000));
    const gpu = el("sel-splat-gpu").value || "H100";
    const rateHr = gpu === "L40S" ? 1.95 : 3.95;
    const estSec = (state.splatSourceMode === "video" ? 8 : 2) + (iters / 2000) * (gpu === "L40S" ? 36 : 20);
    el("splat-est-cost").textContent = `~$${((estSec / 3600) * rateHr).toFixed(3)} (${gpu})`;
  }
  el("inp-splat-iters").addEventListener("input", updateSplatCostEstimate);
  el("sel-splat-gpu").addEventListener("change", updateSplatCostEstimate);

  el("btn-train-splat").addEventListener("click", async () => {
    const knownCamVal = el("sel-splat-known-cams")?.value || "";
    const payload = {
      source_mode: state.splatSourceMode,
      iterations: Math.max(200, Math.min(30000, Number(el("inp-splat-iters").value || 5000))),
      gpu: el("sel-splat-gpu").value || "H100",
      sh_degree: Number(el("sel-splat-sh").value || 1),
      render_mode: el("sel-splat-rendermode").value || "ellipsoids_viewport",
      auto_import: el("chk-splat-autoimport").checked,
      lambda_dssim: Number(el("inp-splat-dssim").value || 0.2),
      opa_reg: Number(el("inp-splat-opareg").value || 0.0005),
      prune_opa: Number(el("inp-splat-pruneopa").value || 0.025),
      opacity_reset_every: Number(el("inp-splat-opareset").value || 2000),
      conf_percentile: Number(el("inp-splat-vggtconf").value || 45),
      max_anisotropy: Math.max(4, Number(el("inp-splat-maxaniso").value || 10) * 0.6),
      rasterize_mode: "antialiased",
      refine_poses: el("chk-splat-refineposes")?.checked ?? true,
      pose_backend: el("sel-splat-pose-backend")?.value || "vggt",
    };
    if (state.splatSourceMode === "video") {
      payload.video_path = el("sel-splat-video").value;
      if (!payload.video_path) return showToast("Select a source video first.", true);
      if (knownCamVal) payload.known_cameras_job = knownCamVal;
    } else {
      payload.num_views = Number(el("inp-splat-views").value || 28);
      payload.targets = el("inp-splat-targets").value.trim();
      payload.transparent_bg = el("chk-splat-rgba").checked;
    }

    const btn = el("btn-train-splat");
    btn.disabled = true;
    try {
      const res = await apiFetch("/api/splats/train", { method: "POST", json: payload });
      if (res.job) handleSplatJobUpdate(res.job);
      showToast(`Launched Modal ${payload.gpu} 3DGS job: ${res.job?.job_id || ""}`);
    } catch (err) {
      showToast(err.message, true);
    } finally {
      btn.disabled = false;
    }
  });

  async function importSplatToBlender(jobId) {
    showToast(`Importing '${jobId}' into Blender 5.3…`);
    try {
      const res = await apiFetch("/api/splats/import", {
        method: "POST",
        json: {
          job_id: jobId,
          render_mode: el("sel-splat-rendermode")?.value || "ellipsoids_viewport",
          max_points: 350000,
          import_cameras: true,
        },
      });
      showToast(`Imported '${res.imported?.object}' into Blender 5.3!`);
      refreshState();
    } catch (err) {
      showToast(err.message, true);
    }
  }

  let splatShowDiff = false;
  el("btn-splat-show-render").addEventListener("click", () => {
    splatShowDiff = false;
    el("splat-live-img").classList.remove("hidden");
    el("splat-live-diff").classList.add("hidden");
  });
  el("btn-splat-show-diff").addEventListener("click", () => {
    splatShowDiff = true;
    el("splat-live-img").classList.add("hidden");
    el("splat-live-diff").classList.remove("hidden");
  });

  const announcedSplatDone = new Set();
  function handleSplatJobUpdate(j) {
    if (!j || !j.job_id) return;
    const prev = state.splatJobs.get(j.job_id);
    state.splatJobs.set(j.job_id, j);
    renderHierarchyTree();

    const active = !["completed", "failed", "cancelled"].includes(j.stage);
    if (active || (prev && prev.stage !== "completed" && j.stage === "completed")) {
      if (active && el("drawer-cloud-splat")) el("drawer-cloud-splat").open = true;
      el("splat-live-card").classList.remove("hidden");
      const pct = j.total_steps > 0 ? Math.min(100, Math.round((j.step / j.total_steps) * 100)) : 5;
      el("splat-prog-fill").style.width = `${j.stage === "completed" ? 100 : pct}%`;
      el("splat-live-title").textContent = `${j.job_id} (${j.step}/${j.total_steps})`;
      el("splat-live-metrics").textContent = `PSNR ${Number(j.psnr || 0).toFixed(1)}dB · ${Number(j.n_gaussians || 0).toLocaleString()} pts`;
      if (j.preview_url) {
        el("splat-live-preview-wrap").classList.remove("hidden");
        el("splat-live-img").src = withTokenUrl(j.preview_url);
        if (j.diff_url) el("splat-live-diff").src = withTokenUrl(j.diff_url);
      }
    }

    if (j.stage === "completed" && prev && prev.stage !== "completed" && !announcedSplatDone.has(j.job_id)) {
      announcedSplatDone.add(j.job_id);
      showToast(`3DGS Splat completed: ${j.job_id}`);
      open3DSceneInCenter(j.job_id, false);
    }
  }

  // `tt-splat` Live Controls
  const wbState = { jobId: null, paused: false, ticks: [], densify: [], surgeries: [], snapshots: [] };
  function activeWbJobId() {
    return wbState.jobId || state.activeSplatJob?.job_id || Array.from(state.splatJobs.keys())[0] || null;
  }

  el("btn-wb-pause")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    wbState.paused = !wbState.paused;
    el("btn-wb-pause").textContent = wbState.paused ? "▶ Resume" : "⏸ Pause";
    await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, { method: "POST", json: { paused: wbState.paused } }).catch(() => {});
  });
  el("btn-wb-saveckpt")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, { method: "POST", json: { save_checkpoint: true } }).catch(() => {});
    showToast("Requested cloud checkpoint");
  });
  el("btn-wb-stop")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch(`/api/splats/cancel/${encodeURIComponent(jid)}`, { method: "POST", json: {} }).catch(() => {});
  });
  el("btn-wb-push-knobs")?.addEventListener("click", async () => {
    const jid = activeWbJobId(); if (!jid) return;
    await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, {
      method: "POST",
      json: { lambda_dssim: Number(el("inp-splat-dssim").value || 0.2), opa_reg: Number(el("inp-splat-opareg").value || 0.0005) },
    }).catch(() => {});
    showToast("Pushed live training knobs");
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

  function drawWbCharts() {
    const ticks = wbState.ticks;
    const c1 = el("canvas-splat-loss-psnr"), c2 = el("canvas-splat-count");
    if (!ticks.length || !c1 || !c2) return;
    el("splat-charts-wrap")?.classList.remove("hidden");
    const g1 = c1.getContext("2d"), W = c1.width, H = c1.height;
    g1.clearRect(0, 0, W, H);
    const losses = ticks.map((t) => Number(t.loss || 0)), psnrs = ticks.map((t) => Number(t.psnr || 0));
    const maxL = Math.max(0.01, ...losses), minP = Math.min(...psnrs, 12), maxP = Math.max(minP + 2, ...psnrs);
    const plot = (ctx, arr, mn, mx, col) => {
      ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.beginPath();
      arr.forEach((v, i) => {
        const x = 10 + (ticks.length > 1 ? (i / (ticks.length - 1)) * (W - 20) : 0);
        const y = 8 + (1 - (v - mn) / Math.max(1e-6, mx - mn)) * (ctx.canvas.height - 16);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };
    plot(g1, losses, 0, maxL, "#fb923c");
    plot(g1, psnrs, minP, maxP, "#4ade80");
    const g2 = c2.getContext("2d"); g2.clearRect(0, 0, c2.width, c2.height);
    const cnts = ticks.map((t) => Number(t.n_gaussians || 0));
    plot(g2, cnts, Math.min(...cnts) * 0.9, Math.max(100, ...cnts) * 1.05, "#a78bfa");
  }

  window._bbPushSplatTelemetry = (ev) => {
    if (!wbState.jobId) wbState.jobId = ev.job_id;
    if (ev.job_id !== wbState.jobId) return;
    if (ev.event === "telemetry") { wbState.ticks.push(ev); drawWbCharts(); }
  };

  // --- Side-by-Side Synced Trajectory Comparator ---
  const cmpState = { active: false, job: null, traj: null, curFrameIdx: 0, orbitYaw: -0.75, orbitPitch: 0.48, dragging: false, lastX: 0, lastY: 0, rafId: 0 };
  window._bbStopComparator = () => {
    cmpState.active = false;
    if (cmpState.rafId) cancelAnimationFrame(cmpState.rafId);
    el("vid-cmp-left")?.pause();
    el("vid-cmp-right")?.pause();
  };

  async function openComparatorForJob(job) {
    hideAllCenterViews();
    state.centerKind = "compare";
    state.activeSplatJob = job;
    cmpState.job = job;
    cmpState.active = true;

    el("stage-kind-badge").textContent = "COMPARE & PATH";
    el("stage-title").textContent = job.job_id;
    el("theater-compare-wrap").classList.remove("hidden");
    syncViewModeButtons();
    renderHierarchyTree();

    try {
      const data = await apiFetch(`/api/splats/trajectory/${encodeURIComponent(job.job_id)}`);
      cmpState.traj = data.trajectory;
      cmpState.curFrameIdx = 0;

      const leftOpts = [];
      const rightOpts = [];
      if (data.source_video_url) {
        leftOpts.push({ label: `Original (${data.source_video_rel || "source"})`, url: data.source_video_url });
        rightOpts.push({ label: `Original (${data.source_video_rel || "source"})`, url: data.source_video_url });
      }
      if (data.modal_render_url) {
        rightOpts.unshift({ label: "Modal H100 3DGS Render", url: data.modal_render_url });
        leftOpts.push({ label: "Modal H100 3DGS Render", url: data.modal_render_url });
      }
      if (data.blender_render_url) {
        rightOpts.push({ label: "Blender 5.3 EEVEE Render", url: data.blender_render_url });
        leftOpts.push({ label: "Blender 5.3 EEVEE Render", url: data.blender_render_url });
      }

      const selL = el("sel-cmp-left"), selR = el("sel-cmp-right");
      selL.innerHTML = leftOpts.map((o, i) => `<option value="${i}">${escapeHtml(o.label)}</option>`).join("") || `<option>—</option>`;
      selR.innerHTML = rightOpts.map((o, i) => `<option value="${i}">${escapeHtml(o.label)}</option>`).join("") || `<option>—</option>`;
      selL._opts = leftOpts; selR._opts = rightOpts;

      const setCmpVid = (vEl, bEl, phEl, opt) => {
        if (!opt?.url) { vEl.pause(); vEl.removeAttribute("src"); phEl?.classList.remove("hidden"); bEl.textContent = "No render yet"; return; }
        phEl?.classList.add("hidden"); bEl.textContent = opt.label; vEl.src = withTokenUrl(opt.url); vEl.load();
      };
      setCmpVid(el("vid-cmp-left"), el("lbl-cmp-left"), null, leftOpts[0]);
      setCmpVid(el("vid-cmp-right"), el("lbl-cmp-right"), el("cmp-right-placeholder"), (data.modal_render_url || data.blender_render_url) ? rightOpts[0] : null);

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

  el("btn-cmp-play").addEventListener("click", () => {
    const vL = el("vid-cmp-left"), vR = el("vid-cmp-right");
    const m = vL.src ? vL : vR.src ? vR : null;
    if (!m) {
      cmpState._virt = !cmpState._virt;
      el("btn-cmp-play").textContent = cmpState._virt ? "⏸ Pause" : "▶ Play";
      return;
    }
    if (m.paused) {
      if (vL.src && vR.src) vR.currentTime = vL.currentTime;
      if (vL.src) vL.play().catch(() => {});
      if (vR.src) vR.play().catch(() => {});
      el("btn-cmp-play").textContent = "⏸ Pause";
    } else {
      vL.pause(); vR.pause(); el("btn-cmp-play").textContent = "▶ Play";
    }
  });
  el("btn-cmp-prevf").addEventListener("click", () => seekCmpFrame(cmpState.curFrameIdx - 1));
  el("btn-cmp-nextf").addEventListener("click", () => seekCmpFrame(cmpState.curFrameIdx + 1));
  el("rng-cmp-scrub").addEventListener("input", (e) => seekCmpFrame(Number(e.target.value || 0)));

  el("btn-cmp-farm-modal").addEventListener("click", async () => {
    if (!cmpState.job) return;
    const btn = el("btn-cmp-farm-modal"); btn.disabled = true;
    try {
      const res = await apiFetch("/api/splats/render", { method: "POST", json: { job_id: cmpState.job.job_id, mode: "modal" } });
      showToast(`Rendered ${res.frames} frames @ ${res.render_fps} fps!`);
      await refreshSplats();
      await openComparatorForJob(state.splatJobs.get(cmpState.job.job_id) || cmpState.job);
    } catch (e) { showToast(e.message, true); } finally { btn.disabled = false; }
  });
  el("btn-cmp-render-blender").addEventListener("click", async () => {
    if (!cmpState.job) return;
    const btn = el("btn-cmp-render-blender"); btn.disabled = true;
    try {
      await apiFetch("/api/splats/render", { method: "POST", json: { job_id: cmpState.job.job_id, mode: "blender" } });
      await refreshSplats();
      await openComparatorForJob(state.splatJobs.get(cmpState.job.job_id) || cmpState.job);
    } catch (e) { showToast(e.message, true); } finally { btn.disabled = false; }
  });

  el("canvas-cmp-path3d").addEventListener("mousedown", (e) => { cmpState.dragging = true; cmpState.lastX = e.clientX; cmpState.lastY = e.clientY; });
  window.addEventListener("mouseup", () => { cmpState.dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!cmpState.dragging || !cmpState.active) return;
    cmpState.orbitYaw -= (e.clientX - cmpState.lastX) * 0.01;
    cmpState.orbitPitch = Math.max(-1.35, Math.min(1.42, cmpState.orbitPitch + (e.clientY - cmpState.lastY) * 0.01));
    cmpState.lastX = e.clientX; cmpState.lastY = e.clientY;
    drawComparatorCanvases();
  });

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

  // =========================================================================
  // MULTI-PASS DEPTH-TESTED WEBGL2 3D / MINECRAFT FPS / SPLAT CULLING ENGINE
  // =========================================================================
  function createUnified3DEngine(canvas) {
    const gl = canvas.getContext("webgl2", {
      antialias: true,
      alpha: false,
      depth: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) return null;
    gl.getExtension("EXT_color_buffer_float");
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    const PRIME_LENSES = [14, 18, 24, 28, 35, 50, 85, 105, 135];
    const VOXEL_SIZES = [0.5, 1.0, 2.0];
    const BRUSH_STEPS = [0.05, 0.10, 0.18, 0.28, 0.45, 0.70, 1.05, 1.50, 2.20, 3.20, 4.50];
    const Z_NEAR = 0.08, Z_FAR = 600.0;
    const DEPTH_A = (Z_FAR + Z_NEAR) / (Z_FAR - Z_NEAR);
    const DEPTH_B = (-2.0 * Z_FAR * Z_NEAR) / (Z_FAR - Z_NEAR);
    const TEX_W = 2048;

    function compileProg(vs, fs) {
      const mk = (t, src) => {
        const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
      gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(p);
      return p;
    }

    const meshProg = compileProg(
      `#version 300 es
      precision highp float;
      layout(location = 0) in vec3 a_pos;
      layout(location = 1) in vec3 a_nrm;
      layout(location = 2) in vec4 a_col;
      uniform mat3 u_W;
      uniform vec3 u_camPos;
      uniform vec2 u_focal;
      uniform vec2 u_viewport;
      uniform vec2 u_depthAB;
      out vec3 v_nrm;
      out vec3 v_world;
      out vec4 v_col;
      out float v_camZ;
      void main() {
        vec3 cam = u_W * (a_pos - u_camPos);
        v_nrm = a_nrm; v_world = a_pos; v_col = a_col; v_camZ = cam.z;
        gl_Position = vec4(cam.x * u_focal.x / (u_viewport.x * 0.5), cam.y * u_focal.y / (u_viewport.y * 0.5), u_depthAB.x * cam.z + u_depthAB.y, cam.z);
      }`,
      `#version 300 es
      precision mediump float;
      in vec3 v_nrm; in vec3 v_world; in vec4 v_col; in float v_camZ;
      out vec4 fragColor;
      void main() {
        if (v_camZ < 0.08) discard;
        vec3 N = normalize(v_nrm);
        float sun = max(dot(N, normalize(vec3(-0.48, -0.62, 0.82))), 0.0);
        float fill = max(dot(N, normalize(vec3(0.55, 0.45, 0.35))), 0.0) * 0.28;
        float hemi = 0.38 + 0.22 * (N.z * 0.5 + 0.5);
        vec3 base = v_col.rgb;
        if (abs(N.z) > 0.95 && abs(v_world.z) < 0.015) {
          vec2 gf = abs(fract(v_world.xy + 0.5) - 0.5);
          float ln = 1.0 - smoothstep(0.0, 0.024, min(gf.x, gf.y));
          base = mix(base, base * 1.45 + vec3(0.04, 0.07, 0.12), ln * 0.55);
        }
        vec3 lit = base * (hemi + sun * 0.68 + fill);
        lit = mix(lit, vec3(0.05, 0.07, 0.11), clamp((v_camZ - 40.0) / 140.0, 0.0, 0.72));
        fragColor = vec4(lit, v_col.a);
      }`
    );

    const splatProg = compileProg(
      `#version 300 es
      precision highp float;
      precision highp int;
      layout(location = 0) in vec2 a_quad;
      layout(location = 1) in uint a_index;
      uniform highp sampler2D u_posColTex;
      uniform highp sampler2D u_covTexA;
      uniform highp sampler2D u_covTexB;
      uniform highp usampler2D u_flagTex;
      uniform mat3 u_W;
      uniform vec3 u_camPos;
      uniform vec2 u_focal;
      uniform vec2 u_viewport;
      uniform vec2 u_depthAB;
      uniform float u_scaleMult;
      uniform int u_visMode;
      out vec4 v_color;
      out vec2 v_uv;
      flat out int v_pointMode;
      vec3 turboRamp(float t) {
        t = clamp(t, 0.0, 1.0);
        return clamp(vec3(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)), 0.08, 1.0);
      }
      void main() {
        ivec2 tc = ivec2(int(a_index & 2047u), int(a_index >> 11u));
        uint flags = texelFetch(u_flagTex, tc, 0).r;
        if ((flags & 1u) != 0u) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
        vec4 posPack = texelFetch(u_posColTex, tc, 0);
        uint rgbaU = floatBitsToUint(posPack.w);
        vec4 col = vec4(float(rgbaU & 255u), float((rgbaU >> 8u) & 255u), float((rgbaU >> 16u) & 255u), float((rgbaU >> 24u) & 255u)) / 255.0;
        vec3 cam = u_W * (posPack.xyz - u_camPos);
        if (cam.z < 0.10 || col.a < 0.008) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
        vec4 cA = texelFetch(u_covTexA, tc, 0);
        vec4 cB = texelFetch(u_covTexB, tc, 0);
        float effScale = u_scaleMult * (u_visMode == 2 ? 0.35 : 1.0);
        float s2 = effScale * effScale;
        float invZ = 1.0 / cam.z;
        vec2 centerNdc = vec2(cam.x * u_focal.x * invZ, cam.y * u_focal.y * invZ) / (u_viewport * 0.5);
        float zNdc = u_depthAB.x + u_depthAB.y * invZ;
        v_pointMode = (u_visMode == 1) ? 1 : 0;
        vec2 offsetNdc;
        if (u_visMode == 1) {
          float pxR = clamp(2.8 * u_scaleMult, 1.5, 9.0);
          offsetNdc = a_quad * pxR / u_viewport;
          v_uv = a_quad * 0.75;
          col.a = max(col.a, 0.85);
        } else {
          mat3 Vrk = mat3(cA.x * s2, cA.y * s2, cA.z * s2, cA.y * s2, cB.x * s2, cB.y * s2, cA.z * s2, cB.y * s2, cB.z * s2);
          float invZ2 = invZ * invZ;
          mat3 J = mat3(u_focal.x * invZ, 0.0, 0.0, 0.0, u_focal.y * invZ, 0.0, -u_focal.x * cam.x * invZ2, -u_focal.y * cam.y * invZ2, 0.0);
          mat3 T = J * u_W;
          mat3 cov2d = T * Vrk * transpose(T);
          float d1 = cov2d[0][0] + 0.3, od = cov2d[0][1], d2 = cov2d[1][1] + 0.3;
          float mid = 0.5 * (d1 + d2), rad = length(vec2((d1 - d2) * 0.5, od));
          float l1 = mid + rad, l2 = max(mid - rad, 0.1);
          vec2 dv = normalize(vec2(od, l1 - d1 + 1e-7));
          vec2 v1 = min(sqrt(2.0 * l1), 1024.0) * dv;
          vec2 v2 = min(sqrt(2.0 * l2), 1024.0) * vec2(dv.y, -dv.x);
          offsetNdc = (a_quad.x * v1 + a_quad.y * v2) * 2.0 / u_viewport;
          v_uv = a_quad * sqrt(2.0);
        }
        if (u_visMode == 3) col.rgb = turboRamp((posPack.z + 0.3) / 3.2);
        else if (u_visMode == 4) {
          float alert = max(clamp((cA.w - 4.0) / 10.0, 0.0, 1.0), clamp((0.28 - col.a) / 0.25, 0.0, 1.0));
          col.rgb = mix(col.rgb * 0.38, vec3(1.0, 0.22, 0.72), alert);
          col.a = max(col.a, alert * 0.85);
        }
        if ((flags & 4u) != 0u) { col.rgb = mix(col.rgb, vec3(1.0, 0.15, 0.65), 0.82); col.a = max(col.a, 0.88); }
        else if ((flags & 2u) != 0u) { col.rgb = mix(col.rgb, vec3(0.18, 0.92, 1.0), 0.72); col.a = max(col.a, 0.85); }
        gl_Position = vec4(centerNdc + offsetNdc, zNdc, 1.0);
        v_color = col;
      }`,
      `#version 300 es
      precision mediump float;
      in vec4 v_color; in vec2 v_uv; flat in int v_pointMode;
      out vec4 fragColor;
      void main() {
        float r2 = dot(v_uv, v_uv);
        if (v_pointMode == 1) { if (r2 > 2.2) discard; fragColor = vec4(v_color.rgb * v_color.a, v_color.a); return; }
        if (r2 > 8.0) discard;
        float a = v_color.a * exp(-0.5 * r2);
        if (a < 0.004) discard;
        fragColor = vec4(v_color.rgb * a, a);
      }`
    );

    const lineProg = compileProg(
      `#version 300 es
      precision highp float;
      layout(location = 0) in vec3 a_pos;
      layout(location = 1) in vec4 a_col;
      uniform mat3 u_W;
      uniform vec3 u_camPos;
      uniform vec2 u_focal;
      uniform vec2 u_viewport;
      uniform vec2 u_depthAB;
      out vec4 v_col;
      void main() {
        vec3 cam = u_W * (a_pos - u_camPos);
        float z = max(0.085, cam.z);
        gl_Position = vec4(vec2(cam.x * u_focal.x / z, cam.y * u_focal.y / z) / (u_viewport * 0.5), u_depthAB.x + u_depthAB.y / z, 1.0);
        v_col = a_col;
      }`,
      `#version 300 es
      precision mediump float;
      in vec4 v_col; out vec4 fragColor;
      void main() { fragColor = vec4(v_col.rgb * v_col.a, v_col.a); }`
    );

    const meshVao = gl.createVertexArray();
    gl.bindVertexArray(meshVao);
    const meshBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, meshBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 40, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 40, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 40, 24);

    const splatVao = gl.createVertexArray();
    gl.bindVertexArray(splatVao);
    const quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const idxBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
    gl.enableVertexAttribArray(1); gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(1, 1);

    const lineVao = gl.createVertexArray();
    gl.bindVertexArray(lineVao);
    const lineBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);

    const mkTex = (unit) => {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    const posColTex = mkTex(0), covTexA = mkTex(1), covTexB = mkTex(2), flagTex = mkTex(3);
    gl.useProgram(splatProg);
    gl.uniform1i(gl.getUniformLocation(splatProg, "u_posColTex"), 0);
    gl.uniform1i(gl.getUniformLocation(splatProg, "u_covTexA"), 1);
    gl.uniform1i(gl.getUniformLocation(splatProg, "u_covTexB"), 2);
    gl.uniform1i(gl.getUniformLocation(splatProg, "u_flagTex"), 3);

    const uMesh = { W: gl.getUniformLocation(meshProg, "u_W"), camPos: gl.getUniformLocation(meshProg, "u_camPos"), focal: gl.getUniformLocation(meshProg, "u_focal"), viewport: gl.getUniformLocation(meshProg, "u_viewport"), depthAB: gl.getUniformLocation(meshProg, "u_depthAB") };
    const uSplat = { W: gl.getUniformLocation(splatProg, "u_W"), camPos: gl.getUniformLocation(splatProg, "u_camPos"), focal: gl.getUniformLocation(splatProg, "u_focal"), viewport: gl.getUniformLocation(splatProg, "u_viewport"), depthAB: gl.getUniformLocation(splatProg, "u_depthAB"), scaleMult: gl.getUniformLocation(splatProg, "u_scaleMult"), visMode: gl.getUniformLocation(splatProg, "u_visMode") };
    const uLine = { W: gl.getUniformLocation(lineProg, "u_W"), camPos: gl.getUniformLocation(lineProg, "u_camPos"), focal: gl.getUniformLocation(lineProg, "u_focal"), viewport: gl.getUniformLocation(lineProg, "u_viewport"), depthAB: gl.getUniformLocation(lineProg, "u_depthAB") };

    const st = {
      activeJobId: "__default_courtyard__",
      hasOrigBackup: false,
      fpsMode: false,
      locomotion: "walk",
      doubleSpaceFly: true,
      autoLand: true,
      lastSpaceMs: 0,
      eye: [2.8, -4.6, 1.65],
      footZ: 0.0,
      vertVel: 0.0,
      yaw: -0.52,
      pitch: -0.12,
      headYaw: 0.0,
      headPitch: 0.0,
      gimbalLocked: false,
      autoRun: false,
      prone: false,
      leanAmt: 0.0,
      rollDeg: 0.0,
      lensMm: 35,
      speedMult: 1.0,
      binoZoom: false,
      adsTarget: null,
      keys: new Set(),
      orbitTarget: [0, 0, 1.0],
      orbitTheta: -1.15,
      orbitPhi: 0.28,
      orbitDist: 5.8,
      orbitDragging: false,
      orbitPan: false,
      lastX: 0,
      lastY: 0,
      slot: 1,
      voxelSizeIdx: 1,
      brushRadius: 0.45,
      brushReach: 3.5,
      fixedReach: false,
      paintSubmode: "select",
      mouseDownBtn: -1,
      splatVisMode: 0,
      splatScaleMult: 1.0,
      showSplats: true,
      showSolid: true,
      showGrid: true,
      showGizmos: true,
      cropBox: null,
      cropCornerA: null,
      fltMinOpa: 0.0,
      fltMaxScale: 1.50,
      fltMaxAniso: 30.0,
      fltMinZ: -3.0,
      camNodes: [],
      subjectNodes: [],
      trackSubject: false,
      voxels: [],
      pings: [],
      pendingPing: null,
      riding: false,
      rideStartT: 0,
      recordingFlight: false,
      lastRecT: 0,
      lastHit: null,
      brushCursorPt: null,
      liveLinkBlender: false,
      lastBlSyncMs: 0,
      blSyncBusy: false,
    };

    let baseMeshFloats = new Float32Array(0);
    let baseColliders = [];
    let gpuMeshVerts = 0;
    let solidDirty = true;

    let texH = 1, splatCount = 0;
    let positions = null, opacities01 = null, maxScales = null, anisotropies = null;
    let rawSplatBytes = null, posColData = null, posColU32 = null, covAData = null, covBData = null, flagData = null;
    let sortedIdx = null, depthKeys = null;
    const depthBuckets = new Int32Array(65536);
    let validSortedCount = 0, dirtySort = true;
    let sceneCenter = [0, 0, 1.0], sceneRadius = 3.5;
    let undoStack = [], redoStack = [], recordedOps = [], activeStroke = null;

    function pushBoxTris(out, c, sz, rgba, skipBot = false) {
      const [cx, cy, cz] = c, [hx, hy, hz] = [0.5 * sz[0], 0.5 * sz[1], 0.5 * sz[2]];
      const faces = [
        [[1,0,0], [[cx+hx,cy-hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx+hx,cy+hy,cz+hz],[cx+hx,cy-hy,cz+hz]]],
        [[-1,0,0], [[cx-hx,cy+hy,cz-hz],[cx-hx,cy-hy,cz-hz],[cx-hx,cy-hy,cz+hz],[cx-hx,cy+hy,cz+hz]]],
        [[0,1,0], [[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz+hz],[cx+hx,cy+hy,cz+hz]]],
        [[0,-1,0], [[cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz+hz],[cx-hx,cy-hy,cz+hz]]],
        [[0,0,1], [[cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz]]],
      ];
      if (!skipBot) faces.push([[0,0,-1], [[cx-hx,cy+hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx-hx,cy-hy,cz-hz]]]);
      faces.forEach(([nrm, q]) => {
        [[0,1,2],[0,2,3]].forEach(([a,b,d]) => {
          [q[a], q[b], q[d]].forEach((p) => out.push(p[0], p[1], p[2], nrm[0], nrm[1], nrm[2], rgba[0], rgba[1], rgba[2], rgba[3]));
        });
      });
    }

    function buildSolidStage(kind) {
      const out = [], cols = [];
      if (kind === "default_cube" || kind === "courtyard" || kind === "blank_grid") {
        const ext = kind === "courtyard" ? 10 : 7;
        for (let ix = -ext; ix < ext; ix++) {
          for (let iy = -ext; iy < ext; iy++) {
            const c = ((ix + iy) & 1) === 0 ? [0.16, 0.20, 0.28, 1] : [0.11, 0.14, 0.20, 1];
            const q = [[ix,iy,0],[ix+1,iy,0],[ix+1,iy+1,0],[ix,iy+1,0]];
            [[0,1,2],[0,2,3]].forEach(([a,b,d]) => [q[a],q[b],q[d]].forEach((p) => out.push(p[0],p[1],p[2],0,0,1,c[0],c[1],c[2],1)));
          }
        }
      }
      if (kind === "default_cube" || kind === "courtyard") {
        pushBoxTris(out, [0, 0, 1.0], [2.0, 2.0, 2.0], [0.78, 0.81, 0.86, 1.0]);
        cols.push({ min: [-1, -1, 0], max: [1, 1, 2] });
        if (kind === "courtyard") {
          for (let s = 0; s < 4; s++) {
            const h = (s + 1) * 0.40, cy = -2.2 + s * 0.9;
            pushBoxTris(out, [-2.8, cy, h * 0.5], [1.4, 0.9, h], [0.34, 0.48, 0.66, 1], true);
            cols.push({ min: [-3.5, cy - 0.45, 0], max: [-2.1, cy + 0.45, h] });
          }
          [[-3.8,-3.8],[3.8,-3.8],[-3.8,3.8],[3.8,3.8]].forEach(([px, py]) => {
            pushBoxTris(out, [px, py, 1.3], [0.64, 0.64, 2.6], [0.32, 0.39, 0.50, 1], true);
            pushBoxTris(out, [px, py, 1.65], [0.70, 0.70, 0.08], [0.22, 0.80, 0.98, 1]);
            cols.push({ min: [px - 0.32, py - 0.32, 0], max: [px + 0.32, py + 0.32, 2.6] });
          });
          pushBoxTris(out, [0, 3.8, 2.85], [8.24, 0.66, 0.50], [0.42, 0.48, 0.60, 1]);
        }
      }
      baseMeshFloats = new Float32Array(out);
      baseColliders = cols;
      solidDirty = true;
    }

    function syncCombinedMesh() {
      const voxOut = [];
      st.voxels.forEach((vx) => pushBoxTris(voxOut, vx.center, vx.size, vx.color || [0.38, 0.75, 0.96, 1]));
      const comb = new Float32Array(baseMeshFloats.length + voxOut.length);
      comb.set(baseMeshFloats, 0);
      if (voxOut.length) comb.set(voxOut, baseMeshFloats.length);
      gl.bindBuffer(gl.ARRAY_BUFFER, meshBuf);
      gl.bufferData(gl.ARRAY_BUFFER, comb, gl.DYNAMIC_DRAW);
      gpuMeshVerts = comb.length / 10;
      solidDirty = false;
    }

    function uploadFlags() {
      if (!flagData) return;
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, flagTex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, texH, gl.RED_INTEGER, gl.UNSIGNED_BYTE, flagData);
    }

    function ingestSplatBuf(buf) {
      const cap = Math.min(Math.floor(buf.byteLength / 32), 350000);
      splatCount = cap;
      rawSplatBytes = new Uint8Array(buf.slice(0, cap * 32));
      const f32 = new Float32Array(rawSplatBytes.buffer);
      const u32 = new Uint32Array(rawSplatBytes.buffer);
      const u8 = rawSplatBytes;

      texH = Math.max(1, Math.ceil(cap / TEX_W));
      posColData = new Float32Array(TEX_W * texH * 4);
      posColU32 = new Uint32Array(posColData.buffer);
      covAData = new Float32Array(TEX_W * texH * 4);
      covBData = new Float32Array(TEX_W * texH * 4);
      flagData = new Uint8Array(TEX_W * texH);
      positions = new Float32Array(cap * 3);
      opacities01 = new Float32Array(cap);
      maxScales = new Float32Array(cap);
      anisotropies = new Float32Array(cap);
      sortedIdx = new Uint32Array(cap);
      depthKeys = new Float32Array(cap);

      const xs = [], ys = [], zs = [];
      const stride = Math.max(1, Math.floor(cap / 2000));
      for (let i = 0; i < cap; i++) {
        const bf = i * 8, bb = i * 32;
        const px = f32[bf], py = f32[bf + 1], pz = f32[bf + 2];
        const sx = f32[bf + 3], sy = f32[bf + 4], sz = f32[bf + 5];
        positions[3 * i] = px; positions[3 * i + 1] = py; positions[3 * i + 2] = pz;
        if (i % stride === 0 && Number.isFinite(px)) { xs.push(px); ys.push(py); zs.push(pz); }
        posColData[4 * i] = px; posColData[4 * i + 1] = py; posColData[4 * i + 2] = pz;
        posColU32[4 * i + 3] = u32[bf + 6];
        opacities01[i] = u8[bb + 27] / 255.0;

        let s0 = sx, s1 = sy, s2 = sz;
        if (s0 > s1) { const t = s0; s0 = s1; s1 = t; }
        if (s1 > s2) { const t = s1; s1 = s2; s2 = t; }
        if (s0 > s1) { const t = s0; s0 = s1; s1 = t; }
        maxScales[i] = s2;
        anisotropies[i] = s2 / Math.max(s1, 1e-6);

        let qw = (u8[bb+28]-127.5)/127.5, qx = (u8[bb+29]-127.5)/127.5, qy = (u8[bb+30]-127.5)/127.5, qz = (u8[bb+31]-127.5)/127.5;
        const qn = Math.hypot(qw, qx, qy, qz) || 1;
        qw /= qn; qx /= qn; qy /= qn; qz /= qn;
        const r00 = 1-2*(qy*qy+qz*qz), r01 = 2*(qx*qy-qw*qz), r02 = 2*(qx*qz+qw*qy);
        const r10 = 2*(qx*qy+qw*qz), r11 = 1-2*(qx*qx+qz*qz), r12 = 2*(qy*qz-qw*qx);
        const r20 = 2*(qx*qz-qw*qy), r21 = 2*(qy*qz+qw*qx), r22 = 1-2*(qx*qx+qy*qy);
        const m00 = r00*sx, m01 = r01*sy, m02 = r02*sz;
        const m10 = r10*sx, m11 = r11*sy, m12 = r12*sz;
        const m20 = r20*sx, m21 = r21*sy, m22 = r22*sz;
        covAData[4*i] = m00*m00+m01*m01+m02*m02; covAData[4*i+1] = m00*m10+m01*m11+m02*m12; covAData[4*i+2] = m00*m20+m01*m21+m02*m22; covAData[4*i+3] = anisotropies[i];
        covBData[4*i] = m10*m10+m11*m11+m12*m12; covBData[4*i+1] = m10*m20+m11*m21+m12*m22; covBData[4*i+2] = m20*m20+m21*m21+m22*m22; covBData[4*i+3] = s2;
        sortedIdx[i] = i;
      }

      if (xs.length > 8) {
        xs.sort((a,b) => a-b); ys.sort((a,b) => a-b); zs.sort((a,b) => a-b);
        const m = xs.length >> 1;
        sceneCenter = [xs[m], ys[m], zs[m]];
        const ds = xs.map((_, k) => Math.hypot(xs[k]-sceneCenter[0], ys[k]-sceneCenter[1], zs[k]-sceneCenter[2])).sort((a,b) => a-b);
        sceneRadius = Math.max(1.4, ds[Math.floor(ds.length * 0.7)] || 3.0);
      }

      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, posColTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, texH, 0, gl.RGBA, gl.FLOAT, posColData);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, covTexA);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, texH, 0, gl.RGBA, gl.FLOAT, covAData);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, covTexB);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, texH, 0, gl.RGBA, gl.FLOAT, covBData);
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, flagTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8UI, TEX_W, texH, 0, gl.RED_INTEGER, gl.UNSIGNED_BYTE, flagData);
      gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
      gl.bufferData(gl.ARRAY_BUFFER, sortedIdx.byteLength, gl.DYNAMIC_DRAW);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, sortedIdx);

      undoStack = []; redoStack = []; recordedOps = []; dirtySort = true;
      updateSplatStatsHud();
    }

    function sortBackToFront(cam) {
      if (!splatCount || !positions || !flagData) return;
      let minZ = 1e9, maxZ = -1e9;
      const ex = cam.eye[0], ey = cam.eye[1], ez = cam.eye[2];
      for (let i = 0; i < splatCount; i++) {
        if ((flagData[i] & 1) !== 0) { depthKeys[i] = -1; continue; }
        const z = (positions[3*i]-ex)*cam.fx + (positions[3*i+1]-ey)*cam.fy + (positions[3*i+2]-ez)*cam.fz;
        depthKeys[i] = z;
        if (z > 0.08) { if (z < minZ) minZ = z; if (z > maxZ) maxZ = z; }
      }
      if (maxZ <= minZ) maxZ = minZ + 1;
      depthBuckets.fill(0);
      const inv = 65535.0 / (maxZ - minZ);
      for (let i = 0; i < splatCount; i++) {
        const z = depthKeys[i];
        const b = z <= 0.08 ? 65535 : Math.max(0, Math.min(65534, 65534 - (((z - minZ) * inv) | 0)));
        depthKeys[i] = b; depthBuckets[b]++;
      }
      let sum = 0;
      for (let b = 0; b < 65536; b++) { const c = depthBuckets[b]; depthBuckets[b] = sum; sum += c; }
      validSortedCount = depthBuckets[65535];
      for (let i = 0; i < splatCount; i++) sortedIdx[depthBuckets[depthKeys[i]]++] = i;
      gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
      if (validSortedCount > 0) gl.bufferSubData(gl.ARRAY_BUFFER, 0, sortedIdx.subarray(0, validSortedCount));
      dirtySort = false;
    }

    function updateSplatStatsHud() {
      el("btn-undo").disabled = undoStack.length === 0;
      el("btn-redo").disabled = redoStack.length === 0;
      el("btn-restore-orig").classList.toggle("hidden", !st.hasOrigBackup);
      if (!splatCount || !flagData) {
        el("badge-splat-stats").textContent = "Solid Stage";
        el("lbl-selected-cnt").textContent = "0";
        return;
      }
      let culled = 0, sel = 0;
      for (let i = 0; i < splatCount; i++) {
        if (flagData[i] & 1) culled++;
        else if (flagData[i] & 2) sel++;
      }
      el("badge-splat-stats").textContent =
        `${(splatCount - culled).toLocaleString()} / ${splatCount.toLocaleString()} pts` + (culled ? ` (${culled.toLocaleString()} culled)` : "");
      el("lbl-selected-cnt").textContent = String(sel);
    }

    function pushUndo(entry) {
      undoStack.push(entry);
      if (undoStack.length > 40) undoStack.shift();
      redoStack = [];
      if (entry.serverOp) recordedOps.push(entry.serverOp);
      updateSplatStatsHud();
    }

    function runUndoRedo(isUndo) {
      const src = isUndo ? undoStack : redoStack;
      const dst = isUndo ? redoStack : undoStack;
      if (!src.length || !flagData) return;
      const e = src.pop(); dst.push(e);
      if (e.kind === "flags") {
        const vals = isUndo ? e.before : e.after;
        for (let k = 0; k < e.indices.length; k++) flagData[e.indices[k]] = vals[k];
        uploadFlags(); dirtySort = true;
      } else if (e.kind === "paint") {
        const cols = isUndo ? e.colBefore : e.colAfter;
        for (let k = 0; k < e.indices.length; k++) {
          const idx = e.indices[k];
          if (cols) {
            const u = cols[k]; posColU32[4 * idx + 3] = u;
            rawSplatBytes[idx*32+24] = u&255; rawSplatBytes[idx*32+25] = (u>>>8)&255; rawSplatBytes[idx*32+26] = (u>>>16)&255; rawSplatBytes[idx*32+27] = (u>>>24)&255;
          }
          if (e.covScaleFactor) {
            const f = isUndo ? 1 / e.covScaleFactor : e.covScaleFactor, f2 = f * f;
            covAData[4*idx]*=f2; covAData[4*idx+1]*=f2; covAData[4*idx+2]*=f2;
            covBData[4*idx]*=f2; covBData[4*idx+1]*=f2; covBData[4*idx+2]*=f2; covBData[4*idx+3]*=f;
            maxScales[idx] *= f;
          }
        }
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, posColTex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, texH, gl.RGBA, gl.FLOAT, posColData);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, covTexA);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, texH, gl.RGBA, gl.FLOAT, covAData);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, covTexB);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, texH, gl.RGBA, gl.FLOAT, covBData);
      }
      if (isUndo && e.serverOp && recordedOps.length) recordedOps.pop();
      else if (!isUndo && e.serverOp) recordedOps.push(e.serverOp);
      updateSplatStatsHud();
    }

    function beginStroke(mode) {
      if (!splatCount || !flagData) return;
      activeStroke = { mode, touched: new Map(), points: [], radius: st.brushRadius };
    }

    function stepStroke(pt) {
      if (!activeStroke || !splatCount || !positions || !pt) return;
      const [cx, cy, cz] = pt, r = st.brushRadius, r2 = r * r;
      const lp = activeStroke.points[activeStroke.points.length - 1];
      if (!lp || Math.hypot(cx - lp[0], cy - lp[1], cz - lp[2]) > r * 0.22) {
        activeStroke.points.push([Number(cx.toFixed(4)), Number(cy.toFixed(4)), Number(cz.toFixed(4))]);
      }
      const mode = activeStroke.mode;
      let chFlags = false, chCol = false, chCov = false;
      const [tr, tg, tb] = hexToRgb01(el("inp-paint-color").value);

      for (let i = 0; i < splatCount; i++) {
        const dx = positions[3*i] - cx; if (dx > r || dx < -r) continue;
        const dy = positions[3*i+1] - cy; if (dy > r || dy < -r) continue;
        const dz = positions[3*i+2] - cz; if (dx*dx + dy*dy + dz*dz > r2) continue;
        const cf = flagData[i];
        if (mode === "cull" && (cf & 1) === 0) {
          if (!activeStroke.touched.has(i)) activeStroke.touched.set(i, cf);
          flagData[i] = (cf | 1) & ~2; chFlags = true;
        } else if (mode === "restore" && (cf & 1) !== 0) {
          if (!activeStroke.touched.has(i)) activeStroke.touched.set(i, cf);
          flagData[i] = cf & ~1; chFlags = true;
        } else if (mode === "select" && (cf & 1) === 0 && (cf & 2) === 0) {
          if (!activeStroke.touched.has(i)) activeStroke.touched.set(i, cf);
          flagData[i] = cf | 2; chFlags = true;
        } else if (mode === "deselect" && (cf & 2) !== 0) {
          if (!activeStroke.touched.has(i)) activeStroke.touched.set(i, cf);
          flagData[i] = cf & ~2; chFlags = true;
        } else if ((cf & 1) === 0 && !activeStroke.touched.has(i)) {
          if (mode === "tint") {
            const pu = posColU32[4*i+3]; activeStroke.touched.set(i, pu);
            const nr = Math.round(((pu&255)/255 * 0.45 + tr * 0.55) * 255);
            const ng = Math.round((((pu>>>8)&255)/255 * 0.45 + tg * 0.55) * 255);
            const nb = Math.round((((pu>>>16)&255)/255 * 0.45 + tb * 0.55) * 255);
            const nu = (((pu>>>24)&255) << 24 | (nb << 16) | (ng << 8) | nr) >>> 0;
            posColU32[4*i+3] = nu; rawSplatBytes[i*32+24]=nr; rawSplatBytes[i*32+25]=ng; rawSplatBytes[i*32+26]=nb;
            chCol = true;
          } else if (mode === "soften") {
            const pu = posColU32[4*i+3]; activeStroke.touched.set(i, pu);
            const ca = Math.max(4, Math.round(((pu>>>24)&255) * 0.65));
            posColU32[4*i+3] = ((ca << 24) | (pu & 0x00ffffff)) >>> 0;
            rawSplatBytes[i*32+27] = ca; opacities01[i] = ca / 255.0; chCol = true;
          } else if (mode === "shrink") {
            const fac = 0.72, f2 = fac * fac;
            activeStroke.touched.set(i, fac);
            covAData[4*i]*=f2; covAData[4*i+1]*=f2; covAData[4*i+2]*=f2;
            covBData[4*i]*=f2; covBData[4*i+1]*=f2; covBData[4*i+2]*=f2; covBData[4*i+3]*=fac;
            maxScales[i] *= fac; chCov = true;
          }
        }
      }
      if (chFlags) { uploadFlags(); if (mode === "restore") dirtySort = true; }
      if (chCol) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, posColTex); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, texH, gl.RGBA, gl.FLOAT, posColData); }
      if (chCov) {
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, covTexA); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, texH, gl.RGBA, gl.FLOAT, covAData);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, covTexB); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, texH, gl.RGBA, gl.FLOAT, covBData);
      }
    }

    function endStroke() {
      if (!activeStroke) return;
      const s = activeStroke; activeStroke = null;
      if (!s.touched.size) return;
      const n = s.touched.size, indices = new Uint32Array(n);
      if (["cull", "restore", "select", "deselect"].includes(s.mode)) {
        const before = new Uint8Array(n), after = new Uint8Array(n);
        let k = 0;
        s.touched.forEach((bv, idx) => { indices[k] = idx; before[k] = bv; after[k] = flagData[idx]; k++; });
        const sop = s.mode === "cull" ? { op: "brush_cull", points: s.points, radius: s.radius } : s.mode === "restore" ? { op: "brush_restore", points: s.points, radius: s.radius } : null;
        pushUndo({ kind: "flags", indices, before, after, serverOp: sop });
        dirtySort = true;
      } else if (s.mode === "tint" || s.mode === "soften") {
        const colBefore = new Uint32Array(n), colAfter = new Uint32Array(n);
        let k = 0;
        s.touched.forEach((bv, idx) => { indices[k] = idx; colBefore[k] = bv; colAfter[k] = posColU32[4*idx+3]; k++; });
        const sop = s.mode === "tint" ? { op: "brush_paint", points: s.points, radius: s.radius, color_rgb: hexToRgb01(el("inp-paint-color").value), blend: 0.55 } : { op: "brush_paint", points: s.points, radius: s.radius, opacity_mult: 0.65 };
        pushUndo({ kind: "paint", indices, colBefore, colAfter, serverOp: sop });
      } else if (s.mode === "shrink") {
        let k = 0; s.touched.forEach((_, idx) => { indices[k++] = idx; });
        pushUndo({ kind: "paint", indices, covScaleFactor: 0.72, serverOp: { op: "brush_paint", points: s.points, radius: s.radius, scale_mult: 0.72 } });
      }
    }

    function mutateFlags(pred, nextFn, serverOp, msg) {
      if (!splatCount || !flagData) return;
      const idxs = [], bef = [], aft = [];
      for (let i = 0; i < splatCount; i++) {
        if (pred(i)) {
          const b = flagData[i], a = nextFn(b);
          if (a !== b) { idxs.push(i); bef.push(b); aft.push(a); flagData[i] = a; }
        }
      }
      if (!idxs.length) return showToast("0 Gaussians matched");
      uploadFlags(); dirtySort = true;
      pushUndo({ kind: "flags", indices: new Uint32Array(idxs), before: new Uint8Array(bef), after: new Uint8Array(aft), serverOp });
      if (msg) showToast(`${msg} (${idxs.length.toLocaleString()} Gaussians)`);
    }

    function isFilterActive() {
      return st.fltMinOpa > 0.005 || st.fltMaxScale < 1.48 || st.fltMaxAniso < 29.5 || st.fltMinZ > -2.95;
    }
    function matchesFilter(i) {
      if ((flagData[i] & 1) !== 0) return false;
      if (st.fltMinOpa > 0.005 && opacities01[i] < st.fltMinOpa) return true;
      if (st.fltMaxScale < 1.48 && maxScales[i] > st.fltMaxScale) return true;
      if (st.fltMaxAniso < 29.5 && anisotropies[i] > st.fltMaxAniso) return true;
      if (st.fltMinZ > -2.95 && positions[3 * i + 2] < st.fltMinZ) return true;
      return false;
    }
    function updateFilterPreview() {
      if (!splatCount || !flagData) return;
      const act = isFilterActive();
      let m = 0;
      for (let i = 0; i < splatCount; i++) {
        if (act && matchesFilter(i)) { flagData[i] |= 4; m++; }
        else flagData[i] &= ~4;
      }
      uploadFlags();
      el("lbl-filter-preview").textContent = `${m.toLocaleString()} marked`;
    }

    // --- Basis, Spline & Raycasting ---
    function basisFromEyeFwd(eye, fwd, roll = 0, lensOv = null) {
      let fx = fwd[0], fy = fwd[1], fz = fwd[2];
      const fn = Math.hypot(fx, fy, fz) || 1; fx /= fn; fy /= fn; fz /= fn;
      let rx = fy, ry = -fx, rz = 0;
      const rn = Math.hypot(rx, ry, rz);
      if (rn < 1e-5) { rx = 1; ry = 0; } else { rx /= rn; ry /= rn; }
      let ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
      if (Math.abs(roll) > 0.01) {
        const rad = (roll * Math.PI) / 180, cr = Math.cos(rad), sr = Math.sin(rad);
        const nrx = rx * cr + ux * sr, nry = ry * cr + uy * sr, nrz = rz * cr + uz * sr;
        ux = -rx * sr + ux * cr; uy = -ry * sr + uy * cr; uz = -rz * sr + uz * cr;
        rx = nrx; ry = nry; rz = nrz;
      }
      return { eye: eye.slice(), rx, ry, rz, ux, uy, uz, fx, fy, fz, W: new Float32Array([rx, ux, fx, ry, uy, fy, rz, uz, fz]), lensMm: lensOv !== null ? lensOv : (st.binoZoom ? 135 : st.lensMm) };
    }

    function catmull3(p0, p1, p2, p3, u) {
      const u2 = u * u, u3 = u2 * u;
      return [0,1,2].map((k) => 0.5 * (2*p1[k] + (-p0[k]+p2[k])*u + (2*p0[k]-5*p1[k]+4*p2[k]-p3[k])*u2 + (-p0[k]+3*p1[k]-3*p2[k]+p3[k])*u3));
    }
    function evalSpline(pts, frac) {
      if (!pts.length) return [0,0,0];
      if (pts.length === 1) return pts[0].slice();
      const f = Math.max(0, Math.min(1, frac)) * (pts.length - 1);
      const i = Math.min(pts.length - 2, Math.floor(f));
      return catmull3(pts[Math.max(0, i-1)], pts[i], pts[i+1], pts[Math.min(pts.length-1, i+2)], f - i);
    }
    function evalRailCam(frac) {
      const n = st.camNodes.length;
      if (!n) return computeCam();
      if (n === 1) return basisFromEyeFwd(st.camNodes[0].position, st.camNodes[0].forward, st.camNodes[0].roll_deg || 0, st.camNodes[0].lens_mm || 35);
      const pos = evalSpline(st.camNodes.map((c) => c.position), frac);
      const fwd = (st.trackSubject && st.subjectNodes.length)
        ? (() => { const sp = evalSpline(st.subjectNodes.map((s) => s.position), frac); return [sp[0]-pos[0], sp[1]-pos[1], sp[2]+1.1-pos[2]]; })()
        : evalSpline(st.camNodes.map((c) => c.forward), frac);
      return basisFromEyeFwd(pos, fwd, 0, st.camNodes[0].lens_mm || 35);
    }

    function computeCam() {
      if (st.riding && st.camNodes.length) {
        const dur = Math.max(1, Number(el("inp-rail-dur").value || 5));
        return evalRailCam((((performance.now() - st.rideStartT) / 1000) % dur) / dur);
      }
      if (!st.fpsMode) {
        const cp = Math.cos(st.orbitPhi), sp = Math.sin(st.orbitPhi), ct = Math.cos(st.orbitTheta), snt = Math.sin(st.orbitTheta);
        const eye = [st.orbitTarget[0] + st.orbitDist * cp * ct, st.orbitTarget[1] + st.orbitDist * cp * snt, st.orbitTarget[2] + st.orbitDist * sp];
        return basisFromEyeFwd(eye, [st.orbitTarget[0] - eye[0], st.orbitTarget[1] - eye[1], st.orbitTarget[2] - eye[2]], 0, st.lensMm);
      }
      const brx = Math.cos(st.yaw), bry = -Math.sin(st.yaw);
      const effEye = [st.eye[0] + brx * st.leanAmt * 0.42, st.eye[1] + bry * st.leanAmt * 0.42, st.eye[2] - Math.abs(st.leanAmt) * 0.06];
      let fwd;
      if (st.adsTarget) {
        fwd = [st.adsTarget[0] - effEye[0], st.adsTarget[1] - effEye[1], st.adsTarget[2] - effEye[2]];
      } else {
        const ly = st.yaw + st.headYaw, lp = Math.max(-1.48, Math.min(1.48, st.pitch + st.headPitch));
        fwd = [Math.sin(ly) * Math.cos(lp), Math.cos(ly) * Math.cos(lp), Math.sin(lp)];
      }
      return basisFromEyeFwd(effEye, fwd, st.rollDeg + st.leanAmt * 6.5);
    }

    function rayBox(eye, fwd, bmin, bmax) {
      let tmin = -Infinity, tmax = Infinity, ha = 2, hs = 1;
      for (let a = 0; a < 3; a++) {
        const d = fwd[a];
        if (Math.abs(d) < 1e-7) { if (eye[a] < bmin[a] || eye[a] > bmax[a]) return null; }
        else {
          let t1 = (bmin[a] - eye[a]) / d, t2 = (bmax[a] - eye[a]) / d, sg = -1;
          if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; sg = 1; }
          if (t1 > tmin) { tmin = t1; ha = a; hs = sg; }
          if (t2 < tmax) tmax = t2;
          if (tmin > tmax) return null;
        }
      }
      if (tmin <= 0.12) return null;
      const nrm = [0,0,0]; nrm[ha] = hs;
      return { t: tmin, normal: nrm };
    }

    function raycast(eye, fwd, includeCulled = false) {
      let bestT = Infinity, bestHit = null;
      st.voxels.forEach((vx, vi) => {
        const c = vx.center, s = vx.size;
        const r = rayBox(eye, fwd, [c[0]-0.5*s[0],c[1]-0.5*s[1],c[2]-0.5*s[2]], [c[0]+0.5*s[0],c[1]+0.5*s[1],c[2]+0.5*s[2]]);
        if (r && r.t < bestT) { bestT = r.t; bestHit = { pt: [eye[0]+fwd[0]*r.t, eye[1]+fwd[1]*r.t, eye[2]+fwd[2]*r.t], dist: r.t, normal: r.normal, kind: "voxel", voxelIdx: vi }; }
      });
      if (st.showSolid) {
        baseColliders.forEach((bx) => {
          const r = rayBox(eye, fwd, bx.min, bx.max);
          if (r && r.t < bestT) { bestT = r.t; bestHit = { pt: [eye[0]+fwd[0]*r.t, eye[1]+fwd[1]*r.t, eye[2]+fwd[2]*r.t], dist: r.t, normal: r.normal, kind: "mesh", voxelIdx: -1 }; }
        });
      }
      if (st.showSplats && splatCount > 0 && positions && flagData) {
        const stride = Math.max(1, Math.floor(splatCount / 25000));
        const hz = [];
        for (let i = 0; i < splatCount; i += stride) {
          if (!includeCulled && (flagData[i] & 1) !== 0) continue;
          const dx = positions[3*i]-eye[0], dy = positions[3*i+1]-eye[1], dz = positions[3*i+2]-eye[2];
          const z = dx*fwd[0] + dy*fwd[1] + dz*fwd[2];
          if (z <= 0.18 || z >= bestT) continue;
          const d2 = dx*dx + dy*dy + dz*dz - z*z;
          const rt = z * 0.038 + 0.06;
          if (d2 <= rt * rt) hz.push(z);
        }
        if (hz.length >= 2) {
          hz.sort((a,b) => a-b);
          const zp = hz[Math.floor(hz.length * 0.12)];
          if (zp < bestT) { bestT = zp; bestHit = { pt: [eye[0]+fwd[0]*zp, eye[1]+fwd[1]*zp, eye[2]+fwd[2]*zp], dist: zp, normal: [0,0,1], kind: "splat", voxelIdx: -1 }; }
        }
      }
      if (!bestHit && fwd[2] < -0.012 && eye[2] > 0) {
        const tg = -eye[2] / fwd[2];
        if (tg > 0.1 && tg < 90) bestHit = { pt: [eye[0]+fwd[0]*tg, eye[1]+fwd[1]*tg, 0], dist: tg, normal: [0,0,1], kind: "floor", voxelIdx: -1 };
      }
      return bestHit;
    }

    function supportAt(x, y, curFootZ) {
      let sup = 0.0;
      const r = 0.28;
      const chk = (bmin, bmax) => {
        if (x + r > bmin[0] && x - r < bmax[0] && y + r > bmin[1] && y - r < bmax[1] && bmax[2] <= curFootZ + 0.56) {
          if (bmax[2] > sup) sup = bmax[2];
        }
      };
      if (st.showSolid) baseColliders.forEach((b) => chk(b.min, b.max));
      st.voxels.forEach((vx) => chk([vx.center[0]-0.5*vx.size[0],vx.center[1]-0.5*vx.size[1],vx.center[2]-0.5*vx.size[2]], [vx.center[0]+0.5*vx.size[0],vx.center[1]+0.5*vx.size[1],vx.center[2]+0.5*vx.size[2]]));
      return sup;
    }

    function setLocomotion(mode, announce = true) {
      st.locomotion = mode;
      if (mode === "walk") st.footZ = Math.max(0, st.eye[2] - 1.65);
      else st.vertVel = 0;
      el("btn-mode-walk").classList.toggle("active-mode", mode === "walk");
      el("btn-mode-fly").classList.toggle("active-mode", mode === "fly");
      el("btn-mode-freecam").classList.toggle("active-mode", mode === "freecam");
      const b = el("badge-locomotion");
      b.textContent = mode === "walk" ? "🚶 WALK 1.65m" : mode === "fly" ? "🛸 CREATIVE FLY" : "🎥 3D FLY";
      b.className = `hud-badge ${mode === "walk" ? "badge-walk" : "badge-fly"}`;
      if (announce) showToast(mode === "walk" ? "🚶 Walk Mode (Space×2 = Creative Fly)" : "🛸 Minecraft Creative Fly (Space=Up · Shift/C=Down · Ctrl=Sprint)");
    }

    function stepPhysics(dt) {
      if (state.centerKind !== "3d" || !st.fpsMode || st.riding) return;
      if (!el("fps-ping-modal").classList.contains("hidden") || !el("fps-help-modal").classList.contains("hidden")) return;

      const alt = st.keys.has("AltLeft") || st.keys.has("AltRight");
      if (!alt && !st.gimbalLocked && (Math.abs(st.headYaw) > 1e-4 || Math.abs(st.headPitch) > 1e-4)) {
        const k = Math.min(1, dt * 14);
        st.headYaw *= 1 - k; st.headPitch *= 1 - k;
        dirtySort = true;
      }

      const shift = st.keys.has("ShiftLeft") || st.keys.has("ShiftRight");
      const ctrl = st.keys.has("ControlLeft") || st.keys.has("ControlRight");
      const cKey = st.keys.has("KeyC");
      const spKey = st.keys.has("Space");
      const flying = st.locomotion !== "walk";

      let tLean = 0;
      if (!shift) { if (st.keys.has("KeyQ")) tLean -= 1; if (st.keys.has("KeyE")) tLean += 1; }
      else if (!flying) { if (st.keys.has("KeyQ")) st.rollDeg -= 30 * dt; if (st.keys.has("KeyE")) st.rollDeg += 30 * dt; }
      st.leanAmt += (tLean - st.leanAmt) * Math.min(1, dt * 13);

      let mult = st.speedMult;
      if (flying) { if (ctrl) mult *= 2.6; }
      else { if (shift) mult *= 2.4; else if (ctrl) mult *= 0.32; if (st.prone) mult *= 0.28; else if (cKey) mult *= 0.58; }
      const spd = Math.max(3.6, sceneRadius * 1.15) * mult;

      const useP = st.locomotion === "freecam";
      const cp = Math.cos(st.pitch);
      const bfx = Math.sin(st.yaw) * (useP ? cp : 1), bfy = Math.cos(st.yaw) * (useP ? cp : 1), bfz = useP ? Math.sin(st.pitch) : 0;
      const brx = Math.cos(st.yaw), bry = -Math.sin(st.yaw);
      if (st.keys.has("KeyS")) st.autoRun = false;

      let vx = 0, vy = 0, vz = 0;
      if (st.keys.has("KeyW") || st.autoRun) { vx += bfx; vy += bfy; vz += bfz; }
      if (st.keys.has("KeyS")) { vx -= bfx; vy -= bfy; vz -= bfz; }
      if (st.keys.has("KeyD")) { vx += brx; vy += bry; }
      if (st.keys.has("KeyA")) { vx -= brx; vy -= bry; }
      if (flying) { if (spKey) vz += 1; if (shift || cKey) vz -= 1; }

      if (vx || vy || vz) {
        const vn = Math.hypot(vx, vy, vz) || 1;
        st.eye[0] += (vx / vn) * spd * dt;
        st.eye[1] += (vy / vn) * spd * dt;
        if (flying) {
          st.eye[2] += (vz / vn) * spd * dt;
          if (st.autoLand && (shift || cKey) && !spKey) {
            const ft = supportAt(st.eye[0], st.eye[1], st.eye[2] - 1.0);
            if (st.eye[2] <= ft + 1.62) { st.footZ = ft; st.eye[2] = ft + 1.65; setLocomotion("walk", true); }
          }
        }
        dirtySort = true;
      }

      if (!flying) {
        const supZ = supportAt(st.eye[0], st.eye[1], st.footZ);
        if (spKey && st.footZ <= supZ + 0.02 && st.vertVel <= 0.01) { st.prone = false; st.vertVel = 5.0; }
        if (st.vertVel !== 0 || st.footZ > supZ) {
          st.footZ += st.vertVel * dt;
          if (st.footZ <= supZ) { st.footZ = supZ; st.vertVel = 0; }
          else st.vertVel -= 13.0 * dt;
          dirtySort = true;
        } else if (st.footZ < supZ) {
          st.footZ += (supZ - st.footZ) * Math.min(1, dt * 16);
          dirtySort = true;
        }
        const stanceH = cKey ? 0.95 : st.prone ? 0.32 : 1.65;
        st.eye[2] += (st.footZ + stanceH - st.eye[2]) * Math.min(1, dt * 15);
      }

      const now = performance.now();
      if (st.keys.has("KeyR") && st.slot === 5) {
        if (!st.recordingFlight) { st.recordingFlight = true; st.camNodes = []; st.lastRecT = 0; }
        if (now - st.lastRecT >= 200) {
          st.lastRecT = now;
          const c = computeCam();
          st.camNodes.push({ position: c.eye.slice(), forward: [c.fx, c.fy, c.fz], lens_mm: st.lensMm, roll_deg: st.rollDeg });
          syncHotbar();
        }
      } else st.recordingFlight = false;
    }

    function snapVoxel(hit, sz) {
      if (hit.kind === "voxel") {
        const b = st.voxels[hit.voxelIdx];
        return [0,1,2].map((a) => b.center[a] + hit.normal[a] * (0.5 * b.size[a] + 0.5 * sz));
      }
      return [Math.round(hit.pt[0] / sz) * sz, Math.round(hit.pt[1] / sz) * sz, Math.max(0.5 * sz, Math.floor((hit.pt[2] + 0.02) / sz) * sz + 0.5 * sz)];
    }

    function drawGizmos(cam, w, h) {
      const verts = [];
      const ex = cam.eye[0], ey = cam.eye[1], ez = cam.eye[2], W = cam.W;
      const zc = (p) => (p[0]-ex)*W[2] + (p[1]-ey)*W[5] + (p[2]-ez)*W[8];
      const seg = (a, b, r, g, bl, al) => {
        const za = zc(a), zb = zc(b);
        if (za < 0.09 && zb < 0.09) return;
        let pa = a, pb = b;
        if (za < 0.09) { const t = (0.09 - za) / (zb - za); pa = [0,1,2].map((k) => a[k] + (b[k] - a[k]) * t); }
        else if (zb < 0.09) { const t = (0.09 - zb) / (za - zb); pb = [0,1,2].map((k) => b[k] + (a[k] - b[k]) * t); }
        verts.push(pa[0],pa[1],pa[2],r,g,bl,al, pb[0],pb[1],pb[2],r,g,bl,al);
      };
      const boxMM = (mn, mx, r, g, bl, al) => {
        const p = [[mn[0],mn[1],mn[2]],[mx[0],mn[1],mn[2]],[mx[0],mx[1],mn[2]],[mn[0],mx[1],mn[2]],[mn[0],mn[1],mx[2]],[mx[0],mn[1],mx[2]],[mx[0],mx[1],mx[2]],[mn[0],mx[1],mx[2]]];
        [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([i,j]) => seg(p[i], p[j], r, g, bl, al));
      };
      const sphere = (c, rad, r, g, bl, al) => {
        for (let k = 0; k < 24; k++) {
          const a0 = (k / 24) * Math.PI * 2, a1 = ((k + 1) / 24) * Math.PI * 2;
          const c0 = Math.cos(a0)*rad, s0 = Math.sin(a0)*rad, c1 = Math.cos(a1)*rad, s1 = Math.sin(a1)*rad;
          seg([c[0]+c0,c[1]+s0,c[2]], [c[0]+c1,c[1]+s1,c[2]], r, g, bl, al);
          seg([c[0]+c0,c[1],c[2]+s0], [c[0]+c1,c[1],c[2]+s1], r, g, bl, al);
          seg([c[0],c[1]+c0,c[2]+s0], [c[0],c[1]+c1,c[2]+s1], r, g, bl, al);
        }
      };

      if (st.showGrid) {
        for (let g = -10; g <= 10; g++) {
          if (g === 0) continue;
          seg([g,-10,0.004],[g,10,0.004], 0.25, 0.38, 0.60, 0.24);
          seg([-10,g,0.004],[10,g,0.004], 0.25, 0.38, 0.60, 0.24);
        }
        seg([-10,0,0.006],[10,0,0.006], 0.92, 0.26, 0.28, 0.72);
        seg([0,-10,0.006],[0,10,0.006], 0.24, 0.86, 0.44, 0.72);
        seg([0,0,0],[0,0,2.8], 0.28, 0.62, 1.0, 0.75);
      }

      if (st.showGizmos) {
        if (st.fpsMode && !st.riding && (st.slot === 1 || st.slot === 3) && st.brushCursorPt) {
          const [br, bg, bb] = st.slot === 1 && st.mouseDownBtn !== 2 ? [1.0, 0.36, 0.36] : [0.22, 0.88, 1.0];
          sphere(st.brushCursorPt, st.brushRadius, br, bg, bb, 0.88);
        }
        if (st.cropBox) boxMM(st.cropBox.min, st.cropBox.max, 1.0, 0.78, 0.18, 0.95);
        if (st.fpsMode && st.slot === 7 && st.lastHit) {
          const vs = VOXEL_SIZES[st.voxelSizeIdx], vc = snapVoxel(st.lastHit, vs);
          boxMM([vc[0]-0.5*vs,vc[1]-0.5*vs,vc[2]-0.5*vs], [vc[0]+0.5*vs,vc[1]+0.5*vs,vc[2]+0.5*vs], 0.35, 1.0, 0.55, 0.75);
        }
        if (st.camNodes.length >= 2) {
          const pts = st.camNodes.map((n) => n.position);
          let prev = pts[0];
          for (let s = 1; s <= pts.length * 12; s++) {
            const cur = evalSpline(pts, s / (pts.length * 12));
            seg(prev, cur, 0.22, 0.80, 1.0, 0.95); prev = cur;
          }
        }
        st.camNodes.forEach((nd) => {
          const L = nd.position;
          boxMM([L[0]-0.1,L[1]-0.1,L[2]-0.1], [L[0]+0.1,L[1]+0.1,L[2]+0.1], 1.0, 0.82, 0.2, 0.95);
        });
        st.pings.forEach((pg) => {
          const p = pg.position;
          seg(p, [p[0], p[1], p[2] + 1.4], 1.0, 0.48, 0.16, 0.95);
        });
      }

      if (!verts.length) return;
      const focalPx = ((cam.lensMm || 35) / 36.0) * Math.max(w, h);
      gl.useProgram(lineProg); gl.bindVertexArray(lineVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.DYNAMIC_DRAW);
      gl.uniformMatrix3fv(uLine.W, false, cam.W);
      gl.uniform3f(uLine.camPos, cam.eye[0], cam.eye[1], cam.eye[2]);
      gl.uniform2f(uLine.focal, focalPx, focalPx);
      gl.uniform2f(uLine.viewport, w, h);
      gl.uniform2f(uLine.depthAB, DEPTH_A, DEPTH_B);
      gl.drawArrays(gl.LINES, 0, verts.length / 7);
    }

    function drawRadar(cam) {
      const rc = el("fps-radar-canvas");
      if (!rc || rc.classList.contains("hidden")) return;
      const g = rc.getContext("2d"), W = rc.width, cx = W * 0.5, cy = W * 0.5, rad = W * 0.45;
      g.clearRect(0, 0, W, W);
      g.strokeStyle = "rgba(56, 189, 248, 0.25)";
      g.beginPath(); g.arc(cx, cy, rad * 0.55, 0, Math.PI * 2); g.stroke();
      const lookAng = Math.atan2(cam.fx, cam.fy) - st.yaw;
      const hf = Math.atan(18.0 / Math.max(12, cam.lensMm || 35));
      g.fillStyle = "rgba(56, 189, 248, 0.24)";
      g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, rad * 0.72, -Math.PI * 0.5 + lookAng - hf, -Math.PI * 0.5 + lookAng + hf); g.closePath(); g.fill();
      g.strokeStyle = "#4ade80"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(cx, cy + 4); g.lineTo(cx, cy - 10); g.stroke();
    }

    function renderPass(cam, w, h, clean = false) {
      if (solidDirty) syncCombinedMesh();
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.032, 0.045, 0.075, 1.0);
      gl.depthMask(true);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      const focalPx = ((cam.lensMm || 35) / 36.0) * Math.max(w, h);

      if (st.showSolid && gpuMeshVerts > 0) {
        gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(true); gl.disable(gl.BLEND);
        gl.useProgram(meshProg); gl.bindVertexArray(meshVao);
        gl.uniformMatrix3fv(uMesh.W, false, cam.W);
        gl.uniform3f(uMesh.camPos, cam.eye[0], cam.eye[1], cam.eye[2]);
        gl.uniform2f(uMesh.focal, focalPx, focalPx);
        gl.uniform2f(uMesh.viewport, w, h);
        gl.uniform2f(uMesh.depthAB, DEPTH_A, DEPTH_B);
        gl.drawArrays(gl.TRIANGLES, 0, gpuMeshVerts);
      }

      if (st.showSplats && splatCount > 0) {
        if (dirtySort || clean) sortBackToFront(cam);
        if (validSortedCount > 0) {
          gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(false);
          gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
          gl.useProgram(splatProg); gl.bindVertexArray(splatVao);
          gl.uniformMatrix3fv(uSplat.W, false, cam.W);
          gl.uniform3f(uSplat.camPos, cam.eye[0], cam.eye[1], cam.eye[2]);
          gl.uniform2f(uSplat.focal, focalPx, focalPx);
          gl.uniform2f(uSplat.viewport, w, h);
          gl.uniform2f(uSplat.depthAB, DEPTH_A, DEPTH_B);
          gl.uniform1f(uSplat.scaleMult, st.splatScaleMult);
          gl.uniform1i(uSplat.visMode, clean ? 0 : st.splatVisMode);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, validSortedCount);
        }
      }

      if (!clean) {
        gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(false);
        gl.enable(gl.BLEND); gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        drawGizmos(cam, w, h);
      }
      gl.depthMask(true);
    }

    function captureCleanPng(camOv) {
      const w = Math.max(64, canvas.width), h = Math.max(64, canvas.height);
      renderPass(camOv || computeCam(), w, h, true);
      dirtySort = true;
      return canvas.toDataURL("image/png");
    }

    function setFpsMode(on, reqLock = false) {
      st.fpsMode = !!on;
      canvas.classList.toggle("fps-cursor", st.fpsMode);
      if (st.fpsMode) {
        const ob = computeCam();
        if (!reqLock) {
          st.eye = ob.eye.slice();
          st.yaw = Math.atan2(ob.fx, ob.fy);
          st.pitch = Math.asin(Math.max(-0.98, Math.min(0.98, ob.fz)));
          st.footZ = Math.max(0, st.eye[2] - 1.65);
        }
      } else if (document.pointerLockElement === canvas) {
        document.exitPointerLock();
      }
      el("fps-crosshair-wrap").classList.toggle("hidden", !st.fpsMode);
      el("fps-radar-canvas").classList.toggle("hidden", !st.fpsMode);
      el("btn-splat-fps-toggle").textContent = st.fpsMode ? "🌐 Orbit View (F)" : "🎮 Enter FPS (F)";
      syncViewModeButtons();
      if (st.fpsMode && reqLock) canvas.requestPointerLock?.();
      dirtySort = true;
    }

    function syncHotbar() {
      el("cnt-hb-cam").textContent = String(st.camNodes.length);
      el("cnt-hb-sub").textContent = String(st.subjectNodes.length);
      el("cnt-hb-vox").textContent = String(st.voxels.length);
      el("cnt-hb-ping").textContent = String(st.pings.length);
      el("fps-hotbar").querySelectorAll(".hotbar-slot").forEach((b) => b.classList.toggle("active", Number(b.dataset.slot) === st.slot));
      el("btn-rail-track-sub").textContent = `🎯 Subj: ${st.trackSubject ? "ON" : "OFF"}`;
      if (st.slot === 2 && el("more-splat-crop")) el("more-splat-crop").open = true;
      else if (st.slot === 3 && el("more-splat-paint")) el("more-splat-paint").open = true;
      else if (st.slot === 4 && el("more-splat-cleaner")) el("more-splat-cleaner").open = true;
      else if (st.slot === 5 && el("sec-fps-rail")) el("sec-fps-rail").open = true;
    }

    function teleportPreset(pov) {
      const c = sceneCenter, r = Math.max(3.8, sceneRadius * 1.6);
      st.headYaw = 0; st.headPitch = 0; st.rollDeg = 0; st.adsTarget = null;
      st.orbitTarget = c.slice(); st.orbitDist = r;
      if (pov === "front") { st.eye = [c[0], c[1] - r, 1.65]; st.yaw = 0; st.pitch = -0.05; st.orbitTheta = -Math.PI * 0.5; st.orbitPhi = 0.15; }
      else if (pov === "hero") { st.eye = [c[0] + r * 0.65, c[1] - r * 0.78, 1.85]; st.yaw = Math.atan2(c[0]-st.eye[0], c[1]-st.eye[1]); st.pitch = -0.14; st.orbitTheta = -1.05; st.orbitPhi = 0.28; }
      else if (pov === "right") { st.eye = [c[0] + r, c[1], 1.65]; st.yaw = -Math.PI * 0.5; st.pitch = -0.04; st.orbitTheta = 0; st.orbitPhi = 0.15; }
      else if (pov === "top") { setLocomotion("fly", false); st.eye = [c[0], c[1] - 0.8, r * 1.8]; st.yaw = 0; st.pitch = -1.35; st.orbitPhi = 1.35; }
      else { st.eye = [c[0], c[1] - r * 1.1, Math.max(1.65, c[2] + 0.5)]; st.yaw = 0; st.pitch = -0.08; st.orbitTheta = -1.15; st.orbitPhi = 0.26; }
      st.footZ = Math.max(0, st.eye[2] - 1.65);
      dirtySort = true;
    }

    async function loadScene(choice) {
      if (choice === "__default_cube_solid__" || choice === "__blank_grid__") {
        splatCount = 0; positions = null; flagData = null; undoStack = []; redoStack = [];
        st.activeJobId = ""; st.hasOrigBackup = false;
        st.showSolid = true; el("chk-show-solid").checked = true;
        buildSolidStage(choice === "__default_cube_solid__" ? "default_cube" : "blank_grid");
        sceneCenter = [0, 0, 1.0]; sceneRadius = 3.4;
        teleportPreset("hero");
        updateSplatStatsHud();
        return;
      }
      if (choice === "__courtyard__" || choice === "__default_cube_splat__") {
        const yard = choice === "__courtyard__";
        st.activeJobId = yard ? "__default_courtyard__" : "__default_cube__";
        st.hasOrigBackup = false;
        if (yard) { buildSolidStage("courtyard"); st.showSolid = true; el("chk-show-solid").checked = true; }
        else { baseMeshFloats = new Float32Array(0); baseColliders = []; solidDirty = true; st.showSolid = false; el("chk-show-solid").checked = false; }
        st.showSplats = true; el("chk-show-splats").checked = true;
        const resp = await fetch(withTokenUrl(`/api/fps/default_splat?variant=${yard ? "courtyard" : "cube"}&fmt=splat`));
        if (resp.ok) ingestSplatBuf(await resp.arrayBuffer());
        teleportPreset("hero");
        return;
      }
      if (choice === "__blender_live__") {
        try {
          const res = await apiFetch("/api/fps/blender_scene");
          if (res.vertex_floats?.length) {
            splatCount = 0; positions = null; flagData = null;
            baseMeshFloats = new Float32Array(res.vertex_floats);
            baseColliders = []; solidDirty = true;
            st.showSolid = true; el("chk-show-solid").checked = true;
            showToast(`🟠 Streamed ${res.triangles.toLocaleString()} triangles from Blender 5.3`);
          } else {
            buildSolidStage("default_cube");
            showToast("Blender scene empty — showing Default Cube");
          }
          updateSplatStatsHud();
        } catch (e) { showToast(e.message, true); }
        return;
      }

      // Workspace Splat Job
      try {
        const ens = await apiFetch(`/api/splats/ensure_splat/${encodeURIComponent(choice)}`, { method: "POST", json: {} });
        st.activeJobId = choice;
        st.hasOrigBackup = !!ens.has_orig;
        baseMeshFloats = new Float32Array(0); baseColliders = []; solidDirty = true;
        st.showSplats = true; el("chk-show-splats").checked = true;
        const resp = await fetch(withTokenUrl(ens.splat_url));
        if (resp.ok) ingestSplatBuf(await resp.arrayBuffer());
        teleportPreset("frame");
      } catch (e) { showToast(e.message, true); }
    }

    async function importLocalSplatFile(file) {
      try {
        showToast(`Importing ${file.name}…`);
        const buf = await file.arrayBuffer();
        const resp = await fetch(`/api/splats/upload?filename=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: Object.assign({ "X-Filename": file.name }, token ? { Authorization: `Bearer ${token}` } : {}),
          body: buf,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        state.splatJobs.set(data.job.job_id, data.job);
        await open3DSceneInCenter(data.job.job_id, true);
        showToast(`Imported '${data.job.job_id}' (${data.job.n_gaussians.toLocaleString()} Gaussians)`);
      } catch (e) { showToast(e.message, true); }
    }

    async function saveSplat(syncBl = false) {
      if (!splatCount || !flagData) return showToast("No splat loaded to save!", true);
      const drop = [];
      for (let i = 0; i < splatCount; i++) if (flagData[i] & 1) drop.push(i);
      try {
        showToast(syncBl ? "Saving & importing to Blender 5.3…" : "Saving .ply + .splat…");
        const res = await apiFetch("/api/splats/edit/apply", {
          method: "POST",
          json: {
            job_id: st.activeJobId,
            save_as_name: el("inp-save-name").value.trim() || undefined,
            ops: recordedOps,
            drop_indices: drop,
            authoritative_mask: true,
            loaded_count: splatCount,
            sync_blender: syncBl,
          },
        });
        st.activeJobId = res.job_id;
        st.hasOrigBackup = !!res.has_orig;
        el("inp-save-name").value = "";
        await refreshSplats();
        if (res.splat_url) {
          const sp = await fetch(withTokenUrl(res.splat_url));
          if (sp.ok) ingestSplatBuf(await sp.arrayBuffer());
        }
        showToast(`💾 Saved '${res.job_id}' (${res.n_after.toLocaleString()} Gaussians)`);
      } catch (e) { showToast(e.message, true); }
    }

    function downloadSplatOrPly(asPly) {
      if (!splatCount || !rawSplatBytes || !flagData) return showToast("No splat loaded!", true);
      const kept = [];
      for (let i = 0; i < splatCount; i++) if ((flagData[i] & 1) === 0) kept.push(i);
      if (!kept.length) return;
      const base = (st.activeJobId || "splat").replace(/^__/, "").replace(/__$/, "");
      let blob;
      if (!asPly) {
        const out = new Uint8Array(kept.length * 32);
        kept.forEach((si, di) => out.set(rawSplatBytes.subarray(si * 32, si * 32 + 32), di * 32));
        blob = new Blob([out], { type: "application/octet-stream" });
      } else {
        const props = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"];
        const hdr = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${kept.length}\n${props.map((p) => `property float ${p}`).join("\n")}\nend_header\n`);
        const body = new ArrayBuffer(kept.length * 68), dv = new DataView(body), sf = new Float32Array(rawSplatBytes.buffer), C0 = 0.28209479177387814;
        kept.forEach((si, di) => {
          const o = di * 68, bf = si * 8, bb = si * 32;
          dv.setFloat32(o, sf[bf], true); dv.setFloat32(o+4, sf[bf+1], true); dv.setFloat32(o+8, sf[bf+2], true);
          dv.setFloat32(o+24, ((rawSplatBytes[bb+24]/255)-0.5)/C0, true);
          dv.setFloat32(o+28, ((rawSplatBytes[bb+25]/255)-0.5)/C0, true);
          dv.setFloat32(o+32, ((rawSplatBytes[bb+26]/255)-0.5)/C0, true);
          const a = Math.max(1e-4, Math.min(1-1e-4, rawSplatBytes[bb+27]/255));
          dv.setFloat32(o+36, Math.log(a/(1-a)), true);
          dv.setFloat32(o+40, Math.log(Math.max(1e-6, sf[bf+3])), true);
          dv.setFloat32(o+44, Math.log(Math.max(1e-6, sf[bf+4])), true);
          dv.setFloat32(o+48, Math.log(Math.max(1e-6, sf[bf+5])), true);
          dv.setFloat32(o+52, (rawSplatBytes[bb+28]-127.5)/127.5, true);
          dv.setFloat32(o+56, (rawSplatBytes[bb+29]-127.5)/127.5, true);
          dv.setFloat32(o+60, (rawSplatBytes[bb+30]-127.5)/127.5, true);
          dv.setFloat32(o+64, (rawSplatBytes[bb+31]-127.5)/127.5, true);
        });
        blob = new Blob([hdr, body], { type: "application/octet-stream" });
      }
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${base}.${asPly ? "ply" : "splat"}`;
      a.click();
    }

    // --- Input & Tool Action Bindings ---
    function handleToolDown(btn) {
      const cam = computeCam(), hit = st.lastHit;
      if (btn === 1) {
        const pt = hit ? hit.pt.slice() : [cam.eye[0]+cam.fx*3.5, cam.eye[1]+cam.fy*3.5, 0];
        st.pings.push({ position: pt, camera_eye: cam.eye.slice(), note: `Ping #${st.pings.length+1}`, image_b64: captureCleanPng(cam) });
        syncHotbar(); showToast(`📍 Ping #${st.pings.length} dropped`);
        return;
      }
      if (st.slot === 1) { beginStroke(btn === 2 ? "restore" : "cull"); stepStroke(st.brushCursorPt); return; }
      if (st.slot === 2) {
        if (btn === 2) { st.cropCornerA = null; st.cropBox = null; return; }
        const pt = hit ? hit.pt.slice() : st.brushCursorPt.slice();
        if (!st.cropCornerA) { st.cropCornerA = pt; showToast("📦 Corner 1 set — click Corner 2!"); }
        else {
          const a = st.cropCornerA; st.cropCornerA = null;
          st.cropBox = { min: [Math.min(a[0],pt[0])-0.1, Math.min(a[1],pt[1])-0.1, Math.min(a[2],pt[2])-0.1], max: [Math.max(a[0],pt[0])+0.1, Math.max(a[1],pt[1])+0.1, Math.max(a[2],pt[2])+1.8] };
          showToast("📦 Crop Box set — Press Delete to Cull Inside!");
        }
        return;
      }
      if (st.slot === 3) { beginStroke(btn === 2 ? "deselect" : st.paintSubmode); stepStroke(st.brushCursorPt); return; }
      if (st.slot === 4) { if (btn === 0 && isFilterActive()) el("btn-flt-apply").click(); return; }
      if (btn === 2 && st.slot !== 7) {
        st.adsTarget = hit ? hit.pt.slice() : [cam.eye[0]+cam.fx*4, cam.eye[1]+cam.fy*4, cam.eye[2]+cam.fz*4];
        return;
      }
      if (st.slot === 5 && btn === 0) {
        st.camNodes.push({ position: cam.eye.slice(), forward: [cam.fx, cam.fy, cam.fz], lens_mm: st.lensMm, roll_deg: st.rollDeg });
        syncHotbar(); showToast(`🎥 Cam Node #${st.camNodes.length}`);
      } else if (st.slot === 6 && btn === 0) {
        const pt = hit ? hit.pt.slice() : [cam.eye[0]+cam.fx*3, cam.eye[1]+cam.fy*3, 0];
        st.subjectNodes.push({ position: pt }); syncHotbar(); showToast(`🏃 Subject Waypoint #${st.subjectNodes.length}`);
      } else if (st.slot === 7) {
        if (btn === 2 && hit?.kind === "voxel") { st.voxels.splice(hit.voxelIdx, 1); solidDirty = true; syncHotbar(); }
        else if (btn === 0) {
          const vs = VOXEL_SIZES[st.voxelSizeIdx];
          const bh = hit || { pt: [cam.eye[0]+cam.fx*3.5, cam.eye[1]+cam.fy*3.5, 0], kind: "floor", normal: [0,0,1] };
          const [vr, vg, vb] = hexToRgb01(el("inp-paint-color").value);
          st.voxels.push({ center: snapVoxel(bh, vs), size: [vs, vs, vs], color: [vr, vg, vb, 1] });
          solidDirty = true; syncHotbar();
        }
      } else if (st.slot === 8 && btn === 0) {
        const pt = hit ? hit.pt.slice() : [cam.eye[0]+cam.fx*3.2, cam.eye[1]+cam.fy*3.2, 0];
        st.pendingPing = { position: pt, camera_eye: cam.eye.slice(), image_b64: captureCleanPng(cam) };
        if (document.pointerLockElement === canvas) document.exitPointerLock();
        el("fps-ping-coords").textContent = `📍 Ping #${st.pings.length + 1}`;
        el("fps-ping-modal").classList.remove("hidden");
        el("inp-fps-ping-note").value = ""; el("inp-fps-ping-note").focus();
      }
    }

    canvas.addEventListener("mousedown", (e) => {
      e.preventDefault();
      if (st.fpsMode) {
        if (document.pointerLockElement !== canvas) { canvas.requestPointerLock?.(); return; }
        st.mouseDownBtn = e.button;
        handleToolDown(e.button);
        return;
      }
      st.orbitDragging = true;
      st.orbitPan = e.button === 2 || e.shiftKey || e.button === 1;
      st.lastX = e.clientX; st.lastY = e.clientY;
    });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    window.addEventListener("mouseup", (e) => {
      st.orbitDragging = false;
      if (activeStroke) endStroke();
      if (e.button === 2) st.adsTarget = null;
      st.mouseDownBtn = -1;
    });
    window.addEventListener("mousemove", (e) => {
      if (state.centerKind !== "3d") return;
      if (st.fpsMode && document.pointerLockElement === canvas) {
        if (!st.adsTarget) {
          const alt = st.keys.has("AltLeft") || st.keys.has("AltRight") || st.gimbalLocked;
          const sens = st.binoZoom ? 0.0008 : 0.0023;
          if (alt) { st.headYaw = Math.max(-3.05, Math.min(3.05, st.headYaw + e.movementX * sens)); st.headPitch = Math.max(-1.44, Math.min(1.44, st.headPitch - e.movementY * sens)); }
          else { st.yaw += e.movementX * sens; st.pitch = Math.max(-1.48, Math.min(1.48, st.pitch - e.movementY * sens)); }
          dirtySort = true;
        }
        return;
      }
      if (!st.orbitDragging || st.fpsMode) return;
      const dx = e.clientX - st.lastX, dy = e.clientY - st.lastY;
      st.lastX = e.clientX; st.lastY = e.clientY;
      if (st.orbitPan) {
        const c = computeCam(), s = st.orbitDist * 0.0018;
        st.orbitTarget[0] -= (c.rx*dx - c.ux*dy) * s; st.orbitTarget[1] -= (c.ry*dx - c.uy*dy) * s; st.orbitTarget[2] -= (c.rz*dx - c.uz*dy) * s;
      } else {
        st.orbitTheta -= dx * 0.007;
        st.orbitPhi = Math.max(-1.48, Math.min(1.48, st.orbitPhi + dy * 0.007));
      }
      dirtySort = true;
    });

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (st.fpsMode) {
        if (e.altKey) {
          st.brushReach = Math.max(0.6, Math.min(16, st.brushReach + (e.deltaY < 0 ? 0.4 : -0.4)));
          el("rng-brush-reach").value = String(st.brushReach);
          el("lbl-brush-reach").textContent = `${st.brushReach.toFixed(1)}m`;
          return;
        }
        const ci = PRIME_LENSES.reduce((b, v, idx) => (Math.abs(v - st.lensMm) < Math.abs(PRIME_LENSES[b] - st.lensMm) ? idx : b), 4);
        st.lensMm = PRIME_LENSES[Math.max(0, Math.min(PRIME_LENSES.length - 1, ci + (e.deltaY < 0 ? 1 : -1)))];
        el("sel-lens").value = String(st.lensMm);
        return;
      }
      st.orbitDist = Math.max(0.3, Math.min(220, st.orbitDist * Math.exp(e.deltaY * 0.0012)));
      dirtySort = true;
    }, { passive: false });

    window.addEventListener("keydown", (e) => {
      const tag = document.activeElement?.tagName || "";
      const inInput = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      if ((e.ctrlKey || e.metaKey) && !inInput && state.centerKind === "3d") {
        if (e.code === "KeyZ") { e.preventDefault(); runUndoRedo(!e.shiftKey); return; }
        if (e.code === "KeyY") { e.preventDefault(); runUndoRedo(false); return; }
        if (e.code === "KeyS") { e.preventDefault(); saveSplat(false); return; }
      }
      if (inInput) return;
      if (e.code === "Tab") { e.preventDefault(); toggleZenMode(); return; }
      if (e.code === "KeyH" && state.centerKind === "3d") { e.preventDefault(); el("fps-help-modal").classList.toggle("hidden"); return; }
      if (e.code === "KeyF" && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        if (state.centerKind !== "3d") open3DSceneInCenter(state.active3DId, true);
        else setFpsMode(!st.fpsMode, !st.fpsMode);
        return;
      }
      if (state.centerKind !== "3d" || !st.fpsMode) return;
      if (["AltLeft", "AltRight", "Space"].includes(e.code)) e.preventDefault();

      if (e.code === "Space" && !e.repeat && st.doubleSpaceFly) {
        const now = performance.now();
        if (now - st.lastSpaceMs < 280) {
          st.lastSpaceMs = 0;
          setLocomotion(st.locomotion === "walk" ? "fly" : "walk", true);
          st.keys.add("Space");
          return;
        }
        st.lastSpaceMs = now;
      }
      if (e.repeat && ["KeyT", "KeyG", "KeyZ", "KeyP", "KeyK", "CapsLock"].includes(e.code)) return;
      st.keys.add(e.code);

      if (/^Digit[1-8]$/.test(e.code)) { st.slot = Number(e.code.slice(-1)); syncHotbar(); }
      else if (e.code === "KeyG") setLocomotion(st.locomotion === "walk" ? "fly" : "walk", true);
      else if (e.code === "KeyT") { st.autoRun = !st.autoRun; showToast(st.autoRun ? "🏃 Auto-Run ON" : "Auto-Run OFF"); }
      else if (e.code === "KeyZ" && st.locomotion === "walk") st.prone = !st.prone;
      else if (e.code === "KeyB") st.binoZoom = true;
      else if (e.code === "KeyR" && (st.slot === 1 || st.slot === 3)) {
        st.fixedReach = !st.fixedReach; el("chk-fixed-reach").checked = st.fixedReach;
      } else if (e.code === "KeyP") el("btn-rail-ride").click();
      else if (e.code === "KeyK") el("btn-rail-track-sub").click();
      else if (e.code === "Delete") {
        if (st.slot === 2 && st.cropBox) el("btn-crop-cull-in").click();
        else el("btn-sel-delete").click();
      } else if (e.code === "BracketLeft" || e.code === "BracketRight") {
        const dir = e.code === "BracketRight" ? 1 : -1;
        if (st.slot === 7) { st.voxelSizeIdx = Math.max(0, Math.min(2, st.voxelSizeIdx + dir)); showToast(`Voxel: ${VOXEL_SIZES[st.voxelSizeIdx]}m`); }
        else {
          const ci = BRUSH_STEPS.findIndex((v) => v >= st.brushRadius - 0.01);
          st.brushRadius = BRUSH_STEPS[Math.max(0, Math.min(BRUSH_STEPS.length - 1, (ci < 0 ? 4 : ci) + dir))];
          el("rng-brush-rad").value = String(st.brushRadius); el("lbl-brush-rad").textContent = `${st.brushRadius.toFixed(2)}m`;
        }
      }
    });
    window.addEventListener("keyup", (e) => { if (e.code === "KeyB") st.binoZoom = false; st.keys.delete(e.code); });

    // Bind Inspector & HUD Controls
    el("btn-splat-fps-toggle").addEventListener("click", () => setFpsMode(!st.fpsMode, !st.fpsMode));
    el("btn-splat-reset-cam").addEventListener("click", () => teleportPreset("hero"));
    el("btn-undo").addEventListener("click", () => runUndoRedo(true));
    el("btn-redo").addEventListener("click", () => runUndoRedo(false));
    el("btn-fps-help").addEventListener("click", () => el("fps-help-modal").classList.toggle("hidden"));
    el("btn-close-fps-help").addEventListener("click", () => el("fps-help-modal").classList.add("hidden"));
    el("fps-hotbar").addEventListener("click", (e) => {
      const b = e.target.closest(".hotbar-slot");
      if (b?.dataset.slot) { st.slot = Number(b.dataset.slot); syncHotbar(); }
    });

    ["inp-cam-x", "inp-cam-y", "inp-cam-z", "inp-cam-yaw", "inp-cam-pitch", "inp-cam-roll"].forEach((id) => {
      el(id).addEventListener("change", () => {
        st.eye = [Number(el("inp-cam-x").value || 0), Number(el("inp-cam-y").value || 0), Number(el("inp-cam-z").value || 1.65)];
        st.yaw = (Number(el("inp-cam-yaw").value || 0) * Math.PI) / 180;
        st.pitch = Math.max(-1.48, Math.min(1.48, (Number(el("inp-cam-pitch").value || 0) * Math.PI) / 180));
        st.rollDeg = Number(el("inp-cam-roll").value || 0);
        dirtySort = true;
      });
    });
    el("sel-lens").addEventListener("change", (e) => { st.lensMm = Number(e.target.value || 35); });
    el("rng-speed").addEventListener("input", (e) => { st.speedMult = Number(e.target.value || 1); el("lbl-speed").textContent = `${st.speedMult.toFixed(1)}×`; });
    el("btn-mode-walk").addEventListener("click", () => setLocomotion("walk", true));
    el("btn-mode-fly").addEventListener("click", () => setLocomotion("fly", true));
    el("btn-mode-freecam").addEventListener("click", () => setLocomotion("freecam", true));
    el("chk-double-space").addEventListener("change", (e) => { st.doubleSpaceFly = e.target.checked; });
    el("chk-auto-land").addEventListener("change", (e) => { st.autoLand = e.target.checked; });
    el("pov-presets").addEventListener("click", (e) => { const b = e.target.closest("[data-pov]"); if (b) teleportPreset(b.dataset.pov); });
    el("btn-copy-pose").addEventListener("click", () => {
      const c = computeCam();
      navigator.clipboard?.writeText(JSON.stringify({ eye: c.eye, forward: [c.fx, c.fy, c.fz], lens_mm: c.lensMm }, null, 2));
      showToast("Copied Camera Pose JSON");
    });

    el("btn-push-bl-cam").addEventListener("click", async () => {
      const c = computeCam();
      await apiFetch("/api/fps/camera_sync", { method: "POST", json: { eye: c.eye, forward: [c.fx, c.fy, c.fz], roll_deg: st.rollDeg, lens_mm: c.lensMm } })
        .then(() => showToast("🎥 Synced Camera Pose → Blender 5.3!"))
        .catch((e) => showToast(e.message, true));
    });
    el("btn-pull-bl-cam").addEventListener("click", async () => {
      await apiFetch("/api/fps/camera_pull")
        .then((r) => { st.eye = r.eye.slice(); st.yaw = r.yaw_rad; st.pitch = r.pitch_rad; st.lensMm = Math.round(r.lens_mm || 35); setFpsMode(true, false); dirtySort = true; showToast(`📥 Pulled '${r.camera}'`); })
        .catch((e) => showToast(e.message, true));
    });
    el("btn-spawn-bl-cube").addEventListener("click", () => apiFetch("/api/fps/spawn_default_cube_blender", { method: "POST", json: {} }).then(() => showToast("🧊 Spawned Default Cube in Blender 5.3!")).catch((e) => showToast(e.message, true)));
    el("btn-pull-bl-mesh").addEventListener("click", () => open3DSceneInCenter("__blender_live__", true));
    el("chk-live-link-bl").addEventListener("change", (e) => { st.liveLinkBlender = e.target.checked; });

    // Rail & Shot Export
    el("btn-rail-add-node").addEventListener("click", () => {
      const c = computeCam();
      st.camNodes.push({ position: c.eye.slice(), forward: [c.fx, c.fy, c.fz], lens_mm: st.lensMm, roll_deg: st.rollDeg });
      syncHotbar(); showToast(`🎥 Added Cam Node #${st.camNodes.length}`);
    });
    el("btn-rail-ride").addEventListener("click", () => {
      if (!st.camNodes.length) return showToast("Add Cam Nodes first!", true);
      st.riding = !st.riding; st.rideStartT = performance.now();
      el("btn-rail-ride").textContent = st.riding ? "⏹ Stop" : "▶ Ride (P)";
    });
    el("btn-rail-track-sub").addEventListener("click", () => { st.trackSubject = !st.trackSubject; syncHotbar(); });
    el("btn-rail-clear").addEventListener("click", () => { st.camNodes = []; st.subjectNodes = []; st.voxels = []; st.pings = []; st.riding = false; solidDirty = true; syncHotbar(); });

    function shotPayload(extra = {}) {
      const nodes = st.camNodes.length ? st.camNodes : [(() => { const c = computeCam(); return { position: c.eye, forward: [c.fx, c.fy, c.fz], lens_mm: st.lensMm, roll_deg: st.rollDeg }; })()];
      const realJob = st.activeJobId && !st.activeJobId.startsWith("__") ? st.activeJobId : null;
      return Object.assign({
        splat_job_id: realJob, cam_nodes: nodes, subject_nodes: st.subjectNodes, track_subject: st.trackSubject,
        voxels: st.voxels, pings: st.pings, grounded_walk: st.locomotion === "walk",
        duration_sec: Math.max(2, Number(el("inp-rail-dur").value || 5)), fps: 24, width: 1280, height: 720,
        first_frame_b64: captureCleanPng(evalRailCam(0)),
        last_frame_b64: nodes.length > 1 ? captureCleanPng(evalRailCam(1)) : null,
      }, extra);
    }

    el("btn-rail-to-seedance").addEventListener("click", async () => {
      try {
        const res = await apiFetch("/api/shots/export", { method: "POST", json: shotPayload() });
        await refreshMedia();
        await refreshShotsList();
        if (res.first_frame_rel) state.firstFrame = { path: res.first_frame_rel, name: "first_frame.png", kind: "image", url: res.first_frame_url };
        if (res.last_frame_rel) state.lastFrame = { path: res.last_frame_rel, name: "last_frame.png", kind: "image", url: res.last_frame_url };
        if (res.brief?.optical_prompt) el("input-prompt").value = res.brief.optical_prompt;
        renderConditioningSlots(); updatePricingAndInspector();
        switchInspectorTab("seedance");
        showToast(`Loaded Rail '${res.shot_id}' into Seedance 2.5!`);
      } catch (e) { showToast(e.message, true); }
    });
    el("btn-rail-farm-modal").addEventListener("click", async () => {
      if (!st.activeJobId || st.activeJobId.startsWith("__")) return showToast("Save Splat first to farm Modal render!", true);
      try {
        const res = await apiFetch("/api/shots/export", { method: "POST", json: shotPayload({ action: "render_modal" }) });
        await refreshMedia();
        if (res.modal_render_url) openMediaInCenter({ path: res.modal_render_rel, name: `${res.shot_id}.mp4`, kind: "video", url: res.modal_render_url });
      } catch (e) { showToast(e.message, true); }
    });
    el("btn-rail-sync-bl").addEventListener("click", async () => {
      try {
        const res = await apiFetch("/api/shots/export", { method: "POST", json: shotPayload({ action: "sync_blender", send_brief_to_agent: st.pings.length > 0 }) });
        showToast(`Synced DirectorRailCam + ${res.brief_rel} → Blender 5.3!`);
      } catch (e) { showToast(e.message, true); }
    });

    // Splat Cull / Crop / Filter Bindings
    el("sel-splat-vis").addEventListener("change", (e) => { st.splatVisMode = Number(e.target.value || 0); });
    el("rng-splat-scale").addEventListener("input", (e) => { st.splatScaleMult = Number(e.target.value || 1); el("lbl-splat-scale").textContent = `${st.splatScaleMult.toFixed(2)}×`; });
    el("chk-show-splats").addEventListener("change", (e) => { st.showSplats = e.target.checked; });
    el("chk-show-solid").addEventListener("change", (e) => { st.showSolid = e.target.checked; });
    el("chk-show-grid").addEventListener("change", (e) => { st.showGrid = e.target.checked; });
    el("chk-show-gizmos").addEventListener("change", (e) => { st.showGizmos = e.target.checked; });
    el("btn-reset-splat-edits").addEventListener("click", () => mutateFlags((i) => (flagData[i] & 3) !== 0, (f) => f & ~3, null, "↺ Unculled all"));
    el("rng-brush-rad").addEventListener("input", (e) => { st.brushRadius = Number(e.target.value); el("lbl-brush-rad").textContent = `${st.brushRadius.toFixed(2)}m`; });
    el("rng-brush-reach").addEventListener("input", (e) => { st.brushReach = Number(e.target.value); el("lbl-brush-reach").textContent = `${st.brushReach.toFixed(1)}m`; });
    el("chk-fixed-reach").addEventListener("change", (e) => { st.fixedReach = e.target.checked; });

    el("paint-submodes").addEventListener("click", (e) => {
      const b = e.target.closest("[data-pmode]"); if (!b) return;
      st.paintSubmode = b.dataset.pmode;
      el("paint-submodes").querySelectorAll(".mode-pill").forEach((x) => x.classList.toggle("active", x === b));
      st.slot = 3; syncHotbar();
    });
    el("btn-sel-delete").addEventListener("click", () => mutateFlags((i) => (flagData[i]&2)!==0 && (flagData[i]&1)===0, (f) => (f|1)&~2, null, "🔥 Culled selected"));
    el("btn-sel-keep").addEventListener("click", () => mutateFlags((i) => (flagData[i]&1)===0 && (flagData[i]&2)===0, (f) => f|1, null, "🎯 Kept selected"));
    el("btn-sel-invert").addEventListener("click", () => { if (!flagData) return; for (let i = 0; i < splatCount; i++) if ((flagData[i]&1)===0) flagData[i] ^= 2; uploadFlags(); updateSplatStatsHud(); });

    el("btn-crop-here").addEventListener("click", () => {
      const c = st.lastHit ? st.lastHit.pt : sceneCenter;
      st.cropBox = { min: [c[0]-1.5, c[1]-1.5, Math.max(-0.2, c[2]-1.0)], max: [c[0]+1.5, c[1]+1.5, c[2]+1.6] };
      st.slot = 2; syncHotbar(); showToast("📦 Spawned 3D Crop Box");
    });
    el("btn-crop-clear").addEventListener("click", () => { st.cropBox = null; st.cropCornerA = null; });
    el("btn-crop-cull-in").addEventListener("click", () => {
      if (!st.cropBox) return showToast("Spawn Crop Box first!", true);
      const lo = st.cropBox.min, hi = st.cropBox.max;
      mutateFlags((i) => (flagData[i]&1)===0 && positions[3*i]>=lo[0] && positions[3*i]<=hi[0] && positions[3*i+1]>=lo[1] && positions[3*i+1]<=hi[1] && positions[3*i+2]>=lo[2] && positions[3*i+2]<=hi[2], (f) => (f|1)&~2, { op: "aabb_cull", bounds_min: lo, bounds_max: hi }, "🔥 Culled inside box");
    });
    el("btn-crop-keep-in").addEventListener("click", () => {
      if (!st.cropBox) return showToast("Spawn Crop Box first!", true);
      const lo = st.cropBox.min, hi = st.cropBox.max;
      mutateFlags((i) => (flagData[i]&1)===0 && !(positions[3*i]>=lo[0] && positions[3*i]<=hi[0] && positions[3*i+1]>=lo[1] && positions[3*i+1]<=hi[1] && positions[3*i+2]>=lo[2] && positions[3*i+2]<=hi[2]), (f) => (f|1)&~2, { op: "aabb_keep", bounds_min: lo, bounds_max: hi }, "🎯 Kept inside box");
    });

    const syncFilterSliders = () => {
      st.fltMinOpa = Number(el("rng-flt-opa").value || 0);
      st.fltMaxScale = Number(el("rng-flt-scale").value || 1.5);
      st.fltMaxAniso = Number(el("rng-flt-aniso").value || 30);
      st.fltMinZ = Number(el("rng-flt-minz").value || -3);
      el("lbl-flt-opa").textContent = st.fltMinOpa > 0.005 ? st.fltMinOpa.toFixed(2) : "OFF";
      el("lbl-flt-scale").textContent = st.fltMaxScale < 1.48 ? `${st.fltMaxScale.toFixed(2)}m` : "OFF";
      el("lbl-flt-aniso").textContent = st.fltMaxAniso < 29.5 ? `${st.fltMaxAniso.toFixed(0)}×` : "OFF";
      el("lbl-flt-minz").textContent = st.fltMinZ > -2.95 ? `${st.fltMinZ.toFixed(2)}m` : "OFF";
      updateFilterPreview();
    };
    ["rng-flt-opa", "rng-flt-scale", "rng-flt-aniso", "rng-flt-minz"].forEach((id) => el(id).addEventListener("input", syncFilterSliders));
    el("btn-flt-reset").addEventListener("click", () => {
      el("rng-flt-opa").value = "0"; el("rng-flt-scale").value = "1.5"; el("rng-flt-aniso").value = "30"; el("rng-flt-minz").value = "-3";
      syncFilterSliders();
    });
    el("btn-flt-apply").addEventListener("click", () => {
      if (!isFilterActive()) return showToast("Adjust a filter slider first!");
      const op = { op: "filter_prune" };
      if (st.fltMinOpa > 0.005) op.min_opacity = st.fltMinOpa;
      if (st.fltMaxScale < 1.48) op.max_world_scale = st.fltMaxScale;
      if (st.fltMaxAniso < 29.5) op.max_anisotropy = st.fltMaxAniso;
      if (st.fltMinZ > -2.95) op.min_z = st.fltMinZ;
      mutateFlags((i) => matchesFilter(i), (f) => (f|1)&~6, op, "🪄 Pruned floaters");
      el("btn-flt-reset").click();
    });

    el("btn-save-splat").addEventListener("click", () => saveSplat(false));
    el("btn-save-sync-bl").addEventListener("click", () => saveSplat(true));
    el("btn-dl-splat").addEventListener("click", () => downloadSplatOrPly(false));
    el("btn-dl-ply").addEventListener("click", () => downloadSplatOrPly(true));
    el("btn-restore-orig").addEventListener("click", async () => {
      if (!st.activeJobId || st.activeJobId.startsWith("__")) return;
      try {
        const r = await apiFetch(`/api/splats/edit/restore/${encodeURIComponent(st.activeJobId)}`, { method: "POST", json: {} });
        const sp = await fetch(withTokenUrl(r.splat_url));
        if (sp.ok) ingestSplatBuf(await sp.arrayBuffer());
        showToast("↺ Restored original unedited splat");
      } catch (e) { showToast(e.message, true); }
    });

    // Ping Modal
    const savePing = () => {
      if (!st.pendingPing) return;
      st.pings.push(Object.assign({}, st.pendingPing, { note: el("inp-fps-ping-note").value.trim() || `Ping #${st.pings.length + 1}` }));
      st.pendingPing = null; el("fps-ping-modal").classList.add("hidden"); syncHotbar();
    };
    el("btn-fps-ping-save").addEventListener("click", savePing);
    el("btn-fps-ping-cancel").addEventListener("click", () => { st.pendingPing = null; el("fps-ping-modal").classList.add("hidden"); });
    el("inp-fps-ping-note").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); savePing(); } });

    // Main Render Loop
    let lastT = performance.now();
    function frame(now = performance.now()) {
      const dt = Math.min(0.08, Math.max(0.001, (now - lastT) / 1000));
      lastT = now;
      if (state.centerKind === "3d") {
        stepPhysics(dt);
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(64, Math.floor(canvas.clientWidth * dpr));
        const h = Math.max(64, Math.floor(canvas.clientHeight * dpr));
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

        const cam = computeCam();
        st.lastHit = raycast(cam.eye, [cam.fx, cam.fy, cam.fz], st.slot === 1 && st.mouseDownBtn === 2);
        st.brushCursorPt = (st.fixedReach || !st.lastHit)
          ? [cam.eye[0] + cam.fx * st.brushReach, cam.eye[1] + cam.fy * st.brushReach, cam.eye[2] + cam.fz * st.brushReach]
          : st.lastHit.pt.slice();
        if (activeStroke && st.brushCursorPt) stepStroke(st.brushCursorPt);

        if (st.fpsMode) {
          const locked = document.pointerLockElement === canvas;
          el("fps-crosshair").classList.toggle("hit", !!st.lastHit);
          const ring = el("fps-brush-ring");
          if ((st.slot === 1 || st.slot === 3) && st.brushCursorPt) {
            const dist = Math.max(0.25, Math.hypot(st.brushCursorPt[0]-cam.eye[0], st.brushCursorPt[1]-cam.eye[1], st.brushCursorPt[2]-cam.eye[2]));
            const focalCss = ((cam.lensMm || 35) / 36.0) * Math.max(canvas.clientWidth, canvas.clientHeight);
            const diam = Math.max(14, Math.min(340, (2 * st.brushRadius * focalCss) / dist));
            ring.style.width = `${diam.toFixed(0)}px`; ring.style.height = `${diam.toFixed(0)}px`;
            ring.classList.toggle("cull", st.slot === 1 && st.mouseDownBtn !== 2);
            ring.classList.remove("hidden");
          } else ring.classList.add("hidden");

          el("fps-reticle-tag").textContent = locked
            ? `${Math.round(cam.lensMm)}mm · ${st.lastHit ? `${st.lastHit.kind} ${st.lastHit.dist.toFixed(1)}m` : `reach ${st.brushReach.toFixed(1)}m`} · r=${st.brushRadius.toFixed(2)}m`
            : `Click viewport to lock mouse · Space×2 = Creative Fly · 1-8 Hotbar`;
          drawRadar(cam);
        }

        const ae = document.activeElement;
        const setInp = (id, v) => { const n = el(id); if (n && ae !== n) n.value = v; };
        setInp("inp-cam-x", cam.eye[0].toFixed(2));
        setInp("inp-cam-y", cam.eye[1].toFixed(2));
        setInp("inp-cam-z", cam.eye[2].toFixed(2));
        setInp("inp-cam-yaw", ((Math.atan2(cam.fx, cam.fy) * 180) / Math.PI).toFixed(1));
        setInp("inp-cam-pitch", ((Math.asin(Math.max(-1, Math.min(1, cam.fz))) * 180) / Math.PI).toFixed(1));
        setInp("inp-cam-roll", st.rollDeg.toFixed(1));

        renderPass(cam, w, h, false);

        if (st.liveLinkBlender && !st.blSyncBusy && now - st.lastBlSyncMs > 140) {
          const sig = `${cam.eye.map((v) => v.toFixed(2)).join(",")}:${cam.fx.toFixed(2)},${cam.fy.toFixed(2)},${cam.fz.toFixed(2)}`;
          if (sig !== st._lastSig) {
            st._lastSig = sig; st.lastBlSyncMs = now; st.blSyncBusy = true;
            apiFetch("/api/fps/camera_sync", { method: "POST", json: { eye: cam.eye, forward: [cam.fx, cam.fy, cam.fz], roll_deg: st.rollDeg, lens_mm: cam.lensMm } })
              .catch(() => {}).finally(() => { st.blSyncBusy = false; });
          }
        }
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    return {
      loadScene,
      setFpsMode,
      isFps: () => st.fpsMode,
      captureCleanPng,
      saveSplat,
      importLocalSplatFile,
    };
  }

  // --- Boot Studio ---
  (async function boot() {
    renderPromptHistoryDropdown();
    populatePricingDropdowns();
    populateSeedreamModels();
    renderConditioningSlots();
    renderSeedreamRefs();
    updatePricingAndInspector();
    updateSeedreamInspector();
    updateSplatCostEstimate();
    switchInspectorTab(state.inspectorTab);

    window._bbEngine = createUnified3DEngine(el("theater-splat-canvas"));
    await Promise.all([refreshState(), refreshMedia(), refreshSplats(), refreshShotsList()]);
    refreshModels();
    connectSSE();

    let start3D = "__courtyard__";
    if (initialSceneParam) {
      if (initialSceneParam === "cube") start3D = "__default_cube_solid__";
      else if (initialSceneParam === "splat_cube") start3D = "__default_cube_splat__";
      else if (initialSceneParam === "blender") start3D = "__blender_live__";
      else if (state.splatJobs.has(initialSceneParam)) start3D = initialSceneParam;
    }
    await open3DSceneInCenter(start3D, bootInFps);
  })();
})();