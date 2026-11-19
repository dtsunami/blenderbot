(() => {
  // --- Auth Token Bootstrap ---
  const urlParams = new URLSearchParams(window.location.search);
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
    if (!resp.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    return data;
  }

  const MODE_SUFFIXES = {
    standard: "",
    orbit_plate:
      " Smooth steady camera arc with clear multi-view parallax, sharp 35mm deep-focus optics, consistent scene lighting and exposure, zero motion blur, no jump cuts.",
    subject_4d:
      " Coherent continuous subject motion, crisp silhouette separation, steady camera framing, consistent lighting, no abrupt cuts.",
  };

  const DURATION_CHOICES = [
    { val: "", label: "Auto (match source video / 5s)", sec: null },
    { val: "-1", label: "Model Auto (-1)", sec: 5 },
    { val: "4", label: "4 sec", sec: 4 },
    { val: "5", label: "5 sec", sec: 5, defaultSelected: true },
    { val: "6", label: "6 sec", sec: 6 },
    { val: "8", label: "8 sec", sec: 8 },
    { val: "10", label: "10 sec", sec: 10 },
    { val: "12", label: "12 sec", sec: 12 },
    { val: "15", label: "15 sec", sec: 15 },
    { val: "20", label: "20 sec", sec: 20 },
    { val: "25", label: "25 sec", sec: 25 },
    { val: "30", label: "30 sec", sec: 30 },
  ];

  const SIZE_CHOICES = [
    { val: "", label: "Auto (match source video / 1280x720)" },
    { val: "1280x720", label: "1280x720 (720p 16:9)", defaultSelected: true },
    { val: "1920x1080", label: "1920x1080 (1080p 16:9)" },
    { val: "854x480", label: "854x480 (480p 16:9)" },
    { val: "2520x1080", label: "2520x1080 (Cinema 21:9)" },
    { val: "1470x630", label: "1470x630 (720p 21:9)" },
    { val: "1080x1080", label: "1080x1080 (1:1 Square)" },
    { val: "960x960", label: "960x960 (1:1 Medium)" },
    { val: "1080x1920", label: "1080x1920 (1080p 9:16 Vertical)" },
    { val: "720x1280", label: "720x1280 (720p 9:16 Vertical)" },
    { val: "1440x1080", label: "1440x1080 (1080p 4:3)" },
    { val: "1080x1440", label: "1080x1440 (1080p 3:4)" },
  ];

  // --- Studio State ---
  const state = {
    mediaItems: [],
    videoModels: [{ id: "bytedance/seedance-2.5", name: "ByteDance: Seedance 2.5", cost_per_sec_usd: 0.1028 }],
    filter: "all",
    search: "",
    mode: "standard",
    costPerSec: 0.1028,
    sourceVideo: null,       // Source Video 1 item
    sourceVideoProbe: null,  // probed duration/width/height for Video 1
    sourceVideo2: null,      // Source Video 2 item (optional 2nd source video!)
    sourceVideo2Probe: null, // probed duration/width/height for Video 2
    dualVideoMode: "bridge_v1_to_v2",
    refVideos: [],           // 3rd+ extra reference video items (`input_references`)
    firstFrame: null,       // explicit first_frame image item
    lastFrame: null,        // explicit last_frame image item
    refImages: [],          // explicit input_references image items
    refAudio: null,         // explicit input_references audio item
    stageItem: null,
    activeSplatJob: null,
    studioTab: window.location.hash === "#splat" ? "splat" : "seedance",
    splatSourceMode: "video",
    splatJobs: new Map(),
    jobs: new Map(),
    promptHistory: JSON.parse(localStorage.getItem("bb_seedance_prompts") || "[]"),
    agentPricing: { input_per_1m_usd: null, output_per_1m_usd: null },
    ledger: null,
    sessionCost: 0.0,
    liveAgentNode: null,
  };

  const el = (id) => document.getElementById(id);
  const pillBlender = el("pill-blender");
  const btnOrKey = el("btn-or-key");
  const pillHealth = el("pill-health");
  const pillCost = el("pill-cost");
  const keyDrawer = el("key-drawer");
  const inputOrKey = el("input-or-key");
  const mediaGrid = el("media-grid");
  const jobsList = el("jobs-list");
  const agentFeed = el("agent-feed");
  const toastEl = el("composer-toast");

  function showToast(msg, isErr = false) {
    toastEl.textContent = msg;
    toastEl.classList.toggle("err", isErr);
    toastEl.classList.remove("hidden");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toastEl.classList.add("hidden"), 6000);
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

  // --- Prompt History Manager (Up/Down Arrow + Dropdown) ---
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
    sel.innerHTML = `<option value="">🕒 Recent prompts (${state.promptHistory.length})…</option>`;
    [...state.promptHistory].reverse().forEach((p, idx) => {
      const opt = document.createElement("option");
      opt.value = String(state.promptHistory.length - 1 - idx);
      opt.textContent = p.length > 68 ? p.slice(0, 65) + "…" : p;
      sel.appendChild(opt);
    });
  }

  el("sel-prompt-history").addEventListener("change", (e) => {
    const idxStr = e.target.value;
    if (idxStr === "") return;
    const chosen = state.promptHistory[Number(idxStr)];
    if (chosen) {
      el("input-prompt").value = chosen;
      updatePricingAndInspector();
      el("input-prompt").focus();
    }
    e.target.value = "";
  });

  function attachArrowHistory(inputEl, isTextarea, badgeEl) {
    let cursor = -1; // -1 means sitting at live draft
    let draft = "";

    function updateBadge() {
      if (!badgeEl) return;
      if (cursor >= 0 && state.promptHistory.length > 0) {
        badgeEl.textContent = `↑/↓ ${cursor + 1}/${state.promptHistory.length}`;
      } else {
        badgeEl.textContent = "↑/↓ history";
      }
    }

    inputEl.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const hist = state.promptHistory;
      if (!hist.length) return;

      if (e.key === "ArrowUp") {
        if (isTextarea) {
          const beforeCursor = inputEl.value.slice(0, inputEl.selectionStart || 0);
          const onFirstLine = !beforeCursor.includes("\n");
          if (!onFirstLine && cursor === -1) return;
        }
        e.preventDefault();
        if (cursor === -1) {
          draft = inputEl.value;
          cursor = hist.length - 1;
        } else if (cursor > 0) {
          cursor -= 1;
        }
        inputEl.value = hist[cursor];
        inputEl.setSelectionRange(inputEl.value.length, inputEl.value.length);
        updateBadge();
        if (isTextarea) updatePricingAndInspector();
      } else if (e.key === "ArrowDown") {
        if (cursor === -1) return;
        if (isTextarea) {
          const afterCursor = inputEl.value.slice(inputEl.selectionEnd || 0);
          const onLastLine = !afterCursor.includes("\n");
          if (!onLastLine) return;
        }
        e.preventDefault();
        if (cursor < hist.length - 1) {
          cursor += 1;
          inputEl.value = hist[cursor];
        } else {
          cursor = -1;
          inputEl.value = draft;
        }
        inputEl.setSelectionRange(inputEl.value.length, inputEl.value.length);
        updateBadge();
        if (isTextarea) updatePricingAndInspector();
      } else if (e.key.length === 1 || e.key === "Backspace" || e.key === "Delete") {
        cursor = -1;
        updateBadge();
      }
    });
  }

  attachArrowHistory(el("input-prompt"), true, el("prompt-hist-badge"));
  attachArrowHistory(el("agent-input"), false, null);

  // --- Dynamic Dropdown Pricing & Live Outbound Context Inspector ---
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
      const effS =
        c.sec !== null
          ? c.sec
          : state.sourceVideoProbe && state.sourceVideoProbe.duration_sec
          ? Math.max(4, Math.min(30, Math.round(state.sourceVideoProbe.duration_sec)))
          : 5;
      const costStr = `$${(effS * rate).toFixed(2)}`;
      opt.textContent =
        c.val === "" ? `Auto (${effS}s) — Est. Output ~${costStr}` : `${c.label} — Output ~${costStr}`;
      if (c.val === prevDur) opt.selected = true;
      durSel.appendChild(opt);
    });
    durSel.dataset.initialized = "1";

    const secNow = activeDurationSec();
    const curEstStr = `$${(secNow * rate).toFixed(2)}`;

    const sizeSel = el("sel-size");
    const prevSize = sizeSel.dataset.initialized ? sizeSel.value : "1280x720";
    sizeSel.innerHTML = "";
    SIZE_CHOICES.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.val;
      opt.textContent = `${s.label} — Output ~${curEstStr} (${secNow}s)`;
      if (s.val === prevSize) opt.selected = true;
      sizeSel.appendChild(opt);
    });
    sizeSel.dataset.initialized = "1";

    // Also update Model dropdown option labels so every model shows rate + output cost!
    const modSel = el("select-video-model");
    const curMod = modSel.value || "bytedance/seedance-2.5";
    modSel.innerHTML = "";
    state.videoModels.forEach((m) => {
      const r = m.cost_per_sec_usd || state.costPerSec || 0.1028;
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.id} · $${r.toFixed(4)}/s (Output ~$${(r * secNow).toFixed(2)} / ${secNow}s)`;
      if (m.id === curMod) opt.selected = true;
      modSel.appendChild(opt);
    });
  }

  function updatePricingAndInspector() {
    populatePricingDropdowns();
    const sec = activeDurationSec();
    const rate = activeRatePerSec();
    const estUsd = sec * rate;

    el("est-cost-label").textContent = `Est. Output: ~$${estUsd.toFixed(3)} USD (${sec}s @ $${rate.toFixed(4)}/s)`;
    el("inspector-cost-pill").textContent = `Est. Output: ~$${estUsd.toFixed(4)}`;

    // Build exact outbound context breakdown
    const modelId = el("select-video-model").value || "bytedance/seedance-2.5";
    const sizeVal = el("sel-size").value || "auto (from source video / 1280x720)";
    const rawPrompt = el("input-prompt").value.trim();
    const suffix = MODE_SUFFIXES[state.mode] || "";
    const compiledPrompt = rawPrompt ? `${rawPrompt}${suffix}` : "";
    const tokEst = Math.max(0, Math.ceil(compiledPrompt.length / 4));

    const bothVids = !!(state.sourceVideo && state.sourceVideo2);
    const dMode = el("sel-dual-mode").value || "bridge_v1_to_v2";
    state.dualVideoMode = dMode;

    let firstDesc = "none";
    if (state.firstFrame) {
      firstDesc = `🖼 ${state.firstFrame.path} (explicit PNG/JPG override)`;
    } else if (bothVids && dMode === "bridge_v1_to_v2") {
      firstDesc = `🎬 Video 1 END frame from ${state.sourceVideo.path}`;
    } else if (state.sourceVideo && (bothVids || el("chk-vid-first").checked)) {
      firstDesc = `🎬 Video 1 START frame (t=0.0s) from ${state.sourceVideo.path}`;
    } else if (!state.sourceVideo && state.sourceVideo2) {
      firstDesc = `🎬 Video 2 START frame (t=0.0s) from ${state.sourceVideo2.path}`;
    }

    let lastDesc = "none";
    if (state.lastFrame) {
      lastDesc = `🖼 ${state.lastFrame.path} (explicit PNG/JPG override)`;
    } else if (bothVids && dMode === "bridge_v1_to_v2") {
      lastDesc = `🎬 Video 2 START frame (t=0.0s) from ${state.sourceVideo2.path}`;
    } else if (bothVids && dMode === "span_v1_to_v2") {
      lastDesc = `🎬 Video 2 END frame from ${state.sourceVideo2.path}`;
    } else if (state.sourceVideo && el("chk-vid-last").checked) {
      lastDesc = `🎬 Video 1 END frame from ${state.sourceVideo.path}`;
    } else if (!state.sourceVideo && state.sourceVideo2) {
      lastDesc = `🎬 Video 2 END frame from ${state.sourceVideo2.path}`;
    }

    const refParts = [];
    const nMid = Number(el("sel-vid-mids").value || 0);
    if (state.sourceVideo && nMid > 0) {
      refParts.push(`${nMid} mid-keyframes from ${state.sourceVideo.name}`);
    }
    if (state.sourceVideo2 && Math.max(1, nMid) > 0) {
      refParts.push(`${Math.max(1, nMid)} mid-keyframes from ${state.sourceVideo2.name}`);
    }
    if ((state.sourceVideo || state.sourceVideo2) && el("chk-vid-audio").checked && !state.refAudio) {
      const audSrc = (state.sourceVideo || state.sourceVideo2).name;
      refParts.push(`MP3 audio from ${audSrc}`);
    }
    if (state.refVideos.length > 0) {
      refParts.push(
        `${state.refVideos.length * 2} keyframes from extra video(s): ${state.refVideos.map((v) => v.name).join(", ")}`
      );
    }
    if (state.refImages.length > 0) {
      refParts.push(`${state.refImages.length} ref image(s): ${state.refImages.map((i) => i.name).join(", ")}`);
    }
    if (state.refAudio) {
      refParts.push(`🔊 audio: ${state.refAudio.name}`);
    }

    const promptPreview = compiledPrompt
      ? `"${compiledPrompt.length > 140 ? compiledPrompt.slice(0, 137) + "…" : compiledPrompt}" (${compiledPrompt.length} chars, ~${tokEst} tok)`
      : "(empty — frame_images required)";

    el("inspector-body").innerHTML = `
      <div class="insp-row"><span class="insp-key">endpoint:</span><span class="insp-val">POST /api/v1/videos · <b>${escapeHtml(modelId)}</b> · ${sec}s · ${escapeHtml(sizeVal)}</span></div>
      <div class="insp-row"><span class="insp-key">first_frame:</span><span class="insp-val ${firstDesc !== "none" ? "ok" : ""}">${escapeHtml(firstDesc)}</span></div>
      <div class="insp-row"><span class="insp-key">last_frame:</span><span class="insp-val ${lastDesc !== "none" ? "ok" : ""}">${escapeHtml(lastDesc)}</span></div>
      <div class="insp-row"><span class="insp-key">references:</span><span class="insp-val ${refParts.length ? "hi" : ""}">${refParts.length ? escapeHtml(refParts.join(" + ")) : "none"}</span></div>
      <div class="insp-row"><span class="insp-key">prompt (${escapeHtml(state.mode)}):</span><span class="insp-val">${escapeHtml(promptPreview)}</span></div>
    `;
  }

  ["sel-duration", "sel-size", "select-video-model", "chk-vid-first", "chk-vid-last", "sel-vid-mids", "chk-vid-audio", "chk-gen-audio"].forEach((id) => {
    el(id).addEventListener("change", updatePricingAndInspector);
  });
  el("input-prompt").addEventListener("input", updatePricingAndInspector);

  // --- Agent LLM Context Bar & Segment Drawer ---
  function renderAgentContextBar() {
    const lg = state.ledger;
    if (!lg) return;
    const mediaSegs = (lg.segments || []).filter((s) => s.type === "media");
    const inRate = state.agentPricing.input_per_1m_usd;
    const outRate = state.agentPricing.output_per_1m_usd;
    const estTurnIn = inRate ? ` · ~$${((lg.total_tokens / 1_000_000) * inRate).toFixed(4)} in` : "";
    const outRateLbl = outRate ? ` ($${outRate}/1M out)` : "";

    el("agent-ctx-summary").textContent =
      `LLM Context: ${lg.total_tokens.toLocaleString()} / ${(lg.limit / 1000).toFixed(0)}k tok (${lg.pct}%) · ${mediaSegs.length} media${estTurnIn}${outRateLbl}`;

    const drawer = el("agent-ctx-drawer");
    const rows = [
      `<div class="ctx-seg-row"><span>[sys] system + Blender grounding</span><span>${lg.system_tokens.toLocaleString()} tok</span></div>`,
    ];
    (lg.segments || []).forEach((s) => {
      rows.push(`
        <div class="ctx-seg-row">
          <span>#${s.id} [${escapeHtml(s.type)}] ${escapeHtml(s.label)} (${s.tokens.toLocaleString()} tok)</span>
          ${s.pinned ? "<span>📌</span>" : `<button data-drop-seg="${s.id}" title="Drop segment from context">✕</button>`}
        </div>
      `);
    });
    drawer.innerHTML = rows.join("");
    drawer.querySelectorAll("button[data-drop-seg]").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const sid = btn.dataset.dropSeg;
        try {
          await apiFetch(`/api/context/${sid}`, { method: "DELETE" });
          refreshState();
        } catch (err) {
          showToast(err.message, true);
        }
      });
    });
  }

  el("agent-ctx-bar").addEventListener("click", () => {
    el("agent-ctx-drawer").classList.toggle("hidden");
  });

  // --- Header & State Refresh ---
  function applyTopState(s) {
    if (s.seedance_spec && s.seedance_spec.cost_per_sec_usd) {
      state.costPerSec = s.seedance_spec.cost_per_sec_usd;
    }
    if (s.agent_pricing) state.agentPricing = s.agent_pricing;
    if (s.ledger) {
      state.ledger = s.ledger;
      renderAgentContextBar();
    }
    if (Array.isArray(s.prompt_history)) {
      mergeHistoryStrings(s.prompt_history);
    }

    if (s.blender && s.blender.connected) {
      const scName = (s.blender.scene && s.blender.scene.name) || "Scene";
      pillBlender.textContent = `Blender 5.3: ${s.blender.mode} (${scName})`;
      pillBlender.className = "pill pill-ok";
    } else {
      pillBlender.textContent = "Blender 5.3: standby";
      pillBlender.className = "pill pill-muted";
    }

    if (s.openrouter && s.openrouter.configured) {
      btnOrKey.textContent = `OpenRouter: ready (${s.openrouter.masked.split(",")[0]})`;
      btnOrKey.className = "pill pill-btn pill-ok";
    } else {
      btnOrKey.textContent = "OpenRouter: Set API Key";
      btnOrKey.className = "pill pill-btn pill-warn";
    }

    if (s.health) {
      const hs = s.health.status || "healthy";
      pillHealth.textContent = `Health: ${hs}`;
      pillHealth.className =
        hs === "healthy" ? "pill pill-ok" : hs === "watch" ? "pill pill-warn" : "pill pill-err";
    }
    if (s.stats && typeof s.stats.cost === "number") {
      state.sessionCost = s.stats.cost;
      pillCost.textContent = `Session: $${s.stats.cost.toFixed(4)}`;
    }
    el("btn-agent-cancel").classList.toggle("hidden", !s.busy);

    if (Array.isArray(s.jobs)) {
      s.jobs.forEach((j) => handleJobUpdate(j, true));
      renderJobs();
    }
    if (Array.isArray(s.splat_jobs)) {
      s.splat_jobs.forEach((sj) => state.splatJobs.set(sj.job_id, sj));
      renderSplatJobs();
    }
    updatePricingAndInspector();
  }

  pillBlender.addEventListener("click", async () => {
    pillBlender.textContent = "Blender 5.3: connecting…";
    try {
      const res = await apiFetch("/api/blender/connect", { method: "POST", json: { mode: "auto" } });
      const b = res.blender || {};
      showToast(`Connected to Blender 5.3 [${b.mode || "auto"}] (PID ${b.pid || "?"})`);
      refreshState();
    } catch (err) {
      showToast(`Blender connect error: ${err.message}`, true);
      refreshState();
    }
  });

  async function refreshState() {
    try {
      const s = await apiFetch("/api/state");
      applyTopState(s);
    } catch (err) {
      showToast(`State error: ${err.message}`, true);
    }
  }

  async function refreshModels() {
    try {
      const data = await apiFetch("/api/seedance/models");
      if (Array.isArray(data.models) && data.models.length) {
        state.videoModels = data.models;
        populatePricingDropdowns();
      }
    } catch (_e) {
      // Keep default seedance-2.5
    }
  }

  // --- OpenRouter Key Drawer ---
  btnOrKey.addEventListener("click", () => {
    keyDrawer.classList.toggle("hidden");
    if (!keyDrawer.classList.contains("hidden")) inputOrKey.focus();
  });
  el("btn-close-key").addEventListener("click", () => keyDrawer.classList.add("hidden"));
  el("btn-save-key").addEventListener("click", async () => {
    const k = inputOrKey.value.trim();
    if (!k) return;
    try {
      await apiFetch("/api/keys/openrouter", { method: "POST", json: { key: k, save: true } });
      inputOrKey.value = "";
      keyDrawer.classList.add("hidden");
      showToast("Saved OPENROUTER_API_KEY to ~/.blenderbot/.env");
      refreshState();
      refreshModels();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  // --- Preview Theater ---
  function stopSplatViewer() {
    if (window._bbSplatViewer) window._bbSplatViewer.pause();
    if (window._bbStopComparator) window._bbStopComparator();
    const sc = el("theater-splat-canvas");
    const hud = el("theater-splat-hud");
    const cmp = el("theater-compare-wrap");
    if (sc) sc.classList.add("hidden");
    if (hud) hud.classList.add("hidden");
    if (cmp) cmp.classList.add("hidden");
    if (el("btn-stage-compare")) el("btn-stage-compare").classList.add("hidden");
  }

  function openInTheater(item) {
    stopSplatViewer();
    state.stageItem = item;
    state.activeSplatJob = null;
    el("stage-title").textContent = item.path || item.name || "";
    el("theater-empty").classList.add("hidden");
    const vid = el("theater-video");
    const img = el("theater-image");
    const aud = el("theater-audio");
    vid.pause();
    aud.pause();
    vid.classList.add("hidden");
    img.classList.add("hidden");
    aud.classList.add("hidden");

    const authedUrl = withTokenUrl(item.url);
    if (item.kind === "video") {
      vid.poster = withTokenUrl(`/api/media/poster?path=${encodeURIComponent(item.path || "")}`);
      vid.src = authedUrl;
      vid.classList.remove("hidden");
      vid.load();
      vid.play().catch(() => {});
    } else if (item.kind === "image") {
      img.src = authedUrl;
      img.classList.remove("hidden");
    } else if (item.kind === "audio") {
      aud.src = authedUrl;
      aud.classList.remove("hidden");
      aud.load();
    }

    el("theater-actions").classList.remove("hidden");
    const sizeLabel = item.size_bytes ? ` · ${formatBytes(item.size_bytes)}` : "";
    const costLabel = item.cost_usd ? ` · Output cost: $${Number(item.cost_usd).toFixed(4)}` : "";
    el("theater-meta").textContent = `${item.kind.toUpperCase()}${sizeLabel}${costLabel}`;
    el("btn-stage-to-splat").classList.toggle("hidden", item.kind !== "video");
    el("btn-stage-import-bl").classList.add("hidden");
    el("btn-stage-use-vid").classList.toggle("hidden", item.kind !== "video");
    el("btn-stage-ref-vid").classList.toggle("hidden", item.kind !== "video");
    el("btn-stage-use-first").classList.toggle("hidden", item.kind !== "image");
    el("btn-stage-at-agent").classList.remove("hidden");
    const dl = el("link-stage-download");
    dl.classList.remove("hidden");
    dl.href = `${authedUrl}&download=1`;
    dl.download = item.name || "media";
  }

  el("btn-stage-to-splat").addEventListener("click", () => {
    if (state.stageItem && state.stageItem.kind === "video") {
      jumpToSplatWithVideo(state.stageItem.path);
    }
  });
  el("btn-stage-import-bl").addEventListener("click", () => {
    if (state.activeSplatJob) {
      importSplatToBlender(state.activeSplatJob.job_id);
    }
  });

  el("btn-stage-use-vid").textContent = "Set Vid 1";
  el("btn-stage-ref-vid").textContent = "Set Vid 2";
  el("btn-stage-use-vid").addEventListener("click", () => {
    if (state.stageItem && state.stageItem.kind === "video") setSourceVideo(state.stageItem);
  });
  el("btn-stage-ref-vid").addEventListener("click", () => {
    if (state.stageItem && state.stageItem.kind === "video") setSourceVideo2(state.stageItem);
  });
  el("btn-stage-use-first").addEventListener("click", () => {
    if (state.stageItem && state.stageItem.kind === "image") setFirstFrame(state.stageItem);
  });
  el("btn-stage-at-agent").addEventListener("click", () => {
    if (!state.stageItem) return;
    const inp = el("agent-input");
    inp.value = `${inp.value ? inp.value + " " : ""}@${state.stageItem.path} `;
    inp.focus();
  });

  // --- Source Video 1 & 2 Dropdowns Sync ---
  function populateSourceVideoDropdown() {
    const videos = state.mediaItems.filter((it) => it.kind === "video");
    const sel1 = el("sel-source-video");
    const sel2 = el("sel-source-video-2");
    const cur1 = state.sourceVideo ? state.sourceVideo.path : "";
    const cur2 = state.sourceVideo2 ? state.sourceVideo2.path : "";

    sel1.innerHTML = `<option value="">— None —</option>`;
    sel2.innerHTML = `<option value="">— None (Single video / optional 2nd clip) —</option>`;

    videos.forEach((it) => {
      const costTag = it.cost_usd ? ` · Output cost $${Number(it.cost_usd).toFixed(4)}` : "";
      const label = `${it.path} (${formatBytes(it.size_bytes)}${costTag})`;

      const o1 = document.createElement("option");
      o1.value = it.path;
      o1.textContent = label;
      if (it.path === cur1) o1.selected = true;
      sel1.appendChild(o1);

      const o2 = document.createElement("option");
      o2.value = it.path;
      o2.textContent = label;
      if (it.path === cur2) o2.selected = true;
      sel2.appendChild(o2);
    });
  }

  el("sel-source-video").addEventListener("change", (e) => {
    const val = e.target.value;
    const item = val ? state.mediaItems.find((it) => it.path === val) || null : null;
    setSourceVideo(item);
    if (item) openInTheater(item);
  });
  el("sel-source-video-2").addEventListener("change", (e) => {
    const val = e.target.value;
    const item = val ? state.mediaItems.find((it) => it.path === val) || null : null;
    setSourceVideo2(item);
    if (item) openInTheater(item);
  });
  el("sel-dual-mode").addEventListener("change", () => {
    state.dualVideoMode = el("sel-dual-mode").value;
    renderConditioningSlots();
    updatePricingAndInspector();
  });
  el("btn-swap-videos").addEventListener("click", () => {
    const tmpV = state.sourceVideo;
    const tmpP = state.sourceVideoProbe;
    state.sourceVideo = state.sourceVideo2;
    state.sourceVideoProbe = state.sourceVideo2Probe;
    state.sourceVideo2 = tmpV;
    state.sourceVideo2Probe = tmpP;
    populateSourceVideoDropdown();
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  });

  // --- Conditioning Slot Mutations ---
  async function setSourceVideo(item) {
    state.sourceVideo = item || null;
    state.sourceVideoProbe = null;
    if (item) {
      if (state.sourceVideo2 && state.sourceVideo2.path === item.path) {
        state.sourceVideo2 = null;
        state.sourceVideo2Probe = null;
      }
      state.refVideos = state.refVideos.filter((v) => v.path !== item.path);
      apiFetch(`/api/workspace/probe?path=${encodeURIComponent(item.path)}`)
        .then((info) => {
          if (state.sourceVideo && state.sourceVideo.path === item.path) {
            state.sourceVideoProbe = info;
            renderConditioningSlots();
            updatePricingAndInspector();
          }
        })
        .catch(() => {});
    }
    populateSourceVideoDropdown();
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  }

  async function setSourceVideo2(item) {
    state.sourceVideo2 = item || null;
    state.sourceVideo2Probe = null;
    if (item) {
      if (state.sourceVideo && state.sourceVideo.path === item.path) {
        state.sourceVideo = null;
        state.sourceVideoProbe = null;
      }
      state.refVideos = state.refVideos.filter((v) => v.path !== item.path);
      apiFetch(`/api/workspace/probe?path=${encodeURIComponent(item.path)}`)
        .then((info) => {
          if (state.sourceVideo2 && state.sourceVideo2.path === item.path) {
            state.sourceVideo2Probe = info;
            renderConditioningSlots();
            updatePricingAndInspector();
          }
        })
        .catch(() => {});
    }
    populateSourceVideoDropdown();
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  }

  function toggleRefVideo(item) {
    if (
      (state.sourceVideo && state.sourceVideo.path === item.path) ||
      (state.sourceVideo2 && state.sourceVideo2.path === item.path)
    ) {
      showToast(`${item.name} is already selected as Video 1 / Video 2`);
      return;
    }
    const idx = state.refVideos.findIndex((v) => v.path === item.path);
    if (idx >= 0) state.refVideos.splice(idx, 1);
    else state.refVideos.push(item);
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  }

  function setFirstFrame(item) {
    state.firstFrame =
      item && state.firstFrame && state.firstFrame.path === item.path ? null : item;
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  }

  function setLastFrame(item) {
    state.lastFrame =
      item && state.lastFrame && state.lastFrame.path === item.path ? null : item;
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  }

  function toggleRefImage(item) {
    const idx = state.refImages.findIndex((r) => r.path === item.path);
    if (idx >= 0) state.refImages.splice(idx, 1);
    else state.refImages.push(item);
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  }

  function setRefAudio(item) {
    state.refAudio =
      item && state.refAudio && state.refAudio.path === item.path ? null : item;
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  }

  function syncPreviewVideoEl(vEl, item) {
    if (!item) {
      vEl.pause();
      delete vEl.dataset.curPath;
      vEl.removeAttribute("src");
      vEl.classList.add("hidden");
      return;
    }
    if (vEl.dataset.curPath !== item.path) {
      vEl.dataset.curPath = item.path;
      vEl.poster = withTokenUrl(`/api/media/poster?path=${encodeURIComponent(item.path)}`);
      vEl.src = withTokenUrl(item.url);
      vEl.load();
    }
    vEl.classList.remove("hidden");
  }

  function renderConditioningSlots() {
    const vCard = el("slot-video-card");
    const vWrap = el("dual-video-previews");
    const vVid1 = el("slot-video-preview");
    const vVid2 = el("slot-video-preview-2");
    const vOpts = el("slot-video-opts");
    const vLbl = el("slot-video-label");
    const hasAnyVid = !!(state.sourceVideo || state.sourceVideo2);
    const hasBoth = !!(state.sourceVideo && state.sourceVideo2);

    vCard.classList.toggle("active", hasAnyVid);
    el("btn-clear-video").classList.toggle("hidden", !state.sourceVideo);
    el("btn-clear-video-2").classList.toggle("hidden", !state.sourceVideo2);
    el("btn-swap-videos").classList.toggle("hidden", !hasBoth);
    el("dual-video-mode-bar").classList.toggle("hidden", !hasBoth);
    vWrap.classList.toggle("hidden", !hasAnyVid);
    vOpts.classList.toggle("hidden", !hasAnyVid);
    el("lbl-vid-first").classList.toggle("hidden", hasBoth);
    el("lbl-vid-last").classList.toggle("hidden", hasBoth && el("sel-dual-mode").value !== "v1_primary_v2_ref");

    syncPreviewVideoEl(vVid1, state.sourceVideo);
    syncPreviewVideoEl(vVid2, state.sourceVideo2);

    if (hasBoth) {
      vLbl.innerHTML = `Dual Video Active: <b>${escapeHtml(state.sourceVideo.name)}</b> + <b>${escapeHtml(state.sourceVideo2.name)}</b>`;
    } else if (state.sourceVideo) {
      const pr = state.sourceVideoProbe;
      const metaExtra = pr && pr.duration_sec ? ` · ${pr.width || "?"}x${pr.height || "?"} · ${pr.duration_sec}s` : "";
      vLbl.innerHTML = `Active Video 1: <b>${escapeHtml(state.sourceVideo.path)}</b> (${formatBytes(state.sourceVideo.size_bytes)}${metaExtra})`;
    } else if (state.sourceVideo2) {
      vLbl.innerHTML = `Active Video 2: <b>${escapeHtml(state.sourceVideo2.path)}</b> (${formatBytes(state.sourceVideo2.size_bytes)})`;
    } else {
      vLbl.innerHTML = `Pick 1 or 2 videos above (or click <b>Vid 1</b> / <b>Vid 2</b> on cards in the Library)`;
    }

    // First Frame
    const fSlot = el("slot-first");
    const fBody = el("slot-first-body");
    el("btn-clear-first").classList.toggle("hidden", !state.firstFrame);
    fSlot.classList.toggle("filled", !!state.firstFrame);
    fBody.innerHTML = state.firstFrame
      ? `<img src="${withTokenUrl(state.firstFrame.url)}" alt="first" /><div class="slot-caption">${escapeHtml(state.firstFrame.name)}</div>`
      : `<span class="empty-placeholder">Click <b>First</b> on any image</span>`;

    // Last Frame
    const lSlot = el("slot-last");
    const lBody = el("slot-last-body");
    el("btn-clear-last").classList.toggle("hidden", !state.lastFrame);
    lSlot.classList.toggle("filled", !!state.lastFrame);
    lBody.innerHTML = state.lastFrame
      ? `<img src="${withTokenUrl(state.lastFrame.url)}" alt="last" /><div class="slot-caption">${escapeHtml(state.lastFrame.name)}</div>`
      : `<span class="empty-placeholder">Click <b>Last</b> on any image</span>`;

    // Extra Reference Videos, Images & Audio Chips
    const chips = el("refs-chips");
    chips.innerHTML = "";
    if (!state.refVideos.length && !state.refImages.length && !state.refAudio) {
      chips.innerHTML = `<span class="empty-placeholder">Optional — click <b>+Ref</b> on images/videos or <b>Audio</b></span>`;
    } else {
      state.refVideos.forEach((vid, idx) => {
        const chip = document.createElement("span");
        chip.className = "ref-chip";
        chip.innerHTML = `🎞 ${escapeHtml(vid.name)} (2 keyframes) <button title="Remove">✕</button>`;
        chip.querySelector("button").addEventListener("click", () => {
          state.refVideos.splice(idx, 1);
          renderConditioningSlots();
          renderMediaGrid();
          updatePricingAndInspector();
        });
        chips.appendChild(chip);
      });
      state.refImages.forEach((img, idx) => {
        const chip = document.createElement("span");
        chip.className = "ref-chip";
        chip.innerHTML = `🖼 ${escapeHtml(img.name)} <button title="Remove">✕</button>`;
        chip.querySelector("button").addEventListener("click", () => {
          state.refImages.splice(idx, 1);
          renderConditioningSlots();
          renderMediaGrid();
          updatePricingAndInspector();
        });
        chips.appendChild(chip);
      });
      if (state.refAudio) {
        const chip = document.createElement("span");
        chip.className = "ref-chip";
        chip.innerHTML = `🔊 ${escapeHtml(state.refAudio.name)} <button title="Remove">✕</button>`;
        chip.querySelector("button").addEventListener("click", () => {
          state.refAudio = null;
          renderConditioningSlots();
          renderMediaGrid();
          updatePricingAndInspector();
        });
        chips.appendChild(chip);
      }
    }
  }

  el("btn-clear-video").addEventListener("click", () => setSourceVideo(null));
  el("btn-clear-video-2").addEventListener("click", () => setSourceVideo2(null));
  el("btn-clear-first").addEventListener("click", () => setFirstFrame(null));
  el("btn-clear-last").addEventListener("click", () => setLastFrame(null));
  el("btn-clear-cond").addEventListener("click", () => {
    state.sourceVideo = null;
    state.sourceVideoProbe = null;
    state.sourceVideo2 = null;
    state.sourceVideo2Probe = null;
    state.refVideos = [];
    state.firstFrame = null;
    state.lastFrame = null;
    state.refImages = [];
    state.refAudio = null;
    populateSourceVideoDropdown();
    renderConditioningSlots();
    renderMediaGrid();
    updatePricingAndInspector();
  });

  // --- Media Library Grid (with Active Role Glow & Vid 1 / Vid 2 / Splat Toggle Buttons) ---
  function rolesForMedia(it) {
    const roles = [];
    const splatSel = el("sel-splat-video") ? el("sel-splat-video").value : "";
    if (state.studioTab === "splat" && state.splatSourceMode === "video" && splatSel && splatSel === it.path) {
      roles.push({ label: "SPLAT SRC", primary: true });
    }
    if (state.sourceVideo && state.sourceVideo.path === it.path) roles.push({ label: "VID 1", primary: true });
    if (state.sourceVideo2 && state.sourceVideo2.path === it.path) roles.push({ label: "VID 2", primary: true });
    if (state.refVideos.some((v) => v.path === it.path)) roles.push({ label: "+REF VID", primary: false });
    if (state.firstFrame && state.firstFrame.path === it.path) roles.push({ label: "FIRST", primary: false });
    if (state.lastFrame && state.lastFrame.path === it.path) roles.push({ label: "LAST", primary: false });
    if (state.refImages.some((r) => r.path === it.path)) roles.push({ label: "+REF", primary: false });
    if (state.refAudio && state.refAudio.path === it.path) roles.push({ label: "AUDIO", primary: false });
    return roles;
  }

  function renderMediaGrid() {
    mediaGrid.innerHTML = "";
    const q = state.search.toLowerCase();
    const filtered = state.mediaItems.filter((it) => {
      if (state.filter === "seedance" && !it.is_seedance) return false;
      if (state.filter !== "all" && state.filter !== "seedance" && it.kind !== state.filter) return false;
      if (q && !it.path.toLowerCase().includes(q)) return false;
      return true;
    });

    if (filtered.length === 0) {
      mediaGrid.innerHTML = `<div class="empty-placeholder" style="grid-column:1/-1">No matching workspace media found. Upload files or capture from Blender!</div>`;
      return;
    }

    filtered.forEach((it) => {
      const activeRoles = rolesForMedia(it);
      const isPrimary = activeRoles.some((r) => r.primary);
      const card = document.createElement("div");
      card.className = `media-card${isPrimary ? " selected-primary" : activeRoles.length ? " selected-slot" : ""}`;
      const u = withTokenUrl(it.url);

      let thumbInner = "";
      if (it.kind === "image") {
        thumbInner = `<img src="${u}" loading="lazy" alt="${escapeHtml(it.name)}" />`;
      } else if (it.kind === "video") {
        const posterU = withTokenUrl(
          `/api/media/poster?path=${encodeURIComponent(it.path)}&v=${it.mtime || 0}`
        );
        thumbInner = `<video src="${u}" poster="${posterU}" preload="none" muted playsinline></video>`;
      } else {
        thumbInner = `<div class="audio-glyph">♫</div>`;
      }

      const roleBadge = activeRoles.length
        ? `<span class="badge-active-role ${isPrimary ? "" : "role-orange"}">✓ ${activeRoles.map((r) => r.label).join(" · ")}</span>`
        : "";
      const costBadge = it.cost_usd
        ? `<span class="badge-cost" title="Billed output cost">$${Number(it.cost_usd).toFixed(3)}</span>`
        : "";

      card.innerHTML = `
        <div class="media-thumb" title="${escapeHtml(it.path)}">
          <span class="badge-kind">${it.is_seedance ? "seedance" : it.kind}</span>
          ${costBadge}
          ${thumbInner}
          ${roleBadge}
        </div>
        <div class="media-info">
          <div class="media-name" title="${escapeHtml(it.path)}">${escapeHtml(it.name)}</div>
          <div class="media-actions"></div>
        </div>
      `;

      const thumb = card.querySelector(".media-thumb");
      thumb.addEventListener("click", () => openInTheater(it));
      if (it.kind === "video") {
        const vEl = thumb.querySelector("video");
        thumb.addEventListener("mouseenter", () => vEl && vEl.play().catch(() => {}));
        thumb.addEventListener("mouseleave", () => {
          if (vEl) {
            vEl.pause();
            vEl.currentTime = 0.05;
          }
        });
      }

      const act = card.querySelector(".media-actions");
      const addBtn = (label, activeCls, fn) => {
        const b = document.createElement("button");
        b.textContent = label;
        if (activeCls) b.className = activeCls;
        b.addEventListener("click", (e) => {
          e.stopPropagation();
          fn();
        });
        act.appendChild(b);
      };

      if (it.kind === "image") {
        const isFirst = state.firstFrame && state.firstFrame.path === it.path;
        const isLast = state.lastFrame && state.lastFrame.path === it.path;
        const isRef = state.refImages.some((r) => r.path === it.path);
        addBtn(isFirst ? "✓ First" : "First", isFirst ? "btn-active-orange" : "", () => setFirstFrame(it));
        addBtn(isLast ? "✓ Last" : "Last", isLast ? "btn-active-orange" : "", () => setLastFrame(it));
        addBtn(isRef ? "✓ Ref" : "+Ref", isRef ? "btn-active-orange" : "", () => toggleRefImage(it));
      } else if (it.kind === "video") {
        const isV1 = state.sourceVideo && state.sourceVideo.path === it.path;
        const isV2 = state.sourceVideo2 && state.sourceVideo2.path === it.path;
        const isRefV = state.refVideos.some((v) => v.path === it.path);
        addBtn(isV1 ? "✓ Vid 1" : "Vid 1", isV1 ? "btn-active-role" : "", () => setSourceVideo(isV1 ? null : it));
        addBtn(isV2 ? "✓ Vid 2" : "Vid 2", isV2 ? "btn-active-role" : "", () => setSourceVideo2(isV2 ? null : it));
        addBtn(isRefV ? "✓ Ref" : "+Ref", isRefV ? "btn-active-orange" : "", () => toggleRefVideo(it));
        addBtn("🌐 Splat", "", () => jumpToSplatWithVideo(it.path));
      } else if (it.kind === "audio") {
        const isAud = state.refAudio && state.refAudio.path === it.path;
        addBtn(isAud ? "✓ Audio" : "Audio", isAud ? "btn-active-orange" : "", () => setRefAudio(it));
        addBtn("Play", "", () => openInTheater(it));
      }

      mediaGrid.appendChild(card);
    });
  }

  async function refreshMedia(autoSelectNewest = false) {
    try {
      const data = await apiFetch("/api/workspace/media");
      state.mediaItems = data.items || [];
      populateSourceVideoDropdown();
      populateSplatVideoDropdown();
      renderMediaGrid();
      if (autoSelectNewest && state.mediaItems.length > 0 && !state.stageItem) {
        const newestSeedance = state.mediaItems.find((m) => m.is_seedance && m.kind === "video");
        openInTheater(newestSeedance || state.mediaItems[0]);
      }
    } catch (err) {
      showToast(`Media scan failed: ${err.message}`, true);
    }
  }

  el("media-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    document.querySelectorAll("#media-tabs .tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    renderMediaGrid();
  });
  el("media-search").addEventListener("input", (e) => {
    state.search = e.target.value;
    renderMediaGrid();
  });
  el("btn-refresh-media").addEventListener("click", () => refreshMedia(false));

  // --- File Upload (Button + Drag & Drop) ---
  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        showToast(`Uploading ${file.name} (${formatBytes(file.size)})…`);
        const buf = await file.arrayBuffer();
        const res = await fetch(`/api/workspace/upload?filename=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: Object.assign(
            { "X-Filename": file.name },
            token ? { Authorization: `Bearer ${token}` } : {}
          ),
          body: buf,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `Upload HTTP ${res.status}`);
        const item = data.item;
        await refreshMedia(false);
        openInTheater(item);
        // Set first uploaded video as Source Video if none selected yet; additional videos can be toggled or added as +RefVid!
        if (item.kind === "video" && !state.sourceVideo) setSourceVideo(item);
        else if (item.kind === "video" && !state.sourceVideo2) setSourceVideo2(item);
        else if (item.kind === "image" && !state.firstFrame) setFirstFrame(item);
        else if (item.kind === "audio" && !state.refAudio) setRefAudio(item);
        showToast(`Uploaded ${item.name}`);
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

  const dz = el("dropzone-library");
  dz.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag-over");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag-over");
    if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
  });

  // --- 1-Click Blender 5.3 Capture ---
  async function captureFromBlender(kind) {
    const btn = kind === "playblast" ? el("btn-cap-playblast") : el("btn-cap-still");
    btn.disabled = true;
    try {
      showToast(`Rendering ${kind} from Blender 5.3…`);
      const res = await apiFetch("/api/blender/capture", { method: "POST", json: { kind } });
      await refreshMedia(false);
      openInTheater(res.item);
      if (res.item.kind === "video") setSourceVideo(res.item);
      else setFirstFrame(res.item);
      showToast(`Captured ${res.item.name} from Blender 5.3`);
    } catch (err) {
      showToast(`Blender capture failed: ${err.message}`, true);
    } finally {
      btn.disabled = false;
    }
  }
  el("btn-cap-still").addEventListener("click", () => captureFromBlender("still"));
  el("btn-cap-playblast").addEventListener("click", () => captureFromBlender("playblast"));

  // --- Mode Presets ---
  el("mode-pills").addEventListener("click", (e) => {
    const p = e.target.closest(".mode-pill");
    if (!p) return;
    document.querySelectorAll("#mode-pills .mode-pill").forEach((b) => b.classList.remove("active"));
    p.classList.add("active");
    state.mode = p.dataset.mode;
    updatePricingAndInspector();
  });

  // --- Submit Seedance 2.5 Generation ---
  async function submitSeedance() {
    const prompt = el("input-prompt").value.trim();
    const durVal = el("sel-duration").value;
    const sizeVal = el("sel-size").value;
    const seedVal = el("input-seed").value.trim();
    const outPath = el("input-outpath").value.trim();

    if (prompt) pushLocalPromptHistory(prompt);

    const body = {
      model: el("select-video-model").value || "bytedance/seedance-2.5",
      prompt,
      mode: state.mode,
      duration: durVal !== "" ? Number(durVal) : null,
      size: sizeVal || null,
      seed: seedVal !== "" ? Number(seedVal) : null,
      output_path: outPath || null,
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
    };

    const btn = el("btn-generate");
    btn.disabled = true;
    try {
      const res = await apiFetch("/api/seedance/generate", { method: "POST", json: body });
      state.jobs.set(res.job.local_id, res.job);
      renderJobs();
      showToast(`Submitted Seedance job ${res.job.local_id} — polling OpenRouter…`);
    } catch (err) {
      showToast(err.message, true);
      if (err.message.includes("OPENROUTER_API_KEY")) {
        keyDrawer.classList.remove("hidden");
        inputOrKey.focus();
      }
    } finally {
      btn.disabled = false;
    }
  }

  el("btn-generate").addEventListener("click", submitSeedance);
  el("input-prompt").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      submitSeedance();
    }
  });

  // --- Render Seedance Jobs List ---
  function renderJobs() {
    const arr = Array.from(state.jobs.values()).sort((a, b) => b.created_at - a.created_at);
    if (!arr.length) {
      jobsList.innerHTML = `<div class="empty-placeholder">No Seedance jobs submitted in this session yet.</div>`;
      return;
    }
    jobsList.innerHTML = "";
    arr.forEach((j) => {
      const active = !["completed", "failed", "cancelled"].includes(j.stage);
      const cls = j.stage === "completed" ? "completed" : j.stage === "failed" ? "failed" : active ? "running" : "";
      const card = document.createElement("div");
      card.className = `job-card ${cls}`;
      const costBadge = j.cost_usd
        ? `<span class="job-cost-pill">Output: $${Number(j.cost_usd).toFixed(4)}</span>`
        : "";
      card.innerHTML = `
        <div class="job-top">
          <span>${escapeHtml(j.local_id)} · <code>${escapeHtml(j.stage)}</code> (${j.elapsed_sec || 0}s)</span>
          <span>${costBadge} ${active ? `<button class="btn btn-ghost btn-xs btn-cancel-job">Cancel</button>` : ""}</span>
        </div>
        <div class="job-prompt">${escapeHtml(j.error ? "⚠ " + j.error : j.prompt || j.output_rel)}</div>
      `;
      const cancelBtn = card.querySelector(".btn-cancel-job");
      if (cancelBtn) {
        cancelBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          await apiFetch(`/api/seedance/cancel/${j.local_id}`, { method: "POST" }).catch(() => {});
        });
      }
      card.addEventListener("click", () => {
        if (j.stage === "completed" && j.media_url) {
          openInTheater({
            path: j.output_rel,
            name: j.output_rel.split("/").pop(),
            kind: "video",
            size_bytes: 0,
            cost_usd: j.cost_usd,
            url: j.media_url,
          });
        }
      });
      jobsList.appendChild(card);
    });
  }

  // --- Agent Co-Pilot Feed & SSE ---
  function appendFeedLine(text, cls = "feed-agent") {
    const div = document.createElement("div");
    div.className = `feed-msg ${cls}`;
    div.textContent = text;
    agentFeed.appendChild(div);
    agentFeed.scrollTop = agentFeed.scrollHeight;
    return div;
  }

  el("agent-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const inp = el("agent-input");
    const txt = inp.value.trim();
    if (!txt) return;
    pushLocalPromptHistory(txt);
    inp.value = "";
    try {
      await apiFetch("/api/chat/submit", { method: "POST", json: { text: txt } });
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-agent-cancel").addEventListener("click", () => {
    apiFetch("/api/chat/cancel", { method: "POST" }).catch(() => {});
  });

  const announcedCompletions = new Set();

  function handleJobUpdate(data, fromPoll = false) {
    if (!data || !data.local_id) return;
    const prev = state.jobs.get(data.local_id);
    state.jobs.set(data.local_id, data);
    if (!fromPoll) renderJobs();

    if (
      data.stage === "completed" &&
      data.media_url &&
      !announcedCompletions.has(data.local_id)
    ) {
      announcedCompletions.add(data.local_id);
      if (prev) {
        refreshMedia(false);
        if (!fromPoll) refreshState();
        openInTheater({
          path: data.output_rel,
          name: data.output_rel.split("/").pop(),
          kind: "video",
          size_bytes: 0,
          cost_usd: data.cost_usd,
          url: data.media_url,
        });
        showToast(
          `Seedance 2.5 finished: ${data.output_rel} (Output cost: $${Number(data.cost_usd || 0).toFixed(4)})`
        );
      }
    } else if (
      data.stage === "failed" &&
      data.error &&
      (!prev || prev.stage !== "failed")
    ) {
      showToast(data.error, true);
    }
  }

  setInterval(() => {
    const hasRunning = Array.from(state.jobs.values()).some(
      (j) => !["completed", "failed", "cancelled"].includes(j.stage)
    );
    if (hasRunning) refreshState();
  }, 2500);

  let lastEventSeq = 0;

  function connectSSE() {
    const es = new EventSource(withTokenUrl(`/api/events?since=${lastEventSeq}`));
    es.onmessage = (msg) => {
      let ev;
      try {
        ev = JSON.parse(msg.data);
      } catch (_e) {
        return;
      }
      if (typeof ev.seq === "number" && ev.seq > lastEventSeq) {
        lastEventSeq = ev.seq;
      }
      const { type, data } = ev;
      if (type === "seedance-job" && data) {
        handleJobUpdate(data, false);
      } else if (type === "splat-job" && data) {
        handleSplatJobUpdate(data);
      } else if (type === "splat-telemetry" && data) {
        if (window._bbPushSplatTelemetry) window._bbPushSplatTelemetry(data);
      } else if (type === "splat-render" && data) {
        if (window._bbOnSplatRenderProgress) window._bbOnSplatRenderProgress(data);
      } else if (type === "workspace-media-added") {
        refreshMedia(false);
      } else if (type === "transcript" && data) {
        if (data.type === "user") appendFeedLine(`❯ ${data.text}`, "feed-user");
        else if (data.message || data.text) {
          appendFeedLine(`· ${data.message || data.text}`, data.type === "warning" ? "feed-warn" : "feed-op");
        }
      } else if (type === "turn-start") {
        state.liveAgentNode = appendFeedLine("", "feed-agent");
        el("btn-agent-cancel").classList.remove("hidden");
      } else if (type === "text") {
        if (!state.liveAgentNode) state.liveAgentNode = appendFeedLine("", "feed-agent");
        state.liveAgentNode.textContent += String(data || "");
        agentFeed.scrollTop = agentFeed.scrollHeight;
      } else if (type === "ops-parsed" && Array.isArray(data)) {
        appendFeedLine(`⚙ ops: ${data.map((o) => o.op).join(", ")}`, "feed-op");
      } else if (type === "turn-end" || type === "done" || type === "cancelled" || type === "error") {
        state.liveAgentNode = null;
        el("btn-agent-cancel").classList.add("hidden");
        if (type === "error" && data && data.message) appendFeedLine(`✗ ${data.message}`, "feed-warn");
        refreshState();
        refreshMedia(false);
      } else if (type === "blender-changed") {
        refreshState();
      }
    };
    es.onerror = () => {
      es.close();
      setTimeout(connectSSE, 3000);
    };
  }

  // =========================================================================
  // 3DGS SPLAT STUDIO & INTERACTIVE WEBGL2 `.splat` 3D VIEWER
  // =========================================================================

  function switchStudioTab(tab) {
    state.studioTab = tab === "splat" ? "splat" : "seedance";
    const isSplat = state.studioTab === "splat";
    el("tab-mode-seedance").classList.toggle("active", !isSplat);
    el("tab-mode-splat").classList.toggle("active", isSplat);
    el("panel-seedance-composer").classList.toggle("hidden", isSplat);
    el("panel-splat-studio").classList.toggle("hidden", !isSplat);
    el("select-video-model").classList.toggle("hidden", isSplat);
    if (isSplat) {
      window.history.replaceState({}, "", "#splat");
      refreshSplats();
    } else if (window.location.hash === "#splat") {
      window.history.replaceState({}, "", window.location.pathname);
    }
  }

  el("tab-mode-seedance").addEventListener("click", () => switchStudioTab("seedance"));
  el("tab-mode-splat").addEventListener("click", () => switchStudioTab("splat"));

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
    if (b && b.dataset.src) setSplatSourceMode(b.dataset.src);
  });

  function jumpToSplatWithVideo(videoPath) {
    switchStudioTab("splat");
    setSplatSourceMode("video");
    populateSplatVideoDropdown();
    const sel = el("sel-splat-video");
    if (videoPath) {
      sel.value = videoPath;
      const item = state.mediaItems.find((it) => it.path === videoPath);
      if (item) openInTheater(item);
      renderMediaGrid();
    }
  }

  el("sel-splat-video").addEventListener("change", (e) => {
    const val = e.target.value;
    const item = val ? state.mediaItems.find((it) => it.path === val) : null;
    if (item) openInTheater(item);
    renderMediaGrid();
  });

  function populateSplatVideoDropdown() {
    const sel = el("sel-splat-video");
    if (!sel) return;
    const prev = sel.value;
    const videos = state.mediaItems.filter((it) => it.kind === "video");
    sel.innerHTML = `<option value="">— Select workspace / Seedance video —</option>`;
    videos.forEach((it) => {
      const opt = document.createElement("option");
      opt.value = it.path;
      opt.textContent = `${it.path} (${formatBytes(it.size_bytes)})`;
      sel.appendChild(opt);
    });
    if (prev && videos.some((v) => v.path === prev)) {
      sel.value = prev;
    } else if (state.sourceVideo) {
      sel.value = state.sourceVideo.path;
    } else if (videos.length > 0) {
      sel.value = videos[0].path;
    }
  }

  function updateSplatCostEstimate() {
    const iters = Math.max(200, Number(el("inp-splat-iters").value || 5000));
    const gpu = el("sel-splat-gpu").value || "H100";
    const rateHr = gpu === "L40S" ? 1.95 : 3.95;
    const poseSec = state.splatSourceMode === "video" ? 8 : 2;
    const estSec = poseSec + (iters / 2000) * (gpu === "L40S" ? 36 : 20);
    const estUsd = (estSec / 3600) * rateHr;
    el("splat-est-cost").textContent = `Est. Modal ${gpu} ~$${estUsd.toFixed(3)} USD (~${Math.round(estSec)}s for ${iters.toLocaleString()} steps)`;
  }
  el("inp-splat-iters").addEventListener("input", () => {
    const v = String(el("inp-splat-iters").value);
    el("splat-iter-pills").querySelectorAll(".mode-pill").forEach((b) => {
      b.classList.toggle("active", b.dataset.iters === v);
    });
    updateSplatCostEstimate();
  });
  el("splat-iter-pills").addEventListener("click", (e) => {
    const b = e.target.closest(".mode-pill");
    if (!b || !b.dataset.iters) return;
    el("inp-splat-iters").value = b.dataset.iters;
    el("splat-iter-pills").querySelectorAll(".mode-pill").forEach((p) => p.classList.toggle("active", p === b));
    updateSplatCostEstimate();
  });
  el("sel-splat-gpu").addEventListener("change", updateSplatCostEstimate);

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

  if (el("btn-pull-cam-prompt")) {
    el("btn-pull-cam-prompt").addEventListener("click", async () => {
      try {
        const res = await apiFetch("/api/splats/camera_prompt", { method: "POST", json: {} });
        if (res.seedance_camera_prompt) {
          const inp = el("input-prompt");
          const cur = inp.value.trim();
          inp.value = cur ? `${cur} ${res.seedance_camera_prompt}` : res.seedance_camera_prompt;
          updatePricingAndInspector();
          showToast(`Pulled Blender camera '${res.camera}' (${res.distance_m}m, Δyaw ${res.yaw_delta_deg}°)`);
        }
      } catch (err) {
        showToast(`Camera prompt failed: ${err.message}`, true);
      }
    });
  }

  el("btn-train-splat").addEventListener("click", async () => {
    const knownCamVal = el("sel-splat-known-cams") ? el("sel-splat-known-cams").value : "";
    const payload = {
      source_mode: state.splatSourceMode,
      iterations: Math.max(200, Math.min(30000, Number(el("inp-splat-iters").value || 5000))),
      gpu: el("sel-splat-gpu").value || "H100",
      sh_degree: Number(el("sel-splat-sh").value || 1),
      render_mode: el("sel-splat-rendermode").value || "ellipsoids",
      auto_import: el("chk-splat-autoimport").checked,
      lambda_dssim: el("inp-splat-dssim") ? Number(el("inp-splat-dssim").value || 0.2) : 0.2,
      opa_reg: Number(el("inp-splat-opareg").value || 0.0005),
      prune_opa: Number(el("inp-splat-pruneopa").value || 0.025),
      opacity_reset_every: Number(el("inp-splat-opareset").value || 2000),
      conf_percentile: Number(el("inp-splat-vggtconf").value || 45),
      max_anisotropy: Math.max(4, Number(el("inp-splat-maxaniso").value || 10) * 0.6),
      rasterize_mode: el("sel-splat-rastmode") ? el("sel-splat-rastmode").value : "antialiased",
      refine_poses: el("chk-splat-refineposes") ? el("chk-splat-refineposes").checked : false,
      pose_backend: el("sel-splat-pose-backend") ? el("sel-splat-pose-backend").value : "vggt",
    };
    if (state.splatSourceMode === "video") {
      payload.video_path = el("sel-splat-video").value;
      if (!payload.video_path) {
        showToast("Select a source MP4 video first.", true);
        return;
      }
      if (knownCamVal) {
        payload.known_cameras_job = knownCamVal;
      }
    } else {
      payload.num_views = Number(el("inp-splat-views").value || 28);
      payload.targets = el("inp-splat-targets").value.trim();
      payload.transparent_bg = el("chk-splat-rgba").checked;
    }

    const btn = el("btn-train-splat");
    btn.disabled = true;
    try {
      const res = await apiFetch("/api/splats/train", { method: "POST", json: payload });
      if (res.job) {
        if (window._bbSelectWorkbenchJob) window._bbSelectWorkbenchJob(res.job.job_id, true);
        handleSplatJobUpdate(res.job);
      }
      showToast(`Launched Modal ${payload.gpu} 3DGS job (${payload.iterations} steps): ${res.job ? res.job.job_id : ""}`);
    } catch (err) {
      showToast(err.message, true);
    } finally {
      btn.disabled = false;
    }
  });

  async function refreshSplats() {
    try {
      const res = await apiFetch("/api/splats/jobs");
      if (Array.isArray(res.jobs)) {
        res.jobs.forEach((j) => state.splatJobs.set(j.job_id, j));
        renderSplatJobs();
      }
    } catch (_e) {}
  }
  el("btn-refresh-splats").addEventListener("click", refreshSplats);

  async function importSplatToBlender(jobId) {
    const mode = el("sel-splat-rendermode").value || "ellipsoids";
    const maxScale = Number(el("inp-splat-maxscale").value || 0.25);
    const maxAniso = Number(el("inp-splat-maxaniso").value || 10);
    const minOpa = Number(el("inp-splat-pruneopa").value || 0.025);
    showToast(`Importing ${jobId} into Blender 5.3 (culling needles >${maxAniso}x, size >${maxScale}m)…`);
    try {
      const res = await apiFetch("/api/splats/import", {
        method: "POST",
        json: {
          job_id: jobId,
          render_mode: mode,
          max_points: 350000,
          min_opacity: minOpa,
          max_world_scale: maxScale,
          max_anisotropy: maxAniso,
          import_cameras: true,
        },
      });
      const imp = res.imported || {};
      showToast(
        `Imported '${imp.object}' (${Number(imp.points_imported || 0).toLocaleString()} clean splats) into Blender 5.3!`
      );
      refreshState();
    } catch (err) {
      showToast(`Import failed: ${err.message}`, true);
    }
  }

  async function polishResumeSplat(jobId) {
    const maxScale = Number(el("inp-splat-maxscale").value || 0.25);
    const minOpa = Math.max(0.03, Number(el("inp-splat-pruneopa").value || 0.03));
    const gpu = el("sel-splat-gpu").value || "H100";
    showToast(`Resuming ${jobId} on ${gpu}: culling floaters + +2,000 polish steps…`);
    try {
      const res = await apiFetch("/api/splats/resume", {
        method: "POST",
        json: {
          job_id: jobId,
          add_steps: 2000,
          gpu,
          prune_opa: minOpa,
          max_world_scale: maxScale,
          opa_reg: Number(el("inp-splat-opareg").value || 0.0005),
        },
      });
      if (res.job) handleSplatJobUpdate(res.job);
    } catch (err) {
      showToast(`Resume failed: ${err.message}`, true);
    }
  }

  const announcedSplatDone = new Set();

  function handleSplatJobUpdate(j) {
    if (!j || !j.job_id) return;
    const prev = state.splatJobs.get(j.job_id);
    state.splatJobs.set(j.job_id, j);
    renderSplatJobs();

    const active = !["completed", "failed", "cancelled"].includes(j.stage);
    const liveCard = el("splat-live-card");
    if (active || (prev && prev.stage !== "completed" && j.stage === "completed")) {
      liveCard.classList.remove("hidden");
      const pct = j.total_steps > 0 ? Math.min(100, Math.round((j.step / j.total_steps) * 100)) : 5;
      el("splat-prog-fill").style.width = `${j.stage === "completed" ? 100 : pct}%`;
      el("splat-live-title").textContent = `${j.job_id} · ${j.stage} (${j.step}/${j.total_steps})`;
      el("splat-live-metrics").textContent =
        `PSNR ${Number(j.psnr || 0).toFixed(2)} dB · ${Number(j.n_gaussians || 0).toLocaleString()} pts` +
        (j.it_per_sec ? ` · ${j.it_per_sec} it/s` : "");

      if (j.preview_url) {
        el("splat-live-preview-wrap").classList.remove("hidden");
        el("splat-live-img").src = withTokenUrl(j.preview_url);
        if (j.diff_url) el("splat-live-diff").src = withTokenUrl(j.diff_url);
        el("splat-live-img").classList.toggle("hidden", splatShowDiff);
        el("splat-live-diff").classList.toggle("hidden", !splatShowDiff);
      }
    }

    if (j.stage === "completed" && prev && prev.stage !== "completed" && !announcedSplatDone.has(j.job_id)) {
      announcedSplatDone.add(j.job_id);
      showToast(
        `3D Splat ready: ${j.job_id} (${Number(j.n_gaussians || 0).toLocaleString()} Gaussians, $${Number(j.cost_usd || 0).toFixed(4)})`
      );
      refreshState();
      if (j.splat_url) openSplatInTheater(j);
    } else if (j.stage === "failed" && j.error && (!prev || prev.stage !== "failed")) {
      showToast(j.error, true);
    }
  }

  function renderSplatJobs() {
    populateKnownCamsDropdown();
    const box = el("splat-jobs-list");
    if (!box) return;
    const list = Array.from(state.splatJobs.values()).sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    if (list.length === 0) {
      box.innerHTML = `<div class="empty-placeholder">No 3D splats yet. Pick a video or Blender scene above and click Train!</div>`;
      return;
    }
    box.innerHTML = "";
    list.forEach((j) => {
      const card = document.createElement("div");
      card.className = "splat-job-card";
      const thumbSrc = j.preview_url ? withTokenUrl(j.preview_url) : "/favicon.svg";
      const pts = Number(j.n_gaussians || 0).toLocaleString();
      const psnrStr = j.psnr ? ` · ${Number(j.psnr).toFixed(1)}dB` : "";
      const costStr = j.cost_usd ? ` · $${Number(j.cost_usd).toFixed(4)}` : "";

      card.innerHTML = `
        <img class="splat-job-thumb" src="${thumbSrc}" alt="" />
        <div class="splat-job-body">
          <div class="splat-job-title" title="${escapeHtml(j.job_id)}">${escapeHtml(j.job_id)}</div>
          <div class="splat-job-sub">${escapeHtml(j.stage)} · ${pts} pts${psnrStr}${costStr}</div>
        </div>
        <div class="splat-job-actions">
          ${j.cameras_rel ? `<button type="button" class="btn btn-accent btn-xs btn-cmp-path" title="Compare Original Video vs Modal Cloud / Blender Render + 3D Camera Path &amp; Rotation">🎬 Compare &amp; Path</button>` : ""}
          <button type="button" class="btn btn-secondary btn-xs btn-open-plots" title="Open tt-splat Loss/PSNR/Count plots, GPU stage timings, snapshot scrubber &amp; surgeries">📊 Plots</button>
          ${j.splat_url ? `<button type="button" class="btn btn-primary btn-xs btn-view-3d">🌐 3D</button>` : ""}
          ${j.ply_rel ? `<button type="button" class="btn btn-primary btn-xs btn-imp-bl" title="Import .ply + cameras + frustum rig + GT overlay into Blender 5.3">📥 Splat+Cams</button>` : ""}
          ${j.cameras_rel ? `<button type="button" class="btn btn-secondary btn-xs btn-imp-cams" title="Import ONLY solved Seedance camera trajectory + 3D frustum cones + live GT overlay into Blender">🎥 Cams Only</button>` : ""}
          ${j.gt_url ? `<button type="button" class="btn btn-secondary btn-xs btn-view-gt" title="View Ground Truth frame">🎯 GT</button>` : ""}
          ${j.diff_url ? `<button type="button" class="btn btn-secondary btn-xs btn-view-diff" title="View 3x Luminance Error heatmap">🔥 Error</button>` : ""}
          ${j.preview_url ? `<button type="button" class="btn btn-secondary btn-xs btn-view-img" title="View 3DGS Render">🖼 Render</button>` : ""}
          ${j.stage === "completed" ? `<button type="button" class="btn btn-secondary btn-xs btn-polish-splat" title="Surgically cull floaters/needles &amp; train +2,000 polish steps on Modal">✨ +2k</button>` : ""}
          ${j.cameras_url ? `<a class="btn btn-ghost btn-xs" href="${withTokenUrl(j.cameras_url)}&download=1" download="${escapeHtml(j.job_id)}_cameras.json" title="Download solved cameras.json">⬇ JSON</a>` : ""}
        </div>
      `;

      const bCmp = card.querySelector(".btn-cmp-path");
      if (bCmp) bCmp.addEventListener("click", () => openComparatorForJob(j));
      const bPlots = card.querySelector(".btn-open-plots");
      if (bPlots) bPlots.addEventListener("click", () => window._bbSelectWorkbenchJob && window._bbSelectWorkbenchJob(j.job_id, false));
      const b3d = card.querySelector(".btn-view-3d");
      if (b3d) b3d.addEventListener("click", () => openSplatInTheater(j));
      const bPol = card.querySelector(".btn-polish-splat");
      if (bPol) bPol.addEventListener("click", () => polishResumeSplat(j.job_id));
      const bGt = card.querySelector(".btn-view-gt");
      if (bGt) {
        bGt.addEventListener("click", () =>
          openInTheater({
            path: `${j.job_id}/gt_frame_00001.jpg`,
            name: `${j.job_id}_gt.jpg`,
            kind: "image",
            size_bytes: 0,
            url: j.gt_url,
          })
        );
      }
      const bDiff = card.querySelector(".btn-view-diff");
      if (bDiff) {
        bDiff.addEventListener("click", () =>
          openInTheater({
            path: `${j.job_id}/latest_diff.jpg`,
            name: `${j.job_id}_error.jpg`,
            kind: "image",
            size_bytes: 0,
            url: j.diff_url,
          })
        );
      }
      const bImg = card.querySelector(".btn-view-img");
      if (bImg) {
        bImg.addEventListener("click", () =>
          openInTheater({
            path: `${j.job_id}/latest_render.jpg`,
            name: `${j.job_id}_render.jpg`,
            kind: "image",
            size_bytes: 0,
            cost_usd: j.cost_usd,
            url: j.preview_url,
          })
        );
      }
      const bImp = card.querySelector(".btn-imp-bl");
      if (bImp) bImp.addEventListener("click", () => importSplatToBlender(j.job_id));
      const bCams = card.querySelector(".btn-imp-cams");
      if (bCams) {
        bCams.addEventListener("click", async () => {
          showToast(`Importing solved camera trajectory + frustums + GT overlay for ${j.job_id}…`);
          try {
            const res = await apiFetch("/api/splats/import_cameras", {
              method: "POST",
              json: { job_id: j.job_id, overlay: "gt" },
            });
            const c = res.cameras || {};
            showToast(
              `Imported ${c.camera} (${c.frames} frames + ${c.frustum_collection || "frustums"} + GT overlay)!`
            );
            refreshState();
          } catch (err) {
            showToast(err.message, true);
          }
        });
      }
      box.appendChild(card);
    });
  }

  // --- Embedded WebGL2 3DGS Viewer + FPS / Minecraft Camera Rail, Subject-Track & Spatial Brief Engine ---
  const PRIME_LENSES = [18, 24, 28, 35, 50, 85, 105, 135];
  const VOXEL_SIZES = [0.5, 1.0, 2.0];

  function createSplatWebGLViewer(canvas, hudInfoEl) {
    const gl = canvas.getContext("webgl2", {
      antialias: false,
      alpha: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
    });
    if (!gl) return null;
    gl.getExtension("EXT_color_buffer_float");

    const vsSrc = `#version 300 es
    precision highp float;
    precision highp int;

    layout(location = 0) in vec2 a_quad;
    layout(location = 1) in uint a_index;

    uniform highp sampler2D u_posColTex;
    uniform highp sampler2D u_covTexA;
    uniform highp sampler2D u_covTexB;
    uniform mat3 u_W;
    uniform vec3 u_camPos;
    uniform vec2 u_focal;
    uniform vec2 u_viewport;

    out vec4 v_color;
    out vec2 v_uv;

    void main() {
      ivec2 tc = ivec2(int(a_index & 2047u), int(a_index >> 11u));
      vec4 posPack = texelFetch(u_posColTex, tc, 0);
      uint rgbaU = floatBitsToUint(posPack.w);
      vec4 col = vec4(
        float(rgbaU & 255u),
        float((rgbaU >> 8u) & 255u),
        float((rgbaU >> 16u) & 255u),
        float((rgbaU >> 24u) & 255u)
      ) / 255.0;

      vec3 rel = posPack.xyz - u_camPos;
      vec3 cam = u_W * rel;
      if (cam.z < 0.15 || col.a < 0.012) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
      }

      vec3 cA = texelFetch(u_covTexA, tc, 0).xyz;
      vec3 cB = texelFetch(u_covTexB, tc, 0).xyz;
      mat3 Vrk = mat3(
        cA.x, cA.y, cA.z,
        cA.y, cB.x, cB.y,
        cA.z, cB.y, cB.z
      );

      float invZ = 1.0 / cam.z;
      float invZ2 = invZ * invZ;
      mat3 J = mat3(
        u_focal.x * invZ, 0.0, 0.0,
        0.0, u_focal.y * invZ, 0.0,
        -u_focal.x * cam.x * invZ2, -u_focal.y * cam.y * invZ2, 0.0
      );
      mat3 T = J * u_W;
      mat3 cov2d = T * Vrk * transpose(T);

      float diag1 = cov2d[0][0] + 0.3;
      float offDiag = cov2d[0][1];
      float diag2 = cov2d[1][1] + 0.3;

      float mid = 0.5 * (diag1 + diag2);
      float radius = length(vec2((diag1 - diag2) * 0.5, offDiag));
      float lambda1 = mid + radius;
      float lambda2 = max(mid - radius, 0.1);

      vec2 diagVec = normalize(vec2(offDiag, lambda1 - diag1 + 1e-7));
      vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagVec;
      vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagVec.y, -diagVec.x);

      vec2 centerNdc = vec2(cam.x * u_focal.x * invZ, cam.y * u_focal.y * invZ) / (u_viewport * 0.5);
      vec2 offsetNdc = (a_quad.x * v1 + a_quad.y * v2) * 2.0 / u_viewport;

      gl_Position = vec4(centerNdc + offsetNdc, 0.0, 1.0);
      v_uv = a_quad * sqrt(2.0);
      v_color = col;
    }`;

    const fsSrc = `#version 300 es
    precision mediump float;
    in vec4 v_color;
    in vec2 v_uv;
    out vec4 fragColor;
    void main() {
      float r2 = dot(v_uv, v_uv);
      if (r2 > 8.0) discard;
      float a = v_color.a * exp(-0.5 * r2);
      if (a < 0.004) discard;
      fragColor = vec4(v_color.rgb * a, a);
    }`;

    // Second pass: 3D Wireframe / Rail / Subject-Capsule / Voxel / Beacon Overlay Shader
    const lineVsSrc = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 a_pos;
    layout(location = 1) in vec4 a_col;
    uniform mat3 u_W;
    uniform vec3 u_camPos;
    uniform vec2 u_focal;
    uniform vec2 u_viewport;
    out vec4 v_col;
    void main() {
      vec3 cam = u_W * (a_pos - u_camPos);
      float z = max(0.08, cam.z);
      vec2 ndc = vec2(cam.x * u_focal.x / z, cam.y * u_focal.y / z) / (u_viewport * 0.5);
      gl_Position = vec4(ndc, 0.0, 1.0);
      v_col = a_col;
    }`;

    const lineFsSrc = `#version 300 es
    precision mediump float;
    in vec4 v_col;
    out vec4 fragColor;
    void main() {
      fragColor = vec4(v_col.rgb * v_col.a, v_col.a);
    }`;

    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(sh));
      }
      return sh;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(prog);

    const lineProg = gl.createProgram();
    gl.attachShader(lineProg, compile(gl.VERTEX_SHADER, lineVsSrc));
    gl.attachShader(lineProg, compile(gl.FRAGMENT_SHADER, lineFsSrc));
    gl.linkProgram(lineProg);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const idxBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(1, 1);

    const lineVao = gl.createVertexArray();
    gl.bindVertexArray(lineVao);
    const lineBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);

    const makeTex = (unit) => {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    const posColTex = makeTex(0);
    const covTexA = makeTex(1);
    const covTexB = makeTex(2);

    const uW = gl.getUniformLocation(prog, "u_W");
    const uCamPos = gl.getUniformLocation(prog, "u_camPos");
    const uFocal = gl.getUniformLocation(prog, "u_focal");
    const uViewport = gl.getUniformLocation(prog, "u_viewport");
    gl.useProgram(prog);
    gl.uniform1i(gl.getUniformLocation(prog, "u_posColTex"), 0);
    gl.uniform1i(gl.getUniformLocation(prog, "u_covTexA"), 1);
    gl.uniform1i(gl.getUniformLocation(prog, "u_covTexB"), 2);

    const luW = gl.getUniformLocation(lineProg, "u_W");
    const luCamPos = gl.getUniformLocation(lineProg, "u_camPos");
    const luFocal = gl.getUniformLocation(lineProg, "u_focal");
    const luViewport = gl.getUniformLocation(lineProg, "u_viewport");

    let count = 0;
    let positions = null;
    let sortedIdx = null;
    let depthBuckets = new Int32Array(65536);
    let depthKeys = null;
    let sceneCenter = [0, 0, 0.8];
    let sceneRadius = 3.0;

    // Orbit state
    let target = [0, 0, 0.8];
    let theta = -1.25;
    let phi = 0.25;
    let dist = 4.5;

    // FPS / Rust / Minecraft Rail & Sandbox state
    const fps = {
      enabled: false,
      eye: [0, -4.0, 1.65],
      yaw: 0.0,            // Body movement heading (0 = +Y forward)
      pitch: -0.04,        // Body pitch (used in Drone Fly mode)
      headYaw: 0.0,        // Rust `Hold Alt` decoupled over-the-shoulder head-look yaw offset!
      headPitch: 0.0,      // Rust `Hold Alt` decoupled head-look pitch offset!
      gimbalLocked: false, // `CapsLock` freezes head-look offset without holding Alt
      autoRun: false,      // Rust `T` toggle: hands-free constant-speed forward dolly cruise!
      prone: false,        // `Z` toggle: 0.32m ground-skimming worms-eye stance
      leanAmt: 0.0,        // `Q` / `E` smoothed [-1..+1] corner peek/lean offset!
      vertVel: 0.0,        // Jump impulse in grounded stance
      floorOffsetZ: 0.0,   // Current height above floor when jumping
      rollDeg: 0.0,
      lensMm: 35,
      grounded: true,      // Starts in Grounded 1.65m Walk (`C`=crouch, `Z`=prone, `G`=toggle Drone Fly!)
      slot: 1,             // Hotbar 1=Cam, 2=Subject, 3=Voxel, 4=Ping
      voxelSizeIdx: 1,     // 1.0m default
      keys: new Set(),
      adsTarget: null,     // `Hold RMB` locks 3D look-at point!
      binoZoom: false,     // `Hold B` temporary 135mm spotter scope
      bookmarks: {},       // `Shift+F1..F4` save POV, `F1..F4` instant teleport
      lastHit: null,       // { pt: [x,y,z], dist, normal: [nx,ny,nz], kind, voxelIdx }
      camNodes: [],        // [{ position, forward, target, lens_mm, roll_deg }]
      subjectNodes: [],    // [{ position }]
      trackSubject: false, // `K` key or UI button toggles Camera Rail auto-tracking Subject
      voxels: [],          // [{ center: [x,y,z], size: [s,s,s], color: [r,g,b,a] }]
      pings: [],           // [{ position: [x,y,z], camera_eye: [x,y,z], note, image_b64 }]
      riding: false,
      rideStartT: 0,
      recordingFlight: false,
      lastRecT: 0,
      pendingPing: null,
    };

    let running = false;
    let dirtySort = true;
    let dragging = false;
    let panMode = false;
    let lastX = 0;
    let lastY = 0;
    let lastFrameT = performance.now();

    function syncHotbarCounts() {
      if (el("cnt-hb-cam")) el("cnt-hb-cam").textContent = String(fps.camNodes.length);
      if (el("cnt-hb-sub")) el("cnt-hb-sub").textContent = String(fps.subjectNodes.length);
      if (el("cnt-hb-vox")) el("cnt-hb-vox").textContent = String(fps.voxels.length);
      if (el("cnt-hb-ping")) el("cnt-hb-ping").textContent = String(fps.pings.length);
      if (el("fps-hotbar")) {
        el("fps-hotbar").querySelectorAll(".hotbar-slot").forEach((b) => {
          b.classList.toggle("active", Number(b.dataset.slot) === fps.slot);
        });
      }
      if (el("btn-rail-track-sub")) {
        el("btn-rail-track-sub").textContent = `🎯 Track Subj: ${fps.trackSubject ? "ON" : "OFF"}`;
        el("btn-rail-track-sub").classList.toggle("btn-primary", fps.trackSubject);
      }
    }

    function setFpsMode(on, requestLock = false) {
      fps.enabled = !!on;
      if (fps.enabled) {
        // Seed FPS eye from current orbit camera so transition is seamless
        const ob = computeOrbitBasis();
        fps.eye = ob.eye.slice();
        fps.yaw = Math.atan2(ob.fx, ob.fy);
        fps.pitch = Math.asin(Math.max(-0.98, Math.min(0.98, ob.fz)));
      } else if (document.pointerLockElement === canvas) {
        document.exitPointerLock();
      }
      el("fps-crosshair-wrap").classList.toggle("hidden", !fps.enabled);
      el("fps-radar-canvas")?.classList.toggle("hidden", !fps.enabled);
      el("fps-rail-topbar").classList.toggle("hidden", !fps.enabled && fps.camNodes.length === 0);
      el("fps-hotbar").classList.toggle("hidden", !fps.enabled);
      if (el("btn-splat-fps-toggle")) {
        el("btn-splat-fps-toggle").textContent = fps.enabled ? "🌐 Orbit Mode" : "🎮 FPS Mode";
      }
      syncHotbarCounts();
      updateHudText();
      if (fps.enabled && requestLock) {
        canvas.requestPointerLock?.();
      }
      dirtySort = true;
    }

    function updateHudText() {
      if (!fps.enabled) {
        hudInfoEl.textContent = count
          ? `${count.toLocaleString()} Gaussians · Drag orbit · Click 🎮 FPS for Rust/Minecraft Controls`
          : `Blank Sandbox · Click 🎮 FPS Mode to build Voxels & Camera Rail`;
        return;
      }
      const locked = document.pointerLockElement === canvas;
      hudInfoEl.textContent = locked
        ? `WASD · T autorun · Alt free-look · C crouch · Z prone · Q/E peek · RMB lock · MMB ping · B bino`
        : `Click viewport to lock mouse · 1-4 hotbar · T autorun · Alt look · P ride`;
    }

    document.addEventListener("pointerlockchange", updateHudText);

    function resetCamera() {
      target = sceneCenter.slice();
      theta = -1.25;
      phi = 0.25;
      dist = Math.max(0.8, sceneRadius * 1.75);
      if (fps.enabled) {
        fps.eye = [sceneCenter[0], sceneCenter[1] - Math.max(2.5, sceneRadius * 1.4), 1.65];
        fps.yaw = 0.0;
        fps.pitch = -0.04;
        fps.headYaw = 0.0;
        fps.headPitch = 0.0;
        fps.autoRun = false;
        fps.prone = false;
        fps.leanAmt = 0.0;
        fps.rollDeg = 0.0;
        fps.adsTarget = null;
      }
      dirtySort = true;
    }

    function computeOrbitBasis() {
      const cp = Math.cos(phi);
      const sp = Math.sin(phi);
      const ct = Math.cos(theta);
      const st = Math.sin(theta);
      const eye = [target[0] + dist * cp * ct, target[1] + dist * cp * st, target[2] + dist * sp];
      let fx = target[0] - eye[0];
      let fy = target[1] - eye[1];
      let fz = target[2] - eye[2];
      const fn = Math.hypot(fx, fy, fz) || 1;
      fx /= fn; fy /= fn; fz /= fn;
      let rx = fy, ry = -fx, rz = 0;
      const rn = Math.hypot(rx, ry, rz) || 1;
      rx /= rn; ry /= rn;
      const ux = ry * fz - rz * fy;
      const uy = rz * fx - rx * fz;
      const uz = rx * fy - ry * fx;
      const W = new Float32Array([rx, ux, fx, ry, uy, fy, rz, uz, fz]);
      return { eye, rx, ry, rz, ux, uy, uz, fx, fy, fz, W, lensMm: 35 };
    }

    function buildBasisFromEyeFwd(eye, fwd, rollDeg = 0, lensOverride = null) {
      let fx = fwd[0], fy = fwd[1], fz = fwd[2];
      const fn = Math.hypot(fx, fy, fz) || 1;
      fx /= fn; fy /= fn; fz /= fn;
      let rx = fy, ry = -fx, rz = 0;
      const rn = Math.hypot(rx, ry, rz);
      if (rn < 1e-5) { rx = 1; ry = 0; rz = 0; } else { rx /= rn; ry /= rn; }
      let ux = ry * fz - rz * fy;
      let uy = rz * fx - rx * fz;
      let uz = rx * fy - ry * fx;
      if (Math.abs(rollDeg) > 0.01) {
        const rad = (rollDeg * Math.PI) / 180.0;
        const cr = Math.cos(rad), sr = Math.sin(rad);
        const nrx = rx * cr + ux * sr, nry = ry * cr + uy * sr, nrz = rz * cr + uz * sr;
        ux = -rx * sr + ux * cr; uy = -ry * sr + uy * cr; uz = -rz * sr + uz * cr;
        rx = nrx; ry = nry; rz = nrz;
      }
      const W = new Float32Array([rx, ux, fx, ry, uy, fy, rz, uz, fz]);
      const effLens = lensOverride !== null ? lensOverride : (fps.binoZoom ? 135 : fps.lensMm);
      return { eye: eye.slice(), rx, ry, rz, ux, uy, uz, fx, fy, fz, W, lensMm: effLens };
    }

    function catmull3(p0, p1, p2, p3, u) {
      const u2 = u * u, u3 = u2 * u;
      return [0, 1, 2].map((k) =>
        0.5 * (
          2 * p1[k] +
          (-p0[k] + p2[k]) * u +
          (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * u2 +
          (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * u3
        )
      );
    }

    function evalSplineAt(pts, frac) {
      if (!pts.length) return [0, 0, 0];
      if (pts.length === 1) return pts[0].slice();
      const f = Math.max(0, Math.min(1, frac)) * (pts.length - 1);
      const i = Math.min(pts.length - 2, Math.floor(f));
      const u = f - i;
      return catmull3(
        pts[Math.max(0, i - 1)],
        pts[i],
        pts[i + 1],
        pts[Math.min(pts.length - 1, i + 2)],
        u
      );
    }

    function evalRailCameraAt(frac) {
      const n = fps.camNodes.length;
      if (n === 0) return computeCameraBasis();
      if (n === 1) {
        const c0 = fps.camNodes[0];
        return buildBasisFromEyeFwd(c0.position, c0.forward, c0.roll_deg || 0, c0.lens_mm || 35);
      }
      const pos = evalSplineAt(fps.camNodes.map((c) => c.position), frac);
      let fwd;
      if (fps.trackSubject && fps.subjectNodes.length > 0) {
        const subPt = evalSplineAt(fps.subjectNodes.map((s) => s.position), frac);
        fwd = [subPt[0] - pos[0], subPt[1] - pos[1], subPt[2] + 1.1 - pos[2]];
      } else {
        fwd = evalSplineAt(fps.camNodes.map((c) => c.forward), frac);
      }
      const fIdx = Math.min(n - 2, Math.floor(Math.max(0, Math.min(0.9999, frac)) * (n - 1)));
      const u = Math.max(0, Math.min(1, frac * (n - 1) - fIdx));
      const lens = (fps.camNodes[fIdx].lens_mm || 35) * (1 - u) + (fps.camNodes[fIdx + 1].lens_mm || 35) * u;
      const roll = (fps.camNodes[fIdx].roll_deg || 0) * (1 - u) + (fps.camNodes[fIdx + 1].roll_deg || 0) * u;
      return buildBasisFromEyeFwd(pos, fwd, roll, lens);
    }

    function computeCameraBasis() {
      if (fps.riding && fps.camNodes.length >= 1) {
        const dur = Math.max(1, Number(el("inp-rail-dur")?.value || 5));
        const elapsed = (performance.now() - fps.rideStartT) / 1000.0;
        const frac = (elapsed % dur) / dur;
        return evalRailCameraAt(frac);
      }
      if (!fps.enabled) return computeOrbitBasis();

      // Apply Q/E tactical corner peek/lean offset along body-right axis!
      const bodyRx = Math.cos(fps.yaw), bodyRy = -Math.sin(fps.yaw);
      const effEye = [
        fps.eye[0] + bodyRx * (fps.leanAmt * 0.42),
        fps.eye[1] + bodyRy * (fps.leanAmt * 0.42),
        fps.eye[2] - Math.abs(fps.leanAmt) * 0.06,
      ];
      const effRoll = fps.rollDeg + fps.leanAmt * 6.5;

      let fwd;
      if (fps.adsTarget) {
        fwd = [
          fps.adsTarget[0] - effEye[0],
          fps.adsTarget[1] - effEye[1],
          fps.adsTarget[2] - effEye[2],
        ];
        const fn = Math.hypot(fwd[0], fwd[1], fwd[2]) || 1;
        // When NOT holding Alt, ADS updates body heading; when holding Alt, ADS tracks over shoulder!
        const altHeld = fps.keys.has("AltLeft") || fps.keys.has("AltRight") || fps.gimbalLocked;
        if (!altHeld) {
          fps.yaw = Math.atan2(fwd[0] / fn, fwd[1] / fn);
          fps.pitch = Math.asin(Math.max(-0.98, Math.min(0.98, fwd[2] / fn)));
        }
      } else {
        const lookYaw = fps.yaw + fps.headYaw;
        const lookPitch = Math.max(-1.48, Math.min(1.48, fps.pitch + fps.headPitch));
        const cp = Math.cos(lookPitch);
        fwd = [Math.sin(lookYaw) * cp, Math.cos(lookYaw) * cp, Math.sin(lookPitch)];
      }
      return buildBasisFromEyeFwd(effEye, fwd, effRoll);
    }

    // Raycast crosshair against Voxels -> 3DGS Splat Point Cloud -> Z=0 Ground Plane!
    function raycastCrosshair(eye, fwd) {
      let bestT = Infinity;
      let bestHit = null;

      // 1. Exact slab ray-AABB test against placed Voxels (enables Minecraft face-stacking!)
      fps.voxels.forEach((vx, vIdx) => {
        const c = vx.center, s = vx.size;
        let tmin = -Infinity, tmax = Infinity;
        let hitAxis = 2, hitSign = 1;
        for (let a = 0; a < 3; a++) {
          const lo = c[a] - 0.5 * s[a], hi = c[a] + 0.5 * s[a];
          const d = fwd[a];
          if (Math.abs(d) < 1e-7) {
            if (eye[a] < lo || eye[a] > hi) return;
          } else {
            let t1 = (lo - eye[a]) / d, t2 = (hi - eye[a]) / d;
            let sgn = -1;
            if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; sgn = 1; }
            if (t1 > tmin) { tmin = t1; hitAxis = a; hitSign = sgn; }
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return;
          }
        }
        if (tmin > 0.15 && tmin < bestT) {
          bestT = tmin;
          const nrm = [0, 0, 0];
          nrm[hitAxis] = hitSign;
          bestHit = {
            pt: [eye[0] + fwd[0] * tmin, eye[1] + fwd[1] * tmin, eye[2] + fwd[2] * tmin],
            dist: tmin,
            normal: nrm,
            kind: "voxel",
            voxelIdx: vIdx,
          };
        }
      });

      // 2. Fast cone ray-pick against 3DGS splat cloud (~20k sampled points, <0.25 ms)
      if (count > 0 && positions) {
        const stride = Math.max(1, Math.floor(count / 20000));
        const hitsZ = [];
        for (let i = 0; i < count; i += stride) {
          const dx = positions[3 * i] - eye[0];
          const dy = positions[3 * i + 1] - eye[1];
          const dz = positions[3 * i + 2] - eye[2];
          const z = dx * fwd[0] + dy * fwd[1] + dz * fwd[2];
          if (z <= 0.25 || z >= bestT) continue;
          const d2 = dx * dx + dy * dy + dz * dz - z * z;
          const rTol = z * 0.035 + 0.05;
          if (d2 <= rTol * rTol) hitsZ.push(z);
        }
        if (hitsZ.length >= 3) {
          hitsZ.sort((a, b) => a - b);
          const zPick = hitsZ[Math.floor(hitsZ.length * 0.15)];
          if (zPick < bestT) {
            bestT = zPick;
            bestHit = {
              pt: [eye[0] + fwd[0] * zPick, eye[1] + fwd[1] * zPick, eye[2] + fwd[2] * zPick],
              dist: zPick,
              normal: [0, 0, 1],
              kind: "splat",
              voxelIdx: -1,
            };
          }
        }
      }

      // 3. Fallback: Z=0 metric floor intersection
      if (!bestHit && fwd[2] < -0.015 && eye[2] > 0) {
        const tg = -eye[2] / fwd[2];
        if (tg > 0.1 && tg < 80.0) {
          bestHit = {
            pt: [eye[0] + fwd[0] * tg, eye[1] + fwd[1] * tg, 0.0],
            dist: tg,
            normal: [0, 0, 1],
            kind: "floor",
            voxelIdx: -1,
          };
        }
      }
      return bestHit;
    }

    function stepFpsMovement(dt) {
      if (!fps.enabled || fps.riding) return;
      if (el("fps-ping-modal") && !el("fps-ping-modal").classList.contains("hidden")) return;

      const altHeld = fps.keys.has("AltLeft") || fps.keys.has("AltRight");
      // Rust head-snap: smoothly spring `headYaw`/`headPitch` back to 0 when Alt is released!
      if (!altHeld && !fps.gimbalLocked && (Math.abs(fps.headYaw) > 1e-4 || Math.abs(fps.headPitch) > 1e-4)) {
        const kSnap = Math.min(1.0, dt * 14.0);
        fps.headYaw *= 1.0 - kSnap;
        fps.headPitch *= 1.0 - kSnap;
        if (Math.abs(fps.headYaw) < 1e-3) fps.headYaw = 0.0;
        if (Math.abs(fps.headPitch) < 1e-3) fps.headPitch = 0.0;
        dirtySort = true;
      }

      // Smooth Q / E corner peek/lean (`-1` left, `+1` right; Shift+Q/E adjusts persistent Dutch roll)
      const shiftHeld = fps.keys.has("ShiftLeft") || fps.keys.has("ShiftRight");
      const ctrlHeld = fps.keys.has("ControlLeft") || fps.keys.has("ControlRight");
      const crouchHeld = fps.keys.has("KeyC");

      let targetLean = 0.0;
      if (!shiftHeld) {
        if (fps.keys.has("KeyQ")) targetLean -= 1.0;
        if (fps.keys.has("KeyE")) targetLean += 1.0;
      } else {
        if (fps.keys.has("KeyQ")) fps.rollDeg = Math.max(-35, fps.rollDeg - 30 * dt);
        if (fps.keys.has("KeyE")) fps.rollDeg = Math.min(35, fps.rollDeg + 30 * dt);
      }
      if (Math.abs(fps.leanAmt - targetLean) > 1e-4) {
        fps.leanAmt += (targetLean - fps.leanAmt) * Math.min(1.0, dt * 13.0);
        dirtySort = true;
      }

      // Speed modifiers: Shift = Sprint (2.6x), Ctrl = Silent Slow-Walk Creep (0.30x), Crouch = 0.55x, Prone = 0.25x
      let spdMult = shiftHeld ? 2.6 : ctrlHeld ? 0.30 : 1.0;
      if (fps.grounded) {
        if (fps.prone) spdMult *= 0.28;
        else if (crouchHeld) spdMult *= 0.58;
      }
      const baseSpd = Math.max(2.2, sceneRadius * 0.9) * spdMult;

      // CRITICAL RUST MECHANIC: Movement vector uses `fps.yaw` (body heading), NOT `headYaw`!
      // Holding `Alt` to look over your shoulder NEVER curves your `WASD` or `T` Auto-Run line!
      const cpBody = Math.cos(fps.pitch);
      const bodyFx = Math.sin(fps.yaw) * (fps.grounded ? 1.0 : cpBody);
      const bodyFy = Math.cos(fps.yaw) * (fps.grounded ? 1.0 : cpBody);
      const bodyFz = fps.grounded ? 0.0 : Math.sin(fps.pitch);
      const bodyRx = Math.cos(fps.yaw), bodyRy = -Math.sin(fps.yaw);

      if (fps.keys.has("KeyS")) fps.autoRun = false; // Backpedal cancels Rust auto-run

      let vx = 0, vy = 0, vz = 0;
      if (fps.keys.has("KeyW") || fps.autoRun) { vx += bodyFx; vy += bodyFy; vz += bodyFz; }
      if (fps.keys.has("KeyS")) { vx -= bodyFx; vy -= bodyFy; vz -= bodyFz; }
      if (fps.keys.has("KeyD")) { vx += bodyRx; vy += bodyRy; }
      if (fps.keys.has("KeyA")) { vx -= bodyRx; vy -= bodyRy; }
      if (!fps.grounded) {
        if (fps.keys.has("Space")) vz += 1.0;
        if (crouchHeld) vz -= 1.0;
      }

      if (vx !== 0 || vy !== 0 || vz !== 0) {
        const vn = Math.hypot(vx, vy, vz) || 1;
        fps.eye[0] += (vx / vn) * baseSpd * dt;
        fps.eye[1] += (vy / vn) * baseSpd * dt;
        fps.eye[2] += (vz / vn) * baseSpd * dt;
        dirtySort = true;
      }

      if (fps.grounded) {
        // Stance height: Standing = 1.65m, C Crouch = 0.95m, Z Prone = 0.32m + Space Jump arc!
        if (fps.keys.has("Space") && fps.floorOffsetZ <= 0.01) {
          fps.prone = false;
          fps.vertVel = 4.4;
        }
        if (fps.vertVel !== 0 || fps.floorOffsetZ > 0) {
          fps.floorOffsetZ = Math.max(0.0, fps.floorOffsetZ + fps.vertVel * dt);
          fps.vertVel = fps.floorOffsetZ > 0 ? fps.vertVel - 11.5 * dt : 0.0;
          dirtySort = true;
        }
        const stanceZ = crouchHeld ? 0.95 : fps.prone ? 0.32 : 1.65;
        const targetEyeZ = stanceZ + fps.floorOffsetZ;
        fps.eye[2] += (targetEyeZ - fps.eye[2]) * Math.min(1.0, dt * 13.0);
        dirtySort = true;
      }

      // 1-Take Steadicam / Auto-Run + Alt Free-Look recording while holding `R` in slot 1!
      const now = performance.now();
      if (fps.keys.has("KeyR") && fps.slot === 1) {
        if (!fps.recordingFlight) {
          fps.recordingFlight = true;
          fps.camNodes = [];
          fps.lastRecT = 0;
        }
        if (now - fps.lastRecT >= 220) {
          fps.lastRecT = now;
          const cNow = computeCameraBasis();
          fps.camNodes.push({
            position: cNow.eye.slice(),
            forward: [cNow.fx, cNow.fy, cNow.fz],
            target: fps.adsTarget ? fps.adsTarget.slice() : null,
            lens_mm: fps.lensMm,
            roll_deg: fps.rollDeg + fps.leanAmt * 6.5,
          });
          el("fps-rail-topbar").classList.remove("hidden");
          syncHotbarCounts();
        }
      } else {
        fps.recordingFlight = false;
      }
    }

    function drawRadarMinimap(cam) {
      const rc = el("fps-radar-canvas");
      if (!rc || !fps.enabled) return;
      const g = rc.getContext("2d");
      const W = rc.width, H = rc.height;
      const cx = W * 0.5, cy = H * 0.5, rad = W * 0.46;
      g.clearRect(0, 0, W, H);

      // Range ring
      g.strokeStyle = "rgba(56, 189, 248, 0.25)";
      g.lineWidth = 1;
      g.beginPath(); g.arc(cx, cy, rad * 0.55, 0, Math.PI * 2); g.stroke();

      // Ego-centric rotation: Up on radar (`-Y_px`) = Body Movement Heading (`fps.yaw`)!
      const scalePx = rad / Math.max(4.0, sceneRadius * 1.6);
      const cY = Math.cos(fps.yaw), sY = Math.sin(fps.yaw);
      function toRadar(pt) {
        const dx = pt[0] - fps.eye[0], dy = pt[1] - fps.eye[1];
        // bodyRight = (cos, -sin), bodyFwd = (sin, cos)
        const rx = dx * cY - dy * sY;
        const fy = dx * sY + dy * cY;
        return [cx + rx * scalePx, cy - fy * scalePx];
      }

      // Voxels (cyan squares)
      g.fillStyle = "rgba(56, 189, 248, 0.7)";
      fps.voxels.forEach((vx) => {
        const [px, py] = toRadar(vx.center);
        if (Math.hypot(px - cx, py - cy) < rad) g.fillRect(px - 2, py - 2, 4, 4);
      });

      // Subject Track (magenta line + dots)
      if (fps.subjectNodes.length > 0) {
        g.strokeStyle = "#f472b6";
        g.fillStyle = "#f472b6";
        g.beginPath();
        fps.subjectNodes.forEach((sn, idx) => {
          const [px, py] = toRadar(sn.position);
          if (idx === 0) g.moveTo(px, py); else g.lineTo(px, py);
        });
        g.stroke();
        fps.subjectNodes.forEach((sn) => {
          const [px, py] = toRadar(sn.position);
          g.beginPath(); g.arc(px, py, 2.5, 0, Math.PI * 2); g.fill();
        });
      }

      // Cam Rail (gold)
      if (fps.camNodes.length > 0) {
        g.strokeStyle = "#facc15";
        g.beginPath();
        fps.camNodes.forEach((cn, idx) => {
          const [px, py] = toRadar(cn.position);
          if (idx === 0) g.moveTo(px, py); else g.lineTo(px, py);
        });
        g.stroke();
      }

      // Pings (orange diamonds)
      g.fillStyle = "#fb923c";
      fps.pings.forEach((pg) => {
        const [px, py] = toRadar(pg.position);
        if (Math.hypot(px - cx, py - cy) < rad) {
          g.beginPath(); g.arc(px, py, 3.0, 0, Math.PI * 2); g.fill();
        }
      });

      // Cyan Sight Cone (rotates relative to Up whenever Holding Alt to look over shoulder!!)
      const lookAng = Math.atan2(cam.fx, cam.fy) - fps.yaw;
      const halfFov = Math.atan(18.0 / Math.max(12, cam.lensMm || 35));
      g.fillStyle = "rgba(56, 189, 248, 0.24)";
      g.beginPath();
      g.moveTo(cx, cy);
      g.arc(cx, cy, rad * 0.72, -Math.PI * 0.5 + lookAng - halfFov, -Math.PI * 0.5 + lookAng + halfFov);
      g.closePath();
      g.fill();

      // Green Body Movement Heading Arrow (always points straight Up on ego radar!)
      g.strokeStyle = fps.autoRun ? "#4ade80" : "#e2e8f0";
      g.lineWidth = 2;
      g.beginPath(); g.moveTo(cx, cy + 4); g.lineTo(cx, cy - 9); g.stroke();
      g.fillStyle = "#4ade80";
      g.beginPath(); g.arc(cx, cy, 3, 0, Math.PI * 2); g.fill();
    }

    // Build dynamic 3D wireframe overlay buffer (`GL_LINES` clipped cleanly at near plane z=0.12)
    function draw3DOverlays(cam, w, h) {
      const verts = [];
      const ex = cam.eye[0], ey = cam.eye[1], ez = cam.eye[2];
      const W = cam.W; // column-major: rx=W[0],ry=W[3],rz=W[6]; ux=W[1],uy=W[4],uz=W[7]; fx=W[2],fy=W[5],fz=W[8]
      const zCam = (p) => (p[0] - ex) * W[2] + (p[1] - ey) * W[5] + (p[2] - ez) * W[8];

      function pushSeg(a, b, r, g, bl, al) {
        let za = zCam(a), zb = zCam(b);
        const zNear = 0.12;
        if (za < zNear && zb < zNear) return;
        let pa = a, pb = b;
        if (za < zNear) {
          const t = (zNear - za) / (zb - za);
          pa = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
        } else if (zb < zNear) {
          const t = (zNear - zb) / (za - zb);
          pb = [b[0] + (a[0] - b[0]) * t, b[1] + (a[1] - b[1]) * t, b[2] + (a[2] - b[2]) * t];
        }
        verts.push(pa[0], pa[1], pa[2], r, g, bl, al, pb[0], pb[1], pb[2], r, g, bl, al);
      }

      function pushBox(c, sz, r, g, bl, al) {
        const hx = 0.5 * sz[0], hy = 0.5 * sz[1], hz = 0.5 * sz[2];
        const p = [
          [c[0] - hx, c[1] - hy, c[2] - hz], [c[0] + hx, c[1] - hy, c[2] - hz],
          [c[0] + hx, c[1] + hy, c[2] - hz], [c[0] - hx, c[1] + hy, c[2] - hz],
          [c[0] - hx, c[1] - hy, c[2] + hz], [c[0] + hx, c[1] - hy, c[2] + hz],
          [c[0] + hx, c[1] + hy, c[2] + hz], [c[0] - hx, c[1] + hy, c[2] + hz],
        ];
        [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([i, j]) =>
          pushSeg(p[i], p[j], r, g, bl, al)
        );
      }

      // 1. Z=0 Metric Grid (in FPS mode or when count===0)
      if (fps.enabled || count === 0) {
        const gRad = 8;
        for (let g = -gRad; g <= gRad; g++) {
          const hi = g === 0 ? 0.36 : 0.14;
          pushSeg([g, -gRad, 0], [g, gRad, 0], 0.25, 0.4, 0.65, hi);
          pushSeg([-gRad, g, 0], [gRad, g, 0], 0.25, 0.4, 0.65, hi);
        }
      }

      // 2. Placed Snap-Grid Voxels
      fps.voxels.forEach((vx) => {
        const col = vx.color || [0.35, 0.78, 0.98, 0.9];
        pushBox(vx.center, vx.size, col[0], col[1], col[2], 0.9);
      });

      // Ghost preview voxel when Slot 3 is active
      if (fps.enabled && !fps.riding && fps.slot === 3 && fps.lastHit) {
        const vsz = VOXEL_SIZES[fps.voxelSizeIdx];
        const ghostC = snapVoxelPlacement(fps.lastHit, vsz);
        pushBox(ghostC, [vsz, vsz, vsz], 0.3, 1.0, 0.55, 0.55);
      }

      // 3. Camera Rail Spline + Gold Frustum Blocks
      if (fps.camNodes.length >= 2) {
        const pts = fps.camNodes.map((n) => n.position);
        const steps = Math.max(24, pts.length * 14);
        let prev = pts[0];
        for (let s = 1; s <= steps; s++) {
          const cur = evalSplineAt(pts, s / steps);
          pushSeg(prev, cur, 0.22, 0.78, 1.0, 0.95);
          prev = cur;
        }
      }
      fps.camNodes.forEach((nd) => {
        const nb = buildBasisFromEyeFwd(nd.position, nd.forward, nd.roll_deg || 0);
        const L = nd.position;
        const fl = 0.28, fw = 0.16, fh = 0.10;
        const c4 = [
          [L[0] + nb.fx * fl - nb.rx * fw - nb.ux * fh, L[1] + nb.fy * fl - nb.ry * fw - nb.uy * fh, L[2] + nb.fz * fl - nb.rz * fw - nb.uz * fh],
          [L[0] + nb.fx * fl + nb.rx * fw - nb.ux * fh, L[1] + nb.fy * fl + nb.ry * fw - nb.uy * fh, L[2] + nb.fz * fl + nb.rz * fw - nb.uz * fh],
          [L[0] + nb.fx * fl + nb.rx * fw + nb.ux * fh, L[1] + nb.fy * fl + nb.ry * fw + nb.uy * fh, L[2] + nb.fz * fl + nb.rz * fw + nb.uz * fh],
          [L[0] + nb.fx * fl - nb.rx * fw + nb.ux * fh, L[1] + nb.fy * fl - nb.ry * fw + nb.uy * fh, L[2] + nb.fz * fl - nb.rz * fw + nb.uz * fh],
        ];
        c4.forEach((c, k) => {
          pushSeg(L, c, 1.0, 0.82, 0.2, 0.95);
          pushSeg(c, c4[(k + 1) % 4], 1.0, 0.82, 0.2, 0.95);
        });
      });

      // 4. Subject / Player Landmark Trajectory + Animated 1.75m Player Capsule!
      if (fps.subjectNodes.length >= 1) {
        const sPts = fps.subjectNodes.map((s) => s.position);
        if (sPts.length >= 2) {
          let spPrev = sPts[0];
          for (let s = 1; s <= sPts.length * 12; s++) {
            const sc = evalSplineAt(sPts, s / (sPts.length * 12));
            pushSeg([spPrev[0], spPrev[1], spPrev[2] + 0.05], [sc[0], sc[1], sc[2] + 0.05], 0.95, 0.35, 0.88, 0.92);
            spPrev = sc;
          }
        }
        fps.subjectNodes.forEach((sn) => {
          pushBox([sn.position[0], sn.position[1], sn.position[2] + 0.88], [0.45, 0.45, 1.75], 0.95, 0.35, 0.88, 0.45);
        });
        // Live animated player capsule during Ride preview!
        if (fps.riding) {
          const dur = Math.max(1, Number(el("inp-rail-dur")?.value || 5));
          const frac = (((performance.now() - fps.rideStartT) / 1000.0) % dur) / dur;
          const liveP = evalSplineAt(sPts, frac);
          pushBox([liveP[0], liveP[1], liveP[2] + 0.88], [0.52, 0.52, 1.75], 0.3, 1.0, 0.55, 1.0);
        }
      }

      // 5. Spatial Brief Pings (`[#1]` Vertical Beacon Poles) + ADS Target-Lock Diamond
      fps.pings.forEach((pg) => {
        const p = pg.position;
        pushSeg(p, [p[0], p[1], p[2] + 1.4], 1.0, 0.48, 0.16, 0.95);
        pushBox([p[0], p[1], p[2] + 1.45], [0.22, 0.22, 0.22], 1.0, 0.48, 0.16, 0.95);
      });

      if (fps.adsTarget) {
        pushBox(fps.adsTarget, [0.18, 0.18, 0.18], 0.22, 0.88, 1.0, 1.0);
      }

      if (!verts.length) return;
      const arr = new Float32Array(verts);
      gl.useProgram(lineProg);
      gl.bindVertexArray(lineVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW);
      gl.uniformMatrix3fv(luW, false, cam.W);
      gl.uniform3f(luCamPos, cam.eye[0], cam.eye[1], cam.eye[2]);
      const focalPx = ((cam.lensMm || 35) / 36.0) * Math.max(w, h);
      gl.uniform2f(luFocal, focalPx, focalPx);
      gl.uniform2f(luViewport, w, h);
      gl.drawArrays(gl.LINES, 0, verts.length / 7);
    }

    function snapVoxelPlacement(hit, sz) {
      if (hit.kind === "voxel") {
        const base = fps.voxels[hit.voxelIdx];
        return [
          base.center[0] + hit.normal[0] * (0.5 * base.size[0] + 0.5 * sz),
          base.center[1] + hit.normal[1] * (0.5 * base.size[1] + 0.5 * sz),
          base.center[2] + hit.normal[2] * (0.5 * base.size[2] + 0.5 * sz),
        ];
      }
      const gx = Math.round(hit.pt[0] / sz) * sz;
      const gy = Math.round(hit.pt[1] / sz) * sz;
      const gz = Math.max(0.5 * sz, Math.floor(hit.pt[2] / sz) * sz + 0.5 * sz);
      return [gx, gy, gz];
    }

    function sortBackToFront(cam) {
      if (!count || !positions) return;
      let minZ = 1e9;
      let maxZ = -1e9;
      const ex = cam.eye[0], ey = cam.eye[1], ez = cam.eye[2];
      const fx = cam.fx, fy = cam.fy, fz = cam.fz;
      for (let i = 0; i < count; i++) {
        const z = (positions[3 * i] - ex) * fx + (positions[3 * i + 1] - ey) * fy + (positions[3 * i + 2] - ez) * fz;
        depthKeys[i] = z;
        if (z > 0.1) {
          if (z < minZ) minZ = z;
          if (z > maxZ) maxZ = z;
        }
      }
      if (maxZ <= minZ) maxZ = minZ + 1.0;
      depthBuckets.fill(0);
      const invSpan = 65535.0 / (maxZ - minZ);
      for (let i = 0; i < count; i++) {
        const z = depthKeys[i];
        const b = z <= 0.1 ? 65535 : Math.max(0, Math.min(65534, 65534 - ((z - minZ) * invSpan) | 0));
        depthKeys[i] = b;
        depthBuckets[b]++;
      }
      let sum = 0;
      for (let b = 0; b < 65536; b++) {
        const c = depthBuckets[b];
        depthBuckets[b] = sum;
        sum += c;
      }
      const validCount = depthBuckets[65535];
      for (let i = 0; i < count; i++) {
        const b = depthKeys[i];
        sortedIdx[depthBuckets[b]++] = i;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, sortedIdx.subarray(0, validCount));
      sortBackToFront._validCount = validCount;
      dirtySort = false;
    }

    function handlePrimaryFpsAction() {
      const cam = computeCameraBasis();
      const hit = raycastCrosshair(cam.eye, [cam.fx, cam.fy, cam.fz]);

      if (fps.slot === 1) {
        // Place Camera Rail Node
        fps.camNodes.push({
          position: cam.eye.slice(),
          forward: [cam.fx, cam.fy, cam.fz],
          target: fps.adsTarget ? fps.adsTarget.slice() : null,
          lens_mm: fps.lensMm,
          roll_deg: fps.rollDeg,
        });
        el("fps-rail-topbar").classList.remove("hidden");
        syncHotbarCounts();
        showToast(`Placed Cam Node #${fps.camNodes.length} (${fps.lensMm}mm)`);
      } else if (fps.slot === 2) {
        // Place Subject / Player Waypoint on landmark/floor
        const pt = hit ? hit.pt.slice() : [cam.eye[0] + cam.fx * 3.0, cam.eye[1] + cam.fy * 3.0, 0.0];
        fps.subjectNodes.push({ position: pt });
        el("fps-rail-topbar").classList.remove("hidden");
        syncHotbarCounts();
        showToast(`Placed Subject Waypoint #${fps.subjectNodes.length} @ (${pt[0].toFixed(1)}, ${pt[1].toFixed(1)}, ${pt[2].toFixed(1)})m`);
      } else if (fps.slot === 3) {
        // Place Snap-Grid Voxel Block
        const vsz = VOXEL_SIZES[fps.voxelSizeIdx];
        const baseHit = hit || { pt: [cam.eye[0] + cam.fx * 3.5, cam.eye[1] + cam.fy * 3.5, 0.0], kind: "floor", normal: [0, 0, 1] };
        const center = snapVoxelPlacement(baseHit, vsz);
        fps.voxels.push({ center, size: [vsz, vsz, vsz], color: [0.38, 0.75, 0.96, 1.0] });
        el("fps-rail-topbar").classList.remove("hidden");
        syncHotbarCounts();
      } else if (fps.slot === 4) {
        // Place Mule-style 3D Spatial Ping + Hero Screenshot!
        const pt = hit ? hit.pt.slice() : [cam.eye[0] + cam.fx * 3.0, cam.eye[1] + cam.fy * 3.0, 0.0];
        const heroPng = renderCleanSnapshotDataUrl(cam);
        fps.pendingPing = { position: pt, camera_eye: cam.eye.slice(), image_b64: heroPng };
        if (document.pointerLockElement === canvas) document.exitPointerLock();
        el("fps-ping-coords").textContent = `📍 Ping #${fps.pings.length + 1} (${pt[0].toFixed(1)}, ${pt[1].toFixed(1)}, ${pt[2].toFixed(1)})m`;
        el("fps-ping-modal").classList.remove("hidden");
        el("inp-fps-ping-note").value = "";
        el("inp-fps-ping-note").focus();
      }
    }

    function handleDemolishAction() {
      const cam = computeCameraBasis();
      const hit = raycastCrosshair(cam.eye, [cam.fx, cam.fy, cam.fz]);
      if (fps.slot === 3 && hit && hit.kind === "voxel") {
        fps.voxels.splice(hit.voxelIdx, 1);
      } else if (fps.slot === 1 && fps.camNodes.length) {
        fps.camNodes.pop();
      } else if (fps.slot === 2 && fps.subjectNodes.length) {
        fps.subjectNodes.pop();
      } else if (fps.slot === 4 && fps.pings.length) {
        fps.pings.pop();
      }
      syncHotbarCounts();
    }

    canvas.addEventListener("mousedown", (e) => {
      if (fps.enabled) {
        if (document.pointerLockElement !== canvas) {
          canvas.requestPointerLock?.();
          return;
        }
        if (e.button === 0) {
          handlePrimaryFpsAction();
        } else if (e.button === 1) {
          // Apex / Rust Middle-Click (`MMB`): Instant Tactical Landmark Ping from ANY hotbar slot!
          e.preventDefault();
          const cam = computeCameraBasis();
          const hit = raycastCrosshair(cam.eye, [cam.fx, cam.fy, cam.fz]);
          const pt = hit ? hit.pt.slice() : [cam.eye[0] + cam.fx * 3.5, cam.eye[1] + cam.fy * 3.5, 0.0];
          const heroPng = renderCleanSnapshotDataUrl(cam);
          fps.pings.push({
            position: pt,
            camera_eye: cam.eye.slice(),
            note: `Tactical Ping #${fps.pings.length + 1} (${hit ? hit.kind : "air"})`,
            image_b64: heroPng,
          });
          el("fps-rail-topbar").classList.remove("hidden");
          syncHotbarCounts();
          showToast(`📍 Tactical Ping #${fps.pings.length} @ (${pt[0].toFixed(1)}, ${pt[1].toFixed(1)}, ${pt[2].toFixed(1)})m`);
        } else if (e.button === 2) {
          const cam = computeCameraBasis();
          const hit = raycastCrosshair(cam.eye, [cam.fx, cam.fy, cam.fz]);
          fps.adsTarget = hit
            ? hit.pt.slice()
            : [cam.eye[0] + cam.fx * 4.0, cam.eye[1] + cam.fy * 4.0, cam.eye[2] + cam.fz * 4.0];
        }
        return;
      }
      dragging = true;
      panMode = e.button === 2 || e.shiftKey;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    window.addEventListener("mouseup", (e) => {
      dragging = false;
      if (fps.enabled && e.button === 2) {
        fps.adsTarget = null;
      }
    });
    window.addEventListener("mousemove", (e) => {
      if (!running) return;
      if (fps.enabled && document.pointerLockElement === canvas) {
        if (!fps.adsTarget) {
          const altHeld = fps.keys.has("AltLeft") || fps.keys.has("AltRight") || fps.gimbalLocked;
          const sens = fps.binoZoom ? 0.0008 : 0.0024;
          if (altHeld) {
            // Rust `Hold Alt` Free-Look: turn head over shoulder up to ±175° without curving movement!
            fps.headYaw = Math.max(-3.05, Math.min(3.05, fps.headYaw + e.movementX * sens));
            fps.headPitch = Math.max(-1.42, Math.min(1.42, fps.headPitch - e.movementY * sens));
          } else {
            fps.yaw += e.movementX * sens;
            fps.pitch = Math.max(-1.48, Math.min(1.48, fps.pitch - e.movementY * sens));
          }
          dirtySort = true;
        }
        return;
      }
      if (!dragging || fps.enabled) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (panMode) {
        const cam = computeCameraBasis();
        const s = dist * 0.0018;
        target[0] -= (cam.rx * dx - cam.ux * dy) * s;
        target[1] -= (cam.ry * dx - cam.uy * dy) * s;
        target[2] -= (cam.rz * dx - cam.uz * dy) * s;
      } else {
        theta -= dx * 0.007;
        phi = Math.max(-1.48, Math.min(1.48, phi + dy * 0.007));
      }
      dirtySort = true;
    });

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (fps.enabled) {
        const curIdx = PRIME_LENSES.reduce(
          (best, v, idx) => (Math.abs(v - fps.lensMm) < Math.abs(PRIME_LENSES[best] - fps.lensMm) ? idx : best),
          3
        );
        const nextIdx = Math.max(0, Math.min(PRIME_LENSES.length - 1, curIdx + (e.deltaY < 0 ? 1 : -1)));
        fps.lensMm = PRIME_LENSES[nextIdx];
        return;
      }
      dist = Math.max(0.25, Math.min(200.0, dist * Math.exp(e.deltaY * 0.0012)));
      dirtySort = true;
    }, { passive: false });

    window.addEventListener("keydown", (e) => {
      if (!running) return;
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if (e.code === "KeyF" && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        setFpsMode(!fps.enabled, !fps.enabled);
        return;
      }
      if (!fps.enabled) return;

      // Prevent Windows browser menu bar from stealing PointerLock when holding/releasing Left Alt!
      if (e.code === "AltLeft" || e.code === "AltRight" || e.code === "Space") {
        e.preventDefault();
      }
      if (e.repeat && ["KeyT", "KeyG", "KeyZ", "KeyP", "KeyK", "CapsLock"].includes(e.code)) return;

      fps.keys.add(e.code);
      if (["Digit1", "Digit2", "Digit3", "Digit4"].includes(e.code)) {
        fps.slot = Number(e.code.slice(-1));
        syncHotbarCounts();
      } else if (e.code === "KeyT") {
        // Rust `T` Auto-Run: hands-free constant-velocity dolly cruise!
        fps.autoRun = !fps.autoRun;
        showToast(fps.autoRun ? "🏃 Auto-Run ON (Hold Alt to free-look over shoulder!)" : "Auto-Run OFF");
      } else if (e.code === "CapsLock") {
        e.preventDefault();
        fps.gimbalLocked = !fps.gimbalLocked;
        showToast(fps.gimbalLocked ? "🔒 Head-Look Gimbal Locked (keeps off-axis angle)" : "Head-Look springs forward on Alt release");
      } else if (e.code === "KeyZ") {
        fps.prone = !fps.prone;
        showToast(fps.prone ? "🐍 Prone 0.32m Worms-Eye Stance" : "Standing 1.65m Stance");
      } else if (e.code === "KeyB") {
        fps.binoZoom = true;
      } else if (e.code === "KeyV") {
        // Rust `V` Push-to-Callout: drop Spatial Ping + open note prompt at crosshair!
        const prevSlot = fps.slot;
        fps.slot = 4;
        handlePrimaryFpsAction();
        fps.slot = prevSlot;
      } else if (e.code === "KeyG") {
        fps.grounded = !fps.grounded;
        showToast(fps.grounded ? "🚶 Grounded 1.65m Walk (C crouch · Z prone)" : "🛸 Drone Free-Fly (Space up · C down)");
      } else if (e.code === "KeyP") {
        toggleRideRail();
      } else if (e.code === "KeyK") {
        fps.trackSubject = !fps.trackSubject;
        syncHotbarCounts();
        showToast(fps.trackSubject ? "🎯 Camera Rail tracking Subject ON" : "Camera Rail tracking Subject OFF");
      } else if (["F1", "F2", "F3", "F4"].includes(e.code)) {
        e.preventDefault();
        if (e.shiftKey) {
          fps.bookmarks[e.code] = { eye: fps.eye.slice(), yaw: fps.yaw, pitch: fps.pitch, lensMm: fps.lensMm };
          showToast(`Saved POV Bookmark ${e.code}`);
        } else if (fps.bookmarks[e.code]) {
          const bm = fps.bookmarks[e.code];
          fps.eye = bm.eye.slice(); fps.yaw = bm.yaw; fps.pitch = bm.pitch; fps.lensMm = bm.lensMm;
          dirtySort = true;
          showToast(`Teleported to ${e.code}`);
        }
      } else if (e.code === "KeyX" || e.code === "Backspace") {
        handleDemolishAction();
      } else if (e.code === "BracketLeft") {
        fps.voxelSizeIdx = Math.max(0, fps.voxelSizeIdx - 1);
        showToast(`Voxel snap size: ${VOXEL_SIZES[fps.voxelSizeIdx]}m`);
      } else if (e.code === "BracketRight") {
        fps.voxelSizeIdx = Math.min(VOXEL_SIZES.length - 1, fps.voxelSizeIdx + 1);
        showToast(`Voxel snap size: ${VOXEL_SIZES[fps.voxelSizeIdx]}m`);
      } else if (e.code === "Digit0") {
        fps.rollDeg = 0.0;
        fps.headYaw = 0.0;
        fps.headPitch = 0.0;
      }
    });
    window.addEventListener("keyup", (e) => {
      if (fps.enabled && (e.code === "AltLeft" || e.code === "AltRight")) e.preventDefault();
      if (e.code === "KeyB") fps.binoZoom = false;
      fps.keys.delete(e.code);
    });

    function toggleRideRail() {
      if (!fps.camNodes.length) {
        showToast("Place at least 2 Cam Nodes (`[1] + LMB`) to ride the rail!", true);
        return;
      }
      fps.riding = !fps.riding;
      fps.rideStartT = performance.now();
      if (el("btn-rail-ride")) {
        el("btn-rail-ride").textContent = fps.riding ? "⏹ Stop Ride" : "▶ Ride (P)";
      }
    }

    function renderSinglePass(cam, w, h, drawOverlays = true) {
      if (dirtySort || drawOverlays === false) sortBackToFront(cam);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.035, 0.048, 0.078, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      const focalPx = ((cam.lensMm || 35) / 36.0) * Math.max(w, h);
      if (count > 0) {
        gl.useProgram(prog);
        gl.bindVertexArray(vao);
        gl.uniformMatrix3fv(uW, false, cam.W);
        gl.uniform3f(uCamPos, cam.eye[0], cam.eye[1], cam.eye[2]);
        gl.uniform2f(uFocal, focalPx, focalPx);
        gl.uniform2f(uViewport, w, h);
        const nDraw = sortBackToFront._validCount || count;
        if (nDraw > 0) gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, nDraw);
      }

      if (drawOverlays) {
        draw3DOverlays(cam, w, h);
      } else if (fps.voxels.length > 0) {
        const savedNodes = fps.camNodes, savedSub = fps.subjectNodes, savedPings = fps.pings, savedEn = fps.enabled;
        fps.camNodes = []; fps.subjectNodes = []; fps.pings = []; fps.enabled = false;
        draw3DOverlays(cam, w, h);
        fps.camNodes = savedNodes; fps.subjectNodes = savedSub; fps.pings = savedPings; fps.enabled = savedEn;
      }
    }

    function renderCleanSnapshotDataUrl(camOverride) {
      const w = Math.max(64, canvas.width);
      const h = Math.max(64, canvas.height);
      renderSinglePass(camOverride, w, h, false);
      dirtySort = true;
      return canvas.toDataURL("image/png");
    }

    function frameLoop(now = performance.now()) {
      if (!running) return;
      const dt = Math.min(0.08, Math.max(0.001, (now - lastFrameT) / 1000.0));
      lastFrameT = now;
      stepFpsMovement(dt);

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(64, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(64, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const cam = computeCameraBasis();
      if (fps.enabled && !fps.riding) {
        fps.lastHit = raycastCrosshair(cam.eye, [cam.fx, cam.fy, cam.fz]);
        const ch = el("fps-crosshair");
        const tag = el("fps-reticle-tag");
        if (ch && tag) {
          ch.classList.toggle("hit", !!fps.lastHit && !fps.adsTarget);
          ch.classList.toggle("locked", !!fps.adsTarget);
          const altHeld = fps.keys.has("AltLeft") || fps.keys.has("AltRight") || fps.gimbalLocked;
          const crouch = fps.keys.has("KeyC");
          let stanceLbl = "🛸 FLY";
          if (fps.grounded) {
            stanceLbl = crouch ? "🧎 CROUCH 0.95m" : fps.prone ? "🐍 PRONE 0.32m" : "🚶 WALK 1.65m";
          }
          const runLbl = fps.autoRun ? " · 🏃 AUTORUN" : "";
          const altLbl = altHeld ? ` · 👀 ALT ${(fps.headYaw * 57.3).toFixed(0)}°` : "";
          const lockLbl = fps.adsTarget ? " · 🎯 LOCK" : "";
          const binoLbl = fps.binoZoom ? " · 🔭 BINO" : "";
          const distLbl = fps.lastHit ? ` · ${fps.lastHit.dist.toFixed(1)}m` : "";
          const recLbl = fps.recordingFlight ? " · 🔴 REC" : "";
          tag.textContent = `${Math.round(cam.lensMm || fps.lensMm)}mm · ${stanceLbl}${runLbl}${altLbl}${lockLbl}${binoLbl}${distLbl}${recLbl}`;
        }
        drawRadarMinimap(cam);
      }

      renderSinglePass(cam, w, h, true);
      requestAnimationFrame(frameLoop);
    }

    async function loadSplatUrl(url, label) {
      hudInfoEl.textContent = `Loading ${label}…`;
      const resp = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
      if (!resp.ok) throw new Error(`Failed to fetch .splat (${resp.status})`);
      const buf = await resp.arrayBuffer();
      const total = Math.floor(buf.byteLength / 32);
      const cap = Math.min(total, 350000);
      count = cap;

      const f32 = new Float32Array(buf);
      const u32 = new Uint32Array(buf);
      const u8 = new Uint8Array(buf);

      const texW = 2048;
      const texH = Math.ceil(cap / texW);
      const posColData = new Float32Array(texW * texH * 4);
      const posColU32 = new Uint32Array(posColData.buffer);
      const covAData = new Float32Array(texW * texH * 4);
      const covBData = new Float32Array(texW * texH * 4);
      positions = new Float32Array(cap * 3);
      sortedIdx = new Uint32Array(cap);
      depthKeys = new Float32Array(cap);

      const xs = [], ys = [], zs = [];
      const sampleStride = Math.max(1, Math.floor(cap / 2000));

      for (let i = 0; i < cap; i++) {
        const baseF = i * 8;
        const baseB = i * 32;
        const px = f32[baseF];
        const py = f32[baseF + 1];
        const pz = f32[baseF + 2];
        const sx = f32[baseF + 3];
        const sy = f32[baseF + 4];
        const sz = f32[baseF + 5];

        positions[3 * i] = px;
        positions[3 * i + 1] = py;
        positions[3 * i + 2] = pz;
        if (i % sampleStride === 0 && Number.isFinite(px)) {
          xs.push(px); ys.push(py); zs.push(pz);
        }

        posColData[4 * i] = px;
        posColData[4 * i + 1] = py;
        posColData[4 * i + 2] = pz;
        posColU32[4 * i + 3] = u32[baseF + 6];

        let qw = (u8[baseB + 28] - 127.5) / 127.5;
        let qx = (u8[baseB + 29] - 127.5) / 127.5;
        let qy = (u8[baseB + 30] - 127.5) / 127.5;
        let qz = (u8[baseB + 31] - 127.5) / 127.5;
        const qn = Math.hypot(qw, qx, qy, qz) || 1;
        qw /= qn; qx /= qn; qy /= qn; qz /= qn;

        const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
        const r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
        const r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);

        const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
        const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
        const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;

        covAData[4 * i] = m00 * m00 + m01 * m01 + m02 * m02;
        covAData[4 * i + 1] = m00 * m10 + m01 * m11 + m02 * m12;
        covAData[4 * i + 2] = m00 * m20 + m01 * m21 + m02 * m22;
        covBData[4 * i] = m10 * m10 + m11 * m11 + m12 * m12;
        covBData[4 * i + 1] = m10 * m20 + m11 * m21 + m12 * m22;
        covBData[4 * i + 2] = m20 * m20 + m21 * m21 + m22 * m22;
        sortedIdx[i] = i;
      }

      if (xs.length > 4) {
        xs.sort((a, b) => a - b);
        ys.sort((a, b) => a - b);
        zs.sort((a, b) => a - b);
        const mid = xs.length >> 1;
        sceneCenter = [xs[mid], ys[mid], zs[mid]];
        const dists = xs.map((_, k) =>
          Math.hypot(xs[k] - sceneCenter[0], ys[k] - sceneCenter[1], zs[k] - sceneCenter[2])
        ).sort((a, b) => a - b);
        sceneRadius = Math.max(0.5, dists[Math.floor(dists.length * 0.65)] || 2.5);
      }

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, posColTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texW, texH, 0, gl.RGBA, gl.FLOAT, posColData);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, covTexA);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texW, texH, 0, gl.RGBA, gl.FLOAT, covAData);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, covTexB);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texW, texH, 0, gl.RGBA, gl.FLOAT, covBData);

      gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
      gl.bufferData(gl.ARRAY_BUFFER, sortedIdx.byteLength, gl.DYNAMIC_DRAW);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, sortedIdx);

      resetCamera();
      updateHudText();
      if (!running) {
        running = true;
        lastFrameT = performance.now();
        requestAnimationFrame(frameLoop);
      }
    }

    function startBlankSandbox() {
      if (!running) {
        running = true;
        lastFrameT = performance.now();
        requestAnimationFrame(frameLoop);
      }
      setFpsMode(true, false);
    }

    function buildExportPayload(extra = {}) {
      const nodes = fps.camNodes.length
        ? fps.camNodes
        : [(() => { const c = computeCameraBasis(); return { position: c.eye, forward: [c.fx, c.fy, c.fz], lens_mm: fps.lensMm, roll_deg: fps.rollDeg }; })()];
      const firstCam = evalRailCameraAt(0.0);
      const lastCam = evalRailCameraAt(1.0);
      const firstB64 = renderCleanSnapshotDataUrl(firstCam);
      const lastB64 = nodes.length > 1 ? renderCleanSnapshotDataUrl(lastCam) : null;

      return Object.assign({
        splat_job_id: state.activeSplatJob ? state.activeSplatJob.job_id : null,
        cam_nodes: nodes,
        subject_nodes: fps.subjectNodes,
        track_subject: fps.trackSubject,
        voxels: fps.voxels,
        pings: fps.pings,
        grounded_walk: fps.grounded,
        duration_sec: Math.max(2, Number(el("inp-rail-dur")?.value || 5)),
        fps: 24,
        width: 1280,
        height: 720,
        first_frame_b64: firstB64,
        last_frame_b64: lastB64,
      }, extra);
    }

    // Bind FPS / Hotbar / Ping / Export UI buttons once
    el("btn-splat-fps-toggle")?.addEventListener("click", () => setFpsMode(!fps.enabled, !fps.enabled));
    el("fps-hotbar")?.addEventListener("click", (e) => {
      const b = e.target.closest(".hotbar-slot");
      if (b && b.dataset.slot) {
        fps.slot = Number(b.dataset.slot);
        syncHotbarCounts();
      }
    });
    el("btn-rail-ride")?.addEventListener("click", toggleRideRail);
    el("btn-rail-track-sub")?.addEventListener("click", () => {
      fps.trackSubject = !fps.trackSubject;
      syncHotbarCounts();
    });
    el("btn-rail-clear")?.addEventListener("click", () => {
      fps.camNodes = [];
      fps.subjectNodes = [];
      fps.voxels = [];
      fps.pings = [];
      fps.riding = false;
      syncHotbarCounts();
    });

    function commitPendingPing() {
      if (!fps.pendingPing) return;
      const note = el("inp-fps-ping-note").value.trim() || `Landmark #${fps.pings.length + 1}`;
      fps.pings.push(Object.assign({}, fps.pendingPing, { note }));
      fps.pendingPing = null;
      el("fps-ping-modal").classList.add("hidden");
      el("fps-rail-topbar").classList.remove("hidden");
      syncHotbarCounts();
      showToast(`Saved Spatial Ping #${fps.pings.length}: "${note}"`);
    }
    el("btn-fps-ping-save")?.addEventListener("click", commitPendingPing);
    el("btn-fps-ping-cancel")?.addEventListener("click", () => {
      fps.pendingPing = null;
      el("fps-ping-modal").classList.add("hidden");
    });
    el("inp-fps-ping-note")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); commitPendingPing(); }
      if (e.key === "Escape") { fps.pendingPing = null; el("fps-ping-modal").classList.add("hidden"); }
    });

    // 1. Export Rail -> Seedance 2.5 Composer + Lock Splat Prior!
    el("btn-rail-to-seedance")?.addEventListener("click", async () => {
      try {
        const res = await apiFetch("/api/shots/export", { method: "POST", json: buildExportPayload() });
        await refreshMedia(false);
        await refreshShotsList();
        const fItem = state.mediaItems.find((m) => m.path === res.first_frame_rel) ||
          (res.first_frame_rel ? { path: res.first_frame_rel, name: "first_frame.png", kind: "image", url: res.first_frame_url } : null);
        const lItem = state.mediaItems.find((m) => m.path === res.last_frame_rel) ||
          (res.last_frame_rel ? { path: res.last_frame_rel, name: "last_frame.png", kind: "image", url: res.last_frame_url } : null);
        state.firstFrame = fItem;
        state.lastFrame = lItem;
        el("sel-duration").value = String(Math.round(res.brief.duration_sec));
        const curP = el("input-prompt").value.trim();
        el("input-prompt").value = curP ? `${curP} ${res.brief.optical_prompt}` : res.brief.optical_prompt;
        if (el("sel-splat-known-cams")) {
          el("sel-splat-known-cams").value = `shot:${res.shot_id}`;
        }
        renderConditioningSlots();
        renderMediaGrid();
        updatePricingAndInspector();
        switchStudioTab("seedance");
        showToast(`Loaded Rail '${res.shot_id}' (First/Last PNGs + Optical/Subject Prompt + Camera Prior) into Seedance 2.5!`);
      } catch (err) {
        showToast(err.message, true);
      }
    });

    // 2. Farm Modal H100 Rail Render (~2s)
    el("btn-rail-farm-modal")?.addEventListener("click", async () => {
      if (!state.activeSplatJob) {
        showToast("Load a 3D Splat first to farm a Modal 3DGS render (or use 🎬 → Seedance / 🧊 → Blender on blank voxel stages!).", true);
        return;
      }
      const btn = el("btn-rail-farm-modal");
      btn.disabled = true;
      btn.textContent = "☁️ Rendering…";
      try {
        const res = await apiFetch("/api/shots/export", {
          method: "POST",
          json: buildExportPayload({ action: "render_modal", gpu: el("sel-splat-gpu")?.value || "H100" }),
        });
        await refreshShotsList();
        await refreshMedia(false);
        if (res.modal_render_url) {
          openInTheater({
            path: res.modal_render_rel,
            name: `${res.shot_id}.mp4`,
            kind: "video",
            size_bytes: 0,
            cost_usd: res.cost_usd,
            url: res.modal_render_url,
          });
        }
        showToast(`Modal H100 rendered rail '${res.shot_id}' @ ${res.render_fps} fps in ${res.wall_s}s!`);
      } catch (err) {
        showToast(err.message, true);
      } finally {
        btn.disabled = false;
        btn.textContent = "☁️ Modal Render";
      }
    });

    // 3. Sync `shot.usda` / `DirectorRailCam` + Voxel `blockout.glb` + Spatial `brief.md` to Blender 5.3 & Agent!
    el("btn-rail-sync-bl")?.addEventListener("click", async () => {
      try {
        const sendAgent = fps.pings.length > 0;
        const res = await apiFetch("/api/shots/export", {
          method: "POST",
          json: buildExportPayload({
            action: "sync_blender",
            send_brief_to_agent: sendAgent,
            task: fps.pings.map((p, i) => `[P${i + 1}] ${p.note}`).join("; "),
          }),
        });
        await refreshShotsList();
        const inp = el("agent-input");
        if (!sendAgent && inp && !inp.value.includes(res.brief_rel)) {
          inp.value = `@${res.brief_rel} `;
        }
        showToast(
          `Synced DirectorRailCam + ${res.brief.voxels_count} voxels + ${res.brief_rel} to Blender 5.3!`
        );
      } catch (err) {
        showToast(err.message, true);
      }
    });

    return {
      load: loadSplatUrl,
      startBlankSandbox,
      resetCamera,
      pause: () => {
        running = false;
        if (document.pointerLockElement === canvas) document.exitPointerLock();
        el("fps-crosshair-wrap")?.classList.add("hidden");
        el("fps-radar-canvas")?.classList.add("hidden");
        el("fps-rail-topbar")?.classList.add("hidden");
      },
    };
  }

  async function openSplatInTheater(job) {
    if (!job || !job.splat_url) return;
    state.stageItem = null;
    state.activeSplatJob = job;

    el("stage-title").textContent = `3D Splat · ${job.job_id}`;
    el("theater-empty").classList.add("hidden");
    el("theater-video").pause();
    el("theater-audio").pause();
    el("theater-video").classList.add("hidden");
    el("theater-image").classList.add("hidden");
    el("theater-audio").classList.add("hidden");

    if (window._bbStopComparator) window._bbStopComparator();
    el("theater-compare-wrap").classList.add("hidden");

    const sc = el("theater-splat-canvas");
    const hud = el("theater-splat-hud");
    sc.classList.remove("hidden");
    hud.classList.remove("hidden");

    el("theater-actions").classList.remove("hidden");
    el("theater-meta").textContent =
      `3DGS SPLAT · ${Number(job.n_gaussians || 0).toLocaleString()} pts` +
      (job.psnr ? ` · PSNR ${Number(job.psnr).toFixed(2)} dB` : "") +
      (job.cost_usd ? ` · $${Number(job.cost_usd).toFixed(4)}` : "");
    el("btn-stage-compare").classList.toggle("hidden", !job.cameras_rel);
    el("btn-stage-to-splat").classList.add("hidden");
    el("btn-stage-import-bl").classList.toggle("hidden", !job.ply_rel);
    el("btn-stage-use-vid").classList.add("hidden");
    el("btn-stage-ref-vid").classList.add("hidden");
    el("btn-stage-use-first").classList.add("hidden");
    el("btn-stage-at-agent").classList.add("hidden");

    const dl = el("link-stage-download");
    dl.classList.remove("hidden");
    dl.href = `${withTokenUrl(job.splat_url)}&download=1`;
    dl.download = `${job.job_id}.splat`;

    if (!window._bbSplatViewer) {
      window._bbSplatViewer = createSplatWebGLViewer(sc, el("splat-hud-info"));
      el("btn-splat-reset-cam").addEventListener("click", () => {
        if (window._bbSplatViewer) window._bbSplatViewer.resetCamera();
      });
    }
    if (window._bbSplatViewer) {
      try {
        await window._bbSplatViewer.load(withTokenUrl(job.splat_url), job.job_id);
      } catch (err) {
        showToast(`3D Viewer error: ${err.message}`, true);
      }
    }
  }

  if (el("btn-stage-compare")) {
    el("btn-stage-compare").addEventListener("click", () => {
      if (state.activeSplatJob) openComparatorForJob(state.activeSplatJob);
    });
  }

  // =========================================================================
  // PHASE 5A: `tt-splat` TELEMETRY PLOTS, STAGE BAR, SCRUBBER & SURGERIES
  // =========================================================================

  const wbState = {
    jobId: null,
    paused: false,
    ticks: [],
    densify: [],
    surgeries: [],
    snapshots: [],
    cameras: [],
    hoverIdx: -1,
  };

  const STAGE_COLORS = [
    { key: "fwd_raster_ms", label: "raster fwd", color: "#38bdf8" },
    { key: "loss_ssim_ms", label: "loss+SSIM", color: "#94a3b8" },
    { key: "bwd_ms", label: "raster bwd", color: "#fb923c" },
    { key: "adam_ms", label: "Adam", color: "#22d3ee" },
    { key: "densify_ms", label: "densify", color: "#4ade80" },
    { key: "host_ms", label: "host", color: "#64748b" },
  ];

  state.exportedShots = [];

  async function refreshShotsList() {
    try {
      const res = await apiFetch("/api/shots/list");
      if (Array.isArray(res.shots)) {
        state.exportedShots = res.shots;
        populateKnownCamsDropdown();
      }
    } catch (_e) {}
  }

  function populateKnownCamsDropdown() {
    const sel = el("sel-splat-known-cams");
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML =
      `<option value="">— Auto Solve Poses (VGGT / COLMAP) —</option>` +
      `<option value="__blender_active__">🎥 Active Blender 5.3 Camera (Zero-Drift Lock / Prior)</option>`;
    (state.exportedShots || []).forEach((sh) => {
      const o = document.createElement("option");
      o.value = `shot:${sh.shot_id}`;
      o.textContent = `🎮 FPS Rail: ${sh.shot_id} (${sh.cam_nodes_count || 0} nodes, ${sh.duration_sec}s)`;
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

  function renderStageBreakdownBar(timings) {
    const wrap = el("splat-stage-wrap");
    const bar = el("splat-stage-bar");
    const leg = el("splat-stage-legend");
    if (!timings || !timings.fwd_raster_ms) {
      wrap.classList.add("hidden");
      return;
    }
    wrap.classList.remove("hidden");
    const total = Math.max(0.001, STAGE_COLORS.reduce((s, it) => s + Number(timings[it.key] || 0), 0));
    bar.innerHTML = STAGE_COLORS.map((it) => {
      const v = Number(timings[it.key] || 0);
      const pct = Math.max(0, (v / total) * 100);
      return `<div class="splat-stage-seg" style="width:${pct.toFixed(1)}%;background:${it.color}" title="${it.label}: ${v.toFixed(2)} ms (${pct.toFixed(0)}%)"></div>`;
    }).join("");
    leg.innerHTML = STAGE_COLORS.map((it) => {
      const v = Number(timings[it.key] || 0);
      return `<span><i style="color:${it.color}">●</i> ${it.label} <b>${v.toFixed(2)}ms</b></span>`;
    }).join("") + `<span>total <b>${Number(timings.step_ms || total).toFixed(2)}ms</b></span>`;
  }

  function drawTelemetryCharts() {
    const wrap = el("splat-charts-wrap");
    const cMain = el("canvas-splat-loss-psnr");
    const cCnt = el("canvas-splat-count");
    const ticks = wbState.ticks;
    if (!ticks.length || !cMain || !cCnt) {
      wrap.classList.add("hidden");
      return;
    }
    wrap.classList.remove("hidden");

    const ctx1 = cMain.getContext("2d");
    const W = cMain.width, H = cMain.height;
    ctx1.clearRect(0, 0, W, H);

    const padL = 38, padR = 38, padT = 10, padB = 18;
    const pw = W - padL - padR, ph = H - padT - padB;
    const minStep = ticks[0].step;
    const maxStep = Math.max(minStep + 1, ticks[ticks.length - 1].step);
    const xOf = (st) => padL + ((st - minStep) / (maxStep - minStep)) * pw;

    const losses = ticks.map((t) => Number(t.loss || 0));
    const psnrs = ticks.map((t) => Number(t.psnr || 0));
    const maxLoss = Math.max(0.01, ...losses);
    const minPsnr = Math.min(...psnrs, 12);
    const maxPsnr = Math.max(minPsnr + 2, ...psnrs);

    // Grid lines
    ctx1.strokeStyle = "#182234";
    ctx1.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {
      const gy = padT + (g / 3) * ph;
      ctx1.beginPath(); ctx1.moveTo(padL, gy); ctx1.lineTo(W - padR, gy); ctx1.stroke();
    }

    // Vertical densify & surgery markers
    ctx1.save();
    ctx1.setLineDash([3, 3]);
    wbState.densify.forEach((d) => {
      if (d.step >= minStep && d.step <= maxStep) {
        const dx = xOf(d.step);
        ctx1.strokeStyle = "rgba(74, 222, 128, 0.28)";
        ctx1.beginPath(); ctx1.moveTo(dx, padT); ctx1.lineTo(dx, H - padB); ctx1.stroke();
      }
    });
    wbState.surgeries.forEach((s) => {
      if (s.step >= minStep && s.step <= maxStep) {
        const sx = xOf(s.step);
        ctx1.strokeStyle = "rgba(250, 204, 21, 0.75)";
        ctx1.beginPath(); ctx1.moveTo(sx, padT); ctx1.lineTo(sx, H - padB); ctx1.stroke();
      }
    });
    ctx1.restore();

    const plotLine = (ctx, vals, vmin, vmax, color, lw = 1.6) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = xOf(ticks[i].step);
        const y = padT + (1.0 - (v - vmin) / Math.max(1e-6, vmax - vmin)) * ph;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    plotLine(ctx1, ticks.map((t) => Number(t.l1 || 0)), 0, maxLoss, "#38bdf8", 1.2);
    plotLine(ctx1, losses, 0, maxLoss, "#fb923c", 1.7);
    plotLine(ctx1, psnrs, minPsnr, maxPsnr, "#4ade80", 1.8);

    // Axis labels
    ctx1.fillStyle = "#fb923c";
    ctx1.font = "10px monospace";
    ctx1.fillText(maxLoss.toFixed(3), 2, padT + 8);
    ctx1.fillText("0.00", 6, H - padB);
    ctx1.fillStyle = "#4ade80";
    ctx1.fillText(`${maxPsnr.toFixed(1)}dB`, W - padR + 3, padT + 8);
    ctx1.fillText(`${minPsnr.toFixed(0)}dB`, W - padR + 3, H - padB);
    ctx1.fillStyle = "#8b98ad";
    ctx1.fillText(`step ${minStep}`, padL, H - 4);
    ctx1.fillText(`step ${maxStep}`, W - padR - 54, H - 4);

    if (wbState.hoverIdx >= 0 && wbState.hoverIdx < ticks.length) {
      const ht = ticks[wbState.hoverIdx];
      const hx = xOf(ht.step);
      ctx1.strokeStyle = "#e2e8f0";
      ctx1.lineWidth = 1;
      ctx1.beginPath(); ctx1.moveTo(hx, padT); ctx1.lineTo(hx, H - padB); ctx1.stroke();
      el("lbl-chart-hover").textContent =
        `step ${ht.step} · loss ${Number(ht.loss).toFixed(4)} · L1 ${Number(ht.l1 || 0).toFixed(4)} · SSIM ${Number(ht.ssim || 0).toFixed(3)} · PSNR ${Number(ht.psnr).toFixed(2)}dB · ${Number(ht.n_gaussians).toLocaleString()} pts`;
    }

    // Gaussian count chart
    const ctx2 = cCnt.getContext("2d");
    const W2 = cCnt.width, H2 = cCnt.height;
    ctx2.clearRect(0, 0, W2, H2);
    const ph2 = H2 - 16;
    const cnts = ticks.map((t) => Number(t.n_gaussians || 0));
    const minC = Math.min(...cnts) * 0.92;
    const maxC = Math.max(minC + 100, ...cnts) * 1.03;

    ctx2.strokeStyle = "#a78bfa";
    ctx2.lineWidth = 1.6;
    ctx2.beginPath();
    cnts.forEach((v, i) => {
      const x = padL + ((ticks[i].step - minStep) / (maxStep - minStep)) * (W2 - padL - padR);
      const y = 6 + (1.0 - (v - minC) / Math.max(1, maxC - minC)) * ph2;
      if (i === 0) ctx2.moveTo(x, y);
      else ctx2.lineTo(x, y);
    });
    ctx2.stroke();
    ctx2.fillStyle = "#a78bfa";
    ctx2.font = "10px monospace";
    ctx2.fillText(`${Math.round(maxC / 1000)}k pts`, 2, 13);
    ctx2.fillText(`${Math.round(minC / 1000)}k`, 2, H2 - 4);
  }

  el("canvas-splat-loss-psnr").addEventListener("mousemove", (e) => {
    if (!wbState.ticks.length) return;
    const rect = e.target.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - 30) / Math.max(40, rect.width - 60)));
    wbState.hoverIdx = Math.min(wbState.ticks.length - 1, Math.round(frac * (wbState.ticks.length - 1)));
    drawTelemetryCharts();
  });

  async function selectWorkbenchJob(jobId, resetLive = false) {
    wbState.jobId = jobId;
    populateKnownCamsDropdown();
    el("splat-live-card").classList.remove("hidden");
    if (resetLive) {
      wbState.ticks = [];
      wbState.densify = [];
      wbState.surgeries = [];
      wbState.snapshots = [];
      drawTelemetryCharts();
      return;
    }
    try {
      const data = await apiFetch(`/api/splats/telemetry/${encodeURIComponent(jobId)}`);
      wbState.ticks = data.ticks || [];
      wbState.densify = data.densify_events || [];
      wbState.surgeries = data.surgery_events || [];
      wbState.snapshots = data.snapshots || [];
      wbState.cameras = data.cameras || [];

      const fSel = el("sel-wb-focus-cam");
      fSel.innerHTML = `<option value="">🎯 Focus: All Views (${wbState.cameras.length})</option>`;
      wbState.cameras.forEach((cn) => {
        const o = document.createElement("option");
        o.value = cn;
        o.textContent = `🎯 ${cn}`;
        fSel.appendChild(o);
      });

      const lastTick = wbState.ticks[wbState.ticks.length - 1];
      if (lastTick) {
        el("splat-live-title").textContent = `${jobId} · Telemetry (${lastTick.step} steps)`;
        el("splat-live-metrics").textContent =
          `PSNR ${Number(lastTick.psnr || 0).toFixed(2)} dB · SSIM ${Number(lastTick.ssim || 0).toFixed(3)} · ${Number(lastTick.n_gaussians || 0).toLocaleString()} pts`;
        el("splat-prog-fill").style.width = "100%";
        renderStageBreakdownBar(lastTick.stage_timings);
      }
      drawTelemetryCharts();

      const snapRow = el("splat-snap-row");
      if (wbState.snapshots.length > 0) {
        snapRow.classList.remove("hidden");
        const rng = el("rng-splat-snap");
        rng.max = String(wbState.snapshots.length - 1);
        rng.value = String(wbState.snapshots.length - 1);
        const sLast = wbState.snapshots[wbState.snapshots.length - 1];
        el("lbl-splat-snap").textContent = `step ${sLast.step}`;
        el("splat-live-preview-wrap").classList.remove("hidden");
        el("splat-live-img").src = withTokenUrl(sLast.url);
      } else {
        snapRow.classList.add("hidden");
      }
    } catch (err) {
      showToast(`Telemetry load failed: ${err.message}`, true);
    }
  }
  window._bbSelectWorkbenchJob = selectWorkbenchJob;

  window._bbPushSplatTelemetry = (ev) => {
    if (!wbState.jobId) wbState.jobId = ev.job_id;
    if (ev.job_id !== wbState.jobId) return;
    if (ev.event === "telemetry") {
      wbState.ticks.push(ev);
      if (ev.stage_timings) renderStageBreakdownBar(ev.stage_timings);
      drawTelemetryCharts();
    } else if (ev.event === "densify") {
      wbState.densify.push(ev);
    } else if (ev.event === "surgery") {
      wbState.surgeries.push(ev);
      showToast(`Surgery @ step ${ev.step}: removed ${ev.removed || 0} Gaussians`);
    } else if (ev.event === "checkpoint_saved") {
      showToast(`Saved cloud checkpoint @ step ${ev.step}`);
    }
  };

  el("rng-splat-snap").addEventListener("input", (e) => {
    const idx = Number(e.target.value || 0);
    const snap = wbState.snapshots[idx];
    if (!snap) return;
    el("lbl-splat-snap").textContent = `step ${snap.step}`;
    el("splat-live-preview-wrap").classList.remove("hidden");
    el("splat-live-img").classList.remove("hidden");
    el("splat-live-diff").classList.add("hidden");
    el("splat-live-img").src = withTokenUrl(snap.url);
  });

  // Live Mid-Train Controls & Surgical Workbench Actions
  function activeWorkbenchJobId() {
    if (wbState.jobId) return wbState.jobId;
    if (state.activeSplatJob) return state.activeSplatJob.job_id;
    const first = Array.from(state.splatJobs.keys())[0];
    return first || null;
  }

  el("btn-wb-pause").addEventListener("click", async () => {
    const jid = activeWorkbenchJobId();
    if (!jid) return;
    wbState.paused = !wbState.paused;
    el("btn-wb-pause").textContent = wbState.paused ? "▶ Resume" : "⏸ Pause";
    try {
      await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, {
        method: "POST",
        json: { paused: wbState.paused },
      });
      showToast(wbState.paused ? `Paused ${jid}` : `Resumed ${jid}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-wb-saveckpt").addEventListener("click", async () => {
    const jid = activeWorkbenchJobId();
    if (!jid) return;
    try {
      await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, {
        method: "POST",
        json: { save_checkpoint: true },
      });
      showToast(`Requested immediate cloud checkpoint for ${jid}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-wb-stop").addEventListener("click", async () => {
    const jid = activeWorkbenchJobId();
    if (!jid) return;
    try {
      await apiFetch(`/api/splats/cancel/${encodeURIComponent(jid)}`, { method: "POST", json: {} });
      showToast(`Sent stop & export to ${jid}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("sel-wb-focus-cam").addEventListener("change", async (e) => {
    const jid = activeWorkbenchJobId();
    if (!jid) return;
    const cam = e.target.value;
    try {
      await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, {
        method: "POST",
        json: { focus_camera: cam },
      });
      showToast(cam ? `Locked focus to ${cam}` : "Cleared camera focus (round-robin)");
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-wb-push-knobs").addEventListener("click", async () => {
    const jid = activeWorkbenchJobId();
    if (!jid) return;
    const lam = Number(el("inp-splat-dssim").value || 0.2);
    const opa = Number(el("inp-splat-opareg").value || 0.0005);
    try {
      await apiFetch(`/api/splats/control/${encodeURIComponent(jid)}`, {
        method: "POST",
        json: { lambda_dssim: lam, opa_reg: opa },
      });
      showToast(`Pushed live λ_dssim=${lam}, opa_reg=${opa} to ${jid}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-surg-reset-opa").addEventListener("click", async () => {
    const jid = activeWorkbenchJobId();
    if (!jid) return showToast("Select a splat job first", true);
    try {
      const res = await apiFetch("/api/splats/resume", {
        method: "POST",
        json: { job_id: jid, add_steps: 1500, reset_opacities: 0.015, gpu: el("sel-splat-gpu").value },
      });
      if (res.job) handleSplatJobUpdate(res.job);
      showToast(`Reset opacities on ${jid} + 1,500 healing steps`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-surg-clamp-scale").addEventListener("click", async () => {
    const jid = activeWorkbenchJobId();
    if (!jid) return showToast("Select a splat job first", true);
    try {
      const res = await apiFetch("/api/splats/resume", {
        method: "POST",
        json: { job_id: jid, add_steps: 1000, clamp_log_scale: -1.6, prune_opa: 0.03, gpu: el("sel-splat-gpu").value },
      });
      if (res.job) handleSplatJobUpdate(res.job);
      showToast(`Clamped overgrown spikes on ${jid} + 1,000 polish steps`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-cropbox-spawn").addEventListener("click", async () => {
    try {
      const res = await apiFetch("/api/splats/crop_box", { method: "POST", json: { action: "spawn" } });
      const b = res.crop_box || {};
      showToast(`Spawned '${b.object}' in Blender 5.3 — position/scale it around floaters or hero subject!`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  async function runCropBoxHeal(action) {
    const jid = activeWorkbenchJobId();
    if (!jid) return showToast("Select a splat job first", true);
    try {
      const res = await apiFetch("/api/splats/crop_box", {
        method: "POST",
        json: { action, job_id: jid, add_steps: 1500, gpu: el("sel-splat-gpu").value },
      });
      if (res.job) handleSplatJobUpdate(res.job);
      showToast(`Launched ${action === "keep_heal" ? "Keep-Inside-Box" : "Cull-Inside-Box"} & Heal on ${jid}!`);
    } catch (err) {
      showToast(err.message, true);
    }
  }
  el("btn-cropbox-cull").addEventListener("click", () => runCropBoxHeal("cull_heal"));
  el("btn-cropbox-keep").addEventListener("click", () => runCropBoxHeal("keep_heal"));

  el("btn-build-proxy").addEventListener("click", async () => {
    const jid = activeWorkbenchJobId();
    if (!jid) return showToast("Select a splat job first", true);
    showToast(`Extracting Surface-Nets proxy_collision.glb for ${jid}…`);
    try {
      const res = await apiFetch(`/api/splats/proxy/${encodeURIComponent(jid)}`, {
        method: "POST",
        json: { import_blender: true, grid_res: 48 },
      });
      showToast(`Built ${res.proxy_glb_rel} & imported into Blender 5.3!`);
      refreshSplats();
    } catch (err) {
      showToast(err.message, true);
    }
  });

  // =========================================================================
  // PHASE 5B: SIDE-BY-SIDE CLIP COMPARATOR & 3D CAMERA PATH / ROTATION HUD
  // =========================================================================

  const cmpState = {
    active: false,
    job: null,
    traj: null,
    curFrameIdx: 0,
    orbitYaw: -0.75,
    orbitPitch: 0.48,
    dragging: false,
    lastX: 0,
    lastY: 0,
    rafId: 0,
  };

  function stopComparator() {
    cmpState.active = false;
    if (cmpState.rafId) cancelAnimationFrame(cmpState.rafId);
    const vL = el("vid-cmp-left");
    const vR = el("vid-cmp-right");
    if (vL) vL.pause();
    if (vR) vR.pause();
  }
  window._bbStopComparator = stopComparator;

  window._bbOnSplatRenderProgress = (ev) => {
    if (!cmpState.active || !cmpState.job || ev.job_id !== cmpState.job.job_id) return;
    if (ev.event === "render_progress") {
      el("btn-cmp-farm-modal").textContent = `☁️ Rendering ${ev.frame}/${ev.total_frames}…`;
    }
  };

  function buildClipChoicesForJob(job, data) {
    const leftOpts = [];
    const rightOpts = [];
    if (data.source_video_url) {
      leftOpts.push({ label: `🎬 Original (${data.source_video_rel || "source"})`, url: data.source_video_url });
      rightOpts.push({ label: `🎬 Original (${data.source_video_rel || "source"})`, url: data.source_video_url });
    }
    if (data.modal_render_url) {
      rightOpts.unshift({ label: "☁️ Modal H100 3DGS Render", url: data.modal_render_url });
      leftOpts.push({ label: "☁️ Modal H100 3DGS Render", url: data.modal_render_url });
    }
    if (data.blender_render_url) {
      rightOpts.push({ label: "🧊 Blender 5.3 EEVEE Render", url: data.blender_render_url });
      leftOpts.push({ label: "🧊 Blender 5.3 EEVEE Render", url: data.blender_render_url });
    }
    state.mediaItems.filter((m) => m.kind === "video").slice(0, 12).forEach((m) => {
      leftOpts.push({ label: `📁 ${m.name}`, url: m.url });
      rightOpts.push({ label: `📁 ${m.name}`, url: m.url });
    });
    return { leftOpts, rightOpts };
  }

  function applyCmpVideoSrc(vidEl, badgeEl, phEl, opt) {
    if (!opt || !opt.url) {
      vidEl.pause();
      vidEl.removeAttribute("src");
      if (phEl) phEl.classList.remove("hidden");
      badgeEl.textContent = "No render yet";
      return;
    }
    if (phEl) phEl.classList.add("hidden");
    badgeEl.textContent = opt.label;
    vidEl.src = withTokenUrl(opt.url);
    vidEl.playbackRate = Number(el("sel-cmp-speed").value || 1);
    vidEl.load();
  }

  async function openComparatorForJob(job) {
    stopSplatViewer();
    state.activeSplatJob = job;
    cmpState.job = job;
    cmpState.active = true;

    el("stage-title").textContent = `Compare & Camera Path · ${job.job_id}`;
    el("theater-empty").classList.add("hidden");
    el("theater-video").classList.add("hidden");
    el("theater-image").classList.add("hidden");
    el("theater-audio").classList.add("hidden");
    el("theater-compare-wrap").classList.remove("hidden");
    el("theater-actions").classList.remove("hidden");
    el("btn-stage-import-bl").classList.toggle("hidden", !job.ply_rel);

    try {
      const data = await apiFetch(`/api/splats/trajectory/${encodeURIComponent(job.job_id)}`);
      cmpState.traj = data.trajectory;
      cmpState.curFrameIdx = 0;

      const { leftOpts, rightOpts } = buildClipChoicesForJob(job, data);
      const selL = el("sel-cmp-left");
      const selR = el("sel-cmp-right");
      selL.innerHTML = leftOpts.length
        ? leftOpts.map((o, i) => `<option value="${i}">${escapeHtml(o.label)}</option>`).join("")
        : `<option value="">— No source clip —</option>`;
      selR.innerHTML = rightOpts.length
        ? rightOpts.map((o, i) => `<option value="${i}">${escapeHtml(o.label)}</option>`).join("")
        : `<option value="">— Click Farm Modal Render or Render Blender —</option>`;

      selL._opts = leftOpts;
      selR._opts = rightOpts;

      applyCmpVideoSrc(el("vid-cmp-left"), el("lbl-cmp-left"), null, leftOpts[0]);
      // Prefer Modal or Blender render on right; if neither rendered yet, show placeholder prompt!
      const hasRecreated = !!(data.modal_render_url || data.blender_render_url);
      if (hasRecreated) {
        applyCmpVideoSrc(el("vid-cmp-right"), el("lbl-cmp-right"), el("cmp-right-placeholder"), rightOpts[0]);
      } else {
        applyCmpVideoSrc(el("vid-cmp-right"), el("lbl-cmp-right"), el("cmp-right-placeholder"), null);
      }

      // Render scrubber range & gold keyframe marker ticks
      const nF = cmpState.traj.total_frames;
      const scrub = el("rng-cmp-scrub");
      scrub.max = String(Math.max(0, nF - 1));
      scrub.value = "0";
      const ticksBox = el("cmp-keyframe-ticks");
      ticksBox.innerHTML = (cmpState.traj.keyframe_markers || []).map((km) => {
        const pct = nF > 1 ? ((km.frame - 1) / (nF - 1)) * 100 : 0;
        return `<span class="cmp-kf-tick" style="left:${pct.toFixed(2)}%" title="Solved Keyframe #${km.key_index + 1} (${km.image_name} @ ${km.pts_sec}s)"></span>`;
      }).join("");

      el("theater-meta").textContent =
        `${nF} frames @ ${cmpState.traj.fps} fps (${cmpState.traj.duration_sec}s) · ${cmpState.traj.solved_keyframes} keyframes · path ${cmpState.traj.path_length_m}m`;

      drawComparatorCanvases();
      startComparatorLoop();
    } catch (err) {
      showToast(`Trajectory load failed: ${err.message}`, true);
    }
  }

  el("sel-cmp-left").addEventListener("change", (e) => {
    const opts = e.target._opts || [];
    applyCmpVideoSrc(el("vid-cmp-left"), el("lbl-cmp-left"), null, opts[Number(e.target.value)]);
  });
  el("sel-cmp-right").addEventListener("change", (e) => {
    const opts = e.target._opts || [];
    applyCmpVideoSrc(el("vid-cmp-right"), el("lbl-cmp-right"), el("cmp-right-placeholder"), opts[Number(e.target.value)]);
  });

  function masterCmpVideo() {
    const vL = el("vid-cmp-left");
    const vR = el("vid-cmp-right");
    return vL && vL.src ? vL : vR && vR.src ? vR : null;
  }

  function seekBothVideosToFrame(fi) {
    if (!cmpState.traj) return;
    const nF = cmpState.traj.total_frames;
    const clamped = Math.max(0, Math.min(nF - 1, fi));
    cmpState.curFrameIdx = clamped;
    const tSec = clamped / Math.max(1, cmpState.traj.fps);
    [el("vid-cmp-left"), el("vid-cmp-right")].forEach((v) => {
      if (v && v.src && Number.isFinite(v.duration)) {
        v.currentTime = Math.min(v.duration - 0.001, tSec);
      }
    });
    el("rng-cmp-scrub").value = String(clamped);
    drawComparatorCanvases();
  }

  el("btn-cmp-play").addEventListener("click", () => {
    const vL = el("vid-cmp-left");
    const vR = el("vid-cmp-right");
    const m = masterCmpVideo();
    if (!m) {
      // Even with no MP4 loaded yet, animate playhead through the 3D camera path!
      cmpState._virtualPlay = !cmpState._virtualPlay;
      el("btn-cmp-play").textContent = cmpState._virtualPlay ? "⏸ Pause" : "▶ Play";
      return;
    }
    if (m.paused) {
      if (vL.src && vR.src) vR.currentTime = vL.currentTime;
      if (vL.src) vL.play().catch(() => {});
      if (vR.src) vR.play().catch(() => {});
      el("btn-cmp-play").textContent = "⏸ Pause";
    } else {
      vL.pause();
      vR.pause();
      el("btn-cmp-play").textContent = "▶ Play";
    }
  });

  el("btn-cmp-prevf").addEventListener("click", () => {
    el("vid-cmp-left").pause();
    el("vid-cmp-right").pause();
    cmpState._virtualPlay = false;
    el("btn-cmp-play").textContent = "▶ Play";
    seekBothVideosToFrame(cmpState.curFrameIdx - 1);
  });
  el("btn-cmp-nextf").addEventListener("click", () => {
    el("vid-cmp-left").pause();
    el("vid-cmp-right").pause();
    cmpState._virtualPlay = false;
    el("btn-cmp-play").textContent = "▶ Play";
    seekBothVideosToFrame(cmpState.curFrameIdx + 1);
  });
  el("rng-cmp-scrub").addEventListener("input", (e) => {
    seekBothVideosToFrame(Number(e.target.value || 0));
  });
  el("sel-cmp-speed").addEventListener("change", (e) => {
    const r = Number(e.target.value || 1);
    el("vid-cmp-left").playbackRate = r;
    el("vid-cmp-right").playbackRate = r;
  });

  el("btn-cmp-farm-modal").addEventListener("click", async () => {
    if (!cmpState.job) return;
    const btn = el("btn-cmp-farm-modal");
    btn.disabled = true;
    btn.textContent = "☁️ Farming on H100…";
    try {
      const res = await apiFetch("/api/splats/render", {
        method: "POST",
        json: { job_id: cmpState.job.job_id, mode: "modal", gpu: el("sel-splat-gpu").value || "H100" },
      });
      showToast(
        `Modal H100 rendered ${res.frames} frames @ ${res.render_fps} fps in ${res.wall_s}s ($${Number(res.cost_usd || 0).toFixed(4)})!`
      );
      await refreshSplats();
      await openComparatorForJob(state.splatJobs.get(cmpState.job.job_id) || cmpState.job);
    } catch (err) {
      showToast(`Modal render failed: ${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "☁️ Farm Modal Render";
    }
  });

  el("btn-cmp-render-blender").addEventListener("click", async () => {
    if (!cmpState.job) return;
    const btn = el("btn-cmp-render-blender");
    btn.disabled = true;
    btn.textContent = "🧊 Rendering EEVEE…";
    try {
      const res = await apiFetch("/api/splats/render", {
        method: "POST",
        json: { job_id: cmpState.job.job_id, mode: "blender" },
      });
      showToast(`Blender 5.3 rendered ${res.frames} matching frames in ${res.wall_s}s!`);
      await refreshSplats();
      await openComparatorForJob(state.splatJobs.get(cmpState.job.job_id) || cmpState.job);
    } catch (err) {
      showToast(`Blender render failed: ${err.message}`, true);
    } finally {
      btn.disabled = false;
      btn.textContent = "🧊 Render Blender";
    }
  });

  // Orbit drag on 3D Camera Trajectory canvas + click-to-seek on Angles canvas
  const cPath3d = el("canvas-cmp-path3d");
  cPath3d.addEventListener("mousedown", (e) => {
    cmpState.dragging = true;
    cmpState.lastX = e.clientX;
    cmpState.lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => { cmpState.dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!cmpState.dragging || !cmpState.active) return;
    cmpState.orbitYaw -= (e.clientX - cmpState.lastX) * 0.01;
    cmpState.orbitPitch = Math.max(-1.35, Math.min(1.42, cmpState.orbitPitch + (e.clientY - cmpState.lastY) * 0.01));
    cmpState.lastX = e.clientX;
    cmpState.lastY = e.clientY;
    drawComparatorCanvases();
  });

  el("canvas-cmp-angles").addEventListener("click", (e) => {
    if (!cmpState.traj) return;
    const rect = e.target.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left - 24) / Math.max(30, rect.width - 34)));
    seekBothVideosToFrame(Math.round(frac * (cmpState.traj.total_frames - 1)));
  });

  function startComparatorLoop() {
    if (cmpState.rafId) cancelAnimationFrame(cmpState.rafId);
    let lastVirt = performance.now();

    function tick(now) {
      if (!cmpState.active || !cmpState.traj) return;
      const vL = el("vid-cmp-left");
      const vR = el("vid-cmp-right");
      const m = masterCmpVideo();
      const nF = cmpState.traj.total_frames;
      const fps = cmpState.traj.fps || 24;

      if (m && !m.paused) {
        // Keep secondary video locked within 45 ms of master
        if (vL.src && vR.src && m === vL && Math.abs(vR.currentTime - vL.currentTime) > 0.045) {
          vR.currentTime = vL.currentTime;
        }
        const fi = Math.max(0, Math.min(nF - 1, Math.floor(m.currentTime * fps)));
        if (fi !== cmpState.curFrameIdx) {
          cmpState.curFrameIdx = fi;
          el("rng-cmp-scrub").value = String(fi);
          drawComparatorCanvases();
        }
      } else if (cmpState._virtualPlay) {
        const dt = (now - lastVirt) / 1000.0;
        const rate = Number(el("sel-cmp-speed").value || 1);
        if (dt >= 1.0 / (fps * rate)) {
          lastVirt = now;
          cmpState.curFrameIdx = (cmpState.curFrameIdx + 1) % nF;
          el("rng-cmp-scrub").value = String(cmpState.curFrameIdx);
          drawComparatorCanvases();
        }
      }
      cmpState.rafId = requestAnimationFrame(tick);
    }
    cmpState.rafId = requestAnimationFrame(tick);
  }

  function drawComparatorCanvases() {
    const traj = cmpState.traj;
    if (!traj || !traj.frames || !traj.frames.length) return;
    const frames = traj.frames;
    const fi = Math.max(0, Math.min(frames.length - 1, cmpState.curFrameIdx));
    const cur = frames[fi];

    el("lbl-cmp-readout").textContent =
      `f ${String(cur.frame).padStart(3, "0")}/${frames.length} · t=${cur.pts_sec.toFixed(2)}s` +
      (cur.is_keyframe ? ` [★ KF #${cur.keyframe_index + 1}]` : "") +
      ` · XYZ=(${cur.location[0].toFixed(2)}, ${cur.location[1].toFixed(2)}, ${cur.location[2].toFixed(2)})m` +
      ` · Yaw ${cur.yaw_deg.toFixed(1)}° Pitch ${cur.pitch_deg.toFixed(1)}° Roll ${cur.roll_deg.toFixed(1)}° · ${cur.lens_mm.toFixed(0)}mm`;

    // --- 1. 3D Orbit Camera Trajectory & Frustum Canvas ---
    const c3 = el("canvas-cmp-path3d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w3 = Math.max(120, Math.floor(c3.clientWidth * dpr));
    const h3 = Math.max(90, Math.floor(c3.clientHeight * dpr));
    if (c3.width !== w3 || c3.height !== h3) { c3.width = w3; c3.height = h3; }
    const g3 = c3.getContext("2d");
    g3.clearRect(0, 0, w3, h3);

    const bmin = traj.bounds_min, bmax = traj.bounds_max;
    const cx = 0.5 * (bmin[0] + bmax[0]);
    const cy = 0.5 * (bmin[1] + bmax[1]);
    const cz = 0.5 * Math.min(0, bmin[2]) + 0.5 * bmax[2];
    const span = Math.max(1.2, Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]) * 0.85);

    const cyaw = Math.cos(cmpState.orbitYaw), syaw = Math.sin(cmpState.orbitYaw);
    const cp = Math.cos(cmpState.orbitPitch), sp = Math.sin(cmpState.orbitPitch);
    const eyeDist = span * 2.1;
    const eye = [cx + eyeDist * cp * cyaw, cy + eyeDist * cp * syaw, cz + eyeDist * sp];
    let fx = cx - eye[0], fy = cy - eye[1], fz = cz - eye[2];
    const fn = Math.hypot(fx, fy, fz) || 1;
    fx /= fn; fy /= fn; fz /= fn;
    let rx = fy, ry = -fx, rz = 0;
    const rn = Math.hypot(rx, ry) || 1;
    rx /= rn; ry /= rn;
    const ux = ry * fz, uy = -rx * fz, uz = rx * fy - ry * fx;
    const focal = Math.min(w3, h3) * 1.15;

    function proj(pt) {
      const dx = pt[0] - eye[0], dy = pt[1] - eye[1], dz = pt[2] - eye[2];
      const zc = dx * fx + dy * fy + dz * fz;
      if (zc < 0.08) return null;
      const xc = dx * rx + dy * ry + dz * rz;
      const yc = dx * ux + dy * uy + dz * uz;
      return [w3 * 0.5 + (xc / zc) * focal, h3 * 0.5 - (yc / zc) * focal, zc];
    }

    // Z=0 Ground Grid
    g3.strokeStyle = "rgba(51, 65, 85, 0.45)";
    g3.lineWidth = 1;
    const gridR = Math.max(2, Math.ceil(span));
    const stepG = Math.max(0.5, gridR / 4);
    for (let gv = -gridR; gv <= gridR + 1e-4; gv += stepG) {
      const a = proj([cx + gv, cy - gridR, 0]), b = proj([cx + gv, cy + gridR, 0]);
      if (a && b) { g3.beginPath(); g3.moveTo(a[0], a[1]); g3.lineTo(b[0], b[1]); g3.stroke(); }
      const c = proj([cx - gridR, cy + gv, 0]), d = proj([cx + gridR, cy + gv, 0]);
      if (c && d) { g3.beginPath(); g3.moveTo(c[0], c[1]); g3.lineTo(d[0], d[1]); g3.stroke(); }
    }

    // Trajectory Ribbon
    g3.lineWidth = 2.0;
    for (let i = 1; i < frames.length; i++) {
      const p0 = proj(frames[i - 1].location);
      const p1 = proj(frames[i].location);
      if (!p0 || !p1) continue;
      g3.strokeStyle = i <= fi ? "#38bdf8" : "rgba(148, 163, 184, 0.38)";
      g3.beginPath(); g3.moveTo(p0[0], p0[1]); g3.lineTo(p1[0], p1[1]); g3.stroke();
    }

    // Solved Keyframe Markers + Mini Optical Ticks
    const coneScale = span * 0.11;
    frames.forEach((f) => {
      if (!f.is_keyframe) return;
      const pc = proj(f.location);
      const tip = proj([
        f.location[0] + f.forward[0] * coneScale * 0.7,
        f.location[1] + f.forward[1] * coneScale * 0.7,
        f.location[2] + f.forward[2] * coneScale * 0.7,
      ]);
      if (pc) {
        g3.fillStyle = "#facc15";
        g3.beginPath(); g3.arc(pc[0], pc[1], 2.6, 0, Math.PI * 2); g3.fill();
        if (tip) {
          g3.strokeStyle = "rgba(250, 204, 21, 0.5)";
          g3.lineWidth = 1;
          g3.beginPath(); g3.moveTo(pc[0], pc[1]); g3.lineTo(tip[0], tip[1]); g3.stroke();
        }
      }
    });

    // Live Active 3D Camera Frustum Pyramid + Optical Ray
    const L = cur.location, F = cur.forward, R = cur.right, U = cur.up;
    const pEye = proj(L);
    const fLen = coneScale * 1.45, fW = fLen * 0.55, fH = fLen * 0.34;
    const corners3d = [
      [L[0] + F[0] * fLen - R[0] * fW - U[0] * fH, L[1] + F[1] * fLen - R[1] * fW - U[1] * fH, L[2] + F[2] * fLen - R[2] * fW - U[2] * fH],
      [L[0] + F[0] * fLen + R[0] * fW - U[0] * fH, L[1] + F[1] * fLen + R[1] * fW - U[1] * fH, L[2] + F[2] * fLen + R[2] * fW - U[2] * fH],
      [L[0] + F[0] * fLen + R[0] * fW + U[0] * fH, L[1] + F[1] * fLen + R[1] * fW + U[1] * fH, L[2] + F[2] * fLen + R[2] * fW + U[2] * fH],
      [L[0] + F[0] * fLen - R[0] * fW + U[0] * fH, L[1] + F[1] * fLen - R[1] * fW + U[1] * fH, L[2] + F[2] * fLen - R[2] * fW + U[2] * fH],
    ];
    const c2d = corners3d.map(proj);
    if (pEye && c2d.every(Boolean)) {
      g3.strokeStyle = "#ff7b29";
      g3.lineWidth = 1.8;
      c2d.forEach((cp2) => {
        g3.beginPath(); g3.moveTo(pEye[0], pEye[1]); g3.lineTo(cp2[0], cp2[1]); g3.stroke();
      });
      g3.beginPath();
      c2d.forEach((cp2, idx) => { if (idx === 0) g3.moveTo(cp2[0], cp2[1]); else g3.lineTo(cp2[0], cp2[1]); });
      g3.closePath();
      g3.fillStyle = "rgba(255, 123, 41, 0.22)";
      g3.fill();
      g3.stroke();
    }
    g3.fillStyle = "#94a3b8";
    g3.font = `${Math.round(9.5 * dpr)}px monospace`;
    g3.fillText("3D Cam Path (Z=0 grid · ★ keyframes)", 6 * dpr, 12 * dpr);

    // --- 2. Synchronized Camera Rotation (Yaw/Pitch/Roll) & Speed Curves ---
    const cA = el("canvas-cmp-angles");
    const wA = Math.max(120, Math.floor(cA.clientWidth * dpr));
    const hA = Math.max(90, Math.floor(cA.clientHeight * dpr));
    if (cA.width !== wA || cA.height !== hA) { cA.width = wA; cA.height = hA; }
    const gA = cA.getContext("2d");
    gA.clearRect(0, 0, wA, hA);

    const pL = 24 * dpr, pR = 10 * dpr, pT = 16 * dpr, pB = 12 * dpr;
    const aw = wA - pL - pR, ah = hA - pT - pB;
    const yaws = frames.map((f) => f.yaw_deg);
    const pitches = frames.map((f) => f.pitch_deg);
    const rolls = frames.map((f) => f.roll_deg);
    const speeds = frames.map((f) => f.speed_m_s);

    const allAng = [...yaws, ...pitches, ...rolls];
    const minAng = Math.min(...allAng, -5);
    const maxAng = Math.max(...allAng, 5);
    const maxSpd = Math.max(0.5, ...speeds);

    const xF = (i) => pL + (frames.length > 1 ? (i / (frames.length - 1)) * aw : 0);
    const drawCurve = (arr, vmin, vmax, col, lw = 1.5) => {
      gA.strokeStyle = col;
      gA.lineWidth = lw * dpr;
      gA.beginPath();
      arr.forEach((v, i) => {
        const x = xF(i);
        const y = pT + (1.0 - (v - vmin) / Math.max(1e-5, vmax - vmin)) * ah;
        if (i === 0) gA.moveTo(x, y); else gA.lineTo(x, y);
      });
      gA.stroke();
    };

    drawCurve(speeds, 0, maxSpd, "rgba(74, 222, 128, 0.55)", 1.1);
    drawCurve(rolls, minAng, maxAng, "#a78bfa", 1.2);
    drawCurve(pitches, minAng, maxAng, "#fb923c", 1.4);
    drawCurve(yaws, minAng, maxAng, "#38bdf8", 1.6);

    // Playhead cursor
    const px = xF(fi);
    gA.strokeStyle = "#ffffff";
    gA.lineWidth = 1.2 * dpr;
    gA.beginPath(); gA.moveTo(px, pT - 2); gA.lineTo(px, hA - pB); gA.stroke();

    gA.font = `${Math.round(9.5 * dpr)}px monospace`;
    gA.fillStyle = "#38bdf8"; gA.fillText("Yaw°", 6 * dpr, 12 * dpr);
    gA.fillStyle = "#fb923c"; gA.fillText("Pitch°", 38 * dpr, 12 * dpr);
    gA.fillStyle = "#a78bfa"; gA.fillText("Roll°", 80 * dpr, 12 * dpr);
    gA.fillStyle = "#4ade80"; gA.fillText("m/s", 116 * dpr, 12 * dpr);
    gA.fillStyle = "#94a3b8";
    gA.fillText(`${maxAng.toFixed(0)}°`, 2 * dpr, pT + 7 * dpr);
    gA.fillText(`${minAng.toFixed(0)}°`, 2 * dpr, hA - pB);
  }

  // Launch FPS / Voxel / Rail Sandbox on active splat OR blank Z=0 grid!
  el("btn-open-fps-sandbox")?.addEventListener("click", () => {
    if (window._bbStopComparator) window._bbStopComparator();
    el("theater-compare-wrap")?.classList.add("hidden");
    el("theater-empty").classList.add("hidden");
    el("theater-video").pause();
    el("theater-audio").pause();
    el("theater-video").classList.add("hidden");
    el("theater-image").classList.add("hidden");
    el("theater-audio").classList.add("hidden");

    const sc = el("theater-splat-canvas");
    const hud = el("theater-splat-hud");
    sc.classList.remove("hidden");
    hud.classList.remove("hidden");
    el("stage-title").textContent = state.activeSplatJob
      ? `FPS Rail · ${state.activeSplatJob.job_id}`
      : "FPS Rail & Voxel Sandbox (Z=0 Grid)";

    if (!window._bbSplatViewer) {
      window._bbSplatViewer = createSplatWebGLViewer(sc, el("splat-hud-info"));
      el("btn-splat-reset-cam").addEventListener("click", () => {
        if (window._bbSplatViewer) window._bbSplatViewer.resetCamera();
      });
    }
    window._bbSplatViewer.startBlankSandbox();
  });

  // --- Init ---
  renderPromptHistoryDropdown();
  populatePricingDropdowns();
  renderConditioningSlots();
  updatePricingAndInspector();
  updateSplatCostEstimate();
  if (state.studioTab === "splat") switchStudioTab("splat");
  refreshState();
  refreshModels();
  refreshMedia(true);
  refreshSplats();
  refreshShotsList();
  connectSSE();
})();