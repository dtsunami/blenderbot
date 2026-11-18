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
    const sc = el("theater-splat-canvas");
    const hud = el("theater-splat-hud");
    if (sc) sc.classList.add("hidden");
    if (hud) hud.classList.add("hidden");
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

  el("btn-train-splat").addEventListener("click", async () => {
    const payload = {
      source_mode: state.splatSourceMode,
      iterations: Math.max(200, Math.min(30000, Number(el("inp-splat-iters").value || 5000))),
      gpu: el("sel-splat-gpu").value || "H100",
      sh_degree: Number(el("sel-splat-sh").value || 1),
      render_mode: el("sel-splat-rendermode").value || "ellipsoids",
      auto_import: el("chk-splat-autoimport").checked,
      opa_reg: Number(el("inp-splat-opareg").value || 0.0005),
      prune_opa: Number(el("inp-splat-pruneopa").value || 0.025),
      opacity_reset_every: Number(el("inp-splat-opareset").value || 2000),
      conf_percentile: Number(el("inp-splat-vggtconf").value || 45),
      max_anisotropy: Math.max(4, Number(el("inp-splat-maxaniso").value || 10) * 0.6),
      pose_backend: el("sel-splat-pose-backend") ? el("sel-splat-pose-backend").value : "vggt",
    };
    if (state.splatSourceMode === "video") {
      payload.video_path = el("sel-splat-video").value;
      if (!payload.video_path) {
        showToast("Select a source MP4 video first.", true);
        return;
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
      if (res.job) handleSplatJobUpdate(res.job);
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

  // --- Embedded Zero-Dependency WebGL2 3D Gaussian Splat Viewer ---
  function createSplatWebGLViewer(canvas, hudInfoEl) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: false });
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

    let count = 0;
    let positions = null;
    let sortedIdx = null;
    let depthBuckets = new Int32Array(65536);
    let depthKeys = null;
    let sceneCenter = [0, 0, 0.5];
    let sceneRadius = 3.0;
    let target = [0, 0, 0.5];
    let theta = -1.25;
    let phi = 0.28;
    let dist = 4.5;
    let running = false;
    let dirtySort = true;
    let dragging = false;
    let panMode = false;
    let lastX = 0;
    let lastY = 0;

    function resetCamera() {
      target = sceneCenter.slice();
      theta = -1.25;
      phi = 0.25;
      dist = Math.max(0.8, sceneRadius * 1.75);
      dirtySort = true;
    }

    function computeCameraBasis() {
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
      // right = normalize(cross(fwd, [0, 0, 1]))
      let rx = fy;
      let ry = -fx;
      let rz = 0;
      const rn = Math.hypot(rx, ry, rz) || 1;
      rx /= rn; ry /= rn;
      // up = cross(right, fwd)
      const ux = ry * fz - rz * fy;
      const uy = rz * fx - rx * fz;
      const uz = rx * fy - ry * fx;
      // GLSL column-major mat3 for row0=right, row1=up, row2=fwd:
      const W = new Float32Array([rx, ux, fx, ry, uy, fy, rz, uz, fz]);
      return { eye, rx, ry, rz, ux, uy, uz, fx, fy, fz, W };
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
        // Back-to-front: largest z maps to bucket 0!
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

    canvas.addEventListener("mousedown", (e) => {
      dragging = true;
      panMode = e.button === 2 || e.shiftKey;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!dragging || !running) return;
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
      dist = Math.max(0.25, Math.min(200.0, dist * Math.exp(e.deltaY * 0.0012)));
      dirtySort = true;
    }, { passive: false });

    function frameLoop() {
      if (!running) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(64, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(64, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const cam = computeCameraBasis();
      if (dirtySort) sortBackToFront(cam);

      gl.viewport(0, 0, w, h);
      gl.clearColor(0.035, 0.048, 0.078, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      gl.useProgram(prog);
      gl.bindVertexArray(vao);
      gl.uniformMatrix3fv(uW, false, cam.W);
      gl.uniform3f(uCamPos, cam.eye[0], cam.eye[1], cam.eye[2]);
      const fy = h * 0.92;
      gl.uniform2f(uFocal, fy, fy);
      gl.uniform2f(uViewport, w, h);

      const nDraw = sortBackToFront._validCount || count;
      if (nDraw > 0) gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, nDraw);
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
      hudInfoEl.textContent = `${cap.toLocaleString()} Gaussians · Drag orbit · Right-drag pan · Scroll zoom`;
      if (!running) {
        running = true;
        requestAnimationFrame(frameLoop);
      }
    }

    return {
      load: loadSplatUrl,
      resetCamera,
      pause: () => { running = false; },
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

    const sc = el("theater-splat-canvas");
    const hud = el("theater-splat-hud");
    sc.classList.remove("hidden");
    hud.classList.remove("hidden");

    el("theater-actions").classList.remove("hidden");
    el("theater-meta").textContent =
      `3DGS SPLAT · ${Number(job.n_gaussians || 0).toLocaleString()} pts` +
      (job.psnr ? ` · PSNR ${Number(job.psnr).toFixed(2)} dB` : "") +
      (job.cost_usd ? ` · $${Number(job.cost_usd).toFixed(4)}` : "");
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
  connectSSE();
})();