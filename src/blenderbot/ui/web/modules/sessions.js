// Agent Sessions Center Stage & Inspector: live streaming chat, past session transcripts,
// in-place continue/new session, `.blend` association banner, approval broker, Journal Undo/Redo & Context Ledger.

import { actions, apiFetch, el, escapeHtml, formatBytes, showToast, state, withTokenUrl } from "./state.js";

function formatRelTime(isoStr) {
  if (!isoStr) return "";
  const ts = Date.parse(isoStr);
  if (Number.isNaN(ts)) return "";
  const sec = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

export async function refreshSessions(allWorkspaces = state.sessionsAllWorkspaces) {
  try {
    const data = await apiFetch(`/api/sessions?all_workspaces=${allWorkspaces ? 1 : 0}`);
    state.activeSessionId = data.active_id || state.activeSessionId;
    if (!state.viewedSessionId) state.viewedSessionId = state.activeSessionId;
    state.sessionsList = data.sessions || [];
    if (data.journal) state.journalState = data.journal;
    if (Array.isArray(data.pending_approvals)) state.pendingApprovals = data.pending_approvals;
    if (data.catalog) state.providerCatalog = data.catalog;
    if (Array.isArray(data.roles)) state.rolesList = data.roles;

    renderSessionSettingsControls();
    renderJournalInspector();
    renderApprovalsBanner();
    actions.renderHierarchyTree?.();
    syncTopbarSessionPill();
  } catch (_e) {}
}

export function syncTopbarSessionPill() {
  const navBtn = el("btn-topbar-session");
  const pill = el("pill-session");
  const sDot = el("dot-session");
  const valEl = el("val-session");
  const liveDot = el("topbar-agent-dot");
  const badge = el("agent-unread-badge");
  const hasApproval = state.pendingApprovals.length > 0;

  navBtn?.classList.toggle("active", state.centerKind === "session");
  pill?.classList.toggle("needs-approval", hasApproval);
  liveDot?.classList.toggle("hidden", !state.agentBusy && !hasApproval);

  if (sDot) {
    sDot.className = `sys-dot ${hasApproval ? "warn" : state.agentBusy ? "busy" : "ok"}`;
  }
  if (valEl) {
    if (hasApproval) valEl.textContent = `Approve (${state.pendingApprovals.length})`;
    else if (state.agentBusy) valEl.textContent = "running…";
    else valEl.textContent = state.activeSessionId ? state.activeSessionId.slice(0, 6) : "ready";
  }

  if (state.centerKind === "session" && state.viewedSessionId === state.activeSessionId) {
    state.agentUnread = 0;
  }
  if (badge) {
    badge.textContent = String(state.agentUnread);
    badge.classList.toggle("hidden", state.agentUnread <= 0);
  }

  const metaBox = el("spop-session-meta");
  if (metaBox) {
    const curS = state.sessionsList.find((x) => x.id === state.activeSessionId);
    metaBox.innerHTML = `
      <div class="insp-row"><span class="insp-key">session_id:</span><span class="insp-val ok">${escapeHtml(state.activeSessionId || "—")}</span></div>
      <div class="insp-row"><span class="insp-key">model / role:</span><span class="insp-val">${escapeHtml(state.agentModelId || "—")} (${escapeHtml(state.agentRole || "builder")})</span></div>
      <div class="insp-row"><span class="insp-key">turns / ops:</span><span class="insp-val">${curS?.turns || 0} turns · ${curS?.ops || 0} bl_ops</span></div>
      <div class="insp-row"><span class="insp-key">context:</span><span class="insp-val">${state.ledger ? `${state.ledger.pct}% (${state.ledger.total_tokens.toLocaleString()} tok)` : "—"}</span></div>
    `;
  }
}

export function toggleSessionInCenter() {
  if (state.centerKind === "session" && state.prevNonSessionView) {
    const prev = state.prevNonSessionView;
    if (prev.kind === "3d") actions.open3DSceneInCenter?.(prev.id || "__courtyard__", false);
    else if (prev.kind === "media" && prev.item) actions.openMediaInCenter?.(prev.item);
    else if (prev.kind === "launch") actions.openSplatLaunchDialog?.();
    else if (prev.kind === "gen" && prev.studio) actions.openGenerativeStudioInCenter?.(prev.studio);
    else actions.open3DSceneInCenter?.(state.active3DId || "__courtyard__", false);
    return;
  }
  openSessionInCenter(state.activeSessionId || "active", true);
}

export async function openSessionInCenter(sessionId = "active", focusInput = false) {
  if (state.centerKind !== "session") {
    state.prevNonSessionView = {
      kind: state.centerKind,
      id: state.active3DId,
      item: state.stageItem,
      studio: state.activeGenStudio,
    };
  }
  const effId = !sessionId || sessionId === "active" || sessionId === "current"
    ? (state.activeSessionId || "active")
    : sessionId;

  actions.hideAllCenterViews?.();
  state.centerKind = "session";
  state.viewedSessionId = effId;

  el("theater-session-wrap")?.classList.remove("hidden");
  el("stage-3d-btns")?.classList.add("hidden");
  el("stage-media-btns")?.classList.add("hidden");

  try {
    const detail = await apiFetch(`/api/sessions/${encodeURIComponent(effId)}`);
    const s = detail.session || {};
    const isAct = !!s.is_active;
    state.viewedSessionId = s.id || effId;
    if (isAct) state.activeSessionId = s.id;
    if (detail.journal && isAct) state.journalState = detail.journal;
    if (detail.live?.pending && isAct) state.pendingApprovals = detail.live.pending;

    el("stage-kind-badge").textContent = isAct ? "ACTIVE SESSION" : "PAST SESSION";
    el("stage-title").textContent = s.first_prompt
      ? `${s.id} — ${s.first_prompt.slice(0, 64)}`
      : `Agent Session ${s.id}`;
    el("theater-meta").textContent = `${s.model || ""} · ${s.turns || 0} turns · ${s.ops || 0} ops · $${Number(s.cost || 0).toFixed(4)}`;

    renderSessionHeaderBar(s, detail.journal);
    renderTranscriptEvents(detail.events || [], isAct, detail.live?.state);
    renderApprovalsBanner();
    renderJournalInspector(detail.journal || state.journalState);
    renderSessionBlendAssociation(s);
  } catch (err) {
    showToast(`Failed to load session: ${err.message}`, true);
  }

  actions.syncInspectorContext?.();
  syncTopbarSessionPill();
  actions.renderHierarchyTree?.();

  if (focusInput && state.viewedSessionId === state.activeSessionId) {
    setTimeout(() => el("agent-input")?.focus(), 20);
  }
}

function renderSessionHeaderBar(s, journal) {
  const bar = el("session-stage-banner");
  if (!bar) return;
  const isAct = !!s.is_active;
  const relT = formatRelTime(s.started_at);
  const blendBadge = s.blend_name
    ? `<span class="sec-meta-pill" title="${escapeHtml(s.blend_filepath)}">🟠 ${escapeHtml(s.blend_name)}</span>`
    : s.scene_name
    ? `<span class="sec-meta-pill">Scene: ${escapeHtml(s.scene_name)}</span>`
    : `<span class="sec-meta-pill">Unsaved Scene</span>`;

  if (isAct) {
    bar.innerHTML = `
      <div class="sess-banner-left">
        <span class="sess-live-pill">● Active <code>${escapeHtml(s.id)}</code></span>
        ${blendBadge}
        <span class="wb-lbl">${escapeHtml(relT)}${s.resumed_from ? ` · continued from ${escapeHtml(s.resumed_from)}` : ""}</span>
      </div>
      <div class="sess-banner-right">
        <button type="button" id="btn-sess-banner-undo" class="btn btn-secondary btn-xs" ${journal?.can_undo ? "" : "disabled"} title="Undo last Blender/file action (/undo)">↶ Undo Action</button>
        <button type="button" id="btn-sess-banner-redo" class="btn btn-secondary btn-xs" ${journal?.can_redo ? "" : "disabled"} title="Redo undone action (/redo)">↷ Redo</button>
        <button type="button" id="btn-sess-banner-new" class="btn btn-ghost btn-xs" title="Start a clean new Agent Session">+ New Session</button>
      </div>
    `;
    el("session-compose-wrap")?.classList.remove("hidden");
    el("btn-sess-banner-undo")?.addEventListener("click", () => runJournalAction("undo"));
    el("btn-sess-banner-redo")?.addEventListener("click", () => runJournalAction("redo"));
    el("btn-sess-banner-new")?.addEventListener("click", () => startNewSession(false));
  } else {
    bar.innerHTML = `
      <div class="sess-banner-left">
        <span class="sec-meta-pill">📜 Archived <code>${escapeHtml(s.id)}</code></span>
        ${blendBadge}
        <span class="wb-lbl">${escapeHtml(relT)} · ${s.turns || 0} turns · $${Number(s.cost || 0).toFixed(4)}</span>
      </div>
      <div class="sess-banner-right">
        ${s.has_blend_state ? `<button type="button" id="btn-sess-cont-blend" class="btn btn-accent btn-sm" title="Restore conversation + undo stack AND open this session's .blend scene in Blender 5.3">▶ Continue + Open .blend</button>` : ""}
        <button type="button" id="btn-sess-cont-only" class="btn btn-primary btn-sm" title="Continue this conversation &amp; undo history with your current Blender scene">▶ Continue Session</button>
        <button type="button" id="btn-sess-back-active" class="btn btn-ghost btn-xs">Back to Active</button>
      </div>
    `;
    el("session-compose-wrap")?.classList.add("hidden");
    el("btn-sess-cont-blend")?.addEventListener("click", () => continuePastSession(s.id, true));
    el("btn-sess-cont-only")?.addEventListener("click", () => continuePastSession(s.id, false));
    el("btn-sess-back-active")?.addEventListener("click", () => openSessionInCenter(state.activeSessionId, true));
  }
}

function renderSessionBlendAssociation(s) {
  const box = el("insp-session-blend-box");
  if (!box) return;
  const curBlFp = state.blenderInfo?.filepath || "";
  const curBlName = curBlFp ? curBlFp.replace(/\\/g, "/").split("/").pop() : "";
  const sessFp = s.blend_filepath || "";
  const sessSnap = s.last_snapshot || "";
  const mismatch = sessFp && curBlFp && sessFp.toLowerCase() !== curBlFp.toLowerCase();

  box.innerHTML = `
    <div class="insp-row"><span class="insp-key">live_blender:</span><span class="insp-val ${state.blenderInfo?.connected ? "ok" : ""}">${state.blenderInfo?.connected ? escapeHtml(curBlName || `Unsaved (${state.blenderInfo?.scene?.name || "Scene"})`) : "disconnected"}</span></div>
    <div class="insp-row"><span class="insp-key">session_blend:</span><span class="insp-val">${escapeHtml(s.blend_name || (sessSnap ? "checkpoint saved" : "unsaved scratch"))}</span></div>
    ${mismatch ? `<div class="sess-blend-mismatch">Scene differs from session's <b>${escapeHtml(s.blend_name)}</b> <button type="button" id="btn-insp-open-sess-blend" class="btn btn-secondary btn-xs">Open in Blender</button></div>` : ""}
    ${!sessFp && sessSnap ? `<div class="sess-blend-mismatch">Session has a saved <code>.blend</code> snapshot <button type="button" id="btn-insp-open-sess-snap" class="btn btn-secondary btn-xs">Restore Snapshot</button></div>` : ""}
  `;
  el("btn-insp-open-sess-blend")?.addEventListener("click", () => openBlendFileInBlender(sessFp));
  el("btn-insp-open-sess-snap")?.addEventListener("click", () => openBlendFileInBlender(sessSnap));
}

export async function openBlendFileInBlender(filepath) {
  if (!filepath) return;
  try {
    showToast(`Opening ${filepath.split(/[\\/]/).pop()} in Blender 5.3…`);
    await apiFetch("/api/blender/open_blend", { method: "POST", json: { path: filepath } });
    await actions.refreshState?.();
    await refreshSessions();
    showToast("Opened .blend scene in Blender 5.3");
  } catch (err) {
    showToast(`Open .blend failed: ${err.message}`, true);
  }
}

export async function saveActiveBlenderScene(customPath = "") {
  try {
    const res = await apiFetch("/api/blender/save_blend", {
      method: "POST",
      json: customPath ? { path: customPath } : {},
    });
    if (Array.isArray(res.blends)) state.blendItems = res.blends;
    await actions.refreshState?.();
    await refreshSessions();
    showToast(`Saved ${res.result?.filepath || ".blend"}`);
  } catch (err) {
    showToast(`Save .blend failed: ${err.message}`, true);
  }
}

function formatMessageBodyHtml(rawText) {
  const escaped = escapeHtml(rawText || "");
  return escaped.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_m, lang, code) => {
    const tag = lang ? `<div class="msg-code-lang">${escapeHtml(lang)}</div>` : "";
    return `<div class="msg-code-block">${tag}<pre><code>${code}</code></pre></div>`;
  });
}

function createTranscriptRow(ev) {
  const div = document.createElement("div");
  if (ev.type === "user" || ev.type === "steer") {
    div.className = `feed-msg feed-user${ev.type === "steer" ? " feed-steer" : ""}`;
    div.innerHTML = `<div class="msg-role-hdr">${ev.type === "steer" ? "⚡ Operator Steer" : "❯ You"}</div><div class="msg-prose">${escapeHtml(ev.text || "")}</div>`;
  } else if (ev.type === "ops") {
    const ok = ev.status === "ok";
    div.className = `feed-msg feed-op ${ok ? "op-ok" : "op-err"}`;
    const opsSummary = (ev.ops || []).join(", ") || `${ev.count || 0} ops`;
    const errTxt = (ev.errors || []).map((e) => e.message || JSON.stringify(e)).join("; ");
    div.innerHTML = `<span>⚙ <b>bl_ops</b> [${escapeHtml(ev.status)}]: <code>${escapeHtml(opsSummary)}</code></span>${errTxt ? `<div class="op-err-line">⚠ ${escapeHtml(errTxt)}</div>` : ""}`;
  } else if (ev.type === "shell") {
    div.className = "feed-msg feed-op";
    div.innerHTML = `<div><code>$ ${escapeHtml(ev.cmd)}</code> (exit ${ev.exit_code})</div>${ev.output ? `<pre class="shell-out-pre">${escapeHtml(ev.output)}</pre>` : ""}`;
  } else if (ev.type === "notice" || ev.type === "warning") {
    div.className = `feed-msg ${ev.type === "warning" ? "feed-warn" : "feed-op"}`;
    div.textContent = `· ${ev.message || ev.text || ""}`;
  } else {
    div.className = "feed-msg feed-agent";
    div.innerHTML = `<div class="msg-role-hdr">🤖 blenderbot${ev.turn ? ` · turn #${ev.turn}` : ""}</div><div class="msg-prose">${formatMessageBodyHtml(ev.text || "")}</div>`;
  }
  return div;
}

function renderTranscriptEvents(events, isActive, liveState) {
  const feed = el("agent-feed");
  if (!feed) return;
  feed.innerHTML = "";
  state.liveAgentNode = null;

  if (!events.length && !liveState?.live) {
    feed.innerHTML = `<div class="feed-empty-welcome">
      <div class="welcome-glyph">🤖</div>
      <h3>Blender 5.3 Agent Session Ready</h3>
      <p>Ask blenderbot to model geometry, rig/animate, set lighting &amp; Principled BSDF materials, audit topology, or render contact sheets. Reference workspace files with <code>@path</code> or run slash commands like <code>/sheet</code>, <code>/audit</code>, <code>/turntable</code>.</p>
    </div>`;
    return;
  }

  events.forEach((ev) => feed.appendChild(createTranscriptRow(ev)));

  if (isActive && liveState?.busy && liveState?.live) {
    const liveRow = createTranscriptRow({ type: "assistant", text: liveState.live });
    state.liveAgentNode = liveRow.querySelector(".msg-prose") || liveRow;
    state.liveAgentRaw = liveState.live;
    feed.appendChild(liveRow);
  }
  feed.scrollTop = feed.scrollHeight;
}

export function appendLiveSessionEvent(kind, payload) {
  const viewingActive = state.centerKind === "session" && state.viewedSessionId === state.activeSessionId;
  const feed = el("agent-feed");

  if (!viewingActive && (kind === "user" || kind === "turn-end")) {
    state.agentUnread += 1;
    syncTopbarSessionPill();
  }
  if (!feed || state.viewedSessionId !== state.activeSessionId) return;

  feed.querySelector(".feed-empty-welcome")?.remove();

  if (kind === "user") {
    feed.appendChild(createTranscriptRow({ type: "user", text: payload }));
  } else if (kind === "notice" || kind === "warning") {
    feed.appendChild(createTranscriptRow({ type: kind, message: payload }));
  } else if (kind === "ops-parsed") {
    const names = Array.isArray(payload) ? payload.map((o) => o.op) : [];
    feed.appendChild(createTranscriptRow({ type: "ops", status: "running", ops: names, count: names.length }));
  } else if (kind === "turn-start") {
    state.liveAgentRaw = "";
    const row = createTranscriptRow({ type: "assistant", text: "…" });
    state.liveAgentNode = row.querySelector(".msg-prose") || row;
    feed.appendChild(row);
  } else if (kind === "text") {
    state.liveAgentRaw = (state.liveAgentRaw || "") + String(payload || "");
    if (!state.liveAgentNode) {
      const row = createTranscriptRow({ type: "assistant", text: "" });
      state.liveAgentNode = row.querySelector(".msg-prose") || row;
      feed.appendChild(row);
    }
    state.liveAgentNode.innerHTML = formatMessageBodyHtml(state.liveAgentRaw);
  } else if (kind === "turn-end") {
    state.liveAgentNode = null;
    state.liveAgentRaw = "";
  }
  feed.scrollTop = feed.scrollHeight;
}

export function renderApprovalsBanner() {
  const box = el("session-approvals-bar");
  if (!box) return;
  if (!state.pendingApprovals.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    syncTopbarSessionPill();
    return;
  }
  box.classList.remove("hidden");
  box.innerHTML = state.pendingApprovals
    .map((ap) => {
      const opObj = ap.op || {};
      const opName = opObj.op || String(opObj);
      const paramsStr = opObj.params ? JSON.stringify(opObj.params) : "";
      return `
        <div class="approval-card" data-apid="${escapeHtml(ap.id)}">
          <div class="approval-info">
            <span class="approval-badge">⚠ APPROVAL REQUIRED</span>
            <code>${escapeHtml(opName)}</code>
            <span class="wb-mono">${escapeHtml(paramsStr.slice(0, 140))}</span>
          </div>
          <div class="approval-btns">
            <button type="button" class="btn btn-accent btn-xs btn-ap-allow" data-apid="${escapeHtml(ap.id)}">✓ Approve (Checkpoint First)</button>
            <button type="button" class="btn btn-ghost btn-xs btn-ap-deny" data-apid="${escapeHtml(ap.id)}">✕ Deny</button>
          </div>
        </div>
      `;
    })
    .join("");

  box.querySelectorAll(".btn-ap-allow").forEach((b) => {
    b.addEventListener("click", () => resolveWebApproval(b.dataset.apid, true));
  });
  box.querySelectorAll(".btn-ap-deny").forEach((b) => {
    b.addEventListener("click", () => resolveWebApproval(b.dataset.apid, false));
  });
  syncTopbarSessionPill();
}

async function resolveWebApproval(apId, allow) {
  try {
    await apiFetch("/api/chat/approve", {
      method: "POST",
      json: { id: apId, allow, checkpoint_first: true },
    });
    state.pendingApprovals = state.pendingApprovals.filter((p) => p.id !== apId);
    renderApprovalsBanner();
  } catch (err) {
    showToast(err.message, true);
  }
}

export async function startNewSession(resetBlender = false) {
  try {
    const res = await apiFetch("/api/sessions/new", {
      method: "POST",
      json: { reset_blender: resetBlender },
    });
    state.activeSessionId = res.session?.id || "";
    state.viewedSessionId = state.activeSessionId;
    state.journalState = res.journal || null;
    await refreshSessions();
    await actions.refreshState?.();
    await openSessionInCenter(state.activeSessionId, true);
    showToast(`Started new Agent Session ${state.activeSessionId}`);
  } catch (err) {
    showToast(err.message, true);
  }
}

export async function continuePastSession(sessionId, openBlend = false) {
  try {
    const res = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/resume`, {
      method: "POST",
      json: { open_blend: openBlend },
    });
    state.activeSessionId = res.session?.id || "";
    state.viewedSessionId = state.activeSessionId;
    state.journalState = res.journal || null;
    await refreshSessions();
    await actions.refreshState?.();
    await openSessionInCenter(state.activeSessionId, true);
    showToast(
      res.opened_blend
        ? `Continued session & opened ${res.opened_blend.split(/[\\/]/).pop()}`
        : `Continued session (${res.session?.turns || 0} prior turns restored)`
    );
  } catch (err) {
    showToast(err.message, true);
  }
}

export async function runJournalAction(dir, entryId = "") {
  try {
    const url = dir === "redo" ? "/api/journal/redo" : "/api/journal/undo";
    const res = await apiFetch(url, {
      method: "POST",
      json: entryId ? { id: entryId } : { steps: 1 },
    });
    if (res.journal) {
      state.journalState = res.journal;
      renderJournalInspector(res.journal);
    }
    await actions.refreshState?.();
    if (state.centerKind === "session" && state.viewedSessionId === state.activeSessionId) {
      const curS = state.sessionsList.find((x) => x.id === state.activeSessionId);
      if (curS) renderSessionHeaderBar(curS, state.journalState);
    }
    showToast(dir === "redo" ? "↷ Redid action" : "↶ Undid action");
  } catch (err) {
    showToast(err.message, true);
  }
}

export function renderJournalInspector(jState = state.journalState) {
  const listEl = el("insp-journal-list");
  const badge = el("badge-journal-cursor");
  if (!listEl) return;

  const acts = jState?.actions || [];
  if (badge) badge.textContent = `${jState?.applied_count || 0}/${acts.length}`;
  if (el("btn-insp-journal-undo")) el("btn-insp-journal-undo").disabled = !jState?.can_undo;
  if (el("btn-insp-journal-redo")) el("btn-insp-journal-redo").disabled = !jState?.can_redo;

  if (!acts.length) {
    listEl.innerHTML = `<div class="empty-placeholder">No scene/file mutations recorded in this session yet.</div>`;
    return;
  }
  listEl.innerHTML = "";
  [...acts].reverse().slice(0, 20).forEach((a) => {
    const row = document.createElement("div");
    const statusCls = a.superseded ? "superseded" : a.applied ? "applied" : "undone";
    row.className = `journal-act-row ${statusCls}`;
    row.innerHTML = `
      <div class="jact-left">
        <code>${escapeHtml(a.id)}</code>
        <span class="jact-summary" title="${escapeHtml(a.summary)}">${escapeHtml(a.summary)}</span>
      </div>
      ${
        !a.superseded
          ? `<button type="button" class="btn btn-ghost btn-xs">${a.applied ? "Undo" : "Redo"}</button>`
          : `<span class="wb-lbl">sup</span>`
      }
    `;
    row.querySelector("button")?.addEventListener("click", () => {
      runJournalAction(a.applied ? "undo" : "redo", a.id);
    });
    listEl.appendChild(row);
  });
}

export function renderSessionSettingsControls() {
  const cat = state.providerCatalog;
  const modSel = el("sel-agent-model");
  if (modSel && cat?.providers) {
    const curId = cat.current_model?.id || state.agentModelId || "";
    modSel.innerHTML = "";
    cat.providers.forEach((p) => {
      (p.models || []).forEach((m) => {
        const opt = document.createElement("option");
        opt.value = `${p.name}::${m.id}`;
        opt.textContent = `${m.id} (${p.name}${p.has_key ? "" : " · no key"})`;
        if (m.id === curId && p.name === cat.current_provider) opt.selected = true;
        modSel.appendChild(opt);
      });
    });
  }
  if (el("sel-agent-effort") && cat?.effort) {
    el("sel-agent-effort").value = cat.effort;
  }
  if (el("sel-agent-role") && state.agentRole) {
    el("sel-agent-role").value = state.agentRole;
  }
}

export function renderBlendCenterView(blendItem) {
  if (!blendItem) return;
  actions.hideAllCenterViews?.();
  state.centerKind = "blend";
  state.stageBlendItem = blendItem;

  el("stage-kind-badge").textContent = "BLENDER .BLEND";
  el("stage-title").textContent = blendItem.name;
  el("theater-meta").textContent = `${blendItem.path} · ${formatBytes(blendItem.size_bytes)}`;
  el("theater-blend-wrap")?.classList.remove("hidden");

  const isCurrentlyOpen =
    state.blenderInfo?.filepath &&
    blendItem.abs_path &&
    state.blenderInfo.filepath.toLowerCase() === blendItem.abs_path.toLowerCase();

  const card = el("blend-stage-card");
  if (card) {
    const sessLinks = (blendItem.session_ids || [])
      .map((sid) => `<button type="button" class="mode-pill btn-jump-sess" data-sid="${escapeHtml(sid)}">🤖 Session ${escapeHtml(sid)}</button>`)
      .join(" ");
    card.innerHTML = `
      <div class="blend-hero-icon">🟠</div>
      <h3>${escapeHtml(blendItem.name)} ${isCurrentlyOpen ? `<span class="sess-live-pill">Currently Open in Blender 5.3</span>` : ""}</h3>
      <div class="wb-mono">${escapeHtml(blendItem.path)} · ${formatBytes(blendItem.size_bytes)}</div>
      <div class="blend-stage-actions">
        <button type="button" id="btn-blend-open-live" class="btn btn-accent btn-lg">${isCurrentlyOpen ? "↻ Reload in Blender 5.3" : "🟠 Open in Blender 5.3"}</button>
        <button type="button" id="btn-blend-stream-3d" class="btn btn-primary btn-sm">🎮 Stream Live 3D / FPS View</button>
        <button type="button" id="btn-blend-grab-still" class="btn btn-secondary btn-sm">📸 Render Still</button>
        <a href="${withTokenUrl(blendItem.url)}&download=1" download="${escapeHtml(blendItem.name)}" class="btn btn-ghost btn-sm">⬇ Download .blend</a>
      </div>
      ${sessLinks ? `<div class="blend-sess-row"><span class="wb-lbl">Authored / Edited by:</span> ${sessLinks}</div>` : ""}
    `;
    el("btn-blend-open-live")?.addEventListener("click", async () => {
      await openBlendFileInBlender(blendItem.path);
      renderBlendCenterView(blendItem);
    });
    el("btn-blend-stream-3d")?.addEventListener("click", async () => {
      if (!isCurrentlyOpen) await openBlendFileInBlender(blendItem.path);
      actions.open3DSceneInCenter?.("__blender_live__", false);
    });
    el("btn-blend-grab-still")?.addEventListener("click", async () => {
      if (!isCurrentlyOpen) await openBlendFileInBlender(blendItem.path);
      actions.captureFromBlender?.("still");
    });
    card.querySelectorAll(".btn-jump-sess").forEach((b) => {
      b.addEventListener("click", () => openSessionInCenter(b.dataset.sid, false));
    });
  }

  actions.syncInspectorContext?.();
  actions.renderHierarchyTree?.();
}

export function initSessionsModule() {
  el("btn-topbar-session")?.addEventListener("click", () => toggleSessionInCenter());
  el("btn-insp-new-session")?.addEventListener("click", () => startNewSession(false));
  el("btn-insp-new-session-clean")?.addEventListener("click", () => startNewSession(true));
  el("btn-insp-journal-undo")?.addEventListener("click", () => runJournalAction("undo"));
  el("btn-insp-journal-redo")?.addEventListener("click", () => runJournalAction("redo"));
  el("btn-bl-insp-undo")?.addEventListener("click", () => runJournalAction("undo"));
  el("btn-bl-insp-redo")?.addEventListener("click", () => runJournalAction("redo"));

  el("btn-save-blend-now")?.addEventListener("click", () => {
    const customP = (el("inp-save-blend-path")?.value || "").trim();
    saveActiveBlenderScene(customP);
  });

  el("sel-agent-model")?.addEventListener("change", async (e) => {
    const [prov, mod] = String(e.target.value || "").split("::");
    if (!mod) return;
    try {
      await apiFetch("/api/session/settings", { method: "POST", json: { provider: prov, model: mod } });
      await actions.refreshState?.();
      await refreshSessions();
      showToast(`Switched Agent model → ${mod}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("sel-agent-effort")?.addEventListener("change", async (e) => {
    try {
      await apiFetch("/api/session/settings", { method: "POST", json: { effort: e.target.value } });
      showToast(`Reasoning effort → ${e.target.value}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("sel-agent-role")?.addEventListener("change", async (e) => {
    try {
      await apiFetch("/api/session/settings", { method: "POST", json: { role: e.target.value } });
      state.agentRole = e.target.value;
      showToast(`Capability role → ${e.target.value}`);
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("btn-agent-compact")?.addEventListener("click", async () => {
    try {
      showToast("Compacting LLM context ledger…");
      await apiFetch("/api/session/settings", { method: "POST", json: { action: "compact" } });
      await actions.refreshState?.();
      showToast("Compacted context ledger");
    } catch (err) {
      showToast(err.message, true);
    }
  });

  el("session-quick-cmds")?.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-slash]");
    if (!b) return;
    const cmd = b.dataset.slash;
    try {
      await apiFetch("/api/chat/submit", { method: "POST", json: { text: cmd } });
    } catch (err) {
      showToast(err.message, true);
    }
  });

  actions.refreshSessions = refreshSessions;
  actions.openSessionInCenter = openSessionInCenter;
  actions.toggleSessionInCenter = toggleSessionInCenter;
  actions.appendLiveSessionEvent = appendLiveSessionEvent;
  actions.renderApprovalsBanner = renderApprovalsBanner;
  actions.renderJournalInspector = renderJournalInspector;
  actions.startNewSession = startNewSession;
  actions.continuePastSession = continuePastSession;
  actions.openBlendCenterView = renderBlendCenterView;
  actions.openBlendFileInBlender = openBlendFileInBlender;
  actions.saveActiveBlenderScene = saveActiveBlenderScene;
  actions.syncTopbarSessionPill = syncTopbarSessionPill;
}