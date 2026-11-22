// VS Code-style Left Explorer Hierarchy Tree, rich hover preview tooltip, and Center Stage view switcher.

import { BUILTIN_3D_STAGES, actions, apiFetch, el, escapeHtml, formatBytes, showToast, state, withTokenUrl } from "./state.js";

const tooltipEl = () => el("hierarchy-tooltip");

export function showHoverTooltip(anchorEl, cfg) {
  const tt = tooltipEl();
  if (!tt || document.body.classList.contains("hide-left")) return;
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

  tt.classList.remove("hidden");
  const rect = anchorEl.getBoundingClientRect();
  const libRect = el("dropzone-library")?.getBoundingClientRect();
  const anchorRight = libRect ? Math.max(libRect.right, rect.right) : rect.right;
  const ttH = tt.offsetHeight || 280;
  const ttW = tt.offsetWidth || 316;
  const topY = Math.max(54, Math.min(window.innerHeight - ttH - 12, rect.top - 16));
  const leftX = Math.max(12, Math.min(window.innerWidth - ttW - 12, anchorRight + 8));
  tt.style.top = `${topY}px`;
  tt.style.left = `${leftX}px`;
}

export function hideHoverTooltip() {
  const tt = tooltipEl();
  if (!tt) return;
  tt.classList.add("hidden");
  const vid = el("ht-preview-box")?.querySelector("video");
  if (vid) vid.pause();
}

function rolesForMedia(it) {
  const roles = [];
  if (state.sourceVideo && state.sourceVideo.path === it.path) roles.push({ label: "VID1", full: "VID 1", primary: true });
  if (state.sourceVideo2 && state.sourceVideo2.path === it.path) roles.push({ label: "VID2", full: "VID 2", primary: true });
  if (state.firstFrame && state.firstFrame.path === it.path) roles.push({ label: "1ST", full: "1ST FRAME", primary: false });
  if (state.lastFrame && state.lastFrame.path === it.path) roles.push({ label: "LAST", full: "LAST FRAME", primary: false });
  if (state.seedreamRefs.some((r) => r.path === it.path)) roles.push({ label: "SR", full: "SEEDREAM REF", primary: state.inspectorTab === "seedream" });
  if (state.meshyInputImage && state.meshyInputImage.path === it.path) roles.push({ label: "3D", full: "MESHY INPUT", primary: state.inspectorTab === "meshy" });
  if (state.meshyTextureImage && state.meshyTextureImage.path === it.path) roles.push({ label: "TEX", full: "MESHY TEXTURE", primary: false });
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
  if (low === "jobs") return "⚡";
  if (low === "cloud") return "☁️";
  if (low === "local") return "🖥️";
  if (low === "modal-3dgs" || low === "splats") return "🌐";
  if (low === "fal-meshy" || low === "meshy") return "🧊";
  if (low === "seedance") return "🎬";
  if (low === "seedream") return "🎨";
  if (low === "blender" || low === "renders") return "🖼";
  if (low === "shots") return "🎥";
  return isOpen ? "📂" : "📁";
}

function folderSortRank(segName) {
  const low = segName.toLowerCase();
  if (low === "jobs") return 0;
  if (low === "cloud") return 1;
  if (low === "local") return 2;
  if (low === "modal-3dgs") return 1;
  if (low === "fal-meshy") return 2;
  if (low === "assets") return 3;
  if (low === "meshy") return 4;
  if (low === "splats") return 5;
  if (low === "seedance") return 6;
  if (low === "seedream") return 7;
  if (low === "renders") return 8;
  if (low === "shots") return 9;
  if (low === "inputs") return 10;
  return 12;
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

export function expandTreePath(folderPath) {
  const parts = String(folderPath || "")
    .replace(/\\/g, "/")
    .split("/")
    .map((s) => s.trim())
    .filter(Boolean);
  state.collapsedGroups.delete("sec:workspace");
  let accum = "";
  for (const seg of parts) {
    accum = accum ? `${accum}/${seg}` : seg;
    state.collapsedGroups.delete(`dir:${accum}`);
  }
}

export function scrollActiveTreeRowIntoView() {
  requestAnimationFrame(() => {
    const activeRow = el("hierarchy-tree")?.querySelector(".vsc-tree-row.active-stage");
    activeRow?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  });
}

export function renderHierarchyTree() {
  const tree = el("hierarchy-tree");
  if (!tree) return;
  hideHoverTooltip();
  tree.innerHTML = "";
  state.allTreeBranchKeys = [];
  const q = state.search.trim().toLowerCase();
  const forceExpand = q.length > 0;
  const flt = state.filter;

  // --- Section 1: Agent Sessions (Active + Past Resumable Conversations) ---
  if (flt === "all" || flt === "sessions") {
    const sessRows = (state.sessionsList || []).filter(
      (s) =>
        !q ||
        String(s.id || "").toLowerCase().includes(q) ||
        String(s.first_prompt || "").toLowerCase().includes(q) ||
        String(s.blend_name || "").toLowerCase().includes(q)
    );
    tree.appendChild(
      buildVscSection("sec:sessions", "Agent Sessions", Math.max(1, sessRows.length), forceExpand, (body) => {
        const newRow = document.createElement("div");
        newRow.className = "vsc-tree-row tree-item tree-action-row";
        newRow.tabIndex = 0;
        newRow.setAttribute("role", "treeitem");
        newRow.innerHTML = `
          ${makeIndentGuidesHtml(1)}
          <span class="vsc-twistie spacer">+</span>
          <span class="vsc-node-icon">✨</span>
          <span class="vsc-label-wrap"><span class="tree-name">New Agent Session</span></span>
          <span class="tree-sub">clean chat</span>
        `;
        newRow.addEventListener("click", () => actions.startNewSession?.(false));
        body.appendChild(newRow);

        sessRows.forEach((s) => {
          const isViewed = state.centerKind === "session" && state.viewedSessionId === s.id;
          const row = document.createElement("div");
          row.className = `vsc-tree-row tree-item${isViewed ? " active-stage" : ""}`;
          row.tabIndex = 0;
          row.setAttribute("role", "treeitem");
          const labelTxt = s.first_prompt
            ? `${s.id} · ${s.first_prompt.slice(0, 34)}`
            : s.is_active
            ? `${s.id} (Active Session)`
            : `Session ${s.id}`;
          const subTxt = s.is_active ? "live" : `${s.turns || 0}t`;
          row.innerHTML = `
            ${makeIndentGuidesHtml(1)}
            <span class="vsc-twistie spacer">·</span>
            <span class="vsc-node-icon">${s.is_active ? "🟢" : "💬"}</span>
            <span class="vsc-label-wrap">
              <span class="tree-name" title="${escapeHtml(s.first_prompt || s.id)}">${escapeHtml(labelTxt)}</span>
              ${s.blend_name ? `<span class="role-pill orange" title="${escapeHtml(s.blend_filepath)}">.blend</span>` : ""}
            </span>
            <span class="tree-sub">${escapeHtml(subTxt)}</span>
            <span class="tree-quick-btns"></span>
          `;
          const qBtns = row.querySelector(".tree-quick-btns");
          if (!s.is_active) {
            const contBtn = document.createElement("button");
            contBtn.type = "button";
            contBtn.textContent = "▶ Resume";
            contBtn.title = "Continue this past session (restores context & undo stack)";
            contBtn.addEventListener("click", (e) => {
              e.stopPropagation();
              actions.continuePastSession?.(s.id, false);
            });
            qBtns.appendChild(contBtn);
          }
          row.addEventListener("click", () => actions.openSessionInCenter?.(s.id, !!s.is_active));
          row.addEventListener("mouseenter", () =>
            showHoverTooltip(row, {
              glyph: s.is_active ? "🤖" : "📜",
              title: `${s.is_active ? "Active" : "Past"} Session ${s.id}`,
              path: s.first_prompt || "(No prompt sent yet)",
              specs: [
                ["Model", s.model || "—"],
                ["Turns / Ops", `${s.turns || 0} turns · ${s.ops || 0} bl_ops`],
                ["Scene", s.blend_name || s.scene_name || "Unsaved Scene"],
                ["Cost", `$${Number(s.cost || 0).toFixed(4)} USD`],
              ],
            })
          );
          row.addEventListener("mouseleave", hideHoverTooltip);
          body.appendChild(row);
        });
      })
    );
  }

  // --- Section 2: Blender 5.3 Live Scene & Workspace `.blend` Files ---
  if (flt === "all" || flt === "splat" || flt === "blend") {
    const liveStage = BUILTIN_3D_STAGES.find((s) => s.id === "__blender_live__");
    const blends = (state.blendItems || []).filter(
      (b) => !q || b.name.toLowerCase().includes(q) || b.path.toLowerCase().includes(q)
    );
    const blCount = (liveStage ? 1 : 0) + blends.length;
    if (blCount > 0) {
      tree.appendChild(
        buildVscSection("sec:blender_scenes", "Blender 5.3 & .blend Scenes", blCount, forceExpand, (body) => {
          if (liveStage && (!q || "live blender 5.3 scene".includes(q))) {
            const isAct = state.centerKind === "3d" && state.active3DId === "__blender_live__";
            const blConn = !!state.blenderInfo?.connected;
            const scLbl =
              blConn && state.blenderInfo?.scene
                ? typeof state.blenderInfo.scene === "string"
                  ? state.blenderInfo.scene
                  : state.blenderInfo.scene.name
                : "standby";
            const row = document.createElement("div");
            row.className = `vsc-tree-row tree-item${isAct ? " active-stage" : ""}`;
            row.tabIndex = 0;
            row.setAttribute("role", "treeitem");
            row.innerHTML = `
              ${makeIndentGuidesHtml(1)}
              <span class="vsc-twistie spacer">·</span>
              <span class="vsc-node-icon">🟠</span>
              <span class="vsc-label-wrap"><span class="tree-name">Live Blender 5.3 Viewport</span></span>
              <span class="tree-sub">${escapeHtml(scLbl)}</span>
              <span class="tree-quick-btns"></span>
            `;
            const qBtns = row.querySelector(".tree-quick-btns");
            const fpsB = document.createElement("button");
            fpsB.type = "button";
            fpsB.textContent = "🎮 FPS";
            fpsB.title = "Fly through Live Blender 5.3 meshes in FPS mode";
            fpsB.addEventListener("click", (e) => {
              e.stopPropagation();
              open3DSceneInCenter("__blender_live__", true);
            });
            qBtns.appendChild(fpsB);
            row.addEventListener("click", () => open3DSceneInCenter("__blender_live__", false));
            row.addEventListener("mouseenter", () =>
              showHoverTooltip(row, {
                glyph: "🟠",
                title: "Live Blender 5.3 Scene",
                path: state.blenderInfo?.filepath || "Connected Blender 5.3 Bridge / Headless Worker",
                specs: [
                  ["Connection", blConn ? `${state.blenderInfo.mode} (connected)` : "standby (click to connect)"],
                  ["Active Camera", state.blenderInfo?.active_camera || "—"],
                ],
              })
            );
            row.addEventListener("mouseleave", hideHoverTooltip);
            body.appendChild(row);
          }

          blends.forEach((bItem) => {
            const isAct = state.centerKind === "blend" && state.stageBlendItem?.path === bItem.path;
            const isOpenLive =
              state.blenderInfo?.filepath &&
              bItem.abs_path &&
              state.blenderInfo.filepath.toLowerCase() === bItem.abs_path.toLowerCase();
            const row = document.createElement("div");
            row.className = `vsc-tree-row tree-item${isAct ? " active-stage" : ""}`;
            row.tabIndex = 0;
            row.setAttribute("role", "treeitem");
            row.innerHTML = `
              ${makeIndentGuidesHtml(1)}
              <span class="vsc-twistie spacer">·</span>
              <span class="vsc-node-icon">📦</span>
              <span class="vsc-label-wrap">
                <span class="tree-name" title="${escapeHtml(bItem.path)}">${escapeHtml(bItem.name)}</span>
                ${isOpenLive ? `<span class="role-pill">OPEN</span>` : ""}
              </span>
              <span class="tree-sub">${escapeHtml(formatBytes(bItem.size_bytes))}</span>
              <span class="tree-quick-btns"></span>
            `;
            const qBtns = row.querySelector(".tree-quick-btns");
            const opBtn = document.createElement("button");
            opBtn.type = "button";
            opBtn.textContent = "🟠 Open";
            opBtn.title = "Open this .blend file in Blender 5.3";
            opBtn.addEventListener("click", (e) => {
              e.stopPropagation();
              actions.openBlendFileInBlender?.(bItem.path);
            });
            qBtns.appendChild(opBtn);

            row.addEventListener("click", () => actions.openBlendCenterView?.(bItem));
            row.addEventListener("mouseenter", () =>
              showHoverTooltip(row, {
                glyph: "🟠",
                title: bItem.name,
                path: bItem.path,
                specs: [
                  ["File Size", formatBytes(bItem.size_bytes)],
                  ["Status", isOpenLive ? "Currently open in Blender 5.3" : "Workspace .blend file"],
                  ...(bItem.session_ids?.length ? [["Sessions", bItem.session_ids.join(", ")]] : []),
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

  // --- Section 2: Nested Workspace Folder & Asset Hierarchy ---
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

  // --- Local & Cloud Compute Jobs (`jobs/cloud/...` & `jobs/local/...`) ---
  if (flt === "all" || flt === "jobs") {
    Array.from(state.splatJobs.values())
      .filter((j) => !q || j.job_id.toLowerCase().includes(q) || String(j.source_label || "").toLowerCase().includes(q))
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
      .forEach((j) => {
        const isLocal = String(j.gpu || "").toLowerCase().includes("local") || String(j.job_id).startsWith("local_");
        const jFolder = isLocal ? "jobs/local/3dgs" : "jobs/cloud/modal-3dgs";
        insertLeafAtFolder(jFolder, { leafType: "job", jobKind: "splat", jobKey: `splat:${j.job_id}`, path: `${jFolder}/${j.job_id}`, job: j });
      });

    Array.from(state.meshyJobs.values())
      .filter((m) => !q || m.job_id.toLowerCase().includes(q) || String(m.prompt || "").toLowerCase().includes(q))
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
      .forEach((m) => {
        const jFolder = "jobs/cloud/fal-meshy";
        insertLeafAtFolder(jFolder, { leafType: "job", jobKind: "meshy", jobKey: `meshy:${m.job_id}`, path: `${jFolder}/${m.job_id}`, job: m });
      });

    Array.from(state.jobs.values())
      .filter((j) => !q || String(j.local_id || "").toLowerCase().includes(q) || String(j.prompt || "").toLowerCase().includes(q) || String(j.output_rel || "").toLowerCase().includes(q))
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
      .forEach((j) => {
        const lid = String(j.local_id || "");
        const isLocal = j.mode === "local" || lid.startsWith("bl-local");
        const isSr = lid.startsWith("sr-") || String(j.model || "").includes("seedream");
        const jFolder = isLocal ? "jobs/local/blender" : isSr ? "jobs/cloud/seedream" : "jobs/cloud/seedance";
        insertLeafAtFolder(jFolder, { leafType: "job", jobKind: isLocal ? "local" : isSr ? "seedream" : "seedance", jobKey: `media:${lid}`, path: `${jFolder}/${lid}`, job: j });
      });
  }

  // --- Completed 3D Gaussian Splats (`assets/splats`) & Meshy GLBs (`assets/meshy`) ---
  if (flt === "all" || flt === "splat") {
    const splats = Array.from(state.splatJobs.values())
      .filter((j) => j.ply_rel && (!q || j.job_id.toLowerCase().includes(q) || String(j.source_label || "").toLowerCase().includes(q)))
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

    splats.forEach((j) => {
      let folder = "assets/splats";
      if (j.ply_rel && j.ply_rel.includes("/")) {
        const parts = j.ply_rel.replace(/\\/g, "/").split("/");
        if (parts.length >= 3) folder = parts.slice(0, -2).join("/");
        else folder = parts.slice(0, -1).join("/");
      }
      insertLeafAtFolder(folder, { leafType: "splat", job: j });
    });

    const meshys = Array.from(state.meshyJobs.values())
      .filter((m) => m.stage === "completed" && (!q || m.job_id.toLowerCase().includes(q) || String(m.prompt || "").toLowerCase().includes(q)))
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

    meshys.forEach((m) => {
      let folder = "assets/meshy";
      if (m.output_rel && m.output_rel.includes("/")) {
        const parts = m.output_rel.replace(/\\/g, "/").split("/");
        if (parts.length >= 3) folder = parts.slice(0, -2).join("/");
        else folder = parts.slice(0, -1).join("/");
      }
      insertLeafAtFolder(folder, { leafType: "meshy", job: m });
    });
  }

  // --- Workspace Media (Videos, Images, Audio) ---
  if (flt !== "splat" && flt !== "jobs") {
    state.mediaItems.forEach((it) => {
      if (flt !== "all" && it.kind !== flt) return;
      if (q && !it.path.toLowerCase().includes(q) && !it.name.toLowerCase().includes(q)) return;
      const folder = it.path.includes("/") ? it.path.split("/").slice(0, -1).join("/") : "";
      insertLeafAtFolder(folder, { leafType: "media", item: it });
    });
  }

  const activeWbJobId = actions.getActiveWbJobId ? actions.getActiveWbJobId() : null;

  const renderLeafRow = (entry, depth) => {
    if (entry.leafType === "job") {
      const j = entry.job;
      const row = document.createElement("div");
      const isAct =
        (state.centerKind === "job" && state.activeJobKey === entry.jobKey) ||
        (entry.jobKind === "splat" && state.centerKind === "kernel" && activeWbJobId === j.job_id);
      row.className = `vsc-tree-row tree-item${isAct ? " active-stage" : ""}`;
      row.tabIndex = 0;
      row.setAttribute("role", "treeitem");

      const stg = String(j.stage || "completed").toLowerCase();
      const dotCls = stg === "completed" ? "completed" : stg === "failed" ? "failed" : stg === "stopped" || stg === "cancelled" ? "stopped" : "running";
      const dispName = entry.jobKind === "splat" || entry.jobKind === "meshy" ? j.job_id : j.local_id;
      let subLbl = stg;
      if (entry.jobKind === "splat") {
        subLbl = stg === "completed" && j.psnr ? `${Number(j.psnr).toFixed(1)}dB` : j.total_steps ? `${j.step || 0}/${j.total_steps}` : stg;
      } else if (entry.jobKind === "meshy") {
        subLbl = stg === "completed" && j.triangles_total ? `${(j.triangles_total / 1000).toFixed(1)}k` : stg;
      } else if (j.cost_usd) {
        subLbl = `$${Number(j.cost_usd).toFixed(3)}`;
      }

      const thumbUrl = j.preview_url || j.thumbnail_url || (j.media_url && /\.(png|jpe?g|webp)/i.test(j.output_rel || "") ? j.media_url : null);
      const glyph = entry.jobKind === "splat" ? "🌐" : entry.jobKind === "meshy" ? "🧊" : entry.jobKind === "seedream" ? "🎨" : entry.jobKind === "local" ? "🖥️" : "🎬";
      const iconHtml = thumbUrl
        ? `<span class="vsc-node-icon has-thumb"><img src="${withTokenUrl(thumbUrl)}" loading="lazy" alt="" /></span>`
        : `<span class="vsc-node-icon">${glyph}</span>`;

      row.innerHTML = `
        ${makeIndentGuidesHtml(depth)}
        <span class="vsc-twistie spacer">·</span>
        ${iconHtml}
        <span class="vsc-label-wrap">
          <span class="job-status-dot ${dotCls}"></span>
          <span class="tree-name" title="${escapeHtml(entry.path)}">${escapeHtml(dispName)}</span>
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
      if (entry.jobKind === "splat") {
        if (j.ply_rel) mkBtn("🎮 3D", "Open trained Splat in 3D/FPS Viewport", () => open3DSceneInCenter(j.job_id, false));
        if (j.cameras_rel) mkBtn("🎬 Path", "Compare Source Video vs Trajectory", () => actions.openComparatorForJob?.(j));
      } else if (entry.jobKind === "meshy" && stg === "completed") {
        mkBtn("🧊 3D", "Open .glb Mesh in 3D Viewport", () => open3DSceneInCenter(`meshy:${j.job_id}`, false));
        mkBtn("→ BL", "Import .glb into Blender 5.3", () => actions.importMeshyToBlender?.(j.job_id));
      } else if (j.media_url && stg === "completed") {
        const isImg = /\.(png|jpe?g|webp)$/i.test(j.output_rel || "") || entry.jobKind === "seedream";
        mkBtn("👁 Media", "Open generated artifact in Media Player", () =>
          openMediaInCenter({
            path: j.output_rel,
            name: (j.output_rel || "").split("/").pop() || dispName,
            kind: isImg ? "image" : "video",
            size_bytes: j.conditioning?.size_bytes || 0,
            width: j.width,
            height: j.height,
            cost_usd: j.cost_usd,
            url: j.media_url,
          })
        );
      }

      row.addEventListener("click", () => actions.openJobInCenter?.(entry.jobKind, j, entry.jobKey, entry.path));
      row.addEventListener("mouseenter", () => {
        const specs = [
          ["Job Path", entry.path],
          ["Status", stg.toUpperCase()],
        ];
        if (entry.jobKind === "splat") {
          specs.push(["GPU / Steps", `${j.gpu || "H100"} · ${j.step || 0}/${j.total_steps || j.iterations || 0}`]);
          if (j.psnr) specs.push(["PSNR / Loss", `${j.psnr} dB · loss ${j.loss || "—"}`]);
          if (j.n_gaussians) specs.push(["Gaussians", Number(j.n_gaussians).toLocaleString()]);
        } else if (entry.jobKind === "meshy") {
          specs.push(["Endpoint", j.model || "meshy/v7.1/image-to-3d"]);
          if (j.triangles_total) specs.push(["Triangles", Number(j.triangles_total).toLocaleString()]);
        } else {
          specs.push(["Model", j.model || entry.jobKind]);
          if (j.output_rel) specs.push(["Artifact", j.output_rel]);
        }
        if (j.cost_usd) specs.push(["Cost", `$${Number(j.cost_usd).toFixed(4)} USD`]);
        showHoverTooltip(row, {
          imageUrl: thumbUrl ? withTokenUrl(thumbUrl) : null,
          glyph,
          title: dispName,
          path: `${entry.path} (Click for Status & Telemetry)`,
          specs,
        });
      });
      row.addEventListener("mouseleave", hideHoverTooltip);
      return row;
    }

    if (entry.leafType === "meshy") {
      const m = entry.job;
      const row = document.createElement("div");
      const sceneKey = `meshy:${m.job_id}`;
      const isAct = state.centerKind === "3d" && state.active3DId === sceneKey;
      row.className = `vsc-tree-row tree-item${isAct ? " active-stage" : ""}`;
      row.tabIndex = 0;
      row.setAttribute("role", "treeitem");

      const subLbl = m.triangles_total ? `${(m.triangles_total / 1000).toFixed(1)}k tris` : "GLB";
      const iconHtml = m.thumbnail_url
        ? `<span class="vsc-node-icon has-thumb"><img src="${withTokenUrl(m.thumbnail_url)}" loading="lazy" alt="" /></span>`
        : `<span class="vsc-node-icon">🧊</span>`;

      row.innerHTML = `
        ${makeIndentGuidesHtml(depth)}
        <span class="vsc-twistie spacer">·</span>
        ${iconHtml}
        <span class="vsc-label-wrap">
          <span class="tree-name" title="${escapeHtml(m.job_id)}">${escapeHtml(m.job_id)}</span>
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
      mkBtn("🎮 FPS", "Inspect 3D Mesh in FPS Mode", () => open3DSceneInCenter(sceneKey, true));
      mkBtn("→ BL", "Import .glb into Blender 5.3", () => actions.importMeshyToBlender?.(m.job_id));

      row.addEventListener("click", () => open3DSceneInCenter(sceneKey, false));
      row.addEventListener("mouseenter", () => {
        const specs = [
          ["Type", "Meshy 7.1 3D Mesh (.glb)"],
          ["Triangles", Number(m.triangles_total || 0).toLocaleString()],
        ];
        if (m.request_id) specs.push(["fal Request", String(m.request_id)]);
        if (m.cost_usd) specs.push(["Est. Cost", `$${Number(m.cost_usd).toFixed(2)}`]);
        showHoverTooltip(row, {
          imageUrl: m.thumbnail_url ? withTokenUrl(m.thumbnail_url) : null,
          glyph: "🧊",
          title: m.job_id,
          path: m.output_rel || `assets/meshy/${m.job_id}/model.glb`,
          specs,
        });
      });
      row.addEventListener("mouseleave", hideHoverTooltip);
      return row;
    }

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
      mkBtn("📊 Stats", "Open Job Status, Render/Error & Telemetry", () => actions.openKernelTelemetryForJob?.(j.job_id));
      if (j.cameras_rel) mkBtn("🎬 Path", "Compare Clip & 3D Camera Path", () => actions.openComparatorForJob?.(j));
      if (j.ply_rel) mkBtn("→ BL", "Import Splat into Blender 5.3", () => actions.importSplatToBlender?.(j.job_id));

      row.addEventListener("click", () => {
        actions.switchInspectorTab?.("fps");
        open3DSceneInCenter(j.job_id, false);
      });
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

    // Media Leaf (Video, Image, Audio)
    const it = entry.item;
    const roles = rolesForMedia(it);
    const isPrimary = roles.some((r) => r.primary);
    const isAct =
      (state.centerKind === "media" && state.stageItem && state.stageItem.path === it.path) ||
      (state.centerKind === "launch" && it.kind === "video" && state.launchVideoPath === it.path);

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
      const isM3d = state.meshyInputImage && state.meshyInputImage.path === it.path;
      addQ(is1 ? "✓ 1st" : "1st", is1 ? "active-orange" : "", "Set as Seedance First Frame", () => actions.setFirstFrame?.(it));
      addQ(isL ? "✓ Last" : "Last", isL ? "active-orange" : "", "Set as Seedance Last Frame", () => actions.setLastFrame?.(it));
      addQ(isSr ? "✓ SR" : "🎨 SR", isSr ? "active-green" : "", "Toggle Seedream Reference", () => actions.toggleSeedreamRef?.(it));
      addQ(isM3d ? "✓ 3D" : "🧊 3D", isM3d ? "active-green" : "", "Send to Meshy 7.1 Image-to-3D", () => {
        actions.setMeshyInputImage?.(it);
        actions.switchInspectorTab?.("meshy");
      });
    } else if (it.kind === "video") {
      addQ("▶ Play", "", "Play Video Clip in Center Media Player", () => openMediaInCenter(it));
      const isV1 = state.sourceVideo && state.sourceVideo.path === it.path;
      addQ(isV1 ? "✓ Vid1" : "Vid1", isV1 ? "active-green" : "", "Set as Seedance Video 1", () => actions.setSourceVideo?.(isV1 ? null : it));
      addQ("🌐 Splat", "active-orange", "Open 3DGS Splat Launcher Dialog in Center Stage", () => actions.openSplatLaunchDialog?.("video", it.path));
    } else if (it.kind === "audio") {
      const isAud = state.refAudio && state.refAudio.path === it.path;
      addQ(isAud ? "✓ Audio" : "Audio", isAud ? "active-orange" : "", "Set as Seedance Audio Reference", () => actions.setRefAudio?.(it));
    }

    // Clicking an input video (e.g. `assets/inputs/*.mp4` or when in Splat workflow) opens the full Center-Stage Splat Launch Dialog!
    row.addEventListener("click", () => {
      const isInputVideo = it.kind === "video" && (it.path.startsWith("assets/inputs/") || state.inspectorTab === "fps" || state.centerKind === "launch");
      if (isInputVideo && actions.openSplatLaunchDialog) {
        actions.openSplatLaunchDialog("video", it.path);
      } else {
        openMediaInCenter(it);
      }
    });
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
      buildVscSection("sec:workspace", "Workspace Files, Jobs & Splats", rootNode.count, forceExpand, (body) => {
        const topDirs = Array.from(rootNode.dirs.values()).sort(
          (a, b) => folderSortRank(a.name) - folderSortRank(b.name) || a.name.localeCompare(b.name)
        );
        topDirs.forEach((td) => renderDirBranch(td, 1, body));
        rootNode.leaves.forEach((lf) => body.appendChild(renderLeafRow(lf, 1)));
      })
    );
  }

  // --- Secondary Collapsed Section: 3D Test Sandboxes & FPS Stages ---
  if (flt === "all" || flt === "splat") {
    const sandboxes = BUILTIN_3D_STAGES.filter(
      (s) => s.id !== "__blender_live__" && (!q || s.name.toLowerCase().includes(q) || s.sub.toLowerCase().includes(q))
    );
    if (sandboxes.length) {
      tree.appendChild(
        buildVscSection("sec:sandboxes", "3D Test Sandboxes (FPS)", sandboxes.length, forceExpand, (body) => {
          sandboxes.forEach((stItem) => {
            const row = document.createElement("div");
            const isAct = state.centerKind === "3d" && state.active3DId === stItem.id;
            row.className = `vsc-tree-row tree-item${isAct ? " active-stage" : ""}`;
            row.tabIndex = 0;
            row.setAttribute("role", "treeitem");
            row.innerHTML = `
              ${makeIndentGuidesHtml(1)}
              <span class="vsc-twistie spacer">·</span>
              <span class="vsc-node-icon">${stItem.glyph}</span>
              <span class="vsc-label-wrap"><span class="tree-name">${escapeHtml(stItem.name)}</span></span>
              <span class="tree-quick-btns"></span>
            `;
            const fpsBtn = document.createElement("button");
            fpsBtn.type = "button";
            fpsBtn.textContent = "🎮 FPS";
            fpsBtn.addEventListener("click", (e) => {
              e.stopPropagation();
              open3DSceneInCenter(stItem.id, true);
            });
            row.querySelector(".tree-quick-btns").appendChild(fpsBtn);
            row.addEventListener("click", () => open3DSceneInCenter(stItem.id, false));
            body.appendChild(row);
          });
        })
      );
    }
  }

  // --- Secondary Collapsed Section: Experimental Cloud Generative Studios ---
  if (flt === "all") {
    const genEntries = [
      { id: "splat_launch", glyph: "🌐", name: "3DGS Cloud Splat Launcher", sub: "Modal H100" },
      { id: "seedream", glyph: "🎨", name: "Seedream 5.5 Image Gen", sub: "Experimental" },
      { id: "seedance", glyph: "🎬", name: "Seedance 2.5 Video Gen", sub: "Experimental" },
      { id: "meshy", glyph: "🧊", name: "Meshy 7.1 Image-to-3D", sub: "Experimental" },
      { id: "how", glyph: "📖", name: "3DGS Kernel & Math Reference", sub: "Docs" },
    ].filter((g) => !q || g.name.toLowerCase().includes(q));

    if (genEntries.length) {
      tree.appendChild(
        buildVscSection("sec:gen_experimental", "Cloud Gen AI & Labs", genEntries.length, forceExpand, (body) => {
          genEntries.forEach((g) => {
            const isAct =
              (g.id === "splat_launch" && state.centerKind === "launch") ||
              (g.id === "how" && state.centerKind === "how") ||
              (state.centerKind === "gen" && state.activeGenStudio === g.id);
            const row = document.createElement("div");
            row.className = `vsc-tree-row tree-item${isAct ? " active-stage" : ""}`;
            row.tabIndex = 0;
            row.setAttribute("role", "treeitem");
            row.innerHTML = `
              ${makeIndentGuidesHtml(1)}
              <span class="vsc-twistie spacer">·</span>
              <span class="vsc-node-icon">${g.glyph}</span>
              <span class="vsc-label-wrap"><span class="tree-name">${escapeHtml(g.name)}</span></span>
              <span class="tree-sub">${escapeHtml(g.sub)}</span>
            `;
            row.addEventListener("click", () => {
              if (g.id === "splat_launch") actions.openSplatLaunchDialog?.();
              else if (g.id === "how") actions.openHowItWorksInCenter?.();
              else actions.openGenerativeStudioInCenter?.(g.id);
            });
            body.appendChild(row);
          });
        })
      );
    }
  }

  if (!tree.children.length) {
    tree.innerHTML = `<div class="empty-placeholder" style="padding:20px">No matching items. Drop files or clear search filter.</div>`;
  }
}

export function syncViewModeButtons() {
  const isSess = state.centerKind === "session";
  const is3DOrbit = state.centerKind === "3d" && !(window._bbEngine && window._bbEngine.isFps());
  const isFpsActive = state.centerKind === "3d" && window._bbEngine && window._bbEngine.isFps();
  const isLaunch = state.centerKind === "launch";
  const isGen = state.centerKind === "gen";
  const isCmp = state.centerKind === "compare";
  const isKdiag = state.centerKind === "kernel" || state.centerKind === "job";
  const isHow = state.centerKind === "how";

  el("btn-topbar-session")?.classList.toggle("active", isSess);
  el("btn-view-inspect")?.classList.toggle("active", is3DOrbit || state.centerKind === "media" || state.centerKind === "blend");
  el("btn-view-fps")?.classList.toggle("active", !!isFpsActive);
  el("btn-view-launch")?.classList.toggle("active", isLaunch);
  el("btn-view-gen")?.classList.toggle("active", isGen);
  el("btn-view-compare")?.classList.toggle("active", isCmp);
  el("btn-view-kernel")?.classList.toggle("active", isKdiag);
  el("btn-view-how")?.classList.toggle("active", isHow);
  el("btn-view-compare")?.classList.toggle("hidden", !(state.activeSplatJob && state.activeSplatJob.cameras_rel));

  const viewingFirst = state.centerKind === "media" && state.stageItem && state.firstFrame && state.stageItem.path === state.firstFrame.path;
  const viewingLast = state.centerKind === "media" && state.stageItem && state.lastFrame && state.stageItem.path === state.lastFrame.path;
  el("slot-first")?.classList.toggle("viewing-center", !!viewingFirst);
  el("slot-last")?.classList.toggle("viewing-center", !!viewingLast);
  actions.syncTopbarSessionPill?.();
}

export function hideAllCenterViews() {
  el("theater-empty")?.classList.add("hidden");
  const vid = el("theater-video");
  const img = el("theater-image");
  const audWrap = el("theater-audio-wrap");
  const aud = el("theater-audio");
  vid?.pause();
  aud?.pause();
  el("launch-video-preview")?.pause();
  vid?.classList.add("hidden");
  img?.classList.add("hidden");
  audWrap?.classList.add("hidden");
  el("theater-session-wrap")?.classList.add("hidden");
  el("theater-blend-wrap")?.classList.add("hidden");
  el("theater-gen-wrap")?.classList.add("hidden");
  el("theater-splat-canvas")?.classList.add("hidden");
  el("fps-top-strip")?.classList.add("hidden");
  el("fps-crosshair-wrap")?.classList.add("hidden");
  el("fps-radar-canvas")?.classList.add("hidden");
  el("theater-splat-hud")?.classList.add("hidden");
  el("theater-compare-wrap")?.classList.add("hidden");
  el("theater-splat-launch-wrap")?.classList.add("hidden");
  el("theater-kernel-wrap")?.classList.add("hidden");
  el("theater-how-it-works-wrap")?.classList.add("hidden");
  el("theater-job-wrap")?.classList.add("hidden");
  if (window._bbStopComparator) window._bbStopComparator();
  if (window._bbEngine && document.pointerLockElement) document.exitPointerLock();
}

export function openMediaInCenter(item) {
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

  el("stage-3d-btns")?.classList.add("hidden");
  el("stage-media-btns")?.classList.remove("hidden");
  el("btn-stage-use-first")?.classList.toggle("hidden", item.kind !== "image");
  el("btn-stage-use-last")?.classList.toggle("hidden", item.kind !== "image");
  el("btn-stage-use-sr")?.classList.toggle("hidden", item.kind !== "image");
  el("btn-stage-use-meshy")?.classList.toggle("hidden", item.kind !== "image");
  el("btn-stage-use-vid")?.classList.toggle("hidden", item.kind !== "video");
  el("btn-stage-ref-vid")?.classList.toggle("hidden", item.kind !== "video");
  el("btn-stage-to-splat")?.classList.toggle("hidden", item.kind !== "video");
  const dl = el("link-stage-download");
  if (dl) {
    dl.href = `${u}&download=1`;
    dl.download = item.name || "asset";
  }

  actions.syncInspectorContext?.();
  syncViewModeButtons();
  renderHierarchyTree();
}

export async function open3DSceneInCenter(sceneId, enterFps = false) {
  hideAllCenterViews();
  state.centerKind = "3d";
  state.stageItem = null;
  state.active3DId = sceneId || "__courtyard__";

  const isMeshy = String(state.active3DId).startsWith("meshy:");
  const meshyId = isMeshy ? String(state.active3DId).slice(6) : "";
  const meshyJob = isMeshy ? state.meshyJobs.get(meshyId) || null : null;
  state.activeMeshyJob = meshyJob;

  const splatJob = !isMeshy ? state.splatJobs.get(state.active3DId) || null : null;
  state.activeSplatJob = splatJob;

  el("theater-splat-canvas")?.classList.remove("hidden");
  el("fps-top-strip")?.classList.remove("hidden");
  el("theater-splat-hud")?.classList.remove("hidden");
  el("stage-media-btns")?.classList.add("hidden");
  el("stage-3d-btns")?.classList.remove("hidden");
  el("btn-stage-save-splat")?.classList.toggle("hidden", isMeshy);

  const glbDl = el("link-stage-download-glb");
  if (glbDl) {
    if (meshyJob && meshyJob.glb_url) {
      glbDl.href = `${withTokenUrl(meshyJob.glb_url)}&download=1`;
      glbDl.download = `${meshyJob.job_id}.glb`;
      glbDl.classList.remove("hidden");
    } else {
      glbDl.classList.add("hidden");
    }
  }

  const builtIn = BUILTIN_3D_STAGES.find((s) => s.id === state.active3DId);
  el("stage-kind-badge").textContent = meshyJob ? "MESHY 3D GLB" : splatJob ? "3DGS SPLAT" : "3D STAGE";
  el("stage-title").textContent = meshyJob ? meshyJob.job_id : builtIn ? builtIn.name : state.active3DId;
  el("theater-meta").textContent = meshyJob
    ? `${Number(meshyJob.triangles_total || 0).toLocaleString()} tris${meshyJob.cost_usd ? ` · $${Number(meshyJob.cost_usd).toFixed(2)}` : ""}`
    : splatJob
    ? `${Number(splatJob.n_gaussians || 0).toLocaleString()} pts${splatJob.psnr ? ` · ${splatJob.psnr} dB` : ""}`
    : builtIn
    ? builtIn.sub
    : "";

  // Populate Right Inspector Active Splat Settings & Resume Panel whenever a completed splat is selected!
  actions.syncInspectorActiveSplatPanel?.(splatJob);

  if (window._bbEngine) {
    await window._bbEngine.loadScene(state.active3DId);
    window._bbEngine.setFpsMode(!!enterFps, false);
  }
  actions.syncInspectorContext?.();
  if (splatJob && window._bbHydrateSplatTelemetry) {
    window._bbHydrateSplatTelemetry(splatJob.job_id);
  }
  syncViewModeButtons();
  renderHierarchyTree();
}

export async function captureCenter3DIntoRole(role) {
  if (!window._bbEngine) return;
  try {
    const dataUrl = window._bbEngine.captureCleanPng();
    const res = await apiFetch("/api/workspace/save_snapshot", {
      method: "POST",
      json: { data_url: dataUrl, prefix: `view_${role}` },
    });
    const item = res.item;
    await actions.refreshMedia?.();
    if (role === "first") {
      state.firstFrame = item;
      actions.renderConditioningSlots?.();
      actions.updatePricingAndInspector?.();
      actions.switchInspectorTab?.("seedance");
      showToast(`📸 Captured 3D View → Seedance 1st Frame (${item.name})`);
    } else if (role === "last") {
      state.lastFrame = item;
      actions.renderConditioningSlots?.();
      actions.updatePricingAndInspector?.();
      actions.switchInspectorTab?.("seedance");
      showToast(`📸 Captured 3D View → Seedance Last Frame (${item.name})`);
    } else if (role === "seedream") {
      if (!state.seedreamRefs.some((r) => r.path === item.path)) {
        state.seedreamRefs.unshift(item);
        state.seedreamRefs = state.seedreamRefs.slice(0, 10);
      }
      actions.renderSeedreamRefs?.();
      actions.updateSeedreamInspector?.();
      actions.switchInspectorTab?.("seedream");
      showToast(`🎨 Captured 3D View → Seedream Reference (${item.name})`);
    } else if (role === "meshy") {
      state.meshyInputImage = item;
      actions.renderMeshySlots?.();
      actions.updateMeshyInspector?.();
      actions.switchInspectorTab?.("meshy");
      showToast(`🧊 Captured 3D View → Meshy 7.1 Input (${item.name})`);
    }
    renderHierarchyTree();
  } catch (err) {
    showToast(`3D capture failed: ${err.message}`, true);
  }
}

export function initExplorerAndStageBar() {
  el("hierarchy-tree")?.addEventListener("scroll", hideHoverTooltip);

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

  el("hierarchy-tree")?.addEventListener("keydown", (e) => {
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

  el("media-tabs")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    document.querySelectorAll("#media-tabs .tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    renderHierarchyTree();
  });
  el("media-search")?.addEventListener("input", (e) => {
    state.search = e.target.value;
    renderHierarchyTree();
  });
  el("btn-refresh-media")?.addEventListener("click", () => {
    actions.refreshSessions?.();
    actions.refreshMedia?.();
    actions.refreshSplats?.();
  });

  // Top Header Center-Stage Mode Switcher
  el("btn-view-inspect")?.addEventListener("click", () => {
    if (state.centerKind === "3d" && window._bbEngine) {
      window._bbEngine.setFpsMode(false, false);
      syncViewModeButtons();
    } else if (state.activeSplatJob) {
      open3DSceneInCenter(state.activeSplatJob.job_id, false);
    } else if (state.blenderInfo?.connected && (!state.active3DId || state.active3DId === "__courtyard__")) {
      open3DSceneInCenter("__blender_live__", false);
    } else {
      open3DSceneInCenter(state.active3DId || "__courtyard__", false);
    }
  });

  el("btn-view-fps")?.addEventListener("click", () => {
    const target3D = state.activeSplatJob ? state.activeSplatJob.job_id : state.active3DId || "__courtyard__";
    open3DSceneInCenter(target3D, true);
  });

  el("btn-view-launch")?.addEventListener("click", () => {
    actions.openSplatLaunchDialog?.();
  });

  el("btn-view-gen")?.addEventListener("click", () => {
    actions.openGenerativeStudioInCenter?.(state.activeGenStudio || "seedream");
  });

  el("btn-view-compare")?.addEventListener("click", () => {
    if (state.activeSplatJob) actions.openComparatorForJob?.(state.activeSplatJob);
  });

  el("btn-view-kernel")?.addEventListener("click", () => {
    const jid = state.activeSplatJob?.job_id || Array.from(state.splatJobs.keys())[0] || null;
    actions.openKernelTelemetryForJob?.(jid);
  });

  el("btn-view-how")?.addEventListener("click", () => {
    actions.openHowItWorksInCenter?.();
  });

  // Center Quick Action Buttons
  el("btn-3d-to-first")?.addEventListener("click", () => captureCenter3DIntoRole("first"));
  el("btn-3d-to-last")?.addEventListener("click", () => captureCenter3DIntoRole("last"));
  el("btn-3d-to-seedream")?.addEventListener("click", () => captureCenter3DIntoRole("seedream"));
  el("btn-3d-to-meshy")?.addEventListener("click", () => captureCenter3DIntoRole("meshy"));

  el("btn-stage-use-first")?.addEventListener("click", () => state.stageItem && actions.setFirstFrame?.(state.stageItem));
  el("btn-stage-use-last")?.addEventListener("click", () => state.stageItem && actions.setLastFrame?.(state.stageItem));
  el("btn-stage-use-sr")?.addEventListener("click", () => {
    if (state.stageItem) {
      actions.toggleSeedreamRef?.(state.stageItem);
      actions.switchInspectorTab?.("seedream");
    }
  });
  el("btn-stage-use-meshy")?.addEventListener("click", () => {
    if (state.stageItem) {
      actions.setMeshyInputImage?.(state.stageItem, true);
      actions.switchInspectorTab?.("meshy");
    }
  });
  el("btn-stage-use-vid")?.addEventListener("click", () => state.stageItem && actions.setSourceVideo?.(state.stageItem));
  el("btn-stage-ref-vid")?.addEventListener("click", () => state.stageItem && actions.setSourceVideo2?.(state.stageItem));
  el("btn-stage-to-splat")?.addEventListener("click", () => state.stageItem && actions.openSplatLaunchDialog?.("video", state.stageItem.path));
  el("btn-stage-at-agent")?.addEventListener("click", () => {
    if (!state.stageItem) return;
    actions.setAgentDrawerOpen?.(true, false);
    const inp = el("agent-input");
    inp.value = `${inp.value ? inp.value + " " : ""}@${state.stageItem.path} `;
    inp.focus();
  });

  el("btn-stage-save-splat")?.addEventListener("click", () => window._bbEngine && window._bbEngine.saveSplat(false));
  el("btn-stage-import-bl")?.addEventListener("click", () => {
    if (state.activeMeshyJob) actions.importMeshyToBlender?.(state.activeMeshyJob.job_id);
    else if (state.activeSplatJob) actions.importSplatToBlender?.(state.activeSplatJob.job_id);
    else if (window._bbEngine) window._bbEngine.saveSplat(true);
  });

  actions.hideHoverTooltip = hideHoverTooltip;
  actions.renderHierarchyTree = renderHierarchyTree;
  actions.syncViewModeButtons = syncViewModeButtons;
  actions.hideAllCenterViews = hideAllCenterViews;
  actions.openMediaInCenter = openMediaInCenter;
  actions.open3DSceneInCenter = open3DSceneInCenter;
  actions.captureCenter3DIntoRole = captureCenter3DIntoRole;
  actions.expandTreePath = expandTreePath;
  actions.scrollActiveTreeRowIntoView = scrollActiveTreeRowIntoView;
}