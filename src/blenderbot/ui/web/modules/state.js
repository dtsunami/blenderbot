// Shared reactive state, auth/fetch helpers, and constants for the Studio Workbench.

const urlParams = new URLSearchParams(window.location.search);
export const initialSceneParam = urlParams.get("scene") || "";
export const bootInFps =
  window.location.pathname === "/fps" ||
  window.location.hash === "#fps" ||
  !!initialSceneParam;

let token =
  urlParams.get("token") ||
  sessionStorage.getItem("bb_token") ||
  localStorage.getItem("bb_token") ||
  "";

if (urlParams.get("token")) {
  try { sessionStorage.setItem("bb_token", token); } catch (_e) {}
  try { localStorage.setItem("bb_token", token); } catch (_e) {}
  urlParams.delete("token");
  const cleanQ = urlParams.toString();
  window.history.replaceState({}, "", window.location.pathname + (cleanQ ? `?${cleanQ}` : "") + (window.location.hash || ""));
} else if (token) {
  try { sessionStorage.setItem("bb_token", token); } catch (_e) {}
  try { localStorage.setItem("bb_token", token); } catch (_e) {}
}

export function getToken() {
  return token;
}

export function setActiveToken(newTok) {
  token = String(newTok || "").trim();
  try { sessionStorage.setItem("bb_token", token); } catch (_e) {}
  try { localStorage.setItem("bb_token", token); } catch (_e) {}
}

export function withTokenUrl(rawUrl) {
  if (!rawUrl || !token) return rawUrl;
  const sep = rawUrl.includes("?") ? "&" : "?";
  return `${rawUrl}${sep}token=${encodeURIComponent(token)}`;
}

export const el = (id) => document.getElementById(id);

export function showToast(msg, isErr = false) {
  const t = el("composer-toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.toggle("err", !!isErr);
  t.classList.remove("hidden");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => t.classList.add("hidden"), 4800);
}

let authWarnShown = false;
export async function apiFetch(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const resp = await fetch(path, Object.assign({}, opts, { headers }));
  const data = await resp.json().catch(() => ({}));
  if (resp.status === 401 && !authWarnShown) {
    authWarnShown = true;
    const fb = el("btn-fal-key");
    if (fb) { fb.textContent = "Session: Auth Needed"; fb.className = "pill pill-btn pill-err"; }
    const ob = el("btn-or-key");
    if (ob) { ob.textContent = "Run /web in REPL"; ob.className = "pill pill-btn pill-warn"; }
    showToast("Web session token missing or expired — run /web or /meshy in blenderbot REPL to open an authenticated tab.", true);
  }
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  authWarnShown = false;
  return data;
}

export function formatBytes(b) {
  if (!b) return "0 B";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

export function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function hexToRgb01(hex) {
  const clean = String(hex || "#ff7b29").replace("#", "");
  const n = parseInt(clean, 16) || 0xff7b29;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export const MODE_SUFFIXES = {
  standard: "",
  orbit_plate:
    " Smooth steady camera arc with clear multi-view parallax, sharp 35mm deep-focus optics, consistent scene lighting and exposure, zero motion blur, no jump cuts.",
  subject_4d:
    " Coherent continuous subject motion, crisp silhouette separation, steady camera framing, consistent lighting, no abrupt cuts.",
};

export const SEEDREAM_MODE_SUFFIXES = {
  standard: "",
  concept_plate:
    " Crisp 35mm deep-focus production concept plate, clean architectural/spatial geometry, coherent physical lighting, zero lens distortion, suitable for 3D projection & Seedance conditioning.",
  pbr_texture:
    " Seamless orthographic flat-lit surface texture, even diffuse illumination, zero cast shadows, zero perspective vignetting, high micro-surface detail.",
  cutout_prop:
    " Isolated clean hero 3D prop asset, crisp silhouette edges, studio three-point rim & key lighting, centered framing with clean alpha separation.",
};

export const SEEDREAM_RES_COST = { "1K": 0.02, "1.5K": 0.03, "2K": 0.04 };

export const DURATION_CHOICES = [
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

export const SIZE_CHOICES = [
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

export const BUILTIN_3D_STAGES = [
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

export const state = {
  mediaItems: [],
  blendItems: [],
  blenderInfo: { connected: false },
  imageModels: [{ id: "bytedance-seed/seedream-5-5", name: "ByteDance Seed: Seedream 5.5", cost_per_image_usd: 0.02 }],
  videoModels: [{ id: "bytedance/seedance-2.5", name: "ByteDance: Seedance 2.5", cost_per_sec_usd: 0.1028 }],
  filter: "all",
  search: "",
  collapsedGroups: new Set(["sec:sandboxes", "sec:gen_experimental"]),

  // Center Stage State: "session" | "3d" | "blend" | "media" | "launch" | "gen" | "compare" | "kernel" | "how" | "job"
  centerKind: "3d",
  active3DId: "__courtyard__",
  activeGenStudio: "seedream", // "seedream" | "seedance" | "meshy"
  activeSplatJob: null,
  activeMeshyJob: null,
  activeJobKey: null,
  stageItem: null,
  stageBlendItem: null,
  launchVideoPath: "",
  prevNonSessionView: null,

  // Sessions & Agent State
  activeSessionId: "",
  viewedSessionId: "",
  sessionsList: [],
  sessionsAllWorkspaces: false,
  journalState: null,
  pendingApprovals: [],
  providerCatalog: null,
  rolesList: ["viewer", "builder", "master"],
  agentRole: "builder",
  agentModelId: "",

  // Context-driven Inspector (legacy alias preserved for callers)
  inspectorTab: "3d",

  mode: "standard",
  seedreamMode: "standard",
  seedreamRefs: [],
  meshyInputImage: null,
  meshyTextureImage: null,
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

  splatSourceMode: "video",
  splatJobs: new Map(),
  meshyJobs: new Map(),
  jobs: new Map(),
  exportedShots: [],
  promptHistory: JSON.parse(localStorage.getItem("bb_agent_prompts") || "[]"),
  genPromptHistory: JSON.parse(localStorage.getItem("bb_gen_prompts") || "[]"),
  agentPricing: { input_per_1m_usd: null, output_per_1m_usd: null },
  ledger: null,
  sessionCost: 0.0,
  liveAgentNode: null,
  liveAgentRaw: "",

  leftPaneW: 300,
  rightPaneW: 390,
  hideLeft: false,
  hideRight: false,
  agentUnread: 0,
  agentBusy: false,
  allTreeBranchKeys: [],
};

// Simple cross-module command hooks (avoids circular imports between stage/inspector/tree)
export const actions = {};