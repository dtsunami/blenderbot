// blenderbot Studio Workbench Entrypoint (ES Module Orchestrator).
// Modules live in `/assets/modules/*.js`:
//   - state.js            : auth, apiFetch, shared reactive state & constants
//   - sessions.js         : Agent Sessions (live chat, past transcripts, continue/new, approvals, `.blend` association & Journal Undo/Redo)
//   - workbench.js        : resizable 3-pane layout, splitters, Context-Driven Inspector & Experimental Gen AI Studio
//   - explorer.js         : Left Hierarchy Browser (Agent Sessions, Blender Scenes, Workspace Assets, Jobs, Sandboxes & Gen AI)
//   - splat_launcher.js   : Center-Stage 3DGS Splat Launch Dialog (`openSplatLaunchDialog`), `evalArcScale` & `wireAdaptiveStrategyArc`
//   - splat_telemetry.js  : Job Status Stage (`openKernelTelemetryForJob`), `openHowItWorksInCenter`, & Trajectory Comparator
//   - composers.js        : Seedance 2.5, Seedream 5.5, Meshy 7.1, `openJobInCenter` & SSE stream
//   - fps_engine.js       : Multi-pass WebGL2 3D / Minecraft FPS / Splat Brush Engine
//   - param_docs.js       : `META_PARAM_DOCS` & `wireMetaParamTooltips`

import {
  connectSSE,
  initComposersAndJobs,
  populatePricingDropdowns,
  populateSeedreamModels,
  refreshMedia,
  refreshModels,
  refreshShotsList,
  refreshSplats,
  refreshState,
  renderConditioningSlots,
  renderMeshySlots,
  renderSeedreamRefs,
  updateMeshyInspector,
  updatePricingAndInspector,
  updateSeedreamInspector,
} from "./modules/composers.js";
import { initExplorerAndStageBar, open3DSceneInCenter } from "./modules/explorer.js";
import { createUnified3DEngine } from "./modules/fps_engine.js";
import { META_PARAM_DOCS, wireMetaParamTooltips } from "./modules/param_docs.js";
import { initSessionsModule, openSessionInCenter, refreshSessions } from "./modules/sessions.js";
import { evalArcScale, initSplatLauncherAndResume, openSplatLaunchDialog, updateSplatCostEstimate, wireAdaptiveStrategyArc } from "./modules/splat_launcher.js";
import { initSplatTelemetryAndComparator, openHowItWorksInCenter, openKernelTelemetryForJob } from "./modules/splat_telemetry.js";
import { bootInFps, el, initialSceneParam, state } from "./modules/state.js";
import { initWorkbenchChrome, openGenerativeStudioInCenter, renderPromptHistoryDropdown, syncInspectorContext } from "./modules/workbench.js";

// Keep key symbols exported / attached for debugging and smoke checks
window._bbStudio = {
  state,
  META_PARAM_DOCS,
  evalArcScale,
  wireAdaptiveStrategyArc,
  wireMetaParamTooltips,
  openSplatLaunchDialog,
  openKernelTelemetryForJob,
  openHowItWorksInCenter,
  openSessionInCenter,
  openGenerativeStudioInCenter,
};

(async function boot() {
  initWorkbenchChrome();
  initSessionsModule();
  initExplorerAndStageBar();
  initSplatLauncherAndResume();
  initSplatTelemetryAndComparator();
  initComposersAndJobs();
  wireMetaParamTooltips();

  renderPromptHistoryDropdown();
  populatePricingDropdowns();
  populateSeedreamModels();
  renderConditioningSlots();
  renderSeedreamRefs();
  renderMeshySlots();
  updatePricingAndInspector();
  updateSeedreamInspector();
  updateMeshyInspector();
  updateSplatCostEstimate();

  window._bbEngine = createUnified3DEngine(el("theater-splat-canvas"));
  await Promise.all([refreshState(), refreshSessions(), refreshMedia(), refreshSplats(), refreshShotsList()]);
  refreshModels();
  connectSSE();

  const hash = window.location.hash;
  if (bootInFps || initialSceneParam) {
    let start3D = "__blender_live__";
    if (initialSceneParam === "cube") start3D = "__default_cube_solid__";
    else if (initialSceneParam === "splat_cube") start3D = "__default_cube_splat__";
    else if (initialSceneParam === "courtyard") start3D = "__courtyard__";
    else if (initialSceneParam === "blender") start3D = "__blender_live__";
    else if (initialSceneParam && state.splatJobs.has(initialSceneParam)) start3D = initialSceneParam;
    else if (!state.blenderInfo?.connected) start3D = "__courtyard__";
    await open3DSceneInCenter(start3D, bootInFps);
  } else if (hash === "#meshy" || hash === "#seedream" || hash === "#seedance") {
    openGenerativeStudioInCenter(hash.slice(1));
  } else if (hash === "#splat") {
    openSplatLaunchDialog();
  } else {
    // Primary blenderbot workflow: open the Active Agent Session in Center Stage (or Live Blender if requested)
    await openSessionInCenter(state.activeSessionId || "active", true);
  }
  syncInspectorContext();
})();