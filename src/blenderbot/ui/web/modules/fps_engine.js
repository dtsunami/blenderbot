// Multi-Pass Depth-Tested WebGL2 3D / Minecraft FPS / Splat Culling & Shot Rail Engine.

import { actions, apiFetch, el, getToken, hexToRgb01, showToast, state, withTokenUrl } from "./state.js";

export function createUnified3DEngine(canvas) {
  if (!canvas) return null;
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
    if (el("btn-undo")) el("btn-undo").disabled = undoStack.length === 0;
    if (el("btn-redo")) el("btn-redo").disabled = redoStack.length === 0;
    el("btn-restore-orig")?.classList.toggle("hidden", !st.hasOrigBackup);
    if (!splatCount || !flagData) {
      if (el("badge-splat-stats")) el("badge-splat-stats").textContent = "Solid Stage";
      if (el("lbl-selected-cnt")) el("lbl-selected-cnt").textContent = "0";
      return;
    }
    let culled = 0, sel = 0;
    for (let i = 0; i < splatCount; i++) {
      if (flagData[i] & 1) culled++;
      else if (flagData[i] & 2) sel++;
    }
    if (el("badge-splat-stats")) {
      el("badge-splat-stats").textContent =
        `${(splatCount - culled).toLocaleString()} / ${splatCount.toLocaleString()} pts` + (culled ? ` (${culled.toLocaleString()} culled)` : "");
    }
    if (el("lbl-selected-cnt")) el("lbl-selected-cnt").textContent = String(sel);
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
    const [tr, tg, tb] = hexToRgb01(el("inp-paint-color")?.value);

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
      const sop = s.mode === "tint" ? { op: "brush_paint", points: s.points, radius: s.radius, color_rgb: hexToRgb01(el("inp-paint-color")?.value), blend: 0.55 } : { op: "brush_paint", points: s.points, radius: s.radius, opacity_mult: 0.65 };
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
    if (el("lbl-filter-preview")) el("lbl-filter-preview").textContent = `${m.toLocaleString()} marked`;
  }

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
      const dur = Math.max(1, Number(el("inp-rail-dur")?.value || 5));
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
    el("btn-mode-walk")?.classList.toggle("active-mode", mode === "walk");
    el("btn-mode-fly")?.classList.toggle("active-mode", mode === "fly");
    el("btn-mode-freecam")?.classList.toggle("active-mode", mode === "freecam");
    const b = el("badge-locomotion");
    if (b) {
      b.textContent = mode === "walk" ? "🚶 WALK 1.65m" : mode === "fly" ? "🛸 CREATIVE FLY" : "🎥 3D FLY";
      b.className = `hud-badge ${mode === "walk" ? "badge-walk" : "badge-fly"}`;
    }
    if (announce) showToast(mode === "walk" ? "🚶 Walk Mode (Space×2 = Creative Fly)" : "🛸 Minecraft Creative Fly (Space=Up · Shift/C=Down · Ctrl=Sprint)");
  }

  function stepPhysics(dt) {
    if (state.centerKind !== "3d" || !st.fpsMode || st.riding) return;
    if (!el("fps-ping-modal")?.classList.contains("hidden") || !el("fps-help-modal")?.classList.contains("hidden")) return;

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
    el("fps-crosshair-wrap")?.classList.toggle("hidden", !st.fpsMode);
    el("fps-radar-canvas")?.classList.toggle("hidden", !st.fpsMode);
    if (el("btn-splat-fps-toggle")) el("btn-splat-fps-toggle").textContent = st.fpsMode ? "🌐 Orbit View (F)" : "🎮 Enter FPS (F)";
    actions.syncViewModeButtons?.();
    if (st.fpsMode && reqLock) canvas.requestPointerLock?.();
    dirtySort = true;
  }

  function syncHotbar() {
    if (el("cnt-hb-cam")) el("cnt-hb-cam").textContent = String(st.camNodes.length);
    if (el("cnt-hb-sub")) el("cnt-hb-sub").textContent = String(st.subjectNodes.length);
    if (el("cnt-hb-vox")) el("cnt-hb-vox").textContent = String(st.voxels.length);
    if (el("cnt-hb-ping")) el("cnt-hb-ping").textContent = String(st.pings.length);
    el("fps-hotbar")?.querySelectorAll(".hotbar-slot").forEach((b) => b.classList.toggle("active", Number(b.dataset.slot) === st.slot));
    if (el("btn-rail-track-sub")) el("btn-rail-track-sub").textContent = `🎯 Subj: ${st.trackSubject ? "ON" : "OFF"}`;
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
      st.showSolid = true; if (el("chk-show-solid")) el("chk-show-solid").checked = true;
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
      if (yard) { buildSolidStage("courtyard"); st.showSolid = true; if (el("chk-show-solid")) el("chk-show-solid").checked = true; }
      else { baseMeshFloats = new Float32Array(0); baseColliders = []; solidDirty = true; st.showSolid = false; if (el("chk-show-solid")) el("chk-show-solid").checked = false; }
      st.showSplats = true; if (el("chk-show-splats")) el("chk-show-splats").checked = true;
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
          st.showSolid = true; if (el("chk-show-solid")) el("chk-show-solid").checked = true;
          showToast(`🟠 Streamed ${res.triangles.toLocaleString()} triangles from Blender 5.3`);
        } else {
          buildSolidStage("default_cube");
          showToast("Blender scene empty — showing Default Cube");
        }
        updateSplatStatsHud();
      } catch (e) { showToast(e.message, true); }
      return;
    }

    if (String(choice).startsWith("meshy:")) {
      const mId = String(choice).slice(6);
      try {
        const res = await apiFetch(`/api/meshy/triangles/${encodeURIComponent(mId)}`);
        splatCount = 0; positions = null; flagData = null; undoStack = []; redoStack = [];
        st.activeJobId = ""; st.hasOrigBackup = false;
        buildSolidStage("blank_grid");
        if (res.vertex_floats?.length) {
          const glbFloats = new Float32Array(res.vertex_floats);
          const merged = new Float32Array(baseMeshFloats.length + glbFloats.length);
          merged.set(baseMeshFloats, 0);
          merged.set(glbFloats, baseMeshFloats.length);
          baseMeshFloats = merged;
          if (res.world_aabb) {
            baseColliders = [{ min: res.world_aabb.min, max: res.world_aabb.max }];
            sceneCenter = res.world_aabb.center.slice();
            const sz = res.world_aabb.size;
            sceneRadius = Math.max(1.2, Math.hypot(sz[0], sz[1], sz[2]) * 0.65);
          }
          solidDirty = true;
        }
        st.showSolid = true; if (el("chk-show-solid")) el("chk-show-solid").checked = true;
        teleportPreset("hero");
        if (el("badge-splat-stats")) el("badge-splat-stats").textContent = `🧊 ${Number(res.total_triangles || res.triangles || 0).toLocaleString()} tris (.glb)`;
      } catch (e) { showToast(e.message, true); }
      return;
    }

    try {
      const ens = await apiFetch(`/api/splats/ensure_splat/${encodeURIComponent(choice)}`, { method: "POST", json: {} });
      st.activeJobId = choice;
      st.hasOrigBackup = !!ens.has_orig;
      baseMeshFloats = new Float32Array(0); baseColliders = []; solidDirty = true;
      st.showSplats = true; if (el("chk-show-splats")) el("chk-show-splats").checked = true;
      const resp = await fetch(withTokenUrl(ens.splat_url));
      if (resp.ok) ingestSplatBuf(await resp.arrayBuffer());
      teleportPreset("frame");

      const jMeta = state.splatJobs.get(choice);
      if (jMeta && jMeta.cameras_url) {
        const cr = await fetch(withTokenUrl(jMeta.cameras_url)).catch(() => null);
        if (cr && cr.ok) {
          const cdoc = await cr.json().catch(() => null);
          const c0 = cdoc?.cameras?.[0];
          if (c0?.location && c0?.matrix_world) {
            const loc = c0.location;
            const mw = c0.matrix_world;
            const fx = -mw[0][2], fy = -mw[1][2], fz = -mw[2][2];
            st.eye = loc.slice();
            st.yaw = Math.atan2(fx, fy);
            st.pitch = Math.asin(Math.max(-0.98, Math.min(0.98, fz)));
            st.lensMm = Math.max(14, Math.min(105, Math.round(c0.lens_mm || 24)));
            if (el("sel-lens")) el("sel-lens").value = String(PRIME_LENSES.reduce((b, v) => (Math.abs(v - st.lensMm) < Math.abs(b - st.lensMm) ? v : b), 24));
            st.orbitTarget = [loc[0] + fx * 2.2, loc[1] + fy * 2.2, loc[2] + fz * 2.2];
            st.orbitDist = 2.2;
            st.orbitTheta = Math.atan2(loc[1] - st.orbitTarget[1], loc[0] - st.orbitTarget[0]);
            st.orbitPhi = Math.asin(Math.max(-0.95, Math.min(0.95, (loc[2] - st.orbitTarget[2]) / 2.2)));
            dirtySort = true;
          }
        }
      }
    } catch (e) { showToast(e.message, true); }
  }

  async function importLocalSplatFile(file) {
    const tok = getToken();
    try {
      showToast(`Importing ${file.name}…`);
      const buf = await file.arrayBuffer();
      const resp = await fetch(`/api/splats/upload?filename=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: Object.assign({ "X-Filename": file.name }, tok ? { Authorization: `Bearer ${tok}` } : {}),
        body: buf,
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      state.splatJobs.set(data.job.job_id, data.job);
      await actions.open3DSceneInCenter?.(data.job.job_id, true);
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
          save_as_name: (el("inp-save-name")?.value || "").trim() || undefined,
          ops: recordedOps,
          drop_indices: drop,
          authoritative_mask: true,
          loaded_count: splatCount,
          sync_blender: syncBl,
        },
      });
      st.activeJobId = res.job_id;
      st.hasOrigBackup = !!res.has_orig;
      if (el("inp-save-name")) el("inp-save-name").value = "";
      await actions.refreshSplats?.();
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
    if (st.slot === 4) { if (btn === 0 && isFilterActive()) el("btn-flt-apply")?.click(); return; }
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
        const [vr, vg, vb] = hexToRgb01(el("inp-paint-color")?.value);
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
        if (el("rng-brush-reach")) el("rng-brush-reach").value = String(st.brushReach);
        if (el("lbl-brush-reach")) el("lbl-brush-reach").textContent = `${st.brushReach.toFixed(1)}m`;
        return;
      }
      const ci = PRIME_LENSES.reduce((b, v, idx) => (Math.abs(v - st.lensMm) < Math.abs(PRIME_LENSES[b] - st.lensMm) ? idx : b), 4);
      st.lensMm = PRIME_LENSES[Math.max(0, Math.min(PRIME_LENSES.length - 1, ci + (e.deltaY < 0 ? 1 : -1)))];
      if (el("sel-lens")) el("sel-lens").value = String(st.lensMm);
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
    if (e.code === "Tab") { e.preventDefault(); actions.toggleZenMode?.(); return; }
    if (e.code === "KeyH" && state.centerKind === "3d") { e.preventDefault(); el("fps-help-modal")?.classList.toggle("hidden"); return; }
    if (e.code === "KeyF" && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      if (state.centerKind !== "3d") actions.open3DSceneInCenter?.(state.active3DId, true);
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
      st.fixedReach = !st.fixedReach; if (el("chk-fixed-reach")) el("chk-fixed-reach").checked = st.fixedReach;
    } else if (e.code === "KeyP") el("btn-rail-ride")?.click();
    else if (e.code === "KeyK") el("btn-rail-track-sub")?.click();
    else if (e.code === "Delete") {
      if (st.slot === 2 && st.cropBox) el("btn-crop-cull-in")?.click();
      else el("btn-sel-delete")?.click();
    } else if (e.code === "BracketLeft" || e.code === "BracketRight") {
      const dir = e.code === "BracketRight" ? 1 : -1;
      if (st.slot === 7) { st.voxelSizeIdx = Math.max(0, Math.min(2, st.voxelSizeIdx + dir)); showToast(`Voxel: ${VOXEL_SIZES[st.voxelSizeIdx]}m`); }
      else {
        const ci = BRUSH_STEPS.findIndex((v) => v >= st.brushRadius - 0.01);
        st.brushRadius = BRUSH_STEPS[Math.max(0, Math.min(BRUSH_STEPS.length - 1, (ci < 0 ? 4 : ci) + dir))];
        if (el("rng-brush-rad")) el("rng-brush-rad").value = String(st.brushRadius);
        if (el("lbl-brush-rad")) el("lbl-brush-rad").textContent = `${st.brushRadius.toFixed(2)}m`;
      }
    }
  });
  window.addEventListener("keyup", (e) => { if (e.code === "KeyB") st.binoZoom = false; st.keys.delete(e.code); });

  el("btn-splat-fps-toggle")?.addEventListener("click", () => setFpsMode(!st.fpsMode, !st.fpsMode));
  el("btn-splat-reset-cam")?.addEventListener("click", () => teleportPreset("hero"));
  el("btn-undo")?.addEventListener("click", () => runUndoRedo(true));
  el("btn-redo")?.addEventListener("click", () => runUndoRedo(false));
  el("btn-fps-help")?.addEventListener("click", () => el("fps-help-modal")?.classList.toggle("hidden"));
  el("btn-close-fps-help")?.addEventListener("click", () => el("fps-help-modal")?.classList.add("hidden"));
  el("fps-hotbar")?.addEventListener("click", (e) => {
    const b = e.target.closest(".hotbar-slot");
    if (b?.dataset.slot) { st.slot = Number(b.dataset.slot); syncHotbar(); }
  });

  ["inp-cam-x", "inp-cam-y", "inp-cam-z", "inp-cam-yaw", "inp-cam-pitch", "inp-cam-roll"].forEach((id) => {
    el(id)?.addEventListener("change", () => {
      st.eye = [Number(el("inp-cam-x")?.value || 0), Number(el("inp-cam-y")?.value || 0), Number(el("inp-cam-z")?.value || 1.65)];
      st.yaw = (Number(el("inp-cam-yaw")?.value || 0) * Math.PI) / 180;
      st.pitch = Math.max(-1.48, Math.min(1.48, (Number(el("inp-cam-pitch")?.value || 0) * Math.PI) / 180));
      st.rollDeg = Number(el("inp-cam-roll")?.value || 0);
      dirtySort = true;
    });
  });
  el("sel-lens")?.addEventListener("change", (e) => { st.lensMm = Number(e.target.value || 35); });
  el("rng-speed")?.addEventListener("input", (e) => { st.speedMult = Number(e.target.value || 1); if (el("lbl-speed")) el("lbl-speed").textContent = `${st.speedMult.toFixed(1)}×`; });
  el("btn-mode-walk")?.addEventListener("click", () => setLocomotion("walk", true));
  el("btn-mode-fly")?.addEventListener("click", () => setLocomotion("fly", true));
  el("btn-mode-freecam")?.addEventListener("click", () => setLocomotion("freecam", true));
  el("chk-double-space")?.addEventListener("change", (e) => { st.doubleSpaceFly = e.target.checked; });
  el("chk-auto-land")?.addEventListener("change", (e) => { st.autoLand = e.target.checked; });
  el("pov-presets")?.addEventListener("click", (e) => { const b = e.target.closest("[data-pov]"); if (b) teleportPreset(b.dataset.pov); });
  el("btn-copy-pose")?.addEventListener("click", () => {
    const c = computeCam();
    navigator.clipboard?.writeText(JSON.stringify({ eye: c.eye, forward: [c.fx, c.fy, c.fz], lens_mm: c.lensMm }, null, 2));
    showToast("Copied Camera Pose JSON");
  });

  el("btn-push-bl-cam")?.addEventListener("click", async () => {
    const c = computeCam();
    await apiFetch("/api/fps/camera_sync", { method: "POST", json: { eye: c.eye, forward: [c.fx, c.fy, c.fz], roll_deg: st.rollDeg, lens_mm: c.lensMm } })
      .then(() => showToast("🎥 Synced Camera Pose → Blender 5.3!"))
      .catch((e) => showToast(e.message, true));
  });
  el("btn-pull-bl-cam")?.addEventListener("click", async () => {
    await apiFetch("/api/fps/camera_pull")
      .then((r) => { st.eye = r.eye.slice(); st.yaw = r.yaw_rad; st.pitch = r.pitch_rad; st.lensMm = Math.round(r.lens_mm || 35); setFpsMode(true, false); dirtySort = true; showToast(`📥 Pulled '${r.camera}'`); })
      .catch((e) => showToast(e.message, true));
  });
  el("btn-spawn-bl-cube")?.addEventListener("click", () => apiFetch("/api/fps/spawn_default_cube_blender", { method: "POST", json: {} }).then(() => showToast("🧊 Spawned Default Cube in Blender 5.3!")).catch((e) => showToast(e.message, true)));
  el("btn-pull-bl-mesh")?.addEventListener("click", () => actions.open3DSceneInCenter?.("__blender_live__", true));
  el("chk-live-link-bl")?.addEventListener("change", (e) => { st.liveLinkBlender = e.target.checked; });

  el("btn-rail-add-node")?.addEventListener("click", () => {
    const c = computeCam();
    st.camNodes.push({ position: c.eye.slice(), forward: [c.fx, c.fy, c.fz], lens_mm: st.lensMm, roll_deg: st.rollDeg });
    syncHotbar(); showToast(`🎥 Added Cam Node #${st.camNodes.length}`);
  });
  el("btn-rail-ride")?.addEventListener("click", () => {
    if (!st.camNodes.length) return showToast("Add Cam Nodes first!", true);
    st.riding = !st.riding; st.rideStartT = performance.now();
    el("btn-rail-ride").textContent = st.riding ? "⏹ Stop" : "▶ Ride (P)";
  });
  el("btn-rail-track-sub")?.addEventListener("click", () => { st.trackSubject = !st.trackSubject; syncHotbar(); });
  el("btn-rail-clear")?.addEventListener("click", () => { st.camNodes = []; st.subjectNodes = []; st.voxels = []; st.pings = []; st.riding = false; solidDirty = true; syncHotbar(); });

  function shotPayload(extra = {}) {
    const nodes = st.camNodes.length ? st.camNodes : [(() => { const c = computeCam(); return { position: c.eye, forward: [c.fx, c.fy, c.fz], lens_mm: st.lensMm, roll_deg: st.rollDeg }; })()];
    const realJob = st.activeJobId && !st.activeJobId.startsWith("__") ? st.activeJobId : null;
    return Object.assign({
      splat_job_id: realJob, cam_nodes: nodes, subject_nodes: st.subjectNodes, track_subject: st.trackSubject,
      voxels: st.voxels, pings: st.pings, grounded_walk: st.locomotion === "walk",
      duration_sec: Math.max(2, Number(el("inp-rail-dur")?.value || 5)), fps: 24, width: 1280, height: 720,
      first_frame_b64: captureCleanPng(evalRailCam(0)),
      last_frame_b64: nodes.length > 1 ? captureCleanPng(evalRailCam(1)) : null,
    }, extra);
  }

  el("btn-rail-to-seedance")?.addEventListener("click", async () => {
    try {
      const res = await apiFetch("/api/shots/export", { method: "POST", json: shotPayload() });
      await actions.refreshMedia?.();
      await actions.refreshShotsList?.();
      if (res.first_frame_rel) state.firstFrame = { path: res.first_frame_rel, name: "first_frame.png", kind: "image", url: res.first_frame_url };
      if (res.last_frame_rel) state.lastFrame = { path: res.last_frame_rel, name: "last_frame.png", kind: "image", url: res.last_frame_url };
      if (res.brief?.optical_prompt && el("input-prompt")) el("input-prompt").value = res.brief.optical_prompt;
      actions.renderConditioningSlots?.(); actions.updatePricingAndInspector?.();
      actions.switchInspectorTab?.("seedance");
      showToast(`Loaded Rail '${res.shot_id}' into Seedance 2.5!`);
    } catch (e) { showToast(e.message, true); }
  });
  el("btn-rail-farm-modal")?.addEventListener("click", async () => {
    if (!st.activeJobId || st.activeJobId.startsWith("__")) return showToast("Save Splat first to farm Modal render!", true);
    try {
      const res = await apiFetch("/api/shots/export", { method: "POST", json: shotPayload({ action: "render_modal" }) });
      await actions.refreshMedia?.();
      if (res.modal_render_url) actions.openMediaInCenter?.({ path: res.modal_render_rel, name: `${res.shot_id}.mp4`, kind: "video", url: res.modal_render_url });
    } catch (e) { showToast(e.message, true); }
  });
  el("btn-rail-sync-bl")?.addEventListener("click", async () => {
    try {
      const res = await apiFetch("/api/shots/export", { method: "POST", json: shotPayload({ action: "sync_blender", send_brief_to_agent: st.pings.length > 0 }) });
      showToast(`Synced DirectorRailCam + ${res.brief_rel} → Blender 5.3!`);
    } catch (e) { showToast(e.message, true); }
  });

  el("sel-splat-vis")?.addEventListener("change", (e) => { st.splatVisMode = Number(e.target.value || 0); });
  el("rng-splat-scale")?.addEventListener("input", (e) => { st.splatScaleMult = Number(e.target.value || 1); if (el("lbl-splat-scale")) el("lbl-splat-scale").textContent = `${st.splatScaleMult.toFixed(2)}×`; });
  el("chk-show-splats")?.addEventListener("change", (e) => { st.showSplats = e.target.checked; });
  el("chk-show-solid")?.addEventListener("change", (e) => { st.showSolid = e.target.checked; });
  el("chk-show-grid")?.addEventListener("change", (e) => { st.showGrid = e.target.checked; });
  el("chk-show-gizmos")?.addEventListener("change", (e) => { st.showGizmos = e.target.checked; });
  el("btn-reset-splat-edits")?.addEventListener("click", () => mutateFlags((i) => (flagData[i] & 3) !== 0, (f) => f & ~3, null, "↺ Unculled all"));
  el("rng-brush-rad")?.addEventListener("input", (e) => { st.brushRadius = Number(e.target.value); if (el("lbl-brush-rad")) el("lbl-brush-rad").textContent = `${st.brushRadius.toFixed(2)}m`; });
  el("rng-brush-reach")?.addEventListener("input", (e) => { st.brushReach = Number(e.target.value); if (el("lbl-brush-reach")) el("lbl-brush-reach").textContent = `${st.brushReach.toFixed(1)}m`; });
  el("chk-fixed-reach")?.addEventListener("change", (e) => { st.fixedReach = e.target.checked; });

  el("paint-submodes")?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-pmode]"); if (!b) return;
    st.paintSubmode = b.dataset.pmode;
    el("paint-submodes").querySelectorAll(".mode-pill").forEach((x) => x.classList.toggle("active", x === b));
    st.slot = 3; syncHotbar();
  });
  el("btn-sel-delete")?.addEventListener("click", () => mutateFlags((i) => (flagData[i]&2)!==0 && (flagData[i]&1)===0, (f) => (f|1)&~2, null, "🔥 Culled selected"));
  el("btn-sel-keep")?.addEventListener("click", () => mutateFlags((i) => (flagData[i]&1)===0 && (flagData[i]&2)===0, (f) => f|1, null, "🎯 Kept selected"));
  el("btn-sel-invert")?.addEventListener("click", () => { if (!flagData) return; for (let i = 0; i < splatCount; i++) if ((flagData[i]&1)===0) flagData[i] ^= 2; uploadFlags(); updateSplatStatsHud(); });

  el("btn-crop-here")?.addEventListener("click", () => {
    const c = st.lastHit ? st.lastHit.pt : sceneCenter;
    st.cropBox = { min: [c[0]-1.5, c[1]-1.5, Math.max(-0.2, c[2]-1.0)], max: [c[0]+1.5, c[1]+1.5, c[2]+1.6] };
    st.slot = 2; syncHotbar(); showToast("📦 Spawned 3D Crop Box");
  });
  el("btn-crop-clear")?.addEventListener("click", () => { st.cropBox = null; st.cropCornerA = null; });
  el("btn-crop-cull-in")?.addEventListener("click", () => {
    if (!st.cropBox) return showToast("Spawn Crop Box first!", true);
    const lo = st.cropBox.min, hi = st.cropBox.max;
    mutateFlags((i) => (flagData[i]&1)===0 && positions[3*i]>=lo[0] && positions[3*i]<=hi[0] && positions[3*i+1]>=lo[1] && positions[3*i+1]<=hi[1] && positions[3*i+2]>=lo[2] && positions[3*i+2]<=hi[2], (f) => (f|1)&~2, { op: "aabb_cull", bounds_min: lo, bounds_max: hi }, "🔥 Culled inside box");
  });
  el("btn-crop-keep-in")?.addEventListener("click", () => {
    if (!st.cropBox) return showToast("Spawn Crop Box first!", true);
    const lo = st.cropBox.min, hi = st.cropBox.max;
    mutateFlags((i) => (flagData[i]&1)===0 && !(positions[3*i]>=lo[0] && positions[3*i]<=hi[0] && positions[3*i+1]>=lo[1] && positions[3*i+1]<=hi[1] && positions[3*i+2]>=lo[2] && positions[3*i+2]<=hi[2]), (f) => (f|1)&~2, { op: "aabb_keep", bounds_min: lo, bounds_max: hi }, "🎯 Kept inside box");
  });

  const syncFilterSliders = () => {
    st.fltMinOpa = Number(el("rng-flt-opa")?.value || 0);
    st.fltMaxScale = Number(el("rng-flt-scale")?.value || 1.5);
    st.fltMaxAniso = Number(el("rng-flt-aniso")?.value || 30);
    st.fltMinZ = Number(el("rng-flt-minz")?.value || -3);
    if (el("lbl-flt-opa")) el("lbl-flt-opa").textContent = st.fltMinOpa > 0.005 ? st.fltMinOpa.toFixed(2) : "OFF";
    if (el("lbl-flt-scale")) el("lbl-flt-scale").textContent = st.fltMaxScale < 1.48 ? `${st.fltMaxScale.toFixed(2)}m` : "OFF";
    if (el("lbl-flt-aniso")) el("lbl-flt-aniso").textContent = st.fltMaxAniso < 29.5 ? `${st.fltMaxAniso.toFixed(0)}×` : "OFF";
    if (el("lbl-flt-minz")) el("lbl-flt-minz").textContent = st.fltMinZ > -2.95 ? `${st.fltMinZ.toFixed(2)}m` : "OFF";
    updateFilterPreview();
  };
  ["rng-flt-opa", "rng-flt-scale", "rng-flt-aniso", "rng-flt-minz"].forEach((id) => el(id)?.addEventListener("input", syncFilterSliders));
  el("btn-flt-reset")?.addEventListener("click", () => {
    el("rng-flt-opa").value = "0"; el("rng-flt-scale").value = "1.5"; el("rng-flt-aniso").value = "30"; el("rng-flt-minz").value = "-3";
    syncFilterSliders();
  });
  el("btn-flt-apply")?.addEventListener("click", () => {
    if (!isFilterActive()) return showToast("Adjust a filter slider first!");
    const op = { op: "filter_prune" };
    if (st.fltMinOpa > 0.005) op.min_opacity = st.fltMinOpa;
    if (st.fltMaxScale < 1.48) op.max_world_scale = st.fltMaxScale;
    if (st.fltMaxAniso < 29.5) op.max_anisotropy = st.fltMaxAniso;
    if (st.fltMinZ > -2.95) op.min_z = st.fltMinZ;
    mutateFlags((i) => matchesFilter(i), (f) => (f|1)&~6, op, "🪄 Pruned floaters");
    el("btn-flt-reset")?.click();
  });

  el("btn-save-splat")?.addEventListener("click", () => saveSplat(false));
  el("btn-save-sync-bl")?.addEventListener("click", () => saveSplat(true));
  el("btn-dl-splat")?.addEventListener("click", () => downloadSplatOrPly(false));
  el("btn-dl-ply")?.addEventListener("click", () => downloadSplatOrPly(true));
  el("btn-restore-orig")?.addEventListener("click", async () => {
    if (!st.activeJobId || st.activeJobId.startsWith("__")) return;
    try {
      const r = await apiFetch(`/api/splats/edit/restore/${encodeURIComponent(st.activeJobId)}`, { method: "POST", json: {} });
      const sp = await fetch(withTokenUrl(r.splat_url));
      if (sp.ok) ingestSplatBuf(await sp.arrayBuffer());
      showToast("↺ Restored original unedited splat");
    } catch (e) { showToast(e.message, true); }
  });

  const savePing = () => {
    if (!st.pendingPing) return;
    st.pings.push(Object.assign({}, st.pendingPing, { note: (el("inp-fps-ping-note")?.value || "").trim() || `Ping #${st.pings.length + 1}` }));
    st.pendingPing = null; el("fps-ping-modal")?.classList.add("hidden"); syncHotbar();
  };
  el("btn-fps-ping-save")?.addEventListener("click", savePing);
  el("btn-fps-ping-cancel")?.addEventListener("click", () => { st.pendingPing = null; el("fps-ping-modal")?.classList.add("hidden"); });
  el("inp-fps-ping-note")?.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); savePing(); } });

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
        el("fps-crosshair")?.classList.toggle("hit", !!st.lastHit);
        const ring = el("fps-brush-ring");
        if (ring) {
          if ((st.slot === 1 || st.slot === 3) && st.brushCursorPt) {
            const dist = Math.max(0.25, Math.hypot(st.brushCursorPt[0]-cam.eye[0], st.brushCursorPt[1]-cam.eye[1], st.brushCursorPt[2]-cam.eye[2]));
            const focalCss = ((cam.lensMm || 35) / 36.0) * Math.max(canvas.clientWidth, canvas.clientHeight);
            const diam = Math.max(14, Math.min(340, (2 * st.brushRadius * focalCss) / dist));
            ring.style.width = `${diam.toFixed(0)}px`; ring.style.height = `${diam.toFixed(0)}px`;
            ring.classList.toggle("cull", st.slot === 1 && st.mouseDownBtn !== 2);
            ring.classList.remove("hidden");
          } else ring.classList.add("hidden");
        }

        if (el("fps-reticle-tag")) {
          el("fps-reticle-tag").textContent = locked
            ? `${Math.round(cam.lensMm)}mm · ${st.lastHit ? `${st.lastHit.kind} ${st.lastHit.dist.toFixed(1)}m` : `reach ${st.brushReach.toFixed(1)}m`} · r=${st.brushRadius.toFixed(2)}m`
            : `Click viewport to lock mouse · Space×2 = Creative Fly · 1-8 Hotbar`;
        }
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