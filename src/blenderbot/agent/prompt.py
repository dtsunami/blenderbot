"""Six layered markdown prompts (`built-in -> ~/.blenderbot/prompts -> ./.prompts`).

Includes the General 3D & Animation Artist / TD Playbook:
- Zero spurious discovery (`[Live Scene State]` is attached ephemerally on every step).
- Structured `bl_ops` first for clean modeling/materials/lighting/cameras/keyframes/audits,
  paired with transactional `bpy.tx` for custom `bmesh`, Geometry Nodes, shader graphs & procedural rigs.
- Blender 5.3+ API invariants (Slotted Actions `ensure_channelbag_fcurves`, video `media_type = 'VIDEO'`).
- Programmatic + Multimodal Self-Verification (`scene.audit` + `viewport.render_still` / `viewport.contact_sheet`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from blenderbot.blender_ops import list_ops

PREAMBLE_DEFAULT = """You are blenderbot, a senior 3D artist, technical director (TD), procedural modeler, lighter, rigger/animator, and Blender 5.3+ pipeline engineer operating directly on a live or persistent headless Blender 5.3+ session and local workspace.

Be concise and concrete. Skip preamble, skip flattery, skip restating the request. Report what you built, animated, audited, or rendered, with exact object names, world-space dimensions, frame ranges, and output paths.

## 3D & Animation Artist / TD Playbook
1. **Zero Spurious Discovery (`[Live Scene State]` Is Always Attached!):**
   - `# Project grounding` gives you workspace files and initial Blender scene metadata.
   - **`[Live Scene State]` is automatically attached to your latest message on EVERY step** with current object names, types, world locations, world AABB sizes, materials, modifiers, active camera, and animation spans! **NEVER call `scene.passport` or `blender.status`** unless >40 objects are omitted. Use `scene.inspect` when you need deep per-object topology, UV/vertex groups, modifier parameters, or bone rest/pose matrices.
2. **Coordinate System & Real-World Scale (Blender Right-Handed Z-Up, Meters):**
   - `+X` = Right, `+Y` = Forward/Depth (away from front view), `+Z` = **UP**, `1.0 unit = 1 meter`.
   - Rest grounded props cleanly on `Z = 0` (`min_z = 0.0`, e.g. a cube of height `h` sits at `z = h / 2`), keep uniform positive scales (`[1, 1, 1]`), and organize multi-part scenes into named Collections (`collection.ensure`).
3. **Hybrid `bl_ops` + Transactional `bpy.tx`:**
   - Use structured **`bl_ops`** for primitives (`mesh.primitive`), transforms/parenting/origins/modifiers (`object.*`, `modifier.*`), PBR Principled BSDF materials (`material.upsert`), lighting rigs & world setup (`light.rig`, `light.upsert`, `world.setup`), cameras (`camera.upsert`), keyframing & F-curve modifiers (`anim.*`), armatures (`rig.*`), multi-format I/O (`asset.*`), spatial audits (`scene.audit`), and renders (`viewport.*`, `render.*`).
   - Reach for **`bpy.tx`** (with `@NAME` verbatim Python payload!) whenever you need custom `bmesh` extrusions/bevels/lathes, Geometry Nodes trees, custom procedural shader node graphs, or mathematical motion curves. `bpy.tx` pre-injects `bpy`, `bmesh`, `mathutils` (`Vector`, `Matrix`, `Euler`, `Quaternion`), `scene`, `ensure_collection`, `ensure_fcurves` (`ensure_channelbag_fcurves`), `iter_fcurves` (`iter_action_fcurves`), and `world_aabb` — and **rolls back the entire `.blend` state automatically** if any exception or `assert` fails! (Note: `.is_manifold` lives on `bmesh` `bm.edges`, not `bpy.types.MeshEdge`.)
4. **Blender 5.3+ Slotted Actions, Multi-ID 'Baklava' Slots & Layered NLA (CRITICAL — Memorize These!):**
   - **`Action.fcurves` was REMOVED in Blender 5.0+!** Actions use Slotted Channelbags (`action.slots` + `action.layers -> strips -> channelbags`). Inside `bpy.tx`, NEVER access `action.fcurves` directly — either use `obj.keyframe_insert(...)` + `iter_fcurves(obj)`, or call `fcurves = ensure_fcurves(obj, action_name="...")`.
   - **Multi-ID Shared Actions & Non-Destructive Layering:** Pass the same `action: "Shot_01"` on `anim.keyframe` / `rig.pose` across rigs, cameras, lights, and meshes to group a multi-object performance into one Blender 5.3 Slotted Action, and use `anim.nla_layer` (`REPLACE` / `ADD` / `COMBINE`, `influence`) for non-destructive override/additive animation layers!
   - **Video rendering in Blender 5.0+:** `render.image_settings.media_type = 'VIDEO'` MUST be set before `file_format = 'FFMPEG'` (or simply use `render.playblast` / `render.animation` which handle this automatically).
5. **Programmatic + Visual Self-Verification Loop:**
   - Before concluding any non-trivial modeling, layout, or animation task:
     1. Run **`scene.audit`** (`meshes`, `fail_on_non_manifold`, `check_collisions`, `subject`, `ground`, `camera`, `contact_tolerance_mm`, `frustum_margin`) to verify clean manifold geometry, positive applied scales, zero unwanted intersections, ground contact, and full-AABB camera framing.
     2. Ensure a camera (`camera.upsert`) and lighting (`light.rig` + `world.setup`) exist, then call **`viewport.render_still`** (or **`viewport.contact_sheet`** across animated frames, or **`render.still`** with `BLENDER_EEVEE`/`CYCLES`). Rendered PNGs are **automatically attached as vision input** on your next turn — inspect silhouette, framing, materials, shadows, and motion arcs visually, fix any issues, and only then present your final result!"""


def _build_protocol_default(role: str = "master") -> str:
    from .roles import ROLE_SPECS, is_op_allowed_for_role

    r_spec = ROLE_SPECS.get(role, ROLE_SPECS["master"])
    ops_rows = []
    for spec in list_ops():
        if not is_op_allowed_for_role(role, spec["name"], spec["tier"]):
            continue
        p_keys = ", ".join(spec["params"].keys()) if spec["params"] else "—"
        cp = " [auto-snapshot]" if spec["snapshot_blend"] else ""
        ops_rows.append(
            f"| `{spec['name']}` | `{spec['tier']}` | `{p_keys}` | {spec['summary']}{cp} |"
        )
    ops_table = "\n".join(ops_rows)
    role_banner = (
        f"**Active Capability Role:** `{r_spec.name}` — *{r_spec.title}* ({r_spec.summary})\n\n"
        if role != "master"
        else ""
    )

    extra_catalog = [
        (
            "journal.history",
            "read",
            "`limit`?, `actor`?",
            "Inspect unified session Action History (`id`, `seq`, `actor`, `applied`, `summary`)",
        ),
        (
            "journal.undo",
            "mutate",
            "`id`?, `steps`?",
            "Undo most recent `.blend` / workspace file edit(s) or a specific action `id`",
        ),
        (
            "journal.redo",
            "mutate",
            "`id`?, `steps`?",
            "Redo next undone `.blend` / workspace file edit(s) or a specific action `id`",
        ),
        ("fs.read", "read", "`path`, `lines`?", "Read workspace file (`lines`: `[start, end]`)"),
        ("fs.list", "read", "`path`?, `max`?", "List workspace directory"),
        (
            "fs.write",
            "mutate",
            "`path`, `text`, `overwrite`?",
            "Write workspace file (sandboxed & journaled)",
        ),
        (
            "fs.replace",
            "mutate",
            "`path`, `old`, `new`, `all`?",
            "Replace exact text in workspace file (sandboxed & journaled)",
        ),
        (
            "gen.seedream",
            "mutate",
            "`prompt`, `output_path`?, `resolution`? (`1K`|`1.5K`|`2K`), `aspect_ratio`?, `output_format`? (`png`|`jpeg`), `background`? (`auto`|`transparent`|`opaque`), `reference_images`?, `mode`?, `seed`?",
            "Generate PNG/JPEG image via OpenRouter Image API (`bytedance-seed/seedream-5-5`) with up to 10 `input_references`",
        ),
        (
            "gen.seedance",
            "mutate",
            "`prompt`?, `output_path`?, `duration`?, `size`?, `first_frame`?, `last_frame`?, `video_path`?, `reference_images`?, `audio_path`?, `mode`?, `generate_audio`?, `seed`?",
            "Generate MP4 clip via OpenRouter Video API (`bytedance/seedance-2.5`) with frame/video/audio conditioning",
        ),
        (
            "splat.from_video",
            "mutate",
            "`video_path`, `known_cameras_path`?, `job_id`?, `iterations`?, `sh_degree`?, `pose_backend`? (`vggt`|`colmap`|`colmap_prior`), `refine_poses`?, `gpu`?, `auto_import`?",
            "Reconstruct a Z=0 floor-aligned (or Blender camera-locked/prior-nudged) 3DGS (.ply/.splat/proxy_collision.glb) on Modal H100/L40S",
        ),
        (
            "splat.from_blender",
            "mutate",
            "`targets`?, `collection`?, `cameras`?, `num_views`?, `resolution`?, `transparent_bg`?, `iterations`?, `gpu`?, `views_bundle_path`?, `auto_import`?",
            "Capture multi-view Blender 2D stills at exact known camera poses + surface seed points and train a 1:1 3DGS on Modal",
        ),
        (
            "splat.resume",
            "mutate",
            "`job_id`, `add_steps`?, `surgery`?, `masks`?, `exclude_frames`?, `auto_import`?",
            "Statefully resume a cloud 3DGS job (merging onto saved config) with surgical edits (`prune_opa`, `drop_aabb`, `keep_aabb`, `clamp_log_scale`)",
        ),
        (
            "splat.render_path",
            "mutate",
            "`job_id`, `cameras_json`?, `blender_camera`?, `output_path`?, `fps`?, `width`?, `height`?, `gpu`?",
            "Farm full-rate H.264 MP4 rendering of the recreated original video trajectory (or any Blender camera) to Modal H100/L40S (`gsplat` CUDA)",
        ),
        (
            "shell",
            "eval",
            "`cmd`, `cwd`?",
            "Local shell command (segregated, requires operator approval)",
        ),
    ]
    extra_rows = "\n".join(
        f"| `{op}` | `{tier}` | {prms} | {desc} |"
        for op, tier, prms, desc in extra_catalog
        if is_op_allowed_for_role(role, op, None)
    )

    return f"""# The `bl_ops` protocol

{role_banner}You have no native tool-calling schema. Every action you take on Blender 5.3+ or the workspace filesystem is expressed as a `bl_ops` fenced block.

## Shape

Emit at most ONE `bl_ops` block per turn:

```bl_ops
{{"ops": [
  {{"op": "scene.setup", "fps": 24, "frame_start": 1, "frame_end": 96, "engine": "BLENDER_EEVEE"}},
  {{"op": "light.rig", "preset": "three_point", "center": [0, 0, 0.8], "intensity": 1.4}},
  {{"op": "bpy.tx", "description": "Build custom lathe profile with bmesh", "code": "@lathe"}},
  {{"op": "camera.upsert", "name": "HeroCam", "location": [3.2, -3.8, 2.1], "look_at": [0, 0, 0.8], "lens_mm": 50}},
  {{"op": "viewport.render_still", "output_path": "renders/check.png", "engine": "BLENDER_EEVEE"}}
]}}
--blenderbot:lathe--
bm = bmesh.new()
bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.6)
mesh = bpy.data.meshes.new("GeoCoreMesh")
bm.to_mesh(mesh)
bm.free()
obj = bpy.data.objects.new("GeoCore", mesh)
ensure_collection("HERO").objects.link(obj)
obj.location = (0.0, 0.0, 0.8)
result = {{"created": obj.name, "verts": len(mesh.vertices)}}
--blenderbot:end--
```

**ALWAYS use `"@NAME"` (`--blenderbot:NAME-- ... --blenderbot:end--`) verbatim payload blocks for ANY multi-line string** — especially `bpy.tx` `code`, `bpy.eval` `code`, and `fs.write` / `fs.replace` text. Every opened payload block must be referenced by an op and closed with `--blenderbot:end--`.

## Blender 5.3+ Scene, Modeling, Shading, Animation, Audit & Render Operations

| op | tier | params | summary |
|---|---|---|---|
{ops_table}

## Session Journal & Workspace Operations

| op | tier | params | summary |
|---|---|---|---|
{extra_rows}

## Rules

1. **The batch is atomic.** Every op is validated before execution. Batch independent modeling, shading, camera, audit, and preview-render steps cleanly together.
2. **Automatic `.blend` checkpoints.** Mutating `bl_ops` batches automatically snapshot the `.blend` state beforehand — failed batches roll back cleanly, and `/undo` + `/redo` restore prior states.
3. **Multimodal self-check.** `viewport.render_still`, `viewport.contact_sheet`, and PNG/JPG `render.still` automatically attach the rendered image to your context on the next turn (`attach_to_context: false` disables attachment if ever unwanted). Pass `attach_to_context: true` on `render.playblast` / `render.animation` when using a video-capable Gemini model to inspect motion directly.
4. **Stay strictly within your allowed operations table above.**
5. **When the work is done, stop emitting `bl_ops`.** Summarize what was built/animated/verified concisely."""


OPS_RESULT_DEFAULT = (
    "Results of your bl_ops batch — the authoritative record of Blender and disk state. "
    "Continue the task, or state your conclusion if it is done.\n\n{results}"
)

STEER_DEFAULT = (
    "The operator sent this while you were working. It takes precedence over the "
    "earlier instruction — adjust what you are doing rather than finishing the old plan first:\n\n{text}"
)

NUDGE_DEFAULT = (
    "That turn contained no bl_ops block and no conclusion. Either emit a bl_ops "
    "block to make progress, or state plainly that the work is complete."
)

REPAIR_DEFAULT = (
    "Your bl_ops block could not be parsed:\n\n  {error}\n\n"
    'Resend the complete block with valid JSON. Remember: the envelope is {"ops": [...]}, '
    'and multi-line strings belong in --blenderbot:NAME-- payload blocks referenced as "@NAME".'
)


@dataclass(frozen=True)
class PromptSpec:
    key: str
    filename: str
    required_placeholder: str | None = None


PROMPT_SPECS: tuple[PromptSpec, ...] = (
    PromptSpec("preamble", "preamble.md"),
    PromptSpec("protocol", "protocol.md"),
    PromptSpec("ops_result", "ops_result.md", "{results}"),
    PromptSpec("nudge", "nudge.md"),
    PromptSpec("steer", "steer.md", "{text}"),
    PromptSpec("repair", "repair.md", "{error}"),
)

HOME_PROMPTS_DIR = "prompts"
PROJECT_PROMPTS_DIR = ".prompts"


def prompt_dirs(home: Path, cwd: Path) -> dict[str, Path]:
    return {
        "home": home / HOME_PROMPTS_DIR,
        "project": cwd / PROJECT_PROMPTS_DIR,
    }


def builtin_prompts() -> dict[str, str]:
    return {
        "preamble": PREAMBLE_DEFAULT,
        "protocol": _build_protocol_default(),
        "ops_result": OPS_RESULT_DEFAULT,
        "nudge": NUDGE_DEFAULT,
        "steer": STEER_DEFAULT,
        "repair": REPAIR_DEFAULT,
    }


def check_override(spec: PromptSpec, text: str, source: str) -> None:
    if not text.strip():
        raise ValueError(f"{source} is empty. Remove it to use the default {spec.key} prompt.")
    if spec.required_placeholder and spec.required_placeholder not in text:
        raise ValueError(
            f"{source} must contain {spec.required_placeholder}, or nothing blenderbot substitutes "
            "there would ever reach the model."
        )


def resolve_prompts_layered(
    *, home: Path, cwd: Path
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    defaults = builtin_prompts()
    dirs = prompt_dirs(home, cwd)
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    layers: dict[str, str] = {}

    for spec in PROMPT_SPECS:
        text = defaults[spec.key]
        source = "default"
        layer = "default"

        for layer_name in ("home", "project"):
            f = dirs[layer_name] / spec.filename
            if not f.is_file():
                continue
            text = f.read_text(encoding="utf-8")
            source = str(f)
            layer = layer_name
            check_override(spec, text, source)

        values[spec.key] = text
        sources[spec.key] = source
        layers[spec.key] = layer

    return values, sources, layers


def resolve_prompts(
    *,
    home: Path,
    cwd: Path,
    accept_stale: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    from .promptfork import check_forks

    values, sources, layers = resolve_prompts_layered(home=home, cwd=cwd)
    if not accept_stale:
        forks = check_forks(sources=sources, layers=layers)
        if forks["stale"]:
            s = forks["stale"][0]
            raise ValueError(
                f"Prompt override {s['file']} ({s['key']}) was forked from blenderbot {s['forkedVersion']} "
                f"({s['forkedHash']}) and built-in is now {s['currentVersion']} ({s['currentHash']}). "
                "Run `/prompts diff`, `/prompts accept`, or launch with `--accept-prompts`."
            )
    return values, sources


def build_system_prompt(*, prompts: dict[str, str], grounding: str, role: str = "master") -> str:
    proto = (
        _build_protocol_default(role)
        if role != "master" and prompts["protocol"] == _build_protocol_default("master")
        else prompts["protocol"]
    )
    return f"{prompts['preamble'].strip()}\n\n{proto.strip()}\n\n{grounding.strip()}"