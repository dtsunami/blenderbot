"""3-Tier Role-Based Capability Surfaces (`blenderbot.agent.roles`).

Lets the operator scope what `blenderbot` can execute — from strictly read-only scene review
(`viewer`), to safe autonomous 3D/animation creation (`builder`), up to full `master` control:

  1. `viewer`  — Read-only Blender inspection & visual review (`scene.passport`, `scene.inspect`,
                 `scene.audit`, `rig.armature_inspect`, `material.list`, `viewport.render_still`,
                 `viewport.contact_sheet`, `fs.read`, `fs.list`, `journal.history`).
  2. `builder` — Safe autonomous 3D modeling, modifiers, materials, lighting, cameras, keyframing,
                 F-curves, armatures, asset import/export, renders/playblasts, transactional `bpy.tx`,
                 `scene.save`, `fs.write`/`fs.replace`, and `journal.undo`/`journal.redo`.
                 Blocks `destructive` ops (`scene.clear`, `scene.open`), `bpy.eval`, and `shell`.
  3. `master`  — Full capability surface including approval-gated `destructive` ops, `bpy.eval`,
                 and `shell`.
"""

from __future__ import annotations

from dataclasses import dataclass

ROLES: tuple[str, ...] = ("viewer", "builder", "master")


@dataclass(frozen=True)
class RoleSpec:
    name: str
    title: str
    summary: str


ROLE_SPECS: dict[str, RoleSpec] = {
    "viewer": RoleSpec(
        "viewer",
        "Read-Only Scene & Visual Reviewer",
        "Inspects objects, topology/collision audits, armatures, viewport stills & contact sheets. Zero mutations.",
    ),
    "builder": RoleSpec(
        "builder",
        "Autonomous 3D Modeler, Lighter & Animator",
        "Full non-destructive modeling, shading, lighting, rigging/animation, exports, renders, and transactional `bpy.tx`.",
    ),
    "master": RoleSpec(
        "master",
        "All-Stage Master 3D TD",
        "Full capability surface including approval-gated `scene.clear`, `scene.open`, `bpy.eval`, and `shell`.",
    ),
}

READ_ONLY_EXTRA_OPS = frozenset(
    {
        "fs.read",
        "fs.list",
        "journal.history",
    }
)

BUILDER_EXTRA_OPS = READ_ONLY_EXTRA_OPS | frozenset(
    {
        "fs.write",
        "fs.replace",
        "journal.undo",
        "journal.redo",
        "render.modal",
        "gen.meshy",
        "gen.seedream",
        "gen.seedance",
        "splat.from_video",
        "splat.from_images",
        "splat.from_blender",
        "splat.resume",
        "splat.render_path",
    }
)


def is_op_allowed_for_role(role: str, op_name: str, blender_tier: str | None = None) -> bool:
    """Return True if `op_name` (with optional `blender_ops` tier) is permitted under `role`."""
    r = role.strip().lower() or "master"
    if r == "master":
        return True

    if blender_tier == "read" or op_name in READ_ONLY_EXTRA_OPS:
        return True

    if r == "viewer":
        return False

    if r == "builder":
        if blender_tier == "mutate":
            return True
        return op_name in BUILDER_EXTRA_OPS

    return False