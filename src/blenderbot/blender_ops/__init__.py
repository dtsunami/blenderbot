"""Stdlib-only Blender 5.3+ domain operations (`blenderbot.blender_ops`).

Runs inside both live GUI Blender (`BridgeTransport`) and persistent background `blender.exe -b`
workers (`HeadlessTransport`) with zero third-party dependencies.
"""

from __future__ import annotations

# Register all op modules on import
from . import (
    anim_ops,
    asset_ops,
    audit_ops,
    modeling_ops,
    render_ops,
    rig_ops,
    scene_ops,
    shading_ops,
    splat_ops,
    tx_op,
)
from .schema import (
    OPS_REGISTRY,
    OpContext,
    OpError,
    OpSpec,
    SafetyTier,
    blender_op,
    dispatch_op,
    get_op,
    list_ops,
)

__all__ = [
    "OPS_REGISTRY",
    "OpContext",
    "OpError",
    "OpSpec",
    "SafetyTier",
    "anim_ops",
    "asset_ops",
    "audit_ops",
    "blender_op",
    "dispatch_op",
    "get_op",
    "list_ops",
    "modeling_ops",
    "render_ops",
    "rig_ops",
    "scene_ops",
    "shading_ops",
    "splat_ops",
    "tx_op",
]