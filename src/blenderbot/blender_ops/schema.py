"""Operation registry, runtime context wrapper, and safety tier metadata (stdlib only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

SafetyTier = Literal["read", "mutate", "destructive", "eval"]


class OpError(Exception):
    """Structured error raised by any blender_op when preconditions or Blender calls fail."""

    def __init__(self, message: str, *, code: str = "OP_ERROR") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class OpContext:
    """Runtime handles passed to every blender_op handler inside Blender."""

    bpy: Any
    mode: str = "headless"  # "gui" | "headless"
    snaps_dir: str | None = None
    progress_fn: Callable[[dict[str, Any]], None] | None = None
    suspend_timeout_fn: Callable[[bool], None] | None = None

    def report_progress(self, data: dict[str, Any]) -> None:
        if self.progress_fn is not None:
            try:
                self.progress_fn(data)
            except Exception:  # noqa: BLE001, S110
                pass

    def set_suspend_timeout(self, enabled: bool = True) -> None:
        if self.suspend_timeout_fn is not None:
            try:
                self.suspend_timeout_fn(bool(enabled))
            except Exception:  # noqa: BLE001, S110
                pass

    @property
    def scene(self) -> Any:
        sc = self.bpy.context.scene
        if sc is None:
            raise OpError("No active Blender scene in bpy.context", code="NO_SCENE")
        return sc

    @property
    def view_layer(self) -> Any:
        vl = self.bpy.context.view_layer
        if vl is None:
            raise OpError("No active ViewLayer in bpy.context", code="NO_VIEW_LAYER")
        return vl


OpHandler = Callable[[OpContext, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class OpSpec:
    name: str
    tier: SafetyTier
    summary: str
    handler: OpHandler
    params_doc: dict[str, str] = field(default_factory=dict)
    snapshot_blend: bool = False


OPS_REGISTRY: dict[str, OpSpec] = {}


def blender_op(
    name: str,
    *,
    tier: SafetyTier = "read",
    summary: str = "",
    params_doc: dict[str, str] | None = None,
    snapshot_blend: bool = False,
) -> Callable[[OpHandler], OpHandler]:
    """Decorator registering a serializable Blender operation (`dict` in -> `dict` out)."""

    def decorator(fn: OpHandler) -> OpHandler:
        OPS_REGISTRY[name] = OpSpec(
            name=name,
            tier=tier,
            summary=summary or (fn.__doc__ or "").strip().splitlines()[0],
            handler=fn,
            params_doc=dict(params_doc or {}),
            snapshot_blend=snapshot_blend,
        )
        return fn

    return decorator


def get_op(name: str) -> OpSpec | None:
    return OPS_REGISTRY.get(name)


def list_ops() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in sorted(OPS_REGISTRY.keys()):
        spec = OPS_REGISTRY[name]
        out.append(
            {
                "name": spec.name,
                "tier": spec.tier,
                "summary": spec.summary,
                "params": dict(spec.params_doc),
                "snapshot_blend": spec.snapshot_blend,
            }
        )
    return out


def dispatch_op(ctx: OpContext, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = OPS_REGISTRY.get(name)
    if spec is None:
        raise OpError(f'Unknown blender_op "{name}"', code="UNKNOWN_OP")
    return spec.handler(ctx, dict(params or {}))