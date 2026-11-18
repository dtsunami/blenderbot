"""Agent harness: `bl_ops` parser, `.blend` + file journal, sandbox, roles, grounding, and streaming loop."""

from .executor import (
    ApprovalDecision,
    BatchOutcome,
    execute_batch,
    format_batch_results,
    plan_batch,
)
from .grounding import (
    build_grounding_text,
    build_static_system_prompt,
    format_live_scene_state,
    poll_live_blender_banner,
    snapshot_blender_grounding,
)
from .journal import ActionRecord, BlendCheckpoint, FileEntry, Journal
from .loop import EVENTS, Agent, EventEmitter, detect_repetition
from .parser import (
    ExtractedTurn,
    FencedBlock,
    OpCall,
    ParseError,
    extract_turn,
    parse_ops_body,
    scan_blocks,
)
from .prompt import (
    PROMPT_SPECS,
    build_system_prompt,
    builtin_prompts,
    resolve_prompts,
    resolve_prompts_layered,
)
from .roles import ROLE_SPECS, ROLES, RoleSpec, is_op_allowed_for_role
from .sandbox import Sandbox, SandboxError

__all__ = [
    "EVENTS",
    "PROMPT_SPECS",
    "ROLES",
    "ROLE_SPECS",
    "ActionRecord",
    "Agent",
    "ApprovalDecision",
    "BatchOutcome",
    "BlendCheckpoint",
    "EventEmitter",
    "ExtractedTurn",
    "FencedBlock",
    "FileEntry",
    "Journal",
    "OpCall",
    "ParseError",
    "RoleSpec",
    "Sandbox",
    "SandboxError",
    "build_grounding_text",
    "build_static_system_prompt",
    "build_system_prompt",
    "builtin_prompts",
    "detect_repetition",
    "execute_batch",
    "extract_turn",
    "format_batch_results",
    "format_live_scene_state",
    "is_op_allowed_for_role",
    "parse_ops_body",
    "plan_batch",
    "poll_live_blender_banner",
    "resolve_prompts",
    "resolve_prompts_layered",
    "scan_blocks",
    "snapshot_blender_grounding",
]