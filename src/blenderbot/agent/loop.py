"""Hand-rolled streaming agent loop (`blenderbot.agent.loop`).

- Explicit `EVENTS` tuple enforced across TUI, Headless, and Web SSE consumers.
- Streams `thinking` and `text` channels, detects degenerate repeating loops mid-stream,
  demotes `effort` on `max_tokens` reasoning blowouts, salvages unexpected native tool calls,
  executes `bl_ops` batches, attaches rendered stills/contact-sheets/playblasts into `Ledger`
  for multimodal visual self-verification, and drains operator `steers` at step boundaries.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from blenderbot.config.env import Config
from blenderbot.context.artifacts import ArtifactStore
from blenderbot.context.compact import compact_ledger
from blenderbot.context.ledger import Ledger
from blenderbot.context.media import load_image_part, load_video_part
from blenderbot.providers.base import ProviderAdapter, ProviderError, StreamEvent, cost_of_usage
from blenderbot.transport.base import BlenderTransport

from .executor import ApprovalDecision, execute_batch, format_batch_results
from .grounding import format_live_scene_state, poll_live_blender_banner
from .journal import Journal
from .parser import OpCall, ParseError, extract_turn
from .sandbox import Sandbox

MAX_STEPS = 50
MAX_REPAIRS = 2
MAX_NUDGES = 2
MAX_EMPTY = 2
MAX_PROVIDER_RETRIES = 2

LOWER_EFFORT: dict[str, str | None] = {
    "max": "high",
    "xhigh": "high",
    "high": "medium",
    "medium": "low",
    "low": None,
}

REPEAT_WINDOW = 2048
REPEAT_PROBE = 64
REPEAT_MAX_PERIOD = 512
REPEAT_MIN_REPEATS = 4
REPEAT_CHECK_EVERY = 1024

INTENT_RE = re.compile(
    r"\b(?:i'?ll|i will|i'?m going to|i am going to|let me|let'?s|first,? i|next,? i|now i'?ll)\b",
    re.IGNORECASE,
)
CONDITIONAL_OFFER_RE = re.compile(
    r"(?:^|[,;:\-–—]\s*)\b(?:if|once|when|whenever|should|drop|send|share|pass|want)\b",
    re.IGNORECASE,
)

EVENTS: tuple[str, ...] = (
    "turn-start",
    "thinking",
    "text",
    "turn-end",
    "usage",
    "ops-parsed",
    "op-progress",
    "ops-result",
    "artifact",
    "approval-request",
    "compact-start",
    "compact-done",
    "nudge",
    "retry",
    "steer-queued",
    "steered",
    "prompts-applied",
    "repair",
    "warning",
    "cancelled",
    "error",
    "done",
)


def detect_repetition(text: str) -> dict[str, Any] | None:
    """Detect exact periodic repetition collapse at the tail of streamed text."""
    if len(text) < REPEAT_WINDOW:
        return None
    tail = text[-REPEAT_WINDOW:]
    probe = tail[-REPEAT_PROBE:]
    prev = tail.rfind(probe, 0, len(tail) - REPEAT_PROBE)
    if prev < 0:
        return None
    period = (len(tail) - REPEAT_PROBE) - prev
    if period <= 0 or period > REPEAT_MAX_PERIOD:
        return None
    unit = tail[-period:]
    repeats = len(tail) // period
    if repeats < REPEAT_MIN_REPEATS:
        return None
    check_len = repeats * period
    if tail[-check_len:] == unit * repeats:
        return {"period": period, "repeats": repeats, "unit": unit[:80]}
    return None


def _salvage_native_tool_calls(calls: list[dict[str, Any]]) -> list[OpCall]:
    salvaged: list[OpCall] = []
    for idx, call in enumerate(calls):
        name = str(call.get("name", "")).strip()
        raw_args = call.get("args", "{}")
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except ValueError:
            continue
        if name in ("bl_ops", "da_ops", "dr_ops") and isinstance(parsed.get("ops"), list):
            for sub_i, op_dict in enumerate(parsed["ops"]):
                if isinstance(op_dict, dict) and op_dict.get("op"):
                    params = {k: v for k, v in op_dict.items() if k != "op"}
                    salvaged.append(OpCall(index=sub_i, op=str(op_dict["op"]), params=params))
        elif name:
            salvaged.append(
                OpCall(index=idx, op=name, params=parsed if isinstance(parsed, dict) else {})
            )
    return salvaged


class EventEmitter:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[[Any], None]]] = {}
        self.event_counts: dict[str, int] = {}
        self.unretired_events: dict[str, int] = {}

    def on(self, event: str, fn: Callable[[Any], None]) -> None:
        self._listeners.setdefault(event, []).append(fn)

    def off(self, event: str, fn: Callable[[Any], None]) -> None:
        lst = self._listeners.get(event)
        if lst and fn in lst:
            lst.remove(fn)

    def emit(self, event: str, payload: Any = None) -> None:
        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        handlers = list(self._listeners.get(event, []))
        active_consumers = [h for h in handlers if not getattr(h, "passive", False)]
        if not active_consumers:
            self.unretired_events[event] = self.unretired_events.get(event, 0) + 1
        for h in handlers:
            h(payload)


class Agent(EventEmitter):
    def __init__(
        self,
        *,
        cfg: Config,
        make_provider: Callable[..., ProviderAdapter],
        ledger: Ledger,
        sandbox: Sandbox,
        journal: Journal,
        artifacts: ArtifactStore,
        prompts: dict[str, str],
        transport: BlenderTransport | None = None,
        session_log: Any = None,
        perf_tracker: Any = None,
        drain_prompts_hook: Callable[[], dict[str, Any] | None] | None = None,
        initial_stats: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.make_provider = make_provider
        self.ledger = ledger
        self.sandbox = sandbox
        self.journal = journal
        self.artifacts = artifacts
        self.prompts = prompts
        self.transport = transport
        self.session_log = session_log
        self.perf = perf_tracker
        self._drain_prompts_hook = drain_prompts_hook

        self.busy = False
        self._cancel_event: asyncio.Event | None = None
        self._active_user_prompt: str = ""
        self.steers: list[str] = []
        self._last_blender_banner: str | None = None

        seed = initial_stats or {}
        self.run_cost: float = 0.0
        self.stats: dict[str, Any] = {
            "turns": int(seed.get("turns", 0) or 0),
            "ops": int(seed.get("ops", 0) or 0),
            "cost": float(seed.get("cost", 0.0) or 0.0),
            "in_tokens": int(seed.get("in_tokens", 0) or 0),
            "out_tokens": int(seed.get("out_tokens", 0) or 0),
            "cache_read": int(seed.get("cache_read", 0) or 0),
            "malformed": int(seed.get("malformed", 0) or 0),
            "nudges": int(seed.get("nudges", 0) or 0),
            "op_failures": int(seed.get("op_failures", 0) or 0),
            "thought_leaks": int(seed.get("thought_leaks", 0) or 0),
            "compactions": int(seed.get("compactions", 0) or 0),
        }
        # Rolling model-stability & long/multimodal context friction since last compaction
        self._rolling_health: dict[str, int] = {
            "consecutive_op_failures": 0,
            "malformed": 0,
            "thought_leaks": 0,
            "salvaged": 0,
            "op_failures": 0,
            "nudges": 0,
            "empty": 0,
        }

    def _has_compressible_context(self) -> bool:
        compressible = self.ledger.segments[:-4] if len(self.ledger.segments) > 4 else []
        return any(
            not s.pinned
            and s.type in ("media", "op-result", "shell", "paste")
            and s.tokens > 250
            for s in compressible
        )

    def _reset_rolling_health(self) -> None:
        for k in self._rolling_health:
            self._rolling_health[k] = 0

    def context_health(self) -> dict[str, Any]:
        """Compute model protocol/op stability & multimodal context-pressure indicator."""
        r = self._rolling_health
        media_segs = [s for s in self.ledger.segments if s.type == "media"]
        video_segs = [s for s in media_segs if s.part.get("type") == "video"]
        media_tokens = sum(s.tokens for s in media_segs)
        ctx_pct = round(self.ledger.usage_fraction * 100.0, 1)

        score = (
            2.0 * r["malformed"]
            + 2.0 * r["empty"]
            + 1.5 * r["thought_leaks"]
            + 1.5 * r["op_failures"]
            + (1.0 if r["consecutive_op_failures"] >= 2 else 0.0)
            + 1.0 * r["nudges"]
            + 1.0 * r["salvaged"]
        )
        if score > 0 and (video_segs or media_tokens >= 12_000 or ctx_pct >= 50.0):
            score += 1.5
        score = round(score, 2)

        if score >= 5.0:
            status = "critical"
        elif score >= 3.0:
            status = "degraded"
        elif score > 0.0:
            status = "watch"
        else:
            status = "healthy"

        if score >= 3.0 and self._has_compressible_context():
            recommended = "compact"
        elif score >= 2.5 or r["consecutive_op_failures"] >= 2 or r["malformed"] >= 1:
            recommended = "nudge"
        else:
            recommended = "none"

        return {
            "status": status,
            "score": score,
            "recommended_action": recommended,
            "context_tokens": self.ledger.total_tokens,
            "context_pct": ctx_pct,
            "media_segments": len(media_segs),
            "video_segments": len(video_segs),
            "media_tokens": media_tokens,
            "rolling": dict(r),
            "totals": {
                "malformed": self.stats["malformed"],
                "nudges": self.stats["nudges"],
                "op_failures": self.stats["op_failures"],
                "thought_leaks": self.stats["thought_leaks"],
                "compactions": self.stats["compactions"],
            },
        }

    def _record_health_signal(self, signal: str, detail: str = "") -> dict[str, Any]:
        snap = self.context_health()
        payload = {
            "turn": self.stats["turns"],
            "provider": self.cfg.provider,
            "model": self.cfg.model.id,
            "signal": signal,
            "detail": detail[:240],
            **snap,
        }
        if self.session_log is not None:
            self.session_log.record("health", payload)
            flight = getattr(self.session_log, "flight", None)
            if flight is not None:
                flight.note(
                    "model-health",
                    signal=signal,
                    status=snap["status"],
                    score=snap["score"],
                    recommended_action=snap["recommended_action"],
                    detail=detail[:160],
                )
        return snap

    def cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()

    def steer(self, text: str) -> bool:
        cleaned = text.strip()
        if not self.busy or not cleaned:
            return False
        if cleaned == self._active_user_prompt or (self.steers and self.steers[-1] == cleaned):
            return False
        self.steers.append(cleaned)
        self.emit("steer-queued", {"text": cleaned, "queued": len(self.steers)})
        return True

    def _drain_steers(self) -> bool:
        if not self.steers:
            return False
        combined = "\n".join(self.steers)
        self.steers.clear()
        wrapped = self.prompts["steer"].replace("{text}", combined)
        self.ledger.add(seg_type="user", label="steer", role="user", text=wrapped)
        if self.session_log is not None:
            self.session_log.record("steer", {"text": combined})
        self.emit("steered", {"text": combined})
        return True

    async def _request_approval(self, call: OpCall) -> ApprovalDecision:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()

        def _respond(decision: dict[str, Any]) -> None:
            if not fut.done():
                fut.set_result(decision)

        req_obj = {
            "op": {"op": call.op, **call.params},
            "index": call.index,
            "respond": _respond,
        }
        self.emit("approval-request", req_obj)
        raw = await fut
        return ApprovalDecision(
            allow=bool(raw.get("allow", False)),
            reason=raw.get("reason"),
            checkpoint_first=bool(raw.get("checkpoint_first", False)),
        )

    async def _stream_single_turn(
        self, cancel_ev: asyncio.Event, *, effort_override: str | None = None
    ) -> dict[str, Any]:
        provider = self.make_provider(self.cfg, effort_override=effort_override)
        for seg in list(self.ledger.segments):
            if seg.part.get("type") == "image" and not seg.part.get("data"):
                self.ledger.drop(seg.id)
        messages = self.ledger.to_messages()

        # Ephemerally attach compact `[Live Scene State]` ONLY to `messages[-1]`!
        live_scene = await format_live_scene_state(self.transport)
        if live_scene and messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": [*messages[-1]["content"], {"type": "text", "text": live_scene}],
            }

        text_chunks: list[str] = []
        think_chunks: list[str] = []
        usage_ev: StreamEvent | None = None
        stop_reason = "end_turn"
        native_calls: list[dict[str, Any]] = []

        t0 = time.perf_counter()
        ttft: float | None = None
        next_rep_check = REPEAT_WINDOW + REPEAT_CHECK_EVERY
        degenerate: dict[str, Any] | None = None

        flight = getattr(self.session_log, "flight", None)
        if flight is not None:
            flight.mark(
                "llm:await-first-byte",
                turn=self.stats["turns"] + 1,
                provider=self.cfg.provider,
                model=self.cfg.model.id,
                context_tokens=self.ledger.total_tokens,
            )

        self.emit(
            "turn-start",
            {
                "turn": self.stats["turns"] + 1,
                "provider": self.cfg.provider,
                "model": self.cfg.model.id,
                "effort": effort_override or self.cfg.effort,
            },
        )
        if self.perf is not None:
            self.perf.on_turn_start(
                model=self.cfg.model.id, effort=effort_override or self.cfg.effort
            )

        async for ev in provider.send(
            system=self.ledger.system,
            messages=messages,
            cancel_event=cancel_ev,
        ):
            if ev.t == "thinking":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    if self.perf is not None:
                        self.perf.on_first_token(ttft)
                    if flight is not None:
                        flight.mark(
                            "llm:streaming",
                            turn=self.stats["turns"] + 1,
                            channel="thinking",
                            ttft_ms=round(ttft * 1000),
                        )
                elif flight is not None:
                    flight.touch()
                think_chunks.append(ev.delta)
                if self.perf is not None:
                    self.perf.on_chars(len(ev.delta))
                self.emit("thinking", ev.delta)
            elif ev.t == "text":
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    if self.perf is not None:
                        self.perf.on_first_token(ttft)
                    if flight is not None:
                        flight.mark(
                            "llm:streaming",
                            turn=self.stats["turns"] + 1,
                            channel="text",
                            ttft_ms=round(ttft * 1000),
                        )
                elif flight is not None:
                    flight.touch()
                text_chunks.append(ev.delta)
                if self.perf is not None:
                    self.perf.on_chars(len(ev.delta))
                self.emit("text", ev.delta)

                joined_len = sum(len(c) for c in text_chunks)
                if joined_len >= next_rep_check:
                    next_rep_check = joined_len + REPEAT_CHECK_EVERY
                    rep = detect_repetition("".join(text_chunks))
                    if rep is not None:
                        degenerate = rep
                        cancel_ev.set()
                        break
            elif ev.t == "tool_calls":
                native_calls.extend(ev.calls)
            elif ev.t == "diagnostic":
                if ev.reason == "thought_leak":
                    self.stats["thought_leaks"] += 1
                    self._rolling_health["thought_leaks"] += 1
                    self._record_health_signal("thought_leak", ev.detail)
            elif ev.t == "usage":
                usage_ev = ev
            elif ev.t == "stop":
                stop_reason = ev.reason

        elapsed = time.perf_counter() - t0
        full_text = "".join(text_chunks)
        full_thinking = "".join(think_chunks)

        turn_cost = 0.0
        if usage_ev is not None:
            turn_cost = cost_of_usage(self.cfg.model, usage_ev)
            self.run_cost += turn_cost
            self.stats["cost"] += turn_cost
            self.stats["in_tokens"] += usage_ev.in_fresh
            self.stats["out_tokens"] += usage_ev.out
            self.stats["cache_read"] += usage_ev.cache_read
            self.ledger.observe_provider_usage(usage_ev.context_in)
            u_payload = {
                "in": usage_ev.in_fresh,
                "out": usage_ev.out,
                "cache_read": usage_ev.cache_read,
                "cache_write": usage_ev.cache_write,
                "context_in": usage_ev.context_in,
                "cost": turn_cost,
                "total_cost": self.stats["cost"],
            }
            self.emit("usage", u_payload)
            if self.session_log is not None:
                self.session_log.record("usage", u_payload)

        self.stats["turns"] += 1
        if self.perf is not None:
            perf_rec = self.perf.on_turn_end(
                elapsed=elapsed,
                out_tokens=usage_ev.out if usage_ev else 0,
                context_in=usage_ev.context_in if usage_ev else self.ledger.total_tokens,
                effort=effort_override or self.cfg.effort,
            )
            if self.session_log is not None:
                self.session_log.record("perf", asdict(perf_rec))

        self.emit(
            "turn-end",
            {
                "turn": self.stats["turns"],
                "elapsed": round(elapsed, 3),
                "ttft": round(ttft, 3) if ttft is not None else None,
                "out_tokens": usage_ev.out if usage_ev else 0,
                "cost": turn_cost,
                "stop_reason": stop_reason,
                "effort": effort_override or self.cfg.effort,
            },
        )

        return {
            "text": full_text,
            "thinking": full_thinking,
            "stop_reason": stop_reason,
            "native_calls": native_calls,
            "degenerate": degenerate,
        }

    async def run(self, user_prompt: str) -> dict[str, Any]:
        """Run the full multi-step agent loop for `user_prompt`."""
        if self.busy:
            raise RuntimeError("Agent is already running a turn")

        self.busy = True
        self._active_user_prompt = user_prompt.strip()
        cancel_ev = asyncio.Event()
        self._cancel_event = cancel_ev

        try:
            banner = await poll_live_blender_banner(self.transport)
            if banner and banner != self._last_blender_banner:
                self._last_blender_banner = banner
                prompt_with_state = f"{banner}\n\n{user_prompt}"
            else:
                prompt_with_state = user_prompt

            self.ledger.add(
                seg_type="user",
                label=user_prompt[:200].replace("\n", " "),
                role="user",
                text=prompt_with_state,
            )
            if self.session_log is not None:
                self.session_log.record("user", {"text": user_prompt, "blender_banner": banner})

            repairs = 0
            nudges = 0
            empty_turns = 0
            cur_effort: str | None = None
            pending_failed_turns: list[dict[str, Any]] = []

            def _compress_superseded() -> None:
                if not pending_failed_turns:
                    return
                for item in pending_failed_turns:
                    prose_head = item["prose"].splitlines()[0][:120] if item["prose"] else ""
                    a_stub = (
                        f"{prose_head}\n[superseded bl_ops attempt omitted: {item['summary']}]"
                        if prose_head
                        else f"[superseded bl_ops attempt omitted: {item['summary']}]"
                    )
                    r_stub = (
                        f"Batch not applied ({item['summary']}). "
                        "Scene and disk were unchanged; resolved in the following turn."
                    )
                    self.ledger.replace_text(
                        item["assistant_id"], text=a_stub, label="superseded attempt"
                    )
                    self.ledger.replace_text(
                        item["result_id"], text=r_stub, label=item["label"]
                    )
                pending_failed_turns.clear()

            for _step in range(MAX_STEPS):
                if cancel_ev.is_set():
                    self.emit("cancelled", {"turn": self.stats["turns"]})
                    return {"status": "cancelled"}

                if self.cfg.max_cost > 0 and self.run_cost >= self.cfg.max_cost:
                    msg = (
                        f"Session run spend (${self.run_cost:.4f}; accumulated ${self.stats['cost']:.4f}) "
                        f"reached BLENDERBOT_MAX_COST (${self.cfg.max_cost:.2f})"
                    )
                    self.emit("error", {"message": msg})
                    return {"status": "over_budget", "message": msg}

                pre_health = self.context_health()
                degraded_compact = pre_health["recommended_action"] == "compact"
                if self.ledger.needs_compaction or degraded_compact:
                    self.emit(
                        "compact-start",
                        {
                            "tokens": self.ledger.total_tokens,
                            "reason": "health-degraded" if degraded_compact else "token-threshold",
                        },
                    )
                    cmp_res = await compact_ledger(self.ledger, force=degraded_compact)
                    self.stats["compactions"] += 1
                    self._reset_rolling_health()
                    self.emit("compact-done", cmp_res)
                    if degraded_compact and cmp_res.get("freed_tokens", 0) > 0:
                        self.emit(
                            "warning",
                            {
                                "message": (
                                    f"Context health {pre_health['status']} (score={pre_health['score']}); "
                                    f"auto-compacted stale media/results (-{cmp_res['freed_tokens']:,} tok)"
                                )
                            },
                        )
                        if self.session_log is not None:
                            self.session_log.record(
                                "compact",
                                {"reason": "health-degraded", "pre_health": pre_health, **cmp_res},
                            )

                turn_out: dict[str, Any] | None = None
                for attempt in range(MAX_PROVIDER_RETRIES + 1):
                    try:
                        turn_out = await self._stream_single_turn(
                            cancel_ev, effort_override=cur_effort
                        )
                        break
                    except ProviderError as err:
                        if (
                            err.retryable
                            and attempt < MAX_PROVIDER_RETRIES
                            and not cancel_ev.is_set()
                        ):
                            delay = min(15.0, 1.5 * (2**attempt))
                            self.emit(
                                "retry",
                                {"attempt": attempt + 1, "delay": delay, "error": str(err)},
                            )
                            await asyncio.sleep(delay)
                            continue
                        self.emit("error", {"message": str(err)})
                        return {"status": "error", "message": str(err)}

                assert turn_out is not None

                if turn_out["degenerate"] is not None:
                    self._record_health_signal(
                        "degenerate_repetition", f"period={turn_out['degenerate']['period']}"
                    )
                    msg = (
                        f"Aborted degenerate repeating generation "
                        f"(period={turn_out['degenerate']['period']} chars)"
                    )
                    self.emit("error", {"message": msg})
                    return {"status": "error", "message": msg}

                if cancel_ev.is_set():
                    self.emit("cancelled", {"turn": self.stats["turns"]})
                    return {"status": "cancelled"}

                text = turn_out["text"]

                if not text.strip() and turn_out["stop_reason"] == "max_tokens":
                    self._rolling_health["empty"] += 1
                    self._record_health_signal("max_tokens_empty", "hit max_tokens during thinking")
                    eff_now = cur_effort or self.cfg.effort
                    lower = LOWER_EFFORT.get(eff_now)
                    if lower:
                        cur_effort = lower
                        self.emit(
                            "warning",
                            {
                                "message": (
                                    f"Turn hit max_tokens during thinking; retrying at effort={lower}"
                                )
                            },
                        )
                        continue

                if not text.strip() and not turn_out["native_calls"]:
                    empty_turns += 1
                    self._rolling_health["empty"] += 1
                    self._record_health_signal("empty_turn", f"attempt={empty_turns}")
                    if empty_turns <= MAX_EMPTY:
                        continue
                    self.emit("error", {"message": "Model returned empty responses repeatedly."})
                    return {"status": "error", "message": "empty_turns"}
                empty_turns = 0

                assistant_seg = self.ledger.add(
                    seg_type="assistant",
                    label=f"turn #{self.stats['turns']}",
                    role="assistant",
                    text=text or "[tool_calls]",
                )
                if self.session_log is not None:
                    self.session_log.record(
                        "assistant", {"turn": self.stats["turns"], "text": text}
                    )

                try:
                    extracted = extract_turn(text)
                except ParseError as p_err:
                    self.stats["malformed"] += 1
                    self._rolling_health["malformed"] += 1
                    repairs += 1
                    self._record_health_signal("malformed_bl_ops", str(p_err))
                    self.emit("repair", {"attempt": repairs, "error": str(p_err)})
                    if repairs > MAX_REPAIRS:
                        self.emit(
                            "error",
                            {
                                "message": (
                                    f"Malformed bl_ops after {MAX_REPAIRS} repair attempts: {p_err}"
                                )
                            },
                        )
                        return {"status": "error", "message": str(p_err)}
                    repair_msg = self.prompts["repair"].replace("{error}", str(p_err))
                    rep_seg = self.ledger.add(
                        seg_type="user", label="repair", role="user", text=repair_msg
                    )
                    pending_failed_turns.append(
                        {
                            "assistant_id": assistant_seg.id,
                            "result_id": rep_seg.id,
                            "prose": "",
                            "summary": f"malformed bl_ops ({str(p_err).splitlines()[0][:80]})",
                            "label": "repair (compressed)",
                        }
                    )
                    continue

                repairs = 0
                if extracted.recovered:
                    self._rolling_health["salvaged"] += 1
                    self._record_health_signal("recovered_fence", extracted.recovered)
                    self.emit(
                        "warning", {"message": f"Recovered bl_ops block ({extracted.recovered})"}
                    )

                for art_blk in extracted.artifacts:
                    if not art_blk.body.strip():
                        continue
                    art = self.artifacts.add(
                        lang=art_blk.lang, body=art_blk.body, turn=self.stats["turns"]
                    )
                    self.emit("artifact", art.to_meta_dict())

                ops = extracted.ops
                if not ops and turn_out["native_calls"]:
                    ops = _salvage_native_tool_calls(turn_out["native_calls"])
                    if ops:
                        self._rolling_health["salvaged"] += 1
                        self._record_health_signal(
                            "salvaged_tool_calls", f"count={len(ops)}"
                        )
                        self.emit(
                            "warning",
                            {"message": f"Salvaged {len(ops)} native tool call(s) as bl_ops"},
                        )

                if not ops:
                    if self._drain_steers():
                        continue
                    last_sentence = text.strip().splitlines()[-1] if text.strip() else ""
                    if (
                        INTENT_RE.search(last_sentence)
                        and not CONDITIONAL_OFFER_RE.search(last_sentence)
                        and nudges < MAX_NUDGES
                    ):
                        nudges += 1
                        self.stats["nudges"] += 1
                        self._rolling_health["nudges"] += 1
                        self._record_health_signal("intent_without_ops", last_sentence[:120])
                        self.emit("nudge", {"attempt": nudges})
                        self.ledger.add(
                            seg_type="user",
                            label="nudge",
                            role="user",
                            text=self.prompts["nudge"],
                        )
                        continue

                    _compress_superseded()
                    self.emit("done", {"turns": self.stats["turns"], "cost": self.stats["cost"]})
                    return {"status": "done", "text": extracted.prose or text}

                nudges = 0
                self.emit(
                    "ops-parsed",
                    [{"index": c.index, "op": c.op, "params": c.params} for c in ops],
                )

                flight = getattr(self.session_log, "flight", None)
                if flight is not None:
                    flight.mark(
                        "ops:execute",
                        turn=self.stats["turns"],
                        count=len(ops),
                        ops=[c.op for c in ops],
                    )

                last_flight_note = 0.0

                def _on_op_progress(pdata: dict[str, Any], _fl: Any = flight) -> None:
                    nonlocal last_flight_note
                    if _fl is not None:
                        _fl.touch()
                        now_m = time.monotonic()
                        if pdata.get("phase") == "frame_done" and (now_m - last_flight_note) >= 8.0:
                            last_flight_note = now_m
                            _fl.note(
                                "render-progress",
                                op=pdata.get("op"),
                                done_frames=pdata.get("done_frames"),
                                total_frames=pdata.get("total_frames"),
                                pct=pdata.get("pct"),
                                eta_s=pdata.get("eta_s"),
                            )
                    self.emit("op-progress", pdata)

                outcome = await execute_batch(
                    ops,
                    sandbox=self.sandbox,
                    journal=self.journal,
                    transport=self.transport,
                    cfg=self.cfg,
                    role=self.cfg.role,
                    yolo=self.cfg.yolo,
                    shell_timeout=self.cfg.shell_timeout,
                    approve=self._request_approval,
                    on_progress=_on_op_progress,
                )
                self.stats["ops"] += len(ops)
                for r_item in outcome.results:
                    res_d = r_item.get("result")
                    if r_item.get("ok") and isinstance(res_d, dict) and res_d.get("cost_usd"):
                        op_cost = float(res_d["cost_usd"] or 0.0)
                        if op_cost > 0:
                            self.run_cost += op_cost
                            self.stats["cost"] += op_cost
                if outcome.status in ("plan-failed", "apply-failed"):
                    self.stats["op_failures"] += 1
                    self._rolling_health["op_failures"] += 1
                    self._rolling_health["consecutive_op_failures"] += 1
                    err_head = (
                        outcome.errors[0]["message"].splitlines()[0][:140]
                        if outcome.errors
                        else outcome.status
                    )
                    h_snap = self._record_health_signal(outcome.status, err_head)
                else:
                    self._rolling_health["consecutive_op_failures"] = 0
                    h_snap = self.context_health()

                formatted = format_batch_results(outcome)
                wrapped_res = self.prompts["ops_result"].replace("{results}", formatted)
                if self._rolling_health["consecutive_op_failures"] >= 2:
                    wrapped_res += (
                        "\n\n[Protocol & Scene Health Hint: Multiple consecutive bl_ops batches failed "
                        f"(health={h_snap['status']}, score={h_snap['score']}). Stop repeating the failing "
                        "approach — check exact op/parameter names in `# The bl_ops protocol` and verify "
                        "existing objects in `[Live Scene State]`, then emit a minimal corrective batch.]"
                    )

                res_seg = self.ledger.add(
                    seg_type="op-result",
                    label=f"batch ({len(ops)} ops: {outcome.status})",
                    role="user",
                    text=wrapped_res,
                )

                # Zero-effect failures (`plan-failed`, or `apply-failed` where `.blend` & files rolled back cleanly
                # with no shell side effects) compress into 1-line stubs once a subsequent turn succeeds!
                has_shell_commit = any(
                    r.get("op") == "shell" for r in outcome.results
                )
                if outcome.status == "plan-failed" or (
                    outcome.status == "apply-failed"
                    and outcome.checkpoint is None
                    and not has_shell_commit
                ):
                    err0 = (
                        outcome.errors[0]["message"].splitlines()[0][:80]
                        if outcome.errors
                        else outcome.status
                    )
                    pending_failed_turns.append(
                        {
                            "assistant_id": assistant_seg.id,
                            "result_id": res_seg.id,
                            "prose": extracted.prose,
                            "summary": f"{outcome.status}: {err0}",
                            "label": f"{outcome.status} (compressed)",
                        }
                    )
                else:
                    _compress_superseded()

                # Attach rendered stills / contact sheets / playblasts into multimodal context!
                for m_item in outcome.attached_media:
                    m_kind = m_item["kind"]
                    m_path = m_item["path"]
                    if m_kind == "image" and self.cfg.model.vision:
                        try:
                            img = load_image_part(m_path)
                            self.ledger.add(
                                seg_type="media",
                                label=f"render preview ({img['name']})",
                                role="user",
                                image=img,
                            )
                            # Keep at most 6 unpinned auto-attached render previews in active context
                            auto_stills = [
                                s
                                for s in self.ledger.segments
                                if s.type == "media"
                                and not s.pinned
                                and s.label.startswith("render preview (")
                            ]
                            for old_s in auto_stills[:-6]:
                                self.ledger.drop(old_s.id)
                        except Exception as img_err:  # noqa: BLE001
                            self.emit(
                                "warning",
                                {"message": f"Skipped unreadable image {m_path}: {img_err}"},
                            )
                    elif m_kind == "video":
                        if self.cfg.model.video:
                            try:
                                vid = load_video_part(m_path)
                                self.ledger.add(
                                    seg_type="media",
                                    label=f"playblast video ({vid['name']}, {vid['delivery']})",
                                    role="user",
                                    video=vid,
                                )
                            except Exception as vid_err:  # noqa: BLE001
                                self.emit(
                                    "warning",
                                    {"message": f"Skipped unreadable video {m_path}: {vid_err}"},
                                )
                        else:
                            self.emit(
                                "warning",
                                {
                                    "message": (
                                        f"Video {Path(m_path).name} rendered, but {self.cfg.model.id} "
                                        "does not accept video natively (use `viewport.contact_sheet` or switch to a Gemini model)."
                                    )
                                },
                            )

                res_event = {
                    "status": outcome.status,
                    "results": outcome.results,
                    "errors": outcome.errors,
                    "checkpoint": (
                        Path(outcome.checkpoint.snapshot_path).name
                        if outcome.checkpoint
                        else None
                    ),
                    "attached_media": outcome.attached_media,
                }
                self.emit("ops-result", res_event)
                if self.session_log is not None:
                    self.session_log.record("ops-result", res_event)

                self._drain_steers()
                if self._drain_prompts_hook is not None:
                    applied_p = self._drain_prompts_hook()
                    if applied_p is not None:
                        self.emit("prompts-applied", applied_p)

            self.emit("error", {"message": f"Reached MAX_STEPS ({MAX_STEPS})"})
            return {"status": "max_steps"}
        finally:
            self.busy = False
            self._active_user_prompt = ""
            self._cancel_event = None
            if self.session_log is not None:
                flight = getattr(self.session_log, "flight", None)
                if flight is not None:
                    flight.mark("idle", turns=self.stats["turns"], ops=self.stats["ops"])
                self.session_log.update_totals(
                    self.stats,
                    provider=self.cfg.provider,
                    model_id=self.cfg.model.id,
                    health=self.context_health(),
                )