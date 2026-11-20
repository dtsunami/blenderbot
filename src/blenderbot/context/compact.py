"""Context auto-compaction policy.

Evicts older unpinned rendered stills/contact sheets (`media`) and bulky `op-result` dumps first so
the cached system prompt prefix stays intact and turn alternation remains valid.
"""

from __future__ import annotations

from typing import Any

from .ledger import PROTECTED_TYPES, Ledger
from .tokens import estimate_text_tokens


async def compact_ledger(
    ledger: Ledger,
    *,
    target_fraction: float = 0.50,
    force: bool = False,
    summarizer: Any = None,
) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        before_tokens = ledger.total_tokens
        evicted = 0
        truncated = 0
        eff_target = (
            min(target_fraction, max(0.01, ledger.usage_fraction * 0.45))
            if force
            else target_fraction
        )

        compressible = ledger.segments[:-4] if len(ledger.segments) > 4 else []

        # Pass 1: Evict stale media attachments & bulky op-result JSON payloads
        for seg in compressible:
            if not force and ledger.usage_fraction <= eff_target:
                break
            if force and evicted > 0 and ledger.usage_fraction <= eff_target:
                break
            if seg.pinned or seg.type in PROTECTED_TYPES:
                continue
            if seg.type in ("media", "op-result") and seg.tokens > 250:
                note = f"[compacted {seg.type}: {seg.label} — re-run op if needed]"
                seg.part = {"type": "text", "text": note}
                seg.tokens = estimate_text_tokens(note)
                evicted += 1

        # Pass 2: Truncate large shell/paste blocks
        for seg in compressible:
            if ledger.usage_fraction <= eff_target:
                break
            if seg.pinned or seg.type in PROTECTED_TYPES:
                continue
            if seg.type in ("shell", "paste") and seg.part.get("type") == "text" and seg.tokens > 500:
                txt = str(seg.part.get("text", ""))
                clipped = txt[:800] + f"\n… [truncated {len(txt) - 800} chars during compaction]"
                seg.part = {"type": "text", "text": clipped}
                seg.tokens = estimate_text_tokens(clipped)
                truncated += 1

        # Pass 3: Condense oldest unpinned turns if still above target.
        # CRITICAL: Never condense the latest `user` instruction (`last_user_idx`), even across
        # long 20+ step autonomous tool loops, and preserve full user request text (not 48-char
        # labels) when summarizing older turns so multi-part instructions are never clipped.
        summarized = 0
        if ledger.usage_fraction > eff_target and len(ledger.segments) > 8:
            last_user_idx = max(
                (i for i, s in enumerate(ledger.segments) if s.type == "user"),
                default=-1,
            )
            candidates = [
                i
                for i, s in enumerate(ledger.segments[:-6])
                if i != last_user_idx and not s.pinned and s.type not in PROTECTED_TYPES
            ]
            if len(candidates) >= 2:
                slice_segs = [ledger.segments[i] for i in candidates[: min(8, len(candidates))]]

                def _clean_user_body(raw_txt: str) -> str:
                    txt = str(raw_txt or "").strip()
                    if txt.startswith("[Blender Live:") and "\n\n" in txt:
                        txt = txt.split("\n\n", 1)[1].strip()
                    return txt[:1500]

                transcript_text = "\n\n".join(
                    f"[{s.role.upper()} - {s.label}]\n{s.part.get('text', '[media]')[:1500]}"
                    for s in slice_segs
                )
                summary_body: str
                if callable(summarizer):
                    summary_body = await summarizer(transcript_text)
                else:
                    lines = []
                    for s in slice_segs:
                        if s.type == "user" and s.part.get("type") == "text":
                            lines.append(f"- user: {_clean_user_body(str(s.part.get('text', ''))) or s.label}")
                        else:
                            lines.append(f"- {s.role}: {s.label}")
                    summary_body = "[Earlier session summary]\n" + "\n".join(lines)

                drop_ids = {s.id for s in slice_segs}
                first_idx = next(i for i, s in enumerate(ledger.segments) if s.id in drop_ids)
                ledger.segments = [s for s in ledger.segments if s.id not in drop_ids]
                seg = ledger.add(
                    seg_type="summary",
                    label=f"summary ({len(slice_segs)} segments)",
                    role="user",
                    text=summary_body,
                )
                ledger.segments.pop()
                ledger.segments.insert(first_idx, seg)
                summarized = len(slice_segs)

        after_tokens = ledger.total_tokens
        return {
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "freed_tokens": max(0, before_tokens - after_tokens),
            "evicted": evicted,
            "truncated": truncated,
            "summarized": summarized,
            "dropped_count": evicted + truncated + summarized,
        }

    return await ledger.with_compaction_lock(_run)