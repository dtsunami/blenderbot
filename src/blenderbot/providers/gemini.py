"""Google Gemini adapter (`google-genai`) with Multimodal Image, Video, Files API & YouTube support."""

from __future__ import annotations

import asyncio
import base64
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from .base import ProviderError, StreamEvent, wrap_provider_error
from .watchdog import DEFAULT_FIRST_CHUNK_TIMEOUT_SEC, iterate_with_idle_timeout

DEFAULT_STREAM_IDLE_TIMEOUT_SEC = DEFAULT_FIRST_CHUNK_TIMEOUT_SEC
LEAKED_THOUGHT_PREFIX_RE = re.compile(r"^[ \t]*(?:düşünceler|sthought)\r?\n")
_LEAK_TOKENS = ("sthought", "düşünceler")
_LEAKED_FENCE_SEAM_RE = re.compile(r"\r?\n(?=```(?:bl_ops|da_ops|dr_ops)\b)")
# When Gemini's `<ctrl94>` misfires as `sthought\n`, `<ctrl95>` (end-of-thought) is stripped by the
# decoder with zero whitespace, gluing sentence punctuation directly to the response start (`concisely.Built`).
_LEAKED_CTRL95_GLUE_RE = re.compile(r"([a-z]{2,}[.!?])([A-Z#*`])")
_MAX_LEAKED_THOUGHT_BUF = 4096
_UPLOADED_FILES_CACHE: dict[str, tuple[str, str]] = {}


def _could_be_leaked_prefix(s: str) -> bool:
    """Return True while `s` can still grow into `^[ \\t]*(?:sthought|düşünceler)\\r?\\n`."""
    stripped = s.lstrip(" \t")
    if not stripped:
        return len(s) < 8
    if "\n" in stripped:
        return bool(LEAKED_THOUGHT_PREFIX_RE.match(s))
    core = stripped.rstrip("\r")
    if len(core) < len(stripped) and stripped[-1] != "\r":
        return False
    return any(tok.startswith(core) for tok in _LEAK_TOKENS)


class GeminiThoughtSanitizer:
    """Routes leaked unflagged `sthought` / `düşünceler` chunks into `thinking` + emits `thought_leak`."""

    def __init__(self) -> None:
        self.saw_flagged_thought = False
        self.saw_text = False
        self.in_leaked_thought = False
        self.leaked_detected = False
        self._prefix_buf = ""
        self._leaked_buf = ""

    def feed(self, txt: str, *, is_thought: bool, part_index: int = 0) -> list[StreamEvent]:
        if not txt:
            return []
        if is_thought:
            self.saw_flagged_thought = True
            return [StreamEvent(t="thinking", delta=txt)]

        if self.saw_text or self.saw_flagged_thought:
            self.saw_text = True
            return [StreamEvent(t="text", delta=txt)]

        events: list[StreamEvent] = []

        # Multi-part boundary within a single candidate while inside a leaked thought:
        # Part 0 was pre-`<ctrl95>` thought, Part 1+ is visible answer!
        if self.in_leaked_thought and part_index > 0:
            if self._leaked_buf:
                events.append(StreamEvent(t="thinking", delta=self._leaked_buf))
                self._leaked_buf = ""
            self.in_leaked_thought = False
            self.saw_text = True
            events.append(StreamEvent(t="text", delta=txt))
            return events

        if not self.in_leaked_thought:
            self._prefix_buf += txt
            m = LEAKED_THOUGHT_PREFIX_RE.match(self._prefix_buf)
            if m is not None:
                remainder = self._prefix_buf[m.end() :]
                self._prefix_buf = ""
                self.in_leaked_thought = True
                if not self.leaked_detected:
                    self.leaked_detected = True
                    events.append(
                        StreamEvent(
                            t="diagnostic",
                            reason="thought_leak",
                            detail="Gemini leaked unflagged sthought/düşünceler prefix",
                        )
                    )
                if not remainder:
                    return events
                txt = remainder
            elif _could_be_leaked_prefix(self._prefix_buf):
                return events
            else:
                out_txt = self._prefix_buf
                self._prefix_buf = ""
                self.saw_text = True
                events.append(StreamEvent(t="text", delta=out_txt))
                return events

        # Currently buffering post-`sthought\n` body until we hit a seam or stream end
        self._leaked_buf += txt
        lstripped = self._leaked_buf.lstrip(" \t\r\n")
        if lstripped.startswith("```"):
            out_txt = self._leaked_buf
            self._leaked_buf = ""
            self.in_leaked_thought = False
            self.saw_text = True
            events.append(StreamEvent(t="text", delta=out_txt))
            return events

        fence_m = _LEAKED_FENCE_SEAM_RE.search(self._leaked_buf)
        if fence_m is not None:
            thought_part = self._leaked_buf[: fence_m.start()]
            answer_part = self._leaked_buf[fence_m.end() :]
            self._leaked_buf = ""
            self.in_leaked_thought = False
            self.saw_text = True
            if thought_part:
                events.append(StreamEvent(t="thinking", delta=thought_part))
            if answer_part:
                events.append(StreamEvent(t="text", delta=answer_part))
            return events

        glue_m = _LEAKED_CTRL95_GLUE_RE.search(self._leaked_buf)
        if glue_m is not None:
            split_at = glue_m.end(1)
            thought_part = self._leaked_buf[:split_at]
            answer_part = self._leaked_buf[split_at:]
            self._leaked_buf = ""
            self.in_leaked_thought = False
            self.saw_text = True
            if thought_part:
                events.append(StreamEvent(t="thinking", delta=thought_part))
            if answer_part:
                events.append(StreamEvent(t="text", delta=answer_part))
            return events

        if len(self._leaked_buf) >= _MAX_LEAKED_THOUGHT_BUF:
            out_txt = self._leaked_buf
            self._leaked_buf = ""
            self.in_leaked_thought = False
            self.saw_text = True
            events.append(StreamEvent(t="text", delta=out_txt))
        return events

    def flush(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if self._prefix_buf:
            m = LEAKED_THOUGHT_PREFIX_RE.match(self._prefix_buf + "\n")
            if m is not None and self._prefix_buf.lstrip(" \t").rstrip("\r") in _LEAK_TOKENS:
                if not self.leaked_detected:
                    self.leaked_detected = True
                    events.append(
                        StreamEvent(
                            t="diagnostic",
                            reason="thought_leak",
                            detail="Gemini leaked trailing sthought/düşünceler token",
                        )
                    )
            else:
                events.append(StreamEvent(t="text", delta=self._prefix_buf))
            self._prefix_buf = ""
        if self._leaked_buf:
            # No seam appeared before stream end (e.g. ` düşünceler\nHello` with empty thought):
            # preserve as visible text so valid responses are never swallowed.
            events.append(StreamEvent(t="text", delta=self._leaked_buf))
            self._leaked_buf = ""
            self.in_leaked_thought = False
        return events


class GeminiAdapter:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        thinking_visible: bool = True,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_visible = thinking_visible
        self.stream_idle_timeout = float(
            os.environ.get("BLENDERBOT_STREAM_IDLE_TIMEOUT")
            or os.environ.get("DAVAG_STREAM_IDLE_TIMEOUT")
            or DEFAULT_STREAM_IDLE_TIMEOUT_SEC
        )
        http_opts = types.HttpOptions(base_url=base_url) if base_url else None
        self._client = (
            genai.Client(api_key=api_key, http_options=http_opts)
            if http_opts is not None
            else genai.Client(api_key=api_key)
        )

    async def _ensure_uploaded(self, file_path: str, media_type: str) -> types.Part:
        key = str(Path(file_path).resolve())
        cached = _UPLOADED_FILES_CACHE.get(key)
        if cached is not None:
            uri, mime = cached
            return types.Part.from_uri(file_uri=uri, mime_type=mime)

        uploaded = await self._client.aio.files.upload(
            file=key,
            config=types.UploadFileConfig(mime_type=media_type),
        )
        file_name = uploaded.name
        assert file_name is not None

        for _ in range(120):
            state_str = str(getattr(uploaded.state, "name", uploaded.state) or "").upper()
            if "ACTIVE" in state_str:
                break
            if "FAILED" in state_str:
                raise ProviderError(
                    f"Gemini Files API processing failed for {Path(key).name}",
                    provider="gemini",
                )
            await asyncio.sleep(1.5)
            uploaded = await self._client.aio.files.get(name=file_name)

        uri = str(uploaded.uri)
        mime = str(uploaded.mime_type or media_type)
        _UPLOADED_FILES_CACHE[key] = (uri, mime)
        return types.Part.from_uri(file_uri=uri, mime_type=mime)

    async def _to_gemini_contents(self, messages: list[dict[str, Any]]) -> list[types.Content]:
        contents: list[types.Content] = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            parts: list[types.Part] = []
            for p in msg.get("content", []):
                ptype = p.get("type", "text")
                if ptype in ("image", "video"):
                    if p.get("uri"):
                        mime = p.get("media_type")
                        if mime:
                            parts.append(types.Part.from_uri(file_uri=str(p["uri"]), mime_type=mime))
                        else:
                            parts.append(types.Part(file_data=types.FileData(file_uri=str(p["uri"]))))
                    elif p.get("file_path"):
                        part = await self._ensure_uploaded(
                            str(p["file_path"]),
                            str(p.get("media_type") or "video/mp4"),
                        )
                        parts.append(part)
                    else:
                        raw_bytes = base64.b64decode(p["data"])
                        parts.append(
                            types.Part.from_bytes(
                                data=raw_bytes,
                                mime_type=str(p.get("media_type") or "image/png"),
                            )
                        )
                else:
                    parts.append(types.Part.from_text(text=str(p.get("text", ""))))
            contents.append(types.Content(role=role, parts=parts))
        return contents

    async def send(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        cancel_event: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        try:
            contents = await self._to_gemini_contents(messages)
            cfg = types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                thinking_config=(
                    types.ThinkingConfig(include_thoughts=True) if self.thinking_visible else None
                ),
            )

            idle_timeout = self.stream_idle_timeout
            try:
                stream = await asyncio.wait_for(
                    self._client.aio.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=cfg,
                    ),
                    timeout=idle_timeout,
                )
            except TimeoutError as t_err:
                raise ProviderError(
                    f"[gemini] Timed out waiting >{idle_timeout:.0f}s for stream open",
                    provider="gemini",
                    status=408,
                    retryable=True,
                ) from t_err

            usage_meta = None
            finish_reason = "end_turn"
            sanitizer = GeminiThoughtSanitizer()
            async for chunk in iterate_with_idle_timeout(
                stream,
                provider="gemini",
                cancel_event=cancel_event,
                first_chunk_sec=idle_timeout,
            ):
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage_meta = chunk.usage_metadata
                cands = getattr(chunk, "candidates", None) or []
                if cands:
                    cand = cands[0]
                    if getattr(cand, "finish_reason", None):
                        finish_reason = str(getattr(cand.finish_reason, "name", cand.finish_reason))
                    content = getattr(cand, "content", None)
                    for p_idx, part in enumerate(getattr(content, "parts", None) or []):
                        txt = getattr(part, "text", None)
                        if txt:
                            is_thought = bool(getattr(part, "thought", False))
                            for out_ev in sanitizer.feed(
                                txt, is_thought=is_thought, part_index=p_idx
                            ):
                                yield out_ev

            if cancel_event is not None and cancel_event.is_set():
                return

            for tail_ev in sanitizer.flush():
                yield tail_ev

            if usage_meta is not None:
                cached = int(getattr(usage_meta, "cached_content_token_count", 0) or 0)
                prompt = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
                cand_out = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
                thought_out = int(getattr(usage_meta, "thoughts_token_count", 0) or 0)
                yield StreamEvent(
                    t="usage",
                    in_fresh=max(0, prompt - cached),
                    out=cand_out + thought_out,
                    cache_read=cached,
                    cache_write=0,
                    context_in=prompt,
                )

            stop = (
                "max_tokens"
                if "MAX_TOKENS" in finish_reason
                else "refusal"
                if any(k in finish_reason for k in ("SAFETY", "PROHIBITED"))
                else "end_turn"
            )
            yield StreamEvent(t="stop", reason=stop)
        except Exception as err:
            raise wrap_provider_error(err, "gemini") from err

    async def count_tokens(
        self, *, system: str, messages: list[dict[str, Any]]
    ) -> int | None:
        try:
            contents = await self._to_gemini_contents(messages)
            res = await self._client.aio.models.count_tokens(
                model=self.model,
                contents=contents,
                config=types.CountTokensConfig(system_instruction=system),
            )
            return int(res.total_tokens) if res.total_tokens is not None else None
        except Exception:  # noqa: BLE001
            return None