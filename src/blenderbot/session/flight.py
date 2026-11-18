"""Flight recorder (`sessions/<id>/breadcrumbs.jsonl`) ported from `davai/src/session/flight.js`.

Records synchronous phase transitions (`mark`), chunk-level liveness (`touch`), and
silent async stall heartbeats (`idle_ms` vs `elapsed_ms`) so half-open sockets or
slow Blender RPCs are immediately diagnosable via `/flight`.
"""

from __future__ import annotations

import atexit
import datetime
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

BREADCRUMB_FILE = "breadcrumbs.jsonl"
DEFAULT_CHECK_SEC = 2.0
DEFAULT_STALL_SEC = 15.0
HUMAN_WAIT_PHASES = frozenset({"idle", "approval-wait", "closed"})


class FlightRecorder:
    def __init__(
        self,
        session_dir: Path,
        *,
        check_sec: float = DEFAULT_CHECK_SEC,
        stall_sec: float = DEFAULT_STALL_SEC,
    ) -> None:
        self.file = session_dir / BREADCRUMB_FILE
        self.check_sec = check_sec
        self.stall_sec = stall_sec
        self._lock = threading.Lock()
        self.closed = False
        self.broken = False

        now = time.monotonic()
        self.phase = "init"
        self.detail: dict[str, Any] = {}
        self.phase_since = now
        self.last_activity_at = now
        self.last_stall_report_at = 0.0

        self._stop_ev = threading.Event()
        self._thread = threading.Thread(
            target=self._watch_loop, name="blenderbot-flight", daemon=True
        )
        self._thread.start()
        atexit.register(self._atexit_close)

    def _atexit_close(self) -> None:
        if not self.closed:
            self.close("process-exit")

    def _write(self, record: dict[str, Any]) -> None:
        if self.broken or self.closed:
            return
        payload = {
            "ts": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "pid": os.getpid(),
            **record,
        }
        try:
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            with open(self.file, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except OSError:
            self.broken = True

    def mark(self, phase: str, **detail: Any) -> None:
        with self._lock:
            if self.closed:
                return
            now = time.monotonic()
            prev_phase = self.phase
            prev_ms = round((now - self.phase_since) * 1000)
            self.phase = phase
            self.detail = dict(detail)
            self.phase_since = now
            self.last_activity_at = now
            self.last_stall_report_at = 0.0
            rec: dict[str, Any] = {"kind": "phase", "phase": phase}
            if prev_phase and prev_phase != "init":
                rec["prev"] = prev_phase
                rec["prev_ms"] = prev_ms
            rec.update(detail)
            self._write(rec)

    def touch(self) -> None:
        """Refresh liveness during active streaming without disk I/O."""
        if not self.closed:
            self.last_activity_at = time.monotonic()

    def note(self, event: str, **detail: Any) -> None:
        with self._lock:
            if self.closed:
                return
            now = time.monotonic()
            self._write(
                {
                    "kind": "note",
                    "event": event,
                    "phase": self.phase,
                    "phase_elapsed_ms": round((now - self.phase_since) * 1000),
                    **detail,
                }
            )

    def tick(self) -> None:
        with self._lock:
            if self.closed or self.phase in HUMAN_WAIT_PHASES:
                return
            now = time.monotonic()
            idle_sec = now - self.last_activity_at
            elapsed_sec = now - self.phase_since
            if idle_sec >= self.stall_sec and (
                self.last_stall_report_at == 0.0
                or (now - self.last_stall_report_at) >= self.stall_sec
            ):
                self.last_stall_report_at = now
                self._write(
                    {
                        "kind": "stall-heartbeat",
                        "phase": self.phase,
                        "elapsed_ms": round(elapsed_sec * 1000),
                        "idle_ms": round(idle_sec * 1000),
                        **self.detail,
                    }
                )

    def _watch_loop(self) -> None:
        while not self._stop_ev.wait(self.check_sec):
            self.tick()

    def close(self, reason: str = "exit", **extra: Any) -> None:
        with self._lock:
            if self.closed:
                return
            self._stop_ev.set()
            now = time.monotonic()
            self._write(
                {
                    "kind": "close",
                    "reason": reason,
                    "last_phase": self.phase,
                    "last_phase_ms": round((now - self.phase_since) * 1000),
                    **extra,
                }
            )
            self.closed = True


def read_breadcrumbs(session_dir: Path, limit: int = 40) -> list[dict[str, Any]]:
    f = session_dir / BREADCRUMB_FILE
    if not f.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out[-limit:] if limit > 0 else out