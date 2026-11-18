"""Unified Undo/Redo Action Journal covering workspace files and `.blend` scene checkpoints.

Persisted to `<session_dir>/actions.jsonl` + `<session_dir>/journal.jsonl` + `snaps/` and
hydrated automatically on `blenderbot --resume` so the full edit history can be undone,
redone, or jumped across sessions via `scene.snapshot_save` / `scene.snapshot_restore`.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from blenderbot.transport.base import BlenderTransport


@dataclass
class FileEntry:
    target: str
    existed: bool
    snap_path: str | None
    mode: int | None
    ts: float


@dataclass
class BlendCheckpoint:
    seq: int
    snapshot_path: str
    original_filepath: str
    ts: float
    after_snapshot_path: str | None = None
    after_filepath: str = ""


@dataclass
class ActionRecord:
    id: str
    seq: int
    ts: float
    actor: str  # "operator" | "agent"
    action_type: str
    summary: str
    strategy: str  # "blend_swap" | "files_only"
    applied: bool = True
    superseded: bool = False
    checkpoint: dict[str, Any] | None = None
    before_files: list[dict[str, Any]] = field(default_factory=list)
    after_files: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _unlock_readonly_win(path: Path) -> None:
    try:
        st = path.stat()
        if not (st.st_mode & stat.S_IWRITE):
            os.chmod(path, st.st_mode | stat.S_IWRITE)
    except OSError:
        pass


class Journal:
    def __init__(
        self,
        session_dir: str | Path,
        *,
        resume_from: str | Path | None = None,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.snaps_dir = self.session_dir / "snaps"
        self.journal_file = self.session_dir / "journal.jsonl"
        self.actions_file = self.session_dir / "actions.jsonl"
        self.snaps_dir.mkdir(parents=True, exist_ok=True)
        self.on_change = on_change
        self._seq = 0
        self._act_seq = 0
        self.blend_checkpoints: list[BlendCheckpoint] = []
        self.actions: list[ActionRecord] = []

        if resume_from is not None:
            self._import_resumed_dir(Path(resume_from))
        self._load_persisted_actions()

    def _import_resumed_dir(self, prev_dir: Path) -> None:
        if prev_dir.resolve() == self.session_dir.resolve() or not prev_dir.is_dir():
            return
        prev_snaps = prev_dir / "snaps"
        if prev_snaps.is_dir():
            for item in prev_snaps.iterdir():
                if item.is_file():
                    dst = self.snaps_dir / item.name
                    if not dst.exists():
                        shutil.copy2(item, dst)
        for fname in ("journal.jsonl", "actions.jsonl"):
            src_f = prev_dir / fname
            dst_f = self.session_dir / fname
            if src_f.is_file() and not dst_f.exists():
                shutil.copy2(src_f, dst_f)

    def _remap_snap_path(self, raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        p = Path(raw_path)
        if p.is_file():
            return str(p)
        local = self.snaps_dir / p.name
        if local.is_file():
            return str(local)
        return raw_path

    def _load_persisted_actions(self) -> None:
        if not self.actions_file.is_file():
            return
        loaded: list[ActionRecord] = []
        max_snap_seq = 0
        for raw in self.actions_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            for flist_key in ("before_files", "after_files"):
                for fe in d.get(flist_key) or []:
                    fe["snap_path"] = self._remap_snap_path(fe.get("snap_path"))
            cp_dict = d.get("checkpoint")
            if isinstance(cp_dict, dict):
                cp_dict["snapshot_path"] = self._remap_snap_path(cp_dict.get("snapshot_path")) or ""
                if cp_dict.get("after_snapshot_path"):
                    cp_dict["after_snapshot_path"] = self._remap_snap_path(
                        cp_dict.get("after_snapshot_path")
                    )
                max_snap_seq = max(max_snap_seq, int(cp_dict.get("seq", 0)))
            rec = ActionRecord(
                id=str(d["id"]),
                seq=int(d["seq"]),
                ts=float(d.get("ts", time.time())),
                actor=str(d.get("actor", "agent")),
                action_type=str(d.get("action_type", "edit")),
                summary=str(d.get("summary", "")),
                strategy=str(d.get("strategy", "blend_swap")),
                applied=bool(d.get("applied", True)),
                superseded=bool(d.get("superseded", False)),
                checkpoint=cp_dict if isinstance(cp_dict, dict) else None,
                before_files=list(d.get("before_files") or []),
                after_files=list(d.get("after_files") or []),
                details=dict(d.get("details") or {}),
            )
            loaded.append(rec)
            if rec.checkpoint and rec.applied and not rec.superseded:
                self.blend_checkpoints.append(BlendCheckpoint(**rec.checkpoint))

        self.actions = loaded
        if loaded:
            self._act_seq = max(r.seq for r in loaded)
        self._seq = max(self._seq, max_snap_seq, self._act_seq * 2)

    def _append_log(self, kind: str, payload: dict[str, Any]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with open(self.journal_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "ts": time.time(), **payload}) + "\n")

    def _persist_actions(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.actions_file.with_name("actions.jsonl.tmp")
        lines = [json.dumps(asdict(r), ensure_ascii=False) for r in self.actions]
        tmp.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        os.replace(tmp, self.actions_file)

    def _notify(self) -> None:
        if callable(self.on_change):
            with contextlib.suppress(Exception):
                self.on_change(self.list_state())

    def _supersede_unapplied_blend_swaps(self) -> None:
        for r in self.actions:
            if not r.applied and not r.superseded and r.strategy == "blend_swap":
                r.superseded = True

    def snapshot_file(self, target: Path, *, log_event: bool = True) -> FileEntry:
        self._seq += 1
        existed = target.is_file()
        snap_path: str | None = None
        mode: int | None = None

        if existed:
            st = target.stat()
            mode = st.st_mode
            snap = self.snaps_dir / f"f-{self._seq:04d}-{target.name}"
            shutil.copy2(target, snap)
            snap_path = str(snap)

        entry = FileEntry(
            target=str(target),
            existed=existed,
            snap_path=snap_path,
            mode=mode,
            ts=time.time(),
        )
        if log_event:
            self._append_log("file_snapshot", asdict(entry))
        return entry

    def rollback_files(self, entries: list[FileEntry]) -> list[str]:
        failures: list[str] = []
        for entry in reversed(entries):
            target = Path(entry.target)
            snap_str = self._remap_snap_path(entry.snap_path)
            try:
                if not entry.existed:
                    if target.exists():
                        _unlock_readonly_win(target)
                        target.unlink()
                elif snap_str:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        _unlock_readonly_win(target)
                    shutil.copy2(snap_str, target)
                    if entry.mode is not None:
                        os.chmod(target, entry.mode)
            except OSError as err:
                failures.append(f"{target}: {err}")
        return failures

    async def create_blend_checkpoint(self, transport: BlenderTransport) -> BlendCheckpoint:
        """Save a `.blend` pre-mutation snapshot via `scene.snapshot_save`."""
        self._seq += 1
        res = await transport.call(
            "scene.snapshot_save",
            {"dir": str(self.snaps_dir), "prefix": f"b-{self._seq:04d}"},
        )
        cp = BlendCheckpoint(
            seq=self._seq,
            snapshot_path=str(res["snapshot_path"]),
            original_filepath=str(res.get("original_filepath") or ""),
            ts=time.time(),
        )
        self.blend_checkpoints.append(cp)
        self._append_log("blend_checkpoint", asdict(cp))
        return cp

    async def restore_blend_checkpoint(
        self, cp: BlendCheckpoint, transport: BlenderTransport
    ) -> dict[str, Any]:
        snap_path = self._remap_snap_path(cp.snapshot_path) or cp.snapshot_path
        return await transport.call(
            "scene.snapshot_restore",
            {
                "snapshot_path": snap_path,
                "original_filepath": cp.original_filepath,
            },
        )

    def record_batch_action(
        self,
        *,
        actor: str,
        action_type: str,
        summary: str,
        checkpoint: BlendCheckpoint | None = None,
        before_files: list[FileEntry] | None = None,
        after_files: list[FileEntry] | None = None,
        details: dict[str, Any] | None = None,
    ) -> ActionRecord:
        if checkpoint is not None:
            self._supersede_unapplied_blend_swaps()
            strategy = "blend_swap"
        else:
            strategy = "files_only"

        self._act_seq += 1
        rec = ActionRecord(
            id=f"act-{self._act_seq:04d}",
            seq=self._act_seq,
            ts=time.time(),
            actor=actor,
            action_type=action_type,
            summary=summary,
            strategy=strategy,
            applied=True,
            superseded=False,
            checkpoint=asdict(checkpoint) if checkpoint is not None else None,
            before_files=[asdict(f) for f in (before_files or [])],
            after_files=[asdict(f) for f in (after_files or [])],
            details=dict(details or {}),
        )
        self.actions.append(rec)
        self._persist_actions()
        self._append_log(
            "action_recorded",
            {"id": rec.id, "actor": actor, "type": action_type, "summary": summary},
        )
        self._notify()
        return rec

    def list_state(self) -> dict[str, Any]:
        active = [r for r in self.actions if not r.superseded]
        applied_list = [r for r in active if r.applied]
        undone_list = [r for r in active if not r.applied]
        cursor = applied_list[-1].seq if applied_list else 0
        max_seq = self.actions[-1].seq if self.actions else 0
        return {
            "actions": [asdict(r) for r in self.actions],
            "total": len(self.actions),
            "applied_count": len(applied_list),
            "undone_count": len(undone_list),
            "cursor": cursor,
            "max_seq": max_seq,
            "can_undo": bool(applied_list),
            "can_redo": bool(undone_list),
            "next_undo": applied_list[-1].summary if applied_list else None,
            "next_redo": undone_list[0].summary if undone_list else None,
        }

    async def _apply_single_undo(
        self, rec: ActionRecord, transport: BlenderTransport | None
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if rec.strategy == "blend_swap" and rec.checkpoint is not None:
            if transport is None:
                raise RuntimeError("Blender is not connected.")
            cp = rec.checkpoint
            # Capture post-mutation snapshot on first undo so Redo works cleanly!
            if not cp.get("after_snapshot_path"):
                self._seq += 1
                aft = await transport.call(
                    "scene.snapshot_save",
                    {"dir": str(self.snaps_dir), "prefix": f"b-after-{cp['seq']:04d}"},
                )
                cp["after_snapshot_path"] = str(aft["snapshot_path"])
                cp["after_filepath"] = str(aft.get("original_filepath") or "")

            snap_path = self._remap_snap_path(str(cp["snapshot_path"])) or str(cp["snapshot_path"])
            st = await transport.call(
                "scene.snapshot_restore",
                {
                    "snapshot_path": snap_path,
                    "original_filepath": str(cp.get("original_filepath") or ""),
                },
            )
            extra = {
                "checkpoint_seq": int(cp["seq"]),
                "scene": (st.get("scene") or {}).get("name"),
                "objects": (st.get("counts") or {}).get("objects"),
            }

        if rec.before_files:
            f_entries = [FileEntry(**fe) for fe in rec.before_files]
            errs = self.rollback_files(f_entries)
            if errs:
                extra["file_warnings"] = errs

        rec.applied = False
        return extra

    async def _apply_single_redo(
        self, rec: ActionRecord, transport: BlenderTransport | None
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if rec.strategy == "blend_swap" and rec.checkpoint is not None:
            if transport is None:
                raise RuntimeError("Blender is not connected.")
            cp = rec.checkpoint
            after_path = self._remap_snap_path(cp.get("after_snapshot_path"))
            if not after_path:
                raise RuntimeError(f"Action {rec.id} has no post-mutation .blend snapshot to redo.")
            st = await transport.call(
                "scene.snapshot_restore",
                {
                    "snapshot_path": after_path,
                    "original_filepath": str(cp.get("after_filepath") or cp.get("original_filepath") or ""),
                },
            )
            extra = {
                "checkpoint_seq": int(cp["seq"]),
                "scene": (st.get("scene") or {}).get("name"),
                "objects": (st.get("counts") or {}).get("objects"),
            }

        if rec.after_files:
            f_entries = [FileEntry(**fe) for fe in rec.after_files]
            errs = self.rollback_files(f_entries)
            if errs:
                extra["file_warnings"] = errs

        rec.applied = True
        return extra

    def _find_action(self, entry_id: str) -> ActionRecord:
        q = entry_id.strip().lower()
        for r in self.actions:
            if r.id.lower() == q or (q.isdigit() and r.seq == int(q)):
                return r
        raise KeyError(f'Action "{entry_id}" not found in journal.')

    async def undo_entry(
        self, entry_id: str, transport: BlenderTransport | None
    ) -> dict[str, Any]:
        rec = self._find_action(entry_id)
        if rec.superseded:
            raise RuntimeError(f"Action {rec.id} was superseded by a later scene branch.")
        if not rec.applied:
            return {"ok": True, "already": "undone", "action": asdict(rec)}
        if rec.strategy == "blend_swap":
            return await self.goto_cursor(rec.seq - 1, transport)

        extra = await self._apply_single_undo(rec, transport)
        self._persist_actions()
        self._append_log("action_undo", {"id": rec.id, "actor": rec.actor, "summary": rec.summary})
        self._notify()
        return {"ok": True, "undone": asdict(rec), **extra}

    async def redo_entry(
        self, entry_id: str, transport: BlenderTransport | None
    ) -> dict[str, Any]:
        rec = self._find_action(entry_id)
        if rec.superseded:
            raise RuntimeError(f"Action {rec.id} was superseded by a later scene branch.")
        if rec.applied:
            return {"ok": True, "already": "applied", "action": asdict(rec)}
        if rec.strategy == "blend_swap":
            return await self.goto_cursor(rec.seq, transport)

        extra = await self._apply_single_redo(rec, transport)
        self._persist_actions()
        self._append_log("action_redo", {"id": rec.id, "actor": rec.actor, "summary": rec.summary})
        self._notify()
        return {"ok": True, "redone": asdict(rec), **extra}

    async def undo_last(
        self, transport: BlenderTransport | None, *, steps: int = 1
    ) -> dict[str, Any]:
        count = max(1, int(steps))
        undone_items: list[dict[str, Any]] = []
        last_extra: dict[str, Any] = {}

        for _ in range(count):
            cand = next(
                (r for r in reversed(self.actions) if r.applied and not r.superseded), None
            )
            if cand is None:
                if not undone_items:
                    raise RuntimeError("No applied actions available to undo in this session.")
                break

            last_extra = await self._apply_single_undo(cand, transport)
            undone_items.append(asdict(cand))
            self._append_log(
                "action_undo", {"id": cand.id, "actor": cand.actor, "summary": cand.summary}
            )

        self._persist_actions()
        self._notify()
        return {
            "ok": True,
            "undone": undone_items[-1],
            "undone_list": undone_items,
            "count": len(undone_items),
            **last_extra,
        }

    async def redo_next(
        self, transport: BlenderTransport | None, *, steps: int = 1
    ) -> dict[str, Any]:
        count = max(1, int(steps))
        redone_items: list[dict[str, Any]] = []
        last_extra: dict[str, Any] = {}

        for _ in range(count):
            applied_seqs = [r.seq for r in self.actions if r.applied and not r.superseded]
            min_seq = max(applied_seqs) if applied_seqs else 0
            cand = next(
                (r for r in self.actions if not r.applied and not r.superseded and r.seq > min_seq),
                None,
            ) or next((r for r in self.actions if not r.applied and not r.superseded), None)

            if cand is None:
                if not redone_items:
                    raise RuntimeError("No undone actions available to redo in this session.")
                break

            last_extra = await self._apply_single_redo(cand, transport)
            redone_items.append(asdict(cand))
            self._append_log(
                "action_redo", {"id": cand.id, "actor": cand.actor, "summary": cand.summary}
            )

        self._persist_actions()
        self._notify()
        return {
            "ok": True,
            "redone": redone_items[-1],
            "redone_list": redone_items,
            "count": len(redone_items),
            **last_extra,
        }

    async def goto_cursor(
        self, target_seq: int, transport: BlenderTransport | None
    ) -> dict[str, Any]:
        target = max(0, int(target_seq))
        undone_ids: list[str] = []
        redone_ids: list[str] = []
        last_extra: dict[str, Any] = {}

        for rec in reversed(self.actions):
            if rec.superseded:
                continue
            if rec.seq > target and rec.applied:
                last_extra = await self._apply_single_undo(rec, transport)
                undone_ids.append(rec.id)
                self._append_log("cursor_undo", {"id": rec.id, "target_seq": target})

        for rec in self.actions:
            if rec.superseded:
                continue
            if rec.seq <= target and not rec.applied:
                last_extra = await self._apply_single_redo(rec, transport)
                redone_ids.append(rec.id)
                self._append_log("cursor_redo", {"id": rec.id, "target_seq": target})

        self._persist_actions()
        self._notify()
        return {
            "ok": True,
            "cursor": target,
            "undone_ids": undone_ids,
            "redone_ids": redone_ids,
            **last_extra,
        }