"""FastAPI routes for Agent Sessions (list, inspect transcript, new, resume/continue, approvals, journal undo/redo & model/role controls)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from blenderbot.agent.grounding import build_grounding_text, snapshot_blender_grounding
from blenderbot.agent.journal import Journal
from blenderbot.agent.prompt import build_system_prompt
from blenderbot.agent.roles import ROLE_SPECS, ROLES
from blenderbot.context.compact import compact_ledger
from blenderbot.context.utf8 import sanitize_utf8_str
from blenderbot.session.log import SessionLog, list_sessions
from blenderbot.session.models_ctl import apply_model_change, provider_catalog_state
from blenderbot.session.resume import find_session_dir, resolve_lineage, restore_ledger_from_lines

from .routes_common import get_web_state

router = APIRouter()


def _parse_transcript_events(lines: list[str], limit: int = 240) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        et = str(ev.get("type") or "")
        data = ev.get("data") or {}
        ts = ev.get("t")
        if et == "user" and isinstance(data, dict) and data.get("text"):
            events.append({"type": "user", "text": sanitize_utf8_str(str(data["text"])), "t": ts})
        elif et == "steer" and isinstance(data, dict) and data.get("text"):
            events.append({"type": "steer", "text": sanitize_utf8_str(str(data["text"])), "t": ts})
        elif et == "assistant" and isinstance(data, dict) and data.get("text"):
            events.append(
                {
                    "type": "assistant",
                    "text": sanitize_utf8_str(str(data["text"])),
                    "turn": data.get("turn"),
                    "t": ts,
                }
            )
        elif et == "ops-result" and isinstance(data, dict):
            res_list = data.get("results") or []
            ops_names = [str(r.get("op") or "") for r in res_list if isinstance(r, dict) and r.get("op")]
            events.append(
                {
                    "type": "ops",
                    "status": data.get("status", "ok"),
                    "ops": ops_names,
                    "count": len(res_list),
                    "errors": data.get("errors") or [],
                    "t": ts,
                }
            )
        elif et == "shell" and isinstance(data, dict):
            events.append(
                {
                    "type": "shell",
                    "cmd": data.get("cmd", ""),
                    "exit_code": data.get("exit_code", 0),
                    "output": str(data.get("output") or "")[:2000],
                    "t": ts,
                }
            )
    return events[-limit:]


def _extract_first_prompt(session_dir: Path, meta: dict[str, Any]) -> str:
    if meta.get("first_prompt"):
        return str(meta["first_prompt"])
    tr = session_dir / "transcript.jsonl"
    if not tr.is_file():
        return ""
    try:
        with tr.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, ln in enumerate(fh):
                if idx > 60:
                    break
                if '"user"' not in ln:
                    continue
                ev = json.loads(ln)
                if ev.get("type") == "user" and isinstance(ev.get("data"), dict):
                    txt = str(ev["data"].get("text") or "").strip()
                    if txt:
                        return txt[:140]
    except Exception:  # noqa: BLE001, S110
        pass
    return ""


def _serialize_session_summary(meta: dict[str, Any], active_id: str, cur_cwd: Path) -> dict[str, Any]:
    sid = str(meta.get("id") or "")
    sdir = Path(str(meta.get("dir") or ""))
    s_cwd = str(meta.get("cwd") or "")
    same_cwd = False
    if s_cwd:
        try:
            same_cwd = Path(s_cwd).resolve() == cur_cwd.resolve()
        except OSError:
            same_cwd = s_cwd.lower() == str(cur_cwd).lower()

    blend_fp = str(meta.get("blend_filepath") or "")
    blend_name = Path(blend_fp).name if blend_fp else ""
    last_snap = str(meta.get("last_snapshot") or "")
    if not last_snap and sdir.is_dir():
        snaps = sorted((sdir / "snaps").glob("b-*.blend")) if (sdir / "snaps").is_dir() else []
        if snaps:
            last_snap = str(snaps[-1])

    totals = meta.get("totals") or {}
    return {
        "id": sid,
        "is_active": sid == active_id,
        "started_at": meta.get("started_at", ""),
        "cwd": s_cwd,
        "same_cwd": same_cwd,
        "provider": meta.get("provider", ""),
        "model": meta.get("model", ""),
        "resumed_from": meta.get("resumed_from"),
        "blend_filepath": blend_fp,
        "blend_name": blend_name,
        "scene_name": meta.get("scene_name", ""),
        "last_snapshot": last_snap,
        "has_blend_state": bool((blend_fp and Path(blend_fp).is_file()) or (last_snap and Path(last_snap).is_file())),
        "first_prompt": _extract_first_prompt(sdir, meta) if sdir.is_dir() else str(meta.get("first_prompt") or ""),
        "turns": int(totals.get("turns", 0) or 0),
        "ops": int(totals.get("ops", 0) or 0),
        "cost": round(float(totals.get("cost", 0.0) or 0.0), 5),
    }


@router.get("/sessions")
async def api_list_sessions(req: Request, all_workspaces: int = 0) -> dict[str, Any]:
    st = get_web_state(req)
    sess = st.session
    sess.log.update_totals(
        sess.agent.stats,
        provider=sess.cfg.provider,
        model_id=sess.cfg.model.id,
        health=sess.agent.context_health(),
    )
    raw_list = await asyncio.to_thread(list_sessions, sess.cfg.home, 45)
    cur_id = sess.log.id
    cur_cwd = sess.cfg.cwd

    items: list[dict[str, Any]] = []
    seen_active = False
    for m in raw_list:
        row = _serialize_session_summary(m, cur_id, cur_cwd)
        if row["id"] == cur_id:
            seen_active = True
            items.insert(0, row)
        elif all_workspaces or row["same_cwd"] or row["turns"] > 0:
            items.append(row)

    if not seen_active:
        active_meta = {**sess.log.meta, "dir": str(sess.log.dir)}
        items.insert(0, _serialize_session_summary(active_meta, cur_id, cur_cwd))

    return {
        "active_id": cur_id,
        "sessions": items[:35],
        "journal": sess.journal.list_state(),
        "pending_approvals": sess.bus.snapshot()["pending"],
        "catalog": provider_catalog_state(sess),
        "roles": list(ROLES),
        "role_specs": {k: {"title": v.title, "summary": v.summary} for k, v in ROLE_SPECS.items()},
    }


@router.get("/sessions/{session_id}")
async def api_get_session_detail(req: Request, session_id: str) -> JSONResponse:
    st = get_web_state(req)
    sess = st.session
    if session_id in ("active", "current", sess.log.id):
        tr_lines = (
            sess.log.transcript_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if sess.log.transcript_path.is_file()
            else []
        )
        summary = _serialize_session_summary({**sess.log.meta, "dir": str(sess.log.dir)}, sess.log.id, sess.cfg.cwd)
        return JSONResponse(
            {
                "session": summary,
                "events": _parse_transcript_events(tr_lines, limit=300),
                "journal": sess.journal.list_state(),
                "live": sess.bus.snapshot(),
            }
        )

    try:
        s_dir, meta = await asyncio.to_thread(find_session_dir, sess.cfg.home, session_id)
        lineage = await asyncio.to_thread(resolve_lineage, sess.cfg.home, s_dir, meta)
        summary = _serialize_session_summary({**meta, "dir": str(s_dir)}, sess.log.id, sess.cfg.cwd)
        past_journal = Journal(s_dir)
        return JSONResponse(
            {
                "session": summary,
                "events": _parse_transcript_events(lineage["transcript_lines"], limit=300),
                "journal": past_journal.list_state(),
            }
        )
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=404)


@router.post("/sessions/new")
async def api_new_session(req: Request) -> JSONResponse:
    """Start a fresh Agent Session in-place while keeping Blender & Web Studio alive."""
    st = get_web_state(req)
    sess = st.session
    if sess.agent.busy:
        return JSONResponse({"error": "Agent is currently running a turn — cancel first."}, status_code=409)

    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    reset_blend = bool(body.get("reset_blender", False))

    sess.log.update_totals(
        sess.agent.stats,
        provider=sess.cfg.provider,
        model_id=sess.cfg.model.id,
        health=sess.agent.context_health(),
    )

    if reset_blend and sess.transport is not None:
        await sess.transport.call("scene.clear", {}, timeout=15.0)

    new_log = SessionLog(
        home=sess.cfg.home,
        cwd=sess.cfg.cwd,
        provider=sess.cfg.provider,
        model_id=sess.cfg.model.id,
        ro_dirs=sess.cfg.ro_dirs,
    )
    new_journal = Journal(new_log.dir)
    new_journal.on_change = lambda st_payload: sess.bus.publish("journal-updated", st_payload)

    if sess.transport is not None:
        try:
            st_bl = await sess.transport.call("blender.status", {}, timeout=4.0)
            sc_info = st_bl.get("scene") if isinstance(st_bl.get("scene"), dict) else {}
            new_log.update_blend_association(
                blend_filepath=str(st_bl.get("filepath") or ""),
                scene_name=str(sc_info.get("name") or ""),
            )
        except Exception:  # noqa: BLE001, S110
            pass

    blender_snap = await snapshot_blender_grounding(sess.transport)
    grounding_text, _, _ = build_grounding_text(
        cfg=sess.cfg, sandbox=sess.sandbox, blender_section=blender_snap
    )
    sys_prompt = build_system_prompt(
        prompts=sess.prompts, grounding=grounding_text, role=sess.cfg.role
    )
    sess.ledger.clear()
    sess.ledger.set_system(sys_prompt)

    sess.log = new_log
    sess.journal = new_journal
    sess.agent.session_log = new_log
    sess.agent.journal = new_journal
    for k in list(sess.agent.stats.keys()):
        sess.agent.stats[k] = 0.0 if k == "cost" else 0
    sess.bus.state["stats"] = dict(sess.agent.stats)
    sess.bus.publish("session-changed", {"active_id": new_log.id, "mode": "new"})

    summary = _serialize_session_summary({**new_log.meta, "dir": str(new_log.dir)}, new_log.id, sess.cfg.cwd)
    return JSONResponse({"ok": True, "session": summary, "journal": new_journal.list_state()})


@router.post("/sessions/{session_id}/resume")
async def api_resume_session(req: Request, session_id: str) -> JSONResponse:
    """Continue a past session in-place (restores Ledger, Journal undo stack, and optionally opens its `.blend`)."""
    st = get_web_state(req)
    sess = st.session
    if sess.agent.busy:
        return JSONResponse({"error": "Agent is currently running a turn — cancel first."}, status_code=409)

    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    open_associated_blend = bool(body.get("open_blend", False))

    try:
        s_dir, meta = await asyncio.to_thread(find_session_dir, sess.cfg.home, session_id)
        lineage = await asyncio.to_thread(resolve_lineage, sess.cfg.home, s_dir, meta)
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=404)

    sess.log.update_totals(
        sess.agent.stats,
        provider=sess.cfg.provider,
        model_id=sess.cfg.model.id,
        health=sess.agent.context_health(),
    )

    resumed_id = str(meta.get("id") or "")
    new_log = SessionLog(
        home=sess.cfg.home,
        cwd=sess.cfg.cwd,
        provider=sess.cfg.provider,
        model_id=sess.cfg.model.id,
        resumed_from=resumed_id,
        ro_dirs=sess.cfg.ro_dirs,
        initial_totals=lineage["totals"],
        initial_transcript_lines=lineage["transcript_lines"],
    )
    prev_blend = str(meta.get("blend_filepath") or "")
    prev_snap = str(meta.get("last_snapshot") or "")
    new_log.update_blend_association(
        blend_filepath=prev_blend or None,
        scene_name=str(meta.get("scene_name") or "") or None,
        last_snapshot=prev_snap or None,
    )
    if meta.get("first_prompt"):
        new_log.meta["first_prompt"] = meta["first_prompt"]
        new_log._flush_meta()

    new_journal = Journal(new_log.dir, resume_from=s_dir)
    new_journal.on_change = lambda st_payload: sess.bus.publish("journal-updated", st_payload)

    opened_blend_note: str | None = None
    target_blend = (
        Path(prev_blend)
        if prev_blend and Path(prev_blend).is_file()
        else (Path(prev_snap) if prev_snap and Path(prev_snap).is_file() else None)
    )
    if open_associated_blend and target_blend is not None and sess.transport is not None:
        try:
            cp = await new_journal.create_blend_checkpoint(sess.transport)
            await sess.transport.call("scene.open", {"filepath": str(target_blend)}, timeout=60.0)
            new_journal.record_batch_action(
                actor="operator",
                action_type="scene.open",
                summary=f"Restored session scene {target_blend.name}",
                checkpoint=cp,
            )
            opened_blend_note = str(target_blend)
        except Exception:  # noqa: BLE001, S110
            pass

    blender_snap = await snapshot_blender_grounding(sess.transport)
    grounding_text, _, _ = build_grounding_text(
        cfg=sess.cfg, sandbox=sess.sandbox, blender_section=blender_snap
    )
    sys_prompt = build_system_prompt(
        prompts=sess.prompts, grounding=grounding_text, role=sess.cfg.role
    )
    sess.ledger.clear()
    sess.ledger.set_system(sys_prompt)
    restore_ledger_from_lines(lineage["transcript_lines"], sess.ledger)

    sess.log = new_log
    sess.journal = new_journal
    sess.agent.session_log = new_log
    sess.agent.journal = new_journal
    for k, v in lineage["totals"].items():
        if k in sess.agent.stats:
            sess.agent.stats[k] = v
    sess.bus.state["stats"] = dict(sess.agent.stats)
    sess.bus.publish("session-changed", {"active_id": new_log.id, "resumed_from": resumed_id})

    summary = _serialize_session_summary({**new_log.meta, "dir": str(new_log.dir)}, new_log.id, sess.cfg.cwd)
    return JSONResponse(
        {
            "ok": True,
            "session": summary,
            "events": _parse_transcript_events(lineage["transcript_lines"], limit=300),
            "journal": new_journal.list_state(),
            "opened_blend": opened_blend_note,
        }
    )


@router.post("/chat/approve")
async def api_chat_approve(req: Request) -> JSONResponse:
    """Resolve a pending `approval-request` from the Web Studio Session view."""
    st = get_web_state(req)
    body = await req.json()
    ap_id = str(body.get("id") or "").strip()
    allow = bool(body.get("allow", False))
    checkpoint_first = bool(body.get("checkpoint_first", True))
    reason = str(body.get("reason") or "").strip() or None
    if not ap_id:
        return JSONResponse({"error": "Missing approval id"}, status_code=400)
    ok = st.session.bus.resolve_approval(
        ap_id,
        {"allow": allow, "checkpoint_first": checkpoint_first, "reason": reason},
    )
    if not ok:
        return JSONResponse({"error": f"Approval {ap_id} is no longer pending"}, status_code=404)
    return JSONResponse({"ok": True, "id": ap_id, "allow": allow})


@router.post("/journal/undo")
async def api_journal_undo(req: Request) -> JSONResponse:
    st = get_web_state(req)
    sess = st.session
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    entry_id = str(body.get("id") or "").strip()
    steps = int(body.get("steps") or 1)
    try:
        if entry_id:
            res = await sess.journal.undo_entry(entry_id, sess.transport)
        else:
            res = await sess.journal.undo_last(sess.transport, steps=steps)
        return JSONResponse({"ok": True, "result": res, "journal": sess.journal.list_state()})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/journal/redo")
async def api_journal_redo(req: Request) -> JSONResponse:
    st = get_web_state(req)
    sess = st.session
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    entry_id = str(body.get("id") or "").strip()
    steps = int(body.get("steps") or 1)
    try:
        if entry_id:
            res = await sess.journal.redo_entry(entry_id, sess.transport)
        else:
            res = await sess.journal.redo_next(sess.transport, steps=steps)
        return JSONResponse({"ok": True, "result": res, "journal": sess.journal.list_state()})
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/session/settings")
async def api_update_session_settings(req: Request) -> JSONResponse:
    """Switch Agent model, provider, reasoning effort, capability role, or compact context."""
    st = get_web_state(req)
    sess = st.session
    body = await req.json()
    action = str(body.get("action") or "").strip()
    try:
        if action == "compact":
            res_c = await compact_ledger(
                ledger=sess.ledger,
                make_provider=sess.agent.make_provider,
                cfg=sess.cfg,
            )
            return JSONResponse({"ok": True, "compacted": res_c})

        if body.get("role"):
            want_role = str(body["role"]).strip().lower()
            if want_role not in ROLES:
                return JSONResponse({"error": f"Unknown role: {want_role}"}, status_code=400)
            sess.cfg.role = want_role
            sess.agent.cfg.role = want_role
            blender_snap = await snapshot_blender_grounding(sess.transport)
            grounding_text, _, _ = build_grounding_text(
                cfg=sess.cfg, sandbox=sess.sandbox, blender_section=blender_snap
            )
            sess.ledger.set_system(
                build_system_prompt(prompts=sess.prompts, grounding=grounding_text, role=want_role)
            )

        if body.get("model") or body.get("provider") or body.get("effort"):
            apply_model_change(
                sess,
                provider=body.get("provider"),
                model_id=body.get("model"),
                effort=body.get("effort"),
                persist=bool(body.get("persist", False)),
            )

        if "yolo" in body:
            sess.cfg.yolo = bool(body["yolo"])
            sess.agent.cfg.yolo = sess.cfg.yolo

        return JSONResponse(
            {
                "ok": True,
                "provider": sess.cfg.provider,
                "model": sess.cfg.model.id,
                "effort": sess.cfg.effort,
                "role": sess.cfg.role,
                "yolo": sess.cfg.yolo,
            }
        )
    except Exception as err:  # noqa: BLE001
        return JSONResponse({"error": str(err)}, status_code=400)


@router.post("/context/{seg_id}/pin")
async def api_toggle_pin_segment(req: Request, seg_id: int) -> JSONResponse:
    st = get_web_state(req)
    pinned = st.session.ledger.toggle_pin(seg_id)
    return JSONResponse({"ok": True, "id": seg_id, "pinned": pinned})