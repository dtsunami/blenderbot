"""Wires Config, BlenderTransport, Grounding, Ledger, Sandbox, Journal, Agent, and SessionBus."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blenderbot.agent.grounding import build_grounding_text, snapshot_blender_grounding
from blenderbot.agent.journal import Journal
from blenderbot.agent.loop import Agent
from blenderbot.agent.prompt import build_system_prompt, resolve_prompts_layered
from blenderbot.agent.promptfork import check_forks
from blenderbot.agent.sandbox import Sandbox
from blenderbot.config import (
    Config,
    load_config,
    load_settings,
    resolve_home,
    settings_to_defaults,
)
from blenderbot.context.artifacts import ArtifactStore
from blenderbot.context.ledger import Ledger
from blenderbot.context.pastes import PasteStore
from blenderbot.providers import ProviderAdapter, create_provider
from blenderbot.transport import BlenderTransport, TransportConnectionError, connect_blender

from .bus import SessionBus
from .history import PromptHistory
from .log import SessionLog
from .perf import PerfTracker
from .promptedit import apply_staged_prompts
from .resume import find_session_dir, resolve_lineage, restore_ledger_from_lines


@dataclass
class Session:
    cfg: Config
    prompts: dict[str, str]
    prompt_sources: dict[str, str]
    prompt_layers: dict[str, str]
    ledger: Ledger
    sandbox: Sandbox
    journal: Journal
    artifacts: ArtifactStore
    pastes: PasteStore
    history: PromptHistory
    perf: PerfTracker
    log: SessionLog
    agent: Agent
    bus: SessionBus
    transport: BlenderTransport | None
    auto_connect_blender: bool = True
    blender_grounding_snapshot: str = ""
    blender_connect_note: str | None = None
    agents_sections: list[dict[str, Any]] = field(default_factory=list)
    grounding_notices: list[str] = field(default_factory=list)
    pending_prompts: dict[str, Any] | None = None
    web_server: Any = None

    async def maybe_auto_connect_blender(self) -> bool:
        """Opportunistically attach if Blender is available and not yet connected."""
        if self.transport is not None or not self.auto_connect_blender:
            return self.transport is not None
        try:
            await self.reconnect_blender()
            return True
        except Exception as err:  # noqa: BLE001
            self.blender_connect_note = str(err).splitlines()[0]
            return False

    async def reconnect_blender(self, mode: str | None = None) -> dict[str, Any]:
        if self.transport is not None:
            await self.transport.close()
            self.transport = None
            self.agent.transport = None
        eff_mode = mode or self.cfg.transport
        self.transport = await connect_blender(
            mode=eff_mode,
            home=self.cfg.home,
            bridge_host=self.cfg.bridge_host,
            bridge_port=self.cfg.bridge_port,
            snaps_dir=self.journal.snaps_dir,
        )
        self.agent.transport = self.transport
        st = await self.transport.call("blender.status", timeout=8.0)
        self.blender_connect_note = None
        self.bus.publish(
            "blender-changed",
            {"connected": True, "mode": self.transport.mode, "status": st},
        )
        return st

    async def close(self) -> None:
        self.log.update_totals(
            self.agent.stats,
            provider=self.cfg.provider,
            model_id=self.cfg.model.id,
            health=self.agent.context_health(),
        )
        self.log.flight.close("exit")
        self.bus.detach()
        if self.web_server is not None:
            with contextlib.suppress(Exception):
                await self.web_server.close()
            self.web_server = None
        if self.transport is not None:
            await self.transport.close()
            self.transport = None


async def bootstrap_session(
    *,
    cwd: str | Path | None = None,
    overrides: dict[str, str] | None = None,
    resume: str | bool = False,
    accept_stale_prompts: bool = False,
    make_provider_fn: Callable[..., ProviderAdapter] = create_provider,
    injected_transport: BlenderTransport | None = None,
    auto_connect_blender: bool = True,
    home: str | Path | None = None,
) -> Session:
    eff_home = Path(home).resolve() if home else resolve_home()
    resumed_id: str | None = None
    resume_dir: Path | None = None

    lineage: dict[str, Any] | None = None
    if resume:
        resume_dir, meta = find_session_dir(eff_home, resume)
        resumed_id = str(meta.get("id"))
        lineage = resolve_lineage(eff_home, resume_dir, meta)

    defaults = settings_to_defaults(load_settings(eff_home))
    cfg = load_config(cwd=cwd, defaults=defaults, overrides=overrides, home=eff_home)

    resumed_cwd_paths: list[Path] = []
    for cand_raw in (lineage["resumed_cwds"] if lineage else []):
        with contextlib.suppress(OSError):
            cand_p = Path(cand_raw).resolve()
            if cand_p.is_dir():
                if cand_p not in resumed_cwd_paths:
                    resumed_cwd_paths.append(cand_p)
                if cand_p != cfg.cwd and cand_p not in cfg.ro_dirs:
                    cfg.ro_dirs.append(cand_p)

    prompts, prompt_sources, prompt_layers = resolve_prompts_layered(home=cfg.home, cwd=cfg.cwd)
    forks = check_forks(sources=prompt_sources, layers=prompt_layers)
    if forks["stale"] and not accept_stale_prompts:
        s = forks["stale"][0]
        raise ValueError(
            f"Prompt override {s['file']} ({s['key']}) was forked from blenderbot {s['forkedVersion']} "
            f"({s['forkedHash']}) and built-in is now {s['currentVersion']} ({s['currentHash']}). "
            "Run `/prompts accept` or pass `--accept-prompts`."
        )

    log = SessionLog(
        home=cfg.home,
        cwd=cfg.cwd,
        provider=cfg.provider,
        model_id=cfg.model.id,
        resumed_from=resumed_id,
        ro_dirs=cfg.ro_dirs,
        resumed_cwds=resumed_cwd_paths,
        initial_totals=lineage["totals"] if lineage else None,
        initial_transcript_lines=lineage["transcript_lines"] if lineage else None,
    )
    journal = Journal(log.dir, resume_from=resume_dir)

    transport: BlenderTransport | None = injected_transport
    connect_note: str | None = None
    if transport is None and auto_connect_blender:
        try:
            transport = await connect_blender(
                mode=cfg.transport,
                home=cfg.home,
                bridge_host=cfg.bridge_host,
                bridge_port=cfg.bridge_port,
                snaps_dir=journal.snaps_dir,
            )
        except TransportConnectionError as err:
            connect_note = str(err).splitlines()[0]

    sandbox = Sandbox(cfg.cwd, ro_dirs=cfg.ro_dirs)
    blender_snap = await snapshot_blender_grounding(transport)
    grounding_text, agents_sections, notices = build_grounding_text(
        cfg=cfg, sandbox=sandbox, blender_section=blender_snap
    )
    cfg.warnings.extend(notices)

    system_prompt = build_system_prompt(prompts=prompts, grounding=grounding_text, role=cfg.role)
    ledger = Ledger(limit=cfg.context_limit, compact_at=cfg.compact_at)
    ledger.set_system(system_prompt)
    if lineage is not None:
        restore_ledger_from_lines(lineage["transcript_lines"], ledger)

    artifacts = ArtifactStore()
    pastes = PasteStore()
    history = PromptHistory(cfg.home)
    perf = PerfTracker(
        initial_turns=lineage["perf_turns"] if lineage else None,
        turn_offset=int(lineage["totals"]["turns"]) if lineage else 0,
    )

    sess_holder: list[Session] = []

    def _drain_prompts() -> dict[str, Any] | None:
        if sess_holder and sess_holder[0].pending_prompts:
            return apply_staged_prompts(sess_holder[0])
        return None

    agent = Agent(
        cfg=cfg,
        make_provider=make_provider_fn,
        ledger=ledger,
        sandbox=sandbox,
        journal=journal,
        artifacts=artifacts,
        prompts=prompts,
        transport=transport,
        session_log=log,
        perf_tracker=perf,
        drain_prompts_hook=_drain_prompts,
        initial_stats=lineage["totals"] if lineage else None,
    )
    bus = SessionBus(agent=agent)
    bus.state["stats"] = dict(agent.stats)
    if lineage is not None:
        for entry in lineage["web_replay"]:
            bus.publish("transcript", entry)
    journal.on_change = lambda st_payload: bus.publish("journal-updated", st_payload)

    session = Session(
        cfg=cfg,
        prompts=prompts,
        prompt_sources=prompt_sources,
        prompt_layers=prompt_layers,
        ledger=ledger,
        sandbox=sandbox,
        journal=journal,
        artifacts=artifacts,
        pastes=pastes,
        history=history,
        perf=perf,
        log=log,
        agent=agent,
        bus=bus,
        transport=transport,
        auto_connect_blender=auto_connect_blender,
        blender_grounding_snapshot=blender_snap,
        blender_connect_note=connect_note,
        agents_sections=agents_sections,
        grounding_notices=notices,
    )
    sess_holder.append(session)
    return session