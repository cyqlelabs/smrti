"""SimEngine: the main simulation engine with the 8-phase tick loop."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from smrti_town.llm import LLMClient

from smrti import Smrti

from smrti_town.agent import Action, Agent
from smrti_town.calendar import SimCalendar
from smrti_town.config import (
    ACTION_EAT,
    ACTION_MOVE,
    ACTION_REPRODUCE,
    ACTION_SLEEP,
    ACTION_STUDY,
    ACTION_TALK,
    ACTION_WAIT,
    ACTION_WANDER,
    ACTION_WORK,
    EPOCH_INTERVAL_HOURS,
    HOURS_PER_YEAR,
    TICK_SKIP,
)
from smrti_town.culture import promote_bridges_to_culture, run_bridge_discovery
from smrti_town.director import Chronos, Director, SystemEvent
from smrti_town.drives import _clamp
from smrti_town.lifecycle import (
    apply_relationship_transition,
    archive_agent,
    check_death,
    check_relationship_gates,
    check_reproduction_gate,
    compute_life_stage,
    spawn_child,
)
from smrti_town.extractor import fire_extraction
from smrti_town.narrator import Narrator
from smrti_town.spatial import TownTopology
from smrti_town.sporadic import (
    SporadicEvent,
    apply_sporadic_effects,
    generate_sporadic_events,
)

logger = logging.getLogger("smrti_town.engine")


@dataclass
class TickResult:
    tick_number: int
    calendar: dict
    director_mode: str
    delta_hours: float
    agents: list[dict]
    events: list[dict]
    conversations: list[dict]
    births: list[dict] = field(default_factory=list)
    deaths: list[str] = field(default_factory=list)
    relationship_changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": "tick",
            "tick_number": self.tick_number,
            "calendar": self.calendar,
            "director_mode": self.director_mode,
            "delta_hours": self.delta_hours,
            "agents": self.agents,
            "events": self.events,
            "conversations": self.conversations,
            "births": self.births,
            "deaths": self.deaths,
            "relationship_changes": self.relationship_changes,
        }


class SimEngine:
    """Owns the tick loop, action resolution, and event broadcasting."""

    def __init__(
        self,
        agents: list[Agent],
        topology: TownTopology,
        calendar: SimCalendar | None = None,
        db_path: str = "~/.smrti/town.db",
        tenant_id: str = "millbrook",
        llm_client: "LLMClient | None" = None,
    ) -> None:
        self.agents = agents
        self.topology = topology
        self.calendar = calendar or SimCalendar()
        self.db_path = db_path
        self.tenant_id = tenant_id

        self.llm_client = llm_client

        self.director = Director()
        self.chronos = Chronos()
        self.narrator = Narrator()

        self.tick_number: int = 0
        self.running: bool = False
        self.paused: bool = False
        self._last_epoch_hours: float = 0.0
        self._epoch_count: int = 0

        # WebSocket broadcast callback
        self._broadcast: Callable[[dict], Any] | None = None

        # Strong references to fire-and-forget background tasks so GC cannot
        # cancel them mid-execution (asyncio only holds weak references).
        self._bg_tasks: set[asyncio.Task] = set()

        # Place smrti instances for socially significant places
        self._place_smrtis: dict[str, Smrti] = {}
        for place in self.topology.places.values():
            if place.has_space:
                self._place_smrtis[place.name] = Smrti(
                    db_path=db_path,
                    personality=place.personality,
                    tenant_id=tenant_id,
                    write_space=place.space_name,
                    read_spaces=[place.space_name, "World_Space", "Space_Culture"],
                )

        # Agents by name for fast lookup
        self._agents_by_name: dict[str, Agent] = {a.name: a for a in agents}

    def set_broadcast(self, callback: Callable[[dict], Any]) -> None:
        self._broadcast = callback

    # ── Main tick loop ───────────────────────────────────────────────

    async def run(self, max_ticks: int = 0) -> None:
        """Run the simulation loop. max_ticks=0 means run until stopped."""
        self.running = True
        self.paused = False
        ticks_run = 0

        while self.running:
            if self.paused:
                await asyncio.sleep(0.05)
                continue

            result = await self.tick()

            if self._broadcast:
                try:
                    await self._broadcast(result.to_dict())
                except Exception as exc:
                    logger.warning("Broadcast error: %s", exc)

            ticks_run += 1
            if max_ticks and ticks_run >= max_ticks:
                break

            # Yield control so WebSocket and HTTP can process
            await asyncio.sleep(0.01)

        self.running = False

    async def tick(self) -> TickResult:
        """Execute one full 8-phase tick."""
        self.tick_number += 1
        events: list[dict] = []
        conversations: list[dict] = []
        births: list[dict] = []
        deaths: list[str] = []
        relationship_changes: list[str] = []

        alive_agents = [a for a in self.agents if a.alive]

        # ── Phase 0: Director + Chronos ──────────────────────────────
        delta = self.director.compute_tick_delta(alive_agents, self.topology.places)
        self.calendar.advance(delta)

        milestone_events = self.chronos.check_milestones(alive_agents, self.calendar)
        birthday_events = self.chronos.check_birthdays(alive_agents, self.calendar, delta)
        for evt in milestone_events + birthday_events:
            events.append(evt.to_dict())
            self._inject_milestone_memory(evt)

        # ── Phase 1: Python state update ─────────────────────────────
        for agent in alive_agents:
            agent.age_hours += delta
            self._update_drives(agent, delta)

            # Death checks
            if check_death(agent, self.calendar, delta):
                death_narratives = archive_agent(agent, self.agents)
                deaths.append(agent.name)
                for narrative in death_narratives:
                    events.append({
                        "type": "death",
                        "description": narrative,
                        "agent": agent.name,
                    })

        # Refresh alive list after deaths
        alive_agents = [a for a in self.agents if a.alive]

        # ── Phase 2: Perception ──────────────────────────────────────
        agent_contexts: dict[str, Any] = {}
        place_agents: dict[str, list[str]] = {}
        for place_name, place in self.topology.places.items():
            place_agents[place_name] = [
                n for n in place.occupants
                if n in self._agents_by_name and self._agents_by_name[n].alive
            ]

        for agent in alive_agents:
            if not agent.can_talk and not agent.can_move:
                continue
            nearby = [
                n for n in place_agents.get(agent.location, [])
                if n != agent.name
            ]
            # Update read_spaces based on current location
            self._update_agent_read_spaces(agent)

            ctx = agent.perceive(
                nearby_agents=nearby,
                time_of_day=self.calendar.time_of_day(),
                season=self.calendar.season,
                calendar_hour=self.calendar.hour_of_day,
            )
            agent_contexts[agent.name] = ctx

        # ── Phase 3: Decision ────────────────────────────────────────
        agent_actions: dict[str, Action] = {}
        available_places = self.topology.all_place_names()
        place_types = {n: p.place_type for n, p in self.topology.places.items()}

        for agent in alive_agents:
            if agent.name not in agent_contexts:
                agent_actions[agent.name] = Action(type=ACTION_WAIT)
                continue
            ctx = agent_contexts[agent.name]
            action = agent.decide(ctx, available_places, place_agents, place_types)
            agent_actions[agent.name] = action

        # ── Phase 3.5: LLM dialogue enrichment (fire-and-forget) ─────
        # Tasks run in the background; the tick is never blocked by LLM latency.
        if self.llm_client and self.llm_client.settings.enabled:
            await self._enrich_dialogue_llm(
                alive_agents, agent_actions, agent_contexts, self.tick_number
            )

        # ── Phase 4: Engine resolution ───────────────────────────────
        for agent in alive_agents:
            action = agent_actions.get(agent.name)
            if not action:
                continue
            self._resolve_action(agent, action)

            # Check for reproduction
            if (
                action.type == ACTION_TALK
                and action.target
                and agent.drives.romance >= 40
            ):
                target_agent = self._agents_by_name.get(action.target)
                if target_agent and check_reproduction_gate(
                    agent, target_agent, self.calendar, len(alive_agents)
                ):
                    # Small probability per interaction when gate is met
                    import random
                    if random.random() < 0.02:
                        child = spawn_child(
                            agent, target_agent, self.agents, self.db_path, self.tenant_id
                        )
                        self.agents.append(child)
                        self._agents_by_name[child.name] = child
                        # Place child at parent's location
                        if agent.location in self.topology.places:
                            self.topology.places[agent.location].add_occupant(child.name)
                        births.append({
                            "child": child.name,
                            "parent_a": agent.name,
                            "parent_b": target_agent.name,
                        })
                        birth_text = self.narrator.narrate_birth(
                            child.name, agent.name, target_agent.name
                        )
                        events.append({
                            "type": "birth",
                            "description": birth_text,
                            "child": child.name,
                        })

        # ── Phase 5: Narrative remember() ────────────────────────────
        for agent in alive_agents:
            action = agent_actions.get(agent.name)
            if not action or action.type == ACTION_WAIT:
                continue
            narrative = self.narrator.narrate_action(agent, action)
            if narrative:
                valence = self._estimate_action_valence(action)
                try:
                    ep_id = agent.smrti.remember(
                        content=narrative,
                        type="episode",
                        valence=valence,
                        metadata={"tick": self.tick_number, "action": action.type},
                    )
                    if action.type == ACTION_TALK:
                        fire_extraction(ep_id, narrative, agent.smrti, self.llm_client, self._bg_tasks)
                except Exception:
                    pass

        # ── Phase 6: Conversation propagation ────────────────────────
        for agent in alive_agents:
            action = agent_actions.get(agent.name)
            if not action or action.type != ACTION_TALK or not action.target:
                continue
            if not action.dialogue:
                continue

            target_agent = self._agents_by_name.get(action.target)
            if not target_agent or not target_agent.alive:
                continue

            # Track interactions
            agent.increment_interaction(action.target)
            target_agent.increment_interaction(agent.name)

            narration = self.narrator.narrate_conversation(
                speaker=agent.name,
                listener=action.target,
                location=agent.location,
                content=action.dialogue,
            )

            # Write to place space
            place_smrti = self._place_smrtis.get(agent.location)
            if place_smrti:
                try:
                    place_ep_id = place_smrti.remember(
                        content=narration["place"],
                        type="episode",
                        valence=0.1,
                    )
                    fire_extraction(place_ep_id, narration["place"], place_smrti, self.llm_client, self._bg_tasks)
                except Exception:
                    pass

            # Epistemic copy to listener
            try:
                listener_ep_id = target_agent.smrti.remember(
                    content=narration["listener"],
                    type="episode",
                    valence=0.1,
                    metadata={"speaker": agent.name, "location": agent.location},
                )
                fire_extraction(listener_ep_id, narration["listener"], target_agent.smrti, self.llm_client, self._bg_tasks)
            except Exception:
                pass

            conversations.append({
                "speaker": agent.name,
                "listener": action.target,
                "location": agent.location,
                "content": action.dialogue,
            })

        # ── Sporadic events ──────────────────────────────────────────
        sporadic_events = generate_sporadic_events(
            alive_agents, self.topology, delta, self.calendar.season
        )
        for sevt in sporadic_events:
            sporadic_ep_ids = apply_sporadic_effects(sevt, self._agents_by_name)
            for agent_name, ep_id in sporadic_ep_ids.items():
                agent = self._agents_by_name.get(agent_name)
                if agent and agent.alive:
                    fire_extraction(ep_id, sevt.description, agent.smrti, self.llm_client, self._bg_tasks)
            events.append(sevt.to_dict())

        # ── Phase 7: Epoch (periodic) ────────────────────────────────
        hours_since_epoch = self.calendar.total_hours - self._last_epoch_hours
        if hours_since_epoch >= EPOCH_INTERVAL_HOURS or delta >= TICK_SKIP:
            self._run_epoch()
            self._last_epoch_hours = self.calendar.total_hours

            # Relationship gate checks (deduplicate pairs)
            seen_pairs: set[tuple[str, str]] = set()
            for agent in alive_agents:
                transitions = check_relationship_gates(agent, self.agents)
                for t in transitions:
                    pair = tuple(sorted([t.agent_name, t.target_name]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    narratives = apply_relationship_transition(t, self._agents_by_name)
                    relationship_changes.extend(narratives)
                    events.append({
                        "type": "relationship",
                        "description": t.detail,
                        "agents": [t.agent_name, t.target_name],
                    })

        # ── Build result ─────────────────────────────────────────────
        agent_dicts = []
        for agent in self.agents:
            action = agent_actions.get(agent.name)
            d = agent.to_dict()
            if action:
                d["action"] = action.type
                d["action_target"] = action.target
                d["dialogue"] = action.dialogue
            agent_dicts.append(d)

        return TickResult(
            tick_number=self.tick_number,
            calendar=self.calendar.to_dict(),
            director_mode=self.director.mode,
            delta_hours=delta,
            agents=agent_dicts,
            events=events,
            conversations=conversations,
            births=births,
            deaths=deaths,
            relationship_changes=relationship_changes,
        )

    # ── Action resolution ────────────────────────────────────────────

    def _resolve_action(self, agent: Agent, action: Action) -> None:
        """Apply action effects to the simulation state."""
        if action.type == ACTION_MOVE or action.type == ACTION_WANDER:
            target = action.target
            if target and target in self.topology.places:
                self.topology.move_agent(agent.name, agent.location, target)
                agent.location = target

        elif action.type == ACTION_EAT:
            agent.drives.reset_hunger()

        elif action.type == ACTION_SLEEP:
            agent.drives.reset_energy()

        elif action.type == ACTION_WORK:
            agent.drives.reset_duty()

        elif action.type == ACTION_STUDY:
            agent.drives.reduce_curiosity()

        elif action.type == ACTION_TALK:
            agent.drives.reduce_social()
            if agent.drives.romance >= 40:
                agent.drives.reduce_romance()
            if action.target:
                target_agent = self._agents_by_name.get(action.target)
                if target_agent:
                    target_agent.drives.reduce_social()

    # ── Drive update ─────────────────────────────────────────────────

    def _update_drives(self, agent: Agent, delta: float) -> None:
        is_work = self.calendar.is_work_hours()
        is_adult = agent.life_stage in ("adult",)
        agent.drives.accumulate(
            delta,
            is_work_hours=is_work,
            is_adult=is_adult,
            energy_decay_mult=agent.energy_decay_mult,
            active_drives=agent.active_drives,
        )

    # ── Read spaces update ───────────────────────────────────────────

    def _update_agent_read_spaces(self, agent: Agent) -> None:
        """Update agent's read_spaces based on current location."""
        spaces = [
            f"Agent_Space_{agent.name}",
            "World_Space",
            "Space_Culture",
        ]
        # Add current place space
        place = self.topology.places.get(agent.location)
        if place and place.has_space:
            spaces.append(place.space_name)
        agent.smrti.read_spaces = spaces

    # ── Milestone injection ──────────────────────────────────────────

    def _inject_milestone_memory(self, event: SystemEvent) -> None:
        agent = self._agents_by_name.get(event.agent_name)
        if not agent or not agent.alive:
            return
        try:
            agent.smrti.remember(
                content=event.detail,
                type="episode",
                valence=0.4,
                metadata={"event_type": event.event_type},
            )
        except Exception:
            pass

    # ── Epoch ────────────────────────────────────────────────────────

    def _run_epoch(self) -> None:
        """Run epoch consolidation on all active spaces."""
        self._epoch_count += 1

        # Run reflect on alive agent spaces + persist interaction counts
        for agent in self.agents:
            if not agent.alive:
                continue
            try:
                agent.smrti.reflect()
            except Exception:
                pass
            agent.persist_interactions()

        # Run reflect on occupied place spaces
        for place_name, place_smrti in self._place_smrtis.items():
            place = self.topology.places.get(place_name)
            if place and place.occupants:
                try:
                    place_smrti.reflect()
                except Exception:
                    pass

        # Bridge discovery every 10th epoch
        if self._epoch_count % 10 == 0:
            agent_spaces = [
                f"Agent_Space_{a.name}" for a in self.agents if a.alive
            ]
            try:
                bridges = run_bridge_discovery(
                    self.tenant_id, self.db_path, agent_spaces
                )
                if bridges > 0:
                    all_spaces_list = self._get_all_spaces()
                    promote_bridges_to_culture(
                        self.tenant_id, self.db_path, all_spaces_list
                    )
            except Exception:
                pass

    def _get_all_spaces(self) -> list[str]:
        """Get all spaces for this tenant from the DB."""
        try:
            smrti = Smrti(
                db_path=self.db_path,
                personality="balanced",
                tenant_id=self.tenant_id,
                write_space="World_Space",
            )
            spaces = smrti.list_spaces()
            smrti.close()
            return spaces
        except Exception:
            return []

    # ── Valence estimation ───────────────────────────────────────────

    def _estimate_action_valence(self, action: Action) -> float:
        """Estimate emotional valence of an action."""
        valence_map = {
            ACTION_TALK: 0.2,
            ACTION_EAT: 0.3,
            ACTION_SLEEP: 0.1,
            ACTION_WORK: 0.0,
            ACTION_STUDY: 0.1,
            ACTION_MOVE: 0.0,
            ACTION_WANDER: 0.1,
            ACTION_WAIT: 0.0,
            ACTION_REPRODUCE: 0.7,
        }
        return valence_map.get(action.type, 0.0)

    # ── LLM dialogue enrichment ──────────────────────────────────────

    async def _enrich_dialogue_llm(
        self,
        alive_agents: list[Agent],
        agent_actions: dict[str, Action],
        agent_contexts: dict[str, Any],
        tick_number: int,
    ) -> None:
        """Fire-and-forget LLM dialogue generation for all TALK actions.

        The tick is NOT blocked — each agent gets a background asyncio.Task.
        When the model responds (which may take many seconds on slow local
        models), a ``dialogue_patch`` message is broadcast to connected clients
        so the event log updates live.  Template dialogue in the tick result is
        always shown immediately as a fallback.
        """
        for agent in alive_agents:
            action = agent_actions.get(agent.name)
            if (
                not action
                or action.type != ACTION_TALK
                or not action.target
                or agent.name not in agent_contexts
            ):
                continue
            ctx = agent_contexts[agent.name]
            # Capture loop variables before launching task
            task = asyncio.create_task(
                self._generate_and_broadcast_dialogue(
                    agent=agent,
                    target=action.target,
                    fallback=action.dialogue,
                    ctx=ctx,
                    tick_number=tick_number,
                )
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)

    async def _generate_and_broadcast_dialogue(
        self,
        agent: Agent,
        target: str,
        fallback: str,
        ctx: Any,
        tick_number: int,
    ) -> None:
        """Background task: call LLM, then broadcast patch to clients."""
        if not self.llm_client:
            return
        enriched = await self.llm_client.generate_dialogue(
            speaker=agent.name,
            target=target,
            location=agent.location,
            time_of_day=ctx.time_of_day,
            season=ctx.season,
            personality=agent.personality_preset,
            urgent_drive=ctx.urgent_drive,
            memories=ctx.memories,
            fallback=fallback,
        )
        if enriched == fallback:
            return  # LLM failed or returned same text — nothing to patch

        if self._broadcast:
            try:
                await self._broadcast({
                    "type": "dialogue_patch",
                    "tick": tick_number,
                    "speaker": agent.name,
                    "target": target,
                    "content": enriched,
                })
            except Exception as exc:
                logger.debug("Broadcast dialogue_patch failed: %s", exc)

    # ── Control ──────────────────────────────────────────────────────

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.running = False

    def skip_week(self) -> None:
        self.director.request_skip()

    # ── State queries ────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "tick_number": self.tick_number,
            "calendar": self.calendar.to_dict(),
            "director_mode": self.director.mode,
            "running": self.running,
            "paused": self.paused,
            "agent_count": len(self.agents),
            "alive_count": sum(1 for a in self.agents if a.alive),
            "places": {
                name: place.to_dict()
                for name, place in self.topology.places.items()
            },
        }

    def get_agent(self, name: str) -> dict | None:
        agent = self._agents_by_name.get(name)
        if not agent:
            return None
        return agent.to_dict()

    def get_agent_memories(self, name: str, query: str = "", top_k: int = 10) -> list[dict]:
        agent = self._agents_by_name.get(name)
        if not agent:
            return []
        q = query or f"What has happened to {name} recently?"
        try:
            results = agent.smrti.recall(query=q, top_k=top_k)
            return [
                {
                    "content": r.atom.content or r.atom.label,
                    "salience": round(r.salience, 3),
                    "type": r.atom.type.value,
                    "valence": r.atom.valence.valence if r.atom.valence else 0.0,
                }
                for r in results
            ]
        except Exception:
            return []
