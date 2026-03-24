"""Millbrook founding scenario: Town Hall, 3 settlers, everything else built via petition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smrti import Smrti

from smrti_town.agent import Agent
from smrti_town.calendar import SimCalendar
from smrti_town.engine import SimEngine
from smrti_town.spatial import TownTopology, build_millbrook_topology

if TYPE_CHECKING:
    from smrti_town.llm import LLMClient


def create_millbrook(
    db_path: str = "~/.smrti/town.db",
    tenant_id: str = "millbrook",
    llm_client: "LLMClient | None" = None,
) -> SimEngine:
    """Create the founding scenario: 3 settlers at Town Hall, nothing else yet."""

    topology = build_millbrook_topology()

    # ── World_Space: founding story only ────────────────────────────
    world_smrti = Smrti(
        db_path=db_path,
        personality="deterministic",
        tenant_id=tenant_id,
        write_space="World_Space",
    )
    world_smrti.remember(
        content="A small group of settlers has gathered at the Town Hall to found a new community.",
        type="concept",
        probability=1.0,
        valence=0.4,
    )
    world_smrti.remember(
        content="Town_Hall is the only building. The settlers must work together to build the rest.",
        type="belief",
        probability=1.0,
        valence=0.1,
        metadata={"entity_type": "location", "place": "Town_Hall"},
    )
    world_smrti.remember(
        content="The town needs food, shelter, and community before it can grow.",
        type="belief",
        probability=1.0,
        valence=0.0,
    )
    world_smrti.close()

    # ── 3 founding settlers ──────────────────────────────────────────
    agents = [
        _create_alice(db_path, tenant_id),
        _create_bob(db_path, tenant_id),
        _create_marco(db_path, tenant_id),
    ]

    for agent in agents:
        agent.location = "Town_Hall"
        topology.places["Town_Hall"].add_occupant(agent.name)

    # Alice and Bob already know each other
    alice, bob = agents[0], agents[1]
    for _ in range(8):
        alice.increment_interaction("Bob")
        bob.increment_interaction("Alice")

    calendar = SimCalendar(total_hours=0.0)

    engine = SimEngine(
        agents=agents,
        topology=topology,
        calendar=calendar,
        db_path=db_path,
        tenant_id=tenant_id,
        llm_client=llm_client,
    )

    return engine


def _create_alice(db_path: str, tenant_id: str) -> Agent:
    """Alice: balanced, 28 — the de-facto organiser of the founding group."""
    agent = Agent(
        name="Alice",
        personality="balanced",
        location="Town_Hall",
        age_years=28.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I helped organise the journey here. We need to build something lasting.",
        type="belief",
        probability=1.0,
        valence=0.4,
    )
    agent.smrti.remember(
        content="Bob came with me. I trust him completely.",
        type="belief",
        probability=1.0,
        valence=0.6,
        metadata={"relation": "friend", "target": "Bob"},
    )
    agent.smrti.remember(
        content="We have no homes yet. Finding shelter is urgent.",
        type="belief",
        probability=1.0,
        valence=-0.2,
    )
    return agent


def _create_bob(db_path: str, tenant_id: str) -> Agent:
    """Bob: curious, 30 — builder and thinker."""
    agent = Agent(
        name="Bob",
        personality="curious",
        location="Town_Hall",
        age_years=30.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I came here to build something new. The land is good.",
        type="belief",
        probability=1.0,
        valence=0.4,
    )
    agent.smrti.remember(
        content="Alice and I planned this together. We need a farm before winter.",
        type="belief",
        probability=0.9,
        valence=0.2,
        metadata={"relation": "friend", "target": "Alice"},
    )
    agent.smrti.remember(
        content="We are hungry. Food must come first.",
        type="belief",
        probability=1.0,
        valence=-0.3,
    )
    return agent


def _create_marco(db_path: str, tenant_id: str) -> Agent:
    """Marco: empathetic, 25 — the community spirit of the group."""
    agent = Agent(
        name="Marco",
        personality="empathetic",
        location="Town_Hall",
        age_years=25.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I joined these settlers because I believe we can build a great community.",
        type="belief",
        probability=0.9,
        valence=0.5,
    )
    agent.smrti.remember(
        content="People need a place to gather and talk. Community is everything.",
        type="belief",
        probability=0.8,
        valence=0.4,
    )
    agent.smrti.remember(
        content="We need food and a place to sleep before we can think about anything else.",
        type="belief",
        probability=0.9,
        valence=-0.1,
    )
    return agent
