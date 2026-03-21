"""Millbrook scenario: 4-6 starting agents, town layout, initial state."""

from __future__ import annotations

from smrti import Smrti

from smrti_town.agent import Agent
from smrti_town.calendar import SimCalendar
from smrti_town.engine import SimEngine
from smrti_town.spatial import TownTopology, build_millbrook_topology


def create_millbrook(
    db_path: str = "~/.smrti/town.db",
    tenant_id: str = "millbrook",
) -> SimEngine:
    """Create the Millbrook scenario with 6 starting agents and full town layout."""

    topology = build_millbrook_topology()

    # ── Initialize World_Space with topology atoms ───────────────────
    world_smrti = Smrti(
        db_path=db_path,
        personality="deterministic",
        tenant_id=tenant_id,
        write_space="World_Space",
    )
    # Store place atoms in world space
    for place_name, place in topology.places.items():
        world_smrti.remember(
            content=f"{place_name} is a location in Millbrook.",
            type="concept",
            probability=1.0,
            valence=0.0,
            metadata={"entity_type": "location", "place": place_name},
        )
        if place.parent:
            world_smrti.remember(
                content=f"{place_name} is part of {place.parent}.",
                type="belief",
                probability=1.0,
                valence=0.0,
                metadata={"relation": "PartOf", "source": place_name, "target": place.parent},
            )
    world_smrti.close()

    # ── Initialize Space_Culture ─────────────────────────────────────
    culture_smrti = Smrti(
        db_path=db_path,
        personality="balanced",
        tenant_id=tenant_id,
        write_space="Space_Culture",
    )
    culture_smrti.remember(
        content="Cafe Rosetta serves excellent coffee and pastries.",
        type="belief",
        probability=0.8,
        valence=0.3,
    )
    culture_smrti.remember(
        content="The Public Library has a large collection of books.",
        type="belief",
        probability=0.9,
        valence=0.2,
    )
    culture_smrti.remember(
        content="The Town Market sells fresh bread and produce.",
        type="belief",
        probability=0.9,
        valence=0.2,
    )
    culture_smrti.remember(
        content="Central Park is a beautiful place to relax.",
        type="belief",
        probability=0.8,
        valence=0.4,
    )
    culture_smrti.close()

    # ── Create starting agents ───────────────────────────────────────
    agents = [
        _create_alice(db_path, tenant_id),
        _create_bob(db_path, tenant_id),
        _create_sofia(db_path, tenant_id),
        _create_marco(db_path, tenant_id),
        _create_elena(db_path, tenant_id),
        _create_yuki(db_path, tenant_id),
    ]

    # Place agents in starting locations
    agent_locations = {
        "Alice": "Alice_Home",
        "Bob": "Alice_Home",
        "Sofia": "Sofia_Home",
        "Marco": "Cafe_Rosetta",
        "Elena": "Public_Library",
        "Yuki": "Central_Park",
    }
    for agent in agents:
        loc = agent_locations.get(agent.name, "Central_Park")
        agent.location = loc
        if loc in topology.places:
            topology.places[loc].add_occupant(agent.name)

    # Pre-seed some relationships (Alice and Bob know each other well)
    alice = agents[0]
    bob = agents[1]
    for _ in range(15):
        alice.increment_interaction("Bob")
        bob.increment_interaction("Alice")

    # Sofia and Elena know each other somewhat
    sofia = agents[2]
    elena = agents[4]
    for _ in range(6):
        sofia.increment_interaction("Elena")
        elena.increment_interaction("Sofia")

    calendar = SimCalendar(total_hours=0.0)

    engine = SimEngine(
        agents=agents,
        topology=topology,
        calendar=calendar,
        db_path=db_path,
        tenant_id=tenant_id,
    )

    return engine


def _create_alice(db_path: str, tenant_id: str) -> Agent:
    """Alice: balanced personality, age 28, lives with Bob."""
    agent = Agent(
        name="Alice",
        personality="balanced",
        location="Alice_Home",
        age_years=28.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I live with Bob at our home on Elm Street.",
        type="belief",
        probability=1.0,
        valence=0.5,
    )
    agent.smrti.remember(
        content="I enjoy spending time at Cafe Rosetta.",
        type="belief",
        probability=0.8,
        valence=0.4,
    )
    agent.smrti.remember(
        content="Bob is my partner and we have been together for years.",
        type="belief",
        probability=1.0,
        valence=0.6,
        metadata={"relation": "romantic", "target": "Bob"},
    )
    return agent


def _create_bob(db_path: str, tenant_id: str) -> Agent:
    """Bob: curious personality, age 30, lives with Alice."""
    agent = Agent(
        name="Bob",
        personality="curious",
        location="Alice_Home",
        age_years=30.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I live with Alice at our home on Elm Street.",
        type="belief",
        probability=1.0,
        valence=0.5,
    )
    agent.smrti.remember(
        content="I love reading books at the Public Library.",
        type="belief",
        probability=0.9,
        valence=0.5,
    )
    agent.smrti.remember(
        content="Alice is my partner and we have been together for years.",
        type="belief",
        probability=1.0,
        valence=0.6,
        metadata={"relation": "romantic", "target": "Alice"},
    )
    return agent


def _create_sofia(db_path: str, tenant_id: str) -> Agent:
    """Sofia: analytical personality, age 35, lives alone."""
    agent = Agent(
        name="Sofia",
        personality="analytical",
        location="Sofia_Home",
        age_years=35.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I live in my home on Elm Street.",
        type="belief",
        probability=1.0,
        valence=0.2,
    )
    agent.smrti.remember(
        content="I work at the Town Market managing accounts.",
        type="belief",
        probability=0.9,
        valence=0.1,
    )
    agent.smrti.remember(
        content="I prefer quiet evenings with a good book.",
        type="belief",
        probability=0.8,
        valence=0.3,
    )
    return agent


def _create_marco(db_path: str, tenant_id: str) -> Agent:
    """Marco: empathetic personality, age 25, works at the cafe."""
    agent = Agent(
        name="Marco",
        personality="empathetic",
        location="Cafe_Rosetta",
        age_years=25.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I work at Cafe Rosetta as a barista.",
        type="belief",
        probability=1.0,
        valence=0.4,
    )
    agent.smrti.remember(
        content="I love meeting new people and hearing their stories.",
        type="belief",
        probability=0.8,
        valence=0.5,
    )
    agent.smrti.remember(
        content="Cafe Rosetta makes the best espresso in town.",
        type="belief",
        probability=0.9,
        valence=0.6,
    )
    return agent


def _create_elena(db_path: str, tenant_id: str) -> Agent:
    """Elena: maverick personality, age 32, the town librarian."""
    agent = Agent(
        name="Elena",
        personality="maverick",
        location="Public_Library",
        age_years=32.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I am the librarian at the Public Library.",
        type="belief",
        probability=1.0,
        valence=0.3,
    )
    agent.smrti.remember(
        content="I know more about the history of Millbrook than anyone.",
        type="belief",
        probability=0.7,
        valence=0.4,
    )
    agent.smrti.remember(
        content="Sofia and I are good friends who often discuss books.",
        type="belief",
        probability=0.8,
        valence=0.4,
        metadata={"relation": "friend", "target": "Sofia"},
    )
    return agent


def _create_yuki(db_path: str, tenant_id: str) -> Agent:
    """Yuki: curious personality, age 22, new in town."""
    agent = Agent(
        name="Yuki",
        personality="curious",
        location="Central_Park",
        age_years=22.0,
        db_path=db_path,
        tenant_id=tenant_id,
    )
    agent.smrti.remember(
        content="I just moved to Millbrook and am exploring the town.",
        type="episode",
        valence=0.3,
    )
    agent.smrti.remember(
        content="I am interested in learning about the people and places here.",
        type="belief",
        probability=0.8,
        valence=0.4,
    )
    agent.smrti.remember(
        content="Central Park is a lovely place to sit and think.",
        type="episode",
        valence=0.5,
    )
    return agent
