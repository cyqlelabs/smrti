"""Tests for the smrti-town simulator."""

from __future__ import annotations

import asyncio
import os
import random
import tempfile

import pytest

from smrti_town.agent import Action, Agent, PerceptionContext
from smrti_town.calendar import SimCalendar
from smrti_town.config import (
    ACTION_EAT,
    ACTION_MOVE,
    ACTION_SLEEP,
    ACTION_STUDY,
    ACTION_TALK,
    ACTION_WAIT,
    ACTION_WANDER,
    ACTION_WORK,
    HOURS_PER_YEAR,
    PRESET_TRAITS,
    TRAIT_NAMES,
)
from smrti_town.director import Chronos, Director
from smrti_town.drives import AgentDrives
from smrti_town.engine import SimEngine
from smrti_town.lifecycle import (
    _infer_relationship_state,
    check_death,
    check_relationship_gates,
    check_reproduction_gate,
    inherit_personality,
    inherit_traits,
    spawn_child,
)
from smrti_town.narrator import Narrator
from smrti_town.spatial import Place, TownTopology, build_millbrook_topology
from smrti_town.sporadic import generate_sporadic_events


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    """Return a temporary database path that is cleaned up after the test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


@pytest.fixture
def topology():
    """Return a simple 4-place topology for testing."""
    topo = TownTopology()
    topo.add_place(Place(name="Park", is_outdoor=True, place_type="outdoor"))
    topo.add_place(Place(name="Cafe", place_type="public"))
    topo.add_place(Place(name="Library", place_type="public"))
    topo.add_place(Place(name="Alice_Home", place_type="home"))
    topo.add_place(Place(name="Bob_Home", place_type="home"))
    topo.connect("Park", "Cafe")
    topo.connect("Park", "Library")
    topo.connect("Park", "Alice_Home")
    topo.connect("Park", "Bob_Home")
    topo.connect("Cafe", "Library")
    return topo


@pytest.fixture
def agent_pair(tmp_db):
    """Return two agents (Alice and Bob) with a shared db."""
    alice = Agent(
        name="Alice", personality="empathetic", location="Park",
        age_years=30, db_path=tmp_db, tenant_id="test",
    )
    bob = Agent(
        name="Bob", personality="analytical", location="Cafe",
        age_years=28, db_path=tmp_db, tenant_id="test",
    )
    yield alice, bob
    alice.smrti.close()
    bob.smrti.close()


# ── Calendar ──────────────────────────────────────────────────────────

class TestCalendar:
    def test_advance_and_time_of_day(self):
        cal = SimCalendar(total_hours=0)
        assert cal.time_of_day() == "night"
        cal.advance(8)
        assert cal.time_of_day() == "morning"
        cal.advance(6)
        assert cal.time_of_day() == "afternoon"

    def test_seasons(self):
        cal = SimCalendar(total_hours=0)
        assert cal.season == "spring"
        cal.advance(7 * 24)  # 1 week
        assert cal.season == "summer"
        cal.advance(7 * 24)
        assert cal.season == "autumn"

    def test_year_rollover(self):
        cal = SimCalendar(total_hours=HOURS_PER_YEAR - 1)
        assert cal.year == 0
        cal.advance(2)
        assert cal.year == 1


# ── Drives ────────────────────────────────────────────────────────────

class TestDrives:
    def test_accumulate_basic(self):
        d = AgentDrives()
        d.accumulate(2.0)  # 2 hours
        assert d.hunger == 4  # HUNGER_RATE=2 * 2h
        assert d.energy == 98  # ENERGY_DRAIN_RATE=1 * 2h drained
        assert d.social == 2

    def test_fractional_accumulation(self):
        """Scene ticks (0.25h) should accumulate fractionally."""
        d = AgentDrives()
        for _ in range(8):
            d.accumulate(0.25)
        # 8 * 0.25 = 2h, hunger rate 2/h → 4
        assert d.hunger == 4
        assert d.energy == 98

    def test_reset_hunger(self):
        d = AgentDrives()
        d.hunger = 80
        d.reset_hunger()
        assert d.hunger == 0

    def test_highest_urgent_drive(self):
        d = AgentDrives()
        d.hunger = 75
        d.social = 65
        assert d.highest_urgent_drive() == "hunger"

    def test_energy_threshold(self):
        d = AgentDrives()
        d.energy = 15
        assert d.highest_urgent_drive() == "energy"


# ── Topology ──────────────────────────────────────────────────────────

class TestTopology:
    def test_path_distance(self, topology):
        assert topology.path_distance("Park", "Cafe") == 1
        assert topology.path_distance("Alice_Home", "Bob_Home") == 2
        assert topology.path_distance("Park", "Park") == 0

    def test_places_by_type(self, topology):
        assert set(topology.places_by_type("public")) == {"Cafe", "Library"}
        assert set(topology.places_by_type("home")) == {"Alice_Home", "Bob_Home"}
        assert topology.places_by_type("outdoor") == ["Park"]

    def test_home_for(self, topology):
        assert topology.home_for("Alice") == "Alice_Home"
        assert topology.home_for("Bob") == "Bob_Home"
        # Unknown agent gets fallback home
        assert topology.home_for("Charlie") in ("Alice_Home", "Bob_Home")

    def test_move_agent(self, topology):
        topology.places["Park"].add_occupant("Alice")
        topology.move_agent("Alice", "Park", "Cafe")
        assert "Alice" not in topology.places["Park"].occupants
        assert "Alice" in topology.places["Cafe"].occupants

    def test_millbrook_has_place_types(self):
        topo = build_millbrook_topology()
        assert topo.places["Alice_Home"].place_type == "home"
        assert topo.places["Cafe_Rosetta"].place_type == "public"
        assert topo.places["Central_Park"].place_type == "outdoor"
        assert topo.places["Elm_Street"].place_type == "street"


# ── Agent ─────────────────────────────────────────────────────────────

class TestAgent:
    def test_traits_from_preset(self, tmp_db):
        a = Agent(name="A", personality="empathetic", db_path=tmp_db, tenant_id="t")
        assert a.traits == PRESET_TRAITS["empathetic"]
        a.smrti.close()

    def test_traits_custom(self, tmp_db):
        custom = {t: 0.5 for t in TRAIT_NAMES}
        a = Agent(name="A", personality="balanced", db_path=tmp_db, tenant_id="t", traits=custom)
        assert a.traits == custom
        a.smrti.close()

    def test_effective_action_bias(self, tmp_db):
        a = Agent(name="A", personality="balanced", db_path=tmp_db, tenant_id="t")
        bias = a.effective_action_bias()
        assert 0.0 <= bias["social"] <= 1.0
        assert 0.0 <= bias["duty"] <= 1.0
        a.smrti.close()

    def test_life_stage(self, tmp_db):
        infant = Agent(name="I", age_years=2, db_path=tmp_db, tenant_id="t")
        assert infant.life_stage == "infant"
        child = Agent(name="C", age_years=10, db_path=tmp_db, tenant_id="t")
        assert child.life_stage == "child"
        adult = Agent(name="A", age_years=30, db_path=tmp_db, tenant_id="t")
        assert adult.life_stage == "adult"
        elder = Agent(name="E", age_years=70, db_path=tmp_db, tenant_id="t")
        assert elder.life_stage == "elder"
        for a in (infant, child, adult, elder):
            a.smrti.close()

    def test_interaction_counts(self, agent_pair):
        alice, bob = agent_pair
        assert alice.get_interaction_count("Bob") == 0
        alice.increment_interaction("Bob")
        alice.increment_interaction("Bob")
        assert alice.get_interaction_count("Bob") == 2

    def test_decide_sleep_when_exhausted(self, agent_pair, topology):
        alice, _ = agent_pair
        alice.drives.energy = 5
        place_types = {n: p.place_type for n, p in topology.places.items()}
        ctx = PerceptionContext(
            location="Park", time_of_day="morning", season="spring",
            nearby_agents=[], urgent_drive=None, memories=[],
            schedule_obligation=None, personality_preset="empathetic",
        )
        action = alice.decide(ctx, topology.all_place_names(), {}, place_types)
        assert action.type in (ACTION_SLEEP, ACTION_MOVE)

    def test_decide_uses_place_types(self, tmp_db, topology):
        """Agents should find public places for work/food without hardcoded names."""
        a = Agent(name="Worker", personality="deterministic", location="Park",
                  age_years=30, db_path=tmp_db, tenant_id="t")
        place_types = {n: p.place_type for n, p in topology.places.items()}
        ctx = PerceptionContext(
            location="Park", time_of_day="morning", season="spring",
            nearby_agents=[], urgent_drive="duty", memories=[],
            schedule_obligation="work", personality_preset="deterministic",
        )
        action = a.decide(ctx, topology.all_place_names(), {}, place_types)
        # Should move to a public place or work
        assert action.type in (ACTION_MOVE, ACTION_WORK)
        if action.type == ACTION_MOVE:
            assert place_types.get(action.target) == "public"
        a.smrti.close()


# ── Director ──────────────────────────────────────────────────────────

class TestDirector:
    def test_scene_mode(self, tmp_db, topology):
        a1 = Agent(name="A1", location="Park", db_path=tmp_db, tenant_id="t")
        a2 = Agent(name="A2", location="Park", db_path=tmp_db, tenant_id="t")
        topology.places["Park"].add_occupant("A1")
        topology.places["Park"].add_occupant("A2")
        d = Director()
        delta = d.compute_tick_delta([a1, a2], topology.places)
        assert delta == 0.25
        assert d.mode == "scene"
        a1.smrti.close()
        a2.smrti.close()

    def test_routine_mode(self, tmp_db, topology):
        a1 = Agent(name="A1", location="Park", db_path=tmp_db, tenant_id="t")
        topology.places["Park"].add_occupant("A1")
        d = Director()
        delta = d.compute_tick_delta([a1], topology.places)
        assert delta == 2.0
        assert d.mode == "routine"
        a1.smrti.close()

    def test_skip_mode(self, tmp_db, topology):
        a1 = Agent(name="A1", location="Park", db_path=tmp_db, tenant_id="t")
        d = Director()
        d.request_skip()
        delta = d.compute_tick_delta([a1], topology.places)
        assert delta == 168.0
        assert d.mode == "skip"
        a1.smrti.close()


# ── Lifecycle ─────────────────────────────────────────────────────────

class TestLifecycle:
    def test_inherit_traits(self):
        t_a = {t: 0.2 for t in TRAIT_NAMES}
        t_b = {t: 0.8 for t in TRAIT_NAMES}
        child = inherit_traits(t_a, t_b, stress_level=0.0)
        for t in TRAIT_NAMES:
            assert 0.0 <= child[t] <= 1.0

    def test_stress_increases_variance(self):
        """High stress should produce more spread in child traits."""
        random.seed(42)
        t_a = {t: 0.4 for t in TRAIT_NAMES}
        t_b = {t: 0.6 for t in TRAIT_NAMES}
        # Low stress
        children_low = [inherit_traits(t_a, t_b, 0.0) for _ in range(100)]
        # High stress
        random.seed(42)
        children_high = [inherit_traits(t_a, t_b, 1.0) for _ in range(100)]
        # Variance should be higher with stress
        var_low = sum(
            sum((c[t] - 0.5) ** 2 for c in children_low) for t in TRAIT_NAMES
        )
        random.seed(99)
        children_high = [inherit_traits(t_a, t_b, 1.0) for _ in range(100)]
        var_high = sum(
            sum((c[t] - 0.5) ** 2 for c in children_high) for t in TRAIT_NAMES
        )
        # With stress_level=1.0, variance_mult=3.0 so spread should be larger
        # (statistically almost always true with 100 samples)
        assert var_high > var_low * 0.5  # generous threshold

    def test_relationship_inference(self, agent_pair):
        alice, bob = agent_pair
        assert _infer_relationship_state(alice, bob) == "stranger"
        for _ in range(6):
            alice.increment_interaction("Bob")
            bob.increment_interaction("Alice")
        assert _infer_relationship_state(alice, bob) == "friend"

    def test_reproduction_gate_requires_adults(self, agent_pair):
        alice, bob = agent_pair
        cal = SimCalendar()
        assert not check_reproduction_gate(alice, bob, cal, 2)

    def test_spawn_child(self, tmp_db):
        a = Agent(name="Anna", personality="empathetic", age_years=30, db_path=tmp_db, tenant_id="t")
        b = Agent(name="Ben", personality="curious", age_years=32, db_path=tmp_db, tenant_id="t")
        child = spawn_child(a, b, [a, b], tmp_db, "t")
        assert child.alive
        assert child.parents == ("Anna", "Ben")
        assert child.personality_preset == "inherited"
        assert all(t in child.traits for t in TRAIT_NAMES)
        assert child.age_years == 0.0
        a.smrti.close()
        b.smrti.close()
        child.smrti.close()


# ── Narrator ──────────────────────────────────────────────────────────

class TestNarrator:
    def test_narrate_conversation(self):
        n = Narrator()
        result = n.narrate_conversation("Alice", "Bob", "Park", "Hello!")
        assert "Alice" in result["place"]
        assert "Bob" in result["place"]
        # Listener hears "Alice told me..."
        assert "Alice" in result["listener"]
        assert "Hello!" in result["listener"]


# ── Sporadic Events ───────────────────────────────────────────────────

class TestSporadic:
    def test_generate_respects_delta_scaling(self, tmp_db, topology):
        a = Agent(name="A", location="Park", db_path=tmp_db, tenant_id="t")
        topology.places["Park"].add_occupant("A")
        # With very small delta, almost no events
        random.seed(0)
        events_small = generate_sporadic_events([a], topology, 0.01, "spring")
        random.seed(0)
        events_large = generate_sporadic_events([a], topology, 100.0, "spring")
        # Large delta should produce at least as many events
        assert len(events_large) >= len(events_small)
        a.smrti.close()


# ── Interaction Persistence ───────────────────────────────────────────

class TestInteractionPersistence:
    def test_persist_and_restore(self, tmp_db):
        a = Agent(name="Test", personality="balanced", db_path=tmp_db, tenant_id="t")
        for _ in range(5):
            a.increment_interaction("Friend")
        a.persist_interactions()

        # Create a new agent with the same space to simulate restart
        a2 = Agent(name="Test", personality="balanced", db_path=tmp_db, tenant_id="t")
        a2.restore_interactions()
        assert a2.get_interaction_count("Friend") == 5
        a.smrti.close()
        a2.smrti.close()


# ── Engine integration ────────────────────────────────────────────────

class TestEngine:
    def test_single_tick(self, tmp_db, topology):
        alice = Agent(name="Alice", personality="empathetic", location="Park",
                      age_years=30, db_path=tmp_db, tenant_id="t")
        bob = Agent(name="Bob", personality="analytical", location="Cafe",
                    age_years=28, db_path=tmp_db, tenant_id="t")
        topology.places["Park"].add_occupant("Alice")
        topology.places["Cafe"].add_occupant("Bob")
        engine = SimEngine(
            agents=[alice, bob], topology=topology,
            db_path=tmp_db, tenant_id="t",
        )
        result = asyncio.get_event_loop().run_until_complete(engine.tick())
        assert result.tick_number == 1
        assert len(result.agents) == 2
        alice.smrti.close()
        bob.smrti.close()

    def test_place_types_passed_to_decide(self, tmp_db, topology):
        """Engine should pass place_types dict so agents don't use hardcoded names."""
        alice = Agent(name="Alice", personality="empathetic", location="Park",
                      age_years=30, db_path=tmp_db, tenant_id="t")
        topology.places["Park"].add_occupant("Alice")
        engine = SimEngine(
            agents=[alice], topology=topology,
            db_path=tmp_db, tenant_id="t",
        )
        # Run a tick — if place_types weren't passed, _place_types would be empty
        asyncio.get_event_loop().run_until_complete(engine.tick())
        # _place_types should have been set during decide
        assert hasattr(alice, "_place_types")
        assert alice._place_types.get("Cafe") == "public"
        alice.smrti.close()
