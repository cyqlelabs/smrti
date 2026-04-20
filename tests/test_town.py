"""Tests for the smrti-town simulator."""

from __future__ import annotations

import os
import tempfile

import pytest

from smrti_town.agent import Action, Citizen, PerceptionContext
from smrti_town.calendar import SimCalendar
from smrti_town.config import (
    ACTION_EAT,
    ACTION_MOVE,
    ACTION_SLEEP,
    ACTION_WAIT,
    ACTION_WANDER,
    ACTION_WORK,
    HOURS_PER_YEAR,
    HUNGER_RATE,
    HUNGER_THRESHOLD,
    NEED_MAX,
    PRESET_TRAITS,
    TICK_ROUTINE,
    TICK_SCENE,
    TICK_SKIP,
    TRAIT_NAMES,
)
from smrti_town.director import Chronos, Director
from smrti_town.drives import CitizenNeeds, MASLOW_ORDER
from smrti_town.lifecycle import (
    check_death,
    check_relationship_progression,
    check_reproduction_eligibility,
    create_child,
)
from smrti_town.spatial import Place, TownTopology


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


@pytest.fixture
def topology():
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
    topo.assign_home("Alice", "Alice_Home")
    topo.assign_home("Bob", "Bob_Home")
    return topo


@pytest.fixture
def citizen_pair(tmp_db):
    alice = Citizen(
        name="Alice", personality="empathetic", location="Park",
        age_years=30, db_path=tmp_db, tenant_id="test",
    )
    bob = Citizen(
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
        assert cal.time_of_day == "night"
        cal.advance(8)
        assert cal.time_of_day == "morning"
        cal.advance(6)
        assert cal.time_of_day == "afternoon"

    def test_seasons(self):
        cal = SimCalendar(total_hours=0)
        assert cal.season == "spring"
        cal.advance(7 * 24)
        assert cal.season == "summer"
        cal.advance(7 * 24)
        assert cal.season == "autumn"

    def test_year_rollover(self):
        cal = SimCalendar(total_hours=HOURS_PER_YEAR - 1)
        assert cal.year == 1
        cal.advance(2)
        assert cal.year == 2


# ── Needs (Maslow) ────────────────────────────────────────────────────

class TestNeeds:
    def test_all_needs_start_at_zero(self):
        n = CitizenNeeds()
        for need in MASLOW_ORDER:
            assert getattr(n, need) == 0.0

    def test_tick_accumulates_hunger(self):
        n = CitizenNeeds()
        n.tick(2.0, "adult", has_home=True, has_job=False)
        assert n.hunger == pytest.approx(HUNGER_RATE * 2.0)

    def test_satisfy_hunger(self):
        n = CitizenNeeds()
        n.hunger = 80.0
        n.satisfy("hunger")
        assert n.hunger == 0.0

    def test_highest_unmet_need(self):
        n = CitizenNeeds()
        n.hunger = HUNGER_THRESHOLD + 1
        assert n.highest_unmet_need("adult") == "hunger"

    def test_highest_unmet_need_none_when_satisfied(self):
        n = CitizenNeeds()
        assert n.highest_unmet_need("adult") is None

    def test_need_urgency(self):
        n = CitizenNeeds()
        n.hunger = 50.0
        assert n.need_urgency("hunger") == pytest.approx(0.5)

    def test_serialization_roundtrip(self):
        n = CitizenNeeds()
        n.tick(4.0, "adult", has_home=True, has_job=False)
        n2 = CitizenNeeds.from_dict(n.to_dict())
        assert abs(n2.hunger - n.hunger) < 0.01


# ── Topology ──────────────────────────────────────────────────────────

class TestTopology:
    def test_path_distance(self, topology):
        assert topology.path_distance("Park", "Cafe") == 1
        assert topology.path_distance("Alice_Home", "Bob_Home") == 2
        assert topology.path_distance("Park", "Park") == 0

    def test_places_by_type(self, topology):
        public_names = {p.name for p in topology.places_by_type("public")}
        assert public_names == {"Cafe", "Library"}
        home_names = {p.name for p in topology.places_by_type("home")}
        assert home_names == {"Alice_Home", "Bob_Home"}

    def test_home_for(self, topology):
        assert topology.home_for("Alice").name == "Alice_Home"
        assert topology.home_for("Bob").name == "Bob_Home"
        assert topology.home_for("Unknown") is None

    def test_move_agent(self, topology):
        topology.places["Park"].add_occupant("Alice")
        topology.move_agent("Alice", "Park", "Cafe")
        assert "Alice" not in topology.places["Park"].occupants
        assert "Alice" in topology.places["Cafe"].occupants


# ── Citizen ───────────────────────────────────────────────────────────

class TestCitizen:
    def test_traits_from_preset(self, tmp_db):
        a = Citizen(name="A", personality="empathetic", db_path=tmp_db, tenant_id="t")
        assert a.traits == PRESET_TRAITS["empathetic"]
        a.smrti.close()

    def test_traits_custom(self, tmp_db):
        custom = {t: 0.5 for t in TRAIT_NAMES}
        a = Citizen(name="A", personality="balanced", db_path=tmp_db, tenant_id="t", traits=custom)
        assert a.traits == custom
        a.smrti.close()

    def test_life_stage(self, tmp_db):
        infant = Citizen(name="I", age_years=2, db_path=tmp_db, tenant_id="t")
        assert infant.life_stage == "infant"
        child = Citizen(name="C", age_years=10, db_path=tmp_db, tenant_id="t")
        assert child.life_stage == "child"
        adult = Citizen(name="A", age_years=30, db_path=tmp_db, tenant_id="t")
        assert adult.life_stage == "adult"
        elder = Citizen(name="E", age_years=70, db_path=tmp_db, tenant_id="t")
        assert elder.life_stage == "elder"
        for a in (infant, child, adult, elder):
            a.smrti.close()

    def test_interaction_counts(self, citizen_pair):
        alice, _ = citizen_pair
        assert alice.interaction_counts.get("Bob", 0) == 0
        alice.record_interaction("Bob")
        alice.record_interaction("Bob")
        assert alice.interaction_counts["Bob"] == 2

    def test_decide_when_hungry(self, citizen_pair, topology):
        alice, _ = citizen_pair
        alice.needs.hunger = NEED_MAX
        ctx = PerceptionContext(
            location="Park", time_of_day="morning", season="spring",
            nearby_agents=[], urgent_need="hunger", memories=[],
            schedule_obligation=None, personality_preset="empathetic",
        )
        action = alice.decide(ctx, topology)
        assert isinstance(action, Action)

    def test_decide_with_work_schedule(self, tmp_db, topology):
        a = Citizen(name="Worker", personality="deterministic", location="Park",
                    age_years=30, db_path=tmp_db, tenant_id="t")
        ctx = PerceptionContext(
            location="Park", time_of_day="morning", season="spring",
            nearby_agents=[], urgent_need=None, memories=[],
            schedule_obligation="work", personality_preset="deterministic",
        )
        action = a.decide(ctx, topology)
        assert action.type in (ACTION_MOVE, ACTION_WORK, ACTION_WANDER, ACTION_WAIT)
        a.smrti.close()


# ── Director ──────────────────────────────────────────────────────────

class TestDirector:
    def test_scene_mode(self, tmp_db, topology):
        a1 = Citizen(name="A1", location="Park", db_path=tmp_db, tenant_id="t")
        a2 = Citizen(name="A2", location="Park", db_path=tmp_db, tenant_id="t")
        topology.places["Park"].add_occupant("A1")
        topology.places["Park"].add_occupant("A2")
        d = Director()
        cal = SimCalendar()
        delta = d.compute_delta([a1, a2], cal)
        assert delta == TICK_SCENE
        assert d.mode == "scene"
        a1.smrti.close()
        a2.smrti.close()

    def test_routine_mode(self, tmp_db, topology):
        a1 = Citizen(name="A1", location="Park", db_path=tmp_db, tenant_id="t")
        topology.places["Park"].add_occupant("A1")
        d = Director()
        cal = SimCalendar(total_hours=10)  # daytime — avoids montage path
        delta = d.compute_delta([a1], cal)
        assert delta == TICK_ROUTINE
        assert d.mode == "routine"
        a1.smrti.close()

    def test_skip_mode(self, tmp_db):
        a1 = Citizen(name="A1", location="Park", db_path=tmp_db, tenant_id="t")
        d = Director()
        d.force_skip()
        cal = SimCalendar()
        delta = d.compute_delta([a1], cal)
        assert delta == TICK_SKIP
        assert d.mode == "skip"
        a1.smrti.close()


# ── Lifecycle ─────────────────────────────────────────────────────────

class TestLifecycle:
    def test_create_child_spec(self, tmp_db):
        a = Citizen(name="Anna", personality="empathetic", age_years=30, db_path=tmp_db, tenant_id="t")
        b = Citizen(name="Ben", personality="curious", age_years=32, db_path=tmp_db, tenant_id="t")
        spec = create_child(a, b)
        assert spec["parents"] == ["Anna", "Ben"]
        assert spec["life_stage"] == "infant"
        assert spec["age"] == 0
        assert all(t in spec["traits"] for t in TRAIT_NAMES)
        a.smrti.close()
        b.smrti.close()

    def test_reproduction_eligibility_no_relationship(self, citizen_pair):
        alice, bob = citizen_pair
        assert not check_reproduction_eligibility(alice, bob)

    def test_relationship_progression_insufficient_interactions(self, citizen_pair):
        alice, bob = citizen_pair
        result = check_relationship_progression(alice, bob, interaction_count=0, shared_valence=0.5)
        assert result is None

