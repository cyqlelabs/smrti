"""Tests for the smrti-town simulator."""

from __future__ import annotations

import dataclasses
import os
import tempfile

import pytest

from smrti_town.agent import Action, Citizen, PerceptionContext
from smrti_town.calendar import SimCalendar
from smrti_town.config import (
    ACTION_EAT,
    ACTION_MOVE,
    ACTION_SLEEP,
    ACTION_TALK,
    ACTION_WAIT,
    ACTION_WANDER,
    ACTION_WORK,
    EXPERIENCE_VALENCE,
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
            nearby_agents=[], urgent_need="hunger",
            schedule_obligation=None, personality_preset="empathetic",
        )
        action = alice.decide(ctx, topology)
        assert isinstance(action, Action)

    def test_decide_with_work_schedule(self, tmp_db, topology):
        a = Citizen(name="Worker", personality="deterministic", location="Park",
                    age_years=30, db_path=tmp_db, tenant_id="t")
        ctx = PerceptionContext(
            location="Park", time_of_day="morning", season="spring",
            nearby_agents=[], urgent_need=None,
            schedule_obligation="work", personality_preset="deterministic",
        )
        action = a.decide(ctx, topology)
        assert action.type in (ACTION_MOVE, ACTION_WORK, ACTION_WANDER, ACTION_WAIT)
        a.smrti.close()


# ── Memory loop ───────────────────────────────────────────────────────

@pytest.fixture
def town():
    topo = TownTopology()
    topo.add_place(Place(name="Town Square", is_outdoor=True, place_type="outdoor"))
    topo.add_place(Place(name="Tavern", place_type="public", building_key="tavern"))
    topo.add_place(Place(name="Bakery", place_type="public", building_key="bakery"))
    topo.add_place(Place(name="Park", is_outdoor=True, place_type="outdoor", building_key="park"))
    topo.connect("Town Square", "Tavern")
    topo.connect("Town Square", "Park")
    topo.connect("Tavern", "Bakery")
    return topo


@pytest.fixture
def alice(tmp_db):
    a = Citizen(name="Alice", location="Town Square", db_path=tmp_db, tenant_id="t")
    yield a
    a.smrti.close()


class TestMemoryLoop:
    """Action → memory → decision: what a citizen lives through changes
    where it goes and whom it talks to."""

    def test_a_quarrel_sours_the_place(self, alice, town):
        assert alice._find_food_source(town) == "Tavern"
        assert alice._find_social_venue(town) == "Tavern"
        alice.experience("Alice quarrelled with Bob at Tavern", EXPERIENCE_VALENCE["quarrel"])
        assert alice._place_valence("Tavern") < 0
        assert alice._place_valence("Bakery") == 0.0
        assert alice._find_food_source(town) == "Bakery"
        assert alice._find_social_venue(town) == "Park"

    def test_routine_fades_and_quarrels_hold(self, alice):
        alice.experience("Alice had a meal at Bakery", EXPERIENCE_VALENCE["meal"])
        alice.experience("Alice quarrelled with Bob at Tavern", EXPERIENCE_VALENCE["quarrel"])
        (meal,) = alice.smrti.recall("Bakery", top_k=1)
        (quarrel,) = alice.smrti.recall("quarrelled with Bob", top_k=1)
        assert meal.atom.metadata["source"] == "agent"
        assert meal.atom.attention.lti == 0.0
        assert quarrel.atom.attention.lti >= 0.5

    def test_a_good_meal_draws_the_citizen_back(self, alice, town):
        alice.experience("Alice had a meal at Bakery", EXPERIENCE_VALENCE["meal"])
        assert alice._find_food_source(town) == "Bakery"

    def test_a_quarrel_sours_the_person(self, alice):
        alice.experience("Alice quarrelled with Bob at Tavern", EXPERIENCE_VALENCE["quarrel"])
        alice.experience("Alice talked with Carol at Tavern", EXPERIENCE_VALENCE["talk"])
        assert alice._person_valence("Bob") < 0 < alice._person_valence("Carol")

    def test_talk_tone(self, citizen_pair, monkeypatch):
        alice, bob = citizen_pair
        monkeypatch.setattr("smrti_town.agent.random.random", lambda: 0.0)
        assert alice.talk_tone(bob) == EXPERIENCE_VALENCE["quarrel"]
        monkeypatch.setattr("smrti_town.agent.random.random", lambda: 1.0)
        assert alice.talk_tone(bob) == EXPERIENCE_VALENCE["talk"]

    def test_resolution_writes_what_each_citizen_lived(self, citizen_pair, town, monkeypatch):
        from smrti_town.economy import EconomyManager
        from smrti_town.server import _resolve_actions

        alice, bob = citizen_pair
        alice.location = bob.location = "Tavern"
        economy = EconomyManager()
        monkeypatch.setattr("smrti_town.agent.random.random", lambda: 0.0)
        talk = {"citizen": "Alice", "action": dataclasses.asdict(Action(type=ACTION_TALK, target="Bob"))}
        assert _resolve_actions([talk], [alice, bob], economy, town, delta=1.0) == [
            (alice, "Alice quarrelled with Bob at Tavern", EXPERIENCE_VALENCE["quarrel"]),
            (bob, "Bob quarrelled with Alice at Tavern", EXPERIENCE_VALENCE["quarrel"]),
        ]
        assert alice.interaction_counts["Bob"] == bob.interaction_counts["Alice"] == 1

        eat = {"citizen": "Alice", "action": dataclasses.asdict(Action(type=ACTION_EAT, target="Tavern"))}
        assert _resolve_actions([eat], [alice], economy, town, delta=1.0) == [
            (alice, "Alice had a meal at Tavern", EXPERIENCE_VALENCE["meal"]),
        ]
        economy.wallets["Alice"] = 0
        assert _resolve_actions([eat], [alice], economy, town, delta=1.0) == [
            (alice, "Alice could not afford a meal at Tavern", EXPERIENCE_VALENCE["unaffordable"]),
        ]

    def test_memory_route_reads_the_atom(self, alice):
        from smrti_town.server import _memory_dict

        alice.experience("Alice had a meal at Bakery", EXPERIENCE_VALENCE["meal"])
        (memory,) = [_memory_dict(r) for r in alice.smrti.recall("Bakery", top_k=1)]
        assert memory["content"] == "Alice had a meal at Bakery"
        assert memory["type"] == "episode"
        assert memory["valence"] == EXPERIENCE_VALENCE["meal"]


# ── Town loops ────────────────────────────────────────────────────────

class TestTownLoops:
    """The daily loops that keep the town running without the player."""

    def test_working_at_a_farm_teaches_agriculture(self, alice, town):
        from smrti_town.economy import EconomyManager
        from smrti_town.server import _resolve_actions

        town.add_place(Place(name="Farm", place_type="industrial", building_key="farm"))
        alice.location = "Farm"
        work = {"citizen": "Alice", "action": dataclasses.asdict(Action(type=ACTION_WORK, target="Farm"))}
        _resolve_actions([work], [alice], EconomyManager(), town, delta=8.0)
        assert alice.skills.level("agriculture") > 0
        assert alice.skills.level("medicine") == 0

    def test_open_jobs_are_filled(self, citizen_pair, town):
        from smrti_town.server import _fill_open_jobs

        alice, bob = citizen_pair
        town.add_place(Place(name="Farm", place_type="industrial", building_key="farm"))
        _fill_open_jobs([alice, bob], town)
        assert alice.workplace is not None and bob.workplace is not None
        assert {alice.workplace, bob.workplace} <= {"Tavern", "Bakery", "Farm"}

    def test_a_skilled_saver_petitions_for_a_business(self, citizen_pair, tmp_db, town):
        from smrti_town.economy import EconomyManager
        from smrti_town.petition import PetitionManager
        from smrti_town.server import _business_petitions

        alice, bob = citizen_pair
        # a general store unlocks at five residents
        others = [Citizen(name=f"C{i}", db_path=tmp_db, tenant_id="test") for i in range(3)]
        citizens = [alice, bob, *others]
        economy = EconomyManager()
        for c in citizens:
            economy.register_citizen(c.name, starting_wallet=1000)
        alice.skills.skills["commerce"] = 0.5
        pm = PetitionManager()
        new = _business_petitions(pm, citizens, economy, town, hours=0.0)
        assert [p.source for p in new] == ["Alice"]
        assert new[0].building_suggestion == "general_store"
        assert _business_petitions(pm, citizens, economy, town, hours=1.0) == []
        for c in others:
            c.smrti.close()


# ── Events ────────────────────────────────────────────────────────────

class TestEvents:
    def test_effects_land_on_needs_wallets_and_memory(self, citizen_pair):
        from smrti_town.economy import EconomyManager
        from smrti_town.events import EventManager, GameEvent

        alice, bob = citizen_pair
        economy = EconomyManager(treasury=5000)
        economy.register_citizen("Alice")
        fire = GameEvent(
            event_type="crisis_fire", description="A fire breaks out!",
            affected_citizens=["Alice"], affected_buildings=["Bakery"],
            effects={"treasury": -2000, "health": -0.3},
        )
        experiences = EventManager.apply_effects(fire, {"Alice": alice, "Bob": bob}, economy)
        assert economy.treasury == 3000
        assert alice.needs.health == pytest.approx(30.0)
        assert bob.needs.health == 0.0
        assert experiences == [(alice, "A fire breaks out! Bakery burned down.", -0.8)]

        coin = GameEvent(event_type="found_item", description="Alice found a coin at Park",
                         affected_citizens=["Alice"], effects={"wallet": 5})
        EventManager.apply_effects(coin, {"Alice": alice}, economy)
        assert economy.wallets["Alice"] == 105

    def test_crime_rate(self, citizen_pair, town):
        from smrti_town.events import EventManager, GameEvent

        alice, bob = citizen_pair
        manager = EventManager()
        assert manager.crime_rate([alice, bob], town) == pytest.approx(0.3)
        alice.workplace = "Bakery"
        assert manager.crime_rate([alice, bob], town) == pytest.approx(0.15)
        manager.active_crises.append(GameEvent(event_type="crisis_crime_wave", description=""))
        assert manager.crime_rate([alice, bob], town) == pytest.approx(0.65)
        town.add_place(Place(name="Watch", place_type="civic", building_key="constabulary"))
        assert manager.crime_rate([alice, bob], town) == pytest.approx(0.325)


# ── Relationships and births ──────────────────────────────────────────

class TestLifecycleLoop:
    def test_good_evenings_make_friends_and_quarrels_unmake_them(self, citizen_pair):
        from smrti_town.lifecycle import update_relationships

        alice, bob = citizen_pair
        for _ in range(6):
            alice.experience("Alice talked with Bob at Tavern", EXPERIENCE_VALENCE["talk"])
            bob.experience("Bob talked with Alice at Tavern", EXPERIENCE_VALENCE["talk"])
            alice.record_interaction("Bob")
            bob.record_interaction("Alice")
        changes = update_relationships([alice, bob], hours=100.0)
        assert [(a.name, b.name, old, new) for a, b, old, new in changes] == [("Alice", "Bob", "acquaintance", "friend")]
        assert alice.relationships["Bob"] == bob.relationships["Alice"] == "friend"
        assert alice.relationship_since["Bob"] == 100.0

        for _ in range(9):
            alice.experience("Alice quarrelled with Bob at Tavern", EXPERIENCE_VALENCE["quarrel"])
            bob.experience("Bob quarrelled with Alice at Tavern", EXPERIENCE_VALENCE["quarrel"])
        changes = update_relationships([alice, bob], hours=200.0)
        assert [(old, new) for _, _, old, new in changes] == [("friend", "acquaintance")]

    def test_a_settled_couple_has_a_child_on_its_parents_genome(self, citizen_pair, tmp_db, town, monkeypatch):
        from smrti_town.lifecycle import check_births
        from smrti_town.population import PopulationManager

        alice, bob = citizen_pair
        town.add_place(Place(name="Home", place_type="residential", building_key="house"))
        for c in (alice, bob):
            c.home = "Home"
            town.assign_home(c.name, "Home")
        alice.relationships["Bob"] = bob.relationships["Alice"] = "romantic"
        alice.relationship_since["Bob"] = bob.relationship_since["Alice"] = 0.0
        monkeypatch.setattr("smrti_town.lifecycle.random.random", lambda: 0.0)
        assert check_births([alice, bob], PopulationManager(), town, hours=HOURS_PER_YEAR / 2) == []
        (spec,) = check_births([alice, bob], PopulationManager(), town, hours=2 * HOURS_PER_YEAR)
        assert spec["parents"] == ["Alice", "Bob"]
        child = Citizen(name=spec["name"], age_years=0.0, personality=spec["personality"],
                        db_path=tmp_db, tenant_id="test", traits=spec["traits"], parents=("Alice", "Bob"))
        child.inherit(spec["personality_params"])
        assert child.personality_params == pytest.approx(spec["personality_params"])
        assert child.life_stage == "infant"
        child.smrti.close()

    def test_satisfaction(self, alice):
        from smrti_town.lifecycle import satisfaction

        assert satisfaction(alice) == 1.0
        alice.needs.hunger = NEED_MAX
        assert satisfaction(alice) < 1.0


# ── LLM settings ──────────────────────────────────────────────────────

def test_llm_endpoint_comes_from_the_environment(monkeypatch):
    from smrti_town.llm import LLMSettings

    monkeypatch.delenv("SMRTI_TOWN_LLM_URL", raising=False)
    assert LLMSettings().base_url == "http://0.0.0.0:8421/v1"
    monkeypatch.setenv("SMRTI_TOWN_LLM_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("SMRTI_TOWN_LLM_MODEL", "qwen")
    settings = LLMSettings()
    assert (settings.base_url, settings.model) == ("http://localhost:8080/v1", "qwen")
    assert LLMSettings.from_dict(settings.to_dict()).model == "qwen"


# ── Culture and persistence ───────────────────────────────────────────

class TestTownMemory:
    def test_a_shared_experience_becomes_culture(self, citizen_pair, tmp_db):
        from smrti import Smrti
        from smrti_town.culture import run_culture_pass

        alice, bob = citizen_pair
        alice.location = bob.location = "Park"
        for c in (alice, bob):
            c.experience("A sudden rainstorm sweeps through Park.", -0.2)
        culture = Smrti(db_path=tmp_db, personality="balanced", tenant_id="test", write_space="Space_Culture")
        bridges, promoted = run_culture_pass([alice, bob], culture)
        assert bridges >= 1 and promoted == 1
        (shared,) = culture.recall("rainstorm Park", top_k=1)
        assert shared.atom.content == "A sudden rainstorm sweeps through Park."
        assert shared.atom.space == "Space_Culture"
        # a second pass promotes nothing new
        assert run_culture_pass([alice, bob], culture)[1] == 0

    def test_snapshot_roundtrip(self, citizen_pair, tmp_db, town):
        from smrti_town import persistence
        from smrti_town.calendar import SimCalendar
        from smrti_town.director import Chronos, Director
        from smrti_town.economy import EconomyManager
        from smrti_town.gridmap import GridMap

        alice, bob = citizen_pair
        alice.relationships["Bob"] = "friend"
        alice.relationship_since["Bob"] = 48.0
        alice.record_interaction("Bob")
        alice.home = "Tavern"
        town.assign_home("Alice", "Tavern")
        grid = GridMap()
        grid.place("town_hall", 10, 10, "Town Hall")
        game = {
            "phase": "gameplay", "tick_count": 7, "calendar": SimCalendar(total_hours=50.0),
            "director": Director(), "chronos": Chronos(), "topology": town, "gridmap": grid,
            "economy": EconomyManager(treasury=1234), "citizens": [alice, bob],
            "mayor": {"name": "Alice"}, "last_petition_check": 24.0,
        }
        persistence.save(alice.smrti.db, "test", persistence.snapshot(game))
        restored: dict = {}
        persistence.restore(restored, persistence.load(alice.smrti.db, "test"), tmp_db, "test")
        assert restored["tick_count"] == 7 and restored["calendar"].total_hours == 50.0
        assert restored["economy"].treasury == 1234
        assert set(restored["topology"].places) == set(town.places)
        assert restored["topology"].path_distance("Town Square", "Bakery") == 2
        assert restored["topology"].home_for("Alice").name == "Tavern"
        assert [b.place_name for b in restored["gridmap"].buildings] == ["Town Hall"]
        alice2 = next(c for c in restored["citizens"] if c.name == "Alice")
        assert alice2.relationships == {"Bob": "friend"} and alice2.relationship_since == {"Bob": 48.0}
        assert alice2.interaction_counts == {"Bob": 1} and alice2.home == "Tavern"
        assert restored["mayor"] == {"name": "Alice"} and restored["last_petition_check"] == 24.0
        persistence.clear(alice.smrti.db, "test")
        assert persistence.load(alice.smrti.db, "test") is None
        for c in restored["citizens"]:
            c.smrti.close()


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

