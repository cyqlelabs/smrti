"""Citizen — the main agent class for smrti-town."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from smrti import Smrti

from smrti_town.config import (
    ACTION_EAT,
    ACTION_INTERACT,
    ACTION_MOVE,
    ACTION_PLAY,
    ACTION_PRAY,
    ACTION_REPRODUCE,
    ACTION_SHOP,
    ACTION_SLEEP,
    ACTION_STUDY,
    ACTION_TALK,
    ACTION_WAIT,
    ACTION_WANDER,
    ACTION_WORK,
    AGENT_SPEED_CHILD,
    AGENT_SPEED_DEFAULT,
    AGENT_SPEED_ELDER,
    BUILDING_CATALOG,
    ELDER_DEATH_PROB_PER_TICK,
    ENTREPRENEURSHIP_COMMERCE_SKILL,
    ENTREPRENEURSHIP_SAVINGS_THRESHOLD,
    HOURS_PER_YEAR,
    LIFE_STAGES,
    NEED_MAX,
    PERSONALITY_ACTION_BIAS,
    PRESET_TRAITS,
    STARVATION_HOURS,
    STARTING_WALLET,
    TRAIT_NAMES,
)
from smrti_town.drives import CitizenNeeds
from smrti_town.skills import SkillSet

if TYPE_CHECKING:
    from smrti_town.calendar import SimCalendar
    from smrti_town.spatial import Place, TownTopology


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class Action:
    type: str
    target: str | None = None
    dialogue: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PerceptionContext:
    location: str
    time_of_day: str
    season: str
    nearby_agents: list[str] = field(default_factory=list)
    urgent_need: str | None = None
    memories: list[dict] = field(default_factory=list)
    schedule_obligation: str | None = None
    personality_preset: str = "balanced"
    current_hour: float = 12.0
    place_building_key: str | None = None
    has_home: bool = False
    has_job: bool = False
    crime_rate: float = 0.0


# ── Visual DNA generator ─────────────────────────────────────────────

_SKIN_TONES = [0xFFDDB4, 0xF1C27D, 0xE0AC69, 0xC68642, 0x8D5524, 0x6B3A2A]
_HAIR_COLORS = [0x090806, 0x2C222B, 0x71635A, 0xB7A69E, 0xD6C4C2, 0xCB6820, 0xE6CEA8]
_SHIRT_COLORS = [0xE74C3C, 0x3498DB, 0x2ECC71, 0xF39C12, 0x9B59B6, 0x1ABC9C, 0xE67E22, 0x7F8C8D]
_PANT_COLORS = [0x2C3E50, 0x34495E, 0x795548, 0x4A235A, 0x1B4F72, 0x1E8449]


def _generate_visual_dna() -> dict:
    return {
        "skin": random.choice(_SKIN_TONES),
        "hair": random.choice(_HAIR_COLORS),
        "shirt": random.choice(_SHIRT_COLORS),
        "pants": random.choice(_PANT_COLORS),
        "hat": random.random() < 0.3,
        "hair_style": random.randint(0, 4),
        "body_type": random.randint(0, 2),
    }


# ── Citizen ───────────────────────────────────────────────────────────

class Citizen:
    """A town citizen with Smrti memory, Maslow needs, skills, and traits."""

    def __init__(
        self,
        name: str,
        personality: str = "balanced",
        location: str = "",
        age_years: float = 25.0,
        db_path: str = "~/.smrti/town.db",
        tenant_id: str = "millbrook",
        parents: tuple[str, str] | None = None,
        traits: dict[str, float] | None = None,
        home: str | None = None,
        workplace: str | None = None,
        council_role: str | None = None,
        initial_skills: dict[str, float] | None = None,
        visual_dna: dict | None = None,
    ) -> None:
        self.name = name
        self.personality_preset = personality
        self.location = location

        # Smrti memory instance
        write_space = f"Agent_Space_{name}"
        self.smrti = Smrti(
            db_path=db_path,
            personality=personality,
            tenant_id=tenant_id,
            write_space=write_space,
            read_spaces=[write_space, "World_Space", "Space_Culture"],
        )

        # Drives and skills
        self.needs = CitizenNeeds()
        self.skills = SkillSet(initial=initial_skills)

        # Economy
        self.wallet: int = STARTING_WALLET

        # Traits
        if traits is not None:
            self.traits: dict[str, float] = {
                t: max(0.0, min(1.0, traits.get(t, 0.5))) for t in TRAIT_NAMES
            }
        elif personality in PRESET_TRAITS:
            self.traits = dict(PRESET_TRAITS[personality])
        else:
            self.traits = dict(PRESET_TRAITS["balanced"])

        # Visual appearance
        self.visual_dna: dict = visual_dna if visual_dna is not None else _generate_visual_dna()

        # Movement state
        self.world_pos: tuple[float, float] = (0.0, 0.0)
        self.path: list[tuple[float, float]] = []
        self.path_index: int = 0
        self.moving: bool = False
        self.speed: float = AGENT_SPEED_DEFAULT
        self.facing: str = "south"

        # Life
        self.alive: bool = True
        self.age_hours: float = age_years * HOURS_PER_YEAR
        self.parents = parents
        self.home = home
        self.workplace = workplace
        self.council_role = council_role

        # Social
        self.relationships: dict[str, str] = {}  # name -> relationship_type
        self.interaction_counts: dict[str, int] = {}

        # Health tracking
        self.starvation_hours: float = 0.0

        # Current action (set by decide())
        self.current_action: Action | None = None

    # ── properties ────────────────────────────────────────────────────

    @property
    def age_years(self) -> float:
        return self.age_hours / HOURS_PER_YEAR

    @property
    def life_stage(self) -> str:
        age = self.age_years
        for stage_name, info in LIFE_STAGES.items():
            lo, hi = info["age_range"]
            if lo <= age < hi:
                return stage_name
        return "elder"

    @property
    def life_stage_info(self) -> dict:
        return LIFE_STAGES.get(self.life_stage, LIFE_STAGES["adult"])

    @property
    def can_move(self) -> bool:
        return self.life_stage_info.get("can_move", True)

    @property
    def can_talk(self) -> bool:
        return self.life_stage_info.get("can_talk", True)

    @property
    def can_work(self) -> bool:
        return self.life_stage_info.get("can_work", True)

    @property
    def can_reproduce(self) -> bool:
        return self.life_stage_info.get("can_reproduce", False)

    # ── tick ──────────────────────────────────────────────────────────

    def tick_state(
        self,
        delta_hours: float,
        crime_rate: float = 0.0,
        nearby_count: int = 0,
    ) -> None:
        """Update age, needs, starvation tracking, and death checks."""
        if not self.alive:
            return

        # Age
        self.age_hours += delta_hours

        # Speed by life stage
        stage = self.life_stage
        if stage == "child":
            self.speed = AGENT_SPEED_CHILD
        elif stage == "elder":
            self.speed = AGENT_SPEED_ELDER
        else:
            self.speed = AGENT_SPEED_DEFAULT

        # Needs — council roles count as meaningful employment
        has_job = self.workplace is not None or self.council_role is not None
        current_action = self.current_action.type if self.current_action else None
        self.needs.tick(
            delta_hours=delta_hours,
            life_stage=stage,
            has_home=self.home is not None,
            has_job=has_job,
            crime_rate=crime_rate,
            current_action=current_action,
            nearby_count=nearby_count,
        )

        # Starvation
        if self.needs.hunger >= NEED_MAX:
            self.starvation_hours += delta_hours
        else:
            self.starvation_hours = max(0.0, self.starvation_hours - delta_hours)

        # Death checks
        if self.starvation_hours >= STARVATION_HOURS:
            self.alive = False
            return

        if stage == "elder" and random.random() < ELDER_DEATH_PROB_PER_TICK:
            self.alive = False

    # ── perception ────────────────────────────────────────────────────

    def perceive(
        self,
        topology: TownTopology,
        calendar: SimCalendar,
        nearby_agents: list[str],
        place: Place | None = None,
        crime_rate: float = 0.0,
    ) -> PerceptionContext:
        """Build a PerceptionContext from the current environment."""
        # Recall relevant memories for decision-making.
        location_name = place.name if place else self.location
        memories: list[dict] = []
        try:
            results = self.smrti.recall(
                f"at {location_name} with {', '.join(nearby_agents[:3])}" if nearby_agents
                else f"at {location_name}",
                top_k=5,
                min_confidence=0.1,
            )
            # A RecallResult wraps the atom; the memory's text, tone and
            # truth live on r.atom. (These used to read r.content and
            # r.valence, attributes the result never had, and the except
            # below swallowed the error — so memories were always empty and
            # no citizen ever decided on one.)
            memories = [
                {
                    "content": r.atom.content or r.atom.label,
                    "valence": r.atom.valence.valence,
                    "probability": r.atom.truth.probability,
                }
                for r in results
            ]
        except Exception:
            pass

        # Schedule obligation
        schedule_obligation: str | None = None
        schedule = self.life_stage_info.get("schedule")
        if schedule:
            hour = calendar.hour
            for obligation, (start, end) in schedule.items():
                if start <= hour < end:
                    schedule_obligation = obligation
                    break

        building_key = place.building_key if place else None

        return PerceptionContext(
            location=location_name,
            time_of_day=calendar.time_of_day,
            season=calendar.season,
            nearby_agents=nearby_agents,
            urgent_need=self.needs.highest_unmet_need(self.life_stage),
            memories=memories,
            schedule_obligation=schedule_obligation,
            personality_preset=self.personality_preset,
            current_hour=calendar.hour,
            place_building_key=building_key,
            has_home=self.home is not None,
            has_job=self.workplace is not None or self.council_role is not None,
            crime_rate=crime_rate,
        )

    # ── decision ──────────────────────────────────────────────────────

    def decide(self, context: PerceptionContext, topology: TownTopology | None = None) -> Action:
        """Rule-based decision engine.  No LLM calls.

        Priority: satisfy highest unmet need first, modulated by personality
        traits and memory valence.
        """
        if not self.alive:
            return Action(type=ACTION_WAIT)

        if not self.can_move:
            return Action(type=ACTION_WAIT)

        # Night time → sleep (unless urgent hunger or health need)
        if context.time_of_day == "night" and context.urgent_need not in ("hunger", "health"):
            if self.home and self.location != self.home:
                return Action(type=ACTION_MOVE, target=self.home, metadata={"reason": "sleep"})
            return Action(type=ACTION_SLEEP)

        # Schedule obligations override non-urgent needs
        if context.schedule_obligation and context.urgent_need not in ("hunger", "health"):
            return self._handle_schedule(context, topology)

        # Address urgent needs in Maslow order
        need = context.urgent_need
        if need:
            return self._handle_need(need, context, topology)

        # No urgent need — personality-driven optional activities
        return self._idle_action(context, topology)

    def _handle_schedule(self, context: PerceptionContext, topology: TownTopology | None) -> Action:
        obligation = context.schedule_obligation
        if obligation == "work" and self.workplace:
            if self.location != self.workplace:
                return Action(type=ACTION_MOVE, target=self.workplace, metadata={"reason": "work"})
            return Action(type=ACTION_WORK, target=self.workplace)
        if obligation == "school" and topology:
            school = self._find_building("school", topology)
            if school and self.location != school:
                return Action(type=ACTION_MOVE, target=school, metadata={"reason": "school"})
            if school:
                return Action(type=ACTION_STUDY, target=school)
        return Action(type=ACTION_WAIT)

    def _handle_need(
        self, need: str, context: PerceptionContext, topology: TownTopology | None
    ) -> Action:
        if need == "hunger":
            return self._handle_hunger(context, topology)
        elif need == "shelter":
            return self._handle_shelter(context, topology)
        elif need == "health":
            return self._handle_health(context, topology)
        elif need == "safety":
            return self._handle_safety(context, topology)
        elif need == "social":
            return self._handle_social(context, topology)
        elif need == "education":
            return self._handle_education(context, topology)
        elif need == "purpose":
            return self._handle_purpose(context, topology)
        elif need == "culture":
            return self._handle_culture(context, topology)
        elif need == "actualization":
            return self._handle_actualization(context, topology)
        return Action(type=ACTION_WAIT)

    # ── need handlers ─────────────────────────────────────────────────

    def _handle_hunger(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        # If at a food source, eat.
        if ctx.place_building_key and self._building_provides_food(ctx.place_building_key):
            return Action(type=ACTION_EAT, target=ctx.location)
        # Move to nearest food source.
        if topo:
            dest = self._find_food_source(topo)
            if dest and dest != ctx.location:
                return Action(type=ACTION_MOVE, target=dest, metadata={"reason": "hunger"})
            if dest:
                return Action(type=ACTION_EAT, target=dest)
        # Fallback: if at home, eat (subsistence).
        if self.home and ctx.location == self.home:
            return Action(type=ACTION_EAT, target=ctx.location)
        if self.home:
            return Action(type=ACTION_MOVE, target=self.home, metadata={"reason": "hunger"})
        return Action(type=ACTION_WAIT, metadata={"reason": "no_food"})

    def _handle_shelter(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        if self.home:
            if ctx.location != self.home:
                return Action(type=ACTION_MOVE, target=self.home, metadata={"reason": "shelter"})
            return Action(type=ACTION_SLEEP)
        # Homeless: go to inn or town hall.
        if topo:
            shelter = self._find_building("inn", topo) or self._find_building("town_hall", topo)
            if shelter and ctx.location != shelter:
                return Action(type=ACTION_MOVE, target=shelter, metadata={"reason": "shelter"})
            if shelter:
                return Action(type=ACTION_SLEEP, target=shelter)
        return Action(type=ACTION_WAIT, metadata={"reason": "homeless"})

    def _handle_health(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        if topo:
            clinic = self._find_building("clinic", topo) or self._find_building("hospital", topo)
            if clinic and ctx.location != clinic:
                return Action(type=ACTION_MOVE, target=clinic, metadata={"reason": "health"})
            if clinic:
                return Action(type=ACTION_INTERACT, target=clinic, metadata={"reason": "health"})
        # No clinic — rest at home.
        if self.home and ctx.location != self.home:
            return Action(type=ACTION_MOVE, target=self.home, metadata={"reason": "rest"})
        return Action(type=ACTION_SLEEP, metadata={"reason": "rest"})

    def _handle_safety(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        # Stay home or move near constabulary.
        if self.home and ctx.location != self.home:
            return Action(type=ACTION_MOVE, target=self.home, metadata={"reason": "safety"})
        if topo:
            safe = self._find_building("constabulary", topo)
            if safe and ctx.location != safe:
                return Action(type=ACTION_MOVE, target=safe, metadata={"reason": "safety"})
        return Action(type=ACTION_WAIT, metadata={"reason": "safety"})

    def _handle_social(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        # Personality bias: shy citizens are less likely to seek social contact.
        if random.random() < self.traits.get("shyness", 0.3) * 0.5:
            return Action(type=ACTION_WAIT, metadata={"reason": "shy"})

        # If already near other agents, talk to someone.
        if ctx.nearby_agents and self.can_talk:
            target = self._pick_social_target(ctx)
            return Action(type=ACTION_TALK, target=target)

        # Move to a social venue.
        if topo:
            venue = self._find_social_venue(topo, ctx)
            if venue and ctx.location != venue:
                return Action(type=ACTION_MOVE, target=venue, metadata={"reason": "social"})
            if venue and ctx.nearby_agents:
                return Action(type=ACTION_TALK, target=self._pick_social_target(ctx))
        return Action(type=ACTION_WANDER, metadata={"reason": "social"})

    def _handle_education(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        if topo:
            school = (
                self._find_building("school", topo)
                or self._find_building("library", topo)
                or self._find_building("university", topo)
            )
            if school and ctx.location != school:
                return Action(type=ACTION_MOVE, target=school, metadata={"reason": "education"})
            if school:
                return Action(type=ACTION_STUDY, target=school)
        return Action(type=ACTION_WAIT, metadata={"reason": "no_school"})

    def _handle_purpose(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        # Laziness trait reduces work probability.
        if random.random() < self.traits.get("laziness", 0.3) * 0.4:
            return self._idle_action(ctx, topo)

        if self.workplace:
            if ctx.location != self.workplace:
                return Action(type=ACTION_MOVE, target=self.workplace, metadata={"reason": "work"})
            return Action(type=ACTION_WORK, target=self.workplace)

        # No assigned workplace — try odd jobs at any commercial/industrial building.
        if topo:
            for bkey in ("farm", "lumber_mill", "quarry", "general_store", "market"):
                place = self._find_building(bkey, topo)
                if place:
                    if ctx.location != place:
                        return Action(type=ACTION_MOVE, target=place, metadata={"reason": "odd_job"})
                    return Action(type=ACTION_WORK, target=place, metadata={"reason": "odd_job"})
        return Action(type=ACTION_WANDER, metadata={"reason": "no_work"})

    def _handle_culture(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        if topo:
            venue = (
                self._find_building("theater", topo)
                or self._find_building("museum", topo)
                or self._find_building("park", topo)
                or self._find_building("festival_grounds", topo)
            )
            if venue and ctx.location != venue:
                return Action(type=ACTION_MOVE, target=venue, metadata={"reason": "culture"})
            if venue:
                return Action(type=ACTION_PLAY, target=venue)
        return Action(type=ACTION_WANDER, metadata={"reason": "culture"})

    def _handle_actualization(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        # Check entrepreneurship eligibility.
        if (
            self.wallet >= ENTREPRENEURSHIP_SAVINGS_THRESHOLD
            and self.skills.level("commerce") >= ENTREPRENEURSHIP_COMMERCE_SKILL
        ):
            return Action(
                type=ACTION_INTERACT,
                metadata={"reason": "entrepreneurship", "eligible": True},
            )
        # Otherwise pursue culture or education as self-improvement.
        if random.random() < 0.5:
            return self._handle_culture(ctx, topo)
        return self._handle_education(ctx, topo)

    # ── idle / personality-driven ─────────────────────────────────────

    def _idle_action(self, ctx: PerceptionContext, topo: TownTopology | None) -> Action:
        """Choose an action based on personality bias when no need is urgent."""
        bias = PERSONALITY_ACTION_BIAS.get(self.personality_preset, PERSONALITY_ACTION_BIAS["balanced"])

        # Build weighted choices.
        choices: list[tuple[str, float]] = []
        for category, weight in bias.items():
            choices.append((category, weight))

        # Pick by weighted random.
        category = self._weighted_choice(choices)

        if category == "social" and ctx.nearby_agents and self.can_talk:
            return Action(type=ACTION_TALK, target=self._pick_social_target(ctx))
        elif category == "social" and topo:
            venue = self._find_social_venue(topo, ctx)
            if venue and ctx.location != venue:
                return Action(type=ACTION_MOVE, target=venue, metadata={"reason": "social_idle"})
        elif category == "education" and topo:
            school = self._find_building("library", topo) or self._find_building("school", topo)
            if school and ctx.location != school:
                return Action(type=ACTION_MOVE, target=school, metadata={"reason": "study_idle"})
            if school:
                return Action(type=ACTION_STUDY, target=school)
        elif category == "purpose" and self.workplace:
            if ctx.location != self.workplace:
                return Action(type=ACTION_MOVE, target=self.workplace, metadata={"reason": "extra_work"})
            return Action(type=ACTION_WORK, target=self.workplace)
        elif category == "culture" and topo:
            venue = (
                self._find_building("park", topo)
                or self._find_building("theater", topo)
                or self._find_building("museum", topo)
            )
            if venue and ctx.location != venue:
                return Action(type=ACTION_MOVE, target=venue, metadata={"reason": "culture_idle"})
            if venue:
                return Action(type=ACTION_PLAY, target=venue)
        elif category == "wander":
            return Action(type=ACTION_WANDER)

        # Default fallback.
        if self.traits.get("adventurous", 0.4) > 0.5:
            return Action(type=ACTION_WANDER)
        return Action(type=ACTION_WAIT)

    # ── helpers ───────────────────────────────────────────────────────

    def _find_building(self, building_key: str, topo: TownTopology) -> str | None:
        """Find the nearest place with the given building_key, avoiding places
        with negative memory valence."""
        places = topo.places_by_building(building_key)
        if not places:
            return None
        if len(places) == 1:
            return places[0].name

        # Score by distance (prefer closer) and memory valence (avoid negative).
        best_name: str | None = None
        best_score = float("-inf")
        for p in places:
            dist = topo.path_distance(self.location, p.name) if self.location else 99
            if dist < 0:
                dist = 99
            # Check memory valence for this place.
            valence_bias = self._place_valence(p.name)
            score = -dist + valence_bias * 3.0
            if score > best_score:
                best_score = score
                best_name = p.name
        return best_name

    def _find_food_source(self, topo: TownTopology) -> str | None:
        """Find a place that provides food."""
        food_places: list[str] = []
        for p in topo.places.values():
            if p.building_key and self._building_provides_food(p.building_key):
                food_places.append(p.name)
        if not food_places:
            return None
        if len(food_places) == 1:
            return food_places[0]
        # Prefer closest.
        food_places.sort(key=lambda n: topo.path_distance(self.location, n) if self.location else 99)
        return food_places[0]

    def _find_social_venue(self, topo: TownTopology, ctx: PerceptionContext) -> str | None:
        """Find a social venue (tavern, park, church, festival_grounds)."""
        for bkey in ("tavern", "park", "church", "festival_grounds"):
            place = self._find_building(bkey, topo)
            if place:
                return place
        return None

    @staticmethod
    def _building_provides_food(building_key: str) -> bool:
        bdef = BUILDING_CATALOG.get(building_key)
        return bdef is not None and bdef.provides_food

    def _pick_social_target(self, ctx: PerceptionContext) -> str:
        """Pick a nearby agent to interact with, preferring positive memory valence."""
        if not ctx.nearby_agents:
            return ""
        if len(ctx.nearby_agents) == 1:
            return ctx.nearby_agents[0]

        # Weight by relationship and memory valence.
        candidates: list[tuple[str, float]] = []
        for name in ctx.nearby_agents:
            weight = 1.0
            # Boost friends.
            rel = self.relationships.get(name)
            if rel in ("friend", "close_friend", "romantic", "married"):
                weight += 2.0
            # Memory valence bias.
            valence = self._person_valence(name)
            weight += valence * 2.0
            weight = max(0.1, weight)
            candidates.append((name, weight))

        return self._weighted_choice(candidates)

    def _place_valence(self, place_name: str) -> float:
        """Return the average mood of memories about a place.  0.0 if none.

        This reads the *absorbed* mood (``valence.valence``), not the tone
        each memory was written with. The engine's own judgements — severity,
        ranking, pruning — read the intrinsic tone so that a concept cannot
        become a "mistake" by keeping bad company; a citizen deciding where
        to go wants the opposite: the tavern concept should carry the mood of
        every evening spent there. This is the consumer of the engine's mood
        propagation, and its ``mood_inertia`` is why an empathetic citizen's
        map of the town changes faster than an analytical one's.
        """
        return self._memory_mood(place_name)

    def _person_valence(self, person_name: str) -> float:
        """Return the average mood of memories about a person.  0.0 if none."""
        return self._memory_mood(person_name)

    def _memory_mood(self, name: str) -> float:
        """Average mood of the memories that actually mention *name*.

        Recall returns the nearest memories whatever they are about, and a
        bad evening at the tavern must not colour the library, so only the
        memories that name the place or person count.
        """
        try:
            results = self.smrti.recall(name, top_k=5, min_confidence=0.1, boost=False)
        except Exception:
            return 0.0
        needle = name.casefold()
        moods = [
            r.atom.valence.valence
            for r in results
            if needle in (r.atom.content or r.atom.label or "").casefold()
        ]
        if not moods:
            return 0.0
        return sum(moods) / len(moods)

    @staticmethod
    def _weighted_choice(items: list[tuple[str, float]]) -> str:
        """Weighted random selection from a list of (value, weight) tuples."""
        if not items:
            return ""
        total = sum(w for _, w in items)
        if total <= 0:
            return items[0][0]
        r = random.random() * total
        cumulative = 0.0
        for value, weight in items:
            cumulative += weight
            if r <= cumulative:
                return value
        return items[-1][0]

    # ── interaction tracking ──────────────────────────────────────────

    def record_interaction(self, other_name: str) -> None:
        self.interaction_counts[other_name] = self.interaction_counts.get(other_name, 0) + 1

    def persist_interactions(self, db) -> None:
        """Save pairwise interaction counts to the DB."""
        rows = [
            (self.name, other, count)
            for other, count in self.interaction_counts.items()
        ]
        if rows:
            db.execute_many(
                "INSERT OR REPLACE INTO citizen_interactions (citizen, other, count) VALUES (?, ?, ?)",
                rows,
            )

    def restore_interactions(self, db) -> None:
        """Reload pairwise interaction counts from the DB."""
        rows = db.fetchall(
            "SELECT other, count FROM citizen_interactions WHERE citizen = ?",
            (self.name,),
        )
        self.interaction_counts = {r["other"]: r["count"] for r in rows} if rows else {}

    # ── serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "personality": self.personality_preset,
            "location": self.location,
            "age_years": round(self.age_years, 2),
            "life_stage": self.life_stage,
            "alive": self.alive,
            "needs": self.needs.to_dict(),
            "skills": self.skills.to_dict(),
            "wallet": self.wallet,
            "traits": self.traits,
            "visual_dna": self.visual_dna,
            "world_pos": list(self.world_pos),
            "moving": self.moving,
            "facing": self.facing,
            "home": self.home,
            "workplace": self.workplace,
            "council_role": self.council_role,
            "relationships": self.relationships,
            "current_action": {
                "type": self.current_action.type,
                "target": self.current_action.target,
            } if self.current_action else None,
        }
