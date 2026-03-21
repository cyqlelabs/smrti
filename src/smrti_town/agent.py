"""Agent: wraps drives + location + inventory + age + smrti instance."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from smrti import Smrti

from smrti_town.config import (
    ACTION_EAT,
    ACTION_INTERACT,
    ACTION_MOVE,
    ACTION_PROPOSE,
    ACTION_REPRODUCE,
    ACTION_SLEEP,
    ACTION_STUDY,
    ACTION_TALK,
    ACTION_WAIT,
    ACTION_WANDER,
    ACTION_WORK,
    CURIOSITY_TOPICS,
    FOOD_TOPICS,
    GREETINGS,
    LIFE_STAGES,
    PERSONALITY_ACTION_BIAS,
    PRESET_TRAITS,
    ROMANTIC_LINES,
    SMALL_TALK,
    TRAIT_NAMES,
)
from smrti_town.drives import AgentDrives


@dataclass
class Action:
    """Structured action output from agent decision."""

    type: str
    target: Optional[str] = None
    dialogue: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"type": self.type}
        if self.target:
            d["target"] = self.target
        if self.dialogue:
            d["dialogue"] = self.dialogue
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class PerceptionContext:
    """Gathered context for decision making."""

    location: str
    time_of_day: str
    season: str
    nearby_agents: list[str]
    urgent_drive: Optional[str]
    memories: list[dict]
    schedule_obligation: Optional[str]
    personality_preset: str

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "time_of_day": self.time_of_day,
            "season": self.season,
            "nearby_agents": self.nearby_agents,
            "urgent_drive": self.urgent_drive,
            "schedule_obligation": self.schedule_obligation,
            "personality": self.personality_preset,
        }


class Agent:
    """A person in the town. Wraps drives (Python) + memory (Smrti)."""

    def __init__(
        self,
        name: str,
        personality: str = "balanced",
        location: str = "Central_Park",
        age_years: float = 25.0,
        db_path: str = "~/.smrti/town.db",
        tenant_id: str = "millbrook",
        parents: tuple[str, str] | None = None,
        traits: dict[str, float] | None = None,
    ) -> None:
        self.name = name
        self.personality_preset = personality
        self.location = location
        self.age_hours: float = age_years * 672.0  # HOURS_PER_YEAR
        self.drives = AgentDrives()
        self.inventory: list[str] = []
        self.alive: bool = True
        self.parents = parents
        self.last_milestone_year: int = int(age_years)
        self.starvation_hours: float = 0.0
        self._interaction_counts: dict[str, int] = {}  # target_name -> count
        self._db_path = db_path
        self._tenant_id = tenant_id

        # Behavioural traits — heritable floats 0..1 that modulate decisions.
        # Initialised from preset if not explicitly provided.
        self.traits: dict[str, float] = traits or dict(
            PRESET_TRAITS.get(personality, PRESET_TRAITS["balanced"])
        )

        # Smrti instance — write to agent's private space, read from
        # agent space + world + culture
        self.smrti = Smrti(
            db_path=db_path,
            personality=personality,
            tenant_id=tenant_id,
            write_space=f"Agent_Space_{name}",
            read_spaces=[
                f"Agent_Space_{name}",
                "World_Space",
                "Space_Culture",
            ],
        )

    # ── Properties ───────────────────────────────────────────────────

    @property
    def age_years(self) -> float:
        return self.age_hours / 672.0

    @property
    def life_stage(self) -> str:
        years = self.age_years
        for stage, info in LIFE_STAGES.items():
            lo, hi = info["age_range"]
            if lo <= years < hi:
                return stage
        return "elder"

    @property
    def life_stage_info(self) -> dict:
        return LIFE_STAGES.get(self.life_stage, LIFE_STAGES["elder"])

    @property
    def active_drives(self) -> list[str]:
        return self.life_stage_info.get("drives", ["hunger", "energy", "social"])

    @property
    def can_move(self) -> bool:
        return self.life_stage_info.get("can_move", False)

    @property
    def can_talk(self) -> bool:
        return self.life_stage_info.get("can_talk", False)

    @property
    def energy_decay_mult(self) -> float:
        return self.life_stage_info.get("energy_decay_mult", 1.0)

    def effective_action_bias(self) -> dict[str, float]:
        """Compute action bias by blending preset defaults with behavioural traits.

        Traits modulate the preset bias:
          - shyness     lowers social bias
          - proactivity raises all non-social biases slightly
          - laziness    lowers duty bias
          - adventurous raises wander bias
          - creativity  raises curiosity bias
          - nurturing   raises romance & social bias
          - leadership  raises social bias (initiator tendency)
        """
        base = dict(PERSONALITY_ACTION_BIAS.get(
            self.personality_preset, PERSONALITY_ACTION_BIAS["balanced"]
        ))
        t = self.traits

        # Shyness inverts social tendency
        base["social"] = max(0.0, min(1.0,
            base["social"] * (1.0 - t.get("shyness", 0.3) * 0.8)
            + t.get("leadership", 0.4) * 0.3
            + t.get("nurturing", 0.5) * 0.2
        ))

        # Laziness suppresses duty
        base["duty"] = max(0.0, min(1.0,
            base["duty"] * (1.0 - t.get("laziness", 0.3) * 0.7)
            + t.get("proactivity", 0.5) * 0.15
        ))

        # Adventurousness and creativity boost wander and curiosity
        base["wander"] = max(0.0, min(1.0,
            base["wander"] + t.get("adventurous", 0.4) * 0.4
        ))
        base["curiosity"] = max(0.0, min(1.0,
            base["curiosity"] + t.get("creativity", 0.5) * 0.2
        ))

        # Nurturing boosts romance
        base["romance"] = max(0.0, min(1.0,
            base["romance"] + t.get("nurturing", 0.5) * 0.2
        ))

        return base

    def increment_interaction(self, target_name: str) -> None:
        self._interaction_counts[target_name] = (
            self._interaction_counts.get(target_name, 0) + 1
        )

    def get_interaction_count(self, target_name: str) -> int:
        return self._interaction_counts.get(target_name, 0)

    # ── Perception ───────────────────────────────────────────────────

    def perceive(
        self,
        nearby_agents: list[str],
        time_of_day: str,
        season: str,
        calendar_hour: float,
    ) -> PerceptionContext:
        """Gather context for decision-making."""
        if not self.alive or not self.can_talk:
            return PerceptionContext(
                location=self.location,
                time_of_day=time_of_day,
                season=season,
                nearby_agents=nearby_agents,
                urgent_drive=None,
                memories=[],
                schedule_obligation=None,
                personality_preset=self.personality_preset,
            )

        # Check schedule obligations
        schedule_obligation = self._check_schedule(calendar_hour)

        # Query smrti for relevant memories
        query = f"I am at {self.location}. It is {time_of_day}, {season}."
        if nearby_agents:
            query += f" {', '.join(nearby_agents)} are here."
        memories = []
        try:
            recall_results = self.smrti.recall(query=query, top_k=5)
            for r in recall_results:
                memories.append({
                    "content": r.atom.content or r.atom.label,
                    "salience": r.salience,
                    "valence": r.atom.valence.valence if r.atom.valence else 0.0,
                })
        except Exception:
            pass

        urgent_drive = self.drives.highest_urgent_drive(self.active_drives)

        return PerceptionContext(
            location=self.location,
            time_of_day=time_of_day,
            season=season,
            nearby_agents=nearby_agents,
            urgent_drive=urgent_drive,
            memories=memories,
            schedule_obligation=schedule_obligation,
            personality_preset=self.personality_preset,
        )

    # ── Decision ─────────────────────────────────────────────────────

    def decide(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
    ) -> Action:
        """Rule-based decision system. No LLM calls.

        Priority order:
        1. Sleep if exhausted or nighttime
        2. Schedule obligations (school/work)
        3. Highest urgent drive above threshold
        4. Personality-influenced idle action
        5. Random wandering
        """
        if not self.alive:
            return Action(type=ACTION_WAIT)
        if not self.can_talk and not self.can_move:
            return Action(type=ACTION_WAIT, dialogue="(infant)")

        # 1. Sleep — very low energy or nighttime
        if self.drives.energy <= 10 or (
            ctx.time_of_day == "night" and self.drives.energy < 60
        ):
            return self._decide_sleep(ctx, available_places)

        # 2. Schedule obligations
        if ctx.schedule_obligation:
            return self._decide_schedule(ctx, available_places)

        # 3. Urgent drive
        if ctx.urgent_drive:
            return self._decide_drive(ctx, available_places, place_agents)

        # 4. Personality-biased idle action
        return self._decide_idle(ctx, available_places, place_agents)

    # ── Private decision helpers ─────────────────────────────────────

    def _decide_sleep(self, ctx: PerceptionContext, available_places: list[str]) -> Action:
        home = f"{self.name.split('_')[0]}_Home" if "_" not in self.name else f"{self.name}_Home"
        # Try to find a home-like place
        home_candidates = [p for p in available_places if "Home" in p]
        if home in available_places and self.location != home:
            return Action(type=ACTION_MOVE, target=home, dialogue="Time to head home and rest.")
        if home_candidates and self.location not in home_candidates:
            return Action(type=ACTION_MOVE, target=home_candidates[0], dialogue="Going home to sleep.")
        return Action(type=ACTION_SLEEP, dialogue=f"{self.name} falls asleep.")

    def _decide_schedule(self, ctx: PerceptionContext, available_places: list[str]) -> Action:
        if ctx.schedule_obligation == "school":
            if "Public_Library" in available_places and self.location != "Public_Library":
                return Action(type=ACTION_MOVE, target="Public_Library", dialogue="Heading to school.")
            return Action(type=ACTION_STUDY, dialogue=f"{self.name} studies diligently.")
        if ctx.schedule_obligation == "work":
            work_places = [p for p in available_places if p in ("Town_Market", "Cafe_Rosetta", "Public_Library")]
            if work_places and self.location not in work_places:
                target = random.choice(work_places)
                return Action(type=ACTION_MOVE, target=target, dialogue="Heading to work.")
            return Action(type=ACTION_WORK, dialogue=f"{self.name} focuses on work.")
        return Action(type=ACTION_WAIT)

    def _decide_drive(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
    ) -> Action:
        drive = ctx.urgent_drive
        bias = self.effective_action_bias()

        if drive == "hunger":
            return self._decide_eat(ctx, available_places)
        if drive == "energy":
            return self._decide_sleep(ctx, available_places)
        if drive == "social":
            return self._decide_social(ctx, available_places, place_agents, bias)
        if drive == "curiosity":
            return self._decide_curiosity(ctx, available_places, bias)
        if drive == "duty":
            return self._decide_schedule(
                PerceptionContext(
                    location=ctx.location,
                    time_of_day=ctx.time_of_day,
                    season=ctx.season,
                    nearby_agents=ctx.nearby_agents,
                    urgent_drive=ctx.urgent_drive,
                    memories=ctx.memories,
                    schedule_obligation="work",
                    personality_preset=ctx.personality_preset,
                ),
                available_places,
            )
        if drive == "romance":
            return self._decide_romance(ctx, available_places, place_agents, bias)

        return Action(type=ACTION_WAIT)

    def _decide_eat(self, ctx: PerceptionContext, available_places: list[str]) -> Action:
        food_places = [p for p in available_places if p in ("Cafe_Rosetta", "Town_Market")]
        # Check memories for food-related places
        for mem in ctx.memories:
            content = mem.get("content", "")
            for place in available_places:
                if place in content and ("food" in content.lower() or "eat" in content.lower() or "bread" in content.lower()):
                    if self.location != place:
                        return Action(type=ACTION_MOVE, target=place, dialogue="I remember there's food there.")

        if food_places and self.location not in food_places:
            target = random.choice(food_places)
            return Action(type=ACTION_MOVE, target=target, dialogue=random.choice(FOOD_TOPICS))
        return Action(
            type=ACTION_EAT,
            dialogue=random.choice(FOOD_TOPICS),
        )

    def _decide_social(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
        bias: dict,
    ) -> Action:
        shyness = self.traits.get("shyness", 0.3)

        # Talk to someone nearby (shy agents are less likely to initiate)
        if ctx.nearby_agents and random.random() > shyness * 0.6:
            # Leadership trait: leaders pick the most-connected agent
            if self.traits.get("leadership", 0.4) > 0.6:
                scored = [(a, self.get_interaction_count(a)) for a in ctx.nearby_agents]
                scored.sort(key=lambda x: x[1])
                target = scored[0][0]  # least known — leaders reach out
            else:
                target = random.choice(ctx.nearby_agents)
            dialogue = self._generate_social_dialogue(target, ctx)
            return Action(type=ACTION_TALK, target=target, dialogue=dialogue)

        # Shy agents avoid crowded places; seek smaller groups or solitude
        social_bias = bias.get("social", 0.5)
        populated = [
            (p, agents) for p, agents in place_agents.items()
            if agents and p != self.location and p in available_places
        ]
        if populated and random.random() < social_bias:
            if shyness > 0.6:
                # Shy: prefer the place with fewest people (still not empty)
                populated.sort(key=lambda x: len(x[1]))
            else:
                populated.sort(key=lambda x: len(x[1]), reverse=True)
            target_place = populated[0][0]
            return Action(type=ACTION_MOVE, target=target_place, dialogue="Going to find some company.")

        # Go to a social hub
        social_places = [p for p in available_places if p in ("Cafe_Rosetta", "Central_Park")]
        if social_places and self.location not in social_places:
            return Action(type=ACTION_MOVE, target=random.choice(social_places), dialogue="Heading out to socialize.")

        return Action(type=ACTION_WAIT, dialogue=f"{self.name} waits, hoping someone comes by.")

    def _decide_curiosity(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        bias: dict,
    ) -> Action:
        curiosity_bias = bias.get("curiosity", 0.5)
        if "Public_Library" in available_places and self.location != "Public_Library" and random.random() < curiosity_bias:
            return Action(type=ACTION_MOVE, target="Public_Library", dialogue="Time to learn something new.")
        if self.location == "Public_Library":
            return Action(type=ACTION_STUDY, dialogue=random.choice(CURIOSITY_TOPICS))
        # Wander to explore
        if random.random() < bias.get("wander", 0.3):
            candidates = [p for p in available_places if p != self.location]
            if candidates:
                target = random.choice(candidates)
                return Action(type=ACTION_WANDER, target=target, dialogue="Exploring the area.")
        return Action(type=ACTION_INTERACT, dialogue=random.choice(CURIOSITY_TOPICS))

    def _decide_romance(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
        bias: dict,
    ) -> Action:
        # Look for romantic partners nearby
        romance_bias = bias.get("romance", 0.5)
        if ctx.nearby_agents and random.random() < romance_bias:
            # Prefer agents with high interaction count
            scored = [
                (a, self.get_interaction_count(a))
                for a in ctx.nearby_agents
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            target = scored[0][0]
            dialogue = random.choice(ROMANTIC_LINES).format(target=target)
            return Action(type=ACTION_TALK, target=target, dialogue=dialogue)

        # Move toward social places to meet people
        social_places = [p for p in available_places if p in ("Cafe_Rosetta", "Central_Park")]
        if social_places and self.location not in social_places:
            return Action(type=ACTION_MOVE, target=random.choice(social_places), dialogue="Going somewhere nice.")

        return Action(type=ACTION_WAIT, dialogue=f"{self.name} daydreams.")

    def _decide_idle(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
    ) -> Action:
        bias = self.effective_action_bias()
        stubbornness = self.traits.get("stubbornness", 0.3)
        proactivity = self.traits.get("proactivity", 0.5)

        # Proactive agents are more likely to do something even when idle
        idle_threshold = 0.5 * (1.0 - proactivity * 0.6)

        # Talk to nearby agents if social-leaning personality
        if ctx.nearby_agents and random.random() < bias.get("social", 0.5):
            target = random.choice(ctx.nearby_agents)
            dialogue = self._generate_social_dialogue(target, ctx)
            return Action(type=ACTION_TALK, target=target, dialogue=dialogue)

        # Stubborn agents are less likely to leave their current location
        wander_chance = bias.get("wander", 0.3) * (1.0 - stubbornness * 0.5)
        if random.random() < wander_chance:
            candidates = [p for p in available_places if p != self.location]
            if candidates:
                target = random.choice(candidates)
                return Action(type=ACTION_WANDER, target=target, dialogue="Just taking a walk.")

        # Study if curiosity-leaning (proactive agents don't need to be at the library)
        curiosity_chance = bias.get("curiosity", 0.5) * 0.3
        if random.random() < curiosity_chance:
            if self.location == "Public_Library" or proactivity > 0.7:
                return Action(type=ACTION_STUDY, dialogue=random.choice(CURIOSITY_TOPICS))

        # Lazy agents are content to do nothing
        if random.random() < idle_threshold:
            return Action(type=ACTION_WAIT, dialogue=f"{self.name} relaxes.")

        return Action(type=ACTION_WAIT, dialogue=f"{self.name} relaxes.")

    def _generate_social_dialogue(self, target: str, ctx: PerceptionContext) -> str:
        """Generate plausible dialogue based on context and memories."""
        # Check for relevant memories about the target
        for mem in ctx.memories:
            content = mem.get("content", "")
            if target.lower() in content.lower() and mem.get("salience", 0) > 0.3:
                return f"Hey {target}, I was thinking about what you said last time."

        # Greeting if low interaction count
        if self.get_interaction_count(target) < 3:
            return random.choice(GREETINGS).format(target=target)

        # Context-dependent dialogue
        if ctx.season == "summer":
            summer_topics = [
                f"Beautiful day, isn't it {target}?",
                f"{target}, have you been enjoying the summer weather?",
            ]
            return random.choice(summer_topics + SMALL_TALK)
        if ctx.season == "winter":
            winter_topics = [
                f"Stay warm out there, {target}.",
                f"{target}, are you looking forward to the holidays?",
            ]
            return random.choice(winter_topics + SMALL_TALK)

        return random.choice(SMALL_TALK)

    def _check_schedule(self, calendar_hour: float) -> str | None:
        schedule = self.life_stage_info.get("schedule")
        if not schedule:
            return None
        for obligation, (start, end) in schedule.items():
            if start <= calendar_hour < end:
                return obligation
        return None

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "location": self.location,
            "age_years": round(self.age_years, 1),
            "life_stage": self.life_stage,
            "alive": self.alive,
            "personality": self.personality_preset,
            "traits": {k: round(v, 3) for k, v in self.traits.items()},
            "drives": self.drives.to_dict(),
            "inventory": self.inventory,
            "parents": list(self.parents) if self.parents else None,
        }
