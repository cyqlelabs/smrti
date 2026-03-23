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
    LIFE_STAGES,
    PERSONALITY_ACTION_BIAS,
    PRESET_TRAITS,
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
    place_types: dict = field(default_factory=dict)

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
        traits: dict | None = None,
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
        self._place_types: dict[str, str] = {}
        self.mood_valence: float = 0.0
        self._relationships: list[dict] = []
        self._regression_cooldown: dict[str, bool] = {}  # target_name → True while cooling
        self.traits: dict[str, float] = (
            traits if traits is not None
            else dict(PRESET_TRAITS.get(personality, PRESET_TRAITS["balanced"]))
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

    def increment_interaction(self, target_name: str) -> None:
        self._interaction_counts[target_name] = (
            self._interaction_counts.get(target_name, 0) + 1
        )

    def get_interaction_count(self, target_name: str) -> int:
        return self._interaction_counts.get(target_name, 0)

    def effective_action_bias(self) -> dict[str, float]:
        """Return action bias dict, blending personality preset with all traits."""
        from smrti_town.config import PERSONALITY_ACTION_BIAS
        base = dict(PERSONALITY_ACTION_BIAS.get(
            self.personality_preset,
            PERSONALITY_ACTION_BIAS["balanced"],
        ))
        shyness = self.traits.get("shyness", 0.3)
        adventurous = self.traits.get("adventurous", 0.4)
        laziness = self.traits.get("laziness", 0.3)
        leadership = self.traits.get("leadership", 0.4)
        creativity = self.traits.get("creativity", 0.5)
        nurturing = self.traits.get("nurturing", 0.5)
        stubbornness = self.traits.get("stubbornness", 0.3)
        # Shyness reduces social, leadership amplifies it
        base["social"] = max(0.0, min(1.0, base.get("social", 0.5) * (1.0 - shyness * 0.5) * (1.0 + leadership * 0.3)))
        # Adventurous boosts wander, stubbornness suppresses it
        base["wander"] = max(0.0, min(1.0, base.get("wander", 0.3) * (1.0 + adventurous * 0.5) * (1.0 - stubbornness * 0.4)))
        # Creativity amplifies curiosity-driven actions
        base["curiosity"] = max(0.0, min(1.0, base.get("curiosity", 0.5) * (1.0 + creativity * 0.4)))
        # Laziness suppresses duty/work
        base["duty"] = max(0.0, min(1.0, base.get("duty", 0.5) * (1.0 - laziness * 0.5)))
        # Nurturing amplifies romance/social warmth
        base["romance"] = max(0.0, min(1.0, base.get("romance", 0.5) * (1.0 + nurturing * 0.2)))
        return base

    def _memory_valence_for_agent(self, agent_name: str, memories: list[dict]) -> float:
        """Return average valence of memories that mention a given agent name."""
        vals = [m["valence"] for m in memories if agent_name in (m.get("content") or "")]
        return sum(vals) / len(vals) if vals else 0.0

    def _memory_preferred_location(self, candidates: list[str], memories: list[dict]) -> str | None:
        """Return the candidate location with the strongest positive memory association."""
        scores: dict[str, float] = {}
        for m in memories:
            content = (m.get("content") or "").lower()
            for place in candidates:
                key = place.lower().replace("_", " ")
                if key in content or place.lower() in content:
                    scores[place] = scores.get(place, 0.0) + m.get("valence", 0.0)
        if scores:
            best = max(scores, key=lambda k: scores[k])
            if scores[best] > 0.2:
                return best
        return None

    def persist_interactions(self) -> None:
        """Save interaction counts directly to the Smrti DB (single transaction)."""
        import json
        label = f"__interactions__{self.name}"
        data = json.dumps(self._interaction_counts)
        try:
            db = self.smrti.db
            db.execute_batch([
                (
                    "DELETE FROM atoms WHERE label = ? AND tenant_id = ? AND space = ?",
                    (label, self._tenant_id, f"Agent_Space_{self.name}"),
                ),
                (
                    "INSERT INTO atoms (id, type, label, content, tenant_id, space, probability, confidence) "
                    "VALUES (?, 'belief', ?, ?, ?, ?, 1.0, 1.0)",
                    (label, label, data, self._tenant_id, f"Agent_Space_{self.name}"),
                ),
            ])
        except Exception:
            pass

    def update_relationships(self, all_agents: list[Agent]) -> None:
        """Cache relationship states for all known agents (called at epoch time)."""
        from smrti_town.lifecycle import _infer_relationship_state
        _STATE_VALENCE = {
            "married": 0.8, "romantic": 0.7, "close_friend": 0.5,
            "friend": 0.3, "acquaintance": 0.1,
        }
        rels = []
        for other in all_agents:
            if other.name == self.name or not other.alive:
                continue
            count = self.get_interaction_count(other.name)
            if count == 0:
                continue
            state = _infer_relationship_state(self, other)
            rels.append({
                "name": other.name,
                "state": state,
                "count": count,
                "valence": _STATE_VALENCE.get(state, 0.0),
            })
        self._relationships = rels

    def restore_interactions(self) -> None:
        """Load interaction counts directly from the Smrti DB."""
        import json
        label = f"__interactions__{self.name}"
        try:
            db = self.smrti.db
            rows = db.fetchall(
                "SELECT content FROM atoms WHERE label = ? AND tenant_id = ? AND space = ?",
                (label, self._tenant_id, f"Agent_Space_{self.name}"),
            )
            if rows:
                self._interaction_counts = json.loads(rows[0]["content"])
        except Exception:
            pass

    # ── Perception ───────────────────────────────────────────────────

    def perceive(
        self,
        nearby_agents: list[str],
        time_of_day: str,
        season: str,
        calendar_hour: float,
        place_types: dict[str, str] | None = None,
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
                place_types=place_types or {},
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
            if memories:
                self.mood_valence = sum(m["valence"] for m in memories) / len(memories)
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
            place_types=place_types or {},
        )

    # ── Decision ─────────────────────────────────────────────────────

    def decide(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
        place_types: dict[str, str] | None = None,
    ) -> Action:
        """Rule-based decision system. No LLM calls.

        Priority order:
        1. Sleep if exhausted or nighttime
        2. Schedule obligations (school/work)
        3. Highest urgent drive above threshold
        4. Personality-influenced idle action
        5. Random wandering
        """
        if place_types is not None:
            self._place_types = place_types
            ctx = PerceptionContext(
                location=ctx.location,
                time_of_day=ctx.time_of_day,
                season=ctx.season,
                nearby_agents=ctx.nearby_agents,
                urgent_drive=ctx.urgent_drive,
                memories=ctx.memories,
                schedule_obligation=ctx.schedule_obligation,
                personality_preset=ctx.personality_preset,
                place_types=place_types,
            )
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
        home_candidates = [p for p in available_places if ctx.place_types.get(p) == "home"]
        if home_candidates and self.location not in home_candidates:
            return Action(type=ACTION_MOVE, target=home_candidates[0])
        return Action(type=ACTION_SLEEP)

    def _decide_schedule(self, ctx: PerceptionContext, available_places: list[str]) -> Action:
        if ctx.schedule_obligation == "school":
            school_places = [p for p in available_places if ctx.place_types.get(p) == "public"]
            if school_places and self.location not in school_places:
                return Action(type=ACTION_MOVE, target=school_places[0])
            return Action(type=ACTION_STUDY)
        if ctx.schedule_obligation == "work":
            work_places = [p for p in available_places if ctx.place_types.get(p) == "public"]
            if work_places and self.location not in work_places:
                target = random.choice(work_places)
                return Action(type=ACTION_MOVE, target=target)
            return Action(type=ACTION_WORK)
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
                    place_types=ctx.place_types,
                ),
                available_places,
            )
        if drive == "romance":
            return self._decide_romance(ctx, available_places, place_agents, bias)

        return Action(type=ACTION_WAIT)

    def _decide_eat(self, ctx: PerceptionContext, available_places: list[str]) -> Action:
        food_places = [p for p in available_places if ctx.place_types.get(p) == "public"]
        if food_places and self.location not in food_places:
            target = random.choice(food_places)
            return Action(type=ACTION_MOVE, target=target)
        return Action(type=ACTION_EAT)

    def _decide_social(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
        bias: dict,
    ) -> Action:
        # Talk to someone nearby — prefer agents with positive memory association
        if ctx.nearby_agents:
            stubbornness = self.traits.get("stubbornness", 0.3)
            candidates = []
            for name in ctx.nearby_agents:
                mem_val = self._memory_valence_for_agent(name, ctx.memories)
                # Stubborn agents avoid those they remember negatively
                if mem_val < -0.3 and random.random() < stubbornness:
                    continue
                candidates.append((name, mem_val))
            if not candidates:
                candidates = [(name, 0.0) for name in ctx.nearby_agents]
            candidates.sort(key=lambda x: x[1], reverse=True)
            return Action(type=ACTION_TALK, target=candidates[0][0])

        # Move toward populated places
        social_bias = bias.get("social", 0.5)
        populated = [
            (p, agents) for p, agents in place_agents.items()
            if agents and p != self.location and p in available_places
        ]
        if populated and random.random() < social_bias:
            # Prefer the place with the most people, biased by positive memories
            populated.sort(key=lambda x: len(x[1]), reverse=True)
            target_place = populated[0][0]
            mem_place = self._memory_preferred_location(
                [p for p, _ in populated], ctx.memories
            )
            if mem_place:
                target_place = mem_place
            return Action(type=ACTION_MOVE, target=target_place)

        # Go to a social hub — prefer one with positive memories
        social_places = [p for p in available_places if ctx.place_types.get(p) in ("public", "outdoor")]
        if social_places and self.location not in social_places:
            mem_place = self._memory_preferred_location(social_places, ctx.memories)
            return Action(type=ACTION_MOVE, target=mem_place or random.choice(social_places))

        return Action(type=ACTION_WAIT)

    def _decide_curiosity(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        bias: dict,
    ) -> Action:
        curiosity_bias = bias.get("curiosity", 0.5)
        curiosity_places = [p for p in available_places if ctx.place_types.get(p) == "public"]
        if curiosity_places and self.location not in curiosity_places and random.random() < curiosity_bias:
            return Action(type=ACTION_MOVE, target=random.choice(curiosity_places))
        if ctx.place_types.get(self.location) == "public":
            return Action(type=ACTION_STUDY)
        # Wander to explore
        if random.random() < bias.get("wander", 0.3):
            candidates = [p for p in available_places if p != self.location]
            if candidates:
                target = random.choice(candidates)
                return Action(type=ACTION_WANDER, target=target)
        return Action(type=ACTION_INTERACT)

    def _decide_romance(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
        bias: dict,
    ) -> Action:
        # Look for romantic partners nearby — weight by interaction history and memory valence
        romance_bias = bias.get("romance", 0.5)
        if ctx.nearby_agents and random.random() < romance_bias:
            scored = []
            for a in ctx.nearby_agents:
                mem_val = self._memory_valence_for_agent(a, ctx.memories)
                # Combine interaction count and memory valence
                score = self.get_interaction_count(a) + mem_val * 5
                scored.append((a, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            target = scored[0][0]
            return Action(type=ACTION_TALK, target=target)

        # Move toward social places to meet people
        social_places = [p for p in available_places if ctx.place_types.get(p) in ("public", "outdoor")]
        if social_places and self.location not in social_places:
            return Action(type=ACTION_MOVE, target=random.choice(social_places))

        return Action(type=ACTION_WAIT)

    def _decide_idle(
        self,
        ctx: PerceptionContext,
        available_places: list[str],
        place_agents: dict[str, list[str]],
    ) -> Action:
        bias = self.effective_action_bias()

        # Children: prefer to stay near a parent
        if self.life_stage == "child" and self.parents:
            for place, occupants in place_agents.items():
                if place == self.location:
                    continue
                for parent_name in self.parents:
                    if parent_name and parent_name in occupants and place in available_places:
                        if random.random() < 0.55:
                            return Action(type=ACTION_MOVE, target=place)

        # Talk to nearby agents if social-leaning personality
        if ctx.nearby_agents and random.random() < bias.get("social", 0.5):
            target = random.choice(ctx.nearby_agents)
            return Action(type=ACTION_TALK, target=target)

        # Wander if wander-leaning
        if random.random() < bias.get("wander", 0.3):
            candidates = [p for p in available_places if p != self.location]
            if candidates:
                target = random.choice(candidates)
                return Action(type=ACTION_WANDER, target=target)

        # Study if curiosity-leaning
        if random.random() < bias.get("curiosity", 0.5) * 0.3:
            if ctx.place_types.get(self.location) == "public":
                return Action(type=ACTION_STUDY)

        return Action(type=ACTION_WAIT)

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
            "drives": self.drives.to_dict(),
            "inventory": self.inventory,
            "parents": list(self.parents) if self.parents else None,
            "mood_valence": round(self.mood_valence, 2),
            "relationships": self._relationships,
        }
