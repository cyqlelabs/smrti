"""EventManager — organic events, crises, and sporadic happenings."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from typing import Any

from smrti_town.config import (
    CRISIS_EVENTS,
    EVENT_VALENCE,
    NEED_MAX,
    SPORADIC_EVENTS,
)


@dataclass
class GameEvent:
    event_type: str
    description: str
    affected_citizens: list[str] = field(default_factory=list)
    affected_buildings: list[str] = field(default_factory=list)
    effects: dict = field(default_factory=dict)  # e.g. {"treasury": -1000, "health": -0.2}
    tick_number: int = 0
    expires_hours: float = 0.0  # crises only: sim time at which the crisis is over


class EventManager:
    """Tracks organic events, crises, and sporadic happenings."""

    def __init__(self) -> None:
        self.active_crises: list[GameEvent] = []
        self.event_history: list[GameEvent] = []

    # ── crisis ──────────────────────────────────────────────────────────

    def roll_crisis(
        self,
        town_state: dict,
        tick_number: int,
    ) -> GameEvent | None:
        """Probabilistic crisis roll from CRISIS_EVENTS.

        *town_state* should contain:
            - existing_buildings: list[str]  (building_keys)
            - population: int
            - treasury: int
            - citizens: list (citizen names or objects)
            - building_places: list[str] (place names)

        Returns a GameEvent if a crisis triggers, else None.
        """
        existing = set(town_state.get("existing_buildings", []))
        population = town_state.get("population", 0)
        building_places = town_state.get("building_places", [])
        citizen_names = []
        for c in town_state.get("citizens", []):
            if isinstance(c, str):
                citizen_names.append(c)
            else:
                name = getattr(c, "name", "")
                if name:
                    citizen_names.append(name)

        for crisis_def in CRISIS_EVENTS:
            prob = crisis_def["prob"]
            # Scale probability slightly with population (more people = more chaos).
            scaled_prob = prob * (1.0 + population * 0.005)

            if random.random() > scaled_prob:
                continue

            crisis_id = crisis_def["id"]
            mitigator = crisis_def.get("mitigated_by")
            has_mitigation = mitigator in existing if mitigator else False

            # Build effects based on crisis type.
            effects: dict = {}
            affected_citizens: list[str] = []
            affected_buildings: list[str] = []

            if crisis_id == "fire":
                effects["treasury"] = -2000
                if building_places and not has_mitigation:
                    affected_buildings = [random.choice(building_places)]
                    effects["building_destroyed"] = True
                if citizen_names:
                    n_affected = random.randint(1, min(3, len(citizen_names)))
                    affected_citizens = random.sample(citizen_names, n_affected)
                    effects["health"] = -0.3

            elif crisis_id == "epidemic":
                n_sick = max(1, population // 3)
                if citizen_names:
                    affected_citizens = random.sample(
                        citizen_names, min(n_sick, len(citizen_names))
                    )
                effects["health"] = -0.4 if not has_mitigation else -0.15
                effects["treasury"] = -1000

            elif crisis_id == "drought":
                effects["food_shortage"] = True
                effects["treasury"] = -1500
                if citizen_names:
                    affected_citizens = list(citizen_names)  # Everyone affected.
                effects["hunger"] = 0.3 if not has_mitigation else 0.1

            elif crisis_id == "crime_wave":
                effects["safety"] = -0.4 if not has_mitigation else -0.15
                effects["treasury"] = -800
                if citizen_names:
                    n_victims = max(1, population // 5)
                    affected_citizens = random.sample(
                        citizen_names, min(n_victims, len(citizen_names))
                    )

            elif crisis_id == "economic_downturn":
                effects["treasury"] = -3000 if not has_mitigation else -1000
                effects["commerce_reduction"] = 0.5 if not has_mitigation else 0.2
                if citizen_names:
                    affected_citizens = list(citizen_names)

            description = crisis_def.get("description", f"Crisis: {crisis_id}")
            if has_mitigation:
                description += f" (mitigated by {mitigator})"

            event = GameEvent(
                event_type=f"crisis_{crisis_id}",
                description=description,
                affected_citizens=affected_citizens,
                affected_buildings=affected_buildings,
                effects=effects,
                tick_number=tick_number,
            )
            self.active_crises.append(event)
            self.event_history.append(event)
            return event

        return None

    # ── sporadic events ─────────────────────────────────────────────────

    def roll_sporadic(
        self,
        agents_by_place: dict[str, list[str]],
        tick_number: int,
    ) -> list[GameEvent]:
        """Roll for random small events from SPORADIC_EVENTS.

        *agents_by_place* — {place_name: [citizen_names_present]}.
        Returns list of triggered events.
        """
        events: list[GameEvent] = []

        for sev in SPORADIC_EVENTS:
            if random.random() > sev["prob"]:
                continue

            outdoor_only = sev.get("outdoor_only", False)
            affects_all = sev.get("affects_all", False)

            # Pick a location with agents present.
            occupied = {
                place: agents
                for place, agents in agents_by_place.items()
                if agents
            }
            if not occupied:
                continue

            location = random.choice(list(occupied.keys()))
            agents_present = occupied[location]

            # Skip outdoor-only events if we can't determine outdoorness
            # (all places are treated as potentially outdoor-eligible here;
            # the engine can refine this with place metadata).

            if affects_all:
                affected = list(agents_present)
            else:
                affected = [random.choice(agents_present)]

            # Pick a template and fill in placeholders.
            template = random.choice(sev["templates"])
            agent_name = affected[0] if affected else "Someone"
            description = template.replace("{location}", location).replace("{agent}", agent_name)

            # Build effects.
            effects: dict = {}
            eid = sev["id"]
            if eid == "accident_trip":
                effects["health"] = -0.05
            elif eid == "found_item":
                effects["wallet"] = 5
            elif eid == "illness_mild":
                effects["health"] = -0.1
            elif eid == "surprise_visitor":
                effects["social"] = 0.1
                effects["commerce_boost"] = 0.05
            elif eid == "gossip":
                effects["social"] = 0.05

            event = GameEvent(
                event_type=eid,
                description=description,
                affected_citizens=affected,
                effects=effects,
                tick_number=tick_number,
            )
            events.append(event)
            self.event_history.append(event)

        return events

    # ── effects ─────────────────────────────────────────────────────────

    @staticmethod
    def apply_effects(event: GameEvent, citizens_by_name: dict[str, Any], economy: Any) -> list[tuple[Any, str, float]]:
        """Apply an event to the town and return the episode every citizen it
        touched remembers it by.

        Effects are fractions of a need: ``health``, ``safety`` and ``social``
        name the good quantity (negative is harm), ``hunger`` names the
        deprivation itself, so a food shortage's ``hunger: 0.3`` raises it.
        """
        fx = event.effects
        if economy is not None and fx.get("treasury"):
            economy.treasury = max(0, economy.treasury + int(fx["treasury"]))
        tone = EVENT_VALENCE.get(event.event_type, 0.0)
        text = event.description
        if event.affected_buildings:
            text = f"{text} {', '.join(event.affected_buildings)} burned down."
        experiences: list[tuple[Any, str, float]] = []
        for name in event.affected_citizens:
            c = citizens_by_name.get(name)
            if c is None or not c.alive:
                continue
            for need, sign in (("health", -1), ("safety", -1), ("social", -1), ("hunger", 1)):
                if need in fx:
                    deprived = getattr(c.needs, need) + sign * fx[need] * NEED_MAX
                    setattr(c.needs, need, max(0.0, min(NEED_MAX, deprived)))
            if economy is not None and fx.get("wallet") and name in economy.wallets:
                economy.wallets[name] += int(fx["wallet"])
            if tone:
                experiences.append((c, text, tone))
        return experiences

    def crime_rate(self, citizens: list[Any], topology: Any) -> float:
        """How unsafe the town is: adults with nothing to do, and a crime
        wave while one is active, halved by a constabulary. The safety need
        rises with it."""
        adults = [c for c in citizens if c.life_stage == "adult"]
        idle = sum(1 for c in adults if c.workplace is None and c.council_role is None)
        rate = 0.3 * idle / len(adults) if adults else 0.0
        if any(e.event_type == "crisis_crime_wave" for e in self.active_crises):
            rate += 0.5
        if topology.places_by_building("constabulary"):
            rate *= 0.5
        return min(1.0, rate)

    # ── crisis resolution ───────────────────────────────────────────────

    def resolve_crisis(self, crisis: GameEvent, has_mitigation: bool) -> dict:
        """Resolve an active crisis.  Returns outcome dict."""
        if crisis in self.active_crises:
            self.active_crises.remove(crisis)

        outcome: dict = {"event_type": crisis.event_type, "resolved": True}

        if has_mitigation:
            # Mitigated: reduce negative effects by 60%.
            outcome["severity"] = "mitigated"
            outcome["effects"] = {}
            for key, val in crisis.effects.items():
                if isinstance(val, (int, float)) and val < 0:
                    outcome["effects"][key] = val * 0.4  # Only 40% of damage remains.
                else:
                    outcome["effects"][key] = val
        else:
            outcome["severity"] = "full"
            outcome["effects"] = dict(crisis.effects)

        return outcome

    # ── serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        def _event_dict(e: GameEvent) -> dict:
            return {
                "event_type": e.event_type,
                "description": e.description,
                "affected_citizens": e.affected_citizens,
                "affected_buildings": e.affected_buildings,
                "effects": e.effects,
                "tick_number": e.tick_number,
                "expires_hours": e.expires_hours,
            }

        return {
            "active_crises": [_event_dict(c) for c in self.active_crises],
            "event_history": [_event_dict(e) for e in self.event_history[-50:]],
        }

    @classmethod
    def from_dict(cls, data: dict) -> EventManager:
        em = cls()
        for e_data in data.get("active_crises", []):
            em.active_crises.append(GameEvent(**e_data))
        for e_data in data.get("event_history", []):
            em.event_history.append(GameEvent(**e_data))
        return em
