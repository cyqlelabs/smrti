"""Sporadic random events: weather, accidents, illness, found items, etc."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from smrti_town.config import OUTDOOR_PLACES, SPORADIC_EVENTS, TICK_ROUTINE

if TYPE_CHECKING:
    from smrti_town.agent import Agent
    from smrti_town.spatial import Place, TownTopology


@dataclass
class SporadicEvent:
    event_id: str
    description: str
    location: str
    affected_agents: list[str] = field(default_factory=list)
    valence: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": "sporadic",
            "event_id": self.event_id,
            "description": self.description,
            "location": self.location,
            "affected_agents": self.affected_agents,
        }


# Valence mapping for event types
_EVENT_VALENCE = {
    "weather_rain": -0.1,
    "weather_sunny": 0.3,
    "weather_wind": -0.05,
    "accident_trip": -0.4,
    "found_item": 0.4,
    "illness_mild": -0.3,
    "surprise_visitor": 0.3,
    "animal_encounter": 0.3,
    "gossip": 0.1,
    "power_outage": -0.2,
    "festival": 0.5,
    "strange_noise": -0.1,
}

# Energy effects for some events
_EVENT_ENERGY_EFFECT = {
    "accident_trip": -10,
    "illness_mild": -15,
    "found_item": 0,
    "festival": -5,
}

# Items granted by found_item events
_FOUND_ITEMS = [
    "shiny coin",
    "old book",
    "wildflower",
    "smooth stone",
    "feather",
    "wooden button",
]


def generate_sporadic_events(
    agents: list[Agent],
    topology: TownTopology,
    delta_hours: float,
    current_season: str,
) -> list[SporadicEvent]:
    """Generate random sporadic events for the current tick.

    Probability is scaled by delta_hours relative to routine tick (2h).
    """
    events: list[SporadicEvent] = []
    prob_scale = delta_hours / TICK_ROUTINE

    # Build location -> alive agents mapping
    loc_agents: dict[str, list[str]] = {}
    for agent in agents:
        if agent.alive:
            loc_agents.setdefault(agent.location, []).append(agent.name)

    for event_def in SPORADIC_EVENTS:
        # Season modifiers
        base_prob = event_def["prob"] * prob_scale
        base_prob = _season_modifier(event_def["id"], base_prob, current_season)

        if random.random() > base_prob:
            continue

        # Pick a location
        if event_def["outdoor_only"]:
            candidate_locs = [
                loc for loc in loc_agents
                if loc in OUTDOOR_PLACES or topology.places.get(loc, None) and topology.places[loc].is_outdoor
            ]
        else:
            candidate_locs = list(loc_agents.keys())

        if not candidate_locs:
            continue

        location = random.choice(candidate_locs)
        agents_here = loc_agents.get(location, [])
        if not agents_here:
            continue

        # Pick affected agent(s)
        if event_def["affects_all"]:
            affected = agents_here[:]
        else:
            affected = [random.choice(agents_here)]

        # Pick template and format
        template = random.choice(event_def["templates"])
        agent_name = affected[0] if affected else "Someone"
        description = template.format(
            agent=agent_name,
            location=_pretty_location(location),
        )

        metadata: dict = {}
        if event_def["id"] == "found_item":
            item = random.choice(_FOUND_ITEMS)
            metadata["item"] = item

        events.append(SporadicEvent(
            event_id=event_def["id"],
            description=description,
            location=location,
            affected_agents=affected,
            valence=_EVENT_VALENCE.get(event_def["id"], 0.0),
            metadata=metadata,
        ))

    return events


def apply_sporadic_effects(
    event: SporadicEvent,
    agents_by_name: dict[str, Agent],
) -> None:
    """Apply mechanical effects of a sporadic event to agents."""
    energy_effect = _EVENT_ENERGY_EFFECT.get(event.event_id, 0)

    for agent_name in event.affected_agents:
        agent = agents_by_name.get(agent_name)
        if not agent or not agent.alive:
            continue

        # Energy effect
        if energy_effect:
            agent.drives.energy = max(0, min(100, agent.drives.energy + energy_effect))

        # Found item — add to inventory
        if event.event_id == "found_item" and event.metadata.get("item"):
            agent.inventory.append(event.metadata["item"])

        # Illness — reduce energy over time (already applied via energy_effect)
        # Gossip — boost social (they heard something interesting)
        if event.event_id == "gossip":
            agent.drives.social = max(0, agent.drives.social - 10)

        # Festival — reduce social need, boost curiosity
        if event.event_id == "festival":
            agent.drives.social = max(0, agent.drives.social - 20)
            agent.drives.curiosity = max(0, agent.drives.curiosity - 10)

    # Write memories for affected agents
    for agent_name in event.affected_agents:
        agent = agents_by_name.get(agent_name)
        if not agent or not agent.alive:
            continue
        try:
            agent.smrti.remember(
                content=event.description,
                type="episode",
                valence=event.valence,
                metadata={"event_type": event.event_id},
            )
        except Exception:
            pass


def _season_modifier(event_id: str, base_prob: float, season: str) -> float:
    """Adjust event probability based on season."""
    if event_id == "weather_rain":
        if season == "autumn":
            return base_prob * 2.0
        if season == "summer":
            return base_prob * 0.5
    if event_id == "weather_sunny":
        if season == "summer":
            return base_prob * 2.0
        if season == "winter":
            return base_prob * 0.3
    if event_id == "festival":
        if season == "summer":
            return base_prob * 3.0
        if season == "winter":
            return base_prob * 2.0  # holiday festivals
    if event_id == "animal_encounter":
        if season == "spring":
            return base_prob * 2.0
        if season == "winter":
            return base_prob * 0.3
    if event_id == "illness_mild":
        if season == "winter":
            return base_prob * 2.5
        if season == "summer":
            return base_prob * 0.5
    return base_prob


def _pretty_location(location: str) -> str:
    """Convert place ID to a readable name."""
    return location.replace("_", " ")
