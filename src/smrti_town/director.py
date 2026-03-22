"""Director: tick pacing.  Chronos: milestone and birthday detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from smrti_town.config import (
    HOURS_PER_YEAR,
    MILESTONES,
    TICK_MONTAGE,
    TICK_ROUTINE,
    TICK_SCENE,
    TICK_SKIP,
)

if TYPE_CHECKING:
    from smrti_town.agent import Agent
    from smrti_town.calendar import SimCalendar
    from smrti_town.spatial import Place


# ── Data carriers ────────────────────────────────────────────────────

@dataclass
class SystemEvent:
    agent_name: str
    event_type: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "type": "milestone",
            "agent": self.agent_name,
            "event_type": self.event_type,
            "detail": self.detail,
        }


# ── Director ─────────────────────────────────────────────────────────

class Director:
    """Decides how much sim-time each tick consumes."""

    def __init__(self) -> None:
        self._skip_requested: bool = False
        self.mode: str = "routine"

    def request_skip(self) -> None:
        self._skip_requested = True

    def compute_tick_delta(
        self,
        agents: list[Agent],
        places: dict[str, Place],
    ) -> float:
        if self._skip_requested:
            self._skip_requested = False
            self.mode = "skip"
            return TICK_SKIP

        # Scene mode: 2+ agents at the same non-home place
        alive_names = {a.name for a in agents if a.alive}
        for place in places.values():
            # Sleeping together at home is montage, not scene
            if place.place_type == "home":
                continue
            living_occupants = [n for n in place.occupants if n in alive_names]
            if len(living_occupants) >= 2:
                self.mode = "scene"
                return TICK_SCENE

        # Montage mode: everyone sleeping or solo
        alive_agents = [a for a in agents if a.alive]
        if alive_agents and all(
            a.drives.energy < 10 or not a.life_stage_info.get("can_talk", False)
            for a in alive_agents
        ):
            self.mode = "montage"
            return TICK_MONTAGE

        self.mode = "routine"
        return TICK_ROUTINE


# ── Chronos ──────────────────────────────────────────────────────────

class Chronos:
    """Fires milestone and birthday events based on agent ages."""

    def check_milestones(
        self,
        agents: list[Agent],
        cal: SimCalendar,
    ) -> list[SystemEvent]:
        events: list[SystemEvent] = []
        for agent in agents:
            if not agent.alive:
                continue
            age_years = cal.to_years(agent.age_hours)
            for year, milestone in MILESTONES.items():
                if agent.last_milestone_year < year <= age_years:
                    detail = _milestone_detail(agent.name, milestone, year)
                    events.append(SystemEvent(
                        agent_name=agent.name,
                        event_type=milestone,
                        detail=detail,
                    ))
                    agent.last_milestone_year = year
        return events

    def check_birthdays(
        self,
        agents: list[Agent],
        cal: SimCalendar,
        delta_hours: float,
    ) -> list[SystemEvent]:
        events: list[SystemEvent] = []
        for agent in agents:
            if not agent.alive:
                continue
            prev_years = int(cal.to_years(agent.age_hours - delta_hours))
            curr_years = int(cal.to_years(agent.age_hours))
            if curr_years > prev_years and curr_years > 0:
                events.append(SystemEvent(
                    agent_name=agent.name,
                    event_type="birthday",
                    detail=f"Today is {agent.name}'s birthday! They are now {curr_years} years old.",
                ))
        return events


def _milestone_detail(name: str, milestone: str, year: int) -> str:
    details = {
        "school_enrollment": f"{name} starts school at age {year}.",
        "adolescence": f"{name} enters adolescence at age {year}.",
        "graduation": f"{name} graduates at age {year}.",
        "career_start": f"{name} begins their career at age {year}.",
        "retirement": f"{name} retires at age {year}.",
    }
    return details.get(milestone, f"{name} reaches milestone '{milestone}' at age {year}.")
