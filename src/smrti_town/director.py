"""Director — adaptive tick pacing and Chronos milestone tracker."""

from __future__ import annotations

import logging
from typing import Any

from smrti_town.config import (
    MILESTONES,
    TICK_MONTAGE,
    TICK_ROUTINE,
    TICK_SCENE,
    TICK_SKIP,
)

log = logging.getLogger(__name__)


class Director:
    """Adaptive tick pacing based on current town activity.

    Modes:
    - ``scene`` (0.25h):  2+ agents together at the same place (social interaction).
    - ``routine`` (2h):   default daily-life pacing.
    - ``montage`` (8h):   everyone sleeping or alone — fast-forward.
    - ``skip`` (168h):    player-requested 1-week jump.
    """

    def __init__(self) -> None:
        self.mode: str = "routine"
        self._skip_requested: bool = False

    def compute_delta(self, agents: list[Any], calendar: Any) -> float:
        """Determine the tick delta in sim-hours based on current agent activity.

        Parameters
        ----------
        agents:
            List of agent objects. Each must have ``location`` (str|None)
            and ``alive`` (bool) attributes.
        calendar:
            SimCalendar instance (used for time_of_day).
        """
        if self._skip_requested:
            self._skip_requested = False
            self.mode = "skip"
            return TICK_SKIP

        alive = [a for a in agents if getattr(a, "alive", True)]
        if not alive:
            self.mode = "routine"
            return TICK_ROUTINE

        # Count occupants per location
        location_counts: dict[str, int] = {}
        for a in alive:
            loc = getattr(a, "location", None)
            if loc:
                location_counts[loc] = location_counts.get(loc, 0) + 1

        # Check for social scene: any location with 2+ agents
        has_social = any(c >= 2 for c in location_counts.values())
        if has_social:
            self.mode = "scene"
            return TICK_SCENE

        # Check for montage: all agents sleeping or alone
        time_of_day = getattr(calendar, "time_of_day", "morning")
        all_solo_or_sleeping = True
        for a in alive:
            loc = getattr(a, "location", None)
            if loc and location_counts.get(loc, 0) > 1:
                all_solo_or_sleeping = False
                break

        if all_solo_or_sleeping and time_of_day == "night":
            self.mode = "montage"
            return TICK_MONTAGE

        self.mode = "routine"
        return TICK_ROUTINE

    def force_skip(self) -> None:
        """Request a 1-week time skip on the next tick."""
        self._skip_requested = True

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "skip_requested": self._skip_requested,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Director:
        director = cls()
        director.mode = data.get("mode", "routine")
        director._skip_requested = data.get("skip_requested", False)
        return director


class Chronos:
    """Milestone and birthday event tracker.

    Checks agent ages against the ``MILESTONES`` table and fires events
    once per agent per milestone. Also detects birthdays (new year-of-age).
    """

    def __init__(self) -> None:
        # (agent_name, milestone_age) tuples that have already been fired.
        self.fired_milestones: set[tuple[str, int]] = set()
        # Track last known year-age per agent for birthday detection.
        self._last_age: dict[str, int] = {}

    def check(self, agents: list[Any], calendar: Any) -> list[dict]:
        """Check all agents for milestone and birthday events.

        Parameters
        ----------
        agents:
            List of agent objects. Each must have ``name`` (str),
            ``age_years`` (int/float), ``alive`` (bool), ``life_stage`` (str).
        calendar:
            SimCalendar instance (unused currently, reserved for anniversary events).

        Returns
        -------
        list[dict]
            List of event dicts: ``{event_type, agent_name, description}``.
        """
        events: list[dict] = []

        for agent in agents:
            if not getattr(agent, "alive", True):
                continue

            name = getattr(agent, "name", "")
            age = int(getattr(agent, "age_years", 0))

            # Birthday detection
            prev_age = self._last_age.get(name)
            if prev_age is not None and age > prev_age:
                events.append({
                    "event_type": "birthday",
                    "agent_name": name,
                    "description": f"{name} turned {age} years old!",
                })
            self._last_age[name] = age

            # Milestone detection
            for milestone_age, event_type in MILESTONES.items():
                if age >= milestone_age and (name, milestone_age) not in self.fired_milestones:
                    self.fired_milestones.add((name, milestone_age))
                    stage = getattr(agent, "life_stage", "adult")
                    desc = self._milestone_description(name, milestone_age, event_type, stage)
                    events.append({
                        "event_type": event_type,
                        "agent_name": name,
                        "description": desc,
                    })

        return events

    def to_dict(self) -> dict:
        return {
            "fired_milestones": sorted(list(pair) for pair in self.fired_milestones),
            "last_age": dict(self._last_age),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Chronos:
        chronos = cls()
        chronos.fired_milestones = {(name, age) for name, age in data.get("fired_milestones", [])}
        chronos._last_age = dict(data.get("last_age", {}))
        return chronos

    @staticmethod
    def _milestone_description(name: str, age: int, event_type: str, life_stage: str) -> str:
        descs = {
            "school_enrollment": f"{name} (age {age}) is old enough to attend school.",
            "adolescence": f"{name} (age {age}) enters adolescence.",
            "graduation": f"{name} (age {age}) graduates and enters adulthood.",
            "career_start": f"{name} (age {age}) is ready to begin a professional career.",
            "retirement": f"{name} (age {age}) retires from active work.",
        }
        return descs.get(event_type, f"{name} reached age {age} milestone: {event_type}.")
