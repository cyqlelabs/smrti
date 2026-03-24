"""Simulation calendar — tracks hour, day, season, year."""

from __future__ import annotations

from dataclasses import dataclass

from smrti_town.config import (
    DAYS_PER_SEASON,
    DAYS_PER_YEAR,
    HOURS_PER_DAY,
    HOURS_PER_YEAR,
    SEASONS,
    TIME_OF_DAY_RANGES,
)


@dataclass
class SimCalendar:
    total_hours: float = 0.0

    @property
    def hour(self) -> float:
        return self.total_hours % HOURS_PER_DAY

    @property
    def day(self) -> int:
        return int(self.total_hours // HOURS_PER_DAY) % DAYS_PER_YEAR + 1

    @property
    def year(self) -> int:
        return int(self.total_hours // HOURS_PER_YEAR) + 1

    @property
    def season_index(self) -> int:
        day_of_year = int(self.total_hours // HOURS_PER_DAY) % DAYS_PER_YEAR
        return day_of_year // DAYS_PER_SEASON

    @property
    def season(self) -> str:
        return SEASONS[self.season_index]

    @property
    def time_of_day(self) -> str:
        h = self.hour
        for name, (start, end) in TIME_OF_DAY_RANGES.items():
            if start <= h < end:
                return name
        return "night"

    def advance(self, delta_hours: float) -> None:
        self.total_hours += delta_hours

    def to_dict(self) -> dict:
        return {
            "total_hours": round(self.total_hours, 2),
            "hour": round(self.hour, 2),
            "day": self.day,
            "year": self.year,
            "season": self.season,
            "time_of_day": self.time_of_day,
        }
