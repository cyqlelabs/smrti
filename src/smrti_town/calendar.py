"""SimCalendar: tracks sim-time (hours, days, seasons, years)."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """Tracks simulation time with a compressed calendar.

    28 days per year, 4 seasons of 7 days each, 24 hours per day.
    """

    total_hours: float = 0.0

    def advance(self, delta_hours: float) -> None:
        self.total_hours += delta_hours

    @property
    def hour_of_day(self) -> float:
        return self.total_hours % HOURS_PER_DAY

    @property
    def day(self) -> int:
        """Day within the current year (0-based)."""
        return int(self.total_hours // HOURS_PER_DAY) % DAYS_PER_YEAR

    @property
    def day_total(self) -> int:
        """Total days elapsed since start."""
        return int(self.total_hours // HOURS_PER_DAY)

    @property
    def year(self) -> int:
        return int(self.total_hours // HOURS_PER_YEAR)

    @property
    def season_index(self) -> int:
        return self.day // DAYS_PER_SEASON

    @property
    def season(self) -> str:
        return SEASONS[min(self.season_index, len(SEASONS) - 1)]

    @property
    def day_of_season(self) -> int:
        return self.day % DAYS_PER_SEASON

    def time_of_day(self) -> str:
        h = self.hour_of_day
        for name, (start, end) in TIME_OF_DAY_RANGES.items():
            if start <= h < end:
                return name
        return "night"

    def to_years(self, hours: float) -> float:
        return hours / HOURS_PER_YEAR

    def is_work_hours(self) -> bool:
        h = self.hour_of_day
        return 8.0 <= h < 17.0

    def is_school_hours(self) -> bool:
        h = self.hour_of_day
        return 8.0 <= h < 14.0

    def is_sleep_time(self) -> bool:
        h = self.hour_of_day
        return h >= 22.0 or h < 6.0

    def format_time(self) -> str:
        hour = int(self.hour_of_day)
        minute = int((self.hour_of_day % 1) * 60)
        return f"{hour:02d}:{minute:02d}"

    def format_date(self) -> str:
        return (
            f"Day {self.day + 1}, {self.season.capitalize()}, Year {self.year + 1} "
            f"({self.format_time()})"
        )

    def to_dict(self) -> dict:
        return {
            "hour": round(self.hour_of_day, 2),
            "day": self.day + 1,
            "day_total": self.day_total,
            "season": self.season,
            "year": self.year + 1,
            "time_of_day": self.time_of_day(),
            "formatted": self.format_date(),
        }
