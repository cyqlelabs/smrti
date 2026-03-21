"""AgentDrives: hunger, energy, social, curiosity, duty, romance."""

from __future__ import annotations

from dataclasses import dataclass

from smrti_town.config import (
    CURIOSITY_RATE,
    CURIOSITY_RESET_AMOUNT,
    CURIOSITY_THRESHOLD,
    DRIVE_MAX,
    DRIVE_MIN,
    DUTY_RATE,
    DUTY_THRESHOLD,
    ENERGY_DRAIN_RATE,
    ENERGY_LOW_THRESHOLD,
    HUNGER_RATE,
    HUNGER_RESET,
    HUNGER_THRESHOLD,
    ROMANCE_RATE,
    ROMANCE_RESET_AMOUNT,
    ROMANCE_THRESHOLD,
    SOCIAL_RATE,
    SOCIAL_RESET_AMOUNT,
    SOCIAL_THRESHOLD,
)


def _clamp(value: int, lo: int = DRIVE_MIN, hi: int = DRIVE_MAX) -> int:
    return max(lo, min(hi, value))


@dataclass
class AgentDrives:
    hunger: int = 0
    energy: int = 100
    social: int = 0
    curiosity: int = 0
    duty: int = 0
    romance: int = 0

    # Fractional accumulators — drives are integers but small deltas
    # (e.g. 0.25h scene ticks) produce sub-1.0 increments. We accumulate
    # the fractional remainder here and add whole units once they reach 1.
    _frac_hunger: float = 0.0
    _frac_energy: float = 0.0
    _frac_social: float = 0.0
    _frac_curiosity: float = 0.0
    _frac_duty: float = 0.0
    _frac_romance: float = 0.0

    def _acc(self, frac_attr: str, drive_attr: str, raw: float, sign: int = 1) -> None:
        """Accumulate a fractional drive change, applying whole units."""
        frac = getattr(self, frac_attr) + raw
        whole = int(frac)
        setattr(self, frac_attr, frac - whole)
        if whole != 0:
            setattr(self, drive_attr, _clamp(getattr(self, drive_attr) + sign * whole))

    def accumulate(
        self,
        delta_hours: float,
        *,
        is_work_hours: bool = False,
        is_adult: bool = True,
        energy_decay_mult: float = 1.0,
        active_drives: list[str] | None = None,
    ) -> None:
        """Time-weighted drive accumulation."""
        allowed = set(active_drives) if active_drives else {
            "hunger", "energy", "social", "curiosity", "duty", "romance",
        }

        if "hunger" in allowed:
            self._acc("_frac_hunger", "hunger", HUNGER_RATE * delta_hours)
        if "energy" in allowed:
            self._acc("_frac_energy", "energy", ENERGY_DRAIN_RATE * delta_hours * energy_decay_mult, sign=-1)
        if "social" in allowed:
            self._acc("_frac_social", "social", SOCIAL_RATE * delta_hours)
        if "curiosity" in allowed:
            self._acc("_frac_curiosity", "curiosity", CURIOSITY_RATE * delta_hours)
        if "duty" in allowed and is_work_hours and is_adult:
            self._acc("_frac_duty", "duty", DUTY_RATE * delta_hours)
        if "romance" in allowed and is_adult:
            self._acc("_frac_romance", "romance", ROMANCE_RATE * delta_hours)

    def reset_hunger(self) -> None:
        self.hunger = HUNGER_RESET

    def reset_energy(self) -> None:
        self.energy = _clamp(100)

    def reduce_social(self) -> None:
        self.social = _clamp(self.social - SOCIAL_RESET_AMOUNT)

    def reduce_curiosity(self) -> None:
        self.curiosity = _clamp(self.curiosity - CURIOSITY_RESET_AMOUNT)

    def reduce_romance(self) -> None:
        self.romance = _clamp(self.romance - ROMANCE_RESET_AMOUNT)

    def reset_duty(self) -> None:
        self.duty = 0

    def highest_urgent_drive(self, active_drives: list[str] | None = None) -> str | None:
        """Return the name of the highest-priority drive above its threshold, or None."""
        allowed = set(active_drives) if active_drives else {
            "hunger", "energy", "social", "curiosity", "duty", "romance",
        }
        candidates: list[tuple[int, str]] = []
        if "energy" in allowed and self.energy <= ENERGY_LOW_THRESHOLD:
            candidates.append((100 - self.energy, "energy"))
        if "hunger" in allowed and self.hunger >= HUNGER_THRESHOLD:
            candidates.append((self.hunger, "hunger"))
        if "duty" in allowed and self.duty >= DUTY_THRESHOLD:
            candidates.append((self.duty, "duty"))
        if "social" in allowed and self.social >= SOCIAL_THRESHOLD:
            candidates.append((self.social, "social"))
        if "romance" in allowed and self.romance >= ROMANCE_THRESHOLD:
            candidates.append((self.romance, "romance"))
        if "curiosity" in allowed and self.curiosity >= CURIOSITY_THRESHOLD:
            candidates.append((self.curiosity, "curiosity"))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def to_dict(self) -> dict:
        return {
            "hunger": self.hunger,
            "energy": self.energy,
            "social": self.social,
            "curiosity": self.curiosity,
            "duty": self.duty,
            "romance": self.romance,
        }
