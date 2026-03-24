"""CitizenNeeds — 9-level Maslow hierarchy for citizen drives."""

from __future__ import annotations

from smrti_town.config import (
    ACTUALIZATION_RATE,
    ACTUALIZATION_THRESHOLD,
    CULTURE_RATE,
    CULTURE_RESET_AMOUNT,
    CULTURE_THRESHOLD,
    EDUCATION_RATE,
    EDUCATION_RESET_AMOUNT,
    EDUCATION_THRESHOLD,
    HEALTH_RATE,
    HEALTH_THRESHOLD,
    HUNGER_RATE,
    HUNGER_RESET,
    HUNGER_THRESHOLD,
    LIFE_STAGES,
    NEED_MAX,
    NEED_MIN,
    PURPOSE_RATE,
    PURPOSE_THRESHOLD,
    SAFETY_RATE,
    SAFETY_THRESHOLD,
    SHELTER_RATE,
    SHELTER_THRESHOLD,
    SOCIAL_RATE,
    SOCIAL_RESET_AMOUNT,
    SOCIAL_THRESHOLD,
)

# Maslow priority order (index 0 = highest priority).
MASLOW_ORDER: list[str] = [
    "hunger",
    "shelter",
    "health",
    "safety",
    "social",
    "education",
    "purpose",
    "culture",
    "actualization",
]

_RATES: dict[str, float] = {
    "hunger": HUNGER_RATE,
    "shelter": SHELTER_RATE,
    "health": HEALTH_RATE,
    "safety": SAFETY_RATE,
    "social": SOCIAL_RATE,
    "education": EDUCATION_RATE,
    "purpose": PURPOSE_RATE,
    "culture": CULTURE_RATE,
    "actualization": ACTUALIZATION_RATE,
}

_THRESHOLDS: dict[str, float] = {
    "hunger": HUNGER_THRESHOLD,
    "shelter": SHELTER_THRESHOLD,
    "health": HEALTH_THRESHOLD,
    "safety": SAFETY_THRESHOLD,
    "social": SOCIAL_THRESHOLD,
    "education": EDUCATION_THRESHOLD,
    "purpose": PURPOSE_THRESHOLD,
    "culture": CULTURE_THRESHOLD,
    "actualization": ACTUALIZATION_THRESHOLD,
}

_RESET_AMOUNTS: dict[str, float] = {
    "hunger": HUNGER_RESET,
    "social": SOCIAL_RESET_AMOUNT,
    "education": EDUCATION_RESET_AMOUNT,
    "culture": CULTURE_RESET_AMOUNT,
}


def _clamp(v: float) -> float:
    return max(NEED_MIN, min(NEED_MAX, v))


class CitizenNeeds:
    """Nine Maslow needs, each 0-100.  Higher value = more deprived."""

    __slots__ = tuple(MASLOW_ORDER)

    def __init__(self) -> None:
        for name in MASLOW_ORDER:
            setattr(self, name, 0.0)

    # ── tick ──────────────────────────────────────────────────────────
    def tick(
        self,
        delta_hours: float,
        life_stage: str,
        has_home: bool,
        has_job: bool,
        crime_rate: float = 0.0,
    ) -> None:
        """Advance all needs by *delta_hours*.

        Only needs listed in the life-stage's ``needs`` array are active;
        others stay at their current value (never decay for irrelevant stages).
        """
        stage_info = LIFE_STAGES.get(life_stage, LIFE_STAGES["adult"])
        active_needs: set[str] = set(stage_info["needs"])
        energy_mult: float = stage_info.get("energy_decay_mult", 1.0)

        for name in MASLOW_ORDER:
            if name not in active_needs:
                continue

            rate = _RATES[name]

            # Special-case adjustments
            if name == "shelter":
                # Shelter need only rises when homeless.
                if has_home:
                    self.shelter = _clamp(self.shelter - 2.0 * delta_hours)
                    continue
                rate = 2.0  # Homeless: shelter need rises fast.
            elif name == "safety":
                rate = rate * (1.0 + crime_rate * 4.0)
            elif name == "purpose":
                if not has_job:
                    rate *= 1.5  # Unemployed adults feel purposelessness faster.
            elif name == "hunger":
                rate *= energy_mult

            current = getattr(self, name)
            setattr(self, name, _clamp(current + rate * delta_hours))

    # ── queries ───────────────────────────────────────────────────────
    def highest_unmet_need(self, life_stage: str = "adult") -> str | None:
        """Return the highest-priority need above its threshold, respecting
        Maslow ordering: lower-priority needs only matter if all higher-priority
        needs are below threshold.

        Returns ``None`` when every active need is satisfied.
        """
        stage_info = LIFE_STAGES.get(life_stage, LIFE_STAGES["adult"])
        active_needs: set[str] = set(stage_info["needs"])

        for name in MASLOW_ORDER:
            if name not in active_needs:
                continue
            if getattr(self, name) >= _THRESHOLDS[name]:
                return name
        return None

    def need_value(self, name: str) -> float:
        return getattr(self, name, 0.0)

    def need_urgency(self, name: str) -> float:
        """0.0 = fully satisfied, 1.0 = maximally deprived."""
        return getattr(self, name, 0.0) / NEED_MAX

    # ── mutations ─────────────────────────────────────────────────────
    def satisfy(self, need_name: str, amount: float | None = None) -> None:
        """Reduce *need_name* by *amount*, or to its reset value if *amount*
        is ``None`` and a reset constant exists."""
        if amount is not None:
            current = getattr(self, need_name, 0.0)
            setattr(self, need_name, _clamp(current - amount))
        elif need_name in _RESET_AMOUNTS:
            reset = _RESET_AMOUNTS[need_name]
            if need_name == "hunger":
                setattr(self, need_name, _clamp(reset))
            else:
                current = getattr(self, need_name, 0.0)
                setattr(self, need_name, _clamp(current - reset))
        else:
            setattr(self, need_name, NEED_MIN)

    # ── serialization ─────────────────────────────────────────────────
    def to_dict(self) -> dict[str, float]:
        return {name: round(getattr(self, name), 2) for name in MASLOW_ORDER}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> CitizenNeeds:
        n = cls()
        for name in MASLOW_ORDER:
            if name in data:
                setattr(n, name, _clamp(float(data[name])))
        return n

    def __repr__(self) -> str:
        parts = [f"{n}={getattr(self, n):.0f}" for n in MASLOW_ORDER]
        return f"CitizenNeeds({', '.join(parts)})"
