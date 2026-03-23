"""CitizenNeeds — 9-level Maslow hierarchy for citizen drives."""

from __future__ import annotations

from smrti_town.config import (
    ACTUALIZATION_RATE,
    ACTUALIZATION_THRESHOLD,
    ACTION_INTERACT,
    ACTION_MOVE,
    ACTION_PLAY,
    ACTION_PRAY,
    ACTION_SLEEP,
    ACTION_STUDY,
    ACTION_TALK,
    ACTION_WAIT,
    ACTION_WANDER,
    ACTION_WORK,
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

# Actions that count as purposeful work — suppress purpose need rise.
_PURPOSEFUL_ACTIONS: frozenset[str] = frozenset({ACTION_WORK, ACTION_STUDY})
# Actions that count as social engagement — suppress social need rise.
# ACTION_PRAY is intentionally excluded: solitary prayer is not social.
_SOCIAL_ACTIONS: frozenset[str] = frozenset({ACTION_TALK, ACTION_INTERACT, ACTION_PLAY})

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
        current_action: str | None = None,
        nearby_count: int = 0,
    ) -> None:
        """Advance all needs by *delta_hours*.

        Rates are reactive to what the citizen is currently doing and their
        environment — not fixed multipliers.  Only needs listed in the
        life-stage's ``needs`` array are active.
        """
        stage_info = LIFE_STAGES.get(life_stage, LIFE_STAGES["adult"])
        active_needs: set[str] = set(stage_info["needs"])
        energy_mult: float = stage_info.get("energy_decay_mult", 1.0)

        for name in MASLOW_ORDER:
            if name not in active_needs:
                continue

            rate = _RATES[name]

            if name == "shelter":
                # Housed: shelter recovers. Homeless: rises fast.
                if has_home:
                    self.shelter = _clamp(self.shelter - 2.0 * delta_hours)
                    continue
                rate = 2.0

            elif name == "hunger":
                # Exertion burns more energy; rest conserves it.
                if current_action == ACTION_SLEEP:
                    rate *= 0.4 * energy_mult
                elif current_action in (ACTION_MOVE, ACTION_WORK):
                    rate *= 1.4 * energy_mult
                else:
                    rate *= energy_mult

            elif name == "safety":
                rate *= 1.0 + crime_rate * 4.0

            elif name == "social":
                # Being around others slows social deprivation; active
                # social actions essentially freeze it.
                if current_action in _SOCIAL_ACTIONS:
                    rate = 0.0
                else:
                    # Each additional neighbour reduces rise; 4+ neighbours
                    # fully suppress the rise (floor at 0.0).
                    rate *= max(0.0, 1.0 - nearby_count * 0.25)

            elif name == "purpose":
                # Purposeful work prevents need rise; idle citizens drift
                # toward purposelessness; civic roles count as meaningful work.
                if current_action in _PURPOSEFUL_ACTIONS:
                    rate = -rate * 0.5  # actively building meaning
                elif has_job:
                    rate *= 0.2  # employed but not currently working — slow drift
                elif current_action in (ACTION_WAIT, ACTION_WANDER, None):
                    rate *= 1.5  # idle and unemployed: fastest purposelessness
                # else: moving/socialising — neutral drift at base rate

            elif name == "culture":
                if current_action in (ACTION_PLAY, ACTION_PRAY):
                    rate = -rate  # cultural activities actively satisfy

            elif name == "education":
                if current_action == ACTION_STUDY:
                    rate = -rate  # studying actively satisfies

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
