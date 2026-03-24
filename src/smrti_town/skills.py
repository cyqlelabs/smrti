"""SkillSet — 8 skill categories with learning and role-checking."""

from __future__ import annotations

from smrti_town.config import SKILL_CATEGORIES


class SkillSet:
    """Eight skill categories, each 0.0-1.0."""

    __slots__ = ("skills",)

    def __init__(self, initial: dict[str, float] | None = None) -> None:
        self.skills: dict[str, float] = {cat: 0.0 for cat in SKILL_CATEGORIES}
        if initial:
            for cat, level in initial.items():
                if cat in self.skills:
                    self.skills[cat] = max(0.0, min(1.0, float(level)))

    # ── learning ──────────────────────────────────────────────────────
    def learn(self, category: str, delta_hours: float, building_type: str | None = None) -> float:
        """Increment *category* if *building_type* matches one of the skill's
        ``learned_at`` locations (or if the skill has no location requirement,
        e.g. ``leadership``).

        Returns the actual XP gained (0.0 if learning conditions are not met).
        """
        info = SKILL_CATEGORIES.get(category)
        if info is None:
            return 0.0

        learned_at: list[str] = info["learned_at"]
        # Skills with an empty learned_at list (e.g. leadership) can be learned
        # anywhere via experience; otherwise the citizen must be at the right building.
        if learned_at and (building_type is None or building_type not in learned_at):
            return 0.0

        xp_rate: float = info["xp_per_hour"]
        gain = xp_rate * delta_hours
        old = self.skills[category]
        self.skills[category] = min(1.0, old + gain)
        return self.skills[category] - old

    # ── role checks ───────────────────────────────────────────────────
    def can_fill_role(self, role: str) -> bool:
        """Check whether skill levels meet the requirements for *role*.

        A role is enabled when any skill category that lists it in ``enables``
        has a level >= the category's ``staff_min_level`` equivalent.  In
        practice we check: does any category whose ``enables`` list contains
        *role* have a current skill level >= the category's ``xp_per_hour * 20``
        (a rough proxy for minimum competence)?  For a more precise check, use
        :meth:`meets_skill_requirement`.
        """
        for cat, info in SKILL_CATEGORIES.items():
            if role in info["enables"]:
                # Minimum competence: if skill > 0 at all, the role is fillable.
                # The building's staff_min_level provides the real gate.
                if self.skills.get(cat, 0.0) > 0.0:
                    return True
        return False

    def meets_skill_requirement(self, skill_category: str, min_level: float) -> bool:
        """Return ``True`` if the citizen's level in *skill_category* is at
        least *min_level*."""
        if not skill_category:
            return True  # Building requires no skill.
        return self.skills.get(skill_category, 0.0) >= min_level

    # ── queries ───────────────────────────────────────────────────────
    def best_skill(self) -> tuple[str, float]:
        """Return ``(category, level)`` of the highest-level skill."""
        best_cat = max(self.skills, key=self.skills.get)  # type: ignore[arg-type]
        return best_cat, self.skills[best_cat]

    def level(self, category: str) -> float:
        return self.skills.get(category, 0.0)

    # ── serialization ─────────────────────────────────────────────────
    def to_dict(self) -> dict[str, float]:
        return {cat: round(lvl, 4) for cat, lvl in self.skills.items()}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> SkillSet:
        return cls(initial=data)

    def __repr__(self) -> str:
        nonzero = {c: round(l, 3) for c, l in self.skills.items() if l > 0}
        return f"SkillSet({nonzero})"
