"""PopulationManager — immigration, natural growth, and departure."""

from __future__ import annotations

import random

from smrti_town.config import (
    BUILDING_CATALOG,
    HOUSING_IMMIGRANT_PROFILES,
    IMMIGRATION_BASE_PROBABILITY,
    LIFE_STAGES,
    MAX_POPULATION,
    PARAM_BOUNDS,
    PERSONALITY_PARAMS,
    PRESET_TRAITS,
    RELATIONSHIP_GATES,
    REPRODUCTION_GATE,
    SATISFACTION_EXODUS_THRESHOLD,
    STARTING_WALLET,
    STRESS_VARIANCE_BASE,
    STRESS_VARIANCE_MAX_MULT,
    TRAIT_BOUNDS,
    TRAIT_NAMES,
)

# ── Name pools (language-agnostic via index — callers can override) ────
_GIVEN_NAMES = [
    "Aiden", "Briar", "Calla", "Dain", "Elara", "Fenn", "Gale",
    "Hazel", "Ilan", "Jory", "Kael", "Linden", "Maren", "Noel",
    "Orin", "Petra", "Quinn", "Rowan", "Sage", "Theron", "Una",
    "Vale", "Wren", "Xan", "Yara", "Zephyr", "Ash", "Blythe",
    "Cedar", "Darcy", "Ember", "Frost", "Glen", "Hollis", "Iris",
    "Jade", "Kai", "Lark", "Moss", "Nym", "Olive", "Pike",
]

_SURNAMES = [
    "Ashford", "Birch", "Clay", "Dale", "Elm", "Fern", "Grove",
    "Heath", "Ivy", "Juniper", "Knoll", "Locke", "Marsh", "North",
    "Oak", "Pine", "Reed", "Stone", "Thorn", "Vale", "Weld",
    "Brook", "Croft", "Dusk", "Flint", "Hale", "Kern", "Lea",
]


def _pick_name(existing: set[str], surname: str | None = None) -> str:
    """Generate a unique name not already in *existing*.

    If *surname* is provided, the generated name uses that surname so
    family members share a last name.
    """
    for _ in range(200):
        first = random.choice(_GIVEN_NAMES)
        last = surname or random.choice(_SURNAMES)
        name = f"{first} {last}"
        if name not in existing:
            return name
    # Fallback with numeric suffix.
    base = f"{random.choice(_GIVEN_NAMES)} {surname or random.choice(_SURNAMES)}"
    suffix = random.randint(100, 999)
    return f"{base} {suffix}"


def _pick_personality() -> str:
    return random.choice(list(PRESET_TRAITS.keys()))


def _random_traits(personality: str) -> dict[str, float]:
    """Generate traits seeded from a personality preset with small jitter."""
    base = PRESET_TRAITS.get(personality, PRESET_TRAITS["balanced"])
    traits: dict[str, float] = {}
    for name in TRAIT_NAMES:
        lo, hi = TRAIT_BOUNDS[name]
        val = base.get(name, 0.5) + random.gauss(0, 0.1)
        traits[name] = max(lo, min(hi, val))
    return traits


def _random_skills() -> dict[str, float]:
    """Generate a sparse skill set for a new immigrant."""
    from smrti_town.config import SKILL_CATEGORIES
    skills: dict[str, float] = {}
    # Each immigrant starts with 1-3 skills at low levels.
    chosen = random.sample(list(SKILL_CATEGORIES.keys()), k=min(3, len(SKILL_CATEGORIES)))
    for sk in chosen:
        skills[sk] = round(random.uniform(0.05, 0.35), 3)
    return skills


class PopulationManager:
    """Handles immigration, natural growth, and citizen departure."""

    def compute_pull_factors(
        self,
        citizens: list,
        buildings: list,
        economy,
    ) -> dict[str, float]:
        """Compute attractiveness scores for potential immigrants.

        Returns dict with keys: housing_available, employment_available,
        services_quality, reputation.  Each 0.0 - 1.0.
        """
        alive = [c for c in citizens if getattr(c, "alive", True)]
        pop = len(alive)

        # Housing availability: count empty housing capacity.
        total_capacity = 0
        housed = 0
        for b in buildings:
            bdef = BUILDING_CATALOG.get(getattr(b, "building_key", None) or "", None)
            if bdef and bdef.provides_housing:
                total_capacity += bdef.capacity
        for c in alive:
            if getattr(c, "home", None):
                housed += 1
        free_housing = max(0, total_capacity - housed)
        housing_score = min(1.0, free_housing / max(1, pop * 0.3))

        # Employment: count businesses that need staff.
        jobs_available = 0
        employed_count = 0
        for b in buildings:
            bdef = BUILDING_CATALOG.get(getattr(b, "building_key", None) or "", None)
            if bdef and bdef.staff_required > 0:
                jobs_available += bdef.staff_required
        for c in alive:
            if getattr(c, "workplace", None):
                employed_count += 1
        free_jobs = max(0, jobs_available - employed_count)
        employment_score = min(1.0, free_jobs / max(1, pop * 0.2))

        # Services: civic and cultural buildings relative to population.
        service_count = 0
        for b in buildings:
            bdef = BUILDING_CATALOG.get(getattr(b, "building_key", None) or "", None)
            if bdef and bdef.category in ("civic", "cultural"):
                service_count += 1
        services_score = min(1.0, service_count / max(1, pop * 0.1))

        # Reputation: treasury health + average citizen wealth.
        treasury = getattr(economy, "treasury", 0)
        avg_wallet = 0
        if alive:
            wallets = getattr(economy, "wallets", {})
            wallet_sum = sum(wallets.get(getattr(c, "name", ""), 0) for c in alive)
            avg_wallet = wallet_sum / len(alive)
        treasury_health = min(1.0, treasury / 50000)
        wealth_health = min(1.0, avg_wallet / (STARTING_WALLET * 3))
        reputation = (treasury_health + wealth_health) / 2.0

        return {
            "housing_available": round(housing_score, 3),
            "employment_available": round(employment_score, 3),
            "services_quality": round(services_score, 3),
            "reputation": round(reputation, 3),
        }

    def check_immigration(
        self,
        pull_factors: dict[str, float],
        available_housing: list[str],
    ) -> dict | None:
        """Probabilistic immigration check.

        *available_housing* — list of building_keys with free capacity
        (e.g. ["cottage", "house"]).

        Returns a family spec dict or None if no immigration occurs.
        """
        if not available_housing:
            return None

        # Aggregate pull score.
        score = (
            pull_factors.get("housing_available", 0) * 0.3
            + pull_factors.get("employment_available", 0) * 0.3
            + pull_factors.get("services_quality", 0) * 0.2
            + pull_factors.get("reputation", 0) * 0.2
        )

        prob = IMMIGRATION_BASE_PROBABILITY * score
        if random.random() > prob:
            return None

        # Pick a housing type from what's available.
        housing_type = random.choice(available_housing)
        profile = HOUSING_IMMIGRANT_PROFILES.get(
            housing_type,
            HOUSING_IMMIGRANT_PROFILES.get("cottage", {"adults": 1, "children": 0}),
        )

        adults_spec = profile.get("adults", 1)
        children_spec = profile.get("children", 0)

        if isinstance(adults_spec, tuple):
            adults = random.randint(adults_spec[0], adults_spec[1])
        else:
            adults = adults_spec

        if isinstance(children_spec, tuple):
            children = random.randint(children_spec[0], children_spec[1])
        else:
            children = children_spec

        return {
            "housing_type": housing_type,
            "adults": adults,
            "children": children,
        }

    def generate_fallback_family(
        self,
        housing_type: str,
        existing_names: set[str] | None = None,
    ) -> list[dict]:
        """Template-based family generation when LLM is unavailable.

        Returns list of citizen specs: {name, age, personality, skills, traits}.
        """
        existing = set(existing_names or [])
        profile = HOUSING_IMMIGRANT_PROFILES.get(
            housing_type,
            {"adults": 1, "children": 0},
        )

        adults_spec = profile.get("adults", 1)
        children_spec = profile.get("children", 0)
        if isinstance(adults_spec, tuple):
            n_adults = random.randint(adults_spec[0], adults_spec[1])
        else:
            n_adults = adults_spec
        if isinstance(children_spec, tuple):
            n_children = random.randint(children_spec[0], children_spec[1])
        else:
            n_children = children_spec

        family: list[dict] = []
        surname = random.choice(_SURNAMES)

        for _ in range(n_adults):
            name = _pick_name(existing, surname=surname)
            existing.add(name)
            personality = _pick_personality()
            family.append({
                "name": name,
                "age": random.randint(20, 45),
                "personality": personality,
                "skills": _random_skills(),
                "traits": _random_traits(personality),
                "life_stage": "adult",
            })

        for _ in range(n_children):
            name = _pick_name(existing, surname=surname)
            existing.add(name)
            personality = _pick_personality()
            age = random.randint(1, 16)
            if age < 5:
                stage = "infant"
            else:
                stage = "child"
            family.append({
                "name": name,
                "age": age,
                "personality": personality,
                "skills": {},
                "traits": _random_traits(personality),
                "life_stage": stage,
            })

        return family

    def check_natural_growth(self, citizens: list) -> list[tuple[str, str]]:
        """Check couples for reproduction eligibility.

        Returns list of (parent1_name, parent2_name) pairs eligible to reproduce.
        """
        eligible_pairs: list[tuple[str, str]] = []
        checked: set[frozenset[str]] = set()

        min_rel = REPRODUCTION_GATE.get("min_relationship", "romantic")
        allow_close = REPRODUCTION_GATE.get("also_allow_close_friend", True)
        required_stage = REPRODUCTION_GATE.get("life_stage", "adult")
        both_energy = REPRODUCTION_GATE.get("both_energy", 70)

        allowed_rels = {min_rel, "married"}
        if allow_close:
            allowed_rels.add("close_friend")

        for c in citizens:
            if not getattr(c, "alive", True):
                continue
            if getattr(c, "life_stage", "") != required_stage:
                continue

            relationships = getattr(c, "relationships", {})
            name_a = getattr(c, "name", "")

            for partner_name, rel_type in relationships.items():
                if rel_type not in allowed_rels:
                    continue
                pair_key = frozenset([name_a, partner_name])
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                # Find partner in citizens list.
                partner = None
                for p in citizens:
                    if getattr(p, "name", "") == partner_name:
                        partner = p
                        break
                if partner is None:
                    continue
                if not getattr(partner, "alive", True):
                    continue
                if getattr(partner, "life_stage", "") != required_stage:
                    continue

                # Energy check — use hunger as inverse energy proxy.
                # Low hunger = high energy.
                needs_a = getattr(c, "needs", None)
                needs_b = getattr(partner, "needs", None)
                if needs_a and needs_b:
                    energy_a = 100.0 - getattr(needs_a, "hunger", 100.0)
                    energy_b = 100.0 - getattr(needs_b, "hunger", 100.0)
                    if energy_a < both_energy or energy_b < both_energy:
                        continue

                eligible_pairs.append((name_a, partner_name))

        return eligible_pairs

    def check_departure(
        self,
        citizens: list,
        satisfaction_scores: dict[str, float],
    ) -> list[str]:
        """Citizens with sustained low satisfaction may leave.

        *satisfaction_scores* — {citizen_name: 0.0-1.0}.
        Returns list of citizen names departing.
        """
        departures: list[str] = []
        for c in citizens:
            if not getattr(c, "alive", True):
                continue
            name = getattr(c, "name", "")
            stage = getattr(c, "life_stage", "adult")
            # Only adults and elders can decide to leave.
            if stage not in ("adult", "elder"):
                continue
            score = satisfaction_scores.get(name, 1.0)
            if score < SATISFACTION_EXODUS_THRESHOLD:
                # Probabilistic departure: lower satisfaction = higher chance.
                depart_prob = (SATISFACTION_EXODUS_THRESHOLD - score) / SATISFACTION_EXODUS_THRESHOLD
                depart_prob *= 0.3  # Dampen so it's not instant.
                if random.random() < depart_prob:
                    departures.append(name)

        return departures
