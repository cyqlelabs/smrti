"""Lifecycle — death, reproduction, relationship progression/regression."""

from __future__ import annotations

import math
import random

from smrti_town.config import (
    DEATH_LOW_ENERGY_MULT,
    ELDER_DEATH_PROB_PER_TICK,
    HOURS_PER_YEAR,
    LIFE_STAGES,
    NEED_MAX,
    PARAM_BOUNDS,
    PERSONALITY_PARAMS,
    PRESET_TRAITS,
    RELATIONSHIP_GATES,
    REPRODUCTION_GATE,
    STARVATION_HOURS,
    STRESS_VARIANCE_BASE,
    STRESS_VARIANCE_MAX_MULT,
    TRAIT_BOUNDS,
    TRAIT_NAMES,
)


# ── Death ───────────────────────────────────────────────────────────────

def check_death(citizen, delta_hours: float) -> bool:
    """Check if a citizen dies this tick.

    Elder death: base probability scaled by delta_hours, increased if
    health/hunger are critical.

    Starvation: any citizen with max hunger for STARVATION_HOURS dies.

    Returns True if the citizen should die.
    """
    if not getattr(citizen, "alive", True):
        return False

    stage = getattr(citizen, "life_stage", "adult")
    needs = getattr(citizen, "needs", None)
    age = getattr(citizen, "age_years", 30.0)

    # Starvation check: hunger at max for extended period.
    if needs:
        hunger = getattr(needs, "hunger", 0.0)
        if hunger >= NEED_MAX:
            # Approximate: citizen is at max hunger, apply starvation chance
            # proportional to how long they've been starving.
            # Since we can't track cumulative starvation time here, use a
            # probability per tick that accumulates to near-certainty over
            # STARVATION_HOURS.
            # P(survive N hours) = (1-p)^N = 0.01 when N = STARVATION_HOURS
            # => p = 1 - 0.01^(1/STARVATION_HOURS)
            p_per_hour = 1.0 - math.pow(0.01, 1.0 / STARVATION_HOURS)
            p_this_tick = 1.0 - math.pow(1.0 - p_per_hour, delta_hours)
            if random.random() < p_this_tick:
                return True

    # Elder death: age-based probability.
    if stage == "elder":
        base_prob = ELDER_DEATH_PROB_PER_TICK * delta_hours
        # Increase probability with age beyond 65.
        age_factor = 1.0 + max(0.0, age - 65) * 0.02
        # Low energy (high hunger) multiplier.
        energy_mult = 1.0
        if needs:
            hunger = getattr(needs, "hunger", 0.0)
            if hunger > 70:
                energy_mult = DEATH_LOW_ENERGY_MULT
            health = getattr(needs, "health", 0.0)
            if health > 70:
                energy_mult *= 1.5
        prob = base_prob * age_factor * energy_mult
        if random.random() < prob:
            return True

    return False


# ── Reproduction ────────────────────────────────────────────────────────

def check_reproduction_eligibility(citizen_a, citizen_b) -> bool:
    """Check REPRODUCTION_GATE requirements for a pair of citizens.

    Assumes both are already identified as being in an eligible relationship
    (checked by PopulationManager.check_natural_growth).
    """
    gate = REPRODUCTION_GATE
    required_stage = gate.get("life_stage", "adult")

    for c in (citizen_a, citizen_b):
        if not getattr(c, "alive", True):
            return False
        if getattr(c, "life_stage", "") != required_stage:
            return False

    # Energy check (inverse of hunger).
    both_energy = gate.get("both_energy", 70)
    for c in (citizen_a, citizen_b):
        needs = getattr(c, "needs", None)
        if needs:
            hunger = getattr(needs, "hunger", NEED_MAX)
            energy = NEED_MAX - hunger
            if energy < both_energy:
                return False

    # Relationship type check.
    min_rel = gate.get("min_relationship", "romantic")
    allow_close = gate.get("also_allow_close_friend", True)
    allowed = {min_rel, "married"}
    if allow_close:
        allowed.add("close_friend")

    name_a = getattr(citizen_a, "name", "")
    name_b = getattr(citizen_b, "name", "")
    rels_a = getattr(citizen_a, "relationships", {})
    rel_type = rels_a.get(name_b)
    if rel_type not in allowed:
        return False

    return True


def create_child(
    parent_a,
    parent_b,
    existing_names: set[str] | None = None,
) -> dict:
    """Generate child spec with inherited personality and traits.

    Personality params: blend both parents with stress-boosted Gaussian mutation.
    Traits: average with random jitter.

    Returns dict: {name, age, personality, skills, traits, personality_params,
                   parents: [name_a, name_b]}.
    """
    from smrti_town.population import _pick_name, _pick_personality

    name_a = getattr(parent_a, "name", "Unknown")
    name_b = getattr(parent_b, "name", "Unknown")
    existing = set(existing_names or [])
    child_name = _pick_name(existing)

    # Inherit personality preset from one parent randomly.
    preset_a = getattr(parent_a, "personality_preset", "balanced")
    preset_b = getattr(parent_b, "personality_preset", "balanced")
    child_preset = random.choice([preset_a, preset_b])

    # Blend personality hyperparameters.
    params_a = getattr(parent_a, "personality_params", None) or {}
    params_b = getattr(parent_b, "personality_params", None) or {}

    # Compute stress as average unmet needs of parents (0-1 scale).
    stress = 0.0
    stress_count = 0
    for parent in (parent_a, parent_b):
        needs = getattr(parent, "needs", None)
        if needs:
            for need_name in ("hunger", "shelter", "health", "safety"):
                val = getattr(needs, need_name, 0.0)
                stress += val / NEED_MAX
                stress_count += 1
    if stress_count > 0:
        stress /= stress_count
    stress_variance = STRESS_VARIANCE_BASE * (1.0 + stress * (STRESS_VARIANCE_MAX_MULT - 1.0))

    child_params: dict[str, float] = {}
    for param in PERSONALITY_PARAMS:
        lo, hi = PARAM_BOUNDS.get(param, (0.0, 1.0))
        val_a = params_a.get(param, (lo + hi) / 2)
        val_b = params_b.get(param, (lo + hi) / 2)
        blended = (val_a + val_b) / 2.0
        mutated = blended + random.gauss(0, 0.05 * stress_variance)
        child_params[param] = max(lo, min(hi, mutated))

    # Blend behavioural traits.
    traits_a = getattr(parent_a, "traits", PRESET_TRAITS.get(preset_a, {}))
    traits_b = getattr(parent_b, "traits", PRESET_TRAITS.get(preset_b, {}))
    child_traits: dict[str, float] = {}
    for trait in TRAIT_NAMES:
        lo, hi = TRAIT_BOUNDS.get(trait, (0.0, 1.0))
        va = traits_a.get(trait, 0.5)
        vb = traits_b.get(trait, 0.5)
        blended = (va + vb) / 2.0
        mutated = blended + random.gauss(0, 0.08 * stress_variance)
        child_traits[trait] = max(lo, min(hi, mutated))

    return {
        "name": child_name,
        "age": 0,
        "personality": child_preset,
        "skills": {},
        "traits": child_traits,
        "personality_params": child_params,
        "life_stage": "infant",
        "parents": [name_a, name_b],
    }


# ── Relationship progression ───────────────────────────────────────────

# Ordered relationship tiers for progression/regression.
_REL_TIERS = ["acquaintance", "friend", "close_friend", "romantic", "married"]
_REL_INDEX = {r: i for i, r in enumerate(_REL_TIERS)}


def check_relationship_progression(
    citizen_a,
    citizen_b,
    interaction_count: int,
    shared_valence: float,
) -> str | None:
    """Check RELATIONSHIP_GATES to see if a pair should progress.

    *interaction_count* — total interactions between the pair.
    *shared_valence* — average valence of shared memories (-1 to 1).

    Returns the new relationship type if progression is warranted, else None.
    """
    name_a = getattr(citizen_a, "name", "")
    name_b = getattr(citizen_b, "name", "")
    rels_a = getattr(citizen_a, "relationships", {})
    current_rel = rels_a.get(name_b, "acquaintance")
    current_idx = _REL_INDEX.get(current_rel, 0)

    if current_idx >= len(_REL_TIERS) - 1:
        return None  # Already at max tier.

    next_tier = _REL_TIERS[current_idx + 1]
    gate = RELATIONSHIP_GATES.get(next_tier, {})

    # Check interaction count.
    if "interaction_count" in gate:
        if interaction_count < gate["interaction_count"]:
            return None

    # Check valence thresholds.
    if "valence" in gate:
        if shared_valence < gate["valence"]:
            return None
    if "mutual_valence" in gate:
        if shared_valence < gate["mutual_valence"]:
            return None

    # Friendship LTI checks — we approximate LTI as a function of
    # interaction count and valence, since the actual LTI is in the
    # Smrti memory graph. Gate values are thresholds on 0-1 scale.
    if "friend_lti" in gate:
        approx_lti = min(1.0, interaction_count / 20.0 * max(0, shared_valence))
        if approx_lti < gate["friend_lti"]:
            return None
    if "close_friend_lti" in gate:
        approx_lti = min(1.0, interaction_count / 30.0 * max(0, shared_valence))
        if approx_lti < gate["close_friend_lti"]:
            return None
    if "romantic_lti" in gate:
        approx_lti = min(1.0, interaction_count / 40.0 * max(0, shared_valence))
        if approx_lti < gate["romantic_lti"]:
            return None

    # Shared episodes check.
    if "shared_episodes" in gate:
        # Approximate: each positive interaction is roughly a shared episode.
        est_shared = int(interaction_count * max(0, shared_valence))
        if est_shared < gate["shared_episodes"]:
            return None

    # Cohabitation check — requires both living in the same home.
    if "cohabitation_years" in gate:
        home_a = getattr(citizen_a, "home", None)
        home_b = getattr(citizen_b, "home", None)
        if not home_a or home_a != home_b:
            return None
        # Approximate cohabitation time from interaction count
        # (assumes ~1 interaction per tick period).
        # This is a rough heuristic; the engine should track actual cohabitation.

    return next_tier


def check_relationship_regression(
    citizen_a,
    citizen_b,
    negative_episodes: int,
) -> str | None:
    """Regress relationship if too many negative episodes accumulate.

    Regression thresholds:
        married -> romantic: 10 negative episodes
        romantic -> close_friend: 7
        close_friend -> friend: 5
        friend -> acquaintance: 3

    Returns the new (lower) relationship type, or None if no regression.
    """
    name_a = getattr(citizen_a, "name", "")
    name_b = getattr(citizen_b, "name", "")
    rels_a = getattr(citizen_a, "relationships", {})
    current_rel = rels_a.get(name_b, "acquaintance")
    current_idx = _REL_INDEX.get(current_rel, 0)

    if current_idx <= 0:
        return None  # Already at lowest tier.

    regression_thresholds = {
        "married": 10,
        "romantic": 7,
        "close_friend": 5,
        "friend": 3,
    }

    threshold = regression_thresholds.get(current_rel, 5)
    if negative_episodes >= threshold:
        return _REL_TIERS[current_idx - 1]

    return None
