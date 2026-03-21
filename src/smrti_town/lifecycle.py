"""Relationship gating, reproduction, aging, death, agent archival."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from smrti.personality.params import PersonalityProfile, load_preset

from smrti_town.config import (
    DEATH_LOW_ENERGY_MULT,
    ELDER_DEATH_PROB_PER_TICK,
    HOURS_PER_YEAR,
    LIFE_STAGES,
    MAX_POPULATION,
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

if TYPE_CHECKING:
    from smrti_town.agent import Agent
    from smrti_town.calendar import SimCalendar


@dataclass
class RelationshipTransition:
    agent_name: str
    target_name: str
    from_state: str
    to_state: str
    detail: str = ""


def compute_life_stage(age_years: float) -> str:
    for stage, info in LIFE_STAGES.items():
        lo, hi = info["age_range"]
        if lo <= age_years < hi:
            return stage
    return "elder"


# ── Death ────────────────────────────────────────────────────────────

def check_death(agent: Agent, cal: SimCalendar, delta_hours: float) -> bool:
    """Return True if the agent should die this tick."""
    if not agent.alive:
        return False
    age_years = cal.to_years(agent.age_hours)

    # Old age — probability increases each year past 65
    if age_years >= 65:
        years_past_elder = age_years - 65
        death_prob = ELDER_DEATH_PROB_PER_TICK * years_past_elder
        if agent.drives.energy < 20:
            death_prob *= DEATH_LOW_ENERGY_MULT
        # Scale by delta to keep probability consistent across tick sizes
        death_prob *= (delta_hours / 2.0)  # normalised to routine tick
        if random.random() < death_prob:
            return True

    # Starvation
    if agent.drives.energy <= 0:
        agent.starvation_hours += delta_hours
        if agent.starvation_hours > STARVATION_HOURS:
            return True
    else:
        agent.starvation_hours = 0.0

    return False


def archive_agent(agent: Agent, all_agents: list[Agent]) -> list[str]:
    """Mark agent as dead and notify kin. Returns list of narrative strings."""
    agent.alive = False
    narratives: list[str] = []
    death_text = f"{agent.name} has passed away."
    narratives.append(death_text)

    # Write death to the agent's own space
    agent.smrti.remember(
        content=death_text,
        type="episode",
        valence=-0.9,
        metadata={"event": "death"},
    )

    # Notify survivors who knew this agent
    for other in all_agents:
        if other.name == agent.name or not other.alive:
            continue
        interaction_count = other.get_interaction_count(agent.name)
        if interaction_count > 0 or (agent.parents and other.name in agent.parents):
            grief_text = f"{agent.name} has passed away."
            other.smrti.remember(
                content=grief_text,
                type="episode",
                valence=-0.8,
                metadata={"event": "death_notification", "deceased": agent.name},
            )

    # Notify children
    for other in all_agents:
        if not other.alive:
            continue
        if other.parents and agent.name in other.parents:
            grief_text = f"My parent {agent.name} has passed away."
            other.smrti.remember(
                content=grief_text,
                type="episode",
                valence=-0.9,
                metadata={"event": "parent_death", "deceased": agent.name},
            )

    return narratives


# ── Reproduction ─────────────────────────────────────────────────────

def check_reproduction_gate(
    agent_a: Agent,
    agent_b: Agent,
    cal: SimCalendar,
    total_population: int,
) -> bool:
    """Check if two agents can reproduce.

    No marriage requirement. Romantic relationship OR close_friend + romance
    drive is sufficient.
    """
    # Population cap — reduce fertility
    if total_population >= MAX_POPULATION:
        if random.random() > 0.1:  # 90% chance to block if over cap
            return False

    # Both must be alive adults
    if not agent_a.alive or not agent_b.alive:
        return False
    if agent_a.life_stage != "adult" or agent_b.life_stage != "adult":
        return False

    # Energy check
    min_energy = REPRODUCTION_GATE["both_energy"]
    if agent_a.drives.energy < min_energy or agent_b.drives.energy < min_energy:
        return False

    # Relationship check — need mutual high interaction count
    a_count = agent_a.get_interaction_count(agent_b.name)
    b_count = agent_b.get_interaction_count(agent_a.name)

    # Either romantic-level bond (high interaction) or close_friend + romance drive
    romantic_threshold = 20  # substantial interaction history
    friend_threshold = 10
    if a_count >= romantic_threshold and b_count >= romantic_threshold:
        return True
    if (
        a_count >= friend_threshold
        and b_count >= friend_threshold
        and agent_a.drives.romance >= 40
        and agent_b.drives.romance >= 40
    ):
        return True

    return False


def inherit_personality(
    parent_a: PersonalityProfile,
    parent_b: PersonalityProfile,
    stress_level: float = 0.0,
) -> PersonalityProfile:
    """Create child personality from parent distributions.

    stress_level: 0.0 (calm) to 1.0 (severe stress). Higher stress increases
    variance in inherited parameters.
    """
    stress = max(0.0, min(1.0, stress_level))
    variance_mult = STRESS_VARIANCE_BASE + stress * (STRESS_VARIANCE_MAX_MULT - STRESS_VARIANCE_BASE)

    child = PersonalityProfile()
    for param in PERSONALITY_PARAMS:
        val_a = getattr(parent_a, param)
        val_b = getattr(parent_b, param)
        mean = (val_a + val_b) / 2.0
        variance = abs(val_a - val_b) * 0.3 * variance_mult
        child_val = random.gauss(mean, max(variance, 0.001))
        lo, hi = PARAM_BOUNDS.get(param, (0.0, 1.0))
        child_val = max(lo, min(hi, child_val))
        setattr(child, param, round(child_val, 4))

    child.preset_name = "inherited"
    return child


def inherit_traits(
    parent_a_traits: dict[str, float],
    parent_b_traits: dict[str, float],
    stress_level: float = 0.0,
) -> dict[str, float]:
    """Create child behavioural traits from parent trait distributions.

    Same Gaussian blend as personality inheritance: mean of parents,
    variance proportional to parental divergence and stress.
    """
    stress = max(0.0, min(1.0, stress_level))
    variance_mult = STRESS_VARIANCE_BASE + stress * (STRESS_VARIANCE_MAX_MULT - STRESS_VARIANCE_BASE)

    child_traits: dict[str, float] = {}
    for trait in TRAIT_NAMES:
        val_a = parent_a_traits.get(trait, 0.5)
        val_b = parent_b_traits.get(trait, 0.5)
        mean = (val_a + val_b) / 2.0
        variance = abs(val_a - val_b) * 0.3 * variance_mult
        child_val = random.gauss(mean, max(variance, 0.01))
        lo, hi = TRAIT_BOUNDS.get(trait, (0.0, 1.0))
        child_traits[trait] = round(max(lo, min(hi, child_val)), 4)
    return child_traits


def spawn_child(
    parent_a: Agent,
    parent_b: Agent,
    all_agents: list[Agent],
    db_path: str,
    tenant_id: str,
) -> Agent:
    """Create a new infant agent from two parents."""
    from smrti_town.agent import Agent as AgentClass

    # Generate name
    child_number = sum(
        1 for a in all_agents
        if a.parents and (parent_a.name in a.parents or parent_b.name in a.parents)
    ) + 1
    # Use a combination of parent name fragments
    name_pool = _generate_child_names(parent_a.name, parent_b.name)
    existing_names = {a.name for a in all_agents}
    child_name = None
    for candidate in name_pool:
        if candidate not in existing_names:
            child_name = candidate
            break
    if not child_name:
        child_name = f"Child_{parent_a.name[:3]}_{parent_b.name[:3]}_{child_number}"

    # Compute stress from parent valence
    stress_a = _get_avg_valence(parent_a)
    stress_b = _get_avg_valence(parent_b)
    stress = max(0.0, -(stress_a + stress_b) / 2.0)

    # Inherit personality hyperparameters (smrti engine tuning)
    pa_profile = load_preset(parent_a.personality_preset) if parent_a.personality_preset != "inherited" else _extract_profile(parent_a)
    pb_profile = load_preset(parent_b.personality_preset) if parent_b.personality_preset != "inherited" else _extract_profile(parent_b)
    child_profile = inherit_personality(pa_profile, pb_profile, stress)

    # Inherit behavioural traits
    child_traits = inherit_traits(parent_a.traits, parent_b.traits, stress)

    child = AgentClass(
        name=child_name,
        personality="balanced",  # placeholder, overridden below
        location=parent_a.location,
        age_years=0.0,
        db_path=db_path,
        tenant_id=tenant_id,
        parents=(parent_a.name, parent_b.name),
        traits=child_traits,
    )
    child.personality_preset = "inherited"
    # Apply inherited personality to smrti
    _apply_profile_to_agent(child, child_profile)

    # Pre-install family bonds
    child.smrti.remember(
        content=f"My parents are {parent_a.name} and {parent_b.name}.",
        type="belief",
        probability=1.0,
        valence=0.6,
        metadata={"relation": "parent", "targets": [parent_a.name, parent_b.name]},
    )

    # Parent gets memory of child
    for parent in (parent_a, parent_b):
        parent.smrti.remember(
            content=f"My child {child_name} was born.",
            type="episode",
            valence=0.8,
            metadata={"event": "child_birth", "child": child_name},
        )

    # Sibling bonds
    for other in all_agents:
        if (
            other.alive
            and other.parents
            and other.name != child_name
            and set(other.parents) == {parent_a.name, parent_b.name}
        ):
            child.smrti.remember(
                content=f"My sibling is {other.name}.",
                type="belief",
                probability=1.0,
                valence=0.5,
                metadata={"relation": "sibling", "target": other.name},
            )
            other.smrti.remember(
                content=f"My new sibling {child_name} was born.",
                type="episode",
                valence=0.5,
                metadata={"event": "sibling_birth", "sibling": child_name},
            )

    return child


def _generate_child_names(parent_a: str, parent_b: str) -> list[str]:
    """Generate candidate child names from parent names."""
    # Simple name generation — blend syllables
    names = [
        f"{parent_a[:2]}{parent_b[-2:]}a",
        f"{parent_b[:2]}{parent_a[-2:]}o",
        f"{parent_a[:3]}el",
        f"{parent_b[:3]}ia",
        f"Li{parent_a[-2:]}",
        f"Ma{parent_b[-2:]}",
        f"{parent_a[:2]}ra",
        f"{parent_b[:2]}na",
    ]
    # Capitalize and deduplicate
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        n = n.capitalize()
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def _get_avg_valence(agent: Agent) -> float:
    """Get average valence from agent's recent memories."""
    try:
        results = agent.smrti.recall(query="how do I feel", top_k=5)
        if not results:
            return 0.0
        total = sum(r.atom.valence.valence for r in results if r.atom.valence)
        return total / len(results)
    except Exception:
        return 0.0


def _extract_profile(agent: Agent) -> PersonalityProfile:
    """Extract current personality profile from an agent's smrti DB."""
    try:
        status = agent.smrti.status()
        personality = status.get("personality", {})
        profile = PersonalityProfile()
        for param in PERSONALITY_PARAMS:
            if param in personality:
                setattr(profile, param, personality[param])
        return profile
    except Exception:
        return PersonalityProfile()


def _apply_profile_to_agent(agent: Agent, profile: PersonalityProfile) -> None:
    """Write a custom personality profile to the agent's smrti space."""
    agent.smrti.db.execute(
        """
        UPDATE personality SET
            confidence_decay_rate=?, confidence_update_lr=?, min_confidence_to_surface=?,
            sti_decay_rate=?, sti_boost_on_access=?, sti_propagation_factor=?,
            lti_promotion_threshold=?, valence_weight=?, valence_propagation=?,
            mood_inertia=?, w_similarity=?, w_sti=?, w_confidence=?, w_lti=?, w_valence=?,
            preset_name=?
        WHERE tenant_id=? AND space=?
        """,
        (
            profile.confidence_decay_rate,
            profile.confidence_update_lr,
            profile.min_confidence_to_surface,
            profile.sti_decay_rate,
            profile.sti_boost_on_access,
            profile.sti_propagation_factor,
            profile.lti_promotion_threshold,
            profile.valence_weight,
            profile.valence_propagation,
            profile.mood_inertia,
            profile.w_similarity,
            profile.w_sti,
            profile.w_confidence,
            profile.w_lti,
            profile.w_valence,
            "inherited",
            agent.smrti.tenant_id,
            agent.smrti.write_space,
        ),
    )


# ── Relationship gates ───────────────────────────────────────────────

def check_relationship_gates(
    agent: Agent,
    all_agents: list[Agent],
) -> list[RelationshipTransition]:
    """Check whether any relationships should transition based on gates."""
    transitions: list[RelationshipTransition] = []
    if not agent.alive or not agent.can_talk:
        return transitions

    for other in all_agents:
        if other.name == agent.name or not other.alive or not other.can_talk:
            continue

        count = agent.get_interaction_count(other.name)
        mutual_count = other.get_interaction_count(agent.name)
        min_count = min(count, mutual_count)

        current_state = _infer_relationship_state(agent, other)
        next_state = _next_relationship_state(current_state, min_count, agent, other)

        if next_state and next_state != current_state:
            transitions.append(RelationshipTransition(
                agent_name=agent.name,
                target_name=other.name,
                from_state=current_state,
                to_state=next_state,
                detail=f"{agent.name} and {other.name}: {current_state} -> {next_state}",
            ))

    return transitions


def _infer_relationship_state(agent: Agent, other: Agent) -> str:
    """Infer current relationship state from interaction count."""
    count = agent.get_interaction_count(other.name)
    mutual = other.get_interaction_count(agent.name)
    min_c = min(count, mutual)

    if min_c >= 20:
        return "romantic"
    if min_c >= 10:
        return "close_friend"
    if min_c >= 5:
        return "friend"
    if min_c >= 1:
        return "acquaintance"
    return "stranger"


def _next_relationship_state(
    current: str,
    min_count: int,
    agent: Agent,
    other: Agent,
) -> str | None:
    """Determine if a relationship should advance."""
    progression = ["stranger", "acquaintance", "friend", "close_friend", "romantic", "married"]
    try:
        idx = progression.index(current)
    except ValueError:
        return None

    if idx >= len(progression) - 1:
        return None

    next_state = progression[idx + 1]
    gate = RELATIONSHIP_GATES.get(next_state, {})

    if next_state == "acquaintance":
        return "acquaintance" if min_count >= 1 else None

    if next_state == "friend":
        needed = gate.get("interaction_count", 5)
        if min_count >= needed:
            return "friend"

    if next_state == "close_friend":
        needed_lti = gate.get("friend_lti", 0.5)
        needed_episodes = gate.get("shared_episodes", 3)
        if min_count >= max(needed_episodes, 8):
            return "close_friend"

    if next_state == "romantic":
        needed_drive = gate.get("romance_drive", 50)
        if (
            min_count >= 15
            and agent.drives.romance >= needed_drive * 0.5
            and other.drives.romance >= needed_drive * 0.5
            and agent.life_stage == "adult"
            and other.life_stage == "adult"
        ):
            return "romantic"

    if next_state == "married":
        if min_count >= 25 and agent.life_stage == "adult" and other.life_stage == "adult":
            return "married"

    return None


def apply_relationship_transition(
    transition: RelationshipTransition,
    agents_by_name: dict[str, Agent],
) -> list[str]:
    """Apply a relationship transition and write memories."""
    narratives: list[str] = []
    agent = agents_by_name.get(transition.agent_name)
    target = agents_by_name.get(transition.target_name)
    if not agent or not target:
        return narratives

    text_a = f"My relationship with {transition.target_name} has deepened. We are now {transition.to_state}s."
    text_b = f"My relationship with {transition.agent_name} has deepened. We are now {transition.to_state}s."

    if transition.to_state == "married":
        text_a = f"I married {transition.target_name}."
        text_b = f"I married {transition.agent_name}."

    agent.smrti.remember(
        content=text_a,
        type="belief" if transition.to_state == "married" else "episode",
        probability=1.0 if transition.to_state == "married" else 0.8,
        valence=0.6,
        metadata={"relation": transition.to_state, "target": transition.target_name},
    )
    target.smrti.remember(
        content=text_b,
        type="belief" if transition.to_state == "married" else "episode",
        probability=1.0 if transition.to_state == "married" else 0.8,
        valence=0.6,
        metadata={"relation": transition.to_state, "target": transition.agent_name},
    )
    narratives.append(transition.detail)
    return narratives
