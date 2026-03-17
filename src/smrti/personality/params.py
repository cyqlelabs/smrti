"""PersonalityProfile dataclass and preset loading."""
from dataclasses import dataclass, field


@dataclass
class PersonalityProfile:
    # Belief dynamics
    confidence_decay_rate: float = 0.02
    confidence_update_lr: float = 0.3
    min_confidence_to_surface: float = 0.1
    # Attention dynamics
    sti_decay_rate: float = 0.1
    sti_boost_on_access: float = 0.5
    sti_propagation_factor: float = 0.15
    lti_promotion_threshold: float = 0.7
    # Emotional dynamics
    valence_weight: float = 0.2
    valence_propagation: float = 0.1
    mood_inertia: float = 0.8
    # Salience weights
    w_similarity: float = 0.35
    w_sti: float = 0.25
    w_confidence: float = 0.20
    w_lti: float = 0.10
    w_valence: float = 0.10
    # Meta
    preset_name: str = "balanced"


PRESETS = {
    "balanced": PersonalityProfile(),
    "analytical": PersonalityProfile(
        confidence_decay_rate=0.01,
        confidence_update_lr=0.15,
        sti_decay_rate=0.05,
        sti_propagation_factor=0.05,
        lti_promotion_threshold=0.9,
        valence_weight=0.05,
        mood_inertia=0.95,
        w_similarity=0.30,
        w_sti=0.15,
        w_confidence=0.40,
        w_lti=0.10,
        w_valence=0.05,
        preset_name="analytical",
    ),
    "curious": PersonalityProfile(
        confidence_decay_rate=0.03,
        confidence_update_lr=0.5,
        sti_decay_rate=0.2,
        sti_boost_on_access=0.8,
        sti_propagation_factor=0.3,
        lti_promotion_threshold=0.5,
        valence_weight=0.15,
        mood_inertia=0.5,
        w_similarity=0.25,
        w_sti=0.35,
        w_confidence=0.15,
        w_lti=0.10,
        w_valence=0.15,
        preset_name="curious",
    ),
    "empathetic": PersonalityProfile(
        confidence_decay_rate=0.02,
        confidence_update_lr=0.4,
        sti_decay_rate=0.08,
        sti_propagation_factor=0.2,
        valence_weight=0.4,
        valence_propagation=0.25,
        mood_inertia=0.4,
        w_similarity=0.25,
        w_sti=0.20,
        w_confidence=0.10,
        w_lti=0.10,
        w_valence=0.35,
        preset_name="empathetic",
    ),
    "maverick": PersonalityProfile(
        confidence_decay_rate=0.005,
        confidence_update_lr=0.1,
        sti_decay_rate=0.15,
        sti_propagation_factor=0.35,
        lti_promotion_threshold=0.4,
        valence_weight=0.25,
        mood_inertia=0.7,
        w_similarity=0.20,
        w_sti=0.30,
        w_confidence=0.15,
        w_lti=0.15,
        w_valence=0.20,
        preset_name="maverick",
    ),
    "deterministic": PersonalityProfile(
        confidence_decay_rate=0.005,
        confidence_update_lr=0.4,
        min_confidence_to_surface=0.3,
        sti_decay_rate=0.08,
        sti_boost_on_access=0.8,
        sti_propagation_factor=0.05,
        lti_promotion_threshold=0.85,
        valence_weight=0.20,
        valence_propagation=0.05,
        mood_inertia=0.95,
        w_similarity=0.35,
        w_sti=0.10,
        w_confidence=0.30,
        w_lti=0.13,
        w_valence=0.12,
        preset_name="deterministic",
    ),
}


def load_preset(name: str) -> PersonalityProfile:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown personality preset: {name}. Choose from: {list(PRESETS.keys())}"
        )
    return PRESETS[name]
