"""PersonalityProfile dataclass and preset loading."""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# The range in which each hyperparameter means something. A profile holding a
# value outside it — a string, a negative decay rate, a surfacing floor of 7 —
# is refused when it is built, and a stored row holding one reads as that
# field's default: one bad write through ``/personality`` used to make every
# later recall in the space fail, across restarts.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "confidence_decay_rate": (0.0, 1.0),
    "confidence_update_lr": (0.0, 1.0),
    "min_confidence_to_surface": (0.0, 1.0),
    "sti_decay_rate": (0.0, 1.0),
    "sti_boost_on_access": (0.0, 3.0),
    "sti_propagation_factor": (0.0, 1.0),
    "lti_promotion_threshold": (0.0, 3.0),
    "lti_decay_rate": (0.0, 1.0),
    "agent_source_trust": (0.0, 1.0),
    "valence_weight": (0.0, 1.0),
    "valence_propagation": (0.0, 1.0),
    "mood_inertia": (0.0, 1.0),
    "w_similarity": (0.0, 1.0),
    "w_sti": (0.0, 1.0),
    "w_confidence": (0.0, 1.0),
    "w_lti": (0.0, 1.0),
    "w_valence": (0.0, 1.0),
}


def _checked(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    low, high = PARAM_RANGES[name]
    # Written as a positive test so NaN, which compares false to everything,
    # is refused too.
    if not low <= value <= high:
        raise ValueError(f"{name} must be within [{low}, {high}], got {value!r}")
    return float(value)


_warned: set[tuple[str, str]] = set()


def param_value(name: str, value, default: float) -> float:
    """A stored hyperparameter, or *default* when it is missing or unusable."""
    if value is None:
        return default
    try:
        return _checked(name, value)
    except ValueError as exc:
        if (name, repr(value)) not in _warned:
            _warned.add((name, repr(value)))
            logger.warning("ignoring stored personality value: %s; using %s", exc, default)
        return default


def personality_params(row) -> dict:
    """A stored personality row with every unusable hyperparameter set to
    None, which every reader already maps to that field's default."""
    params = dict(row) if row else {}
    for name in PARAM_RANGES:
        if params.get(name) is not None:
            checked = param_value(name, params[name], None)
            params[name] = checked
    return params


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
    lti_decay_rate: float = 0.01
    # Provenance — how much weight agent-authored content carries relative to
    # the user's own. Scales initial truth/attention of atoms extracted from
    # agent turns and accelerates their decay, so unincorporated model output
    # fades while user-stated facts persist.
    agent_source_trust: float = 0.5
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

    def __post_init__(self) -> None:
        for name in PARAM_RANGES:
            setattr(self, name, _checked(name, getattr(self, name)))


PRESETS = {
    "balanced": PersonalityProfile(),
    "analytical": PersonalityProfile(
        confidence_decay_rate=0.01,
        confidence_update_lr=0.15,
        sti_decay_rate=0.05,
        sti_propagation_factor=0.05,
        lti_promotion_threshold=0.9,
        lti_decay_rate=0.005,
        agent_source_trust=0.3,
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
        lti_decay_rate=0.02,
        agent_source_trust=0.7,
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
        lti_decay_rate=0.02,
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
        lti_decay_rate=0.003,
        agent_source_trust=0.3,
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
