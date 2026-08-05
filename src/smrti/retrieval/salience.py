"""Salience scoring formula for memory retrieval."""
from __future__ import annotations


def compute_salience(
    similarity: float,
    sti: float,
    confidence: float,
    lti: float,
    valence: float,
    intensity: float,
    w_similarity: float = 0.35,
    w_sti: float = 0.25,
    w_confidence: float = 0.20,
    w_lti: float = 0.10,
    w_valence: float = 0.10,
    valence_weight: float = 0.2,
) -> float:
    """
    S = (w_sim  x cosine_similarity)
      + (w_sti  x normalized_sti)
      + (w_conf x confidence)
      + (w_lti  x normalized_lti)
      + (w_val  x |valence| x intensity)

    STI and LTI are normalized by dividing by 2.0 and clamping to [0, 1].

    ``valence_weight`` is the global scaling factor for emotional influence:
    it controls how aggressively the dynamic weight shift operates when
    valence < -0.5 (higher = stronger shift from STI toward valence).
    """
    # Dynamic weight scaling: severe negative-valence atoms shift weight
    # from STI to valence so old-but-critical errors outrank recent trivia.
    # valence_weight controls the strength of this shift per personality.
    # The shift is mass-conserving: valence gains exactly what STI loses.
    if valence < -0.5:
        boost = min(abs(valence) * intensity * valence_weight, w_sti)
        w_sti = w_sti - boost
        w_valence = w_valence + boost

    return (
        w_similarity * similarity
        + w_sti * min(sti / 2.0, 1.0)
        + w_confidence * confidence
        + w_lti * min(lti / 2.0, 1.0)
        + w_valence * abs(valence) * intensity
    )
