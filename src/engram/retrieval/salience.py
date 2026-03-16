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
) -> float:
    """
    S = (w_sim  x cosine_similarity)
      + (w_sti  x normalized_sti)
      + (w_conf x confidence)
      + (w_lti  x normalized_lti)
      + (w_val  x |valence| x intensity)

    STI and LTI are normalized by dividing by 2.0 and clamping to [0, 1].
    """
    return (
        w_similarity * similarity
        + w_sti * min(sti / 2.0, 1.0)
        + w_confidence * confidence
        + w_lti * min(lti / 2.0, 1.0)
        + w_valence * abs(valence) * intensity
    )
