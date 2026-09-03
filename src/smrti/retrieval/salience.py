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
    standing_scale: float = 1.0,
) -> float:
    """
    S = similarity x ( w_sim
                     + w_sti  x normalized_sti
                     + w_conf x confidence
                     + w_lti  x normalized_lti
                     + w_val  x |valence| x intensity )

    Relevance gates standing. The four standing terms say how much the graph
    has come to trust an atom; similarity says how much the atom is about the
    question. Adding the two let an atom with high standing and nothing to do
    with the query outrank a relevant one — a well-connected person concept
    at similarity zero scored above every episode that actually answered —
    so standing is scaled by relevance instead: an atom that is not about the
    question cannot be salient to it, however important it is otherwise, and
    among atoms that are about it, standing decides the order.

    STI and LTI are normalized by dividing by 2.0 and clamping to [0, 1].

    ``valence_weight`` is the global scaling factor for emotional influence:
    it controls how aggressively the dynamic weight shift operates when
    valence < -0.5 (higher = stronger shift from STI toward valence).

    ``standing_scale`` discounts the standing terms alone. Recall passes the
    personality's ``agent_source_trust`` for agent-authored atoms: what the
    graph has come to trust about a model's own output counts for less, while
    how much it is about the question is a property of the query and is left
    whole.
    """
    # Dynamic weight scaling: severe negative-valence atoms shift weight
    # from STI to valence so old-but-critical errors outrank recent trivia.
    # valence_weight controls the strength of this shift per personality.
    # The shift is mass-conserving: valence gains exactly what STI loses.
    if valence < -0.5:
        boost = min(abs(valence) * intensity * valence_weight, w_sti)
        w_sti = w_sti - boost
        w_valence = w_valence + boost

    standing = (
        w_sti * min(sti / 2.0, 1.0)
        + w_confidence * confidence
        + w_lti * min(lti / 2.0, 1.0)
        + w_valence * abs(valence) * intensity
    )
    return max(0.0, similarity) * (w_similarity + standing_scale * standing)
