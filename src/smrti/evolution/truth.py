"""Truth maintenance: one revision rule for every piece of evidence.

A truth value is a strength (``probability``) and a confidence. Confidence is
read as an evidence count through ``c = n / (n + 1)``: confidence 0.5 is one
unit of evidence, 0.9 is nine, and 0 is none. Revision adds the new
observation's weight to that count and takes the count-weighted mean of the
strengths, which is the PLN revision rule with ``k = 1`` and the same
arithmetic :meth:`smrti.core.models.TruthValue.merge` performs when two
whole truth values are combined. One formula, used by the epoch's evidence
pass, by reinforcement, and by bridge merging, so "confidence" means the same
thing wherever it is read.
"""
from __future__ import annotations

from smrti.core.models import TruthValue

# Confidence can never reach exactly 1.0 through revision (that would be an
# infinite count), so a stored 1.0 is read as a very large count instead.
_EPSILON = 1e-9


def evidence_count(confidence: float) -> float:
    """The number of evidence units a confidence stands for."""
    c = max(0.0, min(confidence, 1.0 - _EPSILON))
    return c / (1.0 - c)


def confidence_from_count(count: float) -> float:
    return max(0.0, min(1.0, count / (count + 1.0)))


def update_truth(
    current: TruthValue,
    evidence_prob: float,
    evidence_weight: float,
    lr: float,
) -> TruthValue:
    """Revise ``current`` with one observation of ``evidence_prob``.

    ``evidence_weight * lr`` is how many evidence units the observation is
    worth: the caller's weight scaled by the personality's learning rate. The
    strength moves to the count-weighted mean, so a heavily-evidenced belief
    barely moves and a fresh one follows the observation; confidence rises
    with the count and converges toward one instead of ratcheting past it.
    """
    w = evidence_weight * lr
    if w <= 0:
        return current
    n = evidence_count(current.confidence)
    n_new = n + w
    new_prob = (n * current.probability + w * evidence_prob) / n_new
    return TruthValue(
        probability=max(0.0, min(1.0, new_prob)),
        confidence=confidence_from_count(n_new),
    )
