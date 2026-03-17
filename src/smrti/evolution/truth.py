"""Bayesian truth maintenance."""
from __future__ import annotations

from smrti.core.models import TruthValue


def update_truth(
    current: TruthValue,
    evidence_prob: float,
    evidence_weight: float,
    lr: float,
) -> TruthValue:
    """Weighted Bayesian-inspired update.

    Blends the current belief with new evidence scaled by weight * learning_rate.
    Confidence grows toward 1.0 asymptotically with accumulated evidence.
    """
    w = evidence_weight * lr
    new_prob = (current.probability * current.confidence + evidence_prob * w) / (
        current.confidence + w + 1e-9
    )
    new_conf = min(1.0, current.confidence + w * (1.0 - current.confidence))
    return TruthValue(
        probability=max(0.0, min(1.0, new_prob)),
        confidence=max(0.0, min(1.0, new_conf)),
    )


def pln_merge(a: TruthValue, b: TruthValue) -> TruthValue:
    """PLN revision rule: merge two independent truth estimates."""
    return a.merge(b)
