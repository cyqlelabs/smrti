"""Severity classification for recall results."""
from __future__ import annotations

from smrti.core.models import RecallResult


def classify_memory(r: RecallResult) -> str:
    """Classify a recall result into a severity level for actionability signaling.

    Returns one of: "critical_warning", "known_antipattern", "context".
    """
    v = r.atom.valence.valence
    i = r.atom.valence.intensity
    p = r.atom.truth.probability
    c = r.atom.truth.confidence
    if v < -0.5 and i > 0.5:
        return "critical_warning"
    if p < 0.3 and c > 0.3:
        return "known_antipattern"
    return "context"
