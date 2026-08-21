"""Severity classification for recall results."""
from __future__ import annotations

from smrti.core.models import AtomType, RecallResult
from smrti.core.provenance import VALENCE_STATED


def classify_memory(r: RecallResult) -> str:
    """Classify a recall result into a severity level for actionability signaling.

    Returns one of: "critical_warning", "known_antipattern", "context".

    A critical warning is read by the agent as a hard constraint, so it takes
    more than a negative reading of the text. Two things bar the way.

    The valence must have been stated by whoever stored the memory. Estimated
    valence scores the mood of the words, and stored conversation is full of
    ordinary frustration — "I didn't understand", "that was terrible" — which
    is a speaker's tone, not a report of a mistake to never repeat. A caller
    that sets the valence deliberately is making exactly that report.

    And the memory must be able to hold a proposition. Concepts are index
    nodes: a bare label carries nothing to avoid doing again.

    The tone read here is the atom's own, never the mood it absorbed from its
    neighbours — see :class:`smrti.core.models.Valence`.
    """
    atom = r.atom
    v = atom.valence.own
    i = atom.valence.own_intensity
    p = atom.truth.probability
    c = atom.truth.confidence
    stated = atom.metadata.get(VALENCE_STATED) is True
    if stated and atom.type != AtomType.CONCEPT and v < -0.5 and i > 0.5:
        return "critical_warning"
    if p < 0.3 and c > 0.3:
        return "known_antipattern"
    return "context"
