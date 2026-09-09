"""Severity classification for recall results."""
from __future__ import annotations

from smrti.core.models import Atom, AtomType, RecallResult
from smrti.core.provenance import VALENCE_STATED


def is_critical_warning(atom: Atom) -> bool:
    """Whether the atom is a stated, severe warning — the one memory kind that
    becomes a hard constraint at recall, and the one retrieval must never
    damp on the way there."""
    return (
        atom.metadata.get(VALENCE_STATED) is True
        and atom.type != AtomType.CONCEPT
        and atom.valence.own < -0.5
        and atom.valence.own_intensity > 0.5
    )


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
    p = atom.truth.probability
    c = atom.truth.confidence
    if is_critical_warning(atom):
        return "critical_warning"
    if p < 0.3 and c > 0.3:
        return "known_antipattern"
    return "context"
