"""Tests for severity classification."""
from unittest.mock import MagicMock

from smrti.core.models import (
    Atom,
    AtomType,
    AttentionValue,
    RecallResult,
    TruthValue,
    Valence,
)
from smrti.core.provenance import VALENCE_STATED
from smrti.retrieval.classify import classify_memory


def _make_result(valence=-0.8, intensity=0.9, probability=0.8, confidence=0.5, stated=True,
                 atom_type=AtomType.EPISODE):
    """Defaults to a stated valence — a caller reporting something to avoid.

    An estimated one is only a reading of the text's mood and cannot reach
    critical_warning; tests/test_severity_gate.py covers that side.
    """
    atom = Atom(
        type=atom_type,
        label="test",
        truth=TruthValue(probability=probability, confidence=confidence),
        attention=AttentionValue(sti=0.5, lti=0.3),
        valence=Valence(valence=valence, intensity=intensity),
        metadata={VALENCE_STATED: True} if stated else {},
    )
    return RecallResult(atom=atom, salience=0.5, similarity=0.7)


def test_critical_warning_negative_valence_high_intensity():
    r = _make_result(valence=-0.8, intensity=0.9)
    assert classify_memory(r) == "critical_warning"


def test_critical_warning_boundary():
    r = _make_result(valence=-0.51, intensity=0.51)
    assert classify_memory(r) == "critical_warning"


def test_not_critical_when_valence_above_threshold():
    r = _make_result(valence=-0.4, intensity=0.9)
    assert classify_memory(r) != "critical_warning"


def test_not_critical_when_intensity_below_threshold():
    r = _make_result(valence=-0.8, intensity=0.4)
    assert classify_memory(r) != "critical_warning"


def test_known_antipattern_low_prob_high_conf():
    r = _make_result(valence=0.0, intensity=0.0, probability=0.2, confidence=0.5,
                     atom_type=AtomType.BELIEF)
    assert classify_memory(r) == "known_antipattern"


def test_a_superseded_claim_edge_is_an_antipattern():
    r = _make_result(valence=0.0, intensity=0.0, probability=0.1, confidence=0.5,
                     atom_type=AtomType.RELATION)
    assert classify_memory(r) == "known_antipattern"


def test_only_propositions_can_be_antipatterns():
    """A low probability on an episode, goal or concept is not a disproven
    claim, and must not reach the agent as something to avoid."""
    for kind in (AtomType.EPISODE, AtomType.GOAL, AtomType.CONCEPT):
        r = _make_result(valence=0.0, intensity=0.0, probability=0.1, confidence=0.9,
                         atom_type=kind)
        assert classify_memory(r) == "context", kind


def test_not_antipattern_when_prob_above_threshold():
    r = _make_result(valence=0.0, intensity=0.0, probability=0.4, confidence=0.5,
                     atom_type=AtomType.BELIEF)
    assert classify_memory(r) != "known_antipattern"


def test_not_antipattern_when_conf_below_threshold():
    r = _make_result(valence=0.0, intensity=0.0, probability=0.2, confidence=0.2,
                     atom_type=AtomType.BELIEF)
    assert classify_memory(r) != "known_antipattern"


def test_context_for_neutral_memory():
    r = _make_result(valence=0.3, intensity=0.5, probability=0.8, confidence=0.7)
    assert classify_memory(r) == "context"


def test_critical_takes_precedence_over_antipattern():
    """When both conditions match, critical_warning wins (checked first)."""
    r = _make_result(valence=-0.8, intensity=0.9, probability=0.1, confidence=0.5)
    assert classify_memory(r) == "critical_warning"
