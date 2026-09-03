"""Tests for salience-scored retrieval."""
import os
import tempfile

import pytest

from smrti import Smrti


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, personality="balanced", tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def test_remember_and_recall(mem):
    mem.remember("User prefers Python over JavaScript", probability=0.9)
    mem.remember("User is working on a machine learning project", probability=0.8)

    results = mem.recall("programming languages")
    assert len(results) > 0
    assert any(
        "Python" in r.atom.label or "Python" in (r.atom.content or "")
        for r in results
    )


def test_recall_returns_salience(mem):
    mem.remember("Alice likes coffee", valence=0.5)
    results = mem.recall("Alice coffee")
    assert all(hasattr(r, "salience") for r in results)
    assert all(r.salience >= 0 for r in results)


def test_min_confidence_filter(mem):
    """The floor excludes, and only excludes, atoms below it.

    Asserted both ways: the earlier form (``empty or all above``) passed on an
    empty result, which is what a broken recall returns.
    """
    atom_id = mem.remember("the vault password rotates monthly", probability=0.3)
    mem.db.execute("UPDATE atoms SET confidence = 0.2 WHERE id = ?", (atom_id,))

    below = mem.recall("vault password rotation", min_confidence=0.9)
    assert all(r.atom.id != atom_id for r in below)

    above = mem.recall("vault password rotation", min_confidence=0.1)
    assert any(r.atom.id == atom_id for r in above)


# ── Salience boost for negative-valence atoms ────────────────────────────────

def test_negative_valence_boosts_salience():
    from smrti.retrieval.salience import compute_salience

    # Same base params, only valence differs
    base = dict(similarity=0.5, sti=0.5, confidence=0.5, lti=0.3)

    neutral = compute_salience(**base, valence=0.0, intensity=0.8)
    negative = compute_salience(**base, valence=-0.8, intensity=0.8)

    # Negative valence atom should score higher due to weight shift
    assert negative > neutral


def test_salience_weight_shift_preserves_total():
    from smrti.retrieval.salience import compute_salience

    # With extreme negative valence, the shifted weight still produces
    # a valid (non-negative) score
    score = compute_salience(
        similarity=0.5, sti=0.0, confidence=0.5,
        lti=0.3, valence=-1.0, intensity=1.0,
    )
    assert score >= 0
