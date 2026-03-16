"""Tests for salience-scored retrieval."""
import os
import tempfile

import pytest

from engram import Engram


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Engram(db_path=db_path, personality="balanced", agent_id="test")
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
    mem.remember("low confidence fact", probability=0.3)
    results = mem.recall("low confidence fact", min_confidence=0.9)
    assert len(results) == 0 or all(r.atom.truth.confidence >= 0.9 for r in results)
