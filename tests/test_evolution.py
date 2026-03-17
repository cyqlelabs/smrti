"""Tests for truth maintenance and consolidation epoch."""
import os
import tempfile

import pytest

from engram import Engram
from engram.core.models import TruthValue
from engram.evolution.truth import update_truth


def test_truth_update_increases_confidence():
    current = TruthValue(probability=0.5, confidence=0.0)
    updated = update_truth(current, 0.9, 1.0, 0.3)
    assert updated.confidence > current.confidence
    assert updated.probability > 0.5


def test_truth_update_contradicting_evidence():
    current = TruthValue(probability=0.9, confidence=0.8)
    updated = update_truth(current, 0.1, 1.0, 0.3)
    assert updated.probability < current.probability


def test_pln_merge():
    a = TruthValue(probability=0.8, confidence=0.5)
    b = TruthValue(probability=0.6, confidence=0.5)
    merged = a.merge(b)
    assert 0.0 <= merged.probability <= 1.0
    assert merged.confidence > 0.5


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Engram(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def test_reflect_runs(mem):
    mem.remember("test memory")
    result = mem.reflect()
    assert result is not None
    assert hasattr(result, "beliefs_updated")
    assert hasattr(result, "atoms_pruned")
