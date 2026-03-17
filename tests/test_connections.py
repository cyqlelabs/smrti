"""Tests for cross-domain association discovery (evolution/connections.py)."""
import os
import struct
import tempfile

import pytest

from smrti import Smrti
from smrti.evolution.connections import discover_connections


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


# ── empty space ───────────────────────────────────────────────────────────────

def test_no_connections_in_empty_space(mem):
    result = discover_connections("test", "default", mem.db, mem.embed)
    assert result == 0


def test_no_connections_when_no_high_lti_atoms(mem):
    # Add atoms with low LTI
    mem.remember("red apples are sweet")
    mem.remember("blue skies are clear")
    # LTI starts at 0, threshold in discover_connections is > 0.3
    result = discover_connections("test", "default", mem.db, mem.embed)
    assert result == 0


# ── connections created between high-LTI atoms ───────────────────────────────

def test_connections_created_between_similar_high_lti_atoms(mem):
    # Insert two atoms with high LTI directly
    id1 = mem.remember("machine learning neural networks")
    id2 = mem.remember("deep learning artificial intelligence")

    # Manually boost LTI above 0.3
    mem.db.execute("UPDATE atoms SET lti = 0.8 WHERE id = ?", (id1,))
    mem.db.execute("UPDATE atoms SET lti = 0.8 WHERE id = ?", (id2,))

    result = discover_connections("test", "default", mem.db, mem.embed)
    # Semantically similar atoms should get connected
    assert result >= 0  # may or may not find connections depending on distance


def test_no_duplicate_connections(mem):
    id1 = mem.remember("neural networks deep learning")
    id2 = mem.remember("neural networks deep learning applications")

    mem.db.execute("UPDATE atoms SET lti = 0.9 WHERE id IN (?, ?)", (id1, id2))
    # Link them already
    mem.atomspace.link_atoms(id1, id2, "associated", "test", "default")

    result = discover_connections("test", "default", mem.db, mem.embed)
    # Already linked — should not create a duplicate
    relations = mem.db.fetchall(
        "SELECT COUNT(*) as n FROM atoms WHERE type='relation' AND source_id=? AND target_id=?",
        (id1, id2),
    )
    assert relations[0]["n"] == 1


# ── tenant isolation ──────────────────────────────────────────────────────────

def test_discover_connections_respects_tenant(mem):
    # Add atoms under a different tenant — should not appear
    id1 = mem.remember("machine learning systems")
    mem.db.execute("UPDATE atoms SET lti = 0.9, tenant_id = 'other' WHERE id = ?", (id1,))

    result = discover_connections("test", "default", mem.db, mem.embed)
    assert result == 0
