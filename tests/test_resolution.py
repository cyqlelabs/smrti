"""Tests for entity resolution."""
import os
import tempfile

import pytest

from smrti.core.db import Database
from smrti.core.embed import EmbeddingProvider
from smrti.extraction.resolve import EntityResolver


@pytest.fixture
def resolver():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = Database(db_path)
    db.initialize()
    embed = EmbeddingProvider()
    res = EntityResolver(db, embed)
    yield res
    db.close()
    os.unlink(db_path)


def test_creates_new_atom(resolver):
    atom_id = resolver.resolve("Python", "tool", "test", "default", ["default"])
    assert atom_id is not None


def test_exact_match_returns_same_id(resolver):
    id1 = resolver.resolve("Alice", "person", "test", "default", ["default"])
    id2 = resolver.resolve("Alice", "person", "test", "default", ["default"])
    assert id1 == id2


def test_fuzzy_match(resolver):
    id1 = resolver.resolve("JavaScript", "tool", "test", "default", ["default"])
    id2 = resolver.resolve("Javascript", "tool", "test", "default", ["default"])
    assert id1 == id2


def test_different_entity_types_create_separate_atoms(resolver):
    """A goal entity must not be absorbed into a semantically similar concept atom."""
    concept_id = resolver.resolve("build a memory system", "concept", "test", "default", ["default"])
    goal_id = resolver.resolve("build a memory system", "goal", "test", "default", ["default"])
    assert concept_id != goal_id

    concept_row = resolver.db.fetchone("SELECT type, entity_type FROM atoms WHERE id = ?", (concept_id,))
    goal_row = resolver.db.fetchone("SELECT type, entity_type FROM atoms WHERE id = ?", (goal_id,))
    assert concept_row["type"] == "concept"
    assert goal_row["type"] == "goal"
    assert goal_row["entity_type"] == "goal"


def test_alias_lookup_returns_same_id(resolver):
    """Tier 1: alias resolution returns the existing atom."""
    id1 = resolver.resolve("NodeJS", "tool", "test", "default", ["default"])
    # Manually register a secondary alias for the same atom
    resolver.aliases.add(id1, "Node.js", "test", "default")
    id2 = resolver.resolve("Node.js", "tool", "test", "default", ["default"])
    assert id1 == id2


def test_create_atom_sets_goal_type(resolver):
    """Goal entity type maps to 'goal' atom type."""
    atom_id = resolver.resolve("ship v2.0", "goal", "test", "default", ["default"])
    row = resolver.db.fetchone("SELECT type FROM atoms WHERE id = ?", (atom_id,))
    assert row["type"] == "goal"


def test_create_atom_sets_preference_as_belief(resolver):
    atom_id = resolver.resolve("prefer dark mode", "preference", "test", "default", ["default"])
    row = resolver.db.fetchone("SELECT type FROM atoms WHERE id = ?", (atom_id,))
    assert row["type"] == "belief"


def test_create_atom_sets_constraint_as_belief(resolver):
    atom_id = resolver.resolve("must use TLS", "constraint", "test", "default", ["default"])
    row = resolver.db.fetchone("SELECT type FROM atoms WHERE id = ?", (atom_id,))
    assert row["type"] == "belief"


def test_boost_sti_on_exact_match(resolver):
    atom_id = resolver.resolve("Python", "tool", "test", "default", ["default"])
    before = resolver.db.fetchone("SELECT sti FROM atoms WHERE id = ?", (atom_id,))
    resolver.resolve("Python", "tool", "test", "default", ["default"])
    after = resolver.db.fetchone("SELECT sti FROM atoms WHERE id = ?", (atom_id,))
    assert after["sti"] >= before["sti"]
