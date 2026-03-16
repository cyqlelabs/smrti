"""Tests for entity resolution."""
import os
import tempfile

import pytest

from engram.core.db import Database
from engram.core.embed import EmbeddingProvider
from engram.extraction.resolve import EntityResolver


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
