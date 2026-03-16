"""Tests for core atomspace operations."""
import os
import tempfile

import pytest

from engram.core.atomspace import AtomSpace
from engram.core.db import Database
from engram.core.embed import EmbeddingProvider
from engram.core.models import Atom, AtomType, EntityType, TruthValue


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = Database(db_path)
    database.initialize()
    yield database
    database.close()
    os.unlink(db_path)


@pytest.fixture
def embed():
    return EmbeddingProvider()


@pytest.fixture
def atomspace(db, embed):
    return AtomSpace(db, embed)


def test_add_and_get_atom(atomspace):
    atom = Atom(type=AtomType.CONCEPT, label="Python", agent_id="test")
    atom_id = atomspace.add_atom(atom)
    assert atom_id is not None

    retrieved = atomspace.get_atom(atom_id, "test")
    assert retrieved is not None
    assert retrieved.label == "Python"
    assert retrieved.type == AtomType.CONCEPT


def test_link_atoms(atomspace):
    a = Atom(type=AtomType.CONCEPT, label="Alice", entity_type=EntityType.PERSON, agent_id="test")
    b = Atom(type=AtomType.CONCEPT, label="Acme Corp", entity_type=EntityType.ORGANIZATION, agent_id="test")
    a_id = atomspace.add_atom(a)
    b_id = atomspace.add_atom(b)

    rel_id = atomspace.link_atoms(a_id, b_id, "works_at", "test")
    assert rel_id is not None

    neighbors = atomspace.get_neighbors(a_id, "test")
    assert any(n.label == "Acme Corp" for n in neighbors)


def test_boost_sti(atomspace):
    atom = Atom(type=AtomType.CONCEPT, label="important", agent_id="test")
    atom_id = atomspace.add_atom(atom)

    atomspace.boost_sti(atom_id, 1.0)
    updated = atomspace.get_atom(atom_id, "test")
    assert updated.attention.sti > 0
