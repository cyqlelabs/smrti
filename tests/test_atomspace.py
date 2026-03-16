"""Tests for core atomspace operations."""
import os
import tempfile

import pytest

from smrti.core.atomspace import AtomSpace
from smrti.core.db import Database
from smrti.core.embed import EmbeddingProvider
from smrti.core.models import Atom, AtomType, EntityType, TruthValue


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
    atom = Atom(type=AtomType.CONCEPT, label="Python", tenant_id="test", space="default")
    atom_id = atomspace.add_atom(atom)
    assert atom_id is not None

    retrieved = atomspace.get_atom(atom_id, "test", "default")
    assert retrieved is not None
    assert retrieved.label == "Python"
    assert retrieved.type == AtomType.CONCEPT


def test_link_atoms(atomspace):
    a = Atom(type=AtomType.CONCEPT, label="Alice", entity_type=EntityType.PERSON, tenant_id="test", space="default")
    b = Atom(type=AtomType.CONCEPT, label="Acme Corp", entity_type=EntityType.ORGANIZATION, tenant_id="test", space="default")
    a_id = atomspace.add_atom(a)
    b_id = atomspace.add_atom(b)

    rel_id = atomspace.link_atoms(a_id, b_id, "works_at", "test", "default")
    assert rel_id is not None

    neighbors = atomspace.get_neighbors(a_id, "test", ["default"])
    assert any(n.label == "Acme Corp" for n in neighbors)


def test_boost_sti(atomspace):
    atom = Atom(type=AtomType.CONCEPT, label="important", tenant_id="test", space="default")
    atom_id = atomspace.add_atom(atom)

    atomspace.boost_sti(atom_id, 1.0)
    updated = atomspace.get_atom(atom_id, "test", "default")
    assert updated.attention.sti > 0


# ── LTI protection for severe negative-valence atoms ─────────────────────────

def test_severe_negative_valence_gets_lti_floor(atomspace):
    from smrti.core.models import Valence
    atom = Atom(
        type=AtomType.EPISODE,
        label="critical mistake",
        tenant_id="test",
        space="default",
        valence=Valence(valence=-0.9, intensity=0.9),
    )
    atom_id = atomspace.add_atom(atom)
    retrieved = atomspace.get_atom(atom_id, "test", "default")
    assert retrieved.attention.lti >= 0.5


def test_mild_negative_valence_no_lti_floor(atomspace):
    from smrti.core.models import Valence
    atom = Atom(
        type=AtomType.EPISODE,
        label="minor issue",
        tenant_id="test",
        space="default",
        valence=Valence(valence=-0.3, intensity=0.5),
    )
    atom_id = atomspace.add_atom(atom)
    retrieved = atomspace.get_atom(atom_id, "test", "default")
    assert retrieved.attention.lti == 0.0


def test_lti_floor_does_not_lower_existing_lti(atomspace):
    from smrti.core.models import AttentionValue, Valence
    atom = Atom(
        type=AtomType.EPISODE,
        label="already high lti",
        tenant_id="test",
        space="default",
        attention=AttentionValue(sti=0.0, lti=0.8),
        valence=Valence(valence=-0.9, intensity=0.9),
    )
    atom_id = atomspace.add_atom(atom)
    retrieved = atomspace.get_atom(atom_id, "test", "default")
    assert retrieved.attention.lti >= 0.8
