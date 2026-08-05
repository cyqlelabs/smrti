"""Tests for KNN-index hygiene, entity_type round-trip, and read clamping."""
import os
import tempfile

import pytest

from smrti.core.atomspace import AtomSpace
from smrti.core.db import Database
from smrti.core.embed import EmbeddingProvider
from smrti.core.models import Atom, AtomType, EntityType, atom_from_row


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


def _vec_rows(db):
    return db.fetchall("SELECT atom_id, label FROM vec_atoms")


def test_relation_atoms_are_not_embedded(atomspace, db):
    a = atomspace.add_atom(Atom(type=AtomType.CONCEPT, label="Rust", tenant_id="t", space="s"))
    b = atomspace.add_atom(Atom(type=AtomType.CONCEPT, label="Cargo", tenant_id="t", space="s"))
    rel_id = atomspace.link_atoms(a, b, "uses", "t", "s")

    ids = {r["atom_id"] for r in _vec_rows(db)}
    assert a in ids and b in ids
    assert rel_id not in ids


def test_add_atom_refreshes_stale_embedding(atomspace, db, embed):
    atom = Atom(type=AtomType.CONCEPT, label="cooking recipes", tenant_id="t", space="s")
    atomspace.add_atom(atom)

    atom.label = "quantum physics"
    atomspace.add_atom(atom)

    rows = _vec_rows(db)
    assert len(rows) == 1
    assert rows[0]["label"] == "quantum physics"

    import struct

    q = embed.embed("quantum physics")
    hit = db.fetchone(
        "SELECT atom_id, distance FROM vec_atoms WHERE embedding MATCH ? AND tenant_id = ? ORDER BY distance LIMIT 1",
        (struct.pack(f"{len(q)}f", *q), "t"),
    )
    assert hit["atom_id"] == atom.id
    assert hit["distance"] < 0.1  # near-identical text under cosine distance


def test_update_atom_refreshes_embedding(atomspace, db):
    atom = Atom(type=AtomType.CONCEPT, label="old label", tenant_id="t", space="s")
    atomspace.add_atom(atom)

    atom.label = "brand new label"
    atomspace.update_atom(atom)

    rows = _vec_rows(db)
    assert len(rows) == 1
    assert rows[0]["label"] == "brand new label"


def test_entity_type_round_trips_losslessly(atomspace):
    atom = Atom(
        type=AtomType.CONCEPT,
        label="FastAPI",
        entity_type=EntityType.TECHNOLOGY,
        tenant_id="t",
        space="s",
    )
    atomspace.add_atom(atom)

    loaded = atomspace.get_atom(atom.id, "t", "s")
    assert loaded.entity_type == EntityType.TECHNOLOGY

    # A read-modify-write cycle must not degrade the stored type.
    atomspace.update_atom(loaded)
    reloaded = atomspace.get_atom(atom.id, "t", "s")
    assert reloaded.entity_type == EntityType.TECHNOLOGY


@pytest.mark.parametrize("etype", ["role", "skill", "topic", "media", "health", "pronoun"])
def test_all_canonical_ner_labels_are_enum_members(etype):
    assert EntityType(etype).value == etype


def test_atom_from_row_clamps_out_of_range_values(db):
    db.execute(
        """INSERT INTO atoms (id, type, label, tenant_id, space, valence, intensity, lti, probability)
           VALUES ('x1', 'concept', 'poisoned', 't', 's', -999.0, 7.0, 1.5, 2.0)"""
    )
    row = db.fetchone("SELECT * FROM atoms WHERE id = 'x1'")
    atom = atom_from_row(row)
    assert atom.valence.valence == -1.0
    assert atom.valence.intensity == 1.0
    assert atom.attention.lti == 1.0
    assert atom.truth.probability == 1.0


def test_search_by_label_escapes_wildcards(atomspace):
    atomspace.add_atom(Atom(type=AtomType.CONCEPT, label="100% coverage", tenant_id="t", space="s"))
    atomspace.add_atom(Atom(type=AtomType.CONCEPT, label="100 units", tenant_id="t", space="s"))

    hits = atomspace.search_by_label("100%", "t", ["s"])
    assert [a.label for a in hits] == ["100% coverage"]
