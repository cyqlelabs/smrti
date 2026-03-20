"""Additional atomspace coverage: update_atom, link_atoms idempotency,
get_neighbors empty, get_relations, search_by_label with entity_type,
add_evidence, mark_evidence_processed."""
from __future__ import annotations

import os
import tempfile

import pytest

from smrti import Smrti
from smrti.core.atomspace import AtomSpace
from smrti.core.models import (
    Atom, AtomType, AttentionValue, Evidence, TruthValue, Valence,
)


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def test_update_atom(mem):
    atom_id = mem.remember("Original content")
    row_before = mem.db.fetchone("SELECT content FROM atoms WHERE id=?", (atom_id,))
    assert row_before["content"] == "Original content"

    atom = mem.atomspace.get_atom(atom_id, mem.tenant_id, mem.write_space)
    atom.label = "Updated"
    atom.truth = TruthValue(probability=0.99, confidence=0.9)
    mem.atomspace.update_atom(atom)

    row_after = mem.db.fetchone("SELECT label, probability FROM atoms WHERE id=?", (atom_id,))
    assert row_after["label"] == "Updated"
    assert abs(row_after["probability"] - 0.99) < 0.001


def test_link_atoms_idempotent(mem):
    a = mem.remember("Node A")
    b = mem.remember("Node B")
    id1 = mem.atomspace.link_atoms(a, b, "relates", mem.tenant_id, mem.write_space)
    id2 = mem.atomspace.link_atoms(a, b, "relates", mem.tenant_id, mem.write_space)
    assert id1 == id2
    # STI should have been boosted on second call
    row = mem.db.fetchone("SELECT sti FROM atoms WHERE id=?", (id1,))
    assert row["sti"] > 0


def test_get_neighbors_empty(mem):
    atom_id = mem.remember("Isolated atom")
    neighbors = mem.atomspace.get_neighbors(atom_id, mem.tenant_id, [mem.write_space])
    assert neighbors == []


def test_get_relations(mem):
    a = mem.remember("Source atom")
    b = mem.remember("Target atom")
    mem.atomspace.link_atoms(a, b, "likes", mem.tenant_id, mem.write_space)

    relations = mem.atomspace.get_relations(a, mem.tenant_id, [mem.write_space])
    assert len(relations) >= 1
    rel = relations[0]
    assert rel.type == AtomType.RELATION
    assert rel.source_id == a
    assert rel.target_id == b


def test_search_by_label_with_entity_type(mem):
    from smrti.core.models import EntityType
    atom = Atom(
        type=AtomType.CONCEPT,
        label="Alice the developer",
        entity_type=EntityType.PERSON,
        tenant_id=mem.tenant_id,
        space=mem.write_space,
    )
    mem.atomspace.add_atom(atom)

    results = mem.atomspace.search_by_label("Alice", mem.tenant_id, [mem.write_space], entity_type="person")
    assert any(a.label == "Alice the developer" for a in results)


def test_search_by_label_no_entity_type(mem):
    mem.remember("Search target content")
    results = mem.atomspace.search_by_label("Search target", mem.tenant_id, [mem.write_space])
    assert len(results) >= 1


def test_add_evidence_and_mark_processed(mem):
    atom_id = mem.remember("A belief")
    ev = Evidence(
        atom_id=atom_id,
        observed_probability=0.9,
        tenant_id=mem.tenant_id,
        space=mem.write_space,
    )
    mem.atomspace.add_evidence(ev)

    pending = mem.atomspace.get_pending_evidence(mem.tenant_id, mem.write_space)
    assert any(e.atom_id == atom_id for e in pending)

    mem.atomspace.mark_evidence_processed(ev.id)
    pending_after = mem.atomspace.get_pending_evidence(mem.tenant_id, mem.write_space)
    assert all(e.id != ev.id for e in pending_after)
