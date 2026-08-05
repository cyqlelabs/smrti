"""Tests for space-scoped KNN retrieval, person-injection caps, and salience conservation."""
import os
import tempfile

import pytest

from smrti.core.atomspace import AtomSpace
from smrti.core.db import Database
from smrti.core.embed import EmbeddingProvider
from smrti.core.models import Atom, AtomType, EntityType, TruthValue
from smrti.retrieval.fan_out import retrieve
from smrti.retrieval.salience import compute_salience


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


def _concept(label, space, **kw):
    return Atom(
        type=AtomType.CONCEPT,
        label=label,
        tenant_id="t",
        space=space,
        truth=TruthValue(probability=0.8, confidence=0.8),
        **kw,
    )


def test_crowded_sibling_space_cannot_starve_read_space(atomspace, db, embed):
    """A tenant space packed with perfect query matches must not consume the
    KNN candidate budget of a recall that reads a different space."""
    query = "deployment pipeline configuration"
    for i in range(55):
        atomspace.add_atom(_concept(f"{query} variant {i}", "crowded"))
    atomspace.add_atom(_concept("pipeline deployment settings", "quiet"))

    results = retrieve(
        query, "t", ["quiet"], db, embed, write_space="quiet", top_k=5
    )
    assert [r.atom.label for r in results] == ["pipeline deployment settings"]
    assert results[0].similarity > 0.5  # true cosine similarity, not L2 leftovers


def test_person_injection_is_capped(atomspace, db, embed):
    """Persons that are not KNN hits enter only via injection, capped at 3."""
    query = "gardening tips"
    # 55 close matches keep the KNN top-50 free of person atoms, so any
    # person in the results arrived through the injection path.
    for i in range(55):
        atomspace.add_atom(_concept(f"gardening tips volume {i}", "s"))
    for i in range(6):
        atomspace.add_atom(
            _concept(f"Zorblax Qynthor {i}", "s", entity_type=EntityType.PERSON)
        )

    results = retrieve(query, "t", ["s"], db, embed, write_space="s", top_k=100)
    persons = [r for r in results if r.atom.entity_type == EntityType.PERSON]
    assert len(persons) <= 3


def test_salience_weight_shift_conserves_mass():
    # Raw boost (|v| * intensity * valence_weight = 1.0) far exceeds w_sti:
    # the shift must cap at w_sti, not mint new weight for w_valence.
    score = compute_salience(
        similarity=0.0,
        sti=2.0,
        confidence=0.0,
        lti=0.0,
        valence=-1.0,
        intensity=1.0,
        w_similarity=0.35,
        w_sti=0.05,
        w_confidence=0.20,
        w_lti=0.10,
        w_valence=0.10,
        valence_weight=1.0,
    )
    assert score == pytest.approx(0.15)  # (0.10 + 0.05) * |v| * intensity


def test_null_personality_column_falls_back_to_default(atomspace, db, embed):
    atomspace.add_atom(_concept("hiking trails", "s"))
    db.execute(
        "INSERT INTO personality (tenant_id, space, w_similarity) VALUES ('t', 's', NULL)"
    )
    # Must not raise TypeError from None arithmetic.
    results = retrieve("hiking", "t", ["s"], db, embed, write_space="s", top_k=3)
    assert results
