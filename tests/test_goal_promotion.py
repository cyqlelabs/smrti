"""Tests for goal promotion via has_goal claims."""
from __future__ import annotations

import uuid

import pytest

from smrti import Smrti
from smrti.extraction.extract import _link_claims, _promote_to_goal


@pytest.fixture
def mem(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Smrti(db_path=db_path, personality="balanced", tenant_id="t1", write_space="s1")


def _add_atom(mem, label: str, atom_type: str = "concept", entity_type: str = "project") -> str:
    atom_id = str(uuid.uuid4())
    mem.db.execute(
        """INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti, valence, intensity)
           VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.5, 0.5, 0.3, 0.0, 0.0)""",
        (atom_id, atom_type, label, entity_type, mem.tenant_id, mem.write_space),
    )
    return atom_id


def test_promote_to_goal(mem):
    """_promote_to_goal changes type and entity_type to goal."""
    atom_id = _add_atom(mem, "permaculture project", "concept", "project")

    _promote_to_goal(atom_id, mem)

    row = mem.db.fetchone("SELECT type, entity_type FROM atoms WHERE id = ?", (atom_id,))
    assert row["type"] == "goal"
    assert row["entity_type"] == "goal"


def test_promote_noop_if_already_goal(mem):
    """_promote_to_goal is a no-op if atom is already a goal."""
    atom_id = _add_atom(mem, "ship mobile app", "goal", "goal")

    _promote_to_goal(atom_id, mem)

    row = mem.db.fetchone("SELECT type, entity_type FROM atoms WHERE id = ?", (atom_id,))
    assert row["type"] == "goal"
    assert row["entity_type"] == "goal"


def test_link_claims_promotes_on_has_goal(mem):
    """_link_claims promotes the object atom to goal when predicate is has_goal."""
    person_id = _add_atom(mem, "Elias", "concept", "person")
    project_id = _add_atom(mem, "permaculture project", "concept", "project")

    entity_ids = {"Elias": person_id, "permaculture project": project_id}
    claims = [{"subject": "Elias", "predicate": "has_goal", "object": "permaculture project", "valence": 0.8}]

    _link_claims(claims, entity_ids, mem)

    row = mem.db.fetchone("SELECT type, entity_type FROM atoms WHERE id = ?", (project_id,))
    assert row["type"] == "goal"
    assert row["entity_type"] == "goal"


def test_link_claims_no_promote_on_other_predicates(mem):
    """_link_claims does not promote atoms for non-has_goal predicates."""
    person_id = _add_atom(mem, "Elias", "concept", "person")
    project_id = _add_atom(mem, "permaculture project", "concept", "project")

    entity_ids = {"Elias": person_id, "permaculture project": project_id}
    claims = [{"subject": "Elias", "predicate": "works_on", "object": "permaculture project"}]

    _link_claims(claims, entity_ids, mem)

    row = mem.db.fetchone("SELECT type, entity_type FROM atoms WHERE id = ?", (project_id,))
    assert row["type"] == "concept"
    assert row["entity_type"] == "project"
