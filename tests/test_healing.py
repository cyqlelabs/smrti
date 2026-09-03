"""Tests for graph healing — orphaned episode repair."""
from __future__ import annotations

import uuid

import pytest

from smrti import Smrti
from smrti.evolution.healing import heal_orphaned_episodes


@pytest.fixture
def mem(tmp_path):
    db_path = str(tmp_path / "test.db")
    return Smrti(db_path=db_path, personality="balanced", tenant_id="t1", write_space="s1")


def _add_concept(mem, label: str, entity_type: str = "concept") -> str:
    atom_id = str(uuid.uuid4())
    mem.db.execute(
        """INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti, valence, intensity)
           VALUES (?, 'concept', ?, ?, ?, ?, 0.8, 0.5, 0.5, 0.5, 0.0, 0.0)""",
        (atom_id, label, entity_type, mem.tenant_id, mem.write_space),
    )
    return atom_id


def _add_episode(mem, content: str) -> str:
    atom_id = str(uuid.uuid4())
    mem.db.execute(
        """INSERT INTO atoms (id, type, label, content, tenant_id, space, probability, confidence, sti, lti, valence, intensity)
           VALUES (?, 'episode', ?, ?, ?, ?, 0.75, 0.5, 0.5, 0.0, 0.0, 0.0)""",
        (atom_id, content[:60], content, mem.tenant_id, mem.write_space),
    )
    return atom_id


def _add_relation(mem, source_id: str, target_id: str, relation: str) -> str:
    rel_id = str(uuid.uuid4())
    mem.db.execute(
        """INSERT INTO atoms (id, type, label, source_id, target_id, relation, tenant_id, space, probability, confidence, sti, lti, valence, intensity)
           VALUES (?, 'relation', ?, ?, ?, ?, ?, ?, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)""",
        (rel_id, relation, source_id, target_id, relation, mem.tenant_id, mem.write_space),
    )
    return rel_id


def test_heals_orphaned_episodes(mem):
    """Episodes mentioning concepts but not a person get linked to the most salient person."""
    person_id = _add_concept(mem, "Elias", "person")
    concept_a = _add_concept(mem, "Python")
    concept_b = _add_concept(mem, "FastAPI")

    ep1 = _add_episode(mem, "I love using Python for web development")
    ep2 = _add_episode(mem, "FastAPI is great for building APIs")

    # Both episodes mention concepts but NOT the person
    _add_relation(mem, ep1, concept_a, "mentions")
    _add_relation(mem, ep2, concept_b, "mentions")

    healed = heal_orphaned_episodes(mem.tenant_id, mem.write_space, mem.db)
    assert healed == 2

    # Check that mentions edges to person were created
    for ep_id in (ep1, ep2):
        row = mem.db.fetchone(
            """SELECT 1 FROM atoms WHERE type = 'relation' AND relation = 'mentions'
               AND source_id = ? AND target_id = ? AND tenant_id = ? AND space = ?""",
            (ep_id, person_id, mem.tenant_id, mem.write_space),
        )
        assert row is not None, f"Episode {ep_id} should now mention person"

    # And nothing else: the placeholder person -> concept edges an earlier
    # version drew made the person a hub joined to every concept in the space.
    for concept_id in (concept_a, concept_b):
        row = mem.db.fetchone(
            """SELECT 1 FROM atoms WHERE type = 'relation'
               AND source_id = ? AND target_id = ? AND tenant_id = ? AND space = ?""",
            (person_id, concept_id, mem.tenant_id, mem.write_space),
        )
        assert row is None, "healing must not turn the person into a hub"


def test_no_healing_when_person_already_linked(mem):
    """Episodes that already mention a person are not healed."""
    person_id = _add_concept(mem, "Elias", "person")
    concept_a = _add_concept(mem, "Python")

    ep1 = _add_episode(mem, "Elias likes Python")
    _add_relation(mem, ep1, concept_a, "mentions")
    _add_relation(mem, ep1, person_id, "mentions")

    healed = heal_orphaned_episodes(mem.tenant_id, mem.write_space, mem.db)
    assert healed == 0


def test_no_healing_without_person(mem):
    """If no person atom exists, nothing is healed."""
    concept_a = _add_concept(mem, "Python")
    ep1 = _add_episode(mem, "Python is great")
    _add_relation(mem, ep1, concept_a, "mentions")

    healed = heal_orphaned_episodes(mem.tenant_id, mem.write_space, mem.db)
    assert healed == 0


def test_idempotent(mem):
    """Running healing twice doesn't create duplicate edges."""
    person_id = _add_concept(mem, "Elias", "person")
    concept_a = _add_concept(mem, "Python")
    ep1 = _add_episode(mem, "I love Python")
    _add_relation(mem, ep1, concept_a, "mentions")

    healed1 = heal_orphaned_episodes(mem.tenant_id, mem.write_space, mem.db)
    assert healed1 == 1

    healed2 = heal_orphaned_episodes(mem.tenant_id, mem.write_space, mem.db)
    assert healed2 == 0
