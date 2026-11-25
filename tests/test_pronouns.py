"""Tests for pronoun detection and merge logic."""
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from smrti import Smrti


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def _mock_ner(pronoun_names=None):
    """Create a mock NERProvider where classify_pronoun returns True for given names."""
    pronoun_set = {n.lower() for n in (pronoun_names or [])}
    mock = MagicMock()
    mock.classify_pronoun.side_effect = lambda name: name.lower() in pronoun_set
    return mock


# ── Batch merge tests ────────────────────────────────────────────────────────
#
# A pronoun is what the tagger typed ``pronoun`` in context; no word list is
# consulted, so a name like "Mia" typed person stays a person.


def test_batch_merge_single_person_drops_the_pronoun():
    """1 named person + 1 pronoun → the pronoun is dropped, not folded into the
    person: "I" may be the speaker talking about someone else."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    entities = [
        {"name": "Elara", "type": "person", "aliases": []},
        {"name": "I", "type": "pronoun", "aliases": ["my"]},
        {"name": "systems strategist", "type": "concept", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities)

    names = [e["name"] for e in result]
    assert "Elara" in names
    assert "I" not in names
    assert "systems strategist" in names
    elara = next(e for e in result if e["name"] == "Elara")
    assert elara["aliases"] == []


def test_batch_merge_multi_person_drops_the_pronoun():
    """2 named persons + pronoun → who it means is ambiguous, so it is dropped."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    entities = [
        {"name": "Elara", "type": "person", "aliases": []},
        {"name": "Dave", "type": "person", "aliases": []},
        {"name": "I", "type": "pronoun", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities)

    names = [e["name"] for e in result]
    assert names == ["Elara", "Dave"]
    assert all("I" not in e["aliases"] for e in result)


def test_batch_merge_no_named_removes_pronoun():
    """Only pronouns → all removed, no orphan atoms."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    entities = [
        {"name": "I", "type": "pronoun", "aliases": []},
        {"name": "me", "type": "pronoun", "aliases": []},
        {"name": "Python", "type": "tool", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities)

    names = [e["name"] for e in result]
    assert "I" not in names
    assert "me" not in names
    assert "Python" in names


def test_batch_merge_nonperson_untouched():
    """Non-person entities (concepts, tools) are never merged."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    entities = [
        {"name": "Python", "type": "tool", "aliases": []},
        {"name": "Django", "type": "tool", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities)
    assert len(result) == 2


def test_batch_merge_keeps_names_a_word_list_took_for_pronouns():
    """"Mia" and "Ella" are Italian and Spanish pronouns too; typed person by
    the tagger, they are people."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    entities = [
        {"name": "Mia", "type": "person", "aliases": []},
        {"name": "Ella", "type": "person", "aliases": []},
        {"name": "Alps", "type": "location", "aliases": []},
    ]
    assert merge_pronoun_entities_in_batch(entities) == entities


def test_batch_merge_dual_typed_pronoun():
    """When the same name is typed both person and pronoun, treat it as a pronoun."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    entities = [
        {"name": "I", "type": "person", "aliases": []},
        {"name": "I", "type": "pronoun", "aliases": []},
        {"name": "slow productivity", "type": "preference", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities)

    names = [e["name"] for e in result]
    # "I" should be removed (no named person to merge into, all pronoun)
    assert "I" not in names
    assert "slow productivity" in names


def test_batch_merge_dual_typed_with_named_person():
    """'I' typed both person and pronoun alongside a real name is a pronoun, and
    is dropped rather than folded into the name."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    entities = [
        {"name": "Elias", "type": "person", "aliases": []},
        {"name": "I", "type": "person", "aliases": []},
        {"name": "I", "type": "pronoun", "aliases": []},
        {"name": "Python", "type": "tool", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities)

    names = [e["name"] for e in result]
    assert "Elias" in names
    assert "I" not in names
    assert "Python" in names
    elias = next(e for e in result if e["name"] == "Elias")
    assert "I" not in elias["aliases"]


def test_a_lone_pronoun_is_dropped_even_when_an_old_alias_names_someone(mem):
    """The alias table used to resolve a lone pronoun, and another text's
    "she" stored as Bob's made "She moved to Berlin." Bob's move."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    import uuid
    bob_id = str(uuid.uuid4())
    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti) VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)",
        (bob_id, "concept", "Bob", "person", "test", "default"),
    )
    mem.db.execute(
        "INSERT OR IGNORE INTO aliases (alias, atom_id, tenant_id, space) VALUES (?, ?, ?, ?)",
        ("she", bob_id, "test", "default"),
    )

    entities = [
        {"name": "She", "type": "pronoun", "aliases": []},
        {"name": "Berlin", "type": "location", "aliases": []},
    ]
    assert merge_pronoun_entities_in_batch(entities) == [entities[1]]


# ── Pronoun gate in resolve loop ─────────────────────────────────────────────


def test_pronoun_gate_skips_creation(mem):
    """Entity with type='pronoun' is skipped in _resolve_ner_entities."""
    from smrti.extraction.extract import _resolve_ner_entities

    episode_id = mem.remember("test episode", type="episode")

    mock_ner = _mock_ner()

    entities = [
        {"name": "Elara", "type": "person", "aliases": []},
        {"name": "I", "type": "pronoun", "aliases": []},
    ]

    with patch("smrti.extraction.ner.get_ner", return_value=mock_ner):
        entity_ids = _resolve_ner_entities(entities, episode_id, mem)

    assert "Elara" in entity_ids
    # "I" is never folded into the person the text names: it may be the speaker.
    assert "I" not in entity_ids

    # No separate atom with label "I" should exist
    row = mem.db.fetchone(
        "SELECT id FROM atoms WHERE label = 'I' AND entity_type = 'person' AND tenant_id = 'test'",
        (),
    )
    assert row is None


# ── Graceful fallback without GLiNER2 ────────────────────────────────────────


def test_no_ner_runtime_graceful_fallback(mem):
    """Without the NER runtime installed, pronoun handling is disabled but resolve works."""
    from smrti.extraction.extract import _resolve_ner_entities

    episode_id = mem.remember("test episode", type="episode")

    entities = [
        {"name": "Elara", "type": "person", "aliases": []},
        {"name": "I", "type": "person", "aliases": []},
    ]

    with patch("smrti.extraction.ner.get_ner", side_effect=ImportError("no gliner2_onnx")):
        entity_ids = _resolve_ner_entities(entities, episode_id, mem)

    # Both should be resolved (no pronoun filtering without GLiNER2)
    assert "Elara" in entity_ids
    assert "I" in entity_ids
