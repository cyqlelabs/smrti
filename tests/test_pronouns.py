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


def test_batch_merge_single_person_absorbs_pronoun():
    """1 named person + 1 pronoun → pronoun merged into named person."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    ner = _mock_ner(["I", "my"])
    entities = [
        {"name": "Elara", "type": "person", "aliases": []},
        {"name": "I", "type": "person", "aliases": ["my"]},
        {"name": "systems strategist", "type": "concept", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities, ner)

    names = [e["name"] for e in result]
    assert "Elara" in names
    assert "I" not in names
    assert "systems strategist" in names
    # Pronoun aliases merged into Elara
    elara = next(e for e in result if e["name"] == "Elara")
    assert "I" in elara["aliases"]
    assert "my" in elara["aliases"]


def test_batch_merge_multi_person_skips():
    """2 named persons + pronoun → pronoun kept (ambiguous)."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    ner = _mock_ner(["I"])
    entities = [
        {"name": "Elara", "type": "person", "aliases": []},
        {"name": "Dave", "type": "person", "aliases": []},
        {"name": "I", "type": "person", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities, ner)

    names = [e["name"] for e in result]
    assert "Elara" in names
    assert "Dave" in names
    assert "I" in names  # kept because ambiguous


def test_batch_merge_no_named_removes_pronoun():
    """Only pronoun persons → all removed, no orphan atoms."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    ner = _mock_ner(["I", "me"])
    entities = [
        {"name": "I", "type": "person", "aliases": []},
        {"name": "me", "type": "person", "aliases": []},
        {"name": "Python", "type": "tool", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities, ner)

    names = [e["name"] for e in result]
    assert "I" not in names
    assert "me" not in names
    assert "Python" in names


def test_batch_merge_nonperson_untouched():
    """Non-person entities (concepts, tools) are never merged."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    ner = _mock_ner(["I"])
    entities = [
        {"name": "Python", "type": "tool", "aliases": []},
        {"name": "Django", "type": "tool", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities, ner)
    assert len(result) == 2


def test_batch_merge_pronoun_type_from_gliner():
    """Entities with type='pronoun' from GLiNER2 are treated as pronouns."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    ner = _mock_ner()  # classify_pronoun returns False for everything
    entities = [
        {"name": "Elara", "type": "person", "aliases": []},
        {"name": "I", "type": "pronoun", "aliases": ["my"]},
    ]
    result = merge_pronoun_entities_in_batch(entities, ner)

    names = [e["name"] for e in result]
    assert "Elara" in names
    assert "I" not in names
    elara = next(e for e in result if e["name"] == "Elara")
    assert "I" in elara["aliases"]


def test_batch_merge_dual_typed_pronoun():
    """When GLiNER emits same name as both person and pronoun, treat as pronoun."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    ner = _mock_ner()  # classify_pronoun returns False for everything
    entities = [
        {"name": "I", "type": "person", "aliases": []},
        {"name": "I", "type": "pronoun", "aliases": []},
        {"name": "slow productivity", "type": "preference", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities, ner)

    names = [e["name"] for e in result]
    # "I" should be removed (no named person to merge into, all pronoun)
    assert "I" not in names
    assert "slow productivity" in names


def test_batch_merge_dual_typed_with_named_person():
    """When GLiNER emits 'I' as both person and pronoun alongside a real name, merge into named."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    ner = _mock_ner()  # classify_pronoun returns False for everything
    entities = [
        {"name": "Elias", "type": "person", "aliases": []},
        {"name": "I", "type": "person", "aliases": []},
        {"name": "I", "type": "pronoun", "aliases": []},
        {"name": "Python", "type": "tool", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(entities, ner)

    names = [e["name"] for e in result]
    assert "Elias" in names
    assert "I" not in names
    assert "Python" in names
    elias = next(e for e in result if e["name"] == "Elias")
    assert "I" in elias["aliases"]


# ── Alias-based pronoun resolution tests ─────────────────────────────────────


def test_batch_merge_resolves_pronoun_via_alias(mem):
    """When only pronouns in batch but 'I' is alias of existing person, resolve to that person."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    # Create Elias as a person atom with "I" as alias
    import uuid
    elias_id = str(uuid.uuid4())
    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti) VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)",
        (elias_id, "concept", "Elias", "person", "test", "default"),
    )
    mem.db.execute(
        "INSERT OR IGNORE INTO aliases (alias, atom_id, tenant_id, space) VALUES (?, ?, ?, ?)",
        ("I", elias_id, "test", "default"),
    )

    ner = _mock_ner(["I", "my"])
    entities = [
        {"name": "I", "type": "person", "aliases": ["my"]},
        {"name": "intellectual humility", "type": "preference", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(
        entities, ner, db=mem.db, tenant_id="test", spaces=["default"],
    )

    names = [e["name"] for e in result]
    assert "Elias" in names
    assert "I" not in names
    assert "intellectual humility" in names
    # "I" and "my" should be in Elias's aliases
    elias = next(e for e in result if e["name"] == "Elias")
    assert "I" in elias["aliases"]
    assert "my" in elias["aliases"]


def test_batch_merge_no_alias_removes_pronoun(mem):
    """When only pronouns in batch and no alias exists, remove them."""
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    ner = _mock_ner(["I"])
    entities = [
        {"name": "I", "type": "person", "aliases": []},
        {"name": "Python", "type": "tool", "aliases": []},
    ]
    result = merge_pronoun_entities_in_batch(
        entities, ner, db=mem.db, tenant_id="test", spaces=["default"],
    )

    names = [e["name"] for e in result]
    assert "I" not in names
    assert "Python" in names


# ── Retroactive merge tests ──────────────────────────────────────────────────


def test_retroactive_merge_same_episode(mem):
    """Pronoun atom co-mentioned with named person in same episode → merged."""
    from smrti.extraction.pronouns import find_and_merge_pronoun_atoms

    # Create an episode
    episode_id = mem.remember("I am Elara", type="episode")

    # Create a pronoun atom and a named atom
    import uuid
    pronoun_id = str(uuid.uuid4())
    named_id = str(uuid.uuid4())

    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti) VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)",
        (pronoun_id, "concept", "I", "person", "test", "default"),
    )
    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti) VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)",
        (named_id, "concept", "Elara", "person", "test", "default"),
    )

    # Create mentions edges: episode → pronoun, episode → named
    mem.atomspace.link_atoms(episode_id, pronoun_id, "mentions", "test", "default")
    mem.atomspace.link_atoms(episode_id, named_id, "mentions", "test", "default")

    # Create a target atom and relation edge from pronoun
    target_id = str(uuid.uuid4())
    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti) VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)",
        (target_id, "concept", "strategist", "concept", "test", "default"),
    )
    mem.atomspace.link_atoms(pronoun_id, target_id, "is", "test", "default")

    ner = _mock_ner(["I"])

    find_and_merge_pronoun_atoms(named_id, episode_id, mem.db, ner, "test", "default")

    # Pronoun atom should be deleted
    row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (pronoun_id,))
    assert row is None

    # "I" should be an alias of the named atom
    alias_row = mem.db.fetchone(
        "SELECT atom_id FROM aliases WHERE alias = 'I' AND tenant_id = 'test'",
        (),
    )
    assert alias_row is not None
    assert alias_row["atom_id"] == named_id


def test_retroactive_merge_different_episode_skipped(mem):
    """Pronoun from different episode → NOT merged."""
    from smrti.extraction.pronouns import find_and_merge_pronoun_atoms

    episode1_id = mem.remember("first episode", type="episode")
    episode2_id = mem.remember("second episode", type="episode")

    import uuid
    pronoun_id = str(uuid.uuid4())
    named_id = str(uuid.uuid4())

    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti) VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)",
        (pronoun_id, "concept", "I", "person", "test", "default"),
    )
    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti) VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)",
        (named_id, "concept", "Elara", "person", "test", "default"),
    )

    # Pronoun mentioned in episode1, named in episode2
    mem.atomspace.link_atoms(episode1_id, pronoun_id, "mentions", "test", "default")
    mem.atomspace.link_atoms(episode2_id, named_id, "mentions", "test", "default")

    ner = _mock_ner(["I"])

    # Looking from episode2's perspective — pronoun not co-mentioned
    find_and_merge_pronoun_atoms(named_id, episode2_id, mem.db, ner, "test", "default")

    # Pronoun atom should still exist
    row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (pronoun_id,))
    assert row is not None


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
    # "I" is absorbed as alias of Elara (batch merge), so it maps to same atom
    assert entity_ids.get("I") == entity_ids["Elara"]

    # No separate atom with label "I" should exist
    row = mem.db.fetchone(
        "SELECT id FROM atoms WHERE label = 'I' AND entity_type = 'person' AND tenant_id = 'test'",
        (),
    )
    assert row is None


# ── Graceful fallback without GLiNER2 ────────────────────────────────────────


def test_no_gliner2_graceful_fallback(mem):
    """Without GLiNER2 installed, pronoun handling is disabled but resolve works."""
    from smrti.extraction.extract import _resolve_ner_entities

    episode_id = mem.remember("test episode", type="episode")

    entities = [
        {"name": "Elara", "type": "person", "aliases": []},
        {"name": "I", "type": "person", "aliases": []},
    ]

    with patch("smrti.extraction.ner.get_ner", side_effect=ImportError("no gliner2")):
        entity_ids = _resolve_ner_entities(entities, episode_id, mem)

    # Both should be resolved (no pronoun filtering without GLiNER2)
    assert "Elara" in entity_ids
    assert "I" in entity_ids
