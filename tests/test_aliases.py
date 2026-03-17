"""Tests for AliasManager (extraction/aliases.py)."""
import os
import tempfile

import pytest

from smrti import Smrti
from smrti.extraction.aliases import AliasManager


@pytest.fixture
def setup():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    am = AliasManager(engine.db)
    yield engine, am
    engine.close()
    os.unlink(db_path)


def _atom(engine, content="alias test atom"):
    return engine.remember(content)


def test_lookup_returns_none_when_empty(setup):
    _, am = setup
    assert am.lookup("JS", "test", ["default"]) is None


def test_add_and_lookup(setup):
    engine, am = setup
    aid = _atom(engine, "JavaScript framework")
    am.add(aid, "JS", "test", "default")
    assert am.lookup("JS", "test", ["default"]) == aid


def test_lookup_case_insensitive(setup):
    engine, am = setup
    aid = _atom(engine, "TypeScript lang")
    am.add(aid, "TypeScript", "test", "default")
    assert am.lookup("typescript", "test", ["default"]) == aid
    assert am.lookup("TYPESCRIPT", "test", ["default"]) == aid


def test_lookup_across_spaces(setup):
    engine, am = setup
    aid = _atom(engine, "Python language")
    am.add(aid, "Python", "test", "default")
    assert am.lookup("Python", "test", ["other-space", "default"]) == aid


def test_lookup_respects_tenant(setup):
    engine, am = setup
    aid = _atom(engine, "Rust language")
    am.add(aid, "Rust", "test", "default")
    # Different tenant should not find it
    assert am.lookup("Rust", "other-tenant", ["default"]) is None


def test_get_all_for_atom_empty(setup):
    _, am = setup
    assert am.get_all_for_atom("nonexistent", "test", "default") == []


def test_get_all_for_atom(setup):
    engine, am = setup
    aid = _atom(engine, "Go language")
    am.add(aid, "go-alias-1", "test", "default")
    am.add(aid, "go-alias-2", "test", "default")
    aliases = am.get_all_for_atom(aid, "test", "default")
    assert set(aliases) == {"go-alias-1", "go-alias-2"}


def test_delete_for_atom(setup):
    engine, am = setup
    aid = _atom(engine, "Kotlin language")
    am.add(aid, "kotlin-alias", "test", "default")
    am.delete_for_atom(aid, "test", "default")
    assert am.lookup("kotlin-alias", "test", ["default"]) is None


def test_delete_does_not_affect_other_atoms(setup):
    engine, am = setup
    aid1 = _atom(engine, "Swift language")
    aid2 = _atom(engine, "Elixir language")
    am.add(aid1, "swift-keep", "test", "default")
    am.add(aid2, "elixir-delete", "test", "default")
    am.delete_for_atom(aid2, "test", "default")
    assert am.lookup("swift-keep", "test", ["default"]) == aid1
    assert am.lookup("elixir-delete", "test", ["default"]) is None


def test_add_duplicate_is_idempotent(setup):
    engine, am = setup
    aid = _atom(engine, "Ruby language")
    am.add(aid, "ruby-dup", "test", "default")
    am.add(aid, "ruby-dup", "test", "default")
    aliases = am.get_all_for_atom(aid, "test", "default")
    assert aliases.count("ruby-dup") == 1
