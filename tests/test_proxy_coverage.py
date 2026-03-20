"""Coverage tests for proxy helper functions: _remember, _enrich_content, _build_query."""
from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from smrti import Smrti
from smrti.core.models import (
    Atom, AtomType, AttentionValue, EntityType, RecallResult, TruthValue, Valence,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="default", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def _recall_result(content, entity_type=None, valence=0.0, atom_type=AtomType.CONCEPT):
    atom = Atom(
        type=atom_type,
        label=content[:80],
        content=content if atom_type != AtomType.CONCEPT else "",
        entity_type=entity_type,
        truth=TruthValue(probability=0.8, confidence=0.8),
        attention=AttentionValue(sti=0.5, lti=0.3),
        valence=Valence(valence=valence, intensity=abs(valence)),
    )
    return RecallResult(atom=atom, salience=0.5, similarity=0.7)


# ── _remember ─────────────────────────────────────────────────────────────────

def test_remember_stores_episode(mem):
    from smrti.servers.proxy import _remember
    with patch("smrti.servers.proxy.get_mem", return_value=mem):
        atom_id = run(_remember("Hello world", "default", "default"))
    assert atom_id


def test_remember_returns_empty_for_ignored_content(mem):
    from smrti.servers.proxy import _remember
    mem._ignore_re = [__import__("re").compile(r"IGNORE_THIS")]
    with patch("smrti.servers.proxy.get_mem", return_value=mem):
        atom_id = run(_remember("IGNORE_THIS content", "default", "default"))
    assert atom_id == ""


def test_remember_deduplicates(mem):
    from smrti.servers.proxy import _remember
    with patch("smrti.servers.proxy.get_mem", return_value=mem):
        id1 = run(_remember("Duplicate content", "default", "default"))
        id2 = run(_remember("Duplicate content", "default", "default"))
    assert id1
    assert id2 == ""  # second call is a dup


def test_remember_assistant_source(mem):
    from smrti.servers.proxy import _remember
    with patch("smrti.servers.proxy.get_mem", return_value=mem):
        atom_id = run(_remember("Assistant reply text", "default", "default", source="agent"))
    assert atom_id


# ── _enrich_content ───────────────────────────────────────────────────────────

def test_enrich_content_with_existing_content(mem):
    from smrti.servers.proxy import _enrich_content
    atom = Atom(
        type=AtomType.EPISODE,
        label="Episode",
        content="Already has content",
        tenant_id="default",
        space="default",
    )
    mem.atomspace.add_atom(atom)
    r = RecallResult(atom=atom, salience=0.5, similarity=0.7)
    result = _enrich_content(r, mem)
    assert result == "Already has content"


def test_enrich_content_concept_no_relations(mem):
    from smrti.servers.proxy import _enrich_content
    atom = Atom(
        type=AtomType.CONCEPT,
        label="Orphan concept",
        content="",
        entity_type=EntityType.PERSON,
        tenant_id="default",
        space="default",
    )
    mem.atomspace.add_atom(atom)
    r = RecallResult(atom=atom, salience=0.5, similarity=0.7)
    result = _enrich_content(r, mem)
    assert "Orphan concept" in result
    assert "person" in result


def test_enrich_content_concept_with_relations(mem):
    from smrti.servers.proxy import _enrich_content
    from smrti.core.models import EntityType
    atom_a = Atom(
        type=AtomType.CONCEPT,
        label="Nico",
        content="",
        entity_type=EntityType.PERSON,
        tenant_id="default",
        space="default",
    )
    mem.atomspace.add_atom(atom_a)

    atom_b = Atom(
        type=AtomType.CONCEPT,
        label="GetProductized",
        content="",
        entity_type=EntityType.ORGANIZATION,
        tenant_id="default",
        space="default",
    )
    mem.atomspace.add_atom(atom_b)

    mem.atomspace.link_atoms(atom_a.id, atom_b.id, "works_for", "default", "default")

    r = RecallResult(atom=atom_a, salience=0.5, similarity=0.7)
    result = _enrich_content(r, mem)
    assert "Nico" in result
    assert "works_for" in result
    assert "GetProductized" in result


def test_enrich_content_non_concept_returns_label(mem):
    from smrti.servers.proxy import _enrich_content
    # Use EPISODE type (not concept/belief/goal) to hit the non-concept branch
    atom = Atom(
        type=AtomType.EPISODE,
        label="ep label",
        content="",
        tenant_id="default",
        space="default",
    )
    mem.atomspace.add_atom(atom)
    r = RecallResult(atom=atom, salience=0.5, similarity=0.7)
    result = _enrich_content(r, mem)
    assert result == "ep label"


# ── _build_query ─────────────────────────────────────────────────────────────

def test_build_query_last_mode_with_user():
    from smrti.servers.proxy import _build_query
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Second question"},
    ]
    with patch("smrti.servers.proxy._QUERY_MODE", "last"):
        query = _build_query(messages)
    assert query == "Second question"


def test_build_query_last_mode_no_user():
    from smrti.servers.proxy import _build_query
    messages = [{"role": "system", "content": "Be helpful."}]
    with patch("smrti.servers.proxy._QUERY_MODE", "last"):
        query = _build_query(messages)
    assert query is None


def test_build_query_last_mode_multimodal_content():
    from smrti.servers.proxy import _build_query
    messages = [{"role": "user", "content": [{"type": "image_url"}]}]
    with patch("smrti.servers.proxy._QUERY_MODE", "last"):
        query = _build_query(messages)
    assert query is None


# ── get_http ──────────────────────────────────────────────────────────────────

def test_get_http_returns_client():
    import smrti.servers.proxy as proxy_mod
    # Reset so we exercise the None branch
    old = proxy_mod._http
    proxy_mod._http = None
    try:
        client = proxy_mod.get_http()
        assert client is not None
    finally:
        proxy_mod._http = old


# ── _inject_context agent episode filtering ───────────────────────────────────

def test_inject_context_filters_agent_episodes():
    """Agent-sourced episodes must not be injected back as context."""
    from smrti.servers.proxy import _inject_context

    agent_atom = Atom(
        type=AtomType.EPISODE,
        label="Agent output",
        content="Agent output",
        truth=TruthValue(probability=0.8, confidence=0.8),
        attention=AttentionValue(sti=0.5, lti=0.3),
        valence=Valence(valence=0.0, intensity=0.0),
        metadata={"source": "agent"},
    )
    agent_result = RecallResult(atom=agent_atom, salience=0.5, similarity=0.7)

    body = {"messages": [{"role": "user", "content": "Hello"}]}

    async def _run():
        with patch("smrti.servers.proxy._recall", return_value=[agent_result]):
            result, ctx, mems = await _inject_context(body, "t1", "s1", ["s1"])
        return result, ctx

    result, ctx = run(_run())
    assert ctx == ""
    assert result == body
