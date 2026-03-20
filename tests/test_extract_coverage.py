"""Coverage tests for extraction/extract.py."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from smrti import Smrti
from smrti.extraction.extract import (
    _apply_thinking_mode,
    _build_entity_context,
    _link_claims,
    _resolve_ner_entities,
    extract_and_link,
    extract_and_link_hybrid,
    extract_and_link_serialized,
    extract_claims_only,
    extract_knowledge,
)


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def run(coro):
    return asyncio.run(coro)


# ── _apply_thinking_mode ────────────────────────────────────────────────────

def test_apply_thinking_disabled():
    body = {}
    _apply_thinking_mode(body, "disabled")
    assert body["chat_template_kwargs"]["enable_thinking"] is False


def test_apply_thinking_enabled():
    body = {}
    _apply_thinking_mode(body, "enabled")
    assert body["chat_template_kwargs"]["enable_thinking"] is True


def test_apply_thinking_auto_noop():
    body = {}
    _apply_thinking_mode(body, "auto")
    assert "chat_template_kwargs" not in body


# ── _build_entity_context ────────────────────────────────────────────────────

def test_build_entity_context_empty(mem):
    ctx = _build_entity_context(mem)
    assert isinstance(ctx, str)
    assert ctx == ""


def test_build_entity_context_with_atoms(mem):
    from smrti.core.models import Atom, AtomType, EntityType
    atom = Atom(
        type=AtomType.CONCEPT,
        label="Alice",
        entity_type=EntityType.PERSON,
        tenant_id=mem.tenant_id,
        space=mem.write_space,
    )
    mem.atomspace.add_atom(atom)
    ctx = _build_entity_context(mem)
    assert "Alice" in ctx
    assert "person" in ctx


# ── _link_claims ─────────────────────────────────────────────────────────────

def test_link_claims_basic(mem):
    a = mem.remember("Alice")
    b = mem.remember("Python")
    entity_ids = {"Alice": a, "Python": b}

    _link_claims(
        [{"subject": "Alice", "predicate": "uses", "object": "Python", "valence": 0.0}],
        entity_ids, mem,
    )
    # Relation should exist
    rows = mem.db.fetchall(
        "SELECT * FROM atoms WHERE type='relation' AND source_id=?", (a,)
    )
    assert len(rows) >= 1


def test_link_claims_negative_valence_propagates(mem):
    ep_id = mem.remember("Episode with negative claim")
    subj_id = mem.remember("Subject atom")
    obj_id = mem.remember("Broken thing")
    entity_ids = {"subject": subj_id, "broken": obj_id}

    _link_claims(
        [{"subject": "subject", "predicate": "broke", "object": "broken", "valence": -0.8}],
        entity_ids, mem, episode_id=ep_id,
    )
    # Object atom should have negative valence
    row = mem.db.fetchone("SELECT valence FROM atoms WHERE id=?", (obj_id,))
    assert row["valence"] <= -0.8

    # Episode should have negative valence
    ep_row = mem.db.fetchone("SELECT valence FROM atoms WHERE id=?", (ep_id,))
    assert ep_row["valence"] <= -0.8


def test_link_claims_auto_creates_missing_object(mem):
    subj_id = mem.remember("Subject")
    entity_ids = {"Subject": subj_id}

    _link_claims(
        [{"subject": "Subject", "predicate": "knows", "object": "NewConcept", "valence": 0.0}],
        entity_ids, mem,
    )
    # NewConcept should have been created
    row = mem.db.fetchone(
        "SELECT id FROM atoms WHERE label='NewConcept' AND tenant_id=?", (mem.tenant_id,)
    )
    assert row is not None


def test_link_claims_has_goal_promotes_to_goal(mem):
    subj_id = mem.remember("Agent")
    entity_ids = {"Agent": subj_id}

    _link_claims(
        [{"subject": "Agent", "predicate": "has_goal", "object": "BuildAI", "valence": 0.0}],
        entity_ids, mem,
    )
    row = mem.db.fetchone(
        "SELECT type FROM atoms WHERE label='BuildAI' AND tenant_id=?", (mem.tenant_id,)
    )
    assert row is not None
    assert row["type"] == "goal"


def test_link_claims_skips_same_subject_object(mem):
    atom_id = mem.remember("Same atom")
    entity_ids = {"Same": atom_id}
    # Subject == object — should be a no-op
    _link_claims(
        [{"subject": "Same", "predicate": "self", "object": "Same", "valence": 0.0}],
        entity_ids, mem,
    )


# ── extract_knowledge (mock HTTP) ────────────────────────────────────────────

def _mock_http_response(payload: dict, status: int = 200):
    msg = {"role": "assistant", "content": json.dumps(payload)}
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"choices": [{"message": msg}]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    return mock_client


def test_extract_knowledge_success():
    payload = {"entities": [{"name": "Alice", "type": "person"}], "claims": []}
    client = _mock_http_response(payload)

    with patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        async def _run():
            return await extract_knowledge(
                "Alice is cool.", client, "http://localhost", "Bearer x", "gpt-4o"
            )
        result = asyncio.run(_run())

    assert result is not None
    assert result["entities"][0]["name"] == "Alice"


def test_extract_knowledge_strips_markdown_fence():
    raw = '```json\n{"entities": [], "claims": []}\n```'
    msg = {"role": "assistant", "content": raw}
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": msg}]}
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)

    with patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        async def _run():
            return await extract_knowledge("test", client, "http://localhost", "", "m")
        result = asyncio.run(_run())

    assert result == {"entities": [], "claims": []}


def test_extract_knowledge_uses_reasoning_content_fallback():
    msg = {"role": "assistant", "content": "", "reasoning_content": '{"entities": [], "claims": []}'}
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": msg}]}
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)

    with patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        async def _run():
            return await extract_knowledge("test", client, "http://localhost", "", "m")
        result = asyncio.run(_run())

    assert result == {"entities": [], "claims": []}


def test_extract_knowledge_returns_none_on_error():
    client = AsyncMock()
    client.post = AsyncMock(side_effect=Exception("network error"))

    with patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        async def _run():
            return await extract_knowledge("test", client, "http://localhost", "", "m")
        result = asyncio.run(_run())

    assert result is None


# ── extract_claims_only (mock HTTP) ─────────────────────────────────────────

def test_extract_claims_only_success():
    payload = {"claims": [{"subject": "A", "predicate": "knows", "object": "B", "valence": 0.0}]}
    client = _mock_http_response(payload)

    with patch("smrti.extraction.extract._get_http", return_value=client), \
         patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        async def _run():
            return await extract_claims_only(
                "A knows B",
                [{"name": "A", "type": "person"}, {"name": "B", "type": "person"}],
                "http://localhost", "", "m",
            )
        result = asyncio.run(_run())

    assert result is not None
    assert len(result["claims"]) == 1


def test_extract_claims_only_with_entity_context():
    payload = {"claims": []}
    client = _mock_http_response(payload)

    with patch("smrti.extraction.extract._get_http", return_value=client), \
         patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        async def _run():
            return await extract_claims_only(
                "He likes it",
                [{"name": "User", "type": "person"}],
                "http://localhost", "", "m",
                entity_context="- User (person)",
            )
        result = asyncio.run(_run())

    assert result == {"claims": []}


def test_extract_claims_only_returns_none_on_error():
    client = AsyncMock()
    client.post = AsyncMock(side_effect=Exception("timeout"))

    with patch("smrti.extraction.extract._get_http", return_value=client), \
         patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        async def _run():
            return await extract_claims_only(
                "test", [{"name": "X", "type": "person"}],
                "http://localhost", "", "m",
            )
        result = asyncio.run(_run())

    assert result is None


# ── extract_and_link ─────────────────────────────────────────────────────────

def test_extract_and_link_noop_on_none_response(mem):
    payload = {"entities": [{"name": "Alice", "type": "person"}], "claims": []}
    client = _mock_http_response(payload)

    async def _run():
        with patch("smrti.extraction.extract._get_http", return_value=client), \
             patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
            ep_id = mem.remember("Alice is here")
            await extract_and_link(ep_id, "Alice is here", mem, "", "m", "http://localhost")

    asyncio.run(_run())


def test_extract_and_link_noop_when_extraction_fails(mem):
    client = AsyncMock()
    client.post = AsyncMock(side_effect=Exception("fail"))

    async def _run():
        with patch("smrti.extraction.extract._get_http", return_value=client), \
             patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
            ep_id = mem.remember("Some text")
            await extract_and_link(ep_id, "Some text", mem, "", "m", "http://localhost")

    asyncio.run(_run())


# ── extract_and_link_hybrid ──────────────────────────────────────────────────

def test_hybrid_llm_mode_delegates_to_extract_and_link(mem):
    called = []

    async def fake_extract_and_link(*args, **kwargs):
        called.append(True)

    async def _run():
        with patch("smrti.extraction.extract.extract_and_link", fake_extract_and_link):
            ep_id = mem.remember("test")
            await extract_and_link_hybrid(ep_id, "test", mem, "", "m", "http://localhost", mode="llm")

    asyncio.run(_run())
    assert called


def test_hybrid_agent_source_delegates_to_extract_and_link(mem):
    called = []

    async def fake_extract_and_link(*args, **kwargs):
        called.append(True)

    async def _run():
        with patch("smrti.extraction.extract.extract_and_link", fake_extract_and_link):
            ep_id = mem.remember("agent output")
            await extract_and_link_hybrid(ep_id, "agent output", mem, "", "m", "http://localhost", source="agent")

    asyncio.run(_run())
    assert called


def test_hybrid_local_mode_no_llm_call(mem):
    extract_called = []

    async def _run():
        mock_ner = MagicMock()
        mock_ner.extract.return_value = [{"name": "Alice", "type": "person"}, {"name": "Bob", "type": "person"}]

        with patch("smrti.extraction.extract.extract_and_link") as mock_llm, \
             patch("smrti.extraction.ner.get_ner", return_value=mock_ner), \
             patch("smrti.extraction.extract.extract_claims_only") as mock_claims:
            ep_id = mem.remember("Alice and Bob")
            await extract_and_link_hybrid(ep_id, "Alice and Bob", mem, "", "m", "http://localhost", mode="local")
            extract_called.append(mock_llm.call_count)
            extract_called.append(mock_claims.call_count)

    asyncio.run(_run())
    # In local mode, LLM should not be called
    assert extract_called[0] == 0  # extract_and_link not called
    assert extract_called[1] == 0  # extract_claims_only not called


def test_hybrid_fallback_on_ner_exception(mem):
    called = []

    async def fake_extract_and_link(*args, **kwargs):
        called.append(True)

    async def _run():
        with patch("smrti.extraction.extract.extract_and_link", fake_extract_and_link), \
             patch("smrti.extraction.ner.get_ner", side_effect=ImportError("no gliner")):
            ep_id = mem.remember("some text")
            await extract_and_link_hybrid(ep_id, "some text", mem, "", "m", "http://localhost", mode="hybrid")

    asyncio.run(_run())
    assert called


def test_hybrid_empty_ner_falls_through_to_llm(mem):
    called = []

    async def fake_extract_and_link(*args, **kwargs):
        called.append(True)

    async def _run():
        mock_ner = MagicMock()
        mock_ner.extract.return_value = []  # NER found nothing

        with patch("smrti.extraction.extract.extract_and_link", fake_extract_and_link), \
             patch("smrti.extraction.ner.get_ner", return_value=mock_ner):
            ep_id = mem.remember("abstract text")
            await extract_and_link_hybrid(ep_id, "abstract text", mem, "", "m", "http://localhost", mode="hybrid")

    asyncio.run(_run())
    assert called


def test_hybrid_fewer_than_2_entities_skips_claims(mem):
    claims_called = []

    async def _run():
        mock_ner = MagicMock()
        # Only 1 unique entity — claims not called
        mock_ner.extract.return_value = [{"name": "Alice", "type": "person"}]

        with patch("smrti.extraction.ner.get_ner", return_value=mock_ner), \
             patch("smrti.extraction.extract.extract_claims_only") as mock_claims:
            ep_id = mem.remember("Alice alone")
            await extract_and_link_hybrid(ep_id, "Alice alone", mem, "", "m", "http://localhost", mode="hybrid")
            claims_called.append(mock_claims.call_count)

    asyncio.run(_run())
    assert claims_called[0] == 0


# ── extract_and_link_serialized ───────────────────────────────────────────────

def test_serialized_acquires_lock_and_runs(mem):
    """Serialized wrapper should complete without error."""
    async def _run():
        mock_ner = MagicMock()
        mock_ner.extract.return_value = []

        with patch("smrti.extraction.ner.get_ner", return_value=mock_ner), \
             patch("smrti.extraction.extract.extract_and_link") as mock_llm:
            ep_id = mem.remember("serialized test")
            await extract_and_link_serialized(ep_id, "serialized test", mem, "", "m", "http://localhost")
            # No error is sufficient

    asyncio.run(_run())
