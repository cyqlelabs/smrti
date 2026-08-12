"""Coverage tests for extraction plumbing not exercised by the happy path.

Covers the per-loop lock registry, the HTTP client cache, entity-context
injection, the claims-only failure modes, and the hybrid path where the
LLM emits brand-new entities alongside its claims.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smrti import Smrti
from smrti.extraction import extract as extract_mod
from smrti.extraction.extract import (
    _PerLoopLocks,
    _agent_trust,
    _get_http,
    _resolve_ner_entities,
    extract_and_link_hybrid,
    extract_claims_only,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def mem(db_path):
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()


def _response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


def _chat_payload(content):
    return {"choices": [{"message": {"content": content}}]}


# ── _PerLoopLocks ─────────────────────────────────────────────────────────────

def test_per_loop_locks_are_reused_within_a_loop():
    locks = _PerLoopLocks()

    async def _run():
        first = locks.get_lock("tenant/space")
        assert locks.get_lock("tenant/space") is first
        assert "tenant/space" in locks
        assert locks["tenant/space"] is first

    run(_run())


def test_per_loop_locks_lookup_raises_for_an_unknown_key():
    locks = _PerLoopLocks()

    async def _run():
        locks.get_lock("known")

    run(_run())
    assert "missing" not in locks
    with pytest.raises(KeyError):
        locks["missing"]


def test_per_loop_locks_evict_closed_loops():
    locks = _PerLoopLocks()
    run(_evict_helper(locks))
    # The first loop is closed; a new loop starts from a clean registry.
    assert run(_evict_helper(locks)) == 1
    locks.clear()
    assert "tenant/space" not in locks


async def _evict_helper(locks):
    locks.get_lock("tenant/space")
    return len(locks._entries)


# ── _get_http ─────────────────────────────────────────────────────────────────

def test_http_client_is_cached_per_loop_and_evicted_when_closed():
    async def _run():
        client = _get_http()
        assert _get_http() is client
        return client

    first = run(_run())  # this loop is closed once asyncio.run returns

    async def _run_again():
        return _get_http()

    second = run(_run_again())
    assert second is not first  # the closed loop's entry was evicted
    assert len(extract_mod._http_clients) == 1
    extract_mod._http_clients.clear()


# ── _agent_trust ──────────────────────────────────────────────────────────────

def test_agent_trust_falls_back_to_the_schema_default(mem):
    assert _agent_trust(mem) == pytest.approx(0.5)
    mem.db.execute(
        "UPDATE personality SET agent_source_trust = NULL WHERE tenant_id = ? AND space = ?",
        (mem.tenant_id, mem.write_space),
    )
    assert _agent_trust(mem) == pytest.approx(0.5)


def test_agent_trust_reads_the_configured_value(mem):
    mem.db.execute(
        "UPDATE personality SET agent_source_trust = 0.9 WHERE tenant_id = ? AND space = ?",
        (mem.tenant_id, mem.write_space),
    )
    assert _agent_trust(mem) == pytest.approx(0.9)


# ── _resolve_ner_entities ─────────────────────────────────────────────────────

def test_resolve_ner_entities_skips_blank_and_pronoun_spans(mem):
    episode_id = mem.remember("She works with Nico at Cyqle")
    entities = [
        {"name": "  ", "type": "person"},
        {"name": "she", "type": "pronoun"},
        {"name": "Cyqle", "type": "organization"},
    ]
    with patch("smrti.extraction.pronouns.merge_pronoun_entities_in_batch", side_effect=lambda e, *a, **k: e):
        entity_ids = _resolve_ner_entities(entities, episode_id, mem)
    assert set(entity_ids) == {"Cyqle", "cyqle"}


def test_resolve_ner_entities_coerces_an_unknown_type_to_concept(mem):
    episode_id = mem.remember("We discussed astrophysics")
    with patch("smrti.extraction.pronouns.merge_pronoun_entities_in_batch", side_effect=lambda e, *a, **k: e):
        entity_ids = _resolve_ner_entities(
            [{"name": "astrophysics", "type": "not-a-real-type"}], episode_id, mem
        )
    atom_id = entity_ids["astrophysics"]
    row = mem.db.fetchone("SELECT entity_type FROM atoms WHERE id = ?", (atom_id,))
    assert row["entity_type"] == "concept"


# ── extract_claims_only ───────────────────────────────────────────────────────

def _claims_only(mem_tenant="test", **kwargs):
    defaults = dict(
        text="Nico works at Cyqle",
        entities=[{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}],
        upstream="http://llm.local",
        auth="Bearer k",
        model="qwen",
        tenant_id=mem_tenant,
    )
    defaults.update(kwargs)
    return run(extract_claims_only(**defaults))


def test_claims_only_returns_none_on_an_upstream_error():
    with patch("smrti.extraction.extract._get_http", return_value=MagicMock()):
        with patch("smrti.extraction.extract._post_chat", new=AsyncMock(return_value=_response({}, status=503))):
            assert _claims_only() is None


def test_claims_only_returns_none_when_the_response_is_not_an_object():
    payload = _chat_payload("[1, 2, 3]")
    with patch("smrti.extraction.extract._get_http", return_value=MagicMock()):
        with patch("smrti.extraction.extract._post_chat", new=AsyncMock(return_value=_response(payload))):
            assert _claims_only() is None


def test_claims_only_returns_none_when_the_request_raises():
    with patch("smrti.extraction.extract._get_http", return_value=MagicMock()):
        with patch("smrti.extraction.extract._post_chat", new=AsyncMock(side_effect=RuntimeError("down"))):
            assert _claims_only() is None


def test_claims_only_strips_a_markdown_code_fence():
    fenced = '```json\n{"claims": [{"subject": "Nico", "predicate": "works_for", "object": "Cyqle"}]}\n```'
    with patch("smrti.extraction.extract._get_http", return_value=MagicMock()):
        with patch("smrti.extraction.extract._post_chat", new=AsyncMock(return_value=_response(_chat_payload(fenced)))):
            parsed = _claims_only()
    assert parsed["claims"][0]["predicate"] == "works_for"


def test_claims_only_prepends_the_known_entity_context():
    captured = {}

    async def _capture(_http, _upstream, _auth, body):
        captured["body"] = body
        return _response(_chat_payload('{"claims": []}'))

    with patch("smrti.extraction.extract._get_http", return_value=MagicMock()):
        with patch("smrti.extraction.extract._post_chat", new=_capture):
            _claims_only(entity_context="Nico [person]")

    user_message = captured["body"]["messages"][1]["content"]
    assert "[Known entities" in user_message
    assert "Nico [person]" in user_message
    assert "[Text to extract]" in user_message


# ── extract_and_link_hybrid: LLM-emitted entities ─────────────────────────────

def _hybrid(mem, content, ner_entities, claims_result):
    ner_instance = MagicMock()
    ner_instance.extract.return_value = ner_entities
    ner_instance.classify_pronoun.return_value = False
    episode_id = mem.remember(content)
    with patch("smrti.extraction.ner.get_ner", return_value=ner_instance):
        with patch("smrti.extraction.extract.extract_claims_only",
                   new=AsyncMock(return_value=claims_result)):
            run(extract_and_link_hybrid(
                episode_id, content, mem, "", "qwen", "http://llm.local", "user",
            ))
    return episode_id


def test_hybrid_stops_when_the_claims_call_fails(mem):
    episode_id = _hybrid(
        mem,
        "Nico deployed smrti at Cyqle",
        [{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}],
        None,
    )
    assert mem.db.fetchall(
        "SELECT id FROM atoms WHERE type = 'relation' AND relation = 'works_for'"
    ) == []
    assert episode_id


def test_hybrid_creates_concept_atoms_for_new_llm_entities(mem):
    _hybrid(
        mem,
        "Nico is learning Rust at Cyqle",
        [{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}],
        {
            "entities": [
                {"name": "Rust", "type": "technology"},
                {"name": "", "type": "technology"},        # dropped: no name
                {"name": "Ship v2", "type": "milestone"},   # dropped: type not allowed
            ],
            "claims": [],
        },
    )
    row = mem.db.fetchone(
        "SELECT type, entity_type FROM atoms WHERE u_lower(label) = u_lower('Rust')"
    )
    assert row["type"] == "concept"
    assert mem.db.fetchone(
        "SELECT id FROM atoms WHERE u_lower(label) = u_lower('Ship v2')"
    ) is None
    # New non-goal entities are linked back to the episode.
    assert mem.db.fetchall(
        "SELECT id FROM atoms WHERE type = 'relation' AND relation = 'mentions'"
    )


def test_hybrid_reclassifies_a_preference_into_a_belief(mem):
    _hybrid(
        mem,
        "Nico prefers dark mode when working at Cyqle",
        [{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}],
        {"entities": [{"name": "dark mode", "type": "preference"}], "claims": []},
    )
    row = mem.db.fetchone(
        "SELECT type, entity_type FROM atoms WHERE u_lower(label) = u_lower('dark mode')"
    )
    assert row["type"] == "belief"
    assert row["entity_type"] == "preference"


def test_hybrid_falls_back_to_the_full_llm_path_when_ner_raises(mem):
    content = "Nico ships smrti"
    episode_id = mem.remember(content)
    ner_instance = MagicMock()
    ner_instance.extract.side_effect = RuntimeError("model missing")
    full = AsyncMock()
    with patch("smrti.extraction.ner.get_ner", return_value=ner_instance):
        with patch("smrti.extraction.extract.extract_and_link", new=full):
            run(extract_and_link_hybrid(
                episode_id, content, mem, "", "qwen", "http://llm.local", "user",
            ))
    full.assert_awaited_once()


def test_local_mode_swallows_an_ner_failure(mem):
    content = "Nico ships smrti"
    episode_id = mem.remember(content)
    ner_instance = MagicMock()
    ner_instance.extract.side_effect = RuntimeError("model missing")
    full = AsyncMock()
    with patch("smrti.extraction.ner.get_ner", return_value=ner_instance):
        with patch("smrti.extraction.extract.extract_and_link", new=full):
            run(extract_and_link_hybrid(
                episode_id, content, mem, "", "qwen", "http://llm.local", "user",
                mode="local",
            ))
    full.assert_not_awaited()
