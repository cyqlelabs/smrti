"""Hardening tests for the extraction layer (audit fixes)."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smrti import Smrti
from smrti.core.db import Database
from smrti.core.embed import EmbeddingProvider
from smrti.extraction.extract import (
    _db_resolve_label,
    _link_claims,
    _validate_extraction,
    extract_and_link_hybrid,
    extract_and_link_serialized,
    extract_knowledge,
)
from smrti.extraction.resolve import EntityResolver


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


@pytest.fixture
def resolver():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = Database(db_path)
    db.initialize()
    embed = EmbeddingProvider()
    res = EntityResolver(db, embed)
    yield res
    db.close()
    os.unlink(db_path)


# ── Claim valence clamping and per-claim isolation ───────────────────────────


def test_claim_valence_clamped_before_db_write(mem):
    ep = mem.remember("episode with wild valence")
    a = mem.remember("Subject")
    b = mem.remember("Object")
    entity_ids = {"Subject": a, "Object": b}

    _link_claims(
        [{"subject": "Subject", "predicate": "broke", "object": "Object", "valence": -999}],
        entity_ids, mem, episode_id=ep,
    )
    rel = mem.db.fetchone(
        "SELECT valence FROM atoms WHERE type='relation' AND source_id=? AND target_id=?",
        (a, b),
    )
    assert rel["valence"] == -1.0
    obj = mem.db.fetchone("SELECT valence, intensity FROM atoms WHERE id=?", (b,))
    assert obj["valence"] == -1.0
    assert obj["intensity"] == 1.0
    ep_row = mem.db.fetchone("SELECT valence, intensity FROM atoms WHERE id=?", (ep,))
    assert ep_row["valence"] == -1.0
    assert ep_row["intensity"] == 1.0


def test_non_numeric_valence_does_not_abort_batch(mem):
    a = mem.remember("Subject")
    b = mem.remember("BadObject")
    c = mem.remember("GoodObject")
    entity_ids = {"Subject": a, "BadObject": b, "GoodObject": c}

    _link_claims(
        [
            {"subject": "Subject", "predicate": "hates", "object": "BadObject", "valence": "very bad"},
            {"subject": "Subject", "predicate": "likes", "object": "GoodObject", "valence": 0.5},
        ],
        entity_ids, mem,
    )
    rel = mem.db.fetchone(
        "SELECT id FROM atoms WHERE type='relation' AND relation='likes' AND source_id=? AND target_id=?",
        (a, c),
    )
    assert rel is not None


def test_db_resolve_label_excludes_relation_and_episode(mem):
    mem.db.execute(
        "INSERT INTO atoms (id, type, label, tenant_id, space) VALUES ('ep-x', 'episode', 'Deploy', 'test', 'default')",
    )
    assert _db_resolve_label("Deploy", {}, mem) is None

    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space) VALUES ('c-x', 'concept', 'Deploy', 'concept', 'test', 'default')",
    )
    assert _db_resolve_label("Deploy", {}, mem) == "c-x"


def test_entity_map_case_collision_keeps_exact_keys(mem):
    """Two same-message names differing only in case keep their own atoms."""
    from smrti.extraction.extract import _register_entity

    entity_ids: dict[str, str] = {}
    _register_entity(entity_ids, "Apple", "atom-org")
    _register_entity(entity_ids, "apple", "atom-fruit")
    assert entity_ids["Apple"] == "atom-org"
    assert entity_ids["apple"] == "atom-fruit"


# ── Malformed LLM responses ──────────────────────────────────────────────────


def test_validate_extraction_top_level_list_rejected():
    assert _validate_extraction([1, 2, 3]) is None
    assert _validate_extraction("nope") is None


def test_validate_extraction_drops_invalid_items():
    parsed = _validate_extraction({
        "entities": ["Alice", {"name": "Bob", "type": "person"}, {"name": 5, "type": "person"}],
        "claims": [
            {"subject": "A", "predicate": "p", "object": "B"},
            "junk",
            {"subject": 1, "predicate": "p", "object": "B"},
        ],
    })
    assert parsed["entities"] == [{"name": "Bob", "type": "person"}]
    assert parsed["claims"] == [{"subject": "A", "predicate": "p", "object": "B"}]


def test_validate_extraction_non_list_sections_emptied():
    parsed = _validate_extraction({"entities": "nope", "claims": {"a": 1}})
    assert parsed["entities"] == []
    assert parsed["claims"] == []


def _mock_response(status: int, content: str | None = None):
    resp = MagicMock()
    resp.status_code = status
    if content is not None:
        resp.json.return_value = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    return resp


def test_extract_knowledge_top_level_list_returns_none():
    client = AsyncMock()
    client.post = AsyncMock(return_value=_mock_response(200, json.dumps([1, 2, 3])))

    with patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        result = asyncio.run(
            extract_knowledge("test", client, "http://localhost", "", "m")
        )
    assert result is None


def test_extract_knowledge_non_2xx_returns_none():
    client = AsyncMock()
    client.post = AsyncMock(return_value=_mock_response(500))

    with patch("smrti.servers.config.EXTRACT_THINKING", "auto"):
        result = asyncio.run(
            extract_knowledge("test", client, "http://localhost", "", "m")
        )
    assert result is None
    assert client.post.call_count == 1  # 5xx is not retried


def test_extract_knowledge_4xx_retries_without_chat_template_kwargs():
    ok = _mock_response(200, json.dumps({"entities": [], "claims": []}))
    client = AsyncMock()
    client.post = AsyncMock(side_effect=[_mock_response(400), ok])

    with patch("smrti.servers.config.EXTRACT_THINKING", "disabled"):
        result = asyncio.run(
            extract_knowledge("test", client, "http://localhost", "", "m")
        )
    assert result == {"entities": [], "claims": []}
    assert client.post.call_count == 2
    first_body = client.post.call_args_list[0].kwargs["json"]
    retry_body = client.post.call_args_list[1].kwargs["json"]
    assert "chat_template_kwargs" in first_body
    assert "chat_template_kwargs" not in retry_body


# ── Per-loop asyncio state ───────────────────────────────────────────────────


def test_session_locks_isolated_per_loop():
    from smrti.extraction import extract as ex

    async def _get():
        return ex._session_locks.get_lock("iso:test")

    loop1 = asyncio.new_event_loop()
    try:
        lock1 = loop1.run_until_complete(_get())
        assert loop1.run_until_complete(_get()) is lock1  # cached within loop
    finally:
        loop1.close()
    loop2 = asyncio.new_event_loop()
    try:
        lock2 = loop2.run_until_complete(_get())
    finally:
        loop2.close()
    assert lock1 is not lock2


def test_http_client_isolated_per_loop_and_acloses():
    from smrti.extraction import extract as ex

    async def _get_use_close():
        client = ex._get_http()
        assert ex._get_http() is client  # cached within loop
        await ex.aclose_extract_clients()
        assert client.is_closed
        return client

    loop1 = asyncio.new_event_loop()
    try:
        c1 = loop1.run_until_complete(_get_use_close())
    finally:
        loop1.close()
    loop2 = asyncio.new_event_loop()
    try:
        c2 = loop2.run_until_complete(_get_use_close())
    finally:
        loop2.close()
    assert c1 is not c2


def test_serialized_works_across_sequential_loops():
    """Two sequential event loops must both complete (no cross-loop hang)."""
    calls = []

    async def fake_hybrid(*args, **kwargs):
        calls.append(True)

    class FakeMem:
        tenant_id = "t-loop"
        write_space = "s-loop"

    with patch("smrti.extraction.extract.extract_and_link_hybrid", fake_hybrid):
        asyncio.run(extract_and_link_serialized("e1", "x", FakeMem(), "", "m", "u"))
        asyncio.run(extract_and_link_serialized("e2", "x", FakeMem(), "", "m", "u"))
    assert len(calls) == 2


# ── Pronoun merge alias hygiene ──────────────────────────────────────────────


def _insert_person(mem, atom_id: str, label: str, space: str = "default") -> None:
    mem.db.execute(
        "INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti) VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)",
        (atom_id, "concept", label, "person", "test", space),
    )


def test_merge_pronoun_does_not_persist_pronoun_alias(mem):
    from smrti.extraction.pronouns import merge_pronoun_into_named

    _insert_person(mem, "pron-1", "I")
    _insert_person(mem, "named-1", "Elara")

    merge_pronoun_into_named("pron-1", "named-1", mem.db, "test", "default")

    assert mem.db.fetchone("SELECT id FROM atoms WHERE id = 'pron-1'", ()) is None
    alias_row = mem.db.fetchone(
        "SELECT atom_id FROM aliases WHERE alias = 'I' AND tenant_id = 'test'", ()
    )
    assert alias_row is None


def test_merge_pronoun_delete_is_space_scoped(mem):
    from smrti.extraction.pronouns import merge_pronoun_into_named

    _insert_person(mem, "pron-other", "I", space="other")
    _insert_person(mem, "named-2", "Elara")

    merge_pronoun_into_named("pron-other", "named-2", mem.db, "test", "default")

    # Pronoun atom lives in a different space — must survive the merge
    row = mem.db.fetchone("SELECT id FROM atoms WHERE id = 'pron-other'", ())
    assert row is not None


# ── Unicode-aware label matching (u_lower) ───────────────────────────────────


def test_u_lower_merges_non_ascii_case_variants(resolver):
    """MÜLLER and müller must resolve to the same atom via tier 0.

    WRatio('MÜLLER', 'müller') is 0.0, so the fuzzy tier can never merge
    them — only Unicode-aware exact matching does.
    """
    id1 = resolver.resolve("müller", "person", "test", "default", ["default"])
    id2 = resolver.resolve("MÜLLER", "person", "test", "default", ["default"])
    assert id1 == id2
    # Tier 0 matched, so no alias may have been persisted
    rows = resolver.db.fetchall("SELECT alias FROM aliases WHERE tenant_id = 'test'", ())
    assert rows == []


# ── Fuzzy / embedding alias persistence gates ────────────────────────────────


def test_fuzzy_high_score_persists_alias(resolver):
    """WRatio('Tensorflow2', 'Tensorflow') is ~95 — above the persist gate."""
    id1 = resolver.resolve("Tensorflow", "technology", "test", "default", ["default"])
    id2 = resolver.resolve("Tensorflow2", "technology", "test", "default", ["default"])
    assert id1 == id2
    row = resolver.db.fetchone(
        "SELECT atom_id FROM aliases WHERE alias = 'Tensorflow2' AND tenant_id = 'test'", ()
    )
    assert row is not None


def test_fuzzy_threshold_score_matches_without_persisting_alias(resolver):
    """WRatio('PostgreSQL Database', 'PostgreSQL') is 90 — matches at the
    fuzzy threshold (85) but must not be persisted as an alias (< 92)."""
    id1 = resolver.resolve("PostgreSQL", "technology", "test", "default", ["default"])
    id2 = resolver.resolve("PostgreSQL Database", "technology", "test", "default", ["default"])
    assert id1 == id2
    row = resolver.db.fetchone(
        "SELECT atom_id FROM aliases WHERE alias = 'PostgreSQL Database' AND tenant_id = 'test'", ()
    )
    assert row is None


def test_embedding_tier_never_persists_alias(resolver):
    id1 = resolver.resolve("machine learning", "topic", "test", "default", ["default"])
    real_vec = resolver.embed_engine.embed("machine learning")

    with patch.object(resolver.embed_engine, "embed", return_value=real_vec):
        resolver.fuzzy_threshold = 101.0  # force past the fuzzy tier
        resolver.cosine_threshold = 1.0
        id2 = resolver.resolve("statistical modelling", "topic", "test", "default", ["default"])

    assert id2 == id1
    row = resolver.db.fetchone(
        "SELECT atom_id FROM aliases WHERE alias = 'statistical modelling' AND tenant_id = 'test'", ()
    )
    assert row is None


def test_embedding_tier_is_space_scoped(resolver):
    """A vector match in another space must not leak across spaces."""
    id_other = resolver.resolve("Alice", "person", "test", "other", ["other"])
    real_vec = resolver.embed_engine.embed("Alice")

    with patch.object(resolver.embed_engine, "embed", return_value=real_vec):
        resolver.fuzzy_threshold = 101.0
        resolver.cosine_threshold = 1.0
        id_default = resolver.resolve("Alice", "person", "test", "default", ["default"])

    assert id_default != id_other


# ── CJK verb-phrase gate ─────────────────────────────────────────────────────


def _span(text, label, score=1.0):
    """One entity as the ONNX runtime returns it."""
    return SimpleNamespace(text=text, label=label, score=score, start=0, end=len(text))


def test_cjk_constraint_span_reaches_classifier():
    from smrti.extraction.ner import _is_verb_phrase

    model = MagicMock()
    model.extract_entities.return_value = [_span("绝对不要删除数据库", "verb_phrase", 0.8)]
    ent = {"name": "绝对不要删除数据库", "type": "constraint"}
    assert _is_verb_phrase(ent, model) is True
    model.extract_entities.assert_called_once()


def test_short_cjk_span_skips_classifier():
    from smrti.extraction.ner import _is_verb_phrase

    model = MagicMock()
    ent = {"name": "冥想", "type": "preference"}
    assert _is_verb_phrase(ent, model) is False
    model.extract_entities.assert_not_called()


def test_two_word_english_span_skips_classifier():
    from smrti.extraction.ner import _is_verb_phrase

    model = MagicMock()
    ent = {"name": "dark mode", "type": "preference"}
    assert _is_verb_phrase(ent, model) is False
    model.extract_entities.assert_not_called()


def test_ner_extract_filters_cjk_verb_phrase():
    from smrti.extraction.ner import NERProvider

    provider = NERProvider()
    mock_model = MagicMock()

    def entities(text, labels, threshold=0.4):
        if labels == ["noun_phrase", "verb_phrase"]:
            return [_span(text, "verb_phrase", 0.8)]
        return [_span("绝对不要删除数据库", "constraint")]

    mock_model.extract_entities.side_effect = entities
    provider._model = mock_model

    results = provider.extract("绝对不要删除数据库")
    assert results == []


def test_ner_extract_surfaces_real_confidence():
    from smrti.extraction.ner import NERProvider

    provider = NERProvider()
    mock_model = MagicMock()
    mock_model.extract_entities.return_value = [_span("Alice", "person", 0.87)]
    provider._model = mock_model

    results = provider.extract("Alice is here")
    assert results == [{"name": "Alice", "type": "person", "score": 0.87}]
    _, kwargs = mock_model.extract_entities.call_args
    assert kwargs == {"threshold": 0.4}


# ── Speaker attribution ──────────────────────────────────────────────────────


def _run_hybrid_with_persons(mem, mock_claims):
    mock_ner = MagicMock()
    mock_ner.extract.return_value = [{"name": "Python", "type": "technology", "score": 0.9}]
    mock_ner.classify_pronoun.return_value = False

    with patch("smrti.extraction.ner.get_ner", return_value=mock_ner), \
         patch("smrti.extraction.extract.extract_claims_only", mock_claims):
        ep = mem.remember("I use Python", type="episode")
        asyncio.run(extract_and_link_hybrid(
            ep, "I use Python", mem, "", "m", "http://localhost", mode="hybrid",
        ))


def test_speaker_attributed_when_exactly_one_person(mem):
    _insert_person(mem, "p-nico", "Nico")

    mock_claims = AsyncMock(return_value={"claims": []})
    _run_hybrid_with_persons(mem, mock_claims)

    mock_claims.assert_called_once()
    entities_arg = mock_claims.call_args.args[1]
    assert any(e["name"] == "Nico" for e in entities_arg)


def test_speaker_attribution_skipped_with_multiple_persons(mem):
    _insert_person(mem, "p-nico", "Nico")
    _insert_person(mem, "p-dave", "Dave")

    mock_claims = AsyncMock(return_value={"claims": []})
    _run_hybrid_with_persons(mem, mock_claims)

    # No speaker injected → only 1 unique entity → claims call skipped
    mock_claims.assert_not_called()


# ── Sentiment initialization and multilingual anchors ────────────────────────


@pytest.fixture
def _reset_sentiment_cache():
    from smrti.extraction import sentiment as smod

    saved = (smod._neg_vecs, smod._pos_vecs)
    smod._neg_vecs = None
    smod._pos_vecs = None
    yield
    smod._neg_vecs, smod._pos_vecs = saved


def test_sentiment_init_survives_failed_first_attempt(_reset_sentiment_cache):
    from smrti.extraction import sentiment as smod
    from smrti.extraction.sentiment import estimate_valence

    class FlakyEmbed:
        def __init__(self):
            self.batch_calls = 0

        def embed_batch(self, texts):
            self.batch_calls += 1
            if self.batch_calls == 2:
                raise RuntimeError("embed fail")  # positive-anchor batch fails
            if self.batch_calls % 2 == 1:
                return [[1.0, 0.0] for _ in texts]
            return [[0.0, 1.0] for _ in texts]

        def embed(self, text):
            return [0.0, 1.0]

    embed = FlakyEmbed()
    with pytest.raises(RuntimeError):
        estimate_valence("great", embed)
    # All-or-nothing: neither side may be cached after the partial failure
    assert smod._neg_vecs is None
    assert smod._pos_vecs is None

    val = estimate_valence("great", embed)
    assert val > 0


@pytest.fixture(scope="module")
def embed_provider():
    return EmbeddingProvider()


def test_german_negative_sentence_gets_negative_valence(embed_provider):
    from smrti.extraction.sentiment import estimate_valence

    val = estimate_valence(
        "Das ist eine Katastrophe, ich hasse dieses schreckliche Ergebnis.",
        embed_provider,
    )
    assert val < 0


def test_chinese_negative_sentence_gets_negative_valence(embed_provider):
    from smrti.extraction.sentiment import estimate_valence

    val = estimate_valence("这是一个可怕的错误，绝对不能再发生。", embed_provider)
    assert val < 0
