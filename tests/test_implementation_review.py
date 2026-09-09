"""The defects the 2026-09-09 implementation review confirmed, pinned down.

Each test replays one of the review's reproductions through the facade on
the deterministic bag-of-words embedder, so it measures bookkeeping and not
the embedding model.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import smrti.core.embed as embed_module
from smrti import Smrti
from smrti.core.db import stable_rowid
from smrti.core.models import Atom, AtomType, RecallResult, TruthValue
from smrti.core.provenance import ATOM_METADATA_JSON
from smrti.extraction.extract import _build_entity_context, _link_claims
from smrti.extraction.resolve import EntityResolver
from smrti.retrieval.classify import classify_memory
from smrti.servers import proxy
from smrti.servers.proxy import _enrich_content, _inject_context, app
from tests.test_engine_claims import _BagOfWords


@pytest.fixture(autouse=True)
def bag_of_words_embedder(monkeypatch):
    monkeypatch.setattr(embed_module.EmbeddingProvider, "_get_model", lambda self: _BagOfWords())


@pytest.fixture
def mem(tmp_path):
    return Smrti(db_path=str(tmp_path / "review.db"), tenant_id="t", write_space="s")


def _row(mem, atom_id):
    return mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (atom_id,))


def _meta(mem, atom_id) -> dict:
    return json.loads(_row(mem, atom_id)["metadata"] or "{}")


def _stamp_forgotten(mem, atom_id) -> None:
    mem.db.execute(
        f"UPDATE atoms SET metadata = json_set({ATOM_METADATA_JSON}, '$.forgotten', json('true')) "
        "WHERE id = ?",
        (atom_id,),
    )


def _entity(mem, name, etype):
    return EntityResolver(mem.db, mem.embed).resolve(name, etype, "t", "s", ["s"])


def _claim(mem, subject, predicate, obj, supersedes=None, entity_ids=None, episode=""):
    claim = {"subject": subject, "predicate": predicate, "object": obj}
    if supersedes:
        claim["supersedes"] = supersedes
    _link_claims([claim], dict(entity_ids or {}), mem, episode_id=episode)


def _live_claims(mem, subject_id) -> set[tuple[str, str]]:
    rows = mem.db.fetchall(
        """SELECT r.relation, t.label FROM atoms r JOIN atoms t ON t.id = r.target_id
           WHERE r.type = 'relation' AND r.source_id = ?
             AND json_extract(r.metadata, '$.superseded_by') IS NULL""",
        (subject_id,),
    )
    return {(r["relation"], r["label"]) for r in rows}


# ── 1. a stated warning survives the echo filter ─────────────────────────────


def test_a_stated_warning_is_recalled_by_the_question_it_forbids(mem):
    warning = mem.remember(
        "Never deploy production without a verified database backup.",
        valence=-0.9,
    )
    hits = {r.atom.id: r for r in mem.recall("Deploy production without a verified database backup")}
    assert warning in hits
    assert classify_memory(hits[warning]) == "critical_warning"


def test_a_plain_restatement_of_the_question_is_still_an_echo(mem):
    echo = mem.remember("Deploy production without a verified database backup")
    assert echo not in {r.atom.id for r in mem.recall("Deploy production without a verified database backup")}


# ── 2. returning to an earlier fact ──────────────────────────────────────────


def test_returning_to_an_earlier_city_makes_it_current_again(mem):
    alice = _entity(mem, "Alice", "person")
    ids = {"Alice": alice}
    _claim(mem, "Alice", "lives_in", "Berlin", entity_ids=ids)
    _claim(mem, "Alice", "lives_in", "Paris", supersedes="Berlin", entity_ids=ids)
    _claim(mem, "Alice", "lives_in", "Berlin", supersedes="Paris", entity_ids=ids)

    assert _live_claims(mem, alice) == {("lives_in", "Berlin")}
    context = _build_entity_context(mem)
    assert "lives_in Berlin" in context
    assert "lives_in Paris" not in context

    alice_atom = mem.atomspace.get_atom(alice, "t", "s")
    enriched = _enrich_content(RecallResult(atom=alice_atom, salience=1.0, similarity=1.0), mem)
    assert "Berlin" in enriched
    assert "Paris" not in enriched


def test_a_revived_claim_is_not_cut_again_at_the_next_epoch(mem):
    alice = _entity(mem, "Alice", "person")
    ids = {"Alice": alice}
    _claim(mem, "Alice", "lives_in", "Berlin", entity_ids=ids)
    _claim(mem, "Alice", "lives_in", "Paris", supersedes="Berlin", entity_ids=ids)
    mem.reflect()
    _claim(mem, "Alice", "lives_in", "Berlin", supersedes="Paris", entity_ids=ids)
    mem.reflect()

    berlin = mem.db.fetchone(
        "SELECT r.probability FROM atoms r JOIN atoms t ON t.id = r.target_id "
        "WHERE r.source_id = ? AND t.label = 'Berlin'",
        (alice,),
    )
    paris = mem.db.fetchone(
        "SELECT r.probability FROM atoms r JOIN atoms t ON t.id = r.target_id "
        "WHERE r.source_id = ? AND t.label = 'Paris'",
        (alice,),
    )
    assert berlin["probability"] > 0.3
    assert paris["probability"] <= 0.1


def test_going_back_to_a_preference_clears_its_antipattern_status(mem):
    alice = _entity(mem, "Alice", "person")
    dark = _entity(mem, "dark mode", "preference")
    light = _entity(mem, "light mode", "preference")
    ids = {"Alice": alice, "dark mode": dark, "light mode": light}
    _claim(mem, "Alice", "prefers", "dark mode", entity_ids=ids)
    _claim(mem, "Alice", "prefers", "light mode", supersedes="dark mode", entity_ids=ids)
    mem.reflect()
    assert _row(mem, dark)["probability"] <= 0.1

    _claim(mem, "Alice", "prefers", "dark mode", supersedes="light mode", entity_ids=ids)
    mem.reflect()

    assert _row(mem, dark)["probability"] > 0.3
    assert _row(mem, light)["probability"] <= 0.1


# ── 3. a shared object belief belongs to every subject ───────────────────────


def test_correcting_one_persons_preference_leaves_the_others_intact(mem):
    alice = _entity(mem, "Alice", "person")
    bob = _entity(mem, "Bob", "person")
    tea = _entity(mem, "tea", "preference")
    ids = {"Alice": alice, "Bob": bob, "tea": tea}
    _claim(mem, "Alice", "prefers", "tea", entity_ids=ids)
    _claim(mem, "Bob", "prefers", "tea", entity_ids=ids)
    before = _row(mem, tea)["probability"]

    _claim(mem, "Alice", "prefers", "coffee", supersedes="tea", entity_ids=ids)
    mem.reflect()

    assert _row(mem, tea)["probability"] == pytest.approx(before)
    assert _live_claims(mem, bob) == {("prefers", "tea")}
    assert _live_claims(mem, alice) == {("prefers", "coffee")}


def test_a_preference_nobody_else_holds_is_still_superseded(mem):
    alice = _entity(mem, "Alice", "person")
    tea = _entity(mem, "tea", "preference")
    ids = {"Alice": alice, "tea": tea}
    _claim(mem, "Alice", "prefers", "tea", entity_ids=ids)
    _claim(mem, "Alice", "prefers", "coffee", supersedes="tea", entity_ids=ids)
    mem.reflect()
    assert _row(mem, tea)["probability"] <= 0.1


# ── 4. forgotten atoms stay out of every rendering ───────────────────────────


def _alice_works_on_secret(mem):
    alice = _entity(mem, "Alice", "person")
    project = _entity(mem, "SensitiveProject", "project")
    mem.atomspace.link_atoms(alice, project, "works_on", "t", "s")
    return alice, project


def test_a_forgotten_atom_is_left_out_of_proxy_enrichment(mem):
    alice, project = _alice_works_on_secret(mem)
    _stamp_forgotten(mem, project)
    alice_atom = mem.atomspace.get_atom(alice, "t", "s")
    enriched = _enrich_content(RecallResult(atom=alice_atom, salience=1.0, similarity=1.0), mem)
    assert "SensitiveProject" not in enriched


def test_a_forgotten_atom_is_left_out_of_the_extraction_context(mem):
    _, project = _alice_works_on_secret(mem)
    assert "SensitiveProject" in _build_entity_context(mem)
    _stamp_forgotten(mem, project)
    assert "SensitiveProject" not in _build_entity_context(mem)


def test_forget_drops_the_atom_from_both_indexes(mem):
    atom_id = mem.remember("the launch codes are in the red binder")
    assert mem.forget("launch codes red binder")
    assert mem.db.fetchone("SELECT 1 FROM vec_atoms WHERE rowid = ?", (stable_rowid(atom_id),)) is None
    assert mem.db.fetchone("SELECT 1 FROM atoms_fts WHERE rowid = ?", (stable_rowid(atom_id),)) is None
    # Reopening must not rebuild the lexical index with the forgotten row in it.
    mem.db._init_atoms_fts()
    assert mem.db.fetchone("SELECT 1 FROM atoms_fts WHERE rowid = ?", (stable_rowid(atom_id),)) is None


# ── 5. deleting a graph that holds supersessions ─────────────────────────────


def _supersession_graph(mem):
    alice = _entity(mem, "Alice", "person")
    dark = _entity(mem, "dark mode", "preference")
    ids = {"Alice": alice, "dark mode": dark}
    _claim(mem, "Alice", "prefers", "dark mode", entity_ids=ids)
    _claim(mem, "Alice", "prefers", "light mode", supersedes="dark mode", entity_ids=ids)


def test_clear_space_survives_a_supersession(mem):
    _supersession_graph(mem)
    assert mem.clear_space() > 0
    assert mem.status()["total_atoms"] == 0
    assert mem.db.fetchone("SELECT COUNT(*) AS n FROM evidence")["n"] == 0


def test_the_pruner_survives_a_supersession(mem):
    _supersession_graph(mem)
    mem.db.execute("UPDATE atoms SET confidence = 0.0, lti = 0.0, sti = 0.0")
    result = mem.reflect()
    assert result.atoms_pruned > 0
    assert mem.db.fetchone("SELECT COUNT(*) AS n FROM atoms WHERE type = 'relation'")["n"] == 0


# ── 6. the proxy keeps the agent's own memories and survives usage chunks ────


def _agent_result(text):
    atom = Atom(
        type=AtomType.EPISODE, label=text, content=text,
        truth=TruthValue(probability=0.75, confidence=0.6),
        metadata={"source": "agent"},
    )
    return RecallResult(atom=atom, salience=0.5, similarity=0.7)


def test_an_assistant_only_recall_still_reaches_the_prompt():
    body = {"messages": [{"role": "user", "content": "What did you recommend?"}]}
    memories = [_agent_result("I recommended sqlite-vec for the vector index")]
    with patch("smrti.servers.proxy._recall", AsyncMock(return_value=memories)):
        with patch("smrti.servers.proxy.get_mem", return_value=MagicMock()):
            result, injection, dicts = asyncio.run(_inject_context(body, "t", "s", ["s"]))
    assert "sqlite-vec" in injection
    assert result["messages"][0]["role"] == "system"
    assert dicts[0]["source"] == "agent"


def _mock_stream_client(sse_lines):
    async def aiter_lines():
        for line in sse_lines:
            yield line

    response = MagicMock()
    response.aiter_lines = aiter_lines
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=None)
    client = MagicMock()
    client.stream = MagicMock(return_value=ctx)
    return client


def _mock_mem():
    m = MagicMock()
    m.tenant_id = "default"
    m.write_space = "default"
    m.recall.return_value = []
    return m


def test_a_usage_only_stream_chunk_is_passed_through(monkeypatch):
    sse = [
        'data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
        'data: {"id":"1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}',
        "data: [DONE]",
    ]
    store = AsyncMock()

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch("smrti.servers.proxy._bootstrap", return_value=("default", "default", "/tmp/t.db")):
                with patch("smrti.servers.proxy.get_mem", return_value=_mock_mem()):
                    with patch("smrti.servers.proxy.get_http", return_value=_mock_stream_client(sse)):
                        with patch("smrti.servers.proxy._store_exchange", store):
                            resp = await client.post(
                                "/v1/chat/completions",
                                json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                            )
                            await asyncio.gather(*proxy._background_tasks)
                            return resp

    resp = asyncio.run(run_test())
    assert "proxy_error" not in resp.text
    assert '"total_tokens":6' in resp.text.replace(" ", "")
    assert store.await_count == 1
    assert store.await_args.args[1] == "Hello"


def test_a_choiceless_non_stream_response_is_forwarded_not_crashed():
    upstream = MagicMock()
    upstream.json.return_value = {"id": "x", "choices": [], "usage": {"total_tokens": 0}}
    upstream.status_code = 200
    upstream.headers = {}
    client = AsyncMock()
    client.post = AsyncMock(return_value=upstream)

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            with patch("smrti.servers.proxy._bootstrap", return_value=("default", "default", "/tmp/t.db")):
                with patch("smrti.servers.proxy.get_mem", return_value=_mock_mem()):
                    with patch("smrti.servers.proxy.get_http", return_value=client):
                        with patch("smrti.servers.proxy._store_exchange", AsyncMock()):
                            return await c.post(
                                "/v1/chat/completions",
                                json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                            )

    resp = asyncio.run(run_test())
    assert resp.status_code == 200
    assert resp.json()["choices"] == []


# ── 7. the candidate pool is not spent on what cannot surface ────────────────


def test_forgotten_neighbours_do_not_crowd_out_the_live_memory(mem):
    for i in range(60):
        _stamp_forgotten(mem, mem.remember(f"alpha beta gamma n{i}"))
    live = mem.remember("alpha beta gamma delta epsilon zeta")
    assert live in {r.atom.id for r in mem.recall("alpha beta gamma")}


def test_sunk_neighbours_do_not_crowd_out_the_live_memory(mem):
    for i in range(60):
        sunk = mem.remember(f"alpha beta gamma n{i}")
        mem.db.execute("UPDATE atoms SET confidence = 0.01 WHERE id = ?", (sunk,))
    live = mem.remember("alpha beta gamma delta epsilon zeta")
    assert live in {r.atom.id for r in mem.recall("alpha beta gamma")}


def test_an_atom_stored_without_a_vector_is_still_recalled_and_then_repaired(mem):
    belief = Atom(
        type=AtomType.BELIEF, label="the build runs on jenkins",
        content="the build runs on jenkins",
        truth=TruthValue(probability=0.8, confidence=0.5), tenant_id="t", space="s",
    )
    mem.atomspace.add_atom(belief, require_vector=False)
    assert mem.db.fetchone("SELECT 1 FROM vec_atoms WHERE rowid = ?", (stable_rowid(belief.id),)) is None

    assert belief.id in {r.atom.id for r in mem.recall("jenkins build", min_confidence=0.0)}

    mem.reflect()
    assert mem.db.fetchone("SELECT 1 FROM vec_atoms WHERE rowid = ?", (stable_rowid(belief.id),)) is not None
    assert "vector_missing" not in _meta(mem, belief.id)


# ── 8. shutdown waits for the exchanges still being stored ───────────────────


def test_the_proxy_lifespan_drains_pending_stores():
    stored = []

    async def slow_store():
        await asyncio.sleep(0.05)
        stored.append(True)

    async def run_test():
        with patch("smrti.servers.proxy._bootstrap", return_value=("default", "default", "/tmp/t.db")):
            with patch("smrti.servers.proxy.run_reflect_loop", new=lambda *_: asyncio.sleep(3600)):
                async with proxy.lifespan(app):
                    proxy._spawn(slow_store())
        return stored

    assert asyncio.run(run_test()) == [True]


# ── performance: what the review measured from the query plans ───────────────


def test_entity_lookups_use_the_normalized_label_indexes(mem):
    plan = " ".join(
        r["detail"]
        for r in mem.db.fetchall(
            "EXPLAIN QUERY PLAN SELECT id FROM atoms WHERE u_lower(label) = u_lower(?) "
            "AND entity_type = ? AND tenant_id = ? AND space IN (?)",
            ("Alice", "person", "t", "s"),
        )
    )
    assert "idx_atoms_label_norm" in plan
    plan = " ".join(
        r["detail"]
        for r in mem.db.fetchall(
            "EXPLAIN QUERY PLAN SELECT atom_id FROM aliases WHERE u_lower(alias) = u_lower(?) "
            "AND tenant_id = ? AND space IN (?)",
            ("Ali", "t", "s"),
        )
    )
    assert "idx_aliases_alias_norm" in plan


def test_space_overlap_reports_how_much_of_each_space_it_considered(mem, monkeypatch):
    import smrti.spaces.set_ops as set_ops

    other = Smrti(db_path=mem.db._db_path, tenant_id="t", write_space="s2")
    for text in ("alpha beta", "gamma delta", "epsilon zeta"):
        mem.remember(text)
        other.remember(text)

    whole = mem.space_overlap("s2")
    assert (whole.sampled_a, whole.size_a, whole.sampled_b, whole.size_b) == (3, 3, 3, 3)
    assert whole.complete
    assert whole.jaccard == pytest.approx(1.0)

    monkeypatch.setattr(set_ops, "SAMPLE_LIMIT", 2)
    head = mem.space_overlap("s2")
    assert (head.sampled_a, head.size_a) == (2, 3)
    assert not head.complete
