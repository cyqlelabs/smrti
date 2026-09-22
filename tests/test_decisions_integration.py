"""The decision tasks at their integration points: recall, extraction,
supersession, entity resolution, and the server surface.

Runs on the deterministic bag-of-words embedder so nothing here depends
on what two sentences mean, only on which words they share; and on a
static provider whose answers the test chooses, so what is measured is
what the engine does with an answer, not what the model would have said.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from rapidfuzz import fuzz

import smrti.core.embed as embed_module
from smrti import Smrti
from smrti.core.provenance import SOURCE_AGENT, VALENCE_STATED
from smrti.decisions import Choice, DecisionEngine, DecisionPolicy, MODE_OFF, Noul, StaticProvider, TASKS, audit
from smrti.decisions.extraction import COMPATIBLE, EXPLICIT_UPDATE, TONE_DECISION, TONE_SPEAKER, TONE_THING
from smrti.extraction.extract import SUPERSESSION_DEFERRED, _link_claims, extract_and_link_hybrid
from smrti.extraction.resolve import EntityResolver
from smrti.servers.mcp import handle_tool

_WORD = re.compile(r"\w+")
_token_vectors: dict[str, np.ndarray] = {}


def _token_vector(token: str) -> np.ndarray:
    vec = _token_vectors.get(token)
    if vec is None:
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        vec = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        _token_vectors[token] = vec
    return vec


class _BagOfWords:
    def embed(self, texts):
        for text in texts:
            tokens = _WORD.findall(text.casefold())
            acc = 0.6 * (len(tokens) ** 0.5) * _token_vector("<common>")
            for token in tokens:
                acc += _token_vector(token)
            norm = np.linalg.norm(acc)
            yield acc / norm if norm else acc + 1e-3


@pytest.fixture(autouse=True)
def bag_of_words_embedder(monkeypatch):
    monkeypatch.setattr(embed_module.EmbeddingProvider, "_get_model", lambda self: _BagOfWords())


@pytest.fixture(autouse=True)
def clean_audit():
    audit.reset()
    yield
    audit.reset()


def run(coro):
    return asyncio.run(coro)


def _engine(provider, **modes) -> DecisionEngine:
    policy = DecisionPolicy(modes={task: MODE_OFF for task in TASKS}).with_modes(**modes)
    return DecisionEngine(policy, provider)


def _mem(tmp_path, engine: DecisionEngine, name="d") -> Smrti:
    return Smrti(db_path=str(tmp_path / f"{name}.db"), tenant_id="t", write_space="s", decisions=engine)


def _row(mem, atom_id):
    return mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (atom_id,))


# ── recall: evidence reranking and the attention boundary ────────────────────


def _evidence_provider(direct_for: str):
    """Answers "direct evidence" only for the candidate whose text holds *direct_for*."""

    def answer(key, question, state):
        ref, kind = key.rsplit("_", 1)
        candidate = next(c for c in state["candidates"] if c["ref"] == ref)
        hit = direct_for in candidate["text"]
        return {"type": "noul", "noul": 0.95 if (kind == "direct" and hit) else 0.02}

    return StaticProvider(answer)


def _three_memories(mem):
    mem.remember("the deploy pipeline builds the docker image on jenkins")
    answer_id = mem.remember("the deploy pipeline is owned by the platform team in oslo")
    mem.remember("the deploy pipeline runs the integration tests on jenkins")
    return answer_id


def test_an_active_rerank_puts_the_judged_evidence_first(tmp_path):
    provider = _evidence_provider("oslo")
    mem = _mem(tmp_path, _engine(provider, rerank="active"))
    answer_id = _three_memories(mem)
    local = mem.recall("who owns the deploy pipeline on jenkins", rerank=False)
    assert [r.atom.id for r in local][0] != answer_id  # the stand-in ranks the jenkins echoes first

    results = mem.recall("who owns the deploy pipeline on jenkins")
    assert results[0].atom.id == answer_id
    assert results[0].evidence == pytest.approx(0.95, abs=0.01)
    assert all(r.evidence is not None for r in results)
    record = audit.get_all()[0]
    assert record["task"] == "rerank" and record["applied"] and record["outcome"] == "reordered"
    assert record["summary"]["sufficiency"] == pytest.approx(0.95, abs=0.01)


def test_a_shadow_rerank_annotates_but_keeps_the_local_order(tmp_path):
    provider = _evidence_provider("oslo")
    mem = _mem(tmp_path, _engine(provider, rerank="shadow"))
    _three_memories(mem)
    local = [r.atom.id for r in mem.recall("who owns the deploy pipeline on jenkins", rerank=False)]
    shadowed = mem.recall("who owns the deploy pipeline on jenkins")
    assert [r.atom.id for r in shadowed] == local
    assert all(r.evidence is not None for r in shadowed)
    assert provider.calls
    assert not audit.get_all()[0]["applied"]


def test_only_the_returned_results_are_boosted(tmp_path):
    provider = _evidence_provider("oslo")
    mem = _mem(tmp_path, _engine(provider, rerank="active"))
    answer_id = _three_memories(mem)
    ids = [r["id"] for r in mem.db.fetchall("SELECT id FROM atoms WHERE type = 'episode'")]

    results = mem.recall("who owns the deploy pipeline on jenkins", top_k=1)
    assert [r.atom.id for r in results] == [answer_id]
    stis = {r["id"]: r["sti"] for r in mem.db.fetchall("SELECT id, sti FROM atoms WHERE type = 'episode'")}
    assert stis[answer_id] > 0.0
    assert all(stis[i] == 0.0 for i in ids if i != answer_id)


def test_a_failed_judgement_keeps_the_local_ranking(tmp_path):
    def broken(_k, _q, _s):
        raise RuntimeError("down")

    mem = _mem(tmp_path, _engine(StaticProvider(broken), rerank="active"))
    _three_memories(mem)
    local = [r.atom.id for r in mem.recall("who owns the deploy pipeline on jenkins", rerank=False)]
    assert [r.atom.id for r in mem.recall("who owns the deploy pipeline on jenkins")] == local
    assert audit.get_all()[0]["outcome"] == "unavailable"


def test_the_evidence_cutoff_never_drops_a_stated_warning(tmp_path):
    provider = _evidence_provider("oslo")
    policy = DecisionPolicy(modes={task: MODE_OFF for task in TASKS}, rerank_min_evidence=0.5).with_modes(rerank="active")
    mem = _mem(tmp_path, DecisionEngine(policy, provider))
    answer_id = _three_memories(mem)
    warning_id = mem.remember(
        "never run the deploy pipeline on jenkins without a backup", valence=-0.9, intensity=0.9
    )
    results = mem.recall("who owns the deploy pipeline on jenkins")
    ids = [r.atom.id for r in results]
    assert answer_id in ids and warning_id in ids
    assert len(ids) == 2  # the two jenkins echoes fell under the line
    assert audit.get_all()[0]["summary"]["dropped"] == 2


def test_forgetting_does_not_ask_the_judge(tmp_path):
    provider = _evidence_provider("oslo")
    mem = _mem(tmp_path, _engine(provider, rerank="active"))
    _three_memories(mem)
    mem.forget("deploy pipeline jenkins")
    assert provider.calls == []


def test_attend_boosts_only_what_the_caller_kept(tmp_path):
    mem = _mem(tmp_path, _engine(None))
    kept = mem.remember("the deploy pipeline builds the docker image")
    other = mem.remember("the deploy pipeline runs the tests")
    foreign = Smrti(db_path=str(tmp_path / "d.db"), tenant_id="t", write_space="elsewhere",
                    decisions=mem.decisions).remember("the deploy pipeline is elsewhere")
    wide = mem.recall("deploy pipeline", top_k=10, boost=False)
    assert {r.atom.id for r in wide} >= {kept, other}
    assert all(_row(mem, i)["sti"] == 0.0 for i in (kept, other))

    assert mem.attend([kept, foreign]) == 2  # two ids written; only the write space one moves
    assert _row(mem, kept)["sti"] > 0.0
    assert _row(mem, other)["sti"] == 0.0
    assert _row(mem, foreign)["sti"] == 0.0
    assert handle_tool(mem, "smrti_attend", {"atom_ids": [other]}) == {"status": "ok", "space": "s", "boosted": 1}
    assert _row(mem, other)["sti"] > 0.0


def test_the_recall_tool_carries_the_evidence_field(tmp_path):
    mem = _mem(tmp_path, _engine(_evidence_provider("oslo"), rerank="active"))
    _three_memories(mem)
    memories = handle_tool(mem, "smrti_recall", {"query": "who owns the deploy pipeline"})["memories"]
    assert memories and all("evidence" in m for m in memories)
    off = _mem(tmp_path, _engine(None), name="off")
    off.remember("the deploy pipeline is owned by oslo")
    assert handle_tool(off, "smrti_recall", {"query": "deploy pipeline"})["memories"][0]["evidence"] is None


# ── extraction routing ───────────────────────────────────────────────────────


def _route_provider(durable: float, corrects: float, novel: float) -> StaticProvider:
    values = {"durable": durable, "corrects": corrects, "novel": novel}
    return StaticProvider(lambda key, _q, _s: {"type": "noul", "noul": values[key]})


def _hybrid(mem, content, ner_entities, source="user", mode="hybrid"):
    """Run the hybrid pipeline with NER and both LLM calls stubbed; returns the stubs."""
    ner_instance = MagicMock()
    ner_instance.extract.return_value = ner_entities
    ner_instance.classify_pronoun.return_value = False
    episode_id = mem.remember(content, metadata={"source": source} if source == "agent" else None)
    claims_only = AsyncMock(return_value={"claims": []})
    full = AsyncMock(return_value=None)
    with patch("smrti.extraction.ner.get_ner", return_value=ner_instance):
        with patch("smrti.extraction.extract.extract_claims_only", new=claims_only):
            with patch("smrti.extraction.extract.extract_and_link", new=full):
                run(extract_and_link_hybrid(
                    episode_id, content, mem, "", "qwen", "http://llm.local", source, mode=mode,
                ))
    return episode_id, claims_only, full, ner_instance


def _mentions(mem, episode_id):
    return mem.db.fetchall(
        "SELECT target_id FROM atoms WHERE type = 'relation' AND relation = 'mentions' AND source_id = ?",
        (episode_id,),
    )


def test_chatter_is_routed_past_the_claims_call_but_still_linked(tmp_path):
    mem = _mem(tmp_path, _engine(_route_provider(0.05, 0.02, 0.05), routing="active"))
    episode_id, claims_only, full, ner = _hybrid(
        mem, "Thanks Nico, that worked at Cyqle",
        [{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}],
    )
    claims_only.assert_not_awaited()
    full.assert_not_awaited()
    assert len(_mentions(mem, episode_id)) == 2  # the episode is kept and linked
    assert _row(mem, episode_id) is not None
    record = next(r for r in audit.get_all() if r["task"] == "routing")
    assert record["outcome"] == "skip" and record["applied"]


def test_a_durable_fact_with_one_entity_is_routed_to_the_llm(tmp_path):
    mem = _mem(tmp_path, _engine(_route_provider(0.95, 0.1, 0.8), routing="active"))
    _, claims_only, full, _ = _hybrid(mem, "Never deploy to production on a Friday", [{"name": "production", "type": "technology"}])
    claims_only.assert_not_awaited()  # fewer than two entities: the claims-only prompt is not used
    full.assert_awaited_once()          # the full extraction is


def test_shadow_routing_changes_nothing_and_records_the_route(tmp_path):
    mem = _mem(tmp_path, _engine(_route_provider(0.05, 0.02, 0.05), routing="shadow"))
    _, claims_only, full, _ = _hybrid(
        mem, "Thanks Nico, that worked at Cyqle",
        [{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}],
    )
    claims_only.assert_awaited_once()
    record = next(r for r in audit.get_all() if r["task"] == "routing")
    assert record["outcome"] == "skip" and not record["applied"] and record["mode"] == "shadow"


def test_an_unavailable_gate_takes_the_existing_path(tmp_path):
    def broken(_k, _q, _s):
        raise RuntimeError("down")

    mem = _mem(tmp_path, _engine(StaticProvider(broken), routing="active"))
    _, claims_only, _, _ = _hybrid(
        mem, "Nico works at Cyqle",
        [{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}],
    )
    claims_only.assert_awaited_once()


def test_an_assistant_echo_is_routed_past_the_full_extraction(tmp_path):
    mem = _mem(tmp_path, _engine(_route_provider(0.1, 0.0, 0.05), routing="active"))
    episode_id, claims_only, full, _ = _hybrid(
        mem, "As you said, you prefer tea.", [{"name": "tea", "type": "preference"}], source="agent",
    )
    full.assert_not_awaited()
    assert _mentions(mem, episode_id)  # linked locally, with agent provenance on the episode
    assert json.loads(_row(mem, episode_id)["metadata"])["source"] == "agent"


def test_local_mode_never_asks_the_gate(tmp_path):
    provider = _route_provider(0.0, 0.0, 0.0)
    mem = _mem(tmp_path, _engine(provider, routing="active"))
    _hybrid(mem, "Nico works at Cyqle", [{"name": "Nico", "type": "person"}], mode="local")
    assert provider.calls == []


def test_the_gate_sees_the_known_context(tmp_path):
    provider = _route_provider(0.5, 0.5, 0.5)
    mem = _mem(tmp_path, _engine(provider, routing="active"))
    EntityResolver(mem.db, mem.embed).resolve("Nico", "person", "t", "s", ["s"])
    _hybrid(mem, "Nico works at Cyqle", [{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}])
    state = provider.calls[0]["state"]
    assert "Nico (person)" in state["known_context"]
    assert state["author"] == "user"


# ── supersession ─────────────────────────────────────────────────────────────


def _choice_provider(label, confidence=0.9, on_call=None):
    def answer(_k, question, state):
        assert isinstance(question, Choice)
        if on_call is not None:
            on_call(state)
        return {"type": "choice", "choice": label, "probabilities": {label: confidence}, "confidence": confidence}

    return StaticProvider(answer)


def _move(mem, source="user"):
    """Nico lived in Córdoba, then says he moved to Rosario; returns (old_edge, new_edge)."""
    resolver = EntityResolver(mem.db, mem.embed)
    ids = {}
    for name, etype in (("Nico", "person"), ("Córdoba", "location"), ("Rosario", "location")):
        ids[name] = resolver.resolve(name, etype, "t", "s", ["s"])
    first = mem.remember("Nico lives in Córdoba")
    _link_claims([{"subject": "Nico", "predicate": "lives_in", "object": "Córdoba"}], ids, mem, first)
    second = mem.remember("Nico moved to Rosario")
    _link_claims(
        [{"subject": "Nico", "predicate": "lives_in", "object": "Rosario", "supersedes": "Córdoba"}],
        ids, mem, second, source,
    )
    old_edge = mem.db.fetchone(
        "SELECT * FROM atoms WHERE type = 'relation' AND relation = 'lives_in' AND target_id = ?", (ids["Córdoba"],)
    )
    new_edge = mem.db.fetchone(
        "SELECT * FROM atoms WHERE type = 'relation' AND relation = 'lives_in' AND target_id = ?", (ids["Rosario"],)
    )
    return old_edge, new_edge


def _contradictions(mem):
    return mem.db.fetchall("SELECT * FROM atoms WHERE type = 'relation' AND relation = 'contradicts'")


def test_a_verified_update_supersedes_as_before(tmp_path):
    mem = _mem(tmp_path, _engine(_choice_provider(EXPLICIT_UPDATE), supersession="active"))
    old_edge, new_edge = _move(mem)
    assert json.loads(old_edge["metadata"])["superseded_by"] == new_edge["id"]
    assert len(_contradictions(mem)) == 1
    record = next(r for r in audit.get_all() if r["task"] == "supersession")
    assert record["outcome"] == EXPLICIT_UPDATE and record["summary"]["allowed"]


def test_a_compatible_verdict_keeps_both_claims_and_says_why(tmp_path):
    mem = _mem(tmp_path, _engine(_choice_provider(COMPATIBLE), supersession="active"))
    old_edge, new_edge = _move(mem)
    assert "superseded_by" not in json.loads(old_edge["metadata"])
    assert _contradictions(mem) == []
    deferred = json.loads(new_edge["metadata"])[SUPERSESSION_DEFERRED]
    assert deferred["old_edge"] == old_edge["id"] and deferred["label"] == COMPATIBLE
    assert mem.db.fetchall("SELECT 1 FROM evidence WHERE atom_id = ?", (old_edge["id"],)) == []


def test_an_unsure_verdict_keeps_both_claims(tmp_path):
    mem = _mem(tmp_path, _engine(_choice_provider(EXPLICIT_UPDATE, confidence=0.2), supersession="active"))
    old_edge, new_edge = _move(mem)
    assert "superseded_by" not in json.loads(old_edge["metadata"])
    assert json.loads(new_edge["metadata"])[SUPERSESSION_DEFERRED]["label"] == "insufficient_context"


def test_shadow_supersession_supersedes_and_records(tmp_path):
    mem = _mem(tmp_path, _engine(_choice_provider(COMPATIBLE), supersession="shadow"))
    old_edge, new_edge = _move(mem)
    assert json.loads(old_edge["metadata"])["superseded_by"] == new_edge["id"]
    assert SUPERSESSION_DEFERRED not in json.loads(new_edge["metadata"])
    record = next(r for r in audit.get_all() if r["task"] == "supersession")
    assert record["outcome"] == COMPATIBLE and not record["applied"]


def test_the_provider_sees_the_source_text_and_both_claims(tmp_path):
    provider = _choice_provider(EXPLICIT_UPDATE)
    mem = _mem(tmp_path, _engine(provider, supersession="active"))
    _move(mem)
    state = provider.calls[0]["state"]
    assert state["source_text"] == "Nico moved to Rosario"
    assert state["earlier_claim"]["object"] == "Córdoba" and state["earlier_claim"]["stated_by"] == "user"
    assert state["later_claim"]["object"] == "Rosario"


def test_a_claim_that_changed_during_verification_is_not_superseded(tmp_path):
    mem = _mem(tmp_path, _engine(None))

    def touch(_state):
        mem.db.execute("UPDATE atoms SET updated_at = '2099-01-01 00:00:00' WHERE relation = 'lives_in'")

    mem.decisions = _engine(_choice_provider(EXPLICIT_UPDATE, on_call=touch), supersession="active")
    old_edge, _ = _move(mem)
    assert "superseded_by" not in json.loads(old_edge["metadata"])
    assert _contradictions(mem) == []


def test_an_agent_claim_never_supersedes_what_the_user_stated(tmp_path):
    mem = _mem(tmp_path, _engine(None))
    old_edge, new_edge = _move(mem, source="agent")
    assert "superseded_by" not in json.loads(old_edge["metadata"])
    assert _contradictions(mem) == []
    assert json.loads(new_edge["metadata"])["source"] == SOURCE_AGENT


def test_an_agent_claim_may_supersede_the_agent_s_own(tmp_path):
    mem = _mem(tmp_path, _engine(None))
    resolver = EntityResolver(mem.db, mem.embed)
    ids = {n: resolver.resolve(n, t, "t", "s", ["s"]) for n, t in
           (("Nico", "person"), ("Córdoba", "location"), ("Rosario", "location"))}
    first = mem.remember("you live in Córdoba", metadata={"source": "agent"})
    _link_claims([{"subject": "Nico", "predicate": "lives_in", "object": "Córdoba"}], ids, mem, first, "agent")
    second = mem.remember("you moved to Rosario", metadata={"source": "agent"})
    _link_claims([{"subject": "Nico", "predicate": "lives_in", "object": "Rosario", "supersedes": "Córdoba"}],
                 ids, mem, second, "agent")
    old_edge = mem.db.fetchone("SELECT metadata FROM atoms WHERE relation = 'lives_in' AND target_id = ?", (ids["Córdoba"],))
    assert "superseded_by" in json.loads(old_edge["metadata"])


# ── entity verification ──────────────────────────────────────────────────────


def _resolver(mem, provider=None, mode="active", context="", **kwargs) -> EntityResolver:
    engine = _engine(provider, entity=mode) if provider is not None else _engine(None)
    return EntityResolver(mem.db, mem.embed, decisions=engine, context=context, **kwargs)


def test_an_exact_match_never_asks(tmp_path):
    provider = _choice_provider("c0")
    mem = _mem(tmp_path, _engine(None))
    resolver = _resolver(mem, provider)
    a = resolver.resolve("Alice", "person", "t", "s", ["s"])
    assert resolver.resolve("alice", "person", "t", "s", ["s"]) == a
    assert provider.calls == []


def test_an_uncertain_fuzzy_match_is_verified_against_the_sentence(tmp_path):
    stored, mention = "Alex Rivera", "Alex"
    score = fuzz.WRatio(mention, stored)
    assert 85 <= score < 92, score  # an uncertain tier-2 match, by construction

    mem = _mem(tmp_path, _engine(None))
    plain = EntityResolver(mem.db, mem.embed)
    rivera = plain.resolve(stored, "person", "t", "s", ["s"])
    acme = plain.resolve("Acme", "organization", "t", "s", ["s"])
    mem.atomspace.link_atoms(rivera, acme, "works_for", "t", "s")

    # "none": the mention is somebody else — a provisional duplicate is made
    provider = _choice_provider("none")
    new_id = _resolver(mem, provider, context="Alex from Globex called").resolve(mention, "person", "t", "s", ["s"])
    assert new_id != rivera
    state = provider.calls[0]["state"]
    assert state["sentence"] == "Alex from Globex called"
    assert state["candidates"][0]["facts"] == ["works_for Acme"]

    # "c0" with confidence: the identity link, as the fuzzy tier would have made
    mem2 = _mem(tmp_path, _engine(None), name="d2")
    rivera2 = EntityResolver(mem2.db, mem2.embed).resolve(stored, "person", "t", "s", ["s"])
    assert _resolver(mem2, _choice_provider("c0")).resolve(mention, "person", "t", "s", ["s"]) == rivera2

    # unsure: a duplicate rather than an unjustified link
    mem3 = _mem(tmp_path, _engine(None), name="d3")
    rivera3 = EntityResolver(mem3.db, mem3.embed).resolve(stored, "person", "t", "s", ["s"])
    assert _resolver(mem3, _choice_provider("c0", confidence=0.2)).resolve(mention, "person", "t", "s", ["s"]) != rivera3

    # shadow: the fuzzy match stands, the verdict is only recorded
    mem4 = _mem(tmp_path, _engine(None), name="d4")
    rivera4 = EntityResolver(mem4.db, mem4.embed).resolve(stored, "person", "t", "s", ["s"])
    assert _resolver(mem4, _choice_provider("none"), mode="shadow").resolve(mention, "person", "t", "s", ["s"]) == rivera4
    assert any(r["task"] == "entity" and not r["applied"] for r in audit.get_all())


def test_a_near_certain_fuzzy_match_is_not_questioned(tmp_path):
    provider = _choice_provider("none")
    mem = _mem(tmp_path, _engine(None))
    js = EntityResolver(mem.db, mem.embed).resolve("JavaScript", "technology", "t", "s", ["s"])
    assert _resolver(mem, provider).resolve("Javascript", "technology", "t", "s", ["s"]) == js
    assert provider.calls == []


def test_an_embedding_match_is_verified_too(tmp_path):
    mem = _mem(tmp_path, _engine(None))
    smith = EntityResolver(mem.db, mem.embed).resolve("Alice Smith", "person", "t", "s", ["s"])
    # fuzzy is switched off so the mention reaches the embedding tier, where the
    # bag-of-words stand-in puts "Alice" within reach of "Alice Smith"
    kwargs = dict(fuzzy_threshold=101.0, cosine_threshold=0.5)
    assert EntityResolver(mem.db, mem.embed, **kwargs).resolve("Alice", "person", "t", "s", ["s"]) == smith

    provider = _choice_provider("none")
    duplicate = _resolver(mem, provider, **kwargs).resolve("Alice", "person", "t", "s", ["s"])
    assert duplicate != smith
    assert provider.calls and provider.calls[0]["state"]["candidates"][0]["label"] == "Alice Smith"


# ── tone ─────────────────────────────────────────────────────────────────────


def _tone_provider(label: str, confidence: float = 0.9) -> StaticProvider:
    def answer(_k, question, _s):
        assert isinstance(question, Choice)
        return {"type": "choice", "choice": label, "probabilities": {label: confidence}, "confidence": confidence}

    return StaticProvider(answer)


@pytest.fixture
def grim_estimate(monkeypatch):
    """The sentiment estimator reads every text as a grave one — the reading
    a curt request or an apology gets, and the one that buys a pruning floor."""
    monkeypatch.setattr("smrti.estimate_valence", lambda text, embed: -0.9)


def test_a_speakers_mood_is_damped_under_every_line_it_used_to_cross(tmp_path, grim_estimate):
    provider = _tone_provider(TONE_SPEAKER)
    mem = _mem(tmp_path, _engine(provider, tone="active"))
    atom_id = mem.remember("cerra el navegador")
    row = _row(mem, atom_id)
    damped = -0.9 * mem.decisions.policy.tone_damping
    # both pairs carry the damped tone, and the intensity follows it
    assert row["valence"] == pytest.approx(damped)
    assert row["intrinsic_valence"] == pytest.approx(damped)
    assert row["intensity"] == pytest.approx(abs(damped))
    assert row["intrinsic_intensity"] == pytest.approx(abs(damped))
    # no pruning floor: the estimate alone would have bought one at −0.9
    assert row["lti"] == 0.0
    meta = json.loads(row["metadata"])
    assert meta[TONE_DECISION] == {"estimate": -0.9, "label": TONE_SPEAKER, "confidence": 0.9}
    # still an estimate: no decision may state a valence
    assert VALENCE_STATED not in meta
    assert provider.calls[0]["state"]["author"] == "user"
    assert audit.counters()[("tone", "active", TONE_SPEAKER)] == 1


def test_a_verdict_on_the_thing_itself_keeps_the_estimate_and_its_floor(tmp_path, grim_estimate):
    mem = _mem(tmp_path, _engine(_tone_provider(TONE_THING), tone="active"))
    row = _row(mem, mem.remember("the deploy wiped the production volume"))
    assert row["valence"] == pytest.approx(-0.9) and row["intrinsic_valence"] == pytest.approx(-0.9)
    assert row["lti"] == 0.5
    assert TONE_DECISION not in json.loads(row["metadata"])


def test_a_stated_valence_is_never_questioned(tmp_path, grim_estimate):
    provider = _tone_provider(TONE_SPEAKER)
    mem = _mem(tmp_path, _engine(provider, tone="active"))
    row = _row(mem, mem.remember("never deploy without a backup", valence=-0.8))
    assert row["valence"] == pytest.approx(-0.8) and row["lti"] == 0.5
    assert json.loads(row["metadata"])[VALENCE_STATED] is True
    assert provider.calls == []


def test_shadow_tone_stores_the_estimate_and_records_the_verdict(tmp_path, grim_estimate):
    mem = _mem(tmp_path, _engine(_tone_provider(TONE_SPEAKER), tone="shadow"))
    row = _row(mem, mem.remember("cerra el navegador"))
    assert row["valence"] == pytest.approx(-0.9) and row["lti"] == 0.5
    assert TONE_DECISION not in json.loads(row["metadata"])
    record = audit.get_all()[0]
    assert record["task"] == "tone" and record["mode"] == "shadow" and not record["applied"]


def test_an_unavailable_tone_judge_stores_the_estimate(tmp_path, grim_estimate):
    class _Down:
        name = "down"
        model = "m"

        def ask(self, *a, **k):
            from smrti.decisions import DecisionUnavailable
            raise DecisionUnavailable("loading")

    mem = _mem(tmp_path, _engine(_Down(), tone="active"))
    row = _row(mem, mem.remember("cerra el navegador"))
    assert row["valence"] == pytest.approx(-0.9)
    assert TONE_DECISION not in json.loads(row["metadata"])


def test_beliefs_and_the_remember_tool_go_through_the_same_judge(tmp_path, grim_estimate):
    provider = _tone_provider(TONE_SPEAKER)
    mem = _mem(tmp_path, _engine(provider, tone="active"))
    row = _row(mem, mem.believe("disculpa por la frustración causada", 0.8, source=SOURCE_AGENT))
    assert row["valence"] == pytest.approx(-0.9 * mem.decisions.policy.tone_damping)
    assert provider.calls[-1]["state"] == {
        "text": "disculpa por la frustración causada", "author": "assistant", "kind": "belief",
    }
    out = handle_tool(mem, "smrti_remember", {"content": "¿y por qué afirmaste que seguía con timeouts?",
                                              "source": "agent"})
    assert provider.calls[-1]["state"]["author"] == "assistant"
    assert TONE_DECISION in json.loads(_row(mem, out["atom_id"])["metadata"])
    # a stated one through the tool is a report, and stands
    out = handle_tool(mem, "smrti_remember", {"content": "no afirmar problemas sin verificarlos primero",
                                              "type": "belief", "valence": -0.8})
    assert _row(mem, out["atom_id"])["valence"] == pytest.approx(-0.8)
    assert len(provider.calls) == 2


# ── the server surface ───────────────────────────────────────────────────────


def test_the_proxy_budget_keeps_constraints_first():
    from smrti.servers.proxy import _within_budget

    warnings = ["- YOU MUST NOT: " + "w" * 30, "- AVOID: " + "a" * 30]
    context = ["- Note: " + "c" * 30, "- Note: " + "d" * 30]
    assert _within_budget(warnings, context, 0) == (warnings, context)
    kept_w, kept_c = _within_budget(warnings, context, 130)
    assert kept_w == warnings and kept_c == context[:1]
    kept_w, kept_c = _within_budget(warnings, context, 10)
    assert kept_w == warnings[:1] and kept_c == []  # one warning is always kept


def test_the_proxy_recall_log_carries_ids_and_evidence(tmp_path):
    from smrti.core.models import Atom, AtomType, RecallResult, TruthValue
    from smrti.servers.proxy import _inject_context

    atom = Atom(id="atom-9", type=AtomType.EPISODE, label="fact", content="fact",
                truth=TruthValue(probability=0.8, confidence=0.8))
    result = RecallResult(atom=atom, salience=0.5, similarity=0.7, evidence=0.83)
    with patch("smrti.servers.proxy._recall", AsyncMock(return_value=[result])):
        with patch("smrti.servers.proxy.get_mem", return_value=MagicMock()):
            _, _, memories = run(_inject_context({"messages": [{"role": "user", "content": "q"}]}, "t", "s", ["s"]))
    assert memories[0]["id"] == "atom-9" and memories[0]["evidence"] == 0.83


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient
    from smrti.servers import rest as rest_mod

    async def _noop_reflect(*_args, **_kwargs):
        return

    instance = Smrti(db_path=str(tmp_path / "rest.db"), tenant_id="default", write_space="default",
                     decisions=_engine(None))
    with patch.object(rest_mod, "get_mem", return_value=instance):
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            with patch("smrti.servers.config.EXTRACT", False):
                with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
                    yield c


def test_the_decision_log_endpoints_and_metrics(client):
    assert client.get("/decisions").json() == []
    audit.record(task="rerank", mode="shadow", tenant_id="default", space="default", provider="static",
                 model="m", outcome="reordered", applied=False)
    records = client.get("/decisions").json()
    assert records[0]["task"] == "rerank" and records[0]["outcome"] == "reordered"
    body = client.get("/metrics").text
    assert 'smrti_decisions_mode{task="rerank"} 0' in body
    assert 'smrti_decisions_total{task="rerank",mode="shadow",outcome="reordered"} 1' in body
    assert client.delete("/decisions").json() == {"status": "ok"}
    assert client.get("/decisions").json() == []


def test_the_attend_route(client):
    atom_id = client.post("/remember", json={"content": "the deploy pipeline builds the image"}).json()["atom_id"]
    resp = client.post("/attend", json={"atom_ids": [atom_id]})
    assert resp.status_code == 200 and resp.json()["boosted"] == 1
    assert client.post("/attend", json={"atom_ids": []}).status_code == 422


def test_in_llm_mode_the_skip_route_touches_no_tagger(tmp_path):
    # llm mode is for machines that cannot hold the 2.3 GB tagger; a message
    # the gate judged not worth a claim must not load it for the mentions.
    mem = _mem(tmp_path, _engine(_route_provider(0.05, 0.02, 0.05), routing="active"))
    episode_id, claims_only, full, ner = _hybrid(
        mem, "Thanks Nico, that worked at Cyqle",
        [{"name": "Nico", "type": "person"}, {"name": "Cyqle", "type": "organization"}],
        mode="llm",
    )
    ner.extract.assert_not_called()
    claims_only.assert_not_awaited()
    full.assert_not_awaited()
    assert _mentions(mem, episode_id) == []
    assert _row(mem, episode_id) is not None  # the episode itself stays
