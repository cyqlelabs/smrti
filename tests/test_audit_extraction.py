"""Regression tests for the extraction half of the audit: date resolution,
the extraction endpoint's credentials, tone writes, supersession and entity
resolution."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from smrti import Smrti
from smrti.extraction import extract, temporal
from smrti.extraction.temporal import annotate, resolve_spans

BASE = datetime(2026, 11, 25, 15, 0, 0)


@pytest.fixture
def mem(tmp_path):
    return Smrti(db_path=str(tmp_path / "memory.db"), tenant_id="t", write_space="s")


class SpanNER:
    """Tags the given spans wherever they occur in the text."""

    def __init__(self, spans: list[tuple[str, str]]) -> None:
        self._spans = spans

    def extract(self, text, labels=None, threshold=0.4):
        return [
            {"name": name, "type": label, "score": 0.99}
            for name, label in self._spans
            if name in text
        ]


# ── date resolution ──────────────────────────────────────────────────────────


def test_the_date_parser_is_built_once_and_resolves_against_each_call_s_base():
    """A parser built per span cost about a second a span; the reused one must
    still read "tomorrow" against the write time of the call it serves."""
    ner = SpanNER([("tomorrow", "date")])

    first = resolve_spans("The review is tomorrow", BASE, ner)
    parser = temporal._parser
    second = resolve_spans("The review is tomorrow", datetime(2020, 2, 28, 9), ner)

    assert first == [("tomorrow", "2026-11-26")]
    assert second == [("tomorrow", "2020-02-29")]
    assert temporal._parser is parser


def test_resolving_a_year_twice_nests_no_annotation():
    """"2019" occurs again inside its own annotation; the second run annotated
    that occurrence and nested a new annotation inside the old one."""
    ner = SpanNER([("2019", "date")])
    stored = "I moved to Berlin in 2019 [resolved: 2019-11-25]."

    assert annotate(stored, BASE, ner) == stored
    assert annotate(annotate("I moved to Berlin in 2019.", BASE, ner), BASE, ner) == (
        "I moved to Berlin in 2019."
    )


def test_a_month_or_a_year_is_never_given_a_day():
    """"March" became the 25th of March and "2019" the 25th of November 2019."""
    for span in ("March", "2019", "last year"):
        assert resolve_spans(f"It happened {span}", BASE, SpanNER([(span, "date")])) == []


def test_a_duration_without_a_direction_is_not_resolved_into_the_past():
    """The tagger cut "in two weeks" to "two weeks", which the parser read as
    two weeks ago."""
    ner = SpanNER([("two weeks", "date")])

    assert resolve_spans("She turns 30 in two weeks", BASE, ner) == []
    assert resolve_spans(
        "She turns 30 in two weeks", BASE, SpanNER([("in two weeks", "date")])
    ) == [("in two weeks", "2026-12-09")]


def test_a_numeric_date_is_resolved_only_when_its_numbers_settle_the_order():
    """"12/03/1990" was read month-first whatever language it was written in."""
    assert resolve_spans("Born on 12/03/1990", BASE, SpanNER([("12/03/1990", "date")])) == []
    assert resolve_spans(
        "Born on 25/03/1990", BASE, SpanNER([("25/03/1990", "date")])
    ) == [("25/03/1990", "1990-03-25")]


# ── the extraction endpoint ──────────────────────────────────────────────────


def _chat_request(monkeypatch, upstream, auth, server_key="", extract_key=""):
    """The (url, Authorization) one extraction call sends."""
    from smrti.servers import config as cfg

    monkeypatch.setattr(cfg, "API_KEY", server_key)
    monkeypatch.setattr(extract, "_EXTRACT_API_KEY", extract_key)
    seen = []

    async def endpoint(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            await extract._post_chat(client, upstream, auth, {"messages": []})

    asyncio.run(call())
    return seen[0]


def test_the_smrti_server_key_is_never_forwarded_to_the_extraction_endpoint(monkeypatch):
    """A client that authenticated to Smrti with its key had that key posted
    to whatever endpoint extraction pointed at."""
    _, sent = _chat_request(monkeypatch, "http://llm", "Bearer server-key", server_key="server-key")
    assert sent is None
    _, sent = _chat_request(monkeypatch, "http://llm", "Bearer upstream-key", server_key="server-key")
    assert sent == "Bearer upstream-key"


def test_an_extraction_api_key_is_the_endpoint_s_credential(monkeypatch):
    _, sent = _chat_request(
        monkeypatch, "http://llm", "Bearer upstream-key", server_key="server-key", extract_key="ek",
    )
    assert sent == "Bearer ek"


def test_an_upstream_that_ends_in_v1_is_not_given_a_second_one(monkeypatch):
    """The documented Docker example names .../v1, and the call went to /v1/v1."""
    url, _ = _chat_request(monkeypatch, "http://host:11434/v1/", "")
    assert url == "http://host:11434/v1/chat/completions"
    url, _ = _chat_request(monkeypatch, "http://host:11434", "")
    assert url == "http://host:11434/v1/chat/completions"


def test_local_mode_calls_no_model_for_an_agent_turn(mem):
    """Agent turns took the full LLM path whatever the mode, sending the
    assistant's text to the configured upstream in a mode promised local."""
    episode = mem.remember("I recommend Docker for the project.", metadata={"source": "agent"})
    ner = MagicMock()
    ner.extract.return_value = [{"name": "Docker", "type": "technology"}]
    post = AsyncMock()
    with patch("smrti.extraction.ner.get_ner", return_value=ner), patch.object(extract, "_post_chat", post):
        asyncio.run(extract.extract_and_link_hybrid(
            episode, "I recommend Docker for the project.", mem, "", "m", "http://llm",
            source="agent", mode="local",
        ))
    assert post.await_count == 0
    assert mem.db.fetchone("SELECT 1 FROM atoms WHERE label = 'Docker'") is not None


def test_a_failed_extraction_call_is_logged_as_a_warning(caplog):
    """Failures went only to the in-process call log, so an endpoint failing
    every call left nothing in the server's own log."""
    post = AsyncMock(return_value=httpx.Response(503))
    with patch.object(extract, "_post_chat", post), caplog.at_level("WARNING", logger="smrti.extract"):
        result = asyncio.run(extract.extract_knowledge("Alice met Bob", None, "http://llm", "", "m"))
    assert result is None
    assert any("HTTP 503" in record.getMessage() for record in caplog.records)


# ── tone ─────────────────────────────────────────────────────────────────────


def _concept(mem, label, entity_type="concept"):
    """An entity atom, created the way extraction creates one."""
    return extract._resolver(mem, "user", "", context="").resolve(label, entity_type, "t", "s", ["s"])


def test_a_claim_never_lowers_a_tone_the_caller_stated(mem):
    """A stated −0.1 note kept its VALENCE_STATED mark when a −0.9 claim
    lowered its tone, and was injected as "YOU MUST NOT: …"."""
    from smrti.core.models import RecallResult
    from smrti.retrieval.classify import classify_memory

    ids = {"Alice": _concept(mem, "Alice", "person"), "Rome": _concept(mem, "Rome", "location")}
    episode = mem.remember("Enforce the security policy.", valence=-0.1)
    extract._link_claims(
        [{"subject": "Alice", "predicate": "lives_in", "object": "Rome", "valence": -0.9}],
        ids, mem, episode_id=episode,
    )

    atom = mem.atomspace.get_atom(episode, "t", "s")
    assert atom.valence.own == pytest.approx(-0.1)
    assert classify_memory(RecallResult(atom=atom, salience=1, similarity=1)) == "context"
    # An estimated tone is still lowered, as before.
    assert mem.atomspace.get_atom(ids["Rome"], "t", "s").valence.own == pytest.approx(-0.9)


# ── supersession and revival ─────────────────────────────────────────────────


def _places(mem, *names):
    ids = {"Alice": _concept(mem, "Alice", "person")}
    ids.update({name: _concept(mem, name, "location") for name in names})
    return ids


def _edge(mem, ids, place, predicate="lives_in"):
    return mem.db.fetchone(
        "SELECT * FROM atoms WHERE type = 'relation' AND relation = ? AND source_id = ? AND target_id = ?",
        (predicate, ids["Alice"], ids[place]),
    )


def _lives_in(place, **extra):
    return {"subject": "Alice", "predicate": "lives_in", "object": place, **extra}


def test_an_agent_restating_a_replaced_claim_does_not_revive_it(mem):
    """_supersede refused agent claims over the user's, but _revive did not:
    the assistant repeating the old city undid the user's correction."""
    ids = _places(mem, "Berlin", "Paris")
    extract._link_claims([_lives_in("Berlin")], ids, mem)
    extract._link_claims([_lives_in("Paris", supersedes="Berlin")], ids, mem)

    extract._link_claims([_lives_in("Berlin")], ids, mem, source="agent")

    assert "superseded_by" in json.loads(_edge(mem, ids, "Berlin")["metadata"])


def test_a_claim_under_another_predicate_is_never_superseded(mem):
    """"works_in Paris, supersedes Madrid" cut "born_in Madrid"."""
    ids = _places(mem, "Madrid", "Paris")
    extract._link_claims([{"subject": "Alice", "predicate": "born_in", "object": "Madrid"}], ids, mem)
    extract._link_claims(
        [{"subject": "Alice", "predicate": "works_in", "object": "Paris", "supersedes": "Madrid"}], ids, mem,
    )

    assert "superseded_by" not in json.loads(_edge(mem, ids, "Madrid", "born_in")["metadata"] or "{}")


def test_a_claim_with_a_structural_or_blank_predicate_draws_no_edge(mem):
    """A model-emitted "contradicts" edge is one the epoch acts on."""
    ids = _places(mem, "Berlin")
    extract._link_claims(
        [_lives_in("Berlin", predicate="contradicts"), _lives_in("Berlin", predicate="  ")], ids, mem,
    )

    assert mem.db.fetchall("SELECT relation FROM atoms WHERE type = 'relation'") == []


def test_a_revived_claim_is_not_pulled_back_down_by_the_supersession_it_undid(mem):
    """The 0.1 evidence the supersession filed was still pending at revival,
    and the next epoch settled the revived claim at 0.67 instead of 0.8."""
    ids = _places(mem, "Berlin", "Paris")
    extract._link_claims([_lives_in("Berlin")], ids, mem)
    extract._link_claims([_lives_in("Paris", supersedes="Berlin")], ids, mem)
    extract._link_claims([_lives_in("Berlin", supersedes="Paris")], ids, mem)

    mem.reflect()

    assert _edge(mem, ids, "Berlin")["probability"] >= 0.8 - 1e-9


# ── entity resolution ────────────────────────────────────────────────────────


def _resolve(mem, name, entity_type):
    return extract._resolver(mem, "user", "", context="").resolve(name, entity_type, "t", "s", ["s"])


def test_a_label_shared_with_a_person_organization_or_place_is_a_homonym(mem):
    """Tier 0b matched a label across any entity type: Java the island became
    Java the language and Apple the company the fruit."""
    assert _resolve(mem, "Java", "technology") != _resolve(mem, "Java", "location")
    assert _resolve(mem, "Apple", "organization") != _resolve(mem, "Apple", "concept")


def test_a_preference_makes_the_concept_of_the_same_name_a_belief(mem):
    """The preference resolved to a fresh belief beside the concept, so the
    reclassification after it never fired and "coffee" existed twice."""
    concept = _resolve(mem, "coffee", "concept")

    assert _resolve(mem, "coffee", "preference") == concept
    row = mem.db.fetchone("SELECT type, entity_type FROM atoms WHERE id = ?", (concept,))
    assert (row["type"], row["entity_type"]) == ("belief", "preference")


def test_a_name_that_only_looks_like_a_known_one_is_a_new_atom(mem):
    """WRatio ≥ 85 merged Paul into Paula and Java into JavaScript, with no
    decision to confirm it."""
    for known, mention, entity_type in [
        ("Paul", "Paula", "person"), ("Jon", "John", "person"), ("Ana", "Anabel", "person"),
        ("Java", "JavaScript", "technology"), ("React", "React Native", "technology"),
    ]:
        assert _resolve(mem, known, entity_type) != _resolve(mem, mention, entity_type), mention
    assert mem.db.fetchall("SELECT alias FROM aliases") == []


def test_an_accent_is_not_a_different_name_but_a_kana_mark_is(mem):
    assert _resolve(mem, "José", "person") == _resolve(mem, "Jose", "person")
    assert _resolve(mem, "かとう", "person") != _resolve(mem, "がとう", "person")


def test_a_claim_naming_a_forgotten_atom_is_about_a_new_one(mem):
    """_db_resolve_label found the forgotten atom the resolver skips."""
    ids = {"Carol": _concept(mem, "Carol", "person")}
    dave = _concept(mem, "Dave", "person")
    mem.forget(atom_ids=[dave])

    extract._link_claims([{"subject": "Carol", "predicate": "met", "object": "Dave"}], ids, mem)

    edge = mem.db.fetchone("SELECT target_id FROM atoms WHERE relation = 'met'")
    assert edge["target_id"] != dave


# ── aliases and pronouns ─────────────────────────────────────────────────────


def test_a_pronoun_alias_an_older_build_stored_is_never_a_match_and_is_purged(mem):
    """"she" stored as Bob's alias made Bob of every later "she"."""
    bob = _concept(mem, "Bob", "person")
    mem.db.execute(
        "INSERT INTO aliases (alias, atom_id, tenant_id, space) VALUES ('she', ?, 't', 's')", (bob,),
    )

    assert _resolve(mem, "She", "person") != bob
    assert mem.db.fetchall("SELECT alias FROM aliases") == []


def test_extraction_stores_no_alias_but_resolves_its_claims_with_them(mem):
    episode = mem.remember("Bob said he would come.")
    entity_ids = extract._resolve_ner_entities(
        [{"name": "Bob", "type": "person", "aliases": ["he", "Robert"]}], episode, mem,
    )

    assert entity_ids["he"] == entity_ids["Robert"] == entity_ids["Bob"]
    assert mem.db.fetchall("SELECT alias FROM aliases") == []


def test_aliases_in_the_model_s_output_are_validated(mem):
    """A string was read one alias per letter; a number aborted the extraction."""
    episode = mem.remember("Bob and Dana are here.")
    entity_ids = extract._resolve_ner_entities(
        [{"name": "Bob", "type": "person", "aliases": "Robert"},
         {"name": "Carl", "type": "person", "aliases": [42]},
         {"name": "Dana", "type": "person"}],
        episode, mem,
    )

    assert entity_ids["Robert"] == entity_ids["Bob"]
    assert "R" not in entity_ids
    assert {"Carl", "Dana"} <= set(entity_ids)


def test_a_pronoun_is_typed_from_the_tagger_in_context_not_a_word_list():
    """The word list missed Dutch "Hij", which became a person, and took the
    names "Mia" and "Ella" for pronouns, which dropped them."""
    from smrti.extraction.ner import NERProvider
    from smrti.extraction.pronouns import merge_pronoun_entities_in_batch

    span = lambda text, label, score: MagicMock(text=text, label=label, score=score)
    provider = NERProvider()
    provider._model = MagicMock()
    provider._model.extract_entities.return_value = [
        span("Hij", "person", 0.93), span("Hij", "pronoun", 0.41), span("Philips", "organization", 1.0),
    ]
    assert {e["name"]: e["type"] for e in provider.extract("Hij werkt bij Philips")} == {
        "Hij": "pronoun", "Philips": "organization",
    }

    people = [{"name": "Mia", "type": "person", "aliases": []}, {"name": "Ella", "type": "person", "aliases": []}]
    assert merge_pronoun_entities_in_batch(people) == people


# ── the session locks ────────────────────────────────────────────────────────


def test_idle_session_locks_are_dropped_but_a_held_one_is_kept():
    """One lock per (tenant, space) ever seen was kept forever."""
    locks = extract._PerLoopLocks()

    async def flood():
        async with locks.session("held"):
            held = locks.get_lock("held")
            for n in range(extract._MAX_SESSION_LOCKS * 2):
                async with locks.session(f"tenant-{n}:space"):
                    pass
            assert locks.get_lock("held") is held
        return len(locks._loop_entry()[1])

    assert asyncio.run(flood()) <= extract._MAX_SESSION_LOCKS
