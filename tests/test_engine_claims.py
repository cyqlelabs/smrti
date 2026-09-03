"""The headline claims, tested end to end through the facade.

Each test here replays a claim the README makes and a way it used to be
false. They run on a deterministic bag-of-words embedder so they measure the
engine's bookkeeping, not the embedding model, and so they run the same
everywhere — none of them depends on what two sentences mean, only on which
words they share.
"""
from __future__ import annotations

import hashlib
import json
import re
from unittest.mock import MagicMock

import numpy as np
import pytest

import smrti.core.embed as embed_module
from smrti import Smrti
from smrti.core.models import Atom, AtomType, EntityType, PERMANENT_PROBABILITY
from smrti.core.provenance import VALENCE_STATED
from smrti.extraction.extract import _build_entity_context, _link_claims, _store_temporal
from smrti.extraction.resolve import EntityResolver
from smrti.evolution.healing import heal_orphaned_episodes
from smrti.retrieval.classify import classify_memory
from smrti.servers.reflect_loop import _was_used

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
    """Shared words, shared direction; nothing else.

    Every vector also carries a common component, because real sentence
    embeddings do: two unrelated texts sit at a small positive cosine with
    the model, never at zero, and a stand-in that put them at zero would
    exercise a branch the model never reaches.
    """

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


@pytest.fixture
def mem(tmp_path):
    return Smrti(db_path=str(tmp_path / "claims.db"), tenant_id="t", write_space="s")


def _row(mem, atom_id):
    return mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (atom_id,))


def _rank(results, atom_id):
    for i, r in enumerate(results):
        if r.atom.id == atom_id:
            return i
    return None


# ── forgetting ────────────────────────────────────────────────────────────────


def test_forget_does_not_raise_the_attention_of_what_it_forgets(mem):
    atom_id = mem.remember("the deploy pipeline uses Jenkins", valence=0.0)
    assert _row(mem, atom_id)["sti"] == 0.0

    mem.forget("deploy pipeline Jenkins")

    assert _row(mem, atom_id)["sti"] == 0.0


def test_a_forgotten_memory_stops_surfacing_at_any_floor(mem):
    atom_id = mem.remember("the deploy pipeline uses Jenkins", valence=0.0)
    assert _rank(mem.recall("deploy pipeline Jenkins"), atom_id) is not None

    mem.forget("deploy pipeline Jenkins")

    assert _rank(mem.recall("deploy pipeline Jenkins", min_confidence=0.0), atom_id) is None
    row = _row(mem, atom_id)
    assert row["confidence"] < mem._surfacing_floor()
    assert json.loads(row["metadata"])["forgotten"] is True


def test_a_forgotten_user_memory_can_be_pruned(mem):
    atom_id = mem.remember("the deploy pipeline uses Jenkins", valence=0.0)
    mem.forget("deploy pipeline Jenkins")

    mem.reflect()

    assert _row(mem, atom_id) is None


# ── the surfacing floor ───────────────────────────────────────────────────────


def test_the_personality_floor_gates_recall_when_the_caller_names_none(tmp_path):
    mem = Smrti(
        db_path=str(tmp_path / "floor.db"), tenant_id="t", write_space="s",
        personality="deterministic",  # min_confidence_to_surface = 0.3
    )
    atom_id = mem.remember("kubernetes ingress rewrites the host header", valence=0.0)
    mem.db.execute("UPDATE atoms SET confidence = 0.15 WHERE id = ?", (atom_id,))

    assert _rank(mem.recall("kubernetes ingress host header"), atom_id) is None
    assert _rank(mem.recall("kubernetes ingress host header", min_confidence=0.1), atom_id) is not None


# ── relevance gates standing ──────────────────────────────────────────────────


def test_a_saturated_person_hub_does_not_outrank_relevant_episodes(mem):
    """The failure replayed: "Alice" was the first result for a Kubernetes query."""
    alice = mem.atomspace.add_atom(Atom(
        type=AtomType.CONCEPT, label="Alice", entity_type=EntityType.PERSON,
        tenant_id="t", space="s",
    ))
    mem.db.execute(
        "UPDATE atoms SET sti = 3.0, lti = 0.9, confidence = 0.9 WHERE id = ?", (alice,)
    )
    episodes = [
        mem.remember(f"kubernetes ingress rule {i} rewrites the host header", valence=0.0)
        for i in range(5)
    ]

    results = mem.recall("kubernetes ingress host header", top_k=5)

    assert results[0].atom.id in episodes
    alice_rank = _rank(results, alice)
    assert alice_rank is None or alice_rank == len(results) - 1
    for r in results:
        if r.atom.id in episodes:
            assert alice_rank is None or _rank(results, r.atom.id) < alice_rank


def test_an_aged_critical_error_outranks_fresh_trivia_through_recall(mem):
    error = mem.remember(
        "the Friday deploy of the payments service broke production and lost orders",
        valence=-0.9,
    )
    # Aged: no attention left, confidence at the floor, LTI held by the
    # critical floor.
    mem.db.execute(
        "UPDATE atoms SET sti = 0.0, confidence = 0.1, lti = 0.5 WHERE id = ?", (error,)
    )
    trivia = mem.remember(
        "the Friday deploy of the payments service went out smoothly as usual",
        valence=0.0,
    )
    mem.db.execute(
        "UPDATE atoms SET sti = 1.0, confidence = 0.5 WHERE id = ?", (trivia,)
    )

    results = mem.recall("payments service Friday deploy", min_confidence=0.0)

    assert _rank(results, error) is not None
    assert _rank(results, error) < _rank(results, trivia)
    assert classify_memory(results[_rank(results, error)]) == "critical_warning"


# ── epochs are units of use ───────────────────────────────────────────────────


def test_the_reflect_loop_skips_a_space_nobody_used(mem):
    assert mem.ops_since_reflect == 0
    assert not _was_used(mem)

    mem.remember("a note", valence=0.0)
    assert _was_used(mem)

    mem.reflect()
    assert mem.ops_since_reflect == 0
    assert not _was_used(mem)


def test_an_instance_that_reports_no_activity_is_consolidated_as_before():
    assert _was_used(MagicMock())


# ── one atom per kind, whatever door it came through ─────────────────────────


def test_remember_with_type_belief_is_believe(mem):
    via_remember = mem.remember("Python suits ML", type="belief",
                                probability=PERMANENT_PROBABILITY, valence=0.0)
    via_believe = mem.believe("Python suits ML", probability=PERMANENT_PROBABILITY, valence=0.0)

    assert _row(mem, via_remember)["confidence"] == _row(mem, via_believe)["confidence"]
    assert _row(mem, via_remember)["confidence"] == pytest.approx(PERMANENT_PROBABILITY)


def test_the_reason_for_a_belief_is_recorded(mem):
    atom_id = mem.believe("cats are mammals", probability=0.85,
                          evidence="the team survey said so", valence=0.0)

    [row] = mem.evidence(atom_id)

    assert row.text == "the team survey said so"
    assert row.source == "user"
    assert row.observed_probability == pytest.approx(0.85)


def test_intensity_is_a_dimension_the_caller_can_state(mem):
    mild = mem.remember("never use the staging keys in prod", valence=-0.9, intensity=0.2)
    grave = mem.remember("never use the staging keys in prod again", valence=-0.9)

    assert _row(mem, mild)["intrinsic_intensity"] == pytest.approx(0.2)
    assert _row(mem, grave)["intrinsic_intensity"] == pytest.approx(0.9)
    hits = {r.atom.id: r for r in mem.recall("staging keys in prod", min_confidence=0.0)}
    assert classify_memory(hits[grave]) == "critical_warning"
    assert classify_memory(hits[mild]) == "context"


def test_an_extracted_concept_carries_its_own_tone(mem):
    """Resolver-created atoms used to have NULL intrinsic columns and drift."""
    resolver = EntityResolver(mem.db, mem.embed)
    concept = resolver.resolve("Postgres", "technology", "t", "s", ["s"])

    row = _row(mem, concept)
    assert row["intrinsic_valence"] == 0.0
    assert row["intrinsic_intensity"] == 0.0

    episode = mem.remember("Postgres corrupted the whole database, never again", valence=-0.9)
    mem.atomspace.link_atoms(episode, concept, "mentions", "t", "s")
    for _ in range(20):
        mem.reflect()
    row = _row(mem, concept)
    assert row["valence"] < 0.0  # the mood moved
    assert row["intrinsic_valence"] == 0.0  # the judged tone did not


# ── supersession: contradictions have a producer ─────────────────────────────


def _person_with_claim(mem, name, predicate, obj):
    resolver = EntityResolver(mem.db, mem.embed)
    person = resolver.resolve(name, "person", "t", "s", ["s"])
    place = resolver.resolve(obj, "location", "t", "s", ["s"])
    edge = mem.atomspace.link_atoms(person, place, predicate, "t", "s")
    return person, place, edge


def test_a_superseding_claim_names_the_old_one_the_loser(mem):
    alice, amsterdam, old_edge = _person_with_claim(mem, "Alice", "lives_in", "Amsterdam")
    assert "lives_in Amsterdam" in _build_entity_context(mem)

    episode = mem.remember("Quick update: I moved to Berlin last month", valence=0.0)
    entity_ids = {"Alice": alice, "Amsterdam": amsterdam}
    _link_claims(
        [{"subject": "Alice", "predicate": "lives_in", "object": "Berlin",
          "supersedes": "Amsterdam"}],
        entity_ids, mem, episode_id=episode,
    )

    contradiction = mem.db.fetchone(
        "SELECT source_id, target_id, metadata FROM atoms "
        "WHERE type = 'relation' AND relation = 'contradicts'"
    )
    assert contradiction is not None
    assert contradiction["target_id"] == old_edge
    assert json.loads(contradiction["metadata"])["loser"] == old_edge
    assert json.loads(_row(mem, old_edge)["metadata"])["superseded_by"] == contradiction["source_id"]

    context = _build_entity_context(mem)
    assert "lives_in Berlin" in context
    assert "lives_in Amsterdam" not in context


def test_a_superseded_claim_loses_probability_and_confidence_at_the_next_epoch(mem):
    alice, amsterdam, old_edge = _person_with_claim(mem, "Alice", "lives_in", "Amsterdam")
    before = _row(mem, old_edge)
    episode = mem.remember("I moved to Berlin", valence=0.0)
    _link_claims(
        [{"subject": "Alice", "predicate": "lives_in", "object": "Berlin",
          "supersedes": "Amsterdam"}],
        {"Alice": alice, "Amsterdam": amsterdam}, mem, episode_id=episode,
    )

    result = mem.reflect()

    after = _row(mem, old_edge)
    assert result.contradictions_resolved == 1
    assert after["probability"] < before["probability"]
    assert after["confidence"] < before["confidence"]


def test_a_superseded_belief_becomes_a_known_antipattern(mem):
    resolver = EntityResolver(mem.db, mem.embed)
    alice = resolver.resolve("Alice", "person", "t", "s", ["s"])
    dark = resolver.resolve("dark mode", "preference", "t", "s", ["s"])  # a belief atom
    mem.atomspace.link_atoms(alice, dark, "prefers", "t", "s")
    mem.db.execute("UPDATE atoms SET confidence = 0.6 WHERE id = ?", (dark,))
    episode = mem.remember("I switched to light mode, dark mode strains my eyes", valence=0.0)

    _link_claims(
        [{"subject": "Alice", "predicate": "prefers", "object": "light mode",
          "supersedes": "dark mode"}],
        {"Alice": alice, "dark mode": dark}, mem, episode_id=episode,
    )
    for _ in range(3):
        mem.reflect()

    hits = {r.atom.id: r for r in mem.recall("dark mode", min_confidence=0.0)}
    assert dark in hits
    assert hits[dark].atom.truth.probability < 0.3
    assert classify_memory(hits[dark]) == "known_antipattern"


# ── one temporal record ───────────────────────────────────────────────────────


def test_the_model_adds_dates_the_parser_missed_and_never_overwrites_its_own(mem):
    episode = mem.remember("the retro is next Friday and the offsite el finde que viene", valence=0.0)
    mem.db.execute(
        "UPDATE atoms SET metadata = json_set(metadata, '$.temporal', json(?)) WHERE id = ?",
        (json.dumps([{"text": "next Friday", "resolved": "2026-09-11"}]), episode),
    )

    _store_temporal(episode, mem, [
        {"text": "next Friday", "resolved": "2026-09-18"},
        {"text": "el finde que viene", "resolved": "2026-09-12"},
    ])

    stored = json.loads(_row(mem, episode)["metadata"])["temporal"]
    assert stored == [
        {"text": "next Friday", "resolved": "2026-09-11"},
        {"text": "el finde que viene", "resolved": "2026-09-12"},
    ]


# ── healing attributes by name ────────────────────────────────────────────────


def test_healing_attributes_an_orphan_to_the_person_it_names(mem):
    resolver = EntityResolver(mem.db, mem.embed)
    alice = resolver.resolve("Alice", "person", "t", "s", ["s"])
    bob = resolver.resolve("Bob", "person", "t", "s", ["s"])
    mem.db.execute("UPDATE atoms SET sti = 3.0, lti = 0.9 WHERE id = ?", (bob,))
    hiking = resolver.resolve("hiking", "topic", "t", "s", ["s"])
    episode = mem.remember("Alice went hiking in the alps", valence=0.0)
    mem.atomspace.link_atoms(episode, hiking, "mentions", "t", "s")

    assert heal_orphaned_episodes("t", "s", mem.db) == 1

    edges = mem.db.fetchall(
        "SELECT target_id FROM atoms WHERE type = 'relation' AND relation = 'mentions' AND source_id = ?",
        (episode,),
    )
    assert {e["target_id"] for e in edges} == {hiking, alice}
    assert mem.db.fetchone(
        "SELECT 1 FROM atoms WHERE type = 'relation' AND source_id = ?", (alice,)
    ) is None  # no person -> concept hub edges


# ── smrti-town reads its memories ─────────────────────────────────────────────


def test_a_citizen_avoids_a_place_it_remembers_badly(tmp_path):
    from smrti_town.agent import Citizen

    citizen = Citizen(
        name="Ana", personality="empathetic", location="Park",
        age_years=30, db_path=str(tmp_path / "town.db"), tenant_id="town",
    )
    citizen.smrti.remember("had a terrible, rotten meal at the Tavern", valence=-0.8)

    assert citizen._place_valence("Tavern") < 0.0
    assert citizen._place_valence("Library") == 0.0
