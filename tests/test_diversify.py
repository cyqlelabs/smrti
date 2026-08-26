"""Tests for the post-ranking diversity cap.

The case being replayed: "qué sabes sobre mi familia" returned five episodes
stored minutes apart, every one of them a restatement of the question. Each
was ranked correctly; the response still carried one moment five times and
zero facts.

The second case, found by the LongMemEval-S harness rather than by a user: a
whole stored session shares one write timestamp, so a cap keyed on time alone
capped the session — turns saying different things, including the ones
holding the answer. Repetition is what has to be capped, and time is only
half of what identifies it.
"""
from __future__ import annotations

import pytest

from smrti import Smrti
from smrti.core.models import Atom, AtomType, RecallResult, TruthValue
from smrti.retrieval.diversify import _cluster_key, diversify
from smrti.retrieval.text import containment, word_set

RESTATEMENT = "qué sabes sobre mi familia"
SAME_MOMENT = "2026-08-25 19:00:00"
LATER = "2026-08-25 21:00:00"


def _result(
    label: str,
    salience: float,
    atom_type: AtomType = AtomType.EPISODE,
    created_at: str | None = SAME_MOMENT,
    confidence: float = 0.5,
) -> RecallResult:
    atom = Atom(
        type=atom_type,
        label=label[:100],
        content=label,
        truth=TruthValue(probability=0.8, confidence=confidence),
        created_at=created_at,
    )
    return RecallResult(atom=atom, salience=salience, similarity=0.5)


def _restatements(count: int, created_at: str = SAME_MOMENT) -> list[RecallResult]:
    """Near-copies of one question, as a live graph accumulates them."""
    variants = [
        RESTATEMENT,
        "qué sabes sobre mi familia?",
        "qué sabes sobre mi familia por favor",
        "qué sabes vos sobre mi familia",
        "che, qué sabes sobre mi familia",
        "qué sabes sobre mi familia entonces",
    ]
    return [
        _result(variants[n % len(variants)], 1.0 - n / 100, created_at=created_at)
        for n in range(count)
    ]


# ── what counts as the same thing said again ─────────────────────────────────


def test_restatements_of_one_question_read_as_near_copies():
    a, b = _restatements(2)
    assert containment(
        word_set(a.atom.content), word_set(b.atom.content)
    ) >= 0.7


def test_two_turns_of_one_conversation_do_not():
    a = word_set("My daughter Esmeralda started school this week")
    b = word_set("I rebuilt the deck over the weekend and my back hurts")
    assert containment(a, b) < 0.7


# ── time clustering ──────────────────────────────────────────────────────────


def test_atoms_written_in_the_same_window_share_a_cluster():
    a = _result("a", 1.0, created_at="2026-08-25 19:00:10")
    b = _result("b", 0.9, created_at="2026-08-25 19:09:59")
    assert _cluster_key(a) == _cluster_key(b)


def test_atoms_written_far_apart_do_not_share_a_cluster():
    a = _result("a", 1.0, created_at=SAME_MOMENT)
    b = _result("b", 0.9, created_at=LATER)
    assert _cluster_key(a) != _cluster_key(b)


def test_an_atom_with_no_timestamp_is_a_cluster_of_one():
    a = _result("a", 1.0, created_at=None)
    b = _result("b", 0.9, created_at=None)
    assert _cluster_key(a) != _cluster_key(b)


def test_an_unparseable_timestamp_is_a_cluster_of_one():
    a = _result("a", 1.0, created_at="not a date")
    b = _result("b", 0.9, created_at="not a date")
    assert _cluster_key(a) != _cluster_key(b)


# ── the cap ──────────────────────────────────────────────────────────────────


def test_one_question_restated_cannot_fill_the_whole_response():
    echoes = _restatements(5)
    elsewhere = [
        _result("Esmeralda started school this week", 0.5, created_at=LATER),
        _result("Roxana repainted the kitchen on Sunday", 0.49, created_at=LATER),
        _result("the car needs new tyres before winter", 0.48, created_at=LATER),
    ]

    kept = diversify(echoes + elsewhere, top_k=5)

    assert sum(1 for r in kept if RESTATEMENT in r.atom.content) == 2
    assert len(kept) == 5


def test_the_top_ranked_restatements_are_the_ones_kept():
    echoes = _restatements(5)
    elsewhere = [
        _result("Esmeralda started school this week", 0.5, created_at=LATER),
        _result("Roxana repainted the kitchen on Sunday", 0.49, created_at=LATER),
        _result("the car needs new tyres before winter", 0.48, created_at=LATER),
    ]

    kept = diversify(echoes + elsewhere, top_k=5)

    assert [r.atom.content for r in kept if RESTATEMENT in r.atom.content] == [
        echoes[0].atom.content,
        echoes[1].atom.content,
    ]


def test_a_whole_session_written_at_one_timestamp_is_not_capped():
    """The LongMemEval regression: distinct turns, one stored session, one clock."""
    session = [
        _result("My daughter Esmeralda started school this week", 1.00),
        _result("I graduated with a degree in marine biology", 0.99),
        _result("we moved to San Benito in the spring", 0.98),
        _result("the deck took me the whole weekend to rebuild", 0.97),
        _result("Roxana is starting a new job in October", 0.96),
        _result("I have been reading about permaculture lately", 0.95),
    ]
    noise = [_result(f"unrelated note number {n}", 0.5 - n / 100, created_at=LATER) for n in range(6)]

    kept = diversify(session + noise, top_k=5)

    assert [r.atom.content for r in kept] == [r.atom.content for r in session[:5]]


def test_repetition_in_different_moments_is_not_capped_together():
    """Two moments, each allowed its own repeats — the cap is per moment."""
    kept = diversify(
        _restatements(3) + _restatements(3, created_at=LATER), top_k=4
    )

    assert len(kept) == 4


def test_beliefs_keep_their_slots_when_episodes_outrank_them():
    episodes = [
        _result(f"a passing remark about topic {n}", 1.0 - n / 100) for n in range(4)
    ]
    facts = [
        _result("Esmeralda is my daughter", 0.2, AtomType.BELIEF),
        _result("Roxana is my partner", 0.19, AtomType.BELIEF),
    ]

    kept = diversify(episodes + facts, top_k=4)

    assert [r.atom.content for r in kept if r.atom.type == AtomType.BELIEF] == [
        "Esmeralda is my daughter",
        "Roxana is my partner",
    ]


def test_beliefs_keep_their_slots_when_concepts_outrank_them():
    """A concept taking the reserved slot would crowd the fact out just as well."""
    episodes = [_result(f"a passing remark about topic {n}", 1.0 - n / 100) for n in range(2)]
    concepts = [
        _result(f"concept number {n}", 0.6 - n / 100, AtomType.CONCEPT) for n in range(6)
    ]
    fact = _result("Esmeralda is my daughter", 0.1, AtomType.BELIEF)

    kept = diversify(episodes + concepts + [fact], top_k=5)

    assert fact in kept


def test_the_reserve_never_takes_more_than_half_the_response():
    episodes = [
        _result(f"a remark about topic {n}", 1.0 - n / 100, created_at=f"2026-08-25 {19 + n}:00:00")
        for n in range(4)
    ]
    facts = [_result(f"standing fact number {n}", 0.1 - n / 100, AtomType.BELIEF) for n in range(4)]

    kept = diversify(episodes + facts, top_k=3)

    assert len([r for r in kept if r.atom.type == AtomType.BELIEF]) <= 1


def test_a_belief_below_the_surfacing_floor_claims_no_reserved_slot():
    faint = _result("barely remembered claim", 0.9, AtomType.BELIEF, confidence=0.01)
    episodes = [
        _result(f"a remark about topic {n}", 0.5 - n / 100, created_at=f"2026-08-25 {19 + n}:00:00")
        for n in range(4)
    ]

    kept = diversify([faint] + episodes, top_k=2, min_confidence=0.1)

    # It still ranks — the cap reserves nothing for it, it does not banish it.
    assert faint in kept


def test_the_reserve_leaves_episodes_only_the_slots_it_did_not_take():
    """Held by pre-selecting the beliefs, not by a second cap on episodes."""
    episodes = [
        _result(f"a remark about topic {n}", 1.0 - n / 100,
                created_at=f"2026-08-25 {19 + n}:00:00")
        for n in range(4)
    ]
    facts = [_result(f"standing fact number {n}", 0.5 - n / 100, AtomType.BELIEF)
             for n in range(2)]

    kept = diversify(episodes + facts, top_k=4)

    # Two slots are reserved, so only two of the four episodes fit.
    assert len([r for r in kept if r.atom.type == AtomType.EPISODE]) == 2
    assert len([r for r in kept if r.atom.type == AtomType.BELIEF]) == 2


def test_a_single_topic_graph_still_answers_with_a_full_top_k():
    kept = diversify(_restatements(8), top_k=5)

    assert len(kept) == 5


def test_capped_atoms_yield_their_slot_in_salience_order():
    echoes = _restatements(5)

    kept = diversify(echoes, top_k=4)

    assert [r.atom.content for r in kept] == [r.atom.content for r in echoes[:4]]


def test_results_stay_ordered_by_salience():
    facts = [_result(f"standing fact number {n}", 0.3 - n / 100, AtomType.BELIEF) for n in range(3)]

    kept = diversify(_restatements(5) + facts, top_k=5)

    assert [r.salience for r in kept] == sorted(
        (r.salience for r in kept), reverse=True
    )


def test_a_pool_no_bigger_than_top_k_is_returned_untouched():
    pool = _restatements(3)

    assert diversify(pool, top_k=5) == pool


def test_asking_for_nothing_returns_nothing():
    assert diversify([_result("a", 1.0)], top_k=0) == []


def test_concepts_and_goals_are_never_capped():
    """Only episodes carry conversational repetition; index nodes do not."""
    same_moment = [
        _result("machine learning", 1.0 - n / 100, AtomType.CONCEPT) for n in range(5)
    ]
    later = [_result("gardening", 0.1, AtomType.CONCEPT, created_at=LATER)]

    kept = diversify(same_moment + later, top_k=4)

    assert len(kept) == 4
    assert all(r.atom.content == "machine learning" for r in kept)


# ── end to end ───────────────────────────────────────────────────────────────


@pytest.fixture
def mem(tmp_path):
    return Smrti(
        db_path=str(tmp_path / "diversify.db"),
        personality="balanced",
        tenant_id="test",
        write_space="default",
    )


def test_the_family_recall_mixes_facts_with_the_asking(mem):
    """Replay of the reported case, end to end."""
    for _ in range(5):
        mem.remember(RESTATEMENT)
    mem.believe("Esmeralda es la hija de Nicolás", probability=0.95)
    mem.believe("Roxana es la pareja de Nicolás", probability=0.95)

    results = mem.recall(RESTATEMENT, top_k=4, min_confidence=0.0)

    types = [r.atom.type for r in results]
    assert types.count(AtomType.BELIEF) == 2
    assert types.count(AtomType.EPISODE) == 2


def test_a_session_of_distinct_turns_survives_recall_intact(mem):
    """The regression the benchmark caught, end to end."""
    turns = [
        "My daughter Esmeralda started school this week",
        "I graduated with a degree in marine biology",
        "We moved to San Benito in the spring",
        "Roxana is starting a new job in October",
    ]
    for turn in turns:
        mem.remember(turn)

    results = mem.recall("what do you know about my studies", top_k=4, min_confidence=0.0)

    assert len(results) == 4
