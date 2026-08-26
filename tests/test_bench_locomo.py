"""Tests for the LoCoMo harness.

The dataset ships separately, so these run against a miniature conversation in
the same schema. Under test: that both speakers are stored as testimony, that
session dates survive, that evidence is matched by turn identity, and that the
adversarial category — where the right answer is to refuse — is kept out of
the headline.
"""
from __future__ import annotations

import json

import pytest

from bench.locomo.adapter import (
    aggregate,
    evaluate_question,
    ingest,
    load_conversations,
    parse_session_date,
    select_questions,
)
from bench.locomo.run import main
from smrti import Smrti


def _sample(sample_id="conv1") -> dict:
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "I adopted a greyhound called Pip"},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "I have been painting sunrises again"},
            ],
            "session_2_date_time": "9:10 am on 3 June, 2023",
            "session_2": [
                {"speaker": "Caroline", "dia_id": "D2:1", "text": "Pip ran his first race today"},
                {"speaker": "Melanie", "dia_id": "D2:2", "text": "The gallery took two of my canvases"},
            ],
        },
        "qa": [
            {"question": "What did Caroline adopt?", "answer": "a greyhound called Pip",
             "evidence": ["D1:1"], "category": 4},
            {"question": "When did Melanie sell paintings?", "answer": "3 June 2023",
             "evidence": ["D2:2"], "category": 2},
            {"question": "What breed did Caroline's dog race against?",
             "adversarial_answer": "whippets", "evidence": ["D2:1"], "category": 5},
        ],
    }


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([_sample()]), encoding="utf-8")
    return str(path)


@pytest.fixture
def mem(tmp_path):
    return Smrti(
        db_path=str(tmp_path / "locomo.db"),
        personality="deterministic",
        tenant_id="locomo",
        write_space="c_conv1",
    )


# ── parsing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1:56 pm on 8 May, 2023", "2023-05-08 13:56:00"),
        ("9:10 am on 3 June, 2023", "2023-06-03 09:10:00"),
    ],
)
def test_session_dates_parse(raw, expected):
    assert str(parse_session_date(raw)) == expected


@pytest.mark.parametrize("raw", ["", None, "last spring"])
def test_an_unparseable_session_date_is_none(raw):
    assert parse_session_date(raw) is None


def test_every_turn_across_every_session_is_loaded(dataset):
    conversation = load_conversations(dataset)[0]
    assert len(conversation.turns) == 4
    assert conversation.speakers == ("Caroline", "Melanie")


def test_sessions_are_ordered_numerically_not_lexically(tmp_path):
    """session_10 must follow session_9, which string ordering gets wrong."""
    sample = _sample()
    sample["conversation"]["session_10"] = [
        {"speaker": "Caroline", "dia_id": "D10:1", "text": "much later"}
    ]
    sample["conversation"]["session_10_date_time"] = "8:00 am on 1 December, 2023"
    path = tmp_path / "many.json"
    path.write_text(json.dumps([sample]), encoding="utf-8")

    ids = [t.dia_id for t in load_conversations(str(path))[0].turns]
    assert ids[-1] == "D10:1"


def test_the_adversarial_reference_is_a_refusal_not_the_trap(dataset):
    conversation = load_conversations(dataset)[0]
    adversarial = next(q for q in conversation.questions if q.is_adversarial)
    assert "does not contain" in adversarial.answer
    assert "whippets" not in adversarial.answer


def test_the_question_subset_is_balanced_across_categories(dataset):
    conversation = load_conversations(dataset)[0]

    picked = select_questions(conversation, 2)

    assert len({q.category for q in picked}) == 2


def test_no_cap_keeps_every_question(dataset):
    conversation = load_conversations(dataset)[0]
    assert len(select_questions(conversation, None)) == 3


# ── ingestion ────────────────────────────────────────────────────────────────


def test_turns_are_stored_with_their_speaker(mem, dataset):
    conversation = load_conversations(dataset)[0]

    stored = ingest(conversation, mem)

    contents = {
        mem.atomspace.get_atom(a, "locomo", "c_conv1").content for a in stored
    }
    assert "Caroline: I adopted a greyhound called Pip" in contents


def test_neither_speaker_is_treated_as_an_assistant(mem, dataset):
    """LoCoMo is two people talking — nothing here earns the source discount."""
    conversation = load_conversations(dataset)[0]

    stored = ingest(conversation, mem)

    sources = {
        mem.atomspace.get_atom(a, "locomo", "c_conv1").metadata.get("source")
        for a in stored
    }
    assert sources == {None}


def test_the_session_date_becomes_the_write_time(mem, dataset):
    conversation = load_conversations(dataset)[0]

    stored = ingest(conversation, mem)

    first = next(a for a, t in stored.items() if t.dia_id == "D1:1")
    row = mem.db.fetchone("SELECT created_at FROM atoms WHERE id = ?", (first,))
    assert row["created_at"] == "2023-05-08 13:56:00"


# ── scoring ──────────────────────────────────────────────────────────────────


def test_a_recall_that_returns_the_evidence_turn_is_a_hit(mem, dataset):
    conversation = load_conversations(dataset)[0]
    stored = ingest(conversation, mem)
    question = conversation.questions[0]

    row = evaluate_question(question, mem, stored, top_k=10, min_confidence=0.0)

    assert row["evidence_hit"] is True
    assert row["category_name"] == "single-hop"


def test_a_recall_that_returns_nothing_is_a_miss(mem, dataset):
    conversation = load_conversations(dataset)[0]
    stored = ingest(conversation, mem)

    row = evaluate_question(
        conversation.questions[0], mem, stored, top_k=0, min_confidence=0.0
    )

    assert row["evidence_hit"] is False


def test_the_headline_excludes_the_adversarial_category():
    rows = [
        {"gold_turns": 1, "evidence_hit": True, "evidence_recall": 1.0,
         "adversarial": False, "category_name": "single-hop"},
        {"gold_turns": 1, "evidence_hit": False, "evidence_recall": 0.0,
         "adversarial": True, "category_name": "adversarial"},
    ]

    summary = aggregate(rows)

    assert summary["questions"] == 2
    assert summary["scored_questions"] == 1
    assert summary["retrieval_hit_rate"] == 1.0


def test_every_category_is_reported_separately():
    rows = [
        {"gold_turns": 1, "evidence_hit": True, "evidence_recall": 1.0,
         "adversarial": False, "category_name": "temporal"},
        {"gold_turns": 1, "evidence_hit": False, "evidence_recall": 0.0,
         "adversarial": False, "category_name": "multi-hop"},
    ]

    summary = aggregate(rows)

    assert summary["by_category"]["temporal"]["retrieval_hit_rate"] == 1.0
    assert summary["by_category"]["multi-hop"]["retrieval_hit_rate"] == 0.0


def test_an_empty_run_scores_zero_rather_than_dividing_by_it():
    assert aggregate([])["retrieval_hit_rate"] == 0.0


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_a_missing_dataset_is_a_clear_failure(capsys):
    assert main(["--dataset", "/nonexistent/locomo10.json"]) == 2
    assert "snap-research/locomo" in capsys.readouterr().err


def test_the_harness_runs_end_to_end(tmp_path, dataset, capsys):
    baseline = tmp_path / "baseline.json"

    code = main([
        "--dataset", dataset,
        "--baseline", str(baseline),
        "--db", str(tmp_path / "run.db"),
        "--update-baseline",
    ])

    assert code == 0
    recorded = json.loads(baseline.read_text(encoding="utf-8"))
    assert recorded["retrieval_hit_rate"] == 1.0
    assert recorded["scored_questions"] == 2
