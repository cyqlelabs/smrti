"""Tests for the HaluMem harness.

What matters here is the three-way verdict. A binary judge scores "I have no
record of that" and a confidently invented birthday identically, and this
benchmark exists precisely to separate them.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from bench.answering import classify_answer
from bench.halumem.adapter import (
    aggregate,
    ingest,
    load_users,
    parse_timestamp,
    recall_for,
    select_questions,
)
from bench.halumem.run import _compare_downward, main
from smrti import Smrti


def _user(uuid="u1") -> dict:
    return {
        "uuid": uuid,
        "sessions": [
            {
                "start_time": "Sep 04, 2025, 18:42:18",
                "end_time": "Sep 04, 2025, 19:02:18",
                "dialogue": [
                    {"role": "user", "content": "My name is Martin Mark and I was born 1996-08-02",
                     "timestamp": "Sep 04, 2025, 18:42:18"},
                    {"role": "assistant", "content": "Good to meet you, Martin.",
                     "timestamp": "Sep 04, 2025, 18:43:00"},
                ],
                "questions": [
                    {"question": "What is Martin Mark's birth date?", "answer": "1996-08-02",
                     "evidence": [{"memory_content": "birth date is 1996-08-02"}],
                     "difficulty": "easy", "question_type": "Basic Fact Recall"},
                    {"question": "What is Martin Mark's middle name?",
                     "answer": "Unknown; not provided by the user.", "evidence": [],
                     "difficulty": "easy", "question_type": "Memory Boundary"},
                ],
            }
        ],
    }


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "halumem.jsonl"
    path.write_text(json.dumps(_user()) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def mem(tmp_path):
    return Smrti(
        db_path=str(tmp_path / "halumem.db"),
        personality="deterministic",
        tenant_id="halumem",
        write_space="u_u1",
    )


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeHTTP:
    def __init__(self, content):
        self._content = content
        self.payloads = []

    async def post(self, url, headers=None, json=None):
        self.payloads.append(json)
        return _FakeResponse(self._content)


# ── parsing ──────────────────────────────────────────────────────────────────


def test_timestamps_parse():
    assert str(parse_timestamp("Sep 04, 2025, 18:42:18")) == "2025-09-04 18:42:18"


@pytest.mark.parametrize("raw", ["", None, "yesterday evening"])
def test_an_unparseable_timestamp_is_none(raw):
    assert parse_timestamp(raw) is None


def test_a_user_carries_their_turns_and_questions(dataset):
    user = load_users(dataset)[0]
    assert len(user.turns) == 2
    assert len(user.questions) == 2


def test_a_question_with_no_evidence_is_flagged_as_a_boundary(dataset):
    user = load_users(dataset)[0]
    boundary = next(q for q in user.questions if q.question_type == "Memory Boundary")
    assert boundary.has_evidence is False


def test_the_user_limit_stops_reading_early(tmp_path):
    path = tmp_path / "many.jsonl"
    path.write_text("\n".join(json.dumps(_user(f"u{n}")) for n in range(5)), encoding="utf-8")

    assert [u.uuid for u in load_users(str(path), limit=2)] == ["u0", "u1"]


def test_the_question_subset_is_balanced_across_types(dataset):
    user = load_users(dataset)[0]

    picked = select_questions(user, 2)

    assert len({q.question_type for q in picked}) == 2


# ── ingestion ────────────────────────────────────────────────────────────────


def test_assistant_turns_are_stored_as_agent_authored(mem, dataset):
    user = load_users(dataset)[0]

    assert ingest(user, mem) == 2

    rows = mem.db.fetchall(
        "SELECT content, metadata FROM atoms WHERE tenant_id='halumem' AND space='u_u1'"
    )
    agent = [r for r in rows if "agent" in (r["metadata"] or "")]
    assert len(agent) == 1
    assert "Good to meet you" in agent[0]["content"]


def test_turn_timestamps_become_the_write_time(mem, dataset):
    user = load_users(dataset)[0]
    ingest(user, mem)

    row = mem.db.fetchone(
        "SELECT created_at FROM atoms WHERE tenant_id='halumem' AND space='u_u1' "
        "ORDER BY created_at LIMIT 1"
    )
    assert row["created_at"] == "2025-09-04 18:42:18"


def test_recalled_memories_reach_the_model_dated(mem, dataset):
    user = load_users(dataset)[0]
    ingest(user, mem)

    memories = recall_for(user.questions[0], mem, top_k=10, min_confidence=0.0)

    assert memories and all(m.startswith("[2025-09-04]") for m in memories)


# ── the three-way verdict ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"verdict": "correct"}', "correct"),
        ('{"verdict": "omission"}', "omission"),
        ('{"verdict": "hallucination"}', "hallucination"),
        ('```json\n{"verdict": "correct"}\n```', "correct"),
    ],
)
def test_the_judge_returns_one_of_three_verdicts(reply, expected):
    verdict = asyncio.run(
        classify_answer(
            _FakeHTTP(reply), "http://localhost/v1", "", "m",
            "birth date?", "1996-08-02", "1996-08-02",
        )
    )
    assert verdict == expected


def test_an_unreadable_verdict_counts_as_a_hallucination():
    """The benefit of the doubt is the one thing this score must not give."""
    verdict = asyncio.run(
        classify_answer(
            _FakeHTTP("looks fine to me"), "http://localhost/v1", "", "m",
            "birth date?", "1996-08-02", "some answer",
        )
    )
    assert verdict == "hallucination"


def test_rates_are_reported_overall_and_per_question_type():
    rows = [
        {"verdict": "correct", "question_type": "Basic Fact Recall", "has_evidence": True},
        {"verdict": "hallucination", "question_type": "Memory Boundary", "has_evidence": False},
        {"verdict": "omission", "question_type": "Basic Fact Recall", "has_evidence": True},
        {"verdict": "correct", "question_type": "Memory Boundary", "has_evidence": False},
    ]

    summary = aggregate(rows)

    assert summary["correct_rate"] == 0.5
    assert summary["hallucination_rate"] == 0.25
    assert summary["omission_rate"] == 0.25
    assert summary["by_question_type"]["Memory Boundary"]["correct_rate"] == 0.5


def test_boundary_questions_get_their_own_hallucination_rate():
    rows = [
        {"verdict": "hallucination", "question_type": "Memory Boundary", "has_evidence": False},
        {"verdict": "correct", "question_type": "Memory Boundary", "has_evidence": False},
        {"verdict": "correct", "question_type": "Basic Fact Recall", "has_evidence": True},
    ]

    summary = aggregate(rows)

    assert summary["boundary_questions"] == 2
    assert summary["boundary_hallucination_rate"] == 0.5


def test_an_empty_run_scores_zero_rather_than_dividing_by_it():
    assert aggregate([])["hallucination_rate"] == 0.0


# ── the gate runs downward ───────────────────────────────────────────────────


def test_more_hallucination_fails_the_run():
    ok, message = _compare_downward(
        {"hallucination_rate": 0.30, "config_hash": "abc"},
        {"hallucination_rate": 0.20, "config_hash": "abc"}, 0.01,
    )
    assert ok is False
    assert "rose" in message


def test_less_hallucination_passes():
    ok, message = _compare_downward(
        {"hallucination_rate": 0.10, "config_hash": "abc"},
        {"hallucination_rate": 0.20, "config_hash": "abc"}, 0.01,
    )
    assert ok is True
    assert "-0.100" in message


def test_a_baseline_from_another_config_is_refused():
    ok, message = _compare_downward(
        {"hallucination_rate": 0.10, "config_hash": "abc"},
        {"hallucination_rate": 0.20, "config_hash": "other"}, 0.01,
    )
    assert ok is False
    assert "different config" in message


def test_an_unrecorded_baseline_passes_and_says_so():
    ok, message = _compare_downward(
        {"hallucination_rate": 0.10, "config_hash": "abc"}, {}, 0.01,
    )
    assert ok is True
    assert "--update-baseline" in message


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_a_missing_dataset_is_a_clear_failure(capsys):
    assert main([
        "--dataset", "/nonexistent/HaluMem-Medium.jsonl",
        "--answer-url", "http://localhost/v1", "--answer-model", "m",
    ]) == 2
    assert "IAAR-Shanghai/HaluMem" in capsys.readouterr().err


def test_running_without_an_answering_model_is_refused(dataset):
    """Hallucination is invisible in a candidate set — there is nothing to score."""
    with pytest.raises(SystemExit):
        main(["--dataset", dataset])
