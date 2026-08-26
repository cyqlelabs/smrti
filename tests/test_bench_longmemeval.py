"""Tests for the LongMemEval-S retrieval harness.

The dataset itself is downloaded separately and is far too large to ship, so
these run against a miniature haystack in the same schema. What is under test
is the harness — that gold evidence is identified by atom identity rather
than by string match, that dates survive ingestion, that the metrics are the
ones claimed, and that a regression against the baseline fails the run.
"""
from __future__ import annotations

import json
import sys

import pytest

from bench.longmemeval.adapter import (
    aggregate,
    evaluate_question,
    ingest,
    load_questions,
    parse_date,
    parse_question,
)
from bench.harness import compare, config_hash, load_baseline, write_baseline
from bench.longmemeval.run import BASELINE_KEYS, main
from smrti import Smrti


def _item(question_id="q1", **overrides) -> dict:
    item = {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "What is my daughter's name?",
        "answer": "Esmeralda",
        "question_date": "2023/05/20 (Sat) 02:36",
        "haystack_dates": ["2023/05/18 (Thu) 09:00", "2023/05/19 (Fri) 11:30"],
        "haystack_session_ids": ["answer_session", "noise_session"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "My daughter Esmeralda started school", "has_answer": True},
                {"role": "assistant", "content": "That's a big milestone", "has_answer": False},
            ],
            [
                {"role": "user", "content": "I rebuilt the deck last weekend", "has_answer": False},
                {"role": "assistant", "content": "Sounds like hard work", "has_answer": False},
            ],
        ],
        "answer_session_ids": ["answer_session"],
    }
    item.update(overrides)
    return item


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "mini.json"
    path.write_text(json.dumps([_item()]), encoding="utf-8")
    return str(path)


@pytest.fixture
def mem(tmp_path):
    return Smrti(
        db_path=str(tmp_path / "bench.db"),
        personality="deterministic",
        tenant_id="longmemeval",
        write_space="q1",
    )


# ── parsing ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2023/05/20 (Sat) 02:36", "2023-05-20 02:36:00"),
        ("2023/05/20", "2023-05-20 00:00:00"),
        ("2023-05-20 02:36:00", "2023-05-20 02:36:00"),
    ],
)
def test_session_dates_parse(raw, expected):
    assert str(parse_date(raw)) == expected


@pytest.mark.parametrize("raw", ["", None, "sometime last year"])
def test_an_unparseable_session_date_is_none(raw):
    assert parse_date(raw) is None


def test_every_turn_becomes_a_turn():
    question = parse_question(_item())
    assert len(question.turns) == 4


def test_the_gold_turns_are_the_ones_the_benchmark_marked():
    question = parse_question(_item())
    assert [t.content for t in question.gold_turns] == [
        "My daughter Esmeralda started school"
    ]


def test_turns_carry_their_session_id_and_date():
    question = parse_question(_item())
    first = question.turns[0]
    assert first.session_id == "answer_session"
    assert str(first.date) == "2023-05-18 09:00:00"


def test_empty_turns_are_skipped():
    item = _item()
    item["haystack_sessions"][0].append({"role": "user", "content": "   "})
    assert len(parse_question(item).turns) == 4


def test_a_session_with_no_id_or_date_still_parses():
    item = _item(haystack_session_ids=[], haystack_dates=[])
    question = parse_question(item)
    assert question.turns[0].session_id == "session_0"
    assert question.turns[0].date is None


def test_a_single_type_subset_is_the_front_of_the_file(tmp_path):
    path = tmp_path / "many.json"
    path.write_text(
        json.dumps([_item(f"q{n}") for n in range(5)]), encoding="utf-8"
    )

    assert [q.question_id for q in load_questions(str(path), limit=2)] == ["q0", "q1"]


def test_the_subset_covers_every_ability_the_benchmark_separates(tmp_path):
    """The file is grouped by type; the front of it is one ability, forty times."""
    path = tmp_path / "grouped.json"
    path.write_text(
        json.dumps(
            [_item(f"a{n}", question_type="single-session-user") for n in range(5)]
            + [_item(f"b{n}", question_type="multi-session") for n in range(5)]
            + [_item(f"c{n}", question_type="temporal-reasoning") for n in range(5)]
        ),
        encoding="utf-8",
    )

    picked = load_questions(str(path), limit=6)

    assert [q.question_id for q in picked] == ["a0", "b0", "c0", "a1", "b1", "c1"]


def test_the_subset_is_the_same_every_run(tmp_path):
    path = tmp_path / "grouped.json"
    path.write_text(
        json.dumps(
            [_item(f"a{n}", question_type="single-session-user") for n in range(4)]
            + [_item(f"b{n}", question_type="multi-session") for n in range(4)]
        ),
        encoding="utf-8",
    )

    first = [q.question_id for q in load_questions(str(path), limit=5)]
    assert first == [q.question_id for q in load_questions(str(path), limit=5)]


def test_a_type_that_runs_out_yields_its_slots_to_the_others(tmp_path):
    path = tmp_path / "lopsided.json"
    path.write_text(
        json.dumps(
            [_item("a0", question_type="single-session-user")]
            + [_item(f"b{n}", question_type="multi-session") for n in range(4)]
        ),
        encoding="utf-8",
    )

    assert [q.question_id for q in load_questions(str(path), limit=4)] == [
        "a0", "b0", "b1", "b2",
    ]


def test_no_limit_loads_everything(dataset):
    assert len(load_questions(dataset)) == 1


# ── ingestion ────────────────────────────────────────────────────────────────


def test_every_turn_is_stored_as_an_episode(mem, dataset):
    question = load_questions(dataset)[0]

    stored = ingest(question, mem)

    assert len(stored) == 4
    assert mem.status()["by_type"]["episode"] == 4


def test_the_session_date_becomes_the_atoms_write_time(mem, dataset):
    question = load_questions(dataset)[0]

    stored = ingest(question, mem)

    gold_id = next(a for a, t in stored.items() if t.has_answer)
    row = mem.db.fetchone("SELECT created_at FROM atoms WHERE id = ?", (gold_id,))
    assert row["created_at"] == "2023-05-18 09:00:00"


def test_assistant_turns_are_stored_as_agent_authored(mem, dataset):
    question = load_questions(dataset)[0]

    stored = ingest(question, mem)

    assistant_id = next(a for a, t in stored.items() if t.role == "assistant")
    atom = mem.atomspace.get_atom(assistant_id, "longmemeval", "q1")
    assert atom.metadata["source"] == "agent"


# ── scoring ──────────────────────────────────────────────────────────────────


def test_a_recall_that_returns_the_gold_turn_is_a_hit(mem, dataset):
    question = load_questions(dataset)[0]
    stored = ingest(question, mem)

    row = evaluate_question(question, mem, stored, top_k=10, min_confidence=0.0)

    assert row["evidence_hit"] is True
    assert row["evidence_recall"] == 1.0
    assert row["session_hit"] is True


def test_a_recall_that_returns_nothing_is_a_miss(mem, dataset):
    question = load_questions(dataset)[0]
    stored = ingest(question, mem)

    row = evaluate_question(question, mem, stored, top_k=0, min_confidence=0.0)

    assert row["evidence_hit"] is False
    assert row["evidence_recall"] == 0.0
    assert row["session_hit"] is False


def test_evidence_is_matched_by_atom_identity_not_by_text(mem, dataset):
    """A near-copy of the gold turn stored elsewhere is not the gold turn."""
    question = load_questions(dataset)[0]
    stored = ingest(question, mem)
    mem.remember("My daughter Esmeralda started school")

    row = evaluate_question(question, mem, stored, top_k=10, min_confidence=0.0)

    gold_ids = {a for a, t in stored.items() if t.has_answer}
    assert set(row["returned_ids"]) & gold_ids


def test_partial_evidence_scores_as_a_fraction():
    rows = [
        {"gold_turns": 4, "evidence_hit": True, "evidence_recall": 0.25, "session_hit": True},
    ]
    assert aggregate(rows)["evidence_recall"] == 0.25
    assert aggregate(rows)["retrieval_hit_rate"] == 1.0


def test_questions_with_no_marked_evidence_are_not_scored():
    rows = [
        {"gold_turns": 0, "evidence_hit": False, "evidence_recall": 0.0, "session_hit": False},
        {"gold_turns": 1, "evidence_hit": True, "evidence_recall": 1.0, "session_hit": True},
    ]
    summary = aggregate(rows)

    assert summary["questions"] == 2
    assert summary["scored_questions"] == 1
    assert summary["retrieval_hit_rate"] == 1.0


def test_an_empty_run_scores_zero_rather_than_dividing_by_it():
    assert aggregate([])["retrieval_hit_rate"] == 0.0


# ── the baseline gate ────────────────────────────────────────────────────────


def test_the_config_fingerprint_ignores_key_order():
    assert config_hash({"top_k": 10, "a": 1}) == config_hash({"a": 1, "top_k": 10})


def test_a_changed_knob_changes_the_fingerprint():
    assert config_hash({"top_k": 10}) != config_hash({"top_k": 50})


def _result(rate: float, config="abc") -> dict:
    return {"retrieval_hit_rate": rate, "config_hash": config}


def test_a_drop_past_the_tolerance_fails_the_run():
    ok, message = compare(_result(0.60), {"retrieval_hit_rate": 0.70, "config_hash": "abc"}, 0.01, "retrieval_hit_rate")

    assert ok is False
    assert "dropped" in message


def test_a_drop_inside_the_tolerance_passes():
    ok, _ = compare(_result(0.695), {"retrieval_hit_rate": 0.70, "config_hash": "abc"}, 0.01, "retrieval_hit_rate")
    assert ok is True


def test_an_improvement_passes():
    ok, message = compare(_result(0.80), {"retrieval_hit_rate": 0.70, "config_hash": "abc"}, 0.01, "retrieval_hit_rate")
    assert ok is True
    assert "+0.100" in message


def test_a_baseline_from_a_different_config_is_refused():
    ok, message = compare(
        _result(0.80), {"retrieval_hit_rate": 0.70, "config_hash": "other"}, 0.01,
        "retrieval_hit_rate",
    )
    assert ok is False
    assert "different config" in message


def test_an_unrecorded_baseline_passes_and_says_so():
    ok, message = compare(_result(0.80), {"retrieval_hit_rate": None}, 0.01, "retrieval_hit_rate")
    assert ok is True
    assert "--update-baseline" in message


def test_the_committed_baseline_names_the_config_it_was_measured_under():
    """A rate with no fingerprint is a number nobody can compare anything to."""
    from bench.longmemeval.run import BASELINE_PATH

    with open(BASELINE_PATH, encoding="utf-8") as handle:
        baseline = json.load(handle)

    if baseline["retrieval_hit_rate"] is not None:
        assert baseline["config_hash"]
        assert baseline["scored_questions"] > 0


def test_a_baseline_file_that_does_not_exist_yet_reads_as_unrecorded(tmp_path):
    """Failing here would throw away the run that just finished."""
    assert load_baseline(str(tmp_path / "never-written.json")) == {}


def test_a_run_against_a_missing_baseline_still_reports(tmp_path, dataset, capsys):
    code = main([
        "--dataset", dataset,
        "--baseline", str(tmp_path / "never-written.json"),
        "--db", str(tmp_path / "run.db"),
    ])

    assert code == 0
    assert "--update-baseline" in capsys.readouterr().err


def test_recording_a_baseline_keeps_only_the_summary(tmp_path):
    path = tmp_path / "baseline.json"
    write_baseline(
        {
            "config_hash": "abc", "recorded_at": "2026-08-25T19:00:00+00:00",
            "questions": 40, "scored_questions": 40, "retrieval_hit_rate": 0.7,
            "evidence_recall": 0.6, "session_hit_rate": 0.8, "rows": [{"big": "row"}],
        },
        str(path),
        BASELINE_KEYS,
    )

    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert "rows" not in recorded
    assert recorded["retrieval_hit_rate"] == 0.7


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_a_missing_dataset_is_a_clear_failure(capsys):
    assert main(["--dataset", "/nonexistent/longmemeval_s.json"]) == 2
    assert "downloaded separately" in capsys.readouterr().err


def test_the_harness_runs_end_to_end_and_records_a_baseline(tmp_path, dataset, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"retrieval_hit_rate": None}), encoding="utf-8")

    code = main([
        "--dataset", dataset,
        "--baseline", str(baseline),
        "--db", str(tmp_path / "run.db"),
        "--update-baseline",
    ])

    assert code == 0
    recorded = json.loads(baseline.read_text(encoding="utf-8"))
    assert recorded["retrieval_hit_rate"] == 1.0
    assert recorded["scored_questions"] == 1


def test_a_regression_against_the_recorded_baseline_exits_nonzero(tmp_path, dataset):
    baseline = tmp_path / "baseline.json"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"top_k": 0, "min_confidence": 0.0, "personality": "deterministic",
                    "question_limit": 1, "tolerance": 0.01}),
        encoding="utf-8",
    )
    # A baseline claiming perfect retrieval, under this exact config.
    from bench.longmemeval.run import load_json

    measured = dict(load_json(str(config)))
    measured.pop("tolerance")
    baseline.write_text(
        json.dumps({"retrieval_hit_rate": 1.0, "config_hash": config_hash(measured)}),
        encoding="utf-8",
    )

    code = main([
        "--dataset", dataset,
        "--config", str(config),
        "--baseline", str(baseline),
        "--db", str(tmp_path / "run.db"),
    ])

    assert code == 1


def test_the_full_result_can_be_written_out(tmp_path, dataset):
    out = tmp_path / "result.json"
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"retrieval_hit_rate": None}), encoding="utf-8")

    main([
        "--dataset", dataset,
        "--baseline", str(baseline),
        "--db", str(tmp_path / "run.db"),
        "--json", str(out),
    ])

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["rows"][0]["question_id"] == "q1"


def test_the_limit_flag_overrides_the_locked_subset(tmp_path, capsys):
    path = tmp_path / "many.json"
    path.write_text(json.dumps([_item(f"q{n}") for n in range(3)]), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"retrieval_hit_rate": None}), encoding="utf-8")

    main([
        "--dataset", str(path),
        "--baseline", str(baseline),
        "--db", str(tmp_path / "run.db"),
        "--limit", "2",
    ])

    assert json.loads(capsys.readouterr().out)["questions"] == 2


# ── the optional answering half ──────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeHTTP:
    def __init__(self, content: str) -> None:
        self._content = content
        self.payloads: list[dict] = []

    async def post(self, url, headers=None, json=None):
        self.payloads.append(json)
        return _FakeResponse(self._content)


def test_answering_sees_only_the_recalled_memories():
    import asyncio

    from bench.answering import answer_question

    http = _FakeHTTP("Esmeralda")
    answer = asyncio.run(
        answer_question(
            http, "http://localhost/v1", "", "m",
            "What is my daughter's name?", ["My daughter Esmeralda started school"],
        )
    )

    assert answer == "Esmeralda"
    assert "Esmeralda started school" in http.payloads[0]["messages"][1]["content"]


def test_the_answering_model_is_told_when_the_question_was_asked():
    """A third of the benchmark is unanswerable without a "now"."""
    import asyncio

    from bench.answering import answer_question

    http = _FakeHTTP("last month")
    asyncio.run(
        answer_question(
            http, "http://localhost/v1", "", "m", "How long ago did I move?",
            ["[2023-05-18] I moved to San Benito"], "2023/06/20 (Tue) 10:00",
        )
    )

    sent = http.payloads[0]["messages"][1]["content"]
    assert "[Question asked on]\n2023/06/20" in sent
    assert sent.index("[Question asked on]") < sent.index("[Memories]")


def test_a_run_without_a_question_date_omits_the_header():
    import asyncio

    from bench.answering import answer_question

    http = _FakeHTTP("Esmeralda")
    asyncio.run(
        answer_question(
            http, "http://localhost/v1", "", "m", "Who?", ["a memory"],
        )
    )

    assert "[Question asked on]" not in http.payloads[0]["messages"][1]["content"]


def test_the_answering_model_is_told_to_make_recommendations():
    """A rubric reference grades the suggestion, not a fact lookup."""
    from bench.answering import ANSWER_PROMPT, JUDGE_PROMPT

    assert "recommendation" in ANSWER_PROMPT
    assert "prefer" in JUDGE_PROMPT
    assert "declines to answer is never correct" in JUDGE_PROMPT


@pytest.mark.parametrize(
    "verdict,expected",
    [
        ('{"correct": true}', True),
        ('{"correct": false}', False),
        ('```json\n{"correct": true}\n```', True),
        ("it looks right to me", False),
    ],
)
def test_the_judge_reads_only_the_json_it_asked_for(verdict, expected):
    import asyncio

    from bench.answering import judge_answer

    assert (
        asyncio.run(
            judge_answer(
                _FakeHTTP(verdict), "http://localhost/v1", "", "m",
                "What is my daughter's name?", "Esmeralda", "Her name is Esmeralda",
            )
        )
        is expected
    )


def test_the_harness_scores_answers_when_a_model_is_named(tmp_path, dataset, capsys, monkeypatch):
    import bench.longmemeval.run as run_module

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"retrieval_hit_rate": None}), encoding="utf-8")
    asked: list[list[str]] = []

    async def _fake_score(answering, items, verdict="binary", concurrency=8):
        asked.extend(item["memories"] for item in items)
        return [{"answer": "Esmeralda", "verdict": True} for _ in items]

    monkeypatch.setattr(run_module, "score_batch", _fake_score)

    run_module.main([
        "--dataset", dataset,
        "--baseline", str(baseline),
        "--db", str(tmp_path / "run.db"),
        "--answer-url", "http://localhost:8421/v1",
        "--answer-model", "test-model",
    ])

    assert json.loads(capsys.readouterr().out)["answer_accuracy"] == 1.0
    # Every memory reaches the model stamped with the day it was recorded.
    assert asked and all(m.startswith("[2023-05-1") for m in asked[0])


def test_answer_accuracy_is_none_when_no_model_is_named(tmp_path, dataset, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"retrieval_hit_rate": None}), encoding="utf-8")

    main([
        "--dataset", dataset,
        "--baseline", str(baseline),
        "--db", str(tmp_path / "run.db"),
    ])

    assert json.loads(capsys.readouterr().out)["answer_accuracy"] is None


def test_naming_an_answer_url_without_a_model_is_refused(tmp_path, dataset):
    with pytest.raises(SystemExit):
        main([
            "--dataset", dataset,
            "--db", str(tmp_path / "run.db"),
            "--answer-url", "http://localhost:8421/v1",
        ])


def test_the_bench_package_is_importable_from_the_repo_root():
    assert "bench.longmemeval.adapter" in sys.modules
