"""Run the HaluMem question-answering harness and compare it to the baseline.

    make bench-halumem DATASET=path/to/HaluMem-Medium.jsonl

Unlike the other two harnesses this one needs an answering model to measure
anything at all: the whole benchmark is about what the system *says*, and
hallucination is not visible in a candidate set. The gated metric is the
hallucination rate, and it is gated downward — a rise fails the run.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone

from smrti import Smrti

from ..answering import score_batch
from ..harness import (
    apply_run_modes,
    build_parser,
    compare,
    config_hash,
    consolidate,
    load_baseline,
    load_json,
    measured_config,
    require_dataset,
    require_decisions,
    resolve_answering,
    resolve_extraction,
    write_baseline,
)
from .adapter import aggregate, ingest, load_users, recall_for, select_questions

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "config.json")
BASELINE_PATH = os.path.join(_HERE, "baseline.json")

GATED_METRIC = "hallucination_rate"
BASELINE_KEYS = (
    "config_hash", "recorded_at", "answer_model", "judge_model", "questions", "correct_rate",
    GATED_METRIC, "omission_rate", "boundary_hallucination_rate",
)
# The models that wrote and graded the answers. They decide the gated number
# as much as retrieval does, yet every HaluMem run names one, so they cannot
# join the fingerprint without refusing the baseline on file for every run:
# they are recorded beside it, and a baseline that names them gates only a
# run made with the same two.
ANSWERING_KEYS = ("answer_model", "judge_model")

DATASET_HINT = "Run `make datasets` to fetch it, or point --dataset at your own copy."


def run(config: dict, dataset: str, db_path: str, answering: dict,
        extraction: dict | None = None) -> dict:
    users = load_users(dataset, config.get("user_limit") or None)
    rows: list[dict] = []
    for position, user in enumerate(users, start=1):
        mem = Smrti(
            db_path=db_path,
            personality=config.get("personality", "deterministic"),
            tenant_id="halumem",
            write_space=f"u_{user.uuid or position}",
        )
        stored = ingest(user, mem, extraction)
        consolidate(mem, config.get("epochs", 0))
        questions = select_questions(user, config.get("questions_per_user") or None)
        items = [
            {
                "question": question.question,
                "reference": question.answer,
                "asked_on": question.asked_on,
                "memories": recall_for(
                    question, mem,
                    top_k=config.get("top_k", 50),
                    min_confidence=config.get("min_confidence", 0.0),
                    boost=config.get("sequential", False),
                ),
            }
            for question in questions
        ]
        scored = asyncio.run(
            score_batch(
                answering, items,
                verdict="three_way",
                concurrency=answering["concurrency"],
            )
        )
        for question, item, verdict in zip(questions, items, scored):
            rows.append({
                "user": user.uuid,
                "question": question.question,
                "reference": question.answer,
                "question_type": question.question_type,
                "difficulty": question.difficulty,
                "has_evidence": question.has_evidence,
                "retrieved": len(item["memories"]),
                "answer": verdict["answer"],
                "verdict": verdict["verdict"],
            })
        halluc = sum(1 for r in rows[-len(questions):] if r["verdict"] == "hallucination")
        print(
            f"[{position}/{len(users)}] {user.uuid} turns={stored} "
            f"questions={len(questions)} hallucinated={halluc}",
            file=sys.stderr,
        )

    return {
        **aggregate(rows),
        "config_hash": config_hash(measured_config(config, extraction)),
        "answer_model": answering.get("model"),
        "judge_model": answering.get("judge_model"),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }


def _compare_downward(result: dict, baseline: dict, tolerance: float) -> tuple[bool, str]:
    """Hallucination is the one metric where up is the regression."""
    recorded = baseline.get(GATED_METRIC)
    if recorded is None:
        return True, (
            f"no baseline recorded — this run measured {GATED_METRIC}="
            f"{result[GATED_METRIC]:.3f}; commit it with --update-baseline"
        )
    for key in ANSWERING_KEYS:
        if baseline.get(key) is not None and baseline[key] != result.get(key):
            return False, (
                f"baseline was answered with {key}={baseline[key]!r}, this run with "
                f"{result.get(key)!r}; the hallucination rates of two models are not one gate"
            )
    mirrored_result = {**result, GATED_METRIC: -result[GATED_METRIC]}
    mirrored_baseline = {**baseline, GATED_METRIC: -recorded}
    ok, _ = compare(mirrored_result, mirrored_baseline, tolerance, GATED_METRIC)
    delta = result[GATED_METRIC] - recorded
    if not ok and baseline.get("config_hash") != result["config_hash"]:
        return False, (
            "baseline was recorded under a different config "
            f"({baseline.get('config_hash')} vs {result['config_hash']}); "
            "re-record it before reading anything into the numbers"
        )
    if not ok:
        return False, (
            f"{GATED_METRIC} rose {delta:.3f} "
            f"({recorded:.3f} → {result[GATED_METRIC]:.3f}), "
            f"past the {tolerance:.3f} tolerance"
        )
    return True, f"{GATED_METRIC} {recorded:.3f} → {result[GATED_METRIC]:.3f} ({delta:+.3f})"


def main(argv: list[str] | None = None) -> int:
    import json as _json

    parser = build_parser("bench.halumem", CONFIG_PATH, BASELINE_PATH)
    args = parser.parse_args(argv)
    if not require_dataset(args.dataset, DATASET_HINT):
        return 2

    answering = resolve_answering(args, parser)
    if answering is None:
        parser.error(
            "HaluMem measures what the system says, so --answer-url and "
            "--answer-model are required"
        )

    config = load_json(args.config)
    if args.limit is not None:
        config["user_limit"] = args.limit
    extraction = resolve_extraction(args)
    config["extraction"] = bool(extraction)
    apply_run_modes(args, config)
    require_decisions(args.decisions)
    tolerance = config.pop("tolerance", 0.01)

    db_path, scratch = args.db, None
    if db_path is None:
        scratch = tempfile.TemporaryDirectory()
        db_path = os.path.join(scratch.name, "halumem.db")
    try:
        result = run(config, args.dataset, db_path, answering, extraction)
    finally:
        if scratch is not None:
            scratch.cleanup()

    print(_json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            _json.dump(result, handle, indent=2)
    if args.update_baseline:
        write_baseline(result, args.baseline, BASELINE_KEYS)
        print(f"baseline updated: {args.baseline}", file=sys.stderr)
        return 0

    ok, message = _compare_downward(result, load_baseline(args.baseline), tolerance)
    print(message, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
