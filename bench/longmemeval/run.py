"""Run the LongMemEval-S retrieval harness and compare it to the baseline.

    make bench DATASET=path/to/longmemeval_s.json

Not a CI gate — it needs a model download the test suite has no business
doing, and a judge key for the optional answering half. It is a required step
in the release checklist for any change that touches retrieval, which is why
it fails loudly rather than printing a number and exiting 0.
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
    build_parser,
    config_hash,
    finish,
    load_json,
    require_dataset,
    resolve_answering,
)
from .adapter import aggregate, evaluate_question, ingest, load_questions

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "config.json")
BASELINE_PATH = os.path.join(_HERE, "baseline.json")

# The metric a release is gated on. Answer accuracy moves with whatever model
# is answering; this one moves only when retrieval does.
GATED_METRIC = "retrieval_hit_rate"
BASELINE_KEYS = (
    "config_hash", "recorded_at", "questions", "scored_questions",
    GATED_METRIC, "evidence_recall", "session_hit_rate",
)

DATASET_HINT = (
    "LongMemEval-S is downloaded separately; point --dataset at the JSON file."
)


def run(config: dict, dataset: str, db_path: str, answering: dict | None = None) -> dict:
    """Ingest every question's history and score its recall."""
    questions = load_questions(dataset, config.get("question_limit") or None)
    rows = []
    for position, question in enumerate(questions, start=1):
        # One space per question: another question's haystack is not this
        # question's distractor set, and the benchmark does not say it is.
        mem = Smrti(
            db_path=db_path,
            personality=config.get("personality", "deterministic"),
            tenant_id="longmemeval",
            write_space=f"q_{question.question_id or position}",
        )
        stored = ingest(question, mem)
        row = evaluate_question(
            question,
            mem,
            stored,
            top_k=config.get("top_k", 50),
            min_confidence=config.get("min_confidence", 0.0),
        )
        if answering:
            recalled = (
                mem.atomspace.get_atom(atom_id, mem.tenant_id, mem.write_space)
                for atom_id in row["returned_ids"]
            )
            # Each memory carries the day it was recorded. A memory store that
            # loses when something was said cannot answer when it happened,
            # and the benchmark asks that of a third of its questions.
            memories = [
                f"[{(atom.created_at or '')[:10]}] {atom.content or atom.label}"
                for atom in recalled
                if atom
            ]
            scored = asyncio.run(
                score_batch(
                    answering,
                    [{
                        "question": question.question,
                        "reference": question.answer,
                        "memories": memories,
                        "asked_on": question.question_date,
                    }],
                    concurrency=answering["concurrency"],
                )
            )[0]
            # The generated answer is kept beside the verdict: a score with no
            # answer under it cannot be argued with.
            row["answer"], row["answer_correct"] = scored["answer"], scored["verdict"]
        rows.append(row)
        print(
            f"[{position}/{len(questions)}] {question.question_id} "
            f"turns={len(stored)} hit={row['evidence_hit']}",
            file=sys.stderr,
        )
    judged = [r for r in rows if "answer_correct" in r]
    return {
        **aggregate(rows),
        "answer_accuracy": (
            sum(1 for r in judged if r["answer_correct"]) / len(judged)
            if judged
            else None
        ),
        "config_hash": config_hash(config),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("bench.longmemeval", CONFIG_PATH, BASELINE_PATH)
    args = parser.parse_args(argv)
    if not require_dataset(args.dataset, DATASET_HINT):
        return 2

    config = load_json(args.config)
    if args.limit is not None:
        config["question_limit"] = args.limit
    # Popped before the fingerprint is taken: the tolerance decides how a
    # comparison is read, not what was measured, so tightening it must not
    # invalidate every baseline on file.
    tolerance = config.pop("tolerance", 0.01)

    answering = resolve_answering(args, parser)
    db_path, scratch = args.db, None
    if db_path is None:
        scratch = tempfile.TemporaryDirectory()
        db_path = os.path.join(scratch.name, "bench.db")
    try:
        result = run(config, args.dataset, db_path, answering)
    finally:
        if scratch is not None:
            scratch.cleanup()

    return finish(args, result, tolerance, GATED_METRIC, BASELINE_KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
