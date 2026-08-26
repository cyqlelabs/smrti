"""Run the LongMemEval-S retrieval harness and compare it to the baseline.

    make bench DATASET=path/to/longmemeval_s.json

Not a CI gate — it needs a model download the test suite has no business
doing, and a judge key for the optional answering half. It is a required step
in the release checklist for any change that touches retrieval, which is why
it fails loudly rather than printing a number and exiting 0.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

from smrti import Smrti

from .adapter import aggregate, evaluate_question, ingest, load_questions
from .answering import answer_question, judge_answer

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "config.json")
BASELINE_PATH = os.path.join(_HERE, "baseline.json")

# The metric a release is gated on. Answer accuracy moves with whatever model
# is answering; this one moves only when retrieval does.
GATED_METRIC = "retrieval_hit_rate"


def config_hash(config: dict) -> str:
    """Fingerprint of the settings a measurement was taken under.

    A baseline recorded at top_k=10 says nothing about a run at top_k=50, and
    comparing them silently would report a ranking regression that is really a
    changed knob.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_baseline(path: str) -> dict:
    """The recorded baseline, or an empty one when there is no file yet.

    A baseline that has never been written reads the same as one that was
    written before anything was measured: nothing to compare against. Failing
    with a traceback instead would throw away the run that just finished.
    """
    try:
        return load_json(path)
    except FileNotFoundError:
        return {}


async def _score_answer(
    answering: dict, question, memories: list[str]
) -> tuple[str, bool]:
    """Generate an answer from the recalled memories and have it judged."""
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http:
        answer = await answer_question(
            http, answering["url"], answering["auth"], answering["model"],
            question.question, memories, question.question_date,
        )
        correct = await judge_answer(
            http, answering["url"], answering["auth"], answering["judge_model"],
            question.question, question.answer, answer,
        )
        return answer, correct


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
            top_k=config.get("top_k", 10),
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
            # The generated answer is kept beside the verdict: a score with no
            # answer under it cannot be argued with.
            row["answer"], row["answer_correct"] = asyncio.run(
                _score_answer(answering, question, memories)
            )
        rows.append(row)
        print(
            f"[{position}/{len(questions)}] {question.question_id} "
            f"turns={len(stored)} hit={rows[-1]['evidence_hit']}",
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


def compare(result: dict, baseline: dict, tolerance: float) -> tuple[bool, str]:
    """(ok, message) for this run against the recorded baseline."""
    recorded = baseline.get(GATED_METRIC)
    if recorded is None:
        return True, (
            f"no baseline recorded — this run measured {GATED_METRIC}="
            f"{result[GATED_METRIC]:.3f}; commit it with --update-baseline"
        )
    if baseline.get("config_hash") != result["config_hash"]:
        return False, (
            "baseline was recorded under a different config "
            f"({baseline.get('config_hash')} vs {result['config_hash']}); "
            "re-record it before reading anything into the numbers"
        )
    delta = result[GATED_METRIC] - recorded
    if delta < -tolerance:
        return False, (
            f"{GATED_METRIC} dropped {abs(delta):.3f} "
            f"({recorded:.3f} → {result[GATED_METRIC]:.3f}), "
            f"past the {tolerance:.3f} tolerance"
        )
    return True, (
        f"{GATED_METRIC} {recorded:.3f} → {result[GATED_METRIC]:.3f} "
        f"({delta:+.3f})"
    )


def write_baseline(result: dict, path: str) -> None:
    keep = (
        "config_hash", "recorded_at", "questions", "scored_questions",
        GATED_METRIC, "evidence_recall", "session_hit_rate",
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({key: result[key] for key in keep}, handle, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.longmemeval")
    parser.add_argument("--dataset", required=True, help="LongMemEval-S JSON file")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--baseline", default=BASELINE_PATH)
    parser.add_argument("--db", default=None, help="scratch DB (default: a temp file)")
    parser.add_argument("--limit", type=int, default=None, help="override question_limit")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--json", dest="json_out", default=None, help="write the full result here")
    # The answering half never gates a release: a model that answers well from
    # a bad candidate set would hide exactly the regression the gate is for.
    parser.add_argument("--answer-url", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:8421/v1")
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--judge-model", default=None, help="defaults to --answer-model")
    args = parser.parse_args(argv)

    if not os.path.exists(args.dataset):
        print(
            f"dataset not found: {args.dataset}\n"
            "LongMemEval-S is downloaded separately; point --dataset at the JSON file.",
            file=sys.stderr,
        )
        return 2

    config = load_json(args.config)
    if args.limit is not None:
        config["question_limit"] = args.limit
    # Popped before the fingerprint is taken: the tolerance decides how a
    # comparison is read, not what was measured, so tightening it must not
    # invalidate every baseline on file.
    tolerance = config.pop("tolerance", 0.01)

    db_path = args.db
    scratch = None
    if db_path is None:
        scratch = tempfile.TemporaryDirectory()
        db_path = os.path.join(scratch.name, "bench.db")
    answering = None
    if args.answer_url:
        if not args.answer_model:
            parser.error("--answer-url also needs --answer-model")
        answering = {
            "url": args.answer_url.rstrip("/"),
            "model": args.answer_model,
            "judge_model": args.judge_model or args.answer_model,
            # Read from the environment, never from a flag: a key in argv is a
            # key in the shell history and in every process listing.
            "auth": (
                f"Bearer {os.environ['SMRTI_BENCH_API_KEY']}"
                if os.environ.get("SMRTI_BENCH_API_KEY")
                else ""
            ),
        }

    try:
        result = run(config, args.dataset, db_path, answering)
    finally:
        if scratch is not None:
            scratch.cleanup()

    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    if args.update_baseline:
        write_baseline(result, args.baseline)
        print(f"baseline updated: {args.baseline}", file=sys.stderr)
        return 0

    ok, message = compare(result, load_baseline(args.baseline), tolerance)
    print(message, file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
