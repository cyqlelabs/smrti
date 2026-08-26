"""Run the LoCoMo retrieval harness and compare it to the baseline.

    make bench-locomo DATASET=path/to/locomo10.json

Ten conversations, each ingested into its own space and questioned through
``recall``. Retrieval hit rate is the gated metric; answering is optional and
gates nothing.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

from smrti import Smrti

from ..answering import score_batch
from ..harness import build_parser, config_hash, finish, load_json, require_dataset
from .adapter import aggregate, evaluate_question, ingest, load_conversations, select_questions

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "config.json")
BASELINE_PATH = os.path.join(_HERE, "baseline.json")

GATED_METRIC = "retrieval_hit_rate"
BASELINE_KEYS = (
    "config_hash", "recorded_at", "questions", "scored_questions",
    GATED_METRIC, "evidence_recall",
)

DATASET_HINT = (
    "LoCoMo is downloaded separately from snap-research/locomo; point "
    "--dataset at data/locomo10.json."
)


def run(config: dict, dataset: str, db_path: str, answering: dict | None = None) -> dict:
    import asyncio

    conversations = load_conversations(dataset, config.get("conversation_limit") or None)
    rows: list[dict] = []
    for position, conversation in enumerate(conversations, start=1):
        # One space per conversation: another conversation's sessions are not
        # this one's distractors, and the benchmark does not say they are.
        mem = Smrti(
            db_path=db_path,
            personality=config.get("personality", "deterministic"),
            tenant_id="locomo",
            write_space=f"c_{conversation.sample_id or position}",
        )
        stored = ingest(conversation, mem)
        questions = select_questions(
            conversation, config.get("questions_per_conversation") or None
        )
        conversation_rows = [
            evaluate_question(
                question, mem, stored,
                top_k=config.get("top_k", 50),
                min_confidence=config.get("min_confidence", 0.0),
            )
            for question in questions
        ]
        if answering:
            items = []
            for row in conversation_rows:
                recalled = (
                    mem.atomspace.get_atom(atom_id, mem.tenant_id, mem.write_space)
                    for atom_id in row["returned_ids"]
                )
                items.append({
                    "question": row["question"],
                    "reference": row["reference"],
                    "memories": [
                        f"[{(atom.created_at or '')[:10]}] {atom.content or atom.label}"
                        for atom in recalled
                        if atom
                    ],
                })
            for row, scored in zip(
                conversation_rows,
                asyncio.run(score_batch(answering, items, concurrency=answering["concurrency"])),
            ):
                row["answer"], row["answer_correct"] = scored["answer"], scored["verdict"]
        rows.extend(conversation_rows)
        hits = sum(1 for r in conversation_rows if r["evidence_hit"])
        print(
            f"[{position}/{len(conversations)}] {conversation.sample_id} "
            f"turns={len(stored)} questions={len(conversation_rows)} hits={hits}",
            file=sys.stderr,
        )

    judged = [r for r in rows if "answer_correct" in r]
    answered = [r for r in judged if not r["adversarial"]]
    refused = [r for r in judged if r["adversarial"]]
    return {
        **aggregate(rows),
        "answer_accuracy": (
            sum(1 for r in answered if r["answer_correct"]) / len(answered)
            if answered
            else None
        ),
        # Scored apart: here a correct response is a refusal, so folding it in
        # would average two opposite skills into one number.
        "adversarial_refusal_rate": (
            sum(1 for r in refused if r["answer_correct"]) / len(refused)
            if refused
            else None
        ),
        "config_hash": config_hash(config),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("bench.locomo", CONFIG_PATH, BASELINE_PATH)
    args = parser.parse_args(argv)
    if not require_dataset(args.dataset, DATASET_HINT):
        return 2

    config = load_json(args.config)
    if args.limit is not None:
        config["conversation_limit"] = args.limit
    tolerance = config.pop("tolerance", 0.01)

    from ..harness import resolve_answering

    answering = resolve_answering(args, parser)
    db_path, scratch = args.db, None
    if db_path is None:
        scratch = tempfile.TemporaryDirectory()
        db_path = os.path.join(scratch.name, "locomo.db")
    try:
        result = run(config, args.dataset, db_path, answering)
    finally:
        if scratch is not None:
            scratch.cleanup()

    return finish(args, result, tolerance, GATED_METRIC, BASELINE_KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
