"""The extraction routing gate against its labeled set.

    make bench-decisions            # uses the core local Laya runtime

The gate exists to avoid LLM claim-extraction calls, and a gate that saves
calls by skipping valuable updates is a regression, so the two numbers are
reported together: the share of messages routed past the LLM, and the
recall of every kind of message that must not be — durable facts,
corrections, constraints — per language. The run fails when any recall
falls under ``--min-recall``.

Every message goes through the same ``route_extraction`` the pipeline
uses, under the same policy lines, so the figures describe the gate as
deployed and not a reimplementation of it. The provider is whatever the
environment configures (``smrti.decisions``); ``--replay`` scores a saved
result instead of calling it, for comparing threshold changes offline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(_HERE, "gate_set.json")

MUST_EXTRACT_KINDS = ("durable", "correction", "constraint")


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["items"])


def score_routes(items: list[dict], routes: list[dict]) -> dict:
    """Fold per-item routes into the report.

    ``routes`` is one dict per item: ``{"decision", "durable", "corrects",
    "novel"}`` (or ``{"decision": None}`` when the gate could not answer,
    which counts as the existing path — never as a skip).
    """
    total = len(items)
    skipped = sum(1 for r in routes if r.get("decision") == "skip")
    forced = sum(1 for r in routes if r.get("decision") == "llm")
    unavailable = sum(1 for r in routes if r.get("decision") is None)

    recall: dict[str, dict] = {}
    by_lang: dict[str, dict] = defaultdict(lambda: {"must_extract": 0, "extracted": 0})
    misses: list[dict] = []
    for kind in MUST_EXTRACT_KINDS:
        must = [(i, r) for i, r in zip(items, routes) if i["label"] == "extract" and i["kind"] == kind]
        kept = [(i, r) for i, r in must if r.get("decision") != "skip"]
        recall[kind] = {
            "n": len(must),
            "recall": (len(kept) / len(must)) if must else None,
            "forced": sum(1 for _, r in kept if r.get("decision") == "llm"),
        }
        for item, route in must:
            if route.get("decision") == "skip":
                misses.append({"text": item["text"], "kind": kind, "lang": item.get("lang"), **_scores(route)})
    for item, route in zip(items, routes):
        if item["label"] != "extract":
            continue
        bucket = by_lang[item.get("lang", "?")]
        bucket["must_extract"] += 1
        if route.get("decision") != "skip":
            bucket["extracted"] += 1
    skippable = [(i, r) for i, r in zip(items, routes) if i["label"] == "skip"]
    calls_avoided = sum(1 for _, r in skippable if r.get("decision") == "skip")
    false_skips = [
        {"text": i["text"], "kind": i["kind"], **_scores(r)}
        for i, r in zip(items, routes) if i["label"] == "extract" and r.get("decision") == "skip"
    ]
    return {
        "items": total,
        "skipped": skipped,
        "forced": forced,
        "unavailable": unavailable,
        "skip_rate": skipped / total if total else 0.0,
        "calls_avoided": {"n": len(skippable), "avoided": calls_avoided,
                          "rate": (calls_avoided / len(skippable)) if skippable else None},
        "recall": recall,
        "recall_by_language": {
            lang: {**b, "recall": (b["extracted"] / b["must_extract"]) if b["must_extract"] else None}
            for lang, b in sorted(by_lang.items())
        },
        "false_skips": false_skips,
        "misses": misses,
    }


def _scores(route: dict) -> dict:
    return {k: round(route[k], 3) for k in ("durable", "corrects", "novel") if isinstance(route.get(k), float)}


async def route_all(items: list[dict], engine, concurrency: int) -> list[dict]:
    from smrti.decisions.extraction import route_extraction

    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(item: dict) -> dict:
        async with gate:
            route = await route_extraction(
                engine, item["text"],
                source="agent" if item.get("author") == "assistant" else "user",
                entity_context=item.get("context") or "",
                tenant_id="bench", space="decisions",
            )
        if route is None:
            return {"decision": None}
        return {"decision": route.decision, "durable": route.durable,
                "corrects": route.corrects, "novel": route.novel}

    return await asyncio.gather(*(one(item) for item in items))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench.decisions.run")
    parser.add_argument("--set", dest="set_path", default=DEFAULT_SET, help="the labeled gate set")
    parser.add_argument("--json", dest="json_out", default=None, help="write the full result here")
    parser.add_argument("--replay", default=None, help="score the routes saved by an earlier --json run")
    parser.add_argument("--min-recall", type=float, default=0.95,
                        help="fail when any must-extract kind's recall is under this")
    parser.add_argument("--concurrency", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_items(args.set_path)

    if args.replay:
        with open(args.replay, encoding="utf-8") as fh:
            saved = json.load(fh)
        routes = saved["routes"]
        if len(routes) != len(items):
            print(f"replay holds {len(routes)} routes for {len(items)} items", file=sys.stderr)
            return 2
        model = saved.get("model", "")
    else:
        from smrti.decisions import DecisionPolicy, DecisionUnavailable, build_engine

        policy = DecisionPolicy.from_env().with_modes(routing="active")
        engine = build_engine(policy)
        if engine.provider is None:
            print("no decision provider configured", file=sys.stderr)
            return 2
        try:
            preload = getattr(engine.provider, "preload", None)
            if preload is not None:
                preload()
        except DecisionUnavailable as exc:
            print(f"could not load local decision model: {exc}", file=sys.stderr)
            return 2
        routes = asyncio.run(route_all(items, engine, args.concurrency))
        model = engine.model
        usage = getattr(engine.provider, "input_tokens", 0)
        print(f"model {model}: {len(items)} messages, {usage} input tokens")

    result = score_routes(items, routes)
    result["model"] = model
    result["routes"] = routes
    result["policy"] = {"skip": os.environ.get("SMRTI_DECISIONS_ROUTING_SKIP", "default"),
                        "force": os.environ.get("SMRTI_DECISIONS_ROUTING_FORCE", "default")}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

    avoided = result["calls_avoided"]
    print(f"calls avoided: {avoided['avoided']}/{avoided['n']} skippable messages "
          f"({(avoided['rate'] or 0.0):.0%}); {result['skipped']} skipped, {result['forced']} forced, "
          f"{result['unavailable']} unavailable of {result['items']}")
    failed = False
    for kind, figures in result["recall"].items():
        recall = figures["recall"]
        shown = "n/a" if recall is None else f"{recall:.0%}"
        print(f"  {kind:<11} recall {shown} over {figures['n']} ({figures['forced']} forced to the LLM)")
        if recall is not None and recall < args.min_recall:
            failed = True
    for lang, figures in result["recall_by_language"].items():
        recall = figures["recall"]
        print(f"  [{lang}] recall {'n/a' if recall is None else f'{recall:.0%}'} over {figures['must_extract']}")
    for miss in result["misses"]:
        print(f"  MISSED {miss['kind']} [{miss.get('lang')}]: {miss['text']!r} {miss}")
    if failed:
        print(f"FAIL: a must-extract kind fell under --min-recall {args.min_recall}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
