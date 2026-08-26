"""The parts every benchmark run shares: config fingerprint, baseline gate, CLI.

Three benchmarks measure three different things, but they all lock a config,
compare one headline metric against a recorded number, and refuse to compare
across configs. That belongs in one place — three copies would drift, and a
gate that behaves differently per benchmark is a gate nobody trusts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def config_hash(config: dict) -> str:
    """Fingerprint of the settings a measurement was taken under.

    A baseline recorded at top_k=10 says nothing about a run at top_k=50, and
    comparing them silently would report a regression that is really a changed
    knob.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_baseline(path: str) -> dict:
    """The recorded baseline, or an empty one when there is no file yet.

    A baseline that has never been written reads the same as one written
    before anything was measured: nothing to compare against. Failing with a
    traceback instead would throw away the run that just finished.
    """
    try:
        return load_json(path)
    except FileNotFoundError:
        return {}


def compare(
    result: dict, baseline: dict, tolerance: float, metric: str
) -> tuple[bool, str]:
    """(ok, message) for this run against the recorded baseline."""
    recorded = baseline.get(metric)
    if recorded is None:
        return True, (
            f"no baseline recorded — this run measured {metric}="
            f"{result[metric]:.3f}; commit it with --update-baseline"
        )
    if baseline.get("config_hash") != result["config_hash"]:
        return False, (
            "baseline was recorded under a different config "
            f"({baseline.get('config_hash')} vs {result['config_hash']}); "
            "re-record it before reading anything into the numbers"
        )
    delta = result[metric] - recorded
    if delta < -tolerance:
        return False, (
            f"{metric} dropped {abs(delta):.3f} "
            f"({recorded:.3f} → {result[metric]:.3f}), "
            f"past the {tolerance:.3f} tolerance"
        )
    return True, f"{metric} {recorded:.3f} → {result[metric]:.3f} ({delta:+.3f})"


def write_baseline(result: dict, path: str, keys: tuple[str, ...]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({key: result.get(key) for key in keys}, handle, indent=2)
        handle.write("\n")


def build_parser(prog: str, config_path: str, baseline_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--dataset", required=True, help="benchmark data file")
    parser.add_argument("--config", default=config_path)
    parser.add_argument("--baseline", default=baseline_path)
    parser.add_argument("--db", default=None, help="scratch DB (default: a temp file)")
    parser.add_argument("--limit", type=int, default=None, help="override the locked subset size")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--json", dest="json_out", default=None, help="write the full result here")
    # The answering half never gates a release: a model that answers well from
    # a bad candidate set would hide exactly the regression the gate is for.
    parser.add_argument("--answer-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--judge-model", default=None, help="defaults to --answer-model")
    parser.add_argument("--concurrency", type=int, default=8, help="answer/judge calls in flight")
    return parser


def resolve_answering(args, parser: argparse.ArgumentParser) -> dict | None:
    """The answering endpoint, or None when the run is retrieval-only."""
    if not args.answer_url:
        return None
    if not args.answer_model:
        parser.error("--answer-url also needs --answer-model")
    return {
        "url": args.answer_url.rstrip("/"),
        "model": args.answer_model,
        "judge_model": args.judge_model or args.answer_model,
        # Read from the environment, never from a flag: a key in argv is a key
        # in the shell history and in every process listing.
        "auth": (
            f"Bearer {os.environ['SMRTI_BENCH_API_KEY']}"
            if os.environ.get("SMRTI_BENCH_API_KEY")
            else ""
        ),
        "concurrency": max(1, args.concurrency),
    }


def require_dataset(path: str, hint: str) -> bool:
    if os.path.exists(path):
        return True
    print(f"dataset not found: {path}\n{hint}", file=sys.stderr)
    return False


def finish(args, result: dict, tolerance: float, metric: str, keys: tuple[str, ...]) -> int:
    """Print the run, record or compare the baseline, and return an exit code."""
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    if args.update_baseline:
        write_baseline(result, args.baseline, keys)
        print(f"baseline updated: {args.baseline}", file=sys.stderr)
        return 0

    ok, message = compare(result, load_baseline(args.baseline), tolerance, metric)
    print(message, file=sys.stderr)
    return 0 if ok else 1
