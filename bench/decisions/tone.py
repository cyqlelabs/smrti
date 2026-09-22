"""The tone check against its labeled set.

    make bench-tone                 # uses the core local Laya runtime

The check exists to keep a speaker's mood from ranking and persisting like
the failure it is about, and a check that damps a real failure is worse than
none, so the two numbers are reported together: the share of speaker-toned
sentences damped at the policy line, per kind and per language, and every
thing-toned sentence that was damped. The run fails on any of the latter.

Every sentence goes through the same ``judge_tone`` the facade uses, under
the same policy lines and with the same grave estimate the sentiment model
gives such text, so the figures describe the check as deployed. The
provider is whatever the environment configures (``smrti.decisions``);
``--replay`` scores a saved result instead of calling it, for comparing
threshold changes offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(_HERE, "tone_set.json")

KEEP = "thing"
DAMP = "speaker"

# What the estimator reads off charged text: grave enough to cross every
# line the engine draws through a tone, which is what makes the verdict
# matter.
ESTIMATE = -0.9


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["items"])


def score_verdicts(items: list[dict], verdicts: list[dict]) -> dict:
    """Fold per-item verdicts into the report.

    ``verdicts`` is one dict per item: ``{"label", "confidence", "damped"}``
    (or ``{"label": None}`` when the check could not answer, which counts as
    the estimate kept — never as a damp).
    """
    total = len(items)
    damped = sum(1 for v in verdicts if v.get("damped"))
    unavailable = sum(1 for v in verdicts if v.get("label") is None)

    by_kind: dict[str, dict] = defaultdict(lambda: {"n": 0, "damped": 0})
    by_lang: dict[str, dict] = defaultdict(lambda: {"n": 0, "damped": 0})
    for item, verdict in zip(items, verdicts):
        if item["label"] != DAMP:
            continue
        for bucket in (by_kind[item["kind"]], by_lang[item.get("lang", "?")]):
            bucket["n"] += 1
            if verdict.get("damped"):
                bucket["damped"] += 1
    speaker = [(i, v) for i, v in zip(items, verdicts) if i["label"] == DAMP]
    false_damps = [
        {"text": i["text"], "kind": i["kind"], "lang": i.get("lang"), "confidence": v.get("confidence")}
        for i, v in zip(items, verdicts) if i["label"] == KEEP and v.get("damped")
    ]
    # the confidence the wrong direction ever reached: the margin under the line
    worst = max((float(v.get("confidence") or 0.0) for i, v in zip(items, verdicts)
                 if i["label"] == KEEP and v.get("raw_label") == DAMP), default=0.0)

    def _rate(b: dict) -> dict:
        return {**b, "rate": (b["damped"] / b["n"]) if b["n"] else None}

    return {
        "items": total,
        "damped": damped,
        "unavailable": unavailable,
        "damp_rate": {"n": len(speaker), "damped": sum(1 for _, v in speaker if v.get("damped")),
                      "rate": (sum(1 for _, v in speaker if v.get("damped")) / len(speaker)) if speaker else None},
        "damped_by_kind": {k: _rate(b) for k, b in sorted(by_kind.items())},
        "damped_by_language": {k: _rate(b) for k, b in sorted(by_lang.items())},
        "false_damps": false_damps,
        "worst_wrong_confidence": worst,
    }


def judge_all(items: list[dict], engine) -> list[dict]:
    from smrti.decisions.extraction import judge_tone

    out = []
    for item in items:
        verdict = judge_tone(
            engine, text=item["text"], estimate=ESTIMATE,
            source="agent" if item.get("author") == "assistant" else "user",
            kind="episode", tenant_id="bench", space="decisions",
        )
        if verdict is None:
            out.append({"label": None})
            continue
        out.append({"label": verdict.label, "confidence": verdict.confidence, "damped": verdict.damped,
                    "raw_label": _raw_label(engine)})
    return out


def _raw_label(engine) -> str | None:
    """The provider's own choice, before the confidence line — from the audit
    record the check just filed."""
    from smrti.decisions import audit

    records = audit.get_all()
    if not records:
        return None
    summary = records[0].get("summary") or {}
    return summary.get("raw_label")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench.decisions.tone")
    parser.add_argument("--set", dest="set_path", default=DEFAULT_SET, help="the labeled tone set")
    parser.add_argument("--json", dest="json_out", default=None, help="write the full result here")
    parser.add_argument("--replay", default=None, help="score the verdicts saved by an earlier --json run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    items = load_items(args.set_path)

    if args.replay:
        with open(args.replay, encoding="utf-8") as fh:
            saved = json.load(fh)
        verdicts = saved["verdicts"]
        if len(verdicts) != len(items):
            print(f"replay holds {len(verdicts)} verdicts for {len(items)} items", file=sys.stderr)
            return 2
        model = saved.get("model", "")
    else:
        from smrti.decisions import DecisionPolicy, DecisionUnavailable, build_engine

        policy = DecisionPolicy.from_env().with_modes(tone="active")
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
        verdicts = judge_all(items, engine)
        model = engine.model
        print(f"model {model}: {len(items)} sentences at line {policy.tone_min_confidence}")

    result = score_verdicts(items, verdicts)
    result["model"] = model
    result["verdicts"] = verdicts
    result["policy"] = {"min_confidence": os.environ.get("SMRTI_DECISIONS_TONE_MIN_CONFIDENCE", "default"),
                        "damping": os.environ.get("SMRTI_DECISIONS_TONE_DAMPING", "default")}
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

    rate = result["damp_rate"]
    print(f"damped: {rate['damped']}/{rate['n']} speaker-toned sentences ({(rate['rate'] or 0.0):.0%}); "
          f"{result['unavailable']} unavailable of {result['items']}")
    for kind, figures in result["damped_by_kind"].items():
        print(f"  {kind:<10} {figures['damped']}/{figures['n']}")
    for lang, figures in result["damped_by_language"].items():
        print(f"  [{lang}] {figures['damped']}/{figures['n']}")
    print(f"  worst confidence of a thing-toned sentence judged speaker: {result['worst_wrong_confidence']:.2f}")
    for miss in result["false_damps"]:
        print(f"  DAMPED {miss['kind']} [{miss.get('lang')}]: {miss['text']!r} at {miss['confidence']}")
    if result["false_damps"]:
        print("FAIL: a thing-toned sentence was damped")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
