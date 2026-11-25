"""The supersession check against its labeled set.

    PYTHONPATH=src:. python -m bench.decisions.supersession

The deterministic writer cuts the earlier claim whenever the extraction
model says a later one ``supersedes`` it, so every ``keep`` item in the set
is a current fact the writer destroys. The check earns its place by blocking
some of those, and loses it by allowing even one: a cut claim is gone, while
a blocked update only leaves both claims standing. So the run reports both —
the ``keep`` items it blocked, the ``allow`` items it let through — and fails
on any ``keep`` item allowed.

Every pair goes through the same ``verify_supersession`` the writer calls,
under the same policy line, so the figures describe the check as deployed.
The provider is whatever the environment configures (``smrti.decisions``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(_HERE, "supersession_set.json")

ALLOW = "allow"
KEEP = "keep"

# When the earlier claim was stated, as the writer passes it: the edge's
# ``created_at``. The set's pairs are about the present, so any past date does.
STATED_AT = "2026-01-15 09:30:00"


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(json.load(fh)["items"])


def score_verdicts(items: list[dict], verdicts: list[dict]) -> dict:
    """Fold per-item verdicts into the report.

    ``verdicts`` is one dict per item: ``{"label", "raw_label", "confidence",
    "allowed"}``, or ``{"label": None}`` when the check could not answer —
    which leaves the writer cutting as it always did, and counts as allowed.
    """
    by_label: dict[str, dict] = defaultdict(lambda: {"n": 0, "allowed": 0})
    by_kind: dict[str, dict] = defaultdict(lambda: {"n": 0, "allowed": 0})
    destructive = []
    for item, verdict in zip(items, verdicts):
        allowed = verdict.get("label") is None or bool(verdict.get("allowed"))
        for bucket in (by_label[item["label"]], by_kind[f"{item['label']}/{item['kind']}"]):
            bucket["n"] += 1
            bucket["allowed"] += int(allowed)
        if item["label"] == KEEP and allowed:
            destructive.append({"text": item["text"], "kind": item["kind"], "lang": item.get("lang"),
                                "raw_label": verdict.get("raw_label"), "confidence": verdict.get("confidence")})
    keep = by_label.get(KEEP, {"n": 0, "allowed": 0})
    allow = by_label.get(ALLOW, {"n": 0, "allowed": 0})
    return {
        "items": len(items),
        "unavailable": sum(1 for v in verdicts if v.get("label") is None),
        "blocked_keep": {"n": keep["n"], "blocked": keep["n"] - keep["allowed"]},
        "allowed_updates": {"n": allow["n"], "allowed": allow["allowed"]},
        "by_label": dict(sorted(by_label.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "destructive": destructive,
    }


def verify_all(items: list[dict], engine) -> list[dict]:
    from smrti.decisions.extraction import verify_supersession

    out = []
    for item in items:
        verdict = verify_supersession(
            engine, episode_text=item["text"], subject=item["subject"], predicate=item["predicate"],
            old_object=item["old"], new_object=item["new"], old_stated_at=STATED_AT,
            old_author="user", new_author="user", tenant_id="bench", space="decisions",
        )
        if verdict is None:
            out.append({"label": None})
            continue
        out.append({"label": verdict.label, "raw_label": _raw_label(), "confidence": verdict.confidence,
                    "allowed": verdict.allow})
    return out


def _raw_label() -> str | None:
    """The provider's own choice, before the confidence line — from the audit
    record the check just filed."""
    from smrti.decisions import audit

    records = audit.get_all()
    return (records[0].get("summary") or {}).get("raw_label") if records else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.decisions.supersession")
    parser.add_argument("--set", dest="set_path", default=DEFAULT_SET, help="the labeled claim pairs")
    parser.add_argument("--json", dest="json_out", default=None, help="write the full result here")
    args = parser.parse_args(argv)
    items = load_items(args.set_path)

    from smrti.decisions import DecisionPolicy, build_engine

    from ..harness import decisions_unavailable

    policy = DecisionPolicy.from_env().with_modes(supersession="active")
    engine = build_engine(policy)
    reason = decisions_unavailable(engine)
    if reason is not None:
        print(f"the decision provider cannot answer: {reason}", file=sys.stderr)
        return 2
    verdicts = verify_all(items, engine)
    result = score_verdicts(items, verdicts)
    result["model"] = engine.model
    result["min_confidence"] = policy.supersession_min_confidence
    result["verdicts"] = verdicts
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

    blocked, allowed = result["blocked_keep"], result["allowed_updates"]
    print(f"model {engine.model}: {len(items)} pairs at line {policy.supersession_min_confidence}")
    print(f"blocked {blocked['blocked']}/{blocked['n']} cuts the writer would make on a current fact; "
          f"allowed {allowed['allowed']}/{allowed['n']} real updates; {result['unavailable']} unavailable")
    for kind, figures in result["by_kind"].items():
        print(f"  {kind:<32} allowed {figures['allowed']}/{figures['n']}")
    for item, verdict in zip(items, verdicts):
        print(f"  [{item['label']:<9}] {verdict.get('raw_label')!s:<20} "
              f"{float(verdict.get('confidence') or 0.0):.2f} {item['text']}")
    for miss in result["destructive"]:
        print(f"  CUT {miss['kind']} [{miss.get('lang')}]: {miss['text']!r} as {miss['raw_label']} at {miss['confidence']}")
    if result["destructive"]:
        print("FAIL: a current fact would be cut")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
