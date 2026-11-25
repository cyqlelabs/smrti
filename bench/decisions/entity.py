"""The entity check against its labeled set.

    PYTHONPATH=src:. python -m bench.decisions.entity

A fuzzy or embedding match the resolver is unsure of goes to this check, and
only a confident choice of one offered candidate merges the mention into
it; anything else makes a provisional duplicate that a later mention can
still merge. A wrong merge cannot be undone — two people become one, and
every fact about either is filed under both — so the run fails on any. A
merge missed is reported beside it: without the check an uncertain true
match is never merged at all.

Every item goes through the same ``verify_entity`` the resolver calls, under
the same policy line, so the figures describe the check as deployed. The
provider is whatever the environment configures (``smrti.decisions``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(_HERE, "entity_set.json")

MERGE = "merge"


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(json.load(fh)["items"])


def score_verdicts(items: list[dict], verdicts: list[dict]) -> dict:
    """Fold per-item verdicts into the report.

    ``verdicts`` is one dict per item: ``{"chosen", "raw_label",
    "confidence"}`` where ``chosen`` is the index of the candidate merged
    into, or None; ``{"chosen": None, "raw_label": None}`` when the check
    could not answer, which merges nothing.
    """
    by_kind: dict[str, dict] = defaultdict(lambda: {"n": 0, "merged": 0, "right": 0})
    by_lang: dict[str, dict] = defaultdict(lambda: {"n": 0, "merged": 0, "right": 0})
    wrong = []
    for item, verdict in zip(items, verdicts):
        chosen = verdict.get("chosen")
        right = chosen is not None and chosen == item["gold"]
        for bucket in (by_kind[item["kind"]], by_lang[item.get("lang", "?")]):
            bucket["n"] += 1
            bucket["merged"] += int(chosen is not None)
            bucket["right"] += int(right)
        if chosen is not None and not right:
            wrong.append({"mention": item["mention"], "into": item["candidates"][chosen]["label"],
                          "lang": item.get("lang"), "confidence": verdict.get("confidence")})
    merges = by_kind.get(MERGE, {"n": 0, "right": 0})
    return {
        "items": len(items),
        "unavailable": sum(1 for v in verdicts if v.get("raw_label") is None),
        "true_merges": {"n": merges["n"], "merged": merges["right"]},
        "by_kind": dict(sorted(by_kind.items())),
        "by_language": dict(sorted(by_lang.items())),
        "wrong_merges": wrong,
    }


def verify_all(items: list[dict], engine) -> list[dict]:
    from smrti.decisions.extraction import verify_entity

    out = []
    for item in items:
        candidates = [{**c, "id": f"cand{i}"} for i, c in enumerate(item["candidates"])]
        verdict = verify_entity(
            engine, name=item["mention"], entity_type=item["type"], context=item["sentence"],
            candidates=candidates, tenant_id="bench", space="decisions",
        )
        if verdict is None:
            out.append({"chosen": None, "raw_label": None})
            continue
        chosen = None if verdict.atom_id is None else int(verdict.atom_id.removeprefix("cand"))
        out.append({"chosen": chosen, "raw_label": _raw_label(), "confidence": verdict.confidence})
    return out


def _raw_label() -> str | None:
    """The provider's own choice, before the confidence line — from the audit
    record the check just filed."""
    from smrti.decisions import audit

    records = audit.get_all()
    return (records[0].get("summary") or {}).get("raw_label") if records else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.decisions.entity")
    parser.add_argument("--set", dest="set_path", default=DEFAULT_SET, help="the labeled uncertain matches")
    parser.add_argument("--json", dest="json_out", default=None, help="write the full result here")
    args = parser.parse_args(argv)
    items = load_items(args.set_path)

    from smrti.decisions import DecisionPolicy, build_engine

    from ..harness import decisions_unavailable

    policy = DecisionPolicy.from_env().with_modes(entity="active")
    engine = build_engine(policy)
    reason = decisions_unavailable(engine)
    if reason is not None:
        print(f"the decision provider cannot answer: {reason}", file=sys.stderr)
        return 2
    if not engine.enabled("entity"):
        print(f"the {engine.provider_name} engine answers no entity question", file=sys.stderr)
        return 2
    verdicts = verify_all(items, engine)
    result = score_verdicts(items, verdicts)
    result["model"] = engine.model
    result["min_confidence"] = policy.entity_min_confidence
    result["verdicts"] = verdicts
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

    merges = result["true_merges"]
    print(f"model {engine.model}: {len(items)} uncertain matches at line {policy.entity_min_confidence}")
    print(f"merged {merges['merged']}/{merges['n']} true matches; {len(result['wrong_merges'])} wrong merges; "
          f"{result['unavailable']} unavailable")
    for kind, figures in result["by_kind"].items():
        print(f"  {kind:<9} merged {figures['merged']}/{figures['n']} ({figures['right']} right)")
    for lang, figures in result["by_language"].items():
        print(f"  [{lang}] merged {figures['merged']}/{figures['n']} ({figures['right']} right)")
    for item, verdict in zip(items, verdicts):
        print(f"  [{item['kind']:<8}] {verdict.get('raw_label')!s:<10} "
              f"{float(verdict.get('confidence') or 0.0):.2f} {item['mention']!r} in {item['sentence']!r}")
    for miss in result["wrong_merges"]:
        print(f"  WRONG {miss['mention']!r} into {miss['into']!r} [{miss.get('lang')}] at {miss['confidence']}")
    if result["wrong_merges"]:
        print("FAIL: an uncertain mention was merged into the wrong entity")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
