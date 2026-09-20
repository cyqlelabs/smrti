"""The routing-gate evaluation: the labeled set is well-formed, the report
folds routes the way the acceptance criteria read them, and the harness
flag enters the config fingerprint."""
from __future__ import annotations

import argparse
import json
import os

import pytest

from bench.decisions.run import DEFAULT_SET, MUST_EXTRACT_KINDS, load_items, main, score_routes
from bench.harness import apply_run_modes, config_hash


def test_the_gate_set_is_labeled_and_bilingual():
    items = load_items(DEFAULT_SET)
    assert len(items) >= 30
    assert all(i["label"] in ("extract", "skip") for i in items)
    assert all(i["author"] in ("user", "assistant") for i in items)
    kinds = {i["kind"] for i in items}
    assert set(MUST_EXTRACT_KINDS) <= kinds and "chatter" in kinds and "echo" in kinds
    assert {i["lang"] for i in items} >= {"en", "es"}
    # constraints that name no entity are the case the entity count misses
    assert any(i["kind"] == "constraint" and not i["context"] for i in items)


def test_the_report_pairs_calls_avoided_with_missed_claims():
    items = [
        {"text": "thanks", "label": "skip", "kind": "chatter", "lang": "en"},
        {"text": "ok", "label": "skip", "kind": "chatter", "lang": "en"},
        {"text": "I moved", "label": "extract", "kind": "correction", "lang": "en"},
        {"text": "never deploy", "label": "extract", "kind": "constraint", "lang": "es"},
        {"text": "I like tea", "label": "extract", "kind": "durable", "lang": "en"},
    ]
    routes = [
        {"decision": "skip", "durable": 0.1, "corrects": 0.0, "novel": 0.1},
        {"decision": "default", "durable": 0.5, "corrects": 0.0, "novel": 0.5},
        {"decision": "llm", "durable": 0.2, "corrects": 0.9, "novel": 0.9},
        {"decision": "skip", "durable": 0.1, "corrects": 0.0, "novel": 0.1},   # the miss
        {"decision": None},                                                    # unavailable: not a skip
    ]
    report = score_routes(items, routes)
    assert report["calls_avoided"] == {"n": 2, "avoided": 1, "rate": 0.5}
    assert report["recall"]["correction"] == {"n": 1, "recall": 1.0, "forced": 1}
    assert report["recall"]["constraint"]["recall"] == 0.0
    assert report["recall"]["durable"]["recall"] == 1.0
    assert report["recall_by_language"]["es"]["recall"] == 0.0
    assert report["misses"][0]["text"] == "never deploy" and report["misses"][0]["durable"] == 0.1
    assert report["unavailable"] == 1


def test_replay_scores_a_saved_run_and_gates_on_recall(tmp_path, capsys):
    items = load_items(DEFAULT_SET)
    routes = [{"decision": "skip" if i["label"] == "skip" else "default"} for i in items]
    saved = tmp_path / "run.json"
    saved.write_text(json.dumps({"routes": routes, "model": "replay"}))
    assert main(["--replay", str(saved)]) == 0
    assert "calls avoided" in capsys.readouterr().out

    routes[next(k for k, i in enumerate(items) if i["kind"] == "constraint")] = {"decision": "skip"}
    saved.write_text(json.dumps({"routes": routes, "model": "replay"}))
    assert main(["--replay", str(saved)]) == 1
    assert "MISSED constraint" in capsys.readouterr().out


def test_a_short_replay_is_refused(tmp_path):
    saved = tmp_path / "run.json"
    saved.write_text(json.dumps({"routes": [{"decision": "skip"}]}))
    assert main(["--replay", str(saved)]) == 2


def test_the_decisions_mode_enters_the_fingerprint(monkeypatch):
    from smrti.decisions import get_decisions, reset_decisions

    monkeypatch.delenv("SMRTI_DECISIONS_RERANK", raising=False)
    base = {"dataset": "x", "top_k": 50}
    disabled = dict(base)
    apply_run_modes(argparse.Namespace(epochs=0, top_k=None, decisions="off"), disabled)
    assert disabled["decisions"] == "off"
    assert config_hash(disabled) != config_hash(base)
    assert get_decisions().policy.mode("rerank") == "off"

    defaulted = dict(base)
    apply_run_modes(argparse.Namespace(epochs=0, top_k=None), defaulted)
    assert defaulted["decisions"] == "active"
    assert get_decisions().policy.mode("rerank") == "active"

    shadowed = dict(base)
    apply_run_modes(argparse.Namespace(epochs=0, top_k=None, decisions="shadow"), shadowed)
    assert shadowed["decisions"] == "shadow"
    assert config_hash(shadowed) != config_hash(base)
    assert get_decisions().policy.mode("rerank") == "shadow"
    os.environ.pop("SMRTI_DECISIONS_RERANK", None)
    reset_decisions(None)
