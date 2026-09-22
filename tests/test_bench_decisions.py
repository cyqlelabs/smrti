"""The routing-gate evaluation: the labeled set is well-formed, the report
folds routes the way the acceptance criteria read them, and the harness
flag enters the config fingerprint."""
from __future__ import annotations

import argparse
import json
import os

import pytest

from bench.decisions import tone
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


# ── the tone set ─────────────────────────────────────────────────────────────


def test_the_tone_set_is_labeled_in_three_languages():
    items = tone.load_items(tone.DEFAULT_SET)
    assert len(items) >= 25
    assert all(i["label"] in (tone.KEEP, tone.DAMP) for i in items)
    assert all(i["author"] in ("user", "assistant") for i in items)
    kinds = {i["kind"] for i in items}
    assert {"failure", "rule", "preference", "apology", "thanks", "request"} <= kinds
    assert {i["lang"] for i in items} >= {"en", "es", "de"}
    # the estimate the check is asked about crosses every line a tone can
    assert tone.ESTIMATE < -0.7


def test_the_tone_report_pairs_damps_with_false_damps():
    items = [
        {"text": "sorry", "label": "speaker", "kind": "apology", "lang": "en"},
        {"text": "gracias", "label": "speaker", "kind": "thanks", "lang": "es"},
        {"text": "the deploy failed", "label": "thing", "kind": "failure", "lang": "en"},
        {"text": "never deploy", "label": "thing", "kind": "rule", "lang": "en"},
        {"text": "close it", "label": "speaker", "kind": "request", "lang": "en"},
    ]
    verdicts = [
        {"label": "speaker", "confidence": 0.6, "damped": True, "raw_label": "speaker"},
        {"label": "thing", "confidence": 0.1, "damped": False, "raw_label": "speaker"},   # under the line
        {"label": "speaker", "confidence": 0.7, "damped": True, "raw_label": "speaker"},  # the false damp
        {"label": "thing", "confidence": 0.3, "damped": False, "raw_label": "speaker"},   # the margin
        {"label": None},                                                                # unavailable
    ]
    report = tone.score_verdicts(items, verdicts)
    assert report["damp_rate"] == {"n": 3, "damped": 1, "rate": pytest.approx(1 / 3)}
    assert report["damped_by_kind"]["apology"] == {"n": 1, "damped": 1, "rate": 1.0}
    assert report["damped_by_kind"]["thanks"]["rate"] == 0.0
    assert report["damped_by_language"]["es"]["damped"] == 0
    assert report["false_damps"] == [{"text": "the deploy failed", "kind": "failure", "lang": "en", "confidence": 0.7}]
    assert report["worst_wrong_confidence"] == 0.7
    assert report["unavailable"] == 1


def test_tone_replay_scores_a_saved_run_and_fails_on_a_false_damp(tmp_path, capsys):
    items = tone.load_items(tone.DEFAULT_SET)
    clean = [{"label": i["label"], "confidence": 0.5, "damped": i["label"] == tone.DAMP, "raw_label": i["label"]}
             for i in items]
    saved = tmp_path / "run.json"
    saved.write_text(json.dumps({"verdicts": clean, "model": "replay"}))
    assert tone.main(["--replay", str(saved)]) == 0
    assert "damped:" in capsys.readouterr().out
    clean[[i["label"] for i in items].index(tone.KEEP)]["damped"] = True
    saved.write_text(json.dumps({"verdicts": clean, "model": "replay"}))
    assert tone.main(["--replay", str(saved)]) == 1
    saved.write_text(json.dumps({"verdicts": clean[:3], "model": "replay"}))
    assert tone.main(["--replay", str(saved)]) == 2
