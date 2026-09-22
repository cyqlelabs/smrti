"""The decisions package on its own: questions, the Laya adapter, policy,
engine, and the pure functions the integrations fold answers with."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from contextlib import contextmanager

import httpx
import pytest
from unittest.mock import patch

from smrti.decisions import (
    Choice,
    DecisionEngine,
    DecisionPolicy,
    DecisionUnavailable,
    Noul,
    Score,
    StaticProvider,
    build_engine,
    audit as _audit_module,
)
from smrti.decisions import audit
from smrti.decisions.engine import DecisionOutcome
from smrti.decisions.extraction import (
    COMPATIBLE,
    EXPLICIT_UPDATE,
    INSUFFICIENT_CONTEXT,
    ROUTE_DEFAULT,
    ROUTE_LLM,
    ROUTE_SKIP,
    decide_route,
    verify_entity,
    verify_supersession,
)
from smrti.decisions.laya import DEFAULT_MODEL, LayaProvider
from smrti.decisions.policies import MODE_ACTIVE, MODE_OFF, MODE_SHADOW, TASK_RERANK, TASKS
from smrti.decisions.provider import parse_response, questions_payload
from smrti.decisions.retrieval import fold_evidence, rerank
from smrti.core.models import Atom, AtomType, RecallResult


@pytest.fixture(autouse=True)
def clean_audit():
    audit.reset()
    yield
    audit.reset()


def run(coro):
    return asyncio.run(coro)


def _policy(**modes) -> DecisionPolicy:
    return DecisionPolicy(modes={task: MODE_OFF for task in TASKS}).with_modes(**modes)


def _yes(_key, question, _state):
    if isinstance(question, Noul):
        return {"type": "noul", "noul": 0.9}
    if isinstance(question, Choice):
        first = next(iter(question.criteria))
        return {"type": "choice", "choice": first, "probabilities": {first: 0.9}, "confidence": 0.9}
    return {"type": "score", "score": 1.0, "probabilities": {"1": 1.0}, "confidence": 0.9}


# ── questions and answers ────────────────────────────────────────────────────


def test_choice_and_score_enforce_the_documented_limits():
    with pytest.raises(ValueError):
        Choice("pick", {"only": "one"})
    with pytest.raises(ValueError):
        Score("rate", [str(i) for i in range(11)])
    Choice("pick", {"a": "x", "b": "y"})
    Score("rate", ["low", "high"])


def test_question_payloads_take_the_wire_shape():
    assert Noul("is it?", true="yes means", false="no means").payload() == {
        "type": "noul", "instructions": "is it?", "criteria": {"true": "yes means", "false": "no means"},
    }
    assert Noul("is it?").payload() == {"type": "noul", "instructions": "is it?"}
    assert Choice("pick", {"a": "x", "b": "y"}).payload()["criteria"] == {"a": "x", "b": "y"}
    assert Score("rate", ["low", "high"]).payload()["criteria"] == ["low", "high"]


def test_parse_response_reads_every_kind_and_the_usage():
    questions = {"n": Noul("n?"), "c": Choice("c?", {"a": "", "b": ""}), "s": Score("s?", ["lo", "mid", "hi"])}
    decisions = parse_response(
        {
            "model": "laya-test",
            "answers": {
                "n": {"type": "noul", "noul": 0.25},
                "c": {"type": "choice", "choice": "b", "probabilities": {"a": 0.3, "b": 0.7}, "confidence": 0.6},
                "s": {"type": "score", "score": 1.5, "probabilities": {"1": 0.5, "2": 0.5}, "confidence": 0.5,
                      "legend": {"0": "lo", "1": "mid", "2": "hi"}},
            },
            "usage": {"input_tokens": 120, "output_tokens": 0},
        },
        questions,
    )
    assert decisions.noul("n") == 0.25
    assert decisions.choice("c").choice == "b"
    assert decisions["s"].normalized == 0.75
    assert decisions.model == "laya-test"
    assert decisions.input_tokens == 120
    assert decisions.compact()["c"] == {"choice": "b", "confidence": 0.6}


@pytest.mark.parametrize(
    "answers",
    [
        {},                                                                       # missing
        {"n": {"type": "noul", "noul": 1.5}},                                     # out of range
        {"n": {"type": "choice", "choice": "a"}},                                 # wrong kind
        {"n": "0.5"},                                                             # not an object
    ],
)
def test_parse_response_refuses_an_answer_the_code_was_not_ready_for(answers):
    with pytest.raises(DecisionUnavailable):
        parse_response({"answers": answers}, {"n": Noul("n?")})


def test_a_choice_the_caller_never_offered_is_refused():
    with pytest.raises(DecisionUnavailable):
        parse_response(
            {"answers": {"c": {"type": "choice", "choice": "z", "probabilities": {}, "confidence": 1.0}}},
            {"c": Choice("c?", {"a": "", "b": ""})},
        )


def test_the_static_provider_validates_its_own_answers():
    provider = StaticProvider(lambda k, q, s: {"type": "noul", "noul": 7})
    with pytest.raises(DecisionUnavailable):
        provider.ask("state", {"n": Noul("n?")}, timeout=1.0)


# ── the Laya adapter ─────────────────────────────────────────────────────────


class _FakeLaya:
    def __init__(self, handler):
        self.handler = handler

    def predict(self, state, questions):
        return self.handler(state, questions)


def _reply(questions_body: dict) -> dict:
    answers = {}
    for key, q in questions_body.items():
        if q["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.8}
        elif q["type"] == "choice":
            first = next(iter(q["criteria"]))
            answers[key] = {
                "type": "choice",
                "choice": first,
                "probabilities": {first: 1.0},
                "confidence": 1.0,
            }
        else:
            answers[key] = {
                "type": "score",
                "score": 0.0,
                "probabilities": {"0": 1.0},
                "confidence": 1.0,
            }
    return {
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }


def _laya(handler, **kwargs) -> LayaProvider:
    return LayaProvider(agent=_FakeLaya(handler), **kwargs)


def test_laya_runs_locally_and_reads_every_answer_kind():
    calls = []

    def handler(state, questions):
        calls.append((state, questions))
        return _reply(questions)

    provider = _laya(handler, model="/models/laya-multilingual", device="cpu")
    questions = {
        "n": Noul("is it?"),
        "c": Choice("pick", {"a": "first", "b": "second"}),
        "s": Score("rate", ["low", "high"]),
    }
    decisions = provider.ask({"text": "hola"}, questions, timeout=2.0)

    # One question per native call — see MAX_QUESTIONS_PER_CALL. The state is
    # repeated with each, since every question is answered against all of it.
    assert len(calls) == 3
    assert [state for state, _ in calls] == [{"text": "hola"}] * 3
    assert all(len(body) == 1 for _, body in calls)
    sent = {key: body for _, asked in calls for key, body in asked.items()}
    assert sent["n"] == {"type": "noul", "instructions": "is it?"}
    assert sent["c"]["criteria"] == {"a": "first", "b": "second"}
    assert decisions.noul("n") == 0.8
    assert decisions.choice("c").choice == "a"
    assert decisions["s"].score == 0.0
    assert decisions.model == "/models/laya-multilingual"
    assert provider.device == "cpu"
    # Usage is summed over the calls one ``ask`` made, not read off the last.
    assert provider.input_tokens == 42 * 3
    assert provider.requests == 1
    provider.close()


def test_laya_stops_asking_once_the_deadline_has_passed():
    """A caller that has given up must not go on paying for the rest.

    Every question has to be answered for a reply to parse, so a run cut
    short is unavailable rather than partial — but the questions after the
    deadline are never asked, which is the point: each one reserves its own
    memory and holds the single worker for as long as it runs.
    """
    asked = []

    def handler(_state, questions):
        asked.extend(questions)
        time.sleep(0.05)
        return _reply(questions)

    provider = _laya(handler)
    questions = {f"q{i}": Noul("?") for i in range(20)}
    with pytest.raises(DecisionUnavailable):
        provider.ask("s", questions, timeout=0.12)
    assert 0 < len(asked) < 20  # it stopped rather than working through all
    provider.close()


def test_laya_refuses_rather_than_queues_behind_an_overrun_call():
    """Inference is serialized on one worker, and a running call cannot be
    cancelled, so queueing behind one that already outran the caller's
    deadline is a wait that cannot succeed."""
    started = threading.Event()
    release = threading.Event()

    def handler(_state, questions):
        started.set()
        release.wait(5.0)
        return _reply(questions)

    provider = _laya(handler)

    def overrun():
        try:
            provider.ask("s", {"q": Noul("?")}, timeout=0.05)
        except DecisionUnavailable:
            pass

    slow = threading.Thread(target=overrun, daemon=True)
    slow.start()
    assert started.wait(2.0)
    time.sleep(0.15)  # the running call is now well past a 0.05s deadline
    with pytest.raises(DecisionUnavailable, match="outran its deadline"):
        provider.ask("s", {"q2": Noul("?")}, timeout=0.05)
    release.set()
    slow.join(5.0)
    provider.close()


def test_laya_reports_model_failures_as_unavailable():
    def handler(_state, _questions):
        raise RuntimeError("weights unavailable")

    provider = _laya(handler)
    with pytest.raises(DecisionUnavailable, match="inference failed"):
        provider.ask("s", {"q": Noul("?")}, timeout=1.0)
    provider.close()


def test_laya_refuses_an_invalid_reply():
    provider = _laya(
        lambda _state, _questions: {
            "answers": {"q": {"type": "noul", "noul": "x"}},
        }
    )
    with pytest.raises(DecisionUnavailable):
        provider.ask("s", {"q": Noul("?")}, timeout=1.0)
    provider.close()


def test_laya_async_path_shares_the_contract():
    provider = _laya(lambda _state, questions: _reply(questions))
    decisions = run(provider.ask_async("s", {"q": Noul("?")}, timeout=1.0))
    assert decisions.noul("q") == 0.8
    run(provider.aclose())


def test_laya_enforces_the_decision_deadline():
    def handler(_state, questions):
        time.sleep(0.05)
        return _reply(questions)

    provider = _laya(handler)
    with pytest.raises(DecisionUnavailable, match="within 0.0s"):
        provider.ask("s", {"q": Noul("?")}, timeout=0.001)
    provider.close()


def test_a_provider_with_no_model_named_resolves_one_when_it_loads():
    # Where the weights are is a question for load time, not construction:
    # the engine has to exist on a box that has none so every decision site
    # can fall back through it.
    provider = LayaProvider("")
    try:
        assert provider.model == ""
        assert not provider.ready
    finally:
        provider.close()

# ── policy ───────────────────────────────────────────────────────────────────


def test_every_task_is_active_by_default():
    policy = DecisionPolicy.from_env({})
    assert all(policy.mode(t) == MODE_ACTIVE for t in TASKS)
    assert policy.any_enabled


def test_the_global_off_switch_restores_the_deterministic_path():
    policy = DecisionPolicy.from_env({"SMRTI_DECISIONS": "off"})
    assert all(policy.mode(t) == MODE_OFF for t in TASKS)
    assert not policy.any_enabled


def test_the_global_mode_sets_every_task_and_a_task_variable_overrides_it():
    policy = DecisionPolicy.from_env({"SMRTI_DECISIONS": "shadow", "SMRTI_DECISIONS_RERANK": "active"})
    assert policy.mode("routing") == MODE_SHADOW
    assert policy.mode(TASK_RERANK) == MODE_ACTIVE
    assert policy.active(TASK_RERANK) and not policy.active("routing")


def test_an_unknown_mode_is_refused_rather_than_read_as_off():
    with pytest.raises(ValueError):
        DecisionPolicy.from_env({"SMRTI_DECISIONS": "on"})


def test_thresholds_and_laya_settings_come_from_the_environment():
    policy = DecisionPolicy.from_env({
        "SMRTI_DECISIONS_MODEL": "/models/laya",
        "SMRTI_DECISIONS_DEVICE": "cpu",
        "SMRTI_DECISIONS_TIMEOUT": "2.5",
        "SMRTI_DECISIONS_RERANK_SHORTLIST": "30", "SMRTI_DECISIONS_ROUTING_SKIP": "0.1",
    })
    assert policy.model == "/models/laya"
    assert policy.device == "cpu"
    assert policy.timeout == 2.5
    assert policy.rerank_shortlist == 30
    assert policy.routing_skip == 0.1


# ── engine ───────────────────────────────────────────────────────────────────


def test_an_off_task_never_asks_the_provider():
    provider = StaticProvider(_yes)
    engine = DecisionEngine(_policy(), provider)
    assert engine.decide("routing", "s", {"q": Noul("?")}, tenant_id="t", space="s") is None
    assert provider.calls == []


def test_shadow_asks_and_marks_the_outcome_not_applied():
    provider = StaticProvider(_yes)
    engine = DecisionEngine(_policy(routing="shadow"), provider)
    outcome = engine.decide("routing", "s", {"q": Noul("?")}, tenant_id="t", space="s")
    assert isinstance(outcome, DecisionOutcome)
    assert outcome.decisions.noul("q") == 0.9
    assert not outcome.applied
    assert len(provider.calls) == 1


def test_the_cache_answers_a_repeated_state_without_the_provider():
    provider = StaticProvider(_yes)
    engine = DecisionEngine(_policy(rerank="active"), provider)
    first = engine.decide("rerank", {"v": 1}, {"q": Noul("?")}, tenant_id="t", space="s")
    second = engine.decide("rerank", {"v": 1}, {"q": Noul("?")}, tenant_id="t", space="s")
    third = engine.decide("rerank", {"v": 2}, {"q": Noul("?")}, tenant_id="t", space="s")
    assert first.applied and second.cached and not third.cached
    assert len(provider.calls) == 2
    engine.invalidate()
    engine.decide("rerank", {"v": 1}, {"q": Noul("?")}, tenant_id="t", space="s")
    assert len(provider.calls) == 3


def test_a_failing_provider_yields_no_decision_and_an_audit_record():
    def broken(_k, _q, _s):
        raise DecisionUnavailable("down")

    engine = DecisionEngine(_policy(routing="active"), StaticProvider(broken))
    assert engine.decide("routing", "s", {"q": Noul("?")}, tenant_id="t", space="sp") is None
    records = audit.get_all()
    assert records and records[0]["outcome"] == "unavailable" and records[0]["error"] == "down"
    assert not records[0]["applied"]
    assert audit.counters()[("routing", "active", "unavailable")] == 1


def test_a_provider_bug_is_contained_too():
    def buggy(_k, _q, _s):
        raise RuntimeError("oops")

    engine = DecisionEngine(_policy(routing="active"), StaticProvider(buggy))
    assert engine.decide("routing", "s", {"q": Noul("?")}, tenant_id="t", space="sp") is None
    assert audit.get_all()[0]["outcome"] == "unavailable"


def test_conclude_files_what_the_caller_did_with_the_answers():
    engine = DecisionEngine(_policy(routing="active"), StaticProvider(_yes))
    outcome = engine.decide("routing", "s", {"q": Noul("?")}, tenant_id="t", space="sp")
    engine.conclude("routing", outcome, tenant_id="t", space="sp", result="skip", applied=True,
                    summary={"text": "thanks"})
    record = audit.get_all()[0]
    assert record["outcome"] == "skip" and record["applied"] and record["answers"] == {"q": 0.9}
    assert record["summary"] == {"text": "thanks"}
    assert record["provider"] == "static"


def test_decisions_are_mirrored_into_the_llm_call_log():
    import smrti.call_log as call_log

    call_log._CALL_LOG.clear()
    engine = DecisionEngine(_policy(routing="active"), StaticProvider(_yes))
    engine.decide("routing", "s", {"q": Noul("?")}, tenant_id="t", space="sp")
    entries = call_log.get_all()
    assert entries and entries[0]["kind"] == "decision" and entries[0]["subkind"] == "routing"
    assert entries[0]["response_parsed"] == {"q": 0.9}
    call_log._CALL_LOG.clear()


def test_an_enabled_policy_builds_the_default_laya_provider():
    engine = build_engine(DecisionPolicy.from_env({"SMRTI_DECISIONS": "active"}))
    assert isinstance(engine.provider, LayaProvider)
    # The model is resolved when it is first loaded, not when the engine is
    # built: importing smrti must not go looking for 343 MB of weights, and
    # on a box that has none the engine still has to exist so every decision
    # site can fall back through it.
    assert engine.model == DEFAULT_MODEL == ""
    assert not engine.provider.ready
    assert engine.mode("rerank") == MODE_ACTIVE
    engine.provider.close()


def test_a_local_model_and_device_configure_the_laya_provider():
    engine = build_engine(DecisionPolicy.from_env({
        "SMRTI_DECISIONS": "shadow",
        "SMRTI_DECISIONS_MODEL": "/models/laya",
        "SMRTI_DECISIONS_DEVICE": "cpu",
    }))
    assert isinstance(engine.provider, LayaProvider)
    assert engine.model == "/models/laya"
    assert engine.provider.device == "cpu"
    assert engine.mode("entity") == MODE_SHADOW
    engine.provider.close()


def test_the_shared_engine_is_built_once_and_can_be_replaced(monkeypatch):
    from smrti.decisions import get_decisions, reset_decisions

    monkeypatch.delenv("SMRTI_DECISIONS", raising=False)
    reset_decisions(None)
    first = get_decisions()
    assert get_decisions() is first
    assert first.mode("rerank") == MODE_ACTIVE
    assert isinstance(first.provider, LayaProvider)
    closed = []
    monkeypatch.setattr(first.provider, "close", lambda: closed.append(True))
    replacement = DecisionEngine(_policy(), None)
    reset_decisions(replacement)
    assert closed == [True]
    assert get_decisions() is replacement
    reset_decisions(None)


# ── folding answers ──────────────────────────────────────────────────────────


def test_evidence_folds_direct_link_contradiction_and_history():
    assert fold_evidence(1.0, 0.0, 0.0, 0.0) == 1.0
    assert fold_evidence(0.0, 1.0, 0.0, 0.0) == pytest.approx(0.6)
    assert fold_evidence(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.9)
    # A historical state discounts, never cancels: for a history question it is the answer.
    assert fold_evidence(1.0, 0.0, 1.0, 0.0) == pytest.approx(0.6)
    assert fold_evidence(0.0, 0.0, 0.0, 0.0) == 0.0


def _result(atom_id: str, salience: float) -> RecallResult:
    atom = Atom(id=atom_id, type=AtomType.EPISODE, label=atom_id, content=atom_id)
    return RecallResult(atom=atom, salience=salience, similarity=0.5)


def test_rerank_blends_evidence_with_the_local_rank():
    results = [_result("a", 1.0), _result("b", 0.8), _result("c", 0.6)]
    # b is the judged answer; c was never judged and keeps its salience share.
    order = [r.atom.id for r in rerank(results, {"a": 0.1, "b": 1.0}, weight=0.5)]
    assert order == ["b", "a", "c"]
    # weight 0 is the local order
    assert [r.atom.id for r in rerank(results, {"a": 0.0, "b": 1.0}, weight=0.0)] == ["a", "b", "c"]
    assert rerank([], {}, 0.5) == []


def test_the_route_lines():
    assert decide_route(0.05, 0.05, 0.1, skip=0.2, force=0.75) == ROUTE_SKIP
    assert decide_route(0.9, 0.0, 0.5, skip=0.2, force=0.75) == ROUTE_LLM
    assert decide_route(0.0, 0.8, 0.0, skip=0.2, force=0.75) == ROUTE_LLM
    assert decide_route(0.5, 0.1, 0.5, skip=0.2, force=0.75) == ROUTE_DEFAULT
    # something new but neither durable nor a correction is not chatter, and not forced
    assert decide_route(0.1, 0.1, 0.9, skip=0.2, force=0.75) == ROUTE_DEFAULT


def _choice_provider(label: str, confidence: float) -> StaticProvider:
    def answer(_k, question, _s):
        assert isinstance(question, Choice)
        return {"type": "choice", "choice": label, "probabilities": {label: confidence}, "confidence": confidence}

    return StaticProvider(answer)


def _supersession_args():
    return dict(
        episode_text="I moved to Rosario last month", subject="Nico", predicate="lives_in",
        old_object="Córdoba", new_object="Rosario", old_stated_at="2026-01-01", old_author="user",
        new_author="user", tenant_id="t", space="s",
    )


def test_supersession_allows_an_update_and_keeps_both_on_compatible():
    engine = DecisionEngine(_policy(supersession="active"), _choice_provider(EXPLICIT_UPDATE, 0.9))
    verdict = verify_supersession(engine, **_supersession_args())
    assert verdict.allow and verdict.applied and verdict.label == EXPLICIT_UPDATE

    engine = DecisionEngine(_policy(supersession="active"), _choice_provider(COMPATIBLE, 0.9))
    verdict = verify_supersession(engine, **_supersession_args())
    assert not verdict.allow and verdict.label == COMPATIBLE


def test_a_supersession_verdict_under_the_confidence_line_is_insufficient_context():
    engine = DecisionEngine(_policy(supersession="active"), _choice_provider(EXPLICIT_UPDATE, 0.3))
    verdict = verify_supersession(engine, **_supersession_args())
    assert verdict.label == INSUFFICIENT_CONTEXT and not verdict.allow
    assert audit.get_all()[0]["summary"]["raw_label"] == EXPLICIT_UPDATE


def test_supersession_is_none_when_off():
    engine = DecisionEngine(_policy(), _choice_provider(EXPLICIT_UPDATE, 0.9))
    assert verify_supersession(engine, **_supersession_args()) is None


def _entity_args():
    return dict(
        name="Alex", entity_type="person", context="Alex from Acme called",
        candidates=[
            {"id": "id-a", "label": "Alex Rivera", "entity_type": "person", "facts": ["works_for Acme"]},
            {"id": "id-b", "label": "Alex Chen", "entity_type": "person", "facts": ["works_for Globex"]},
        ],
        tenant_id="t", space="s",
    )


def test_entity_verification_maps_a_reference_back_to_the_offered_id():
    engine = DecisionEngine(_policy(entity="active"), _choice_provider("c1", 0.9))
    verdict = verify_entity(engine, **_entity_args())
    assert verdict.matched and verdict.atom_id == "id-b" and verdict.label == "match"
    # the provider saw the facts, not the ids
    state = engine.provider.calls[0]["state"]
    assert state["candidates"][1]["facts"] == ["works_for Globex"]
    assert "id-b" not in json.dumps(state)


def test_entity_verification_prefers_a_duplicate_when_unsure():
    for label, confidence in (("none", 0.9), ("ambiguous", 0.9), ("c0", 0.2)):
        engine = DecisionEngine(_policy(entity="active"), _choice_provider(label, confidence))
        verdict = verify_entity(engine, **_entity_args())
        assert not verdict.matched, label
    assert verify_entity(DecisionEngine(_policy(), None), **_entity_args()) is None
    assert verify_entity(DecisionEngine(_policy(entity="active"), _choice_provider("none", 1.0)),
                         **{**_entity_args(), "candidates": []}) is None


def test_audit_records_are_newest_first_and_clear_keeps_the_counters():
    audit.record(task="a", mode="active", tenant_id="t", space="s", provider="p", model="m",
                 outcome="x", applied=True)
    audit.record(task="a", mode="active", tenant_id="t", space="s", provider="p", model="m",
                 outcome="y", applied=False)
    assert [r["outcome"] for r in audit.get_all()] == ["y", "x"]
    audit.clear()
    assert audit.get_all() == []
    assert audit.counters()[("a", "active", "x")] == 1
    assert _audit_module is audit


# ── the load never runs on a caller's thread ─────────────────────────────────
#
# A checkpoint load imports torch and may download 322M parameters. Paid on
# the caller, it sat behind the decision deadline and put a fixed wall in
# front of every recall — 30 seconds against a retrieval that takes 48ms,
# which is longer than any client waits, so the answer arrived after every
# one of them had given up. These pin that it is never paid there again.


class _SlowLoad(LayaProvider):
    """A provider whose checkpoint load is still running."""

    def __init__(self, gate: threading.Event, **kwargs):
        super().__init__(**kwargs)
        self.gate = gate
        self.loads = 0

    def _load(self):
        self.loads += 1
        self.gate.wait(30)
        return self._agent


class _FailingLoad(LayaProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loads = 0

    def _load(self):
        self.loads += 1
        raise DecisionUnavailable("no such checkpoint")


def test_a_decision_asked_before_the_model_is_loaded_does_not_wait_for_it():
    gate = threading.Event()
    provider = _SlowLoad(gate, model="convaiinnovations/laya-multilingual")
    try:
        started = time.monotonic()
        with pytest.raises(DecisionUnavailable, match="still loading"):
            # A 30s deadline: the point is that none of it is spent.
            provider.ask("s", {"q": Noul("?")}, timeout=30.0)
        assert time.monotonic() - started < 1.0
        assert not provider.ready
    finally:
        gate.set()
        provider.close()


def test_the_load_is_started_once_however_many_callers_arrive():
    gate = threading.Event()
    provider = _SlowLoad(gate, model="m")
    try:
        for _ in range(5):
            with pytest.raises(DecisionUnavailable):
                provider.ask("s", {"q": Noul("?")}, timeout=1.0)
        gate.set()
        provider.preload(timeout=5)
        assert provider.loads == 1
    finally:
        gate.set()
        provider.close()


def test_a_failed_load_backs_off_instead_of_being_retried_per_request():
    provider = _FailingLoad(model="m")
    try:
        for _ in range(4):
            with pytest.raises(DecisionUnavailable):
                provider.ask("s", {"q": Noul("?")}, timeout=1.0)
            # The loader thread has to finish before the next ask sees the error.
            deadline = time.monotonic() + 5
            while provider._loader is not None and time.monotonic() < deadline:
                time.sleep(0.01)
        # Each attempt imports torch before it can fail; one is enough.
        assert provider.loads == 1
    finally:
        provider.close()


def test_a_loaded_model_answers_as_before():
    provider = _laya(lambda state, questions: _reply(questions), model="m")
    assert provider.ready
    assert provider.preload() is True
    assert provider.ask("s", {"q": Noul("?")}, timeout=2.0).noul("q") == 0.8


def test_a_stalled_load_does_not_keep_the_process_from_exiting():
    # The loader is a daemon thread on purpose: a non-daemon executor worker
    # stuck in a download is joined by the interpreter's atexit hook, which
    # is how a stopped engine leaves a process that will not die.
    gate = threading.Event()
    provider = _SlowLoad(gate, model="m")
    try:
        with pytest.raises(DecisionUnavailable):
            provider.ask("s", {"q": Noul("?")}, timeout=1.0)
        deadline = time.monotonic() + 5
        while provider._loader is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert provider._loader is not None and provider._loader.daemon
    finally:
        gate.set()
        provider.close()


# ── the engine stops asking a provider that just failed ──────────────────────


class _Unavailable:
    name = "down"

    def __init__(self):
        self.calls = 0

    def ask(self, state, questions, *, timeout):
        self.calls += 1
        raise DecisionUnavailable("down")


def test_the_engine_pays_one_deadline_per_cooldown_not_one_per_request():
    provider = _Unavailable()
    engine = DecisionEngine(DecisionPolicy(cooldown=60.0), provider)
    for i in range(5):
        # A distinct state each time, so the answer cache cannot be what
        # spares the provider.
        assert engine.decide(
            TASK_RERANK, {"q": i}, {"n": Noul("?")}, tenant_id="t", space="s"
        ) is None
    assert provider.calls == 1


def test_the_cooldown_lifts_and_a_working_provider_clears_it():
    provider = _Unavailable()
    engine = DecisionEngine(DecisionPolicy(cooldown=60.0), provider)
    assert engine.decide(TASK_RERANK, {"q": 0}, {"n": Noul("?")}, tenant_id="t", space="s") is None
    assert engine._offline()

    engine._retry_at = time.monotonic() - 1  # the window has passed
    assert not engine._offline()
    engine.provider = _FakeProviderThatAnswers()
    assert engine.decide(TASK_RERANK, {"q": 1}, {"n": Noul("?")}, tenant_id="t", space="s") is not None
    assert not engine._offline()


class _FakeProviderThatAnswers:
    name = "ok"

    def ask(self, state, questions, *, timeout):
        from smrti.decisions.provider import parse_response
        return parse_response(_reply(questions_payload(questions)), questions)


def test_a_zero_cooldown_restores_asking_every_time():
    provider = _Unavailable()
    engine = DecisionEngine(DecisionPolicy(cooldown=0.0), provider)
    for i in range(3):
        engine.decide(TASK_RERANK, {"q": i}, {"n": Noul("?")}, tenant_id="t", space="s")
    assert provider.calls == 3


def test_recall_keeps_its_local_ranking_and_its_speed_when_the_model_is_loading():
    from smrti.core.models import Atom, AtomType, AttentionValue, RecallResult, TruthValue, Valence
    from smrti.decisions.retrieval import make_judge

    gate = threading.Event()
    provider = _SlowLoad(gate, model="m")
    engine = DecisionEngine(DecisionPolicy(), provider)
    judge = make_judge(engine, "default", "main")
    assert judge is not None  # rerank is active by default

    results = [
        RecallResult(
            atom=Atom(
                type=AtomType.EPISODE, label=f"a{i}", content=f"m{i}",
                tenant_id="default", space="main",
                truth=TruthValue(probability=0.8, confidence=0.5),
                attention=AttentionValue(sti=0.1, lti=0.1),
                valence=Valence(valence=0.0, intensity=0.0),
            ),
            salience=1.0 - i * 0.01, similarity=0.5,
        )
        for i in range(30)
    ]
    try:
        started = time.monotonic()
        got = judge("where do we deploy", results)
        assert time.monotonic() - started < 1.0
        assert [r.atom.id for r in got] == [r.atom.id for r in results]
    finally:
        gate.set()
        provider.close()


# ── the real load path ───────────────────────────────────────────────────────
#
# Every test above replaces `_load`, which left the one function the fix is
# about — the import and the checkpoint load — without a line of coverage.
# These drive the real one against a stand-in `laya` module.


@contextmanager
def _fake_laya(loader):
    """Install a module named `edgejev` whose `Agent` is *loader*, then remove
    it. The runtime is EdgeJev; the model it serves is still Laya."""
    module = types.ModuleType("edgejev")
    module.Agent = loader
    previous = sys.modules.get("edgejev")
    sys.modules["edgejev"] = module
    try:
        yield module
    finally:
        if previous is None:
            sys.modules.pop("edgejev", None)
        else:
            sys.modules["edgejev"] = previous


def test_the_real_load_imports_the_runtime_and_serves_the_next_decision():
    seen = {}

    def loader(model_dir, threads=None, provider=None):
        seen["model"], seen["provider"] = model_dir, provider
        return _FakeLaya(lambda state, questions: _reply(questions))

    provider = LayaProvider(model="checkpoint-x", device="cpu")
    try:
        with _fake_laya(loader):
            # The first caller is turned away rather than made to wait...
            with pytest.raises(DecisionUnavailable):
                provider.ask("s", {"q": Noul("?")}, timeout=1.0)
            assert provider.preload(timeout=5) is True
        assert seen == {"model": "checkpoint-x", "provider": "cpu"}
        assert provider.ready
        # ...and the one after the load lands gets the model.
        assert provider.ask("s", {"q": Noul("?")}, timeout=2.0).noul("q") == 0.8
    finally:
        provider.close()


def test_a_missing_runtime_is_reported_as_unavailable():
    provider = LayaProvider(model="m")
    previous = sys.modules.get("edgejev")
    sys.modules["edgejev"] = None  # an import of this name now raises ImportError
    try:
        with pytest.raises(DecisionUnavailable, match="missing from the Smrti installation"):
            provider._load()
    finally:
        if previous is None:
            sys.modules.pop("edgejev", None)
        else:
            sys.modules["edgejev"] = previous
        provider.close()


def test_a_checkpoint_that_will_not_load_is_reported_with_its_model():
    def loader(model_dir, threads=None, provider=None):
        raise RuntimeError("no such directory")

    provider = LayaProvider(model="bad/checkpoint")
    try:
        with _fake_laya(loader):
            with pytest.raises(DecisionUnavailable, match="could not load the decision model at 'bad/checkpoint'"):
                provider._load()
            assert not provider.ready
    finally:
        provider.close()


def test_an_already_loaded_agent_short_circuits_the_load():
    agent = _FakeLaya(lambda state, questions: _reply(questions))
    provider = LayaProvider(agent=agent, model="m")
    try:
        # No `laya` module installed: reaching the import would raise.
        assert provider._load() is agent
    finally:
        provider.close()


# ── the deadline, and the guards around one call ─────────────────────────────


def test_the_deadline_defaults_rejects_nonsense_and_refuses_an_expired_one():
    provider = _laya(lambda state, questions: _reply(questions))
    try:
        assert provider._deadline(None) == 30.0
        assert provider._deadline(2) == 2.0
        with pytest.raises(ValueError, match="must be a number"):
            provider._deadline("soon")
        with pytest.raises(DecisionUnavailable, match="already expired"):
            provider._deadline(0)
        with pytest.raises(DecisionUnavailable, match="already expired"):
            provider._deadline(-1)
    finally:
        provider.close()


def test_a_decision_with_no_questions_is_a_programming_error():
    provider = _laya(lambda state, questions: _reply(questions))
    try:
        with pytest.raises(ValueError, match="at least one question"):
            provider._predict("s", {})
    finally:
        provider.close()


def test_the_async_path_bounds_a_model_that_does_not_answer():
    started = threading.Event()
    release = threading.Event()

    def handler(state, questions):
        started.set()
        release.wait(30)
        return _reply(questions)

    provider = _laya(handler)
    try:
        with pytest.raises(DecisionUnavailable, match="within 0.0s"):
            run(provider.ask_async("s", {"q": Noul("?")}, timeout=0.01))
    finally:
        release.set()
        provider.close()


# ── the cooldown holds on the async door too ─────────────────────────────────


class _UnavailableAsync:
    name = "down"

    def __init__(self):
        self.calls = 0

    async def ask_async(self, state, questions, *, timeout):
        self.calls += 1
        raise DecisionUnavailable("down")


class _AnswersAsync:
    name = "ok"

    async def ask_async(self, state, questions, *, timeout):
        return parse_response(_reply(questions_payload(questions)), questions)


def test_decide_async_pays_one_deadline_per_cooldown():
    provider = _UnavailableAsync()
    engine = DecisionEngine(DecisionPolicy(cooldown=60.0), provider)

    async def drive():
        for i in range(4):
            assert await engine.decide_async(
                TASK_RERANK, {"q": i}, {"n": Noul("?")}, tenant_id="t", space="s"
            ) is None

    run(drive())
    assert provider.calls == 1
    assert engine._offline()


def test_decide_async_clears_the_cooldown_when_the_provider_answers():
    engine = DecisionEngine(DecisionPolicy(cooldown=60.0), _AnswersAsync())
    outcome = run(
        engine.decide_async(TASK_RERANK, {"q": 0}, {"n": Noul("?")}, tenant_id="t", space="s")
    )
    assert outcome is not None and not engine._offline()


def test_predict_passes_the_unready_signal_through_unchanged():
    # `_predict` re-raises DecisionUnavailable rather than wrapping it as an
    # inference failure: "the model is not loaded yet" is not a bad answer.
    provider = LayaProvider(model="m")
    try:
        # Whether the background load is still running or has already failed
        # decides the wording, so assert on what must never change: the signal
        # arrives as-is and not wrapped as an inference failure.
        with pytest.raises(DecisionUnavailable) as raised:
            provider._predict("s", {"q": Noul("?")})
        assert "inference failed" not in str(raised.value)
        # Leave nothing in flight: a loader thread still running when the
        # interpreter exits is what aborts the process.
        deadline = time.monotonic() + 5
        while provider._loader is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert provider._loader is None
    finally:
        provider.close()


def test_decide_async_is_off_and_cached_on_the_same_terms_as_decide():
    engine = DecisionEngine(_policy(rerank=MODE_OFF), _AnswersAsync())
    assert run(
        engine.decide_async(TASK_RERANK, {"q": 0}, {"n": Noul("?")}, tenant_id="t", space="s")
    ) is None

    engine = DecisionEngine(DecisionPolicy(), _AnswersAsync())
    first = run(engine.decide_async(TASK_RERANK, {"q": 1}, {"n": Noul("?")}, tenant_id="t", space="s"))
    second = run(engine.decide_async(TASK_RERANK, {"q": 1}, {"n": Noul("?")}, tenant_id="t", space="s"))
    assert first is not None and not first.cached
    assert second is not None and second.cached


# ── the validation boundary, refusing every shape it is meant to refuse ──────


def test_a_malformed_environment_names_the_variable_it_could_not_read():
    with pytest.raises(ValueError, match="SMRTI_DECISIONS_TIMEOUT must be a number"):
        DecisionPolicy.from_env({"SMRTI_DECISIONS_TIMEOUT": "soon"})
    with pytest.raises(ValueError, match="SMRTI_DECISIONS_COOLDOWN must be a number"):
        DecisionPolicy.from_env({"SMRTI_DECISIONS_COOLDOWN": "a while"})
    with pytest.raises(ValueError, match="SMRTI_DECISIONS_CACHE must be an integer"):
        DecisionPolicy.from_env({"SMRTI_DECISIONS_CACHE": "lots"})
    with pytest.raises(ValueError, match="SMRTI_DECISIONS_RERANK_SHORTLIST must be an integer"):
        DecisionPolicy.from_env({"SMRTI_DECISIONS_RERANK_SHORTLIST": "twenty"})


def test_with_modes_refuses_a_task_that_does_not_exist():
    with pytest.raises(ValueError, match="unknown decision task 'reranking'"):
        DecisionPolicy().with_modes(reranking=MODE_OFF)


def test_every_answer_reports_its_own_value():
    questions = {"n": Noul("?"), "c": Choice("pick", {"a": "first", "b": "second"}), "s": Score("rate", ["low", "high"])}
    decisions = parse_response(
        {
            "answers": {
                "n": {"type": "noul", "noul": 0.25},
                "c": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 0.9},
                "s": {"type": "score", "score": 1.0, "probabilities": {"1": 1.0}, "confidence": 0.8},
            }
        },
        questions,
    )
    assert decisions["n"].value == 0.25
    assert decisions["c"].value == "a"
    assert decisions["s"].value == 1.0


@pytest.mark.parametrize(
    "question, raw, message",
    [
        (Choice("pick", {"a": "first", "b": "second"}), {"type": "noul", "choice": "a"}, "expected choice"),
        (Score("rate", ["low", "high"]), {"type": "noul", "score": 0}, "expected score"),
        (Score("rate", ["low", "high"]), {"type": "score", "score": 7}, "outside the rubric"),
        (Score("rate", ["low", "high"]), {"type": "score", "score": -1}, "outside the rubric"),
        (Noul("?"), {"type": "noul"}, "is not readable"),
        (Choice("pick", {"a": "first", "b": "second"}), {"type": "choice"}, "is not readable"),
    ],
)
def test_an_answer_of_the_wrong_shape_is_refused(question, raw, message):
    with pytest.raises(DecisionUnavailable, match=message):
        parse_response({"answers": {"q": raw}}, {"q": question})


@pytest.mark.parametrize(
    "data, message",
    [
        ("not an object", "reply is not an object"),
        ({"usage": {}}, "carries no answers object"),
        ({"answers": "nope"}, "carries no answers object"),
    ],
)
def test_a_reply_that_is_not_a_reply_is_refused(data, message):
    with pytest.raises(DecisionUnavailable, match=message):
        parse_response(data, {"q": Noul("?")})


def test_unreadable_usage_counts_as_none_rather_than_failing_the_decision():
    questions = {"q": Noul("?")}
    answers = {"answers": {"q": {"type": "noul", "noul": 0.5}}}

    # usage of the wrong type entirely, and usage whose numbers are not numbers
    for usage in ("plenty", {"input_tokens": "many", "output_tokens": None}):
        decisions = parse_response({**answers, "usage": usage}, questions)
        assert decisions.input_tokens == 0 and decisions.output_tokens == 0


# ── the engine's cache, and its refusal to fail over its own logging ─────────


def test_the_cache_can_be_switched_off_entirely():
    provider = StaticProvider(lambda k, q, s: {"type": "noul", "noul": 0.5})
    engine = DecisionEngine(DecisionPolicy(cache_size=0), provider)
    first = engine.decide(TASK_RERANK, {"q": 1}, {"n": Noul("?")}, tenant_id="t", space="s")
    second = engine.decide(TASK_RERANK, {"q": 1}, {"n": Noul("?")}, tenant_id="t", space="s")
    assert first is not None and second is not None
    assert not first.cached and not second.cached


def test_the_cache_evicts_the_least_recently_used_answer():
    provider = StaticProvider(lambda k, q, s: {"type": "noul", "noul": 0.5})
    engine = DecisionEngine(DecisionPolicy(cache_size=2), provider)
    for i in range(3):
        engine.decide(TASK_RERANK, {"q": i}, {"n": Noul("?")}, tenant_id="t", space="s")
    assert len(engine._cache) == 2
    # The oldest is gone, so asking it again is a fresh call.
    again = engine.decide(TASK_RERANK, {"q": 0}, {"n": Noul("?")}, tenant_id="t", space="s")
    assert again is not None and not again.cached


def test_active_reads_the_mode_per_task():
    engine = DecisionEngine(_policy(rerank=MODE_ACTIVE, routing=MODE_SHADOW),
                            StaticProvider(lambda k, q, s: {"type": "noul", "noul": 0.5}))
    assert engine.active(TASK_RERANK)
    assert not engine.active("routing")


def test_a_call_log_that_raises_never_fails_the_decision(monkeypatch):
    import smrti.call_log as call_log

    def explode(*args, **kwargs):
        raise RuntimeError("the log is full")

    monkeypatch.setattr(call_log, "append", explode)
    provider = StaticProvider(lambda k, q, s: {"type": "noul", "noul": 0.5})
    engine = DecisionEngine(DecisionPolicy(), provider)
    outcome = engine.decide(TASK_RERANK, {"q": 1}, {"n": Noul("?")}, tenant_id="t", space="s")
    assert outcome is not None


# ── what a candidate tells the provider about itself ─────────────────────────


def _candidate(**kw) -> RecallResult:
    from smrti.core.models import AttentionValue, TruthValue, Valence

    atom = Atom(
        type=kw.pop("type", AtomType.EPISODE),
        label=kw.pop("label", "a"),
        content=kw.pop("content", "some memory"),
        tenant_id="default",
        space="main",
        truth=TruthValue(probability=kw.pop("probability", 0.8), confidence=0.5),
        attention=AttentionValue(sti=0.1, lti=0.1),
        valence=Valence(valence=0.0, intensity=0.0),
        **kw,
    )
    return RecallResult(atom=atom, salience=1.0, similarity=0.5)


def test_a_candidate_carries_its_type_its_standing_and_its_dates():
    from smrti.core.models import EntityType
    from smrti.decisions.retrieval import _candidate_state

    typed = _candidate_state("c0", _candidate(type=AtomType.CONCEPT, entity_type=EntityType.PERSON))
    assert typed["entity_type"] == EntityType.PERSON.value

    current = _candidate_state("c1", _candidate(type=AtomType.BELIEF, probability=0.8))
    superseded = _candidate_state("c2", _candidate(type=AtomType.BELIEF, probability=0.1))
    assert current["status"] == "current" and superseded["status"] == "superseded"

    dated = _candidate(
        metadata={"temporal": [
            {"text": "mañana", "resolved": "2026-08-27"},
            {"text": "no resolution"},        # incomplete entries are left out
            "not a mapping",
        ]}
    )
    assert _candidate_state("c3", dated)["dates"] == ["mañana = 2026-08-27"]

    assert "dates" not in _candidate_state("c4", _candidate())


def test_the_judge_keeps_the_local_ranking_when_the_provider_says_nothing():
    from smrti.decisions.retrieval import judge_evidence, make_judge

    results = [_candidate(label=f"a{i}") for i in range(3)]

    # The provider is there but answers nothing usable, so decide() is None.
    class _Down:
        name = "down"

        def ask(self, state, questions, *, timeout):
            raise DecisionUnavailable("down")

    engine = DecisionEngine(DecisionPolicy(), _Down())
    assert judge_evidence(engine, "q", results, tenant_id="t", space="s") == results


def test_a_judge_that_raises_does_not_lose_the_ranking_it_was_handed():
    from smrti.decisions.retrieval import make_judge
    import smrti.decisions.retrieval as retrieval_module

    results = [_candidate(label=f"a{i}") for i in range(3)]
    engine = DecisionEngine(DecisionPolicy(), StaticProvider(lambda k, q, s: {"type": "noul", "noul": 0.5}))
    judge = make_judge(engine, "t", "s")
    assert judge is not None

    # Anything at all going wrong inside the judgement is caught: the local
    # ranking is the thing that must survive.
    original = retrieval_module.judge_evidence
    try:
        retrieval_module.judge_evidence = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        assert judge("q", results) == results
    finally:
        retrieval_module.judge_evidence = original


# ── the three write-path tasks, silent when they are off ─────────────────────


def test_the_write_path_tasks_say_nothing_when_they_are_off():
    engine = DecisionEngine(_policy(), StaticProvider(lambda k, q, s: {"type": "noul", "noul": 0.5}))
    assert engine.mode(TASK_RERANK) == MODE_OFF  # _policy() turns every task off

    from smrti.decisions.extraction import route_extraction

    assert run(route_extraction(
        engine, "Alice moved to Paris", source="user", entity_context="", tenant_id="t", space="s"
    )) is None
    assert verify_supersession(
        engine, episode_text="t", subject="Alice", predicate="lives_in", old_object="Berlin",
        new_object="Paris", old_stated_at="", old_author="user", new_author="user",
        tenant_id="t", space="s",
    ) is None
    assert verify_entity(
        engine, name="Alice", entity_type="person", context="", candidates=[{"id": "1", "label": "Alice"}],
        tenant_id="t", space="s",
    ) is None


def test_a_verification_with_no_usable_choice_keeps_both_claims():
    # The provider answers, but not the question that was asked, so the
    # choice is absent and the code preserves rather than merges.
    class _Blank:
        name = "blank"

        def ask(self, state, questions, *, timeout):
            return parse_response(
                {"answers": {k: _reply({k: q.payload()})["answers"][k] for k, q in questions.items()}},
                questions,
            )

    engine = DecisionEngine(DecisionPolicy(), _Blank())
    verdict = verify_supersession(
        engine, episode_text="t", subject="Alice", predicate="lives_in", old_object="Berlin",
        new_object="Paris", old_stated_at="", old_author="user", new_author="user",
        tenant_id="t", space="s",
    )
    assert verdict is not None


def test_an_empty_shortlist_is_returned_untouched():
    from smrti.decisions.retrieval import judge_evidence

    engine = DecisionEngine(DecisionPolicy(), StaticProvider(lambda k, q, s: {"type": "noul", "noul": 0.5}))
    assert judge_evidence(engine, "q", [], tenant_id="t", space="s") == []


class _Refuses:
    name = "down"

    def ask(self, state, questions, *, timeout):
        raise DecisionUnavailable("down")


def test_a_verification_the_provider_could_not_answer_changes_nothing():
    engine = DecisionEngine(DecisionPolicy(), _Refuses())
    assert verify_supersession(
        engine, episode_text="t", subject="Alice", predicate="lives_in", old_object="Berlin",
        new_object="Paris", old_stated_at="", old_author="user", new_author="user",
        tenant_id="t", space="s",
    ) is None
    assert verify_entity(
        engine, name="Alice", entity_type="person", context="", candidates=[{"id": "1", "label": "Alice"}],
        tenant_id="t", space="s",
    ) is None


@pytest.mark.parametrize("key", ["relation", "identity"])
def test_a_verification_answered_with_the_wrong_kind_preserves_rather_than_merges(monkeypatch, key):
    # The choice is validated on the way in, so this shape should be
    # unreachable — but the guard is what keeps an unreadable answer from
    # being read as a merge, which is the one direction that cannot be undone.
    wrong_kind = parse_response({"answers": {key: {"type": "noul", "noul": 0.5}}}, {key: Noul("?")})
    engine = DecisionEngine(DecisionPolicy(), _Refuses())
    monkeypatch.setattr(
        engine, "decide",
        lambda *a, **k: DecisionOutcome(decisions=wrong_kind, mode=MODE_ACTIVE),
    )
    if key == "relation":
        assert verify_supersession(
            engine, episode_text="t", subject="Alice", predicate="lives_in", old_object="Berlin",
            new_object="Paris", old_stated_at="", old_author="user", new_author="user",
            tenant_id="t", space="s",
        ) is None
    else:
        assert verify_entity(
            engine, name="Alice", entity_type="person", context="",
            candidates=[{"id": "1", "label": "Alice"}], tenant_id="t", space="s",
        ) is None


def test_replacing_the_shared_engine_survives_a_provider_that_will_not_close():
    from smrti.decisions import reset_decisions

    class _StubbornProvider:
        name = "stubborn"

        def ask(self, state, questions, *, timeout):
            raise DecisionUnavailable("down")

        def close(self):
            raise RuntimeError("still busy")

    reset_decisions(DecisionEngine(DecisionPolicy(), _StubbornProvider()))
    # Releasing it must not propagate: the replacement is already in place.
    reset_decisions(None)


def test_the_suite_can_never_start_a_real_checkpoint_load():
    """The guard in conftest is the reason chunk one stopped aborting.

    A bare LayaProvider begins a real load on a background thread, and in
    CI — where laya is installed — that thread imports torch. Killed
    mid-import when the interpreter exits, it takes the process with it
    (SIGABRT, exit 134) after every test has already passed. This pins the
    stub in place so that cannot come back unnoticed.
    """
    import laya

    # The stub is a bare module object, so it has no file on disk at all.
    assert getattr(laya, "__file__", None) is None
    with pytest.raises(RuntimeError, match="never loads a real checkpoint"):
        laya.load("convaiinnovations/laya-multilingual")


# --- where the weights come from -------------------------------------------


def _artifact(tmp_path, *, graph=b"a graph", config=None):
    """A tarball shaped like the published one."""
    import io, json as _json, tarfile

    payload = {
        "edgejev.json": (config if config is not None else _json.dumps(
            {"max_len": 1024, "head_max_len": 256, "onnx_file": "model.onnx"}
        )).encode(),
        "model.onnx": graph,
    }
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as tar:
        for name, body in payload.items():
            info = tarfile.TarInfo("./" + name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return blob.getvalue()


def test_the_model_is_resolved_explicit_then_shared_then_own(tmp_path, monkeypatch):
    from smrti.decisions import model as model_source

    explicit, shared, own = tmp_path / "explicit", tmp_path / "factor", tmp_path / "own"
    for directory in (explicit, shared, own):
        directory.mkdir()
        (directory / "edgejev.json").write_text('{"max_len": 1024, "onnx_file": "model.onnx"}')
        (directory / "model.onnx").write_bytes(b"graph")

    monkeypatch.setenv("FACTOR_HOME", str(tmp_path))
    monkeypatch.setenv("SMRTI_HOME", str(tmp_path / "smrti_home"))
    (tmp_path / "decision-model").mkdir()
    for name in ("edgejev.json", "model.onnx"):
        (tmp_path / "decision-model" / name).write_bytes((shared / name).read_bytes())

    # An explicit directory wins over everything.
    monkeypatch.setenv("SMRTI_DECISIONS_MODEL", str(explicit))
    assert model_source.resolve() == explicit

    # Without one, the copy Factor already has is used rather than a second
    # 343 MB of the same weights.
    monkeypatch.delenv("SMRTI_DECISIONS_MODEL")
    assert model_source.resolve() == tmp_path / "decision-model"

    # An explicit directory that holds no model is an error, not a fallback:
    # somebody said where it is and was wrong, and quietly downloading
    # another copy would hide that.
    monkeypatch.setenv("SMRTI_DECISIONS_MODEL", str(tmp_path / "nothing-here"))
    with pytest.raises(FileNotFoundError):
        model_source.resolve()


def test_the_artifact_is_checked_before_it_is_unpacked(tmp_path, monkeypatch):
    import hashlib
    from smrti.decisions import model as model_source

    blob = _artifact(tmp_path)
    served = tmp_path / "served.tar.gz"
    served.write_bytes(blob)
    url = served.as_uri()

    destination = tmp_path / "model"
    with pytest.raises(ValueError, match="checksum"):
        model_source.fetch(destination, url=url, sha256="0" * 64)
    assert not destination.exists(), "a model that failed its checksum was unpacked anyway"

    model_source.fetch(destination, url=url, sha256=hashlib.sha256(blob).hexdigest())
    assert model_source.is_ready(destination)
    assert (destination / "model.onnx").read_bytes() == b"a graph"


def test_a_half_unpacked_model_does_not_read_as_installed(tmp_path):
    from smrti.decisions import model as model_source

    directory = tmp_path / "model"
    directory.mkdir()
    assert not model_source.is_ready(directory)

    (directory / "edgejev.json").write_text('{"max_len": 1024, "onnx_file": "model.onnx"}')
    assert not model_source.is_ready(directory), "metadata without a graph is not a model"

    (directory / "model.onnx").write_bytes(b"")
    assert not model_source.is_ready(directory), "an empty graph is not a model"

    (directory / "model.onnx").write_bytes(b"graph")
    assert model_source.is_ready(directory)


def test_the_archive_cannot_write_outside_its_directory(tmp_path):
    import io, tarfile
    from smrti.decisions import model as model_source

    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as tar:
        info = tarfile.TarInfo("../escaped")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    blob.seek(0)

    directory = tmp_path / "unpack"
    directory.mkdir()
    with tarfile.open(fileobj=blob) as tar:
        with pytest.raises(ValueError, match="outside the archive"):
            model_source._extract(tar, directory)
    assert not (tmp_path / "escaped").exists()


# ── the remote provider ──────────────────────────────────────────────────────


def _remote(handler, **kwargs):
    from smrti.decisions.remote import RemoteProvider

    return RemoteProvider(
        "http://127.0.0.1:8731/",
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _serve(calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        return httpx.Response(200, json={**_reply(body["questions"]), "model": "laya-int8"})

    return handler


def test_the_remote_provider_speaks_the_systemone_route_one_question_at_a_time():
    calls: list = []
    provider = _remote(_serve(calls))
    assert provider.base_url == "http://127.0.0.1:8731"
    decisions = provider.ask(
        "the deploy pipeline is owned by oslo",
        {"durable": Noul("is it durable?"), "which": Choice("pick", {"a": "first", "b": "second"})},
        timeout=5.0,
    )
    assert [path for path, _ in calls] == ["/v1/systemone", "/v1/systemone"]
    assert [list(body["questions"]) for _, body in calls] == [["durable"], ["which"]]
    assert all(body["model"] == "laya" and body["state"] == "the deploy pipeline is owned by oslo" for _, body in calls)
    assert decisions.noul("durable") == 0.8
    assert decisions.choice("which").choice == "a"
    assert decisions.input_tokens == 84  # 42 a call, two calls
    assert decisions.model == provider.model == "laya-int8"  # what the server loaded, not what was asked
    provider.close()


def test_the_remote_provider_async_path_shares_the_contract():
    calls: list = []
    provider = _remote(_serve(calls))
    decisions = asyncio.run(provider.ask_async({"text": "hola"}, {"q": Noul("x")}, timeout=5.0))
    assert decisions.noul("q") == 0.8 and len(calls) == 1
    asyncio.run(provider.aclose())


@pytest.mark.parametrize("status", [400, 503])
def test_a_server_that_refuses_or_is_loading_is_unavailable(status):
    provider = _remote(lambda request: httpx.Response(status, json={"error": {"message": "still loading"}}))
    with pytest.raises(DecisionUnavailable, match=f"HTTP {status}"):
        provider.ask("s", {"q": Noul("x")}, timeout=5.0)
    provider.close()


def test_a_server_that_is_down_is_unavailable_not_an_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    provider = _remote(refuse)
    with pytest.raises(DecisionUnavailable, match="unreachable"):
        provider.ask("s", {"q": Noul("x")}, timeout=5.0)
    with pytest.raises(DecisionUnavailable, match="unreachable"):
        asyncio.run(provider.ask_async("s", {"q": Noul("x")}, timeout=5.0))
    provider.close()


def test_a_reply_that_is_not_json_or_not_an_object_is_unavailable():
    provider = _remote(lambda request: httpx.Response(200, content=b"<html>"))
    with pytest.raises(DecisionUnavailable, match="not JSON"):
        provider.ask("s", {"q": Noul("x")}, timeout=5.0)
    provider = _remote(lambda request: httpx.Response(200, json=[1, 2]))
    with pytest.raises(DecisionUnavailable, match="not an object"):
        provider.ask("s", {"q": Noul("x")}, timeout=5.0)


def test_the_remote_provider_stops_asking_once_the_deadline_has_passed():
    clock = iter([0.0, 0.0, 10.0])  # started, first remaining, second remaining
    provider = _remote(_serve([]))
    with patch("smrti.decisions.remote.time.monotonic", side_effect=lambda: next(clock, 10.0)):
        with pytest.raises(DecisionUnavailable, match="before the deadline"):
            provider.ask("s", {"a": Noul("x"), "b": Noul("y")}, timeout=5.0)
    with pytest.raises(DecisionUnavailable, match="expired"):
        provider.ask("s", {"a": Noul("x")}, timeout=0)
    with pytest.raises(ValueError):
        provider.ask("s", {"a": Noul("x")}, timeout="soon")
    with pytest.raises(ValueError, match="at least one question"):
        provider.ask("s", {}, timeout=5.0)
    provider.close()


def test_a_decision_url_builds_the_remote_provider_and_loads_nothing_here():
    from smrti.decisions.remote import RemoteProvider

    engine = build_engine(DecisionPolicy.from_env({
        "SMRTI_DECISIONS": "active",
        "SMRTI_DECISIONS_URL": "http://127.0.0.1:8731/ ",
        "SMRTI_DECISIONS_MODEL": "/models/laya",
    }))
    assert isinstance(engine.provider, RemoteProvider)
    assert engine.provider.base_url == "http://127.0.0.1:8731"
    assert engine.provider_name == "edgejev"
    engine.provider.close()
    # Off is off, whatever server is named.
    assert build_engine(DecisionPolicy.from_env({"SMRTI_DECISIONS": "off", "SMRTI_DECISIONS_URL": "http://x"})).provider is None
