"""The decisions package on its own: questions, the Laya adapter, policy,
engine, and the pure functions the integrations fold answers with."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

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
    seen = {}

    def handler(state, questions):
        seen["state"] = state
        seen["questions"] = questions
        return _reply(questions)

    provider = _laya(handler, model="/models/laya-multilingual", device="cpu")
    questions = {
        "n": Noul("is it?"),
        "c": Choice("pick", {"a": "first", "b": "second"}),
        "s": Score("rate", ["low", "high"]),
    }
    decisions = provider.ask({"text": "hola"}, questions, timeout=2.0)

    assert seen["state"] == {"text": "hola"}
    assert seen["questions"]["n"] == {"type": "noul", "instructions": "is it?"}
    assert seen["questions"]["c"]["criteria"] == {"a": "first", "b": "second"}
    assert decisions.noul("n") == 0.8
    assert decisions.choice("c").choice == "a"
    assert decisions["s"].score == 0.0
    assert decisions.model == "/models/laya-multilingual"
    assert provider.device == "cpu"
    assert provider.input_tokens == 42
    assert provider.requests == 1
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


def test_laya_needs_a_model_id_or_path():
    with pytest.raises(ValueError):
        LayaProvider("")

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
    assert engine.model == DEFAULT_MODEL
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
