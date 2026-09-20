"""The decisions package on its own: questions, the Jev adapter, policy,
engine, and the pure functions the integrations fold answers with."""
from __future__ import annotations

import asyncio
import json

import httpx
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
from smrti.decisions.jev import JevProvider
from smrti.decisions.policies import MODE_ACTIVE, MODE_OFF, MODE_SHADOW, TASK_RERANK, TASKS
from smrti.decisions.provider import parse_response
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
    return DecisionPolicy().with_modes(**modes)


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
            "model": "jev-1.13.0",
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
    assert decisions.model == "jev-1.13.0"
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


# ── the Jev adapter ──────────────────────────────────────────────────────────


def _jev(handler, **kwargs) -> JevProvider:
    return JevProvider(
        "key-123",
        transport=httpx.MockTransport(handler),
        async_transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _reply(questions_body: dict, model="jev-test") -> dict:
    answers = {}
    for key, q in questions_body.items():
        if q["type"] == "noul":
            answers[key] = {"type": "noul", "noul": 0.8}
        elif q["type"] == "choice":
            first = next(iter(q["criteria"]))
            answers[key] = {"type": "choice", "choice": first, "probabilities": {first: 1.0}, "confidence": 1.0}
        else:
            answers[key] = {"type": "score", "score": 0.0, "probabilities": {"0": 1.0}, "confidence": 1.0}
    return {"model": model, "answers": answers, "usage": {"input_tokens": 42, "output_tokens": 0}}


def test_jev_posts_the_systemone_request_and_reads_the_answers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_reply(seen["body"]["questions"]))

    provider = _jev(handler, base_url="https://jev.example/")
    decisions = provider.ask({"text": "hi"}, {"q": Noul("is it?")}, timeout=2.0)

    assert seen["url"] == "https://jev.example/v1/systemone"
    assert seen["auth"] == "Bearer key-123"
    assert seen["body"]["state"] == {"text": "hi"}
    assert seen["body"]["model"] == "jev-latest"
    assert seen["body"]["questions"]["q"] == {"type": "noul", "instructions": "is it?"}
    assert decisions.noul("q") == 0.8
    assert decisions.model == "jev-test"
    assert provider.input_tokens == 42
    assert provider.requests == 1
    assert provider.estimated_cost_usd == pytest.approx(42 / 1e6 * 0.042)


def test_jev_retries_once_on_a_rate_limit_and_then_answers():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "0"})
        return httpx.Response(200, json=_reply(json.loads(request.content)["questions"]))

    decisions = _jev(handler).ask("s", {"q": Noul("?")}, timeout=2.0)
    assert decisions.noul("q") == 0.8
    assert len(calls) == 2


def test_jev_does_not_retry_a_rejected_key():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(DecisionUnavailable, match="rejected"):
        _jev(handler).ask("s", {"q": Noul("?")}, timeout=2.0)
    assert len(calls) == 1


def test_jev_reports_a_dropped_connection_as_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(DecisionUnavailable, match="failed"):
        _jev(handler, retries=0).ask("s", {"q": Noul("?")}, timeout=1.0)


def test_jev_refuses_an_invalid_reply():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": "x"}}})

    with pytest.raises(DecisionUnavailable):
        _jev(handler).ask("s", {"q": Noul("?")}, timeout=1.0)


def test_jev_async_path_shares_the_contract():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reply(json.loads(request.content)["questions"]))

    provider = _jev(handler)
    decisions = run(provider.ask_async("s", {"q": Noul("?")}, timeout=1.0))
    assert decisions.noul("q") == 0.8
    run(provider.aclose())


def test_jev_needs_a_key():
    with pytest.raises(ValueError):
        JevProvider("")


# ── policy ───────────────────────────────────────────────────────────────────


def test_every_task_is_off_by_default():
    policy = DecisionPolicy.from_env({})
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


def test_thresholds_and_the_key_come_from_the_environment():
    policy = DecisionPolicy.from_env({
        "TYPESAFE_API_KEY": "k", "SMRTI_DECISIONS_TIMEOUT": "2.5",
        "SMRTI_DECISIONS_RERANK_SHORTLIST": "30", "SMRTI_DECISIONS_ROUTING_SKIP": "0.1",
    })
    assert policy.api_key == "k"
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


def test_an_enabled_policy_without_a_key_builds_an_engine_with_every_task_off(caplog):
    engine = build_engine(DecisionPolicy.from_env({"SMRTI_DECISIONS": "active"}))
    assert engine.provider is None
    assert engine.mode("rerank") == MODE_OFF
    assert "no API key" in caplog.text


def test_a_key_builds_the_jev_provider():
    engine = build_engine(DecisionPolicy.from_env({
        "SMRTI_DECISIONS": "shadow", "TYPESAFE_API_KEY": "k",
        "SMRTI_DECISIONS_URL": "https://jev.example", "SMRTI_DECISIONS_MODEL": "jev-1.13.0",
    }))
    assert isinstance(engine.provider, JevProvider)
    assert engine.provider.base_url == "https://jev.example"
    assert engine.model == "jev-1.13.0"
    assert engine.mode("entity") == MODE_SHADOW


def test_the_shared_engine_is_built_once_and_can_be_replaced(monkeypatch):
    from smrti.decisions import get_decisions, reset_decisions

    monkeypatch.delenv("SMRTI_DECISIONS", raising=False)
    reset_decisions(None)
    first = get_decisions()
    assert get_decisions() is first
    assert first.mode("rerank") == MODE_OFF
    replacement = DecisionEngine(_policy(), None)
    reset_decisions(replacement)
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
