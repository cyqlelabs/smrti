"""The student decision model: registry matching, the wire server, the
in-process provider, and the engine choice."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from smrti.decisions import build_engine
from smrti.decisions.extraction import ROUTING_QUESTIONS, TONE_QUESTION
from smrti.decisions.policies import DecisionPolicy
from smrti.decisions.provider import Choice, DecisionUnavailable, DecisionUnsupported, Noul
from smrti.decisions.remote import RemoteProvider
from smrti.decisions.retrieval import evidence_questions
from smrti.decisions.student import hardware, registry
from smrti.decisions.student.provider import StudentProvider
from smrti.decisions.student.runtime import questions_from_payload
from smrti.decisions.student.serve import _handler


# ── registry ───────────────────────────────────────────────────────────────


def test_every_engine_question_matches_one_head():
    routing = registry.match(ROUTING_QUESTIONS)
    assert [m.spec.name for m in routing] == ["routing"]
    assert routing[0].names == {"durable": "durable", "corrects": "corrects", "novel": "novel"}

    rerank = registry.match(evidence_questions(["c3"]))
    assert rerank[0].spec.name == "rerank"
    assert rerank[0].names["c3_contradicts"] == "contradicts"

    assert registry.match({"tone": TONE_QUESTION})[0].spec.name == "tone"
    factor = Choice({"rules": registry.COMPLETION_RULES}, registry.COMPLETION_CRITERIA)
    assert registry.match({"completion": factor})[0].spec.name == "completion"


def test_a_question_no_head_answers_is_refused_whole():
    with pytest.raises(LookupError, match="anything"):
        registry.match({**ROUTING_QUESTIONS, "anything": Noul("is it?")})


def test_a_reworded_question_with_the_same_keys_still_matches(caplog):
    reworded = Choice("different words", registry.COMPLETION_CRITERIA)
    with caplog.at_level("WARNING"):
        assert registry.match({"completion": reworded})[0].spec.name == "completion"
    assert "wording differs" in caplog.text


def test_the_encoder_reads_the_task_and_the_state_and_never_the_question():
    text = registry.encoder_text("routing", {"message": "hola", "author": "user", "known_context": "",
                                             "facts": ["a", "b"]})
    assert text == 'routing:\nmessage: hola\nauthor: user\nfacts: ["a","b"]'


def test_wire_questions_are_validated_by_name():
    qs = questions_from_payload({"d": {"type": "noul", "instructions": "x", "criteria": {"true": "t", "false": "f"}},
                                 "t": {"type": "choice", "instructions": "y", "criteria": {"a": "1", "b": "2"}}})
    assert isinstance(qs["d"], Noul) and qs["d"].true == "t"
    assert isinstance(qs["t"], Choice)
    with pytest.raises(ValueError, match="question 'z'"):
        questions_from_payload({"z": {"type": "choice", "instructions": "", "criteria": ["not", "an", "object"]}})


# ── a stand-in student ─────────────────────────────────────────────────────


class _Stub:
    model_name = "stub-student"

    def predict(self, state, payload):
        questions = questions_from_payload(payload)
        try:
            matches = registry.match(questions)
        except LookupError as exc:
            raise DecisionUnsupported(str(exc)) from exc
        answers = {}
        for m in matches:
            if m.spec.kind == registry.NOULS:
                for name in m.names:
                    answers[name] = {"type": "noul", "noul": 0.8, "confidence": 0.8}
            else:
                name = next(iter(m.names))
                keys = m.spec.keys
                answers[name] = {"type": "choice", "choice": keys[0], "confidence": 0.7,
                                 "probabilities": {k: (0.7 if i == 0 else 0.3 / (len(keys) - 1)) for i, k in enumerate(keys)}}
        return {"model": self.model_name, "answers": answers, "usage": {"input_tokens": 12, "output_tokens": 0}}


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(_Stub(), threading.Lock()))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_the_server_reports_health_with_the_window_a_caller_sizes_against(server):
    with urllib.request.urlopen(server + "/health") as r:
        health = json.load(r)
    assert health["ok"] and health["model"] == "stub-student"
    assert health["max_len"] - health["head_max_len"] == 512


def test_the_server_answers_the_systemone_contract_the_remote_provider_speaks(server):
    provider = RemoteProvider(server)
    decisions = provider.ask({"message": "vivo en Rosario", "author": "user", "known_context": ""},
                             ROUTING_QUESTIONS, timeout=5.0)
    assert decisions.noul("durable") == 0.8 and decisions.model == "stub-student"
    tone = provider.ask({"text": "se cayó el servidor"}, {"tone": TONE_QUESTION}, timeout=5.0).choice("tone")
    assert tone.choice == "thing" and tone.confidence == 0.7
    provider.close()


def test_the_server_refuses_a_question_outside_the_registry_so_the_caller_falls_back(server):
    provider = RemoteProvider(server)
    with pytest.raises(DecisionUnsupported, match="does not answer this question"):
        provider.ask("s", {"whatever": Noul("is it?")}, timeout=5.0)
    provider.close()


def test_the_provider_answers_a_whole_head_in_one_call():
    provider = StudentProvider(model="stub", agent=_Stub())
    decisions = provider.ask({"message": "hi"}, ROUTING_QUESTIONS, timeout=5.0)
    assert set(decisions.answers) == {"durable", "corrects", "novel"}
    assert provider.requests == 1 and provider.input_tokens == 12
    provider.close()


# ── engine choice ──────────────────────────────────────────────────────────


def test_the_engine_is_chosen_by_setting_or_by_hardware(monkeypatch):
    assert type(build_engine(DecisionPolicy.from_env({"SMRTI_DECISIONS_ENGINE": "student"})).provider).__name__ == "StudentProvider"
    assert type(build_engine(DecisionPolicy.from_env({"SMRTI_DECISIONS_ENGINE": "laya"})).provider).__name__ == "LayaProvider"
    monkeypatch.setattr(hardware, "prefers_student", lambda: (True, "no avx2"))
    assert type(build_engine(DecisionPolicy.from_env({})).provider).__name__ == "StudentProvider"
    with pytest.raises(ValueError, match="SMRTI_DECISIONS_ENGINE"):
        DecisionPolicy.from_env({"SMRTI_DECISIONS_ENGINE": "gpt"})


def test_hardware_reads_the_kernel(tmp_path, monkeypatch):
    cpu = tmp_path / "cpuinfo"
    cpu.write_text("processor : 0\nflags : fpu sse sse2 ssse3 sse4a\n")
    assert "avx2" not in hardware.cpu_flags(str(cpu))
    mem = tmp_path / "meminfo"
    mem.write_text("MemTotal: 3620000 kB\nMemAvailable: 512000 kB\n")
    assert hardware.total_mb(str(mem)) == 3535
    monkeypatch.setattr(hardware, "cpu_flags", lambda: {"avx2", "sse4_2"})
    monkeypatch.setattr(hardware, "total_mb", lambda: 16000)
    assert hardware.prefers_student() == (False, "")
    monkeypatch.setattr(hardware, "total_mb", lambda: 3535)
    assert hardware.prefers_student()[0]


# ── refusals ───────────────────────────────────────────────────────────────


def test_two_candidates_in_one_request_are_refused():
    with pytest.raises(LookupError, match="one candidate per request"):
        registry.match({**evidence_questions(["c0"]), **evidence_questions(["c1"])})


def test_an_unsupported_question_opens_no_cooldown(server):
    from smrti.decisions import DecisionEngine
    from smrti.decisions.policies import DecisionPolicy

    engine = DecisionEngine(DecisionPolicy.from_env({"SMRTI_DECISIONS_URL": server}), RemoteProvider(server))
    assert engine.decide("entity", "s", {"identity": Choice("which?", {"c0": "a", "none": "b"})},
                         tenant_id="t", space="s") is None
    # The next question, one the student answers, is asked rather than
    # skipped for a cooldown.
    outcome = engine.decide("routing", {"message": "hola"}, ROUTING_QUESTIONS, tenant_id="t", space="s")
    assert outcome is not None and outcome.decisions.noul("durable") == 0.8


def test_the_remote_provider_sends_a_student_a_whole_head_at_once(server):
    provider = RemoteProvider(server)
    provider.ask({"message": "hola"}, ROUTING_QUESTIONS, timeout=5.0)
    assert provider._per_call() == 256
    provider.close()


def test_the_server_refuses_a_body_that_is_not_an_object(server):
    req = urllib.request.Request(server + "/v1/systemone", data=b"[1, 2]", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 400 and b"JSON object" in err.value.read()
