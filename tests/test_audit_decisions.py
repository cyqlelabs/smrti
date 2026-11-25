"""Regressions from the audit of the decision layer, the space operations
and the benchmark harness: one test per defect fixed."""
from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import threading
import time

import pytest

from smrti.decisions import DecisionEngine, DecisionPolicy, Noul, StaticProvider, audit
from smrti.decisions.laya import LayaProvider
from smrti.decisions.policies import MODE_OFF, TASKS
from smrti.decisions.provider import DecisionNotReady, DecisionUnavailable


def _policy(**modes) -> DecisionPolicy:
    return DecisionPolicy(modes={task: MODE_OFF for task in TASKS}).with_modes(**modes)


def _noul(_key, _question, _state):
    return {"type": "noul", "noul": 0.9}


class _FakeAgent:
    def __init__(self, delay: float):
        self.delay = delay

    def predict(self, state, questions):
        time.sleep(self.delay)
        return {"answers": {k: {"type": "noul", "noul": 0.5} for k in questions}, "usage": {}}


# ── the decision engine ─────────────────────────────────────────────────────


def test_a_bad_reply_to_one_task_does_not_silence_the_others():
    """One malformed rerank reply used to open an engine-wide cooldown, and
    tone, supersession, entity and routing went quiet for a minute."""

    def answer(key, question, state):
        if state == "rerank":
            return {"type": "noul", "noul": 1.7}
        return _noul(key, question, state)

    provider = StaticProvider(answer)
    engine = DecisionEngine(_policy(rerank="active", tone="active"), provider)
    assert engine.decide("rerank", "rerank", {"q": Noul("?")}, tenant_id="t", space="s") is None
    assert engine.offline_for("rerank") and not engine.offline
    assert engine.decide("tone", "tone", {"q": Noul("?")}, tenant_id="t", space="s") is not None
    calls = len(provider.calls)
    assert engine.decide("rerank", "rerank again", {"q": Noul("?")}, tenant_id="t", space="s") is None
    assert len(provider.calls) == calls  # rerank itself is cooling down


def test_a_model_still_loading_opens_no_cooldown():
    """The first caller during a load opened the cooldown, so a model ready a
    second later stayed unused for a minute."""

    class _Loading:
        name = "loading"
        model = "m"

        def __init__(self):
            self.ready = False

        def ask(self, state, questions, *, timeout):
            if not self.ready:
                raise DecisionNotReady("the decision model is still loading")
            return StaticProvider(_noul).ask(state, questions, timeout=timeout)

    provider = _Loading()
    engine = DecisionEngine(_policy(tone="active"), provider)
    assert engine.decide("tone", "a", {"q": Noul("?")}, tenant_id="t", space="s") is None
    assert not engine.offline_for("tone")
    assert audit.get_all()[0]["outcome"] == "loading"
    provider.ready = True
    assert engine.decide("tone", "b", {"q": Noul("?")}, tenant_id="t", space="s") is not None


def test_queue_wait_that_spends_a_deadline_is_the_callers_budget():
    """With one inference worker, a caller queued behind another ran out of
    its deadline in the queue and that was read as the model being down."""
    provider = LayaProvider(agent=_FakeAgent(0.6), model="m")
    engine = DecisionEngine(_policy(tone="active"), provider)
    try:
        first = threading.Thread(
            target=engine.decide, args=("tone", "a", {"q": Noul("?")}),
            kwargs={"tenant_id": "t", "space": "s", "timeout": 2.0},
        )
        first.start()
        time.sleep(0.1)
        assert engine.decide("tone", "b", {"q": Noul("?")}, tenant_id="t", space="s", timeout=0.3) is None
        first.join()
        assert not engine.offline_for("tone")
        assert "budget" in [r["outcome"] for r in audit.get_all()]
    finally:
        provider.close()


def test_a_model_slow_with_the_worker_to_itself_still_opens_the_cooldown():
    """Contention is the caller's budget; a model that cannot answer alone in
    time is down, and every task stops asking it."""
    provider = LayaProvider(agent=_FakeAgent(0.5), model="m")
    engine = DecisionEngine(_policy(tone="active", routing="active"), provider)
    try:
        assert engine.decide("tone", "a", {"q": Noul("?")}, tenant_id="t", space="s", timeout=0.1) is None
        assert engine.offline and engine.offline_for("routing")
    finally:
        provider.close()


def test_a_hung_inference_does_not_hold_the_process_open():
    """The inference worker was a ThreadPoolExecutor thread, joined at exit:
    a 15 s inference held the interpreter open 15 s after the decision had
    given up after one."""
    script = (
        "import time\n"
        "from smrti.decisions.laya import LayaProvider\n"
        "from smrti.decisions import Noul, DecisionUnavailable\n"
        "class Hang:\n"
        "    def predict(self, s, q):\n"
        "        time.sleep(30)\n"
        "p = LayaProvider(agent=Hang(), model='m')\n"
        "try:\n"
        "    p.ask('s', {'q': Noul('?')}, timeout=0.2)\n"
        "except DecisionUnavailable:\n"
        "    pass\n"
    )
    started = time.monotonic()
    subprocess.run([sys.executable, "-c", script], check=True, timeout=20)
    assert time.monotonic() - started < 10


def test_the_async_health_probe_never_blocks_the_event_loop():
    """The remote provider's /health probe was a blocking GET with its own
    five seconds, run inside ``ask_async``: a hung server stalled the event
    loop for all five, past the decision's one-second deadline."""
    from smrti.decisions.remote import RemoteProvider

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    held = []

    def accept():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            held.append(conn)

    threading.Thread(target=accept, daemon=True).start()
    provider = RemoteProvider(f"http://127.0.0.1:{server.getsockname()[1]}")

    async def run() -> float:
        gaps: list[float] = []
        done = False

        async def tick():
            last = time.monotonic()
            while not done:
                await asyncio.sleep(0.02)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        ticker = asyncio.create_task(tick())
        with pytest.raises(DecisionUnavailable):
            await provider.ask_async("s", {"q": Noul("?")}, timeout=0.5)
        done = True
        await ticker
        return max(gaps)

    try:
        assert asyncio.run(run()) < 0.3
        # The failed probe is remembered rather than paid again next call.
        assert provider._health_retry_at > time.monotonic()
    finally:
        server.close()
        for conn in held:
            conn.close()
        provider.close()


def test_inference_holds_four_threads_by_default(monkeypatch):
    """Unset, the runtime spread one small inference over every core and was
    slower than four threads (85 against 53 ms per noul)."""
    import os

    from smrti.decisions.laya import _threads

    monkeypatch.delenv("SMRTI_DECISIONS_THREADS", raising=False)
    assert _threads() == min(4, os.cpu_count() or 1)
    monkeypatch.setenv("SMRTI_DECISIONS_THREADS", "2")
    assert _threads() == 2


@pytest.mark.parametrize("damping", ["1.5", "-1.0"])
def test_a_tone_damping_off_the_unit_interval_is_refused(damping):
    """-1.0 flipped a complaint into praise and 1.5 made every remember of a
    mood fail validation; both were accepted without a word."""
    with pytest.raises(ValueError, match="SMRTI_DECISIONS_TONE_DAMPING"):
        DecisionPolicy.from_env({"SMRTI_DECISIONS_TONE_DAMPING": damping})


def test_decision_records_can_be_read_per_tenant():
    """``GET /decisions`` served every tenant's records, query text included."""
    audit.reset()
    for tenant in ("t1", "t2"):
        audit.record(task="tone", mode="active", tenant_id=tenant, space="s", provider="p", model="m",
                     outcome="thing", applied=True)
    assert {r["tenant_id"] for r in audit.records(["t1"])} == {"t1"}
    assert {r["tenant_id"] for r in audit.records()} == {"t1", "t2"}


def test_an_unconfigured_recall_never_reaches_the_model(tmp_path, monkeypatch):
    """With every task on by default a 17 ms recall took about 4.5 s; the
    recall path's one task, rerank, is off unless asked for, and the model
    stays unloaded until a task that is on asks it something."""
    from smrti import Smrti
    from smrti.decisions import get_decisions, reset_decisions

    for task in TASKS:
        monkeypatch.delenv(f"SMRTI_DECISIONS_{task.upper()}", raising=False)
    monkeypatch.delenv("SMRTI_DECISIONS", raising=False)
    monkeypatch.setenv("SMRTI_DECISIONS_ENGINE", "laya")
    reset_decisions(None)
    audit.reset()
    mem = Smrti(db_path=str(tmp_path / "m.db"), tenant_id="t", write_space="s")
    mem.remember("The staging cluster runs on three nodes.", valence=0.0)
    mem.recall("how many nodes does staging have?")
    engine = get_decisions()
    assert not engine.enabled("rerank")
    assert engine.provider is not None and not engine.provider.ready
    assert engine.provider._loader is None
    assert audit.get_all() == []
