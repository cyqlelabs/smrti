"""Regressions from the audit of the decision layer, the space operations
and the benchmark harness: one test per defect fixed."""
from __future__ import annotations

import asyncio
import json
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


# ── the measurements behind the defaults ────────────────────────────────────


def test_the_supersession_set_counts_an_unanswered_check_as_the_writers_cut():
    """The supersession default rests on this set: a ``keep`` pair the check
    allowed, or could not answer, is a current fact the writer destroys."""
    from bench.decisions.supersession import DEFAULT_SET, load_items, score_verdicts

    items = load_items(DEFAULT_SET)
    keep = [i for i, item in enumerate(items) if item["label"] == "keep"]
    verdicts = [{"label": "compatible", "allowed": False} for _ in items]
    assert score_verdicts(items, verdicts)["destructive"] == []
    verdicts[keep[0]] = {"label": None}
    report = score_verdicts(items, verdicts)
    assert len(report["destructive"]) == 1 and report["blocked_keep"]["blocked"] == len(keep) - 1


def test_the_entity_set_fails_on_a_merge_into_the_wrong_candidate():
    """The entity default rests on this set: merging a mention into anything
    but its gold candidate is the one error that cannot be undone."""
    from bench.decisions.entity import DEFAULT_SET, load_items, score_verdicts

    items = load_items(DEFAULT_SET)
    verdicts = [{"chosen": item["gold"], "raw_label": "c0"} for item in items]
    report = score_verdicts(items, verdicts)
    assert report["wrong_merges"] == [] and report["true_merges"]["merged"] == report["true_merges"]["n"]
    separate = next(i for i, item in enumerate(items) if item["gold"] is None)
    verdicts[separate] = {"chosen": 0, "raw_label": "c0"}
    assert len(score_verdicts(items, verdicts)["wrong_merges"]) == 1


# ── space operations and bridges ────────────────────────────────────────────


def _spaces(tmp_path, *names):
    from smrti import Smrti

    path = str(tmp_path / "spaces.db")
    return [Smrti(db_path=path, tenant_id="t", write_space=name) for name in names]


def _bridge_atoms(mem, space):
    return mem.db.fetchall(
        "SELECT id, metadata FROM atoms WHERE tenant_id = 't' AND space = ? AND type != 'relation'", (space,)
    )


def test_set_operations_never_render_a_forgotten_atom(tmp_path):
    """Union and difference returned forgotten content and the Jaccard
    counted it: forgetting hid an atom from recall but not from set ops."""
    a, b = _spaces(tmp_path, "a", "b")
    secret = a.remember("The vault code is 4417.", valence=0.0)
    a.remember("The staging cluster runs on three nodes.", valence=0.0)
    b.remember("The staging cluster runs on three nodes.", valence=0.0)
    a.forget(atom_ids=[secret])
    assert secret not in {x.id for x in a.space_union("b").atoms}
    assert secret not in {x.id for x in a.space_difference("b").atoms}
    overlap = a.space_overlap("b")
    assert (overlap.sampled_a, overlap.size_a, overlap.jaccard) == (1, 1, 1.0)


def test_rebridging_leaves_a_forgotten_bridge_forgotten(tmp_path):
    """Refreshing a bridge replaced the bridge atom's metadata, forget stamp
    included, and brought it back."""
    left, right, bridge = _spaces(tmp_path, "a", "b", "a_x_b")
    left.believe("Preserve snapshots before every deploy", 0.8, valence=0.0)
    right.believe("Preserve snapshots before every deploy", 0.8, valence=0.0)
    assert left.materialize_bridge("b") == 1
    (row,) = _bridge_atoms(bridge, "a_x_b")
    bridge.forget(atom_ids=[row["id"]])
    assert left.materialize_bridge("b") == 0
    (again,) = _bridge_atoms(bridge, "a_x_b")  # the pair is a tombstone, not re-created
    assert again["id"] == row["id"] and json.loads(again["metadata"])["forgotten"] is True


def test_a_bridge_earns_its_own_warning_and_authorship():
    """Bridges merged both parents' metadata, so a stated valence or an agent
    source on either parent leaked into the bridge: two ``context`` parents
    produced a ``critical_warning`` bridge."""
    from smrti.core.models import Atom, AtomPair, AtomType, Valence
    from smrti.spaces.emergence import _merge_pair

    stated = Atom(type=AtomType.EPISODE, label="x", space="a", valence=Valence(valence=-0.8, intensity=0.9),
                  metadata={"valence_stated": True, "source": "agent"})
    plain = Atom(type=AtomType.EPISODE, label="x", space="b", metadata={"source": "agent"})
    user = Atom(type=AtomType.EPISODE, label="x", space="b")
    assert "valence_stated" not in _merge_pair(AtomPair(atom_a=stated, atom_b=plain, similarity=1.0), "a_x_b", "t").metadata
    assert _merge_pair(AtomPair(atom_a=stated, atom_b=plain, similarity=1.0), "a_x_b", "t").metadata["source"] == "agent"
    assert "source" not in _merge_pair(AtomPair(atom_a=stated, atom_b=user, similarity=1.0), "a_x_b", "t").metadata


def test_a_bridge_takes_its_tone_from_its_parents_tone_not_their_mood():
    """The bridge blended the parents' drifting mood into its own tone, so a
    neutral memory that had absorbed a sour mood gave its bridge a pruning
    floor neither parent was written with."""
    from smrti.core.models import Atom, AtomPair, AtomType, Valence
    from smrti.core.provenance import is_severe_retained
    from smrti.spaces.emergence import _merge_pair

    drifted = Valence(valence=-0.9, intensity=0.9, intrinsic_valence=0.0, intrinsic_intensity=0.0)
    a = Atom(type=AtomType.EPISODE, label="x", space="a", valence=drifted)
    b = Atom(type=AtomType.EPISODE, label="x", space="b", valence=drifted)
    bridged = _merge_pair(AtomPair(atom_a=a, atom_b=b, similarity=1.0), "a_x_b", "t")
    assert bridged.valence.valence == pytest.approx(-0.9) and bridged.valence.own == 0.0
    assert not is_severe_retained(bridged)


def test_a_bridge_is_the_same_whichever_parent_asks():
    """The bridge name is commutative but the content was not: label, type
    and metadata came from whichever parent was passed first."""
    from smrti.core.models import Atom, AtomPair, AtomType, TruthValue
    from smrti.spaces.emergence import _merge_pair

    x = Atom(type=AtomType.BELIEF, label="X", space="a", truth=TruthValue(probability=0.8, confidence=0.5),
             metadata={"key": "left"})
    y = Atom(type=AtomType.GOAL, label="Y", space="b", truth=TruthValue(probability=0.8, confidence=0.5),
             metadata={"key": "right"})
    xy = _merge_pair(AtomPair(atom_a=x, atom_b=y, similarity=1.0), "a_x_b", "t")
    yx = _merge_pair(AtomPair(atom_a=y, atom_b=x, similarity=1.0), "a_x_b", "t")
    assert (xy.label, xy.type, xy.metadata) == (yx.label, yx.type, yx.metadata)


def test_two_pairs_of_spaces_cannot_share_one_bridge_name(tmp_path):
    """("a_x_b", "c") and ("a", "b_x_c") both name "a_x_b_x_c"; the second
    bridge was written into the first one's space."""
    ab, c, a, bc = _spaces(tmp_path, "a_x_b", "c", "a", "b_x_c")
    for mem in (ab, c, a, bc):
        mem.remember("The office closes at six on Fridays.", valence=0.0)
    assert ab.materialize_bridge("c") == 1
    with pytest.raises(ValueError, match="share a bridge name"):
        a.materialize_bridge("b_x_c")


def test_a_bridge_atom_is_never_written_without_its_edges(tmp_path):
    """The bridge atom and its two edges were three transactions, so an edge
    that failed left a bridge atom joined to nothing."""
    from smrti.spaces.emergence import materialize_bridge
    from smrti.spaces.set_ops import space_overlap

    a, b = _spaces(tmp_path, "a", "b")
    a.remember("The office closes at six on Fridays.", valence=0.0)
    gone = b.remember("The office closes at six on Fridays.", valence=0.0)
    overlap = space_overlap("t", "a", "b", a.db, 0.85, a.embed)
    b.atomspace.delete_atoms([gone], "t")  # the edge to it can no longer be drawn
    with pytest.raises(Exception):
        materialize_bridge(overlap, "t", a.db, a.embed, a.atomspace)
    assert _bridge_atoms(a, "a_x_b") == []


def test_refreshing_a_bridge_redraws_an_edge_it_lost(tmp_path):
    """The update path refreshed the bridge atom and never looked at its
    edges, so one lost was lost for good."""
    a, b = _spaces(tmp_path, "a", "b")
    a.remember("The office closes at six on Fridays.", valence=0.0)
    b.remember("The office closes at six on Fridays.", valence=0.0)
    a.materialize_bridge("b")
    edges = "SELECT id FROM atoms WHERE tenant_id = 't' AND space = 'a_x_b' AND relation = 'bridge'"
    first = a.db.fetchall(edges)[0]["id"]
    a.atomspace.delete_atoms([first], "t")
    assert len(a.db.fetchall(edges)) == 1
    a.materialize_bridge("b")
    assert len(a.db.fetchall(edges)) == 2


def test_a_missing_neighbourhood_is_not_read_as_agreement(tmp_path):
    """Two untyped "Java" atoms scored 1.0 when only one had neighbours: the
    missing signal handed its weight to the embedding of a bare name."""
    from smrti.core.models import Atom, AtomType
    from smrti.spaces.set_ops import space_overlap

    (mem,) = _spaces(tmp_path, "prog")
    java = mem.atomspace.add_atom(Atom(type=AtomType.CONCEPT, label="Java", tenant_id="t", space="prog"))
    for name in ("JVM", "Spring Boot", "Kotlin"):
        other = mem.atomspace.add_atom(Atom(type=AtomType.CONCEPT, label=name, tenant_id="t", space="prog"))
        mem.atomspace.link_atoms(java, other, "related_to", "t", "prog")
    mem.atomspace.add_atom(Atom(type=AtomType.CONCEPT, label="Java", tenant_id="t", space="travel"))
    overlap = space_overlap("t", "prog", "travel", mem.db, 0.85, mem.embed)
    assert not [p for p in overlap.pairs if p.atom_a.label == "Java"]


def test_an_overlap_embeds_every_neighbourhood_in_one_call(tmp_path):
    """Each candidate with neighbours cost its own encoder pass, and each
    sampled atom its own vector query."""
    from smrti.spaces.set_ops import space_overlap

    a, b = _spaces(tmp_path, "a", "b")
    for mem in (a, b):
        for i in range(4):
            x = mem.remember(f"Deploy step {i} runs on the staging cluster.", valence=0.0)
            y = mem.remember(f"Rollback plan {i} for the staging cluster.", valence=0.0)
            mem.atomspace.link_atoms(x, y, "related_to", "t", mem.write_space)

    class Counting:
        def __init__(self, inner):
            self.inner, self.single, self.batches = inner, 0, 0

        def embed(self, text):
            self.single += 1
            return self.inner.embed(text)

        def embed_batch(self, texts):
            self.batches += 1
            return self.inner.embed_batch(texts)

    engine = Counting(a.embed)
    overlap = space_overlap("t", "a", "b", a.db, 0.85, engine)
    assert overlap.pairs and engine.single == 0 and engine.batches == 1
