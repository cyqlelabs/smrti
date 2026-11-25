"""Regressions from the evolution and forgetting audit: final forgetting, epochs
safe across processes and failures, current claims kept alive, and epochs that
write only what changes."""
from __future__ import annotations

import asyncio
import json
import time
import types
from unittest.mock import MagicMock, patch

import pytest

import smrti.servers.reflect_loop as reflect_loop_module
from smrti import Smrti
from smrti.core.db import Database
from smrti.core.models import Evidence, TruthValue
from smrti.evolution import epoch as epoch_module
from smrti.evolution.attention import propagate_sti
from smrti.evolution.connections import discover_connections
from smrti.evolution.epoch import LeaseLost, _apply_evidence, _propagate
from smrti.evolution.healing import heal_orphaned_episodes
from smrti.evolution.truth import update_truth
from smrti.evolution.valence import nudge_params, propagate_valence
from smrti.personality.params import personality_params
from smrti.servers.reflect_loop import run_reflect_loop

T, S = "t", "s"


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "audit_evolution.db")


@pytest.fixture
def mem(db_path):
    return Smrti(db_path=db_path, tenant_id=T, write_space=S)


def _row(mem, atom_id):
    return mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (atom_id,))


def _meta(mem, atom_id):
    return json.loads(_row(mem, atom_id)["metadata"] or "{}")


def _epoch_count(mem):
    return mem.db.fetchone(
        "SELECT epoch_count FROM personality WHERE tenant_id = ? AND space = ?", (T, S)
    )["epoch_count"]


def _lease(mem):
    return mem.db.fetchone(
        "SELECT * FROM epoch_lease WHERE tenant_id = ? AND space = ?", (T, S)
    )


def _file(mem, atom_id, observed=1.0, weight=1.0):
    mem.atomspace.add_evidence(Evidence(
        atom_id=atom_id, observed_probability=observed, weight=weight,
        tenant_id=T, space=S,
    ))


def _mentions(mem, episode, *entities):
    for entity in entities:
        mem.atomspace.link_atoms(episode, entity, "mentions", T, S)


# ── forgetting is final ──────────────────────────────────────────────────────

def test_pending_evidence_never_lifts_a_forgotten_atom(mem):
    """Evidence filed before a forget used to be applied at the next epoch,
    lifting the atom from 0.05 back over the surfacing floor, where the floor
    then held it forever. It is retired by the forget and the atom is pruned."""
    atom_id = mem.believe("The user dislikes cilantro", 0.8, valence=0.0)
    _file(mem, atom_id)
    mem.forget(atom_ids=[atom_id])

    pending = mem.db.fetchone(
        "SELECT COUNT(*) n FROM evidence WHERE atom_id = ? AND processed = 0", (atom_id,)
    )["n"]
    assert pending == 0
    mem.reflect()
    assert _row(mem, atom_id) is None


def test_forget_zeroes_attention(mem):
    """A forgotten atom kept its STI, so expansion budgets and propagation
    still spent on it until the pruner got there."""
    atom_id = mem.remember("My old locker code was 4471", valence=0.0)
    mem.db.execute("UPDATE atoms SET sti = 2.0 WHERE id = ?", (atom_id,))
    mem.forget(atom_ids=[atom_id])
    assert _row(mem, atom_id)["sti"] == 0.0


def test_propagation_neither_credits_nor_nudges_a_forgotten_atom(mem):
    """Propagation handed attention and mood back to forgotten neighbours."""
    source = mem.remember("The release crashed and corrupted user files", valence=-0.9)
    forgotten = mem.remember("Release notes draft for version two", valence=0.0)
    mem.atomspace.link_atoms(source, forgotten, "associated", T, S)
    mem.forget(atom_ids=[forgotten])
    # Kept above the prune line so the epoch leaves it in place to inspect.
    mem.db.execute("UPDATE atoms SET lti = 0.5 WHERE id = ?", (forgotten,))
    mem.db.execute("UPDATE atoms SET sti = 1.0 WHERE id = ?", (source,))

    mem.reflect()

    row = _row(mem, forgotten)
    assert row["sti"] == 0.0
    assert row["valence"] == 0.0


def test_association_discovery_never_links_a_forgotten_atom(mem):
    """forget() drops the vector, so a forgotten atom that still had LTI was
    re-embedded and linked to its paraphrase every tenth epoch."""
    forgotten = mem.remember(
        "Running the migration without a backup destroyed the orders table", valence=0.0
    )
    paraphrase = mem.remember(
        "The migration ran without a backup and the orders table was destroyed", valence=0.0
    )
    mem.db.execute("UPDATE atoms SET lti = 0.6 WHERE id IN (?, ?)", (forgotten, paraphrase))
    mem.forget(atom_ids=[forgotten])

    discover_connections(T, S, mem.db, mem.embed)

    edges = mem.db.fetchall(
        "SELECT id FROM atoms WHERE type = 'relation' AND (source_id = ? OR target_id = ?)",
        (forgotten, forgotten),
    )
    assert edges == []


def test_healing_never_files_an_episode_under_a_forgotten_person(mem):
    """The sole person was linked to every orphaned episode even after the
    caller forgot it."""
    person = mem.remember("Alice", type="concept", valence=0.0)
    mem.db.execute("UPDATE atoms SET entity_type = 'person' WHERE id = ?", (person,))
    episode = mem.remember("Went climbing at the gym", valence=0.0)
    gym = mem.remember("climbing gym", type="concept", valence=0.0)
    _mentions(mem, episode, gym)
    mem.forget(atom_ids=[person])

    assert heal_orphaned_episodes(T, S, mem.db) == 0


def test_forget_cascade_removes_what_only_the_episode_supported(mem):
    """Forgetting an episode left the concepts and claim extracted from it,
    so recall kept returning what the caller asked to be rid of."""
    episode = mem.remember("The prod DB key is stored in vault path secret/prod", valence=0.0)
    key = mem.remember("prod DB key", type="concept", valence=0.0)
    vault = mem.remember("secret/prod", type="concept", valence=0.0)
    _mentions(mem, episode, key, vault)
    claim = mem.atomspace.link_atoms(key, vault, "stored_in", T, S, metadata={"episodes": [episode]})

    labels = mem.forget(atom_ids=[episode], cascade=True)

    assert {"prod DB key", "secret/prod"} <= set(labels)
    assert _row(mem, claim) is None
    assert _meta(mem, key).get("forgotten") and _meta(mem, vault).get("forgotten")
    mem.reflect()
    assert _row(mem, key) is None and _row(mem, vault) is None


def test_forget_cascade_keeps_what_a_live_episode_supports(mem):
    """The cascade must not take an entity or claim another episode still
    supports — only the claim the forgotten episode alone stated."""
    gone = mem.remember("Alice lives in Paris", valence=0.0)
    kept = mem.remember("Alice works at Acme", valence=0.0)
    alice = mem.remember("Alice", type="concept", valence=0.0)
    paris = mem.remember("Paris", type="concept", valence=0.0)
    acme = mem.remember("Acme", type="concept", valence=0.0)
    _mentions(mem, gone, alice, paris)
    _mentions(mem, kept, alice, acme)
    lives_in = mem.atomspace.link_atoms(alice, paris, "lives_in", T, S, metadata={"episodes": [gone]})
    works_at = mem.atomspace.link_atoms(alice, acme, "works_at", T, S, metadata={"episodes": [kept]})

    mem.forget(atom_ids=[gone], cascade=True)

    assert _row(mem, lives_in) is None
    assert _meta(mem, paris).get("forgotten")
    for survivor in (alice, acme, works_at):
        assert not _meta(mem, survivor).get("forgotten")


def test_forget_cascade_keeps_a_claim_another_episode_asserted(mem):
    """Co-mention was read as provenance: forgetting an episode that merely
    mentioned Alice and tea deleted the claim another episode stated."""
    stated = mem.remember("Alice said she likes it", valence=0.0)
    chatter = mem.remember("Alice and the tea were both at the party", valence=0.0)
    alice = mem.remember("Alice", type="concept", valence=0.0)
    tea = mem.remember("tea", type="concept", valence=0.0)
    _mentions(mem, stated, alice)
    _mentions(mem, chatter, alice, tea)
    likes = mem.atomspace.link_atoms(alice, tea, "likes", T, S, metadata={"episodes": [stated]})

    mem.forget(atom_ids=[chatter], cascade=True)

    assert _row(mem, likes) is not None
    assert not _meta(mem, tea).get("forgotten")


def test_forget_cascade_keeps_a_claim_nothing_records_an_assertion_for(mem):
    """A claim written before claims recorded who asserted them is kept: when
    in doubt an atom stays."""
    episode = mem.remember("Bob moved to Lisbon", valence=0.0)
    bob = mem.remember("Bob", type="concept", valence=0.0)
    lisbon = mem.remember("Lisbon", type="concept", valence=0.0)
    _mentions(mem, episode, bob, lisbon)
    lives_in = mem.atomspace.link_atoms(bob, lisbon, "lives_in", T, S)

    mem.forget(atom_ids=[episode], cascade=True)

    assert _row(mem, lives_in) is not None


def test_forget_cascade_works_in_chunks(mem, monkeypatch):
    """Lookups over many entities exceeded SQLite's 999-variable limit after
    the episode had already been stamped."""
    from smrti.evolution import forgetting

    monkeypatch.setattr(forgetting, "_CHUNK", 1)
    episode = mem.remember("Carol keeps the spare keys in the blue box", valence=0.0)
    carol = mem.remember("Carol", type="concept", valence=0.0)
    box = mem.remember("blue box", type="concept", valence=0.0)
    _mentions(mem, episode, carol, box)
    claim = mem.atomspace.link_atoms(carol, box, "keeps_keys_in", T, S, metadata={"episodes": [episode]})

    mem.forget(atom_ids=[episode], cascade=True)

    assert _row(mem, claim) is None
    assert _meta(mem, carol).get("forgotten") and _meta(mem, box).get("forgotten")


def test_a_stale_lease_holder_cannot_commit_its_decay(mem):
    """The lease was checked between phases, so a holder that stalled past its
    lease and resumed after a takeover decayed the space a second time."""
    from smrti.evolution.epoch import _decay, _fence

    atom_id = mem.remember("The kettle is in the left cupboard", valence=0.0)
    mem.reflect()
    mem.db.execute(
        "UPDATE epoch_lease SET owner = 'someone-else', expires_at = ? WHERE tenant_id = ? AND space = ?",
        (time.time() + 300, T, S),
    )
    before = _row(mem, atom_id)["confidence"]
    p = personality_params(mem.db.fetchone("SELECT * FROM personality WHERE tenant_id = ? AND space = ?", (T, S)))

    with pytest.raises(LeaseLost):
        _decay(T, S, mem.db, p, fence=lambda conn: _fence(conn, T, S, "me"))
    assert _row(mem, atom_id)["confidence"] == before


# ── epochs across processes and failures ────────────────────────────────────

def test_a_space_leased_by_another_process_is_left_untouched(mem):
    """Two processes consolidating one space decayed and propagated it twice.
    A run that cannot take the lease changes nothing and keeps the activity
    count, so the loop tries again next interval."""
    atom_id = mem.remember("I walked the dog this morning", valence=0.0)
    mem.reflect()
    mem.remember("Another walk today", valence=0.0)
    mem.db.execute(
        "UPDATE epoch_lease SET owner = 'elsewhere', expires_at = ? WHERE tenant_id = ? AND space = ?",
        (time.time() + 600, T, S),
    )
    before, count = dict(_row(mem, atom_id)), _epoch_count(mem)

    result = mem.reflect()

    assert result.atoms_decayed == 0 and result.beliefs_updated == 0
    assert _epoch_count(mem) == count
    assert dict(_row(mem, atom_id)) == before
    assert mem.ops_since_reflect > 0


def test_an_expired_lease_is_taken_over_and_completion_recorded(mem):
    """A holder that died must not block the space forever, and the epoch
    that completes is recorded apart from the count of epochs started."""
    mem.remember("I walked the dog this morning", valence=0.0)
    mem.reflect()
    mem.db.execute(
        "UPDATE epoch_lease SET owner = 'dead', expires_at = ? WHERE tenant_id = ? AND space = ?",
        (time.time() - 1, T, S),
    )

    mem.reflect()

    lease = _lease(mem)
    assert lease["owner"] is None
    assert lease["completed_epoch"] == _epoch_count(mem) == 2


def test_an_epoch_that_loses_its_lease_stops_and_is_not_recorded(mem):
    """An epoch that outlived its lease must stop before its next write
    rather than run alongside the process that took the space over."""
    mem.remember("I walked the dog this morning", valence=0.0)
    original = epoch_module._decay

    def taken_over(*args):
        original(*args)
        mem.db.execute(
            "UPDATE epoch_lease SET owner = 'usurper' WHERE tenant_id = ? AND space = ?", (T, S)
        )

    with patch.object(epoch_module, "_decay", taken_over):
        with pytest.raises(LeaseLost):
            mem.reflect()
    assert _lease(mem)["completed_epoch"] == 0
    assert _lease(mem)["owner"] == "usurper"
    assert mem.ops_since_reflect > 0


def test_two_connections_cannot_apply_the_same_evidence(db_path, mem):
    """The evidence pass read pending rows and marked them processed in
    separate transactions, so a second process reading in between applied
    them again (0 -> 0.583 instead of 0.412). Lease aside, the claim alone
    must hold: the second reader finds nothing pending."""
    atom_id = mem.believe("A proposition", 0.0, valence=0.0)
    _file(mem, atom_id)
    initial = _row(mem, atom_id)
    expected = update_truth(
        TruthValue(probability=initial["probability"], confidence=initial["confidence"]),
        1.0, 1.0, 0.3,
    )
    other = Database(db_path)
    other.initialize()
    original = mem.db.fetchall

    def interleave(sql, params=()):
        rows = original(sql, params)
        if "FROM evidence WHERE processed = 0" in sql:
            mem.db.fetchall = original
            _apply_evidence(T, S, other, 0.3)
        return rows

    mem.db.fetchall = interleave
    try:
        _apply_evidence(T, S, mem.db, 0.3)
    finally:
        mem.db.fetchall = original
        other.close()

    assert _row(mem, atom_id)["probability"] == pytest.approx(expected.probability)
    assert _row(mem, atom_id)["confidence"] == pytest.approx(expected.confidence)


def test_a_failed_vector_repair_still_completes_the_epoch(mem):
    """An encoder failure in the last phase raised after decay had committed
    and left the activity count set, so an idle space aged on every interval
    for as long as the encoder stayed down."""
    atom_id = mem.remember("I bought a new bike", valence=0.0)
    mem.db.execute(
        "UPDATE atoms SET metadata = json_set(metadata, '$.vector_missing', 1) WHERE id = ?",
        (atom_id,),
    )
    with patch.object(mem.embed, "embed", side_effect=RuntimeError("encoder down")):
        mem.reflect()

    assert mem.ops_since_reflect == 0
    assert reflect_loop_module._was_used(mem) is False
    assert _lease(mem)["completed_epoch"] == _epoch_count(mem)
    assert _meta(mem, atom_id).get("vector_missing")


def test_the_loop_consolidates_an_instance_evicted_with_pending_activity():
    """The activity count lives on the instance, so a server evicting it from
    its cache dropped the count and the activity was never consolidated."""
    mem = types.SimpleNamespace(tenant_id=T, write_space=S, ops_since_reflect=0)
    mem.reflect = MagicMock(side_effect=lambda: setattr(mem, "ops_since_reflect", 0))
    ticks = [0]

    async def fake_sleep(_):
        ticks[0] += 1
        if ticks[0] == 2:
            mem.ops_since_reflect = 3  # used, then evicted before this pass
        if ticks[0] > 3:
            raise asyncio.CancelledError

    async def _run():
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await run_reflect_loop(lambda: [mem] if ticks[0] == 1 else [])

    with patch.object(reflect_loop_module, "REFLECT_INTERVAL", 1):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(_run())

    assert mem.reflect.call_count == 1
    assert mem.ops_since_reflect == 0


# ── current claims stay alive ───────────────────────────────────────────────

def _ripe_for_pruning(mem, *atom_ids):
    mem.db.execute(
        f"UPDATE atoms SET confidence = 0.01, lti = 0.0 WHERE id IN ({','.join('?' * len(atom_ids))})",
        atom_ids,
    )


def test_a_current_user_claim_keeps_its_endpoints(mem):
    """Untouched concepts fell under both floors around epoch 80 and the
    prune cascade deleted the user's current lives_in claim with them."""
    alice = mem.remember("Alice", type="concept", valence=0.0)
    paris = mem.remember("Paris", type="concept", valence=0.0)
    loner = mem.remember("Kubernetes", type="concept", valence=0.0)
    claim = mem.atomspace.link_atoms(alice, paris, "lives_in", T, S)
    _ripe_for_pruning(mem, alice, paris, loner)

    mem.reflect()

    assert _row(mem, alice) and _row(mem, paris) and _row(mem, claim)
    assert _row(mem, loner) is None


def test_agent_superseded_or_forgotten_claims_protect_nothing(mem):
    """Only a current claim the user made keeps an endpoint: model output the
    user never adopted, history, and what the caller forgot can all leave."""
    ids = {}
    for name, metadata in (
        ("agent", {"source": "agent"}),
        ("superseded", {"superseded_by": "later-edge"}),
        ("forgotten", {"forgotten": True}),
    ):
        subject = mem.remember(f"{name} subject", type="concept", valence=0.0)
        obj = mem.remember(f"{name} object", type="concept", valence=0.0)
        mem.atomspace.link_atoms(subject, obj, "related_to", T, S, metadata=metadata)
        ids[name] = (subject, obj)
    _ripe_for_pruning(mem, *(atom_id for pair in ids.values() for atom_id in pair))

    mem.reflect()

    for subject, obj in ids.values():
        assert _row(mem, subject) is None and _row(mem, obj) is None


# ── epochs write only what changes ──────────────────────────────────────────

def _graph(path):
    """A small linked graph with active sources, a twice-linked neighbour, a
    relation source with tone, and a bridge edge leaving the space."""
    m = Smrti(db_path=path, tenant_id=T, write_space=S)
    hub = m.remember("hub", type="concept", valence=-0.8)
    a = m.remember("neighbour a", type="concept", valence=0.2)
    b = m.remember("neighbour b", type="concept", valence=0.0)
    c = m.remember("neighbour c", type="concept", valence=0.5)
    outside = Smrti(db_path=path, tenant_id=T, write_space="elsewhere").remember(
        "outside", type="concept", valence=0.0
    )
    m.atomspace.link_atoms(hub, a, "associated", T, S)
    m.atomspace.link_atoms(hub, b, "associated", T, S)
    m.atomspace.link_atoms(b, hub, "mentions", T, S)
    m.atomspace.link_atoms(c, a, "associated", T, S)
    claim = m.atomspace.link_atoms(c, b, "likes", T, S, valence=-0.9)
    m.atomspace.link_atoms(hub, outside, "bridge", T, S)
    m.db.execute(
        "UPDATE atoms SET sti = 2.5, intensity = 0.9 WHERE id IN (?, ?)", (hub, c)
    )
    m.db.execute("UPDATE atoms SET sti = 0.9, intensity = 0.9 WHERE id = ?", (claim,))
    return m, {"hub": hub, "a": a, "b": b, "c": c, "outside": outside, "claim": claim}


def test_batched_propagation_matches_the_per_atom_loop(tmp_path):
    """Propagation now commits once per epoch; the values must land exactly
    where the per-atom loop put them."""
    batched, batched_ids = _graph(str(tmp_path / "batched.db"))
    looped, looped_ids = _graph(str(tmp_path / "looped.db"))
    p = personality_params(batched.db.fetchone("SELECT * FROM personality WHERE space = ?", (S,)))

    def values(m, ids):
        out = {}
        for name, atom_id in ids.items():
            row = m.db.fetchone("SELECT sti, valence, intensity FROM atoms WHERE id = ?", (atom_id,))
            out.update({f"{name}.{column}": row[column] for column in row.keys()})
        return out

    before = values(looped, looped_ids)
    _propagate(T, S, batched.db, p)
    active = looped.db.fetchall(
        "SELECT id, sti, valence, intensity FROM atoms WHERE tenant_id = ? AND space = ? "
        "AND (sti > 0.3 OR (ABS(valence) > 0.3 AND intensity > 0.3))",
        (T, S),
    )
    for row in active:
        if row["sti"] > 0.3:
            propagate_sti(row["id"], row["sti"], p["sti_propagation_factor"], looped.db, T, S)
        if abs(row["valence"]) > 0.3 and row["intensity"] > 0.3:
            propagate_valence(
                row["id"], row["valence"], row["intensity"], p["valence_propagation"],
                looped.db, T, S, mood_inertia=p["mood_inertia"],
            )

    expected = values(looped, looped_ids)
    assert expected != before
    assert values(batched, batched_ids) == pytest.approx(expected)


def test_propagation_commits_once_per_epoch(mem):
    """The per-atom loop committed a transaction per active atom."""
    sources = [mem.remember(f"active {i}", type="concept", valence=-0.8) for i in range(5)]
    sink = mem.remember("sink", type="concept", valence=0.0)
    for source in sources:
        mem.atomspace.link_atoms(source, sink, "associated", T, S)
    mem.db.execute("UPDATE atoms SET sti = 1.0, intensity = 0.9 WHERE label LIKE 'active%'")
    p = personality_params(mem.db.fetchone("SELECT * FROM personality WHERE space = ?", (S,)))

    with patch.object(mem.db, "transaction", wraps=mem.db.transaction) as transaction, \
            patch.object(mem.db, "execute_batch", wraps=mem.db.execute_batch) as batch, \
            patch.object(mem.db, "execute_many", wraps=mem.db.execute_many) as many, \
            patch.object(mem.db, "execute", wraps=mem.db.execute) as single:
        _propagate(T, S, mem.db, p)

    assert transaction.call_count == 1
    assert batch.call_count == many.call_count == single.call_count == 0


def test_decay_leaves_settled_rows_alone(mem):
    """Decay rewrote every row in the space each epoch, including rows whose
    values could no longer change."""
    settled = mem.remember("an old memory", valence=0.0)
    floor = mem.db.fetchone(
        "SELECT min_confidence_to_surface f FROM personality WHERE space = ?", (S,)
    )["f"]
    mem.db.execute(
        "UPDATE atoms SET sti = 0.0, lti = 0.0, confidence = ?, updated_at = '2000-01-01 00:00:00' WHERE id = ?",
        (floor, settled),
    )
    moving = mem.remember("a fresh memory", valence=0.0)

    mem.reflect()

    assert _row(mem, settled)["updated_at"] == "2000-01-01 00:00:00"
    assert _row(mem, moving)["confidence"] < 0.5


# ── smaller items ───────────────────────────────────────────────────────────

def test_a_stated_reason_does_not_raise_a_beliefs_confidence(mem):
    """believe(evidence=...) filed the assertion again as evidence, so giving
    a reason raised a new belief's confidence from 0.3 to 0.42."""
    plain = mem.believe("The user prefers green tea", 0.8, valence=0.0)
    reasoned = mem.believe("The user prefers oolong", 0.8, evidence="said so", valence=0.0)

    mem.reflect()

    assert _row(mem, reasoned)["confidence"] == pytest.approx(_row(mem, plain)["confidence"])
    assert mem.evidence(reasoned)[0].text == "said so"


def test_a_twice_linked_neighbour_is_nudged_once():
    """An atom linked twice to the source was nudged twice per epoch."""
    params = nudge_params(-0.8, 0.9, 0.5, ["n1", "n1", "n2"], T, S)
    assert [p[4] for p in params] == ["n1", "n2"]


def test_clear_space_removes_atoms_written_while_it_runs(mem):
    """The final sweep deleted atoms written after the id read without their
    vectors and lexical rows."""
    mem.remember("first", valence=0.0)
    original = mem.atomspace.delete_atoms
    late = []

    def write_meanwhile(ids, tenant_id):
        removed = original(ids, tenant_id)
        if not late:
            late.append(mem.remember("written while clearing", valence=0.0))
        return removed

    with patch.object(mem.atomspace, "delete_atoms", side_effect=write_meanwhile):
        assert mem.clear_space() == 2

    assert mem.db.fetchone("SELECT COUNT(*) n FROM atoms WHERE space = ?", (S,))["n"] == 0
    assert mem.db.fetchone(
        "SELECT COUNT(*) n FROM vec_atoms WHERE atom_id = ?", (late[0],)
    )["n"] == 0
