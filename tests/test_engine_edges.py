"""Edge paths of the engine changes: migrations, fallbacks and skips.

Companion to ``test_engine_claims.py``, which replays the headline claims;
this file walks the branches those claims do not reach — an old database
being upgraded, a migration that fails halfway, an alias that is a pronoun,
a supersession that names nothing the graph holds.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import types
from unittest.mock import MagicMock, patch

import pytest

import smrti.core.embed as embed_module
import smrti.servers.reflect_loop as reflect_loop_module
from smrti import Smrti
from smrti.core.db import Database, _make_connection, close_database, get_database
from smrti.extraction.aliases import AliasManager
from smrti.extraction.extract import _existing_temporal, _link_claims, _store_temporal
from smrti.extraction.resolve import EntityResolver
from smrti.evolution.healing import heal_orphaned_episodes
from smrti.servers.reflect_loop import run_reflect_loop
from tests.test_engine_claims import _BagOfWords


@pytest.fixture(autouse=True)
def bag_of_words_embedder(monkeypatch):
    monkeypatch.setattr(embed_module.EmbeddingProvider, "_get_model", lambda self: _BagOfWords())


@pytest.fixture
def mem(tmp_path):
    return Smrti(db_path=str(tmp_path / "edges.db"), tenant_id="t", write_space="s")


def _row(mem, atom_id):
    return mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (atom_id,))


def _contradictions(mem):
    return mem.db.fetchall(
        "SELECT source_id, target_id FROM atoms WHERE type = 'relation' AND relation = 'contradicts'"
    )


# ── the evidence log of an older database ─────────────────────────────────────

_OLD_EVIDENCE_COLUMNS = (
    "id, atom_id, observed_probability, weight, source_episode_id, "
    "tenant_id, space, created_at, processed"
)


def _old_format_db(path: str) -> None:
    """A database whose evidence table predates the text and source columns."""
    get_database(path)
    close_database(path)
    conn = sqlite3.connect(path)
    conn.executescript(
        f"""CREATE TABLE evidence_old AS SELECT {_OLD_EVIDENCE_COLUMNS} FROM evidence;
            DROP TABLE evidence;
            ALTER TABLE evidence_old RENAME TO evidence;"""
    )
    conn.commit()
    conn.close()


def _evidence_columns(path: str) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(evidence)")}
    finally:
        conn.close()


def test_an_older_evidence_log_gains_its_columns_on_open(tmp_path):
    path = str(tmp_path / "old.db")
    _old_format_db(path)
    assert not {"text", "source"} & _evidence_columns(path)

    get_database(path)

    assert {"text", "source"} <= _evidence_columns(path)
    assert (tmp_path / "old.db.pre-migration.bak").exists()
    # And the upgraded log takes the new fields.
    mem = Smrti(db_path=path, tenant_id="t", write_space="s")
    atom_id = mem.believe("cats are mammals", probability=0.8, evidence="a survey", valence=0.0)
    assert mem.evidence(atom_id)[0].text == "a survey"


class _RefusingAlter:
    """A connection that refuses ALTER TABLE, so the migration must roll back."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, *args):
        if sql.lstrip().upper().startswith("ALTER TABLE EVIDENCE"):
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_a_failed_evidence_migration_rolls_back_and_raises(tmp_path):
    path = str(tmp_path / "old.db")
    _old_format_db(path)
    db = Database(path)
    real = _make_connection(path)
    db._write_conn = _RefusingAlter(real)
    db._migration_backup_done = True

    with pytest.raises(sqlite3.OperationalError):
        db._migrate_evidence_columns()

    assert not real.in_transaction
    real.close()
    assert not {"text", "source"} & _evidence_columns(path)


# ── rows an earlier release wrote are repaired on open ────────────────────────


def _legacy_entity(conn, atom_id, label, entity_type="technology", tenant="t", space="s"):
    """An extracted entity as the old resolver wrote it: no content, no intrinsic tone."""
    conn.execute(
        """INSERT INTO atoms (id, type, label, entity_type, tenant_id, space,
                              probability, confidence, sti, lti, valence, intensity)
           VALUES (?, 'concept', ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3, ?, ?)""",
        (atom_id, label, entity_type, tenant, space, -0.45, 0.45),  # drifted mood
    )


def _legacy_edge(conn, edge_id, source, target, relation, valence=0.0, confidence=0.5):
    conn.execute(
        """INSERT INTO atoms (id, type, label, source_id, target_id, relation, tenant_id, space,
                              probability, confidence, valence, intensity)
           VALUES (?, 'relation', ?, ?, ?, ?, 't', 's', 0.5, ?, ?, ?)""",
        (edge_id, relation, source, target, relation, confidence, valence, abs(valence)),
    )


def _reopen(path):
    close_database(path)
    return get_database(path)


def test_an_old_extracted_entity_gets_the_tone_its_claims_gave_it(tmp_path):
    path = str(tmp_path / "old.db")
    get_database(path)
    close_database(path)
    conn = sqlite3.connect(path)
    _legacy_entity(conn, "build", "build times")
    _legacy_entity(conn, "dave", "Dave", entity_type="person")
    _legacy_entity(conn, "cake", "cake")
    _legacy_edge(conn, "e1", "dave", "build", "dislikes", valence=-0.6)
    _legacy_edge(conn, "e2", "dave", "build", "tolerates", valence=-0.2)  # not a lowering claim
    _legacy_edge(conn, "e3", "dave", "cake", "mentions", valence=-0.9)  # structural, ignored
    conn.commit()
    conn.close()

    db = _reopen(path)

    build = db.fetchone("SELECT intrinsic_valence, intrinsic_intensity, valence FROM atoms WHERE id = 'build'")
    assert build["intrinsic_valence"] == pytest.approx(-0.6)
    assert build["intrinsic_intensity"] == pytest.approx(0.6)
    assert build["valence"] == pytest.approx(-0.45)  # the mood is left as it was
    cake = db.fetchone("SELECT intrinsic_valence, intrinsic_intensity FROM atoms WHERE id = 'cake'")
    assert (cake["intrinsic_valence"], cake["intrinsic_intensity"]) == (0.0, 0.0)
    assert (tmp_path / "old.db.pre-migration.bak").exists()


def test_an_old_episode_keeps_reading_as_it_always_did(tmp_path):
    """Nothing records the tone a pre-split episode was stored with."""
    path = str(tmp_path / "old.db")
    get_database(path)
    close_database(path)
    conn = sqlite3.connect(path)
    conn.execute(
        """INSERT INTO atoms (id, type, label, content, tenant_id, space, probability, confidence, valence, intensity)
           VALUES ('ep', 'episode', 'a note', 'a note', 't', 's', 0.8, 0.5, -0.4, 0.4)"""
    )
    conn.commit()
    conn.close()

    db = _reopen(path)

    row = db.fetchone("SELECT intrinsic_valence FROM atoms WHERE id = 'ep'")
    assert row["intrinsic_valence"] is None


def test_old_healing_hub_edges_are_removed_and_real_claims_kept(tmp_path):
    path = str(tmp_path / "old.db")
    get_database(path)
    close_database(path)
    conn = sqlite3.connect(path)
    _legacy_entity(conn, "alice", "Alice", entity_type="person")
    _legacy_entity(conn, "python", "Python")
    _legacy_entity(conn, "chess", "chess")
    _legacy_edge(conn, "hub", "alice", "python", "associated", confidence=0.2)
    _legacy_edge(conn, "claim", "alice", "chess", "associated", confidence=0.9)
    _legacy_edge(conn, "para", "python", "chess", "associated", confidence=0.1)  # not from a person
    conn.commit()
    conn.close()

    db = _reopen(path)

    remaining = {r["id"] for r in db.fetchall("SELECT id FROM atoms WHERE type = 'relation'")}
    assert remaining == {"claim", "para"}


def test_atoms_forgotten_by_the_old_forget_are_sunk_below_the_floor(tmp_path):
    path = str(tmp_path / "old.db")
    mem = Smrti(db_path=path, tenant_id="t", write_space="s")
    atom_id = mem.remember("the vault password rotates weekly", valence=0.0)
    # The old forget: confidence * 0.3 and the stamp, leaving 0.15 above the 0.1 floor.
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.15, metadata = '{\"forgotten\": true}' WHERE id = ?",
        (atom_id,),
    )

    db = _reopen(path)

    assert db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (atom_id,))["confidence"] == pytest.approx(0.05)
    mem = Smrti(db_path=path, tenant_id="t", write_space="s")
    mem.reflect()
    assert mem.db.fetchone("SELECT 1 FROM atoms WHERE id = ?", (atom_id,)) is None


class _RefusingRepairWrite:
    """A connection that refuses the repair's first write, so it must roll back."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, *args):
        if "SET" in sql and "intrinsic_valence = COALESCE" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_a_failed_repair_rolls_back_every_change_and_raises(tmp_path):
    path = str(tmp_path / "old.db")
    get_database(path)
    close_database(path)
    conn = sqlite3.connect(path)
    _legacy_entity(conn, "alice", "Alice", entity_type="person")
    _legacy_entity(conn, "python", "Python")
    _legacy_edge(conn, "hub", "alice", "python", "associated", confidence=0.2)
    conn.commit()
    conn.close()
    db = Database(path)
    real = _make_connection(path)
    db._write_conn = _RefusingRepairWrite(real)
    db._migration_backup_done = True

    with pytest.raises(sqlite3.OperationalError):
        db._repair_legacy_rows()

    assert not real.in_transaction
    real.close()
    check = sqlite3.connect(path)
    try:
        assert check.execute("SELECT intrinsic_valence FROM atoms WHERE id = 'python'").fetchone()[0] is None
        assert check.execute("SELECT COUNT(*) FROM atoms WHERE id = 'hub'").fetchone()[0] == 1
    finally:
        check.close()


def test_repairs_do_nothing_on_a_file_with_no_graph_yet(tmp_path):
    path = str(tmp_path / "empty.db")
    db = Database(path)
    db._write_conn = _make_connection(path)

    db._repair_legacy_rows()

    assert not (tmp_path / "empty.db.pre-migration.bak").exists()
    db._write_conn.close()


def test_repairs_are_idempotent_and_leave_a_current_graph_untouched(tmp_path):
    path = str(tmp_path / "current.db")
    mem = Smrti(db_path=path, tenant_id="t", write_space="s")
    mem.remember("Alice prefers TypeScript", valence=0.3)
    EntityResolver(mem.db, mem.embed).resolve("TypeScript", "technology", "t", "s", ["s"])
    before = mem.db.fetchall("SELECT id, confidence, intrinsic_valence FROM atoms ORDER BY id")

    _reopen(path)
    _reopen(path)

    after = get_database(path).fetchall("SELECT id, confidence, intrinsic_valence FROM atoms ORDER BY id")
    assert [tuple(r) for r in after] == [tuple(r) for r in before]
    assert not (tmp_path / "current.db.pre-migration.bak").exists()


# ── healing recognises aliases, never pronouns ────────────────────────────────


def _person(mem, name):
    return EntityResolver(mem.db, mem.embed).resolve(name, "person", "t", "s", ["s"])


def _orphan(mem, text):
    concept = EntityResolver(mem.db, mem.embed).resolve("hiking", "topic", "t", "s", ["s"])
    episode = mem.remember(text, valence=0.0)
    mem.atomspace.link_atoms(episode, concept, "mentions", "t", "s")
    return episode


def _mentioned_persons(mem, episode):
    rows = mem.db.fetchall(
        """SELECT p.id FROM atoms r JOIN atoms p ON p.id = r.target_id
           WHERE r.type = 'relation' AND r.relation = 'mentions' AND r.source_id = ?
             AND p.entity_type = 'person'""",
        (episode,),
    )
    return {r["id"] for r in rows}


def test_healing_recognises_a_person_by_alias_but_never_by_pronoun(mem):
    alice = _person(mem, "Alice")
    bob = _person(mem, "Bob")
    aliases = AliasManager(mem.db)
    aliases.add(alice, "Ali", "t", "s")
    aliases.add(alice, "I", "t", "s")  # the resolver registers these for the speaker
    aliases.add(bob, "my", "t", "s")

    by_alias = _orphan(mem, "went hiking with Ali on Sunday")
    by_pronoun_only = _orphan(mem, "I went hiking and my knees hurt")
    both_named = _orphan(mem, "Alice and Bob went hiking")

    healed = heal_orphaned_episodes("t", "s", mem.db)

    assert healed == 1
    assert _mentioned_persons(mem, by_alias) == {alice}
    assert _mentioned_persons(mem, by_pronoun_only) == set()
    assert _mentioned_persons(mem, both_named) == set()


# ── temporal merge edge cases ─────────────────────────────────────────────────


def test_temporal_of_an_unknown_episode_is_empty_and_storing_to_it_is_harmless(mem):
    assert _existing_temporal("no-such-atom", mem) == []
    _store_temporal("no-such-atom", mem, [{"text": "tomorrow", "resolved": "2026-09-04"}])


def test_unreadable_temporal_metadata_reads_as_none(mem):
    episode = mem.remember("the retro is tomorrow", valence=0.0)
    mem.db.execute("UPDATE atoms SET metadata = 'not json' WHERE id = ?", (episode,))
    assert _existing_temporal(episode, mem) == []

    mem.db.execute("UPDATE atoms SET metadata = ? WHERE id = ?",
                   (json.dumps({"temporal": "tomorrow"}), episode))
    assert _existing_temporal(episode, mem) == []


def test_storing_the_same_resolutions_twice_writes_once(mem):
    episode = mem.remember("the retro is tomorrow", valence=0.0)
    items = [{"text": "tomorrow", "resolved": "2026-09-04"}]
    _store_temporal(episode, mem, items)
    first = _row(mem, episode)["metadata"]

    _store_temporal(episode, mem, items)

    assert _row(mem, episode)["metadata"] == first
    assert json.loads(first)["temporal"] == items


# ── supersession that names nothing, or names it differently ──────────────────


def _alice_in(mem, place, predicate="lives_in"):
    resolver = EntityResolver(mem.db, mem.embed)
    alice = resolver.resolve("Alice", "person", "t", "s", ["s"])
    where = resolver.resolve(place, "location", "t", "s", ["s"])
    mem.atomspace.link_atoms(alice, where, predicate, "t", "s")
    return alice, where


def _move(mem, alice, to, supersedes, entity_ids):
    episode = mem.remember(f"I moved to {to}", valence=0.0)
    _link_claims(
        [{"subject": "Alice", "predicate": "lives_in", "object": to, "supersedes": supersedes}],
        entity_ids, mem, episode_id=episode,
    )


def test_superseding_a_place_the_graph_never_recorded_changes_nothing(mem):
    alice, amsterdam = _alice_in(mem, "Amsterdam")
    _move(mem, alice, "Berlin", "Atlantis", {"Alice": alice, "Amsterdam": amsterdam})
    assert _contradictions(mem) == []


def test_a_claim_cannot_supersede_its_own_object(mem):
    alice, amsterdam = _alice_in(mem, "Amsterdam")
    _move(mem, alice, "Amsterdam", "Amsterdam", {"Alice": alice, "Amsterdam": amsterdam})
    assert _contradictions(mem) == []


def test_superseding_a_place_with_no_claim_edge_changes_nothing(mem):
    resolver = EntityResolver(mem.db, mem.embed)
    alice = resolver.resolve("Alice", "person", "t", "s", ["s"])
    amsterdam = resolver.resolve("Amsterdam", "location", "t", "s", ["s"])  # known, unlinked
    _move(mem, alice, "Berlin", "Amsterdam", {"Alice": alice, "Amsterdam": amsterdam})
    assert _contradictions(mem) == []


def test_supersession_finds_the_old_claim_under_a_different_predicate(mem):
    alice, amsterdam = _alice_in(mem, "Amsterdam", predicate="resides_in")
    old_edge = mem.db.fetchone(
        "SELECT id FROM atoms WHERE type = 'relation' AND relation = 'resides_in'"
    )["id"]

    _move(mem, alice, "Berlin", "Amsterdam", {"Alice": alice, "Amsterdam": amsterdam})

    [edge] = _contradictions(mem)
    assert edge["target_id"] == old_edge
    assert json.loads(_row(mem, old_edge)["metadata"])["superseded_by"] == edge["source_id"]


# ── the resolver's unknown type ───────────────────────────────────────────────


def test_an_unknown_entity_type_becomes_a_concept(mem):
    atom_id = EntityResolver(mem.db, mem.embed)._create_atom("Thing", "gadget", "t", "s")
    row = _row(mem, atom_id)
    assert row["type"] == "concept"
    assert row["entity_type"] == "concept"
    assert row["intrinsic_valence"] == 0.0


# ── the reflect loop skips what nobody used ───────────────────────────────────


def test_the_loop_reflects_only_the_instances_that_were_used():
    idle = types.SimpleNamespace(tenant_id="t", write_space="idle", ops_since_reflect=0,
                                 reflect=MagicMock())
    busy = types.SimpleNamespace(tenant_id="t", write_space="busy", ops_since_reflect=3,
                                 reflect=MagicMock())
    ticks = [0]

    async def fake_sleep(_):
        ticks[0] += 1
        if ticks[0] >= 2:
            raise asyncio.CancelledError

    async def _run():
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await run_reflect_loop(lambda: [idle, busy])

    with patch.object(reflect_loop_module, "REFLECT_INTERVAL", 1):
        try:
            asyncio.run(_run())
        except asyncio.CancelledError:
            pass

    assert busy.reflect.called
    assert not idle.reflect.called
