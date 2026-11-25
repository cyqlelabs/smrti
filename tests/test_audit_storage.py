"""Storage regressions from the audit: indexes, statistics, repairs, deletes.

Each test pins one defect the audit reproduced in ``core/db.py``,
``core/atomspace.py`` or ``core/models.py``.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid

import pytest

import smrti.core.embed as embed_module
import smrti.core.models as models_module
from smrti import Smrti
from smrti.core.atomspace import AtomSpace
from smrti.core.db import Database, _make_connection, close_database, get_database
from smrti.core.models import Atom, AtomType, EntityType, TruthValue, atom_from_row
from tests.test_engine_claims import _BagOfWords


@pytest.fixture(autouse=True)
def bag_of_words_embedder(monkeypatch):
    monkeypatch.setattr(embed_module.EmbeddingProvider, "_get_model", lambda self: _BagOfWords())


class _Stub:
    def embed(self, text):
        return [1.0] + [0.0] * 383


def _plan(conn, sql, params):
    return " | ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params))


def _episodes(db, tenant, space, n):
    db.execute_many(
        "INSERT INTO atoms (id, type, label, tenant_id, space) VALUES (?, 'episode', ?, ?, ?)",
        [(str(uuid.uuid4()), f"note {i}", tenant, space) for i in range(n)],
    )


def test_an_edge_walk_inside_a_space_uses_the_edge_indexes(tmp_path):
    """Propagation looked edges up through a (tenant_id, space) index and
    scanned the whole space per atom: 116 s per epoch at twenty thousand
    atoms. The lookup and link_atoms' existence check must use the edge
    indexes, and the unread sti index must be gone."""
    db = get_database(str(tmp_path / "plans.db"))
    with db._read_conn() as conn:
        walk = _plan(
            conn,
            "SELECT target_id FROM atoms WHERE source_id = ? AND type = 'relation' "
            "AND tenant_id = ? AND space = ?",
            ("a", "t", "s"),
        )
        back = _plan(
            conn,
            "SELECT source_id FROM atoms WHERE target_id = ? AND type = 'relation' "
            "AND tenant_id = ? AND space = ?",
            ("a", "t", "s"),
        )
        link = _plan(
            conn,
            "SELECT id FROM atoms WHERE type = 'relation' AND source_id = ? AND target_id = ? "
            "AND relation = ? AND tenant_id = ? AND space = ?",
            ("a", "b", "r", "t", "s"),
        )
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}

    assert "idx_atoms_edge_out (tenant_id=? AND space=? AND source_id=?)" in walk
    assert "idx_atoms_edge_in (tenant_id=? AND space=? AND target_id=?)" in back
    assert "AND target_id=? AND relation=?" in link
    assert "idx_atoms_sti" not in indexes


def test_readers_plan_with_the_statistics_optimize_gathers(tmp_path):
    """No ANALYZE ever ran, so ``id IN (...)`` within a space scanned the space
    instead of probing the primary key; and statistics gathered on the write
    connection never reached the pooled readers, which loaded the schema
    before them."""
    path = str(tmp_path / "stats.db")
    db = get_database(path)
    _episodes(db, "t", "s", 300)
    _episodes(db, "t", "other", 300)
    db.execute("DROP TABLE IF EXISTS sqlite_stat1")
    ids = [r["id"] for r in db.fetchall("SELECT id FROM atoms WHERE space = 's' LIMIT 5")]
    sql = f"SELECT * FROM atoms WHERE id IN ({','.join('?' * 5)}) AND tenant_id = ? AND space = ?"

    close_database(path)
    db = get_database(path)
    assert db.fetchone("SELECT 1 FROM sqlite_stat1 WHERE tbl = 'atoms'") is not None
    with db._read_conn() as conn:
        assert "sqlite_autoindex_atoms_1 (id=?)" in _plan(conn, sql, (*ids, "t", "s"))

    # Statistics a later ANALYZE writes must reach readers that already loaded the schema.
    _episodes(db, "t", "s", 300)
    for _ in range(4):
        db.fetchall(sql, (*ids, "t", "s"))
    db.execute("ANALYZE")
    db.optimize()

    for _ in range(4):
        with db._read_conn() as conn:
            assert "sqlite_autoindex_atoms_1 (id=?)" in _plan(conn, sql, (*ids, "t", "s"))


def test_a_database_another_process_is_writing_still_opens(tmp_path, caplog):
    """Statistics need the write lock, and an open that had to gather them
    while another process held it failed instead of opening."""
    path = str(tmp_path / "busy.db")
    db = get_database(path)
    _episodes(db, "t", "s", 50)
    db.execute("DROP TABLE sqlite_stat1")
    db._write_conn.execute("PRAGMA busy_timeout=50")
    other = sqlite3.connect(path)
    other.execute("BEGIN IMMEDIATE")
    try:
        with caplog.at_level(logging.WARNING, logger="smrti.core.db"):
            db.optimize()
    finally:
        other.rollback()
        other.close()

    assert any("not refreshed" in r.message for r in caplog.records)
    assert db.fetchone("SELECT COUNT(*) AS n FROM atoms")["n"] == 50


def _person_with_discovery_edge(path):
    sp = AtomSpace(get_database(path), _Stub())
    alice = sp.add_atom(Atom(type=AtomType.CONCEPT, label="Alice", entity_type=EntityType.PERSON,
                             tenant_id="t", space="s"))
    chess = sp.add_atom(Atom(type=AtomType.CONCEPT, label="chess", tenant_id="t", space="s"))
    # What association discovery writes today, out of a person or not.
    return sp.link_atoms(alice, chess, "associated", "t", "s",
                         truth=TruthValue(probability=0.5, confidence=0.1))


def test_reopening_keeps_discovery_edges_and_the_downgrade_backup(tmp_path):
    """Every open deleted the ``associated`` edges discovery draws at 0.1 out
    of a person, and each such open replaced ``.pre-migration.bak`` with a
    copy of the already-upgraded file."""
    path = str(tmp_path / "live.db")
    edge = _person_with_discovery_edge(path)
    backup = path + ".pre-migration.bak"
    with open(backup, "w") as f:
        f.write("the snapshot an earlier migration took")
    os.utime(backup, (1_000_000_000, 1_000_000_000))
    close_database(path)
    conn = _make_connection(path)
    # A row in a shape only older code writes, so a repair runs on open.
    conn.execute(
        "INSERT INTO atoms (id, type, label, tenant_id, space) VALUES ('old', 'concept', 'old', 't', 's')"
    )
    conn.commit()
    conn.close()

    for _ in range(2):
        close_database(path)
        db = get_database(path)

    assert db.fetchone("SELECT 1 FROM atoms WHERE id = ?", (edge,)) is not None
    assert db.fetchone("SELECT intrinsic_valence FROM atoms WHERE id = 'old'")["intrinsic_valence"] == 0.0
    assert os.path.getmtime(backup) == 1_000_000_000


def test_the_one_time_hub_cleanup_takes_what_leans_on_a_hub_edge(tmp_path):
    """The cleanup deleted hub edges by hand, so a ``contradicts`` edge or
    evidence filed against one failed the foreign-key check and the file
    could not be opened again."""
    path = str(tmp_path / "old.db")
    db = get_database(path)
    sp = AtomSpace(db, _Stub())
    alice = sp.add_atom(Atom(type=AtomType.CONCEPT, label="Alice", entity_type=EntityType.PERSON,
                             tenant_id="t", space="s"))
    python = sp.add_atom(Atom(type=AtomType.CONCEPT, label="Python", tenant_id="t", space="s"))
    hub = sp.link_atoms(alice, python, "associated", "t", "s",
                        truth=TruthValue(probability=0.5, confidence=0.2))
    claim = sp.link_atoms(alice, python, "works_with", "t", "s")
    on_hub = sp.link_atoms(claim, hub, "contradicts", "t", "s")
    db.execute(
        "INSERT INTO evidence (id, atom_id, observed_probability, tenant_id, space) VALUES ('ev', ?, 0.1, 't', 's')",
        (hub,),
    )
    db.execute("DELETE FROM smrti_meta WHERE key = ?", (Database._HUB_CLEANUP_KEY,))
    close_database(path)

    db = get_database(path)

    relations = {r["id"] for r in db.fetchall("SELECT id FROM atoms WHERE type = 'relation'")}
    assert relations == {claim}
    assert on_hub not in relations
    assert db.meta_get(Database._HUB_CLEANUP_KEY) == "1"


def test_delete_atoms_ignores_another_tenants_ids(tmp_path):
    """The closure was seeded with whatever ids it was given, so a delete in
    one tenant removed another tenant's atom."""
    mem = Smrti(db_path=str(tmp_path / "tenants.db"), tenant_id="owner", write_space="s")
    victim = mem.remember("the owner's private note")

    assert mem.atomspace.delete_atoms([victim], "outsider") == 0
    assert mem.db.fetchone("SELECT 1 FROM atoms WHERE id = ?", (victim,)) is not None


def test_no_edge_can_be_committed_between_a_delete_and_its_closure(tmp_path, monkeypatch):
    """The closure was read through the read pool before the write lock, and an
    edge committed in between pointed at a deleted atom: the commit failed
    its foreign-key check."""
    path = str(tmp_path / "race.db")
    db = get_database(path)
    sp = AtomSpace(db, _Stub())
    doomed = sp.add_atom(Atom(type=AtomType.CONCEPT, label="doomed", tenant_id="t", space="s"))
    other = sp.add_atom(Atom(type=AtomType.CONCEPT, label="other", tenant_id="t", space="s"))
    outcome = {}
    original = Database._cascade_delete

    def racing(self, conn, atom_ids, tenant_id):
        writer = _make_connection(path)
        writer.execute("PRAGMA busy_timeout=0")
        try:
            writer.execute(
                "INSERT INTO atoms (id, type, label, source_id, target_id, relation, tenant_id, space) "
                "VALUES ('late', 'relation', 'late', ?, ?, 'mentions', 't', 's')",
                (other, doomed),
            )
            writer.commit()
            outcome["late_edge"] = "committed"
        except sqlite3.OperationalError:
            outcome["late_edge"] = "locked out"
        finally:
            writer.close()
        return original(self, conn, atom_ids, tenant_id)

    monkeypatch.setattr(Database, "_cascade_delete", racing)

    assert sp.delete_atoms([doomed], "t") == 1
    assert outcome["late_edge"] == "locked out"
    assert db.fetchone("SELECT 1 FROM atoms WHERE id = ?", (doomed,)) is None


def test_an_in_memory_database_reads_back_what_it_wrote(monkeypatch):
    """Each pooled connection to ``:memory:`` opened a private empty database,
    so the first read after a write found no such table."""
    db = get_database(":memory:")
    sp = AtomSpace(db, _Stub())

    atom_id = sp.add_atom(Atom(type=AtomType.CONCEPT, label="kept", tenant_id="t", space="s"))

    assert sp.get_atom(atom_id, "t", "s").label == "kept"
    close_database(":memory:")


def test_connections_sync_at_normal(tmp_path):
    """Under WAL, FULL synced the log on every commit for no added safety."""
    db = get_database(str(tmp_path / "sync.db"))

    assert db._write_conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_an_unreadable_metadata_row_does_not_abort_recall(tmp_path, monkeypatch, caplog):
    """One metadata value that is not a JSON object raised out of every recall
    that reached the atom."""
    monkeypatch.setattr(models_module, "_unreadable_metadata_logged", False)
    mem = Smrti(db_path=str(tmp_path / "meta.db"), tenant_id="t", write_space="s")
    mem.remember("the build server lives in the basement")
    bad = [mem.remember(f"the build server note {i}") for i in range(3)]
    for atom_id, raw in zip(bad, ("not json", "[1, 2]", '"text"')):
        mem.db.execute("UPDATE atoms SET metadata = ? WHERE id = ?", (raw, atom_id))

    with caplog.at_level(logging.WARNING, logger="smrti.core.models"):
        results = mem.recall("build server", top_k=10, boost=False)
        mem.recall("build server", top_k=10, boost=False)

    returned = {r.atom.id: r.atom.metadata for r in results}
    assert all(returned.get(atom_id) == {} for atom_id in bad)
    assert len([r for r in caplog.records if "not a JSON object" in r.message]) == 1
    assert atom_from_row(
        {"id": "x", "type": "concept", "label": "x", "metadata": None}
    ).metadata == {}


def test_forget_on_an_in_memory_database_does_not_wait_on_itself():
    """forget() read the surfacing floor inside its own write transaction,
    and in memory a read takes the write lock: the call never returned."""
    from smrti import Smrti

    mem = Smrti(db_path=":memory:")
    atom_id = mem.remember("The spare key is under the blue pot")
    assert mem.forget(atom_ids=[atom_id], cascade=True) == ["The spare key is under the blue pot"]
    with mem.db.transaction():
        assert mem.db.fetchone("SELECT COUNT(*) AS n FROM atoms")["n"] == 1
