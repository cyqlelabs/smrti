"""Coverage tests for the Database registry, rollbacks and teardown (core/db.py).

`test_db_migration.py` covers the happy-path migrations; this module covers
the failure paths — every write helper rolls back before re-raising — plus
registry bookkeeping and connection teardown.
"""
from __future__ import annotations

import contextlib
import os
import queue
import sqlite3
import struct
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from smrti.core.db import (
    Database,
    _make_connection,
    _registry,
    _resolve_path,
    clear_registry,
    close_database,
    get_database,
)


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    for suffix in ("", "-wal", "-shm", ".pre-migration.bak"):
        try:
            os.unlink(path + suffix)
        except OSError:
            pass


@pytest.fixture
def db(db_path):
    database = get_database(db_path)
    yield database
    close_database(db_path)


# ── registry ──────────────────────────────────────────────────────────────────

def test_resolve_path_leaves_memory_databases_alone():
    assert _resolve_path(":memory:") == ":memory:"


def test_resolve_path_expands_and_normalizes(db_path):
    aliased = os.path.join(os.path.dirname(db_path), ".", os.path.basename(db_path))
    assert _resolve_path(aliased) == _resolve_path(db_path)


def test_get_database_shares_one_instance_per_path(db, db_path):
    assert get_database(db_path) is db


def test_close_database_is_a_no_op_for_an_unregistered_path(tmp_path):
    close_database(str(tmp_path / "never-opened.db"))  # must not raise


def test_clear_registry_closes_every_open_database(db_path):
    database = get_database(db_path)
    assert _resolve_path(db_path) in _registry
    clear_registry()
    assert _registry == {}
    assert database._write_conn is None


# ── write helpers roll back before re-raising ─────────────────────────────────

class _RecordingConn:
    """Delegates to a real connection while counting rollbacks.

    sqlite3.Connection is a C type whose methods cannot be patched, so the
    connection is wrapped rather than monkeypatched.
    """

    def __init__(self, conn, **overrides):
        self._conn = conn
        self._overrides = overrides
        self.rollbacks = 0

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._conn, name)

    def rollback(self):
        self.rollbacks += 1
        return self._conn.rollback()


@contextlib.contextmanager
def _wrapped_write_conn(database, **overrides):
    real = database._write_conn
    proxy = _RecordingConn(real, **overrides)
    database._write_conn = proxy
    try:
        yield proxy
    finally:
        database._write_conn = real


def test_execute_rolls_back_on_error(db):
    with _wrapped_write_conn(db) as conn:
        with pytest.raises(sqlite3.Error):
            db.execute("INSERT INTO atoms (id, nope) VALUES (?, ?)", ("a", "b"))
        assert conn.rollbacks == 1


def test_execute_many_rolls_back_on_error(db):
    with _wrapped_write_conn(db) as conn:
        with pytest.raises(sqlite3.Error):
            db.execute_many("INSERT INTO no_such_table (id) VALUES (?)", [("a",), ("b",)])
        assert conn.rollbacks == 1


def test_execute_batch_rolls_back_every_statement(db):
    with pytest.raises(sqlite3.Error):
        db.execute_batch([
            ("INSERT INTO atoms (id, type, label, tenant_id, space) VALUES (?, ?, ?, ?, ?)",
             ("batch-1", "concept", "kept?", "t", "s")),
            ("INSERT INTO no_such_table (id) VALUES (?)", ("batch-2",)),
        ])
    assert db.fetchone("SELECT id FROM atoms WHERE id = ?", ("batch-1",)) is None


def test_execute_batch_commits_all_statements_on_success(db):
    db.execute_batch([
        ("INSERT INTO atoms (id, type, label, tenant_id, space) VALUES (?, ?, ?, ?, ?)",
         ("batch-3", "concept", "one", "t", "s")),
        ("INSERT INTO atoms (id, type, label, tenant_id, space) VALUES (?, ?, ?, ?, ?)",
         ("batch-4", "concept", "two", "t", "s")),
    ])
    rows = db.fetchall("SELECT id FROM atoms WHERE id IN ('batch-3', 'batch-4')")
    assert len(rows) == 2


# ── migrations roll back on failure ───────────────────────────────────────────

def _fresh_connection(path: str) -> Database:
    """A Database wrapper over an existing file, without running initialize()."""
    database = Database(path)
    database._write_conn = _make_connection(path)
    return database


def test_content_hash_migration_rolls_back(db_path):
    get_database(db_path)
    close_database(db_path)
    # Drop the column so the migration runs again on the next open.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX IF EXISTS idx_atoms_content_hash")
    conn.execute("ALTER TABLE atoms DROP COLUMN content_hash")
    conn.commit()
    conn.close()

    database = _fresh_connection(db_path)
    try:
        boom = MagicMock(side_effect=sqlite3.Error("boom"))
        with _wrapped_write_conn(database, executemany=boom):
            with pytest.raises(sqlite3.Error):
                database._migrate_content_hash()
        cols = {r["name"] for r in database._write_conn.execute("PRAGMA table_info(atoms)")}
        assert "content_hash" not in cols  # the ALTER was rolled back too
    finally:
        database.close()


def test_personality_migration_rolls_back(db_path):
    get_database(db_path)
    close_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE personality DROP COLUMN agent_source_trust")
    conn.commit()
    conn.close()

    database = _fresh_connection(db_path)
    try:
        boom = MagicMock(side_effect=sqlite3.Error("boom"))
        with _wrapped_write_conn(database, executemany=boom):
            with pytest.raises(sqlite3.Error):
                database._migrate_personality_columns()
        cols = {r["name"] for r in database._write_conn.execute("PRAGMA table_info(personality)")}
        assert "agent_source_trust" not in cols
    finally:
        database.close()


def test_vec_atoms_migration_rolls_back(db_path):
    get_database(db_path)
    close_database(db_path)
    conn = sqlite3.connect(db_path)
    # A leftover backup table forces the migration to run on the next open,
    # and it needs a row that joins to a real atom or there is nothing to
    # copy and nothing to fail on.
    conn.execute("CREATE TABLE _vec_atoms_migrate (atom_id TEXT, embedding BLOB, tenant_id TEXT)")
    conn.execute(
        "INSERT INTO atoms (id, type, label, tenant_id, space) VALUES ('a1', 'concept', 'Alice', 't', 's')"
    )
    conn.execute(
        "INSERT INTO _vec_atoms_migrate (atom_id, embedding, tenant_id) VALUES ('a1', ?, 't')",
        (struct.pack("384f", *([0.1] * 384)),),
    )
    conn.commit()
    conn.close()

    database = _fresh_connection(db_path)
    try:
        # The rows are copied back through executemany now, so the stand-in
        # has to fail there rather than on a single execute.
        def _fail_on_insert(sql, *args):
            raise sqlite3.Error("boom")

        with _wrapped_write_conn(database, executemany=_fail_on_insert):
            with pytest.raises(sqlite3.Error):
                database._migrate_vec_atoms()
        # The backup table survives a failed migration — the rows are recoverable.
        assert database._write_conn.execute(
            "SELECT name FROM sqlite_master WHERE name = '_vec_atoms_migrate'"
        ).fetchone() is not None
    finally:
        database.close()


def test_migration_backup_is_skipped_for_memory_databases():
    database = Database(":memory:")
    database.initialize()
    try:
        database._backup_before_migration()  # no file to snapshot; must not raise
        assert database._migration_backup_done is False
    finally:
        database.close()


def test_migration_backup_runs_at_most_once(db_path):
    database = get_database(db_path)
    database._migration_backup_done = True
    database._backup_before_migration()
    assert not os.path.exists(_resolve_path(db_path) + ".pre-migration.bak")
    close_database(db_path)


def test_migration_backup_replaces_a_stale_snapshot(db_path):
    database = get_database(db_path)
    backup_path = _resolve_path(db_path) + ".pre-migration.bak"
    with open(backup_path, "w") as f:
        f.write("corrupt leftover, not a database")
    database._migration_backup_done = False
    database._backup_before_migration()
    # A readable SQLite snapshot replaced the junk file.
    conn = sqlite3.connect(backup_path)
    try:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'atoms'"
        ).fetchone() is not None
    finally:
        conn.close()
        os.unlink(backup_path)
        close_database(db_path)


# ── close ─────────────────────────────────────────────────────────────────────

def test_close_drains_the_read_pool(db_path):
    database = Database(db_path)
    database.initialize()
    database.close()
    assert database._write_conn is None
    assert database._read_pool.empty()
    database.close()  # idempotent


def test_close_stops_when_the_read_pool_drains_concurrently(db_path):
    """A pool emptied by another closer must end the drain, not raise."""
    database = Database(db_path)
    database.initialize()
    try:
        with patch.object(database._read_pool, "empty", return_value=False):
            with patch.object(database._read_pool, "get_nowait", side_effect=queue.Empty):
                database.close()
    finally:
        while not database._read_pool.empty():
            database._read_pool.get_nowait().close()
    assert database._write_conn is None


def test_context_manager_closes_the_database(db_path):
    with Database(db_path) as database:
        database.initialize()
        assert database._write_conn is not None
    assert database._write_conn is None
