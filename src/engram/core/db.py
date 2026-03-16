from __future__ import annotations

import queue
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any

import sqlite_vec


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS atoms (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    label       TEXT NOT NULL,
    content     TEXT,
    probability REAL DEFAULT 0.5,
    confidence  REAL DEFAULT 0.0,
    sti         REAL DEFAULT 0.0,
    lti         REAL DEFAULT 0.0,
    valence     REAL DEFAULT 0.0,
    intensity   REAL DEFAULT 0.0,
    source_id   TEXT REFERENCES atoms(id),
    target_id   TEXT REFERENCES atoms(id),
    relation    TEXT,
    agent_id    TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')),
    metadata    TEXT DEFAULT '{}',
    entity_type TEXT
);

CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(type);
CREATE INDEX IF NOT EXISTS idx_atoms_agent ON atoms(agent_id);
CREATE INDEX IF NOT EXISTS idx_atoms_entity_type ON atoms(entity_type);
CREATE INDEX IF NOT EXISTS idx_atoms_source ON atoms(source_id);
CREATE INDEX IF NOT EXISTS idx_atoms_target ON atoms(target_id);
CREATE INDEX IF NOT EXISTS idx_atoms_label ON atoms(label);
CREATE INDEX IF NOT EXISTS idx_atoms_sti ON atoms(sti DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_atoms USING vec0(
    atom_id     TEXT,
    embedding   float[384],
    agent_id    TEXT partition key,
    +label      TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    id                  TEXT PRIMARY KEY,
    atom_id             TEXT NOT NULL REFERENCES atoms(id),
    observed_probability REAL NOT NULL,
    weight              REAL DEFAULT 1.0,
    source_episode_id   TEXT,
    agent_id            TEXT NOT NULL,
    created_at          TEXT DEFAULT (datetime('now')),
    processed           INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_evidence_pending ON evidence(processed, agent_id);
CREATE INDEX IF NOT EXISTS idx_evidence_atom ON evidence(atom_id);

CREATE TABLE IF NOT EXISTS personality (
    agent_id                TEXT PRIMARY KEY,
    confidence_decay_rate   REAL DEFAULT 0.02,
    confidence_update_lr    REAL DEFAULT 0.3,
    min_confidence_to_surface REAL DEFAULT 0.1,
    sti_decay_rate          REAL DEFAULT 0.1,
    sti_boost_on_access     REAL DEFAULT 0.5,
    sti_propagation_factor  REAL DEFAULT 0.15,
    lti_promotion_threshold REAL DEFAULT 0.7,
    valence_weight          REAL DEFAULT 0.2,
    valence_propagation     REAL DEFAULT 0.1,
    mood_inertia            REAL DEFAULT 0.8,
    w_similarity            REAL DEFAULT 0.35,
    w_sti                   REAL DEFAULT 0.25,
    w_confidence            REAL DEFAULT 0.20,
    w_lti                   REAL DEFAULT 0.10,
    w_valence               REAL DEFAULT 0.10,
    preset_name             TEXT DEFAULT 'balanced',
    epoch_count             INTEGER DEFAULT 0,
    created_at              TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS aliases (
    alias       TEXT NOT NULL,
    atom_id     TEXT NOT NULL REFERENCES atoms(id),
    agent_id    TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (alias, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_aliases_atom ON aliases(atom_id);
"""

_READ_POOL_SIZE = 4


def _make_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA cache_size=-64000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class Database:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._write_conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()
        self._read_pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=_READ_POOL_SIZE)

    def initialize(self) -> None:
        self._write_conn = _make_connection(self._db_path)
        for _ in range(_READ_POOL_SIZE):
            self._read_pool.put(_make_connection(self._db_path))
        with self._write_lock:
            for statement in _SCHEMA_SQL.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    self._write_conn.execute(stmt)
            self._write_conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Row:
        with self._write_lock:
            cursor = self._write_conn.execute(sql, params)
            self._write_conn.commit()
            return cursor

    def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        with self._write_lock:
            self._write_conn.executemany(sql, params_list)
            self._write_conn.commit()

    @contextmanager
    def _read_conn(self):
        conn = self._read_pool.get()
        try:
            yield conn
        finally:
            self._read_pool.put(conn)

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._read_conn() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._read_conn() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchone()

    def close(self) -> None:
        if self._write_conn:
            self._write_conn.close()
            self._write_conn = None
        while not self._read_pool.empty():
            try:
                conn = self._read_pool.get_nowait()
                conn.close()
            except queue.Empty:
                break

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
