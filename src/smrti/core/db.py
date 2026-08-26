from __future__ import annotations

import hashlib
import os
import queue
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any

import sqlite_vec

_registry: dict[str, "Database"] = {}
_registry_lock = threading.Lock()


def _resolve_path(db_path: str) -> str:
    """Normalize a db path so aliases of the same file share one registry entry."""
    if db_path == ":memory:":
        return db_path
    return os.path.realpath(os.path.expanduser(db_path))


def get_database(db_path: str) -> "Database":
    """Return a shared Database for db_path, creating it on first call."""
    resolved = _resolve_path(db_path)
    with _registry_lock:
        if resolved not in _registry:
            db = Database(resolved)
            db.initialize()
            _registry[resolved] = db
        return _registry[resolved]


def close_database(db_path: str) -> None:
    """Close and evict a database from the registry. Safe to call even if not registered."""
    resolved = _resolve_path(db_path)
    with _registry_lock:
        db = _registry.pop(resolved, None)
    if db is not None:
        db.close()


def clear_registry() -> None:
    """Close all databases and empty the registry. Intended for test teardown."""
    with _registry_lock:
        entries = list(_registry.items())
        _registry.clear()
    for _, db in entries:
        db.close()


_VEC_SCHEMA_SQL = """CREATE VIRTUAL TABLE IF NOT EXISTS vec_atoms USING vec0(
    atom_id     TEXT,
    embedding   float[384] distance_metric=cosine,
    tenant_id   TEXT partition key,
    space       TEXT partition key,
    +label      TEXT
)"""

_SCHEMA_SQL = f"""
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
    intrinsic_valence   REAL,
    intrinsic_intensity REAL,
    source_id   TEXT REFERENCES atoms(id),
    target_id   TEXT REFERENCES atoms(id),
    relation    TEXT,
    tenant_id   TEXT NOT NULL,
    space       TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')),
    metadata    TEXT DEFAULT '{{}}',
    entity_type TEXT,
    content_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(type);
CREATE INDEX IF NOT EXISTS idx_atoms_tenant_space ON atoms(tenant_id, space);
CREATE INDEX IF NOT EXISTS idx_atoms_entity_type ON atoms(entity_type);
CREATE INDEX IF NOT EXISTS idx_atoms_source ON atoms(source_id);
CREATE INDEX IF NOT EXISTS idx_atoms_target ON atoms(target_id);
CREATE INDEX IF NOT EXISTS idx_atoms_label ON atoms(label);
CREATE INDEX IF NOT EXISTS idx_atoms_sti ON atoms(sti DESC);
CREATE INDEX IF NOT EXISTS idx_atoms_content_hash ON atoms(tenant_id, space, content_hash);

{_VEC_SCHEMA_SQL};

CREATE TABLE IF NOT EXISTS evidence (
    id                   TEXT PRIMARY KEY,
    atom_id              TEXT NOT NULL REFERENCES atoms(id),
    observed_probability REAL NOT NULL,
    weight               REAL DEFAULT 1.0,
    source_episode_id    TEXT,
    tenant_id            TEXT NOT NULL,
    space                TEXT NOT NULL,
    created_at           TEXT DEFAULT (datetime('now')),
    processed            INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_evidence_pending ON evidence(processed, tenant_id, space);
CREATE INDEX IF NOT EXISTS idx_evidence_atom ON evidence(atom_id);

CREATE TABLE IF NOT EXISTS personality (
    tenant_id               TEXT NOT NULL,
    space                   TEXT NOT NULL,
    confidence_decay_rate   REAL DEFAULT 0.02,
    confidence_update_lr    REAL DEFAULT 0.3,
    min_confidence_to_surface REAL DEFAULT 0.1,
    sti_decay_rate          REAL DEFAULT 0.1,
    sti_boost_on_access     REAL DEFAULT 0.5,
    sti_propagation_factor  REAL DEFAULT 0.15,
    lti_promotion_threshold REAL DEFAULT 0.7,
    lti_decay_rate          REAL DEFAULT 0.01,
    agent_source_trust      REAL DEFAULT 0.5,
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
    created_at              TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (tenant_id, space)
);

CREATE TABLE IF NOT EXISTS aliases (
    alias       TEXT NOT NULL,
    atom_id     TEXT NOT NULL REFERENCES atoms(id),
    tenant_id   TEXT NOT NULL,
    space       TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (alias, tenant_id, space)
);

CREATE INDEX IF NOT EXISTS idx_aliases_atom ON aliases(atom_id);
"""

_READ_POOL_SIZE = 4


def _u_lower(value: Any) -> Any:
    """Unicode-aware lowercase for SQL — SQLite's LOWER() only folds ASCII."""
    return value.lower() if isinstance(value, str) else value


def _make_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA cache_size=-64000")
    conn.create_function("u_lower", 1, _u_lower, deterministic=True)
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
        self._migration_backup_done = False

    def initialize(self) -> None:
        self._write_conn = _make_connection(self._db_path)
        for _ in range(_READ_POOL_SIZE):
            self._read_pool.put(_make_connection(self._db_path))
        with self._write_lock:
            self._migrate_vec_atoms()
            self._migrate_content_hash()
            self._migrate_personality_columns()
            self._migrate_intrinsic_valence()
            for statement in _SCHEMA_SQL.strip().split(";"):
                stmt = statement.strip()
                if stmt:
                    self._write_conn.execute(stmt)
            self._write_conn.commit()

    def _migrate_content_hash(self) -> None:
        """Add the content_hash column to pre-existing DBs and backfill episodes."""
        conn = self._write_conn
        if (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'atoms'"
            ).fetchone()
            is None
        ):
            return
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(atoms)")}
        if "content_hash" in cols:
            return
        self._backup_before_migration()
        try:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ALTER TABLE atoms ADD COLUMN content_hash TEXT")
            rows = conn.execute(
                "SELECT id, content FROM atoms WHERE type = 'episode' AND content IS NOT NULL"
            ).fetchall()
            conn.executemany(
                "UPDATE atoms SET content_hash = ? WHERE id = ?",
                [
                    (hashlib.sha256(r["content"].encode()).hexdigest(), r["id"])
                    for r in rows
                ],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _migrate_intrinsic_valence(self) -> None:
        """Add the columns holding an atom's tone as written.

        Deliberately not backfilled. Propagation has already moved the stored
        valence away from what each old atom said, and nothing recovers the
        original: re-estimating it from the text would overwrite the handful
        that a caller stated on purpose, and copying the current value would
        just relabel the drift as intrinsic. NULL reads as "use what is there",
        so an existing graph behaves exactly as before and only new atoms carry
        the clean signal.
        """
        conn = self._write_conn
        if (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'atoms'"
            ).fetchone()
            is None
        ):
            return
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(atoms)")}
        missing = [c for c in ("intrinsic_valence", "intrinsic_intensity") if c not in cols]
        if not missing:
            return
        self._backup_before_migration()
        try:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            for column in missing:
                conn.execute(f"ALTER TABLE atoms ADD COLUMN {column} REAL")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _backup_before_migration(self) -> None:
        """Snapshot the whole DB file before the first schema-altering migration.

        Each migration is transactional, but a file-level copy is the only
        rollback that also survives downgrading the package — restore the
        ``.pre-migration.bak`` file over the DB and the previous release runs
        as before. The SQLite backup API is used instead of a filesystem copy
        so the snapshot is consistent even with WAL frames not yet
        checkpointed into the main file. At most one snapshot per open; a
        later release's migration overwrites it, by which point this
        snapshot's state has already been validated by a successful upgrade.
        """
        if self._migration_backup_done or self._db_path == ":memory:":
            return
        backup_path = self._db_path + ".pre-migration.bak"
        try:
            # A stale or corrupt leftover is not a valid backup destination.
            os.remove(backup_path)
        except FileNotFoundError:
            pass
        dest = sqlite3.connect(backup_path)
        try:
            self._write_conn.backup(dest)
        finally:
            dest.close()
        self._migration_backup_done = True

    _PERSONALITY_COLUMNS = (
        ("lti_decay_rate", 0.01),
        ("agent_source_trust", 0.5),
    )

    def _migrate_personality_columns(self) -> None:
        """Add personality columns introduced after a DB was first created.

        ``CREATE TABLE IF NOT EXISTS`` never revises an existing table, so
        columns are appended here and backfilled from the row's own preset —
        a maverick row must get maverick's value, not the column default.
        """
        conn = self._write_conn
        if (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'personality'"
            ).fetchone()
            is None
        ):
            return
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(personality)")}
        missing = [(c, d) for c, d in self._PERSONALITY_COLUMNS if c not in existing]
        if not missing:
            return

        from smrti.personality.params import PRESETS

        self._backup_before_migration()
        try:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            for column, default in missing:
                conn.execute(
                    f"ALTER TABLE personality ADD COLUMN {column} REAL DEFAULT {default}"
                )
                conn.executemany(
                    f"UPDATE personality SET {column} = ? WHERE preset_name = ?",
                    [
                        (getattr(profile, column), name)
                        for name, profile in PRESETS.items()
                    ],
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _migrate_vec_atoms(self) -> None:
        """Rebuild vec_atoms in place when created by a pre-space / L2 schema.

        Embeddings are copied out through a backup table, the table is
        recreated with the cosine metric and space partition key, and rows are
        restored joined to atoms for the space value — dropping relation-atom
        vectors and orphaned vectors in the same pass. Stored vectors are
        unchanged, only the distance metric interpretation moves to cosine.
        """
        conn = self._write_conn
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_atoms'"
        ).fetchone()
        have_backup = (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = '_vec_atoms_migrate'"
            ).fetchone()
            is not None
        )
        if row is None and not have_backup:
            return
        sql = (row["sql"] or "") if row else ""
        current = "space" in sql and "distance_metric=cosine" in sql
        if current and not have_backup:
            return
        self._backup_before_migration()
        # Explicit transaction: SQLite DDL is transactional, but Python's
        # legacy autocommit mode would otherwise commit each DDL statement
        # individually — a crash mid-migration must leave the old index (or a
        # restorable backup table) intact, never a silently empty one.
        try:
            conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            if not have_backup:
                conn.execute(
                    "CREATE TABLE _vec_atoms_migrate AS SELECT atom_id, embedding, tenant_id FROM vec_atoms"
                )
            if row is not None:
                conn.execute("DROP TABLE vec_atoms")
            conn.execute(_VEC_SCHEMA_SQL)
            conn.execute(
                """INSERT INTO vec_atoms (atom_id, embedding, tenant_id, space, label)
                   SELECT m.atom_id, m.embedding, m.tenant_id, a.space, a.label
                   FROM _vec_atoms_migrate m
                   JOIN atoms a ON a.id = m.atom_id
                   WHERE a.type != 'relation'"""
            )
            conn.execute("DROP TABLE _vec_atoms_migrate")
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Row:
        with self._write_lock:
            try:
                cursor = self._write_conn.execute(sql, params)
                self._write_conn.commit()
                return cursor
            except Exception:
                self._write_conn.rollback()
                raise

    def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        with self._write_lock:
            try:
                self._write_conn.executemany(sql, params_list)
                self._write_conn.commit()
            except Exception:
                self._write_conn.rollback()
                raise

    def execute_batch(self, statements: list[tuple]) -> None:
        """Execute multiple (sql, params) pairs in a single transaction."""
        with self._write_lock:
            try:
                for sql, params in statements:
                    self._write_conn.execute(sql, params)
                self._write_conn.commit()
            except Exception:
                self._write_conn.rollback()
                raise

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
