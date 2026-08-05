"""Tests for the vec_atoms schema migration, transaction safety, and u_lower."""
import sqlite3
import struct

import pytest
import sqlite_vec

from smrti.core.db import close_database, get_database

DIM = 384


def _vec(seed: float) -> bytes:
    raw = [((seed * (i + 1)) % 7) + 0.1 for i in range(DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return struct.pack(f"{DIM}f", *[x / norm for x in raw])


def _insert_atom(db, atom_id, type_, label, space, tenant="t1"):
    db.execute(
        "INSERT INTO atoms (id, type, label, tenant_id, space) VALUES (?, ?, ?, ?, ?)",
        (atom_id, type_, label, tenant, space),
    )


def _downgrade_vec_schema(path: str) -> None:
    """Rewrite vec_atoms to the pre-space / L2 schema, keeping its rows."""
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE TABLE _bkp AS SELECT atom_id, embedding, tenant_id, label FROM vec_atoms"
    )
    conn.execute("DROP TABLE vec_atoms")
    conn.execute(
        """CREATE VIRTUAL TABLE vec_atoms USING vec0(
               atom_id TEXT, embedding float[384], tenant_id TEXT partition key, +label TEXT)"""
    )
    conn.execute(
        "INSERT INTO vec_atoms (atom_id, embedding, tenant_id, label) "
        "SELECT atom_id, embedding, tenant_id, label FROM _bkp"
    )
    conn.execute("DROP TABLE _bkp")
    conn.commit()
    conn.close()


@pytest.fixture
def old_schema_db(tmp_path):
    """A DB in the old vec schema holding: two concepts in different spaces,
    one relation atom with a vector, and one orphaned vector."""
    path = str(tmp_path / "old.db")
    db = get_database(path)
    _insert_atom(db, "c1", "concept", "alpha", "s1")
    _insert_atom(db, "c2", "concept", "beta", "s2")
    _insert_atom(db, "r1", "relation", "relation(c1, c2)", "s1")
    for atom_id, seed, space in (("c1", 1.0, "s1"), ("c2", 2.0, "s2"), ("r1", 3.0, "s1")):
        db.execute(
            "INSERT INTO vec_atoms (atom_id, embedding, tenant_id, space, label) VALUES (?, ?, ?, ?, ?)",
            (atom_id, _vec(seed), "t1", space, "L"),
        )
    db.execute(
        "INSERT INTO vec_atoms (atom_id, embedding, tenant_id, space, label) VALUES (?, ?, ?, ?, ?)",
        ("ghost", _vec(4.0), "t1", "s1", "L"),
    )
    close_database(path)
    _downgrade_vec_schema(path)
    return path


def test_fresh_db_gets_new_schema(tmp_path):
    db = get_database(str(tmp_path / "fresh.db"))
    sql = db.fetchone(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_atoms'"
    )["sql"]
    assert "space" in sql
    assert "distance_metric=cosine" in sql


def test_migration_rebuilds_old_schema(old_schema_db):
    db = get_database(old_schema_db)

    sql = db.fetchone(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_atoms'"
    )["sql"]
    assert "space" in sql and "distance_metric=cosine" in sql

    rows = db.fetchall("SELECT atom_id, space FROM vec_atoms")
    by_id = {r["atom_id"]: r["space"] for r in rows}
    # Relation vectors and orphaned vectors are purged; spaces restored from atoms.
    assert by_id == {"c1": "s1", "c2": "s2"}

    # Backup scratch table must not survive.
    assert (
        db.fetchone("SELECT name FROM sqlite_master WHERE name = '_vec_atoms_migrate'")
        is None
    )


def test_migrated_knn_is_space_scoped_cosine(old_schema_db):
    db = get_database(old_schema_db)
    hits = db.fetchall(
        """SELECT atom_id, distance FROM vec_atoms
           WHERE embedding MATCH ? AND tenant_id = ? AND space = ?
           ORDER BY distance LIMIT 10""",
        (_vec(1.0), "t1", "s1"),
    )
    assert [h["atom_id"] for h in hits] == ["c1"]
    # Cosine distance of a vector with itself is ~0.
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_migration_is_idempotent(old_schema_db):
    get_database(old_schema_db)
    close_database(old_schema_db)
    db = get_database(old_schema_db)  # second open: no-op migration
    assert len(db.fetchall("SELECT atom_id FROM vec_atoms")) == 2


def test_execute_many_rolls_back_on_failure(tmp_path):
    db = get_database(str(tmp_path / "tx.db"))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute_many(
            "INSERT INTO atoms (id, type, label, tenant_id, space) VALUES (?, ?, ?, ?, ?)",
            [
                ("a1", "concept", "one", "t1", "s1"),
                ("a2", "concept", "two", "t1", "s1"),
                ("a1", "concept", "dup", "t1", "s1"),
            ],
        )
    # The rows written before the failure must not leak into a later commit.
    db.execute("UPDATE personality SET epoch_count = epoch_count WHERE 1 = 0")
    assert db.fetchall("SELECT id FROM atoms") == []


def test_registry_normalizes_path_aliases(tmp_path):
    canonical = str(tmp_path / "reg.db")
    db1 = get_database(canonical)
    db2 = get_database(str(tmp_path / "sub" / ".." / "reg.db"))
    assert db1 is db2


def test_migration_recovers_from_stranded_backup(tmp_path):
    """A crash-interrupted migration leaves _vec_atoms_migrate behind; the next
    open must restore from it instead of trusting an empty new-schema table."""
    path = str(tmp_path / "stranded.db")
    db = get_database(path)
    _insert_atom(db, "c1", "concept", "alpha", "s1")
    close_database(path)

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # Simulate the crash window: backup populated, vec_atoms empty (new schema).
    conn.execute(
        "CREATE TABLE _vec_atoms_migrate (atom_id TEXT, embedding BLOB, tenant_id TEXT)"
    )
    conn.execute(
        "INSERT INTO _vec_atoms_migrate VALUES (?, ?, ?)", ("c1", _vec(1.0), "t1")
    )
    conn.execute("DELETE FROM vec_atoms")
    conn.commit()
    conn.close()

    db = get_database(path)
    rows = db.fetchall("SELECT atom_id, space FROM vec_atoms")
    assert [(r["atom_id"], r["space"]) for r in rows] == [("c1", "s1")]
    assert (
        db.fetchone("SELECT name FROM sqlite_master WHERE name = '_vec_atoms_migrate'")
        is None
    )


def test_content_hash_backfill_on_old_db(tmp_path):
    import hashlib

    path = str(tmp_path / "hash.db")
    db = get_database(path)
    db.execute(
        "INSERT INTO atoms (id, type, label, content, tenant_id, space) "
        "VALUES ('e1', 'episode', 'ep', 'hello world', 't1', 's1')"
    )
    close_database(path)

    conn = sqlite3.connect(path)
    conn.execute("DROP INDEX idx_atoms_content_hash")
    conn.execute("ALTER TABLE atoms DROP COLUMN content_hash")
    conn.commit()
    conn.close()

    db = get_database(path)
    row = db.fetchone("SELECT content_hash FROM atoms WHERE id = 'e1'")
    assert row["content_hash"] == hashlib.sha256(b"hello world").hexdigest()


def test_u_lower_is_unicode_aware(tmp_path):
    db = get_database(str(tmp_path / "ul.db"))
    _insert_atom(db, "m1", "concept", "MÜLLER", "s1")
    row = db.fetchone(
        "SELECT id FROM atoms WHERE u_lower(label) = ?", ("müller",)
    )
    assert row is not None and row["id"] == "m1"
    # Plain LOWER() would miss this (ASCII-only folding) — guard the guard.
    assert db.fetchone("SELECT id FROM atoms WHERE LOWER(label) = ?", ("müller",)) is None
