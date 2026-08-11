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


# ── personality column migration ──────────────────────────────────────────────

NEW_PERSONALITY_COLUMNS = ("lti_decay_rate", "agent_source_trust")


def _downgrade_personality_schema(path: str) -> None:
    """Strip columns added after release so the table looks like an older DB."""
    conn = sqlite3.connect(path)
    for column in NEW_PERSONALITY_COLUMNS:
        conn.execute(f"ALTER TABLE personality DROP COLUMN {column}")
    conn.commit()
    conn.close()


def _legacy_db(tmp_path, name, preset="maverick", tenant="t1", space="s1"):
    """A DB carrying a personality row but no post-release personality columns."""
    from smrti.personality.params import load_preset

    path = str(tmp_path / name)
    db = get_database(path)
    profile = load_preset(preset)
    db.execute(
        """INSERT INTO personality (tenant_id, space, preset_name, sti_decay_rate, epoch_count)
           VALUES (?, ?, ?, ?, ?)""",
        (tenant, space, preset, profile.sti_decay_rate, 7),
    )
    _insert_atom(db, "keep1", "concept", "existing data", space, tenant=tenant)
    # A bare insert leaves confidence at 0.0, which is legitimately prunable;
    # give it the profile of an established atom so survival is meaningful.
    db.execute("UPDATE atoms SET confidence = 0.7, lti = 0.4 WHERE id = 'keep1'")
    close_database(path)
    _downgrade_personality_schema(path)
    return path


def test_legacy_db_without_new_personality_columns_opens(tmp_path):
    """Opening a pre-upgrade DB must add the columns rather than raise."""
    path = _legacy_db(tmp_path, "legacy.db")
    db = get_database(path)
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(personality)")}
    assert set(NEW_PERSONALITY_COLUMNS) <= cols


def test_personality_migration_backfills_from_the_rows_own_preset(tmp_path):
    """A maverick row must get maverick's values, not the generic column default."""
    from smrti.personality.params import load_preset

    path = _legacy_db(tmp_path, "preset.db", preset="maverick")
    db = get_database(path)
    row = db.fetchone("SELECT * FROM personality WHERE tenant_id = 't1'")
    expected = load_preset("maverick")
    assert row["lti_decay_rate"] == pytest.approx(expected.lti_decay_rate)
    assert row["agent_source_trust"] == pytest.approx(expected.agent_source_trust)
    # maverick deliberately differs from the schema default here — guard the guard.
    assert expected.lti_decay_rate != 0.01


def test_personality_migration_preserves_existing_data(tmp_path):
    """Migration must not disturb atoms or the columns already on the row."""
    path = _legacy_db(tmp_path, "preserve.db")
    db = get_database(path)
    row = db.fetchone("SELECT * FROM personality WHERE tenant_id = 't1'")
    assert row["epoch_count"] == 7
    assert row["preset_name"] == "maverick"
    assert db.fetchone("SELECT label FROM atoms WHERE id = 'keep1'")["label"] == "existing data"


def test_personality_migration_defaults_unknown_presets(tmp_path):
    """A custom-tuned row has no preset to backfill from and takes the default."""
    path = str(tmp_path / "custom.db")
    db = get_database(path)
    db.execute(
        "INSERT INTO personality (tenant_id, space, preset_name) VALUES ('t1', 's1', 'custom')"
    )
    close_database(path)
    _downgrade_personality_schema(path)

    db = get_database(path)
    row = db.fetchone("SELECT * FROM personality WHERE tenant_id = 't1'")
    assert row["lti_decay_rate"] == pytest.approx(0.01)
    assert row["agent_source_trust"] == pytest.approx(0.5)


def test_personality_migration_is_idempotent(tmp_path):
    path = _legacy_db(tmp_path, "idem.db")
    get_database(path)
    close_database(path)
    db = get_database(path)  # second open: columns already present
    assert db.fetchone("SELECT epoch_count FROM personality")["epoch_count"] == 7


def test_partially_migrated_db_gains_only_the_missing_column(tmp_path):
    """A half-applied upgrade must complete, not trip over the column it has."""
    path = str(tmp_path / "partial.db")
    db = get_database(path)
    db.execute(
        "INSERT INTO personality (tenant_id, space, preset_name) VALUES ('t1', 's1', 'balanced')"
    )
    close_database(path)
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE personality DROP COLUMN agent_source_trust")
    conn.commit()
    conn.close()

    db = get_database(path)
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(personality)")}
    assert set(NEW_PERSONALITY_COLUMNS) <= cols


def test_migrated_db_supports_personality_writes(tmp_path):
    """The INSERT and UPDATE paths name every column explicitly.

    Without the migration these raise "no such column" on every pre-existing
    database the moment a new space is initialised or a preset is applied.
    """
    from smrti import Smrti
    from smrti.personality.params import load_preset

    path = _legacy_db(tmp_path, "writes.db", tenant="t1", space="s1")
    close_database(path)

    # A new space on a migrated DB exercises the INSERT column list.
    mem = Smrti(db_path=path, personality="analytical", tenant_id="t1", write_space="fresh")
    row = mem.db.fetchone(
        "SELECT * FROM personality WHERE tenant_id = 't1' AND space = 'fresh'"
    )
    assert row["agent_source_trust"] == pytest.approx(
        load_preset("analytical").agent_source_trust
    )

    # set_personality_profile exercises the UPDATE column list.
    mem.set_personality("curious")
    row = mem.db.fetchone(
        "SELECT * FROM personality WHERE tenant_id = 't1' AND space = 'fresh'"
    )
    assert row["lti_decay_rate"] == pytest.approx(load_preset("curious").lti_decay_rate)


def test_migrated_legacy_db_can_run_an_epoch(tmp_path):
    """End-to-end: an upgraded institutional DB must still consolidate."""
    from smrti import Smrti

    path = _legacy_db(tmp_path, "epoch.db", tenant="t1", space="s1")
    close_database(path)

    mem = Smrti(db_path=path, personality="maverick", tenant_id="t1", write_space="s1")
    mem.remember("a fact worth keeping")
    result = mem.reflect()
    assert result.atoms_decayed >= 1
    assert mem.db.fetchone("SELECT label FROM atoms WHERE id = 'keep1'") is not None


def test_u_lower_is_unicode_aware(tmp_path):
    db = get_database(str(tmp_path / "ul.db"))
    _insert_atom(db, "m1", "concept", "MÜLLER", "s1")
    row = db.fetchone(
        "SELECT id FROM atoms WHERE u_lower(label) = ?", ("müller",)
    )
    assert row is not None and row["id"] == "m1"
    # Plain LOWER() would miss this (ASCII-only folding) — guard the guard.
    assert db.fetchone("SELECT id FROM atoms WHERE LOWER(label) = ?", ("müller",)) is None


def test_new_columns_have_defaults_for_older_writers(tmp_path):
    """An older smrti build names columns explicitly and omits the new ones.

    Mixed-version fleets are normal during a rollout, so a row written by the
    previous release must still come out valid rather than NULL.
    """
    db = get_database(str(tmp_path / "mixed.db"))
    db.execute(
        """INSERT INTO personality (
               tenant_id, space, confidence_decay_rate, confidence_update_lr,
               min_confidence_to_surface, sti_decay_rate, sti_boost_on_access,
               sti_propagation_factor, lti_promotion_threshold, valence_weight,
               valence_propagation, mood_inertia, w_similarity, w_sti, w_confidence,
               w_lti, w_valence, preset_name
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("t1", "s1", 0.02, 0.3, 0.1, 0.1, 0.5, 0.15, 0.7, 0.2, 0.1, 0.8,
         0.35, 0.25, 0.2, 0.1, 0.1, "balanced"),
    )
    row = db.fetchone("SELECT * FROM personality WHERE tenant_id = 't1'")
    assert row["lti_decay_rate"] == pytest.approx(0.01)
    assert row["agent_source_trust"] == pytest.approx(0.5)


def test_epoch_tolerates_a_personality_row_missing_new_values(tmp_path):
    """NULLs in the new columns must fall back, not poison the decay arithmetic."""
    from smrti import Smrti

    path = str(tmp_path / "nulls.db")
    mem = Smrti(db_path=path, personality="balanced", tenant_id="t1", write_space="s1")
    atom_id = mem.remember("a fact", type="concept")
    mem.db.execute("UPDATE atoms SET confidence = 0.8 WHERE id = ?", (atom_id,))
    mem.db.execute(
        "UPDATE personality SET lti_decay_rate = NULL, agent_source_trust = NULL "
        "WHERE tenant_id = 't1' AND space = 's1'"
    )

    mem.reflect()  # must not raise on NULL * float
    row = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (atom_id,))
    assert row["confidence"] < 0.8
