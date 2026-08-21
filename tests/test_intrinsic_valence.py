"""An atom's own tone survives the mood it absorbs from its neighbours."""
import os
import sqlite3
import tempfile

import pytest

from smrti import Smrti
from smrti.core.db import close_database
from smrti.core.models import Valence
from smrti.core.provenance import VALENCE_STATED
from smrti.retrieval.classify import classify_memory


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def _row(mem, atom_id):
    return mem.db.fetchone(
        "SELECT valence, intensity, intrinsic_valence, intrinsic_intensity "
        "FROM atoms WHERE id = ?",
        (atom_id,),
    )


def test_a_new_atom_records_the_tone_it_was_written_with(mem):
    atom_id = mem.remember("I broke the build.", valence=-0.9)
    row = _row(mem, atom_id)
    assert row["intrinsic_valence"] == pytest.approx(-0.9)
    assert row["intrinsic_intensity"] == pytest.approx(0.9)


def test_propagation_moves_the_mood_and_leaves_the_atom_alone(mem):
    atom_id = mem.remember("Roxana studies pharmacy.", valence=0.0)
    mem.db.execute(
        "UPDATE atoms SET valence = -0.9, intensity = 0.9 WHERE id = ?", (atom_id,)
    )
    atom = mem.atomspace.get_atom(atom_id, "test", "default")
    assert atom.valence.valence == pytest.approx(-0.9)
    assert atom.valence.own == pytest.approx(0.0)


def test_absorbed_mood_cannot_make_a_memory_severe(mem):
    """The regression: a word became a mistake by appearing in complaints."""
    atom_id = mem.remember("The gateway writes to its own log file.", valence=0.0)
    mem.db.execute(
        "UPDATE atoms SET valence = -0.95, intensity = 0.95, "
        "metadata = ? WHERE id = ?",
        ('{"%s": true}' % VALENCE_STATED, atom_id),
    )
    hits = [h for h in mem.recall("gateway log", top_k=10, min_confidence=0.0)
            if h.atom.id == atom_id]
    assert hits, "the atom should still be recallable"
    assert classify_memory(hits[0]) == "context"


def test_a_stated_severe_tone_still_reads_as_severe(mem):
    atom_id = mem.remember("I force-pushed over main.", valence=-0.9)
    mem.db.execute(
        "UPDATE atoms SET metadata = ? WHERE id = ?",
        ('{"%s": true}' % VALENCE_STATED, atom_id),
    )
    hits = [h for h in mem.recall("force push main", top_k=10, min_confidence=0.0)
            if h.atom.id == atom_id]
    assert classify_memory(hits[0]) == "critical_warning"


def test_an_atom_predating_the_columns_reads_as_it_always_did():
    """NULL means "use what is there" — nothing is invented for an old graph."""
    v = Valence(valence=-0.7, intensity=0.8)
    assert v.own == pytest.approx(-0.7)
    assert v.own_intensity == pytest.approx(0.8)


def test_the_migration_adds_the_columns_without_backfilling():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
        atom_id = engine.remember("an old memory", valence=-0.6)
        close_database(db_path)

        conn = sqlite3.connect(db_path)
        for column in ("intrinsic_valence", "intrinsic_intensity"):
            conn.execute(f"UPDATE atoms SET {column} = NULL")
        conn.commit()
        conn.close()

        engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
        atom = engine.atomspace.get_atom(atom_id, "test", "default")
        assert atom.valence.intrinsic_valence is None
        assert atom.valence.own == pytest.approx(-0.6)
        engine.close()
    finally:
        for suffix in ("", "-wal", "-shm", ".pre-migration.bak"):
            if os.path.exists(db_path + suffix):
                os.unlink(db_path + suffix)
