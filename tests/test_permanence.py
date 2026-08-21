"""Permanent beliefs: born certain, exempt from decay, repaired on upgrade."""
import os
import sqlite3
import tempfile

import pytest

from smrti import Smrti
from smrti.core.db import close_database
from smrti.core.models import PERMANENT_PROBABILITY
from smrti.servers.mcp import handle_tool


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


def _truth(mem, atom_id):
    row = mem.db.fetchone(
        "SELECT probability, confidence FROM atoms WHERE id = ?", (atom_id,)
    )
    return row["probability"], row["confidence"]


def test_permanent_belief_is_born_certain(mem):
    atom_id = mem.believe("Nicolás lives in San Benito.", PERMANENT_PROBABILITY)
    assert _truth(mem, atom_id)[1] == pytest.approx(PERMANENT_PROBABILITY)


def test_ordinary_belief_is_still_born_unsure(mem):
    atom_id = mem.believe("It will probably rain.", 0.6)
    assert _truth(mem, atom_id)[1] == pytest.approx(0.3)


def test_permanent_belief_holds_confidence_across_epochs(mem):
    atom_id = mem.believe("Nicolás's daughter is Esmeralda.", PERMANENT_PROBABILITY)
    for _ in range(200):
        mem.reflect()
    assert _truth(mem, atom_id)[1] == pytest.approx(PERMANENT_PROBABILITY)


def test_ordinary_belief_still_decays(mem):
    atom_id = mem.believe("It will probably rain.", 0.6)
    before = _truth(mem, atom_id)[1]
    for _ in range(50):
        mem.reflect()
    assert _truth(mem, atom_id)[1] < before


def test_agent_cannot_mint_a_permanent_belief(mem):
    atom_id = mem.believe(
        "My own answer was correct.", PERMANENT_PROBABILITY, source="agent"
    )
    before = _truth(mem, atom_id)[1]
    for _ in range(50):
        mem.reflect()
    assert _truth(mem, atom_id)[1] < before


def test_permanent_episode_is_not_exempt(mem):
    """Permanence is asserted, not inferred: only a belief carries it."""
    atom_id = mem.remember("something happened", probability=PERMANENT_PROBABILITY)
    for _ in range(200):
        mem.reflect()
    probability, confidence = _truth(mem, atom_id)
    assert probability == pytest.approx(PERMANENT_PROBABILITY)
    assert confidence == pytest.approx(0.1)


def test_forget_still_lowers_a_permanent_belief(mem):
    atom_id = mem.believe("Nicolás lives in San Benito.", PERMANENT_PROBABILITY)
    mem.forget("San Benito")
    lowered = _truth(mem, atom_id)[1]
    assert lowered < PERMANENT_PROBABILITY
    mem.reflect()
    assert _truth(mem, atom_id)[1] == pytest.approx(lowered)


def test_belief_records_source_and_valence(mem):
    result = handle_tool(
        mem,
        "smrti_remember",
        {
            "content": "I broke the build again.",
            "type": "belief",
            "probability": 0.9,
            "valence": -0.8,
            "source": "agent",
        },
    )
    row = mem.db.fetchone(
        "SELECT valence, metadata FROM atoms WHERE id = ?", (result["atom_id"],)
    )
    assert row["valence"] == pytest.approx(-0.8)
    assert '"source": "agent"' in row["metadata"]


def test_user_belief_leaves_source_absent(mem):
    """An absent source reads as the user, which is how remember() writes it."""
    result = handle_tool(
        mem,
        "smrti_remember",
        {"content": "Roxana studies pharmacy.", "type": "belief", "probability": 0.9},
    )
    row = mem.db.fetchone(
        "SELECT metadata FROM atoms WHERE id = ?", (result["atom_id"],)
    )
    assert "source" not in (row["metadata"] or "")


def _drowned_graph(db_path, confidence, probability=PERMANENT_PROBABILITY, source=None):
    """Build a graph as a pre-permanence release would have left it."""
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    atom_id = engine.believe("Nicolás lives in San Benito.", probability)
    metadata = '{"source": "agent"}' if source == "agent" else "{}"
    engine.db.execute(
        "UPDATE atoms SET confidence = ?, metadata = ? WHERE id = ?",
        (confidence, metadata, atom_id),
    )
    close_database(db_path)
    # A release predating the repair left no ledger behind, which is what
    # makes the upgrade path run at all.
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS applied_repairs")
    conn.commit()
    conn.close()
    return atom_id


def _restart(db_path):
    """Open the graph the way a fresh process would, re-running migrations."""
    close_database(db_path)
    return Smrti(db_path=db_path, tenant_id="test", write_space="default")


def _confidence(db_path, atom_id):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT confidence FROM atoms WHERE id = ?", (atom_id,)
        ).fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    for suffix in ("", "-wal", "-shm", ".pre-migration.bak"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


def test_repair_lifts_a_belief_decayed_into_the_floor(db_path):
    atom_id = _drowned_graph(db_path, confidence=0.1)
    _restart(db_path).close()
    assert _confidence(db_path, atom_id) == pytest.approx(PERMANENT_PROBABILITY)


def test_repair_leaves_a_forgotten_belief_sunk(db_path):
    """Below the floor is where forget() puts things, so it is left alone."""
    atom_id = _drowned_graph(db_path, confidence=0.02)
    _restart(db_path).close()
    assert _confidence(db_path, atom_id) == pytest.approx(0.02)


def test_repair_ignores_ordinary_and_agent_beliefs(db_path):
    ordinary = _drowned_graph(db_path, confidence=0.1, probability=0.6)
    agent = _drowned_graph(db_path, confidence=0.1, source="agent")
    _restart(db_path).close()
    assert _confidence(db_path, ordinary) == pytest.approx(0.1)
    assert _confidence(db_path, agent) == pytest.approx(0.1)


def test_repair_runs_once_so_a_later_forget_stands(db_path):
    atom_id = _drowned_graph(db_path, confidence=0.1)
    engine = _restart(db_path)
    assert _confidence(db_path, atom_id) == pytest.approx(PERMANENT_PROBABILITY)
    engine.forget("San Benito")
    forgotten = _confidence(db_path, atom_id)
    _restart(db_path).close()
    assert _confidence(db_path, atom_id) == pytest.approx(forgotten)
