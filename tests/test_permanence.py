"""Permanent beliefs: born certain, exempt from decay, healed by the epoch."""
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


def test_forget_stamps_the_atom_as_deliberately_sunk(mem):
    """The stamp is what tells a forget from decay drowning, so the heal
    can lift the latter without resurrecting the former."""
    atom_id = mem.believe("Nicolás lives in San Benito.", PERMANENT_PROBABILITY)
    mem.forget("San Benito")
    row = mem.db.fetchone("SELECT metadata FROM atoms WHERE id = ?", (atom_id,))
    assert '"forgotten"' in row["metadata"]


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


def _drown(mem, atom_id, confidence, source=None):
    """Leave an atom as a pre-permanence writer would have: low confidence,
    no forgotten stamp — indistinguishable from decay because it was decay."""
    metadata = '{"source": "agent"}' if source == "agent" else "{}"
    mem.db.execute(
        "UPDATE atoms SET confidence = ?, metadata = ? WHERE id = ?",
        (confidence, metadata, atom_id),
    )


def _restart(db_path):
    """Open the graph the way a fresh process would."""
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
    close_database(path)
    for suffix in ("", "-wal", "-shm", ".pre-migration.bak"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


def test_epoch_heals_a_belief_drowned_into_the_floor(mem):
    atom_id = mem.believe("Nicolás lives in San Benito.", PERMANENT_PROBABILITY)
    _drown(mem, atom_id, confidence=0.1)
    mem.reflect()
    assert _truth(mem, atom_id)[1] == pytest.approx(PERMANENT_PROBABILITY)


def test_epoch_heals_damage_done_after_the_graph_was_first_opened(db_path):
    """A stale pre-fix process kept minting drowned beliefs for days after the
    package on disk was fixed; a one-time startup repair had already burned its
    ledger and never lifted them. The heal must not care when the damage
    happened, only that it is there."""
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    atom_id = engine.believe("Nicolás lives in San Benito.", PERMANENT_PROBABILITY)
    close_database(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE atoms SET confidence = 0.1 WHERE id = ?", (atom_id,))
    conn.commit()
    conn.close()
    engine = _restart(db_path)
    engine.reflect()
    assert _confidence(db_path, atom_id) == pytest.approx(PERMANENT_PROBABILITY)


def test_heal_leaves_a_belief_sunk_below_the_floor(mem):
    """Below the floor is where a pre-stamp forget() left things; nothing can
    tell those from decay victims, so they stay where they are."""
    atom_id = mem.believe("Nicolás lives in San Benito.", PERMANENT_PROBABILITY)
    _drown(mem, atom_id, confidence=0.02)
    mem.reflect()
    assert _truth(mem, atom_id)[1] == pytest.approx(0.02)


def test_heal_ignores_ordinary_and_agent_beliefs(mem):
    ordinary = mem.believe("It will probably rain.", 0.6)
    agent = mem.believe(
        "My own answer was correct.", PERMANENT_PROBABILITY, source="agent"
    )
    _drown(mem, ordinary, confidence=0.1)
    _drown(mem, agent, confidence=0.1, source="agent")
    mem.reflect()
    assert _truth(mem, ordinary)[1] == pytest.approx(0.1)
    # Pruning only reaches agent-authored atoms, so the drowned agent belief
    # may simply be gone; either way it was not lifted.
    row = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (agent,))
    assert row is None or row["confidence"] < 0.1


def test_a_forget_stands_across_restarts_and_epochs(db_path):
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    atom_id = engine.believe("Nicolás lives in San Benito.", PERMANENT_PROBABILITY)
    engine.forget("San Benito")
    forgotten = _confidence(db_path, atom_id)
    assert forgotten < PERMANENT_PROBABILITY
    engine = _restart(db_path)
    for _ in range(3):
        engine.reflect()
    assert _confidence(db_path, atom_id) == pytest.approx(forgotten)
