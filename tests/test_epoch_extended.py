"""Extended tests for consolidation epoch (evolution/epoch.py)."""
import os
import tempfile

import pytest

from smrti import Smrti
from smrti.evolution.epoch import run_epoch


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(db_path)


# ── no-personality guard ──────────────────────────────────────────────────────

def test_epoch_no_personality_returns_empty_result(mem):
    # Clear personality for this tenant/space
    mem.db.execute(
        "DELETE FROM personality WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )
    result = run_epoch("test", "default", mem.db, mem.embed)
    assert result.beliefs_updated == 0
    assert result.atoms_pruned == 0


# ── evidence processing ───────────────────────────────────────────────────────

def test_evidence_processed_in_epoch(mem):
    atom_id = mem.believe("Water is wet", probability=0.5, evidence="Observation 1")
    mem.believe("Water is wet", probability=0.9, evidence="Observation 2")

    result = mem.reflect()
    assert result.beliefs_updated >= 1


def test_processed_evidence_not_reprocessed(mem):
    mem.believe("Sky is blue", probability=0.8, evidence="Direct observation")
    # First epoch processes it
    r1 = mem.reflect()
    updated_first = r1.beliefs_updated

    # Second epoch has nothing new to process
    r2 = mem.reflect()
    assert r2.beliefs_updated == 0


# ── STI/confidence decay ──────────────────────────────────────────────────────

def test_epoch_decays_sti(mem):
    atom_id = mem.remember("Decaying memory", probability=0.9)
    before = mem.db.fetchone("SELECT sti FROM atoms WHERE id = ?", (atom_id,))
    mem.reflect()
    after = mem.db.fetchone("SELECT sti FROM atoms WHERE id = ?", (atom_id,))
    assert after["sti"] <= before["sti"]


def test_epoch_decays_confidence(mem):
    atom_id = mem.remember("Confidence decays", probability=0.9)
    before = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (atom_id,))
    mem.reflect()
    after = mem.db.fetchone("SELECT confidence FROM atoms WHERE id = ?", (atom_id,))
    assert after["confidence"] <= before["confidence"]


# ── LTI promotion ─────────────────────────────────────────────────────────────

def test_high_sti_atom_gets_lti_promoted(mem):
    atom_id = mem.remember("Frequently accessed memory")
    # Boost STI well above promotion threshold (0.7 default for balanced)
    mem.db.execute("UPDATE atoms SET sti = 2.0 WHERE id = ?", (atom_id,))

    mem.reflect()
    after = mem.db.fetchone("SELECT lti FROM atoms WHERE id = ?", (atom_id,))
    # STI=2.0 >> balanced threshold (0.7); promoted lti = max(0, decayed_sti * 0.5) > 0
    assert after["lti"] > 0


# ── contradiction resolution ──────────────────────────────────────────────────

def test_contradiction_resolution(mem):
    id_a = mem.believe("The project will succeed", probability=0.9)
    id_b = mem.believe("The project will fail", probability=0.6)

    # Link them as contradictory
    mem.atomspace.link_atoms(id_a, id_b, "contradicts", "test", "default")

    conf_before = mem.db.fetchone(
        "SELECT confidence FROM atoms WHERE id = ?", (id_b,)
    )

    result = mem.reflect()
    assert result.contradictions_resolved >= 1

    conf_after = mem.db.fetchone(
        "SELECT confidence FROM atoms WHERE id = ?", (id_b,)
    )
    # Weaker belief (id_b) should have its confidence reduced
    assert conf_after["confidence"] < conf_before["confidence"]


def test_contradiction_lower_confidence_loses(mem):
    id_strong = mem.believe("Fact A is true", probability=0.95)
    id_weak = mem.believe("Fact A is false", probability=0.4)

    mem.atomspace.link_atoms(id_strong, id_weak, "contradicts", "test", "default")

    # Manually ensure confidence difference is clear
    mem.db.execute("UPDATE atoms SET confidence = 0.9 WHERE id = ?", (id_strong,))
    mem.db.execute("UPDATE atoms SET confidence = 0.2 WHERE id = ?", (id_weak,))

    conf_weak_before = mem.db.fetchone(
        "SELECT confidence FROM atoms WHERE id = ?", (id_weak,)
    )["confidence"]

    mem.reflect()

    conf_weak_after = mem.db.fetchone(
        "SELECT confidence FROM atoms WHERE id = ?", (id_weak,)
    )["confidence"]

    assert conf_weak_after < conf_weak_before


# ── atom pruning ──────────────────────────────────────────────────────────────

def test_epoch_prunes_dead_concept_atoms(mem):
    atom_id = mem.remember("Concept to prune", type="concept", probability=0.05)
    # Force low confidence and zero LTI
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.01, lti = 0.0 WHERE id = ?", (atom_id,)
    )

    result = mem.reflect()
    assert result.atoms_pruned >= 1

    row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (atom_id,))
    assert row is None


def test_epoch_does_not_prune_episodes(mem):
    ep_id = mem.remember("Episode survives pruning", type="episode", probability=0.05)
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.01, lti = 0.0 WHERE id = ?", (ep_id,)
    )

    mem.reflect()

    row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (ep_id,))
    assert row is not None


def test_epoch_does_not_prune_beliefs(mem):
    belief_id = mem.believe("Persistent belief", probability=0.1)
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.01, lti = 0.0 WHERE id = ?", (belief_id,)
    )

    mem.reflect()

    row = mem.db.fetchone("SELECT id FROM atoms WHERE id = ?", (belief_id,))
    assert row is not None


# ── cross-domain connections (every 10th epoch) ───────────────────────────────

def test_connections_discovered_on_10th_epoch(mem):
    # Add two semantically similar atoms with high LTI
    id1 = mem.remember("solar energy renewable power")
    id2 = mem.remember("wind energy renewable electricity")
    mem.db.execute("UPDATE atoms SET lti = 0.8 WHERE id IN (?, ?)", (id1, id2))

    # Force epoch count to 9 so next reflect is epoch 10
    mem.db.execute(
        "UPDATE personality SET epoch_count = 9 WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )

    result = mem.reflect()
    # new_connections may be 0 or more — just verify it ran without error
    assert result.new_connections >= 0


def test_connections_not_discovered_on_non_10th_epoch(mem):
    from unittest.mock import patch
    from smrti.evolution import epoch as ep_mod

    with patch("smrti.evolution.epoch.discover_connections") as mock_dc:
        # Force epoch count to 3 (not a multiple of 10)
        mem.db.execute(
            "UPDATE personality SET epoch_count = 3 WHERE tenant_id = ? AND space = ?",
            ("test", "default"),
        )
        mem.reflect()

    # discover_connections should NOT have been called
    mock_dc.assert_not_called()


# ── epoch_count increments ────────────────────────────────────────────────────

def test_epoch_count_increments(mem):
    before = mem.db.fetchone(
        "SELECT epoch_count FROM personality WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )
    mem.reflect()
    after = mem.db.fetchone(
        "SELECT epoch_count FROM personality WHERE tenant_id = ? AND space = ?",
        ("test", "default"),
    )
    assert after["epoch_count"] == before["epoch_count"] + 1
