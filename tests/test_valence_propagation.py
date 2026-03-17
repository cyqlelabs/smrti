"""Tests for emotional valence propagation (evolution/valence.py)."""
from unittest.mock import MagicMock, call

from smrti.evolution.valence import propagate_valence


def _mock_db(forward_ids=None, backward_ids=None):
    db = MagicMock()
    forward = [{"target_id": tid} for tid in (forward_ids or [])]
    backward = [{"source_id": sid} for sid in (backward_ids or [])]
    db.fetchall.side_effect = [forward, backward]
    return db


# ── early-exit when spread is negligible ──────────────────────────────────────

def test_no_propagation_when_valence_too_small():
    db = _mock_db(forward_ids=["n1"])
    propagate_valence("a1", valence=0.05, intensity=0.5, propagation_factor=0.1,
                      db=db, tenant_id="t")
    # spread_v = 0.05 * 0.1 = 0.005 < 0.01
    db.execute.assert_not_called()


def test_no_propagation_when_factor_zero():
    db = _mock_db(forward_ids=["n1"])
    propagate_valence("a1", valence=1.0, intensity=1.0, propagation_factor=0.0,
                      db=db, tenant_id="t")
    db.execute.assert_not_called()


# ── propagates to neighbors ───────────────────────────────────────────────────

def test_propagates_to_forward_neighbors():
    db = _mock_db(forward_ids=["n1", "n2"])
    propagate_valence("a1", valence=0.8, intensity=0.6, propagation_factor=0.3,
                      db=db, tenant_id="t")
    assert db.execute.call_count == 2


def test_propagates_to_backward_neighbors():
    db = _mock_db(backward_ids=["b1"])
    propagate_valence("a1", valence=0.8, intensity=0.6, propagation_factor=0.3,
                      db=db, tenant_id="t")
    assert db.execute.call_count == 1


def test_propagates_to_both_directions():
    db = _mock_db(forward_ids=["f1"], backward_ids=["b1"])
    propagate_valence("a1", valence=-0.5, intensity=0.7, propagation_factor=0.5,
                      db=db, tenant_id="t")
    assert db.execute.call_count == 2


# ── 80/20 blend ───────────────────────────────────────────────────────────────

def test_sql_uses_80_20_blend():
    db = _mock_db(forward_ids=["n1"])
    propagate_valence("a1", valence=0.6, intensity=0.4, propagation_factor=0.5,
                      db=db, tenant_id="t")
    sql_call = db.execute.call_args
    sql = sql_call[0][0]
    assert "0.8" in sql and "0.2" in sql


# ── None IDs filtered ─────────────────────────────────────────────────────────

def test_none_ids_are_filtered():
    db = MagicMock()
    db.fetchall.side_effect = [
        [{"target_id": None}, {"target_id": "n1"}],
        [{"source_id": None}],
    ]
    propagate_valence("a1", valence=0.8, intensity=0.5, propagation_factor=0.5,
                      db=db, tenant_id="t")
    assert db.execute.call_count == 1


# ── Negative valence propagates (value passes through) ───────────────────────

def test_negative_valence_propagates():
    db = _mock_db(forward_ids=["n1"])
    propagate_valence("a1", valence=-0.8, intensity=0.9, propagation_factor=0.5,
                      db=db, tenant_id="t")
    # spread_v = -0.8 * 0.5 = -0.4 — abs(-0.4) >= 0.01, should propagate
    assert db.execute.call_count == 1
    args = db.execute.call_args[0][1]
    # First param is spread_v = -0.4
    assert args[0] < 0
