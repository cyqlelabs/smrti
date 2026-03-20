"""Tests for STI/LTI decay and propagation (evolution/attention.py)."""
from unittest.mock import MagicMock

from smrti.evolution.attention import decay_sti, promote_lti, propagate_sti


# ── decay_sti ─────────────────────────────────────────────────────────────────

def test_decay_sti_reduces_value():
    result = decay_sti(1.0, 0.1)
    assert abs(result - 0.9) < 1e-9


def test_decay_sti_zero_rate():
    result = decay_sti(0.8, 0.0)
    assert result == 0.8


def test_decay_sti_full_rate():
    result = decay_sti(0.5, 1.0)
    assert result == 0.0


def test_decay_sti_small_value():
    result = decay_sti(0.01, 0.5)
    assert abs(result - 0.005) < 1e-9


# ── promote_lti ───────────────────────────────────────────────────────────────

def test_promote_lti_when_sti_above_threshold():
    result = promote_lti(sti=1.0, lti=0.2, threshold=0.5)
    assert result == max(0.2, 1.0 * 0.5)  # = 0.5


def test_promote_lti_no_op_when_below_threshold():
    result = promote_lti(sti=0.3, lti=0.6, threshold=0.5)
    assert result == 0.6


def test_promote_lti_does_not_decrease_existing_lti():
    # If existing LTI > sti*0.5, keep existing
    result = promote_lti(sti=0.8, lti=0.9, threshold=0.5)
    assert result == 0.9


def test_promote_lti_exact_threshold():
    # sti > threshold (strictly), so just above boundary triggers promotion
    result = promote_lti(sti=0.5001, lti=0.0, threshold=0.5)
    assert result == 0.5001 * 0.5


def test_promote_lti_at_threshold_no_promote():
    # sti == threshold is NOT > threshold
    result = promote_lti(sti=0.5, lti=0.0, threshold=0.5)
    assert result == 0.0


# ── propagate_sti ─────────────────────────────────────────────────────────────

def _mock_db(forward_ids=None, backward_ids=None):
    db = MagicMock()
    forward = [{"target_id": tid} for tid in (forward_ids or [])]
    backward = [{"source_id": sid} for sid in (backward_ids or [])]
    db.fetchall.side_effect = [forward, backward]
    return db


def test_propagate_sti_updates_neighbors():
    db = _mock_db(forward_ids=["n1", "n2"])
    propagate_sti("atom1", boost=1.0, propagation_factor=0.2, db=db, tenant_id="t", space="s")
    # Should update both neighbors
    assert db.execute.call_count == 2


def test_propagate_sti_no_op_when_spread_too_small():
    db = _mock_db(forward_ids=["n1"])
    propagate_sti("atom1", boost=0.04, propagation_factor=0.2, db=db, tenant_id="t", space="s")
    # spread = 0.04 * 0.2 = 0.008 < 0.01
    db.execute.assert_not_called()


def test_propagate_sti_spread_exactly_at_boundary():
    # spread = 0.1 * 0.1 = 0.01 — borderline (< 0.01 is skipped, so 0.01 propagates)
    db = _mock_db(forward_ids=["n1"])
    propagate_sti("atom1", boost=0.1, propagation_factor=0.1, db=db, tenant_id="t", space="s")
    # 0.01 is NOT < 0.01, so it should propagate
    assert db.execute.call_count == 1


def test_propagate_sti_no_neighbors():
    db = _mock_db()
    propagate_sti("atom1", boost=1.0, propagation_factor=0.5, db=db, tenant_id="t", space="s")
    db.execute.assert_not_called()


def test_propagate_sti_none_ids_filtered():
    db = MagicMock()
    db.fetchall.side_effect = [
        [{"target_id": None}, {"target_id": "n1"}],
        [{"source_id": None}],
    ]
    propagate_sti("atom1", boost=1.0, propagation_factor=0.2, db=db, tenant_id="t", space="s")
    # Only "n1" is a valid neighbor
    assert db.execute.call_count == 1
