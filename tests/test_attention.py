"""Tests for STI/LTI decay and propagation (evolution/attention.py)."""
from unittest.mock import MagicMock

from smrti.evolution.attention import decay_lti, decay_sti, promote_lti, propagate_sti


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
    # Both neighbors, plus the deduction from the source.
    assert db.execute.call_count == 3


def test_propagate_sti_no_op_when_spread_too_small():
    db = _mock_db(forward_ids=["n1"])
    propagate_sti("atom1", boost=0.04, propagation_factor=0.2, db=db, tenant_id="t", space="s")
    # spread = 0.04 * 0.2 = 0.008 < 0.01
    db.execute.assert_not_called()


def test_propagate_sti_spread_exactly_at_boundary():
    # spread = 0.1 * 0.1 = 0.01 — borderline (< 0.01 is skipped, so 0.01 propagates)
    db = _mock_db(forward_ids=["n1"])
    propagate_sti("atom1", boost=0.1, propagation_factor=0.1, db=db, tenant_id="t", space="s")
    # 0.01 is NOT < 0.01, so it should propagate: one neighbor + source deduction.
    assert db.execute.call_count == 2


def test_propagate_sti_no_neighbors():
    db = _mock_db()
    propagate_sti("atom1", boost=1.0, propagation_factor=0.5, db=db, tenant_id="t", space="s")
    db.execute.assert_not_called()


def test_propagate_sti_divides_budget_across_fan_out():
    """A hub must split one budget among its neighbors, not hand each the full share.

    A flat per-neighbor boost lets a high-fan-out atom emit more STI than it
    holds, which is what let a single noisy extraction pin its whole cluster.
    """
    db = _mock_db(forward_ids=["n1", "n2", "n3", "n4"])
    propagate_sti("hub", boost=1.0, propagation_factor=0.4, db=db, tenant_id="t", space="s")
    amounts = [c.args[1][0] for c in db.execute.call_args_list]
    neighbor_shares, deducted = amounts[:-1], amounts[-1]
    assert neighbor_shares == [0.1, 0.1, 0.1, 0.1]  # 0.4 budget / 4 neighbors
    assert deducted == 0.4


def test_propagate_sti_is_conservative():
    """What the neighbors gain is exactly what the source gives up.

    Without this, propagation is a net source of STI: total activation grows
    faster than decay removes it, every linked atom saturates at the ceiling,
    and LTI can never fall back to the prune floor.
    """
    db = _mock_db(forward_ids=["n1", "n2"], backward_ids=["n3"])
    propagate_sti("hub", boost=2.0, propagation_factor=0.3, db=db, tenant_id="t", space="s")
    calls = db.execute.call_args_list
    gained = sum(c.args[1][0] for c in calls[:-1])
    given_up = calls[-1].args[1][0]
    assert given_up == gained
    assert calls[-1].args[1][1] == "hub"
    assert "sti - ?" in calls[-1].args[0]


def test_propagate_sti_deduplicates_neighbors():
    """An atom linked to the source twice collects one share, not two."""
    db = _mock_db(forward_ids=["n1", "n1"], backward_ids=["n1"])
    propagate_sti("hub", boost=1.0, propagation_factor=0.3, db=db, tenant_id="t", space="s")
    # One neighbor update plus the source deduction.
    assert db.execute.call_count == 2
    assert db.execute.call_args_list[0].args[1][0] == 0.3


def test_propagate_sti_skips_when_per_neighbor_share_underflows():
    """A budget that clears the floor can still leave each neighbor below it."""
    db = _mock_db(forward_ids=[f"n{i}" for i in range(20)])
    # budget = 0.1 (>= 0.01), but 0.1 / 20 = 0.005 per neighbor.
    propagate_sti("hub", boost=1.0, propagation_factor=0.1, db=db, tenant_id="t", space="s")
    db.execute.assert_not_called()


# ── decay_lti ─────────────────────────────────────────────────────────────────

def test_decay_lti_reduces_value():
    assert decay_lti(0.8, 0.02) == 0.8 * 0.98


def test_decay_lti_zero_rate_is_identity():
    assert decay_lti(0.4, 0.0) == 0.4


def test_promote_lti_clamps_saturated_sti():
    """STI saturates at 3.0; an unclamped sti * 0.5 would pin LTI to its ceiling.

    Once LTI is pinned no amount of decay brings it back under the prune floor,
    so the graph can never shed anything it briefly found interesting.
    """
    assert promote_lti(sti=3.0, lti=0.0, threshold=0.4) == 0.5
    assert promote_lti(sti=2.0, lti=0.0, threshold=0.4) == 0.5


def test_promote_lti_and_decay_reach_a_fixed_point_below_ceiling():
    """Repeated promotion at saturated STI must not ratchet LTI upward forever."""
    lti = 0.0
    for _ in range(50):
        lti = decay_lti(promote_lti(sti=3.0, lti=lti, threshold=0.4), 0.02)
    assert lti < 0.5


def test_propagate_sti_none_ids_filtered():
    db = MagicMock()
    db.fetchall.side_effect = [
        [{"target_id": None}, {"target_id": "n1"}],
        [{"source_id": None}],
    ]
    propagate_sti("atom1", boost=1.0, propagation_factor=0.2, db=db, tenant_id="t", space="s")
    # Only "n1" is a valid neighbor, plus the deduction from the source.
    assert db.execute.call_count == 2
