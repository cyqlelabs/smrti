"""STI/LTI decay and propagation."""
from __future__ import annotations


def decay_sti(sti: float, decay_rate: float) -> float:
    """Apply multiplicative decay to short-term importance."""
    return sti * (1.0 - decay_rate)


def promote_lti(sti: float, lti: float, threshold: float) -> float:
    """Promote STI to LTI when STI exceeds threshold."""
    if sti > threshold:
        return max(lti, sti * 0.5)
    return lti


def propagate_sti(
    atom_id: str, boost: float, propagation_factor: float, db, agent_id: str
) -> None:
    """Spread a fraction of STI to 1-hop neighbors.

    Single-hop only — no recursion, no oscillation risk.
    """
    spread = boost * propagation_factor
    if spread < 0.01:
        return

    forward = db.fetchall(
        "SELECT target_id FROM atoms WHERE source_id = ? AND type = 'relation' AND agent_id = ?",
        (atom_id, agent_id),
    )
    backward = db.fetchall(
        "SELECT source_id FROM atoms WHERE target_id = ? AND type = 'relation' AND agent_id = ?",
        (atom_id, agent_id),
    )

    neighbor_ids = [r["target_id"] for r in forward if r["target_id"]]
    neighbor_ids += [r["source_id"] for r in backward if r["source_id"]]

    for nid in neighbor_ids:
        db.execute(
            "UPDATE atoms SET sti = MIN(sti + ?, 3.0) WHERE id = ?",
            (spread, nid),
        )
