"""Emotional valence propagation."""
from __future__ import annotations


def propagate_valence(
    atom_id: str,
    valence: float,
    intensity: float,
    propagation_factor: float,
    db,
    agent_id: str,
) -> None:
    """Propagate emotional valence to 1-hop connected atoms, attenuated by factor.

    Uses an 80/20 weighted blend: existing valence contributes 80%, the incoming
    propagated signal contributes 20%, keeping emotional drift gradual.
    """
    spread_v = valence * propagation_factor
    spread_i = intensity * propagation_factor

    if abs(spread_v) < 0.01:
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
            """UPDATE atoms SET
                   valence   = (valence   * 0.8 + ? * 0.2),
                   intensity = MIN(intensity * 0.8 + ? * 0.2, 1.0)
               WHERE id = ?""",
            (spread_v, spread_i, nid),
        )
