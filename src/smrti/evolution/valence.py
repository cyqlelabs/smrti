"""Emotional valence propagation."""
from __future__ import annotations


def propagate_valence(
    atom_id: str,
    valence: float,
    intensity: float,
    propagation_factor: float,
    db,
    tenant_id: str,
    mood_inertia: float = 0.8,
) -> None:
    """Propagate emotional valence to 1-hop connected atoms, attenuated by factor.

    Uses a weighted blend controlled by ``mood_inertia`` (0–1): existing valence
    contributes ``mood_inertia`` fraction, incoming signal contributes the rest,
    keeping emotional drift gradual for high-inertia presets and more reactive
    for low-inertia ones (e.g. empathetic=0.4).
    """
    spread_v = valence * propagation_factor
    spread_i = intensity * propagation_factor

    if abs(spread_v) < 0.01:
        return

    keep = max(0.0, min(1.0, mood_inertia))
    absorb = 1.0 - keep

    # Relation atoms store their endpoints in source_id/target_id columns rather
    # than as further relation edges, so the standard forward/backward query
    # finds nothing for them.  Use the endpoints directly instead.
    row = db.fetchone("SELECT type, source_id, target_id FROM atoms WHERE id = ?", (atom_id,))
    if row and row["type"] == "relation":
        neighbor_ids = [x for x in (row["source_id"], row["target_id"]) if x]
    else:
        forward = db.fetchall(
            "SELECT target_id FROM atoms WHERE source_id = ? AND type = 'relation' AND tenant_id = ?",
            (atom_id, tenant_id),
        )
        backward = db.fetchall(
            "SELECT source_id FROM atoms WHERE target_id = ? AND type = 'relation' AND tenant_id = ?",
            (atom_id, tenant_id),
        )
        neighbor_ids = [r["target_id"] for r in forward if r["target_id"]]
        neighbor_ids += [r["source_id"] for r in backward if r["source_id"]]

    for nid in neighbor_ids:
        db.execute(
            """UPDATE atoms SET
                   valence   = (valence   * ? + ? * ?),
                   intensity = MIN(intensity * ? + ? * ?, 1.0)
               WHERE id = ?""",
            (keep, spread_v, absorb, keep, spread_i, absorb, nid),
        )
