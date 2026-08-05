"""Emotional valence propagation."""
from __future__ import annotations


def propagate_valence(
    atom_id: str,
    valence: float,
    intensity: float,
    propagation_factor: float,
    db,
    tenant_id: str,
    space: str,
    mood_inertia: float = 0.8,
) -> None:
    """Propagate emotional valence to 1-hop connected atoms within the same space.

    Nudges each neighbor toward the source valence with step size
    ``propagation_factor * (1 - mood_inertia)`` so the blend fixpoint is the
    source valence itself — strong neighbor valence is reinforced, never eroded
    toward zero.  High-inertia presets drift slowly; low-inertia ones
    (e.g. empathetic=0.4) react faster.
    """
    if abs(valence) * propagation_factor < 0.01:
        return

    keep = max(0.0, min(1.0, mood_inertia))
    step = propagation_factor * (1.0 - keep)

    # Relation atoms store their endpoints in source_id/target_id columns rather
    # than as further relation edges, so the standard forward/backward query
    # finds nothing for them.  Use the endpoints directly instead.
    row = db.fetchone("SELECT type, source_id, target_id FROM atoms WHERE id = ?", (atom_id,))
    if row and row["type"] == "relation":
        neighbor_ids = [x for x in (row["source_id"], row["target_id"]) if x]
    else:
        forward = db.fetchall(
            "SELECT target_id FROM atoms WHERE source_id = ? AND type = 'relation' AND tenant_id = ? AND space = ?",
            (atom_id, tenant_id, space),
        )
        backward = db.fetchall(
            "SELECT source_id FROM atoms WHERE target_id = ? AND type = 'relation' AND tenant_id = ? AND space = ?",
            (atom_id, tenant_id, space),
        )
        neighbor_ids = [r["target_id"] for r in forward if r["target_id"]]
        neighbor_ids += [r["source_id"] for r in backward if r["source_id"]]

    for nid in neighbor_ids:
        db.execute(
            """UPDATE atoms SET
                   valence   = MAX(-1.0, MIN(1.0, valence   + ? * (? - valence))),
                   intensity = MAX(0.0,  MIN(1.0, intensity + ? * (? - intensity)))
               WHERE id = ? AND tenant_id = ? AND space = ?""",
            (step, valence, step, intensity, nid, tenant_id, space),
        )
