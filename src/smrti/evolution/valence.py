"""Emotional valence propagation."""
from __future__ import annotations

from smrti.core.provenance import ATOM_FORGOTTEN

# One neighbour's nudge toward the source's tone. A forgotten atom is left
# alone: forgetting it must not leave it absorbing the moods around it.
NUDGE_SQL = f"""UPDATE atoms SET
       valence   = MAX(-1.0, MIN(1.0, valence   + ? * (? - valence))),
       intensity = MAX(0.0,  MIN(1.0, intensity + ? * (? - intensity)))
   WHERE id = ? AND tenant_id = ? AND space = ? AND NOT {ATOM_FORGOTTEN}"""


def nudge_params(
    valence: float,
    intensity: float,
    propagation_factor: float,
    neighbor_ids: list[str],
    tenant_id: str,
    space: str,
    mood_inertia: float = 0.8,
) -> list[tuple]:
    """The :data:`NUDGE_SQL` parameters that nudge each neighbour once.

    The step is ``propagation_factor * (1 - mood_inertia)``, so the blend's
    fixpoint is the source valence itself, not zero. High-inertia presets
    drift slowly; low-inertia ones (e.g. empathetic=0.4) react faster. An
    atom linked twice is nudged once: the second edge is not a second mood.
    """
    if abs(valence) * propagation_factor < 0.01:
        return []
    keep = max(0.0, min(1.0, mood_inertia))
    step = propagation_factor * (1.0 - keep)
    return [
        (step, valence, step, intensity, nid, tenant_id, space)
        for nid in dict.fromkeys(nid for nid in neighbor_ids if nid)
    ]


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
    """Propagate one atom's emotional valence to its 1-hop neighbours in the same space.

    The epoch propagates from every active atom at once (see
    ``epoch._propagate``); this nudges one atom's neighbourhood on its own,
    through the same :func:`nudge_params`.
    """
    # Bail before spending a query on the neighbour lookup.
    if abs(valence) * propagation_factor < 0.01:
        return

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

    params = nudge_params(
        valence, intensity, propagation_factor, neighbor_ids, tenant_id, space, mood_inertia
    )
    # One statement for the whole neighbourhood rather than a commit per
    # neighbour.
    if params:
        db.execute_many(NUDGE_SQL, params)
