"""STI/LTI decay and propagation."""
from __future__ import annotations

from smrti.core.provenance import ATOM_FORGOTTEN


def decay_sti(sti: float, decay_rate: float) -> float:
    """Apply multiplicative decay to short-term importance."""
    return sti * (1.0 - decay_rate)


def decay_lti(lti: float, decay_rate: float) -> float:
    """Apply multiplicative decay to long-term importance."""
    return lti * (1.0 - decay_rate)


def promote_lti(sti: float, lti: float, threshold: float) -> float:
    """Promote STI to LTI when STI exceeds threshold.

    STI is clamped to 1.0 before scaling: STI saturates at 3.0, so an unclamped
    ``sti * 0.5`` pins LTI to its own ceiling on a single promotion and no
    amount of decay can ever bring it back down.
    """
    if sti > threshold:
        return max(lti, min(sti, 1.0) * 0.5)
    return lti


def sti_moves(
    atom_id: str,
    boost: float,
    propagation_factor: float,
    neighbor_ids: list[str],
    tenant_id: str,
    space: str,
) -> list[tuple]:
    """The statements that move a share of *atom_id*'s STI to its neighbours.

    *neighbor_ids* must already be the ones this space can credit (see
    :func:`propagate_sti`). Single-hop only — no recursion, no oscillation
    risk.

    Activation is *moved*, not copied: the budget is divided among the
    neighbors and deducted from the source. Copying makes propagation a net
    source of STI — total activation then grows faster than decay removes it,
    every atom in a linked cluster saturates at the ceiling, and because
    promotion re-asserts an LTI floor for anything above the threshold, LTI can
    never fall back to the prune floor. Spreading a fixed budget also means
    fan-out dilutes: a node with 47 children gives each 1/47th, so a noisy
    extraction can no longer keep its whole cluster maximally salient.

    The credits and the debit are one move of activation and belong in one
    transaction: committing them one neighbour at a time cost a commit per
    edge and could leave the source charged for shares not yet delivered.
    """
    budget = boost * propagation_factor
    if budget < 0.01:
        return []
    # Deduplicate: an atom linked twice must not collect twice the share.
    unique_ids = list(dict.fromkeys(nid for nid in neighbor_ids if nid))
    if not unique_ids:
        return []
    spread = budget / len(unique_ids)
    if spread < 0.01:
        return []
    return [
        *(
            (
                "UPDATE atoms SET sti = MIN(sti + ?, 3.0) WHERE id = ? AND tenant_id = ? AND space = ?",
                (spread, nid, tenant_id, space),
            )
            for nid in unique_ids
        ),
        (
            "UPDATE atoms SET sti = MAX(sti - ?, 0.0) WHERE id = ? AND tenant_id = ? AND space = ?",
            (spread * len(unique_ids), atom_id, tenant_id, space),
        ),
    ]


def propagate_sti(
    atom_id: str, boost: float, propagation_factor: float, db, tenant_id: str, space: str
) -> None:
    """Spread a fraction of one atom's STI to its 1-hop neighbors within the same space.

    The epoch propagates from every active atom at once (see
    ``epoch._propagate``); this moves one atom's share on its own, through
    the same :func:`sti_moves`.
    """
    # Dividing only shrinks the per-neighbor share, so a budget under the floor
    # can never clear it — bail before spending a query on the neighbor lookup.
    if boost * propagation_factor < 0.01:
        return

    # Relation atoms store their endpoints in source_id/target_id columns —
    # the standard forward/backward query finds nothing for them.
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

    unique_ids = list(dict.fromkeys(neighbor_ids))
    # Keep only neighbors this space can actually credit. A relation atom's
    # endpoints — and a bridge edge's in particular — may sit in another space,
    # where the UPDATE is a no-op; charging the source for a share nobody
    # received would make propagation destroy activation instead of moving it.
    # A forgotten atom is not credited either: forget() zeroed its attention.
    if unique_ids:
        ph = ",".join("?" * len(unique_ids))
        in_space = {
            r["id"]
            for r in db.fetchall(
                f"SELECT id FROM atoms WHERE id IN ({ph}) AND tenant_id = ? AND space = ? AND NOT {ATOM_FORGOTTEN}",
                (*unique_ids, tenant_id, space),
            )
        }
        unique_ids = [nid for nid in unique_ids if nid in in_space]
    statements = sti_moves(atom_id, boost, propagation_factor, unique_ids, tenant_id, space)
    if statements:
        db.execute_batch(statements)
