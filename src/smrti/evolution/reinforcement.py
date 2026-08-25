"""Treating "this memory was used" as evidence.

Access boosts STI and nothing else, so confidence has only ever had one way
up: the caller restating the fact. Everything else rides it down to the
surfacing floor — and an atom below that floor can never be recalled, so it
can never be restated, so nothing ever lifts it back. The permanence floor
slowed that spiral; it did not close it.

A client that notices a recalled memory shaped the reply it just produced
knows something the engine cannot see, and reporting it here turns that into
evidence of the ordinary kind. What it is *not* is independent: the reply
used the atom because recall surfaced it, so use partly measures retrieval
rather than truth. Three things bound the damage — the weight is small, the
update is asymptotic (``conf + w·(1−conf)`` converges rather than
ratcheting), and a per-epoch cap stops a memory recalled every turn from
compounding. A wrong memory still sinks the moment the user contradicts it,
because direct testimony outweighs consumption by an order of magnitude.
"""
from __future__ import annotations

import json

from smrti.core.models import TruthValue
from smrti.core.provenance import ATOM_METADATA_JSON, SOURCE_AGENT
from smrti.evolution.truth import update_truth

# Deliberately an order of magnitude below the weight of a stated fact: being
# used is weak evidence, and the point is to hold a memory above the surfacing
# floor, not to promote it past what the user actually said.
DEFAULT_WEIGHT = 0.1

# How many reports one atom may bank between consolidations. Without it, a
# memory the client reports every turn would climb on conversation volume
# alone — a measure of how chatty the session is, not of how true anything is.
CAP_PER_EPOCH = 3


def _param(personality: dict, key: str, default: float) -> float:
    value = personality.get(key)
    return default if value is None else value


def reinforce_atoms(
    atom_ids: list[str],
    tenant_id: str,
    space: str,
    db,
    weight: float = DEFAULT_WEIGHT,
) -> dict:
    """Apply use-evidence to each atom, returning what was and was not applied.

    Every atom is looked up inside ``(tenant_id, space)``, so an id from
    another partition is simply unknown here rather than reachable.
    """
    weight = max(0.0, min(1.0, weight))
    personality = db.fetchone(
        "SELECT * FROM personality WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    p = dict(personality) if personality else {}
    lr = _param(p, "confidence_update_lr", 0.3)
    agent_trust = _param(p, "agent_source_trust", 0.5)
    epoch = int(_param(p, "epoch_count", 0))

    reinforced: list[str] = []
    skipped: list[dict] = []

    for atom_id in dict.fromkeys(atom_ids):
        row = db.fetchone(
            "SELECT probability, confidence, metadata FROM atoms "
            "WHERE id = ? AND tenant_id = ? AND space = ?",
            (atom_id, tenant_id, space),
        )
        if row is None:
            skipped.append({"id": atom_id, "reason": "unknown"})
            continue
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}

        # forget() sank this on purpose. Consumption evidence must not undo a
        # deliberate act, the same way the epoch's permanence lift does not.
        if metadata.get("forgotten"):
            skipped.append({"id": atom_id, "reason": "forgotten"})
            continue

        # The count is engine-written, but so is the metadata this module
        # already refuses to trust; a string here would raise on the compare.
        used = 0
        if metadata.get("reinforced_epoch") == epoch:
            try:
                used = int(metadata.get("reinforced_count", 0))
            except (TypeError, ValueError):
                used = 0
        if used >= CAP_PER_EPOCH:
            skipped.append({"id": atom_id, "reason": "capped"})
            continue

        applied = weight
        if metadata.get("source") == SOURCE_AGENT:
            applied *= agent_trust

        current = TruthValue(
            probability=row["probability"] or 0.0,
            confidence=row["confidence"] or 0.0,
        )
        # The atom's own probability is the observation: using a memory says
        # it mattered, not that it is truer than it claimed to be. Only
        # confidence moves.
        updated = update_truth(current, current.probability, applied, lr)

        db.execute(
            f"""UPDATE atoms SET
                    confidence = ?,
                    metadata = json_set(
                        json_set({ATOM_METADATA_JSON}, '$.reinforced_epoch', ?),
                        '$.reinforced_count', ?),
                    updated_at = datetime('now')
                WHERE id = ? AND tenant_id = ? AND space = ?""",
            (updated.confidence, epoch, used + 1, atom_id, tenant_id, space),
        )
        reinforced.append(atom_id)

    return {"reinforced": reinforced, "skipped": skipped}
