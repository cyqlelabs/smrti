"""Consolidation epoch: the main memory evolution loop."""
from __future__ import annotations

from engram.core.models import EpochResult, TruthValue
from engram.evolution.connections import discover_connections
from engram.evolution.truth import update_truth


def run_epoch(tenant_id: str, space: str, db, embed_engine) -> EpochResult:
    """Single deterministic consolidation pass for a (tenant_id, space) pair.

    Executes in order:
      1. Apply pending evidence to update belief probabilities/confidence
      2. Decay STI and confidence for all atoms in this space
      3. Promote high-STI atoms to LTI
      4. Resolve contradictions by weakening the less confident belief
      5. Discover cross-domain connections (every 10th epoch)
      6. Prune dead atoms below confidence and LTI floors
    """
    db.execute(
        "UPDATE personality SET epoch_count = epoch_count + 1 WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    epoch_row = db.fetchone(
        "SELECT epoch_count FROM personality WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    epoch_count = epoch_row["epoch_count"] if epoch_row else 1

    beliefs_updated = 0
    atoms_pruned = 0
    lti_promoted = 0
    new_connections = 0
    contradictions_resolved = 0

    personality = db.fetchone(
        "SELECT * FROM personality WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    if not personality:
        return EpochResult(
            beliefs_updated=0,
            atoms_decayed=0,
            atoms_pruned=0,
            lti_promoted=0,
            new_connections=0,
            contradictions_resolved=0,
        )

    p = dict(personality)
    lr = p.get("confidence_update_lr", 0.3)

    # 1. Process pending evidence
    pending = db.fetchall(
        "SELECT * FROM evidence WHERE processed = 0 AND tenant_id = ? AND space = ? ORDER BY created_at",
        (tenant_id, space),
    )
    for ev in pending:
        atom_row = db.fetchone(
            "SELECT * FROM atoms WHERE id = ?", (ev["atom_id"],)
        )
        if atom_row:
            current = TruthValue(
                probability=atom_row["probability"],
                confidence=atom_row["confidence"],
            )
            updated = update_truth(
                current, ev["observed_probability"], ev["weight"], lr
            )
            db.execute(
                "UPDATE atoms SET probability = ?, confidence = ?, updated_at = datetime('now') WHERE id = ?",
                (updated.probability, updated.confidence, ev["atom_id"]),
            )
            db.execute(
                "UPDATE evidence SET processed = 1 WHERE id = ?", (ev["id"],)
            )
            beliefs_updated += 1

    # 2. Decay STI and confidence for all atoms in this space
    db.execute(
        """UPDATE atoms SET
               sti        = sti        * (1.0 - ?),
               confidence = confidence * (1.0 - ?),
               updated_at = datetime('now')
           WHERE tenant_id = ? AND space = ?""",
        (p["sti_decay_rate"], p["confidence_decay_rate"], tenant_id, space),
    )

    # 3. Promote high-STI atoms to LTI
    before_lti = db.fetchone(
        "SELECT COUNT(*) as n FROM atoms WHERE tenant_id = ? AND space = ? AND lti > 0",
        (tenant_id, space),
    )
    db.execute(
        "UPDATE atoms SET lti = MAX(lti, sti * 0.5) WHERE tenant_id = ? AND space = ? AND sti > ?",
        (tenant_id, space, p["lti_promotion_threshold"]),
    )
    after_lti = db.fetchone(
        "SELECT COUNT(*) as n FROM atoms WHERE tenant_id = ? AND space = ? AND lti > 0",
        (tenant_id, space),
    )
    lti_promoted = max(
        0,
        (after_lti["n"] if after_lti else 0) - (before_lti["n"] if before_lti else 0),
    )

    # 4. Resolve contradictions within this space
    contradictions = db.fetchall(
        """SELECT id, source_id, target_id FROM atoms
           WHERE type = 'relation' AND relation = 'contradicts'
             AND tenant_id = ? AND space = ?""",
        (tenant_id, space),
    )
    for c in contradictions:
        if not c["source_id"] or not c["target_id"]:
            continue
        src = db.fetchone(
            "SELECT probability, confidence FROM atoms WHERE id = ?", (c["source_id"],)
        )
        tgt = db.fetchone(
            "SELECT probability, confidence FROM atoms WHERE id = ?", (c["target_id"],)
        )
        if src and tgt:
            loser_id = (
                c["source_id"]
                if src["confidence"] < tgt["confidence"]
                else c["target_id"]
            )
            db.execute(
                "UPDATE atoms SET confidence = confidence * 0.8 WHERE id = ?",
                (loser_id,),
            )
            contradictions_resolved += 1

    # 5. Cross-domain connection discovery (every 10th epoch)
    if epoch_count % 10 == 0:
        new_connections = discover_connections(tenant_id, space, db, embed_engine)

    # 6. Prune atoms below both confidence and LTI floors (episodes are exempt)
    min_conf = p.get("min_confidence_to_surface", 0.1)
    dead_rows = db.fetchall(
        """SELECT id FROM atoms
           WHERE tenant_id = ? AND space = ?
             AND confidence < ? AND lti < 0.05 AND type != 'episode'""",
        (tenant_id, space, min_conf),
    )
    atoms_pruned = len(dead_rows)

    for row in dead_rows:
        atom_id = row["id"]
        db.execute(
            "DELETE FROM atoms WHERE type = 'relation' AND (source_id = ? OR target_id = ?)",
            (atom_id, atom_id),
        )
        db.execute("DELETE FROM vec_atoms WHERE atom_id = ?", (atom_id,))
        db.execute("DELETE FROM evidence WHERE atom_id = ?", (atom_id,))
        db.execute("DELETE FROM atoms WHERE id = ?", (atom_id,))

    total = db.fetchone(
        "SELECT COUNT(*) as n FROM atoms WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    atoms_decayed = total["n"] if total else 0

    return EpochResult(
        beliefs_updated=beliefs_updated,
        atoms_decayed=atoms_decayed,
        atoms_pruned=atoms_pruned,
        lti_promoted=lti_promoted,
        new_connections=new_connections,
        contradictions_resolved=contradictions_resolved,
    )
