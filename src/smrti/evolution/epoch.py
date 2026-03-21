"""Consolidation epoch: the main memory evolution loop."""
from __future__ import annotations

from smrti.core.models import EpochResult, TruthValue
from smrti.evolution.attention import propagate_sti
from smrti.evolution.connections import discover_connections
from smrti.evolution.healing import heal_orphaned_episodes
from smrti.evolution.truth import update_truth
from smrti.evolution.valence import propagate_valence
from smrti.spaces.set_ops import space_overlap
from smrti.spaces.emergence import materialize_bridge


def run_epoch(tenant_id: str, space: str, db, embed_engine) -> EpochResult:
    """Single deterministic consolidation pass for a (tenant_id, space) pair.

    Executes in order:
      1. Apply pending evidence to update belief probabilities/confidence
      2. Decay STI and confidence for all atoms in this space
      3. Propagate STI and valence to 1-hop neighbors of active atoms
      4. Heal orphaned episodes (link to most salient person)
      5. Promote high-STI atoms to LTI
      6. Resolve contradictions by weakening the less confident belief
      7. Discover cross-domain connections (every 10th epoch)
      8. Prune dead atoms below confidence and LTI floors
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
    decay_count_row = db.fetchone(
        "SELECT COUNT(*) as n FROM atoms WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    atoms_decayed = decay_count_row["n"] if decay_count_row else 0
    db.execute(
        """UPDATE atoms SET
               sti        = sti        * (1.0 - ?),
               confidence = confidence * (1.0 - ?),
               updated_at = datetime('now')
           WHERE tenant_id = ? AND space = ?""",
        (p["sti_decay_rate"], p["confidence_decay_rate"], tenant_id, space),
    )

    # 2b. Propagate STI and valence to 1-hop neighbors
    propagation_factor = p.get("sti_propagation_factor", 0.15)
    valence_prop_factor = p.get("valence_propagation", 0.1)
    mood_inertia = p.get("mood_inertia", 0.8)
    if propagation_factor > 0 or valence_prop_factor > 0:
        active = db.fetchall(
            "SELECT id, sti, valence, intensity FROM atoms WHERE tenant_id = ? AND space = ? AND (sti > 0.3 OR (valence < -0.3 AND intensity > 0.3))",
            (tenant_id, space),
        )
        for row in active:
            if propagation_factor > 0 and row["sti"] > 0.3:
                propagate_sti(row["id"], row["sti"], propagation_factor, db, tenant_id, space)
            if valence_prop_factor > 0 and abs(row["valence"]) > 0.3 and row["intensity"] > 0.3:
                propagate_valence(row["id"], row["valence"], row["intensity"], valence_prop_factor, db, tenant_id, space, mood_inertia=mood_inertia)

    # 2c. Heal orphaned episodes (link to most salient person)
    orphans_healed = heal_orphaned_episodes(tenant_id, space, db)

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

    # 5b. Cross-space bridge emergence (every 10th epoch)
    bridges_created = 0
    if epoch_count % 10 == 0:
        bridges_created = _discover_bridges(tenant_id, space, db, embed_engine)

    # 6. Prune atoms below both confidence and LTI floors.
    # Episodes, beliefs, and relations are exempt from direct pruning.
    # Relations are cascade-deleted when their endpoint atoms are pruned (see loop below).
    min_conf = p.get("min_confidence_to_surface", 0.1)
    dead_rows = db.fetchall(
        """SELECT id FROM atoms
           WHERE tenant_id = ? AND space = ?
             AND confidence < ? AND lti < 0.05 AND type NOT IN ('episode', 'belief', 'relation')""",
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
        db.execute("DELETE FROM aliases WHERE atom_id = ?", (atom_id,))
        db.execute("DELETE FROM atoms WHERE id = ?", (atom_id,))

    return EpochResult(
        beliefs_updated=beliefs_updated,
        atoms_decayed=atoms_decayed,
        atoms_pruned=atoms_pruned,
        lti_promoted=lti_promoted,
        new_connections=new_connections,
        contradictions_resolved=contradictions_resolved,
        orphans_healed=orphans_healed,
        bridges_created=bridges_created,
    )


def _discover_bridges(tenant_id: str, space: str, db, embed_engine) -> int:
    """Find other spaces in this tenant and materialize bridges where overlap is significant."""
    from smrti.core.atomspace import AtomSpace

    all_spaces_rows = db.fetchall(
        "SELECT DISTINCT space FROM atoms WHERE tenant_id = ? AND space != ?",
        (tenant_id, space),
    )
    if not all_spaces_rows:
        return 0

    atomspace = AtomSpace(db, embed_engine)
    total = 0

    for row in all_spaces_rows:
        other = row["space"]
        # Skip already-materialized bridge spaces to avoid recursion
        if "_x_" in other:
            continue
        overlap = space_overlap(tenant_id, space, other, db, threshold=0.85)
        total += materialize_bridge(
            overlap, tenant_id, db, embed_engine, atomspace, min_jaccard=0.1,
        )

    return total
