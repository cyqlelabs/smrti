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

    # 1. Process pending evidence — each atom's truth update and its evidence
    # marks commit in one transaction so a crash cannot double-count evidence.
    pending = db.fetchall(
        "SELECT * FROM evidence WHERE processed = 0 AND tenant_id = ? AND space = ? ORDER BY created_at",
        (tenant_id, space),
    )
    pending_by_atom: dict[str, list] = {}
    for ev in pending:
        pending_by_atom.setdefault(ev["atom_id"], []).append(ev)
    for atom_id, evs in pending_by_atom.items():
        atom_row = db.fetchone(
            "SELECT * FROM atoms WHERE id = ?", (atom_id,)
        )
        if not atom_row:
            continue
        current = TruthValue(
            probability=atom_row["probability"],
            confidence=atom_row["confidence"],
        )
        for ev in evs:
            current = update_truth(
                current, ev["observed_probability"], ev["weight"], lr
            )
        statements = [
            (
                "UPDATE atoms SET probability = ?, confidence = ?, updated_at = datetime('now') WHERE id = ?",
                (current.probability, current.confidence, atom_id),
            ),
        ]
        statements += [
            ("UPDATE evidence SET processed = 1 WHERE id = ?", (ev["id"],))
            for ev in evs
        ]
        db.execute_batch(statements)
        beliefs_updated += len(evs)

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
            "SELECT id, sti, valence, intensity FROM atoms WHERE tenant_id = ? AND space = ? AND (sti > 0.3 OR (ABS(valence) > 0.3 AND intensity > 0.3))",
            (tenant_id, space),
        )
        for row in active:
            if propagation_factor > 0 and row["sti"] > 0.3:
                propagate_sti(row["id"], row["sti"], propagation_factor, db, tenant_id, space)
            if valence_prop_factor > 0 and abs(row["valence"]) > 0.3 and row["intensity"] > 0.3:
                propagate_valence(row["id"], row["valence"], row["intensity"], valence_prop_factor, db, tenant_id, space, mood_inertia=mood_inertia)

    # 2c. Heal orphaned episodes (link to most salient person)
    orphans_healed = heal_orphaned_episodes(tenant_id, space, db)

    # 3. Promote high-STI atoms to LTI (capped at 1.0)
    promoted_rows = db.fetchall(
        "SELECT id FROM atoms WHERE tenant_id = ? AND space = ? AND sti > ?",
        (tenant_id, space, p["lti_promotion_threshold"]),
    )
    lti_promoted = len(promoted_rows)
    if promoted_rows:
        db.execute(
            "UPDATE atoms SET lti = MIN(MAX(lti, sti * 0.5), 1.0) WHERE tenant_id = ? AND space = ? AND sti > ?",
            (tenant_id, space, p["lti_promotion_threshold"]),
        )

    # 4. Resolve contradictions within this space — each edge is weakened once,
    # then marked resolved atomically so it is never re-penalized.
    contradictions = db.fetchall(
        """SELECT id, source_id, target_id FROM atoms
           WHERE type = 'relation' AND relation = 'contradicts'
             AND tenant_id = ? AND space = ?
             AND json_extract(metadata, '$.resolved') IS NULL""",
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
            db.execute_batch([
                (
                    "UPDATE atoms SET confidence = confidence * 0.8 WHERE id = ?",
                    (loser_id,),
                ),
                (
                    "UPDATE atoms SET metadata = json_set(COALESCE(metadata, '{}'), '$.resolved', 1) WHERE id = ?",
                    (c["id"],),
                ),
            ])
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

    if dead_rows:
        atom_ids = [row["id"] for row in dead_rows]
        ph = ",".join("?" * len(atom_ids))
        # Relation cascade stays tenant-scoped but cross-space so bridge edges
        # referencing pruned atoms are cleaned too; all deletes commit together
        # so a crash cannot leave atoms invisible to KNN.
        db.execute_batch([
            (
                f"DELETE FROM atoms WHERE type = 'relation' AND tenant_id = ? AND (source_id IN ({ph}) OR target_id IN ({ph}))",
                (tenant_id, *atom_ids, *atom_ids),
            ),
            (f"DELETE FROM vec_atoms WHERE atom_id IN ({ph})", tuple(atom_ids)),
            (f"DELETE FROM evidence WHERE atom_id IN ({ph})", tuple(atom_ids)),
            (f"DELETE FROM aliases WHERE atom_id IN ({ph})", tuple(atom_ids)),
            (f"DELETE FROM atoms WHERE id IN ({ph})", tuple(atom_ids)),
        ])

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

    # Bridge spaces must not initiate bridging (would recurse into meta-bridges)
    if "_x_" in space:
        return 0

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
        overlap = space_overlap(tenant_id, space, other, db, threshold=0.85, embed_engine=embed_engine)
        total += materialize_bridge(
            overlap, tenant_id, db, embed_engine, atomspace, min_jaccard=0.1,
        )

    return total
