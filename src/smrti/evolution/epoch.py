"""Consolidation epoch: the main memory evolution loop."""
from __future__ import annotations

from smrti.core.db import fts_delete, vec_delete
from smrti.core.models import (
    PERMANENT_PROBABILITY,
    SUPERSEDED_PROBABILITY,
    EpochResult,
    TruthValue,
)
from smrti.core.provenance import (
    ATOM_FORGOTTEN,
    ATOM_METADATA_JSON,
    ATOM_OWN_INTENSITY,
    ATOM_OWN_VALENCE,
    ATOM_SOURCE,
)
from smrti.evolution.attention import propagate_sti
from smrti.evolution.connections import discover_connections
from smrti.evolution.healing import heal_orphaned_episodes
from smrti.evolution.truth import update_truth
from smrti.evolution.valence import propagate_valence

# Atoms are pruned below an LTI of 0.05, so a floor above that line makes a
# memory permanent. What the user stated keeps one; what the model volunteered
# does not, and decays to nothing unless the user later adopts it by bringing
# it up themselves. Without this asymmetry both terms of the prune predicate
# fall monotonically for every atom — confidence only rises on new evidence and
# LTI only on promotion — so an unmentioned fact is on a one-way trip to
# deletion, and core identity facts are exactly what goes unmentioned longest.
_LTI_FLOOR_USER = 0.1
_LTI_FLOOR_AGENT = 0.0

# Severe negative-valence atoms carry a higher floor from creation (see
# atomspace.add_atom) that keeps past failures out of reach of the pruner.
_LTI_FLOOR_CRITICAL = 0.5

# Permanence is a property of testimony, so it is withheld from the agent's own
# beliefs by passing a threshold no probability can reach. Otherwise a model
# asserting certainty about its own output could mint itself a memory that
# never fades — the one thing agent-source decay exists to prevent.
_PERMANENCE_UNREACHABLE = 2.0


def _param(personality: dict, key: str, default: float) -> float:
    """Read a personality hyperparameter, tolerating absent and NULL columns.

    ``dict.get(key, default)`` substitutes the default only when the key is
    missing; a column that exists but holds NULL yields None and poisons the
    arithmetic downstream. Both shapes occur on databases carried across
    versions, so both must fall back.
    """
    value = personality.get(key)
    return default if value is None else value


def run_epoch(tenant_id: str, space: str, db, embed_engine) -> EpochResult:
    """Single deterministic consolidation pass for a (tenant_id, space) pair.

    Executes in order:
      1. Apply pending evidence to update belief probabilities/confidence
      2. Decay STI, LTI, and confidence for all atoms in this space
         (user-stated episodes/beliefs hold a confidence floor at the
         surfacing threshold so they stay recallable, and one asserted as
         permanent keeps the confidence it was asserted with)
      3. Propagate STI and valence to 1-hop neighbors of active atoms
      4. Heal orphaned episodes (link to most salient person)
      5. Promote high-STI atoms to LTI
      6. Resolve contradictions: weaken the loser the edge names, else the
         less confident endpoint
      7. Discover associations between similar high-LTI atoms (every 10th
         epoch)
      8. Prune dead atoms below confidence and LTI floors

    Bridge spaces are no longer grown here. An earlier version compared the
    space against every other space in the tenant every tenth epoch and
    materialised ``a_x_b`` atoms wherever the overlap cleared a threshold
    nobody had set — a quadratic scan on a timer that wrote atoms into spaces
    nobody had asked for. Bridging is an explicit call now: ``space_merge``.
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
    lr = _param(p, "confidence_update_lr", 0.3)

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
        # The atom must live in the space being consolidated. Evidence filed
        # against an atom elsewhere — an overlay read recorded before the
        # resolver was scoped, or a row left behind by a since-moved atom —
        # would otherwise let this epoch rewrite another space's truth values.
        # Such rows are retired rather than skipped: leaving them pending makes
        # every future epoch re-scan them forever.
        atom_row = db.fetchone(
            "SELECT * FROM atoms WHERE id = ? AND tenant_id = ? AND space = ?",
            (atom_id, tenant_id, space),
        )
        if not atom_row:
            db.execute_many(
                "UPDATE evidence SET processed = 1 WHERE id = ?",
                [(ev["id"],) for ev in evs],
            )
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
                "UPDATE atoms SET probability = ?, confidence = ?, updated_at = datetime('now') "
                "WHERE id = ? AND tenant_id = ? AND space = ?",
                (current.probability, current.confidence, atom_id, tenant_id, space),
            ),
        ]
        statements += [
            ("UPDATE evidence SET processed = 1 WHERE id = ?", (ev["id"],))
            for ev in evs
        ]
        db.execute_batch(statements)
        beliefs_updated += len(evs)

    # 2. Decay STI, LTI, and confidence for all atoms in this space.
    # LTI must decay: promotion only ever raises it, so without a downward
    # force any atom that trends briefly salient is pinned above the prune
    # floor permanently and the graph can never shed anything.
    decay_count_row = db.fetchone(
        "SELECT COUNT(*) as n FROM atoms WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    atoms_decayed = decay_count_row["n"] if decay_count_row else 0
    # Agent-authored content decays faster in proportion to how little it is
    # trusted, so model output the user never picked up fades on its own while
    # user-stated facts persist. Atoms with no recorded source predate
    # provenance tracking and are treated as user-authored.
    agent_multiplier = 2.0 - _param(p, "agent_source_trust", 0.5)
    base_rates = (
        _param(p, "sti_decay_rate", 0.1),
        _param(p, "lti_decay_rate", 0.01),
        _param(p, "confidence_decay_rate", 0.02),
    )
    # LTI decays toward a floor rather than toward zero. The floor only holds
    # an atom that already reached it — one below keeps decaying freely — so
    # this preserves a memory that earned long-term importance without ever
    # granting it to one that did not.
    #
    # Confidence gates surfacing the way LTI gates pruning: recall filters
    # out atoms below min_confidence_to_surface, so once decay carries an
    # atom past that line it can never be recalled — and what cannot surface
    # cannot be restated, so no new evidence ever lifts it back. That is the
    # LTI floor's one-way trip again, aimed at visibility instead of
    # deletion. User-stated episodes and beliefs therefore decay toward the
    # surfacing line, not zero: direct testimony never stops being grounds
    # for a belief. Like the LTI floor, it only holds an atom still at or
    # above it — forget() sinks memories below on purpose, and a floor that
    # reached down would undo every deliberate forget one epoch later.
    # Concepts and goals are derived index nodes, not testimony, and keep
    # decaying freely, as does everything agent-authored.
    #
    # A belief asserted at PERMANENT_PROBABILITY is not merely exempt from
    # confidence decay: the epoch lifts it back to the confidence it was
    # asserted with. The floor keeps a memory reachable; it does not keep it
    # competitive, because it pins every aged atom to the same value while
    # anything stored in the last hour still carries several times that — and
    # confidence is the heaviest term in salience on most presets. A permanent
    # fact that recall can find but never ranks is one the caller experiences
    # as forgotten. Exemption alone is not enough either: it protects what the
    # current code writes, but a writer running pre-permanence code — a stale
    # process serving old code loaded before an upgrade landed on disk — mints
    # such beliefs drowned, and a one-time startup repair cannot lift damage
    # done after it has run. The lift is therefore part of the epoch itself,
    # so the damage heals whenever a current process next touches the space.
    # forget() still wins: it stamps the atoms it sinks, the stamp is what
    # tells a forget from decay drowning, and a stamped atom is never lifted.
    # An unstamped atom below the surfacing floor predates stamping and is
    # left where it is — nothing tells a legacy forget from a decay victim.
    #
    # The stamp also releases the LTI floors. A forgotten memory is one the
    # caller asked to be rid of; holding it at the testimony floor, or at the
    # critical-error floor, would keep it out of the pruner's reach forever.
    min_conf = _param(p, "min_confidence_to_surface", 0.1)
    decay_sql = f"""UPDATE atoms SET
               sti        = sti        * (1.0 - ?),
               lti        = MAX(lti * (1.0 - ?),
                                CASE WHEN {ATOM_FORGOTTEN} THEN 0.0
                                     WHEN {ATOM_OWN_VALENCE} < -0.7
                                          AND {ATOM_OWN_INTENSITY} > 0.7 THEN ?
                                     WHEN lti >= ? THEN ?
                                     ELSE 0.0 END),
               confidence = MAX(CASE WHEN type = 'belief' AND probability >= ?
                                     THEN CASE
                                         WHEN {ATOM_FORGOTTEN} THEN confidence
                                         WHEN confidence >= ?
                                             THEN MAX(confidence, probability)
                                         ELSE confidence END
                                     ELSE confidence * (1.0 - ?) END,
                                CASE WHEN type IN ('episode', 'belief')
                                          AND confidence >= ? THEN ?
                                     ELSE 0.0 END),
               updated_at = datetime('now')
           WHERE tenant_id = ? AND space = ?
             AND {ATOM_SOURCE} {{}} 'agent'"""
    for comparison, multiplier, floor, conf_floor, permanent_at in (
        ("!=", 1.0, _LTI_FLOOR_USER, min_conf, PERMANENT_PROBABILITY),
        ("=", agent_multiplier, _LTI_FLOOR_AGENT, 0.0, _PERMANENCE_UNREACHABLE),
    ):
        sti_rate, lti_rate, conf_rate = (
            min(rate * multiplier, 1.0) for rate in base_rates
        )
        db.execute(
            decay_sql.format(comparison),
            (sti_rate, lti_rate, _LTI_FLOOR_CRITICAL, floor, floor,
             permanent_at, conf_floor, conf_rate, conf_floor, conf_floor,
             tenant_id, space),
        )

    # 2b. Propagate STI and valence to 1-hop neighbors
    propagation_factor = _param(p, "sti_propagation_factor", 0.15)
    valence_prop_factor = _param(p, "valence_propagation", 0.1)
    mood_inertia = _param(p, "mood_inertia", 0.8)
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
        (tenant_id, space, _param(p, "lti_promotion_threshold", 0.7)),
    )
    lti_promoted = len(promoted_rows)
    if promoted_rows:
        # STI is clamped to 1.0 before scaling — it saturates at 3.0, so an
        # unclamped sti * 0.5 pins LTI to its ceiling on a single promotion,
        # which decay can then never walk back.
        db.execute(
            "UPDATE atoms SET lti = MIN(MAX(lti, MIN(sti, 1.0) * 0.5), 1.0) WHERE tenant_id = ? AND space = ? AND sti > ?",
            (tenant_id, space, _param(p, "lti_promotion_threshold", 0.7)),
        )

    # 4. Resolve contradictions within this space — each edge is weakened once,
    # then marked resolved atomically so it is never re-penalized.
    #
    # Who loses: the endpoint the edge names in ``$.loser`` when it names one,
    # else the less confident endpoint. The extractor names one when it
    # records a supersession — the user moved, changed jobs, changed their
    # mind — because there the older claim loses by definition, and it is
    # usually the *more* confident of the two, having had time to be mentioned
    # again; the confidence rule alone would weaken the update instead.
    contradictions = db.fetchall(
        """SELECT id, source_id, target_id,
                  CASE WHEN json_valid(metadata)
                       THEN json_extract(metadata, '$.loser') END AS loser
           FROM atoms
           WHERE type = 'relation' AND relation = 'contradicts'
             AND tenant_id = ? AND space = ?
             AND (CASE WHEN json_valid(metadata)
                       THEN json_extract(metadata, '$.resolved') END) IS NULL""",
        (tenant_id, space),
    )
    for c in contradictions:
        if not c["source_id"] or not c["target_id"]:
            continue
        # Both endpoints must be in this space. A contradiction edge can point
        # at an atom in another space (bridge edges do exactly that), and
        # weakening it here would let one space adjudicate another's beliefs.
        src = db.fetchone(
            "SELECT probability, confidence FROM atoms WHERE id = ? AND tenant_id = ? AND space = ?",
            (c["source_id"], tenant_id, space),
        )
        tgt = db.fetchone(
            "SELECT probability, confidence FROM atoms WHERE id = ? AND tenant_id = ? AND space = ?",
            (c["target_id"], tenant_id, space),
        )
        if src and tgt:
            # A named loser is a supersession: the fact stopped being true,
            # so its probability is cut to the superseded line as well as
            # its confidence. An unnamed contradiction only says the two
            # disagree, and there confidence alone gives way.
            named = c["loser"] in (c["source_id"], c["target_id"])
            if named:
                loser_id = c["loser"]
                weaken = (
                    "UPDATE atoms SET confidence = confidence * 0.8, "
                    "probability = MIN(probability, ?) "
                    "WHERE id = ? AND tenant_id = ? AND space = ?",
                    (SUPERSEDED_PROBABILITY, loser_id, tenant_id, space),
                )
            else:
                loser_id = (
                    c["source_id"]
                    if src["confidence"] < tgt["confidence"]
                    else c["target_id"]
                )
                weaken = (
                    "UPDATE atoms SET confidence = confidence * 0.8 "
                    "WHERE id = ? AND tenant_id = ? AND space = ?",
                    (loser_id, tenant_id, space),
                )
            db.execute_batch([
                weaken,
                (
                    f"UPDATE atoms SET metadata = json_set({ATOM_METADATA_JSON}, '$.resolved', 1) WHERE id = ?",
                    (c["id"],),
                ),
            ])
            contradictions_resolved += 1

    # 5. Association discovery between similar high-LTI atoms (every 10th epoch)
    if epoch_count % 10 == 0:
        new_connections = discover_connections(tenant_id, space, db, embed_engine)

    # 6. Prune atoms below both confidence and LTI floors.
    # User-authored episodes and beliefs are exempt from direct pruning unless
    # the caller forgot them; their agent-authored counterparts are not, so a
    # model turn the user never picked up can leave the graph once it has
    # decayed. Relations are never pruned directly — they cascade with their
    # endpoints (see loop below).
    dead_rows = db.fetchall(
        f"""SELECT id FROM atoms
           WHERE tenant_id = ? AND space = ?
             AND confidence < ? AND lti < 0.05
             AND type != 'relation'
             AND (type NOT IN ('episode', 'belief')
                  OR {ATOM_SOURCE} = 'agent'
                  OR {ATOM_FORGOTTEN})""",
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
            *vec_delete(atom_ids),
            *fts_delete(db, atom_ids),
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
    )
