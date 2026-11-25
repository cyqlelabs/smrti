"""Consolidation epoch: the main memory evolution loop."""
from __future__ import annotations

import logging
import time
import uuid

from smrti.core.atomspace import VECTOR_MISSING, AtomSpace, embedding_text
from smrti.core.models import (
    PERMANENT_PROBABILITY,
    STRUCTURAL_RELATIONS,
    SUPERSEDED_PROBABILITY,
    EpochResult,
    TruthValue,
    atom_from_row,
)
from smrti.core.provenance import (
    ATOM_FORGOTTEN,
    ATOM_METADATA_JSON,
    ATOM_SOURCE,
    SEVERE_RETAINED_SQL,
    SOURCE_AGENT,
    SOURCE_USER,
    claim_current_sql,
    forgotten_sql,
)
from smrti.evolution.attention import sti_moves
from smrti.evolution.connections import discover_connections
from smrti.evolution.healing import heal_orphaned_episodes
from smrti.evolution.truth import update_truth
from smrti.evolution.valence import NUDGE_SQL, nudge_params
from smrti.personality.params import personality_params

logger = logging.getLogger("smrti.epoch")

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

# A decayed attention or confidence value under this is stored as zero.
# Multiplicative decay never reaches zero by itself, so without the cut every
# atom that ever held attention was rewritten on every epoch for thousands of
# epochs, each write changing a number too small to move any ranking.
_NEGLIGIBLE = 1e-6

# How long a reflector may go without renewing its hold on a space before
# another process may take the space over. The epoch renews it between
# phases, so this bounds one phase, not the whole epoch.
_LEASE_SECONDS = 300.0

# The relations that record how the graph is built rather than a fact about
# the world, as an SQL list.
_STRUCTURAL = ", ".join(f"'{relation}'" for relation in STRUCTURAL_RELATIONS)

_ID_CHUNK = 400


class LeaseLost(RuntimeError):
    """Another process took over the space while this epoch was running."""


def _chunks(ids: list[str]):
    for start in range(0, len(ids), _ID_CHUNK):
        yield ids[start : start + _ID_CHUNK]


def _take_lease(tenant_id: str, space: str, db) -> tuple[str, int] | None:
    """Claim the space for one epoch and count it, or None when another holds it.

    One reflector per space at a time, across processes: every step of an
    epoch moves values by a fraction of what they are, so two epochs that
    interleave on one space decay it twice and propagate twice. A lease
    whose holder died stops blocking once it expires. The epoch is counted
    in the same transaction, so a refused run leaves no trace.
    """
    owner = uuid.uuid4().hex
    now = time.time()
    with db.transaction() as conn:
        held = conn.execute(
            "SELECT owner, expires_at FROM epoch_lease WHERE tenant_id = ? AND space = ?",
            (tenant_id, space),
        ).fetchone()
        if held is not None and held["owner"] is not None and held["expires_at"] > now:
            return None
        conn.execute(
            """INSERT INTO epoch_lease (tenant_id, space, owner, expires_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (tenant_id, space) DO UPDATE
               SET owner = excluded.owner, expires_at = excluded.expires_at""",
            (tenant_id, space, owner, now + _LEASE_SECONDS),
        )
        conn.execute(
            "UPDATE personality SET epoch_count = epoch_count + 1 WHERE tenant_id = ? AND space = ?",
            (tenant_id, space),
        )
        row = conn.execute(
            "SELECT epoch_count FROM personality WHERE tenant_id = ? AND space = ?",
            (tenant_id, space),
        ).fetchone()
    return owner, (row["epoch_count"] if row else 1)


def _hold_lease(tenant_id: str, space: str, db, owner: str) -> None:
    """Extend the lease, or stop the epoch if another process has taken it.

    Called between phases, so a holder that outlived its lease stops before
    its next write rather than decaying the space a second time alongside
    the process that took over.
    """
    cursor = db.execute(
        "UPDATE epoch_lease SET expires_at = ? WHERE tenant_id = ? AND space = ? AND owner = ?",
        (time.time() + _LEASE_SECONDS, tenant_id, space, owner),
    )
    if cursor.rowcount != 1:
        raise LeaseLost(f"epoch lease on tenant={tenant_id} space={space} was taken over")


def _release_lease(
    tenant_id: str, space: str, db, owner: str, completed_epoch: int | None
) -> None:
    """Give the space back, recording the epoch when it ran to completion.

    The recorded number is what tells a duplicate run from the next one: the
    personality row counts every epoch that started, this one only those
    that finished.
    """
    if completed_epoch is None:
        db.execute(
            "UPDATE epoch_lease SET owner = NULL, expires_at = 0 "
            "WHERE tenant_id = ? AND space = ? AND owner = ?",
            (tenant_id, space, owner),
        )
        return
    db.execute(
        """UPDATE epoch_lease SET owner = NULL, expires_at = 0,
                  completed_epoch = ?, completed_at = datetime('now')
           WHERE tenant_id = ? AND space = ? AND owner = ?""",
        (completed_epoch, tenant_id, space, owner),
    )


def empty_result() -> EpochResult:
    """What an epoch that changed nothing reports."""
    return EpochResult(
        beliefs_updated=0,
        atoms_decayed=0,
        atoms_pruned=0,
        lti_promoted=0,
        new_connections=0,
        contradictions_resolved=0,
    )


def _param(personality: dict, key: str, default: float) -> float:
    """Read a personality hyperparameter, tolerating absent and NULL columns.

    ``dict.get(key, default)`` substitutes the default only when the key is
    missing; a column that exists but holds NULL yields None and poisons the
    arithmetic downstream. Both shapes occur on databases carried across
    versions, so both must fall back.
    """
    value = personality.get(key)
    return default if value is None else value


def _repair_vectors(tenant_id: str, space: str, db, embed_engine) -> None:
    """File the vectors of atoms stored without one.

    A failure here is logged and ends the repair, never the epoch: decay has
    already been applied by the time it runs, and an epoch that raised after
    it left the space marked as used, so an idle space aged again on every
    interval for as long as the encoder stayed down. The atoms keep their
    mark and are retried next epoch.
    """
    rows = db.fetchall(
        f"""SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'
            AND NOT {ATOM_FORGOTTEN}
            AND (CASE WHEN json_valid(metadata)
                      THEN json_extract(metadata, '$.{VECTOR_MISSING}') END) IS NOT NULL""",
        (tenant_id, space),
    )
    atomspace = AtomSpace(db, embed_engine)
    for row in rows:
        atom = atom_from_row(row)
        try:
            atomspace.file_vector(atom, list(embed_engine.embed(embedding_text(atom))))
        except Exception:
            logger.warning(
                "vector repair stopped at atom %s in tenant=%s space=%s; "
                "the rest stay marked for the next epoch",
                atom.id, tenant_id, space, exc_info=True,
            )
            return


def _apply_evidence(tenant_id: str, space: str, db, lr: float) -> int:
    """Revise truth values with the space's pending evidence.

    The rows are read, the atoms they bear on are read and revised, and the
    rows are marked processed, all in one write transaction: no other
    process can read them as pending in between, so none can apply them a
    second time. Evidence against an atom outside the space, or against one
    forget() sank, is retired unapplied — leaving it pending would make every
    later epoch re-scan it, and applying it to a forgotten atom lifted it
    back over the surfacing floor, where the floor then held it forever.

    Returns how many rows bore on an atom here.
    """
    # Most epochs find nothing pending, and a plain read says so without
    # taking the write lock.
    if not db.fetchall(
        "SELECT 1 FROM evidence WHERE processed = 0 AND tenant_id = ? AND space = ? LIMIT 1",
        (tenant_id, space),
    ):
        return 0
    applied = 0
    with db.transaction() as conn:
        pending = conn.execute(
            """SELECT id, atom_id, observed_probability, weight FROM evidence
               WHERE processed = 0 AND tenant_id = ? AND space = ? ORDER BY created_at""",
            (tenant_id, space),
        ).fetchall()
        by_atom: dict[str, list] = {}
        for ev in pending:
            by_atom.setdefault(ev["atom_id"], []).append(ev)
        atoms = {}
        for chunk in _chunks(list(by_atom)):
            ph = ",".join("?" * len(chunk))
            for row in conn.execute(
                f"""SELECT id, probability, confidence FROM atoms
                    WHERE id IN ({ph}) AND tenant_id = ? AND space = ? AND NOT {ATOM_FORGOTTEN}""",
                (*chunk, tenant_id, space),
            ):
                atoms[row["id"]] = row
        revised = []
        for atom_id, evs in by_atom.items():
            row = atoms.get(atom_id)
            if row is None:
                continue
            current = TruthValue(probability=row["probability"], confidence=row["confidence"])
            for ev in evs:
                current = update_truth(current, ev["observed_probability"], ev["weight"], lr)
            revised.append((current.probability, current.confidence, atom_id, tenant_id, space))
            applied += len(evs)
        conn.executemany(
            "UPDATE atoms SET probability = ?, confidence = ?, updated_at = datetime('now') "
            "WHERE id = ? AND tenant_id = ? AND space = ?",
            revised,
        )
        conn.executemany(
            "UPDATE evidence SET processed = 1 WHERE id = ?",
            [(ev["id"],) for ev in pending],
        )
    return applied


def _decay(tenant_id: str, space: str, db, p: dict) -> None:
    """Decay STI, LTI and confidence, writing only the rows whose values change.

    Each new value is computed by one expression, and the same expression,
    compared with the stored value, decides whether the row is written at
    all. Rewriting every row every epoch made the epoch's cost, and the
    write-ahead log it grew, proportional to the whole graph however little
    of it was still moving.
    """
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
    # deletion. User-stated episodes, beliefs and goals therefore decay toward the
    # surfacing line, not zero: direct testimony never stops being grounds
    # for a belief. Like the LTI floor, it only holds an atom still at or
    # above it — forget() sinks memories below on purpose, and a floor that
    # reached down would undo every deliberate forget one epoch later.
    # A goal the user stated is testimony too — "I want to learn Rust" is a
    # report about the user, and a goal that silently stops surfacing (and
    # then, with LTI at zero, is pruned around epoch 80 on the default rates)
    # is one the agent stops working toward without anyone deciding so.
    # Concepts are derived index nodes, not testimony, and keep decaying
    # freely, as does everything agent-authored.
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
    # The stamp also releases every floor. A forgotten memory is one the
    # caller asked to be rid of; holding it at the testimony floor, or at the
    # critical-error floor, would keep it out of the pruner's reach forever.
    def decayed(column: str, rate: str) -> str:
        return (
            f"(CASE WHEN {column} * (1.0 - :{rate}) < :negligible THEN 0.0 "
            f"ELSE {column} * (1.0 - :{rate}) END)"
        )

    new_sti = decayed("sti", "sti_rate")
    new_lti = f"""MAX({decayed("lti", "lti_rate")},
                      CASE WHEN {ATOM_FORGOTTEN} THEN 0.0
                           WHEN {SEVERE_RETAINED_SQL} THEN :critical_floor
                           WHEN lti >= :lti_floor THEN :lti_floor
                           ELSE 0.0 END)"""
    new_confidence = f"""MAX(CASE WHEN type = 'belief' AND probability >= :permanent_at
                                  THEN CASE
                                      WHEN {ATOM_FORGOTTEN} THEN confidence
                                      WHEN confidence >= :conf_floor
                                          THEN MAX(confidence, probability)
                                      ELSE confidence END
                                  ELSE {decayed("confidence", "conf_rate")} END,
                             CASE WHEN type IN ('episode', 'belief', 'goal')
                                       AND NOT {ATOM_FORGOTTEN}
                                       AND confidence >= :conf_floor THEN :conf_floor
                                  ELSE 0.0 END)"""
    min_conf = _param(p, "min_confidence_to_surface", 0.1)
    for comparison, multiplier, lti_floor, conf_floor, permanent_at in (
        ("!=", 1.0, _LTI_FLOOR_USER, min_conf, PERMANENT_PROBABILITY),
        ("=", agent_multiplier, _LTI_FLOOR_AGENT, 0.0, _PERMANENCE_UNREACHABLE),
    ):
        sti_rate, lti_rate, conf_rate = (
            min(rate * multiplier, 1.0) for rate in base_rates
        )
        db.execute(
            f"""UPDATE atoms SET
                    sti = {new_sti}, lti = {new_lti}, confidence = {new_confidence},
                    updated_at = datetime('now')
                WHERE tenant_id = :tenant_id AND space = :space
                  AND {ATOM_SOURCE} {comparison} 'agent'
                  AND (sti != {new_sti} OR lti != {new_lti}
                       OR confidence != {new_confidence})""",
            {
                "tenant_id": tenant_id, "space": space,
                "sti_rate": sti_rate, "lti_rate": lti_rate, "conf_rate": conf_rate,
                "negligible": _NEGLIGIBLE, "critical_floor": _LTI_FLOOR_CRITICAL,
                "lti_floor": lti_floor, "conf_floor": conf_floor,
                "permanent_at": permanent_at,
            },
        )


def _far_ends(
    tenant_id: str, space: str, db, ids: list[str], near: str, far: str
) -> dict[str, list[str]]:
    """For each of *ids*, the *far* end of every edge in this space whose
    *near* end it is, in the order the edges were written."""
    ends: dict[str, list[str]] = {}
    for chunk in _chunks(ids):
        ph = ",".join("?" * len(chunk))
        for edge in db.fetchall(
            f"""SELECT {near} AS near, {far} AS far FROM atoms
                WHERE {near} IN ({ph}) AND type = 'relation' AND tenant_id = ? AND space = ?
                ORDER BY rowid""",
            (*chunk, tenant_id, space),
        ):
            if edge["far"]:
                ends.setdefault(edge["near"], []).append(edge["far"])
    return ends


def _neighbours(tenant_id: str, space: str, db, active) -> dict[str, list[str]]:
    """Each active atom's 1-hop neighbours, read a chunk of atoms at a time.

    A relation atom's neighbours are its endpoints, which it stores in its
    own columns; any other atom's are the far ends of the edges in this
    space that touch it, outgoing first. Looking these up two queries per
    atom was what an epoch spent nearly all its time on — 17,000 lookups and
    three to four minutes at twenty thousand atoms.
    """
    found = {
        row["id"]: [x for x in (row["source_id"], row["target_id"]) if x]
        for row in active
        if row["type"] == "relation"
    }
    plain = [row["id"] for row in active if row["type"] != "relation"]
    outgoing = _far_ends(tenant_id, space, db, plain, "source_id", "target_id")
    incoming = _far_ends(tenant_id, space, db, plain, "target_id", "source_id")
    for atom_id in plain:
        found[atom_id] = outgoing.get(atom_id, []) + incoming.get(atom_id, [])
    return found


def _creditable(tenant_id: str, space: str, db, ids: list[str]) -> set[str]:
    """Which of *ids* this space can credit: atoms in it that were not forgotten.

    A bridge edge's far end sits in another space, where an update is a
    no-op, and charging the source for a share nobody received would
    destroy activation. A forgotten atom receives nothing either: forget()
    zeroed its attention, and propagation handing it back would undo that.
    """
    found: set[str] = set()
    for chunk in _chunks(ids):
        ph = ",".join("?" * len(chunk))
        found.update(
            r["id"]
            for r in db.fetchall(
                f"""SELECT id FROM atoms
                    WHERE id IN ({ph}) AND tenant_id = ? AND space = ? AND NOT {ATOM_FORGOTTEN}""",
                (*chunk, tenant_id, space),
            )
        )
    return found


def _propagation_statements(
    tenant_id: str, space: str, active, neighbours, creditable, p: dict
) -> list[tuple]:
    """Each active atom's moves and nudges, source by source, as statements."""
    sti_factor = _param(p, "sti_propagation_factor", 0.15)
    valence_factor = _param(p, "valence_propagation", 0.1)
    mood_inertia = _param(p, "mood_inertia", 0.8)
    statements: list[tuple] = []
    for row in active:
        ids = [n for n in neighbours.get(row["id"], []) if n in creditable]
        if sti_factor > 0 and row["sti"] > 0.3:
            statements += sti_moves(row["id"], row["sti"], sti_factor, ids, tenant_id, space)
        if valence_factor > 0 and abs(row["valence"]) > 0.3 and row["intensity"] > 0.3:
            nudges = nudge_params(
                row["valence"], row["intensity"], valence_factor, ids,
                tenant_id, space, mood_inertia,
            )
            statements += [(NUDGE_SQL, params) for params in nudges]
    return statements


def _propagate(tenant_id: str, space: str, db, p: dict) -> None:
    """Spread STI and valence one hop from every active atom, in one transaction.

    The statements are the ones the per-atom propagation issues, in the same
    order — each source's moves read from one snapshot of the active atoms,
    the blends applied in turn — so the values land exactly where they would
    one atom at a time, and commit once. Forgotten atoms neither spread nor
    receive.
    """
    if (
        _param(p, "sti_propagation_factor", 0.15) <= 0
        and _param(p, "valence_propagation", 0.1) <= 0
    ):
        return
    active = db.fetchall(
        f"""SELECT id, type, source_id, target_id, sti, valence, intensity FROM atoms
            WHERE tenant_id = ? AND space = ? AND NOT {ATOM_FORGOTTEN}
              AND (sti > 0.3 OR (ABS(valence) > 0.3 AND intensity > 0.3))""",
        (tenant_id, space),
    )
    if not active:
        return
    neighbours = _neighbours(tenant_id, space, db, active)
    creditable = _creditable(
        tenant_id, space, db, list(dict.fromkeys(n for ids in neighbours.values() for n in ids))
    )
    statements = _propagation_statements(tenant_id, space, active, neighbours, creditable, p)
    if statements:
        db.execute_batch(statements)


# Whether a current claim the user made, not forgotten, names the atom in the
# outer query as an endpoint. Two EXISTS rather than one with an OR, so each
# is answered from its own index.
def _user_claim_names(endpoint: str) -> str:
    return f"""EXISTS (
        SELECT 1 FROM atoms claim
        WHERE claim.{endpoint} = atoms.id AND claim.type = 'relation'
          AND claim.tenant_id = atoms.tenant_id
          AND claim.relation NOT IN ({_STRUCTURAL})
          AND {claim_current_sql('claim')}
          AND NOT {forgotten_sql('claim')}
          AND COALESCE(CASE WHEN json_valid(claim.metadata)
                            THEN json_extract(claim.metadata, '$.source') END,
                       '{SOURCE_USER}') != '{SOURCE_AGENT}')"""


_USER_CLAIM_ENDPOINT = f"({_user_claim_names('source_id')} OR {_user_claim_names('target_id')})"


def run_epoch(tenant_id: str, space: str, db, embed_engine) -> EpochResult:
    """One consolidation pass for a (tenant_id, space) pair.

    Returns an empty result, having changed nothing, when another process is
    consolidating the space (see :func:`try_epoch`).
    """
    result = try_epoch(tenant_id, space, db, embed_engine)
    return empty_result() if result is None else result


def try_epoch(tenant_id: str, space: str, db, embed_engine) -> EpochResult | None:
    """Run one epoch under the space's lease, or return None when another holds it.

    The lease is released however the epoch ends, and the epoch is recorded
    as completed only when it ran to the end.
    """
    lease = _take_lease(tenant_id, space, db)
    if lease is None:
        return None
    owner, epoch_count = lease
    completed = None
    try:
        result = _consolidate(
            tenant_id, space, db, embed_engine, epoch_count,
            hold=lambda: _hold_lease(tenant_id, space, db, owner),
        )
        completed = epoch_count
        return result
    finally:
        _release_lease(tenant_id, space, db, owner, completed)


def _consolidate(
    tenant_id: str, space: str, db, embed_engine, epoch_count: int, hold
) -> EpochResult:
    """Single deterministic consolidation pass for a (tenant_id, space) pair.

    Executes in order:
      1. Apply pending evidence to update belief probabilities/confidence
      2. Decay STI, LTI, and confidence for all atoms in this space
         (user-stated episodes/beliefs/goals hold a confidence floor at the
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
      9. File the vectors of atoms stored without one

    Bridge spaces are no longer grown here. An earlier version compared the
    space against every other space in the tenant every tenth epoch and
    materialised ``a_x_b`` atoms wherever the overlap cleared a threshold
    nobody had set — a quadratic scan on a timer that wrote atoms into spaces
    nobody had asked for. Bridging is an explicit call now: ``space_merge``.
    """
    atoms_pruned = 0
    lti_promoted = 0
    new_connections = 0
    contradictions_resolved = 0

    personality = db.fetchone(
        "SELECT * FROM personality WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    if not personality:
        return empty_result()

    p = personality_params(personality)

    # 1. Process pending evidence.
    beliefs_updated = _apply_evidence(
        tenant_id, space, db, _param(p, "confidence_update_lr", 0.3)
    )

    # 2. Decay STI, LTI, and confidence for all atoms in this space.
    # LTI must decay: promotion only ever raises it, so without a downward
    # force any atom that trends briefly salient is pinned above the prune
    # floor permanently and the graph can never shed anything.
    hold()
    decay_count_row = db.fetchone(
        "SELECT COUNT(*) as n FROM atoms WHERE tenant_id = ? AND space = ?",
        (tenant_id, space),
    )
    atoms_decayed = decay_count_row["n"] if decay_count_row else 0
    _decay(tenant_id, space, db, p)

    # 2b. Propagate STI and valence to 1-hop neighbors
    hold()
    _propagate(tenant_id, space, db, p)

    # 2c. Heal orphaned episodes (link to most salient person)
    hold()
    orphans_healed = heal_orphaned_episodes(tenant_id, space, db)

    # 3. Promote high-STI atoms to LTI (capped at 1.0). A forgotten atom is
    # never promoted: its attention was zeroed, and anything that raised it
    # since must not buy it a long-term floor.
    hold()
    promoted_rows = db.fetchall(
        f"SELECT id FROM atoms WHERE tenant_id = ? AND space = ? AND sti > ? AND NOT {ATOM_FORGOTTEN}",
        (tenant_id, space, _param(p, "lti_promotion_threshold", 0.7)),
    )
    lti_promoted = len(promoted_rows)
    if promoted_rows:
        # STI is clamped to 1.0 before scaling — it saturates at 3.0, so an
        # unclamped sti * 0.5 pins LTI to its ceiling on a single promotion,
        # which decay can then never walk back.
        db.execute(
            f"UPDATE atoms SET lti = MIN(MAX(lti, MIN(sti, 1.0) * 0.5), 1.0) "
            f"WHERE tenant_id = ? AND space = ? AND sti > ? AND NOT {ATOM_FORGOTTEN}",
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
    hold()
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

    # 5. Association discovery between similar high-LTI atoms (every 10th
    # epoch), and on the same cadence a refresh of the statistics the query
    # planner chooses indexes by — without them it can pick the partition
    # index over the edge indexes and scan the whole space per lookup.
    if epoch_count % 10 == 0:
        hold()
        new_connections = discover_connections(tenant_id, space, db, embed_engine)
        db.optimize()

    # 6. Prune atoms below both confidence and LTI floors.
    # User-authored episodes, beliefs and goals are exempt from direct pruning unless
    # the caller forgot them; their agent-authored counterparts are not, so a
    # model turn the user never picked up can leave the graph once it has
    # decayed. Relations are never pruned directly — they cascade with their
    # endpoints (see loop below).
    #
    # Nor is an endpoint of a current claim the user made. Pruning a concept
    # takes every edge on it, and an extracted entity nobody mentions again —
    # the city the user lives in — fell under both floors around epoch 80 and
    # took the user's own lives_in claim with it, though nothing had replaced
    # it. A superseded claim is history and protects nothing, and neither does
    # a claim read from an agent turn: model output the user never adopted
    # must still be able to leave the graph.
    hold()
    min_conf = _param(p, "min_confidence_to_surface", 0.1)
    dead_rows = db.fetchall(
        f"""SELECT id FROM atoms
           WHERE tenant_id = ? AND space = ?
             AND confidence < ? AND lti < 0.05
             AND type != 'relation'
             AND ({ATOM_FORGOTTEN}
                  OR ((type NOT IN ('episode', 'belief', 'goal')
                       OR {ATOM_SOURCE} = 'agent')
                      AND NOT {_USER_CLAIM_ENDPOINT}))""",
        (tenant_id, space, min_conf),
    )
    atoms_pruned = len(dead_rows)

    if dead_rows:
        # The cascade is tenant-scoped but cross-space so bridge edges
        # referencing pruned atoms are cleaned too, and it commits as one so
        # a crash cannot leave atoms invisible to KNN.
        AtomSpace(db, embed_engine).delete_atoms(
            [row["id"] for row in dead_rows], tenant_id
        )

    # 7. File the vectors of atoms stored without one. A write whose encoder
    # failed keeps the atom and marks it; until the mark is cleared the atom
    # is reachable by word alone and embedded afresh on every recall that
    # finds it.
    hold()
    _repair_vectors(tenant_id, space, db, embed_engine)

    return EpochResult(
        beliefs_updated=beliefs_updated,
        atoms_decayed=atoms_decayed,
        atoms_pruned=atoms_pruned,
        lti_promoted=lti_promoted,
        new_connections=new_connections,
        contradictions_resolved=contradictions_resolved,
        orphans_healed=orphans_healed,
    )
