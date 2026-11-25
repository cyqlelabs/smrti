"""Bridge space materialization: when two spaces overlap, grow a new subspace.

A bridge space contains atoms that represent the conceptual intersection of
two parent spaces.  Each bridge atom merges truth values (via PLN), averages
attention weights, and blends valence from its two source atoms.  Relation
edges connect the bridge atom back to both parents so graph traversal can
flow across space boundaries.

Bridge spaces are named ``{sorted_a}_x_{sorted_b}`` so the operation is
commutative (A∩B == B∩A always produces the same space name), and the two
parents of each pair are merged in the same sorted order, so the bridge
atom is the same whichever parent asked for it.
"""
from __future__ import annotations

import json

from smrti.core.atomspace import AtomSpace
from smrti.core.models import (
    Atom,
    AtomPair,
    AtomType,
    AttentionValue,
    SpaceOverlap,
    TruthValue,
    Valence,
)
from smrti.core.provenance import SOURCE_AGENT, SOURCE_USER, VALENCE_STATED


def _blend(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Two (valence, intensity) pairs weighted by intensity: the stronger
    signal wins, and the result is as intense as the more intense one."""
    total = a[1] + b[1]
    if total <= 1e-9:
        return 0.0, 0.0
    valence = (a[0] * a[1] + b[0] * b[1]) / total
    return max(-1.0, min(1.0, valence)), min(1.0, max(a[1], b[1]))


def _merge_pair(pair: AtomPair, bridge_space: str, tenant_id: str) -> Atom:
    """Create a single bridge atom from a matched pair."""
    a, b = sorted((pair.atom_a, pair.atom_b), key=lambda atom: (atom.space, atom.id))

    # PLN merge for truth value
    merged_truth = a.truth.merge(b.truth)

    # Average attention (bridge starts with combined salience signal)
    merged_attention = AttentionValue(
        sti=(a.attention.sti + b.attention.sti) / 2.0,
        lti=max(a.attention.lti, b.attention.lti),
    )

    # The mood and the tone each blend from their own kind. Blending the
    # parents' mood into the bridge's own tone would hand it a tone neither
    # parent was written with, and a pruning floor with it.
    mood, mood_intensity = _blend(
        (a.valence.valence, a.valence.intensity), (b.valence.valence, b.valence.intensity)
    )
    own, own_intensity = _blend(
        (a.valence.own, a.valence.own_intensity), (b.valence.own, b.valence.own_intensity)
    )

    # Use the higher-confidence atom's label
    label = a.label if a.truth.confidence >= b.truth.confidence else b.label
    content = a.content if a.truth.confidence >= b.truth.confidence else b.content

    # Preserve entity_type from the higher-confidence source
    entity_type = a.entity_type if a.truth.confidence >= b.truth.confidence else b.entity_type

    # Merge metadata from both sources — except what a bridge has to earn
    # itself. A bridge is the engine's own inference that two memories match,
    # so no caller stated its valence: inheriting the flag from either parent
    # turned two ordinary memories into a critical warning. And it is
    # agent-authored only when both parents are, since user testimony on
    # either side is what it rests on.
    metadata = {**a.metadata, **b.metadata}
    metadata.pop(VALENCE_STATED, None)
    if all(p.metadata.get("source", SOURCE_USER) == SOURCE_AGENT for p in (a, b)):
        metadata["source"] = SOURCE_AGENT
    else:
        metadata.pop("source", None)
    metadata["bridge_source_a"] = a.id
    metadata["bridge_source_b"] = b.id
    metadata["bridge_similarity"] = pair.similarity

    return Atom(
        type=a.type if a.type != AtomType.RELATION else AtomType.CONCEPT,
        label=label,
        content=content,
        truth=merged_truth,
        attention=merged_attention,
        valence=Valence(
            valence=mood, intensity=mood_intensity,
            intrinsic_valence=own, intrinsic_intensity=own_intensity,
        ),
        entity_type=entity_type,
        tenant_id=tenant_id,
        space=bridge_space,
        metadata=metadata,
    )


class _Deferred:
    """A database handle that collects writes instead of running them.

    The bridge atom and its two edges each go through ``AtomSpace``, the one
    door every atom comes through, and through this handle the three land in
    one transaction. Written as three, an edge that failed left a bridge atom
    joined to neither parent. Reads go to the real database, which is what
    the writes are built against.
    """

    def __init__(self, db) -> None:
        self._db = db
        self.statements: list[tuple] = []

    def __getattr__(self, name: str):
        return getattr(self._db, name)

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.statements.append((sql, params))

    def execute_batch(self, statements: list[tuple]) -> None:
        self.statements.extend(statements)


def _refuse_foreign_bridge(
    bridge_space: str, sources: set[str], overlap: SpaceOverlap, tenant_id: str, db
) -> None:
    """Refuse to write into a bridge space that bridges another pair.

    The name is the two parent names joined by ``_x_``, so ``a_x_b`` with
    ``c`` and ``a`` with ``b_x_c`` both name ``a_x_b_x_c``; bridging the
    second into it would mix two unrelated intersections in one space.
    """
    if not sources:
        return
    rows = db.fetchall(
        "SELECT DISTINCT space FROM atoms WHERE tenant_id = ? AND id IN (SELECT value FROM json_each(?))",
        (tenant_id, json.dumps(sorted(sources))),
    )
    parents = {r["space"] for r in rows}
    if parents - {overlap.space_a, overlap.space_b}:
        raise ValueError(
            f"bridge space {bridge_space!r} already bridges {' and '.join(sorted(parents))}, "
            f"not {overlap.space_a!r} and {overlap.space_b!r}: the two pairs share a bridge name"
        )


def _existing_bridges(
    db, tenant_id: str, bridge_space: str
) -> tuple[dict[tuple[str, str], str], set[tuple[str, str]]]:
    """The bridge atoms already in *bridge_space*, by source pair, and the
    pairs whose bridge atom was forgotten.

    Keys are order-normalized so discovery from either parent space finds
    the same stored bridge atom. A forgotten one is a tombstone: its pair is
    neither refreshed nor bridged again.
    """
    rows = db.fetchall(
        "SELECT id, metadata FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        (tenant_id, bridge_space),
    )
    bridged: dict[tuple[str, str], str] = {}
    tombstones: set[tuple[str, str]] = set()
    for row in rows:
        meta = json.loads(row["metadata"] or "{}")
        src_a = meta.get("bridge_source_a")
        src_b = meta.get("bridge_source_b")
        if not (src_a and src_b):
            continue
        key = tuple(sorted((src_a, src_b)))
        if meta.get("forgotten"):
            tombstones.add(key)
        else:
            bridged[key] = row["id"]
    return bridged, tombstones


def _write_bridge(
    pair: AtomPair, bridge_id: str | None, bridge_space: str, tenant_id: str, db, embed_engine, atomspace
) -> None:
    """Write one pair's bridge atom — new, or refreshed in place when
    *bridge_id* names the existing one — and whichever of its two edges it
    lacks, in one transaction."""
    bridge_atom = _merge_pair(pair, bridge_space, tenant_id)
    deferred = _Deferred(db)
    writer = AtomSpace(deferred, embed_engine)
    linked: set[str] = set()
    if bridge_id is None:
        writer.add_atom(bridge_atom)
    else:
        bridge_atom.id = bridge_id
        writer.update_atom(bridge_atom)
        linked = {
            edge.target_id
            for edge in atomspace.get_relations(bridge_id, tenant_id, [bridge_space])
            if edge.relation == "bridge" and edge.source_id == bridge_id
        }
    for source in (pair.atom_a, pair.atom_b):
        if source.id not in linked:
            writer.link_atoms(
                source_id=bridge_atom.id,
                target_id=source.id,
                relation="bridge",
                tenant_id=tenant_id,
                space=bridge_space,
                truth=TruthValue(probability=pair.similarity, confidence=0.8),
            )
    db.execute_batch(deferred.statements)


def materialize_bridge(
    overlap: SpaceOverlap,
    tenant_id: str,
    db,
    embed_engine,
    atomspace,
    min_jaccard: float = 0.1,
) -> int:
    """Materialize a bridge space from a SpaceOverlap result.

    Creates bridge atoms and ``bridge`` relation edges connecting them back to
    their source atoms in both parent spaces.  Existing bridge atoms (identified
    by ``bridge_source_a`` / ``bridge_source_b`` metadata) are updated rather
    than duplicated, and an edge missing from one is drawn again. A bridge
    atom that was forgotten is left alone and keeps its pair: refreshing it
    used to replace its metadata, stamp and all, and bring it back.

    Args:
        overlap: The SpaceOverlap result from ``space_overlap()``.
        tenant_id: Tenant partition key.
        db: Database handle.
        embed_engine: Embedding provider for new atoms.
        atomspace: AtomSpace instance, read for the edges a bridge atom has.
        min_jaccard: Minimum Jaccard threshold to trigger materialization.

    Returns:
        Number of bridge atoms created or updated.

    Raises:
        ValueError: the bridge space already bridges a different pair.
    """
    if overlap.jaccard < min_jaccard or not overlap.pairs:
        return 0

    # A space overlaps itself completely, so bridging it would mint a duplicate
    # of every atom into an "X_x_X" space that means nothing.
    if overlap.space_a == overlap.space_b:
        return 0

    bridge_space = overlap.bridge_space_name
    bridged, tombstones = _existing_bridges(db, tenant_id, bridge_space)
    sources = {atom_id for key in (*bridged, *tombstones) for atom_id in key}
    _refuse_foreign_bridge(bridge_space, sources, overlap, tenant_id, db)

    count = 0
    for pair in overlap.pairs:
        key = tuple(sorted((pair.atom_a.id, pair.atom_b.id)))
        if key in tombstones:
            continue
        _write_bridge(pair, bridged.get(key), bridge_space, tenant_id, db, embed_engine, atomspace)
        count += 1
    return count
