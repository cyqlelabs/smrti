"""Bridge space materialization: when two spaces overlap, grow a new subspace.

A bridge space contains atoms that represent the conceptual intersection of
two parent spaces.  Each bridge atom merges truth values (via PLN), averages
attention weights, and blends valence from its two source atoms.  Relation
edges connect the bridge atom back to both parents so graph traversal can
flow across space boundaries.

Bridge spaces are named ``{sorted_a}_x_{sorted_b}`` so the operation is
commutative (A∩B == B∩A always produces the same space name).
"""
from __future__ import annotations

import json
import uuid

from smrti.core.models import (
    Atom,
    AtomPair,
    AtomType,
    AttentionValue,
    SpaceOverlap,
    TruthValue,
    Valence,
)


def _merge_pair(pair: AtomPair, bridge_space: str, tenant_id: str) -> Atom:
    """Create a single bridge atom from a matched pair."""
    a, b = pair.atom_a, pair.atom_b

    # PLN merge for truth value
    merged_truth = a.truth.merge(b.truth)

    # Average attention (bridge starts with combined salience signal)
    merged_attention = AttentionValue(
        sti=(a.attention.sti + b.attention.sti) / 2.0,
        lti=max(a.attention.lti, b.attention.lti),
    )

    # Blend valence: weighted by intensity (stronger signal wins)
    total_intensity = a.valence.intensity + b.valence.intensity
    if total_intensity > 1e-9:
        blended_valence = (
            a.valence.valence * a.valence.intensity
            + b.valence.valence * b.valence.intensity
        ) / total_intensity
        blended_intensity = max(a.valence.intensity, b.valence.intensity)
    else:
        blended_valence = 0.0
        blended_intensity = 0.0

    # Use the higher-confidence atom's label
    label = a.label if a.truth.confidence >= b.truth.confidence else b.label
    content = a.content if a.truth.confidence >= b.truth.confidence else b.content

    # Preserve entity_type from the higher-confidence source
    entity_type = a.entity_type if a.truth.confidence >= b.truth.confidence else b.entity_type

    # Merge metadata from both sources
    metadata = {**a.metadata, **b.metadata}
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
            valence=max(-1.0, min(1.0, blended_valence)),
            intensity=min(1.0, blended_intensity),
        ),
        entity_type=entity_type,
        tenant_id=tenant_id,
        space=bridge_space,
        metadata=metadata,
    )


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
    than duplicated.

    Args:
        overlap: The SpaceOverlap result from ``space_overlap()``.
        tenant_id: Tenant partition key.
        db: Database handle.
        embed_engine: Embedding provider for new atoms.
        atomspace: AtomSpace instance for atom/relation creation.
        min_jaccard: Minimum Jaccard threshold to trigger materialization.

    Returns:
        Number of bridge atoms created or updated.
    """
    if overlap.jaccard < min_jaccard or not overlap.pairs:
        return 0

    # A space overlaps itself completely, so bridging it would mint a duplicate
    # of every atom into an "X_x_X" space that means nothing.
    if overlap.space_a == overlap.space_b:
        return 0

    bridge_space = overlap.bridge_space_name
    count = 0

    # Index existing bridge atoms by their source pair so we can update.
    # Keys are order-normalized so discovery from either parent space finds
    # the same stored bridge atom.
    existing = db.fetchall(
        "SELECT id, metadata FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        (tenant_id, bridge_space),
    )
    source_pair_to_id: dict[tuple[str, str], str] = {}
    for row in existing:
        meta = json.loads(row["metadata"] or "{}")
        src_a = meta.get("bridge_source_a")
        src_b = meta.get("bridge_source_b")
        if src_a and src_b:
            source_pair_to_id[tuple(sorted((src_a, src_b)))] = row["id"]

    for pair in overlap.pairs:
        key = tuple(sorted((pair.atom_a.id, pair.atom_b.id)))
        bridge_atom = _merge_pair(pair, bridge_space, tenant_id)

        if key in source_pair_to_id:
            # Update existing bridge atom
            bridge_atom.id = source_pair_to_id[key]
            atomspace.update_atom(bridge_atom)
            count += 1
        else:
            # Create new bridge atom + relation edges
            atom_id = atomspace.add_atom(bridge_atom)

            # Bridge → source_a
            atomspace.link_atoms(
                source_id=atom_id,
                target_id=pair.atom_a.id,
                relation="bridge",
                tenant_id=tenant_id,
                space=bridge_space,
                truth=TruthValue(probability=pair.similarity, confidence=0.8),
            )

            # Bridge → source_b
            atomspace.link_atoms(
                source_id=atom_id,
                target_id=pair.atom_b.id,
                relation="bridge",
                tenant_id=tenant_id,
                space=bridge_space,
                truth=TruthValue(probability=pair.similarity, confidence=0.8),
            )

            count += 1

    return count
