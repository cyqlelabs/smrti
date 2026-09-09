"""Core set theory operations on memory spaces.

Atoms across spaces are matched using **contextual similarity** — a weighted
blend of three signals:

  1. **Embedding similarity** (w=0.6) — cosine distance between atom
     embeddings.  Language-agnostic but blind to homonyms.
  2. **Entity-type compatibility** (w=0.2) — hard penalty when entity types
     clash (e.g. ``location`` vs ``technology``), bonus when they agree.
     Prevents "Java the island" from matching "Java the language".
  3. **Neighborhood similarity** (w=0.2) — cosine distance between the
     concatenated neighbor-label embeddings of each atom.  Two atoms named
     "Java" will diverge because one's neighbors are "Indonesia", "Bali"
     and the other's are "JVM", "Spring".

All three components are language-agnostic (no English word lists).
"""
from __future__ import annotations

import struct

import numpy as np

from smrti.core.db import stable_rowid
from smrti.core.models import (
    Atom,
    AtomPair,
    SpaceOverlap,
    SpaceSetResult,
    atom_from_row,
)

# ── Weights for the contextual similarity blend ──────────────────────
W_EMBEDDING: float = 0.6
W_ENTITY_TYPE: float = 0.2
W_NEIGHBORHOOD: float = 0.2

# How many atoms of each space an operation considers: the most salient
# ones, by attention. Every result reports the sample against the space's
# size (``SpaceOverlap.sampled_a`` / ``size_a``), so a Jaccard over a
# 500-atom head of a 20,000-atom space is read as what it is.
SAMPLE_LIMIT = 500


def _get_space_atoms(tenant_id: str, space: str, db) -> list[Atom]:
    """Return the most salient non-relation atoms in a space, up to SAMPLE_LIMIT."""
    rows = db.fetchall(
        """SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'
           ORDER BY (sti + lti) DESC LIMIT ?""",
        (tenant_id, space, SAMPLE_LIMIT),
    )
    return [atom_from_row(r) for r in rows]


def _space_size(tenant_id: str, space: str, db) -> int:
    row = db.fetchone(
        "SELECT COUNT(*) AS n FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        (tenant_id, space),
    )
    return row["n"] if row else 0


def _unit_rows(vectors: list[list[float]]) -> np.ndarray:
    """The vectors as rows of unit length; a zero vector stays zero."""
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return matrix / norms


def _get_embedding(atom_id: str, db) -> list[float] | None:
    """Fetch the stored embedding for an atom."""
    row = db.fetchone(
        "SELECT embedding FROM vec_atoms WHERE rowid = ?",
        (stable_rowid(atom_id),),
    )
    if row is None:
        return None
    blob = row["embedding"]
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _entity_type_score(a: Atom, b: Atom) -> float | None:
    """Return 1.0 if entity types match, 0.0 if they clash, None when unknown.

    None means the signal is absent — its weight is redistributed to
    embedding similarity by ``_contextual_similarity``.
    """
    if a.entity_type is None or b.entity_type is None:
        return None
    return 1.0 if a.entity_type == b.entity_type else 0.0


def _get_neighbor_labels(atom_id: str, tenant_id: str, space: str, db) -> list[str]:
    """Get labels of 1-hop neighbors (via relation edges) for context."""
    rows = db.fetchall(
        """SELECT a.label FROM atoms a
           INNER JOIN atoms r ON (
               (r.source_id = ? AND r.target_id = a.id)
               OR (r.target_id = ? AND r.source_id = a.id)
           )
           WHERE r.type = 'relation' AND r.tenant_id = ? AND r.space = ?
             AND a.tenant_id = ? AND a.space = ?
             AND a.type != 'relation'
           LIMIT 20""",
        (atom_id, atom_id, tenant_id, space, tenant_id, space),
    )
    return [r["label"] for r in rows]


def _neighbor_context_embedding(
    atom: Atom, db, embed_engine, cache: dict
) -> list[float] | None:
    """Embed the concatenated neighbor labels for an atom, memoized per call.

    Returns None when the atom has no neighbors.
    """
    if atom.id in cache:
        return cache[atom.id]
    labels = _get_neighbor_labels(atom.id, atom.tenant_id, atom.space, db)
    vec = list(embed_engine.embed(" ".join(labels))) if labels else None
    cache[atom.id] = vec
    return vec


def _neighborhood_similarity(
    atom_a: Atom, atom_b: Atom, db, embed_engine, cache: dict
) -> float | None:
    """Compute similarity between atoms' graph neighborhoods.

    Concatenates neighbor labels into a single string per atom and compares
    their embeddings.  Returns None when either atom has no neighbors — the
    signal is absent and its weight is redistributed to embedding similarity.
    """
    vec_a = _neighbor_context_embedding(atom_a, db, embed_engine, cache)
    vec_b = _neighbor_context_embedding(atom_b, db, embed_engine, cache)
    if vec_a is None or vec_b is None:
        return None
    return _cosine_similarity(vec_a, vec_b)


def _contextual_similarity(
    atom_a: Atom,
    atom_b: Atom,
    emb_sim: float,
    db,
    embed_engine=None,
    neighbor_cache: dict | None = None,
) -> float:
    """Compute multi-signal contextual similarity between two atoms.

    Blends embedding similarity, entity-type compatibility, and neighborhood
    similarity.  When a signal is absent (entity type unknown on either side,
    no embed engine, or either neighborhood empty) its weight is redistributed
    to embedding similarity, so two identical atoms always score 1.0.
    """
    w_emb = W_EMBEDDING
    score = 0.0

    et_score = _entity_type_score(atom_a, atom_b)
    if et_score is None:
        w_emb += W_ENTITY_TYPE
    else:
        score += W_ENTITY_TYPE * et_score

    ns = None
    if embed_engine is not None:
        if neighbor_cache is None:
            neighbor_cache = {}
        ns = _neighborhood_similarity(atom_a, atom_b, db, embed_engine, neighbor_cache)
    if ns is None:
        w_emb += W_NEIGHBORHOOD
    else:
        score += W_NEIGHBORHOOD * ns

    return max(0.0, min(1.0, w_emb * emb_sim + score))


def _match_atoms(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    db,
    threshold: float,
    embed_engine=None,
) -> tuple[list[AtomPair], set[str], set[str]]:
    """Find best contextual-similarity matches between two atom sets.

    Uses a three-signal blend (embedding + entity-type + neighborhood) so that
    homonyms like "Java" (island) and "Java" (language) are correctly
    distinguished when they have different entity types or different graph
    neighborhoods.

    Returns:
        pairs: matched AtomPair list (contextual similarity >= threshold)
        matched_a_ids: IDs from atoms_a that were matched
        matched_b_ids: IDs from atoms_b that were matched
    """
    # Pre-fetch embeddings
    emb_a: dict[str, list[float]] = {}
    for atom in atoms_a:
        vec = _get_embedding(atom.id, db)
        if vec is not None:
            emb_a[atom.id] = vec

    emb_b: dict[str, list[float]] = {}
    for atom in atoms_b:
        vec = _get_embedding(atom.id, db)
        if vec is not None:
            emb_b[atom.id] = vec

    pairs: list[AtomPair] = []
    matched_a: set[str] = set()
    matched_b: set[str] = set()
    atom_b_map = {a.id: a for a in atoms_b}
    atom_a_map = {a.id: a for a in atoms_a}

    # Pre-filter: only compute expensive contextual similarity for pairs whose
    # raw embedding similarity is above a looser pre-filter (threshold - 0.15).
    # This avoids neighborhood embedding calls for clearly-unrelated pairs.
    # The raw similarities come from one matrix product — two full samples
    # are a quarter of a million pairs, which as Python cosines was seconds
    # per operation.
    pre_filter = max(0.0, threshold - 0.15)

    neighbor_cache: dict[str, list[float] | None] = {}
    candidates: list[tuple[float, str, str]] = []
    if emb_a and emb_b:
        ids_a, ids_b = list(emb_a), list(emb_b)
        sims = _unit_rows([emb_a[i] for i in ids_a]) @ _unit_rows([emb_b[i] for i in ids_b]).T
        for i, j in zip(*np.nonzero(sims >= pre_filter)):
            aid, bid = ids_a[i], ids_b[j]
            ctx_sim = _contextual_similarity(
                atom_a_map[aid], atom_b_map[bid], float(sims[i, j]), db, embed_engine,
                neighbor_cache=neighbor_cache,
            )
            if ctx_sim >= threshold:
                candidates.append((ctx_sim, aid, bid))

    # Sort by contextual similarity descending, greedily assign
    candidates.sort(key=lambda x: x[0], reverse=True)
    for sim, aid, bid in candidates:
        if aid in matched_a or bid in matched_b:
            continue
        pairs.append(AtomPair(
            atom_a=atom_a_map[aid],
            atom_b=atom_b_map[bid],
            similarity=sim,
        ))
        matched_a.add(aid)
        matched_b.add(bid)

    return pairs, matched_a, matched_b


def space_overlap(
    tenant_id: str,
    space_a: str,
    space_b: str,
    db,
    threshold: float = 0.85,
    embed_engine=None,
) -> SpaceOverlap:
    """Compute the overlap between two spaces.

    Returns a SpaceOverlap with matched pairs and Jaccard similarity.
    When ``embed_engine`` is provided, neighborhood similarity is included
    in the contextual score (recommended for disambiguation).
    """
    atoms_a = _get_space_atoms(tenant_id, space_a, db)
    atoms_b = _get_space_atoms(tenant_id, space_b, db)
    coverage = dict(
        sampled_a=len(atoms_a),
        sampled_b=len(atoms_b),
        size_a=_space_size(tenant_id, space_a, db),
        size_b=_space_size(tenant_id, space_b, db),
    )

    if not atoms_a or not atoms_b:
        return SpaceOverlap(space_a=space_a, space_b=space_b, jaccard=0.0, **coverage)

    pairs, matched_a, matched_b = _match_atoms(
        atoms_a, atoms_b, db, threshold, embed_engine,
    )

    # Jaccard = |A ∩ B| / |A ∪ B| = matched / (|A| + |B| - matched), over
    # the sampled atoms.
    intersection_size = len(pairs)
    union_size = len(atoms_a) + len(atoms_b) - intersection_size
    jaccard = intersection_size / union_size if union_size > 0 else 0.0

    return SpaceOverlap(
        space_a=space_a,
        space_b=space_b,
        jaccard=jaccard,
        pairs=pairs,
        **coverage,
    )


def space_intersection(
    tenant_id: str,
    space_a: str,
    space_b: str,
    db,
    threshold: float = 0.85,
    embed_engine=None,
) -> SpaceSetResult:
    """Return atoms that exist in both spaces (by contextual similarity).

    The returned atoms are from space_a (the "left" side of the intersection).
    The overlap field contains the matched pairs for reference.
    """
    overlap = space_overlap(tenant_id, space_a, space_b, db, threshold, embed_engine)
    atoms = [p.atom_a for p in overlap.pairs]
    return SpaceSetResult(
        operation="intersection",
        spaces=[space_a, space_b],
        atoms=atoms,
        overlap=overlap,
    )


def space_difference(
    tenant_id: str,
    space_a: str,
    space_b: str,
    db,
    threshold: float = 0.85,
    embed_engine=None,
) -> SpaceSetResult:
    """Return atoms in space_a that have no match in space_b."""
    atoms_a = _get_space_atoms(tenant_id, space_a, db)
    atoms_b = _get_space_atoms(tenant_id, space_b, db)

    if not atoms_b:
        return SpaceSetResult(
            operation="difference",
            spaces=[space_a, space_b],
            atoms=atoms_a,
        )

    _, matched_a, _ = _match_atoms(atoms_a, atoms_b, db, threshold, embed_engine)
    unique = [a for a in atoms_a if a.id not in matched_a]

    return SpaceSetResult(
        operation="difference",
        spaces=[space_a, space_b],
        atoms=unique,
    )


def space_union(
    tenant_id: str,
    space_a: str,
    space_b: str,
    db,
    threshold: float = 0.85,
    embed_engine=None,
) -> SpaceSetResult:
    """Return deduplicated union of atoms from both spaces.

    For matched pairs, the atom from space_a is kept (dedup representative).
    """
    atoms_a = _get_space_atoms(tenant_id, space_a, db)
    atoms_b = _get_space_atoms(tenant_id, space_b, db)

    if not atoms_a:
        return SpaceSetResult(
            operation="union",
            spaces=[space_a, space_b],
            atoms=atoms_b,
        )
    if not atoms_b:
        return SpaceSetResult(
            operation="union",
            spaces=[space_a, space_b],
            atoms=atoms_a,
        )

    overlap = space_overlap(tenant_id, space_a, space_b, db, threshold, embed_engine)
    matched_b_ids = {p.atom_b.id for p in overlap.pairs}

    # All of A + unmatched from B
    union_atoms = list(atoms_a) + [a for a in atoms_b if a.id not in matched_b_ids]

    return SpaceSetResult(
        operation="union",
        spaces=[space_a, space_b],
        atoms=union_atoms,
        overlap=overlap,
    )


def space_symmetric_difference(
    tenant_id: str,
    space_a: str,
    space_b: str,
    db,
    threshold: float = 0.85,
    embed_engine=None,
) -> SpaceSetResult:
    """Return atoms that are in one space but not the other."""
    atoms_a = _get_space_atoms(tenant_id, space_a, db)
    atoms_b = _get_space_atoms(tenant_id, space_b, db)

    if not atoms_a and not atoms_b:
        return SpaceSetResult(
            operation="symmetric_difference",
            spaces=[space_a, space_b],
            atoms=[],
        )
    if not atoms_a:
        return SpaceSetResult(
            operation="symmetric_difference",
            spaces=[space_a, space_b],
            atoms=atoms_b,
        )
    if not atoms_b:
        return SpaceSetResult(
            operation="symmetric_difference",
            spaces=[space_a, space_b],
            atoms=atoms_a,
        )

    _, matched_a, matched_b = _match_atoms(
        atoms_a, atoms_b, db, threshold, embed_engine,
    )
    unique = [a for a in atoms_a if a.id not in matched_a]
    unique += [a for a in atoms_b if a.id not in matched_b]

    return SpaceSetResult(
        operation="symmetric_difference",
        spaces=[space_a, space_b],
        atoms=unique,
    )
