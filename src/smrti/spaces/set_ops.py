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
    AtomType,
    SpaceOverlap,
    SpaceSetResult,
    atom_from_row,
)
from smrti.core.provenance import ATOM_FORGOTTEN, forgotten_sql

# ── Weights for the contextual similarity blend ──────────────────────
W_EMBEDDING: float = 0.6
W_ENTITY_TYPE: float = 0.2
W_NEIGHBORHOOD: float = 0.2

# What a signal the two atoms do not both carry contributes once either of
# them carries context: the midpoint, neither agreement nor a clash. Handing
# its weight to the embedding instead read "unknown" as "agrees", and two
# atoms labelled "Java" scored a perfect 1.0 whenever one had no neighbours.
NEUTRAL: float = 0.5

# How many atoms of each space an operation considers: the most salient
# ones, by attention. Every result reports the sample against the space's
# size (``SpaceOverlap.sampled_a`` / ``size_a``), so a Jaccard over a
# 500-atom head of a 20,000-atom space is read as what it is. Forgotten
# atoms are neither sampled nor counted: a set operation renders the space,
# and forgetting hides an atom from every rendering.
SAMPLE_LIMIT = 500

# Vectors fetched per query, under SQLite's oldest default limit of 999
# bound parameters.
_VECTOR_CHUNK = 500


def _get_space_atoms(tenant_id: str, space: str, db) -> list[Atom]:
    """Return the most salient non-relation atoms in a space, up to SAMPLE_LIMIT."""
    rows = db.fetchall(
        f"""SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'
              AND NOT {ATOM_FORGOTTEN}
            ORDER BY (sti + lti) DESC LIMIT ?""",
        (tenant_id, space, SAMPLE_LIMIT),
    )
    return [atom_from_row(r) for r in rows]


def _space_size(tenant_id: str, space: str, db) -> int:
    row = db.fetchone(
        f"""SELECT COUNT(*) AS n FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'
              AND NOT {ATOM_FORGOTTEN}""",
        (tenant_id, space),
    )
    return row["n"] if row else 0


def _unit_rows(vectors: list[list[float]]) -> np.ndarray:
    """The vectors as rows of unit length; a zero vector stays zero."""
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return matrix / norms


def _get_embeddings(atom_ids: list[str], db) -> dict[str, list[float]]:
    """The stored embeddings of *atom_ids*, by id; an atom with no vector is
    left out. One query per chunk rather than one per atom: two full samples
    were a thousand round trips before the first similarity was taken."""
    by_rowid = {stable_rowid(atom_id): atom_id for atom_id in atom_ids}
    rowids = list(by_rowid)
    found: dict[str, list[float]] = {}
    for start in range(0, len(rowids), _VECTOR_CHUNK):
        chunk = rowids[start : start + _VECTOR_CHUNK]
        rows = db.fetchall(
            f"SELECT rowid, embedding FROM vec_atoms WHERE rowid IN ({','.join('?' * len(chunk))})",
            tuple(chunk),
        )
        for row in rows:
            blob = row["embedding"]
            found[by_rowid[row["rowid"]]] = list(struct.unpack(f"{len(blob) // 4}f", blob))
    # In sample order, so the similarity matrix is laid out the same way
    # whatever order the index returned the rows in.
    return {atom_id: found[atom_id] for atom_id in atom_ids if atom_id in found}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _entity_type_score(a: Atom, b: Atom) -> float | None:
    """Return 1.0 if entity types match, 0.0 if they clash, None when unknown."""
    if a.entity_type is None or b.entity_type is None:
        return None
    return 1.0 if a.entity_type == b.entity_type else 0.0


def _get_neighbor_labels(atom_id: str, tenant_id: str, space: str, db) -> list[str]:
    """Get labels of 1-hop neighbors (via relation edges) for context.

    One branch per direction rather than one join on ``source_id = ? OR
    target_id = ?``: SQLite answers the OR with a scan of the whole space,
    34–39 ms a call at 48k rows, while each branch names the tenant, the
    space and one endpoint and is answered from an index.
    """
    neighbour = f"a.tenant_id = ? AND a.space = ? AND a.type != 'relation' AND NOT {forgotten_sql('a')}"
    rows = db.fetchall(
        f"""SELECT a.label FROM atoms r JOIN atoms a ON a.id = r.target_id
           WHERE r.tenant_id = ? AND r.space = ? AND r.source_id = ? AND r.type = 'relation'
             AND {neighbour}
           UNION ALL
           SELECT a.label FROM atoms r JOIN atoms a ON a.id = r.source_id
           WHERE r.tenant_id = ? AND r.space = ? AND r.target_id = ? AND r.type = 'relation'
             AND {neighbour}
           LIMIT 20""",
        (tenant_id, space, atom_id, tenant_id, space, tenant_id, space, atom_id, tenant_id, space),
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


def _prefetch_neighbor_contexts(atoms: list[Atom], db, embed_engine, cache: dict) -> None:
    """Fill *cache* for *atoms* with one batched embedding call rather than
    one encoder pass per atom that has neighbours."""
    texts: dict[str, str] = {}
    for atom in atoms:
        if atom.id in cache:
            continue
        labels = _get_neighbor_labels(atom.id, atom.tenant_id, atom.space, db)
        if labels:
            texts[atom.id] = " ".join(labels)
        else:
            cache[atom.id] = None
    if texts:
        for atom_id, vec in zip(texts, embed_engine.embed_batch(list(texts.values()))):
            cache[atom_id] = list(vec)


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
    similarity. When neither atom carries any context — no entity type, and
    no neighbours on either side — the embedding is the only evidence there
    is and decides alone, so two identical memories still score 1.0. Once
    either carries context, a signal the two do not both carry counts as
    ``NEUTRAL``: one "Java" linked to the JVM and another linked to an
    island, both untyped, are not known to be the same thing, and handing
    the type's weight to the embedding of a bare name — which is exactly
    what a homonym fools — merged them. Without an embed engine the
    neighbourhood is not measured at all, and its weight goes to the
    embedding as before.
    """
    vec_a = vec_b = None
    if embed_engine is not None:
        cache = {} if neighbor_cache is None else neighbor_cache
        vec_a = _neighbor_context_embedding(atom_a, db, embed_engine, cache)
        vec_b = _neighbor_context_embedding(atom_b, db, embed_engine, cache)
    if _context_free(atom_a, atom_b, vec_a, vec_b):
        return _unit(emb_sim)

    type_score = _entity_type_score(atom_a, atom_b)
    if type_score is None and AtomType.CONCEPT not in (atom_a.type, atom_b.type):
        # Entity types describe entities. An episode, belief or goal never
        # has one, and reading that as half a disagreement held two copies of
        # the same sentence under the threshold; their text alone decides.
        type_score = emb_sim
    score = W_EMBEDDING * emb_sim + W_ENTITY_TYPE * _or_neutral(type_score)
    if embed_engine is None:
        return _unit(score + W_NEIGHBORHOOD * emb_sim)
    both_linked = vec_a is not None and vec_b is not None
    return _unit(score + W_NEIGHBORHOOD * _or_neutral(_cosine_similarity(vec_a, vec_b) if both_linked else None))


def _context_free(atom_a: Atom, atom_b: Atom, vec_a: list[float] | None, vec_b: list[float] | None) -> bool:
    """Whether neither atom carries an entity type or neighbours."""
    return atom_a.entity_type is None and atom_b.entity_type is None and vec_a is None and vec_b is None


def _or_neutral(score: float | None) -> float:
    return NEUTRAL if score is None else score


def _unit(score: float) -> float:
    return max(0.0, min(1.0, score))


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
    atom_a_map = {a.id: a for a in atoms_a}
    atom_b_map = {a.id: a for a in atoms_b}
    pairs: list[AtomPair] = []
    matched_a: set[str] = set()
    matched_b: set[str] = set()

    # Sort by contextual similarity descending, greedily assign
    candidates = _candidates(atom_a_map, atom_b_map, db, threshold, embed_engine)
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


def _raw_similarities(
    atom_a_map: dict[str, Atom], atom_b_map: dict[str, Atom], db
) -> tuple[list[str], list[str], np.ndarray]:
    """The ids of the sampled atoms that have vectors, and the cosine
    similarity of every a × b pair of them."""
    emb_a = _get_embeddings(list(atom_a_map), db)
    emb_b = _get_embeddings(list(atom_b_map), db)
    ids_a, ids_b = list(emb_a), list(emb_b)
    if not ids_a or not ids_b:
        return ids_a, ids_b, np.zeros((len(ids_a), len(ids_b)), dtype=np.float32)
    return ids_a, ids_b, _unit_rows([emb_a[i] for i in ids_a]) @ _unit_rows([emb_b[i] for i in ids_b]).T


def _candidates(
    atom_a_map: dict[str, Atom],
    atom_b_map: dict[str, Atom],
    db,
    threshold: float,
    embed_engine=None,
) -> list[tuple[float, str, str]]:
    """Every pair whose contextual similarity reaches *threshold*, as
    ``(similarity, a id, b id)``.

    Pre-filter: only compute expensive contextual similarity for pairs whose
    raw embedding similarity is above a looser pre-filter (threshold - 0.15).
    This avoids neighborhood embedding calls for clearly-unrelated pairs.
    The raw similarities come from one matrix product — two full samples
    are a quarter of a million pairs, which as Python cosines was seconds
    per operation.
    """
    ids_a, ids_b, sims = _raw_similarities(atom_a_map, atom_b_map, db)
    rows, cols = np.nonzero(sims >= max(0.0, threshold - 0.15))

    neighbor_cache: dict[str, list[float] | None] = {}
    if embed_engine is not None and len(rows):
        involved = [atom_a_map[ids_a[i]] for i in np.unique(rows)] + [atom_b_map[ids_b[j]] for j in np.unique(cols)]
        _prefetch_neighbor_contexts(involved, db, embed_engine, neighbor_cache)
    found: list[tuple[float, str, str]] = []
    for i, j in zip(rows, cols):
        aid, bid = ids_a[i], ids_b[j]
        ctx_sim = _contextual_similarity(
            atom_a_map[aid], atom_b_map[bid], float(sims[i, j]), db, embed_engine,
            neighbor_cache=neighbor_cache,
        )
        if ctx_sim >= threshold:
            found.append((ctx_sim, aid, bid))
    return found


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
