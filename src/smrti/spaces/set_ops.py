"""Core set theory operations on memory spaces.

Atoms across spaces are matched by embedding cosine similarity.
Two atoms are considered equivalent when their similarity exceeds a threshold
(default 0.85), making the operations language-agnostic.
"""
from __future__ import annotations

import struct

from smrti.core.models import (
    Atom,
    AtomPair,
    SpaceOverlap,
    SpaceSetResult,
    atom_from_row,
)


def _get_space_atoms(tenant_id: str, space: str, db) -> list[Atom]:
    """Return all non-relation atoms in a space."""
    rows = db.fetchall(
        "SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation'",
        (tenant_id, space),
    )
    return [atom_from_row(r) for r in rows]


def _get_embedding(atom_id: str, db) -> list[float] | None:
    """Fetch the stored embedding for an atom."""
    row = db.fetchone(
        "SELECT embedding FROM vec_atoms WHERE atom_id = ?",
        (atom_id,),
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


def _match_atoms(
    atoms_a: list[Atom],
    atoms_b: list[Atom],
    db,
    threshold: float,
) -> tuple[list[AtomPair], set[str], set[str]]:
    """Find best embedding matches between two atom sets.

    Returns:
        pairs: matched AtomPair list (similarity >= threshold)
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

    # For each atom in A, find best match in B
    pairs: list[AtomPair] = []
    matched_a: set[str] = set()
    matched_b: set[str] = set()
    atom_b_map = {a.id: a for a in atoms_b}
    atom_a_map = {a.id: a for a in atoms_a}

    # Greedy best-match: for each A atom, find closest B atom not yet taken
    candidates: list[tuple[float, str, str]] = []
    for aid, vec_a in emb_a.items():
        for bid, vec_b in emb_b.items():
            sim = _cosine_similarity(vec_a, vec_b)
            if sim >= threshold:
                candidates.append((sim, aid, bid))

    # Sort by similarity descending, greedily assign
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
) -> SpaceOverlap:
    """Compute the overlap between two spaces.

    Returns a SpaceOverlap with matched pairs and Jaccard similarity.
    """
    atoms_a = _get_space_atoms(tenant_id, space_a, db)
    atoms_b = _get_space_atoms(tenant_id, space_b, db)

    if not atoms_a or not atoms_b:
        return SpaceOverlap(space_a=space_a, space_b=space_b, jaccard=0.0)

    pairs, matched_a, matched_b = _match_atoms(atoms_a, atoms_b, db, threshold)

    # Jaccard = |A ∩ B| / |A ∪ B| = matched / (|A| + |B| - matched)
    intersection_size = len(pairs)
    union_size = len(atoms_a) + len(atoms_b) - intersection_size
    jaccard = intersection_size / union_size if union_size > 0 else 0.0

    return SpaceOverlap(
        space_a=space_a,
        space_b=space_b,
        jaccard=jaccard,
        pairs=pairs,
    )


def space_intersection(
    tenant_id: str,
    space_a: str,
    space_b: str,
    db,
    threshold: float = 0.85,
) -> SpaceSetResult:
    """Return atoms that exist in both spaces (by embedding similarity).

    The returned atoms are from space_a (the "left" side of the intersection).
    The overlap field contains the matched pairs for reference.
    """
    overlap = space_overlap(tenant_id, space_a, space_b, db, threshold)
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

    _, matched_a, _ = _match_atoms(atoms_a, atoms_b, db, threshold)
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

    overlap = space_overlap(tenant_id, space_a, space_b, db, threshold)
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

    _, matched_a, matched_b = _match_atoms(atoms_a, atoms_b, db, threshold)
    unique = [a for a in atoms_a if a.id not in matched_a]
    unique += [a for a in atoms_b if a.id not in matched_b]

    return SpaceSetResult(
        operation="symmetric_difference",
        spaces=[space_a, space_b],
        atoms=unique,
    )
