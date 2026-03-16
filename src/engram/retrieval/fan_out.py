"""Salience-scored fan-out retrieval."""
from __future__ import annotations

import struct

from engram.core.models import RecallResult, atom_from_row
from engram.retrieval.salience import compute_salience


def retrieve(
    query: str,
    agent_id: str,
    db,
    embed_engine,
    top_k: int = 10,
    min_confidence: float = 0.1,
) -> list[RecallResult]:
    """
    Full retrieval pipeline:
      1. Embed query
      2. KNN search in vec_atoms (top 50)
      3. 1-hop graph expansion via relation atoms
      4. Score all candidates by salience (personality-weighted)
      5. Return top_k sorted by descending salience
    """
    query_vec = embed_engine.embed(query)
    vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)

    # Load personality weights, fall back to defaults
    personality = db.fetchone(
        "SELECT * FROM personality WHERE agent_id = ?", (agent_id,)
    )
    p = dict(personality) if personality else {}
    w_similarity = p.get("w_similarity", 0.35)
    w_sti = p.get("w_sti", 0.25)
    w_confidence = p.get("w_confidence", 0.20)
    w_lti = p.get("w_lti", 0.10)
    w_valence = p.get("w_valence", 0.10)

    # Step 1: KNN entry points
    knn_rows = db.fetchall(
        """SELECT atom_id, distance FROM vec_atoms
           WHERE embedding MATCH ? AND agent_id = ?
           ORDER BY distance
           LIMIT 50""",
        (vec_bytes, agent_id),
    )

    if not knn_rows:
        return []

    knn_ids = [r["atom_id"] for r in knn_rows]
    knn_distances = {r["atom_id"]: r["distance"] for r in knn_rows}

    # Step 2: 1-hop expansion via relation atoms
    placeholders = ",".join("?" * len(knn_ids))
    expanded_ids: set[str] = set(knn_ids)

    forward = db.fetchall(
        f"SELECT target_id FROM atoms WHERE source_id IN ({placeholders}) AND type = 'relation' AND agent_id = ?",
        (*knn_ids, agent_id),
    )
    expanded_ids.update(r["target_id"] for r in forward if r["target_id"])

    backward = db.fetchall(
        f"SELECT source_id FROM atoms WHERE target_id IN ({placeholders}) AND type = 'relation' AND agent_id = ?",
        (*knn_ids, agent_id),
    )
    expanded_ids.update(r["source_id"] for r in backward if r["source_id"])
    expanded_ids.discard(None)

    if not expanded_ids:
        return []

    # Step 3: Fetch all candidate atoms (non-relation, above confidence floor)
    exp_list = list(expanded_ids)
    exp_placeholders = ",".join("?" * len(exp_list))
    atoms_rows = db.fetchall(
        f"""SELECT * FROM atoms
            WHERE id IN ({exp_placeholders}) AND agent_id = ? AND type != 'relation'
              AND confidence >= ?""",
        (*exp_list, agent_id, min_confidence),
    )

    # Step 4: Score each candidate by salience
    results: list[RecallResult] = []
    for row in atoms_rows:
        atom = atom_from_row(row)
        atom_id = atom.id

        # KNN atoms get a real similarity score; expanded-only atoms score 0
        if atom_id in knn_distances:
            similarity = max(0.0, 1.0 - knn_distances[atom_id])
        else:
            similarity = 0.0

        salience = compute_salience(
            similarity=similarity,
            sti=atom.attention.sti,
            confidence=atom.truth.confidence,
            lti=atom.attention.lti,
            valence=atom.valence.valence,
            intensity=atom.valence.intensity,
            w_similarity=w_similarity,
            w_sti=w_sti,
            w_confidence=w_confidence,
            w_lti=w_lti,
            w_valence=w_valence,
        )
        results.append(RecallResult(atom=atom, salience=salience, similarity=similarity))

    results.sort(key=lambda r: r.salience, reverse=True)
    return results[:top_k]
