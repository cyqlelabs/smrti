"""Salience-scored fan-out retrieval."""
from __future__ import annotations

import struct

from smrti.core.models import RecallResult, atom_from_row
from smrti.retrieval.salience import compute_salience


def retrieve(
    query: str,
    tenant_id: str,
    read_spaces: list[str],
    db,
    embed_engine,
    write_space: str,
    top_k: int = 10,
    min_confidence: float = 0.1,
) -> list[RecallResult]:
    """
    Full retrieval pipeline:
      1. Embed query
      2. KNN search in vec_atoms across all read_spaces (partitioned by tenant_id)
      3. 1-hop graph expansion via relation atoms within read_spaces
      4. Score all candidates by salience (personality-weighted from write_space)
      5. Return top_k sorted by descending salience
    """
    query_vec = embed_engine.embed(query)
    vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)

    # Load personality weights from write_space, fall back to defaults
    personality = db.fetchone(
        "SELECT * FROM personality WHERE tenant_id = ? AND space = ?",
        (tenant_id, write_space),
    )
    p = dict(personality) if personality else {}
    w_similarity = p.get("w_similarity", 0.35)
    w_sti = p.get("w_sti", 0.25)
    w_confidence = p.get("w_confidence", 0.20)
    w_lti = p.get("w_lti", 0.10)
    w_valence = p.get("w_valence", 0.10)
    valence_weight = p.get("valence_weight", 0.2)
    sti_boost = p.get("sti_boost_on_access", 0.5)

    # Step 1: KNN entry points — search across the full tenant partition
    knn_rows = db.fetchall(
        """SELECT atom_id, distance FROM vec_atoms
           WHERE embedding MATCH ? AND tenant_id = ?
           ORDER BY distance
           LIMIT 50""",
        (vec_bytes, tenant_id),
    )

    if not knn_rows:
        return []

    knn_ids = [r["atom_id"] for r in knn_rows]
    knn_distances = {r["atom_id"]: r["distance"] for r in knn_rows}

    # Step 2: Filter KNN hits to those within read_spaces, then 1-hop expand
    spaces_ph = ",".join("?" * len(read_spaces))

    id_ph = ",".join("?" * len(knn_ids))
    expanded_ids: set[str] = set(knn_ids)

    forward = db.fetchall(
        f"SELECT target_id FROM atoms WHERE source_id IN ({id_ph}) AND type = 'relation' AND tenant_id = ? AND space IN ({spaces_ph})",
        (*knn_ids, tenant_id, *read_spaces),
    )
    expanded_ids.update(r["target_id"] for r in forward if r["target_id"])

    backward = db.fetchall(
        f"SELECT source_id FROM atoms WHERE target_id IN ({id_ph}) AND type = 'relation' AND tenant_id = ? AND space IN ({spaces_ph})",
        (*knn_ids, tenant_id, *read_spaces),
    )
    expanded_ids.update(r["source_id"] for r in backward if r["source_id"])
    expanded_ids.discard(None)

    if not expanded_ids:
        return []

    # Step 3: Fetch candidate atoms — space-filtered here (overlay boundary)
    exp_list = list(expanded_ids)
    exp_ph = ",".join("?" * len(exp_list))
    atoms_rows = db.fetchall(
        f"""SELECT * FROM atoms
            WHERE id IN ({exp_ph})
              AND tenant_id = ?
              AND space IN ({spaces_ph})
              AND type IN ('concept', 'belief', 'episode', 'goal')
              AND confidence >= ?""",
        (*exp_list, tenant_id, *read_spaces, min_confidence),
    )

    # Step 4: Score each candidate by salience
    results: list[RecallResult] = []
    for row in atoms_rows:
        atom = atom_from_row(row)
        similarity = max(0.0, 1.0 - knn_distances[atom.id]) if atom.id in knn_distances else 0.0
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
            valence_weight=valence_weight,
        )
        results.append(RecallResult(atom=atom, salience=salience, similarity=similarity))

    results.sort(key=lambda r: r.salience, reverse=True)
    top_results = results[:top_k]

    # Boost STI on accessed atoms so recalled memories gain short-term importance
    if sti_boost > 0 and top_results:
        for r in top_results:
            db.execute(
                "UPDATE atoms SET sti = MIN(sti + ?, 3.0), updated_at = datetime('now') WHERE id = ?",
                (sti_boost, r.atom.id),
            )

    return top_results
