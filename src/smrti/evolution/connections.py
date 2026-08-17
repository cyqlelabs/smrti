"""Cross-domain association discovery ("charisma" engine)."""
from __future__ import annotations

import struct
import uuid


def discover_connections(tenant_id: str, space: str, db, embed_engine) -> int:
    """Find surprising associations between unconnected high-LTI atoms.

    For each of the top 50 atoms by LTI in the given space, load the atom's
    stored embedding (re-embedding only when the vector row is missing) and
    search for semantically similar atoms within the same space that are not
    yet directly linked. Creates a weak 'associated' relation atom for each
    new pair found within the cosine distance threshold of 0.4 (cosine
    distance 0.4 corresponds to cosine similarity 0.6).

    Returns the count of new relation atoms created.
    """
    high_lti = db.fetchall(
        """SELECT id, label, content FROM atoms
           WHERE tenant_id = ? AND space = ? AND lti > 0.3 AND type != 'relation'
           ORDER BY lti DESC LIMIT 50""",
        (tenant_id, space),
    )

    new_count = 0

    for atom in high_lti:
        vec_row = db.fetchone(
            "SELECT embedding FROM vec_atoms WHERE atom_id = ?", (atom["id"],)
        )
        if vec_row is not None:
            vec_bytes = vec_row["embedding"]
        else:
            text = atom["content"] or atom["label"]
            vec = embed_engine.embed(text)
            vec_bytes = struct.pack(f"{len(vec)}f", *vec)

        knn = db.fetchall(
            """SELECT atom_id, distance FROM vec_atoms
               WHERE embedding MATCH ? AND tenant_id = ? AND space = ?
               ORDER BY distance LIMIT 10""",
            (vec_bytes, tenant_id, space),
        )

        existing_rows = db.fetchall(
            """SELECT target_id AS neighbor FROM atoms
               WHERE source_id = ? AND type = 'relation' AND tenant_id = ? AND space = ?
               UNION
               SELECT source_id AS neighbor FROM atoms
               WHERE target_id = ? AND type = 'relation' AND tenant_id = ? AND space = ?""",
            (atom["id"], tenant_id, space, atom["id"], tenant_id, space),
        )
        existing_ids = {r["neighbor"] for r in existing_rows if r["neighbor"]}

        for candidate in knn:
            cid = candidate["atom_id"]
            if cid == atom["id"] or cid in existing_ids:
                continue
            if candidate["distance"] >= 0.4:
                continue

            link_id = str(uuid.uuid4())
            db.execute(
                """INSERT OR IGNORE INTO atoms
                       (id, type, label, source_id, target_id, relation, tenant_id, space, probability, confidence)
                   VALUES (?, 'relation', 'associated', ?, ?, 'associated', ?, ?, 0.5, 0.1)""",
                (link_id, atom["id"], cid, tenant_id, space),
            )
            new_count += 1

    return new_count
