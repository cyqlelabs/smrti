"""Cross-domain association discovery ("charisma" engine)."""
from __future__ import annotations

import struct
import uuid


def discover_connections(agent_id: str, db, embed_engine) -> int:
    """Find surprising associations between unconnected high-LTI atoms.

    For each of the top 50 atoms by LTI, embed the atom's content and search
    for semantically similar atoms that are not yet directly linked. Creates a
    weak 'associated' relation atom for each new pair found within the cosine
    distance threshold of 0.4.

    Returns the count of new relation atoms created.
    """
    high_lti = db.fetchall(
        """SELECT id, label, content FROM atoms
           WHERE agent_id = ? AND lti > 0.3 AND type != 'relation'
           ORDER BY lti DESC LIMIT 50""",
        (agent_id,),
    )

    new_count = 0

    for atom in high_lti:
        text = atom["content"] or atom["label"]
        vec = embed_engine.embed(text)
        vec_bytes = struct.pack(f"{len(vec)}f", *vec)

        knn = db.fetchall(
            """SELECT atom_id, distance FROM vec_atoms
               WHERE embedding MATCH ? AND agent_id = ?
               ORDER BY distance LIMIT 10""",
            (vec_bytes, agent_id),
        )

        # Collect existing 1-hop neighbors so we don't duplicate edges
        existing_rows = db.fetchall(
            """SELECT target_id AS neighbor FROM atoms WHERE source_id = ? AND type = 'relation'
               UNION
               SELECT source_id AS neighbor FROM atoms WHERE target_id = ? AND type = 'relation'""",
            (atom["id"], atom["id"]),
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
                       (id, type, label, source_id, target_id, relation, agent_id, probability, confidence)
                   VALUES (?, 'relation', 'associated', ?, ?, 'associated', ?, 0.5, 0.1)""",
                (link_id, atom["id"], cid, agent_id),
            )
            new_count += 1

    return new_count
