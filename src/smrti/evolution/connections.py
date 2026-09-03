"""Association discovery between similar high-importance atoms.

Every tenth epoch, each of the space's most important atoms is compared
against its nearest neighbours in embedding space, and a weak ``associated``
edge is drawn to any neighbour it is not already linked to. The edge gives
one-hop expansion a path between two memories that say nearly the same thing
in different words — a belief and the episode it was distilled from, a
concept and its paraphrase — which the entity resolver, working one mention
at a time, never saw side by side.

This is paraphrase linking, and it is named for what it does. An earlier
docstring called it cross-domain discovery of surprising associations; two
atoms within cosine 0.4 of each other are the opposite of surprising.
"""
from __future__ import annotations

import struct

from smrti.core.atomspace import AtomSpace
from smrti.core.db import stable_rowid
from smrti.core.models import TruthValue

# Cosine distance ceiling for a pair to be linked (similarity >= 0.6).
_MAX_DISTANCE = 0.4

# An association is the weakest edge the graph draws: an extracted relation
# or a healed mention should always outrank it.
_ASSOCIATION_TRUTH = TruthValue(probability=0.5, confidence=0.1)


def discover_connections(tenant_id: str, space: str, db, embed_engine) -> int:
    """Link each of the top 50 atoms by LTI to its unlinked near neighbours.

    Each atom's stored embedding is used (re-embedding only when the vector
    row is missing) to search for similar atoms within the same space that
    are not yet directly linked. Creates a weak 'associated' relation atom for
    each new pair within cosine distance 0.4.

    Returns the count of new relation atoms created.
    """
    high_lti = db.fetchall(
        """SELECT id, label, content FROM atoms
           WHERE tenant_id = ? AND space = ? AND lti > 0.3 AND type != 'relation'
           ORDER BY lti DESC LIMIT 50""",
        (tenant_id, space),
    )

    atomspace = AtomSpace(db, embed_engine)
    new_count = 0

    for atom in high_lti:
        vec_row = db.fetchone(
            "SELECT embedding FROM vec_atoms WHERE rowid = ?", (stable_rowid(atom["id"]),)
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
            if candidate["distance"] >= _MAX_DISTANCE:
                continue
            atomspace.link_atoms(
                atom["id"], cid, "associated", tenant_id, space,
                truth=_ASSOCIATION_TRUTH,
            )
            existing_ids.add(cid)
            new_count += 1

    return new_count
