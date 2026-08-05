"""Graph healing: detect and repair orphaned episodes during consolidation."""
from __future__ import annotations

import struct
import uuid


def heal_orphaned_episodes(tenant_id: str, space: str, db) -> int:
    """Find episodes that mention concepts but no person, and link them to a person.

    When the space contains exactly one person atom, every orphaned episode is
    attributed to it.  With multiple person atoms, each episode is attributed
    to the person whose stored embedding is most similar to the episode's
    stored embedding (cosine >= 0.3); episodes with no sufficiently similar
    person are skipped rather than mis-attributed.

    For each healed episode:
      1. Create ``episode -> mentions -> person`` edge
      2. For each concept the episode mentions, create ``person -> associated -> concept``
         with low confidence (0.2) so LLM-extracted relations supersede

    Returns the number of healed episodes.
    """
    person_rows = db.fetchall(
        """SELECT id, label FROM atoms
           WHERE tenant_id = ? AND space = ? AND entity_type = 'person'
             AND source_id IS NULL AND type IN ('concept', 'belief', 'goal')
           ORDER BY (sti + lti) DESC""",
        (tenant_id, space),
    )
    if not person_rows:
        return 0

    # Find episodes that have at least one mentions edge to a non-person atom
    # but NO mentions edge to any person atom
    orphaned = db.fetchall(
        """SELECT DISTINCT r.source_id AS episode_id
           FROM atoms r
           JOIN atoms ep ON ep.id = r.source_id
           WHERE r.type = 'relation' AND r.relation = 'mentions'
             AND r.tenant_id = ? AND r.space = ?
             AND ep.type = 'episode'
             AND NOT EXISTS (
                 SELECT 1 FROM atoms r2
                 JOIN atoms target ON target.id = r2.target_id
                 WHERE r2.type = 'relation' AND r2.relation = 'mentions'
                   AND r2.tenant_id = ? AND r2.space = ?
                   AND r2.source_id = r.source_id
                   AND target.entity_type = 'person'
             )""",
        (tenant_id, space, tenant_id, space),
    )

    if not orphaned:
        return 0

    single_person_id = person_rows[0]["id"] if len(person_rows) == 1 else None
    person_vecs: list[tuple[str, tuple]] = []
    if single_person_id is None:
        for row in person_rows:
            vec = _stored_embedding(db, row["id"])
            if vec is not None:
                person_vecs.append((row["id"], vec))

    healed = 0
    for row in orphaned:
        episode_id = row["episode_id"]

        if single_person_id is not None:
            person_id = single_person_id
        else:
            person_id = _best_person(_stored_embedding(db, episode_id), person_vecs)
            if person_id is None:
                continue

        # Link episode -> mentions -> person
        _create_relation(db, episode_id, person_id, "mentions", tenant_id, space)

        # Find concepts this episode mentions and create person -> associated -> concept
        concept_edges = db.fetchall(
            """SELECT target_id FROM atoms
               WHERE type = 'relation' AND relation = 'mentions'
                 AND source_id = ? AND tenant_id = ? AND space = ?""",
            (episode_id, tenant_id, space),
        )
        for edge in concept_edges:
            target_id = edge["target_id"]
            if target_id != person_id:
                _create_relation(
                    db, person_id, target_id, "associated",
                    tenant_id, space, confidence=0.2,
                )

        healed += 1

    return healed


def _stored_embedding(db, atom_id: str) -> tuple | None:
    """Read an atom's stored embedding from vec_atoms as a float tuple."""
    row = db.fetchone(
        "SELECT embedding FROM vec_atoms WHERE atom_id = ?", (atom_id,)
    )
    if row is None:
        return None
    blob = row["embedding"]
    return struct.unpack(f"{len(blob) // 4}f", blob)


def _cosine(a: tuple, b: tuple) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _best_person(
    episode_vec: tuple | None,
    person_vecs: list[tuple[str, tuple]],
    min_sim: float = 0.3,
) -> str | None:
    """Pick the person most similar to the episode, requiring cosine >= min_sim."""
    if episode_vec is None:
        return None
    best_id = None
    best_sim = None
    for person_id, vec in person_vecs:
        sim = _cosine(episode_vec, vec)
        if sim >= min_sim and (best_sim is None or sim > best_sim):
            best_id, best_sim = person_id, sim
    return best_id


def _create_relation(
    db, source_id: str, target_id: str, relation: str,
    tenant_id: str, space: str, confidence: float = 0.5,
) -> None:
    """Create a relation atom if one does not already exist."""
    existing = db.fetchone(
        """SELECT id FROM atoms WHERE type = 'relation' AND source_id = ? AND target_id = ?
           AND relation = ? AND tenant_id = ? AND space = ?""",
        (source_id, target_id, relation, tenant_id, space),
    )
    if existing:
        return
    link_id = str(uuid.uuid4())
    db.execute(
        """INSERT OR IGNORE INTO atoms
               (id, type, label, source_id, target_id, relation, tenant_id, space, probability, confidence)
           VALUES (?, 'relation', ?, ?, ?, ?, ?, ?, 0.5, ?)""",
        (link_id, relation, source_id, target_id, relation, tenant_id, space, confidence),
    )
