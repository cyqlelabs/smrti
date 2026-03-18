"""Graph healing: detect and repair orphaned episodes during consolidation."""
from __future__ import annotations

import uuid


def heal_orphaned_episodes(tenant_id: str, space: str, db) -> int:
    """Find episodes that mention concepts but no person, and link them to the most salient person.

    For each orphaned episode:
      1. Create ``episode -> mentions -> person`` edge
      2. For each concept the episode mentions, create ``person -> associated -> concept``
         with low confidence (0.2) so LLM-extracted relations supersede

    Returns the number of healed episodes.
    """
    # Find the most salient person atom in this space
    person_row = db.fetchone(
        """SELECT id, label FROM atoms
           WHERE tenant_id = ? AND space = ? AND entity_type = 'person'
             AND source_id IS NULL AND type IN ('concept', 'belief', 'goal')
           ORDER BY (sti + lti) DESC LIMIT 1""",
        (tenant_id, space),
    )
    if not person_row:
        return 0

    person_id = person_row["id"]

    # Find episodes that have at least one mentions edge to a non-person atom
    # but NO mentions edge to any person atom
    orphaned = db.fetchall(
        """SELECT DISTINCT r.source_id AS episode_id
           FROM atoms r
           JOIN atoms ep ON ep.id = r.source_id
           WHERE r.type = 'relation' AND r.relation = 'mentions'
             AND r.tenant_id = ? AND r.space = ?
             AND ep.type = 'episode'
             AND r.source_id NOT IN (
                 SELECT r2.source_id FROM atoms r2
                 JOIN atoms target ON target.id = r2.target_id
                 WHERE r2.type = 'relation' AND r2.relation = 'mentions'
                   AND r2.tenant_id = ? AND r2.space = ?
                   AND target.entity_type = 'person'
             )""",
        (tenant_id, space, tenant_id, space),
    )

    if not orphaned:
        return 0

    healed = 0
    for row in orphaned:
        episode_id = row["episode_id"]

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
