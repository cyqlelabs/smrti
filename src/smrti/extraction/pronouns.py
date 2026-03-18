"""Pronoun detection and merge logic for entity resolution."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smrti.extraction.ner import NERProvider


def merge_pronoun_entities_in_batch(
    entities: list[dict],
    ner: NERProvider,
) -> list[dict]:
    """Merge pronoun-like person entities into named persons within a batch.

    Rules:
    - 1 named person + 1+ pronoun persons → merge pronouns into the named person
    - 0 named persons → remove all pronoun persons (no orphan atoms)
    - 2+ named persons → leave pronoun persons as-is (ambiguous)
    - Non-person entities are never touched
    """
    persons = []
    non_persons = []
    for ent in entities:
        if ent.get("type") == "person" or ent.get("type") == "pronoun":
            persons.append(ent)
        else:
            non_persons.append(ent)

    if not persons:
        return entities

    pronoun_persons = []
    named_persons = []
    for p in persons:
        if p.get("type") == "pronoun" or ner.classify_pronoun(p.get("name", "")):
            pronoun_persons.append(p)
        else:
            named_persons.append(p)

    if not pronoun_persons:
        return entities

    if len(named_persons) == 0:
        # No named persons — remove all pronoun persons
        return non_persons + named_persons

    if len(named_persons) == 1:
        # Merge pronoun aliases into the single named person
        target = named_persons[0]
        existing_aliases = set(a.lower() for a in target.get("aliases", []))
        merged_aliases = list(target.get("aliases", []))
        for pp in pronoun_persons:
            pname = pp.get("name", "")
            if pname and pname.lower() not in existing_aliases:
                merged_aliases.append(pname)
                existing_aliases.add(pname.lower())
            for alias in pp.get("aliases", []):
                if alias and alias.lower() not in existing_aliases:
                    merged_aliases.append(alias)
                    existing_aliases.add(alias.lower())
        target["aliases"] = merged_aliases
        return non_persons + named_persons

    # 2+ named persons → ambiguous, leave pronoun persons as-is
    return non_persons + named_persons + pronoun_persons


def merge_pronoun_into_named(
    pronoun_atom_id: str,
    named_atom_id: str,
    db,
    tenant_id: str,
    write_space: str,
) -> None:
    """Merge a pronoun atom into a named atom: reassign edges, transfer aliases, delete."""
    # Get pronoun label to register as alias
    row = db.fetchone("SELECT label FROM atoms WHERE id = ?", (pronoun_atom_id,))
    pronoun_label = row["label"] if row else None

    # Reassign relation edges (source_id and target_id)
    db.execute(
        "UPDATE atoms SET source_id = ? WHERE source_id = ? AND tenant_id = ?",
        (named_atom_id, pronoun_atom_id, tenant_id),
    )
    db.execute(
        "UPDATE atoms SET target_id = ? WHERE target_id = ? AND tenant_id = ?",
        (named_atom_id, pronoun_atom_id, tenant_id),
    )

    # Transfer aliases
    db.execute(
        "UPDATE aliases SET atom_id = ? WHERE atom_id = ? AND tenant_id = ?",
        (named_atom_id, pronoun_atom_id, tenant_id),
    )

    # Register pronoun label as alias of named entity
    if pronoun_label:
        db.execute(
            "INSERT OR IGNORE INTO aliases (alias, atom_id, tenant_id, space) VALUES (?, ?, ?, ?)",
            (pronoun_label, named_atom_id, tenant_id, write_space),
        )

    # Delete self-referencing edges (e.g. "I→is→I" became "Elara→is→Elara")
    db.execute(
        "DELETE FROM atoms WHERE type = 'relation' AND source_id = ? AND target_id = ? AND tenant_id = ?",
        (named_atom_id, named_atom_id, tenant_id),
    )

    # Delete duplicate relation edges (same source, target, relation triple)
    db.execute(
        """DELETE FROM atoms WHERE id IN (
            SELECT a2.id FROM atoms a1
            JOIN atoms a2 ON a1.source_id = a2.source_id
                AND a1.target_id = a2.target_id
                AND a1.relation = a2.relation
                AND a1.tenant_id = a2.tenant_id
                AND a1.id < a2.id
            WHERE a1.type = 'relation' AND a2.type = 'relation'
                AND a1.tenant_id = ?
                AND a1.source_id = ?
        )""",
        (tenant_id, named_atom_id),
    )

    # Delete pronoun atom and its vector entry
    db.execute("DELETE FROM vec_atoms WHERE atom_id = ?", (pronoun_atom_id,))
    db.execute("DELETE FROM atoms WHERE id = ?", (pronoun_atom_id,))


def find_and_merge_pronoun_atoms(
    named_atom_id: str,
    episode_id: str,
    db,
    ner: NERProvider,
    tenant_id: str,
    write_space: str,
) -> None:
    """Find pronoun atoms co-mentioned with named_atom_id in the same episode and merge them."""
    # Find person-type atoms that share a mentions edge with the same episode
    rows = db.fetchall(
        """SELECT DISTINCT a.id, a.label FROM atoms a
           JOIN atoms m ON m.type = 'relation' AND m.relation = 'mentions'
               AND m.source_id = ? AND m.target_id = a.id AND m.tenant_id = ?
           WHERE a.entity_type = 'person' AND a.tenant_id = ?
               AND a.id != ?""",
        (episode_id, tenant_id, tenant_id, named_atom_id),
    )
    for row in rows:
        if ner.classify_pronoun(row["label"]):
            merge_pronoun_into_named(row["id"], named_atom_id, db, tenant_id, write_space)
