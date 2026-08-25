"""Pronoun detection and merge logic for entity resolution."""
from __future__ import annotations

from typing import TYPE_CHECKING

from smrti.core.db import fts_delete

if TYPE_CHECKING:
    from smrti.extraction.ner import NERProvider


def resolve_pronouns_via_aliases(
    pronoun_persons: list[dict],
    db,
    tenant_id: str,
    spaces: list[str],
) -> list[dict]:
    """Try to resolve pronoun entities to existing named persons via alias table.

    For each pronoun person, checks if its name is a known alias pointing to
    a person atom. If so, replaces the pronoun entity with the resolved person.
    Returns the list of successfully resolved entities (pronouns become named).
    """
    from .aliases import AliasManager
    aliases = AliasManager(db)
    resolved = []
    for pp in pronoun_persons:
        pname = (pp.get("name") or "").strip()
        if not pname:
            continue
        atom_id = aliases.lookup(pname, tenant_id, spaces)
        if not atom_id:
            continue
        # Verify the target atom is a person
        row = db.fetchone(
            "SELECT label, entity_type FROM atoms WHERE id = ? AND entity_type = 'person'",
            (atom_id,),
        )
        if row:
            resolved.append({
                "name": row["label"],
                "type": "person",
                "aliases": pp.get("aliases", []) + [pname] if pname.lower() != row["label"].lower() else pp.get("aliases", []),
            })
    return resolved


def merge_pronoun_entities_in_batch(
    entities: list[dict],
    ner: NERProvider,
    db=None,
    tenant_id: str = "",
    spaces: list[str] | None = None,
) -> list[dict]:
    """Merge pronoun-like person entities into named persons within a batch.

    Rules:
    - 1 named person + 1+ pronoun persons → merge pronouns into the named person
    - 0 named persons in batch → try alias table to resolve pronouns to existing persons
    - 0 named persons after alias lookup → remove all pronoun persons (no orphan atoms)
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

    # Collect names that GLiNER explicitly typed as pronoun — if the same name
    # also appears as type=person, it's NER noise and should be treated as a pronoun.
    explicit_pronoun_names = {
        (p.get("name") or "").strip().lower()
        for p in persons
        if p.get("type") == "pronoun"
    }

    pronoun_persons = []
    named_persons = []
    for p in persons:
        name = (p.get("name") or "").strip()
        if (
            p.get("type") == "pronoun"
            or ner.classify_pronoun(name)
            or name.lower() in explicit_pronoun_names
        ):
            pronoun_persons.append(p)
        else:
            named_persons.append(p)

    if not pronoun_persons:
        return entities

    if len(named_persons) == 0:
        # No named persons in batch — check alias table for existing person atoms
        if db is not None and tenant_id and spaces:
            resolved = resolve_pronouns_via_aliases(pronoun_persons, db, tenant_id, spaces)
            if resolved:
                # Deduplicate resolved persons by name
                seen = set()
                for r in resolved:
                    if r["name"].lower() not in seen:
                        named_persons.append(r)
                        seen.add(r["name"].lower())
                # If we resolved to exactly 1 person, merge remaining pronouns into it
                if len(named_persons) == 1:
                    return _merge_pronouns_into_target(named_persons[0], pronoun_persons, non_persons, named_persons)
                elif len(named_persons) > 1:
                    # Multiple resolved persons — ambiguous
                    return non_persons + named_persons + pronoun_persons
        # Still no named persons — remove all pronoun persons
        return non_persons

    if len(named_persons) == 1:
        return _merge_pronouns_into_target(named_persons[0], pronoun_persons, non_persons, named_persons)

    # 2+ named persons → ambiguous, leave pronoun persons as-is
    return non_persons + named_persons + pronoun_persons


def _merge_pronouns_into_target(
    target: dict,
    pronoun_persons: list[dict],
    non_persons: list[dict],
    named_persons: list[dict],
) -> list[dict]:
    """Merge pronoun aliases into a single named person target."""
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


def merge_pronoun_into_named(
    pronoun_atom_id: str,
    named_atom_id: str,
    db,
    tenant_id: str,
    write_space: str,
) -> None:
    """Merge a pronoun atom into a named atom: reassign edges, transfer aliases, delete.

    The pronoun label itself is never registered as an alias — "he"/"I" would
    become a permanent alias of whoever merged first, poisoning tier-1
    resolution for every later speaker.
    """
    # Reassign relation edges (source_id and target_id) within write_space only
    db.execute(
        "UPDATE atoms SET source_id = ? WHERE source_id = ? AND tenant_id = ? AND space = ?",
        (named_atom_id, pronoun_atom_id, tenant_id, write_space),
    )
    db.execute(
        "UPDATE atoms SET target_id = ? WHERE target_id = ? AND tenant_id = ? AND space = ?",
        (named_atom_id, pronoun_atom_id, tenant_id, write_space),
    )

    # Transfer aliases within write_space only
    db.execute(
        "UPDATE aliases SET atom_id = ? WHERE atom_id = ? AND tenant_id = ? AND space = ?",
        (named_atom_id, pronoun_atom_id, tenant_id, write_space),
    )

    # Delete self-referencing edges (e.g. "I→is→I" became "Elara→is→Elara")
    db.execute(
        "DELETE FROM atoms WHERE type = 'relation' AND source_id = ? AND target_id = ? AND tenant_id = ? AND space = ?",
        (named_atom_id, named_atom_id, tenant_id, write_space),
    )

    # Delete duplicate relation edges (same source, target, relation triple)
    db.execute(
        """DELETE FROM atoms WHERE id IN (
            SELECT a2.id FROM atoms a1
            JOIN atoms a2 ON a1.source_id = a2.source_id
                AND a1.target_id = a2.target_id
                AND a1.relation = a2.relation
                AND a1.tenant_id = a2.tenant_id
                AND a1.space = a2.space
                AND a1.id < a2.id
            WHERE a1.type = 'relation' AND a2.type = 'relation'
                AND a1.tenant_id = ?
                AND a1.space = ?
                AND a1.source_id = ?
        )""",
        (tenant_id, write_space, named_atom_id),
    )

    # Delete pronoun atom and its vector entry — scoped to tenant_id + space
    # like the edge reassignment above, so atoms in other spaces are untouched
    row = db.fetchone(
        "SELECT 1 FROM atoms WHERE id = ? AND tenant_id = ? AND space = ?",
        (pronoun_atom_id, tenant_id, write_space),
    )
    if row:
        db.execute("DELETE FROM vec_atoms WHERE atom_id = ?", (pronoun_atom_id,))
        db.execute_batch(fts_delete(db, [pronoun_atom_id]))
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
               AND m.source_id = ? AND m.target_id = a.id AND m.tenant_id = ? AND m.space = ?
           WHERE a.entity_type = 'person' AND a.tenant_id = ? AND a.space = ?
               AND a.id != ?""",
        (episode_id, tenant_id, write_space, tenant_id, write_space, named_atom_id),
    )
    for row in rows:
        if ner.classify_pronoun(row["label"]):
            merge_pronoun_into_named(row["id"], named_atom_id, db, tenant_id, write_space)
