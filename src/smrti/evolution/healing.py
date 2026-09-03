"""Graph healing: attribute orphaned episodes to the person they are about.

An orphaned episode mentions concepts but no person, which happens when
extraction ran without a model that could resolve the speaker, or when the
name was written in a form the tagger missed. Healing files the episode under
a person so that expansion from the person can reach it.

It creates that one edge and nothing else. An earlier version also drew a
low-confidence ``associated`` edge from the person to every concept the
episode mentioned, "so LLM-extracted relations supersede"; those placeholder
edges made the person a hub joined to everything in the space, and a hub is
what expansion and ranking then found on every query.
"""
from __future__ import annotations

from smrti.core.atomspace import AtomSpace
from smrti.core.models import TruthValue


def heal_orphaned_episodes(tenant_id: str, space: str, db) -> int:
    """Find episodes that mention concepts but no person, and link them to a person.

    When the space contains exactly one person atom, every orphaned episode is
    attributed to it — the sole person is the speaker, which is the same
    assumption extraction makes. With several persons, an episode is
    attributed to the one person whose label or alias appears in its text;
    when none or more than one does, it is left alone rather than guessed.
    (An earlier rule compared the episode's embedding against the person's
    *name* embedding at cosine 0.3, which on a paraphrase model is close to
    chance.)

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
        """SELECT DISTINCT r.source_id AS episode_id, ep.content AS content, ep.label AS label
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
    names: list[tuple[str, list[str]]] = []
    if single_person_id is None:
        names = _person_names(person_rows, tenant_id, space, db)

    atomspace = AtomSpace(db, None)
    healed = 0
    for row in orphaned:
        if single_person_id is not None:
            person_id = single_person_id
        else:
            person_id = _named_person(row["content"] or row["label"] or "", names)
            if person_id is None:
                continue
        atomspace.link_atoms(
            row["episode_id"], person_id, "mentions", tenant_id, space,
            truth=TruthValue(probability=0.5, confidence=0.5),
        )
        healed += 1

    return healed


def _person_names(person_rows, tenant_id: str, space: str, db) -> list[tuple[str, list[str]]]:
    """Each person's id with the names it can be recognised by in text.

    Labels plus the aliases the resolver registered, minus pronouns: "I" and
    "my" are aliases of whoever spoke first, and as substrings they would
    match every episode in the space.
    """
    from smrti.extraction.ner import _PRONOUNS

    ids = [r["id"] for r in person_rows]
    ph = ",".join("?" * len(ids))
    alias_rows = db.fetchall(
        f"SELECT alias, atom_id FROM aliases WHERE tenant_id = ? AND space = ? AND atom_id IN ({ph})",
        (tenant_id, space, *ids),
    )
    aliases: dict[str, list[str]] = {}
    for r in alias_rows:
        alias = (r["alias"] or "").strip()
        if alias and alias.casefold() not in _PRONOUNS:
            aliases.setdefault(r["atom_id"], []).append(alias)
    out: list[tuple[str, list[str]]] = []
    for r in person_rows:
        label = (r["label"] or "").strip()
        candidates = [n for n in [label, *aliases.get(r["id"], [])] if n and n.casefold() not in _PRONOUNS]
        out.append((r["id"], [n.casefold() for n in candidates]))
    return out


def _named_person(text: str, names: list[tuple[str, list[str]]]) -> str | None:
    """The one person named in the text, or None when none or several are."""
    folded = text.casefold()
    matched = [pid for pid, forms in names if any(form in folded for form in forms)]
    return matched[0] if len(matched) == 1 else None
