"""What else goes when an episode is forgotten with ``cascade=True``.

Derivation is read from the ``mentions`` edges extraction draws from an
episode to each entity it names, since a claim edge records no episode of its
own. A claim edge between two entities goes when a forgotten episode mentions
both and no live episode does — so forgetting "Alice lives in Paris" takes the
``lives_in`` edge even though another episode mentions Alice. A concept goes
when no live episode mentions it and no surviving claim names it. When in
doubt an atom stays, as everywhere forgetting is decided: a claim whose object
extraction created without a mention is kept, since nothing says which
episode stated it. Only what lives in the write space is touched, and an
entity extraction later meets again is a new one — the resolver never matches
a forgotten atom.
"""
from __future__ import annotations

from smrti.core.models import STRUCTURAL_RELATIONS
from smrti.core.provenance import ATOM_FORGOTTEN, forgotten_sql


def derived_only_from(
    episode_ids: list[str], tenant_id: str, space: str, db
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The concepts and the claim edges *episode_ids* gave rise to that nothing
    still remembered supports, as two ``(id, label)`` lists.

    The episodes must already carry the forgotten stamp: it is what separates
    the mentions that still count from the ones that no longer do.
    """
    mentioned = _mentioned_entities(episode_ids, tenant_id, space, db)
    if not mentioned:
        return [], []
    entity_ids = {atom_id for atom_id, _ in mentioned}
    claims = _claims_on(list(entity_ids), tenant_id, space, db)
    live, gone = _mentioners(list(entity_ids | _endpoints(claims)), tenant_id, db)
    source = set(episode_ids)
    doomed = {
        c["id"]: c["label"]
        for c in claims
        if _named_by_both(gone, c) & source and not _named_by_both(live, c)
    }
    held = _endpoints([c for c in claims if c["id"] not in doomed])
    concepts = [
        (atom_id, label)
        for atom_id, label in mentioned
        if atom_id not in live and atom_id not in held
    ]
    return concepts, list(doomed.items())


def _named_by_both(mentioners: dict[str, set[str]], claim) -> set[str]:
    """The episodes among *mentioners* that mention both ends of *claim*."""
    return mentioners.get(claim["source_id"], set()) & mentioners.get(claim["target_id"], set())


def _endpoints(claims) -> set[str]:
    return {atom_id for c in claims for atom_id in (c["source_id"], c["target_id"]) if atom_id}


def _mentioned_entities(episode_ids, tenant_id, space, db) -> list[tuple[str, str]]:
    """The unforgotten concepts, and extracted beliefs and goals, the episodes
    mention in *space*. A belief or goal the caller wrote has content of its
    own; one extraction made from a mention does not."""
    ph = ", ".join("?" for _ in episode_ids)
    return [
        (row["id"], row["label"])
        for row in db.fetchall(
            f"""SELECT DISTINCT t.id, t.label FROM atoms m
                JOIN atoms t ON t.id = m.target_id
                WHERE m.type = 'relation' AND m.relation = 'mentions'
                  AND m.tenant_id = ? AND m.source_id IN ({ph})
                  AND t.tenant_id = ? AND t.space = ? AND NOT {forgotten_sql('t')}
                  AND (t.type = 'concept'
                       OR (t.type IN ('belief', 'goal') AND t.content IS NULL))""",
            (tenant_id, *episode_ids, tenant_id, space),
        )
    ]


def _claims_on(entity_ids, tenant_id, space, db) -> list:
    """The unforgotten claim edges in *space* with an endpoint among *entity_ids*."""
    ph = ", ".join("?" for _ in entity_ids)
    structural = ", ".join("?" for _ in STRUCTURAL_RELATIONS)
    return db.fetchall(
        f"""SELECT id, label, source_id, target_id FROM atoms
            WHERE type = 'relation' AND tenant_id = ? AND space = ?
              AND relation NOT IN ({structural}) AND NOT {ATOM_FORGOTTEN}
              AND (source_id IN ({ph}) OR target_id IN ({ph}))""",
        (tenant_id, space, *STRUCTURAL_RELATIONS, *entity_ids, *entity_ids),
    )


def _mentioners(entity_ids, tenant_id, db) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """The episodes that mention each entity, split into the live and the forgotten."""
    live: dict[str, set[str]] = {}
    gone: dict[str, set[str]] = {}
    ph = ", ".join("?" for _ in entity_ids)
    for row in db.fetchall(
        f"""SELECT m.source_id AS episode, m.target_id AS entity,
                   {forgotten_sql('ep')} AS forgotten
            FROM atoms m JOIN atoms ep ON ep.id = m.source_id
            WHERE m.type = 'relation' AND m.relation = 'mentions'
              AND m.tenant_id = ? AND m.target_id IN ({ph})""",
        (tenant_id, *entity_ids),
    ):
        side = gone if row["forgotten"] else live
        side.setdefault(row["entity"], set()).add(row["episode"])
    return live, gone
