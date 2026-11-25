"""What else goes when an episode is forgotten with ``cascade=True``.

A claim edge records the episodes that asserted it (``$.episodes``, written by
extraction), and it goes when every one of them is forgotten. Co-mention is
not provenance: an episode that mentions Alice and tea did not necessarily
state that Alice likes tea, and reading it as if it had deleted another
episode's fact. A claim with no record — written before extraction kept one —
is kept, as everywhere forgetting is decided: when in doubt an atom stays.

A concept goes when no live episode mentions it and no surviving claim names
it. Only what lives in the write space is touched, and an entity extraction
later meets again is a new one — the resolver never matches a forgotten atom.
"""
from __future__ import annotations

import json

from smrti.core.models import STRUCTURAL_RELATIONS
from smrti.core.provenance import ATOM_FORGOTTEN, CLAIM_EPISODES, UNRECORDED_ASSERTION, forgotten_sql

# Ids per statement: SQLite before 3.32 allows 999 bound variables.
_CHUNK = 400


def _chunks(ids: list[str]):
    for start in range(0, len(ids), _CHUNK):
        yield ids[start : start + _CHUNK]


def derived_only_from(
    episode_ids: list[str], tenant_id: str, space: str, db
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The concepts and the claim edges *episode_ids* gave rise to that nothing
    still remembered supports, as two ``(id, label)`` lists.

    The episodes are treated as forgotten whether or not they carry the stamp
    yet, so the caller can decide everything first and then stamp and delete
    in one transaction.
    """
    source = set(episode_ids)
    mentioned = _mentioned_entities(episode_ids, tenant_id, space, db)
    if not mentioned:
        return [], []
    entity_ids = {atom_id for atom_id, _ in mentioned}
    claims = _claims_on(list(entity_ids), tenant_id, space, db)
    supporters = {c["id"]: _asserted_by(c) for c in claims}
    live_supporters = _live(
        [e for episodes in supporters.values() for e in episodes if e not in source],
        tenant_id, db,
    )

    def unsupported(episodes: set[str]) -> bool:
        # A forgotten episode asserted it, no live one did, and nothing
        # unrecorded might have.
        return (
            bool(episodes & source)
            and UNRECORDED_ASSERTION not in episodes
            and not (episodes - source) & live_supporters
        )

    doomed = {c["id"]: c["label"] for c in claims if unsupported(supporters[c["id"]])}
    held = {
        atom_id
        for c in claims if c["id"] not in doomed
        for atom_id in (c["source_id"], c["target_id"]) if atom_id
    }
    mentioned_live = _mentioned_by_live(list(entity_ids), source, tenant_id, db)
    concepts = [
        (atom_id, label)
        for atom_id, label in mentioned
        if atom_id not in mentioned_live and atom_id not in held
    ]
    return concepts, list(doomed.items())


def _asserted_by(claim) -> set[str]:
    """The episodes recorded as asserting *claim*; empty when none were."""
    try:
        meta = json.loads(claim["metadata"] or "{}")
    except (TypeError, ValueError):
        return set()
    episodes = meta.get(CLAIM_EPISODES) if isinstance(meta, dict) else None
    return {e for e in episodes if isinstance(e, str)} if isinstance(episodes, list) else set()


def _mentioned_entities(episode_ids, tenant_id, space, db) -> list[tuple[str, str]]:
    """The unforgotten concepts, and extracted beliefs and goals, the episodes
    mention in *space*. A belief or goal the caller wrote has content of its
    own; one extraction made from a mention does not."""
    found: dict[str, str] = {}
    for chunk in _chunks(episode_ids):
        ph = ", ".join("?" for _ in chunk)
        for row in db.fetchall(
            f"""SELECT DISTINCT t.id, t.label FROM atoms m
                JOIN atoms t ON t.id = m.target_id
                WHERE m.type = 'relation' AND m.relation = 'mentions'
                  AND m.tenant_id = ? AND m.source_id IN ({ph})
                  AND t.tenant_id = ? AND t.space = ? AND NOT {forgotten_sql('t')}
                  AND (t.type = 'concept'
                       OR (t.type IN ('belief', 'goal') AND t.content IS NULL))""",
            (tenant_id, *chunk, tenant_id, space),
        ):
            found[row["id"]] = row["label"]
    return list(found.items())


def _claims_on(entity_ids, tenant_id, space, db) -> list:
    """The unforgotten claim edges in *space* with an endpoint among *entity_ids*."""
    structural = ", ".join("?" for _ in STRUCTURAL_RELATIONS)
    claims: dict[str, object] = {}
    for chunk in _chunks(entity_ids):
        ph = ", ".join("?" for _ in chunk)
        for column in ("source_id", "target_id"):
            for row in db.fetchall(
                f"""SELECT id, label, source_id, target_id, metadata FROM atoms
                    WHERE type = 'relation' AND tenant_id = ? AND space = ?
                      AND relation NOT IN ({structural}) AND NOT {ATOM_FORGOTTEN}
                      AND {column} IN ({ph})""",
                (tenant_id, space, *STRUCTURAL_RELATIONS, *chunk),
            ):
                claims[row["id"]] = row
    return list(claims.values())


def _live(episode_ids, tenant_id, db) -> set[str]:
    """Those of *episode_ids* that exist and are not forgotten."""
    live: set[str] = set()
    for chunk in _chunks(list(set(episode_ids))):
        ph = ", ".join("?" for _ in chunk)
        live.update(
            row["id"]
            for row in db.fetchall(
                f"SELECT id FROM atoms WHERE id IN ({ph}) AND tenant_id = ? AND NOT {ATOM_FORGOTTEN}",
                (*chunk, tenant_id),
            )
        )
    return live


def _mentioned_by_live(entity_ids, source, tenant_id, db) -> set[str]:
    """The entities among *entity_ids* that a live episode outside *source* mentions."""
    mentioned: set[str] = set()
    for chunk in _chunks(entity_ids):
        ph = ", ".join("?" for _ in chunk)
        for row in db.fetchall(
            f"""SELECT m.source_id AS episode, m.target_id AS entity
                FROM atoms m JOIN atoms ep ON ep.id = m.source_id
                WHERE m.type = 'relation' AND m.relation = 'mentions'
                  AND m.tenant_id = ? AND m.target_id IN ({ph})
                  AND NOT {forgotten_sql('ep')}""",
            (tenant_id, *chunk),
        ):
            if row["episode"] not in source:
                mentioned.add(row["entity"])
    return mentioned
