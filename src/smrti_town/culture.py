"""Bridge discovery and culture promotion for smrti-town.

What two citizens remember alike — an evening both were at, a storm both
stood in — is materialised as a bridge space, and what a bridge holds
firmly is copied up to ``Space_Culture``, which every citizen reads. That
is how a shared experience becomes the town's memory, and how a place the
town has soured on stays soured for a newcomer who was never there.
"""

from __future__ import annotations

import logging
from typing import Any

from smrti.core.models import atom_from_row

log = logging.getLogger(__name__)


def run_bridge_discovery(agents: list[Any], bridge_threshold: float = 0.3) -> int:
    """Materialise a bridge space for every pair of citizens sharing a place
    whose memories overlap by at least *bridge_threshold* (Jaccard).

    Returns the number of bridge atoms written. Pairs are limited to
    citizens in the same place because the overlap is an all-pairs cosine
    in pure Python — seconds per pair on a few hundred atoms.
    """
    by_location: dict[str, list[Any]] = {}
    for a in agents:
        if getattr(a, "alive", True) and getattr(a, "smrti", None) and getattr(a, "location", None):
            by_location.setdefault(a.location, []).append(a)
    written = 0
    for group in by_location.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                try:
                    written += a.smrti.materialize_bridge(b.smrti.write_space, min_jaccard=bridge_threshold)
                except Exception:
                    log.debug("Bridge discovery failed for %s <-> %s", a.name, b.name, exc_info=True)
    return written


def promote_bridges_to_culture(agents: list[Any], culture_smrti: Any, confidence_min: float = 0.5) -> int:
    """Copy every bridge atom held at *confidence_min* or above into
    ``Space_Culture``, once per distinct text. Returns the number promoted.

    A bridge atom merges two citizens' truth values, so one shared memory
    already clears 0.5; the culture space is what every citizen reads, so
    the copy carries the tone the memory was written with.
    """
    sample = next((a.smrti for a in agents if getattr(a, "smrti", None)), None)
    if sample is None or culture_smrti is None:
        return 0
    promoted = 0
    for space in sample.list_spaces():
        if "_x_" not in space:
            continue
        rows = sample.db.fetchall(
            "SELECT * FROM atoms WHERE tenant_id = ? AND space = ? AND type != 'relation' AND confidence >= ?",
            (sample.tenant_id, space, confidence_min),
        )
        for row in rows:
            atom = atom_from_row(row)
            content = atom.content or atom.label
            if not content:
                continue
            known = culture_smrti.db.fetchone(
                "SELECT 1 FROM atoms WHERE tenant_id = ? AND space = ? AND content = ?",
                (culture_smrti.tenant_id, culture_smrti.write_space, content),
            )
            if known:
                continue
            culture_smrti.remember(
                content,
                type=atom.type.value,
                probability=atom.truth.probability,
                valence=atom.valence.own,
                metadata={"source_bridge": space, "source_atom": atom.id},
            )
            promoted += 1
    if promoted:
        log.info("Promoted %d shared memories to Space_Culture", promoted)
    return promoted


def run_culture_pass(agents: list[Any], culture_smrti: Any) -> tuple[int, int]:
    """One pass of the town's culture: bridges between citizens who share a
    place, then promotion of what the bridges hold. Returns both counts."""
    return run_bridge_discovery(agents), promote_bridges_to_culture(agents, culture_smrti)
