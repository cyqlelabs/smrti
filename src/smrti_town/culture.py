"""Bridge discovery and culture promotion for smrti-town.

Periodically discovers shared knowledge between agent memory spaces via
smrti's bridge-space machinery, then promotes high-confidence bridge atoms
up to ``Space_Culture`` where they become shared town beliefs.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


async def run_bridge_discovery(
    agents: list[Any],
    topology: Any,
    bridge_threshold: float = 0.3,
) -> int:
    """Run bridge space discovery between all pairs of agent smrti instances.

    For each pair of agents that share a location (or have interacted recently),
    compute the overlap between their write spaces using smrti's space_overlap.
    If Jaccard >= *bridge_threshold*, materialize the bridge space.

    Parameters
    ----------
    agents:
        Agent objects with ``.name`` (str), ``.smrti`` (Smrti), ``.location`` (str|None).
    topology:
        TownTopology with ``.places`` dict for occupant checks.
    bridge_threshold:
        Minimum Jaccard index to create a bridge space.

    Returns
    -------
    int
        Number of bridge spaces created or updated.
    """
    bridges_created = 0
    alive = [a for a in agents if getattr(a, "alive", True) and getattr(a, "smrti", None)]

    if len(alive) < 2:
        return 0

    # Group agents by location for proximity-based bridge discovery.
    by_location: dict[str, list[Any]] = {}
    for a in alive:
        loc = getattr(a, "location", None)
        if loc:
            by_location.setdefault(loc, []).append(a)

    # Also consider agents with place-based smrti spaces.
    checked_pairs: set[tuple[str, str]] = set()

    for loc, group in by_location.items():
        for i, a1 in enumerate(group):
            for a2 in group[i + 1:]:
                pair_key = tuple(sorted([a1.name, a2.name]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                try:
                    result = a1.smrti.space_overlap(
                        other_space=a2.smrti.write_space,
                        tenant_id=a1.smrti.tenant_id,
                    )
                    if result and result.get("jaccard", 0) >= bridge_threshold:
                        bridge_result = a1.smrti.materialize_bridge(
                            other_space=a2.smrti.write_space,
                            tenant_id=a1.smrti.tenant_id,
                        )
                        if bridge_result:
                            bridges_created += 1
                            log.debug(
                                "Bridge created between %s and %s (jaccard=%.2f)",
                                a1.name, a2.name, result.get("jaccard", 0),
                            )
                except Exception:
                    log.debug(
                        "Bridge discovery failed for %s <-> %s",
                        a1.name, a2.name,
                        exc_info=True,
                    )

    return bridges_created


async def promote_bridges_to_culture(
    agents: list[Any],
    culture_smrti: Any,
    confidence_min: float = 0.5,
) -> int:
    """Promote high-confidence atoms from bridge spaces to Space_Culture.

    Scans all bridge spaces (named ``{a}_x_{b}``) in the database and copies
    atoms with confidence >= *confidence_min* into the culture space as shared
    beliefs.

    Parameters
    ----------
    agents:
        Agent objects with ``.smrti`` (Smrti).
    culture_smrti:
        The Smrti instance for ``Space_Culture``.
    confidence_min:
        Minimum confidence threshold for promotion.

    Returns
    -------
    int
        Number of atoms promoted.
    """
    promoted = 0

    if not agents or culture_smrti is None:
        return 0

    # Use the first agent's smrti to query for bridge spaces.
    sample_smrti = None
    for a in agents:
        s = getattr(a, "smrti", None)
        if s:
            sample_smrti = s
            break

    if sample_smrti is None:
        return 0

    try:
        spaces = sample_smrti.list_spaces()
    except Exception:
        log.debug("Failed to list spaces for bridge promotion", exc_info=True)
        return 0

    bridge_spaces = [s for s in spaces if "_x_" in s]

    for space_name in bridge_spaces:
        try:
            # Recall high-confidence atoms from the bridge space.
            results = sample_smrti.recall(
                query="shared knowledge beliefs values",
                top_k=20,
                min_confidence=confidence_min,
                read_spaces=[space_name],
            )

            for r in results:
                content = getattr(r, "content", "") or getattr(r, "label", "")
                if not content:
                    continue

                truth = getattr(r, "truth", None)
                conf = getattr(truth, "confidence", 0) if truth else 0
                if conf < confidence_min:
                    continue

                valence_obj = getattr(r, "valence", None)
                val = getattr(valence_obj, "valence", 0.0) if valence_obj else 0.0
                prob = getattr(truth, "probability", 0.8) if truth else 0.8

                culture_smrti.remember(
                    content,
                    type="belief",
                    probability=prob,
                    valence=val,
                    metadata={"source_bridge": space_name},
                )
                promoted += 1

        except Exception:
            log.debug("Failed to promote from bridge space %s", space_name, exc_info=True)

    if promoted:
        log.info("Promoted %d atoms to Space_Culture from %d bridge spaces", promoted, len(bridge_spaces))

    return promoted
