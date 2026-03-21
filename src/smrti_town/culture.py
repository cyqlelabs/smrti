"""Space_Culture promotion from bridge spaces, bridge threshold tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smrti import Smrti

from smrti_town.config import BRIDGE_THRESHOLD, CULTURE_CONFIDENCE_MIN

if TYPE_CHECKING:
    pass


def promote_bridges_to_culture(
    tenant_id: str,
    db_path: str,
    all_spaces: list[str],
) -> int:
    """Scan bridge spaces and promote high-confidence atoms to Space_Culture.

    Uses the Smrti recall API to read bridge space atoms and the culture
    space for deduplication — no direct DB queries.

    Returns the number of atoms promoted.
    """
    culture_smrti = Smrti(
        db_path=db_path,
        personality="balanced",
        tenant_id=tenant_id,
        write_space="Space_Culture",
        read_spaces=["Space_Culture"],
    )

    # Pre-fetch existing culture labels for dedup
    existing_labels: set[tuple[str, str]] = set()
    try:
        culture_results = culture_smrti.recall(query="*", top_k=200)
        for r in culture_results:
            existing_labels.add((r.atom.label, r.atom.type.value))
    except Exception:
        pass

    promoted = 0
    bridge_spaces = [s for s in all_spaces if "_x_" in s]

    for bridge_space in bridge_spaces:
        # Read bridge space atoms via a dedicated Smrti reader
        bridge_reader = Smrti(
            db_path=db_path,
            personality="balanced",
            tenant_id=tenant_id,
            write_space=bridge_space,
            read_spaces=[bridge_space],
        )
        try:
            results = bridge_reader.recall(query="*", top_k=100)
        except Exception:
            bridge_reader.close()
            continue

        for r in results:
            atom = r.atom
            # Filter: only beliefs/concepts with sufficient confidence
            if atom.type.value not in ("belief", "concept"):
                continue
            if atom.truth.confidence < CULTURE_CONFIDENCE_MIN:
                continue
            # Dedup: skip if already in culture
            if (atom.label, atom.type.value) in existing_labels:
                continue

            culture_smrti.remember(
                content=atom.content or atom.label,
                type=atom.type.value,
                probability=atom.truth.probability,
                valence=atom.valence.valence if atom.valence else 0.0,
                metadata={"promoted_from": bridge_space},
            )
            existing_labels.add((atom.label, atom.type.value))
            promoted += 1

        bridge_reader.close()

    culture_smrti.close()
    return promoted


def run_bridge_discovery(
    tenant_id: str,
    db_path: str,
    agent_spaces: list[str],
) -> int:
    """Run bridge discovery between all pairs of agent spaces.

    Uses a higher threshold (0.3) than default to prevent base-ontology
    explosion. Returns total bridges created.
    """
    total_bridges = 0
    processed: set[tuple[str, str]] = set()

    for i, space_a in enumerate(agent_spaces):
        for space_b in agent_spaces[i + 1:]:
            pair = tuple(sorted([space_a, space_b]))
            if pair in processed:
                continue
            processed.add(pair)

            try:
                smrti_a = Smrti(
                    db_path=db_path,
                    personality="balanced",
                    tenant_id=tenant_id,
                    write_space=space_a,
                    read_spaces=[space_a],
                )
                bridges = smrti_a.materialize_bridge(
                    space_b,
                    threshold=BRIDGE_THRESHOLD,
                    min_jaccard=0.1,
                )
                total_bridges += bridges
                smrti_a.close()
            except Exception:
                pass

    return total_bridges
