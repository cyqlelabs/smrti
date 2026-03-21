"""Space_Culture promotion from bridge spaces, bridge threshold tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smrti import Smrti
from smrti.core.models import atom_from_row

from smrti_town.config import BRIDGE_THRESHOLD, CULTURE_CONFIDENCE_MIN

if TYPE_CHECKING:
    pass


def promote_bridges_to_culture(
    tenant_id: str,
    db_path: str,
    all_spaces: list[str],
) -> int:
    """Scan bridge spaces and promote high-confidence atoms to Space_Culture.

    Returns the number of atoms promoted.
    """
    culture_smrti = Smrti(
        db_path=db_path,
        personality="balanced",
        tenant_id=tenant_id,
        write_space="Space_Culture",
        read_spaces=["Space_Culture"],
    )

    promoted = 0
    bridge_spaces = [s for s in all_spaces if "_x_" in s]

    for bridge_space in bridge_spaces:
        rows = culture_smrti.db.fetchall(
            """
            SELECT * FROM atoms
            WHERE tenant_id = ? AND space = ?
            AND confidence >= ?
            AND type IN ('belief', 'concept')
            """,
            (tenant_id, bridge_space, CULTURE_CONFIDENCE_MIN),
        )
        for row in rows:
            atom = atom_from_row(row)
            # Check if already in culture
            existing = culture_smrti.db.fetchone(
                """
                SELECT id FROM atoms
                WHERE tenant_id = ? AND space = 'Space_Culture'
                AND label = ? AND type = ?
                """,
                (tenant_id, atom.label, atom.type.value),
            )
            if existing:
                continue

            culture_smrti.remember(
                content=atom.content or atom.label,
                type=atom.type.value,
                probability=atom.truth.probability,
                valence=atom.valence.valence if atom.valence else 0.0,
                metadata={"promoted_from": bridge_space},
            )
            promoted += 1

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
