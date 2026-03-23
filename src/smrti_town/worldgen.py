"""Opening sequence orchestrator — generates the founding council and initial town layout."""

from __future__ import annotations

import logging
import os

from smrti import Smrti

from smrti_town.config import (
    BUILDING_CATALOG,
    COUNCIL_ROLES,
    STARTING_TREASURY,
)
from smrti_town.economy import EconomyManager
from smrti_town.gridmap import GridMap, PlacedBuilding
from smrti_town.llm import LLMClient
from smrti_town.scenarios.fallback import create_fallback_council
from smrti_town.spatial import Place, TownTopology

log = logging.getLogger(__name__)


def _create_world_smrti(db_path: str, tenant_id: str) -> Smrti:
    """Create the World_Space Smrti instance."""
    return Smrti(
        db_path=db_path,
        personality="deterministic",
        tenant_id=tenant_id,
        write_space="World_Space",
    )


def _create_culture_smrti(db_path: str, tenant_id: str) -> Smrti:
    """Create the Space_Culture Smrti instance."""
    return Smrti(
        db_path=db_path,
        personality="balanced",
        tenant_id=tenant_id,
        write_space="Space_Culture",
    )


def _seed_world_space(world_smrti: Smrti, topology: TownTopology, council_specs: list[dict]) -> None:
    """Seed the World_Space with topology facts and founding council info."""
    # Record all places
    for name, place in topology.places.items():
        world_smrti.remember(
            f"{name} is a {place.place_type} place in town",
            type="concept",
            probability=1.0,
            valence=0.1,
            metadata={"place_name": name, "place_type": place.place_type, "building_key": place.building_key},
        )

    # Record council
    for cs in council_specs:
        world_smrti.remember(
            f"{cs['name']} serves as {COUNCIL_ROLES.get(cs['role'], {}).get('title', cs['role'])} of the town",
            type="concept",
            probability=1.0,
            valence=0.2,
            metadata={"citizen": cs["name"], "council_role": cs["role"]},
        )

    world_smrti.remember(
        "The town was founded by a council of five: a mayor, sheriff, superintendent, doctor, and treasurer.",
        type="episode",
        probability=1.0,
        valence=0.3,
    )


def _seed_culture(culture_smrti: Smrti) -> None:
    """Seed Space_Culture with founding values."""
    culture_smrti.remember(
        "The people of this town value hard work, community, and mutual respect.",
        type="belief",
        probability=0.9,
        valence=0.4,
    )
    culture_smrti.remember(
        "Every citizen deserves shelter, food, and a chance to contribute.",
        type="belief",
        probability=0.85,
        valence=0.3,
    )


async def generate_opening(
    llm_client: LLMClient,
    db_path: str,
    tenant_id: str,
    grid_x: int,
    grid_y: int,
    candidate_index: int | None = None,
    candidates: list[dict] | None = None,
) -> dict:
    """Orchestrate the opening sequence.

    Steps:
    1. If no candidates provided, generate mayor candidates via LLM.
    2. Select the mayor (by index or default to 0).
    3. Build fallback council for remaining roles.
    4. Create topology with Town Hall placed at (grid_x, grid_y).
    5. Seed World_Space and Space_Culture.

    Returns:
        {
            "candidates": list of all generated candidates,
            "mayor": selected mayor dict,
            "council_specs": list of council member specs,
            "citizen_specs": list of citizen specs,
            "topology": TownTopology,
            "gridmap": GridMap,
            "economy": EconomyManager,
            "world_smrti": Smrti,
            "culture_smrti": Smrti,
        }
    """
    # Step 1: Generate or use provided candidates
    if candidates is None:
        candidates = await llm_client.generate_mayor_candidates(
            theme=llm_client.settings.world_theme,
        )

    # Step 2: Select mayor
    idx = candidate_index if candidate_index is not None else 0
    idx = max(0, min(idx, len(candidates) - 1))
    mayor = candidates[idx]

    # Step 3: Build the full council using the fallback template for remaining roles,
    # but replace the mayor with the chosen candidate.
    fallback_council, fallback_citizens = create_fallback_council()

    council_specs = []
    citizen_specs = []

    # Mayor from chosen candidate
    mayor_spec = {
        "name": mayor["name"],
        "role": "mayor",
        "personality": mayor.get("personality", "balanced"),
        "governing_style": mayor.get("governing_style", "moderate"),
        "traits": mayor.get("traits", fallback_council[0]["traits"]),
    }
    council_specs.append(mayor_spec)
    citizen_specs.append({
        "name": mayor["name"],
        "age": 45,
        "personality": mayor.get("personality", "balanced"),
        "skills": {"leadership": 0.5, "literacy": 0.3, "commerce": 0.2},
        "bio": mayor.get("bio", "The elected mayor."),
        "council_role": "mayor",
        "traits": mayor.get("traits", fallback_council[0]["traits"]),
    })

    # Remaining roles from fallback
    for fc_spec, fc_citizen in zip(fallback_council[1:], fallback_citizens[1:]):
        council_specs.append(fc_spec)
        citizen_specs.append(fc_citizen)

    # Step 4: Create topology and gridmap
    gridmap = GridMap()
    topology = TownTopology()

    # Place Town Hall
    hall_name = "Town Hall"
    pb = gridmap.place("town_hall", grid_x, grid_y, place_name=hall_name)
    hall_place = Place(
        name=hall_name,
        place_type="civic",
        building_key="town_hall",
        grid_x=grid_x,
        grid_y=grid_y,
    )
    topology.add_place(hall_place)

    # Add a Town Square (outdoor) adjacent to Town Hall
    sq_name = "Town Square"
    sq_place = Place(
        name=sq_name,
        place_type="outdoor",
        building_key=None,
        is_outdoor=True,
        grid_x=grid_x + 2,
        grid_y=grid_y,
    )
    topology.add_place(sq_place)
    topology.connect(hall_name, sq_name)

    # Add surrounding fields
    fields_name = "Surrounding Fields"
    fields_place = Place(
        name=fields_name,
        place_type="outdoor",
        building_key=None,
        is_outdoor=True,
        grid_x=grid_x - 3,
        grid_y=grid_y,
    )
    topology.add_place(fields_place)
    topology.connect(sq_name, fields_name)

    # All council members start at Town Hall
    for cs in citizen_specs:
        hall_place.add_occupant(cs["name"])

    # Step 5: Seed smrti spaces
    world_smrti = _create_world_smrti(db_path, tenant_id)
    culture_smrti = _create_culture_smrti(db_path, tenant_id)

    _seed_world_space(world_smrti, topology, council_specs)
    _seed_culture(culture_smrti)

    # Step 6: Economy
    economy = EconomyManager(treasury=STARTING_TREASURY)
    bdef = BUILDING_CATALOG.get("town_hall")
    if bdef:
        economy.register_building(hall_name, bdef)
    for cs in citizen_specs:
        economy.register_citizen(cs["name"])

    return {
        "candidates": candidates,
        "mayor": mayor,
        "council_specs": council_specs,
        "citizen_specs": citizen_specs,
        "topology": topology,
        "gridmap": gridmap,
        "economy": economy,
        "world_smrti": world_smrti,
        "culture_smrti": culture_smrti,
    }
