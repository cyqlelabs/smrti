"""GridMap — isometric grid for building placement."""

from __future__ import annotations

from dataclasses import dataclass, field

from smrti_town.config import BUILDING_CATALOG, GRID_HEIGHT, GRID_WIDTH


@dataclass
class PlacedBuilding:
    building_key: str
    grid_x: int
    grid_y: int
    place_name: str
    sprite_variant: int = 0

    def to_dict(self) -> dict:
        bdef = BUILDING_CATALOG.get(self.building_key)
        d = {
            "building_key": self.building_key,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "place_name": self.place_name,
            "sprite_variant": self.sprite_variant,
            "sprite_key": bdef.sprite_key if bdef else self.building_key,
            # Static catalog metadata
            "description": bdef.description if bdef else "",
            "category": bdef.category if bdef else "",
            "cost": bdef.cost if bdef else 0,
            "maintenance": bdef.maintenance if bdef else 0,
            "capacity": bdef.capacity if bdef else 0,
            "revenue_per_hour": bdef.revenue_per_hour if bdef else 0,
            "staff_required": bdef.staff_required if bdef else 0,
            "provides_food": bool(bdef and bdef.provides_food),
            "provides_housing": bool(bdef and bdef.provides_housing),
            "provides_goods": bool(bdef and getattr(bdef, "provides_goods", False)),
            # Live stats — populated by server before broadcast
            "citizens_here": 0,
            "citizens_home": 0,
            "citizens_work": 0,
            "transactions": 0,
            "revenue": 0,
        }
        return d


class GridMap:
    """2D grid tracking which cells are occupied by buildings."""

    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> None:
        self.width = width
        self.height = height
        # (x, y) -> building_key or None.  Only occupied cells are stored.
        self.cells: dict[tuple[int, int], str | None] = {}
        self.buildings: list[PlacedBuilding] = []
        self._by_pos: dict[tuple[int, int], PlacedBuilding] = {}

    # ── footprint helpers ─────────────────────────────────────────────
    @staticmethod
    def _footprint(building_key: str, gx: int, gy: int) -> list[tuple[int, int]]:
        """Return the list of grid cells a building occupies.

        Most buildings are 1x1; large buildings could be bigger in the future
        but the current catalog uses single-cell placement.
        """
        # All current buildings occupy a single cell.
        return [(gx, gy)]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    # ── placement ─────────────────────────────────────────────────────
    def can_place(self, building_key: str, grid_x: int, grid_y: int) -> bool:
        """Check if a building can be placed at (grid_x, grid_y)."""
        if building_key not in BUILDING_CATALOG:
            return False
        cells = self._footprint(building_key, grid_x, grid_y)
        for cx, cy in cells:
            if not self._in_bounds(cx, cy):
                return False
            if (cx, cy) in self.cells:
                return False
        return True

    def place(self, building_key: str, grid_x: int, grid_y: int, place_name: str = "",
              sprite_variant: int = 0) -> PlacedBuilding:
        """Place a building.  Raises ``ValueError`` if the cell is occupied or
        the building key is unknown."""
        if not self.can_place(building_key, grid_x, grid_y):
            raise ValueError(
                f"Cannot place {building_key} at ({grid_x}, {grid_y}): "
                "cell occupied or out of bounds"
            )
        pb = PlacedBuilding(
            building_key=building_key,
            grid_x=grid_x,
            grid_y=grid_y,
            place_name=place_name or f"{building_key}_{grid_x}_{grid_y}",
            sprite_variant=sprite_variant,
        )
        cells = self._footprint(building_key, grid_x, grid_y)
        for cx, cy in cells:
            self.cells[(cx, cy)] = building_key
        self.buildings.append(pb)
        self._by_pos[(grid_x, grid_y)] = pb
        return pb

    def demolish(self, grid_x: int, grid_y: int) -> PlacedBuilding | None:
        """Remove the building at (grid_x, grid_y).  Returns the removed
        ``PlacedBuilding`` or ``None``."""
        pb = self._by_pos.pop((grid_x, grid_y), None)
        if pb is None:
            return None
        cells = self._footprint(pb.building_key, pb.grid_x, pb.grid_y)
        for cx, cy in cells:
            self.cells.pop((cx, cy), None)
        self.buildings = [b for b in self.buildings if b is not pb]
        return pb

    def building_at(self, grid_x: int, grid_y: int) -> PlacedBuilding | None:
        return self._by_pos.get((grid_x, grid_y))

    def buildable_cells(self) -> list[tuple[int, int]]:
        """Return all unoccupied cells within bounds."""
        result: list[tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.cells:
                    result.append((x, y))
        return result

    # ── serialization ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "buildings": [b.to_dict() for b in self.buildings],
        }
