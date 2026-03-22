"""Tile-based grid map with building/road placement."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from smrti_town.spatial import Place


class CellType(Enum):
    road = "road"
    sidewalk = "sidewalk"
    building = "building"
    grass = "grass"
    door = "door"
    water = "water"


# ── Building definitions ─────────────────────────────────────────────

@dataclass
class BuildingDef:
    """Blueprint for a building type."""
    grid_size: tuple[int, int]   # width x height in cells
    min_population: int          # population required to unlock
    staff_role: str              # role generated for staff
    staff_count: int = 1
    sprite_key: str = ""
    must_be_first: bool = False


BUILDING_DEFS: dict[str, BuildingDef] = {
    "city_hall": BuildingDef(
        grid_size=(6, 5), min_population=0, staff_role="mayor",
        staff_count=3, sprite_key="city_hall", must_be_first=True,
    ),
    "house": BuildingDef(
        grid_size=(4, 3), min_population=0, staff_role="resident",
        sprite_key="house",
    ),
    "farm": BuildingDef(
        grid_size=(5, 4), min_population=3, staff_role="farmer",
        sprite_key="farm",
    ),
    "market": BuildingDef(
        grid_size=(5, 4), min_population=5, staff_role="merchant",
        sprite_key="market",
    ),
    "school": BuildingDef(
        grid_size=(5, 4), min_population=8, staff_role="teacher",
        sprite_key="school",
    ),
    "workshop": BuildingDef(
        grid_size=(4, 4), min_population=10, staff_role="craftsperson",
        sprite_key="workshop",
    ),
    "clinic": BuildingDef(
        grid_size=(4, 3), min_population=12, staff_role="doctor",
        sprite_key="clinic",
    ),
    "tavern": BuildingDef(
        grid_size=(5, 4), min_population=15, staff_role="barkeeper",
        sprite_key="tavern",
    ),
    "church": BuildingDef(
        grid_size=(5, 5), min_population=15, staff_role="priest",
        sprite_key="church",
    ),
    "library": BuildingDef(
        grid_size=(5, 4), min_population=20, staff_role="librarian",
        sprite_key="library",
    ),
}


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class PlacedBuilding:
    """A building placed on the grid map."""
    place: Place
    grid_origin: tuple[int, int]   # top-left grid cell
    grid_size: tuple[int, int]     # width x height in cells
    door_cell: tuple[int, int]     # entry point
    sprite_key: str
    building_type: str = ""


@dataclass
class RoadSegment:
    """A road segment between two grid coordinates."""
    start: tuple[int, int]
    end: tuple[int, int]


# ── Bresenham line ────────────────────────────────────────────────────

def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return all grid cells along the line from (x0,y0) to (x1,y1)."""
    cells: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


# ── GridMap ───────────────────────────────────────────────────────────

@dataclass
class GridMap:
    """Tile-based world map with building and road placement."""

    width: int = 150
    height: int = 100
    cells: dict[tuple[int, int], CellType] = field(default_factory=dict)
    buildings: list[PlacedBuilding] = field(default_factory=list)
    roads: list[RoadSegment] = field(default_factory=list)

    # ── Query helpers ─────────────────────────────────────────────────

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def cell_at(self, x: int, y: int) -> CellType:
        return self.cells.get((x, y), CellType.grass)

    def cells_for_building(
        self, grid_x: int, grid_y: int, bw: int, bh: int,
    ) -> list[tuple[int, int]]:
        """Return all cells occupied by a building footprint."""
        return [
            (grid_x + dx, grid_y + dy)
            for dy in range(bh)
            for dx in range(bw)
        ]

    # ── Unlocking ─────────────────────────────────────────────────────

    @staticmethod
    def get_unlocked_buildings(population: int) -> list[str]:
        """Return building types unlocked at the given population."""
        return [
            name for name, bdef in BUILDING_DEFS.items()
            if population >= bdef.min_population
        ]

    # ── Validation ────────────────────────────────────────────────────

    def validate_placement(
        self, building_type: str, grid_x: int, grid_y: int,
    ) -> bool:
        """Check whether a building can be placed at (grid_x, grid_y)."""
        if building_type not in BUILDING_DEFS:
            return False
        bdef = BUILDING_DEFS[building_type]
        bw, bh = bdef.grid_size

        # Must be first building if must_be_first
        if bdef.must_be_first and self.buildings:
            return False

        # Bounds check
        if not self.in_bounds(grid_x, grid_y):
            return False
        if not self.in_bounds(grid_x + bw - 1, grid_y + bh - 1):
            return False

        # Overlap check — all footprint cells must be empty (grass)
        footprint = self.cells_for_building(grid_x, grid_y, bw, bh)
        for cx, cy in footprint:
            if self.cell_at(cx, cy) != CellType.grass:
                return False

        # Door cell — bottom-center of the building, one row below
        door_x = grid_x + bw // 2
        door_y = grid_y + bh  # one row below the building
        if not self.in_bounds(door_x, door_y):
            return False
        door_type = self.cell_at(door_x, door_y)
        if door_type not in (CellType.grass, CellType.road, CellType.sidewalk):
            return False

        # Road connectivity check — skip for the very first building
        # (it has no road network to connect to yet)
        if self.roads:
            if not self._can_reach_road(door_x, door_y):
                return False

        return True

    def _can_reach_road(self, start_x: int, start_y: int) -> bool:
        """BFS from a cell to any road tile. Max search radius 50 cells."""
        if self.cell_at(start_x, start_y) == CellType.road:
            return True
        visited: set[tuple[int, int]] = {(start_x, start_y)}
        queue: deque[tuple[int, int, int]] = deque([(start_x, start_y, 0)])
        max_dist = 50
        while queue:
            cx, cy, dist = queue.popleft()
            if dist >= max_dist:
                continue
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if not self.in_bounds(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                cell = self.cell_at(nx, ny)
                if cell == CellType.road:
                    return True
                if cell in (CellType.grass, CellType.sidewalk, CellType.door):
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
        return False

    # ── Building placement ────────────────────────────────────────────

    def place_building(
        self, building_type: str, grid_x: int, grid_y: int,
        place: Place | None = None,
    ) -> PlacedBuilding:
        """Validate and place a building. Returns PlacedBuilding or raises ValueError."""
        if building_type not in BUILDING_DEFS:
            raise ValueError(f"Unknown building type: {building_type}")

        bdef = BUILDING_DEFS[building_type]

        if not self.validate_placement(building_type, grid_x, grid_y):
            raise ValueError(
                f"Cannot place {building_type} at ({grid_x}, {grid_y}): "
                "overlap, out of bounds, or no road connectivity"
            )

        bw, bh = bdef.grid_size
        door_x = grid_x + bw // 2
        door_y = grid_y + bh

        # Create Place if not provided
        if place is None:
            place = Place(
                name=f"{building_type}_{grid_x}_{grid_y}",
                place_type="home" if building_type == "house" else "public",
                x=grid_x, y=grid_y, w=bw, h=bh,
            )

        # Stamp footprint cells
        footprint = self.cells_for_building(grid_x, grid_y, bw, bh)
        for cx, cy in footprint:
            self.cells[(cx, cy)] = CellType.building

        # Stamp door cell
        self.cells[(door_x, door_y)] = CellType.door

        placed = PlacedBuilding(
            place=place,
            grid_origin=(grid_x, grid_y),
            grid_size=(bw, bh),
            door_cell=(door_x, door_y),
            sprite_key=bdef.sprite_key or building_type,
            building_type=building_type,
        )
        self.buildings.append(placed)

        # Auto-connect road for non-first buildings
        if self.roads:
            self.auto_connect_road((door_x, door_y))

        return placed

    # ── Road placement ────────────────────────────────────────────────

    def place_road(
        self, start: tuple[int, int], end: tuple[int, int],
    ) -> RoadSegment:
        """Place a road segment using Bresenham's line algorithm.

        Auto-generates sidewalk on adjacent cells that are grass.
        """
        sx, sy = start
        ex, ey = end
        if not self.in_bounds(sx, sy) or not self.in_bounds(ex, ey):
            raise ValueError(f"Road endpoints out of bounds: {start} -> {end}")

        line_cells = _bresenham(sx, sy, ex, ey)

        # Stamp road cells
        for cx, cy in line_cells:
            current = self.cell_at(cx, cy)
            if current in (CellType.grass, CellType.sidewalk):
                self.cells[(cx, cy)] = CellType.road

        # Generate sidewalk on adjacent grass cells
        for cx, cy in line_cells:
            for nx, ny in (
                (cx + 1, cy), (cx - 1, cy),
                (cx, cy + 1), (cx, cy - 1),
                (cx + 1, cy + 1), (cx - 1, cy - 1),
                (cx + 1, cy - 1), (cx - 1, cy + 1),
            ):
                if self.in_bounds(nx, ny) and self.cell_at(nx, ny) == CellType.grass:
                    self.cells[(nx, ny)] = CellType.sidewalk

        segment = RoadSegment(start=start, end=end)
        self.roads.append(segment)
        return segment

    # ── Auto-connect road ─────────────────────────────────────────────

    def auto_connect_road(
        self, door_cell: tuple[int, int],
    ) -> RoadSegment | None:
        """Connect a door cell to the nearest existing road tile via a new road."""
        dx, dy = door_cell
        if self.cell_at(dx, dy) == CellType.road:
            return None  # already on a road

        # BFS to find nearest road tile
        visited: set[tuple[int, int]] = {(dx, dy)}
        queue: deque[tuple[int, int, int]] = deque([(dx, dy, 0)])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {(dx, dy): None}
        target: tuple[int, int] | None = None
        max_search = 80

        while queue:
            cx, cy, dist = queue.popleft()
            if dist >= max_search:
                continue
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if not self.in_bounds(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                cell = self.cell_at(nx, ny)
                if cell == CellType.road:
                    parent[(nx, ny)] = (cx, cy)
                    target = (nx, ny)
                    break
                if cell in (CellType.grass, CellType.sidewalk, CellType.door):
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny, dist + 1))
            if target is not None:
                break

        if target is None:
            return None

        # Trace path back from target to door
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = target
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()

        # Need at least door + one intermediate + road endpoint
        if len(path) < 2:
            return None

        # Walk the BFS path cell-by-cell instead of Bresenham (which would
        # draw a straight line that may cut through buildings).
        for cx, cy in path:
            cell = self.cell_at(cx, cy)
            if cell in (CellType.grass, CellType.sidewalk):
                self.cells[(cx, cy)] = CellType.road

        # Generate sidewalk around newly placed road cells
        for cx, cy in path:
            if self.cell_at(cx, cy) == CellType.road:
                for nx, ny in (
                    (cx + 1, cy), (cx - 1, cy),
                    (cx, cy + 1), (cx, cy - 1),
                    (cx + 1, cy + 1), (cx - 1, cy - 1),
                    (cx + 1, cy - 1), (cx - 1, cy + 1),
                ):
                    if self.in_bounds(nx, ny) and self.cell_at(nx, ny) == CellType.grass:
                        self.cells[(nx, ny)] = CellType.sidewalk

        segment = RoadSegment(start=path[0], end=path[-1])
        self.roads.append(segment)
        return segment

    # ── Demolish ──────────────────────────────────────────────────────

    def demolish(self, place_name: str) -> None:
        """Remove a building by its Place name, clearing cells back to grass."""
        target: PlacedBuilding | None = None
        for b in self.buildings:
            if b.place.name == place_name:
                target = b
                break
        if target is None:
            raise ValueError(f"No building found with place name: {place_name}")

        # Clear footprint cells
        ox, oy = target.grid_origin
        bw, bh = target.grid_size
        for cx, cy in self.cells_for_building(ox, oy, bw, bh):
            self.cells.pop((cx, cy), None)

        # Clear door cell
        self.cells.pop(target.door_cell, None)

        self.buildings.remove(target)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the grid map for frontend consumption."""
        return {
            "width": self.width,
            "height": self.height,
            "cells": {
                f"{x},{y}": cell.value
                for (x, y), cell in self.cells.items()
            },
            "buildings": [
                {
                    "place_name": b.place.name,
                    "building_type": b.building_type,
                    "grid_origin": list(b.grid_origin),
                    "grid_size": list(b.grid_size),
                    "door_cell": list(b.door_cell),
                    "sprite_key": b.sprite_key,
                }
                for b in self.buildings
            ],
            "roads": [
                {
                    "start": list(r.start),
                    "end": list(r.end),
                }
                for r in self.roads
            ],
        }
