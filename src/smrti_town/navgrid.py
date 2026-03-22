"""Tile-based NavGrid with A* pathfinding for agent movement."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from smrti_town.spatial import Place, TownTopology


# ── Grid constants ────────────────────────────────────────────────────
CELL_SIZE = 16  # pixels per cell
WORLD_W = 2400
WORLD_H = 1600
GRID_W = WORLD_W // CELL_SIZE   # 150
GRID_H = WORLD_H // CELL_SIZE   # 100

# Movement costs
COST_ROAD = 1
COST_SIDEWALK = 2
COST_DOOR = 2
COST_GRASS = 5
COST_IMPASSABLE = float("inf")

# Diagonal movement cost multiplier
_SQRT2 = math.sqrt(2)

# 8-directional neighbors (dx, dy)
_DIRECTIONS = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]


class OccupantType(IntEnum):
    GRASS = 0
    ROAD = 1
    SIDEWALK = 2
    DOOR = 3
    BUILDING = 4
    WATER = 5


@dataclass
class Cell:
    cost: float = COST_GRASS
    occupant_type: OccupantType = OccupantType.GRASS


def grid_to_world(gx: int, gy: int) -> tuple[float, float]:
    return gx * CELL_SIZE + CELL_SIZE / 2, gy * CELL_SIZE + CELL_SIZE / 2


def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
    gx = max(0, min(int(wx // CELL_SIZE), GRID_W - 1))
    gy = max(0, min(int(wy // CELL_SIZE), GRID_H - 1))
    return gx, gy


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
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


class NavGrid:
    def __init__(self) -> None:
        self.cells: list[list[Cell]] = [
            [Cell() for _ in range(GRID_H)] for _ in range(GRID_W)
        ]
        self._path_cache: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {}
        self._agent_cells: dict[str, tuple[int, int]] = {}
        # Spatial hash: grid cell -> set of agent names
        self._spatial_hash: dict[tuple[int, int], set[str]] = {}
        # Door cells per place name
        self._doors: dict[str, tuple[int, int]] = {}

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < GRID_W and 0 <= gy < GRID_H

    def _set_cell(self, gx: int, gy: int, cost: float, occupant_type: OccupantType) -> None:
        if self._in_bounds(gx, gy):
            self.cells[gx][gy].cost = cost
            self.cells[gx][gy].occupant_type = occupant_type

    def _set_cell_if_lower(self, gx: int, gy: int, cost: float, occupant_type: OccupantType) -> None:
        if self._in_bounds(gx, gy):
            cell = self.cells[gx][gy]
            if cell.occupant_type in (OccupantType.BUILDING, OccupantType.WATER):
                return
            if cost < cell.cost:
                cell.cost = cost
                cell.occupant_type = occupant_type

    # ── Bake pipeline ─────────────────────────────────────────────────

    def bake(self, topology: TownTopology) -> None:
        self._path_cache.clear()
        self._doors.clear()

        # Step 1: reset all cells to grass
        for gx in range(GRID_W):
            for gy in range(GRID_H):
                self.cells[gx][gy].cost = COST_GRASS
                self.cells[gx][gy].occupant_type = OccupantType.GRASS

        # Step 2: rasterize buildings as impassable
        buildings: list[Place] = []
        for place in topology.places.values():
            if not place.display:
                continue
            if place.place_type in ("street", "outdoor"):
                continue
            buildings.append(place)
            self._rasterize_building(place)

        # Step 3: rasterize roads between connected places
        connections = topology.all_connections()
        for a_name, b_name in connections:
            a = topology.places.get(a_name)
            b = topology.places.get(b_name)
            if not a or not b:
                continue
            self._rasterize_road(a, b)

        # Step 4: designate door cells for buildings
        for place in buildings:
            self._designate_door(place)

    def _rasterize_building(self, place: Place) -> None:
        gx0, gy0 = world_to_grid(place.x, place.y)
        gx1, gy1 = world_to_grid(place.x + place.w, place.y + place.h)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                self._set_cell(gx, gy, COST_IMPASSABLE, OccupantType.BUILDING)

    def _rasterize_road(self, a: Place, b: Place) -> None:
        ax, ay = world_to_grid(a.x + a.w // 2, a.y + a.h // 2)
        bx, by = world_to_grid(b.x + b.w // 2, b.y + b.h // 2)
        road_cells = _bresenham(ax, ay, bx, by)
        for gx, gy in road_cells:
            self._set_cell_if_lower(gx, gy, COST_ROAD, OccupantType.ROAD)
            # Sidewalk: adjacent cells
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                self._set_cell_if_lower(gx + dx, gy + dy, COST_SIDEWALK, OccupantType.SIDEWALK)

    def _designate_door(self, place: Place) -> None:
        cx, cy = world_to_grid(place.x + place.w // 2, place.y + place.h // 2)
        gx0, gy0 = world_to_grid(place.x, place.y)
        gx1, gy1 = world_to_grid(place.x + place.w, place.y + place.h)

        # Scan perimeter cells and find the one nearest to a road
        best: Optional[tuple[int, int]] = None
        best_dist = float("inf")

        perimeter: list[tuple[int, int]] = []
        for gx in range(gx0, gx1 + 1):
            perimeter.append((gx, gy0))
            perimeter.append((gx, gy1))
        for gy in range(gy0 + 1, gy1):
            perimeter.append((gx0, gy))
            perimeter.append((gx1, gy))

        for px, py in perimeter:
            # Check adjacent cells (outside the building) for road
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if not self._in_bounds(nx, ny):
                    continue
                if self.cells[nx][ny].occupant_type in (OccupantType.ROAD, OccupantType.SIDEWALK):
                    dist = abs(px - cx) + abs(py - cy)
                    if dist < best_dist:
                        best_dist = dist
                        best = (px, py)
                    break

        if best is None:
            # Fallback: bottom-center of building
            best = (cx, gy1)

        self._set_cell(best[0], best[1], COST_DOOR, OccupantType.DOOR)
        self._doors[place.name] = best

    def door_for(self, place_name: str) -> Optional[tuple[int, int]]:
        return self._doors.get(place_name)

    # ── A* pathfinding ────────────────────────────────────────────────

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        cache_key = (start, goal)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        path = self._astar(start, goal)
        if path:
            self._path_cache[cache_key] = path
        return path

    def find_path_world(
        self,
        start_wx: float,
        start_wy: float,
        goal_wx: float,
        goal_wy: float,
    ) -> list[tuple[float, float]]:
        start = world_to_grid(start_wx, start_wy)
        goal = world_to_grid(goal_wx, goal_wy)
        grid_path = self.find_path(start, goal)
        return [grid_to_world(gx, gy) for gx, gy in grid_path]

    def find_path_between_places(
        self,
        origin: str,
        destination: str,
    ) -> list[tuple[float, float]]:
        origin_door = self._doors.get(origin)
        dest_door = self._doors.get(destination)
        if origin_door is None or dest_door is None:
            return []
        grid_path = self.find_path(origin_door, dest_door)
        return [grid_to_world(gx, gy) for gx, gy in grid_path]

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        if start == goal:
            return [start]

        sx, sy = start
        gx, gy = goal

        # open set: (f_score, counter, x, y)
        counter = 0
        open_set: list[tuple[float, int, int, int]] = []
        heapq.heappush(open_set, (0.0, counter, sx, sy))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        closed: set[tuple[int, int]] = set()

        while open_set:
            _, _, cx, cy = heapq.heappop(open_set)
            current = (cx, cy)

            if current == goal:
                return self._reconstruct(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            current_g = g_score[current]

            for dx, dy in _DIRECTIONS:
                nx, ny = cx + dx, cy + dy
                if not self._in_bounds(nx, ny):
                    continue

                cell = self.cells[nx][ny]
                if cell.cost == float("inf"):
                    continue

                # Diagonal movement costs sqrt(2) * cell cost
                move_cost = cell.cost * (_SQRT2 if dx != 0 and dy != 0 else 1.0)

                # Prevent diagonal movement through impassable corners
                if dx != 0 and dy != 0:
                    if self.cells[cx + dx][cy].cost == float("inf") or \
                       self.cells[cx][cy + dy].cost == float("inf"):
                        continue

                tentative_g = current_g + move_cost
                neighbor = (nx, ny)

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    # Octile distance heuristic
                    hdx = abs(nx - gx)
                    hdy = abs(ny - gy)
                    h = max(hdx, hdy) + (_SQRT2 - 1) * min(hdx, hdy)
                    f = tentative_g + h
                    counter += 1
                    heapq.heappush(open_set, (f, counter, nx, ny))

        return []  # no path found

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ── Nearest road lookup ───────────────────────────────────────────

    def find_nearest_road_cell(self, gx: int, gy: int) -> Optional[tuple[int, int]]:
        if self._in_bounds(gx, gy) and self.cells[gx][gy].occupant_type == OccupantType.ROAD:
            return (gx, gy)

        # BFS outward from (gx, gy)
        visited: set[tuple[int, int]] = {(gx, gy)}
        queue: list[tuple[int, int]] = [(gx, gy)]
        idx = 0
        while idx < len(queue):
            cx, cy = queue[idx]
            idx += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if not self._in_bounds(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                if self.cells[nx][ny].occupant_type == OccupantType.ROAD:
                    return (nx, ny)
                queue.append((nx, ny))
        return None

    # ── Agent spatial tracking ────────────────────────────────────────

    def update_agent_position(self, agent_name: str, wx: float, wy: float) -> None:
        old = self._agent_cells.get(agent_name)
        new = world_to_grid(wx, wy)

        if old == new:
            return

        if old is not None:
            bucket = self._spatial_hash.get(old)
            if bucket:
                bucket.discard(agent_name)
                if not bucket:
                    del self._spatial_hash[old]

        self._agent_cells[agent_name] = new
        self._spatial_hash.setdefault(new, set()).add(agent_name)

    def remove_agent(self, agent_name: str) -> None:
        old = self._agent_cells.pop(agent_name, None)
        if old is not None:
            bucket = self._spatial_hash.get(old)
            if bucket:
                bucket.discard(agent_name)
                if not bucket:
                    del self._spatial_hash[old]

    def agents_near(self, agent_name: str, radius: int = 1) -> list[str]:
        cell = self._agent_cells.get(agent_name)
        if cell is None:
            return []

        cx, cy = cell
        nearby: list[str] = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                bucket = self._spatial_hash.get((cx + dx, cy + dy))
                if bucket:
                    for name in bucket:
                        if name != agent_name:
                            nearby.append(name)
        return nearby

    def agents_share_cell(self, a: str, b: str) -> bool:
        ca = self._agent_cells.get(a)
        cb = self._agent_cells.get(b)
        if ca is None or cb is None:
            return False
        return ca == cb

    def agents_adjacent(self, a: str, b: str) -> bool:
        ca = self._agent_cells.get(a)
        cb = self._agent_cells.get(b)
        if ca is None or cb is None:
            return False
        return abs(ca[0] - cb[0]) <= 1 and abs(ca[1] - cb[1]) <= 1
