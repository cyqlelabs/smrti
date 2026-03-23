"""NavGrid — walkability grid with A* pathfinding and coordinate conversion."""

from __future__ import annotations

import heapq
import math

from smrti_town.config import CELL_SIZE, GRID_HEIGHT, GRID_WIDTH


class NavGrid:
    """Grid-based walkability map with A* pathfinding.

    Coordinates are integer grid cells.  Use :meth:`grid_to_world` /
    :meth:`world_to_grid` to convert between grid and pixel-world space.
    """

    def __init__(self, width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> None:
        self.width = width
        self.height = height
        self.blocked: set[tuple[int, int]] = set()

    # ── mutations ─────────────────────────────────────────────────────
    def block(self, x: int, y: int) -> None:
        self.blocked.add((x, y))

    def unblock(self, x: int, y: int) -> None:
        self.blocked.discard((x, y))

    def block_rect(self, x: int, y: int, w: int, h: int) -> None:
        """Block a rectangular region of cells."""
        for dy in range(h):
            for dx in range(w):
                self.blocked.add((x + dx, y + dy))

    # ── queries ───────────────────────────────────────────────────────
    def is_walkable(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False
        return (x, y) not in self.blocked

    # ── A* pathfinding ────────────────────────────────────────────────
    # 8-directional movement.
    _NEIGHBORS = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]
    _SQRT2 = math.sqrt(2)

    def find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """A* from *start* to *goal*.  Returns the path as a list of ``(x, y)``
        grid cells (inclusive of start and goal), or an empty list if no path
        exists.

        Uses octile distance as the heuristic for 8-directional movement.
        """
        sx, sy = start
        gx, gy = goal

        if not self.is_walkable(gx, gy):
            # Goal is blocked — find the closest walkable neighbor instead.
            best = self._closest_walkable(gx, gy)
            if best is None:
                return []
            gx, gy = best

        if start == (gx, gy):
            return [start]

        # Octile distance heuristic.
        def _h(x: int, y: int) -> float:
            dx = abs(x - gx)
            dy = abs(y - gy)
            return max(dx, dy) + (self._SQRT2 - 1) * min(dx, dy)

        open_set: list[tuple[float, int, int, int]] = []  # (f, tiebreak, x, y)
        heapq.heappush(open_set, (_h(sx, sy), 0, sx, sy))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
        g_score: dict[tuple[int, int], float] = {(sx, sy): 0.0}
        counter = 1  # tiebreaker

        while open_set:
            _, _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                return self._reconstruct(came_from, (gx, gy))

            current_g = g_score[(cx, cy)]

            for dx, dy in self._NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if not self.is_walkable(nx, ny):
                    continue

                # Prevent diagonal movement through blocked corners.
                if dx != 0 and dy != 0:
                    if not self.is_walkable(cx + dx, cy) or not self.is_walkable(cx, cy + dy):
                        continue

                step_cost = self._SQRT2 if (dx != 0 and dy != 0) else 1.0
                tentative_g = current_g + step_cost

                if tentative_g < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + _h(nx, ny)
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_set, (f, counter, nx, ny))
                    counter += 1

        return []  # No path found.

    def _closest_walkable(self, x: int, y: int) -> tuple[int, int] | None:
        """Find the closest walkable cell to (x, y) within a small radius."""
        for r in range(1, 6):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = x + dx, y + dy
                        if self.is_walkable(nx, ny):
                            return (nx, ny)
        return None

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    # ── coordinate conversion ─────────────────────────────────────────
    @staticmethod
    def grid_to_world(gx: int, gy: int) -> tuple[float, float]:
        """Convert grid cell to world-space pixel center."""
        return (gx + 0.5) * CELL_SIZE, (gy + 0.5) * CELL_SIZE

    @staticmethod
    def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
        """Convert world-space pixel position to grid cell."""
        return int(wx // CELL_SIZE), int(wy // CELL_SIZE)
