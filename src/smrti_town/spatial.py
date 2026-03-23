"""TownTopology + Place — adjacency graph for town navigation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smrti import Smrti


@dataclass
class Place:
    name: str
    place_type: str  # civic, commercial, residential, industrial, cultural, infrastructure, outdoor
    building_key: str | None = None
    is_outdoor: bool = False
    occupants: set[str] = field(default_factory=set)
    grid_x: int = 0
    grid_y: int = 0
    smrti: "Smrti | None" = None
    _home_of: set[str] = field(default_factory=set)
    _workplace_of: set[str] = field(default_factory=set)

    @property
    def space_name(self) -> str:
        return f"Place_Space_{self.name}"

    def add_occupant(self, name: str) -> None:
        self.occupants.add(name)

    def remove_occupant(self, name: str) -> None:
        self.occupants.discard(name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "place_type": self.place_type,
            "building_key": self.building_key,
            "is_outdoor": self.is_outdoor,
            "occupants": sorted(self.occupants),
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
        }


class TownTopology:
    """Undirected adjacency graph of named Places with BFS path distance."""

    def __init__(self) -> None:
        self.places: dict[str, Place] = {}
        self._adjacency: dict[str, set[str]] = {}

    # ── graph mutations ───────────────────────────────────────────────
    def add_place(self, place: Place) -> None:
        self.places[place.name] = place
        self._adjacency.setdefault(place.name, set())

    def connect(self, a: str, b: str) -> None:
        """Create a bidirectional edge between two places."""
        self._adjacency.setdefault(a, set()).add(b)
        self._adjacency.setdefault(b, set()).add(a)

    def remove_place(self, name: str) -> Place | None:
        place = self.places.pop(name, None)
        if place is None:
            return None
        neighbors = self._adjacency.pop(name, set())
        for nb in neighbors:
            self._adjacency.get(nb, set()).discard(name)
        return place

    # ── queries ───────────────────────────────────────────────────────
    def neighbors(self, place_name: str) -> list[str]:
        return sorted(self._adjacency.get(place_name, set()))

    def path_distance(self, a: str, b: str) -> int:
        """BFS shortest path distance.  Returns -1 if unreachable."""
        if a == b:
            return 0
        if a not in self._adjacency or b not in self._adjacency:
            return -1
        visited: set[str] = {a}
        queue: deque[tuple[str, int]] = deque([(a, 0)])
        while queue:
            current, dist = queue.popleft()
            for nb in self._adjacency.get(current, ()):
                if nb == b:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return -1

    def find_path(self, a: str, b: str) -> list[str]:
        """BFS shortest path returning list of place names from *a* to *b*
        (inclusive).  Returns empty list if unreachable."""
        if a == b:
            return [a]
        if a not in self._adjacency or b not in self._adjacency:
            return []
        visited: set[str] = {a}
        queue: deque[tuple[str, list[str]]] = deque([(a, [a])])
        while queue:
            current, path = queue.popleft()
            for nb in self._adjacency.get(current, ()):
                if nb == b:
                    return path + [nb]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return []

    def places_by_type(self, place_type: str) -> list[Place]:
        return [p for p in self.places.values() if p.place_type == place_type]

    def places_by_building(self, building_key: str) -> list[Place]:
        return [p for p in self.places.values() if p.building_key == building_key]

    def home_for(self, citizen_name: str) -> Place | None:
        for p in self.places.values():
            if citizen_name in p._home_of:
                return p
        return None

    def workplace_for(self, citizen_name: str) -> Place | None:
        for p in self.places.values():
            if citizen_name in p._workplace_of:
                return p
        return None

    def assign_home(self, citizen_name: str, place_name: str) -> None:
        # Remove from previous home first.
        for p in self.places.values():
            p._home_of.discard(citizen_name)
        place = self.places.get(place_name)
        if place:
            place._home_of.add(citizen_name)

    def assign_workplace(self, citizen_name: str, place_name: str) -> None:
        for p in self.places.values():
            p._workplace_of.discard(citizen_name)
        place = self.places.get(place_name)
        if place:
            place._workplace_of.add(citizen_name)

    def unassign_home(self, citizen_name: str) -> None:
        for p in self.places.values():
            p._home_of.discard(citizen_name)

    def unassign_workplace(self, citizen_name: str) -> None:
        for p in self.places.values():
            p._workplace_of.discard(citizen_name)

    def move_agent(self, name: str, from_place: str | None, to_place: str) -> None:
        if from_place and from_place in self.places:
            self.places[from_place].remove_occupant(name)
        if to_place in self.places:
            self.places[to_place].add_occupant(name)

    def all_place_names(self) -> list[str]:
        return sorted(self.places.keys())

    def all_connections(self) -> list[list[str]]:
        """Return deduplicated edge list as [[a, b], ...]."""
        seen: set[tuple[str, str]] = set()
        edges: list[list[str]] = []
        for a, nbs in self._adjacency.items():
            for b in nbs:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    edges.append(list(key))
        return edges

    # ── serialization ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "places": {name: p.to_dict() for name, p in self.places.items()},
            "connections": self.all_connections(),
        }
