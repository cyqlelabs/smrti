"""Town topology: Place class, path distance, occupant tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Place:
    """A location in the town."""

    name: str
    parent: Optional[str] = None
    personality: str = "balanced"
    is_outdoor: bool = False
    occupants: set[str] = field(default_factory=set)
    children: list[str] = field(default_factory=list)
    # A Place_Space is only created for socially significant places.
    # Leaf rooms use their parent's space.
    has_space: bool = True
    # Layout / rendering fields
    place_type: str = "other"   # "street", "public", "outdoor", "home", "other"
    display: bool = True        # False = root/virtual node, not shown on map
    x: int = 400
    y: int = 300
    w: int = 120
    h: int = 90
    color: str = "#888888"
    icon: str = ""
    label: str = ""

    def add_occupant(self, agent_name: str) -> None:
        self.occupants.add(agent_name)

    def remove_occupant(self, agent_name: str) -> None:
        self.occupants.discard(agent_name)

    @property
    def space_name(self) -> str:
        return f"Place_Space_{self.name}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parent": self.parent,
            "personality": self.personality,
            "is_outdoor": self.is_outdoor,
            "occupants": sorted(self.occupants),
            "children": self.children,
            "place_type": self.place_type,
            "display": self.display,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "color": self.color,
            "icon": self.icon,
            "label": self.label,
        }


class TownTopology:
    """Spatial graph of places with path distance calculation."""

    def __init__(self) -> None:
        self.places: dict[str, Place] = {}
        self._adjacency: dict[str, set[str]] = {}

    def add_place(self, place: Place) -> None:
        self.places[place.name] = place
        if place.name not in self._adjacency:
            self._adjacency[place.name] = set()
        if place.parent and place.parent in self.places:
            self.places[place.parent].children.append(place.name)
            self._adjacency.setdefault(place.parent, set()).add(place.name)
            self._adjacency[place.name].add(place.parent)

    def connect(self, a: str, b: str) -> None:
        """Add a bidirectional edge between two places."""
        self._adjacency.setdefault(a, set()).add(b)
        self._adjacency.setdefault(b, set()).add(a)

    def path_distance(self, a: str, b: str) -> int:
        """BFS shortest path distance between two places. -1 if unreachable."""
        if a == b:
            return 0
        if a not in self._adjacency or b not in self._adjacency:
            return -1
        visited: set[str] = {a}
        queue: list[tuple[str, int]] = [(a, 0)]
        idx = 0
        while idx < len(queue):
            current, dist = queue[idx]
            idx += 1
            for neighbor in self._adjacency.get(current, set()):
                if neighbor == b:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return -1

    def neighbors(self, place_name: str) -> list[str]:
        return sorted(self._adjacency.get(place_name, set()))

    def reachable_places(self, from_place: str, max_dist: int = 1) -> list[str]:
        """Return places reachable within max_dist hops."""
        result: list[str] = []
        visited: set[str] = {from_place}
        queue: list[tuple[str, int]] = [(from_place, 0)]
        idx = 0
        while idx < len(queue):
            current, dist = queue[idx]
            idx += 1
            if dist > 0:
                result.append(current)
            if dist < max_dist:
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return result

    def all_place_names(self) -> list[str]:
        return sorted(self.places.keys())

    def all_connections(self) -> list[list[str]]:
        """Return all unique bidirectional edges as [a, b] pairs."""
        seen: set[frozenset] = set()
        result: list[list[str]] = []
        for a, neighbors in self._adjacency.items():
            for b in neighbors:
                key = frozenset((a, b))
                if key not in seen:
                    seen.add(key)
                    result.append([a, b])
        return result

    def places_of_type(self, place_type: str) -> list[str]:
        """Return names of all places with a given place_type."""
        return [name for name, p in self.places.items() if p.place_type == place_type]

    def places_by_type(self, place_type: str) -> list[str]:
        """Alias for places_of_type."""
        return self.places_of_type(place_type)

    def home_for(self, agent_name: str) -> str | None:
        """Return the home place whose name contains agent_name, or any home as fallback."""
        homes = self.places_of_type("home")
        if not homes:
            return None
        for name in homes:
            if agent_name in name:
                return name
        return homes[0]

    def move_agent(self, agent_name: str, from_place: str, to_place: str) -> None:
        if from_place in self.places:
            self.places[from_place].remove_occupant(agent_name)
        if to_place in self.places:
            self.places[to_place].add_occupant(agent_name)


def build_millbrook_topology() -> TownTopology:
    """Founding scenario: a single Town Hall where settlers begin."""
    topo = TownTopology()
    topo.add_place(Place(
        name="Town_Hall",
        personality="balanced",
        is_outdoor=False,
        has_space=True,
        place_type="public",
        display=True,
        x=360, y=220, w=200, h=140,
        color="#5D3A1A",
        icon="\U0001f3db\ufe0f",
        label="Town Hall",
    ))
    return topo
