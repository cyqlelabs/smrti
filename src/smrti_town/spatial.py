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
    """Construct the Millbrook town layout from the plan."""
    topo = TownTopology()

    # Root — virtual node, not shown on map
    topo.add_place(Place(
        name="Town_Millbrook",
        is_outdoor=True,
        place_type="other",
        display=False,
    ))

    # Streets
    topo.add_place(Place(
        name="Elm_Street",
        parent="Town_Millbrook",
        is_outdoor=True,
        has_space=False,
        place_type="street",
        x=160, y=290, w=160, h=30,
        color="#C4B898", icon="", label="Elm St",
    ))
    topo.add_place(Place(
        name="Main_Street",
        parent="Town_Millbrook",
        is_outdoor=True,
        has_space=False,
        place_type="street",
        x=440, y=290, w=320, h=36,
        color="#B8A88A", icon="", label="Main Street",
    ))

    # Connect streets to town root
    topo.connect("Town_Millbrook", "Elm_Street")
    topo.connect("Town_Millbrook", "Main_Street")
    topo.connect("Elm_Street", "Main_Street")

    # Elm Street — homes
    topo.add_place(Place(
        name="Alice_Home",
        parent="Elm_Street",
        personality="balanced",
        place_type="home",
        x=140, y=380, w=130, h=100,
        color="#D4A03C", icon="🏠", label="Alice's Home",
    ))
    topo.add_place(Place(
        name="Sofia_Home",
        parent="Elm_Street",
        personality="balanced",
        place_type="home",
        x=140, y=160, w=130, h=100,
        color="#C4873C", icon="🏡", label="Sofia's Home",
    ))
    topo.connect("Elm_Street", "Alice_Home")
    topo.connect("Elm_Street", "Sofia_Home")

    # Main Street — public buildings
    topo.add_place(Place(
        name="Cafe_Rosetta",
        parent="Main_Street",
        personality="curious",
        place_type="public",
        x=320, y=180, w=160, h=110,
        color="#E8734A", icon="☕", label="Cafe Rosetta",
    ))
    topo.add_place(Place(
        name="Public_Library",
        parent="Main_Street",
        personality="analytical",
        place_type="public",
        x=560, y=160, w=160, h=110,
        color="#2A7B7F", icon="📚", label="Library",
    ))
    topo.add_place(Place(
        name="Town_Market",
        parent="Main_Street",
        personality="maverick",
        place_type="public",
        x=720, y=360, w=160, h=110,
        color="#8B4C8B", icon="🏪", label="Market",
    ))
    topo.connect("Main_Street", "Cafe_Rosetta")
    topo.connect("Main_Street", "Public_Library")
    topo.connect("Main_Street", "Town_Market")

    # Central Park connects to Main Street and Elm Street
    topo.add_place(Place(
        name="Central_Park",
        parent="Town_Millbrook",
        personality="empathetic",
        is_outdoor=True,
        place_type="outdoor",
        x=440, y=400, w=200, h=140,
        color="#5C9E5C", icon="🌳", label="Central Park",
    ))
    topo.connect("Town_Millbrook", "Central_Park")
    topo.connect("Main_Street", "Central_Park")
    topo.connect("Elm_Street", "Central_Park")

    return topo
