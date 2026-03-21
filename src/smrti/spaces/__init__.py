"""Space set theory: intersection, union, difference, and emergent bridge spaces."""

from smrti.spaces.set_ops import (
    space_intersection,
    space_difference,
    space_union,
    space_symmetric_difference,
    space_overlap,
)
from smrti.spaces.emergence import materialize_bridge

__all__ = [
    "space_intersection",
    "space_difference",
    "space_union",
    "space_symmetric_difference",
    "space_overlap",
    "materialize_bridge",
]
