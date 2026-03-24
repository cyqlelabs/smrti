"""PetitionManager — citizen-driven requests and needs-based petition generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from smrti_town.config import (
    BUILDING_CATALOG,
    NEED_MAX,
    PETITION_MAX_AGE_HOURS,
    PETITION_SIMILARITY_THRESHOLD,
)


@dataclass
class Petition:
    text: str
    source: str  # "council" or citizen name
    category: str  # "housing", "food", "safety", "education", "health", "culture", "infrastructure"
    building_suggestion: str | None = None
    signatures: list[str] = field(default_factory=list)
    created_at_hours: float = 0.0
    urgency: float = 0.5  # 0.0-1.0
    status: str = "active"  # "active", "approved", "dismissed", "expired"


# ── Need-to-petition mapping ───────────────────────────────────────────

_NEED_PETITION_MAP: dict[str, list[dict]] = {
    "hunger": [
        {
            "category": "food",
            "text": "Citizens are going hungry. We need a source of food.",
            "buildings": ["farm", "bakery", "butcher", "market"],
            "threshold": 60,
        },
    ],
    "shelter": [
        {
            "category": "housing",
            "text": "Citizens lack housing. We need more homes.",
            "buildings": ["cottage", "house", "apartment"],
            "threshold": 40,
        },
    ],
    "health": [
        {
            "category": "health",
            "text": "Citizens are falling ill without medical care.",
            "buildings": ["clinic", "hospital"],
            "threshold": 50,
        },
        {
            "category": "infrastructure",
            "text": "We need clean water infrastructure.",
            "buildings": ["well", "water_tower"],
            "threshold": 40,
        },
    ],
    "safety": [
        {
            "category": "safety",
            "text": "Crime and disorder threaten our community. We need law enforcement.",
            "buildings": ["constabulary", "jail"],
            "threshold": 50,
        },
    ],
    "social": [
        {
            "category": "culture",
            "text": "Citizens feel isolated. We need social gathering places.",
            "buildings": ["park", "tavern", "church", "festival_grounds"],
            "threshold": 60,
        },
    ],
    "education": [
        {
            "category": "education",
            "text": "Children need education. We should build a school.",
            "buildings": ["school", "library"],
            "threshold": 50,
        },
    ],
    "purpose": [
        {
            "category": "infrastructure",
            "text": "Adults lack meaningful work. We need more businesses.",
            "buildings": ["general_store", "blacksmith", "tailor", "trading_post"],
            "threshold": 50,
        },
    ],
    "culture": [
        {
            "category": "culture",
            "text": "Our town lacks cultural enrichment.",
            "buildings": ["theater", "museum", "bookstore"],
            "threshold": 55,
        },
    ],
}


class PetitionManager:
    """Manages citizen petitions: creation, signing, merging, and lifecycle."""

    def __init__(self) -> None:
        self.petitions: list[Petition] = []

    # ── creation ────────────────────────────────────────────────────────

    def add_petition(
        self,
        text: str,
        source: str,
        category: str,
        building_suggestion: str | None = None,
        urgency: float = 0.5,
        current_hours: float = 0.0,
    ) -> Petition:
        """Add a new petition.  Returns the created Petition."""
        pet = Petition(
            text=text,
            source=source,
            category=category,
            building_suggestion=building_suggestion,
            signatures=[source] if source != "council" else [],
            created_at_hours=current_hours,
            urgency=max(0.0, min(1.0, urgency)),
        )
        self.petitions.append(pet)
        return pet

    # ── needs scanning ──────────────────────────────────────────────────

    def check_citizen_needs(
        self,
        citizens: list,
        topology,
        economy,
        current_hours: float = 0.0,
    ) -> list[Petition]:
        """Scan citizens for unmet needs and generate petitions.

        Only generates a petition if:
        1. Enough citizens share the same unmet need (>=30% or >=3).
        2. No existing active petition already covers it.
        3. The suggested building is not already present.
        """
        alive = [c for c in citizens if getattr(c, "alive", True)]
        if not alive:
            return []

        # Collect existing building keys in the topology.
        existing_buildings: set[str] = set()
        places = getattr(topology, "places", {})
        if isinstance(places, dict):
            for place in places.values():
                bkey = getattr(place, "building_key", None)
                if bkey:
                    existing_buildings.add(bkey)

        # Tally need values across citizens.
        need_tallies: dict[str, list[float]] = {}
        for c in alive:
            needs = getattr(c, "needs", None)
            if not needs:
                continue
            for need_name in _NEED_PETITION_MAP:
                val = getattr(needs, need_name, 0.0)
                need_tallies.setdefault(need_name, []).append(val)

        active_categories = {
            p.category for p in self.petitions if p.status == "active"
        }

        new_petitions: list[Petition] = []
        pop = len(alive)
        threshold_count = max(3, int(pop * 0.3))

        for need_name, entries in _NEED_PETITION_MAP.items():
            values = need_tallies.get(need_name, [])
            if not values:
                continue

            for entry in entries:
                need_threshold = entry["threshold"]
                affected = [v for v in values if v >= need_threshold]
                if len(affected) < min(threshold_count, pop):
                    continue

                category = entry["category"]
                if category in active_categories:
                    continue

                # Find a building suggestion that is not yet built and
                # whose population unlock is met.
                suggestion: str | None = None
                for bkey in entry["buildings"]:
                    if bkey in existing_buildings:
                        continue
                    bdef = BUILDING_CATALOG.get(bkey)
                    if bdef and bdef.unlock_population <= pop:
                        # Check prerequisite buildings.
                        if bdef.unlock_buildings and not all(
                            ub in existing_buildings for ub in bdef.unlock_buildings
                        ):
                            continue
                        suggestion = bkey
                        break

                if suggestion is None:
                    # All candidate buildings already exist or are locked.
                    continue

                avg_severity = sum(affected) / len(affected) / NEED_MAX
                urgency = min(1.0, avg_severity + len(affected) / pop * 0.3)

                pet = self.add_petition(
                    text=entry["text"],
                    source="citizens",
                    category=category,
                    building_suggestion=suggestion,
                    urgency=urgency,
                    current_hours=current_hours,
                )
                # Auto-sign with affected citizens.
                for c in alive:
                    needs_obj = getattr(c, "needs", None)
                    if needs_obj and getattr(needs_obj, need_name, 0.0) >= need_threshold:
                        name = getattr(c, "name", "")
                        if name and name not in pet.signatures:
                            pet.signatures.append(name)

                new_petitions.append(pet)
                active_categories.add(category)

        return new_petitions

    # ── signatures ──────────────────────────────────────────────────────

    def add_signature(self, petition_index: int, citizen_name: str) -> bool:
        """Add a citizen's signature to a petition.  Returns True if added."""
        if petition_index < 0 or petition_index >= len(self.petitions):
            return False
        pet = self.petitions[petition_index]
        if pet.status != "active":
            return False
        if citizen_name in pet.signatures:
            return False
        pet.signatures.append(citizen_name)
        return True

    # ── merging ─────────────────────────────────────────────────────────

    def merge_similar(self) -> int:
        """Merge petitions about similar topics (same category and building suggestion).

        Returns number of merges performed.
        """
        active = [p for p in self.petitions if p.status == "active"]
        merged_count = 0
        seen: dict[tuple[str, str | None], Petition] = {}

        for pet in active:
            key = (pet.category, pet.building_suggestion)
            if key in seen:
                target = seen[key]
                # Merge signatures.
                for sig in pet.signatures:
                    if sig not in target.signatures:
                        target.signatures.append(sig)
                # Take higher urgency.
                target.urgency = max(target.urgency, pet.urgency)
                pet.status = "expired"  # Mark as consumed.
                merged_count += 1
            else:
                seen[key] = pet

        return merged_count

    # ── expiration ──────────────────────────────────────────────────────

    def expire_old(self, current_hours: float) -> int:
        """Expire petitions older than PETITION_MAX_AGE_HOURS.  Returns count expired."""
        count = 0
        for pet in self.petitions:
            if pet.status != "active":
                continue
            age = current_hours - pet.created_at_hours
            if age > PETITION_MAX_AGE_HOURS:
                pet.status = "expired"
                count += 1
        return count

    # ── ranking ─────────────────────────────────────────────────────────

    def rank_by_urgency(self) -> list[Petition]:
        """Return active petitions sorted by urgency (highest first), then
        by number of signatures."""
        active = [p for p in self.petitions if p.status == "active"]
        return sorted(
            active,
            key=lambda p: (p.urgency, len(p.signatures)),
            reverse=True,
        )

    # ── resolution ──────────────────────────────────────────────────────

    def approve(self, index: int) -> Petition | None:
        """Mark petition as approved.  Returns the petition or None."""
        if index < 0 or index >= len(self.petitions):
            return None
        pet = self.petitions[index]
        pet.status = "approved"
        return pet

    def dismiss(self, index: int) -> Petition | None:
        """Mark petition as dismissed.  Returns the petition or None."""
        if index < 0 or index >= len(self.petitions):
            return None
        pet = self.petitions[index]
        pet.status = "dismissed"
        return pet

    # ── serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "petitions": [
                {
                    "text": p.text,
                    "source": p.source,
                    "category": p.category,
                    "building_suggestion": p.building_suggestion,
                    "signatures": list(p.signatures),
                    "created_at_hours": p.created_at_hours,
                    "urgency": round(p.urgency, 3),
                    "status": p.status,
                }
                for p in self.petitions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PetitionManager:
        pm = cls()
        for p_data in data.get("petitions", []):
            pet = Petition(
                text=p_data.get("text", ""),
                source=p_data.get("source", ""),
                category=p_data.get("category", ""),
                building_suggestion=p_data.get("building_suggestion"),
                signatures=list(p_data.get("signatures", [])),
                created_at_hours=p_data.get("created_at_hours", 0.0),
                urgency=p_data.get("urgency", 0.5),
                status=p_data.get("status", "active"),
            )
            pm.petitions.append(pet)
        return pm
