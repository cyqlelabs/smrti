"""Petition system: detects community needs from Space_Culture and suggests buildings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from smrti.core.db import get_database
from smrti.core.embed import get_embedding_provider

# ── Need taxonomy ────────────────────────────────────────────────────
# Trigger phrases are embedded at startup and averaged into anchor
# vectors.  Detection uses cosine similarity — no hardcoded language
# matching.

NEED_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "food": {
        "triggers": ["hungry", "no food", "need bread", "starving"],
        "resolves_to": ["farm", "market", "bakery"],
    },
    "health": {
        "triggers": ["sick", "injured", "no doctor", "ill"],
        "resolves_to": ["clinic"],
    },
    "education": {
        "triggers": ["learn", "study", "school", "children bored"],
        "resolves_to": ["school", "library"],
    },
    "entertainment": {
        "triggers": ["bored", "nothing to do", "fun"],
        "resolves_to": ["tavern", "park", "theater"],
    },
    "commerce": {
        "triggers": ["buy", "sell", "trade", "goods", "shop"],
        "resolves_to": ["market", "workshop"],
    },
    "housing": {
        "triggers": ["homeless", "crowded", "no room"],
        "resolves_to": ["house"],
    },
    "spiritual": {
        "triggers": ["pray", "meaning", "community", "faith"],
        "resolves_to": ["church"],
    },
}

# ── Similarity / urgency thresholds ──────────────────────────────────
SIMILARITY_THRESHOLD = 0.4   # uses max-over-triggers, not mean-anchor
CONFIDENCE_THRESHOLD = 0.35
DEFAULT_MAX_AGE_HOURS = 720  # 30 sim-days


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class Petition:
    """A community petition requesting a new building."""

    need_category: str
    building_type: str
    petitioners: list[str]
    urgency: float
    evidence: list[str]
    created_at_hours: float
    status: str  # "pending" | "fulfilled" | "expired"
    rationale: str = ""


class PetitionManager:
    """Monitors Space_Culture for consensus needs and generates petitions."""

    def __init__(self, db_path: str, tenant_id: str) -> None:
        self._db_path = db_path
        self._tenant_id = tenant_id
        self._anchors: dict[str, np.ndarray] = {}  # category -> mean embedding
        self._petitions: list[Petition] = []
        self._fulfilled_types: set[str] = set()
        self._seen_atom_ids: set[str] = set()  # avoid re-scanning atoms

    # ── Anchor caching ───────────────────────────────────────────────

    def _cache_anchors(self) -> None:
        """Embed each category's trigger phrases and cache all vectors (not mean).

        Similarity uses max-over-triggers so a single strong signal fires the
        category even when the candidate text is not close to the mean anchor.
        Language-agnostic: embedding cosine similarity handles all locales.
        """
        if self._anchors:
            return
        embed = get_embedding_provider()
        for category, spec in NEED_CATEGORIES.items():
            vecs = embed.embed_batch(spec["triggers"])
            arr = np.array(vecs, dtype=np.float32)
            # Normalise rows so dot product == cosine similarity
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._anchors[category] = arr / norms  # shape (n_triggers, dim)

    # ── Culture scanning ─────────────────────────────────────────────

    def scan_culture(
        self,
        current_hours: float,
        existing_building_types: set[str],
    ) -> list[Petition]:
        """Scan new Space_Culture atoms for need signals.

        Returns newly created petitions (if any).
        """
        self._cache_anchors()

        db = get_database(self._db_path)
        rows = db.fetchall(
            "SELECT id, label, content, confidence, sti "
            "FROM atoms "
            "WHERE tenant_id = ? AND space = 'Space_Culture' "
            "  AND type IN ('belief', 'concept')",
            (self._tenant_id,),
        )

        # Collect only unseen atoms that pass the confidence gate
        candidates: list[dict[str, Any]] = []
        for row in rows:
            atom_id: str = row["id"]
            if atom_id in self._seen_atom_ids:
                continue
            self._seen_atom_ids.add(atom_id)
            if row["confidence"] < CONFIDENCE_THRESHOLD:
                continue
            candidates.append({
                "id": atom_id,
                "label": row["label"],
                "content": row["content"],
                "confidence": row["confidence"],
                "sti": row["sti"],
            })

        if not candidates:
            return []

        # Embed candidate texts
        embed = get_embedding_provider()
        texts = [c["content"] or c["label"] for c in candidates]
        embeddings = embed.embed_batch(texts)

        # Build combined set of types already placed or fulfilled
        unavailable = existing_building_types | self._fulfilled_types
        # Also exclude building types that already have a pending petition
        for p in self._petitions:
            if p.status == "pending":
                unavailable.add(p.building_type)

        new_petitions: list[Petition] = []

        # Match candidates against anchors (max-over-triggers)
        for candidate, vec in zip(candidates, embeddings):
            vec_arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(vec_arr)
            if norm > 0:
                vec_arr = vec_arr / norm
            for category, trigger_matrix in self._anchors.items():
                # trigger_matrix: (n_triggers, dim), each row normalised
                sims = trigger_matrix @ vec_arr  # (n_triggers,)
                sim = float(sims.max())
                if sim < SIMILARITY_THRESHOLD:
                    continue

                # Find first available building type
                building_type = None
                for bt in NEED_CATEGORIES[category]["resolves_to"]:
                    if bt not in unavailable:
                        building_type = bt
                        break
                if building_type is None:
                    continue

                urgency = min(1.0, max(0.0,
                    candidate["confidence"] * 0.6 + candidate["sti"] * 0.4,
                ))

                # Try to extract petitioner names from promoted_from metadata
                petitioners = self._extract_petitioners(candidate["id"])

                petition = Petition(
                    need_category=category,
                    building_type=building_type,
                    petitioners=petitioners,
                    urgency=urgency,
                    evidence=[candidate["label"]],
                    created_at_hours=current_hours,
                    status="pending",
                )
                self._petitions.append(petition)
                new_petitions.append(petition)
                # Mark this building type as claimed
                unavailable.add(building_type)

        return new_petitions

    def _extract_petitioners(self, atom_id: str) -> list[str]:
        """Best-effort extraction of agent names from bridge provenance."""
        db = get_database(self._db_path)
        row = db.fetchone(
            "SELECT metadata FROM atoms WHERE id = ?",
            (atom_id,),
        )
        if not row or not row["metadata"]:
            return []
        try:
            meta = json.loads(row["metadata"])
        except (json.JSONDecodeError, TypeError):
            return []
        promoted_from = meta.get("promoted_from", "")
        if not promoted_from:
            return []
        # Bridge spaces are named Agent_Space_{a}_x_Agent_Space_{b}
        parts = promoted_from.split("_x_")
        names: list[str] = []
        for part in parts:
            if part.startswith("Agent_Space_"):
                names.append(part.removeprefix("Agent_Space_"))
        return names

    # ── Bootstrap ────────────────────────────────────────────────────

    def seed_needs(
        self,
        existing_building_types: set[str],
        current_hours: float = 0.0,
        max_seeds: int = 3,
    ) -> list[Petition]:
        """Create starter petitions for missing critical building types.

        Called once at engine startup so the player immediately sees what the
        town needs — before Space_Culture has any atoms from bridge discovery.
        Skips categories already covered by existing buildings or pending petitions.
        """
        # Priority order: food first, then housing, health, etc.
        priority_order = ["food", "housing", "health", "education", "entertainment",
                          "commerce", "spiritual"]

        unavailable = set(existing_building_types) | self._fulfilled_types
        for p in self._petitions:
            if p.status == "pending":
                unavailable.add(p.building_type)

        seeded: list[Petition] = []
        for category in priority_order:
            if len(seeded) >= max_seeds:
                break
            spec = NEED_CATEGORIES.get(category)
            if not spec:
                continue
            building_type = next(
                (bt for bt in spec["resolves_to"] if bt not in unavailable),
                None,
            )
            if building_type is None:
                continue
            petition = Petition(
                need_category=category,
                building_type=building_type,
                petitioners=[],
                urgency=0.4,
                evidence=[f"Community lacks a {building_type}"],
                created_at_hours=current_hours,
                status="pending",
                rationale=f"The town has no {building_type} yet.",
            )
            self._petitions.append(petition)
            seeded.append(petition)
            unavailable.add(building_type)

        return seeded

    # ── Mutation ─────────────────────────────────────────────────────

    def fulfill(self, petition_idx: int) -> None:
        """Mark a petition as fulfilled."""
        if 0 <= petition_idx < len(self._petitions):
            p = self._petitions[petition_idx]
            p.status = "fulfilled"
            self._fulfilled_types.add(p.building_type)

    def dismiss(self, petition_idx: int) -> None:
        """Mark a petition as expired (dismissed by player/engine)."""
        if 0 <= petition_idx < len(self._petitions):
            self._petitions[petition_idx].status = "expired"

    def expire_old(
        self,
        current_hours: float,
        max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    ) -> None:
        """Expire pending petitions older than *max_age_hours*."""
        for p in self._petitions:
            if p.status != "pending":
                continue
            if current_hours - p.created_at_hours > max_age_hours:
                p.status = "expired"

    # ── Queries ──────────────────────────────────────────────────────

    def pending(self) -> list[Petition]:
        """Return all pending petitions."""
        return [p for p in self._petitions if p.status == "pending"]

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialise all petitions for the REST/WebSocket API."""
        out: list[dict[str, Any]] = []
        for i, p in enumerate(self._petitions):
            out.append({
                "idx": i,
                "need_category": p.need_category,
                "building_type": p.building_type,
                "petitioners": p.petitioners,
                "urgency": p.urgency,
                "evidence": p.evidence,
                "created_at_hours": p.created_at_hours,
                "status": p.status,
                "rationale": p.rationale,
            })
        return out
