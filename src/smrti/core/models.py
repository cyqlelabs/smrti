from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AtomType(str, Enum):
    CONCEPT = "concept"
    BELIEF = "belief"
    EPISODE = "episode"
    GOAL = "goal"
    RELATION = "relation"


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    ROLE = "role"
    TOOL = "tool"  # legacy rows only — NER now maps software tools to 'technology'
    TECHNOLOGY = "technology"
    SKILL = "skill"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    LOCATION = "location"
    EVENT = "event"
    TOPIC = "topic"
    MEDIA = "media"
    HEALTH = "health"
    CONCEPT = "concept"
    GOAL = "goal"
    PRONOUN = "pronoun"


# A belief asserted at this probability or above is the caller's standing
# testimony rather than an estimate that might grow stale. It is born already
# certain and confidence decay leaves it alone, because "permanent" that does
# not survive the passage of time means nothing. Anything below is an ordinary
# claim, born unsure and earning confidence through evidence.
PERMANENT_PROBABILITY = 0.95


class TruthValue(BaseModel):
    probability: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    def merge(self, other: TruthValue) -> TruthValue:
        """PLN revision rule: combine two independent truth estimates."""
        w_a = self.confidence / (1.0 - self.confidence + 1e-9)
        w_b = other.confidence / (1.0 - other.confidence + 1e-9)
        w_total = w_a + w_b
        if w_total < 1e-9:
            return TruthValue(
                probability=(self.probability + other.probability) / 2.0,
                confidence=0.0,
            )
        merged_s = (w_a * self.probability + w_b * other.probability) / w_total
        merged_c = w_total / (w_total + 1.0)
        return TruthValue(probability=merged_s, confidence=min(1.0, merged_c))


class AttentionValue(BaseModel):
    sti: float = Field(0.0, ge=0.0)
    lti: float = Field(0.0, ge=0.0)


class Valence(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    intensity: float = Field(0.0, ge=0.0, le=1.0)


class Atom(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: AtomType
    label: str
    content: Optional[str] = None
    truth: TruthValue = Field(default_factory=TruthValue)
    attention: AttentionValue = Field(default_factory=AttentionValue)
    valence: Valence = Field(default_factory=Valence)
    entity_type: Optional[EntityType] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    relation: Optional[str] = None
    tenant_id: str = "default"
    space: str = "default"
    metadata: dict = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    atom_id: str
    observed_probability: float
    weight: float = 1.0
    source_episode_id: Optional[str] = None
    tenant_id: str = "default"
    space: str = "default"


class RecallResult(BaseModel):
    atom: Atom
    salience: float
    similarity: float


class EpochResult(BaseModel):
    beliefs_updated: int
    atoms_decayed: int
    atoms_pruned: int
    lti_promoted: int
    new_connections: int
    contradictions_resolved: int
    orphans_healed: int = 0
    bridges_created: int = 0


class AtomPair(BaseModel):
    """A matched pair of atoms across two spaces with their similarity score."""
    atom_a: Atom
    atom_b: Atom
    similarity: float


class SpaceOverlap(BaseModel):
    """Result of computing overlap between two spaces."""
    space_a: str
    space_b: str
    jaccard: float = 0.0
    pairs: list[AtomPair] = Field(default_factory=list)

    @property
    def bridge_space_name(self) -> str:
        """Canonical bridge space name (sorted so A∩B == B∩A)."""
        a, b = sorted([self.space_a, self.space_b])
        return f"{a}_x_{b}"


class SpaceSetResult(BaseModel):
    """Result of a set operation on spaces."""
    operation: str
    spaces: list[str]
    atoms: list[Atom] = Field(default_factory=list)
    overlap: Optional[SpaceOverlap] = None


def _safe_entity_type(value: str | None) -> EntityType | None:
    if not value:
        return None
    try:
        return EntityType(value)
    except ValueError:
        return EntityType.CONCEPT


def _clamp(value: float | None, lo: float, hi: float, default: float = 0.0) -> float:
    if value is None:
        value = default
    return min(hi, max(lo, value))


def atom_from_row(row) -> Atom:
    d = dict(row)
    return Atom(
        id=d["id"],
        type=AtomType(d["type"]),
        label=d["label"],
        content=d.get("content"),
        truth=TruthValue(
            probability=_clamp(d.get("probability"), 0.0, 1.0, 0.5),
            confidence=_clamp(d.get("confidence"), 0.0, 1.0),
        ),
        attention=AttentionValue(
            sti=max(0.0, d.get("sti") or 0.0),
            lti=_clamp(d.get("lti"), 0.0, 1.0),
        ),
        valence=Valence(
            valence=_clamp(d.get("valence"), -1.0, 1.0),
            intensity=_clamp(d.get("intensity"), 0.0, 1.0),
        ),
        entity_type=_safe_entity_type(d.get("entity_type")),
        source_id=d.get("source_id"),
        target_id=d.get("target_id"),
        relation=d.get("relation"),
        tenant_id=d.get("tenant_id", "default"),
        space=d.get("space", "default"),
        metadata=json.loads(d.get("metadata") or "{}"),
        created_at=d.get("created_at"),
        updated_at=d.get("updated_at"),
    )
