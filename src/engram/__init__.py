"""Engram: AtomSpace-inspired memory + personality engine for AI agents."""

from __future__ import annotations

import os

from engram.core.db import Database
from engram.core.embed import EmbeddingProvider
from engram.core.atomspace import AtomSpace
from engram.core.models import (
    Atom,
    AtomType,
    AttentionValue,
    EntityType,
    Evidence,
    EpochResult,
    RecallResult,
    TruthValue,
    Valence,
    atom_from_row,
)
from engram.extraction.resolve import EntityResolver
from engram.retrieval.fan_out import retrieve
from engram.evolution.epoch import run_epoch
from engram.personality.params import PersonalityProfile, load_preset


class Engram:
    def __init__(
        self,
        db_path: str = "~/.engram/memory.db",
        personality: str = "balanced",
        agent_id: str = "default",
        extractor=None,
    ) -> None:
        db_path = os.path.expanduser(db_path)
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db = Database(db_path)
        self.db.initialize()
        self.embed = EmbeddingProvider()
        self.atomspace = AtomSpace(self.db, self.embed)
        self.agent_id = agent_id
        self.extractor = extractor
        self._ensure_personality(personality)

    def _ensure_personality(self, preset_name: str) -> None:
        existing = self.db.fetchone(
            "SELECT agent_id FROM personality WHERE agent_id = ?",
            (self.agent_id,),
        )
        if not existing:
            profile = load_preset(preset_name)
            self.db.execute(
                """
                INSERT INTO personality (
                    agent_id, confidence_decay_rate, confidence_update_lr,
                    min_confidence_to_surface, sti_decay_rate, sti_boost_on_access,
                    sti_propagation_factor, lti_promotion_threshold, valence_weight,
                    valence_propagation, mood_inertia, w_similarity, w_sti, w_confidence,
                    w_lti, w_valence, preset_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.agent_id,
                    profile.confidence_decay_rate,
                    profile.confidence_update_lr,
                    profile.min_confidence_to_surface,
                    profile.sti_decay_rate,
                    profile.sti_boost_on_access,
                    profile.sti_propagation_factor,
                    profile.lti_promotion_threshold,
                    profile.valence_weight,
                    profile.valence_propagation,
                    profile.mood_inertia,
                    profile.w_similarity,
                    profile.w_sti,
                    profile.w_confidence,
                    profile.w_lti,
                    profile.w_valence,
                    preset_name,
                ),
            )

    def remember(
        self,
        content: str,
        type: str = "episode",
        probability: float = 0.8,
        valence: float = 0.0,
    ) -> str:
        atom = Atom(
            type=AtomType(type),
            label=content[:100],
            content=content,
            truth=TruthValue(probability=probability, confidence=0.5),
            valence=Valence(valence=valence, intensity=abs(valence)),
            agent_id=self.agent_id,
        )
        return self.atomspace.add_atom(atom)

    def recall(
        self,
        query: str,
        top_k: int = 10,
        min_confidence: float = 0.1,
    ) -> list:
        return retrieve(
            query,
            self.agent_id,
            self.db,
            self.embed,
            top_k=top_k,
            min_confidence=min_confidence,
        )

    def believe(
        self,
        statement: str,
        probability: float,
        evidence: str = None,
    ) -> str:
        atom = Atom(
            type=AtomType.BELIEF,
            label=statement[:100],
            content=statement,
            truth=TruthValue(probability=probability, confidence=0.3),
            agent_id=self.agent_id,
        )
        atom_id = self.atomspace.add_atom(atom)
        if evidence:
            ev = Evidence(
                atom_id=atom_id,
                observed_probability=probability,
                agent_id=self.agent_id,
            )
            self.atomspace.add_evidence(ev)
        return atom_id

    def reflect(self) -> EpochResult:
        return run_epoch(self.agent_id, self.db, self.embed)

    def set_personality(self, preset_name: str) -> None:
        profile = load_preset(preset_name)
        self.db.execute(
            """
            UPDATE personality SET
                confidence_decay_rate=?, confidence_update_lr=?, min_confidence_to_surface=?,
                sti_decay_rate=?, sti_boost_on_access=?, sti_propagation_factor=?,
                lti_promotion_threshold=?, valence_weight=?, valence_propagation=?,
                mood_inertia=?, w_similarity=?, w_sti=?, w_confidence=?, w_lti=?, w_valence=?,
                preset_name=?
            WHERE agent_id=?
            """,
            (
                profile.confidence_decay_rate,
                profile.confidence_update_lr,
                profile.min_confidence_to_surface,
                profile.sti_decay_rate,
                profile.sti_boost_on_access,
                profile.sti_propagation_factor,
                profile.lti_promotion_threshold,
                profile.valence_weight,
                profile.valence_propagation,
                profile.mood_inertia,
                profile.w_similarity,
                profile.w_sti,
                profile.w_confidence,
                profile.w_lti,
                profile.w_valence,
                preset_name,
                self.agent_id,
            ),
        )

    def status(self) -> dict:
        total = self.db.fetchone(
            "SELECT COUNT(*) as n FROM atoms WHERE agent_id=?",
            (self.agent_id,),
        )
        by_type = self.db.fetchall(
            "SELECT type, COUNT(*) as n FROM atoms WHERE agent_id=? GROUP BY type",
            (self.agent_id,),
        )
        personality = self.db.fetchone(
            "SELECT * FROM personality WHERE agent_id=?",
            (self.agent_id,),
        )
        return {
            "total_atoms": total["n"] if total else 0,
            "by_type": {row["type"]: row["n"] for row in by_type},
            "personality": dict(personality) if personality else {},
        }

    def close(self) -> None:
        self.db.close()
