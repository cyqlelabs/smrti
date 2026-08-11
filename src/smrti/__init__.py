"""Smrti: AtomSpace-inspired memory + personality engine for AI agents."""

from __future__ import annotations

import os
import re
import threading

try:
    from smrti._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.0.dev0"

from smrti.core.db import Database, get_database
from smrti.core.embed import EmbeddingProvider, get_embedding_provider
from smrti.core.atomspace import AtomSpace
from smrti.core.models import (
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
from smrti.extraction.resolve import EntityResolver
from smrti.retrieval.fan_out import retrieve
from smrti.evolution.epoch import run_epoch
from smrti.personality.params import PersonalityProfile, load_preset
from smrti.spaces.set_ops import (
    space_overlap as _space_overlap,
    space_intersection as _space_intersection,
    space_difference as _space_difference,
    space_union as _space_union,
    space_symmetric_difference as _space_symmetric_difference,
)
from smrti.spaces.emergence import materialize_bridge as _materialize_bridge

# Per-(tenant_id, space) locks so concurrent reflect() calls (background loop +
# REST /reflect) cannot interleave epochs on the same space.
_reflect_locks: dict[tuple[str, str], threading.Lock] = {}
_reflect_locks_guard = threading.Lock()


def _get_reflect_lock(tenant_id: str, space: str) -> threading.Lock:
    key = (tenant_id, space)
    with _reflect_locks_guard:
        lock = _reflect_locks.get(key)
        if lock is None:
            lock = _reflect_locks[key] = threading.Lock()
    return lock


class Smrti:
    def __init__(
        self,
        db_path: str = "~/.smrti/memory.db",
        personality: str = "balanced",
        tenant_id: str = "default",
        write_space: str = "default",
        read_spaces: list[str] | None = None,
        extractor=None,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        db_path = os.path.expanduser(db_path)
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.db = get_database(db_path)
        self.embed = get_embedding_provider()
        self.atomspace = AtomSpace(self.db, self.embed)
        self.tenant_id = tenant_id
        self.write_space = write_space
        self.read_spaces = read_spaces if read_spaces is not None else [write_space]
        self.extractor = extractor
        try:
            self._ignore_re: list[re.Pattern] = [
                re.compile(p, re.MULTILINE) for p in (ignore_patterns or [])
            ]
        except re.error as exc:
            raise ValueError(f"SMRTI_IGNORE_PATTERNS contains an invalid regex: {exc}") from exc
        self._ensure_personality(personality)

    def _ensure_personality(self, preset_name: str) -> None:
        existing = self.db.fetchone(
            "SELECT preset_name FROM personality WHERE tenant_id = ? AND space = ?",
            (self.tenant_id, self.write_space),
        )
        if existing:
            if existing["preset_name"] != preset_name and os.environ.get("SMRTI_PERSONALITY"):
                self.set_personality(preset_name)
            return
        if not existing:
            profile = load_preset(preset_name)
            self.db.execute(
                """
                INSERT OR IGNORE INTO personality (
                    tenant_id, space, confidence_decay_rate, confidence_update_lr,
                    min_confidence_to_surface, sti_decay_rate, sti_boost_on_access,
                    sti_propagation_factor, lti_promotion_threshold, lti_decay_rate,
                    agent_source_trust, valence_weight,
                    valence_propagation, mood_inertia, w_similarity, w_sti, w_confidence,
                    w_lti, w_valence, preset_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.tenant_id,
                    self.write_space,
                    profile.confidence_decay_rate,
                    profile.confidence_update_lr,
                    profile.min_confidence_to_surface,
                    profile.sti_decay_rate,
                    profile.sti_boost_on_access,
                    profile.sti_propagation_factor,
                    profile.lti_promotion_threshold,
                    profile.lti_decay_rate,
                    profile.agent_source_trust,
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
            # If explicitly configured, win the race: force-update even if INSERT was ignored
            if os.environ.get("SMRTI_PERSONALITY"):
                self.set_personality(preset_name)

    def is_ignored(self, content: str) -> bool:
        return bool(self._ignore_re) and any(rx.search(content) for rx in self._ignore_re)

    def remember(
        self,
        content: str,
        type: str = "episode",
        probability: float = 0.8,
        valence: float = 0.0,
        metadata: dict | None = None,
    ) -> str:
        if self.is_ignored(content):
            return ""
        atom = Atom(
            type=AtomType(type),
            label=content[:100],
            content=content,
            truth=TruthValue(probability=probability, confidence=0.5),
            valence=Valence(valence=valence, intensity=abs(valence)),
            tenant_id=self.tenant_id,
            space=self.write_space,
            metadata=metadata or {},
        )
        return self.atomspace.add_atom(atom)

    def recall(
        self,
        query: str,
        top_k: int = 10,
        min_confidence: float = 0.1,
        read_spaces: list[str] | None = None,
    ) -> list:
        spaces = read_spaces if read_spaces is not None else self.read_spaces
        return retrieve(
            query,
            self.tenant_id,
            spaces,
            self.db,
            self.embed,
            write_space=self.write_space,
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
            tenant_id=self.tenant_id,
            space=self.write_space,
        )
        atom_id = self.atomspace.add_atom(atom)
        if evidence:
            ev = Evidence(
                atom_id=atom_id,
                observed_probability=probability,
                tenant_id=self.tenant_id,
                space=self.write_space,
            )
            self.atomspace.add_evidence(ev)
        return atom_id

    def reflect(self) -> EpochResult:
        with _get_reflect_lock(self.tenant_id, self.write_space):
            return run_epoch(self.tenant_id, self.write_space, self.db, self.embed)

    def forget(self, query: str, top_k: int = 5) -> list[str]:
        """Soften memories matching query by reducing their confidence."""
        results = self.recall(query=query, top_k=top_k)
        forgotten = []
        for r in results:
            if r.atom.space != self.write_space:
                continue
            self.db.execute(
                "UPDATE atoms SET confidence = confidence * 0.3 WHERE id = ? AND tenant_id = ? AND space = ?",
                (r.atom.id, self.tenant_id, self.write_space),
            )
            forgotten.append(r.atom.label)
        return forgotten

    def set_personality(self, preset_name: str) -> None:
        profile = load_preset(preset_name)
        self.set_personality_profile(profile, preset_name)

    def set_personality_profile(self, profile: PersonalityProfile, preset_name: str = "custom") -> None:
        """Apply an arbitrary PersonalityProfile to this space's personality row."""
        self.db.execute(
            """
            UPDATE personality SET
                confidence_decay_rate=?, confidence_update_lr=?, min_confidence_to_surface=?,
                sti_decay_rate=?, sti_boost_on_access=?, sti_propagation_factor=?,
                lti_promotion_threshold=?, lti_decay_rate=?, agent_source_trust=?,
                valence_weight=?, valence_propagation=?,
                mood_inertia=?, w_similarity=?, w_sti=?, w_confidence=?, w_lti=?, w_valence=?,
                preset_name=?
            WHERE tenant_id=? AND space=?
            """,
            (
                profile.confidence_decay_rate,
                profile.confidence_update_lr,
                profile.min_confidence_to_surface,
                profile.sti_decay_rate,
                profile.sti_boost_on_access,
                profile.sti_propagation_factor,
                profile.lti_promotion_threshold,
                profile.lti_decay_rate,
                profile.agent_source_trust,
                profile.valence_weight,
                profile.valence_propagation,
                profile.mood_inertia,
                profile.w_similarity,
                profile.w_sti,
                profile.w_confidence,
                profile.w_lti,
                profile.w_valence,
                preset_name,
                self.tenant_id,
                self.write_space,
            ),
        )

    def status(self) -> dict:
        total = self.db.fetchone(
            "SELECT COUNT(*) as n FROM atoms WHERE tenant_id=? AND space=?",
            (self.tenant_id, self.write_space),
        )
        by_type = self.db.fetchall(
            "SELECT type, COUNT(*) as n FROM atoms WHERE tenant_id=? AND space=? GROUP BY type",
            (self.tenant_id, self.write_space),
        )
        personality = self.db.fetchone(
            "SELECT * FROM personality WHERE tenant_id=? AND space=?",
            (self.tenant_id, self.write_space),
        )
        return {
            "total_atoms": total["n"] if total else 0,
            "by_type": {row["type"]: row["n"] for row in by_type},
            "personality": dict(personality) if personality else {},
        }

    def clear_space(self) -> int:
        """Hard-delete all atoms, evidence, and aliases in write_space. Returns deleted atom count."""
        count_row = self.db.fetchone(
            "SELECT COUNT(*) as n FROM atoms WHERE tenant_id=? AND space=?",
            (self.tenant_id, self.write_space),
        )
        count = count_row["n"] if count_row else 0

        ids = self.db.fetchall(
            "SELECT id FROM atoms WHERE tenant_id=? AND space=?",
            (self.tenant_id, self.write_space),
        )
        for row in ids:
            self.db.execute("DELETE FROM vec_atoms WHERE atom_id=?", (row["id"],))

        self.db.execute(
            "DELETE FROM evidence WHERE tenant_id=? AND space=?",
            (self.tenant_id, self.write_space),
        )
        self.db.execute(
            "DELETE FROM aliases WHERE tenant_id=? AND space=?",
            (self.tenant_id, self.write_space),
        )
        self.db.execute(
            "DELETE FROM atoms WHERE tenant_id=? AND space=?",
            (self.tenant_id, self.write_space),
        )
        return count

    # ── Space set theory ──────────────────────────────────────────────

    def space_overlap(self, other_space: str, threshold: float = 0.85):
        """Compute overlap (Jaccard + matched pairs) between write_space and another space."""
        return _space_overlap(self.tenant_id, self.write_space, other_space, self.db, threshold, self.embed)

    def space_intersection(self, other_space: str, threshold: float = 0.85):
        """Return atoms that exist in both write_space and other_space."""
        return _space_intersection(self.tenant_id, self.write_space, other_space, self.db, threshold, self.embed)

    def space_difference(self, other_space: str, threshold: float = 0.85):
        """Return atoms in write_space that have no match in other_space."""
        return _space_difference(self.tenant_id, self.write_space, other_space, self.db, threshold, self.embed)

    def space_union(self, other_space: str, threshold: float = 0.85):
        """Return deduplicated union of atoms from write_space and other_space."""
        return _space_union(self.tenant_id, self.write_space, other_space, self.db, threshold, self.embed)

    def space_symmetric_difference(self, other_space: str, threshold: float = 0.85):
        """Return atoms that are in one space but not the other."""
        return _space_symmetric_difference(self.tenant_id, self.write_space, other_space, self.db, threshold, self.embed)

    def materialize_bridge(self, other_space: str, threshold: float = 0.85, min_jaccard: float = 0.1) -> int:
        """Compute overlap and materialize a bridge space if Jaccard >= min_jaccard.

        Returns the number of bridge atoms created.
        """
        overlap = _space_overlap(self.tenant_id, self.write_space, other_space, self.db, threshold, self.embed)
        return _materialize_bridge(overlap, self.tenant_id, self.db, self.embed, self.atomspace, min_jaccard)

    def list_spaces(self) -> list[str]:
        """Return all spaces for this tenant."""
        rows = self.db.fetchall(
            "SELECT DISTINCT space FROM atoms WHERE tenant_id = ?",
            (self.tenant_id,),
        )
        return [r["space"] for r in rows]

    def close(self) -> None:
        pass  # db is registry-owned; lifetime is process-scoped
