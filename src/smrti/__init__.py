"""Smrti: AtomSpace-inspired memory + personality engine for AI agents."""

from __future__ import annotations

import os
import re
import threading

try:
    from smrti._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.0.dev0"

from smrti.core.db import Database, fts_delete, get_database
from smrti.core.embed import EmbeddingProvider, get_embedding_provider
from smrti.core.atomspace import AtomSpace
from smrti.core.models import (
    PERMANENT_PROBABILITY,
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
from smrti.core.provenance import ATOM_METADATA_JSON, SOURCE_AGENT, VALENCE_STATED
from smrti.extraction.sentiment import estimate_valence
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


def _write_metadata(source: str, valence_stated: bool) -> dict:
    """Atom metadata for a caller-issued write.

    Only what is true is recorded: an absent source reads as the user, and an
    absent valence flag as estimated, so the common write stays an empty
    object exactly as earlier releases stored it.
    """
    metadata = {}
    if source == SOURCE_AGENT:
        metadata["source"] = SOURCE_AGENT
    if valence_stated:
        metadata[VALENCE_STATED] = True
    return metadata


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
        temporal: bool = False,
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
        # Copied, not aliased: the caller's list must not be able to change
        # this instance's read overlay after construction.
        self.read_spaces = (
            list(read_spaces) if read_spaces is not None else [write_space]
        )
        self.extractor = extractor
        # Resolving relative dates costs an NER pass per write, so the library
        # facade leaves it off and every server mode turns it on (see
        # SMRTI_TEMPORAL). A direct caller who wants it asks for it.
        self._temporal = temporal
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
        valence: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        if self.is_ignored(content):
            return ""
        content = self._resolve_deixis(content)
        valence, stated = self._resolve_valence(content, valence)
        atom = Atom(
            type=AtomType(type),
            label=content[:100],
            content=content,
            truth=TruthValue(probability=probability, confidence=0.5),
            valence=Valence(valence=valence, intensity=abs(valence)),
            tenant_id=self.tenant_id,
            space=self.write_space,
            metadata={**_write_metadata("", stated), **(metadata or {})},
        )
        return self.atomspace.add_atom(atom)

    def _resolve_deixis(self, content: str) -> str:
        """Annotate relative dates in *content* with what they resolve to.

        Runs before the text is embedded and before the valence estimate, so
        the resolved date is part of what the atom stores and what any search
        of it matches. Off unless the caller asked for it, and never fatal:
        text nothing could resolve is stored exactly as written.
        """
        if not self._temporal:
            return content
        from smrti.extraction.temporal import annotate

        return annotate(content)

    def _resolve_valence(self, content: str, valence: float | None) -> tuple[float, bool]:
        """Return the atom's tone and whether the caller was the one who set it.

        Estimation lives here rather than in each caller so the answer to "did
        someone report this, or did a model read the mood of the words?" is
        decided once. Only the first can become a behavioral constraint at
        recall, and a caller that had to remember to say so would forget.
        """
        if valence is None:
            return estimate_valence(content, self.embed), False
        return valence, True

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
        valence: float | None = None,
        source: str = "user",
    ) -> str:
        # A permanent assertion is born certain. Starting it at the ordinary
        # 0.3 leaves a fact the caller stated as settled ranked below the
        # conversational froth stored beside it, which reads to the caller as
        # never having stored it at all.
        confidence = probability if probability >= PERMANENT_PROBABILITY else 0.3
        statement = self._resolve_deixis(statement)
        valence, stated = self._resolve_valence(statement, valence)
        atom = Atom(
            type=AtomType.BELIEF,
            label=statement[:100],
            content=statement,
            truth=TruthValue(probability=probability, confidence=confidence),
            valence=Valence(valence=valence, intensity=abs(valence)),
            tenant_id=self.tenant_id,
            space=self.write_space,
            metadata=_write_metadata(source, stated),
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
        """Soften memories matching query by reducing their confidence.

        Each atom is also stamped as deliberately sunk: the epoch lifts a
        drowned permanent belief back to its asserted probability, and the
        stamp is what keeps that lift from undoing a forget.
        """
        results = self.recall(query=query, top_k=top_k)
        forgotten = []
        for r in results:
            if r.atom.space != self.write_space:
                continue
            self.db.execute(
                f"""UPDATE atoms SET
                        confidence = confidence * 0.3,
                        metadata = json_set({ATOM_METADATA_JSON},
                                            '$.forgotten', json('true'))
                    WHERE id = ? AND tenant_id = ? AND space = ?""",
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

    # Deleting in chunks keeps the generated ``IN (...)`` lists well under
    # SQLite's variable limit on spaces holding tens of thousands of atoms.
    _CLEAR_CHUNK = 400

    def clear_space(self) -> int:
        """Hard-delete all atoms, evidence, and aliases in write_space. Returns deleted atom count."""
        ids = [
            row["id"]
            for row in self.db.fetchall(
                "SELECT id FROM atoms WHERE tenant_id=? AND space=?",
                (self.tenant_id, self.write_space),
            )
        ]
        count = len(ids)

        # Rows outside this space can still reference its atoms: a bridge space
        # links back to both parents, and evidence or aliases may have been
        # filed elsewhere. ``atoms.source_id``/``target_id``, ``evidence`` and
        # ``aliases`` are all real foreign keys with enforcement on, so those
        # references have to go first or the delete aborts with an integrity
        # error and the space is left half-cleared. The cascade stays inside
        # this tenant — another tenant can never hold a reference to begin with.
        for start in range(0, count, self._CLEAR_CHUNK):
            chunk = ids[start : start + self._CLEAR_CHUNK]
            ph = ",".join("?" * len(chunk))
            self.db.execute_batch([
                (
                    f"DELETE FROM atoms WHERE type = 'relation' AND tenant_id = ? "
                    f"AND (source_id IN ({ph}) OR target_id IN ({ph}))",
                    (self.tenant_id, *chunk, *chunk),
                ),
                (f"DELETE FROM vec_atoms WHERE atom_id IN ({ph})", tuple(chunk)),
                *fts_delete(self.db, chunk),
                (f"DELETE FROM evidence WHERE atom_id IN ({ph})", tuple(chunk)),
                (f"DELETE FROM aliases WHERE atom_id IN ({ph})", tuple(chunk)),
                (f"DELETE FROM atoms WHERE id IN ({ph})", tuple(chunk)),
            ])

        # Relation atoms are in ``ids`` too, but rows filed against the space
        # rather than against a surviving atom still need sweeping.
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
            "SELECT DISTINCT space FROM atoms WHERE tenant_id = ? ORDER BY space",
            (self.tenant_id,),
        )
        return [r["space"] for r in rows]

    def close(self) -> None:
        pass  # db is registry-owned; lifetime is process-scoped
