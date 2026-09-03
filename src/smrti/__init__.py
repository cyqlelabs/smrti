"""Smrti: AtomSpace-inspired memory + personality engine for AI agents."""

from __future__ import annotations

import os
import re
import threading

try:
    from smrti._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.0.dev0"

from smrti.core.db import Database, fts_delete, get_database, vec_delete
from smrti.core.embed import EmbeddingProvider, get_embedding_provider
from smrti.core.atomspace import AtomSpace
from smrti.core.models import (
    INITIAL_CONFIDENCE,
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
from smrti.core.provenance import (
    ATOM_METADATA_JSON,
    SOURCE_AGENT,
    SOURCE_USER,
    VALENCE_STATED,
)
from smrti.extraction.sentiment import estimate_valence
from smrti.extraction.resolve import EntityResolver
from smrti.retrieval.fan_out import retrieve
from smrti.evolution.epoch import run_epoch
from smrti.evolution.reinforcement import DEFAULT_WEIGHT as _REINFORCE_WEIGHT, reinforce_atoms
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
        # Memory operations since the last consolidation. The reflect loop
        # reads it to skip a space nobody used: an epoch is a unit of the
        # agent's activity, not of the server's uptime.
        self._ops_since_reflect = 0
        try:
            self._ignore_re: list[re.Pattern] = [
                re.compile(p, re.MULTILINE) for p in (ignore_patterns or [])
            ]
        except re.error as exc:
            raise ValueError(f"SMRTI_IGNORE_PATTERNS contains an invalid regex: {exc}") from exc
        self._ensure_personality(personality)

    @property
    def ops_since_reflect(self) -> int:
        """Remember, believe, recall, reinforce and forget calls since the last epoch."""
        return self._ops_since_reflect

    def _note_activity(self) -> None:
        self._ops_since_reflect += 1

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
        intensity: float | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store *content* as an atom of the given type.

        ``type="belief"`` is handed to :meth:`believe`, so a belief is the same
        atom whichever door it came through — born unsure, or born certain
        when asserted at ``PERMANENT_PROBABILITY``. It used to be a different
        one: confidence 0.5 and no permanence, depending on which method the
        caller happened to pick.

        ``intensity`` is how strongly the tone is felt, independent of its
        sign; left unset it is ``|valence|``, which is all the engine can read
        from a bare number.
        """
        if self.is_ignored(content):
            return ""
        if type == "belief":
            meta = dict(metadata or {})
            return self.believe(
                content,
                probability,
                valence=valence,
                intensity=intensity,
                source=meta.get("source", SOURCE_USER),
                metadata=meta,
            )
        self._note_activity()
        content, temporal = self._resolve_deixis(content)
        valence, stated = self._resolve_valence(content, valence)
        atom = Atom(
            type=AtomType(type),
            label=content[:100],
            content=content,
            truth=TruthValue(
                probability=probability,
                confidence=INITIAL_CONFIDENCE.get(type, INITIAL_CONFIDENCE["episode"]),
            ),
            valence=Valence(
                valence=valence, intensity=self._resolve_intensity(valence, intensity)
            ),
            tenant_id=self.tenant_id,
            space=self.write_space,
            metadata=self._with_temporal(
                {**_write_metadata("", stated), **(metadata or {})}, temporal
            ),
        )
        return self.atomspace.add_atom(atom)

    def _resolve_deixis(self, content: str) -> tuple[str, list[dict]]:
        """Annotate relative dates in *content* and return what they resolved to.

        Runs before the text is embedded and before the valence estimate, so
        the resolved date is part of what the atom stores and what any search
        of it matches. The resolutions are also returned so they can be filed
        in ``metadata.$.temporal``, the one place recall renders dates from;
        the extraction model adds to that list later, never replaces it. Off
        unless the caller asked for it, and never fatal: text nothing could
        resolve is stored exactly as written.
        """
        if not self._temporal:
            return content, []
        from smrti.extraction.temporal import resolve

        return resolve(content)

    @staticmethod
    def _with_temporal(metadata: dict, temporal: list[dict]) -> dict:
        if temporal and "temporal" not in metadata:
            metadata["temporal"] = temporal
        return metadata

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

    @staticmethod
    def _resolve_intensity(valence: float, intensity: float | None) -> float:
        """How strongly the tone is felt: the caller's number, else ``|valence|``.

        The two are separate dimensions in the model — a mild dislike and a
        line never to cross can share a sign — but only a caller can tell
        them apart, and one that says nothing gets the magnitude of the tone.
        """
        if intensity is None:
            return abs(valence)
        return max(0.0, min(1.0, float(intensity)))

    def _surfacing_floor(self) -> float:
        row = self.db.fetchone(
            "SELECT min_confidence_to_surface AS f FROM personality WHERE tenant_id = ? AND space = ?",
            (self.tenant_id, self.write_space),
        )
        return 0.1 if row is None or row["f"] is None else float(row["f"])

    def recall(
        self,
        query: str,
        top_k: int = 10,
        min_confidence: float | None = None,
        read_spaces: list[str] | None = None,
        boost: bool = True,
    ) -> list:
        """Recall memories relevant to *query*, most salient first.

        ``min_confidence`` left unset means the personality's
        ``min_confidence_to_surface``: the floor the preset promises is the
        floor a caller gets. ``boost=False`` reads without raising the
        attention of what came back.
        """
        self._note_activity()
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
            boost=boost,
        )

    def believe(
        self,
        statement: str,
        probability: float,
        evidence: str = None,
        valence: float | None = None,
        intensity: float | None = None,
        source: str = "user",
        metadata: dict | None = None,
    ) -> str:
        """Assert a probabilistic fact.

        ``evidence`` is why the caller believes it. It is recorded on the
        evidence row, so the log holds the observation and not only the fact
        that one was made, and the atom's provenance can be listed later
        through :meth:`evidence`.
        """
        self._note_activity()
        # A permanent assertion is born certain. Starting it at the ordinary
        # belief confidence leaves a fact the caller stated as settled ranked
        # below the conversational froth stored beside it, which reads to the
        # caller as never having stored it at all.
        confidence = (
            probability
            if probability >= PERMANENT_PROBABILITY
            else INITIAL_CONFIDENCE["belief"]
        )
        statement, temporal = self._resolve_deixis(statement)
        valence, stated = self._resolve_valence(statement, valence)
        atom = Atom(
            type=AtomType.BELIEF,
            label=statement[:100],
            content=statement,
            truth=TruthValue(probability=probability, confidence=confidence),
            valence=Valence(
                valence=valence, intensity=self._resolve_intensity(valence, intensity)
            ),
            tenant_id=self.tenant_id,
            space=self.write_space,
            metadata=self._with_temporal(
                {**_write_metadata(source, stated), **(metadata or {})}, temporal
            ),
        )
        atom_id = self.atomspace.add_atom(atom)
        if evidence:
            ev = Evidence(
                atom_id=atom_id,
                observed_probability=probability,
                text=evidence,
                source=source,
                tenant_id=self.tenant_id,
                space=self.write_space,
            )
            self.atomspace.add_evidence(ev)
        return atom_id

    def evidence(self, atom_id: str) -> list[Evidence]:
        """Every observation filed against an atom in the write space, oldest first."""
        return self.atomspace.get_evidence(atom_id, self.tenant_id, self.write_space)

    def reinforce(
        self, atom_ids: list[str], weight: float = _REINFORCE_WEIGHT
    ) -> dict:
        """Record that these memories were used, as evidence for their truth.

        The caller decides what "used" means — a cheap proxy is that
        distinctive words from a recalled atom turned up in the reply it
        informed. The engine trusts the report the way it trusts any other
        evidence, and no further: the weight is small, the update converges,
        and each atom banks at most a few reports per consolidation.

        Returns the ids that took the evidence and, for each one that did
        not, why — unknown in this space, deliberately forgotten, or already
        capped this epoch.
        """
        self._note_activity()
        return reinforce_atoms(
            atom_ids, self.tenant_id, self.write_space, self.db, weight
        )

    def reflect(self) -> EpochResult:
        with _get_reflect_lock(self.tenant_id, self.write_space):
            result = run_epoch(self.tenant_id, self.write_space, self.db, self.embed)
            self._ops_since_reflect = 0
            return result

    def forget(self, query: str, top_k: int = 5) -> list[str]:
        """Stop the memories matching *query* from surfacing.

        Three things happen to each match in the write space. Its confidence
        is sunk below the surfacing floor, so no floor a caller might pass
        finds it. It is stamped ``$.forgotten``, which excludes it from every
        recall outright, keeps the epoch from lifting it back (a drowned
        permanent belief is normally restored to its asserted confidence),
        keeps reinforcement from lifting it, and drops the long-term floors
        that exempt user testimony from pruning — so the next consolidation
        may remove it. And nothing else: the retrieval that finds the matches
        runs without the access boost, because forgetting a memory must not
        make it more prominent, which is what it did when this method called
        the ordinary recall.
        """
        self._note_activity()
        results = self.recall(query=query, top_k=top_k, boost=False)
        # Below the floor by a margin, never merely at it: the decay floor
        # holds an atom that is still at or above the line.
        sunk_to = self._surfacing_floor() * 0.5
        forgotten = []
        for r in results:
            if r.atom.space != self.write_space:
                continue
            self.db.execute(
                f"""UPDATE atoms SET
                        confidence = MIN(confidence * 0.3, ?),
                        metadata = json_set({ATOM_METADATA_JSON},
                                            '$.forgotten', json('true')),
                        updated_at = datetime('now')
                    WHERE id = ? AND tenant_id = ? AND space = ?""",
                (sunk_to, r.atom.id, self.tenant_id, self.write_space),
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
                *vec_delete(chunk),
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
