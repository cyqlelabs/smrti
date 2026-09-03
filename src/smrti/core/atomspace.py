from __future__ import annotations

import hashlib
import json
import struct
from typing import Optional

from smrti.core.db import Database, fts_write, stable_rowid, vec_delete, vec_insert
from smrti.core.embed import EmbeddingProvider
from smrti.core.models import (
    Atom,
    AtomType,
    Evidence,
    TruthValue,
    atom_from_row,
)


class AtomSpace:
    def __init__(self, db: Database, embed: EmbeddingProvider) -> None:
        self._db = db
        self._embed = embed

    def add_atom(
        self,
        atom: Atom,
        embedding: list[float] | None = None,
        *,
        require_vector: bool = True,
    ) -> str:
        """Write an atom and its index rows.

        Every atom the engine creates comes through here — the facade, the
        entity resolver, healing, association discovery and bridging alike —
        so the columns that need writing on every row are written on every
        row. The intrinsic valence pair in particular: an atom inserted with
        those columns NULL reads its judged tone from the drifting one, and
        drifts with its neighbours' mood, which is the one thing the
        intrinsic split exists to prevent.

        ``embedding`` lets a caller that already embedded the text (the
        resolver probes the vector index with it first) hand the vector over
        instead of paying for the encoding twice. ``require_vector=False``
        lets a background writer whose encoder just failed store the row
        without one rather than lose the atom; the lexical index still
        carries it, and a later text change re-embeds it.
        """
        prior = self._db.fetchone(
            "SELECT label, content, tenant_id, space FROM atoms WHERE id = ?",
            (atom.id,),
        )
        # ``INSERT OR REPLACE`` keys on the primary key alone, so re-adding an
        # atom under a different partition silently moves it — carrying its
        # vector row with it — and one tenant's memory lands in another's graph.
        # IDs are UUIDs, so this only happens when a caller reuses one, and when
        # it does the write is a mistake, not a relocation request.
        if prior is not None and (
            prior["tenant_id"] != atom.tenant_id or prior["space"] != atom.space
        ):
            raise ValueError(
                f"atom {atom.id} already exists in "
                f"tenant={prior['tenant_id']!r} space={prior['space']!r}; "
                f"refusing to move it to tenant={atom.tenant_id!r} space={atom.space!r}"
            )
        statements: list[tuple] = [
            (
                """
                INSERT OR REPLACE INTO atoms (
                    id, type, label, content, probability, confidence,
                    sti, lti, valence, intensity,
                    intrinsic_valence, intrinsic_intensity,
                    source_id, target_id, relation,
                    tenant_id, space, metadata, entity_type, content_hash,
                    created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    COALESCE(
                        (SELECT created_at FROM atoms WHERE id = ?),
                        datetime('now')
                    ),
                    datetime('now')
                )
                """,
                (
                    atom.id,
                    atom.type.value,
                    atom.label,
                    atom.content,
                    atom.truth.probability,
                    atom.truth.confidence,
                    atom.attention.sti,
                    atom.attention.lti,
                    atom.valence.valence,
                    atom.valence.intensity,
                    # What the atom itself says, kept out of propagation's reach.
                    atom.valence.own,
                    atom.valence.own_intensity,
                    atom.source_id,
                    atom.target_id,
                    atom.relation,
                    atom.tenant_id,
                    atom.space,
                    json.dumps(atom.metadata),
                    atom.entity_type.value if atom.entity_type else None,
                    hashlib.sha256(atom.content.encode()).hexdigest()
                    if atom.content
                    else None,
                    atom.id,
                ),
            )
        ]

        # Protect severe negative-valence atoms from epoch pruning
        if atom.valence.valence < -0.7 and atom.valence.intensity > 0.7:
            statements.append(
                ("UPDATE atoms SET lti = MAX(lti, 0.5) WHERE id = ?", (atom.id,))
            )

        # Relation atoms carry synthetic labels — keep them out of the KNN and
        # lexical indexes. The lexical write is unconditional where the vector
        # write is not: it is two integer-keyed statements, and skipping it
        # when the vector already exists would leave an atom permanently
        # unsearchable by word if its index row was ever lost.
        if atom.type != AtomType.RELATION:
            statements.extend(fts_write(self._db, atom.id, atom.label, atom.content))
            # By rowid, never by atom_id: a vec0 table answers a non-rowid
            # predicate with a full scan of every vector in the database, so
            # this one query used to make each write cost O(graph size).
            existing_vec = self._db.fetchone(
                "SELECT 1 FROM vec_atoms WHERE rowid = ?",
                (stable_rowid(atom.id),),
            )
            content_changed = prior is not None and (
                prior["label"] != atom.label or prior["content"] != atom.content
            )
            if (not existing_vec or content_changed) and (
                embedding is not None or require_vector
            ):
                if embedding is None:
                    text_to_embed = atom.label
                    if atom.content:
                        text_to_embed = f"{atom.label} {atom.content}"
                    embedding = self._embed.embed(text_to_embed)
                vec_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                if existing_vec:
                    statements.extend(vec_delete([atom.id]))
                statements.append(
                    vec_insert(atom.id, vec_bytes, atom.tenant_id, atom.space, atom.label)
                )

        self._db.execute_batch(statements)
        return atom.id

    def get_atom(self, atom_id: str, tenant_id: str, space: str) -> Atom | None:
        row = self._db.fetchone(
            "SELECT * FROM atoms WHERE id = ? AND tenant_id = ? AND space = ?",
            (atom_id, tenant_id, space),
        )
        if row is None:
            return None
        return atom_from_row(row)

    def update_atom(self, atom: Atom) -> None:
        prior = self._db.fetchone(
            "SELECT label, content FROM atoms WHERE id = ? AND tenant_id = ? AND space = ?",
            (atom.id, atom.tenant_id, atom.space),
        )
        statements: list[tuple] = [
            (
                """
                UPDATE atoms SET
                    type = ?, label = ?, content = ?,
                    probability = ?, confidence = ?,
                    sti = ?, lti = ?,
                    valence = ?, intensity = ?,
                    intrinsic_valence = ?, intrinsic_intensity = ?,
                    source_id = ?, target_id = ?, relation = ?,
                    metadata = ?, entity_type = ?,
                    updated_at = datetime('now')
                WHERE id = ? AND tenant_id = ? AND space = ?
                """,
                (
                    atom.type.value,
                    atom.label,
                    atom.content,
                    atom.truth.probability,
                    atom.truth.confidence,
                    atom.attention.sti,
                    atom.attention.lti,
                    atom.valence.valence,
                    atom.valence.intensity,
                    atom.valence.own,
                    atom.valence.own_intensity,
                    atom.source_id,
                    atom.target_id,
                    atom.relation,
                    json.dumps(atom.metadata),
                    atom.entity_type.value if atom.entity_type else None,
                    atom.id,
                    atom.tenant_id,
                    atom.space,
                ),
            )
        ]

        # Keep the KNN and lexical indexes in sync when the text changes.
        if (
            prior is not None
            and atom.type != AtomType.RELATION
            and (prior["label"] != atom.label or prior["content"] != atom.content)
        ):
            statements.extend(fts_write(self._db, atom.id, atom.label, atom.content))
            text_to_embed = atom.label
            if atom.content:
                text_to_embed = f"{atom.label} {atom.content}"
            embedding = self._embed.embed(text_to_embed)
            vec_bytes = struct.pack(f"{len(embedding)}f", *embedding)
            statements.extend(vec_delete([atom.id]))
            statements.append(
                vec_insert(atom.id, vec_bytes, atom.tenant_id, atom.space, atom.label)
            )

        self._db.execute_batch(statements)

    def link_atoms(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        tenant_id: str,
        space: str,
        truth: Optional[TruthValue] = None,
        valence: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> str:
        # Idempotent: boost STI and return existing relation if already present
        existing = self._db.fetchone(
            """SELECT id FROM atoms WHERE type = 'relation' AND source_id = ? AND target_id = ?
               AND relation = ? AND tenant_id = ? AND space = ?""",
            (source_id, target_id, relation, tenant_id, space),
        )
        if existing:
            self._db.execute(
                "UPDATE atoms SET sti = MIN(sti + 0.2, 3.0), updated_at = datetime('now') WHERE id = ?",
                (existing["id"],),
            )
            return existing["id"]

        if truth is None:
            truth = TruthValue(probability=0.8, confidence=0.5)
        label = f"{relation}({source_id[:8]}, {target_id[:8]})"
        from smrti.core.models import Valence as ValenceModel
        link_atom = Atom(
            type=AtomType.RELATION,
            label=label,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            truth=truth,
            tenant_id=tenant_id,
            space=space,
            valence=ValenceModel(valence=max(-1.0, min(1.0, valence)), intensity=abs(valence)),
            metadata=dict(metadata or {}),
        )
        return self.add_atom(link_atom)

    def get_neighbors(
        self,
        atom_id: str,
        tenant_id: str,
        spaces: list[str],
        direction: str = "both",
    ) -> list[Atom]:
        ph = ",".join("?" * len(spaces))
        neighbor_ids: list[str] = []

        if direction in ("out", "both"):
            rows = self._db.fetchall(
                f"SELECT target_id FROM atoms WHERE source_id = ? AND tenant_id = ? AND space IN ({ph}) AND type = 'relation' AND target_id IS NOT NULL",
                (atom_id, tenant_id, *spaces),
            )
            neighbor_ids.extend(r["target_id"] for r in rows)

        if direction in ("in", "both"):
            rows = self._db.fetchall(
                f"SELECT source_id FROM atoms WHERE target_id = ? AND tenant_id = ? AND space IN ({ph}) AND type = 'relation' AND source_id IS NOT NULL",
                (atom_id, tenant_id, *spaces),
            )
            neighbor_ids.extend(r["source_id"] for r in rows)

        if not neighbor_ids:
            return []

        seen: set[str] = set()
        unique_ids = []
        for nid in neighbor_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)

        id_ph = ",".join("?" * len(unique_ids))
        rows = self._db.fetchall(
            f"SELECT * FROM atoms WHERE id IN ({id_ph}) AND tenant_id = ? AND space IN ({ph})",
            (*unique_ids, tenant_id, *spaces),
        )
        return [atom_from_row(r) for r in rows]

    def get_relations(self, atom_id: str, tenant_id: str, spaces: list[str]) -> list[Atom]:
        ph = ",".join("?" * len(spaces))
        rows = self._db.fetchall(
            f"""
            SELECT * FROM atoms
            WHERE type = 'relation'
              AND tenant_id = ?
              AND space IN ({ph})
              AND (source_id = ? OR target_id = ?)
            """,
            (tenant_id, *spaces, atom_id, atom_id),
        )
        return [atom_from_row(r) for r in rows]

    def search_by_label(
        self,
        label: str,
        tenant_id: str,
        spaces: list[str],
        entity_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[Atom]:
        ph = ",".join("?" * len(spaces))
        escaped = (
            label.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        if entity_type:
            rows = self._db.fetchall(
                f"SELECT * FROM atoms WHERE label LIKE ? ESCAPE '\\' AND tenant_id = ? AND space IN ({ph}) AND entity_type = ? LIMIT ?",
                (f"%{escaped}%", tenant_id, *spaces, entity_type, limit),
            )
        else:
            rows = self._db.fetchall(
                f"SELECT * FROM atoms WHERE label LIKE ? ESCAPE '\\' AND tenant_id = ? AND space IN ({ph}) LIMIT ?",
                (f"%{escaped}%", tenant_id, *spaces, limit),
            )
        return [atom_from_row(r) for r in rows]

    def boost_sti(
        self,
        atom_id: str,
        amount: float = 0.5,
        tenant_id: str | None = None,
        space: str | None = None,
    ) -> None:
        """Raise an atom's STI, optionally constrained to a tenant/space.

        Passing ``tenant_id``/``space`` makes the write a no-op for atoms
        outside that partition, so a caller holding an ID from an overlay space
        cannot reach into it.
        """
        if tenant_id is not None and space is not None:
            self._db.execute(
                "UPDATE atoms SET sti = MIN(sti + ?, 3.0), updated_at = datetime('now') "
                "WHERE id = ? AND tenant_id = ? AND space = ?",
                (amount, atom_id, tenant_id, space),
            )
            return
        self._db.execute(
            "UPDATE atoms SET sti = MIN(sti + ?, 3.0), updated_at = datetime('now') WHERE id = ?",
            (amount, atom_id),
        )

    def add_evidence(self, evidence: Evidence) -> None:
        self._db.execute(
            """
            INSERT INTO evidence (id, atom_id, observed_probability, weight,
                                  source_episode_id, text, source, tenant_id, space)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.id,
                evidence.atom_id,
                evidence.observed_probability,
                evidence.weight,
                evidence.source_episode_id,
                evidence.text,
                evidence.source,
                evidence.tenant_id,
                evidence.space,
            ),
        )

    @staticmethod
    def _evidence_from_row(r) -> Evidence:
        keys = r.keys()
        return Evidence(
            id=r["id"],
            atom_id=r["atom_id"],
            observed_probability=r["observed_probability"],
            weight=r["weight"],
            source_episode_id=r["source_episode_id"],
            text=r["text"] if "text" in keys else None,
            source=r["source"] if "source" in keys else None,
            tenant_id=r["tenant_id"],
            space=r["space"],
            created_at=r["created_at"] if "created_at" in keys else None,
            processed=bool(r["processed"]) if "processed" in keys else False,
        )

    def get_pending_evidence(self, tenant_id: str, space: str) -> list[Evidence]:
        rows = self._db.fetchall(
            "SELECT * FROM evidence WHERE processed = 0 AND tenant_id = ? AND space = ? ORDER BY created_at ASC",
            (tenant_id, space),
        )
        return [self._evidence_from_row(r) for r in rows]

    def get_evidence(self, atom_id: str, tenant_id: str, space: str) -> list[Evidence]:
        """Every observation filed against an atom, oldest first.

        This is what makes the log a provenance record: a belief can list
        why it is believed, not only how confident the engine has become.
        """
        rows = self._db.fetchall(
            "SELECT * FROM evidence WHERE atom_id = ? AND tenant_id = ? AND space = ? "
            "ORDER BY created_at ASC, rowid ASC",
            (atom_id, tenant_id, space),
        )
        return [self._evidence_from_row(r) for r in rows]

    def mark_evidence_processed(self, evidence_id: str) -> None:
        self._db.execute(
            "UPDATE evidence SET processed = 1 WHERE id = ?",
            (evidence_id,),
        )
