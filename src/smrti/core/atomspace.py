from __future__ import annotations

import hashlib
import json
import struct
from typing import Optional

from smrti.core.db import Database, fts_delete, fts_write, stable_rowid, vec_delete, vec_insert
from smrti.core.embed import EmbeddingProvider
from smrti.core.models import (
    Atom,
    AtomType,
    Evidence,
    TruthValue,
    atom_from_row,
)
from smrti.core.provenance import ATOM_METADATA_JSON

# Stamped on an atom stored without a vector, so the epoch can find what
# needs re-embedding without checking every atom against the index.
VECTOR_MISSING = "vector_missing"

# Chunk size for generated ``IN (...)`` lists, well under SQLite's variable
# limit on spaces holding tens of thousands of atoms.
_ID_CHUNK = 400


def embedding_text(atom: Atom) -> str:
    """The text an atom is embedded on: its label, plus its content when it has one."""
    return f"{atom.label} {atom.content}" if atom.content else atom.label


def _chunks(ids: list[str]):
    for start in range(0, len(ids), _ID_CHUNK):
        yield ids[start : start + _ID_CHUNK]


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
                    embedding = self._embed.embed(embedding_text(atom))
                vec_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                if existing_vec:
                    statements.extend(vec_delete([atom.id]))
                statements.append(
                    vec_insert(atom.id, vec_bytes, atom.tenant_id, atom.space, atom.label)
                )
            elif not existing_vec:
                # Stored without a vector: mark it, so the epoch's repair pass
                # can find it and recall can score it in the meantime.
                statements.append(
                    (
                        f"UPDATE atoms SET metadata = json_set({ATOM_METADATA_JSON}, '$.{VECTOR_MISSING}', 1) WHERE id = ?",
                        (atom.id,),
                    )
                )

        self._db.execute_batch(statements)
        return atom.id

    def file_vector(self, atom: Atom, embedding: list[float]) -> None:
        """Give an atom stored without a vector its index row, and clear the mark."""
        vec_bytes = struct.pack(f"{len(embedding)}f", *embedding)
        self._db.execute_batch([
            *vec_delete([atom.id]),
            vec_insert(atom.id, vec_bytes, atom.tenant_id, atom.space, atom.label),
            (
                f"UPDATE atoms SET metadata = json_remove({ATOM_METADATA_JSON}, '$.{VECTOR_MISSING}') WHERE id = ?",
                (atom.id,),
            ),
        ])

    def delete_atoms(self, atom_ids: list[str], tenant_id: str) -> int:
        """Hard-delete atoms and every row that references them. Returns the count removed.

        The set grows to closure first: the relation edges on the atoms, the
        edges on those edges (a ``contradicts`` edge between two claim
        edges), and so on — an edge can sit in any space of the tenant, since
        a bridge space points back into both parents. Evidence and aliases
        are filed against relation atoms too (a supersession files negative
        evidence against the edge it replaced), and all three references are
        enforced foreign keys, so the whole closure goes in one transaction
        with the checks deferred to its commit: a delete that ordered the
        rows by hand left the space half-cleared on the first shape it had
        not foreseen.
        """
        doomed: dict[str, None] = dict.fromkeys(atom_ids)
        frontier = list(doomed)
        while frontier:
            found: list[str] = []
            for chunk in _chunks(frontier):
                ph = ",".join("?" * len(chunk))
                found.extend(
                    r["id"]
                    for r in self._db.fetchall(
                        f"""SELECT id FROM atoms WHERE type = 'relation' AND tenant_id = ?
                            AND (source_id IN ({ph}) OR target_id IN ({ph}))""",
                        (tenant_id, *chunk, *chunk),
                    )
                    if r["id"] not in doomed
                )
            doomed.update(dict.fromkeys(found))
            frontier = found

        ids = list(doomed)
        statements: list[tuple] = [("PRAGMA defer_foreign_keys = ON", ())]
        for chunk in _chunks(ids):
            ph = ",".join("?" * len(chunk))
            statements.extend([
                *vec_delete(chunk),
                *fts_delete(self._db, chunk),
                (f"DELETE FROM evidence WHERE atom_id IN ({ph})", tuple(chunk)),
                (f"DELETE FROM aliases WHERE atom_id IN ({ph})", tuple(chunk)),
                (f"DELETE FROM atoms WHERE id IN ({ph})", tuple(chunk)),
            ])
        self._db.execute_batch(statements)
        return len(ids)

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
            embedding = self._embed.embed(embedding_text(atom))
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
