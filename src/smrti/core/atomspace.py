from __future__ import annotations

import json
from typing import Optional

from smrti.core.db import Database
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

    def add_atom(self, atom: Atom) -> str:
        self._db.execute(
            """
            INSERT OR REPLACE INTO atoms (
                id, type, label, content, probability, confidence,
                sti, lti, valence, intensity,
                source_id, target_id, relation,
                tenant_id, space, metadata, entity_type,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
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
                atom.source_id,
                atom.target_id,
                atom.relation,
                atom.tenant_id,
                atom.space,
                json.dumps(atom.metadata),
                atom.entity_type.value if atom.entity_type else None,
                atom.id,
            ),
        )

        # Protect severe negative-valence atoms from epoch pruning
        if atom.valence.valence < -0.7 and atom.valence.intensity > 0.7:
            self._db.execute(
                "UPDATE atoms SET lti = MAX(lti, 0.5) WHERE id = ?",
                (atom.id,),
            )

        existing_vec = self._db.fetchone(
            "SELECT atom_id FROM vec_atoms WHERE atom_id = ?",
            (atom.id,),
        )
        if not existing_vec:
            text_to_embed = atom.label
            if atom.content:
                text_to_embed = f"{atom.label} {atom.content}"
            embedding = self._embed.embed(text_to_embed)
            self._db.execute(
                "INSERT INTO vec_atoms (atom_id, embedding, tenant_id, label) VALUES (?, ?, ?, ?)",
                (atom.id, json.dumps(embedding), atom.tenant_id, atom.label),
            )

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
        self._db.execute(
            """
            UPDATE atoms SET
                type = ?, label = ?, content = ?,
                probability = ?, confidence = ?,
                sti = ?, lti = ?,
                valence = ?, intensity = ?,
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

    def link_atoms(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        tenant_id: str,
        space: str,
        truth: Optional[TruthValue] = None,
    ) -> str:
        if truth is None:
            truth = TruthValue(probability=0.8, confidence=0.5)
        label = f"{relation}({source_id[:8]}, {target_id[:8]})"
        link_atom = Atom(
            type=AtomType.RELATION,
            label=label,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            truth=truth,
            tenant_id=tenant_id,
            space=space,
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
    ) -> list[Atom]:
        ph = ",".join("?" * len(spaces))
        if entity_type:
            rows = self._db.fetchall(
                f"SELECT * FROM atoms WHERE label LIKE ? AND tenant_id = ? AND space IN ({ph}) AND entity_type = ?",
                (f"%{label}%", tenant_id, *spaces, entity_type),
            )
        else:
            rows = self._db.fetchall(
                f"SELECT * FROM atoms WHERE label LIKE ? AND tenant_id = ? AND space IN ({ph})",
                (f"%{label}%", tenant_id, *spaces),
            )
        return [atom_from_row(r) for r in rows]

    def boost_sti(self, atom_id: str, amount: float = 0.5) -> None:
        self._db.execute(
            "UPDATE atoms SET sti = sti + ?, updated_at = datetime('now') WHERE id = ?",
            (amount, atom_id),
        )

    def add_evidence(self, evidence: Evidence) -> None:
        self._db.execute(
            """
            INSERT INTO evidence (id, atom_id, observed_probability, weight, source_episode_id, tenant_id, space)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.id,
                evidence.atom_id,
                evidence.observed_probability,
                evidence.weight,
                evidence.source_episode_id,
                evidence.tenant_id,
                evidence.space,
            ),
        )

    def get_pending_evidence(self, tenant_id: str, space: str) -> list[Evidence]:
        rows = self._db.fetchall(
            "SELECT * FROM evidence WHERE processed = 0 AND tenant_id = ? AND space = ? ORDER BY created_at ASC",
            (tenant_id, space),
        )
        return [
            Evidence(
                id=r["id"],
                atom_id=r["atom_id"],
                observed_probability=r["observed_probability"],
                weight=r["weight"],
                source_episode_id=r["source_episode_id"],
                tenant_id=r["tenant_id"],
                space=r["space"],
            )
            for r in rows
        ]

    def mark_evidence_processed(self, evidence_id: str) -> None:
        self._db.execute(
            "UPDATE evidence SET processed = 1 WHERE id = ?",
            (evidence_id,),
        )
