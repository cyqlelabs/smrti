from __future__ import annotations

import json
from typing import Optional

from engram.core.db import Database
from engram.core.embed import EmbeddingProvider
from engram.core.models import (
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
                agent_id, metadata, entity_type,
                created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
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
                atom.agent_id,
                json.dumps(atom.metadata),
                atom.entity_type.value if atom.entity_type else None,
                atom.id,
            ),
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
                "INSERT INTO vec_atoms (atom_id, embedding, agent_id, label) VALUES (?, ?, ?, ?)",
                (atom.id, json.dumps(embedding), atom.agent_id, atom.label),
            )

        return atom.id

    def get_atom(self, atom_id: str, agent_id: str) -> Atom | None:
        row = self._db.fetchone(
            "SELECT * FROM atoms WHERE id = ? AND agent_id = ?",
            (atom_id, agent_id),
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
            WHERE id = ? AND agent_id = ?
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
                atom.agent_id,
            ),
        )

    def link_atoms(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        agent_id: str,
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
            agent_id=agent_id,
        )
        return self.add_atom(link_atom)

    def get_neighbors(
        self,
        atom_id: str,
        agent_id: str,
        direction: str = "both",
    ) -> list[Atom]:
        neighbor_ids: list[str] = []

        if direction in ("out", "both"):
            rows = self._db.fetchall(
                "SELECT target_id FROM atoms WHERE source_id = ? AND agent_id = ? AND type = 'relation' AND target_id IS NOT NULL",
                (atom_id, agent_id),
            )
            neighbor_ids.extend(r["target_id"] for r in rows)

        if direction in ("in", "both"):
            rows = self._db.fetchall(
                "SELECT source_id FROM atoms WHERE target_id = ? AND agent_id = ? AND type = 'relation' AND source_id IS NOT NULL",
                (atom_id, agent_id),
            )
            neighbor_ids.extend(r["source_id"] for r in rows)

        if not neighbor_ids:
            return []

        seen = set()
        unique_ids = []
        for nid in neighbor_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)

        placeholders = ",".join("?" * len(unique_ids))
        rows = self._db.fetchall(
            f"SELECT * FROM atoms WHERE id IN ({placeholders}) AND agent_id = ?",
            (*unique_ids, agent_id),
        )
        return [atom_from_row(r) for r in rows]

    def get_relations(self, atom_id: str, agent_id: str) -> list[Atom]:
        rows = self._db.fetchall(
            """
            SELECT * FROM atoms
            WHERE type = 'relation'
              AND agent_id = ?
              AND (source_id = ? OR target_id = ?)
            """,
            (agent_id, atom_id, atom_id),
        )
        return [atom_from_row(r) for r in rows]

    def search_by_label(
        self,
        label: str,
        agent_id: str,
        entity_type: Optional[str] = None,
    ) -> list[Atom]:
        if entity_type:
            rows = self._db.fetchall(
                "SELECT * FROM atoms WHERE label LIKE ? AND agent_id = ? AND entity_type = ?",
                (f"%{label}%", agent_id, entity_type),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM atoms WHERE label LIKE ? AND agent_id = ?",
                (f"%{label}%", agent_id),
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
            INSERT INTO evidence (id, atom_id, observed_probability, weight, source_episode_id, agent_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.id,
                evidence.atom_id,
                evidence.observed_probability,
                evidence.weight,
                evidence.source_episode_id,
                evidence.agent_id,
            ),
        )

    def get_pending_evidence(self, agent_id: str) -> list[Evidence]:
        rows = self._db.fetchall(
            "SELECT * FROM evidence WHERE processed = 0 AND agent_id = ? ORDER BY created_at ASC",
            (agent_id,),
        )
        return [
            Evidence(
                id=r["id"],
                atom_id=r["atom_id"],
                observed_probability=r["observed_probability"],
                weight=r["weight"],
                source_episode_id=r["source_episode_id"],
                agent_id=r["agent_id"],
            )
            for r in rows
        ]

    def mark_evidence_processed(self, evidence_id: str) -> None:
        self._db.execute(
            "UPDATE evidence SET processed = 1 WHERE id = ?",
            (evidence_id,),
        )
