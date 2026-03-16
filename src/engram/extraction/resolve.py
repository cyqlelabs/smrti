"""Entity resolution: exact -> alias -> fuzzy -> embedding -> create."""
from __future__ import annotations

import struct
import uuid

from rapidfuzz import fuzz, process


class EntityResolver:
    """Resolves extracted entity names to existing atoms or creates new ones.

    Resolution tiers (ordered by cost):
      0. Exact label match      — indexed, < 1ms
      1. Alias table lookup     — indexed, < 1ms
      2. Fuzzy match (RapidFuzz) — in-process, < 5ms
      3. Embedding cosine sim   — ONNX inference, < 20ms
      4. Create new atom        — write path
    """

    def __init__(
        self,
        db,
        embed_engine,
        fuzzy_threshold: float = 85.0,
        cosine_threshold: float = 0.3,
    ) -> None:
        self.db = db
        self.embed_engine = embed_engine
        self.fuzzy_threshold = fuzzy_threshold
        self.cosine_threshold = cosine_threshold

        from engram.extraction.aliases import AliasManager
        self.aliases = AliasManager(db)

    def resolve(self, name: str, entity_type: str, agent_id: str) -> str:
        """Return atom_id for the named entity, creating one if needed."""

        # Tier 0: exact label match
        row = self.db.fetchone(
            "SELECT id FROM atoms WHERE LOWER(label) = LOWER(?) AND entity_type = ? AND agent_id = ?",
            (name, entity_type, agent_id),
        )
        if row:
            self._boost_sti(row["id"])
            return row["id"]

        # Tier 1: alias table
        atom_id = self.aliases.lookup(name, agent_id)
        if atom_id:
            self._boost_sti(atom_id)
            return atom_id

        # Tier 2: fuzzy match within same entity_type
        candidates = self.db.fetchall(
            "SELECT id, label FROM atoms WHERE entity_type = ? AND agent_id = ? AND type != 'relation'",
            (entity_type, agent_id),
        )
        if candidates:
            names_map = {r["id"]: r["label"] for r in candidates}
            match = process.extractOne(name, names_map, scorer=fuzz.WRatio)
            if match and match[1] >= self.fuzzy_threshold:
                matched_id = match[2]
                self.aliases.add(matched_id, name, agent_id)
                self._boost_sti(matched_id)
                return matched_id

        # Tier 3: embedding cosine similarity via sqlite-vec
        query_vec = self.embed_engine.embed(name)
        vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
        vec_match = self.db.fetchone(
            """SELECT atom_id, distance FROM vec_atoms
               WHERE embedding MATCH ? AND agent_id = ?
               ORDER BY distance LIMIT 1""",
            (vec_bytes, agent_id),
        )
        if vec_match and vec_match["distance"] < self.cosine_threshold:
            self.aliases.add(vec_match["atom_id"], name, agent_id)
            self._boost_sti(vec_match["atom_id"])
            return vec_match["atom_id"]

        # Tier 4: create new atom
        return self._create_atom(name, entity_type, agent_id)

    def _boost_sti(self, atom_id: str) -> None:
        self.db.execute(
            "UPDATE atoms SET sti = MIN(sti + 0.5, 3.0) WHERE id = ?",
            (atom_id,),
        )

    def _create_atom(self, name: str, entity_type: str, agent_id: str) -> str:
        atom_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO atoms (id, type, label, entity_type, agent_id, probability, confidence)
               VALUES (?, 'concept', ?, ?, ?, 0.5, 0.2)""",
            (atom_id, name, entity_type, agent_id),
        )

        try:
            vec = self.embed_engine.embed(name)
            vec_bytes = struct.pack(f"{len(vec)}f", *vec)
            self.db.execute(
                "INSERT INTO vec_atoms (atom_id, embedding, agent_id, label) VALUES (?, ?, ?, ?)",
                (atom_id, vec_bytes, agent_id, name),
            )
        except Exception:
            pass  # embedding failure must not block atom creation

        return atom_id
