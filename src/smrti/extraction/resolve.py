"""Entity resolution: exact -> alias -> fuzzy -> embedding -> create."""
from __future__ import annotations

import struct
import uuid

from rapidfuzz import fuzz, process


class EntityResolver:
    """Resolves extracted entity names to existing atoms or creates new ones.

    Reads are scoped to read_spaces (overlay); writes go to write_space only.

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

        from smrti.extraction.aliases import AliasManager
        self.aliases = AliasManager(db)

    def resolve(
        self,
        name: str,
        entity_type: str,
        tenant_id: str,
        write_space: str,
        read_spaces: list[str],
    ) -> str:
        """Return atom_id for the named entity, creating one if needed.

        Searches across read_spaces; new atoms are created in write_space.
        """
        spaces_ph = ",".join("?" * len(read_spaces))

        # Tier 0: exact label match across read_spaces
        row = self.db.fetchone(
            f"SELECT id FROM atoms WHERE LOWER(label) = LOWER(?) AND entity_type = ? AND tenant_id = ? AND space IN ({spaces_ph})",
            (name, entity_type, tenant_id, *read_spaces),
        )
        if row:
            self._boost_sti(row["id"])
            return row["id"]

        # Tier 1: alias table across read_spaces
        atom_id = self.aliases.lookup(name, tenant_id, read_spaces)
        if atom_id:
            self._boost_sti(atom_id)
            return atom_id

        # Tier 2: fuzzy match within same entity_type across read_spaces
        candidates = self.db.fetchall(
            f"SELECT id, label FROM atoms WHERE entity_type = ? AND tenant_id = ? AND space IN ({spaces_ph}) AND type != 'relation'",
            (entity_type, tenant_id, *read_spaces),
        )
        if candidates:
            names_map = {r["id"]: r["label"] for r in candidates}
            match = process.extractOne(name, names_map, scorer=fuzz.WRatio)
            if match and match[1] >= self.fuzzy_threshold:
                matched_id = match[2]
                self.aliases.add(matched_id, name, tenant_id, write_space)
                self._boost_sti(matched_id)
                return matched_id

        # Tier 3: embedding cosine similarity via sqlite-vec (tenant scope)
        query_vec = self.embed_engine.embed(name)
        vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
        vec_match = self.db.fetchone(
            """SELECT atom_id, distance FROM vec_atoms
               WHERE embedding MATCH ? AND tenant_id = ?
               ORDER BY distance LIMIT 1""",
            (vec_bytes, tenant_id),
        )
        if vec_match and vec_match["distance"] < self.cosine_threshold:
            self.aliases.add(vec_match["atom_id"], name, tenant_id, write_space)
            self._boost_sti(vec_match["atom_id"])
            return vec_match["atom_id"]

        # Tier 4: create new atom in write_space
        return self._create_atom(name, entity_type, tenant_id, write_space)

    def _boost_sti(self, atom_id: str) -> None:
        self.db.execute(
            "UPDATE atoms SET sti = MIN(sti + 0.5, 3.0) WHERE id = ?",
            (atom_id,),
        )

    def _create_atom(self, name: str, entity_type: str, tenant_id: str, space: str) -> str:
        atom_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti)
               VALUES (?, 'concept', ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)""",
            (atom_id, name, entity_type, tenant_id, space),
        )

        try:
            vec = self.embed_engine.embed(name)
            vec_bytes = struct.pack(f"{len(vec)}f", *vec)
            self.db.execute(
                "INSERT INTO vec_atoms (atom_id, embedding, tenant_id, label) VALUES (?, ?, ?, ?)",
                (atom_id, vec_bytes, tenant_id, name),
            )
        except Exception:
            pass

        return atom_id
