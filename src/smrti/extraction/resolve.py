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

        # Tier 0: exact label match across read_spaces (u_lower: Unicode-aware
        # case folding — SQLite's LOWER() only folds ASCII)
        row = self.db.fetchone(
            f"SELECT id FROM atoms WHERE u_lower(label) = u_lower(?) AND entity_type = ? AND tenant_id = ? AND space IN ({spaces_ph})",
            (name, entity_type, tenant_id, *read_spaces),
        )
        if row:
            self._boost_sti(row["id"])
            return row["id"]

        # Tier 0b: cross-type exact label match — prevents duplicate atoms when
        # the same span is classified under different entity_types (e.g. GLiNER
        # tagging "technology" as both "tool" and "concept"). Only matches atoms
        # that share the same underlying atom type so that goal/belief atoms are
        # never merged into concept atoms.
        atom_type = self._ENTITY_TYPE_TO_ATOM_TYPE.get(entity_type, "concept")
        row = self.db.fetchone(
            f"SELECT id FROM atoms WHERE u_lower(label) = u_lower(?) AND type = ? AND tenant_id = ? AND space IN ({spaces_ph})",
            (name, atom_type, tenant_id, *read_spaces),
        )
        if row:
            self._boost_sti(row["id"])
            return row["id"]

        # Tier 1: alias table across read_spaces
        atom_id = self.aliases.lookup(name, tenant_id, read_spaces)
        if atom_id:
            self._boost_sti(atom_id)
            return atom_id

        # Tier 2: fuzzy match within same entity_type across read_spaces,
        # bounded to the most salient candidates so the scan can't blow up
        candidates = self.db.fetchall(
            f"SELECT id, label FROM atoms WHERE entity_type = ? AND tenant_id = ? AND space IN ({spaces_ph}) AND type != 'relation' ORDER BY (sti + lti) DESC LIMIT 500",
            (entity_type, tenant_id, *read_spaces),
        )
        if candidates:
            names_map = {r["id"]: r["label"] for r in candidates}
            match = process.extractOne(name, names_map, scorer=fuzz.WRatio)
            if match and match[1] >= self.fuzzy_threshold:
                matched_id = match[2]
                # Only persist the alias on near-certain matches — a threshold-level
                # fuzzy hit would otherwise poison tier-1 resolution permanently.
                if match[1] >= self._ALIAS_PERSIST_SCORE:
                    self.aliases.add(matched_id, name, tenant_id, write_space)
                self._boost_sti(matched_id)
                return matched_id

        # Tier 3: embedding cosine similarity via sqlite-vec. KNN filters
        # support equality only, so probe each read space and keep the best.
        query_vec = self.embed_engine.embed(name)
        vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
        vec_match = None
        for space in read_spaces:
            row = self.db.fetchone(
                """SELECT atom_id, distance FROM vec_atoms
                   WHERE embedding MATCH ? AND tenant_id = ? AND space = ?
                   ORDER BY distance LIMIT 1""",
                (vec_bytes, tenant_id, space),
            )
            if row and (vec_match is None or row["distance"] < vec_match["distance"]):
                vec_match = row
        if vec_match and vec_match["distance"] < self.cosine_threshold:
            atom_row = self.db.fetchone(
                "SELECT entity_type FROM atoms WHERE id = ?",
                (vec_match["atom_id"],),
            )
            if atom_row and atom_row["entity_type"] == entity_type:
                # Embedding is the least reliable tier — never persist aliases here.
                self._boost_sti(vec_match["atom_id"])
                return vec_match["atom_id"]

        # Tier 4: create new atom in write_space (reusing the probe vector)
        return self._create_atom(name, entity_type, tenant_id, write_space, vec=query_vec)

    def _boost_sti(self, atom_id: str) -> None:
        self.db.execute(
            "UPDATE atoms SET sti = MIN(sti + 0.5, 3.0) WHERE id = ?",
            (atom_id,),
        )

    # Persisting an alias below this WRatio score risks poisoning tier-1
    # resolution — matches at fuzzy_threshold still resolve, they just
    # aren't remembered as aliases.
    _ALIAS_PERSIST_SCORE = 92.0

    _ENTITY_TYPE_TO_ATOM_TYPE = {
        "goal": "goal",
        "preference": "belief",
        "constraint": "belief",
    }

    def _create_atom(
        self,
        name: str,
        entity_type: str,
        tenant_id: str,
        space: str,
        vec: list[float] | None = None,
    ) -> str:
        atom_id = str(uuid.uuid4())
        atom_type = self._ENTITY_TYPE_TO_ATOM_TYPE.get(entity_type, "concept")
        self.db.execute(
            """INSERT INTO atoms (id, type, label, entity_type, tenant_id, space, probability, confidence, sti, lti)
               VALUES (?, ?, ?, ?, ?, ?, 0.8, 0.6, 1.0, 0.3)""",
            (atom_id, atom_type, name, entity_type, tenant_id, space),
        )

        try:
            if vec is None:
                vec = self.embed_engine.embed(name)
            vec_bytes = struct.pack(f"{len(vec)}f", *vec)
            self.db.execute(
                "INSERT INTO vec_atoms (atom_id, embedding, tenant_id, space, label) VALUES (?, ?, ?, ?, ?)",
                (atom_id, vec_bytes, tenant_id, space, name),
            )
        except Exception:
            pass

        return atom_id
