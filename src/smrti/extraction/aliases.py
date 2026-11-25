"""Alias table management for fast entity resolution."""
from __future__ import annotations

from smrti.extraction.ner import _PRONOUNS

# Older builds stored pronouns as aliases of person atoms only: the pronoun
# merge folded them into the one person a text named, and the model was asked
# to list a person's pronouns among their aliases. The word list is matched
# against nothing else, because case-folded it also holds "US", "IT", "EU" and
# "LA", which are real aliases of places and organizations.
_PERSON_TARGET = "atom_id IN (SELECT id FROM atoms WHERE entity_type = 'person')"


class AliasManager:
    def __init__(self, db) -> None:
        self.db = db

    def lookup(self, alias: str, tenant_id: str, spaces: list[str]) -> str | None:
        """Returns atom_id if alias is known in any of the given spaces, else None.

        Uses u_lower for Unicode-aware case folding (SQLite LOWER is ASCII-only).
        A pronoun is never a person's alias, whatever the table says: older
        builds stored "she" as the alias of whoever it first stood for, and
        every later "she" became that person.
        """
        ph = ",".join("?" * len(spaces))
        query = f"SELECT atom_id FROM aliases WHERE u_lower(alias) = u_lower(?) AND tenant_id = ? AND space IN ({ph})"
        if alias.strip().casefold() in _PRONOUNS:
            query += f" AND NOT {_PERSON_TARGET}"
        row = self.db.fetchone(query, (alias, tenant_id, *spaces))
        return row["atom_id"] if row else None

    def purge_pronouns(self, tenant_id: str, space: str) -> None:
        """Delete the pronoun aliases older builds stored for persons in *space*.

        Idempotent, and a read when there is nothing to delete, so it can run
        whenever a resolver starts rather than once: a process on older code
        sharing the file may still be writing them.
        """
        listed = sorted(_PRONOUNS)
        where = (
            f"tenant_id = ? AND space = ? AND u_lower(alias) IN ({','.join('?' * len(listed))})"
            f" AND {_PERSON_TARGET}"
        )
        params = (tenant_id, space, *listed)
        if self.db.fetchone(f"SELECT 1 FROM aliases WHERE {where} LIMIT 1", params):
            self.db.execute(f"DELETE FROM aliases WHERE {where}", params)

    def add(self, atom_id: str, alias: str, tenant_id: str, space: str) -> None:
        """Register a new alias for an atom in the given space."""
        self.db.execute(
            "INSERT OR IGNORE INTO aliases (alias, atom_id, tenant_id, space) VALUES (?, ?, ?, ?)",
            (alias, atom_id, tenant_id, space),
        )

    def get_all_for_atom(self, atom_id: str, tenant_id: str, space: str) -> list[str]:
        rows = self.db.fetchall(
            "SELECT alias FROM aliases WHERE atom_id = ? AND tenant_id = ? AND space = ?",
            (atom_id, tenant_id, space),
        )
        return [r["alias"] for r in rows]

    def delete_for_atom(self, atom_id: str, tenant_id: str, space: str) -> None:
        self.db.execute(
            "DELETE FROM aliases WHERE atom_id = ? AND tenant_id = ? AND space = ?",
            (atom_id, tenant_id, space),
        )
