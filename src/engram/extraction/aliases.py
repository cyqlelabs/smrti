"""Alias table management for fast entity resolution."""
from __future__ import annotations


class AliasManager:
    def __init__(self, db) -> None:
        self.db = db

    def lookup(self, alias: str, agent_id: str) -> str | None:
        """Returns atom_id if alias is known, else None."""
        row = self.db.fetchone(
            "SELECT atom_id FROM aliases WHERE LOWER(alias) = LOWER(?) AND agent_id = ?",
            (alias, agent_id),
        )
        return row["atom_id"] if row else None

    def add(self, atom_id: str, alias: str, agent_id: str) -> None:
        """Register a new alias for an atom."""
        self.db.execute(
            "INSERT OR IGNORE INTO aliases (alias, atom_id, agent_id) VALUES (?, ?, ?)",
            (alias, atom_id, agent_id),
        )

    def get_all_for_atom(self, atom_id: str, agent_id: str) -> list[str]:
        rows = self.db.fetchall(
            "SELECT alias FROM aliases WHERE atom_id = ? AND agent_id = ?",
            (atom_id, agent_id),
        )
        return [r["alias"] for r in rows]

    def delete_for_atom(self, atom_id: str, agent_id: str) -> None:
        self.db.execute(
            "DELETE FROM aliases WHERE atom_id = ? AND agent_id = ?",
            (atom_id, agent_id),
        )
