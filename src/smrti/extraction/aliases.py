"""Alias table management for fast entity resolution."""
from __future__ import annotations


class AliasManager:
    def __init__(self, db) -> None:
        self.db = db

    def lookup(self, alias: str, tenant_id: str, spaces: list[str]) -> str | None:
        """Returns atom_id if alias is known in any of the given spaces, else None."""
        ph = ",".join("?" * len(spaces))
        row = self.db.fetchone(
            f"SELECT atom_id FROM aliases WHERE LOWER(alias) = LOWER(?) AND tenant_id = ? AND space IN ({ph})",
            (alias, tenant_id, *spaces),
        )
        return row["atom_id"] if row else None

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
