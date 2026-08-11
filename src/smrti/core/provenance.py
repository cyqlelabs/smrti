"""Atom provenance: who authored an atom, and how that is read back from SQL.

Provenance lives in ``atoms.metadata`` under ``$.source``. An atom with no
recorded source predates provenance tracking and reads as user-authored, so
upgrading never retroactively distrusts an existing graph.
"""
from __future__ import annotations

SOURCE_USER = "user"
SOURCE_AGENT = "agent"

# SQL for "who authored this atom", defaulting to the user.
#
# json_extract raises "malformed JSON" rather than returning NULL when the
# column is not valid JSON, which on a whole-table pass would abort the epoch
# for every atom in the space because of one bad row. CASE is used rather than
# `json_valid(...) AND ...` because only CASE guarantees the guarded branch is
# never evaluated.
ATOM_SOURCE = (
    "COALESCE(CASE WHEN json_valid(metadata) "
    "THEN json_extract(metadata, '$.source') END, 'user')"
)

# The same column as a writable JSON object: json_set also raises on malformed
# input, so unreadable metadata is replaced rather than appended to.
ATOM_METADATA_JSON = "CASE WHEN json_valid(metadata) THEN metadata ELSE '{}' END"
