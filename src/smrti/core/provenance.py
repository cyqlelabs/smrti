"""Atom provenance: who authored an atom, and how that is read back from SQL.

Provenance lives in ``atoms.metadata`` under ``$.source``. An atom with no
recorded source predates provenance tracking and reads as user-authored, so
upgrading never retroactively distrusts an existing graph.
"""
from __future__ import annotations

SOURCE_USER = "user"
SOURCE_AGENT = "agent"

# Provenance of an atom's emotional tone: set when the caller stated the
# valence rather than letting it be estimated from the text. Only a stated
# valence can raise a memory to a behavioral constraint at recall. Sentiment
# estimated over ordinary conversation is a reading of tone, not a report of a
# mistake, and treating it as one turns "I didn't understand you" into
# something the agent must never do again.
VALENCE_STATED = "valence_stated"

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

def _metadata_column(table: str) -> str:
    return f"{table}.metadata" if table else "metadata"


def forgotten_sql(table: str = "") -> str:
    """Whether forget() deliberately sank this atom, read from *table*'s row.

    The stamp is what tells a forget from decay drowning: the epoch lifts a
    drowned permanent belief back to its asserted probability, and without
    the stamp that lift would undo every deliberate forget one epoch later.
    An atom with no stamp predates stamping and reads as never forgotten.

    Every reader that renders or resolves an atom applies this — recall,
    the proxy's entity enrichment, the extraction context, entity resolution
    — so that forgetting one atom hides it wherever it is represented.
    """
    column = _metadata_column(table)
    return (
        f"COALESCE(CASE WHEN json_valid({column}) "
        f"THEN json_extract({column}, '$.forgotten') END, 0)"
    )


ATOM_FORGOTTEN = forgotten_sql()

# Set on a claim edge that a later claim about the same subject replaced,
# naming the edge that replaced it. A marked edge is the entity's history;
# rendering reads its current state from the unmarked ones.
SUPERSEDED_BY = "superseded_by"


def claim_current_sql(table: str = "") -> str:
    """Whether a claim edge in *table* has not been superseded."""
    column = _metadata_column(table)
    return (
        f"(CASE WHEN json_valid({column}) "
        f"THEN json_extract({column}, '$.{SUPERSEDED_BY}') END) IS NULL"
    )

# An atom's own tone read back from SQL, falling back to the current value for
# rows written before the columns existed. Only propagation reads the drifting
# pair; everything that judges a memory reads these.
ATOM_OWN_VALENCE = "COALESCE(intrinsic_valence, valence)"
ATOM_OWN_INTENSITY = "COALESCE(intrinsic_intensity, intensity)"
