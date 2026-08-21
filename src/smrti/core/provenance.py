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

# An atom's own tone read back from SQL, falling back to the current value for
# rows written before the columns existed. Only propagation reads the drifting
# pair; everything that judges a memory reads these.
ATOM_OWN_VALENCE = "COALESCE(intrinsic_valence, valence)"
ATOM_OWN_INTENSITY = "COALESCE(intrinsic_intensity, intensity)"
