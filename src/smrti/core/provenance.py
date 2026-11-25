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

# The episodes that asserted a claim edge, appended each time extraction
# draws or reuses it. Forgetting with cascade removes a claim only when every
# episode recorded here is forgotten: co-mention says nothing about who
# stated a fact.
CLAIM_EPISODES = "episodes"


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

# Two tone lines, one meaning each, read by every judgement that turns on tone.
#
# A *stated* tone past the warning line, on anything that can hold a
# proposition, is a critical warning: recall raises it to a constraint, and the
# epoch keeps it as long as it keeps anything. Classification and retention
# once used different lines, so a stated warning at -0.6 was handed to the
# agent as a rule and then pruned.
#
# An *estimated* tone past the severe line is a failure reported in an ordinary
# sentence — not a constraint, but kept from the pruner when the user wrote it.
# An agent's own estimated tone earns no such floor: a model's apology scored
# at -0.8 otherwise outlived every real memory and ranked first at a
# confidence of nothing.
WARNING_VALENCE = -0.5
WARNING_INTENSITY = 0.5
SEVERE_VALENCE = -0.7
SEVERE_INTENSITY = 0.7


def valence_stated_sql(table: str = "") -> str:
    """Whether the caller stated the tone of the atom in *table*'s row."""
    column = _metadata_column(table)
    return (
        f"COALESCE(CASE WHEN json_valid({column}) "
        f"THEN json_extract({column}, '$.{VALENCE_STATED}') END, 0) = 1"
    )


# A warning a later claim replaced is history, not a rule: the replacement is
# the one the agent must follow, and injecting both told it to obey each.
CRITICAL_WARNING_SQL = (
    f"({valence_stated_sql()} AND type != 'concept' "
    f"AND {ATOM_OWN_VALENCE} < {WARNING_VALENCE} "
    f"AND {ATOM_OWN_INTENSITY} > {WARNING_INTENSITY} "
    f"AND {claim_current_sql()})"
)

# What the epoch's severe floor holds and what recall lets past the confidence
# floor: whatever is a warning, or a severe failure the user reported.
SEVERE_RETAINED_SQL = (
    f"({CRITICAL_WARNING_SQL} OR ({ATOM_SOURCE} != '{SOURCE_AGENT}' "
    f"AND {ATOM_OWN_VALENCE} < {SEVERE_VALENCE} "
    f"AND {ATOM_OWN_INTENSITY} > {SEVERE_INTENSITY}))"
)


def _type_name(atom) -> str:
    return getattr(atom.type, "value", atom.type)


def is_critical_warning(atom) -> bool:
    """:data:`CRITICAL_WARNING_SQL` for an atom already in memory."""
    return (
        atom.metadata.get(VALENCE_STATED) is True
        and _type_name(atom) != "concept"
        and atom.valence.own < WARNING_VALENCE
        and atom.valence.own_intensity > WARNING_INTENSITY
        and not atom.metadata.get(SUPERSEDED_BY)
    )


def is_severe_retained(atom) -> bool:
    """:data:`SEVERE_RETAINED_SQL` for an atom already in memory."""
    if is_critical_warning(atom):
        return True
    return (
        atom.metadata.get("source", SOURCE_USER) != SOURCE_AGENT
        and atom.valence.own < SEVERE_VALENCE
        and atom.valence.own_intensity > SEVERE_INTENSITY
    )
