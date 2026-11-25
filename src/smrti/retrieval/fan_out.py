"""Salience-scored fan-out retrieval."""
from __future__ import annotations

import logging
import sqlite3
import struct
import time
import weakref
from dataclasses import dataclass
from typing import Callable

import numpy as np

from smrti.core.models import AtomType, RecallResult, atom_from_row
from smrti.core.atomspace import embedding_text
from smrti.core.db import stable_rowid
from smrti.core.provenance import (
    ATOM_FORGOTTEN,
    ATOM_OWN_INTENSITY,
    ATOM_OWN_VALENCE,
    CRITICAL_WARNING_SQL,
    SEVERE_RETAINED_SQL,
    SUPERSEDED_BY,
    WARNING_INTENSITY,
    WARNING_VALENCE,
    claim_current_sql,
    forgotten_sql,
)
from smrti.personality.params import personality_params
from smrti.retrieval.classify import is_critical_warning
from smrti.retrieval.diversify import diversify
from smrti.retrieval.salience import compute_salience
from smrti.retrieval.text import comparable, coverage, lexical_terms, word_set

logger = logging.getLogger("smrti.retrieval.fan_out")

# The KNN entry pool scales with the graph. A fixed pool lets conversational
# froth crowd the gate long before the graph is large: a fact ranked just past
# the cutoff by raw distance never reaches scoring, so truth, attention, and
# valence never get a vote on it. 4·√N grows the pool with the space without
# ever approaching a full scan.
_KNN_POOL_MIN = 50
_KNN_POOL_MAX = 256

# An episode that is a stored copy of the question is an echo, not an
# answer: utterances of "name my family" are always closer to that query
# than any fact about the family, so pure similarity ranks the asking above
# the knowing. An echo carries zero information the asker does not already
# hold, so its similarity term is zeroed — and since relevance gates
# standing, an echo scores nothing at all: the question is never the answer.
#
# Calling one takes two signals, because neither suffices alone. Word
# overlap is measured against the larger of the two word sets
# (``coverage``) with a high bar, so an answer that contains every word of a
# three-word query is not a copy of it while a stored question that gained a
# "please" is — but a short answer shares nearly every word of its question
# too: "My sister's name is Laura." covers 0.83 of "What is my sister's
# name?", the same as the padded echo. The encoder separates those two when
# it is shown the texts themselves: restatements sit at 0.93 and above (a
# "please", a "che,", a dropped question mark) and short answers that share
# their question's words at 0.79 and below, in English, Spanish, Hindi,
# German and French. So a suspect must clear both lines. It is embedded
# again on its own words for the verdict: the stored vector embeds label and
# content together, for an episode its text twice, which pulls a verbatim
# copy of the query down to 0.86 and leaves the answer level with the
# echoes. Suspects are rare — no LongMemEval-S candidate is one — so the
# encoder runs only where a verdict is due.
#
# A pair too short for the overlap bar to mean anything ("deploy friday"
# against "deploy on friday?", where one dropped word is a third of the
# text) needs only half its words in common before the encoder decides,
# which at this line is near-identity; such pairs used to be exempt, and the
# two-word restatement outranked the answer. The words still gate the
# encoder there: a two-word query is short against every candidate, and
# asking the encoder about each one cost 160 embeddings a recall. A text
# whose words are exactly the query's says nothing the query does not,
# whatever the encoder makes of its punctuation ("Kubernetes!" scores 0.58
# against "kubernetes").
#
# Beliefs and concepts are exempt: searching for a fact by stating it must
# return the fact first. So is a stated warning: "never deploy without a
# backup" covers every word of "deploy without a backup", and the overlap
# cannot see the negation that makes the one the answer to the other — the
# one memory the engine promises to deliver is the one the filter was
# discarding. The gate on the stored similarity just skips the work where
# zeroing could not change the ranking anyway.
_ECHO_OVERLAP = 0.8
_ECHO_SHORT_OVERLAP = 0.5
_ECHO_IDENTITY = 0.9
_ECHO_MIN_SIMILARITY = 0.5

# Whether an atom may surface at a given confidence floor. Applied before
# any pool is cut, not after: a forgotten or sunk neighbour that enters a
# bounded pool spends a slot a live memory needed, and enough of them —
# a space where the user forgot a whole topic — left recall empty while
# the answer sat just past the cutoff. The confidence floor is bypassed for
# what the epoch's severe floor keeps — a stated warning, or a severe failure
# the user reported: a decayed warning is still a warning. A forgotten atom
# never passes, whatever its tone.
_ELIGIBLE = f"NOT {ATOM_FORGOTTEN} AND (confidence >= ? OR {SEVERE_RETAINED_SQL})"
# How much wider each repeated KNN probe casts when the filtered pool came
# up short, and how many ids one eligibility query carries.
_REFILL_FACTOR = 4
_ID_CHUNK = 400
# The most neighbours sqlite-vec returns from one probe; asking for more is
# an error, not a shorter answer.
_KNN_MAX_K = 4096

# How long a counted space size stands in for a fresh count. The size only
# sets the KNN pool, 4·√N clamped to 50–256, so a count a few writes old picks
# the same pool, while counting on every recall scanned the whole partition:
# 8–16 ms of a recall at twenty thousand atoms.
_SPACE_SIZE_TTL_SECONDS = 30.0
_counted_sizes: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# 1-hop expansion budget per direction. Ordered by the standing of the atom
# on the far end, so a hub with a hundred edges to trivia cannot evict the
# one edge that leads somewhere worth recalling.
_EXPANSION_EDGES = 100

# The kinds of atom recall returns. Relation edges are how candidates are
# reached, not candidates themselves.
_CANDIDATE_TYPES = "('concept', 'belief', 'episode', 'goal')"

# Reciprocal Rank Fusion constant. Vector distance and BM25 are not on a
# common scale and no normalization between them is stable across graphs, so
# the two lists are fused on rank alone: an atom scores 1/(k + rank) in each
# list it appears in, and the sums decide the entry pool. k=60 is the value
# the original RRF paper settled on; it is large enough that the top of one
# list cannot swamp the whole of the other, so an atom both halves rank
# moderately well beats one that only a single half loves.
_RRF_K = 60

# Cap on how many query terms reach the lexical index. A pasted stack trace is
# still a query, and every term is a separate posting-list walk.
_FTS_MAX_TERMS = 32

# How many BM25 candidates join the fused pool. The lexical half is a
# high-precision supplement, not a co-equal ranking: its head holds the
# proper-noun matches the embedding missed, and its tail is every atom
# sharing a stop-word with the query. Fusing that tail dilutes the scored
# pool — measured on LongMemEval-S, a full-pool fusion cost five points of
# retrieval hit rate and a ten-candidate head cost none, while the
# cross-language recall the index exists for lives entirely in the head.
_FTS_POOL = 10

# The surfacing floor when neither the caller nor the personality names one.
_DEFAULT_MIN_CONFIDENCE = 0.1


def _stored_vectors(atom_ids: list[str], db) -> dict[str, np.ndarray]:
    """The stored vectors of *atom_ids*; an atom stored without one is absent.

    One statement per chunk, joined through a list of values so that vec0
    still answers each id as a point lookup: it answers ``rowid IN (...)`` by
    scanning every vector in the table, six times slower at twenty thousand
    atoms than the one-statement-per-candidate fetch it would replace.
    """
    by_rowid = {stable_rowid(atom_id): atom_id for atom_id in atom_ids}
    rowids = list(by_rowid)
    vectors: dict[str, np.ndarray] = {}
    for start in range(0, len(rowids), _ID_CHUNK):
        chunk = rowids[start : start + _ID_CHUNK]
        values = ",".join("(?)" for _ in chunk)
        for row in db.fetchall(
            f"""WITH wanted(id) AS (VALUES {values})
                SELECT v.rowid AS rowid, v.embedding AS embedding
                FROM wanted CROSS JOIN vec_atoms v ON v.rowid = wanted.id""",
            tuple(chunk),
        ):
            if row["embedding"] is not None:
                vectors[by_rowid[row["rowid"]]] = np.frombuffer(row["embedding"], dtype=np.float32)
    return vectors


def _term_list(text: str) -> list[str]:
    """Query terms in order of first appearance, deduplicated.

    Ordered, because an FTS5 query string has to be built the same way every
    time or the same question returns different candidate pools on different
    runs; and unfolded, because the index's tokenizer does the folding — see
    :func:`lexical_terms`. Spellings that differ only in case query the same
    postings, so the first one stands for the rest.
    """
    seen: dict[str, str] = {}
    for token in lexical_terms(text):
        seen.setdefault(token.lower(), token)
    return list(seen.values())[:_FTS_MAX_TERMS]


def _fts_query(terms: list[str]) -> str:
    """An FTS5 MATCH expression that ORs the query's terms.

    Every term is double-quoted, which in FTS5 makes it a literal string
    rather than syntax — a query containing ``NEAR``, ``OR`` or ``*`` is then
    searched for, not obeyed. Terms are ORed rather than ANDed because recall
    wants candidates: BM25 already ranks the atom carrying more of them first.
    """
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _rrf_fuse(ranked_lists: list[list[str]], limit: int) -> list[str]:
    """Reciprocal Rank Fusion over several ranked id lists.

    Ties break on the id so the same graph and the same query always produce
    the same pool — retrieval that reshuffles under equal evidence is
    untestable and, on a benchmark, unmeasurable.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, atom_id in enumerate(ranked):
            scores[atom_id] = scores.get(atom_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [atom_id for atom_id, _ in ordered[:limit]]


def _is_echo(query_vec: list[float], query_tokens: set[str], text: str, embed_engine) -> bool:
    """True when the text is the query restated, in words and in meaning.

    Word overlap against the larger side: an echo may add a word ("please")
    or drop one and still cover most of both texts, while an answer that
    merely contains the query's words is much longer than the query and does
    not. A text that passes must also be near-identical to the query on the
    encoder — see ``_ECHO_IDENTITY``.
    """
    words = word_set(text)
    if words == query_tokens:
        return True
    bar = _ECHO_OVERLAP if comparable(query_tokens, words) else _ECHO_SHORT_OVERLAP
    if coverage(query_tokens, words) < bar:
        return False
    return _cosine(query_vec, embed_engine.embed(text)) >= _ECHO_IDENTITY


def _cosine(a, b) -> float:
    """Cosine of two vectors; 0 when either has no length, as a zero vector
    is about nothing."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norms = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if norms < 1e-12:
        return 0.0
    return float(a @ b) / norms


def _restates_query(
    atom, similarity: float, query_vec: list[float], query_tokens: set[str], embed_engine
) -> bool:
    """Whether the candidate is an episode echoing the query, which scores
    nothing — unless it is a stated warning; see ``_ECHO_OVERLAP``."""
    return (
        atom.type == AtomType.EPISODE
        and similarity >= _ECHO_MIN_SIMILARITY
        and not is_critical_warning(atom)
        and _is_echo(query_vec, query_tokens, atom.content or atom.label, embed_engine)
    )


def _lexical_entry_points(
    query: str, tenant_id: str, read_spaces: list[str], db, limit: int,
    min_confidence: float,
) -> list[str]:
    """Atom ids for the query's BM25 ranking, best first.

    ``bm25()`` returns a score that is more negative the better the match, so
    ascending order is descending relevance. Only eligible atoms are ranked,
    so the head of the list is never spent on what could not surface.

    Empty when the build has no FTS5, when the query has no searchable terms,
    or when nothing matches — in every case retrieval carries on with the
    vector half alone, which is what it did before the lexical index existed.
    """
    if not db.fts_enabled:
        return []
    terms = _term_list(query)
    if not terms:
        return []
    spaces_ph = ",".join("?" * len(read_spaces))
    rows = db.fetchall(
        f"""SELECT f.atom_id AS atom_id FROM atoms_fts f
            JOIN atoms a ON a.id = f.atom_id
            WHERE atoms_fts MATCH ?
              AND a.tenant_id = ? AND a.space IN ({spaces_ph})
              AND a.type != 'relation'
              AND {_ELIGIBLE}
            ORDER BY bm25(atoms_fts)
            LIMIT ?""",
        (_fts_query(terms), tenant_id, *read_spaces, min_confidence, limit),
    )
    return [r["atom_id"] for r in rows]


def _eligible_ids(
    atom_ids: list[str], tenant_id: str, space: str, db, min_confidence: float
) -> set[str]:
    """The subset of *atom_ids* that may surface in *space* at this floor."""
    eligible: set[str] = set()
    for start in range(0, len(atom_ids), _ID_CHUNK):
        chunk = atom_ids[start : start + _ID_CHUNK]
        ph = ",".join("?" * len(chunk))
        eligible.update(
            r["id"]
            for r in db.fetchall(
                # The unary plus keeps the planner on the primary key: offered
                # the (tenant_id, space) index it scanned the whole partition,
                # 13 ms at twenty thousand atoms against 0.4 ms by id.
                f"SELECT id FROM atoms WHERE id IN ({ph}) AND +tenant_id = ? AND +space = ? AND {_ELIGIBLE}",
                (*chunk, tenant_id, space, min_confidence),
            )
        )
    return eligible


def _knn_entry_points(
    vec_bytes: bytes, tenant_id: str, space: str, db, pool: int, min_confidence: float,
) -> list:
    """The nearest *pool* eligible atoms in one space, ascending distance.

    A KNN probe is cut to its limit before anything looks at the rows, so a
    space whose nearest neighbours are forgotten or sunk hands back a pool
    with nothing in it that can surface. The probe is repeated wider —
    ``_REFILL_FACTOR`` times — for as long as the filtered pool came up short
    and the probe came back full, which on a healthy graph is never and on a
    decayed one a handful of times. A probe that returned fewer rows than it
    asked for has seen the whole space; one that asked for ``_KNN_MAX_K``
    has asked for all sqlite-vec will give.

    vec0 reports no distance at all for a stored vector with no length, and
    such a row is about nothing: it is dropped here, where sorting it would
    have aborted the whole recall.
    """
    limit = pool
    while True:
        rows = db.fetchall(
            """SELECT atom_id, distance FROM vec_atoms
               WHERE embedding MATCH ? AND tenant_id = ? AND space = ?
               ORDER BY distance
               LIMIT ?""",
            (vec_bytes, tenant_id, space, limit),
        )
        eligible = _eligible_ids([r["atom_id"] for r in rows], tenant_id, space, db, min_confidence)
        kept = [r for r in rows if r["distance"] is not None and r["atom_id"] in eligible]
        if len(kept) >= pool or len(rows) < limit or limit >= _KNN_MAX_K:
            return kept[:pool]
        limit = min(limit * _REFILL_FACTOR, _KNN_MAX_K)


def _graph_size(db, tenant_id: str, read_spaces: list[str]) -> int:
    """How many candidate atoms the read spaces hold, counted at most once
    per ``_SPACE_SIZE_TTL_SECONDS`` per space and database."""
    now = time.monotonic()
    counted = _counted_sizes.setdefault(db, {})
    stale = [
        space for space in read_spaces
        if now - counted.get((tenant_id, space), (float("-inf"), 0))[0] > _SPACE_SIZE_TTL_SECONDS
    ]
    if stale:
        ph = ",".join("?" * len(stale))
        sizes = {
            r["space"]: r["n"]
            for r in db.fetchall(
                f"""SELECT space, COUNT(*) AS n FROM atoms
                    WHERE tenant_id = ? AND space IN ({ph}) AND type != 'relation'
                    GROUP BY space""",
                (tenant_id, *stale),
            )
        }
        for space in stale:
            counted[(tenant_id, space)] = (now, sizes.get(space, 0))
    return sum(counted[(tenant_id, space)][1] for space in read_spaces)


# A hook between ranking and selection: given the query and the salience-
# ranked candidates, returns them in the order selection should cut them.
# The ``rerank`` decision task is one; a caller's own reranker is another.
Judge = Callable[[str, list[RecallResult]], list[RecallResult]]


def boost_attention(db, atom_ids: list[str], tenant_id: str, write_space: str, sti_boost: float) -> int:
    """Raise the STI of *atom_ids* within the write space — the access boost.

    Reading a memory is attention, and this is the one write recall makes.
    It is its own function so that a caller who takes a wider candidate set
    (``recall(top_k=40, boost=False)``) and keeps five can boost the five
    it read and not the thirty-five it discarded — and so that nobody
    reaches for ``reinforce()`` in its place, which is evidence about
    truth, not attention.

    A forgotten atom is never raised: attention is what the epoch promotes
    into a long-term floor, and boosting one kept it from the pruner that
    forgetting hands it to. Nor is a relation edge, which recall never
    returns. Returns how many atoms were raised.

    The boost is the one write a read makes, and the read is already done:
    when another writer holds the database past the busy timeout, the
    boost is logged and dropped rather than turning a finished recall into
    an error.
    """
    ids = [atom_id for atom_id in dict.fromkeys(atom_ids) if atom_id]
    if sti_boost <= 0 or not ids:
        return 0
    raised = 0
    try:
        for start in range(0, len(ids), _ID_CHUNK):
            chunk = ids[start : start + _ID_CHUNK]
            ph = ",".join("?" * len(chunk))
            raised += db.execute(
                f"""UPDATE atoms SET sti = MIN(sti + ?, 3.0), updated_at = datetime('now')
                    WHERE id IN ({ph}) AND +tenant_id = ? AND +space = ?
                      AND type != 'relation' AND NOT {ATOM_FORGOTTEN}""",
                (sti_boost, *chunk, tenant_id, write_space),
            ).rowcount
    except sqlite3.OperationalError as exc:
        # SQLite's own message for SQLITE_BUSY and SQLITE_LOCKED; the error
        # code attribute only exists from Python 3.11.
        if "locked" not in str(exc):
            raise
        logger.warning("access boost dropped, the database is locked: %s", exc)
    return raised


def retrieve(
    query: str,
    tenant_id: str,
    read_spaces: list[str],
    db,
    embed_engine,
    write_space: str,
    top_k: int = 10,
    min_confidence: float | None = None,
    boost: bool = True,
    judge: Judge | None = None,
) -> list[RecallResult]:
    """
    Full retrieval pipeline, in three stages — candidates, selection,
    attention — so that what is boosted is what was selected and nothing
    that was merely considered:
      1. Embed query
      2. KNN search in vec_atoms per read_space (tenant + space partitioned),
         merged by cosine distance, fused by Reciprocal Rank Fusion with a
         BM25 search of the same spaces; pool size scales with the graph
      3. 1-hop graph expansion via relation atoms within read_spaces,
         highest-standing endpoints first
      4. Score all candidates by salience (personality-weighted from
         write_space); expanded candidates are scored on their true stored
         similarity, near-verbatim episode echoes of the query are damped,
         and agent-authored atoms are discounted by source trust
      5. Hand the ranked candidates to ``judge`` when one is given (the
         evidence reranker), cap near-duplicate episodes and reserve slots
         for relevant beliefs, then return top_k in the judge's order
         (descending score without one)
      6. Boost the STI of the returned results only, in the write space

    ``min_confidence`` is the surfacing floor. When the caller passes none,
    the write space's personality decides (``min_confidence_to_surface``),
    which is what that parameter is for: a preset that promises a 0.3 floor
    has to deliver it to a caller who did not repeat the number.

    ``boost`` is whether being recalled raises the STI of what came back.
    Reading a memory is attention, so it does by default; a caller that
    retrieves in order to forget passes ``False``, because forgetting a
    memory must not make it more prominent.

    Atoms stamped ``$.forgotten`` are never candidates. ``forget()`` sinks
    them below the floor as well, but the stamp is the guarantee: a memory
    the caller asked to forget stops surfacing at every floor.
    """
    results, sti_boost, floor = rank_candidates(
        query, tenant_id, read_spaces, db, embed_engine, write_space, min_confidence
    )
    top_results = select(query, results, top_k, floor, judge)
    if boost and top_results:
        boost_attention(db, [r.atom.id for r in top_results], tenant_id, write_space, sti_boost)
    return top_results


def select(
    query: str,
    results: list[RecallResult],
    top_k: int,
    min_confidence: float,
    judge: Judge | None = None,
) -> list[RecallResult]:
    """The final results from the ranked candidates: judged, capped, cut.

    ``min_confidence`` here is only the diversity cap's belief floor; the
    surfacing floor was applied while the candidates were gathered.
    """
    if judge is not None and results:
        results = judge(query, results)
    return diversify(results, top_k, min_confidence)


def rank_candidates(
    query: str,
    tenant_id: str,
    read_spaces: list[str],
    db,
    embed_engine,
    write_space: str,
    min_confidence: float | None = None,
    damp_echoes: bool = True,
) -> tuple[list[RecallResult], float, float]:
    """Every eligible candidate the query reaches, ranked by salience, with
    the write space's access boost and the surfacing floor that was applied
    — the first stage of :func:`retrieve`, which makes no write.

    ``damp_echoes=False`` keeps an episode that restates the query; see
    :func:`_restates_query`."""
    # One KNN probe is issued per read space, so a repeated name is repeated
    # work — and read_spaces can arrive straight from a request header.
    # Deduplicating in order also keeps the ``space IN (...)`` lists tight.
    read_spaces = list(dict.fromkeys(read_spaces))
    spaces_ph = ",".join("?" * len(read_spaces))

    query_vec = embed_engine.embed(query)
    vec_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
    query_tokens = word_set(query)

    # Load personality weights from write_space, fall back to defaults.
    # A NULL column must fall back too, not propagate None into arithmetic.
    personality = db.fetchone(
        "SELECT * FROM personality WHERE tenant_id = ? AND space = ?",
        (tenant_id, write_space),
    )
    p = personality_params(personality)

    def _pget(key: str, default: float) -> float:
        value = p.get(key)
        return default if value is None else value

    w_similarity = _pget("w_similarity", 0.35)
    w_sti = _pget("w_sti", 0.25)
    w_confidence = _pget("w_confidence", 0.20)
    w_lti = _pget("w_lti", 0.10)
    w_valence = _pget("w_valence", 0.10)
    valence_weight = _pget("valence_weight", 0.2)
    sti_boost = _pget("sti_boost_on_access", 0.5)
    agent_trust = _pget("agent_source_trust", 0.5)
    if min_confidence is None:
        min_confidence = _pget("min_confidence_to_surface", _DEFAULT_MIN_CONFIDENCE)

    # Step 1: entry points — the vector and lexical searches run side by side
    # and their ranked lists are fused. One probe per read_space (the space
    # partition key only supports equality during KNN), merged by ascending
    # distance so no space can starve the others out of the candidate budget.
    graph_size = _graph_size(db, tenant_id, read_spaces)
    knn_pool = max(_KNN_POOL_MIN, min(_KNN_POOL_MAX, 4 * int(graph_size**0.5)))
    knn_rows: list = []
    # A query vector with no length is near nothing — vec0 gives every row
    # a NULL distance against it — so the lexical half searches alone.
    if np.linalg.norm(query_vec) > 0.0:
        for space in read_spaces:
            knn_rows.extend(
                _knn_entry_points(vec_bytes, tenant_id, space, db, knn_pool, min_confidence)
            )
    knn_rows.sort(key=lambda r: r["distance"])
    knn_rows = knn_rows[:knn_pool]

    knn_ids = [r["atom_id"] for r in knn_rows]
    knn_distances = {r["atom_id"]: r["distance"] for r in knn_rows}

    # The lexical half. It earns its place on the queries the embedding gets
    # wrong: a fact stored in one language sits a long way from the question
    # that asks for it in another, while the proper nouns both of them carry
    # are byte-identical.
    lexical_ids = _lexical_entry_points(
        query, tenant_id, read_spaces, db, min(_FTS_POOL, knn_pool), min_confidence
    )
    entry_ids = _rrf_fuse([knn_ids, lexical_ids], knn_pool)

    if not entry_ids:
        return [], sti_boost, min_confidence

    # Step 2: 1-hop expansion via relation atoms, capped so a hub atom cannot
    # pull an unbounded neighborhood into the scoring set. The budget goes to
    # the highest-standing endpoints first: an unordered LIMIT hands it to
    # whatever the scan meets, which in a chatty graph is froth. And it goes
    # only to endpoints that could be returned — eligible, a candidate kind,
    # in a read space — since a hub's hundred forgotten or sunk neighbours
    # used to take the whole budget and be dropped afterwards, while its one
    # live neighbour waited past the cut. An endpoint reached by several
    # edges takes one place.
    #
    # Nothing else enters. An earlier version also added the three
    # highest-standing person atoms to every candidate set "to anchor the
    # graph"; scored on their true similarity they carried none for most
    # queries, and their standing alone put them above every relevant
    # episode — a person concept at similarity zero was the first result for
    # a question about Kubernetes. Candidacy comes from the query or from an
    # edge out of something the query found.
    id_ph = ",".join("?" * len(entry_ids))
    expanded_ids: set[str] = set(entry_ids)
    for near, far in (("source_id", "target_id"), ("target_id", "source_id")):
        expanded_ids.update(
            r["id"]
            for r in db.fetchall(
                f"""SELECT id FROM atoms
                    WHERE id IN (SELECT {far} FROM atoms
                                 WHERE {near} IN ({id_ph}) AND +type = 'relation'
                                   AND +tenant_id = ? AND +space IN ({spaces_ph}))
                      AND +tenant_id = ? AND +space IN ({spaces_ph})
                      AND +type IN {_CANDIDATE_TYPES} AND {_ELIGIBLE}
                    ORDER BY (sti + lti + confidence) DESC LIMIT {_EXPANSION_EDGES}""",
                (*entry_ids, tenant_id, *read_spaces, tenant_id, *read_spaces, min_confidence),
            )
        )

    # Step 3: Fetch candidate atoms — space-filtered here (overlay boundary),
    # and eligibility-filtered again for what the lexical half found.
    exp_list = list(expanded_ids)
    exp_ph = ",".join("?" * len(exp_list))
    atoms_rows = db.fetchall(
        f"""SELECT * FROM atoms
            WHERE id IN ({exp_ph})
              AND +tenant_id = ?
              AND +space IN ({spaces_ph})
              AND +type IN {_CANDIDATE_TYPES}
              AND {_ELIGIBLE}""",
        (*exp_list, tenant_id, *read_spaces, min_confidence),
    )

    # Step 4: Score each candidate by salience.
    #
    # Similarity comes from the probe's distance when the probe found the
    # candidate. One that entered through the graph or the lexical index is
    # scored on its real stored embedding — handing it similarity 0 lets a
    # sticky concept with no bearing on the query outrank the fact one edge
    # away from a hit. One stored with no vector at all (its encoder failed
    # at write time, and the lexical index is how it got here) is embedded
    # now: at similarity 0 it has no salience and is dropped, which made the
    # fallback that kept the atom a fallback that kept it unreachable. The
    # epoch files the vector for good.
    query = np.asarray(query_vec, dtype=np.float32)
    stored = _stored_vectors([r["id"] for r in atoms_rows if r["id"] not in knn_distances], db)
    results: list[RecallResult] = []
    for row in atoms_rows:
        atom = None
        if row["id"] in knn_distances:
            similarity = max(0.0, 1.0 - knn_distances[row["id"]])
        else:
            vector = stored.get(row["id"])
            if vector is None:
                atom = atom_from_row(row)
                vector = embed_engine.embed(embedding_text(atom))
            similarity = max(0.0, _cosine(query, vector))
        # A candidate the query does not reach is no result, and is dropped
        # before its row is turned into an atom — most of what a candidate
        # costs. The KNN probe returns its nearest neighbours however far they
        # are, and on a small graph that is every atom in it.
        if similarity <= 0.0:
            continue
        atom = atom or atom_from_row(row)
        # An episode that echoes the query scores nothing, unless the caller
        # is looking for the memory itself rather than an answer
        # (``damp_echoes=False``, which is forget's case: quoting a memory is
        # the natural way to name the one to drop).
        if damp_echoes and _restates_query(atom, similarity, query_vec, query_tokens, embed_engine):
            continue
        # The engine already trusts agent-authored content less at decay and
        # prune time; ranking is where that asymmetry reaches the reader. An
        # agent's stored reply competes with the user testimony it was
        # derived from — and when the reply was wrong, ranking it first
        # re-serves the mistake as memory.
        #
        # The discount reaches the standing terms only. Attention, confidence
        # and valence say how much the graph has come to trust the atom, and
        # an agent's say in that is what the discount exists to shrink;
        # similarity says how much the atom is *about the question*, which is
        # a property of the query. Discounting it too buried the one memory
        # that held the answer whenever that memory was the agent's own reply
        # — "what did you recommend?" has no user-authored answer, and
        # measured on LongMemEval the whole single-session-assistant category
        # scored zero. On equal relevance the user's version still wins.
        standing_scale = agent_trust if atom.metadata.get("source") == "agent" else 1.0
        salience = compute_salience(
            similarity=similarity,
            sti=atom.attention.sti,
            confidence=atom.truth.confidence,
            lti=atom.attention.lti,
            valence=atom.valence.own,
            intensity=atom.valence.own_intensity,
            w_similarity=w_similarity,
            w_sti=w_sti,
            w_confidence=w_confidence,
            w_lti=w_lti,
            w_valence=w_valence,
            valence_weight=valence_weight,
            standing_scale=standing_scale,
        )
        # A result with no salience is not a result, whatever weights the
        # personality carries.
        if salience <= 0.0:
            continue
        results.append(RecallResult(atom=atom, salience=salience, similarity=similarity))

    results.sort(key=lambda r: r.salience, reverse=True)
    # The diversity cap and the cut to top_k run in ``select``: ranking is
    # per-atom and has no opinion about the shape of the set it produces,
    # which is how five copies of a single exchange came to be a whole
    # response. The access boost runs after that, on the write space only —
    # reading from a foreign space must not mutate that space's attention
    # weights, and reading a candidate the cut discarded is not reading.
    return results, sti_boost, min_confidence


# How similar a stated warning must be to the query for find_warnings to
# deliver it. Measured on the multilingual encoder, rules and the requests
# they govern sit at 0.31–0.57 — "never run migrations without a backup"
# against "run the database migration", an English rule against the Spanish
# request to deploy — and unrelated requests at 0.22 and below.
WARNING_MIN_SIMILARITY = 0.3


def find_warnings(
    query: str,
    tenant_id: str,
    read_spaces: list[str],
    db,
    embed_engine,
    limit: int = 3,
    min_similarity: float = WARNING_MIN_SIMILARITY,
) -> list[RecallResult]:
    """The stated warnings in the read spaces that bear on *query*, most
    similar first, at most *limit*.

    Found apart from recall's candidate pool. A warning seventy memories
    further from the query than the nearest ones never entered the KNN pool,
    and one that did could still be cut at top_k, so the one memory the
    engine promises to deliver was the one a busy graph lost. Every
    critical warning in the read spaces is scored against the query on its
    stored vector instead — there are few of them — and those at or above
    *min_similarity* are returned. Similarity alone ranks them, and is what
    each result carries as its salience: a rule that bears on the request
    applies however little attention it has had. Forgotten warnings are
    never returned, and nothing is written.

    The scan reads the whole partition, cheap tone columns first: 12 ms at
    twenty thousand atoms.
    """
    read_spaces = list(dict.fromkeys(read_spaces))
    if limit <= 0 or not read_spaces:
        return []
    spaces_ph = ",".join("?" * len(read_spaces))
    rows = db.fetchall(
        f"""SELECT * FROM atoms
            WHERE tenant_id = ? AND space IN ({spaces_ph})
              AND +type IN {_CANDIDATE_TYPES}
              AND {ATOM_OWN_VALENCE} < {WARNING_VALENCE}
              AND {ATOM_OWN_INTENSITY} > {WARNING_INTENSITY}
              AND NOT {ATOM_FORGOTTEN} AND {CRITICAL_WARNING_SQL}""",
        (tenant_id, *read_spaces),
    )
    if not rows:
        return []
    query_vec = embed_engine.embed(query)
    stored = _stored_vectors([r["id"] for r in rows], db)
    found: list[RecallResult] = []
    for row in rows:
        atom = None
        vector = stored.get(row["id"])
        if vector is None:
            atom = atom_from_row(row)
            vector = embed_engine.embed(embedding_text(atom))
        similarity = _cosine(query_vec, vector)
        if similarity >= min_similarity:
            atom = atom or atom_from_row(row)
            found.append(RecallResult(atom=atom, salience=similarity, similarity=similarity))
    found.sort(key=lambda r: (-r.similarity, r.atom.id))
    return found[:limit]


@dataclass(frozen=True)
class SupersededClaim:
    """A claim a later one replaced: *subject* *predicate* *old_object*,
    replaced by the same claim about *new_object* (None when the replacing
    claim is gone or its object forgotten)."""

    edge_id: str
    subject: str
    predicate: str
    old_object: str
    new_object: str | None


def superseded_claims(
    atom_ids: list[str], tenant_id: str, read_spaces: list[str], db, limit: int = 3
) -> dict[str, list[SupersededClaim]]:
    """The superseded claims about each of *atom_ids*, keyed by the atom.

    Recall never returns claim edges, so a claim the user replaced — "lives
    in Berlin", now "lives in Paris" — could not reach the caller at all,
    and the city it named came back looking current. This names it beside
    the recalled atoms it was about, its subject or its old object, as an
    annotation for the caller to render rather than a result taking a top_k
    slot: at most *limit* per atom, newest claim first. Atoms with no
    superseded claim are absent. A forgotten claim, subject or object is
    never named.
    """
    ids = list(dict.fromkeys(atom_id for atom_id in atom_ids if atom_id))
    read_spaces = list(dict.fromkeys(read_spaces))
    if not ids or not read_spaces or limit <= 0:
        return {}
    claims: dict[str, list[SupersededClaim]] = {}
    # Each id is bound twice, so a chunk holds half of what an id list may.
    for start in range(0, len(ids), _ID_CHUNK // 2):
        chunk = ids[start : start + _ID_CHUNK // 2]
        _collect_superseded(chunk, tenant_id, read_spaces, db, limit, claims)
    return claims


def _collect_superseded(
    ids: list[str], tenant_id: str, read_spaces: list[str], db, limit: int,
    claims: dict[str, list[SupersededClaim]],
) -> None:
    """Add the superseded claims about *ids* to *claims*, *limit* per atom.

    Only the ids of this chunk are credited: an edge whose other endpoint
    sits in another chunk is credited to it when that chunk is read.
    """
    ids_ph = ",".join("?" * len(ids))
    spaces_ph = ",".join("?" * len(read_spaces))
    rows = db.fetchall(
        f"""SELECT r.id, r.source_id, r.target_id, r.relation,
                   s.label AS subject, o.label AS old_object, n.label AS new_object
            FROM atoms r
            JOIN atoms s ON s.id = r.source_id
            JOIN atoms o ON o.id = r.target_id
            LEFT JOIN atoms e ON e.tenant_id = r.tenant_id AND e.id =
                (CASE WHEN json_valid(r.metadata) THEN json_extract(r.metadata, '$.{SUPERSEDED_BY}') END)
            LEFT JOIN atoms n ON n.id = e.target_id AND NOT {forgotten_sql('n')}
            WHERE (r.source_id IN ({ids_ph}) OR r.target_id IN ({ids_ph}))
              AND +r.type = 'relation' AND +r.tenant_id = ? AND +r.space IN ({spaces_ph})
              AND NOT {claim_current_sql('r')}
              AND NOT {forgotten_sql('r')} AND NOT {forgotten_sql('s')} AND NOT {forgotten_sql('o')}
            ORDER BY r.created_at DESC, r.id""",
        (*ids, *ids, tenant_id, *read_spaces),
    )
    wanted = set(ids)
    for row in rows:
        claim = SupersededClaim(
            edge_id=row["id"], subject=row["subject"], predicate=row["relation"],
            old_object=row["old_object"], new_object=row["new_object"],
        )
        for atom_id in dict.fromkeys((row["source_id"], row["target_id"])):
            if atom_id in wanted and len(claims.setdefault(atom_id, [])) < limit:
                claims[atom_id].append(claim)
