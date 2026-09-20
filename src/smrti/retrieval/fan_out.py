"""Salience-scored fan-out retrieval."""
from __future__ import annotations

import struct
from typing import Callable

from smrti.core.models import AtomType, RecallResult, atom_from_row
from smrti.core.atomspace import embedding_text
from smrti.core.db import stable_rowid
from smrti.core.provenance import (
    ATOM_FORGOTTEN,
    ATOM_OWN_INTENSITY,
    ATOM_OWN_VALENCE,
)
from smrti.retrieval.classify import is_critical_warning
from smrti.retrieval.diversify import diversify
from smrti.retrieval.salience import compute_salience
from smrti.retrieval.text import coverage, word_set, words

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
# the knowing. Embedding distance cannot tell an echo from an answer that
# shares the query's vocabulary — both sit high — but an echo is a
# near-duplicate of the query as *text*, so word overlap separates them
# where cosine cannot. An echo carries zero information the asker does not
# already hold, so its similarity term is zeroed — and since relevance gates
# standing, an echo scores nothing at all: the question is never the answer.
#
# Because the verdict is now final, the overlap is measured against the
# larger of the two word sets (``coverage``), not the smaller, and the bar is
# high: an answer that contains every word of a three-word query is not a
# copy of it, and neither is one that adds two words to a seven-word
# question, while a stored question that gained a "please" still clears it.
# Beliefs and concepts are exempt: searching for a fact by stating it must
# return the fact first. So is a stated warning: "never deploy without a
# backup" covers every word of "deploy without a backup", and the overlap
# cannot see the negation that makes the one the answer to the other — the
# one memory the engine promises to deliver is the one the filter was
# discarding. The similarity gate below just skips the token work where
# zeroing could not change the ranking anyway.
_ECHO_OVERLAP = 0.8
_ECHO_MIN_SIMILARITY = 0.5

# Whether an atom may surface at a given confidence floor. Applied before
# any pool is cut, not after: a forgotten or sunk neighbour that enters a
# bounded pool spends a slot a live memory needed, and enough of them —
# a space where the user forgot a whole topic — left recall empty while
# the answer sat just past the cutoff. The confidence floor is bypassed for
# an atom whose own tone is severely negative: a decayed warning is still a
# warning. A forgotten atom never passes, whatever its tone.
_ELIGIBLE = (
    f"NOT {ATOM_FORGOTTEN} AND (confidence >= ? "
    f"OR ({ATOM_OWN_VALENCE} < -0.5 AND {ATOM_OWN_INTENSITY} > 0.5))"
)
# How much wider each repeated KNN probe casts when the filtered pool came
# up short, and how many ids one eligibility query carries.
_REFILL_FACTOR = 4
_ID_CHUNK = 400

# 1-hop expansion budget per direction. Ordered by the standing of the atom
# on the far end, so a hub with a hundred edges to trivia cannot evict the
# one edge that leads somewhere worth recalling.
_EXPANSION_EDGES = 100

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


def _blob_to_vec(blob) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _term_list(text: str) -> list[str]:
    """Query terms in order of first appearance, deduplicated.

    Same tokenization as the echo test, but ordered: an FTS5 query string has
    to be built the same way every time or the same question returns different
    candidate pools on different runs.
    """
    seen: dict[str, None] = {}
    for token in words(text):
        seen.setdefault(token, None)
    return list(seen)[:_FTS_MAX_TERMS]


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


def _is_echo(query_tokens: set[str], text: str) -> bool:
    """True when the text is substantially the query restated.

    Word overlap against the larger side: an echo may add a word ("please")
    or drop one and still cover most of both texts, while an answer that
    merely contains the query's words is much longer than the query and does
    not. Very short texts are never echoes — two words in common prove
    nothing, which ``coverage`` reports as zero overlap.
    """
    return coverage(query_tokens, word_set(text)) >= _ECHO_OVERLAP


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _similarity(
    atom, query_vec: list[float], query_tokens: set[str], knn_distances: dict[str, float],
    db, embed_engine,
) -> float:
    """How much the candidate is about the query.

    From the probe's distance when the probe found it. A candidate that
    entered through the graph instead is scored on its real stored embedding
    — handing it similarity 0 lets a sticky concept with no bearing on the
    query outrank the fact one edge away from a hit. One stored with no
    vector at all (its encoder failed at write time, and the lexical index
    is how it got here) is embedded now: at similarity 0 it has no salience
    and is dropped, which made the fallback that kept the atom a fallback
    that kept it unreachable. The epoch files the vector for good.

    An episode that echoes the query scores nothing, unless it is a stated
    warning — see ``_ECHO_OVERLAP``.
    """
    if atom.id in knn_distances:
        similarity = max(0.0, 1.0 - knn_distances[atom.id])
    else:
        emb_row = db.fetchone(
            "SELECT embedding FROM vec_atoms WHERE rowid = ?",
            (stable_rowid(atom.id),),
        )
        if emb_row and emb_row["embedding"] is not None:
            stored = _blob_to_vec(emb_row["embedding"])
        else:
            stored = embed_engine.embed(embedding_text(atom))
        similarity = max(0.0, _cosine(query_vec, stored))
    if (
        atom.type == AtomType.EPISODE
        and similarity >= _ECHO_MIN_SIMILARITY
        and not is_critical_warning(atom)
        and _is_echo(query_tokens, atom.content or atom.label)
    ):
        return 0.0
    return similarity


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
                f"SELECT id FROM atoms WHERE id IN ({ph}) AND tenant_id = ? AND space = ? AND {_ELIGIBLE}",
                (*chunk, tenant_id, space, min_confidence),
            )
        )
    return eligible


def _knn_entry_points(
    vec_bytes: bytes, tenant_id: str, space: str, db, pool: int,
    min_confidence: float, space_size: int,
) -> list:
    """The nearest *pool* eligible atoms in one space, ascending distance.

    A KNN probe is cut to its limit before anything looks at the rows, so a
    space whose nearest neighbours are forgotten or sunk hands back a pool
    with nothing in it that can surface. The probe is repeated wider —
    ``_REFILL_FACTOR`` times — for as long as the filtered pool came up short
    and the space had rows past the cutoff to offer, which on a healthy graph
    is never and on a decayed one a handful of times.
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
        kept = [r for r in rows if r["atom_id"] in eligible]
        if len(kept) >= pool or len(rows) < limit or limit >= space_size:
            return kept[:pool]
        limit = min(limit * _REFILL_FACTOR, space_size)


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
    truth, not attention. Returns how many ids were written.
    """
    ids = [atom_id for atom_id in dict.fromkeys(atom_ids) if atom_id]
    if sti_boost <= 0 or not ids:
        return 0
    db.execute_many(
        "UPDATE atoms SET sti = MIN(sti + ?, 3.0), updated_at = datetime('now') WHERE id = ? AND tenant_id = ? AND space = ?",
        [(sti_boost, atom_id, tenant_id, write_space) for atom_id in ids],
    )
    return len(ids)


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
         for beliefs, then return top_k sorted by descending score
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
) -> tuple[list[RecallResult], float, float]:
    """Every eligible candidate the query reaches, ranked by salience, with
    the write space's access boost and the surfacing floor that was applied
    — the first stage of :func:`retrieve`, which makes no write."""
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
    p = dict(personality) if personality else {}

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
    space_sizes = {
        r["space"]: r["n"]
        for r in db.fetchall(
            f"""SELECT space, COUNT(*) AS n FROM atoms
                WHERE tenant_id = ? AND space IN ({spaces_ph}) AND type != 'relation'
                GROUP BY space""",
            (tenant_id, *read_spaces),
        )
    }
    graph_size = sum(space_sizes.values())
    knn_pool = max(_KNN_POOL_MIN, min(_KNN_POOL_MAX, 4 * int(graph_size**0.5)))
    knn_rows: list = []
    for space in read_spaces:
        knn_rows.extend(
            _knn_entry_points(
                vec_bytes, tenant_id, space, db, knn_pool, min_confidence,
                space_sizes.get(space, 0),
            )
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
    # whatever the scan meets, which in a chatty graph is froth.
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

    forward = db.fetchall(
        f"""SELECT r.target_id FROM atoms r JOIN atoms t ON t.id = r.target_id
            WHERE r.source_id IN ({id_ph}) AND r.type = 'relation'
              AND r.tenant_id = ? AND r.space IN ({spaces_ph})
            ORDER BY (t.sti + t.lti + t.confidence) DESC LIMIT {_EXPANSION_EDGES}""",
        (*entry_ids, tenant_id, *read_spaces),
    )
    expanded_ids.update(r["target_id"] for r in forward if r["target_id"])

    backward = db.fetchall(
        f"""SELECT r.source_id FROM atoms r JOIN atoms s ON s.id = r.source_id
            WHERE r.target_id IN ({id_ph}) AND r.type = 'relation'
              AND r.tenant_id = ? AND r.space IN ({spaces_ph})
            ORDER BY (s.sti + s.lti + s.confidence) DESC LIMIT {_EXPANSION_EDGES}""",
        (*entry_ids, tenant_id, *read_spaces),
    )
    expanded_ids.update(r["source_id"] for r in backward if r["source_id"])
    expanded_ids.discard(None)

    if not expanded_ids:
        return [], sti_boost, min_confidence

    # Step 3: Fetch candidate atoms — space-filtered here (overlay boundary),
    # and eligibility-filtered again for what the expansion pulled in.
    exp_list = list(expanded_ids)
    exp_ph = ",".join("?" * len(exp_list))
    atoms_rows = db.fetchall(
        f"""SELECT * FROM atoms
            WHERE id IN ({exp_ph})
              AND tenant_id = ?
              AND space IN ({spaces_ph})
              AND type IN ('concept', 'belief', 'episode', 'goal')
              AND {_ELIGIBLE}""",
        (*exp_list, tenant_id, *read_spaces, min_confidence),
    )

    # Step 4: Score each candidate by salience.
    results: list[RecallResult] = []
    for row in atoms_rows:
        atom = atom_from_row(row)
        similarity = _similarity(atom, query_vec, query_tokens, knn_distances, db, embed_engine)
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
        # An atom with no relevance has no salience, and a result with no
        # salience is not a result: the KNN probe always returns its nearest
        # neighbours however far they are, and on a small graph that is
        # every atom in it.
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
