"""Salience-scored fan-out retrieval."""
from __future__ import annotations

import struct

from smrti.core.models import AtomType, RecallResult, atom_from_row
from smrti.core.provenance import ATOM_OWN_INTENSITY, ATOM_OWN_VALENCE
from smrti.retrieval.diversify import diversify
from smrti.retrieval.salience import compute_salience
from smrti.retrieval.text import containment, word_set, words

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
# near-duplicate of the query as *text*, so token containment separates
# them cleanly where cosine cannot. An echo carries zero information the
# asker does not already hold, so its similarity term is zeroed — it stays
# findable on attention, confidence, and valence, just never on being the
# question. Beliefs and concepts are exempt: searching for a fact by
# stating it must return the fact first. The similarity gate below just
# skips the token work where zeroing could not change the ranking anyway.
_ECHO_OVERLAP = 0.7
_ECHO_MIN_SIMILARITY = 0.5

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

    Containment coefficient over word sets: an echo may add a word
    ("please") or drop one, so the overlap is measured against the smaller
    side. Very short texts are never echoes — two words in common prove
    nothing, which ``containment`` reports as zero overlap.
    """
    return containment(query_tokens, word_set(text)) >= _ECHO_OVERLAP


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_entry_points(
    query: str, tenant_id: str, read_spaces: list[str], db, limit: int
) -> list[str]:
    """Atom ids for the query's BM25 ranking, best first.

    ``bm25()`` returns a score that is more negative the better the match, so
    ascending order is descending relevance.

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
            ORDER BY bm25(atoms_fts)
            LIMIT ?""",
        (_fts_query(terms), tenant_id, *read_spaces, limit),
    )
    return [r["atom_id"] for r in rows]


def retrieve(
    query: str,
    tenant_id: str,
    read_spaces: list[str],
    db,
    embed_engine,
    write_space: str,
    top_k: int = 10,
    min_confidence: float = 0.1,
) -> list[RecallResult]:
    """
    Full retrieval pipeline:
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
      5. Cap near-duplicate episodes and reserve slots for beliefs, then
         return top_k sorted by descending salience
    """
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

    # Step 1: entry points — the vector and lexical searches run side by side
    # and their ranked lists are fused. One probe per read_space (the space
    # partition key only supports equality during KNN), merged by ascending
    # distance so no space can starve the others out of the candidate budget.
    size_row = db.fetchone(
        f"SELECT COUNT(*) AS n FROM atoms WHERE tenant_id = ? AND space IN ({spaces_ph}) AND type != 'relation'",
        (tenant_id, *read_spaces),
    )
    graph_size = size_row["n"] if size_row else 0
    knn_pool = max(_KNN_POOL_MIN, min(_KNN_POOL_MAX, 4 * int(graph_size**0.5)))
    knn_rows: list = []
    for space in read_spaces:
        knn_rows.extend(
            db.fetchall(
                """SELECT atom_id, distance FROM vec_atoms
                   WHERE embedding MATCH ? AND tenant_id = ? AND space = ?
                   ORDER BY distance
                   LIMIT ?""",
                (vec_bytes, tenant_id, space, knn_pool),
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
    lexical_ids = _lexical_entry_points(query, tenant_id, read_spaces, db, knn_pool)
    entry_ids = _rrf_fuse([knn_ids, lexical_ids], knn_pool)

    if not entry_ids:
        return []

    # Step 2: 1-hop expansion via relation atoms, capped so a hub atom cannot
    # pull an unbounded neighborhood into the scoring set. The budget goes to
    # the highest-standing endpoints first: an unordered LIMIT hands it to
    # whatever the scan meets, which in a chatty graph is froth.
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

    # Include the most salient person atoms — they anchor the knowledge graph
    # ("who are these memories about?") regardless of query similarity. Capped
    # so person-heavy spaces (e.g. simulations) cannot flood the candidate set.
    person_rows = db.fetchall(
        f"""SELECT id FROM atoms WHERE entity_type = 'person' AND tenant_id = ? AND space IN ({spaces_ph})
            AND type IN ('concept', 'belief', 'goal')
            ORDER BY (sti + lti) DESC LIMIT 3""",
        (tenant_id, *read_spaces),
    )
    expanded_ids.update(r["id"] for r in person_rows)

    if not expanded_ids:
        return []

    # Step 3: Fetch candidate atoms — space-filtered here (overlay boundary)
    exp_list = list(expanded_ids)
    exp_ph = ",".join("?" * len(exp_list))
    atoms_rows = db.fetchall(
        f"""SELECT * FROM atoms
            WHERE id IN ({exp_ph})
              AND tenant_id = ?
              AND space IN ({spaces_ph})
              AND type IN ('concept', 'belief', 'episode', 'goal')
              AND (confidence >= ?
                   OR ({ATOM_OWN_VALENCE} < -0.5 AND {ATOM_OWN_INTENSITY} > 0.5))""",
        (*exp_list, tenant_id, *read_spaces, min_confidence),
    )

    # Step 4: Score each candidate by salience. Candidates that entered
    # through the graph rather than KNN are scored on their real stored
    # embedding — handing them similarity 0 lets a sticky concept with no
    # bearing on the query outrank the fact one edge away from a hit.
    results: list[RecallResult] = []
    for row in atoms_rows:
        atom = atom_from_row(row)
        if atom.id in knn_distances:
            similarity = max(0.0, 1.0 - knn_distances[atom.id])
        else:
            emb_row = db.fetchone(
                "SELECT embedding FROM vec_atoms WHERE atom_id = ?", (atom.id,)
            )
            similarity = (
                max(0.0, _cosine(query_vec, _blob_to_vec(emb_row["embedding"])))
                if emb_row and emb_row["embedding"] is not None
                else 0.0
            )
        if (
            atom.type == AtomType.EPISODE
            and similarity >= _ECHO_MIN_SIMILARITY
            and _is_echo(query_tokens, atom.content or atom.label)
        ):
            similarity = 0.0
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
        )
        # The engine already trusts agent-authored content less at decay and
        # prune time; ranking is where that asymmetry reaches the reader. An
        # agent's stored reply competes with the user testimony it was
        # derived from — and when the reply was wrong, ranking it first
        # re-serves the mistake as memory.
        if atom.metadata.get("source") == "agent":
            salience *= agent_trust
        results.append(RecallResult(atom=atom, salience=salience, similarity=similarity))

    results.sort(key=lambda r: r.salience, reverse=True)
    # Step 5: cap how much of the answer one conversational moment may fill.
    # Ranking is per-atom and has no opinion about the shape of the set it
    # produces, which is how five copies of a single exchange came to be a
    # whole response.
    top_results = diversify(results, top_k, min_confidence)

    # Boost STI on accessed atoms within write_space only — reading from a
    # foreign space must not mutate that space's attention weights.
    if sti_boost > 0 and top_results:
        db.execute_many(
            "UPDATE atoms SET sti = MIN(sti + ?, 3.0), updated_at = datetime('now') WHERE id = ? AND tenant_id = ? AND space = ?",
            [(sti_boost, r.atom.id, tenant_id, write_space) for r in top_results],
        )

    return top_results
