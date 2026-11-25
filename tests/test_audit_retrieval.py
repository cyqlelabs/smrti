"""Regression tests for the retrieval defects the audit reproduced.

The echo tests run on the real multilingual encoder, since the line they pin
was calibrated on it. Everything else uses an embedder answering from a
table, so each similarity is exactly the one the scenario needs.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random

import pytest

from smrti import Smrti
from smrti.core.models import Atom, AtomType, AttentionValue, TruthValue, Valence


DIM = 384


def _at(similarity: float, axis: int) -> list[float]:
    """A unit vector whose cosine with the query vector (axis 0) is *similarity*."""
    vec = [0.0] * DIM
    vec[0] = similarity
    vec[axis] = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return vec


QUERY = _at(1.0, 1)


class _TableEmbedder:
    """Embeds a text from a table, or as a deterministic random unit vector."""

    def __init__(self) -> None:
        self.table: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text in self.table:
            return list(self.table[text])
        rng = random.Random(int(hashlib.sha256(text.encode()).hexdigest(), 16))
        vec = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def mem(tmp_path):
    return Smrti(db_path=str(tmp_path / "audit.db"), personality="balanced",
                 tenant_id="t", write_space="s")


@pytest.fixture
def table_mem(tmp_path):
    m = Smrti(db_path=str(tmp_path / "table.db"), personality="balanced",
              tenant_id="t", write_space="s")
    m.embed = m.atomspace._embed = _TableEmbedder()
    return m


_axes = itertools.cycle(range(2, DIM))


def _add(m, text: str, similarity: float, type: str = "episode", **cols) -> str:
    """Store *text* at exactly *similarity* to ``QUERY``, then set *cols*."""
    vec = _at(similarity, next(_axes) if similarity < 1.0 else 1)
    metadata = cols.pop("metadata", {})
    atom = Atom(type=AtomType(type), label=text[:100], content=text,
                truth=TruthValue(probability=cols.pop("probability", 0.5),
                                 confidence=cols.get("confidence", 0.5)),
                attention=AttentionValue(), valence=Valence(
                    valence=cols.pop("valence", 0.0), intensity=cols.pop("intensity", 0.0)),
                tenant_id=m.tenant_id, space=cols.pop("space", m.write_space), metadata=metadata)
    atom_id = m.atomspace.add_atom(atom, embedding=vec)
    if cols:
        sets = ", ".join(f"{k} = ?" for k in cols)
        m.db.execute(f"UPDATE atoms SET {sets} WHERE id = ?", (*cols.values(), atom_id))
    m.embed.table[text] = vec
    return atom_id


# ── T11: a short answer is not an echo of its question ────────────────────────

@pytest.mark.parametrize("question, answer", [
    ("What is my sister's name?", "My sister's name is Laura."),
    ("¿Cómo se llama mi hermana?", "Mi hermana se llama Laura."),
    ("मेरी बहन का नाम क्या है", "मेरी बहन का नाम लौरा है"),
])
def test_a_short_answer_sharing_the_questions_words_is_returned_first(mem, question, answer):
    """The answer covers 0.83 of its question's words, as a padded echo does,
    and was zeroed as one; it sits far below the near-identity of a real
    restatement."""
    answer_id = mem.remember(answer)
    mem.remember("The team offsite is planned for October.")
    results = mem.recall(question, min_confidence=0.0, boost=False)
    assert results and results[0].atom.id == answer_id


def test_a_restatement_too_short_to_compare_is_still_an_echo(mem):
    """Two words in common prove nothing, so short pairs were never echoes and
    "deploy friday" outranked the answer; near-identity now decides them, and
    a short answer that is not near-identical still surfaces."""
    echo = mem.remember("deploy friday")
    answer = mem.remember("we never deploy on friday because the on-call is thin")
    hits = {r.atom.id for r in mem.recall("deploy on friday?", min_confidence=0.0, boost=False)}
    assert echo not in hits and answer in hits

    short_answer = mem.remember("favorite color: blue")
    hits = {r.atom.id for r in mem.recall("favorite color", min_confidence=0.0, boost=False)}
    assert short_answer in hits



def test_a_short_query_asks_the_encoder_only_about_restatements(table_mem):
    """With fewer than three words a query is too short to compare against
    any candidate, and every one of them was re-embedded for the echo
    verdict: 160 encoder calls a recall at twenty thousand atoms."""
    for n in range(30):
        _add(table_mem, f"redis benchmark note {n} with numbers and a longer story", 0.6 + n / 1000)
    table_mem.embed.table["Redis benchmark"] = QUERY
    calls = []
    embed = table_mem.embed.embed
    table_mem.embed.embed = lambda text: (calls.append(text), embed(text))[1]
    assert len(table_mem.recall("Redis benchmark", top_k=10, boost=False)) == 10
    assert calls == ["Redis benchmark"]

# ── text: words keep their marks, lexical terms fold as the index does ───────

def test_combining_marks_stay_inside_the_word_they_modify():
    """``\\w`` drops combining marks, which cut Hindi and vowelled Arabic
    into single letters that any two sentences share."""
    from smrti.retrieval.text import words

    assert words("मेरी बहन का नाम लौरा है") == ["मेरी", "बहन", "का", "नाम", "लौरा", "है"]
    assert words("أُخْتِي اسْمُهَا لَوْرَا") == ["أُخْتِي", "اسْمُهَا", "لَوْرَا"]


@pytest.mark.parametrize("stored, query", [
    ("Ich wohne in der Hauptstraße zwölf", "Hauptstraße"),
    ("İstanbul'da yaşıyorum", "İstanbul"),
    ("the ﬁnal report", "ﬁnal"),
    ("मेरी बहन का नाम लौरा है", "लौरा"),
])
def test_a_lexical_query_term_finds_the_word_it_names(table_mem, stored, query):
    """Python's casefold is not the index tokenizer's folding: "ß" became
    "ss", "ﬁ" became "fi" and "İ" an "i" plus a combining dot, so the
    lexical half never found the atom holding the very word asked for."""
    from smrti.retrieval.fan_out import _lexical_entry_points

    atom_id = _add(table_mem, stored, 0.1)
    _add(table_mem, "an unrelated note about the weather", 0.1)
    hits = _lexical_entry_points(query, "t", ["s"], table_mem.db, 10, 0.0)
    assert hits == [atom_id]


# ── selection: the judge's order and the belief reserve ──────────────────────

def _ranked(label: str, salience: float, similarity: float, atom_type=AtomType.EPISODE, day: int = 1):
    from smrti.core.models import RecallResult

    atom = Atom(type=atom_type, label=label, content=label,
                truth=TruthValue(probability=0.8, confidence=0.5),
                created_at=f"2026-08-{day:02d} 10:00:00")
    return RecallResult(atom=atom, salience=salience, similarity=similarity)


def test_selection_keeps_the_judges_order_when_candidates_exceed_top_k():
    """Selection re-sorted its survivors by raw salience, so a reranker's
    order survived only when there were no more candidates than slots."""
    from smrti.retrieval.fan_out import select

    ranked = [_ranked(f"distinct memory {n} about topic {n}", 1.0 - n / 10, 0.6, day=n + 1)
              for n in range(8)]
    kept = select("q", ranked, top_k=5, min_confidence=0.0,
                  judge=lambda _q, results: list(reversed(results)))
    assert [r.atom.label for r in kept] == [r.atom.label for r in reversed(ranked)][:5]


def test_a_belief_with_no_bearing_on_the_question_claims_no_reserved_slot():
    """Beliefs at similarity 0.05 took two of five slots from episodes at 0.70."""
    from smrti.retrieval.diversify import diversify

    episodes = [_ranked(f"relevant episode {n} about topic {n}", 0.5 - n / 100, 0.7, day=n + 1)
                for n in range(6)]
    stray = [_ranked(f"unrelated fact {n}", 0.03, 0.05, AtomType.BELIEF) for n in range(2)]
    assert not [r for r in diversify(episodes + stray, top_k=5) if r.atom.type == AtomType.BELIEF]

    terse = _ranked("a terse fact on the topic", 0.08, 0.25, AtomType.BELIEF)
    assert terse in diversify(episodes + [terse], top_k=5)


# ── T15: forgotten atoms neither take attention nor spend the budget ──────────

def test_attend_raises_only_live_atoms_and_reports_what_it_raised(table_mem):
    """attend() boosted forgotten atoms, which kept them from the pruner, and
    reported every id it was handed whether or not a row moved."""
    live = _add(table_mem, "a live memory", 0.5)
    gone = _add(table_mem, "a forgotten memory", 0.5)
    other = _add(table_mem, "a neighbour", 0.5)
    edge = table_mem.atomspace.link_atoms(live, other, "related", "t", "s")
    foreign = _add(table_mem, "a memory in another space", 0.5, space="elsewhere")
    table_mem.forget(atom_ids=[gone])
    before = {r["id"]: r["sti"] for r in table_mem.db.fetchall("SELECT id, sti FROM atoms")}

    assert table_mem.attend([live, gone, edge, foreign, "no-such-id"]) == 1

    after = {r["id"]: r["sti"] for r in table_mem.db.fetchall("SELECT id, sti FROM atoms")}
    assert after[live] > before[live]
    assert all(after[i] == before[i] for i in (gone, edge, foreign))


def test_sunk_neighbours_do_not_spend_the_expansion_budget(table_mem):
    """A hub's hundred sunk, high-attention neighbours took the whole budget
    of 100 edges and were dropped afterwards; the live neighbour holding the
    answer never became a candidate."""
    hub = _add(table_mem, "project atlas hub", 0.9)
    useful = _add(table_mem, "atlas ships on the third of march", 0.10, confidence=0.2)
    table_mem.atomspace.link_atoms(hub, useful, "related", "t", "s")
    for n in range(60):
        _add(table_mem, f"filler note {n}", 0.40 - n / 1000)
    for n in range(100):
        stale = _add(table_mem, f"stale atlas note {n}", 0.05, sti=3.0, lti=0.5, confidence=0.01)
        table_mem.atomspace.link_atoms(hub, stale, "related", "t", "s")
    table_mem.embed.table["launch date?"] = QUERY

    ids = {r.atom.id for r in table_mem.recall("launch date?", top_k=60, boost=False)}
    assert useful in ids


def test_a_locked_database_does_not_fail_a_finished_recall(table_mem, monkeypatch):
    """The access boost is the one write a read makes; a writer holding the
    database past the busy timeout turned a finished recall into an error."""
    import sqlite3

    atom_id = _add(table_mem, "a memory worth reading", 0.8)
    table_mem.embed.table["q"] = QUERY

    def locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(table_mem.db, "execute", locked)
    assert [r.atom.id for r in table_mem.recall("q")] == [atom_id]
    assert table_mem.attend([atom_id]) == 0


# ── T22: recall overhead ──────────────────────────────────────────────────────

def test_a_recall_costs_the_same_statements_however_many_candidates_it_scores(table_mem, monkeypatch):
    """Each recall counted the whole partition, and fetched the vector of
    every candidate the graph reached with a statement of its own."""
    hub = _add(table_mem, "project atlas hub", 0.9)
    for n in range(60):
        neighbour = _add(table_mem, f"atlas detail {n}", 0.2 + n / 1000)
        table_mem.atomspace.link_atoms(hub, neighbour, "related", "t", "s")
    for n in range(60):
        _add(table_mem, f"filler note {n}", 0.5 - n / 1000)
    table_mem.embed.table["atlas?"] = QUERY
    table_mem.recall("atlas?", top_k=60, boost=False)

    statements: list[str] = []
    for name in ("fetchall", "fetchone"):
        original = getattr(table_mem.db, name)

        def counted(sql, params=(), _original=original):
            statements.append(" ".join(sql.split()))
            return _original(sql, params)

        monkeypatch.setattr(table_mem.db, name, counted)
    results = table_mem.recall("atlas?", top_k=60, boost=False)

    assert len(results) == 60
    assert not [s for s in statements if "COUNT(*)" in s or "WHERE rowid = ?" in s]
    assert len(statements) <= 10


def test_the_knn_refill_never_asks_for_more_than_sqlite_vec_returns():
    """A space whose nearest neighbours were all sunk widened its probe by 4x
    up to the space size, and sqlite-vec refuses any k over 4096."""
    from smrti.retrieval.fan_out import _knn_entry_points

    class AllSunk:
        fts_enabled = False

        def __init__(self):
            self.asked: list[int] = []

        def fetchall(self, sql, params=()):
            if "vec_atoms" not in sql:
                return []
            self.asked.append(params[-1])
            return [{"atom_id": f"a{n}", "distance": n / 10_000} for n in range(params[-1])]

    db = AllSunk()
    assert _knn_entry_points(b"", "t", "s", db, 50, 0.1) == []
    assert max(db.asked) == 4096

# ── T13: stated warnings are found apart from the candidate pool ─────────────

def test_find_warnings_reaches_a_stated_warning_the_candidate_pool_misses(table_mem):
    """Seventy memories nearer the query filled the KNN pool, so a stated
    warning never became a candidate and no top_k could deliver it."""
    from smrti.retrieval.fan_out import find_warnings

    stated = {"valence_stated": True}
    warning = _add(table_mem, "never erase recovery snapshots", 0.7,
                   valence=-1.0, intensity=1.0, metadata=stated)
    for n in range(70):
        _add(table_mem, f"filler note {n}", 0.8)
    _add(table_mem, "never post the api key in chat", 0.1,
         valence=-0.9, intensity=0.9, metadata=stated)
    _add(table_mem, "the deploy failed badly last night", 0.75, valence=-0.9, intensity=0.9)
    forgotten = _add(table_mem, "never skip the nightly backup", 0.72,
                     valence=-0.9, intensity=0.9, metadata=stated)
    table_mem.forget(atom_ids=[forgotten])
    table_mem.embed.table["disaster policy"] = QUERY

    found = find_warnings("disaster policy", "t", ["s"], table_mem.db, table_mem.embed)
    assert [r.atom.id for r in found] == [warning]
    assert found[0].similarity == pytest.approx(0.7, abs=1e-3)

