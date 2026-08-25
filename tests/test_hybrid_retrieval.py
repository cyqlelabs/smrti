"""Tests for the lexical half of recall and its fusion with vector search.

The failure being covered: a query and the atom that answers it share their
proper nouns byte for byte, and the embedding still ranks the atom out of
reach — most visibly when the two are written in different languages. BM25
finds shared words in any language; RRF decides which candidates salience
gets to see.
"""
from __future__ import annotations

import sqlite3

import pytest

from smrti import Smrti
from smrti.core.db import close_database, fts_rowid, get_database
from smrti.retrieval.fan_out import (
    _fts_query,
    _lexical_entry_points,
    _rrf_fuse,
    _term_list,
    retrieve,
)


@pytest.fixture
def mem(tmp_path):
    return Smrti(
        db_path=str(tmp_path / "hybrid.db"),
        personality="balanced",
        tenant_id="test",
        write_space="default",
    )


def _fts_rows(mem) -> list[sqlite3.Row]:
    return mem.db.fetchall("SELECT atom_id, label, content FROM atoms_fts")


def _lexical(mem, query: str, limit: int = 50) -> list[str]:
    return _lexical_entry_points(query, mem.tenant_id, [mem.write_space], mem.db, limit)


# ── index maintenance ────────────────────────────────────────────────────────


def test_fts5_is_available_in_this_build(mem):
    assert mem.db.fts_enabled is True


def test_remembering_indexes_the_text_for_lexical_search(mem):
    atom_id = mem.remember("Roxana lives in San Benito")

    rows = _fts_rows(mem)
    assert [r["atom_id"] for r in rows] == [atom_id]
    assert rows[0]["content"] == "Roxana lives in San Benito"


def test_the_lexical_row_is_keyed_by_a_rowid_derived_from_the_atom_id(mem):
    atom_id = mem.remember("Roxana lives in San Benito")

    row = mem.db.fetchone(
        "SELECT atom_id FROM atoms_fts WHERE rowid = ?", (fts_rowid(atom_id),)
    )
    assert row["atom_id"] == atom_id


def test_relation_atoms_stay_out_of_the_lexical_index(mem):
    a = mem.remember("Roxana lives in San Benito")
    b = mem.remember("Esmeralda goes to school")
    mem.atomspace.link_atoms(a, b, "mentions", mem.tenant_id, mem.write_space)

    assert sorted(r["atom_id"] for r in _fts_rows(mem)) == sorted([a, b])


def test_rewriting_an_atom_replaces_its_lexical_row(mem):
    from smrti.core.models import Atom, AtomType

    atom = Atom(
        type=AtomType.BELIEF,
        label="first",
        content="the deploy pipeline is green",
        tenant_id="test",
        space="default",
    )
    mem.atomspace.add_atom(atom)
    atom.content = "the deploy pipeline is red"
    mem.atomspace.add_atom(atom)

    rows = _fts_rows(mem)
    assert len(rows) == 1
    assert rows[0]["content"] == "the deploy pipeline is red"
    assert _lexical(mem, "red") == [atom.id]
    assert _lexical(mem, "green") == []


def test_updating_an_atoms_text_updates_the_lexical_index(mem):
    atom_id = mem.remember("Roxana lives in San Benito")
    atom = mem.atomspace.get_atom(atom_id, mem.tenant_id, mem.write_space)
    atom.label = "Roxana moved"
    atom.content = "Roxana lives in Montevideo"
    mem.atomspace.update_atom(atom)

    assert _lexical(mem, "Montevideo") == [atom_id]
    assert _lexical(mem, "Benito") == []


def test_clearing_a_space_empties_its_lexical_rows(mem):
    mem.remember("Roxana lives in San Benito")
    mem.remember("Esmeralda goes to school")

    mem.clear_space()

    assert _fts_rows(mem) == []


def test_pruned_atoms_leave_the_lexical_index(mem):
    atom_id = mem.remember("a passing thought about nothing")
    # Sink it below both prune floors and mark it agent-authored, which is what
    # makes an episode eligible for pruning at all.
    mem.db.execute(
        "UPDATE atoms SET confidence = 0.0, lti = 0.0, metadata = '{\"source\": \"agent\"}' "
        "WHERE id = ?",
        (atom_id,),
    )

    result = mem.reflect()

    assert result.atoms_pruned == 1
    assert _fts_rows(mem) == []


def test_entities_created_by_the_resolver_are_indexed(mem, tmp_path):
    from smrti.extraction.resolve import EntityResolver

    resolver = EntityResolver(mem.db, mem.embed)
    atom_id = resolver.resolve(
        "Esmeralda", "person", mem.tenant_id, mem.write_space, [mem.write_space]
    )

    assert _lexical(mem, "Esmeralda") == [atom_id]


def test_a_graph_written_before_the_index_existed_is_backfilled(tmp_path):
    db_path = str(tmp_path / "backfill.db")
    mem = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    atom_id = mem.remember("Roxana lives in San Benito")

    # Simulate a database whose atoms were all written by a build with no
    # lexical index: the table is there, the rows are not.
    mem.db.execute("DELETE FROM atoms_fts", ())
    close_database(db_path)

    db = get_database(db_path)
    try:
        rows = db.fetchall("SELECT atom_id FROM atoms_fts")
        assert [r["atom_id"] for r in rows] == [atom_id]
    finally:
        close_database(db_path)


def test_backfill_leaves_an_already_indexed_graph_alone(tmp_path):
    db_path = str(tmp_path / "noop.db")
    mem = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    mem.remember("Roxana lives in San Benito")
    before = [dict(r) for r in mem.db.fetchall("SELECT rowid, atom_id FROM atoms_fts")]
    close_database(db_path)

    db = get_database(db_path)
    try:
        after = [dict(r) for r in db.fetchall("SELECT rowid, atom_id FROM atoms_fts")]
        assert after == before
    finally:
        close_database(db_path)


# ── query construction ───────────────────────────────────────────────────────


def test_terms_are_deduplicated_in_order_of_first_appearance():
    assert _term_list("San Benito, san benito, Roxana") == ["san", "benito", "roxana"]


def test_term_list_is_capped():
    assert len(_term_list(" ".join(str(n) for n in range(200)))) == 32


def test_fts_operators_in_a_query_are_searched_for_not_obeyed(mem):
    literal = mem.remember("the NEAR miss report")
    mem.remember("an unrelated note")

    # Unquoted, "NEAR" would be FTS5 syntax and the query a syntax error.
    assert _lexical(mem, "NEAR") == [literal]
    assert _fts_query(["near", 'say "hi"']) == '"near" OR "say ""hi"""'


def test_a_query_with_no_searchable_terms_yields_no_lexical_candidates(mem):
    mem.remember("Roxana lives in San Benito")
    assert _lexical(mem, "!!! ???") == []


def test_lexical_search_is_skipped_when_the_build_has_no_fts5(mem, monkeypatch):
    mem.remember("Roxana lives in San Benito")
    monkeypatch.setattr(mem.db, "fts_enabled", False)

    assert _lexical(mem, "Roxana") == []


def test_writes_still_work_when_the_build_has_no_fts5(mem, monkeypatch):
    """The index table does not exist there — a write naming it would fail."""
    monkeypatch.setattr(mem.db, "fts_enabled", False)

    atom_id = mem.remember("Roxana lives in San Benito")
    mem.clear_space()

    assert atom_id
    assert mem.status()["total_atoms"] == 0


def test_deleting_no_atoms_writes_no_statement(mem):
    from smrti.core.db import fts_delete

    assert fts_delete(mem.db, []) == []


# ── cross-language and proper-noun recall ────────────────────────────────────


def test_a_spanish_query_reaches_an_english_atom_through_its_proper_nouns(mem):
    atom_id = mem.remember(
        "Nicolas lives in San Benito with Roxana and their daughter Esmeralda"
    )
    mem.remember("the weather has been mild all week")

    assert _lexical(mem, "decime qué sabes sobre Esmeralda") == [atom_id]


def test_an_accented_name_is_reachable_without_its_accents(mem):
    atom_id = mem.remember("Nicolás vive en San Benito")

    assert _lexical(mem, "Nicolas") == [atom_id]


def test_the_lexical_half_is_scoped_to_the_tenant(tmp_path):
    shared = str(tmp_path / "tenants.db")
    mine = Smrti(db_path=shared, tenant_id="mine", write_space="default")
    theirs = Smrti(db_path=shared, tenant_id="theirs", write_space="default")
    theirs.remember("Esmeralda is their daughter")

    assert _lexical(mine, "Esmeralda") == []


def test_the_lexical_half_is_scoped_to_the_read_spaces(tmp_path):
    shared = str(tmp_path / "spaces.db")
    work = Smrti(db_path=shared, tenant_id="test", write_space="work")
    home = Smrti(db_path=shared, tenant_id="test", write_space="home")
    home.remember("Esmeralda is my daughter")

    assert _lexical(work, "Esmeralda") == []


def test_recall_surfaces_the_atom_the_dense_pool_ranked_out(mem, monkeypatch):
    """The whole point of fusion, on one graph with the lexical half toggled.

    The vector pool is squeezed to four entries and filled with English ways
    of asking the Spanish question — the embedding puts every one of them
    nearer the query than the English fact that actually answers it. None of
    them shares a word with the query; the fact shares the daughter's name.
    """
    import smrti.retrieval.fan_out as fan_out

    monkeypatch.setattr(fan_out, "_KNN_POOL_MIN", 4)
    monkeypatch.setattr(fan_out, "_KNN_POOL_MAX", 4)

    target = mem.remember(
        "Nicolas lives in San Benito with Roxana and their daughter Esmeralda"
    )
    for filler in (
        "what do you know about my family",
        "share everything you know about my family",
        "tell me what you know about my relatives",
        "tell me about my relatives please",
    ):
        mem.remember(filler)

    query = "qué sabés sobre Esmeralda y mi familia"

    def _recall():
        return [
            r.atom.id
            for r in retrieve(
                query, mem.tenant_id, [mem.write_space], mem.db, mem.embed,
                write_space=mem.write_space, top_k=10, min_confidence=0.0,
            )
        ]

    monkeypatch.setattr(mem.db, "fts_enabled", False)
    assert target not in _recall()

    monkeypatch.setattr(mem.db, "fts_enabled", True)
    assert target in _recall()


def test_recall_still_works_when_nothing_matches_lexically(mem):
    mem.remember("Roxana lives in San Benito")

    results = mem.recall("dónde vive la familia", min_confidence=0.0)

    assert results


# ── reciprocal rank fusion ───────────────────────────────────────────────────


def test_an_atom_both_halves_like_outranks_one_only_a_single_half_tops():
    fused = _rrf_fuse([["a", "shared"], ["b", "shared"]], limit=3)
    assert fused[0] == "shared"


def test_fusion_keeps_candidates_only_one_half_found():
    assert set(_rrf_fuse([["a"], ["b"]], limit=5)) == {"a", "b"}


def test_fusion_respects_the_pool_limit():
    assert len(_rrf_fuse([["a", "b", "c"], ["d", "e"]], limit=2)) == 2


def test_fusion_breaks_ties_deterministically():
    first = _rrf_fuse([["z", "y", "x"], ["x", "y", "z"]], limit=3)
    second = _rrf_fuse([["z", "y", "x"], ["x", "y", "z"]], limit=3)
    assert first == second


def test_fusion_of_nothing_is_nothing():
    assert _rrf_fuse([[], []], limit=10) == []
