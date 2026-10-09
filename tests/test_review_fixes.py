"""Regressions for an external review of the engine's safety claims.

Each test pins one finding: forget() reaching past what was asked about, a
proxied conversation having no way to record a warning, user-stated goals
decaying out of the graph, low-probability episodes read as antipatterns,
and Japanese and Chinese text being one token per sentence. They run on a
deterministic bag-of-words embedder (over the engine's own tokenizer, so
CJK bigrams are shared words too): they measure the engine's bookkeeping,
not what two sentences mean.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.requests import Request as StarletteRequest

import smrti.core.embed as embed_module
from smrti import Smrti
from smrti.core.db import close_database
from smrti.core.provenance import VALENCE_STATED
from smrti.retrieval.classify import classify_memory
from smrti.retrieval.text import coverage, lexical_text, word_set, words

_token_vectors: dict[str, np.ndarray] = {}


def _token_vector(token: str) -> np.ndarray:
    vec = _token_vectors.get(token)
    if vec is None:
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        vec = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        _token_vectors[token] = vec
    return vec


class _BagOfWords:
    """Shared words, shared direction, plus the small common component real
    sentence embeddings give every pair of texts."""

    def embed(self, texts):
        for text in texts:
            tokens = words(text)
            acc = 0.6 * (len(tokens) ** 0.5) * _token_vector("<common>")
            for token in tokens:
                acc += _token_vector(token)
            norm = np.linalg.norm(acc)
            yield acc / norm if norm else acc + 1e-3


@pytest.fixture(autouse=True)
def bag_of_words_embedder(monkeypatch):
    monkeypatch.setattr(embed_module.EmbeddingProvider, "_get_model", lambda self: _BagOfWords())


@pytest.fixture
def mem(tmp_path):
    return Smrti(db_path=str(tmp_path / "review.db"), tenant_id="t", write_space="s")


def _row(mem, atom_id):
    return mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (atom_id,))


def _forgotten(mem, atom_id) -> bool:
    row = _row(mem, atom_id)
    return row is not None and json.loads(row["metadata"] or "{}").get("forgotten") is True


# ── forget() forgets what was asked about, and nothing else ──────────────────


def test_forget_leaves_unrelated_memories_alone(mem):
    target = mem.remember("the deploy pipeline uses Jenkins", valence=0.0)
    others = [
        mem.remember("my sister lives in Lisbon", valence=0.0),
        mem.remember("the user prefers green tea in the morning", valence=0.0),
        mem.remember("the database password rotates monthly", valence=0.0),
        mem.remember("our standup moved to ten thirty", valence=0.0),
    ]

    forgotten = mem.forget("deploy pipeline Jenkins")

    assert forgotten == ["the deploy pipeline uses Jenkins"]
    assert _forgotten(mem, target)
    assert not any(_forgotten(mem, other) for other in others)
    assert len(mem.recall("sister Lisbon", boost=False)) >= 1


def test_forget_takes_only_matches_close_to_the_best_one(mem):
    exact = mem.remember("the deploy pipeline uses Jenkins on the build server", valence=0.0)
    loose = mem.remember("the server room is cold", valence=0.0)

    mem.forget("deploy pipeline Jenkins build server")

    assert _forgotten(mem, exact)
    assert not _forgotten(mem, loose)


def test_forget_finds_a_memory_quoted_back_verbatim(mem):
    """Recall zeroes an episode that restates the query — the question is not
    the answer — but quoting a memory is how one names the one to forget."""
    atom_id = mem.remember("I told my manager I would quit in March", valence=0.0)

    assert mem.forget("I told my manager I would quit in March")
    assert _forgotten(mem, atom_id)


def test_forget_by_id_forgets_exactly_those(mem, tmp_path):
    keep = mem.remember("the deploy pipeline builds the docker image", valence=0.0)
    drop = mem.remember("the deploy pipeline runs the tests", valence=0.0)
    elsewhere = Smrti(
        db_path=str(tmp_path / "review.db"), tenant_id="t", write_space="other"
    ).remember("the deploy pipeline is elsewhere", valence=0.0)

    forgotten = mem.forget(atom_ids=[drop, elsewhere, "no-such-id"])

    assert forgotten == ["the deploy pipeline runs the tests"]
    assert _forgotten(mem, drop)
    assert not _forgotten(mem, keep)
    assert not _forgotten(mem, elsewhere)  # another space is never touched


def test_forget_needs_something_to_forget(mem):
    with pytest.raises(ValueError):
        mem.forget("   ")
    assert mem.forget(atom_ids=[]) == []


def test_the_forget_tool_takes_ids(mem):
    from smrti.servers.mcp import handle_tool

    drop = mem.remember("the launch codes are in the red binder", valence=0.0)
    keep = mem.remember("the red binder also holds the menu", valence=0.0)

    result = handle_tool(mem, "smrti_forget", {"atom_ids": [drop]})

    assert result["softened"] == ["the launch codes are in the red binder"]
    assert _forgotten(mem, drop) and not _forgotten(mem, keep)
    assert "error" in handle_tool(mem, "smrti_forget", {})


# ── a user-stated goal is testimony ──────────────────────────────────────────


def test_a_user_stated_goal_outlives_inactivity(mem):
    goal = mem.remember("learn to read Japanese", type="goal", valence=0.0)
    agent_goal = mem.remember(
        "suggest a Rust course", type="goal", valence=0.0, metadata={"source": "agent"}
    )
    floor = mem._surfacing_floor()

    for _ in range(150):
        mem.reflect()

    row = _row(mem, goal)
    assert row is not None, "an untouched user goal was pruned"
    assert row["confidence"] >= floor
    assert any(r.atom.id == goal for r in mem.recall("Japanese", boost=False))
    assert _row(mem, agent_goal) is None


def test_a_forgotten_goal_can_still_be_pruned(mem):
    goal = mem.remember("learn to read Japanese", type="goal", valence=0.0)
    mem.forget(atom_ids=[goal])

    mem.reflect()

    assert _row(mem, goal) is None


# ── only a proposition can be an antipattern ─────────────────────────────────


def test_an_old_low_probability_episode_is_context(mem):
    atom_id = mem.remember("we tried the new office coffee", probability=0.1, valence=0.0)
    hit = next(r for r in mem.recall("office coffee", min_confidence=0.0, boost=False)
               if r.atom.id == atom_id)
    assert classify_memory(hit) == "context"


# ── the proxy can record a warning a client states ───────────────────────────


def _request(headers: dict[str, str]) -> StarletteRequest:
    return StarletteRequest({
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "query_string": b"",
    })


def test_the_proxy_reads_a_stated_tone():
    from smrti.servers.proxy import _stated_tone

    assert _stated_tone(_request({})) is None
    assert _stated_tone(_request({"X-Smrti-Valence": "-0.9"})) == (-0.9, None)
    assert _stated_tone(
        _request({"X-Smrti-Valence": "-0.9", "X-Smrti-Intensity": "0.8"})
    ) == (-0.9, 0.8)
    for bad in (
        {"X-Smrti-Valence": "very bad"},
        {"X-Smrti-Valence": "-3"},
        {"X-Smrti-Valence": "-0.9", "X-Smrti-Intensity": "2"},
        {"X-Smrti-Intensity": "0.9"},
    ):
        with pytest.raises(HTTPException) as err:
            _stated_tone(_request(bad))
        assert err.value.status_code == 400


def test_a_stated_tone_makes_the_user_turn_a_critical_warning(mem):
    from smrti.servers import proxy

    messages = [{"role": "user", "content": "never run the migration without a backup"}]
    with patch.object(proxy, "get_mem", lambda tenant_id, write_space: mem), \
         patch("smrti.servers.config.EXTRACT", False):
        asyncio.run(proxy._store_exchange(
            messages, "Understood, I will take a backup first.", "t", "s",
            tone=(-0.9, 0.9),
        ))

    rows = {r["content"]: r for r in mem.db.fetchall("SELECT * FROM atoms WHERE type = 'episode'")}
    user = rows["never run the migration without a backup"]
    reply = rows["Understood, I will take a backup first."]
    assert json.loads(user["metadata"]).get(VALENCE_STATED) is True
    assert not json.loads(reply["metadata"] or "{}").get(VALENCE_STATED)

    hit = next(r for r in mem.recall("run the migration", boost=False)
               if r.atom.content == "never run the migration without a backup")
    assert classify_memory(hit) == "critical_warning"


# ── Chinese and Japanese are segmented ───────────────────────────────────────


def test_unspaced_scripts_are_cut_into_bigrams():
    assert words("東京に住んでいます") == ["東京", "京に", "に住", "住ん", "んで", "でい", "いま", "ます"]
    assert words("iPhone15を買った") == ["iphone15", "を買", "買っ", "った"]
    assert words("Café déjà vu") == ["café", "déjà", "vu"]  # spaced scripts are untouched
    same = word_set("私は東京に住んでいます")
    assert coverage(same, same) == 1.0  # a restatement is now comparable
    assert lexical_text("plain text") == "plain text"


def test_a_japanese_memory_is_found_by_a_word_inside_it(mem):
    if not mem.db.fts_enabled:
        pytest.skip("this SQLite build has no FTS5")
    atom_id = mem.remember("私は東京に住んでいます", valence=0.0)
    mem.remember("the weather in Lisbon is mild", valence=0.0)

    from smrti.retrieval.fan_out import _fts_query, _term_list

    hits = mem.db.fetchall(
        "SELECT atom_id FROM atoms_fts WHERE atoms_fts MATCH ?",
        (_fts_query(_term_list("東京")),),
    )
    assert [h["atom_id"] for h in hits] == [atom_id]


def test_an_index_written_before_segmentation_is_rebuilt(tmp_path):
    path = str(tmp_path / "old.db")
    mem = Smrti(db_path=path, tenant_id="t", write_space="s")
    if not mem.db.fts_enabled:
        pytest.skip("this SQLite build has no FTS5")
    atom_id = mem.remember("私は東京に住んでいます", valence=0.0)
    # What an older build left behind: the sentence as one token, no marker.
    mem.db.execute("UPDATE atoms_fts SET content = ? WHERE atom_id = ?",
                   ("私は東京に住んでいます", atom_id))
    mem.db.execute("DELETE FROM smrti_meta WHERE key = 'fts_format'")
    mem.close()
    close_database(path)

    reopened = Smrti(db_path=path, tenant_id="t", write_space="s")
    hits = reopened.db.fetchall(
        "SELECT atom_id FROM atoms_fts WHERE atoms_fts MATCH ?", ('"東京"',)
    )
    assert [h["atom_id"] for h in hits] == [atom_id]
