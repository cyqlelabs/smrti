"""Coverage tests for engine internals reached only by uncommon inputs.

Grouped by module: the embedding provider's lazy load, cross-domain
connection discovery when a vector row is missing, and the set-operation
edge cases (empty spaces, vectorless atoms, degenerate similarity).
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from smrti import Smrti
from smrti.core.embed import EmbeddingProvider, get_embedding_provider
from smrti.core.models import EntityType, _clamp, _safe_entity_type


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def mem(db_path):
    engine = Smrti(db_path=db_path, tenant_id="test", write_space="default")
    yield engine
    engine.close()


# ── core/embed.py ─────────────────────────────────────────────────────────────

class _StubModel:
    def __init__(self, model_name=None, **kwargs):
        self.model_name = model_name
        self.seen: list[list[str]] = []

    def embed(self, texts, **kwargs):
        self.seen.append(list(texts))
        for i, _ in enumerate(texts):
            yield _StubVector([float(i)] * 3)


class _StubVector(list):
    def tolist(self):
        return list(self)


def test_provider_loads_the_model_lazily_and_once():
    provider = EmbeddingProvider(model_name="stub/model")
    assert provider._model is None
    with patch("fastembed.TextEmbedding", _StubModel) as _:
        model = provider._get_model()
        assert model.model_name == "stub/model"
        assert provider._get_model() is model  # cached, not reloaded


def test_provider_embeds_a_single_text():
    provider = EmbeddingProvider()
    with patch("fastembed.TextEmbedding", _StubModel):
        assert provider.embed("hello") == [0.0, 0.0, 0.0]
        assert provider._get_model().seen[-1] == ["hello"]


def test_provider_embeds_a_batch():
    provider = EmbeddingProvider()
    with patch("fastembed.TextEmbedding", _StubModel):
        vectors = provider.embed_batch(["a", "b"])
    assert vectors == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_provider_reports_its_dimensions():
    assert EmbeddingProvider().dimensions == 384


def test_embedding_provider_is_a_singleton():
    assert get_embedding_provider() is get_embedding_provider()


# ── core/models.py ────────────────────────────────────────────────────────────

def test_unknown_entity_type_degrades_to_concept():
    assert _safe_entity_type("not-a-real-type") is EntityType.CONCEPT


def test_entity_type_of_none_stays_none():
    assert _safe_entity_type(None) is None
    assert _safe_entity_type("") is None


def test_clamp_substitutes_the_default_for_none():
    assert _clamp(None, 0.0, 1.0, default=0.5) == 0.5
    assert _clamp(None, 0.0, 1.0) == 0.0
    assert _clamp(3.0, 0.0, 1.0) == 1.0


# ── evolution/connections.py ──────────────────────────────────────────────────

def _promote(mem, atom_id, lti=0.6):
    mem.db.execute("UPDATE atoms SET lti = ? WHERE id = ?", (lti, atom_id))


def test_discover_connections_links_similar_high_lti_atoms(mem):
    from smrti.evolution.connections import discover_connections

    a = mem.remember("Kubernetes rollout strategy for the API tier", type="concept")
    b = mem.remember("Kubernetes rollout strategy for the worker tier", type="concept")
    _promote(mem, a)
    _promote(mem, b)

    created = discover_connections(mem.tenant_id, mem.write_space, mem.db, mem.embed)
    assert created >= 1
    rows = mem.db.fetchall(
        "SELECT source_id, target_id FROM atoms WHERE type = 'relation' AND relation = 'associated'"
    )
    assert {(r["source_id"], r["target_id"]) for r in rows} & {(a, b), (b, a)}


def test_discover_connections_re_embeds_an_atom_with_no_vector_row(mem):
    from smrti.evolution.connections import discover_connections

    a = mem.remember("Postgres vacuum tuning for the events table", type="concept")
    b = mem.remember("Postgres vacuum tuning for the sessions table", type="concept")
    _promote(mem, a)
    _promote(mem, b)
    mem.db.execute("DELETE FROM vec_atoms WHERE atom_id = ?", (a,))

    embed_spy = MagicMock(side_effect=mem.embed.embed)
    engine = MagicMock()
    engine.embed = embed_spy
    created = discover_connections(mem.tenant_id, mem.write_space, mem.db, engine)

    embed_spy.assert_called_once()  # only the vectorless atom is re-embedded
    assert created >= 1


def test_discover_connections_skips_already_linked_atoms(mem):
    from smrti.evolution.connections import discover_connections

    a = mem.remember("Redis eviction policy for the cache tier", type="concept")
    b = mem.remember("Redis eviction policy for the queue tier", type="concept")
    _promote(mem, a)
    _promote(mem, b)
    mem.atomspace.link_atoms(a, b, "relates_to", mem.tenant_id, mem.write_space)

    assert discover_connections(mem.tenant_id, mem.write_space, mem.db, mem.embed) == 0


def test_discover_connections_ignores_low_lti_atoms(mem):
    from smrti.evolution.connections import discover_connections

    mem.remember("Terraform state locking in the shared bucket", type="concept")
    mem.remember("Terraform state locking in the staging bucket", type="concept")
    mem.db.execute("UPDATE atoms SET lti = 0.1")
    assert discover_connections(mem.tenant_id, mem.write_space, mem.db, mem.embed) == 0


# ── spaces/set_ops.py ─────────────────────────────────────────────────────────

def test_get_embedding_returns_none_without_a_vector_row(mem):
    from smrti.spaces.set_ops import _get_embedding

    atom_id = mem.remember("An atom that loses its vector", type="concept")
    mem.db.execute("DELETE FROM vec_atoms WHERE atom_id = ?", (atom_id,))
    assert _get_embedding(atom_id, mem.db) is None


def test_get_embedding_unpacks_the_stored_blob(mem):
    from smrti.spaces.set_ops import _get_embedding

    atom_id = mem.remember("An atom that keeps its vector", type="concept")
    vector = _get_embedding(atom_id, mem.db)
    assert vector is not None and len(vector) == 384


def test_cosine_similarity_of_a_zero_vector_is_zero():
    from smrti.spaces.set_ops import _cosine_similarity

    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_neighbor_context_embedding_is_memoized(mem):
    from smrti.spaces.set_ops import _neighbor_context_embedding
    from smrti.core.models import atom_from_row

    a = mem.remember("Grafana dashboards", type="concept")
    b = mem.remember("Prometheus scrape config", type="concept")
    mem.atomspace.link_atoms(a, b, "relates_to", mem.tenant_id, mem.write_space)
    atom = atom_from_row(mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (a,)))

    embed_spy = MagicMock(side_effect=mem.embed.embed)
    engine = MagicMock()
    engine.embed = embed_spy
    cache: dict = {}
    first = _neighbor_context_embedding(atom, mem.db, engine, cache)
    second = _neighbor_context_embedding(atom, mem.db, engine, cache)
    assert first is second
    embed_spy.assert_called_once()


def test_contextual_similarity_redistributes_the_neighborhood_weight(mem):
    """With no neighbors to compare, the embedding signal absorbs that weight."""
    from smrti.core.models import atom_from_row
    from smrti.spaces.set_ops import (
        W_EMBEDDING, W_ENTITY_TYPE, W_NEIGHBORHOOD, _contextual_similarity,
    )

    a = mem.remember("Isolated concept about sailing", type="concept")
    b = mem.remember("Isolated concept about sailing boats", type="concept")
    atom_a = atom_from_row(mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (a,)))
    atom_b = atom_from_row(mem.db.fetchone("SELECT * FROM atoms WHERE id = ?", (b,)))

    with_engine = _contextual_similarity(atom_a, atom_b, 0.9, mem.db, mem.embed, {})
    without_engine = _contextual_similarity(atom_a, atom_b, 0.9, mem.db, None)
    # Neither atom has neighbors, so both paths fall back to the same weighting.
    assert with_engine == pytest.approx(without_engine)
    # Absent signals fold into the embedding weight: 0.9 x the full weight budget.
    assert with_engine == pytest.approx(0.9 * (W_EMBEDDING + W_NEIGHBORHOOD + W_ENTITY_TYPE))


def test_overlap_matches_each_atom_at_most_once(db_path, mem):
    """Greedy assignment: a single B atom cannot claim two A atoms."""
    from smrti.spaces.set_ops import space_overlap

    mem.remember("Continuous integration pipeline", type="concept")
    mem.remember("Continuous integration pipelines", type="concept")
    other = Smrti(db_path=db_path, tenant_id="test", write_space="other")
    other.remember("Continuous integration pipeline", type="concept")
    try:
        overlap = space_overlap(
            "test", "default", "other", mem.db, threshold=0.5, embed_engine=mem.embed,
        )
    finally:
        other.close()
    assert len(overlap.pairs) == 1
    assert len({p.atom_b.id for p in overlap.pairs}) == 1


def test_union_of_an_empty_space_returns_the_other_side(db_path, mem):
    from smrti.spaces.set_ops import space_union

    mem.remember("Only atom in this space", type="concept")
    empty_first = space_union("test", "empty", "default", mem.db, embed_engine=mem.embed)
    assert empty_first.operation == "union"
    assert [a.space for a in empty_first.atoms] == ["default"]

    empty_second = space_union("test", "default", "empty", mem.db, embed_engine=mem.embed)
    assert [a.space for a in empty_second.atoms] == ["default"]


def test_symmetric_difference_handles_empty_spaces(db_path, mem):
    from smrti.spaces.set_ops import space_symmetric_difference

    both_empty = space_symmetric_difference("test", "empty-a", "empty-b", mem.db)
    assert both_empty.operation == "symmetric_difference"
    assert both_empty.atoms == []

    mem.remember("Only atom in this space", type="concept")
    left_empty = space_symmetric_difference(
        "test", "empty-a", "default", mem.db, embed_engine=mem.embed
    )
    assert [a.space for a in left_empty.atoms] == ["default"]

    right_empty = space_symmetric_difference(
        "test", "default", "empty-a", mem.db, embed_engine=mem.embed
    )
    assert [a.space for a in right_empty.atoms] == ["default"]
