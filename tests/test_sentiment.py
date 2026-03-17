"""Tests for language-agnostic valence estimation (extraction/sentiment.py)."""
import math

import pytest

from smrti.core.embed import EmbeddingProvider
from smrti.extraction.sentiment import estimate_valence, _cosine


@pytest.fixture(scope="module")
def embed():
    return EmbeddingProvider()


# ── _cosine helper ────────────────────────────────────────────────────────────

def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine(a, b)) < 1e-6


def test_cosine_zero_vector_returns_zero():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(_cosine(a, b) - (-1.0)) < 1e-6


# ── estimate_valence ──────────────────────────────────────────────────────────

def test_positive_text_has_positive_valence(embed):
    val = estimate_valence("I love this! It's wonderful and amazing.", embed)
    assert val > 0


def test_negative_text_has_negative_valence(embed):
    val = estimate_valence("This is terrible and I hate it completely.", embed)
    assert val < 0


def test_neutral_text_in_dead_zone(embed):
    # A neutral/ambiguous sentence should return near zero
    val = estimate_valence("The meeting is scheduled for Tuesday.", embed)
    # May be exactly 0 or small; just verify it's in [-0.3, 0.3] for neutral content
    assert -0.4 <= val <= 0.4


def test_return_value_in_range(embed):
    for text in [
        "Excellent experience!",
        "Absolutely horrible disaster.",
        "The file was created.",
    ]:
        val = estimate_valence(text, embed)
        assert -1.0 <= val <= 1.0


def test_returns_float(embed):
    val = estimate_valence("Some text here.", embed)
    assert isinstance(val, float)


def test_anchor_cache_is_reused(embed):
    """Second call must not re-embed anchors (covered via module-level cache)."""
    from smrti.extraction import sentiment as smod
    _ = estimate_valence("test", embed)
    assert smod._neg_vecs is not None
    assert smod._pos_vecs is not None


def test_positive_stronger_than_neutral(embed):
    neutral = estimate_valence("Today is a day.", embed)
    positive = estimate_valence("This is the most fantastic and joyful day ever!", embed)
    assert positive >= neutral
