"""Language-agnostic valence estimation using embedding similarity."""
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smrti.core.embed import EmbeddingProvider

_NEGATIVE_ANCHORS = [
    "error failure crash bug vulnerability exploit dangerous",
    "mistake wrong broken never avoid warning incident",
]
_POSITIVE_ANCHORS = [
    "success correct reliable stable proven works great",
    "prefer recommend good safe secure love best",
]

_lock = threading.Lock()
_neg_vecs: list[list[float]] | None = None
_pos_vecs: list[list[float]] | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def _ensure_anchors(embed: EmbeddingProvider) -> tuple[list[list[float]], list[list[float]]]:
    global _neg_vecs, _pos_vecs
    if _neg_vecs is None:
        with _lock:
            if _neg_vecs is None:
                _neg_vecs = embed.embed_batch(_NEGATIVE_ANCHORS)
                _pos_vecs = embed.embed_batch(_POSITIVE_ANCHORS)
    return _neg_vecs, _pos_vecs


def estimate_valence(text: str, embed: EmbeddingProvider) -> float:
    """Return a valence estimate in [-1.0, 1.0] using embedding similarity.

    Compares the text embedding against positive and negative anchor
    phrases.  Language-agnostic — works for any language the embedding
    model can encode.  Returns 0.0 when positive and negative signals
    are balanced.
    """
    neg_vecs, pos_vecs = _ensure_anchors(embed)
    vec = embed.embed(text)
    neg_sim = max(_cosine(vec, nv) for nv in neg_vecs)
    pos_sim = max(_cosine(vec, pv) for pv in pos_vecs)
    diff = pos_sim - neg_sim
    # Scale so a clear signal (e.g. 0.15 diff) maps to ~±0.7
    scaled = max(-1.0, min(1.0, diff * 5.0))
    # Dead-zone: if both similarities are low or very close, return 0
    if abs(diff) < 0.03:
        return 0.0
    return round(scaled, 3)
