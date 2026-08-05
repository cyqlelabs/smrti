"""Language-agnostic valence estimation using embedding similarity."""
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smrti.core.embed import EmbeddingProvider

# Anchor sentences span English, Spanish, German, French, Chinese, and
# Japanese so the polarity signal is not biased toward English inputs.
# Counts are kept balanced per language across polarities for symmetric
# mean-pooling.
_NEGATIVE_ANCHORS = [
    "This is a terrible, awful, and completely negative experience.",
    "I hate this, it is horrible, bad, and very frustrating.",
    "Never do this, it is extremely dangerous and must be avoided at all costs.",
    "This is a critical mistake that caused serious damage and must not happen again.",
    "Esto es horrible, lo odio y es una experiencia terrible.",
    "Nunca hagas esto, es muy peligroso y debe evitarse a toda costa.",
    "Das ist schrecklich, ich hasse es, eine furchtbare Erfahrung.",
    "Das war ein schwerer Fehler und darf nie wieder passieren.",
    "C'est horrible, je déteste ça, une expérience terrible.",
    "Ne faites jamais cela, c'est très dangereux et à éviter absolument.",
    "这太糟糕了，我讨厌它，是非常可怕的经历。",
    "千万不要这样做，这非常危险，必须避免。",
    "これはひどい、大嫌いで、最悪の経験です。",
    "絶対にしないでください、とても危険で避けるべきです。",
]
_POSITIVE_ANCHORS = [
    "This is an excellent, wonderful, and completely positive experience.",
    "I love this, it is great, fantastic, and very enjoyable.",
    "This is safe, reliable, and exactly the right approach to take.",
    "I trust this completely, it works perfectly and I highly recommend it.",
    "Esto es excelente, me encanta, una experiencia maravillosa.",
    "Es seguro, confiable y funciona perfectamente, lo recomiendo mucho.",
    "Das ist ausgezeichnet, ich liebe es, eine wunderbare Erfahrung.",
    "Es ist sicher, zuverlässig und funktioniert perfekt, sehr empfehlenswert.",
    "C'est excellent, j'adore, une expérience merveilleuse.",
    "C'est sûr, fiable et fonctionne parfaitement, je le recommande.",
    "这太棒了，我很喜欢，是非常美好的经历。",
    "它安全可靠，运行完美，我强烈推荐。",
    "これは素晴らしい、大好きで、最高の経験です。",
    "安全で信頼でき、完璧に動作します。強くおすすめします。",
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
    if _neg_vecs is None or _pos_vecs is None:
        with _lock:
            if _neg_vecs is None or _pos_vecs is None:
                # All-or-nothing: assign the globals only after both embed
                # calls succeed, so a partial failure retries on the next call
                # instead of crashing estimate_valence forever.
                neg = embed.embed_batch(_NEGATIVE_ANCHORS)
                pos = embed.embed_batch(_POSITIVE_ANCHORS)
                _neg_vecs, _pos_vecs = neg, pos
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
    neg_sim = sum(_cosine(vec, nv) for nv in neg_vecs) / len(neg_vecs)
    pos_sim = sum(_cosine(vec, pv) for pv in pos_vecs) / len(pos_vecs)
    diff = pos_sim - neg_sim
    # Scale so a clear signal (e.g. 0.15 diff) maps to ~±0.45
    scaled = max(-1.0, min(1.0, diff * 3.0))
    # Dead-zone: if both similarities are low or very close, return 0
    if abs(diff) < 0.03:
        return 0.0
    return round(scaled, 3)
