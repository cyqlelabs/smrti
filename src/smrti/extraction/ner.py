"""Zero-shot NER via GLiNER — lazy-loaded, thread-safe singleton."""
from __future__ import annotations

import os
import threading
from typing import Optional


class NERProvider:
    """Wraps GLiNER with lazy initialization. Thread-safe singleton."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.environ.get(
            "SMRTI_NER_MODEL", "urchade/gliner_multi-v2.1"
        )
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from gliner import GLiNER

                    self._model = GLiNER.from_pretrained(self._model_name)
        return self._model

    def extract(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float = 0.4,
    ) -> list[dict]:
        """Extract entity spans from text.

        Returns list of {"name": str, "type": str, "score": float}.
        """
        if labels is None:
            labels = _DEFAULT_LABELS
        model = self._get_model()
        entities = model.predict_entities(text, labels, threshold=threshold)
        # Deduplicate: keep highest-score span per (text_lower, label)
        best: dict[tuple[str, str], dict] = {}
        for ent in entities:
            key = (ent["text"].lower(), ent["label"])
            existing = best.get(key)
            if existing is None or ent["score"] > existing["score"]:
                best[key] = {
                    "name": ent["text"],
                    "type": ent["label"],
                    "score": ent["score"],
                }
        return list(best.values())


_DEFAULT_LABELS = [
    "person",
    "organization",
    "project",
    "tool",
    "preference",
    "constraint",
    "location",
    "event",
    "concept",
    "goal",
]

_instance: Optional[NERProvider] = None
_instance_lock = threading.Lock()


def get_ner() -> NERProvider:
    """Return the module-level NERProvider singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NERProvider()
    return _instance
