"""Zero-shot NER via GLiNER2 — lazy-loaded, thread-safe singleton."""
from __future__ import annotations

import os
import threading
from typing import Optional


class NERProvider:
    """Wraps GLiNER2 with lazy initialization. Thread-safe singleton."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.environ.get(
            "SMRTI_NER_MODEL", "fastino/gliner2-multi-v1"
        )
        self._model = None
        self._lock = threading.Lock()
        self._has_classify = False

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from gliner2 import GLiNER2

                    try:
                        self._model = GLiNER2.from_pretrained(
                            self._model_name, local_files_only=True
                        )
                    except Exception:
                        self._model = GLiNER2.from_pretrained(self._model_name)
                    self._has_classify = hasattr(self._model, "classify_text")
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
        raw = model.extract_entities(text, labels)
        # GLiNER2 returns {"entities": {type: [names]}}
        entities_dict = raw.get("entities", {}) if isinstance(raw, dict) else {}
        # Deduplicate: keep one entry per name_lower, preferring the
        # highest-priority (most specific) entity type when a span matches
        # multiple labels.
        best: dict[str, dict] = {}
        for etype, names in entities_dict.items():
            for name in names:
                key = name.lower()
                priority = _TYPE_PRIORITY.get(etype, len(_DEFAULT_LABELS))
                existing = best.get(key)
                if existing is None or priority < _TYPE_PRIORITY.get(existing["type"], len(_DEFAULT_LABELS)):
                    best[key] = {
                        "name": name,
                        "type": etype,
                        "score": 1.0,
                    }
        return list(best.values())

    def classify_pronoun(self, name: str) -> bool:
        """Return True if `name` is a pronoun rather than a proper name.

        Uses GLiNER2's classify_text when available, otherwise returns False.
        """
        if not name or not name.strip():
            return False
        try:
            model = self._get_model()
        except Exception:
            return False
        if not self._has_classify:
            return False
        try:
            result = model.classify_text(name.strip(), {"type": ["proper_name", "pronoun"]})
            if isinstance(result, dict):
                return result.get("type") == "pronoun"
            return False
        except Exception:
            return False


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
    "pronoun",
]

# When GLiNER tags the same span under multiple labels, keep the most specific.
# Lower index = higher priority.
_TYPE_PRIORITY: dict[str, int] = {t: i for i, t in enumerate(_DEFAULT_LABELS)}

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
