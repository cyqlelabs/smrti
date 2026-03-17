from __future__ import annotations

import threading


class EmbeddingProvider:
    """Wraps FastEmbed with lazy initialization. Thread-safe singleton per model name."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding
                    self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._get_model()
        embeddings = list(model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        return [e.tolist() for e in model.embed(texts)]

    @property
    def dimensions(self) -> int:
        return 384  # bge-small-en-v1.5
