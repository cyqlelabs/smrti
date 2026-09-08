from __future__ import annotations

import threading
import warnings
from pathlib import Path

_singleton: "EmbeddingProvider | None" = None
_singleton_lock = threading.Lock()


def get_embedding_provider() -> "EmbeddingProvider":
    """Return the process-wide EmbeddingProvider singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = EmbeddingProvider()
    return _singleton


def _mapped_model_dir(model_name: str) -> Path:
    """Download the model as FastEmbed would and return a directory holding it with its weights mapped."""
    from fastembed import TextEmbedding
    from fastembed.common.utils import define_cache_dir

    from .weights import externalized

    description = TextEmbedding._get_model_description(model_name)
    model_dir = TextEmbedding.download_model(description, str(define_cache_dir()))
    return externalized(model_dir / description.model_file, with_siblings=True).parent


class EmbeddingProvider:
    """Wraps FastEmbed with lazy initialization. Thread-safe singleton per model name."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    from .weights import release_heap

                    mapped = _mapped_model_dir(self._model_name)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*now uses mean pooling.*")
                        self._model = TextEmbedding(
                            model_name=self._model_name,
                            specific_model_path=str(mapped),
                            enable_cpu_mem_arena=False,
                        )
                    release_heap()
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
        return 384  # paraphrase-multilingual-MiniLM-L12-v2
