"""
Embedding model behind an interface, so the vector store and pipeline never
import sentence-transformers directly. This is what makes the pipeline
testable without downloading a model, and swappable (e.g. to an API-based
embedding model) without touching retrieval code.
"""
from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbedder(EmbeddingModel):
    """Local, offline embedding model (no per-call API cost or network
    dependency once the model weights are cached on disk)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, show_progress_bar=False, normalize_embeddings=True
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


_singleton: EmbeddingModel | None = None


def get_embedder(model_name: str, batch_size: int = 32) -> EmbeddingModel:
    global _singleton
    if _singleton is None:
        _singleton = SentenceTransformerEmbedder(model_name, batch_size)
    return _singleton


def reset_embedder_for_testing(embedder: EmbeddingModel) -> None:
    """Test hook: inject a fake embedder so unit tests never load a real model."""
    global _singleton
    _singleton = embedder
