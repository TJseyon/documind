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
    def embed(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbedder(EmbeddingModel):
    """Local, offline embedding model (no per-call API cost or network
    dependency once the model weights are cached on disk). Runs the model
    in this same process, which means it needs enough RAM to hold it --
    fine on a laptop, often too tight on a free/small hosting tier (see
    GeminiEmbedder below for that case)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        # task_type is a no-op here -- local sentence-transformer models don't
        # distinguish "embedding a document" from "embedding a query" the way
        # Gemini's API does. Accepted anyway so both embedders share one
        # interface and calling code doesn't need to know which is active.
        if not texts:
            return []
        vectors = self._model.encode(
            texts, batch_size=self._batch_size, show_progress_bar=False, normalize_embeddings=True
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class GeminiEmbedder(EmbeddingModel):
    """Calls Google's embedding API instead of running a model locally.

    Why this exists: SentenceTransformerEmbedder loads PyTorch and a model
    into this process's own memory -- on a free hosting tier with ~512MB of
    RAM, that alone can exceed the limit and crash the server. This class
    does the same job (text -> vector) but the heavy computation happens on
    Google's servers, so this process only needs enough memory to make an
    HTTP call. The trade-off: it needs network access and counts against
    Gemini's free-tier request quota, instead of being fully offline.

    Uses "RETRIEVAL_DOCUMENT" vs "RETRIEVAL_QUERY" task types on purpose --
    Gemini's embedding model produces slightly different vectors depending on
    whether the text is something to be searched (a document chunk) or
    something doing the searching (a question), which measurably improves
    retrieval accuracy over treating them identically.
    """

    _DIMENSION = 768  # gemini-embedding-001's default output size

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        from google import genai  # lazy import
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model

    def embed(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        if not texts:
            return []
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=self._types.EmbedContentConfig(task_type=task_type),
        )
        return [list(e.values) for e in result.embeddings]

    @property
    def dimension(self) -> int:
        return self._DIMENSION


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
