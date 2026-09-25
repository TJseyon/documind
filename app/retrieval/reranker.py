"""
Reranking stage.

Why rerank at all when we already fused two ranked lists: RRF is a cheap,
rank-only signal that ignores the actual text of the query and the
candidate. A cross-encoder reads the query and each candidate chunk
*together* (rather than as independently-embedded vectors) and scores their
true relevance -- much more accurate, but too slow to run over the whole
corpus, which is why it only runs on the top ~10 candidates that fusion
already narrowed down, not all chunks.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """candidates: list of (chunk_id, chunk_text). Returns (chunk_id, score)
        sorted descending by relevance."""
        ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked]


class NoOpReranker(Reranker):
    """Passes fused order through unchanged. Used when use_reranker=False in
    config, or in tests/environments without the cross-encoder model
    available."""

    def rerank(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        return [(cid, 1.0 - (i * 0.01)) for i, (cid, _) in enumerate(candidates)]
