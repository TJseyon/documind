"""
Hybrid search fusion.

Why hybrid at all: vector search finds semantically similar chunks but
misses exact terms it hasn't learned to associate (SKU codes, exact error
strings, proper nouns, acronyms). BM25 finds exact keyword matches but
misses paraphrases and synonyms. Neither is sufficient alone for the kind of
mixed factual/semantic questions this project targets.

Fusion method: Reciprocal Rank Fusion (RRF). RRF combines ranked lists using
only rank position (not raw scores), which sidesteps the fact that BM25
scores and cosine-similarity scores live on completely different, incomparable
scales -- there's no principled way to average them directly, but their rank
orders can be combined safely. score(d) = sum(1 / (k + rank(d))) over every
list that contains d; k=60 is the standard constant from the original RRF
paper (it damps the influence of any single very-high rank).

When vector and BM25 disagree: a chunk that's mediocre-but-present in both
lists usually outranks a chunk that's #1 in one list and absent from the
other, because RRF rewards the sum of appearances across lists, not a single
strong signal in isolation.
"""
from dataclasses import dataclass
from typing import Dict, List

from app.retrieval.bm25_index import KeywordMatch
from app.retrieval.vector_store import VectorMatch


@dataclass
class FusedResult:
    chunk_id: str
    fused_score: float
    vector_rank: int | None
    bm25_rank: int | None


def reciprocal_rank_fusion(
    vector_results: List[VectorMatch],
    bm25_results: List[KeywordMatch],
    k: int = 60,
) -> List[FusedResult]:
    scores: Dict[str, float] = {}
    vector_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}

    for rank, match in enumerate(vector_results):
        scores[match.chunk_id] = scores.get(match.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        vector_ranks[match.chunk_id] = rank + 1

    for rank, match in enumerate(bm25_results):
        scores[match.chunk_id] = scores.get(match.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        bm25_ranks[match.chunk_id] = rank + 1

    fused = [
        FusedResult(
            chunk_id=cid,
            fused_score=score,
            vector_rank=vector_ranks.get(cid),
            bm25_rank=bm25_ranks.get(cid),
        )
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda r: r.fused_score, reverse=True)
    return fused
