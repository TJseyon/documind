import time
from typing import List

from app.config import Settings
from app.db.metadata_store import MetadataStore
from app.exceptions import EmbeddingError, NoDocumentsIndexedError
from app.generation.grounding import check_groundedness
from app.generation.llm_client import LLMClient
from app.models import Citation, QueryResponse
from app.retrieval.bm25_index import KeywordIndex
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore


class QueryPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        reranker: Reranker,
        llm_client: LLMClient,
        metadata_store: MetadataStore,
    ):
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._reranker = reranker
        self._llm_client = llm_client
        self._metadata_store = metadata_store

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        start = time.perf_counter()
        s = self._settings
        final_k = top_k or s.top_k_final

        if self._vector_store.count() == 0:
            raise NoDocumentsIndexedError("No documents have been ingested yet. Call /ingest first.")

        try:
            query_embedding = self._embedder.embed([question], task_type="RETRIEVAL_QUERY")[0]
        except Exception as e:  # noqa: BLE001 - convert ANY embedding failure into a clear, readable error
            raise EmbeddingError(f"Could not process your question: {e}") from e
        vector_matches = self._vector_store.query(query_embedding, top_k=s.top_k_vector)
        bm25_matches = self._keyword_index.search(question, top_k=s.top_k_bm25)

        fused = reciprocal_rank_fusion(vector_matches, bm25_matches, k=s.rrf_k)[: s.top_k_fused]

        # Build a lookup so we can recover chunk text/metadata after fusion,
        # which only carries chunk_id + score. Chunks that BM25 surfaced but
        # that weren't in the vector similarity results still live in the
        # vector store (every chunk is written there on ingest) -- hydrate
        # those via a direct ID lookup rather than dropping them.
        by_id = {m.chunk_id: m for m in vector_matches}
        missing_ids = [f.chunk_id for f in fused if f.chunk_id not in by_id]
        if missing_ids:
            for m in self._vector_store.get_by_ids(missing_ids):
                by_id[m.chunk_id] = m

        candidate_ids = [f.chunk_id for f in fused if f.chunk_id in by_id]
        candidates = [(cid, by_id[cid].text) for cid in candidate_ids]
        reranked = self._reranker.rerank(question, candidates)[:final_k]

        text_by_id = dict(candidates)
        meta_by_id = {cid: by_id[cid].metadata for cid in candidate_ids}

        tagged_chunks = [(f"chunk_{i+1}", text_by_id[cid]) for i, (cid, _) in enumerate(reranked)]
        answer_text = self._llm_client.generate(question, tagged_chunks)

        grounding = check_groundedness(
            answer_text,
            context_chunks=[t for _, t in tagged_chunks],
            require_citation=s.require_citation_marker,
        )

        citations = [
            Citation(
                chunk_id=cid,
                document_id=meta_by_id[cid].get("document_id", ""),
                source_filename=meta_by_id[cid].get("source_filename", "unknown"),
                text_snippet=(text_by_id[cid][:240] + "...") if len(text_by_id[cid]) > 240 else text_by_id[cid],
                score=score,
            )
            for cid, score in reranked
        ]

        latency_ms = int((time.perf_counter() - start) * 1000)

        warnings: List[str] = []
        if not grounding.has_citation:
            warnings.append("Model did not cite any chunk; answer is treated as unsupported.")
        if grounding.unsupported_claims:
            warnings.append(
                f"Answer contains values not found in retrieved context: {grounding.unsupported_claims}"
            )

        self._metadata_store.log_query(
            question=question,
            answer=answer_text,
            citations=[c.model_dump() for c in citations],
            grounded=grounding.grounded,
            unsupported_claims=grounding.unsupported_claims,
            latency_ms=latency_ms,
        )

        return QueryResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            grounded=grounding.grounded,
            unsupported_claims=grounding.unsupported_claims,
            warnings=warnings,
            latency_ms=latency_ms,
        )
