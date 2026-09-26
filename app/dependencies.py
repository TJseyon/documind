"""
FastAPI dependency providers. Centralizing these (rather than instantiating
inside route handlers) is what lets tests override individual pieces --
swap in a FakeLLMClient, an InMemoryVectorStore, etc. -- via
`app.dependency_overrides`, without monkeypatching internals.
"""
from functools import lru_cache

from app.config import Settings, get_settings
from app.db.metadata_store import MetadataStore
from app.generation.llm_client import AnthropicLLMClient, GeminiLLMClient, LLMClient
from app.ingestion.pipeline import IngestionPipeline
from app.query_pipeline import QueryPipeline
from app.retrieval.bm25_index import BM25Index, KeywordIndex
from app.retrieval.embeddings import EmbeddingModel, GeminiEmbedder, get_embedder
from app.retrieval.reranker import CrossEncoderReranker, NoOpReranker, Reranker
from app.retrieval.vector_store import ChromaVectorStore, VectorStore


@lru_cache
def get_metadata_store() -> MetadataStore:
    return MetadataStore(get_settings().metadata_db_path)


@lru_cache
def get_vector_store() -> VectorStore:
    return ChromaVectorStore(get_settings().vector_store_path)


@lru_cache
def get_keyword_index() -> KeywordIndex:
    return BM25Index(get_settings().bm25_corpus_path)


def build_embedding_model(s: Settings) -> EmbeddingModel:
    """Pure factory function, separated from get_embedding_model so provider
    selection can be unit tested without going through the lru_cache'd
    get_settings() singleton (same pattern as build_llm_client above)."""
    if s.embedding_provider == "gemini":
        return GeminiEmbedder(api_key=s.gemini_api_key, model=s.gemini_embedding_model)
    return get_embedder(s.embedding_model_name, s.embedding_batch_size)


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return build_embedding_model(get_settings())


@lru_cache
def get_reranker() -> Reranker:
    s = get_settings()
    if not s.use_reranker:
        return NoOpReranker()
    try:
        return CrossEncoderReranker(s.reranker_model_name)
    except Exception:  # noqa: BLE001 - degrade gracefully if model can't be loaded
        return NoOpReranker()


def build_llm_client(s: Settings) -> LLMClient:
    """Pure factory function, separated from get_llm_client so provider
    selection can be unit tested without going through the lru_cache'd
    get_settings() singleton."""
    if s.llm_provider == "gemini":
        return GeminiLLMClient(api_key=s.gemini_api_key, model=s.gemini_model, max_tokens=s.llm_max_tokens)
    return AnthropicLLMClient(
        api_key=s.anthropic_api_key, model=s.llm_model, max_tokens=s.llm_max_tokens, timeout=s.llm_timeout_seconds
    )


def get_llm_client() -> LLMClient:
    return build_llm_client(get_settings())


def get_ingestion_pipeline() -> IngestionPipeline:
    s = get_settings()
    return IngestionPipeline(
        settings=s,
        embedder=get_embedding_model(),
        vector_store=get_vector_store(),
        keyword_index=get_keyword_index(),
        metadata_store=get_metadata_store(),
    )


def get_query_pipeline() -> QueryPipeline:
    s = get_settings()
    return QueryPipeline(
        settings=s,
        embedder=get_embedding_model(),
        vector_store=get_vector_store(),
        keyword_index=get_keyword_index(),
        reranker=get_reranker(),
        llm_client=get_llm_client(),
        metadata_store=get_metadata_store(),
    )
