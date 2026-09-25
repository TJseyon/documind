"""
Test fixtures.

The whole point of the interface-based design in app/retrieval and
app/generation is exercised here: API-route tests run against an
InMemoryVectorStore, a hash-based FakeEmbedder, a NoOpReranker, and a
FakeLLMClient -- meaning the full /ingest -> /query flow can be tested with
zero network calls and zero downloaded ML models.
"""
import hashlib
from typing import List

import pytest
from fastapi.testclient import TestClient

from app.db.metadata_store import MetadataStore
from app.dependencies import (
    get_embedding_model,
    get_ingestion_pipeline,
    get_keyword_index,
    get_llm_client,
    get_metadata_store,
    get_query_pipeline,
    get_reranker,
    get_vector_store,
)
from app.generation.llm_client import FakeLLMClient
from app.ingestion.pipeline import IngestionPipeline
from app.query_pipeline import QueryPipeline
from app.retrieval.bm25_index import BM25Index
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.reranker import NoOpReranker
from app.retrieval.vector_store import InMemoryVectorStore


class FakeEmbedder(EmbeddingModel):
    """Deterministic, dependency-free stand-in: hashes text into a small
    fixed-size vector. Not semantically meaningful, but stable -- identical
    text always produces identical vectors, which is all these tests need."""

    _DIM = 16

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [b / 255.0 for b in digest[: self._DIM]]
            vectors.append(vec)
        return vectors

    @property
    def dimension(self) -> int:
        return self._DIM


@pytest.fixture()
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture()
def in_memory_vector_store():
    return InMemoryVectorStore()


@pytest.fixture()
def bm25_index():
    return BM25Index(persist_path=None)


@pytest.fixture()
def metadata_store(tmp_path):
    return MetadataStore(str(tmp_path / "test_metadata.db"))


@pytest.fixture()
def fake_llm_client():
    return FakeLLMClient()


@pytest.fixture()
def test_client(fake_embedder, in_memory_vector_store, bm25_index, metadata_store, fake_llm_client):
    from app.main import app
    from app.config import get_settings

    settings = get_settings()

    def _ingestion_pipeline():
        return IngestionPipeline(
            settings=settings,
            embedder=fake_embedder,
            vector_store=in_memory_vector_store,
            keyword_index=bm25_index,
            metadata_store=metadata_store,
        )

    def _query_pipeline():
        return QueryPipeline(
            settings=settings,
            embedder=fake_embedder,
            vector_store=in_memory_vector_store,
            keyword_index=bm25_index,
            reranker=NoOpReranker(),
            llm_client=fake_llm_client,
            metadata_store=metadata_store,
        )

    app.dependency_overrides[get_embedding_model] = lambda: fake_embedder
    app.dependency_overrides[get_vector_store] = lambda: in_memory_vector_store
    app.dependency_overrides[get_keyword_index] = lambda: bm25_index
    app.dependency_overrides[get_metadata_store] = lambda: metadata_store
    app.dependency_overrides[get_reranker] = lambda: NoOpReranker()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm_client
    app.dependency_overrides[get_ingestion_pipeline] = _ingestion_pipeline
    app.dependency_overrides[get_query_pipeline] = _query_pipeline

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
