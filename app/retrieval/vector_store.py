"""
Vector store behind an interface.

Chroma is used here (embedded, on-disk, zero extra infra to run) rather than
a Qdrant/pgvector server, because for a solo project the operational cost of
running a separate database process isn't worth it. The interface is the
part that matters for the resume story: swapping to Qdrant or pgvector in
production is a new class implementing `VectorStore`, not a rewrite of the
retrieval or ingestion code.
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class VectorMatch:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    score: float  # higher is better (similarity, not distance)


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict],
    ) -> None:
        ...

    @abstractmethod
    def query(self, embedding: List[float], top_k: int) -> List[VectorMatch]:
        ...

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[VectorMatch]:
        """Direct lookup, not a similarity search. Needed to hydrate text for
        chunks that a keyword-only (BM25) hit surfaced but that didn't rank
        highly enough to appear in the vector similarity results -- every
        chunk lives in the vector store regardless of which retriever found
        it, so this is always safe to call."""
        ...

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_path: str, collection_name: str = "documind_chunks"):
        import chromadb  # lazy import

        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding: List[float], top_k: int) -> List[VectorMatch]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(query_embeddings=[embedding], n_results=min(top_k, self._collection.count()))
        matches = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns squared L2 distance on normalized vectors by default;
            # convert to a similarity score in a stable, monotonic way.
            similarity = 1.0 / (1.0 + dist)
            matches.append(
                VectorMatch(
                    chunk_id=cid,
                    document_id=meta.get("document_id", ""),
                    text=doc,
                    metadata=meta,
                    score=similarity,
                )
            )
        return matches

    def get_by_ids(self, ids: List[str]) -> List[VectorMatch]:
        if not ids:
            return []
        result = self._collection.get(ids=ids)
        matches = []
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            matches.append(VectorMatch(chunk_id=cid, document_id=meta.get("document_id", ""), text=doc, metadata=meta, score=0.0))
        return matches

    def delete_by_document_id(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self._collection.count()


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine similarity store. No external dependency, no disk
    I/O -- used in unit tests so retrieval logic can be verified without
    chromadb or a downloaded embedding model."""

    def __init__(self):
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []
        self._documents: List[str] = []
        self._metadatas: List[dict] = []

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        for cid, vec, doc, meta in zip(ids, embeddings, documents, metadatas):
            if cid in self._ids:
                i = self._ids.index(cid)
                self._vectors[i], self._documents[i], self._metadatas[i] = vec, doc, meta
            else:
                self._ids.append(cid)
                self._vectors.append(vec)
                self._documents.append(doc)
                self._metadatas.append(meta)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-8
        norm_b = math.sqrt(sum(y * y for y in b)) or 1e-8
        return dot / (norm_a * norm_b)

    def query(self, embedding: List[float], top_k: int) -> List[VectorMatch]:
        scored = [
            VectorMatch(
                chunk_id=cid,
                document_id=meta.get("document_id", ""),
                text=doc,
                metadata=meta,
                score=self._cosine(embedding, vec),
            )
            for cid, vec, doc, meta in zip(self._ids, self._vectors, self._documents, self._metadatas)
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def get_by_ids(self, ids: List[str]) -> List[VectorMatch]:
        wanted = set(ids)
        return [
            VectorMatch(chunk_id=cid, document_id=meta.get("document_id", ""), text=doc, metadata=meta, score=0.0)
            for cid, doc, meta in zip(self._ids, self._documents, self._metadatas)
            if cid in wanted
        ]

    def delete_by_document_id(self, document_id: str) -> None:
        keep = [i for i, m in enumerate(self._metadatas) if m.get("document_id") != document_id]
        self._ids = [self._ids[i] for i in keep]
        self._vectors = [self._vectors[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]

    def count(self) -> int:
        return len(self._ids)
