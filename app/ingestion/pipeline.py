"""
Ties extraction -> chunking -> embedding -> indexing together.

Re-indexing without a full rebuild (the "standout add-on"): document_id is
derived deterministically from the filename. Re-ingesting a file with the
same name is therefore treated as an update: old chunks for that
document_id are deleted from the vector store (a targeted metadata-filtered
delete, not a full collection rebuild) before the new chunks are added. The
BM25 side still rebuilds its in-memory index on every write -- see the
docstring in bm25_index.py for why that's a documented, acceptable
trade-off at this scale rather than a design gap.
"""
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.config import Settings
from app.db.metadata_store import MetadataStore
from app.exceptions import EmbeddingError
from app.ingestion.chunking import chunk_text
from app.ingestion.extractors import extract_text
from app.retrieval.bm25_index import KeywordIndex
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore

# Fixed namespace so document IDs are stable across process restarts.
_DOC_NAMESPACE = uuid.UUID("f2a293b0-2b3a-4a34-9f0f-5f2f8f4e2b10")


def document_id_for_filename(filename: str) -> str:
    return str(uuid.uuid5(_DOC_NAMESPACE, filename))


@dataclass
class IngestResult:
    document_id: str
    filename: str
    chunk_count: int
    warnings: List[str]


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        metadata_store: MetadataStore,
    ):
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._metadata_store = metadata_store

    def ingest_document(self, file_path: Path, original_filename: str) -> IngestResult:
        extracted = extract_text(file_path, self._settings.supported_extensions)

        chunks = chunk_text(
            extracted.text,
            chunk_size=self._settings.chunk_size_chars,
            overlap=self._settings.chunk_overlap_chars,
            min_chunk_chars=self._settings.min_chunk_chars,
        )
        warnings = list(extracted.warnings)
        if not chunks:
            warnings.append("Document produced no chunks above the minimum length threshold.")

        document_id = document_id_for_filename(original_filename)

        # Incremental re-index: wipe this document's old chunks first.
        existing = self._metadata_store.get_document_by_filename(original_filename)
        if existing is not None:
            self._vector_store.delete_by_document_id(document_id)
            old_chunk_ids = [f"{document_id}::{i}" for i in range(existing["chunk_count"])]
            self._keyword_index.delete_by_ids(old_chunk_ids)

        if chunks:
            chunk_ids = [f"{document_id}::{c.index}" for c in chunks]
            texts = [c.text for c in chunks]
            try:
                embeddings = self._embedder.embed(texts, task_type="RETRIEVAL_DOCUMENT")
            except Exception as e:  # noqa: BLE001 - convert ANY embedding failure into a clear, readable error
                raise EmbeddingError(
                    f"Could not generate embeddings for '{original_filename}': {e}. "
                    "If this is the first document you've ingested, this usually means the "
                    "embedding model couldn't be downloaded -- check your internet connection."
                ) from e
            metadatas = [
                {"document_id": document_id, "source_filename": original_filename, "chunk_index": c.index}
                for c in chunks
            ]
            self._vector_store.upsert(ids=chunk_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            self._keyword_index.add_documents(ids=chunk_ids, texts=texts)

        self._metadata_store.upsert_document(
            document_id=document_id,
            filename=original_filename,
            chunk_count=len(chunks),
            warnings=warnings,
        )

        return IngestResult(document_id=document_id, filename=original_filename, chunk_count=len(chunks), warnings=warnings)
