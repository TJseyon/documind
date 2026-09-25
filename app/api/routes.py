import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.config import Settings, get_settings
from app.db.metadata_store import MetadataStore
from app.dependencies import (
    get_ingestion_pipeline,
    get_keyword_index,
    get_metadata_store,
    get_query_pipeline,
    get_vector_store,
)
from app.exceptions import DocumentNotFoundError, FileTooLargeError
from app.ingestion.pipeline import IngestionPipeline, document_id_for_filename
from app.models import DocumentSummary, HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.query_pipeline import QueryPipeline
from app.retrieval.bm25_index import KeywordIndex
from app.retrieval.vector_store import VectorStore

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    metadata_store: MetadataStore = Depends(get_metadata_store),
    vector_store: VectorStore = Depends(get_vector_store),
) -> HealthResponse:
    docs = metadata_store.list_documents()
    return HealthResponse(status="ok", documents_indexed=len(docs), chunks_indexed=vector_store.count())


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileTooLargeError(
            f"'{file.filename}' is {size_mb:.1f}MB, over the {settings.max_upload_size_mb}MB limit."
        )

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = pipeline.ingest_document(tmp_path, original_filename=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
        status="indexed",
        warnings=result.warnings,
    )


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(metadata_store: MetadataStore = Depends(get_metadata_store)) -> list[DocumentSummary]:
    import json

    rows = metadata_store.list_documents()
    return [
        DocumentSummary(
            document_id=r["document_id"],
            filename=r["filename"],
            ingested_at=r["ingested_at"],
            chunk_count=r["chunk_count"],
            warnings=json.loads(r["warnings"]),
        )
        for r in rows
    ]


@router.delete("/documents/{filename}", status_code=204)
def delete_document(
    filename: str,
    metadata_store: MetadataStore = Depends(get_metadata_store),
    vector_store: VectorStore = Depends(get_vector_store),
    keyword_index: KeywordIndex = Depends(get_keyword_index),
) -> None:
    existing = metadata_store.get_document_by_filename(filename)
    if existing is None:
        raise DocumentNotFoundError(f"No indexed document with filename '{filename}'.")
    document_id = document_id_for_filename(filename)
    vector_store.delete_by_document_id(document_id)
    chunk_ids = [f"{document_id}::{i}" for i in range(existing["chunk_count"])]
    keyword_index.delete_by_ids(chunk_ids)
    metadata_store.delete_document(document_id)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, pipeline: QueryPipeline = Depends(get_query_pipeline)) -> QueryResponse:
    return pipeline.answer(request.question, top_k=request.top_k)
