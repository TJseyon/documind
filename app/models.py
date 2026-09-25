"""
Request/response schemas. Validation lives here so bad input is rejected at
the API boundary with a clear 422, instead of failing confusingly three
layers deep in the retrieval pipeline.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str
    warnings: List[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    ingested_at: datetime
    chunk_count: int
    warnings: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

    @field_validator("question")
    @classmethod
    def question_must_be_meaningful(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must be at least 3 characters")
        if len(v) > 2000:
            raise ValueError("question must be under 2000 characters")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_in_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 20):
            raise ValueError("top_k must be between 1 and 20")
        return v


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    source_filename: str
    text_snippet: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    grounded: bool
    unsupported_claims: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_indexed: int


class ErrorResponse(BaseModel):
    error_code: str
    detail: str
