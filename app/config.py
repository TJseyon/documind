"""
Central settings layer.

Nothing in the rest of the codebase should hardcode a model name, chunk size,
path, or threshold. Everything tunable lives here and is overridable via
environment variables / a .env file. This is what lets you change chunk size
or swap models without touching business logic, and it's the first thing an
interviewer will look for when they ask "how would I configure this
differently in prod?".
"""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "DocuMind"
    environment: str = "development"
    log_level: str = "INFO"

    # --- LLM (generation) ---
    llm_provider: str = "anthropic"  # "anthropic" or "gemini"

    anthropic_api_key: str = Field(default="", description="Required only when llm_provider=anthropic")
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 30

    gemini_api_key: str = Field(default="", description="Required only when llm_provider=gemini")
    gemini_model: str = "gemini-3.6-flash"

    # --- Embeddings ---
    embedding_provider: str = "local"  # "local" (sentence-transformers, needs ~500MB+ RAM) or "gemini" (API call, low memory -- use this on free/small hosting)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    gemini_embedding_model: str = "gemini-embedding-001"

    # --- Reranking ---
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # --- Chunking ---
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    min_chunk_chars: int = 20  # chunks smaller than this are dropped (near-empty noise)

    # --- Retrieval ---
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_fused: int = 10  # how many survive hybrid fusion before reranking
    top_k_final: int = 4  # how many go into the LLM context after reranking
    rrf_k: int = 60  # reciprocal rank fusion constant

    # --- Ingestion limits / validation ---
    max_upload_size_mb: int = 20
    supported_extensions: List[str] = [".pdf", ".docx", ".txt", ".md"]

    # --- Storage ---
    data_dir: str = "./data"
    vector_store_path: str = "./data/chroma"
    metadata_db_path: str = "./data/metadata.db"
    bm25_corpus_path: str = "./data/bm25_corpus.jsonl"

    # --- Groundedness / hallucination checks ---
    require_citation_marker: bool = True
    flag_unsupported_numbers: bool = True

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
