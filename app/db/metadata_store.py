"""
Lightweight metadata store using stdlib sqlite3 -- deliberately not a second
service to run. Tracks (a) which documents are indexed, for /documents and
for re-ingestion/dedup logic, and (b) every query/answer pair, which is what
the evaluation harness and the "log every input/output" requirement need.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional


class MetadataStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    warnings TEXT NOT NULL DEFAULT '[]'
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    grounded INTEGER NOT NULL,
                    unsupported_claims_json TEXT NOT NULL DEFAULT '[]',
                    latency_ms INTEGER NOT NULL
                )"""
            )

    def upsert_document(self, document_id: str, filename: str, chunk_count: int, warnings: List[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO documents (document_id, filename, ingested_at, chunk_count, warnings)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                        filename=excluded.filename,
                        ingested_at=excluded.ingested_at,
                        chunk_count=excluded.chunk_count,
                        warnings=excluded.warnings""",
                (document_id, filename, datetime.now(timezone.utc).isoformat(), chunk_count, json.dumps(warnings)),
            )

    def delete_document(self, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def get_document_by_filename(self, filename: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
            return cur.fetchone()

    def list_documents(self) -> List[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
            return cur.fetchall()

    def total_chunk_count(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) AS total FROM documents")
            return cur.fetchone()["total"]

    def log_query(
        self,
        question: str,
        answer: str,
        citations: list,
        grounded: bool,
        unsupported_claims: List[str],
        latency_ms: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO query_logs
                   (timestamp, question, answer, citations_json, grounded, unsupported_claims_json, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    question,
                    answer,
                    json.dumps(citations),
                    int(grounded),
                    json.dumps(unsupported_claims),
                    latency_ms,
                ),
            )

    def get_query_logs(self, limit: int = 100) -> List[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,))
            return cur.fetchall()
