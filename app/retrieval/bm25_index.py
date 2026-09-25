"""
BM25 keyword index.

Honest limitation, worth stating out loud in an interview rather than
hiding: rank_bm25's BM25Okapi has no incremental-add API -- every write
rebuilds the in-memory index from the full corpus. For a personal/portfolio
corpus (hundreds to low thousands of chunks) this is milliseconds and a
non-issue. At real production scale you'd swap this for OpenSearch/
Elasticsearch (which do support incremental indexing) behind the same
`KeywordIndex` interface -- that's the trade-off this class exists to make
visible, not hide.
"""
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class KeywordMatch:
    chunk_id: str
    score: float


class KeywordIndex(ABC):
    @abstractmethod
    def add_documents(self, ids: List[str], texts: List[str]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[KeywordMatch]:
        ...

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> None:
        ...


class BM25Index(KeywordIndex):
    def __init__(self, persist_path: str | None = None):
        self._persist_path = Path(persist_path) if persist_path else None
        self._corpus: Dict[str, str] = {}  # chunk_id -> raw text
        self._bm25: BM25Okapi | None = None
        if self._persist_path and self._persist_path.exists():
            self._load()
        self._rebuild()

    def _load(self) -> None:
        with open(self._persist_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                self._corpus[record["id"]] = record["text"]

    def _persist(self) -> None:
        if not self._persist_path:
            return
        with open(self._persist_path, "w", encoding="utf-8") as f:
            for cid, text in self._corpus.items():
                f.write(json.dumps({"id": cid, "text": text}) + "\n")

    def _rebuild(self) -> None:
        if not self._corpus:
            self._bm25 = None
            self._ordered_ids: List[str] = []
            return
        self._ordered_ids = list(self._corpus.keys())
        tokenized = [_tokenize(self._corpus[cid]) for cid in self._ordered_ids]
        self._bm25 = BM25Okapi(tokenized)

    def add_documents(self, ids: List[str], texts: List[str]) -> None:
        for cid, text in zip(ids, texts):
            self._corpus[cid] = text
        self._rebuild()
        self._persist()

    def delete_by_ids(self, ids: List[str]) -> None:
        for cid in ids:
            self._corpus.pop(cid, None)
        self._rebuild()
        self._persist()

    def search(self, query: str, top_k: int) -> List[KeywordMatch]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._ordered_ids, scores), key=lambda x: x[1], reverse=True)
        return [KeywordMatch(chunk_id=cid, score=float(s)) for cid, s in ranked[:top_k] if s > 0]
