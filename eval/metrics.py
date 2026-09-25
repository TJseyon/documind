"""
Automatic, deterministic scoring functions.

These deliberately do NOT try to judge whether an answer is "good" in any
holistic sense -- that's what the manual_score column in the CSV output is
for. What they do check, cheaply and objectively:

  - retrieval_hit: did the correct source document actually show up among
    the chunks retrieved for this question? (retrieval precision, the thing
    most RAG demos never measure)
  - contains_answer: does the generated answer contain the expected
    substring? (a crude but honest automatic proxy for correctness)
  - grounded: did the groundedness checker (app/generation/grounding.py)
    flag this answer as citing its sources and not inventing numbers?
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalCaseResult:
    case_id: str
    question: str
    expected_source_document: str
    expected_answer_substring: str
    retrieved_documents: List[str]
    answer: str
    grounded: bool
    unsupported_claims: List[str]
    latency_ms: int
    manual_score: Optional[str] = None  # left blank for a human to fill in

    @property
    def retrieval_hit(self) -> bool:
        return self.expected_source_document in self.retrieved_documents

    @property
    def contains_answer(self) -> bool:
        return self.expected_answer_substring.lower() in self.answer.lower()


def aggregate_metrics(results: List[EvalCaseResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"n_cases": 0}
    return {
        "n_cases": n,
        "retrieval_hit_rate": round(sum(r.retrieval_hit for r in results) / n, 3),
        "contains_answer_rate": round(sum(r.contains_answer for r in results) / n, 3),
        "groundedness_rate": round(sum(r.grounded for r in results) / n, 3),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
    }
