"""
Hallucination handling.

Two independent, deterministic checks -- neither relies on trusting the LLM's
own claim that it's grounded:

1. Citation check: the answer must contain at least one [chunk_N] tag. If the
   model didn't cite anything, we treat the whole answer as unsupported,
   regardless of how confident it sounds.
2. Numeric/date check: every number, percentage, currency amount, or date
   mentioned in the answer must appear (verbatim, after light normalization)
   somewhere in the retrieved context. This specifically targets the most
   common and most dangerous hallucination pattern -- a fabricated statistic
   that reads as authoritative. It will not catch every kind of hallucination
   (a fabricated *claim* with no numbers slips through), which is exactly why
   this is paired with the citation check and, on the eval side, a manual
   "is this actually accurate" review column.
"""
import re
from dataclasses import dataclass, field
from typing import List

_CITATION_RE = re.compile(r"\[chunk_\w+\]")

# Numbers (incl. decimals, %, currency), and common date formats.
_NUMBER_RE = re.compile(
    r"\$?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b"
)
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)

_TRIVIAL_NUMBERS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}


@dataclass
class GroundingResult:
    grounded: bool
    has_citation: bool
    unsupported_claims: List[str] = field(default_factory=list)


def _normalize(value: str) -> str:
    return value.replace(",", "").replace("$", "").replace("%", "").strip().lower()


def _extract_candidates(text: str) -> List[str]:
    return _NUMBER_RE.findall(text) + _DATE_RE.findall(text)


def check_groundedness(answer: str, context_chunks: List[str], require_citation: bool = True) -> GroundingResult:
    has_citation = bool(_CITATION_RE.search(answer))

    if "don't have enough information" in answer.lower():
        # The model correctly declined -- that's the grounded behavior we want.
        return GroundingResult(grounded=True, has_citation=has_citation, unsupported_claims=[])

    context_blob = _normalize(" ".join(context_chunks))
    unsupported = []
    for candidate in _extract_candidates(answer):
        normalized = _normalize(candidate)
        if normalized in _TRIVIAL_NUMBERS:
            continue  # small numbers are usually list markers / counts of citations, not claims
        if normalized and normalized not in context_blob:
            unsupported.append(candidate)

    grounded = True
    if require_citation and not has_citation:
        grounded = False
    if unsupported:
        grounded = False

    return GroundingResult(grounded=grounded, has_citation=has_citation, unsupported_claims=unsupported)
