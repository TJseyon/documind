"""
Chunking strategy.

Design choice (this is the thing an interviewer will ask about): chunk on
character count with a separator hierarchy, not fixed-size character
slicing. Splitting mid-sentence is what destroys retrieval quality --
a chunk boundary that cuts a sentence in half means neither half embeds
to something that resembles the original meaning. So we try to split on the
"softest" available boundary first (paragraph breaks), and only fall back to
harder boundaries (single newline, sentence-ending punctuation, whitespace)
when a segment is still too big.

Overlap exists so that an answer sitting right at a chunk boundary isn't
silently split across two chunks with neither one containing the full
context.

Why characters instead of tokens: token count depends on which tokenizer the
downstream embedding model uses, and that tokenizer isn't always available
without a network call to fetch its vocab file. Character count is a stable,
dependency-free proxy (~4 chars/token for English is the common rule of
thumb) and is what this project uses. Swapping in a real tokenizer-based
counter later is a one-function change (see `chunk_size_chars` in config).
"""
from dataclasses import dataclass
from typing import List

# Ordered from "softest" to "hardest" split point.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]


@dataclass
class Chunk:
    text: str
    index: int


def _split_on_separator(text: str, separator: str) -> List[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # No separator left that helps -- hard-cut at chunk_size.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, remaining_seps = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep)

    # Re-glue pieces back together up to chunk_size, using the separator we
    # split on (so we don't lose it), recursing into any piece still too big.
    results: List[str] = []
    buffer = ""
    for piece in pieces:
        candidate = (buffer + sep + piece) if buffer else piece
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                results.append(buffer)
            if len(piece) > chunk_size:
                results.extend(_recursive_split(piece, chunk_size, remaining_seps))
                buffer = ""
            else:
                buffer = piece
    if buffer:
        results.append(buffer)
    return results


def _add_overlap(pieces: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    overlapped = [pieces[0]]
    for prev, current in zip(pieces, pieces[1:]):
        tail = prev[-overlap:] if len(prev) > overlap else prev
        overlapped.append(tail + current)
    return overlapped


def chunk_text(text: str, chunk_size: int, overlap: int, min_chunk_chars: int = 20) -> List[Chunk]:
    """Split `text` into overlapping chunks, dropping any chunk that's just
    whitespace/noise shorter than `min_chunk_chars` (an edge case -- otherwise
    a stray page-break artifact becomes a "chunk" that pollutes retrieval)."""
    text = text.strip()
    if not text:
        return []

    raw_pieces = _recursive_split(text, chunk_size, list(_SEPARATORS))
    raw_pieces = [p.strip() for p in raw_pieces if p.strip()]
    overlapped = _add_overlap(raw_pieces, overlap)

    chunks = [
        Chunk(text=piece, index=i)
        for i, piece in enumerate(overlapped)
        if len(piece) >= min_chunk_chars
    ]
    return chunks
