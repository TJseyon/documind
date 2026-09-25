"""
Prompt construction. The system prompt is the first line of defense against
hallucination: the model is instructed to answer only from the provided
context and to tag every factual claim with the chunk it came from. The
second line of defense is `grounding.py`, which checks the model actually
did that instead of trusting it.
"""
from typing import List, Tuple

SYSTEM_PROMPT = """You are a document question-answering assistant. You must \
answer ONLY using the numbered context chunks provided below. Follow these \
rules strictly:

1. Every factual claim in your answer must be immediately followed by the \
tag of the chunk it came from, like [chunk_2].
2. If the answer is not contained in the provided chunks, say exactly: \
"I don't have enough information in the provided documents to answer this." \
Do not guess, and do not use outside knowledge.
3. Do not invent numbers, dates, names, or figures that do not appear \
verbatim in the context.
4. Be concise. Do not repeat the question back."""


def build_context_block(chunks: List[Tuple[str, str]]) -> str:
    """chunks: list of (chunk_tag, chunk_text), e.g. ('chunk_1', '...')."""
    blocks = [f"[{tag}]\n{text}" for tag, text in chunks]
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: List[Tuple[str, str]]) -> str:
    context = build_context_block(chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, citing chunk tags as instructed."
    )
