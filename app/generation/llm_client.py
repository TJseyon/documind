"""
LLM client, behind an interface so the generation step can be unit-tested
with a fake client and no network/API key required.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

from app.exceptions import DownstreamLLMError
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt


class LLMClient(ABC):
    @abstractmethod
    def generate(self, question: str, chunks: List[Tuple[str, str]]) -> str:
        ...


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024, timeout: int = 30):
        if not api_key:
            raise DownstreamLLMError(
                "ANTHROPIC_API_KEY is not set. /query requires an LLM API key; "
                "/ingest does not."
            )
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, question: str, chunks: List[Tuple[str, str]]) -> str:
        user_prompt = build_user_prompt(question, chunks)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:  # noqa: BLE001 - any API failure becomes a clean 502
            raise DownstreamLLMError(f"LLM call failed: {e}") from e

        text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "\n".join(text_blocks).strip()


class GeminiLLMClient(LLMClient):
    """Google Gemini has a genuine free tier (no card required to start),
    which is the practical option if this project needs to be runnable by
    other people (recruiters, friends) at zero cost to you. Swap to it by
    setting LLM_PROVIDER=gemini and GEMINI_API_KEY in .env -- nothing else
    in the pipeline (retrieval, reranking, grounding checks) changes,
    because generation only ever talks to the LLMClient interface."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        if not api_key:
            raise DownstreamLLMError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey"
            )
        from google import genai  # lazy import
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, question: str, chunks: List[Tuple[str, str]]) -> str:
        user_prompt = build_user_prompt(question, chunks)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=self._max_tokens,
                ),
            )
        except Exception as e:  # noqa: BLE001 - any API failure becomes a clean 502
            raise DownstreamLLMError(f"Gemini call failed: {e}") from e

        return (response.text or "").strip()


class FakeLLMClient(LLMClient):
    """Deterministic stand-in for tests -- echoes back a canned answer that
    cites the first chunk, so grounding-check tests can be exercised without
    hitting a real API."""

    def __init__(self, canned_answer: str | None = None):
        self._canned_answer = canned_answer

    def generate(self, question: str, chunks: List[Tuple[str, str]]) -> str:
        if self._canned_answer is not None:
            return self._canned_answer
        if not chunks:
            return "I don't have enough information in the provided documents to answer this."
        tag, text = chunks[0]
        return f"Based on the documents, {text[:80]} [{tag}]"
