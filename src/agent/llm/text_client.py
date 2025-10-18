"""Ollama-backed text generation client."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, List, Optional

from agent.config import settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from ollama import Client as OllamaClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OllamaClient = None  # type: ignore


class LLMGenerationError(RuntimeError):
    """Raised when the text generation client cannot return an answer."""


class TextLLMClient:
    """Thread-safe helper around the Ollama client for text-only models."""

    def __init__(self) -> None:
        self._client: Optional[object] = None
        self._lock = Lock()
        self._unavailable = False

    @property
    def available(self) -> bool:
        return not self._unavailable

    def ensure_client(self) -> bool:
        """Attempt to initialize the underlying Ollama client if needed."""
        return self._get_client() is not None

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> str:
        client = self._get_client()
        if client is None:
            raise LLMGenerationError("text llm client unavailable")

        messages: List[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = {
            "temperature": temperature,
            "num_predict": max(1, max_tokens),
        }

        try:
            response = client.chat(  # type: ignore[operator]
                model=settings.text_llm_model,
                messages=messages,
                options=options,
                stream=False,
            )
        except Exception as exc:  # pragma: no cover - network dependency
            logger.error("Text LLM request failed: %s", exc)
            raise LLMGenerationError(str(exc)) from exc

        message = response.get("message") if isinstance(response, dict) else None
        if not message:
            raise LLMGenerationError("text llm returned empty message")

        content = message.get("content")
        if not content:
            raise LLMGenerationError("text llm returned empty content")

        return str(content).strip()

    def _get_client(self) -> Optional[object]:
        if self._unavailable:
            return None
        if self._client is not None:
            return self._client
        if OllamaClient is None:
            logger.warning("Ollama client not installed; text generation disabled")
            self._unavailable = True
            return None

        with self._lock:
            if self._client is not None or self._unavailable:
                return self._client
            try:
                self._client = OllamaClient(host=settings.ollama_base)
                logger.debug("Initialized Ollama client for text generation")
            except Exception as exc:  # pragma: no cover - network dependency
                logger.warning("Failed to connect to Ollama at %s: %s", settings.ollama_base, exc)
                self._client = None
                self._unavailable = True
        return self._client


text_llm_client = TextLLMClient()
