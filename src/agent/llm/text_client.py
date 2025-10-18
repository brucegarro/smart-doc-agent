"""Ollama-backed text generation client."""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any, List, Optional

from agent.config import settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from ollama import Client as OllamaClient, ResponseError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OllamaClient = None  # type: ignore
    ResponseError = None  # type: ignore


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
        messages: List[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = {
            "temperature": temperature,
            "num_predict": max(1, max_tokens),
        }

        if "num_ctx" not in options:
            try:
                options["num_ctx"] = max(1, int(os.getenv("OLLAMA_CONTEXT_LENGTH", "4096")))
            except ValueError:
                options["num_ctx"] = 4096

        response: Optional[dict[str, Any]] = None
        last_error: Optional[Exception] = None

        for attempt in range(2):
            client = self._get_client()
            if client is None:
                break
            try:
                response = client.chat(  # type: ignore[operator]
                    model=settings.text_llm_model,
                    messages=messages,
                    options=options,
                    stream=False,
                )
                break
            except Exception as exc:  # pragma: no cover - network dependency
                last_error = exc
                if ResponseError is not None and isinstance(exc, ResponseError):
                    status_code = getattr(exc, "status_code", None)
                    logger.error(
                        "Text LLM request failed (status %s, attempt %s): %s",
                        status_code,
                        attempt + 1,
                        exc,
                    )
                    if attempt == 0 and status_code is not None and status_code >= 500:
                        self._reset_client()
                        time.sleep(0.5)
                        continue
                else:
                    logger.error("Text LLM request failed (attempt %s): %s", attempt + 1, exc)
                raise LLMGenerationError(str(exc)) from exc

        if response is None:
            if last_error is not None:
                raise LLMGenerationError(str(last_error)) from last_error
            raise LLMGenerationError("text llm client unavailable")

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

    def _reset_client(self) -> None:
        with self._lock:
            if self._client is not None:
                close = getattr(self._client, "close", None)
                if callable(close):  # pragma: no cover - optional cleanup
                    try:
                        close()
                    except Exception:
                        logger.debug("Error while closing Ollama client", exc_info=True)
            self._client = None
            self._unavailable = False


text_llm_client = TextLLMClient()
