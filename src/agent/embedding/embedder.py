"""Sentence-transformer embedding client with lazy initialization."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Iterable, List, Optional

from agent.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Thin wrapper around sentence-transformers for shared usage."""

    def __init__(self) -> None:
        self.model_name = settings.embedder_model
        self.device = settings.embedder_device
        self.normalize = settings.embedder_normalize
        self._lock = Lock()
        self._model = None
        self._available = True

    @property
    def available(self) -> bool:
        return self._available and self._model is not False

    def _load_model(self):  # type: ignore[override]
        if self._model is False:
            return None

        if self._model is not None:
            return self._model

        with self._lock:
            if self._model not in (None, False):
                return self._model

            try:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    "Loading embedding model '%s' on device '%s'",
                    self.model_name,
                    self.device,
                )
                model = SentenceTransformer(self.model_name, device=self.device)
                self._model = model
            except Exception as exc:  # pragma: no cover - optional dependency
                fallback_attempted = False
                if self.device != "cpu":
                    logger.warning(
                        "Embedder device '%s' unavailable (%s); retrying on CPU",
                        self.device,
                        exc,
                    )
                    try:
                        from sentence_transformers import SentenceTransformer

                        self.device = "cpu"
                        model = SentenceTransformer(self.model_name, device=self.device)
                        self._model = model
                        fallback_attempted = True
                    except Exception as cpu_exc:  # pragma: no cover - optional dependency
                        logger.warning(
                            "Failed to load embedder '%s' on CPU: %s",
                            self.model_name,
                            cpu_exc,
                        )
                        self._model = False
                        self._available = False
                        return None

                if not fallback_attempted:
                    logger.warning("Failed to load embedder '%s': %s", self.model_name, exc)
                    self._model = False
                    self._available = False
                    return None

        return self._model

    def embed(self, texts: Iterable[str], batch_size: int = 32) -> Optional[List[List[float]]]:
        all_texts = list(texts)
        if not all_texts:
            return []

        # Normalize casing so semantically identical spans embed identically.
        text_list = [text.lower() for text in all_texts if text]
        if not text_list:
            return []

        model = self._load_model()
        if model is None:
            return None

        try:
            embeddings = model.encode(  # type: ignore[attr-defined]
                text_list,
                batch_size=batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.error("Embedding model failed to encode batch: %s", exc)
            return None

        # model.encode returns numpy.ndarray when convert_to_numpy=True
        if hasattr(embeddings, "tolist"):
            embeddings_list: List[List[float]] = embeddings.tolist()
        else:
            embeddings_list = [list(vec) for vec in embeddings]

        # When some texts were empty, we skipped them; pad results back to original order
        result: List[List[float]] = []
        iterator = iter(embeddings_list)
        for original in all_texts:
            if not original:
                result.append([])
                continue
            result.append(next(iterator))

        return result


embedding_client = EmbeddingClient()
