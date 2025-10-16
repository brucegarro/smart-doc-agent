"""Embedding and chunking utilities for smart-doc-agent."""

from .embedder import embedding_client
from .chunker import build_chunks, ChunkPayload

__all__ = [
    "embedding_client",
    "build_chunks",
    "ChunkPayload",
]
