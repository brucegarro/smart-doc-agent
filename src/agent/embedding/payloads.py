"""Dataclasses shared across embedding chunk generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkPayload:
    """Represents a chunk that will be stored in the chunks table."""

    content: str
    content_type: str
    chunk_index: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    token_count: int = 0
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
