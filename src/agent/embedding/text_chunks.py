"""Utilities for generating text-based retrieval chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.ingestion.udr import UnifiedDocumentRepresentation

from .payloads import ChunkPayload

logger = logging.getLogger(__name__)

_MAX_TOKENS_PER_CHUNK = 240
_TOKEN_OVERLAP = 40
_MIN_CHARS_PER_CHUNK = 30


@dataclass
class ChunkBuffer:
    """Mutable accumulator for text chunks."""

    texts: List[str] = field(default_factory=list)
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    sections: List[Optional[str]] = field(default_factory=list)
    tokens_per_block: List[int] = field(default_factory=list)
    total_tokens: int = 0

    def add(
        self,
        text: str,
        block: Dict[str, Any],
        page_number: int,
        section_title: Optional[str],
        token_count: int,
    ) -> None:
        if not text:
            return
        self.texts.append(text)
        self.blocks.append(block)
        self.pages.append(page_number)
        self.sections.append(section_title)
        self.tokens_per_block.append(token_count)
        self.total_tokens += token_count

    def clear(self) -> None:
        self.texts.clear()
        self.blocks.clear()
        self.pages.clear()
        self.sections.clear()
        self.tokens_per_block.clear()
        self.total_tokens = 0

    def extend(self, other: "ChunkBuffer") -> None:
        self.texts.extend(other.texts)
        self.blocks.extend(other.blocks)
        self.pages.extend(other.pages)
        self.sections.extend(other.sections)
        self.tokens_per_block.extend(other.tokens_per_block)
        self.total_tokens += other.total_tokens

    def is_empty(self) -> bool:
        return not self.texts

    def block_count(self) -> int:
        return len(self.blocks)


def generate_text_chunks(
    udr: UnifiedDocumentRepresentation,
    block_section_map: Dict[str, str],
) -> List[ChunkPayload]:
    buffer = ChunkBuffer()
    chunks: List[ChunkPayload] = []

    for page in udr.pages:
        for block in page.blocks:
            block_text = (block.text or "").strip()
            if not block_text:
                continue

            block_info = {
                "block_id": block.block_id,
                "block_type": block.block_type,
                "page_number": page.page_num,
                "reading_order": block.reading_order,
            }

            block_section = block_section_map.get(block.block_id)

            if block.block_type == "heading":
                _append_if_payload(chunks, _flush_buffer(buffer, force=False, keep_overlap=False))

            if _section_changed(buffer, block_section):
                _append_if_payload(chunks, _flush_buffer(buffer, force=False, keep_overlap=False))

            block_tokens = estimate_token_count(block_text)
            if buffer.total_tokens and buffer.total_tokens + block_tokens > _MAX_TOKENS_PER_CHUNK:
                _append_if_payload(chunks, _flush_buffer(buffer, force=False, keep_overlap=True))

            buffer.add(
                text=block_text,
                block=block_info,
                page_number=page.page_num,
                section_title=block_section,
                token_count=block_tokens,
            )

    _append_if_payload(chunks, _flush_buffer(buffer, force=True, keep_overlap=False))
    return chunks


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _append_if_payload(chunks: List[ChunkPayload], payload: Optional[ChunkPayload]) -> None:
    if payload is not None:
        chunks.append(payload)


def _section_changed(buffer: ChunkBuffer, candidate: Optional[str]) -> bool:
    if buffer.is_empty() or not candidate:
        return False
    for section in reversed(buffer.sections):
        if section:
            return section != candidate
    return False


def _flush_buffer(
    buffer: ChunkBuffer,
    *,
    force: bool,
    keep_overlap: bool,
) -> Optional[ChunkPayload]:
    if buffer.is_empty():
        return None

    content = _buffer_content(buffer)
    if content is None:
        return None

    if _should_defer_flush(content, buffer, force):
        return None

    overlap_buffer = _prepare_overlap(buffer, keep_overlap)
    payload = _build_text_payload(buffer, content)

    buffer.clear()
    if overlap_buffer:
        buffer.extend(overlap_buffer)

    return payload


def _buffer_content(buffer: ChunkBuffer) -> Optional[str]:
    content = "\n\n".join(part for part in buffer.texts if part).strip()
    if content:
        return content
    buffer.clear()
    return None


def _should_defer_flush(content: str, buffer: ChunkBuffer, force: bool) -> bool:
    if force:
        return False
    return len(content) < _MIN_CHARS_PER_CHUNK and buffer.block_count() > 1


def _prepare_overlap(buffer: ChunkBuffer, keep_overlap: bool) -> Optional[ChunkBuffer]:
    if not keep_overlap or _TOKEN_OVERLAP <= 0:
        return None
    return _capture_overlap(buffer)


def _build_text_payload(buffer: ChunkBuffer, content: str) -> ChunkPayload:
    chunk_section = next((title for title in buffer.sections if title), None)
    page_number = min(buffer.pages) if buffer.pages else None
    metadata = {
        "source": "text_blocks",
        "blocks": [dict(block) for block in buffer.blocks],
    }
    return ChunkPayload(
        content=content,
        content_type="text",
        chunk_index=0,
        page_number=page_number,
        section_title=chunk_section,
        token_count=estimate_token_count(content),
        char_count=len(content),
        metadata=metadata,
    )


def _capture_overlap(buffer: ChunkBuffer) -> Optional[ChunkBuffer]:
    overlap = ChunkBuffer()
    accumulated_tokens = 0
    candidates: List[Tuple[str, Dict[str, Any], int, Optional[str], int]] = []

    for index in range(len(buffer.texts) - 1, -1, -1):
        candidates.append(
            (
                buffer.texts[index],
                buffer.blocks[index],
                buffer.pages[index],
                buffer.sections[index],
                buffer.tokens_per_block[index],
            )
        )
        accumulated_tokens += buffer.tokens_per_block[index]
        if accumulated_tokens >= _TOKEN_OVERLAP:
            break

    for text, block, page, section, token_count in reversed(candidates):
        overlap.add(text, dict(block), page, section, token_count)

    return overlap if not overlap.is_empty() else None
