"""Utilities for converting UDR documents into retrieval chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from agent.ingestion.udr import (
    BoundingBox,
    Figure,
    Reference,
    Section,
    Table,
    UnifiedDocumentRepresentation,
)

logger = logging.getLogger(__name__)

# Chunking heuristics
_MAX_TOKENS_PER_CHUNK = 240
_MIN_CHARS_PER_CHUNK = 30
_TABLE_PREVIEW_ROWS = 6


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


def build_chunks(udr: UnifiedDocumentRepresentation) -> List[ChunkPayload]:
    """Generate retrieval chunks from the unified document representation."""
    chunk_index = 0
    chunks: List[ChunkPayload] = []

    block_section_map = _build_block_section_map(udr.sections)
    page_section_map = _build_page_section_map(udr.sections)

    buffer_text: List[str] = []
    buffer_blocks: List[Dict[str, Any]] = []
    buffer_pages: List[int] = []
    buffer_sections: List[Optional[str]] = []
    buffer_tokens = 0

    def flush_buffer(force: bool = False) -> None:
        nonlocal chunk_index, buffer_text, buffer_blocks, buffer_pages
        nonlocal buffer_sections, buffer_tokens

        if not buffer_text:
            return

        content = "\n\n".join(part for part in buffer_text if part).strip()
        if not content:
            buffer_text.clear()
            buffer_blocks.clear()
            buffer_pages.clear()
            buffer_sections.clear()
            buffer_tokens = 0
            return

        if not force and len(content) < _MIN_CHARS_PER_CHUNK and len(buffer_blocks) > 1:
            # Try to keep small fragments attached to future blocks
            return

        chunk_section = next((title for title in buffer_sections if title), None)
        page_number = min(buffer_pages) if buffer_pages else None
        token_count = _estimate_token_count(content)

        chunk_metadata: Dict[str, Any] = {
            "source": "text_blocks",
            "blocks": buffer_blocks.copy(),
        }

        chunks.append(
            ChunkPayload(
                content=content,
                content_type="text",
                chunk_index=chunk_index,
                page_number=page_number,
                section_title=chunk_section,
                token_count=token_count,
                char_count=len(content),
                metadata=chunk_metadata,
            )
        )

        chunk_index += 1
        buffer_text.clear()
        buffer_blocks.clear()
        buffer_pages.clear()
        buffer_sections.clear()
        buffer_tokens = 0

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

            if block.block_type == "heading":
                flush_buffer()

            block_tokens = _estimate_token_count(block_text)
            if buffer_tokens and buffer_tokens + block_tokens > _MAX_TOKENS_PER_CHUNK:
                flush_buffer()

            buffer_text.append(block_text)
            buffer_blocks.append(block_info)
            buffer_pages.append(page.page_num)
            buffer_sections.append(block_section_map.get(block.block_id))
            buffer_tokens += block_tokens

    flush_buffer(force=True)

    # Tables
    for table in udr.tables:
        chunk_content = _build_table_content(table)
        if not chunk_content:
            continue

        section_title = _section_for_page(
            table.page,
            page_section_map,
        )

        chunks.append(
            ChunkPayload(
                content=chunk_content,
                content_type="table",
                chunk_index=chunk_index,
                page_number=table.page,
                section_title=section_title,
                token_count=_estimate_token_count(chunk_content),
                char_count=len(chunk_content),
                metadata=_table_metadata(table),
            )
        )
        chunk_index += 1

    # Figures
    for figure in udr.figures:
        chunk_content = _build_figure_content(figure)
        if not chunk_content:
            continue

        section_title = _section_for_page(
            figure.page,
            page_section_map,
        )

        chunks.append(
            ChunkPayload(
                content=chunk_content,
                content_type="figure",
                chunk_index=chunk_index,
                page_number=figure.page,
                section_title=section_title,
                token_count=_estimate_token_count(chunk_content),
                char_count=len(chunk_content),
                metadata=_figure_metadata(figure),
            )
        )
        chunk_index += 1

    # References
    for reference in udr.references:
        content = (reference.text or "").strip()
        if not content:
            continue

        chunks.append(
            ChunkPayload(
                content=content,
                content_type="reference",
                chunk_index=chunk_index,
                page_number=reference.page,
                section_title="References",
                token_count=_estimate_token_count(content),
                char_count=len(content),
                metadata={
                    "source": "reference_list",
                    "reference_id": reference.reference_id,
                },
            )
        )
        chunk_index += 1

    logger.debug("Generated %s chunk(s)", len(chunks))
    return chunks


def _build_block_section_map(sections: Sequence[Section]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for section in sections:
        for block_id in section.block_ids:
            if block_id not in mapping:
                mapping[block_id] = section.title
    return mapping


def _build_page_section_map(sections: Sequence[Section]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for section in sections:
        if section.page_start is None or section.page_end is None:
            continue
        for page in range(section.page_start, section.page_end + 1):
            mapping.setdefault(page, section.title)
    return mapping


def _section_for_page(page: Optional[int], page_section_map: Dict[int, str]) -> Optional[str]:
    if page is None:
        return None
    return page_section_map.get(page)


def _build_table_content(table: Table) -> str:
    parts: List[str] = []
    if table.caption:
        parts.append(table.caption.strip())

    if table.text:
        parts.append(table.text.strip())

    content = "\n".join(part for part in parts if part).strip()
    return content


def _table_metadata(table: Table) -> Dict[str, Any]:
    preview = table.data[:_TABLE_PREVIEW_ROWS] if table.data else []
    bbox_dict = _bbox_to_dict(table.bbox)

    return {
        "source": "table",
        "table_id": table.table_id,
        "page": table.page,
        "caption": table.caption,
        "row_count": len(table.data) if table.data else 0,
        "col_count": max((len(row) for row in table.data), default=0) if table.data else 0,
        "header_rows": table.header_rows,
        "header_cols": table.header_cols,
        "confidence": table.confidence,
        "bbox": bbox_dict,
        "artifacts": table.artifacts,
        "data_preview": preview,
    }


def _build_figure_content(figure: Figure) -> str:
    parts: List[str] = []
    if figure.caption:
        parts.append(figure.caption.strip())

    description = figure.artifacts.get("qwen_vl_description") if figure.artifacts else None
    if description:
        parts.append(description.strip())

    content = "\n".join(part for part in parts if part).strip()
    return content


def _figure_metadata(figure: Figure) -> Dict[str, Any]:
    bbox_dict = _bbox_to_dict(figure.bbox)

    return {
        "source": "figure",
        "figure_id": figure.figure_id,
        "page": figure.page,
        "caption": figure.caption,
        "subtype": figure.subtype,
        "confidence": figure.confidence,
        "bbox": bbox_dict,
        "artifacts": figure.artifacts,
    }


def _bbox_to_dict(bbox: Optional[BoundingBox]) -> Optional[Dict[str, float]]:
    if not bbox:
        return None
    return {
        "x0": bbox.x0,
        "y0": bbox.y0,
        "x1": bbox.x1,
        "y1": bbox.y1,
        "page": bbox.page,
    }


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    # Lightweight heuristic: approximate tokens as whitespace-delimited words
    return max(1, len(text.split()))
