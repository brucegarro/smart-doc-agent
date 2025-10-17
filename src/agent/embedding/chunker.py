"""Utilities for converting UDR documents into retrieval chunks."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from agent.ingestion.udr import (
    BoundingBox,
    Figure,
    Reference,
    Section,
    Table,
    UnifiedDocumentRepresentation,
)

from .payloads import ChunkPayload
from .text_chunks import estimate_token_count, generate_text_chunks

logger = logging.getLogger(__name__)

_TABLE_PREVIEW_ROWS = 6


def build_chunks(udr: UnifiedDocumentRepresentation) -> List[ChunkPayload]:
    """Generate retrieval chunks from the unified document representation."""
    block_section_map = _build_block_section_map(udr.sections)
    page_section_map = _build_page_section_map(udr.sections)
    chunks: List[ChunkPayload] = []

    chunks.extend(generate_text_chunks(udr, block_section_map))
    chunks.extend(_generate_table_chunks(udr, page_section_map))
    chunks.extend(_generate_figure_chunks(udr, page_section_map))
    chunks.extend(_generate_reference_chunks(udr))

    _assign_chunk_indices(chunks)

    logger.debug("Generated %s chunk(s)", len(chunks))
    return chunks


def _generate_table_chunks(
    udr: UnifiedDocumentRepresentation,
    page_section_map: Dict[int, str],
) -> List[ChunkPayload]:
    chunks: List[ChunkPayload] = []
    for table in udr.tables:
        chunk_content = _build_table_content(table)
        if not chunk_content:
            continue

        section_title = _section_for_page(table.page, page_section_map)
        chunks.append(
            ChunkPayload(
                content=chunk_content,
                content_type="table",
                chunk_index=0,
                page_number=table.page,
                section_title=section_title,
                token_count=estimate_token_count(chunk_content),
                char_count=len(chunk_content),
                metadata=_table_metadata(table),
            )
        )
    return chunks


def _generate_figure_chunks(
    udr: UnifiedDocumentRepresentation,
    page_section_map: Dict[int, str],
) -> List[ChunkPayload]:
    chunks: List[ChunkPayload] = []
    for figure in udr.figures:
        chunk_content = _build_figure_content(figure)
        if not chunk_content:
            continue

        section_title = _section_for_page(figure.page, page_section_map)
        chunks.append(
            ChunkPayload(
                content=chunk_content,
                content_type="figure",
                chunk_index=0,
                page_number=figure.page,
                section_title=section_title,
                token_count=estimate_token_count(chunk_content),
                char_count=len(chunk_content),
                metadata=_figure_metadata(figure),
            )
        )
    return chunks


def _generate_reference_chunks(udr: UnifiedDocumentRepresentation) -> List[ChunkPayload]:
    chunks: List[ChunkPayload] = []
    for reference in udr.references:
        content = (reference.text or "").strip()
        if not content:
            continue

        chunks.append(
            ChunkPayload(
                content=content,
                content_type="reference",
                chunk_index=0,
                page_number=reference.page,
                section_title="References",
                token_count=estimate_token_count(content),
                char_count=len(content),
                metadata={
                    "source": "reference_list",
                    "reference_id": reference.reference_id,
                },
            )
        )
    return chunks


def _assign_chunk_indices(chunks: Sequence[ChunkPayload]) -> None:
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index


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
