"""Vector similarity search helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from agent.db import get_db_connection
from agent.embedding.embedder import embedding_client
from agent.retrieval.query_parser import QueryContext, build_query_context

_whitespace_re = re.compile(r"\s+")


DOCUMENT_FILTER_MIN_SCORE = 0.65


@dataclass
class ChunkResult:
    """Represents a chunk returned from similarity search."""

    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: int
    content_type: str
    content: str
    snippet: str
    fingerprint: str
    cosine_similarity: float


def _vector_literal(values: Sequence[float]) -> str:
    # Build pgvector literal from raw floats.
    return "[" + ",".join(repr(value) for value in values) + "]"


def _clean_snippet(snippet: Optional[str], *, max_length: int = 200) -> str:
    if not snippet:
        return ""
    collapsed = _whitespace_re.sub(" ", snippet).strip()
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[: max_length - 3].rstrip() + "..."


def _fingerprint_text(text: str) -> str:
    """Build a stable fingerprint for chunk matching across runs."""
    normalized = _whitespace_re.sub(" ", (text or "").strip()).lower()
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:24]


def search_chunks(
    query: str,
    *,
    limit: int = 5,
    content_types: Optional[Iterable[str]] = ("text",),
    query_context: Optional[QueryContext] = None,
) -> List[ChunkResult]:
    """Search for similar chunks given free-text query."""

    if not query.strip():
        raise ValueError("Query cannot be empty")
    if limit <= 0:
        raise ValueError("Limit must be positive")

    context = query_context or build_query_context(query)

    embeddings = embedding_client.embed([query])
    if not embeddings or not embeddings[0]:
        raise ValueError("Failed to generate embedding for query")

    vector = embeddings[0]
    vector_value = _vector_literal(vector)

    filter_clauses: List[str] = []
    params: List[object] = [vector_value]

    document_ids = context.document_ids(min_score=DOCUMENT_FILTER_MIN_SCORE)
    if document_ids:
        filter_clauses.append("c.document_id = ANY(%s)")
        params.append(document_ids)

    if content_types:
        types_list = list(content_types)
        filter_clauses.append("c.content_type = ANY(%s)")
        params.append(types_list)

    params.append(limit)

    filter_clause = ""
    if filter_clauses:
        filter_clause = "WHERE " + " AND ".join(filter_clauses)

    query_sql = f"""
        WITH query_vec AS (
            SELECT %s::vector AS vec
        )
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.chunk_index,
            c.page_number,
            c.content_type,
            c.content,
            1 - (c.embedding <=> query_vec.vec) AS cosine_similarity
        FROM chunks AS c, query_vec
        {filter_clause}
        ORDER BY c.embedding <=> query_vec.vec
        LIMIT %s;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()

    results: List[ChunkResult] = []
    for row in rows:
        content = row["content"] or ""
        results.append(
            ChunkResult(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                chunk_index=row["chunk_index"],
                page_number=row["page_number"],
                content_type=row["content_type"],
                content=content,
                snippet=_clean_snippet(content),
                fingerprint=_fingerprint_text(content),
                cosine_similarity=row["cosine_similarity"],
            )
        )

    return results
