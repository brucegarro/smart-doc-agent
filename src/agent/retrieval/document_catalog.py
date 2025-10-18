"""Document metadata catalog helpers for retrieval workflows."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Tuple

from agent.db import get_db_connection

logger = logging.getLogger(__name__)

_non_alnum_re = re.compile(r"[^a-z0-9]+")
_parenthetical_re = re.compile(r"\([^)]*\)")
_whitespace_re = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentMetadata:
    """Lightweight view of a document record and its search aliases."""

    document_id: str
    title: str
    filename: str
    authors: Tuple[str, ...]
    aliases: Tuple[str, ...]


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    ascii_bytes = value.encode("ascii", "ignore")
    ascii_value = ascii_bytes.decode("ascii")
    lowered = ascii_value.lower()
    cleaned = _non_alnum_re.sub(" ", lowered)
    normalized = _whitespace_re.sub(" ", cleaned).strip()
    return normalized


def _dedupe_aliases(candidates: Sequence[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        normalized = _normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _candidate_aliases(
    document_id: str,
    title: str,
    filename: str,
    authors: Sequence[str],
) -> Tuple[str, ...]:
    aliases: list[str] = []

    if title:
        aliases.append(title)
        if ":" in title:
            aliases.append(title.split(":", 1)[0])
        if " - " in title:
            aliases.append(title.split(" - ", 1)[0])
        stripped = _parenthetical_re.sub("", title)
        if stripped != title:
            aliases.append(stripped)
        words = title.split()
        if len(words) >= 3:
            aliases.append(" ".join(words[:4]))

    if filename:
        stem = filename
        if stem.lower().endswith(".pdf"):
            stem = stem[:-4]
        stem = stem.replace("_", " ")
        aliases.append(stem)

    if document_id:
        aliases.append(document_id.replace("_", " "))

    if authors:
        aliases.append(" ".join(authors[:2]))
        aliases.append(authors[0])

    return _dedupe_aliases(aliases)


@lru_cache(maxsize=1)
def _load_document_catalog() -> Tuple[DocumentMetadata, ...]:
    logger.debug("Loading document catalog from database")
    query = """
        SELECT
            id,
            COALESCE(title, '') AS title,
            COALESCE(filename, '') AS filename,
            COALESCE(authors, ARRAY[]::text[]) AS authors
        FROM documents
        WHERE processing_status IN ('ingested', 'indexed')
        ORDER BY ingested_at DESC NULLS LAST
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(query)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to load document catalog: %s", exc)
                raise
            rows = cur.fetchall()

    documents: list[DocumentMetadata] = []
    for row in rows:
        if hasattr(row, "get"):
            authors_field = row.get("authors")
            doc_id = str(row.get("id"))
            title = str(row.get("title", ""))
            filename = str(row.get("filename", ""))
        else:
            authors_field = row[3] if len(row) > 3 else None
            doc_id = str(row[0])
            title = str(row[1])
            filename = str(row[2])

        if isinstance(authors_field, (list, tuple)):
            authors = tuple(str(item) for item in authors_field if item)
        else:
            authors = tuple()

        aliases = _candidate_aliases(doc_id, title, filename, authors)
        documents.append(
            DocumentMetadata(
                document_id=doc_id,
                title=title,
                filename=filename,
                authors=authors,
                aliases=aliases,
            )
        )
    return tuple(documents)


def list_document_metadata() -> Tuple[DocumentMetadata, ...]:
    """Return the cached catalog of document metadata."""

    return _load_document_catalog()


def invalidate_document_catalog_cache() -> None:
    """Clear the cached catalog so new ingestions get picked up."""

    _load_document_catalog.cache_clear()
    logger.debug("Document catalog cache cleared")
