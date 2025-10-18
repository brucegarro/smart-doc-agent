"""Natural language query parsing helpers for retrieval."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Optional, Sequence, Tuple

from agent.retrieval.document_catalog import DocumentMetadata, list_document_metadata

logger = logging.getLogger(__name__)

_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")

MIN_CANDIDATE_SCORE = 0.5
MAX_DOCUMENT_MATCHES = 6


@dataclass(frozen=True)
class DocumentCandidate:
    document_id: str
    title: str
    score: float
    alias: str


@dataclass(frozen=True)
class QueryContext:
    raw_query: str
    normalized_query: str
    candidate_phrases: Tuple[str, ...]
    document_candidates: Tuple[DocumentCandidate, ...]

    def top_documents(self, min_score: float = 0.0, limit: Optional[int] = None) -> List[DocumentCandidate]:
        matches = [candidate for candidate in self.document_candidates if candidate.score >= min_score]
        if limit is not None:
            return matches[:limit]
        return matches

    def document_ids(self, min_score: float = 0.0, limit: Optional[int] = None) -> List[str]:
        return [candidate.document_id for candidate in self.top_documents(min_score=min_score, limit=limit)]

    def is_comparison(self, min_score: float = MIN_CANDIDATE_SCORE) -> bool:
        return len(self.document_ids(min_score=min_score, limit=None)) >= 2


def build_query_context(
    query: str,
    *,
    catalog: Optional[Sequence[DocumentMetadata]] = None,
    min_score: float = MIN_CANDIDATE_SCORE,
    max_matches: int = MAX_DOCUMENT_MATCHES,
) -> QueryContext:
    if not query:
        raise ValueError("Query cannot be empty")

    normalized = _normalize(query)
    phrases = _collect_candidate_phrases(query)
    catalog_items = tuple(catalog) if catalog is not None else list_document_metadata()
    candidates = _score_document_matches(normalized, phrases, catalog_items, min_score, max_matches)

    return QueryContext(
        raw_query=query,
        normalized_query=normalized,
        candidate_phrases=phrases,
        document_candidates=candidates,
    )


def _normalize(text: str) -> str:
    ascii_bytes = text.encode("ascii", "ignore")
    ascii_text = ascii_bytes.decode("ascii")
    lowered = ascii_text.lower()
    collapsed = _WHITESPACE_RE.sub(" ", lowered)
    return collapsed.strip()


def _collect_candidate_phrases(query: str) -> Tuple[str, ...]:
    phrases: List[str] = []

    for match in _QUOTED_RE.finditer(query):
        phrase = match.group(1) or match.group(2)
        normalized = _normalize(phrase)
        if normalized:
            phrases.append(normalized)

    normalized_query = _normalize(query)
    if normalized_query:
        phrases.append(normalized_query)

    tokens = _TOKEN_RE.findall(normalized_query)
    max_window = min(len(tokens), 6)
    for window in range(1, max_window + 1):
        for start in range(0, len(tokens) - window + 1):
            phrase = " ".join(tokens[start : start + window])
            phrases.append(phrase)

    deduped: List[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase and phrase not in seen:
            deduped.append(phrase)
            seen.add(phrase)
    return tuple(deduped)


def _score_document_matches(
    normalized_query: str,
    phrases: Sequence[str],
    catalog: Sequence[DocumentMetadata],
    min_score: float,
    max_matches: int,
) -> Tuple[DocumentCandidate, ...]:
    scored: List[DocumentCandidate] = []

    for metadata in catalog:
        score, alias = _score_single_document(metadata, normalized_query, phrases)
        if score < min_score:
            continue
        title = metadata.title or metadata.filename or metadata.document_id
        scored.append(DocumentCandidate(metadata.document_id, title, score, alias))

    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    if max_matches > 0:
        scored = scored[:max_matches]
    return tuple(scored)


def _score_single_document(
    metadata: DocumentMetadata,
    normalized_query: str,
    phrases: Sequence[str],
) -> Tuple[float, str]:
    best_score = 0.0
    best_alias = ""

    for alias in metadata.aliases:
        score = _alias_score(alias, normalized_query, phrases)
        if score > best_score:
            best_score = score
            best_alias = alias
        if best_score >= 1.0:
            break

    return best_score, best_alias


def _alias_score(alias: str, normalized_query: str, phrases: Sequence[str]) -> float:
    if not alias:
        return 0.0

    if alias in normalized_query:
        return 1.0

    best = SequenceMatcher(None, normalized_query, alias).ratio()
    for phrase in phrases:
        candidate_score = SequenceMatcher(None, phrase, alias).ratio()
        if candidate_score > best:
            best = candidate_score
    return best


def describe_candidates(candidates: Iterable[DocumentCandidate]) -> List[str]:
    """Return lightweight summaries for logging or debug output."""

    return [f"{candidate.title} ({candidate.score:.2f})" for candidate in candidates]
