"""Background ingestion tasks executed by Redis workers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from agent.config import settings
from agent.ingestion.pdf_parser import ParseOptions
from agent.ingestion.processor import processor

logger = logging.getLogger(__name__)


def _deserialize_parse_options(payload: Optional[dict[str, bool]]) -> Optional[ParseOptions]:
    if not payload:
        return None
    try:
        return ParseOptions(**payload)
    except TypeError as exc:  # pragma: no cover - defensive
        logger.warning("Invalid parse options payload %s: %s", payload, exc)
        return None


def process_pdf_task(
    staged_pdf_path: str,
    source_name: Optional[str] = None,
    parse_options_payload: Optional[dict[str, bool]] = None,
) -> dict[str, Optional[str]]:
    """Execute the ingestion pipeline for a single staged PDF file."""

    pdf_path = Path(staged_pdf_path)
    parse_options = _deserialize_parse_options(parse_options_payload)
    queue_dir = Path(settings.ingest_queue_dir)

    try:
        doc_id = processor.process_pdf(pdf_path, source_name=source_name, parse_options=parse_options)
        return {"status": "succeeded", "doc_id": doc_id}
    except ValueError as exc:
        logger.info("Skipping PDF %s: %s", pdf_path.name, exc)
        return {"status": "skipped", "message": str(exc)}
    except Exception as exc:  # pragma: no cover - ingestion failure surfaces to worker logs
        logger.exception("Ingestion failed for %s", pdf_path)
        raise
    finally:
        try:
            if pdf_path.exists() and pdf_path.is_relative_to(queue_dir):
                pdf_path.unlink()
        except Exception as cleanup_error:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to remove staged file %s: %s", pdf_path, cleanup_error)
