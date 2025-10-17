"""Ingestion scenario execution for the evaluator."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Dict, Any

from agent.db import get_db_connection
from agent.ingestion.processor import DocumentProcessor

from agent.evaluator.models import IngestionRecord, ScenarioResult

if TYPE_CHECKING:  # pragma: no cover - type check helper
    from agent.evaluator.harness import EvaluatorConfig

logger = logging.getLogger(__name__)


class IngestionScenarioRunner:
    """Process fixture PDFs and produce ingestion metrics."""

    def __init__(self, config: "EvaluatorConfig", processor: DocumentProcessor) -> None:
        self._config = config
        self._processor = processor

    def run(self) -> Tuple[ScenarioResult, List[str], List[float]]:
        docs = self._ingestion_fixtures()
        if not docs:
            logger.warning("No PDF fixtures found in %s; skipping ingestion", self._config.fixtures_docs)
            return (
                ScenarioResult(
                    name="ingestion",
                    status="skipped",
                    metrics={"documents": 0},
                    details=["no fixtures"],
                ),
                [],
                [],
            )

        logger.info("Ingestion scenario starting with %s document(s)", len(docs))
        records = [self._ingest_document(path) for path in docs]

        doc_ids = [record.doc_id for record in records if record.doc_id]
        error_records = [record for record in records if record.error]
        duplicate_records = [record for record in records if record.duplicate]
        timings = [record.elapsed for record in records]

        metrics = self._ingestion_metrics(records, doc_ids, timings)
        details = self._ingestion_details(doc_ids, error_records, duplicate_records)

        status, budget_detail = self._ingestion_status(doc_ids, error_records, metrics)
        duration = sum(timings)

        if budget_detail:
            details.append(budget_detail)

        scenario = ScenarioResult(
            name="ingestion",
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=duration,
            artifacts={"documents": doc_ids},
        )
        return scenario, doc_ids, timings

    def _ingestion_fixtures(self) -> List[Path]:
        return sorted(self._config.fixtures_docs.glob("*.pdf"))

    def _ingest_document(self, doc_path: Path) -> IngestionRecord:
        start = time.perf_counter()
        try:
            doc_id = self._processor.process_pdf(doc_path, source_name="eval-fixture")
            elapsed = time.perf_counter() - start
            logger.info("Ingested %s → %s in %.2fs", doc_path.name, doc_id, elapsed)
            return IngestionRecord(path=doc_path, elapsed=elapsed, doc_id=doc_id)
        except ValueError as dup_err:
            elapsed = time.perf_counter() - start
            logger.info("Skipping duplicate fixture %s (%s)", doc_path.name, dup_err)
            return IngestionRecord(path=doc_path, elapsed=elapsed, duplicate=True)
        except Exception as exc:  # noqa: BLE001 - evaluation should continue collecting failures
            elapsed = time.perf_counter() - start
            failure_msg = f"{doc_path.name}:{exc}"[:512]
            logger.exception("Failed to ingest %s", doc_path.name)
            return IngestionRecord(path=doc_path, elapsed=elapsed, error=failure_msg)

    def _ingestion_metrics(
        self,
        records: Sequence[IngestionRecord],
        doc_ids: Sequence[str],
        timings: Sequence[float],
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "documents_attempted": len(records),
            "documents_ingested": len(doc_ids),
            "documents_failed": len([record for record in records if record.error]),
            "durations_sec": list(timings),
            "document_ids": list(doc_ids),
        }

        if doc_ids:
            page_counts = self._fetch_page_counts(doc_ids)
            total_pages = sum(page_counts.values())
            total_time = sum(timings)
            metrics.update(
                {
                    "total_pages": total_pages,
                    "total_time_sec": total_time,
                    "ingest_time_per_page_sec": (total_time / total_pages) if total_pages else None,
                    "avg_time_per_doc_sec": (total_time / len(doc_ids)) if doc_ids else None,
                }
            )
        else:
            metrics.update(
                {
                    "total_pages": 0,
                    "total_time_sec": sum(timings),
                    "avg_time_per_doc_sec": None,
                    "ingest_time_per_page_sec": None,
                }
            )

        return metrics

    def _ingestion_details(
        self,
        doc_ids: Sequence[str],
        failures: Sequence[IngestionRecord],
        duplicates: Sequence[IngestionRecord],
    ) -> List[str]:
        details = [f"ingested:{doc_id}" for doc_id in doc_ids]
        details.extend(f"failed:{failure.error}" for failure in failures if failure.error)
        if duplicates:
            details.extend(f"duplicate:{record.path.name}" for record in duplicates)
        return details

    def _ingestion_status(
        self,
        doc_ids: Sequence[str],
        failures: Sequence[IngestionRecord],
        metrics: Dict[str, Any],
    ) -> Tuple[str, Optional[str]]:
        status = (
            "passed"
            if doc_ids and not any(record.error for record in failures)
            else ("failed" if any(record.error for record in failures) else "skipped")
        )
        time_budget = self._config.ingest_time_per_page_budget_sec
        time_per_page = metrics.get("ingest_time_per_page_sec")
        budget_detail: Optional[str] = None
        if time_budget and time_per_page is not None and time_per_page > time_budget:
            if status == "passed":
                status = "failed"
            budget_detail = (
                f"ingest time per page {time_per_page:.2f}s exceeds budget {time_budget:.2f}s"
            )
        return status, budget_detail

    def _fetch_page_counts(self, doc_ids: Iterable[str]) -> Dict[str, int]:
        if not doc_ids:
            return {}
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, COALESCE(num_pages, 0) AS num_pages
                    FROM documents
                    WHERE id = ANY(%s)
                    """,
                    (list(doc_ids),),
                )
                rows = cur.fetchall()
        return {row["id"]: row["num_pages"] for row in rows}
