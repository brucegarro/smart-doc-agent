"""Batch ingestion orchestration utilities backed by Redis queues."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from redis import Redis
from rq.job import Job

from agent.config import settings
from agent.ingestion.pdf_parser import ParseOptions
from agent.ingestion.processor import DocumentProcessor, ParallelIngestOutcome, ParallelIngestSummary

logger = logging.getLogger(__name__)


@dataclass
class BatchJobResult:
    """Represents the terminal outcome for a queued ingestion job."""

    pdf_path: Path
    status: str
    elapsed: float
    doc_id: Optional[str] = None
    message: Optional[str] = None
    job_id: Optional[str] = None
    duplicate: bool = False


class BatchIngestionRunner:
    """Orchestrates PDF ingestion via Redis-backed worker queues."""

    def __init__(self, processor: Optional[DocumentProcessor] = None) -> None:
        self._processor = processor or DocumentProcessor()
        self._redis_url = settings.redis_url

    def enqueue(
        self,
        pdf_paths: Iterable[Path | str],
        *,
        source_name: Optional[str] = None,
        parse_options: Optional[ParseOptions] = None,
        max_workers: int = 4,
    ) -> ParallelIngestSummary:
        effective_workers = max(1, max_workers)
        return self._processor.process_pdfs_parallel(
            pdf_paths,
            source_name=source_name,
            parse_options=parse_options,
            max_workers=effective_workers,
        )

    def wait_for_jobs(
        self,
        summary: ParallelIngestSummary,
        *,
        timeout: float = 600.0,
        poll_interval: float = 5.0,
    ) -> List[BatchJobResult]:
        outcomes = summary.items
        results: List[BatchJobResult] = []

        job_map = {item.job_id: item for item in outcomes if item.job_id}

        for outcome in outcomes:
            if outcome.job_id:
                continue
            if outcome.status == "skipped":
                results.append(
                    BatchJobResult(
                        pdf_path=outcome.pdf_path,
                        status="skipped",
                        elapsed=0.0,
                        message=outcome.message,
                        duplicate=True,
                    )
                )
            else:
                results.append(
                    BatchJobResult(
                        pdf_path=outcome.pdf_path,
                        status="failed",
                        elapsed=0.0,
                        message=outcome.message or "Failed before enqueue",
                    )
                )

        if not job_map:
            return results

        connection = Redis.from_url(self._redis_url)
        pending_jobs: dict[str, Job] = {}
        for job_id in job_map:
            try:
                job = Job.fetch(job_id, connection=connection)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Unable to fetch job %s: %s", job_id, exc)
                results.append(
                    BatchJobResult(
                        pdf_path=job_map[job_id].pdf_path,
                        status="failed",
                        elapsed=0.0,
                        job_id=job_id,
                        message=f"Unable to fetch job: {exc}",
                    )
                )
                continue
            pending_jobs[job_id] = job

        deadline = time.monotonic() + timeout
        completed: dict[str, Job] = {}

        while pending_jobs and time.monotonic() < deadline:
            for job_id, job in list(pending_jobs.items()):
                try:
                    status = job.get_status(refresh=True)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to refresh job %s: %s", job_id, exc)
                    continue

                if status in {"finished", "failed", "stopped", "canceled", "deferred"}:
                    completed[job_id] = job
                    pending_jobs.pop(job_id, None)
            if pending_jobs:
                time.sleep(poll_interval)

        timed_out = set(pending_jobs.keys())

        for job_id, job in completed.items():
            results.append(self._build_result(job_map[job_id], job))

        for job_id in timed_out:
            job = pending_jobs[job_id]
            results.append(
                BatchJobResult(
                    pdf_path=job_map[job_id].pdf_path,
                    status="timeout",
                    job_id=job_id,
                    elapsed=self._job_elapsed(job),
                    message=f"Timed out after {timeout:.0f}s",
                )
            )

        return results

    def _build_result(self, outcome: ParallelIngestOutcome, job: Job) -> BatchJobResult:
        status = job.get_status()
        result_payload = job.result if hasattr(job, "result") else None
        elapsed = self._job_elapsed(job)

        if status == "finished" and isinstance(result_payload, dict):
            payload_status = result_payload.get("status")
            if payload_status == "succeeded" and result_payload.get("doc_id"):
                return BatchJobResult(
                    pdf_path=outcome.pdf_path,
                    status="succeeded",
                    elapsed=elapsed,
                    doc_id=result_payload.get("doc_id"),
                    job_id=job.id,
                )
            if payload_status == "skipped":
                return BatchJobResult(
                    pdf_path=outcome.pdf_path,
                    status="skipped",
                    elapsed=elapsed,
                    message=result_payload.get("message"),
                    job_id=job.id,
                    duplicate=True,
                )
            return BatchJobResult(
                pdf_path=outcome.pdf_path,
                status=payload_status or "failed",
                elapsed=elapsed,
                message=result_payload.get("message") if isinstance(result_payload, dict) else None,
                job_id=job.id,
            )

        error_message = None
        if job.exc_info:
            error_message = job.exc_info.splitlines()[-1].strip()

        return BatchJobResult(
            pdf_path=outcome.pdf_path,
            status="failed" if status != "canceled" else "canceled",
            elapsed=elapsed,
            message=error_message or f"Job ended with status {status}",
            job_id=job.id,
        )

    @staticmethod
    def _job_elapsed(job: Job) -> float:
        timestamps: list[datetime] = []
        for attr in ("started_at", "ended_at", "enqueued_at"):
            value = getattr(job, attr, None)
            if isinstance(value, datetime):
                timestamps.append(value)

        if job.started_at and job.ended_at:
            return max((job.ended_at - job.started_at).total_seconds(), 0.0)
        if job.enqueued_at and job.ended_at:
            return max((job.ended_at - job.enqueued_at).total_seconds(), 0.0)
        return 0.0
