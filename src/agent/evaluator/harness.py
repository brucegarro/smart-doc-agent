"""End-to-end evaluation harness for smart-doc-agent."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import psycopg
from psycopg import sql
from psycopg.types.json import Json
from redis import Redis

from agent.config import settings
from agent.db import get_db_connection, db
from agent.ingestion.processor import DocumentProcessor
from agent.retrieval.search import ChunkResult, search_chunks

logger = logging.getLogger(__name__)


@dataclass
class ServiceCheck:
    name: str
    status: str
    latency_seconds: float
    detail: str = ""


@dataclass
class ScenarioResult:
    name: str
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorConfig:
    fixtures_root: Path
    fixtures_docs: Path
    fixtures_queries: Path
    fixtures_fields: Path
    fixtures_math: Path
    results_root: Path
    run_id: str
    git_sha: str
    wait_timeout: float = 600.0
    poll_interval: float = 5.0
    retrieval_k: int = 5
    retrieval_hit_threshold: float = 0.7
    retrieval_ndcg_threshold: float = 0.6
    query_latency_budget_ms: float = 2000.0
    ingest_time_per_page_budget_sec: float = 5.0

    @classmethod
    def from_env(cls) -> "EvaluatorConfig":
        fixtures_root = Path(os.getenv("EVAL_FIXTURES_DIR", "/eval/fixtures")).resolve()
        results_root = Path(os.getenv("EVAL_RESULTS_DIR", "/eval/results")).resolve()

        run_id = os.getenv("EVAL_RUN_ID")
        if not run_id:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            sha = cls._detect_git_sha()
            run_id = f"{timestamp}-{sha}"

        git_sha = cls._detect_git_sha()

        retrieval_hit_threshold = float(os.getenv("EVAL_RETRIEVAL_HIT_THRESHOLD", "0.7"))
        retrieval_ndcg_threshold = float(os.getenv("EVAL_RETRIEVAL_NDCG_THRESHOLD", "0.6"))
        query_latency_budget_ms = float(os.getenv("EVAL_QUERY_P95_BUDGET_MS", "2000"))
        ingest_time_per_page_budget_sec = float(os.getenv("EVAL_INGEST_TIME_PER_PAGE_BUDGET_SEC", "5"))
        wait_timeout = float(os.getenv("EVAL_WAIT_TIMEOUT", "180"))
        poll_interval = float(os.getenv("EVAL_POLL_INTERVAL", "5"))

        return cls(
            fixtures_root=fixtures_root,
            fixtures_docs=fixtures_root / "docs",
            fixtures_queries=fixtures_root / "queries.jsonl",
            fixtures_fields=fixtures_root / "fields.jsonl",
            fixtures_math=fixtures_root / "math.jsonl",
            results_root=results_root,
            run_id=run_id,
            git_sha=git_sha,
            wait_timeout=wait_timeout,
            poll_interval=poll_interval,
            retrieval_k=int(os.getenv("EVAL_RETRIEVAL_K", "5")),
            retrieval_hit_threshold=retrieval_hit_threshold,
            retrieval_ndcg_threshold=retrieval_ndcg_threshold,
            query_latency_budget_ms=query_latency_budget_ms,
            ingest_time_per_page_budget_sec=ingest_time_per_page_budget_sec,
        )

    @staticmethod
    def _detect_git_sha() -> str:
        for key in ("EVAL_GIT_SHA", "GIT_SHA", "CI_COMMIT_SHA"):
            value = os.getenv(key)
            if value:
                return value[:12]
        repo_root = Path("/workspace")
        candidates: Iterable[Path] = [Path.cwd(), Path.cwd().parent, repo_root]
        for candidate in candidates:
            head_path = candidate / ".git" / "HEAD"
            if not head_path.exists():
                continue
            try:
                raw = head_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue
            if raw.startswith("ref:"):
                ref = raw.split(" ", 1)[1]
                ref_path = candidate / ".git" / ref
                if ref_path.exists():
                    try:
                        return ref_path.read_text(encoding="utf-8").strip()[:12]
                    except (OSError, UnicodeDecodeError):
                        continue
            elif raw:
                return raw[:12]
        return "unknown"


class Evaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config
        self.processor = DocumentProcessor()
        self.results_dir = (self.config.results_root / self.config.run_id).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._perf_samples: Dict[str, Any] = {}
        self._start = time.time()
        self._original_db_name = settings.db_name
        self._ephemeral_db_name: Optional[str] = None

    def run(self) -> int:
        logger.info("Starting evaluator run %s", self.config.run_id)

        scenario_results: Dict[str, ScenarioResult] = {}
        ingestion_doc_ids: List[str] = []

        try:
            boot_result = self._run_boot_checks()
            scenario_results[boot_result.name] = boot_result
            if boot_result.status == "failed":
                logger.error("Boot checks failed; aborting evaluation")
                return self._finalize_run(scenario_results, ingestion_doc_ids)

            db_setup_result = self._prepare_test_database()
            scenario_results[db_setup_result.name] = db_setup_result
            if db_setup_result.status != "passed":
                logger.error("Test database setup failed; aborting evaluation")
                return self._finalize_run(scenario_results, ingestion_doc_ids)

            try:
                ingestion_result, ingestion_doc_ids = self._run_ingestion()
                scenario_results[ingestion_result.name] = ingestion_result

                query_result = self._run_query_suite(ingestion_doc_ids)
                scenario_results[query_result.name] = query_result

                extraction_result = self._run_extraction_suite()
                scenario_results[extraction_result.name] = extraction_result

                math_result = self._run_math_suite()
                scenario_results[math_result.name] = math_result

                perf_result = self._summarize_performance()
                scenario_results[perf_result.name] = perf_result
            except Exception as exc:  # noqa: BLE001 - ensure scorecard and cleanup still run
                logger.exception("Evaluator encountered an unexpected error: %s", exc)
                scenario_results["unexpected_error"] = ScenarioResult(
                    name="unexpected_error",
                    status="failed",
                    metrics={},
                    details=[str(exc)],
                    duration_seconds=0.0,
                )
        except Exception as outer_exc:  # noqa: BLE001 - catch earlier failures
            logger.exception("Evaluator failed before scenarios completed: %s", outer_exc)
            scenario_results["unexpected_error"] = ScenarioResult(
                name="unexpected_error",
                status="failed",
                metrics={},
                details=[str(outer_exc)],
                duration_seconds=0.0,
            )

        return self._finalize_run(scenario_results, ingestion_doc_ids)

    def _finalize_run(self, scenarios: Dict[str, ScenarioResult], doc_ids: List[str]) -> int:
        scoreboard: Optional[Dict[str, Any]] = None
        exit_code = 1
        try:
            scoreboard = self._build_scorecard(scenarios)
            self._write_scorecard(scoreboard)
            self._persist_scorecard(scoreboard, doc_ids)
            exit_code = 0 if scoreboard["gates"].get("all", True) else 1
            return exit_code
        finally:
            cleanup_result = self._teardown_test_database()
            if cleanup_result:
                logger.info(
                    "Test database cleanup status: %s (%s)",
                    cleanup_result.status,
                    ";".join(cleanup_result.details) if cleanup_result.details else "no details",
                )
            if scoreboard is not None:
                logger.info("Evaluator finished with exit code %s", exit_code)
            else:
                logger.info("Evaluator finished with exit code %s (scorecard unavailable)", exit_code)

    # ------------------------------------------------------------------
    # Boot checks
    # ------------------------------------------------------------------
    def _run_boot_checks(self) -> ScenarioResult:
        services = [
            ("postgres", self._wait_for_postgres),
            ("minio", self._wait_for_minio),
            ("ollama", self._wait_for_ollama),
            ("redis", self._wait_for_redis),
            ("app", self._wait_for_app_runtime),
            ("worker", self._wait_for_worker),
        ]

        checks: List[ServiceCheck] = []
        failures = 0
        warnings = 0

        start = time.perf_counter()
        for name, fn in services:
            svc_start = time.perf_counter()
            try:
                detail = fn()
                latency = time.perf_counter() - svc_start
                checks.append(ServiceCheck(name=name, status="passed", latency_seconds=latency, detail=detail))
                logger.info("Service %s ready in %.2fs", name, latency)
            except TimeoutError as exc:
                latency = time.perf_counter() - svc_start
                failures += 1
                detail = str(exc)
                logger.error("Service %s failed readiness check after %.2fs: %s", name, latency, detail)
                checks.append(ServiceCheck(name=name, status="failed", latency_seconds=latency, detail=detail))
            except RuntimeError as exc:
                latency = time.perf_counter() - svc_start
                warnings += 1
                detail = str(exc)
                logger.warning("Service %s degraded (%.2fs): %s", name, latency, detail)
                checks.append(ServiceCheck(name=name, status="warn", latency_seconds=latency, detail=detail))

        total_duration = time.perf_counter() - start
        status = "passed"
        if failures:
            status = "failed"
        elif warnings:
            status = "warn"

        metrics = {
            "services_checked": len(services),
            "services_passed": len([c for c in checks if c.status == "passed"]),
            "services_warn": warnings,
            "services_failed": failures,
        }
        details = [f"{c.name}:{c.status}" for c in checks]
        artifacts = {"checks": [asdict(c) for c in checks]}

        return ScenarioResult(
            name="boot",
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=total_duration,
            artifacts=artifacts,
        )

    def _wait_for_postgres(self) -> str:
        deadline = time.time() + self.config.wait_timeout
        last_error = "database not reachable"
        while time.time() < deadline:
            try:
                with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                        cur.fetchone()
                return "postgres ready"
            except Exception as exc:  # noqa: BLE001 - best effort loop
                last_error = str(exc)
                time.sleep(self.config.poll_interval)
        raise TimeoutError(last_error)

    def _wait_for_minio(self) -> str:
        base_url = settings.s3_endpoint.rstrip("/")
        url = os.getenv("EVAL_MINIO_HEALTHCHECK", f"{base_url}/minio/health/ready")
        return self._poll_http(url, "minio")

    def _wait_for_ollama(self) -> str:
        base_url = settings.ollama_base.rstrip("/")
        url = os.getenv("EVAL_OLLAMA_HEALTHCHECK", f"{base_url}/api/tags")
        return self._poll_http(url, "ollama")

    def _wait_for_redis(self) -> str:
        deadline = time.time() + self.config.wait_timeout
        client = Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=5)
        last_error = "redis not reachable"
        while time.time() < deadline:
            try:
                if client.ping():
                    return "redis ready"
            except Exception as exc:  # noqa: BLE001 - best effort loop
                last_error = str(exc)
            time.sleep(self.config.poll_interval)
        raise TimeoutError(last_error)

    def _wait_for_app_runtime(self) -> str:
        url = os.getenv("EVAL_APP_HEALTHCHECK_URL", "http://app:8080/healthz")
        if not url:
            raise RuntimeError("app healthcheck disabled")
        try:
            return self._poll_http(url, "app", tolerate_404=True)
        except TimeoutError as exc:
            raise RuntimeError(f"app healthcheck timed out ({exc})") from exc

    def _wait_for_worker(self) -> str:
        # Worker readiness is approximated via Redis queue visibility.
        client = Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=5)
        deadline = time.time() + self.config.wait_timeout
        last_error = "worker queue not reachable"
        while time.time() < deadline:
            try:
                info = client.info(section="clients")
                if info and info.get("connected_clients", 0) >= 1:
                    return "worker assumed ready"
            except Exception as exc:  # noqa: BLE001 - best effort loop
                last_error = str(exc)
            time.sleep(self.config.poll_interval)
        raise RuntimeError(last_error)

    def _poll_http(self, url: str, name: str, *, tolerate_404: bool = False) -> str:
        deadline = time.time() + self.config.wait_timeout
        last_error = f"{name} http poll failed"
        while time.time() < deadline:
            try:
                response = httpx.get(url, timeout=5.0)
                if response.status_code == 200:
                    return f"{name} ready ({response.status_code})"
                if tolerate_404 and response.status_code == 404:
                    return f"{name} responded 404 (treated as ready)"
                last_error = f"status {response.status_code}"
            except Exception as exc:  # noqa: BLE001 - best effort loop
                last_error = str(exc)
            time.sleep(self.config.poll_interval)
        raise TimeoutError(last_error)

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------
    def _run_ingestion(self) -> Tuple[ScenarioResult, List[str]]:
        docs = sorted(self.config.fixtures_docs.glob("*.pdf"))
        if not docs:
            logger.warning("No PDF fixtures found in %s; skipping ingestion", self.config.fixtures_docs)
            return (
                ScenarioResult(
                    name="ingestion",
                    status="skipped",
                    metrics={"documents": 0},
                    details=["no fixtures"],
                ),
                [],
            )

        logger.info("Ingestion scenario starting with %s document(s)", len(docs))
        doc_ids: List[str] = []
        failures: List[str] = []
        timings: List[float] = []

        for doc_path in docs:
            start = time.perf_counter()
            try:
                doc_id = self.processor.process_pdf(doc_path, source_name="eval-fixture")
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
                doc_ids.append(doc_id)
                logger.info("Ingested %s → %s in %.2fs", doc_path.name, doc_id, elapsed)
            except ValueError as dup_err:
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
                detail = f"duplicate:{doc_path.name}:{dup_err}"
                logger.info("Skipping duplicate fixture %s (%s)", doc_path.name, dup_err)
            except Exception as exc:  # noqa: BLE001 - evaluation should continue collecting failures
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
                failure_msg = f"{doc_path.name}:{exc}"[:512]
                failures.append(failure_msg)
                logger.exception("Failed to ingest %s", doc_path.name)

        metrics: Dict[str, Any] = {
            "documents_attempted": len(docs),
            "documents_ingested": len(doc_ids),
            "documents_failed": len(failures),
            "durations_sec": timings,
        }

        if doc_ids:
            page_counts = self._fetch_page_counts(doc_ids)
            total_pages = sum(page_counts.values())
            total_time = sum(timings)
            metrics["total_pages"] = total_pages
            metrics["total_time_sec"] = total_time
            metrics["ingest_time_per_page_sec"] = (total_time / total_pages) if total_pages else None
            metrics["avg_time_per_doc_sec"] = (total_time / len(doc_ids)) if doc_ids else None
        else:
            metrics.update({"total_pages": 0, "total_time_sec": sum(timings), "avg_time_per_doc_sec": None, "ingest_time_per_page_sec": None})

        metrics["document_ids"] = doc_ids
        details = [f"ingested:{doc_id}" for doc_id in doc_ids]
        details.extend(f"failed:{msg}" for msg in failures)

        status = "passed" if doc_ids and not failures else ("failed" if failures else "skipped")

        time_budget = self.config.ingest_time_per_page_budget_sec
        time_per_page = metrics.get("ingest_time_per_page_sec")
        if time_budget and time_per_page is not None:
            if time_per_page > time_budget:
                status = "failed" if status == "passed" else status
                details.append(
                    f"ingest time per page {time_per_page:.2f}s exceeds budget {time_budget:.2f}s"
                )

        duration = sum(timings)
        self._perf_samples["ingestion_durations"] = timings

        return (
            ScenarioResult(
                name="ingestion",
                status=status,
                metrics=metrics,
                details=details,
                duration_seconds=duration,
                artifacts={"documents": doc_ids},
            ),
            doc_ids,
        )

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

    def _run_query_suite(self, doc_ids: List[str]) -> ScenarioResult:
        if not doc_ids:
            logger.warning("No ingested documents available; skipping query evaluation")
            return ScenarioResult(name="queries", status="skipped", metrics={}, details=["no documents"], artifacts={})

        if not self.config.fixtures_queries.exists():
            logger.warning("Query fixtures missing at %s", self.config.fixtures_queries)
            return ScenarioResult(name="queries", status="skipped", metrics={}, details=["no fixtures"], artifacts={})

        fixtures = self._load_jsonl(self.config.fixtures_queries)
        fixtures = [item for item in fixtures if item.get("query")]
        if not fixtures:
            logger.warning("Query fixture file empty; skipping queries scenario")
            return ScenarioResult(name="queries", status="skipped", metrics={}, details=["no fixtures"], artifacts={})

        logger.info("Executing %s query fixture(s)", len(fixtures))

        k = self.config.retrieval_k
        latencies_ms: List[float] = []
        hits: List[float] = []
        ndcgs: List[float] = []
        per_query_records: List[Dict[str, Any]] = []

        for record in fixtures:
            query = record["query"]
            gold_chunks: List[str] = [str(cid) for cid in record.get("gold_chunk_ids", [])]
            gold_set = set(gold_chunks)
            gold_available = bool(gold_set)

            start = time.perf_counter()
            try:
                results = search_chunks(query, limit=k)
            except Exception as exc:  # noqa: BLE001 - log and continue
                latency = (time.perf_counter() - start) * 1000
                latencies_ms.append(latency)
                detail = f"error:{record.get('id', query[:32])}:{exc}"[:256]
                logger.exception("Query execution failed: %s", detail)
                per_query_records.append({
                    "id": record.get("id"),
                    "query": query,
                    "error": str(exc),
                    "latency_ms": latency,
                })
                continue

            latency = (time.perf_counter() - start) * 1000
            latencies_ms.append(latency)

            retrieved_chunk_ids = [res.chunk_id for res in results]
            hit = self._hit_at_k(retrieved_chunk_ids, gold_set, k) if gold_available else None
            ndcg = self._ndcg_at_k(retrieved_chunk_ids, gold_set, k) if gold_available else None

            if hit is not None:
                hits.append(hit)
            if ndcg is not None:
                ndcgs.append(ndcg)

            per_query_records.append({
                "id": record.get("id"),
                "query": query,
                "latency_ms": latency,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "gold_chunk_ids": gold_chunks,
                "hit_at_k": hit,
                "ndcg_at_k": ndcg,
            })

        metrics: Dict[str, Any] = {
            "queries_run": len(per_query_records),
            "latency_ms_all": latencies_ms,
        }
        metrics["latency_p50_ms"] = self._percentile(latencies_ms, 0.5)
        metrics["latency_p95_ms"] = self._percentile(latencies_ms, 0.95)
        metrics["hit_at_k_avg"] = sum(hits) / len(hits) if hits else None
        metrics["ndcg_at_k_avg"] = sum(ndcgs) / len(ndcgs) if ndcgs else None

        status = "passed"
        details: List[str] = []
        if metrics["hit_at_k_avg"] is not None and metrics["hit_at_k_avg"] < self.config.retrieval_hit_threshold:
            status = "failed"
            details.append(f"hit@{k} below threshold {metrics['hit_at_k_avg']:.3f}< {self.config.retrieval_hit_threshold}")
        if metrics["ndcg_at_k_avg"] is not None and metrics["ndcg_at_k_avg"] < self.config.retrieval_ndcg_threshold:
            status = "failed"
            details.append(f"ndcg@{k} below threshold {metrics['ndcg_at_k_avg']:.3f}< {self.config.retrieval_ndcg_threshold}")
        if metrics["latency_p95_ms"] is not None and metrics["latency_p95_ms"] > self.config.query_latency_budget_ms:
            status = "warn" if status != "failed" else status
            details.append(f"p95 latency {metrics['latency_p95_ms']:.1f}ms exceeds budget {self.config.query_latency_budget_ms}ms")

        if not hits and not ndcgs:
            status = "warn" if metrics["queries_run"] else "skipped"
            if status == "warn":
                details.append("no gold chunk ids provided; metrics skipped")

        self._perf_samples["query_latencies_ms"] = latencies_ms
        artifacts = {"queries": per_query_records}

        return ScenarioResult(
            name="queries",
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=sum(latencies_ms) / 1000 if latencies_ms else 0.0,
            artifacts=artifacts,
        )

    def _run_extraction_suite(self) -> ScenarioResult:
        if not self.config.fixtures_fields.exists():
            logger.warning("Extraction fixtures missing at %s", self.config.fixtures_fields)
            return ScenarioResult(name="extraction", status="skipped", metrics={}, details=["no fixtures"], artifacts={})

        fixtures = self._load_jsonl(self.config.fixtures_fields)
        evaluable = [item for item in fixtures if item.get("document_id")]
        if not evaluable:
            return ScenarioResult(
                name="extraction",
                status="skipped",
                metrics={"fixtures_total": len(fixtures)},
                details=["no document ids in fixtures"],
                artifacts={},
            )

        logger.info("Extraction fixtures present but evaluation logic deferred.")
        return ScenarioResult(
            name="extraction",
            status="warn",
            metrics={"fixtures_total": len(fixtures)},
            details=["extraction scoring not implemented"],
            artifacts={"fixtures": evaluable[:3]},
        )

    def _run_math_suite(self) -> ScenarioResult:
        if not self.config.fixtures_math.exists():
            logger.warning("Math fixtures missing at %s", self.config.fixtures_math)
            return ScenarioResult(name="math", status="skipped", metrics={}, details=["no fixtures"], artifacts={})

        fixtures = self._load_jsonl(self.config.fixtures_math)
        evaluable = [item for item in fixtures if item.get("latex_gold")]
        if not evaluable:
            return ScenarioResult(
                name="math",
                status="skipped",
                metrics={"fixtures_total": len(fixtures)},
                details=["no gold latex provided"],
                artifacts={},
            )

        logger.info("Math fixtures present but evaluation logic deferred.")
        return ScenarioResult(
            name="math",
            status="warn",
            metrics={"fixtures_total": len(fixtures)},
            details=["math scoring not implemented"],
            artifacts={"fixtures": evaluable[:3]},
        )

    def _summarize_performance(self) -> ScenarioResult:
        ingestion_times = self._perf_samples.get("ingestion_durations", []) or []
        query_latencies = self._perf_samples.get("query_latencies_ms", []) or []

        metrics: Dict[str, Any] = {
            "ingestion_total_time_sec": sum(ingestion_times) if ingestion_times else None,
            "ingestion_avg_time_sec": (sum(ingestion_times) / len(ingestion_times)) if ingestion_times else None,
            "query_p50_ms": self._percentile(query_latencies, 0.5),
            "query_p95_ms": self._percentile(query_latencies, 0.95),
        }

        status = "passed"
        details: List[str] = []
        if metrics["query_p95_ms"] is not None and metrics["query_p95_ms"] > self.config.query_latency_budget_ms:
            status = "warn"
            details.append(f"query p95 {metrics['query_p95_ms']:.1f}ms exceeds budget {self.config.query_latency_budget_ms}ms")

        if not ingestion_times and not query_latencies:
            status = "skipped"
            details.append("no performance samples collected")

        return ScenarioResult(
            name="perf",
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=0.0,
            artifacts={},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_test_db_name(self) -> str:
        base = f"{self._original_db_name}_eval_{self.config.run_id}".lower()
        sanitized = re.sub(r"[^a-z0-9_]+", "_", base)
        if len(sanitized) > 63:
            sanitized = sanitized[:63]
        return sanitized.rstrip("_") or f"{self._original_db_name}_eval"

    def _prepare_test_database(self) -> ScenarioResult:
        start = time.perf_counter()
        db_name = self._build_test_db_name()
        details: List[str] = []
        metrics: Dict[str, Any] = {"database": db_name}
        status = "passed"
        try:
            self._create_fresh_database(db_name)
            self._apply_schema_to_database(db_name)
            self._switch_database(db_name)
            self._ephemeral_db_name = db_name
            details.append(f"created:{db_name}")
        except Exception as exc:  # noqa: BLE001 - setup must not crash evaluator
            logger.exception("Failed to prepare test database %s", db_name)
            status = "failed"
            details.append(f"error:{exc}")
            try:
                self._drop_database(db_name)
            except Exception as cleanup_exc:  # pragma: no cover - best effort rollback
                details.append(f"drop_cleanup_error:{cleanup_exc}")
            # Ensure we fall back to original database in case of partial setup
            self._restore_original_database()
        duration = time.perf_counter() - start
        return ScenarioResult(
            name="db_setup",
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=duration,
        )

    def _teardown_test_database(self) -> Optional[ScenarioResult]:
        if not self._ephemeral_db_name:
            return None

        start = time.perf_counter()
        details: List[str] = []
        status = "passed"
        db_name = self._ephemeral_db_name

        try:
            db.close()
        except Exception as exc:  # pragma: no cover - defensive close
            status = "warn"
            details.append(f"pool_close:{exc}")

        try:
            self._drop_database(db_name)
            details.append(f"dropped:{db_name}")
        except Exception as exc:  # pragma: no cover - best effort cleanup
            status = "failed"
            details.append(f"drop_error:{exc}")

        self._restore_original_database()
        duration = time.perf_counter() - start
        self._ephemeral_db_name = None

        return ScenarioResult(
            name="db_cleanup",
            status=status,
            metrics={"database": db_name},
            details=details,
            duration_seconds=duration,
        )

    def _restore_original_database(self) -> None:
        db.close()
        object.__setattr__(settings, "db_name", self._original_db_name)
        os.environ["DB_NAME"] = self._original_db_name
        os.environ["DATABASE_URL"] = settings.database_url

    def _switch_database(self, db_name: str) -> None:
        db.close()
        object.__setattr__(settings, "db_name", db_name)
        os.environ["DB_NAME"] = db_name
        os.environ["DATABASE_URL"] = settings.database_url

    def _create_fresh_database(self, db_name: str) -> None:
        conninfo = self._build_conninfo(self._original_db_name)
        with psycopg.connect(conninfo, autocommit=True) as conn:
            drop_sql = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(db_name))
            create_sql = sql.SQL(
                "CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'"
            ).format(sql.Identifier(db_name), sql.Identifier(settings.db_user))
            conn.execute(drop_sql)
            conn.execute(create_sql)

    def _drop_database(self, db_name: str) -> None:
        conninfo = self._build_conninfo(self._original_db_name)
        with psycopg.connect(conninfo, autocommit=True) as conn:
            drop_sql = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(db_name))
            conn.execute(drop_sql)

    def _apply_schema_to_database(self, db_name: str) -> None:
        configured_dir = os.getenv("EVAL_SCHEMA_DIR")
        candidate_dirs = []
        if configured_dir:
            candidate_dirs.append(Path(configured_dir))
        candidate_dirs.extend(
            [
                Path("/app/sql"),
                Path(__file__).resolve().parents[2] / "sql",
                Path(__file__).resolve().parents[3] / "sql",
                Path.cwd() / "sql",
            ]
        )

        schema_dir: Optional[Path] = None
        for candidate in candidate_dirs:
            if candidate.exists():
                schema_dir = candidate
                break

        if not schema_dir:
            raise FileNotFoundError("No schema scripts directory found")

        scripts = sorted(schema_dir.glob("*.sql"))
        if not scripts:
            raise FileNotFoundError(f"No schema scripts found in {schema_dir}")

        conninfo = self._build_conninfo(db_name)
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                for script_path in scripts:
                    sql_text = script_path.read_text(encoding="utf-8").strip()
                    if not sql_text:
                        continue
                    cur.execute(sql_text)
            conn.commit()

    def _build_conninfo(self, database: str) -> str:
        return (
            f"postgresql://{settings.db_user}:{settings.db_password}"
            f"@{settings.db_host}:{settings.db_port}/{database}"
        )
    def _build_scorecard(self, scenarios: Dict[str, ScenarioResult]) -> Dict[str, Any]:
        scenario_payload = {name: self._scenario_to_dict(result) for name, result in scenarios.items()}

        gating_names = ["boot", "ingestion", "queries", "extraction", "math", "perf"]
        gates: Dict[str, bool] = {}
        hard_fail = False
        for name in gating_names:
            result = scenarios.get(name)
            if not result:
                gates[name] = False
                hard_fail = True
                continue
            if result.status == "failed":
                gates[name] = False
                hard_fail = True
            elif result.status == "skipped":
                gates[name] = False
            else:
                gates[name] = True

        gates["all"] = not hard_fail and all(gates.values())

        scored_results = [result for name, result in scenarios.items() if name in gating_names and result.status in {"passed", "failed"}]
        if scored_results:
            total = len(scored_results)
            passed = len([res for res in scored_results if res.status == "passed"])
            overall_score = passed / total if total else None
        else:
            overall_score = None

        metrics = {
            "overall_score": overall_score,
            "runtime_seconds": time.time() - self._start,
        }

        payload = {
            "run_id": self.config.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "env": {
                "git_sha": self.config.git_sha,
                "text_llm_model": settings.text_llm_model,
                "vlm_model": settings.vlm_model,
                "embedder_model": settings.embedder_model,
                "retrieval_k": self.config.retrieval_k,
                "ingest_time_per_page_budget_sec": self.config.ingest_time_per_page_budget_sec,
            },
            "gates": gates,
            "metrics": metrics,
            "scenarios": scenario_payload,
        }
        return payload

    def _scenario_to_dict(self, scenario: ScenarioResult) -> Dict[str, Any]:
        return {
            "status": scenario.status,
            "metrics": scenario.metrics,
            "details": scenario.details,
            "duration_seconds": scenario.duration_seconds,
            "artifacts": scenario.artifacts,
        }

    def _write_scorecard(self, scoreboard: Dict[str, Any]) -> None:
        output_path = self.results_dir / "scorecard.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(scoreboard, handle, indent=2, sort_keys=True)
        logger.info("Scorecard written to %s", output_path)

    def _persist_scorecard(self, scoreboard: Dict[str, Any], doc_ids: List[str]) -> None:
        if not doc_ids:
            logger.warning("No document ids from ingestion; skipping eval_results insert")
            return

        metric_value = scoreboard.get("metrics", {}).get("overall_score")
        metric_value_text = None
        if metric_value is not None:
            metric_value_text = f"overall={metric_value:.3f}"

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO eval_results (
                            document_id,
                            metric_name,
                            metric_value,
                            metric_value_text,
                            dataset,
                            model_name,
                            task,
                            metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            doc_ids[0],
                            "e2e_score",
                            metric_value,
                            metric_value_text,
                            os.getenv("EVAL_DATASET", "eval-fixtures"),
                            os.getenv("EVAL_MODEL_NAME", "smart-doc-agent"),
                            "end_to_end_eval",
                            Json(scoreboard),
                        ),
                    )
                    conn.commit()
            logger.info("Persisted scorecard summary to eval_results for document %s", doc_ids[0])
        except Exception as exc:  # noqa: BLE001 - persistence best effort
            logger.exception("Failed to persist scorecard: %s", exc)

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping invalid JSON line in %s: %s", path, exc)
        except FileNotFoundError:
            logger.warning("Fixture file not found: %s", path)
        return records

    def _hit_at_k(self, predicted: Iterable[str], gold: Iterable[str], k: int) -> float:
        if not gold:
            return 0.0
        gold_set = set(gold)
        for idx, chunk_id in enumerate(predicted):
            if idx >= k:
                break
            if chunk_id in gold_set:
                return 1.0
        return 0.0

    def _ndcg_at_k(self, predicted: List[str], gold: Iterable[str], k: int) -> float:
        gold_set = set(gold)
        if not gold_set:
            return 0.0
        gains = []
        for idx in range(min(k, len(predicted))):
            chunk_id = predicted[idx]
            gains.append(1.0 if chunk_id in gold_set else 0.0)
        if not gains:
            return 0.0
        dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))
        ideal_gains = sorted(gains, reverse=True)
        idcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(ideal_gains))
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _percentile(self, values: List[float], percentile: float) -> Optional[float]:
        if not values:
            return None
        if not 0 <= percentile <= 1:
            raise ValueError("Percentile must be between 0 and 1")
        sorted_vals = sorted(values)
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        index = percentile * (len(sorted_vals) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return sorted_vals[int(index)]
        lower_val = sorted_vals[lower]
        upper_val = sorted_vals[upper]
        fraction = index - lower
        return lower_val + (upper_val - lower_val) * fraction


def main() -> None:
    log_level = os.getenv("EVAL_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="[%(asctime)s] %(levelname)s %(message)s")
    config = EvaluatorConfig.from_env()
    evaluator = Evaluator(config)
    exit_code = evaluator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
