"""Boot scenario runner responsible for service readiness checks."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Callable, List, Tuple

import httpx
import psycopg
from redis import Redis

from agent.config import settings

from agent.evaluator.models import ScenarioResult, ServiceCheck

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from agent.evaluator.harness import EvaluatorConfig


logger = logging.getLogger(__name__)


class BootScenarioRunner:
    """Run readiness checks for dependent services before evaluation."""

    def __init__(self, config: "EvaluatorConfig") -> None:
        self._config = config

    def run(self) -> ScenarioResult:
        services: List[Tuple[str, Callable[[], str]]] = [
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
        status = self._status_from_results(failures=failures, warnings=warnings)
        metrics = self._metrics_for_checks(services, checks, failures, warnings)
        details = [f"{check.name}:{check.status}" for check in checks]
        artifacts = {"checks": [asdict(check) for check in checks]}

        return ScenarioResult(
            name="boot",
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=total_duration,
            artifacts=artifacts,
        )

    def _status_from_results(self, *, failures: int, warnings: int) -> str:
        if failures:
            return "failed"
        if warnings:
            return "warn"
        return "passed"

    def _metrics_for_checks(
        self,
        services: List[Tuple[str, Callable[[], str]]],
        checks: List[ServiceCheck],
        failures: int,
        warnings: int,
    ) -> dict:
        return {
            "services_checked": len(services),
            "services_passed": len([check for check in checks if check.status == "passed"]),
            "services_warn": warnings,
            "services_failed": failures,
        }

    def _wait_for_postgres(self) -> str:
        deadline = time.time() + self._config.wait_timeout
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
                time.sleep(self._config.poll_interval)
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
        deadline = time.time() + self._config.wait_timeout
        client = Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=5)
        last_error = "redis not reachable"
        while time.time() < deadline:
            try:
                if client.ping():
                    return "redis ready"
            except Exception as exc:  # noqa: BLE001 - best effort loop
                last_error = str(exc)
            time.sleep(self._config.poll_interval)
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
        client = Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=5)
        deadline = time.time() + self._config.wait_timeout
        last_error = "worker queue not reachable"
        while time.time() < deadline:
            try:
                info = client.info(section="clients")
                if info and info.get("connected_clients", 0) >= 1:
                    return "worker assumed ready"
            except Exception as exc:  # noqa: BLE001 - best effort loop
                last_error = str(exc)
            time.sleep(self._config.poll_interval)
        raise RuntimeError(last_error)

    def _poll_http(self, url: str, name: str, *, tolerate_404: bool = False) -> str:
        deadline = time.time() + self._config.wait_timeout
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
            time.sleep(self._config.poll_interval)
        raise TimeoutError(last_error)
