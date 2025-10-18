"""End-to-end evaluation harness for smart-doc-agent."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycopg.types.json import Json

from agent.config import settings
from agent.db import get_db_connection
from agent.ingestion.processor import DocumentProcessor
from agent.evaluator.boot_runner import BootScenarioRunner
from agent.evaluator.db_manager import DatabaseManager
from agent.evaluator.ingestion_runner import IngestionScenarioRunner
from agent.evaluator.models import ScenarioResult
from agent.evaluator.quality_runner import CodeQualityRunner
from agent.evaluator.query_runner import QueryScenarioRunner
from agent.evaluator.answer_runner import AnswerScenarioRunner

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    fixtures_root: Path
    fixtures_docs: Path
    fixtures_queries: Path
    fixtures_fields: Path
    fixtures_math: Path
    fixtures_quality: Path
    results_root: Path
    run_id: str
    git_sha: str
    project_root: Path
    code_quality_paths: Tuple[Path, ...]
    wait_timeout: float = 600.0
    poll_interval: float = 5.0
    retrieval_k: int = 5
    retrieval_hit_threshold: float = 0.7
    retrieval_ndcg_threshold: float = 0.6
    retrieval_text_match_threshold: float = 0.82
    query_latency_budget_ms: float = 2000.0
    ingest_time_per_page_budget_sec: float = 5.0
    complexity_function_threshold: float = 10.0
    complexity_average_threshold: float = 5.0
    maintainability_threshold: float = 65.0
    answer_context_limit: int = 3
    answer_max_tokens: int = 256
    answer_temperature: float = 0.2
    answer_similarity_threshold: float = 0.55
    answer_improvement_margin: float = 0.05
    answer_improvement_rate_threshold: float = 0.5
    answer_judge_pass_threshold: float = 0.7
    answer_judge_pass_rate_threshold: float = 0.6

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
        default_project_root = cls._detect_project_root()
        project_root = Path(os.getenv("EVAL_PROJECT_ROOT", str(default_project_root))).resolve()

        quality_path_tokens = os.getenv("EVAL_CODE_QUALITY_PATHS", "src,agent").split(",")
        code_quality_paths: List[Path] = []
        for token in quality_path_tokens:
            raw = token.strip()
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = project_root / candidate
            candidate = candidate.resolve()
            if candidate not in code_quality_paths:
                code_quality_paths.append(candidate)

        retrieval_hit_threshold = float(os.getenv("EVAL_RETRIEVAL_HIT_THRESHOLD", "0.7"))
        retrieval_ndcg_threshold = float(os.getenv("EVAL_RETRIEVAL_NDCG_THRESHOLD", "0.6"))
        retrieval_text_match_threshold = float(os.getenv("EVAL_RETRIEVAL_TEXT_MATCH_THRESHOLD", "0.82"))
        query_latency_budget_ms = float(os.getenv("EVAL_QUERY_P95_BUDGET_MS", "2000"))
        ingest_time_per_page_budget_sec = float(os.getenv("EVAL_INGEST_TIME_PER_PAGE_BUDGET_SEC", "5"))
        wait_timeout = float(os.getenv("EVAL_WAIT_TIMEOUT", "180"))
        poll_interval = float(os.getenv("EVAL_POLL_INTERVAL", "5"))
        complexity_function_threshold = float(os.getenv("EVAL_COMPLEXITY_FUNCTION_MAX", "10"))
        complexity_average_threshold = float(os.getenv("EVAL_COMPLEXITY_AVG_MAX", "5"))
        maintainability_threshold = float(os.getenv("EVAL_MAINTAINABILITY_MIN", "65"))
        answer_context_limit = int(os.getenv("EVAL_ANSWER_CONTEXT_LIMIT", "3"))
        answer_max_tokens = int(os.getenv("EVAL_ANSWER_MAX_TOKENS", "256"))
        answer_temperature = float(os.getenv("EVAL_ANSWER_TEMPERATURE", "0.2"))
        answer_similarity_threshold = float(os.getenv("EVAL_ANSWER_SIMILARITY_MIN", "0.55"))
        answer_improvement_margin = float(os.getenv("EVAL_ANSWER_IMPROVEMENT_MARGIN", "0.05"))
        answer_improvement_rate_threshold = float(os.getenv("EVAL_ANSWER_IMPROVEMENT_RATE_MIN", "0.5"))
        answer_judge_pass_threshold = float(os.getenv("EVAL_ANSWER_JUDGE_PASS_THRESHOLD", "0.7"))
        answer_judge_pass_rate_threshold = float(os.getenv("EVAL_ANSWER_JUDGE_PASS_RATE_MIN", "0.6"))

        return cls(
            fixtures_root=fixtures_root,
            fixtures_docs=fixtures_root / "docs",
            fixtures_queries=fixtures_root / "queries.jsonl",
            fixtures_fields=fixtures_root / "fields.jsonl",
            fixtures_math=fixtures_root / "math.jsonl",
            fixtures_quality=fixtures_root / "quality.json",
            results_root=results_root,
            run_id=run_id,
            git_sha=git_sha,
            project_root=project_root,
            code_quality_paths=tuple(code_quality_paths),
            wait_timeout=wait_timeout,
            poll_interval=poll_interval,
            retrieval_k=int(os.getenv("EVAL_RETRIEVAL_K", "5")),
            retrieval_hit_threshold=retrieval_hit_threshold,
            retrieval_ndcg_threshold=retrieval_ndcg_threshold,
            retrieval_text_match_threshold=retrieval_text_match_threshold,
            query_latency_budget_ms=query_latency_budget_ms,
            ingest_time_per_page_budget_sec=ingest_time_per_page_budget_sec,
            complexity_function_threshold=complexity_function_threshold,
            complexity_average_threshold=complexity_average_threshold,
            maintainability_threshold=maintainability_threshold,
            answer_context_limit=answer_context_limit,
            answer_max_tokens=answer_max_tokens,
            answer_temperature=answer_temperature,
            answer_similarity_threshold=answer_similarity_threshold,
            answer_improvement_margin=answer_improvement_margin,
            answer_improvement_rate_threshold=answer_improvement_rate_threshold,
            answer_judge_pass_threshold=answer_judge_pass_threshold,
            answer_judge_pass_rate_threshold=answer_judge_pass_rate_threshold,
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

    @staticmethod
    def _detect_project_root() -> Path:
        start = Path(__file__).resolve()
        markers = ("pyproject.toml", "requirements.txt", "README.md", ".git")
        for candidate in [start.parent, *start.parents]:
            try:
                if any((candidate / marker).exists() for marker in markers):
                    return candidate
            except OSError:
                continue
        try:
            return Path.cwd().resolve()
        except OSError:
            return start.parent


class Evaluator:
    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config
        self.processor = DocumentProcessor()
        self.results_dir = (self.config.results_root / self.config.run_id).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._perf_samples: Dict[str, Any] = {}
        self._start = time.time()
        self._boot_runner = BootScenarioRunner(self.config)
        self._db_manager = DatabaseManager(self.config, settings.db_name)
        self._ingestion_runner = IngestionScenarioRunner(self.config, self.processor)
        self._query_runner = QueryScenarioRunner(self.config, self._load_jsonl, self._percentile)
        self._quality_runner = CodeQualityRunner(self.config)
        self._answer_runner = AnswerScenarioRunner(self.config, self._load_jsonl)

    def run(self) -> int:
        logger.info("Starting evaluator run %s", self.config.run_id)

        scenario_results: Dict[str, ScenarioResult] = {}
        ingestion_doc_ids: List[str] = []

        try:
            boot_result = self._boot_runner.run()
            scenario_results[boot_result.name] = boot_result
            if boot_result.status == "failed":
                logger.error("Boot checks failed; aborting evaluation")
                return self._finalize_run(scenario_results, ingestion_doc_ids)

            quality_result = self._run_code_quality_suite()
            scenario_results[quality_result.name] = quality_result
            if quality_result.status == "failed":
                logger.error("Code quality checks failed; continuing to capture downstream metrics")

            db_setup_result = self._db_manager.prepare()
            scenario_results[db_setup_result.name] = db_setup_result
            if db_setup_result.status != "passed":
                logger.error("Test database setup failed; aborting evaluation")
                return self._finalize_run(scenario_results, ingestion_doc_ids)

            try:
                ingestion_result, ingestion_doc_ids, ingestion_times = self._ingestion_runner.run()
                scenario_results[ingestion_result.name] = ingestion_result
                self._perf_samples["ingestion_durations"] = ingestion_times

                query_result = self._run_query_suite(ingestion_doc_ids)
                scenario_results[query_result.name] = query_result

                answer_result = self._run_answer_suite(query_result.artifacts.get("queries", []) if query_result.artifacts else [])
                scenario_results[answer_result.name] = answer_result

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
            cleanup_result = self._db_manager.teardown()
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
    # Scenarios
    # ------------------------------------------------------------------
    def _run_code_quality_suite(self) -> ScenarioResult:
        start = time.perf_counter()
        fixture = self._load_quality_fixture()
        outcome = self._quality_runner.evaluate(fixture)
        duration = time.perf_counter() - start
        return ScenarioResult(
            name="code_quality",
            status=outcome.status,
            metrics=outcome.metrics,
            details=outcome.details,
            duration_seconds=duration,
            artifacts=outcome.artifacts,
        )

    def _run_query_suite(self, doc_ids: List[str]) -> ScenarioResult:
        outcome = self._query_runner.run(doc_ids)
        self._perf_samples["query_latencies_ms"] = outcome.latencies_ms
        return ScenarioResult(
            name="queries",
            status=outcome.status,
            metrics=outcome.metrics,
            details=outcome.details,
            duration_seconds=outcome.duration_seconds,
            artifacts=outcome.artifacts,
        )

    def _run_answer_suite(self, query_artifacts: List[Dict[str, Any]]) -> ScenarioResult:
        outcome = self._answer_runner.run(query_artifacts)
        return ScenarioResult(
            name="answers",
            status=outcome.status,
            metrics=outcome.metrics,
            details=outcome.details,
            duration_seconds=outcome.duration_seconds,
            artifacts=outcome.artifacts,
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

    def _build_scorecard(self, scenarios: Dict[str, ScenarioResult]) -> Dict[str, Any]:
        scenario_payload = {name: self._scenario_to_dict(result) for name, result in scenarios.items()}

        gating_names = ["boot", "code_quality", "ingestion", "queries", "answers", "extraction", "math", "perf"]
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
                "answer_judge_model": settings.answer_judge_model,
                "vlm_model": settings.vlm_model,
                "embedder_model": settings.embedder_model,
                "retrieval_k": self.config.retrieval_k,
                "ingest_time_per_page_budget_sec": self.config.ingest_time_per_page_budget_sec,
                "code_quality_paths": self._code_quality_env_paths(),
                "complexity_function_threshold": self.config.complexity_function_threshold,
                "complexity_average_threshold": self.config.complexity_average_threshold,
                "maintainability_threshold": self.config.maintainability_threshold,
                "answer_context_limit": self.config.answer_context_limit,
                "answer_max_tokens": self.config.answer_max_tokens,
                "answer_temperature": self.config.answer_temperature,
                "answer_similarity_threshold": self.config.answer_similarity_threshold,
                "answer_improvement_margin": self.config.answer_improvement_margin,
                "answer_improvement_rate_threshold": self.config.answer_improvement_rate_threshold,
            },
            "gates": gates,
            "metrics": metrics,
            "scenarios": scenario_payload,
        }
        return payload

    def _code_quality_env_paths(self) -> List[str]:
        paths: List[str] = []
        for path in self.config.code_quality_paths:
            try:
                paths.append(str(path.relative_to(self.config.project_root)))
            except ValueError:
                paths.append(str(path))
        return paths

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

    def _load_quality_fixture(self) -> Dict[str, Any]:
        path = self.config.fixtures_quality
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    return data
                logger.warning("Quality fixture %s is not a JSON object; ignoring", path)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in quality fixture %s: %s", path, exc)
        except OSError as exc:
            logger.warning("Unable to read quality fixture %s: %s", path, exc)
        return {}

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
