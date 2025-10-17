"""Query scenario execution helpers for the evaluator harness."""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

from agent.retrieval.search import ChunkResult, search_chunks

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from agent.evaluator.harness import EvaluatorConfig


logger = logging.getLogger(__name__)


@dataclass
class QueryEvaluation:
    record_id: Optional[str]
    query: str
    latency_ms: float
    hit: Optional[float]
    ndcg: Optional[float]
    similarity_scores: Optional[List[float]]
    best_similarity: Optional[float]
    retrieved_fingerprints: List[str]
    retrieved_snippets: List[str]
    match_strategy: str
    error: Optional[str] = None


@dataclass
class QueryCollections:
    latencies: List[float]
    hits: List[float]
    ndcgs: List[float]
    best_similarity_scores: List[float]


@dataclass
class QueryScenarioOutcome:
    status: str
    metrics: Dict[str, Any]
    details: List[str]
    duration_seconds: float
    artifacts: Dict[str, Any]
    latencies_ms: List[float]


class QueryScenarioRunner:
    """Coordinate query scenario execution and metric aggregation."""

    def __init__(
        self,
    config: "EvaluatorConfig",
    load_jsonl: Callable[[Path], List[Dict[str, Any]]],
    percentile_fn: Callable[[List[float], float], Optional[float]],
    ) -> None:
        self._config = config
        self._load_jsonl = load_jsonl
        self._percentile = percentile_fn

    def run(self, doc_ids: List[str]) -> QueryScenarioOutcome:
        skip_outcome = self._maybe_skip(doc_ids)
        if skip_outcome is not None:
            return skip_outcome

        fixtures = self._prepare_fixtures()
        if not fixtures:
            logger.warning("Query fixture file empty; skipping queries scenario")
            return QueryScenarioOutcome(
                status="skipped",
                metrics={},
                details=["no fixtures"],
                duration_seconds=0.0,
                artifacts={},
                latencies_ms=[],
            )

        logger.info("Executing %s query fixture(s)", len(fixtures))

        evaluations, collections = self._execute_queries(fixtures)
        metrics = self._aggregate_metrics(evaluations, collections)
        status, details = self._query_status(metrics, collections)

        artifacts = {
            "queries": [self._query_artifact(evaluation) for evaluation in evaluations]
        }
        duration = sum(collections.latencies) / 1000 if collections.latencies else 0.0

        return QueryScenarioOutcome(
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=duration,
            artifacts=artifacts,
            latencies_ms=collections.latencies,
        )

    def _maybe_skip(self, doc_ids: List[str]) -> Optional[QueryScenarioOutcome]:
        if not doc_ids:
            logger.warning("No ingested documents available; skipping query evaluation")
            return QueryScenarioOutcome(
                status="skipped",
                metrics={},
                details=["no documents"],
                duration_seconds=0.0,
                artifacts={},
                latencies_ms=[],
            )

        if not self._config.fixtures_queries.exists():
            logger.warning("Query fixtures missing at %s", self._config.fixtures_queries)
            return QueryScenarioOutcome(
                status="skipped",
                metrics={},
                details=["no fixtures"],
                duration_seconds=0.0,
                artifacts={},
                latencies_ms=[],
            )
        return None

    def _prepare_fixtures(self) -> List[Dict[str, Any]]:
        fixtures = self._load_jsonl(self._config.fixtures_queries)
        return [item for item in fixtures if item.get("query")]

    def _execute_queries(self, fixtures: Sequence[Dict[str, Any]]) -> Tuple[List[QueryEvaluation], QueryCollections]:
        evaluations = [self._evaluate_record(record, self._config.retrieval_k) for record in fixtures]
        latencies = [evaluation.latency_ms for evaluation in evaluations]
        hits = [evaluation.hit for evaluation in evaluations if evaluation.hit is not None]
        ndcgs = [evaluation.ndcg for evaluation in evaluations if evaluation.ndcg is not None]
        best_similarity_scores = [
            evaluation.best_similarity for evaluation in evaluations if evaluation.best_similarity is not None
        ]

        collections = QueryCollections(
            latencies=latencies,
            hits=hits,
            ndcgs=ndcgs,
            best_similarity_scores=best_similarity_scores,
        )
        return evaluations, collections

    def _evaluate_record(self, record: Dict[str, Any], k: int) -> QueryEvaluation:
        query = record["query"]
        gold_passages: List[str] = [str(text) for text in record.get("gold_passages", []) if text]
        gold_chunks: List[str] = [str(cid) for cid in record.get("gold_chunk_ids", [])]
        gold_set = set(gold_chunks)
        use_text_matching = bool(gold_passages)
        gold_available = bool(gold_passages if use_text_matching else gold_set)

        start = time.perf_counter()
        try:
            results = search_chunks(query, limit=k)
        except Exception as exc:  # noqa: BLE001 - log and continue
            latency = (time.perf_counter() - start) * 1000
            detail = f"error:{record.get('id', query[:32])}:{exc}"[:256]
            logger.exception("Query execution failed: %s", detail)
            return QueryEvaluation(
                record_id=record.get("id"),
                query=query,
                latency_ms=latency,
                hit=None,
                ndcg=None,
                similarity_scores=None,
                best_similarity=None,
                retrieved_fingerprints=[],
                retrieved_snippets=[],
                match_strategy="error",
                error=str(exc),
            )

        latency = (time.perf_counter() - start) * 1000

        retrieved_fingerprints = [res.fingerprint for res in results]
        retrieved_snippets = [res.snippet for res in results]

        hit: Optional[float] = None
        ndcg: Optional[float] = None
        similarity_scores: Optional[List[float]] = None
        match_strategy = "chunk_id"
        best_similarity: Optional[float] = None

        if use_text_matching:
            match_strategy = "passage"
            similarity_scores = self._scores_for_text_matches(results, gold_passages)
            hit = self._hit_from_scores(similarity_scores, self._config.retrieval_text_match_threshold, k)
            ndcg = self._ndcg_from_scores(similarity_scores, k)
            if similarity_scores:
                best_similarity = max(similarity_scores)
        elif gold_available:
            retrieved_chunk_ids = [res.chunk_id for res in results]
            hit = self._hit_at_k(retrieved_chunk_ids, gold_set, k)
            ndcg = self._ndcg_at_k(retrieved_chunk_ids, gold_set, k)

        return QueryEvaluation(
            record_id=record.get("id"),
            query=query,
            latency_ms=latency,
            hit=hit,
            ndcg=ndcg,
            similarity_scores=similarity_scores,
            best_similarity=best_similarity,
            retrieved_fingerprints=retrieved_fingerprints,
            retrieved_snippets=retrieved_snippets,
            match_strategy=match_strategy,
            error=None,
        )

    def _aggregate_metrics(self, evaluations: Sequence[QueryEvaluation], collections: QueryCollections) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "queries_run": len(evaluations),
            "latency_ms_all": list(collections.latencies),
            "latency_p50_ms": self._percentile(list(collections.latencies), 0.5),
            "latency_p95_ms": self._percentile(list(collections.latencies), 0.95),
            "hit_at_k_avg": (sum(collections.hits) / len(collections.hits)) if collections.hits else None,
            "ndcg_at_k_avg": (sum(collections.ndcgs) / len(collections.ndcgs)) if collections.ndcgs else None,
            "top_similarity_avg": (sum(collections.best_similarity_scores) / len(collections.best_similarity_scores))
            if collections.best_similarity_scores
            else None,
            "top_similarity_all": list(collections.best_similarity_scores),
        }
        return metrics

    def _query_status(self, metrics: Dict[str, Any], collections: QueryCollections) -> Tuple[str, List[str]]:
        violations = self._query_threshold_violations(metrics)
        status = self._status_from_violations(violations)
        details = [message for _, message in violations]
        return self._apply_missing_gold(status, details, metrics["queries_run"], collections.hits, collections.ndcgs)

    def _status_from_violations(self, violations: Sequence[Tuple[str, str]]) -> str:
        status = "passed"
        for severity, _ in violations:
            if severity == "fail":
                return "failed"
            if severity == "warn":
                status = "warn"
        return status

    def _apply_missing_gold(
        self,
        status: str,
        details: List[str],
        queries_run: int,
        hits: Sequence[float],
        ndcgs: Sequence[float],
    ) -> Tuple[str, List[str]]:
        if hits or ndcgs:
            return status, details
        if queries_run == 0:
            return "skipped", details
        if status != "failed":
            status = "warn"
        details.append("no gold references provided; metrics skipped")
        return status, details

    def _query_threshold_violations(self, metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
        violations: List[Tuple[str, str]] = []
        k = self._config.retrieval_k

        hit_avg = metrics.get("hit_at_k_avg")
        if hit_avg is not None and hit_avg < self._config.retrieval_hit_threshold:
            violations.append(
                (
                    "fail",
                    f"hit@{k} below threshold {hit_avg:.3f}< {self._config.retrieval_hit_threshold}",
                )
            )

        ndcg_avg = metrics.get("ndcg_at_k_avg")
        if ndcg_avg is not None and ndcg_avg < self._config.retrieval_ndcg_threshold:
            violations.append(
                (
                    "fail",
                    f"ndcg@{k} below threshold {ndcg_avg:.3f}< {self._config.retrieval_ndcg_threshold}",
                )
            )

        similarity_avg = metrics.get("top_similarity_avg")
        if similarity_avg is not None and similarity_avg < self._config.retrieval_text_match_threshold:
            violations.append(
                (
                    "fail",
                    f"avg top similarity {similarity_avg:.3f}< {self._config.retrieval_text_match_threshold}",
                )
            )

        latency_p95 = metrics.get("latency_p95_ms")
        if latency_p95 is not None and latency_p95 > self._config.query_latency_budget_ms:
            violations.append(
                (
                    "warn",
                    f"p95 latency {latency_p95:.1f}ms exceeds budget {self._config.query_latency_budget_ms}ms",
                )
            )

        return violations

    def _query_artifact(self, evaluation: QueryEvaluation) -> Dict[str, Any]:
        artifact: Dict[str, Any] = {
            "id": evaluation.record_id,
            "query": evaluation.query,
            "latency_ms": evaluation.latency_ms,
            "retrieved_fingerprints": evaluation.retrieved_fingerprints,
            "retrieved_snippets": evaluation.retrieved_snippets,
            "match_strategy": evaluation.match_strategy,
            "hit_at_k": evaluation.hit,
            "ndcg_at_k": evaluation.ndcg,
        }
        if evaluation.similarity_scores is not None:
            artifact["similarity_scores"] = evaluation.similarity_scores
        if evaluation.best_similarity is not None:
            artifact["top_similarity"] = evaluation.best_similarity
        if evaluation.error:
            artifact["error"] = evaluation.error
        return artifact

    def _scores_for_text_matches(self, results: Sequence[ChunkResult], gold_passages: Sequence[str]) -> List[float]:
        normalized_gold = [self._normalize_text(passage) for passage in gold_passages if passage]
        scores: List[float] = []
        for chunk in results:
            candidate = self._normalize_text(chunk.content or chunk.snippet)
            scores.append(self._best_similarity(candidate, normalized_gold))
        return scores

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _best_similarity(self, candidate: str, normalized_gold: Sequence[str]) -> float:
        if not candidate or not normalized_gold:
            return 0.0
        best = 0.0
        for reference in normalized_gold:
            if not reference:
                continue
            if reference in candidate:
                return 1.0
            best = max(best, self._text_similarity(candidate, reference))
        return best

    def _text_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()

    def _hit_from_scores(self, scores: Sequence[float], threshold: float, k: int) -> Optional[float]:
        if not scores:
            return None
        top_scores = list(scores[:k])
        if not top_scores:
            return None
        return 1.0 if any(score >= threshold for score in top_scores) else 0.0

    def _ndcg_from_scores(self, scores: Sequence[float], k: int) -> Optional[float]:
        if not scores:
            return None
        top_scores = list(scores[:k])
        if not top_scores:
            return None
        dcg = sum(score / math.log2(idx + 2) for idx, score in enumerate(top_scores) if score > 0)
        ideal = sorted((score for score in top_scores if score > 0), reverse=True)
        if not ideal:
            return 0.0
        idcg = sum(score / math.log2(idx + 2) for idx, score in enumerate(ideal))
        if idcg == 0:
            return 0.0
        return dcg / idcg

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
