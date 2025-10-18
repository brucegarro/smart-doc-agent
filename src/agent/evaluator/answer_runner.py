"""Final answer generation and evaluation runner."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from agent.llm import LLMGenerationError, text_llm_client
from agent.retrieval.search import search_chunks

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent.evaluator.harness import EvaluatorConfig

logger = logging.getLogger(__name__)


@dataclass
class AnswerEvaluation:
    record_id: Optional[str]
    query: str
    context_snippets: List[str]
    baseline_answer: Optional[str]
    contextual_answer: Optional[str]
    baseline_similarity: Optional[float]
    contextual_similarity: Optional[float]
    improvement: Optional[float]
    baseline_latency_ms: Optional[float]
    contextual_latency_ms: Optional[float]
    error: Optional[str] = None
    missing_context: bool = False


@dataclass
class AnswerScenarioOutcome:
    status: str
    metrics: Dict[str, Any]
    details: List[str]
    duration_seconds: float
    artifacts: Dict[str, Any]


class AnswerScenarioRunner:
    """Generate answers with/without context and score them against gold responses."""

    def __init__(self, config: "EvaluatorConfig", load_jsonl) -> None:
        self._config = config
        self._load_jsonl = load_jsonl
        self._system_prompt = (
            "You are a precise research assistant. Answer questions about scientific papers in"
            " at most two sentences. Cite facts from the provided context when available; if"
            " the context is insufficient, say you do not know."
        )

    def run(self, query_artifacts: Sequence[Dict[str, Any]]) -> AnswerScenarioOutcome:
        fixtures = self._load_jsonl(self._config.fixtures_queries)
        evaluable = [item for item in fixtures if item.get("query") and item.get("gold_answers")]
        if not evaluable:
            logger.warning("Answer evaluation fixtures missing gold answers; skipping scenario")
            return AnswerScenarioOutcome(
                status="skipped",
                metrics={"fixtures_total": len(fixtures)},
                details=["no gold answers"],
                duration_seconds=0.0,
                artifacts={},
            )

        if not text_llm_client.ensure_client():
            logger.error("Text LLM client unavailable; cannot evaluate final answers")
            return AnswerScenarioOutcome(
                status="failed",
                metrics={},
                details=["text llm unavailable"],
                duration_seconds=0.0,
                artifacts={},
            )

        artifact_map = {
            artifact.get("id"): artifact
            for artifact in query_artifacts
            if isinstance(artifact, dict) and artifact.get("id")
        }

        start = time.perf_counter()
        evaluations: List[AnswerEvaluation] = []
        total_baseline_latency = 0.0
        total_context_latency = 0.0
        baseline_latency_count = 0
        context_latency_count = 0

        for record in evaluable:
            record_id = record.get("id")
            context_snippets = self._collect_context_snippets(record, artifact_map.get(record_id))
            evaluation = self._evaluate_record(record, context_snippets)
            evaluations.append(evaluation)
            if evaluation.baseline_latency_ms is not None:
                baseline_latency_count += 1
                total_baseline_latency += evaluation.baseline_latency_ms
            if evaluation.contextual_latency_ms is not None:
                context_latency_count += 1
                total_context_latency += evaluation.contextual_latency_ms

        baseline_scores = [ev.baseline_similarity for ev in evaluations if ev.baseline_similarity is not None]
        context_scores = [ev.contextual_similarity for ev in evaluations if ev.contextual_similarity is not None]
        improvements = [ev.improvement for ev in evaluations if ev.improvement is not None]
        avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
        avg_context = sum(context_scores) / len(context_scores) if context_scores else None
        mean_improvement = sum(improvements) / len(improvements) if improvements else None
        positive_improvements = [imp for imp in improvements if imp >= self._config.answer_improvement_margin]
        negative_improvements = [imp for imp in improvements if imp <= -self._config.answer_improvement_margin]
        improvement_rate = (len(positive_improvements) / len(improvements)) if improvements else None
        degrade_rate = (len(negative_improvements) / len(improvements)) if improvements else None

        metrics: Dict[str, Any] = {
            "fixtures_total": len(fixtures),
            "fixtures_evaluated": len(evaluable),
            "evaluations": len(evaluations),
            "baseline_similarity_avg": avg_baseline,
            "context_similarity_avg": avg_context,
            "mean_improvement": mean_improvement,
            "positive_improvement_rate": improvement_rate,
            "negative_improvement_rate": degrade_rate,
            "baseline_similarity_all": baseline_scores,
            "context_similarity_all": context_scores,
            "improvements_all": improvements,
            "baseline_latency_ms_avg": (total_baseline_latency / baseline_latency_count) if baseline_latency_count else None,
            "context_latency_ms_avg": (total_context_latency / context_latency_count) if context_latency_count else None,
            "generation_failures": len([ev for ev in evaluations if ev.error]),
            "missing_context": len([ev for ev in evaluations if ev.missing_context]),
        }

        violations: List[str] = []
        status = "passed"

        if avg_context is None:
            status = "failed"
            violations.append("no contextual answers generated")
        elif avg_context < self._config.answer_similarity_threshold:
            status = "failed"
            violations.append(
                f"avg similarity with context {avg_context:.3f}< {self._config.answer_similarity_threshold}"
            )

        if improvement_rate is None:
            status = "failed"
            violations.append("unable to compute improvement rate")
        elif improvement_rate < self._config.answer_improvement_rate_threshold:
            status = "failed"
            violations.append(
                f"context improvement rate {improvement_rate:.3f}< {self._config.answer_improvement_rate_threshold}"
            )

        if degrade_rate and degrade_rate > 0:
            violations.append(f"context degraded answers for {degrade_rate * 100:.1f}% of evaluated queries")
            if status != "failed":
                status = "warn"

        failures = metrics.get("generation_failures", 0)
        if failures:
            violations.append(f"{failures} generation attempts encountered errors")
            if status != "failed":
                status = "warn"

        missing_context = metrics.get("missing_context", 0)
        if missing_context:
            violations.append(f"missing context for {missing_context} queries")
            if status != "failed":
                status = "warn"

        details = violations or ["context answers met thresholds"]
        artifacts = {
            "evaluations": [self._evaluation_artifact(ev) for ev in evaluations],
        }

        duration = time.perf_counter() - start
        return AnswerScenarioOutcome(
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=duration,
            artifacts=artifacts,
        )

    def _collect_context_snippets(self, record: Dict[str, Any], artifact: Optional[Dict[str, Any]]) -> List[str]:
        snippets: List[str] = []
        limit = max(1, self._config.answer_context_limit)
        if artifact:
            raw_snippets = artifact.get("retrieved_snippets") or []
            for snippet in raw_snippets:
                if not snippet:
                    continue
                snippets.append(self._trim_text(str(snippet)))
                if len(snippets) >= limit:
                    break
        if snippets:
            return snippets
        query = record.get("query")
        if not query:
            return []
        try:
            fallback = search_chunks(query, limit=self._config.retrieval_k)
        except Exception as exc:  # noqa: BLE001 - reuse search errors best-effort
            logger.warning("Fallback retrieval failed for query '%s': %s", query, exc)
            return []
        target_doc = str(record.get("document_id") or "")
        prioritized: List[str] = []
        secondary: List[str] = []
        for chunk in fallback:
            text = chunk.snippet or chunk.content
            if not text:
                continue
            trimmed = self._trim_text(text)
            if target_doc and str(chunk.document_id) == target_doc:
                prioritized.append(trimmed)
            else:
                secondary.append(trimmed)
        for candidate in prioritized + secondary:
            snippets.append(candidate)
            if len(snippets) >= limit:
                break
        return snippets

    def _evaluate_record(self, record: Dict[str, Any], context_snippets: Sequence[str]) -> AnswerEvaluation:
        gold_answers = [self._normalize_text(ans) for ans in record.get("gold_answers", []) if ans]
        query = str(record.get("query", "")).strip()
        record_id = record.get("id")

        baseline_answer: Optional[str] = None
        contextual_answer: Optional[str] = None
        baseline_latency: Optional[float] = None
        contextual_latency: Optional[float] = None
        errors: List[str] = []
        missing_context = False

        try:
            baseline_answer, baseline_latency = self._generate_answer(query, [])
        except LLMGenerationError as exc:
            errors.append(f"baseline:{exc}")
        except Exception as exc:  # noqa: BLE001 - defensive guard
            errors.append(f"baseline:{exc}")

        if context_snippets:
            try:
                contextual_answer, contextual_latency = self._generate_answer(query, context_snippets)
            except LLMGenerationError as exc:
                errors.append(f"context:{exc}")
            except Exception as exc:  # noqa: BLE001 - defensive guard
                errors.append(f"context:{exc}")
        else:
            missing_context = True

        baseline_similarity = self._best_similarity(baseline_answer, gold_answers) if baseline_answer else None
        contextual_similarity = self._best_similarity(contextual_answer, gold_answers) if contextual_answer else None
        improvement = None
        if baseline_similarity is not None and contextual_similarity is not None:
            improvement = contextual_similarity - baseline_similarity

        evaluation = AnswerEvaluation(
            record_id=record_id,
            query=query,
            context_snippets=[self._trim_text(snippet) for snippet in context_snippets],
            baseline_answer=self._trim_text(baseline_answer) if baseline_answer else None,
            contextual_answer=self._trim_text(contextual_answer) if contextual_answer else None,
            baseline_similarity=baseline_similarity,
            contextual_similarity=contextual_similarity,
            improvement=improvement,
            baseline_latency_ms=baseline_latency,
            contextual_latency_ms=contextual_latency,
            error=";".join(errors) if errors else None,
            missing_context=missing_context,
        )
        return evaluation

    def _generate_answer(self, query: str, context_snippets: Sequence[str]) -> tuple[str, float]:
        prompt = self._build_prompt(query, context_snippets)
        start = time.perf_counter()
        response = text_llm_client.generate(
            prompt,
            system=self._system_prompt,
            temperature=self._config.answer_temperature,
            max_tokens=self._config.answer_max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return response, latency_ms

    def _build_prompt(self, query: str, context_snippets: Sequence[str]) -> str:
        lines = [
            "Answer the question about the research paper succinctly.",
            "If the context is insufficient, respond with 'I don't know.'",
        ]
        if context_snippets:
            lines.append("Context snippets:")
            for idx, snippet in enumerate(context_snippets, start=1):
                lines.append(f"{idx}. {snippet}")
        else:
            lines.append("No context snippets are available. Rely on prior knowledge but note uncertainty if unsure.")
        lines.append(f"Question: {query}")
        lines.append("Answer:")
        return "\n".join(lines)

    def _best_similarity(self, answer: Optional[str], normalized_gold: Sequence[str]) -> Optional[float]:
        if not answer or not normalized_gold:
            return None
        candidate = self._normalize_text(answer)
        if not candidate:
            return 0.0
        best = 0.0
        for gold in normalized_gold:
            if not gold:
                continue
            if gold in candidate:
                return 1.0
            best = max(best, SequenceMatcher(None, candidate, gold).ratio())
        return best

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _trim_text(self, text: str, limit: int = 320) -> str:
        if not text:
            return ""
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[: limit - 3].rstrip() + "..."

    def _evaluation_artifact(self, evaluation: AnswerEvaluation) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": evaluation.record_id,
            "query": evaluation.query,
            "context_snippets": evaluation.context_snippets,
            "baseline_answer": evaluation.baseline_answer,
            "contextual_answer": evaluation.contextual_answer,
            "baseline_similarity": evaluation.baseline_similarity,
            "contextual_similarity": evaluation.contextual_similarity,
            "improvement": evaluation.improvement,
            "baseline_latency_ms": evaluation.baseline_latency_ms,
            "contextual_latency_ms": evaluation.contextual_latency_ms,
            "missing_context": evaluation.missing_context,
        }
        if evaluation.error:
            payload["error"] = evaluation.error
        return payload
