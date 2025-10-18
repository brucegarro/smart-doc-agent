"""Final answer generation and evaluation runner."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
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
    gold_answer: Optional[str]
    context_snippets: List[str]
    baseline_answer: Optional[str]
    contextual_answer: Optional[str]
    baseline_similarity: Optional[float]
    contextual_similarity: Optional[float]
    improvement: Optional[float]
    baseline_latency_ms: Optional[float]
    contextual_latency_ms: Optional[float]
    contextual_judge_score: Optional[float]
    contextual_judge_reason: Optional[str]
    error: Optional[str] = None
    missing_context: bool = False


@dataclass
class AnswerScenarioOutcome:
    status: str
    metrics: Dict[str, Any]
    details: List[str]
    duration_seconds: float
    artifacts: Dict[str, Any]


@dataclass
class _MetricsAggregator:
    config: "EvaluatorConfig"
    fixtures_total: int
    fixtures_evaluated: int
    evaluations: List[AnswerEvaluation] = field(default_factory=list)
    baseline_latency_sum: float = 0.0
    baseline_latency_count: int = 0
    contextual_latency_sum: float = 0.0
    contextual_latency_count: int = 0
    judge_scores: List[float] = field(default_factory=list)

    def add(self, evaluation: AnswerEvaluation) -> None:
        self.evaluations.append(evaluation)
        if evaluation.baseline_latency_ms is not None:
            self.baseline_latency_sum += evaluation.baseline_latency_ms
            self.baseline_latency_count += 1
        if evaluation.contextual_latency_ms is not None:
            self.contextual_latency_sum += evaluation.contextual_latency_ms
            self.contextual_latency_count += 1
        if evaluation.contextual_judge_score is not None:
            self.judge_scores.append(evaluation.contextual_judge_score)

    def build_metrics(self) -> Dict[str, Any]:
        baseline_scores, context_scores, improvements = self._score_vectors()
        judge_scores = self.judge_scores
        positive_margin = self.config.answer_improvement_margin
        positives = [value for value in improvements if value >= positive_margin]
        negatives = [value for value in improvements if value <= -positive_margin]
        metrics: Dict[str, Any] = {
            "fixtures_total": self.fixtures_total,
            "fixtures_evaluated": self.fixtures_evaluated,
            "evaluations": len(self.evaluations),
            "baseline_similarity_all": baseline_scores,
            "context_similarity_all": context_scores,
            "improvements_all": improvements,
            "contextual_judge_scores": judge_scores,
            "baseline_similarity_avg": self._average(baseline_scores),
            "context_similarity_avg": self._average(context_scores),
            "contextual_judge_avg": self._average(judge_scores),
            "mean_improvement": self._average(improvements),
            "positive_improvement_rate": self._rate(positives, improvements),
            "negative_improvement_rate": self._rate(negatives, improvements),
            "baseline_latency_ms_avg": self._latency_avg(self.baseline_latency_sum, self.baseline_latency_count),
            "context_latency_ms_avg": self._latency_avg(self.contextual_latency_sum, self.contextual_latency_count),
            "generation_failures": sum(1 for item in self.evaluations if item.error),
            "missing_context": sum(1 for item in self.evaluations if item.missing_context),
            "contextual_judge_pass_rate": self._pass_rate(
                [score for score in judge_scores if score >= self.config.answer_judge_pass_threshold],
                judge_scores,
            ),
        }
        return metrics

    def _score_vectors(self) -> tuple[List[float], List[float], List[float]]:
        baseline = [item.baseline_similarity for item in self.evaluations if item.baseline_similarity is not None]
        contextual = [item.contextual_similarity for item in self.evaluations if item.contextual_similarity is not None]
        improvements = [item.improvement for item in self.evaluations if item.improvement is not None]
        return baseline, contextual, improvements

    def _average(self, values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    def _rate(self, subset: Sequence[float], universe: Sequence[float]) -> Optional[float]:
        if not universe:
            return None
        return len(subset) / len(universe)

    def _latency_avg(self, latency_sum: float, count: int) -> Optional[float]:
        if not count:
            return None
        return latency_sum / count

    def _pass_rate(self, subset: Sequence[float], universe: Sequence[float]) -> Optional[float]:
        if not universe:
            return None
        return len(subset) / len(universe)


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
        self._judge_system_prompt = (
            "You are an expert evaluator. Compare the candidate answer to the provided gold"
            " answer(s) and decide if the candidate is correct."
        )

    def run(self, query_artifacts: Sequence[Dict[str, Any]]) -> AnswerScenarioOutcome:
        fixtures = self._load_jsonl(self._config.fixtures_queries)
        evaluable = self._select_evaluable_fixtures(fixtures)
        if not evaluable:
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

        start = time.perf_counter()
        artifact_index = self._build_artifact_index(query_artifacts)
        aggregator = _MetricsAggregator(
            config=self._config,
            fixtures_total=len(fixtures),
            fixtures_evaluated=len(evaluable),
        )

        for record in evaluable:
            record_id = record.get("id")
            context_snippets = self._collect_context_snippets(record, artifact_index.get(record_id))
            evaluation = self._evaluate_record(record, context_snippets)
            aggregator.add(evaluation)

        metrics = aggregator.build_metrics()
        status, details = self._derive_status(metrics)
        duration = time.perf_counter() - start
        artifacts = {
            "evaluations": [self._evaluation_artifact(item) for item in aggregator.evaluations],
        }
        return AnswerScenarioOutcome(
            status=status,
            metrics=metrics,
            details=details,
            duration_seconds=duration,
            artifacts=artifacts,
        )

    def _collect_context_snippets(self, record: Dict[str, Any], artifact: Optional[Dict[str, Any]]) -> List[str]:
        limit = max(1, self._config.answer_context_limit)
        artifact_snippets = self._artifact_snippets(artifact, limit)
        if artifact_snippets:
            return artifact_snippets
        return self._fallback_snippets(record, limit)

    def _evaluate_record(self, record: Dict[str, Any], context_snippets: Sequence[str]) -> AnswerEvaluation:
        raw_gold_answers = [str(ans) for ans in record.get("gold_answers", []) if ans]
        gold_answers = [self._normalize_text(ans) for ans in raw_gold_answers]
        gold_display = self._trim_text(raw_gold_answers[0]) if raw_gold_answers else None
        query = str(record.get("query", "")).strip()
        record_id = record.get("id")

        baseline_answer, baseline_latency, baseline_error = self._safe_generate(query, [], "baseline")
        contextual_answer: Optional[str] = None
        contextual_latency: Optional[float] = None
        contextual_error: Optional[str] = None
        missing_context = False

        if context_snippets:
            contextual_answer, contextual_latency, contextual_error = self._safe_generate(
                query,
                context_snippets,
                "context",
            )
        else:
            missing_context = True

        baseline_similarity = self._best_similarity(baseline_answer, gold_answers) if baseline_answer else None
        contextual_similarity = self._best_similarity(contextual_answer, gold_answers) if contextual_answer else None
        improvement = None
        if baseline_similarity is not None and contextual_similarity is not None:
            improvement = contextual_similarity - baseline_similarity

        contextual_judge_score = None
        contextual_judge_reason = None
        judge_error: Optional[str] = None
        if contextual_answer and raw_gold_answers:
            contextual_judge_score, contextual_judge_reason, judge_error = self._judge_answer(
                query,
                raw_gold_answers,
                contextual_answer,
            )

        errors = [message for message in (baseline_error, contextual_error) if message]
        if judge_error:
            errors.append(judge_error)
        return AnswerEvaluation(
            record_id=record_id,
            query=query,
            gold_answer=gold_display,
            context_snippets=[self._trim_text(snippet) for snippet in context_snippets],
            baseline_answer=self._trim_text(baseline_answer) if baseline_answer else None,
            contextual_answer=self._trim_text(contextual_answer) if contextual_answer else None,
            baseline_similarity=baseline_similarity,
            contextual_similarity=contextual_similarity,
            improvement=improvement,
            baseline_latency_ms=baseline_latency,
            contextual_latency_ms=contextual_latency,
            contextual_judge_score=contextual_judge_score,
            contextual_judge_reason=contextual_judge_reason,
            error=";".join(errors) if errors else None,
            missing_context=missing_context,
        )

    def _safe_generate(
        self, query: str, context_snippets: Sequence[str], label: str
    ) -> tuple[Optional[str], Optional[float], Optional[str]]:
        try:
            answer, latency = self._generate_answer(query, context_snippets)
            return answer, latency, None
        except LLMGenerationError as exc:
            return None, None, f"{label}:{exc}"
        except Exception as exc:  # noqa: BLE001 - defensive guard
            return None, None, f"{label}:{exc}"

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

    def _judge_answer(
        self,
        query: str,
        gold_answers: Sequence[str],
        candidate_answer: str,
    ) -> tuple[Optional[float], Optional[str], Optional[str]]:
        prompt = self._build_judge_prompt(query, gold_answers, candidate_answer)
        try:
            response = text_llm_client.generate(
                prompt,
                system=self._judge_system_prompt,
                temperature=0.0,
                max_tokens=128,
            )
        except LLMGenerationError as exc:
            return None, None, f"judge:{exc}"
        except Exception as exc:  # noqa: BLE001 - defensive guard
            return None, None, f"judge:{exc}"

        score, rationale = self._parse_judge_response(response)
        if score is None:
            return None, rationale, "judge:unable to parse score"
        return score, rationale, None

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

    def _build_judge_prompt(self, query: str, gold_answers: Sequence[str], candidate_answer: str) -> str:
        lines = [
            "Question:",
            query,
            "",
            "Gold answer(s):",
        ]
        for idx, answer in enumerate(gold_answers, start=1):
            lines.append(f"{idx}. {answer}")
        lines.extend(
            [
                "",
                "Candidate answer:",
                candidate_answer,
                "",
                "Respond with a single line of JSON in the format",
                '{"score": <0 to 1>, "explanation": "..."}.',
                "Score 1 if the candidate is fully correct and grounded in the gold answer,",
                "0 if it is incorrect, and use intermediate values when it is partially correct.",
            ]
        )
        return "\n".join(lines)

    def _parse_judge_response(self, response: str) -> tuple[Optional[float], Optional[str]]:
        text = response.strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to recover from extra text by finding first JSON object.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    payload = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None, text
            else:
                return None, text

        score = payload.get("score")
        explanation = payload.get("explanation")
        try:
            if score is not None:
                score = float(score)
                score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return None, text
        return score, explanation

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
            "gold_answer": evaluation.gold_answer,
            "context_snippets": evaluation.context_snippets,
            "baseline_answer": evaluation.baseline_answer,
            "contextual_answer": evaluation.contextual_answer,
            "baseline_similarity": evaluation.baseline_similarity,
            "contextual_similarity": evaluation.contextual_similarity,
            "improvement": evaluation.improvement,
            "baseline_latency_ms": evaluation.baseline_latency_ms,
            "contextual_latency_ms": evaluation.contextual_latency_ms,
            "contextual_judge_score": evaluation.contextual_judge_score,
            "contextual_judge_reason": evaluation.contextual_judge_reason,
            "missing_context": evaluation.missing_context,
        }
        if evaluation.error:
            payload["error"] = evaluation.error
        return payload

    def _select_evaluable_fixtures(self, fixtures: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        evaluable = [item for item in fixtures if item.get("query") and item.get("gold_answers")]
        if not evaluable:
            logger.warning("Answer evaluation fixtures missing gold answers; skipping scenario")
        return evaluable

    def _build_artifact_index(
        self, query_artifacts: Sequence[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        return {
            artifact.get("id"): artifact
            for artifact in query_artifacts
            if isinstance(artifact, dict) and artifact.get("id")
        }

    def _derive_status(self, metrics: Dict[str, Any]) -> tuple[str, List[str]]:
        observations: List[str] = []
        status = "passed"

        avg_context = metrics.get("context_similarity_avg")
        improvement_rate = metrics.get("positive_improvement_rate")
        degrade_rate = metrics.get("negative_improvement_rate")
        judge_pass_rate = metrics.get("contextual_judge_pass_rate")

        if avg_context is None:
            status = "failed"
            observations.append("no contextual answers generated")
        else:
            observations.append(f"avg similarity with context {avg_context:.3f}")

        if improvement_rate is None:
            status = "failed"
            observations.append("unable to compute improvement rate")
        elif improvement_rate < self._config.answer_improvement_rate_threshold:
            status = "failed"
            observations.append(
                f"context improvement rate {improvement_rate:.3f}< {self._config.answer_improvement_rate_threshold}"
            )
        else:
            observations.append(f"context improvement rate {improvement_rate:.3f}")

        if judge_pass_rate is None:
            status = "failed"
            observations.append("judge could not score contextual answers")
        elif judge_pass_rate < self._config.answer_judge_pass_rate_threshold:
            status = "failed"
            observations.append(
                f"judge pass rate {judge_pass_rate:.3f}< {self._config.answer_judge_pass_rate_threshold}"
            )
        else:
            observations.append(f"judge pass rate {judge_pass_rate:.3f}")

        if degrade_rate:
            observations.append(f"context degraded answers for {degrade_rate * 100:.1f}% of evaluated queries")
            if status == "passed":
                status = "warn"

        failures = metrics.get("generation_failures", 0)
        if failures:
            observations.append(f"{failures} generation attempts encountered errors")
            if status == "passed":
                status = "warn"

        missing_context = metrics.get("missing_context", 0)
        if missing_context:
            observations.append(f"missing context for {missing_context} queries")
            if status == "passed":
                status = "warn"

        return status, observations or ["context answers met thresholds"]

    def _artifact_snippets(self, artifact: Optional[Dict[str, Any]], limit: int) -> List[str]:
        if not artifact:
            return []
        snippets: List[str] = []
        for raw_snippet in artifact.get("retrieved_snippets") or []:
            if not raw_snippet:
                continue
            snippets.append(self._trim_text(str(raw_snippet)))
            if len(snippets) >= limit:
                break
        return snippets

    def _fallback_snippets(self, record: Dict[str, Any], limit: int) -> List[str]:
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

        snippets: List[str] = []
        for candidate in prioritized + secondary:
            snippets.append(candidate)
            if len(snippets) >= limit:
                break
        return snippets
