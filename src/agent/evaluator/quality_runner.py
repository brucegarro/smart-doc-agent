"""Code quality evaluation runner for the evaluator harness."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from radon.complexity import cc_rank, cc_visit
from radon.metrics import mi_visit

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from agent.evaluator.harness import EvaluatorConfig


@dataclass
class QualityThresholds:
    function: float
    average: float
    maintainability: float


@dataclass
class ComplexityRecord:
    score: float
    path: Path
    name: str
    lineno: int


@dataclass
class MaintainabilityRecord:
    score: float
    path: Path


@dataclass
class QualityAnalysis:
    analyzed_files: List[Path] = field(default_factory=list)
    complexities: List[ComplexityRecord] = field(default_factory=list)
    maintainabilities: List[MaintainabilityRecord] = field(default_factory=list)
    read_errors: List[str] = field(default_factory=list)
    missing_paths: List[str] = field(default_factory=list)


@dataclass
class QualitySummary:
    complexity_values: List[float] = field(default_factory=list)
    maintainability_values: List[float] = field(default_factory=list)
    avg_complexity: float = 0.0
    max_complexity: float = 0.0
    worst_rank: Optional[str] = None
    min_maintainability: Optional[float] = None
    avg_maintainability: Optional[float] = None


@dataclass
class QualityScenarioOutcome:
    status: str
    details: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


class CodeQualityRunner:
    """Encapsulates Radon-driven code quality analysis for the evaluator."""

    def __init__(self, config: "EvaluatorConfig") -> None:
        self.config = config

    def evaluate(self, fixture: Any) -> QualityScenarioOutcome:
        candidate_paths, ignore_patterns, thresholds = self._extract_quality_settings(fixture)
        analysis = self._collect_quality_analysis(candidate_paths, ignore_patterns)
        status, details, metrics, artifacts = self._evaluate_quality_results(
            analysis, thresholds, ignore_patterns
        )
        return QualityScenarioOutcome(status=status, details=details, metrics=metrics, artifacts=artifacts)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _extract_quality_settings(self, fixture: Any) -> Tuple[List[Path], List[str], QualityThresholds]:
        path_tokens = fixture.get("paths") if isinstance(fixture, dict) else None
        ignore_patterns: List[str] = []
        if isinstance(fixture, dict):
            raw_patterns = fixture.get("ignore", [])
            if isinstance(raw_patterns, list):
                ignore_patterns = [
                    pattern.strip()
                    for pattern in raw_patterns
                    if isinstance(pattern, str) and pattern.strip()
                ]
        candidate_paths = (
            self._normalize_quality_paths(path_tokens)
            if isinstance(path_tokens, list)
            else list(self.config.code_quality_paths)
        )
        thresholds = self._quality_thresholds_from_fixture(fixture)
        return candidate_paths, ignore_patterns, thresholds

    def _normalize_quality_paths(self, tokens: Sequence[Any]) -> List[Path]:
        paths: List[Path] = []
        for token in tokens:
            if not isinstance(token, str):
                continue
            raw = token.strip()
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.config.project_root / candidate
            try:
                candidate = candidate.resolve()
            except OSError:
                continue
            if candidate not in paths:
                paths.append(candidate)
        return paths

    def _quality_thresholds_from_fixture(self, fixture: Any) -> QualityThresholds:
        base = QualityThresholds(
            function=self.config.complexity_function_threshold,
            average=self.config.complexity_average_threshold,
            maintainability=self.config.maintainability_threshold,
        )
        if not isinstance(fixture, dict):
            return base
        return QualityThresholds(
            function=self._safe_float(fixture.get("max_function_ccn"), base.function),
            average=self._safe_float(fixture.get("max_average_ccn"), base.average),
            maintainability=self._safe_float(fixture.get("min_maintainability_index"), base.maintainability),
        )

    # ------------------------------------------------------------------
    # Analysis collection
    # ------------------------------------------------------------------
    def _collect_quality_analysis(self, paths: Sequence[Path], ignore_patterns: Sequence[str]) -> QualityAnalysis:
        analysis = QualityAnalysis()
        seen: set[Path] = set()
        for directory in paths:
            if not directory.exists():
                analysis.missing_paths.append(str(directory))
                continue
            for file_path in directory.rglob("*.py"):
                resolved = file_path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                relative = self._relative_quality_path(file_path)
                if self._should_ignore_path(relative, ignore_patterns):
                    continue
                source = self._read_source_text(file_path, relative, analysis)
                if source is None:
                    continue
                analysis.analyzed_files.append(file_path)
                self._record_complexity(source, relative, analysis)
                self._record_maintainability(source, relative, analysis)
        return analysis

    def _relative_quality_path(self, file_path: Path) -> Path:
        try:
            return file_path.relative_to(self.config.project_root)
        except ValueError:
            return file_path.resolve()

    def _should_ignore_path(self, relative: Path, ignore_patterns: Sequence[str]) -> bool:
        if not ignore_patterns:
            return False
        relative_str = str(relative)
        return any(fnmatch.fnmatch(relative_str, pattern) for pattern in ignore_patterns)

    def _read_source_text(self, file_path: Path, relative: Path, analysis: QualityAnalysis) -> Optional[str]:
        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            analysis.read_errors.append(f"{relative}:{exc}")
            return None

    def _record_complexity(self, source: str, relative: Path, analysis: QualityAnalysis) -> None:
        try:
            for block in cc_visit(source):
                analysis.complexities.append(
                    ComplexityRecord(score=block.complexity, path=relative, name=block.name, lineno=block.lineno)
                )
        except Exception as exc:  # noqa: BLE001 - surface radon failures in artifacts
            analysis.read_errors.append(f"cc:{relative}:{exc}")

    def _record_maintainability(self, source: str, relative: Path, analysis: QualityAnalysis) -> None:
        try:
            score = mi_visit(source, True)
        except Exception as exc:  # noqa: BLE001 - capture radon parser failures
            analysis.read_errors.append(f"mi:{relative}:{exc}")
            return
        analysis.maintainabilities.append(MaintainabilityRecord(score=score, path=relative))

    # ------------------------------------------------------------------
    # Result evaluation
    # ------------------------------------------------------------------
    def _evaluate_quality_results(
        self,
        analysis: QualityAnalysis,
        thresholds: QualityThresholds,
        ignore_patterns: Sequence[str],
    ) -> Tuple[str, List[str], Dict[str, Any], Dict[str, Any]]:
        summary = self._summarize_quality_analysis(analysis)
        complexity_violations, maintainability_violations = self._quality_violations(analysis, summary, thresholds)
        status, details = self._quality_status(analysis, summary, thresholds)
        metrics = self._quality_metrics(summary, thresholds, analysis)
        artifacts = {
            "complexity_violations": complexity_violations,
            "maintainability_violations": maintainability_violations,
            "ignore_patterns": list(ignore_patterns),
            "analysis_errors": analysis.read_errors,
            "missing_paths": analysis.missing_paths,
        }

        if not details and analysis.analyzed_files:
            details.append(f"analyzed {len(analysis.analyzed_files)} python file(s)")

        return status, details, metrics, artifacts

    def _summarize_quality_analysis(self, analysis: QualityAnalysis) -> QualitySummary:
        complexity_values = [record.score for record in analysis.complexities]
        maintainability_values = [record.score for record in analysis.maintainabilities]
        avg_complexity = sum(complexity_values) / len(complexity_values) if complexity_values else 0.0
        max_complexity = max(complexity_values) if complexity_values else 0.0
        min_maintainability = min(maintainability_values) if maintainability_values else None
        avg_maintainability = (
            sum(maintainability_values) / len(maintainability_values) if maintainability_values else None
        )
        worst_rank = cc_rank(max_complexity) if complexity_values else None
        return QualitySummary(
            complexity_values=complexity_values,
            maintainability_values=maintainability_values,
            avg_complexity=avg_complexity,
            max_complexity=max_complexity,
            worst_rank=worst_rank,
            min_maintainability=min_maintainability,
            avg_maintainability=avg_maintainability,
        )

    def _quality_violations(
        self,
        analysis: QualityAnalysis,
        summary: QualitySummary,
        thresholds: QualityThresholds,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        complexity_violations = [
            {
                "path": str(record.path),
                "name": record.name,
                "lineno": record.lineno,
                "complexity": record.score,
                "rank": cc_rank(record.score),
            }
            for record in analysis.complexities
            if record.score > thresholds.function
        ]
        maintainability_violations = [
            {
                "path": str(record.path),
                "maintainability_index": record.score,
            }
            for record in analysis.maintainabilities
            if record.score < thresholds.maintainability
        ]
        return complexity_violations, maintainability_violations

    def _quality_status(
        self,
        analysis: QualityAnalysis,
        summary: QualitySummary,
        thresholds: QualityThresholds,
    ) -> Tuple[str, List[str]]:
        status = "passed"
        details: List[str] = []

        if analysis.missing_paths:
            details.append(f"missing paths: {', '.join(analysis.missing_paths)}")
            if not analysis.analyzed_files:
                status = "failed"

        if not analysis.analyzed_files:
            status = "failed"
            details.append("no python files discovered for code quality analysis")

        if summary.complexity_values and summary.avg_complexity > thresholds.average:
            status = "failed"
            details.append(
                f"average cyclomatic complexity {summary.avg_complexity:.2f} exceeds threshold {thresholds.average:.2f}"
            )

        if (
            summary.avg_maintainability is not None
            and summary.avg_maintainability < thresholds.maintainability
        ):
            status = "failed"
            details.append(
                f"average maintainability index {summary.avg_maintainability:.2f} below threshold {thresholds.maintainability:.2f}"
            )

        if analysis.read_errors:
            status = "failed"
            details.append(f"analysis errors in {len(analysis.read_errors)} file(s)")

        return status, details

    def _quality_metrics(
        self,
        summary: QualitySummary,
        thresholds: QualityThresholds,
        analysis: QualityAnalysis,
    ) -> Dict[str, Any]:
        return {
            "files_analyzed": len(analysis.analyzed_files),
            "blocks_analyzed": len(analysis.complexities),
            "cyclomatic_complexity_avg": round(summary.avg_complexity, 2)
            if summary.complexity_values
            else 0.0,
            "cyclomatic_complexity_max": round(summary.max_complexity, 2)
            if summary.complexity_values
            else 0.0,
            "cyclomatic_complexity_rank": summary.worst_rank,
            "maintainability_index_min": round(summary.min_maintainability, 2)
            if summary.min_maintainability is not None
            else None,
            "maintainability_index_avg": round(summary.avg_maintainability, 2)
            if summary.avg_maintainability is not None
            else None,
            "threshold_function": thresholds.function,
            "threshold_average": thresholds.average,
            "threshold_maintainability": thresholds.maintainability,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _safe_float(self, value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default