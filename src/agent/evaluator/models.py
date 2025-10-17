"""Common data models for evaluator scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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
class IngestionRecord:
    path: Path
    elapsed: float
    doc_id: Optional[str] = None
    error: Optional[str] = None
    duplicate: bool = False
