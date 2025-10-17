# Quality Guide

This project ships an end-to-end evaluator that exercises ingest, retrieval, extraction, math parsing, and performance gates. The evaluator produces a JSON scorecard and a database record so CI can make release decisions.

## Release Gates

- **Boot**: All core services (Postgres, MinIO, Ollama, Redis/worker, app runtime) report healthy before scenarios execute.
- **Ingestion**: Every fixture document must ingest, chunk, and embed successfully, and time per page must stay under the documented budget.
- **Retrieval**: Queries must meet the Hit@k and nDCG thresholds defined in `EVAL_RUNBOOK.md`.
- **Extraction**: Title, author list, and abstract F1 >= target on the fixture set.
- **Math**: Equation LaTeX exact-match rate >= target.
- **Performance**: Ingest and query latency must stay within the documented budgets.

Any hard gate failure forces a non-zero exit code from the evaluator container, causing CI to fail.

## Scoring Overview

The scorecard surfaces per-scenario metrics and an overall score (0.0-1.0). Skipped tests are excluded from the denominator but produce warnings. The evaluator also captures contextual metadata (git SHA, model versions, run id) to support trend analysis.

## Developer Checklist

1. Keep the fixture corpus representative but lightweight. Add larger suites behind the `eval-full` profile.
2. Update thresholds when core models change and document the rationale in `EVAL_RUNBOOK.md`.
3. Backfill missing fixtures before enabling new quality gates.
4. Prefer deterministic checks before subjective model-based judgments.
5. Record regressions in `eval_results` with run ids so we can compare builds historically.
