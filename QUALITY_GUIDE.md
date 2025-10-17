# Quality Guide

This project ships an end-to-end evaluator that exercises ingest, retrieval, extraction, math parsing, and performance gates. The evaluator produces a JSON scorecard and a database record so CI can make release decisions.

## Release Gates

- **Boot**: All core services (Postgres, MinIO, Ollama, Redis/worker, app runtime) report healthy before scenarios execute.
- **Code Quality**: Static analysis (Radon + lint bundle) must satisfy the thresholds documented below before runtime tests execute.
- **Ingestion**: Every fixture document must ingest, chunk, and embed successfully, and time per page must stay under the documented budget.
- **Retrieval**: Queries must meet the Hit@k and nDCG thresholds defined in `EVAL_RUNBOOK.md`.
- **Extraction**: Title, author list, and abstract F1 >= target on the fixture set.
- **Math**: Equation LaTeX exact-match rate >= target.
- **Performance**: Ingest and query latency must stay within the documented budgets.

Any hard gate failure forces a non-zero exit code from the evaluator container, causing CI to fail.

## Readability & Naming

- Follow intent-revealing naming: `snake_case` for functions/variables, `PascalCase` for classes, and avoid ambiguous abbreviations.
- Group modules logically (ingestion, retrieval, evaluation) and keep public APIs documented with module-level docstrings.
- Write targeted comments only when context is non-obvious (integration boundaries, algorithm trade-offs); remove stale noise immediately.
- Keep imports ordered (`stdlib`, third-party, project) and prefer explicit exports over wildcard imports.

## DRY Structure & Layout

- Promote shared helpers anytime two call-sites share more than ~5 lines of logic—extend existing utilities before forking new code paths.
- Centralise configuration in `settings` or dedicated fixtures instead of scattering literals across modules.
- Keep functions focused on a single responsibility; break out parsing, validation, and persistence steps into small, testable units.
- Avoid temporal coupling: prefer pure functions that accept inputs and return outputs over stateful singletons.

## Complexity Budgets

- Cyclomatic complexity per function/method must stay ≤ 10 (Radon grade A/B). Exceeding this requires either refactoring or an explicit fixture override reviewed in PR.
- Average cyclomatic complexity per file should stay ≤ 5.0; decompose large modules into cohesive packages when the average drifts higher.
- Maintainability Index per module must stay ≥ 65. Modules below the floor need restructuring before merge.
- Document justified exceptions in `eval/fixtures/quality.json` so the evaluator understands intentional hot spots.

## Static Analysis & Harness Enforcement

- `radon` runs before ingestion in the evaluator; failing thresholds cause the entire evaluation to abort.
- New code must land with passing Radon metrics and lint; add targeted tests for any helper you extract to satisfy the DRY guidance.
- Do not silence static analysis without an inline explanation that links to an issue or design document.
- Keep the fixture config (`eval/fixtures/quality.json`) updated when thresholds or ignored paths change; every change requires rationale in the PR description.

## Scoring Overview

The scorecard surfaces per-scenario metrics and an overall score (0.0-1.0). Skipped tests are excluded from the denominator but produce warnings. The evaluator also captures contextual metadata (git SHA, model versions, run id) to support trend analysis.

## Developer Checklist

1. Keep the fixture corpus representative but lightweight. Add larger suites behind the `eval-full` profile.
2. Update thresholds when core models change and document the rationale in `EVAL_RUNBOOK.md`.
3. Backfill missing fixtures before enabling new quality gates.
4. Prefer deterministic checks before subjective model-based judgments.
5. Record regressions in `eval_results` with run ids so we can compare builds historically.
