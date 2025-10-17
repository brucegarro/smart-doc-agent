# Evaluation Runbook

## Purpose

`docker compose --profile eval up --abort-on-container-exit --exit-code-from evaluator` drives an end-to-end harness that ingests fixture documents, runs retrieval/extraction/math checks, and emits `eval/results/<run_id>/scorecard.json` plus a database record.

## Directory Layout

- `eval/fixtures/docs/` — PDFs and images seeded into MinIO and ingested.
- `eval/fixtures/queries.jsonl` — JSON Lines of queries with gold chunk ids and answers.
- `eval/fixtures/fields.jsonl` — JSON Lines of field-level gold data (`title`, `authors`, etc.).
- `eval/fixtures/math.jsonl` — JSON Lines of math crops and gold LaTeX strings.
- `eval/results/` — Writable artifacts directory containing one subfolder per run id.

## Typical Workflow

1. Populate fixtures and update gold references as the product evolves.
2. Run the stack with `docker compose up -d db minio redis ollama app worker` (or reuse an existing dev instance).
3. Launch the evaluator using the compose profile command above.
4. Inspect `eval/results/<run_id>/scorecard.json` for metrics; CI will also surface the exit code.
5. The evaluator writes a summary row to `eval_results` so you can audit runs with SQL.

## Thresholds (initial)

| Scenario    | Metric            | Target |
|-------------|-------------------|--------|
| Ingestion   | Success rate      | 1.0    |
| Ingestion   | Time/page (s)     | ≤ 5.0  |
| Retrieval   | Hit@5             | 0.7    |
| Extraction  | Title F1          | 0.9    |
| Math        | LaTeX exact match | 0.6    |
| Performance | Query p95 (s)     | 2.0    |

Tune these thresholds as coverage improves. Document changes along with fixture updates.

## Troubleshooting

- **Boot failures**: check dependent service logs (`docker compose logs <service>`). The evaluator exits early.
- **Ingestion stalls**: confirm worker is running and Redis queues are draining.
- **Scorecard missing**: ensure `eval/results` is writable in the compose service definition.
- **DB write errors**: run migrations (`scripts/migrate.sh`) and verify the `eval_results` table exists.
