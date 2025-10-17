# Architecture Overview

## Services

- **app** — Python CLI/runtime that orchestrates ingestion and retrieval workloads. Shares code with the evaluator.
- **worker** — Background worker consuming Redis queues for long-running ingestion tasks (table OCR, embeddings).
- **db** — PostgreSQL + pgvector for metadata, chunk storage, and evaluation history.
- **minio** — S3-compatible object storage that stores raw PDFs and intermediate artifacts.
- **ollama** — Hosts local LLM/VLM models for reasoning, extraction, and math.
- **redis** — Queue backing the worker and acting as ephemeral coordination.
- **evaluator** — Profiled compose service that waits for the stack to be healthy, runs end-to-end scenarios, and pushes scorecards.

## Data Flow (CRUDE)

1. PDFs arrive via CLI ingestion and are uploaded to MinIO.
2. Parsed content and derived embeddings land in Postgres (`documents`, `chunks`).
3. Retrieval queries hit the embedding service and fetch candidate chunks from Postgres.
4. Extraction/math routines read stored UDR data and, when needed, call Ollama.
5. Evaluator orchestrates ingestion, queries, and comparisons against gold data; results persist in `eval_results`.

## Key Paths

- `src/agent/ingestion/` — PDF parsing, chunking, embedding.
- `src/agent/retrieval/` — Similarity search utilities.
- `src/agent/evaluator/` — Evaluation harness (introduced in this change).
- `eval/fixtures/` — Inputs that drive evaluation scenarios.
- `eval/results/` — Writable output used by CI.

## Extending the System

- Add new scenario modules under `src/agent/evaluator/` and wire them into the harness.
- Extend compose profiles (`eval-fast`, `eval-full`) for different coverage levels.
- Use `PROMPTS/` for canonical system/developer prompts consumed by downstream agents.
