# Database Schema Snapshot

The evaluator writes metrics alongside ingestion artifacts using the existing Postgres schema. Key tables:

- `documents` — One row per ingested PDF. Contains metadata, parsed content (UDR), and processing status.
- `chunks` — Vector-searchable text/table/equation segments linked to documents.
- `artifacts` — Binary or structured outputs stored alongside documents (tables, figures, etc.).
- `eval_results` — Evaluation metrics keyed by document, metric name, dataset, and model.

Helper views defined in `sql/06_views.sql` join documents with recent evaluation results so you can trend performance by run id or dataset.

Before running evaluations, execute the migrations in `sql/` or run `scripts/migrate.sh` to ensure the schema exists.
