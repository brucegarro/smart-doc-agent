# Evaluation Fixtures

Seed data for end-to-end evaluator runs. Populate these files with representative examples before running a full evaluation.

- `docs/` — PDF and image assets used for ingestion scenarios.
- `queries.jsonl` — One JSON object per line describing retrieval eval prompts.
- `fields.jsonl` — Targeted field extraction gold labels.
- `math.jsonl` — Math expression crops + gold-standard LaTeX.

All files can start with the stub entries committed here. Replace or extend them as your evaluation coverage grows.
