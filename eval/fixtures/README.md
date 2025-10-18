# Evaluation Fixtures

Seed data for end-to-end evaluator runs. Populate these files with representative examples before running a full evaluation.

- `docs/` — PDF and image assets used for ingestion scenarios.
- `queries.jsonl` — One JSON object per line describing retrieval eval prompts.
- `fields.jsonl` — Targeted field extraction gold labels.
- `math.jsonl` — Math expression crops + gold-standard LaTeX.
- `quality.json` — Code-quality thresholds (cyclomatic complexity, maintainability, ignored paths).

## Quick Mode for Fast Iteration

For faster development cycles, minimal fixture sets are available with the `-quick` suffix:
- `queries-quick.jsonl` (3 samples)
- `fields-quick.jsonl` (1 sample)
- `math-quick.jsonl` (1 sample)

**Enable quick mode** by setting `EVAL_QUICK_MODE=true` (default in docker-compose.yml). The evaluator will automatically use the `-quick` fixture files.

**Run full evaluation** by setting `EVAL_QUICK_MODE=false`:
```bash
EVAL_QUICK_MODE=false docker compose --profile eval up --abort-on-container-exit
```

All files can start with the stub entries committed here. Replace or extend them as your evaluation coverage grows.
