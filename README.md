# smart-doc-agent

A smart document agent project.

## Getting Started

This project is currently under development.

### End-to-End Evaluation (CI Harness)

Run the evaluator profile to spin up the stack, execute the scenario suite, and capture a JSON scorecard:

```bash
docker compose --profile eval up --abort-on-container-exit --exit-code-from evaluator
```

Results land in `eval/results/<run_id>/scorecard.json` and a summary row is inserted into `eval_results`. See `EVAL_RUNBOOK.md` for fixture layout and thresholds.

