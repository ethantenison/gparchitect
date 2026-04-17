# benchmark_v1 results

This directory contains benchmark run artifacts:

- `raw_results.jsonl` — incremental JSONL of RunRecord objects (one per line)
- `summary.csv`       — flat CSV generated at the end of each run
- `aggregated.csv`    — per-(dataset, noise, model_id) aggregated metrics
- `report.md`         — markdown benchmark report

To generate these files, run from the **repository root** (source checkout):

```bash
PYTHONPATH=src python -m benchmark_v1.run_benchmark
PYTHONPATH=src python -m benchmark_v1.analyze_results results/benchmark_v1
```

> **Note:** `benchmark_v1` is at the repository root and is not part of the
> installed `gparchitect` package. These commands require a source checkout with
> `src/` on `PYTHONPATH`.
