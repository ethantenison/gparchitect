# benchmark_v1 results

This directory contains benchmark run artifacts:

- `raw_results.jsonl` — incremental JSONL of RunRecord objects (one per line)
- `summary.csv`       — flat CSV generated at the end of each run
- `aggregated.csv`    — per-(dataset, noise, model_id) aggregated metrics
- `report.md`         — markdown benchmark report

To generate these files, run:

```bash
python -m benchmark_v1.run_benchmark
python -m benchmark_v1.analyze_results results/benchmark_v1
```
