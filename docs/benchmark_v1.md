# benchmark_v1 — GPArchitect Empirical Benchmark Suite

## Overview

`benchmark_v1/` is a reproducible benchmark suite that tests whether
GPArchitect's natural-language-to-DSL model construction improves predictive
performance and/or robustness relative to strong GP baselines.

The suite is designed to answer:

> *Does GPArchitect's instruction pathway provide measurable value over simple
> GP defaults on a controlled benchmark, and where does it currently fail?*

---

## Quick Start

> **Note:** `benchmark_v1` is located at the repository root and is not
> included in the installed `gparchitect` package.  All commands below must be
> run from the **repository root** with the source checkout on `PYTHONPATH`, for
> example:
> ```bash
> cd /path/to/gparchitect
> PYTHONPATH=src python -m benchmark_v1.run_benchmark ...
> ```

```bash
# Run the full suite (may take 30–90 minutes)
PYTHONPATH=src python -m benchmark_v1.run_benchmark

# Run only Tier 2 datasets at a single seed/noise level (fast)
PYTHONPATH=src python -m benchmark_v1.run_benchmark --tier 2 --seed 0 --noise 0.05

# Dry-run: print the plan without fitting
PYTHONPATH=src python -m benchmark_v1.run_benchmark --dry-run

# Generate the report from saved results
PYTHONPATH=src python -m benchmark_v1.analyze_results results/benchmark_v1 \
    --report docs/benchmark_v1_report.md
```

---

## Repository Structure

```
benchmark_v1/
├── __init__.py
├── registry.py              # Fixed benchmark registry (datasets, seeds, noise levels)
├── run_benchmark.py         # Runner CLI
├── analyze_results.py       # Aggregation and report generation
├── datasets/
│   ├── __init__.py
│   ├── synthetic.py         # Tier 2: named-column synthetic tabular datasets
│   └── botorch_functions.py # Tier 1: BoTorch test-function adapters
├── prompts/
│   ├── __init__.py
│   └── variants.py          # Prompt variants (aligned / vague / misleading)
└── baselines/
    ├── __init__.py
    └── configs.py           # Baseline GPSpec factories

results/benchmark_v1/        # Output directory (created at runtime)
    raw_results.jsonl        # Incremental run records
    summary.csv              # Flat results table
    aggregated.csv           # Per-group aggregated metrics
    report.md                # Markdown benchmark report
```

---

## Dataset Tiers

### Tier 2 — Named-Column Synthetic Tabular Datasets (Primary)

These datasets are the most important for `benchmark_v1` because they
directly exercise GPArchitect's named-column parsing and kernel selection
claims.

| Dataset | Input Columns | Generating Equation |
|---------|---------------|---------------------|
| `additive` | `x_smooth`, `x_trend`, `x_scale`, `x_irrelevant_1`, `x_irrelevant_2` | Additive sum of three smooth terms; two irrelevant features |
| `periodic_decay` | `seasonality_index`, `system_age` | Periodic × exponential decay |
| `interaction` | `material_hardness`, `process_temperature`, `cooldown_rate` | Product-dominant; `cooldown_rate` is irrelevant |
| `ard_stress` | `x_signal_1`, `x_signal_2`, `x_weak_1`–`x_weak_4`, `x_irrelevant` | Two relevant + four weak + one noise feature |

### Tier 1 — BoTorch Test Functions (Controlled Reference)

These datasets use standard analytic test functions as a controlled layer.

| Dataset | Dimensions | Function |
|---------|-----------|----------|
| `branin` | 2 | Branin — 3 global optima |
| `hartmann6` | 6 | Hartmann6 — 1 global optimum |
| `rosenbrock` | 4 | Rosenbrock — banana-shaped valley (log-scaled output) |

---

## Prompt Variants

Each dataset has three prompt variants:

| Variant | Description |
|---------|-------------|
| `aligned` | Instruction matches the known generating structure |
| `vague` | Generic instruction with no structural guidance |
| `misleading` | Plausible but structurally incorrect instruction |

**Hypothesis**: `aligned` should produce better models than `vague`, which in
turn should match the default baselines.  `misleading` tests the resilience
of the revision/recovery mechanism.

---

## Baselines

| Baseline | Description |
|----------|-------------|
| `default_singletask` | `SingleTaskGP` + Matern52, no ARD, default BoTorch settings |
| `matern52_ard` | `SingleTaskGP` + Matern52, full ARD over all inputs |

Baselines bypass the natural-language translator and are constructed directly
as `GPSpec` objects, ensuring fair comparison.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error on held-out test set |
| MAE | Mean absolute error |
| NLL | Negative log predictive density (Gaussian) |
| Coverage 95% | Empirical coverage of 95% predictive intervals |
| Interval width | Mean width of 95% predictive intervals |
| Success rate | Fraction of seeds where fitting succeeded |
| Retry count | Mean number of DSL revision attempts (GPArchitect only) |
| Wall time | Seconds per run |

---

## Registry Configuration

The registry is defined in `benchmark_v1/registry.py`.  Key defaults:

- **Seeds**: `0, 1, 2`
- **Noise levels**: `0.0` (noiseless), `0.05` (low), `0.15` (moderate)
- **n_train / n_test**: 60/40 (Tier 2 and Branin), 80/40 (Hartmann6 and Rosenbrock)
- **Max retries** (GPArchitect): 5

---

## Runner CLI Reference

```
Usage: python -m benchmark_v1.run_benchmark [OPTIONS]

Options:
  --output-dir TEXT    Output directory. [default: results/benchmark_v1]
  --dataset TEXT       Dataset(s) to include (repeatable).
  --tier INTEGER       Restrict to tier 1 or 2.
  --seed INTEGER       Seed(s) to include (repeatable).
  --noise FLOAT        Noise level(s) to include (repeatable).
  --dry-run            Print run plan without executing.
  -v, --verbose        Enable debug logging.
```

---

## Expected Output

After a full run:

```
results/benchmark_v1/
├── raw_results.jsonl    # One JSON line per (dataset, seed, noise, model) combination
├── summary.csv          # Flat CSV of all records
├── aggregated.csv       # Aggregated over seeds per group
└── report.md            # Benchmark report
```

---

## Interpretation Guide

**When does aligned natural language help?**

Compare `aligned` vs `default_singletask` RMSE.  A meaningful improvement
(> 1 std) on datasets like `periodic_decay` or `interaction` indicates the
instruction pathway is adding value.

**How much degradation from vague/misleading prompts?**

Compare `vague` and `misleading` RMSE against `aligned`.  `vague` should
roughly match the default baseline.  `misleading` tests recovery.

**Are retries helping?**

Check `mean_retry_count` for runs that succeeded after retries versus those
that failed outright.

**Which datasets expose current weaknesses?**

Datasets where `aligned` underperforms `matern52_ard` (despite correct
instructions) reveal gaps in the translator or builder.

---

## Non-Goals for benchmark_v1

- No LLM-based judging
- No large real-world benchmark zoo
- No new kernel families
- No Bayesian optimisation benchmarking
- No overengineered experiment tracking
