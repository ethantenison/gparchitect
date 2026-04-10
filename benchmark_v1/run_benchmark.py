"""
benchmark_v1.run_benchmark — reproducible benchmark runner for benchmark_v1.

Purpose:
    Executes GPArchitect and baseline GP models on all registered benchmark
    datasets, saves results as JSON/CSV, and produces a summary CSV.

Role in benchmark pipeline:
    Registry → **Runner** → results/benchmark_v1/ → analyze_results.py

Usage::

    python -m benchmark_v1.run_benchmark
    python -m benchmark_v1.run_benchmark --output-dir results/benchmark_v1
    python -m benchmark_v1.run_benchmark --dataset additive --seed 0 --noise 0.05
    python -m benchmark_v1.run_benchmark --tier 2 --dry-run

CLI options:
    --output-dir  Where to write result files (default: results/benchmark_v1)
    --dataset     Run only this dataset (can be repeated)
    --tier        Run only this tier (1 or 2)
    --seed        Run only this seed (can be repeated)
    --noise       Run only this noise level (can be repeated)
    --dry-run     Print the run plan without executing

Non-obvious design decisions:
    - Each (dataset, noise, seed, prompt_variant) tuple is one "run" that produces
      one row in the results CSV.
    - Baseline runs use the GPArchitect builder/fitter directly (no translation),
      so their only source of variance is the data split.
    - Wall-clock time is measured with ``time.perf_counter`` to avoid OS effects.
    - Results are written incrementally so that partial runs are not lost.
    - Predictive metrics are computed on the held-out test set; the model is
      not re-trained on test data.

What this module does NOT do:
    - It does not generate the markdown report (see analyze_results.py).
    - It does not persist model checkpoints.
    - It does not perform cross-validation.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One row of benchmark results for a single model evaluation.

    Attributes:
        dataset_name: Name of the benchmark dataset.
        tier: Dataset tier (1 or 2).
        seed: Random seed used.
        noise_std: Observation noise standard deviation.
        model_type: ``"gparchitect"`` or ``"baseline"``.
        model_id: Prompt variant name (for gparchitect) or baseline name.
        fit_success: Whether fitting succeeded.
        retry_count: Number of revision attempts used (gparchitect only).
        rmse: Root mean squared error on the test set.
        mae: Mean absolute error on the test set.
        nll: Negative log predictive density on the test set.
        coverage_95: Empirical coverage of 95% predictive intervals.
        interval_width_95: Mean width of 95% predictive intervals.
        wall_time_s: Wall-clock time in seconds for the full run.
        error_message: Error description if fit_success is False.
    """

    dataset_name: str
    tier: int
    seed: int
    noise_std: float
    model_type: str
    model_id: str
    fit_success: bool
    retry_count: int
    rmse: float | None
    mae: float | None
    nll: float | None
    coverage_95: float | None
    interval_width_95: float | None
    wall_time_s: float
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_mean: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float | None]:
    """Compute predictive metrics given posterior mean, std, and true values.

    Args:
        y_mean: Posterior predictive mean, shape (N,).
        y_std: Posterior predictive standard deviation, shape (N,).
        y_true: Ground-truth test targets, shape (N,).

    Returns:
        Dict with keys: ``rmse``, ``mae``, ``nll``, ``coverage_95``,
        ``interval_width_95``.
    """
    residuals = y_true - y_mean
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    # Negative log predictive density (Gaussian)
    eps = 1e-8
    std_safe = np.maximum(y_std, eps)
    nll = float(np.mean(0.5 * np.log(2 * math.pi * std_safe**2) + 0.5 * (residuals / std_safe) ** 2))

    # 95% interval coverage and width
    z95 = 1.96
    lower = y_mean - z95 * y_std
    upper = y_mean + z95 * y_std
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage_95 = float(np.mean(in_interval))
    interval_width_95 = float(np.mean(upper - lower))

    return {
        "rmse": rmse,
        "mae": mae,
        "nll": nll,
        "coverage_95": coverage_95,
        "interval_width_95": interval_width_95,
    }


def _predict_metrics(model: Any, test_X: Any, y_true: np.ndarray) -> dict[str, float | None]:
    """Run a fitted model on test inputs and compute metrics.

    Args:
        model: A fitted BoTorch GP model.
        test_X: Test input tensor.
        y_true: Ground-truth test targets as numpy array.

    Returns:
        Metrics dict (or dict with None values on failure).
    """
    try:
        import gpytorch
        import torch

        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(test_X)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            variance = posterior.variance.squeeze(-1).cpu().numpy()
            std = np.sqrt(np.maximum(variance, 0.0))

        return _compute_metrics(mean, std, y_true)

    except Exception as exc:
        logger.warning("Prediction failed: %s", exc)
        return {
            "rmse": None,
            "mae": None,
            "nll": None,
            "coverage_95": None,
            "interval_width_95": None,
        }


# ---------------------------------------------------------------------------
# Dataset loading helper
# ---------------------------------------------------------------------------


def _load_dataset(dataset_name: str, seed: int, noise_std: float, n_train: int, n_test: int) -> Any:
    """Load a dataset split for the given configuration.

    Args:
        dataset_name: Name of the dataset.
        seed: Random seed.
        noise_std: Noise standard deviation.
        n_train: Number of training samples.
        n_test: Number of test samples.

    Returns:
        DatasetSplit object.
    """
    from benchmark_v1.datasets.botorch_functions import BOTORCH_GENERATORS
    from benchmark_v1.datasets.synthetic import SYNTHETIC_GENERATORS

    generators = {**SYNTHETIC_GENERATORS, **BOTORCH_GENERATORS}
    if dataset_name not in generators:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {sorted(generators)}")

    gen_fn = generators[dataset_name]
    return gen_fn(seed=seed, n_train=n_train, n_test=n_test, noise_std=noise_std)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# GPArchitect run
# ---------------------------------------------------------------------------


def run_gparchitect_variant(
    split: Any,
    prompt: str,
    prompt_variant: str,
    entry: Any,
) -> RunRecord:
    """Run GPArchitect with a specific prompt variant on a dataset split.

    Args:
        split: DatasetSplit with train/test DataFrames.
        prompt: Natural-language instruction string.
        prompt_variant: Variant label (``"aligned"``, ``"vague"``, ``"misleading"``).
        entry: BenchmarkEntry from the registry.

    Returns:
        RunRecord with metrics and metadata.
    """
    from gparchitect.api import run_gparchitect

    t0 = time.perf_counter()
    try:
        model, experiment_log = run_gparchitect(
            dataframe=split.train,
            instruction=prompt,
            input_columns=split.input_columns,
            output_columns=[split.output_column],
            max_retries=5,
        )
        wall_time_s = time.perf_counter() - t0
        fit_success = experiment_log.final_success
        retry_count = max(0, len(experiment_log.attempts) - 1)
        error_message = ""

        metrics: dict[str, float | None] = {
            "rmse": None,
            "mae": None,
            "nll": None,
            "coverage_95": None,
            "interval_width_95": None,
        }

        if fit_success and model is not None:
            # Prepare test tensor (use same scaling as training)
            from gparchitect.builders.data import prepare_data

            test_bundle = prepare_data(
                split.test,
                split.input_columns,
                [split.output_column],
                scale_inputs=True,
            )
            y_true = split.test[split.output_column].to_numpy()
            metrics = _predict_metrics(model, test_bundle.train_X, y_true)

    except Exception as exc:
        wall_time_s = time.perf_counter() - t0
        fit_success = False
        retry_count = 0
        error_message = f"{type(exc).__name__}: {exc}"
        metrics = {
            "rmse": None,
            "mae": None,
            "nll": None,
            "coverage_95": None,
            "interval_width_95": None,
        }
        logger.error("GPArchitect run failed: %s", error_message)

    return RunRecord(
        dataset_name=split.name,
        tier=entry.tier,
        seed=split.seed,
        noise_std=split.noise_std,
        model_type="gparchitect",
        model_id=prompt_variant,
        fit_success=fit_success,
        retry_count=retry_count,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        nll=metrics["nll"],
        coverage_95=metrics["coverage_95"],
        interval_width_95=metrics["interval_width_95"],
        wall_time_s=wall_time_s,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Baseline run
# ---------------------------------------------------------------------------


def run_baseline(
    split: Any,
    baseline_name: str,
    entry: Any,
) -> RunRecord:
    """Run a baseline model on a dataset split.

    Args:
        split: DatasetSplit with train/test DataFrames.
        baseline_name: Name of the baseline configuration.
        entry: BenchmarkEntry from the registry.

    Returns:
        RunRecord with metrics and metadata.
    """
    from benchmark_v1.baselines.configs import BASELINE_FACTORIES
    from gparchitect.builders.builder import build_model_from_dsl
    from gparchitect.builders.data import prepare_data
    from gparchitect.fitting.fitter import fit_and_validate

    t0 = time.perf_counter()
    try:
        factory = BASELINE_FACTORIES[baseline_name]  # type: ignore[index]
        input_dim = len(split.input_columns)
        spec = factory(input_dim=input_dim, output_dim=1)  # type: ignore[operator]

        train_bundle = prepare_data(
            split.train,
            split.input_columns,
            [split.output_column],
            scale_inputs=True,
        )
        model = build_model_from_dsl(spec, train_bundle.train_X, train_bundle.train_Y)
        fit_result = fit_and_validate(model, train_bundle.train_X, train_bundle.train_Y)

        wall_time_s = time.perf_counter() - t0
        fit_success = fit_result.success
        error_message = fit_result.error_message or ""

        metrics: dict[str, float | None] = {
            "rmse": None,
            "mae": None,
            "nll": None,
            "coverage_95": None,
            "interval_width_95": None,
        }

        if fit_success and fit_result.model is not None:
            test_bundle = prepare_data(
                split.test,
                split.input_columns,
                [split.output_column],
                scale_inputs=True,
            )
            y_true = split.test[split.output_column].to_numpy()
            metrics = _predict_metrics(fit_result.model, test_bundle.train_X, y_true)

    except Exception as exc:
        wall_time_s = time.perf_counter() - t0
        fit_success = False
        error_message = f"{type(exc).__name__}: {exc}"
        metrics = {
            "rmse": None,
            "mae": None,
            "nll": None,
            "coverage_95": None,
            "interval_width_95": None,
        }
        logger.error("Baseline %s failed: %s", baseline_name, error_message)

    return RunRecord(
        dataset_name=split.name,
        tier=entry.tier,
        seed=split.seed,
        noise_std=split.noise_std,
        model_type="baseline",
        model_id=baseline_name,
        fit_success=fit_success,
        retry_count=0,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        nll=metrics["nll"],
        coverage_95=metrics["coverage_95"],
        interval_width_95=metrics["interval_width_95"],
        wall_time_s=wall_time_s,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def build_run_plan(
    dataset_filter: list[str] | None,
    tier_filter: int | None,
    seed_filter: list[int] | None,
    noise_filter: list[float] | None,
) -> list[dict[str, Any]]:
    """Build a list of (entry, seed, noise, model_id, model_type) tuples.

    Args:
        dataset_filter: If set, only include these datasets.
        tier_filter: If set, only include this tier.
        seed_filter: If set, only include these seeds.
        noise_filter: If set, only include these noise levels.

    Returns:
        List of run descriptor dicts.
    """
    from benchmark_v1.registry import REGISTRY

    plan: list[dict[str, Any]] = []
    for entry in REGISTRY:
        if dataset_filter and entry.dataset_name not in dataset_filter:
            continue
        if tier_filter is not None and entry.tier != tier_filter:
            continue
        for seed in entry.seeds:
            if seed_filter and seed not in seed_filter:
                continue
            for noise_std in entry.noise_levels:
                if noise_filter and not any(abs(noise_std - nf) < 1e-9 for nf in noise_filter):
                    continue
                for variant in entry.prompt_variants:
                    plan.append({
                        "entry": entry,
                        "seed": seed,
                        "noise_std": noise_std,
                        "model_type": "gparchitect",
                        "model_id": variant,
                    })
                for baseline in entry.baselines:
                    plan.append({
                        "entry": entry,
                        "seed": seed,
                        "noise_std": noise_std,
                        "model_type": "baseline",
                        "model_id": baseline,
                    })
    return plan


def save_record(record: RunRecord, output_dir: Path) -> None:
    """Append a RunRecord to the incremental JSONL file.

    Args:
        record: The run record to save.
        output_dir: Directory where results are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "raw_results.jsonl"
    with jsonl_path.open("a") as fh:
        fh.write(json.dumps(record.to_dict()) + "\n")


def load_records(output_dir: Path) -> list[RunRecord]:
    """Load all RunRecords from the JSONL file in output_dir.

    Args:
        output_dir: Directory containing ``raw_results.jsonl``.

    Returns:
        List of RunRecord objects.
    """
    jsonl_path = output_dir / "raw_results.jsonl"
    if not jsonl_path.exists():
        return []

    records: list[RunRecord] = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(RunRecord(**data))
    return records


def save_summary_csv(records: list[RunRecord], output_dir: Path) -> None:
    """Write aggregated summary CSV to output_dir.

    Args:
        records: List of RunRecord objects.
        output_dir: Directory where the CSV is written.
    """
    if not records:
        logger.warning("No records to summarise.")
        return

    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Summary CSV written to %s (%d rows).", csv_path, len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--output-dir", default="results/benchmark_v1", show_default=True, help="Output directory.")
@click.option("--dataset", "datasets", multiple=True, help="Dataset(s) to include (repeatable).")
@click.option("--tier", type=int, default=None, help="Restrict to tier 1 or 2.")
@click.option("--seed", "seeds", multiple=True, type=int, help="Seed(s) to include (repeatable).")
@click.option("--noise", "noise_levels", multiple=True, type=float, help="Noise level(s) to include (repeatable).")
@click.option("--dry-run", is_flag=True, default=False, help="Print run plan without executing.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
def main(
    output_dir: str,
    datasets: tuple[str, ...],
    tier: int | None,
    seeds: tuple[int, ...],
    noise_levels: tuple[float, ...],
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run the benchmark_v1 suite and save results.

    Executes GPArchitect and baseline GP models on all registered benchmark
    datasets, writing incremental results to OUTPUT_DIR/raw_results.jsonl
    and a summary to OUTPUT_DIR/summary.csv.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_path = Path(output_dir)
    dataset_filter = list(datasets) if datasets else None
    seed_filter = list(seeds) if seeds else None
    noise_filter = list(noise_levels) if noise_levels else None

    plan = build_run_plan(dataset_filter, tier, seed_filter, noise_filter)

    logger.info("Run plan: %d runs scheduled.", len(plan))

    if dry_run:
        click.echo(f"Dry run: {len(plan)} runs planned.")
        for item in plan:
            entry = item["entry"]
            click.echo(
                f"  {entry.dataset_name:20s}  tier={entry.tier}  "
                f"seed={item['seed']}  noise={item['noise_std']:.3f}  "
                f"{item['model_type']:12s}  {item['model_id']}"
            )
        return

    from benchmark_v1.prompts.variants import get_prompts

    records: list[RunRecord] = []
    for i, item in enumerate(plan, start=1):
        entry = item["entry"]
        seed = item["seed"]
        noise_std = item["noise_std"]
        model_type = item["model_type"]
        model_id = item["model_id"]

        logger.info(
            "[%d/%d] dataset=%s tier=%d seed=%d noise=%.3f type=%s id=%s",
            i, len(plan), entry.dataset_name, entry.tier, seed, noise_std, model_type, model_id,
        )

        try:
            split = _load_dataset(entry.dataset_name, seed, noise_std, entry.n_train, entry.n_test)
        except Exception as exc:
            logger.error("Failed to load dataset %s: %s", entry.dataset_name, exc)
            continue

        if model_type == "gparchitect":
            try:
                prompts = get_prompts(entry.dataset_name)
                prompt = prompts.as_dict()[model_id]
            except KeyError:
                logger.error("No prompt variant '%s' for dataset '%s'", model_id, entry.dataset_name)
                continue
            record = run_gparchitect_variant(split, prompt, model_id, entry)
        else:
            record = run_baseline(split, model_id, entry)

        records.append(record)
        save_record(record, output_path)

        status = "OK" if record.fit_success else "FAIL"
        rmse_str = f"rmse={record.rmse:.4f}" if record.rmse is not None else "rmse=N/A"
        logger.info("  → %s  %s  t=%.1fs", status, rmse_str, record.wall_time_s)

    save_summary_csv(records, output_path)
    n_ok = sum(1 for r in records if r.fit_success)
    click.echo(f"\nBenchmark complete: {n_ok}/{len(records)} runs succeeded. Results in {output_path}/")


if __name__ == "__main__":
    main()
