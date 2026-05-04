"""
benchmark_v1.analyze_results — aggregation and report generation for benchmark_v1.

Purpose:
    Loads raw results from a benchmark run, computes aggregated statistics,
    and writes a markdown report answering the key benchmark questions.

Role in benchmark pipeline:
    results/benchmark_v1/ → **analyze_results** → report

Usage::

    python -m benchmark_v1.analyze_results results/benchmark_v1
    python -m benchmark_v1.analyze_results results/benchmark_v1 --report docs/benchmark_v1_report.md

Non-obvious design decisions:
    - Aggregation is over seeds within each (dataset, noise, model_id) cell.
    - The markdown report uses GitHub-flavoured markdown tables for readability.
    - ``None`` metric values (from failed runs) are excluded from averages but
      counted separately as failure counts.

What this module does NOT do:
    - It does not re-run models.
    - It does not produce plots (figures are out of scope for benchmark_v1).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all benchmark results from a results directory.

    Reads either ``raw_results.jsonl`` (incremental JSONL) or
    ``summary.csv`` if the JSONL is absent.

    Args:
        results_dir: Path to the benchmark results directory.

    Returns:
        DataFrame with one row per run record.

    Raises:
        FileNotFoundError: If no result files are found.
    """
    jsonl_path = results_dir / "raw_results.jsonl"
    csv_path = results_dir / "summary.csv"

    if jsonl_path.exists():
        import json

        rows: list[dict[str, Any]] = []
        with jsonl_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise FileNotFoundError(f"No records found in {jsonl_path}")
        return pd.DataFrame(rows)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"No result files found in {results_dir}. Run 'python -m benchmark_v1.run_benchmark' first."
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std metrics aggregated over seeds.

    Args:
        df: Raw results DataFrame.

    Returns:
        Aggregated DataFrame with one row per (dataset, noise, model_type, model_id).
    """
    metric_cols = ["rmse", "mae", "nll", "coverage_95", "interval_width_95", "wall_time_s"]
    group_cols = ["dataset_name", "tier", "noise_std", "model_type", "model_id"]

    agg_rows: list[dict[str, Any]] = []
    for key, grp in df.groupby(group_cols, sort=True):
        row: dict[str, Any] = dict(zip(group_cols, key))
        row["n_runs"] = len(grp)
        row["n_success"] = int(grp["fit_success"].sum())
        row["success_rate"] = float(row["n_success"]) / row["n_runs"]
        row["mean_retry_count"] = float(grp["retry_count"].mean())
        for col in metric_cols:
            vals = grp[col].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            else:
                row[f"{col}_mean"] = None
                row[f"{col}_std"] = None
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _fmt(value: float | None, decimals: int = 4) -> str:
    """Format a float for table display."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _build_rmse_table(agg: pd.DataFrame) -> str:
    """Build a markdown table of RMSE by (dataset, noise, model_id).

    Args:
        agg: Aggregated metrics DataFrame.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Dataset | Noise | Model | RMSE (mean ± std) | Success |",
        "|---------|-------|-------|-------------------|---------|",
    ]
    for _, row in agg.sort_values(["dataset_name", "noise_std", "model_id"]).iterrows():
        mean_str = _fmt(row["rmse_mean"])
        std_str = _fmt(row["rmse_std"])
        cell = f"{mean_str} ± {std_str}"
        success = f"{row['n_success']}/{row['n_runs']}"
        lines.append(f"| {row['dataset_name']} | {row['noise_std']:.3f} | {row['model_id']} | {cell} | {success} |")
    return "\n".join(lines)


def _build_nll_table(agg: pd.DataFrame) -> str:
    """Build a markdown table of NLL by (dataset, noise, model_id).

    Args:
        agg: Aggregated metrics DataFrame.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Dataset | Noise | Model | NLL (mean ± std) | Coverage 95% |",
        "|---------|-------|-------|------------------|--------------|",
    ]
    for _, row in agg.sort_values(["dataset_name", "noise_std", "model_id"]).iterrows():
        nll_str = f"{_fmt(row['nll_mean'])} ± {_fmt(row['nll_std'])}"
        cov_str = _fmt(row["coverage_95_mean"], decimals=3)
        lines.append(f"| {row['dataset_name']} | {row['noise_std']:.3f} | {row['model_id']} | {nll_str} | {cov_str} |")
    return "\n".join(lines)


def _build_failure_table(agg: pd.DataFrame) -> str:
    """Build a markdown table of failure rates.

    Args:
        agg: Aggregated metrics DataFrame.

    Returns:
        Markdown table string.
    """
    failed = agg[agg["success_rate"] < 1.0].sort_values("success_rate")
    if failed.empty:
        return "*All runs succeeded.*"

    lines = [
        "| Dataset | Noise | Model | Success Rate | Mean Retries |",
        "|---------|-------|-------|--------------|--------------|",
    ]
    for _, row in failed.iterrows():
        lines.append(
            f"| {row['dataset_name']} | {row['noise_std']:.3f} | {row['model_id']} "
            f"| {row['success_rate']:.2f} | {_fmt(row['mean_retry_count'], 2)} |"
        )
    return "\n".join(lines)


def _best_model_per_dataset(agg: pd.DataFrame) -> str:
    """Summarise which model achieved the lowest RMSE per (dataset, noise).

    Args:
        agg: Aggregated metrics DataFrame.

    Returns:
        Markdown table string.
    """
    lines = [
        "| Dataset | Noise | Best Model | RMSE |",
        "|---------|-------|------------|------|",
    ]
    for (ds, noise), grp in agg.groupby(["dataset_name", "noise_std"]):
        valid = grp.dropna(subset=["rmse_mean"])
        if valid.empty:
            best_model = "N/A"
            best_rmse = "N/A"
        else:
            idx = valid["rmse_mean"].idxmin()
            best_model = valid.loc[idx, "model_id"]
            best_rmse = _fmt(valid.loc[idx, "rmse_mean"])
        lines.append(f"| {ds} | {noise:.3f} | {best_model} | {best_rmse} |")
    return "\n".join(lines)


def generate_report(df: pd.DataFrame, agg: pd.DataFrame, output_dir: Path) -> str:
    """Generate a full markdown benchmark report.

    Args:
        df: Raw results DataFrame.
        agg: Aggregated metrics DataFrame.
        output_dir: Directory where raw results were saved (used for paths).

    Returns:
        Markdown string.
    """
    n_total = len(df)
    n_success = int(df["fit_success"].sum())
    n_datasets = df["dataset_name"].nunique()
    seeds = sorted(df["seed"].unique().tolist())
    noise_levels = sorted(df["noise_std"].unique().tolist())

    lines: list[str] = [
        "# GPArchitect benchmark_v1 Report",
        "",
        "## Overview",
        "",
        f"- **Total runs**: {n_total}",
        f"- **Successful fits**: {n_success} ({100 * n_success / max(n_total, 1):.1f}%)",
        f"- **Datasets**: {n_datasets}",
        f"- **Seeds**: {seeds}",
        f"- **Noise levels**: {noise_levels}",
        f"- **Results directory**: `{output_dir}`",
        "",
        "---",
        "",
        "## 1. RMSE Summary",
        "",
        "Mean RMSE (± std across seeds) for each model on each dataset and noise level.",
        "",
        _build_rmse_table(agg),
        "",
        "---",
        "",
        "## 2. Predictive Log-Likelihood and Coverage",
        "",
        "Negative log predictive density (NLL, lower is better) and empirical 95% coverage.",
        "",
        _build_nll_table(agg),
        "",
        "---",
        "",
        "## 3. Best Model per Dataset",
        "",
        "Model with the lowest mean RMSE at each (dataset, noise) combination.",
        "",
        _best_model_per_dataset(agg),
        "",
        "---",
        "",
        "## 4. Failure Rate and Retries",
        "",
        "Runs with success rate < 1.0 or elevated retry counts.",
        "",
        _build_failure_table(agg),
        "",
        "---",
        "",
        "## 5. Key Findings",
        "",
        _generate_key_findings(df, agg),
        "",
        "---",
        "",
        "## 6. Methodology",
        "",
        "- **Tier 1** datasets: BoTorch analytic test functions (Branin, Hartmann6, Rosenbrock).",
        "- **Tier 2** datasets: repository-generated synthetic tabular datasets with named columns.",
        "- Each (dataset, seed, noise) configuration is evaluated with:",
        "  - Three GPArchitect prompt variants: `aligned`, `vague`, `misleading`.",
        "  - Two baselines: `default_singletask` (Matern52, no ARD) and `matern52_ard` (Matern52 + ARD).",
        "- Metrics are computed on a held-out test set (not used in fitting).",
        "- GPArchitect uses up to 5 revision retries.",
        "",
        "---",
        "",
        "*Generated by `benchmark_v1/analyze_results.py`.*",
    ]

    return "\n".join(lines)


def _generate_key_findings(df: pd.DataFrame, agg: pd.DataFrame) -> str:
    """Generate a plain-English key findings section.

    Args:
        df: Raw results DataFrame.
        agg: Aggregated metrics DataFrame.

    Returns:
        Markdown string with bullet-point findings.
    """
    lines: list[str] = []

    # Compare aligned vs vague prompts vs baselines on RMSE
    gpa = agg[agg["model_type"] == "gparchitect"]
    baselines = agg[agg["model_type"] == "baseline"]

    aligned = gpa[gpa["model_id"] == "aligned"]
    vague = gpa[gpa["model_id"] == "vague"]
    misleading = gpa[gpa["model_id"] == "misleading"]

    if not aligned.empty and not baselines.empty:
        # Compute per-(dataset, noise) group comparisons so that the
        # reduction is consistent across models.
        best_baseline_per_group = (
            baselines.groupby(["dataset_name", "noise_std"])["rmse_mean"]
            .min()
            .reset_index()
            .rename(columns={"rmse_mean": "best_baseline_rmse"})
        )
        aligned_per_group = aligned[["dataset_name", "noise_std", "rmse_mean"]].rename(
            columns={"rmse_mean": "aligned_rmse"}
        )
        merged = aligned_per_group.merge(best_baseline_per_group, on=["dataset_name", "noise_std"], how="inner").dropna(
            subset=["aligned_rmse", "best_baseline_rmse"]
        )

        if not merged.empty:
            aligned_mean = float(merged["aligned_rmse"].mean())
            best_baseline_mean = float(merged["best_baseline_rmse"].mean())

            lines.append(
                f"- **Aligned vs default baseline**: "
                f"mean RMSE `aligned`={aligned_mean:.4f} vs "
                f"best baseline={best_baseline_mean:.4f} (averaged over matched dataset/noise groups). "
                + (
                    "Aligned prompts beat the best baseline."
                    if aligned_mean < best_baseline_mean
                    else "Best baseline is competitive with aligned prompts."
                )
            )

        if not vague.empty:
            vague_per_group = vague[["dataset_name", "noise_std", "rmse_mean"]].rename(
                columns={"rmse_mean": "vague_rmse"}
            )
            merged_v = aligned_per_group.merge(vague_per_group, on=["dataset_name", "noise_std"], how="inner").dropna(
                subset=["aligned_rmse", "vague_rmse"]
            )
            if not merged_v.empty:
                delta = float(merged_v["vague_rmse"].mean()) - float(merged_v["aligned_rmse"].mean())
                lines.append(
                    f"- **Aligned vs vague**: vague prompts show {delta:+.4f} RMSE delta "
                    f"({'worse' if delta > 0 else 'better'}) relative to aligned prompts."
                )

        if not misleading.empty:
            misleading_per_group = misleading[["dataset_name", "noise_std", "rmse_mean"]].rename(
                columns={"rmse_mean": "misleading_rmse"}
            )
            merged_m = aligned_per_group.merge(
                misleading_per_group, on=["dataset_name", "noise_std"], how="inner"
            ).dropna(subset=["aligned_rmse", "misleading_rmse"])
            if not merged_m.empty:
                delta_m = float(merged_m["misleading_rmse"].mean()) - float(merged_m["aligned_rmse"].mean())
                lines.append(
                    f"- **Aligned vs misleading**: misleading prompts show {delta_m:+.4f} RMSE delta "
                    f"({'worse' if delta_m > 0 else 'better'}) relative to aligned prompts."
                )

    # Failure analysis
    n_total = len(df)
    n_failed = n_total - int(df["fit_success"].sum())
    if n_failed > 0:
        fail_datasets = df[~df["fit_success"]]["dataset_name"].value_counts()
        lines.append(
            f"- **Failures**: {n_failed}/{n_total} runs failed. "
            f"Highest failure rate on: {fail_datasets.index[0] if not fail_datasets.empty else 'N/A'}."
        )
    else:
        lines.append("- **Failures**: all runs completed successfully (no fit failures).")

    # Retry analysis
    gpa_df = df[df["model_type"] == "gparchitect"]
    if not gpa_df.empty:
        mean_retries = gpa_df["retry_count"].mean()
        lines.append(f"- **Retries**: GPArchitect used on average {mean_retries:.2f} retries per run.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--report",
    "report_path",
    default=None,
    help="Path to write the markdown report (default: <results_dir>/report.md).",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
def main(results_dir: str, report_path: str | None, verbose: bool) -> None:
    """Analyse benchmark_v1 results and generate a markdown report.

    RESULTS_DIR should be the directory containing raw_results.jsonl or summary.csv.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results_path = Path(results_dir)
    df = load_results(results_path)
    logger.info("Loaded %d records from %s.", len(df), results_path)

    agg = aggregate_metrics(df)

    # Write aggregated CSV
    agg_csv = results_path / "aggregated.csv"
    agg.to_csv(agg_csv, index=False)
    logger.info("Aggregated CSV written to %s.", agg_csv)

    report_md = generate_report(df, agg, results_path)

    if report_path is None:
        report_out = results_path / "report.md"
    else:
        report_out = Path(report_path)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report_md)
    click.echo(f"Report written to {report_out}")


if __name__ == "__main__":
    main()
