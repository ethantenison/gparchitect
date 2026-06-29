"""Walk-forward Riskfolio optimization from GP posterior scenarios."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import riskfolio as rp
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_RUNNER = EXPERIMENT_DIR / "run_experiment.py"
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"
STARTING_VALUE = 10_000.0
PERIODS_PER_YEAR = 12

SPEC = importlib.util.spec_from_file_location("tvp_run_experiment", SOURCE_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load source runner from {SOURCE_RUNNER}")
exp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exp
SPEC.loader.exec_module(exp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, default=exp.ARTIFACT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--last-n-windows", type=int, default=24)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--botorch-default-optimizer", action="store_true")
    parser.add_argument("--task-kernel", choices=["positive", "signed_no_prior"], required=True)
    parser.add_argument("--scaling", choices=["botorch_normalize"], default="botorch_normalize")
    parser.add_argument("--posterior-scenarios", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--upperlng", type=float, default=0.20)
    parser.add_argument("--nea", type=int, default=10)
    parser.add_argument("--etf-universe", nargs="+", default=exp.ALL_18_ETFS)
    parser.add_argument(
        "--portfolio-exclude-assets",
        nargs="*",
        default=["BND", "MGK"],
        help="Assets fit by the GP but excluded from final Riskfolio weights, matching BayesFolio helper assets.",
    )
    parser.add_argument("--train-window-months", type=int, default=60)
    parser.add_argument("--save-scenarios", action="store_true")
    return parser.parse_args()


def git_info() -> dict[str, str]:
    def run(cmd: list[str]) -> str:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def optimize_riskfolio(
    returns: pd.DataFrame,
    *,
    method_mu: str,
    method_cov: str,
    upperlng: float,
    nea: int,
) -> pd.Series:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any").dropna(axis=0, how="any")
    if clean.empty or clean.shape[1] < 2:
        return equal_weight(clean.columns.tolist())
    try:
        n_assets = clean.shape[1]
        portfolio = rp.Portfolio(returns=clean, nea=max(1, min(int(nea), n_assets - 1)))
        portfolio.upperlng = max(float(upperlng), 1.0 / n_assets)
        portfolio.lowerlng = 0.0
        portfolio.card = None
        portfolio.alpha = 0.5
        portfolio.assets_stats(method_mu=method_mu, method_cov=method_cov)
        weights_df = portfolio.optimization(
            model="Classic",
            rm="CVaR",
            obj="Sharpe",
            rf=0.0,
            hist=True,
        )
        if weights_df is None or weights_df.empty:
            raise RuntimeError("Riskfolio returned empty weights")
        weights = weights_df.iloc[:, 0].astype(float)
    except Exception:
        weights = equal_weight(clean.columns.tolist())
    weights = weights.reindex(clean.columns).fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return equal_weight(clean.columns.tolist())
    return weights / total


def realized_return(weights: pd.Series, eval_returns: pd.Series) -> float:
    aligned = weights.reindex(eval_returns.index).fillna(0.0)
    return float(np.dot(aligned.to_numpy(dtype=float), eval_returns.to_numpy(dtype=float)))


def information_coefficient(predictions: pd.DataFrame, final_universe: list[str]) -> float:
    frame = predictions[predictions["asset_id"].isin(final_universe)]
    if frame["y_pred"].nunique() < 2 or frame["y_true"].nunique() < 2:
        return math.nan
    return float(frame["y_pred"].corr(frame["y_true"], method="spearman"))


def equal_weight(final_universe: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(final_universe), index=final_universe)


def performance_stats(returns: pd.Series, weights: pd.DataFrame) -> dict[str, float]:
    clean_returns = returns.dropna().astype(float)
    if clean_returns.empty:
        return {
            "n_rebalances": 0.0,
            "cumulative_return": math.nan,
            "cagr": math.nan,
            "annualized_vol": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
            "terminal_value": math.nan,
            "mean_monthly_return": math.nan,
            "hit_rate": math.nan,
            "avg_turnover": math.nan,
            "max_weight": math.nan,
        }
    equity = (1.0 + clean_returns).cumprod()
    years = len(clean_returns) / PERIODS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else math.nan
    ann_vol = float(clean_returns.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR))
    drawdown = equity / equity.cummax() - 1.0
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    if len(turnover) > 0:
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    return {
        "n_rebalances": float(len(clean_returns)),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe": float(cagr / ann_vol) if ann_vol > 0 else math.nan,
        "max_drawdown": float(drawdown.min()),
        "terminal_value": float(STARTING_VALUE * equity.iloc[-1]),
        "mean_monthly_return": float(clean_returns.mean()),
        "hit_rate": float((clean_returns > 0).mean()),
        "avg_turnover": float(turnover.mean()),
        "max_weight": float(weights.max(axis=1).max()),
    }


def markdown_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_root / args.run_id
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    if args.save_scenarios:
        (output_dir / "scenarios").mkdir(parents=True, exist_ok=True)

    panel = exp.load_panel(args.artifact_path, args.etf_universe)
    windows = exp.infer_windows(panel, args.last_n_windows)
    final_universe = [asset for asset in args.etf_universe if asset not in set(args.portfolio_exclude_assets)]
    optimizer_max_iter = None if args.botorch_default_optimizer else args.max_iter
    variant = exp.Variant(name="plain", parameterization="none", target="none")

    manifest = {
        "schema": "gparchitect.gp_scenario_riskfolio.v1",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_path": str(args.artifact_path),
        "output_dir": str(output_dir),
        "windows": [
            {"train_end_date": train_end, "forecast_date": forecast_date} for train_end, forecast_date in windows
        ],
        "training_universe": args.etf_universe,
        "portfolio_universe": final_universe,
        "portfolio_exclude_assets": args.portfolio_exclude_assets,
        "task_kernel": args.task_kernel,
        "rank": args.rank,
        "scaling": args.scaling,
        "posterior_scenarios": args.posterior_scenarios,
        "upperlng": args.upperlng,
        "nea": args.nea,
        "train_window_months": args.train_window_months,
        "max_iter": optimizer_max_iter,
        "botorch_default_optimizer": args.botorch_default_optimizer,
        "git": git_info(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    prediction_rows: list[pd.DataFrame] = []
    ic_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    weight_frames: dict[str, list[pd.Series]] = {
        "gp_scenarios_riskfolio": [],
        "historical_y_ewma2_riskfolio": [],
        "equal_weight": [],
    }

    for window_index, (train_end_s, forecast_date_s) in enumerate(windows):
        print(f"rebalance {forecast_date_s}", flush=True)
        exp.set_seed()
        torch.manual_seed(args.seed + window_index)
        np.random.seed(args.seed + window_index)

        train_end = pd.Timestamp(train_end_s)
        forecast_date = pd.Timestamp(forecast_date_s)
        train_df = exp.training_frame(panel, train_end, train_window_months=args.train_window_months)
        eval_df = panel.loc[(panel["date"] == forecast_date) & panel["y_excess_lead"].notna()].copy()
        if eval_df.empty:
            continue
        train_x, train_y, _ = exp.prepare_multitask_gp_data_with_task_feature(
            train_df[[*exp.MULTITASK_INPUT_COLUMNS, "asset_id", "y_excess_lead"]],
            target_col="y_excess_lead",
            asset_col="asset_id",
            drop_cols=[],
            dtype=torch.float64,
        )
        eval_x, _, _ = exp.prepare_multitask_gp_data_with_task_feature(
            eval_df[[*exp.MULTITASK_INPUT_COLUMNS, "asset_id", "y_excess_lead"]],
            target_col="y_excess_lead",
            asset_col="asset_id",
            drop_cols=[],
            dtype=torch.float64,
        )
        pred_mean, pred_std, model = exp.fit_multitask(
            train_x,
            train_y,
            eval_x,
            variant,
            args.rank,
            optimizer_max_iter,
            args.task_kernel,
            args.scaling,
        )
        model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            posterior = model.posterior(eval_x, observation_noise=True)
            scenario_samples = posterior.rsample(torch.Size([args.posterior_scenarios])).squeeze(-1).cpu().numpy()

        assets = eval_df["asset_id"].astype(str).tolist()
        scenarios = pd.DataFrame(scenario_samples, columns=assets)
        predictions = pd.DataFrame(
            {
                "date": forecast_date.date().isoformat(),
                "asset_id": assets,
                "y_true": eval_df["y_excess_lead"].to_numpy(dtype=float),
                "y_pred": pred_mean,
                "pred_std": pred_std,
                "score": pred_mean / np.clip(pred_std, 1e-12, None),
            }
        )
        prediction_rows.append(predictions)
        ic_rows.append(
            {
                "date": forecast_date.date().isoformat(),
                "strategy": "gp_scenarios_riskfolio",
                "ic": information_coefficient(predictions, final_universe),
            }
        )
        if args.save_scenarios:
            scenarios.loc[:, final_universe].to_csv(
                output_dir / "scenarios" / f"gp_scenarios_{forecast_date.date().isoformat()}.csv",
                index=False,
            )

        eval_returns = eval_df.set_index(eval_df["asset_id"].astype(str))["y_excess_lead"].reindex(final_universe)
        gp_weights = optimize_riskfolio(
            scenarios.loc[:, final_universe],
            method_mu="hist",
            method_cov="hist",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        hist_panel = train_df.pivot(index="date", columns="asset_id", values="y_excess_lead").reindex(
            columns=final_universe
        )
        hist_weights = optimize_riskfolio(
            hist_panel,
            method_mu="ewma2",
            method_cov="ewma2",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        weights_by_strategy = {
            "gp_scenarios_riskfolio": gp_weights.reindex(final_universe).fillna(0.0),
            "historical_y_ewma2_riskfolio": hist_weights.reindex(final_universe).fillna(0.0),
            "equal_weight": equal_weight(final_universe),
        }
        for strategy, weights in weights_by_strategy.items():
            weight_frames[strategy].append(pd.Series(weights, name=forecast_date))
            return_rows.append(
                {
                    "date": forecast_date.date().isoformat(),
                    "strategy": strategy,
                    "return": realized_return(weights, eval_returns),
                    "gp_ic": ic_rows[-1]["ic"] if strategy == "gp_scenarios_riskfolio" else math.nan,
                }
            )

    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    predictions_df.to_csv(output_dir / "gp_predictions.csv", index=False)
    pd.DataFrame(ic_rows).to_csv(output_dir / "gp_ic_by_window.csv", index=False)
    returns_df = pd.DataFrame(return_rows)
    returns_df.to_csv(output_dir / "portfolio_returns.csv", index=False)

    weight_dfs = []
    for strategy, frames in weight_frames.items():
        weights = pd.DataFrame(frames)
        weights.index.name = "date"
        weights = weights.reset_index()
        weights.insert(0, "strategy", strategy)
        weight_dfs.append(weights)
    weights_df = pd.concat(weight_dfs, ignore_index=True)
    weights_df.to_csv(output_dir / "portfolio_weights.csv", index=False)

    summary_rows = []
    for strategy, group in returns_df.groupby("strategy"):
        strategy_returns = group.set_index("date")["return"].astype(float)
        strategy_weights = (
            weights_df[weights_df["strategy"] == strategy].drop(columns=["strategy"]).set_index("date").astype(float)
        )
        summary = performance_stats(strategy_returns, strategy_weights)
        summary["strategy"] = strategy
        if strategy == "gp_scenarios_riskfolio":
            summary["mean_ic"] = float(pd.DataFrame(ic_rows)["ic"].mean())
        else:
            summary["mean_ic"] = math.nan
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows).sort_values("terminal_value", ascending=False)
    summary_df.to_csv(output_dir / "portfolio_summary.csv", index=False)

    report = [
        "# GP Scenario Riskfolio Report",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Task kernel: `{args.task_kernel}`",
        f"- Scaling: `{args.scaling}`",
        f"- Posterior scenarios per rebalance: `{args.posterior_scenarios}`",
        f"- Portfolio universe: `{', '.join(final_universe)}`",
        "",
        "## Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Notes",
        "",
        "- `gp_scenarios_riskfolio` optimizes Riskfolio directly on GP posterior scenarios.",
        "- `historical_y_ewma2_riskfolio` is the historical-return Riskfolio baseline.",
        "- `equal_weight` is an allocation baseline.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    run(parse_args())
