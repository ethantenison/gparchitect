from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRanker

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path("/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet")
BASELINE_RUN = Path(__file__).resolve().parent / "outputs" / "20260625_tvls_24w_iter15"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

ETF_UNIVERSE = ["SPY", "VTV", "IWM", "VEA", "IEF"]
TIME_COLS = ["t_index"]
ETF_COLS = [
    "baspread",
    "ret_kurt",
    "chmom",
    "mom12m",
    "mom36m",
    "cs_mom_rank",
    "max_dd_6m",
    "ma_signal",
    "ret_autocorr",
    "vol_z",
]
MACRO_COLS = [
    "hy_spread",
    "hy_spread_chg_1m",
    "hy_spread_z_12m",
    "vix_slope",
    "vix_ts_z_12m",
    "vix",
    "spy_flow_z_12m",
    "spy_ret",
    "erp",
    "cpi_yoy",
    "cpi_mom",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
]
FEATURE_COLS = [*TIME_COLS, *ETF_COLS, *MACRO_COLS]
FEATURE_SET_CHOICES = ["base", "xsec", "xsec_lag", "xsec_regime", "full_tree"]
LAG_FEATURE_COLS = ["baspread", "chmom", "mom12m", "mom36m", "max_dd_6m", "ma_signal", "ret_autocorr", "vol_z"]
REGIME_INTERACTIONS = [
    ("mom12m_cs_z", "vix_ts_z_12m"),
    ("mom36m_cs_z", "vix_ts_z_12m"),
    ("chmom_cs_z", "vix_slope"),
    ("ma_signal_cs_z", "pct_above_50dma"),
    ("max_dd_6m_cs_z", "vix_ts_z_12m"),
    ("vol_z_cs_z", "vix_ts_z_12m"),
    ("cs_mom_rank_cs_z", "erp"),
    ("baspread_cs_z", "hy_spread_z_12m"),
]


@dataclass(frozen=True)
class Window:
    train_end_date: str
    forecast_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested Optuna XGBRanker ETF ranking experiment.")
    parser.add_argument("--artifact-path", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--etf-universe", nargs="+", default=ETF_UNIVERSE)
    parser.add_argument("--last-n-windows", type=int, default=24)
    parser.add_argument("--train-date-start", type=str, default=None)
    parser.add_argument("--train-window-months", type=int, default=None)
    parser.add_argument("--forecast-date-start", type=str, default=None)
    parser.add_argument("--forecast-date-end", type=str, default=None)
    parser.add_argument(
        "--time-scale-reference",
        type=str,
        default="artifact",
        choices=["artifact", "post-train-filter"],
        help=(
            "Reference panel used to min-max scale t_index. Use 'artifact' to keep "
            "the same t_index scale across train-date filters and rolling windows."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--inner-val-months", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["rank:pairwise", "rank:ndcg", "rank:map"],
        choices=["rank:pairwise", "rank:ndcg", "rank:map"],
    )
    parser.add_argument("--feature-set", type=str, default="base", choices=FEATURE_SET_CHOICES)
    parser.add_argument(
        "--fixed-params-json",
        type=str,
        default=None,
        help="JSON object of XGBRanker params. When set, skip Optuna and reuse these params in every window.",
    )
    return parser.parse_args()


def now_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_info() -> dict[str, str]:
    def run(cmd: list[str]) -> str:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()

    return {"commit": run(["git", "rev-parse", "HEAD"]), "status_short": run(["git", "status", "--short"])}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_panel(path: Path, etf_universe: list[str]) -> pd.DataFrame:
    cols = sorted(set(["date", "asset_id", "y_excess_lead", *FEATURE_COLS]))
    df = pd.read_parquet(path)
    df = df[df["asset_id"].isin(etf_universe)].copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in [c for c in cols if c not in {"date", "asset_id"} and c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df[[c for c in cols if c in df.columns]].sort_values(["date", "asset_id"]).reset_index(drop=True)


def add_cross_sectional_features(panel: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy()
    added: list[str] = []
    for col in ETF_COLS:
        rank_col = f"{col}_cs_rank"
        z_col = f"{col}_cs_z"
        rel_col = f"{col}_rel_median"
        grouped = out.groupby("date")[col]
        out[rank_col] = grouped.rank(method="average", pct=True)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        out[z_col] = ((out[col] - mean) / std).fillna(0.0)
        out[rel_col] = out[col] - grouped.transform("median")
        added.extend([rank_col, z_col, rel_col])
    return out, [*feature_cols, *added]


def add_lag_change_features(panel: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = panel.sort_values(["asset_id", "date"]).copy()
    added: list[str] = []
    grouped = out.groupby("asset_id", sort=False)
    for col in LAG_FEATURE_COLS:
        lag1_col = f"{col}_chg_1m"
        lag3_col = f"{col}_chg_3m"
        out[lag1_col] = grouped[col].diff(1)
        out[lag3_col] = out[col] - grouped[col].shift(3)
        added.extend([lag1_col, lag3_col])
    return out.sort_values(["date", "asset_id"]).reset_index(drop=True), [*feature_cols, *added]


def add_regime_interaction_features(panel: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy()
    added: list[str] = []
    for left, right in REGIME_INTERACTIONS:
        if left not in out or right not in out:
            continue
        interaction_col = f"{left}_x_{right}"
        out[interaction_col] = out[left] * out[right]
        added.append(interaction_col)
    return out, [*feature_cols, *added]


def apply_feature_set(panel: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy()
    feature_cols = list(FEATURE_COLS)
    if feature_set in {"xsec", "xsec_lag", "xsec_regime", "full_tree"}:
        out, feature_cols = add_cross_sectional_features(out, feature_cols)
    if feature_set in {"xsec_lag", "full_tree"}:
        out, feature_cols = add_lag_change_features(out, feature_cols)
    if feature_set in {"xsec_regime", "full_tree"}:
        out, feature_cols = add_regime_interaction_features(out, feature_cols)
    return out, feature_cols


def apply_global_time_minmax(
    panel: pd.DataFrame, reference: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = panel.copy()
    reference_panel = out if reference is None else reference
    t_min = float(reference_panel["t_index"].min())
    t_max = float(reference_panel["t_index"].max())
    denom = t_max - t_min
    if denom <= 0:
        raise ValueError("Cannot scale t_index with zero range")
    out["t_index"] = (out["t_index"] - t_min) / denom
    return out, {"min": t_min, "max": t_max, "denom": denom}


def infer_windows(
    panel: pd.DataFrame,
    last_n: int,
    forecast_date_start: str | None = None,
    forecast_date_end: str | None = None,
) -> list[Window]:
    labeled_dates = sorted(panel.loc[panel["y_excess_lead"].notna(), "date"].dropna().unique())
    dates = [pd.Timestamp(d) for d in labeled_dates]
    pairs = [Window(str(dates[i - 1].date()), str(dates[i].date())) for i in range(1, len(dates))]
    if forecast_date_start is not None:
        start = pd.Timestamp(forecast_date_start)
        pairs = [window for window in pairs if pd.Timestamp(window.forecast_date) >= start]
    if forecast_date_end is not None:
        end = pd.Timestamp(forecast_date_end)
        pairs = [window for window in pairs if pd.Timestamp(window.forecast_date) <= end]
    return pairs[-last_n:]


def add_relevance_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["relevance"] = out.groupby("date")["y_excess_lead"].rank(method="first", ascending=True) - 1
    return out


def impute_features(
    train: pd.DataFrame, other: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train2 = train.copy()
    other2 = other.copy()
    medians: dict[str, float] = {}
    for col in feature_cols:
        median = finite_float(train2[col].median())
        if median is None:
            median = 0.0
        train2[col] = train2[col].fillna(median)
        other2[col] = other2[col].fillna(median)
        medians[col] = median
    return train2, other2, medians


def group_sizes(df: pd.DataFrame) -> list[int]:
    return [int(size) for size in df.groupby("date", sort=True).size().to_list()]


def fit_ranker(train: pd.DataFrame, params: dict[str, Any], seed: int, feature_cols: list[str]) -> XGBRanker:
    train = train.sort_values(["date", "asset_id"]).copy()
    params = params.copy()
    map_top_k = int(params.pop("map_top_k", 1))
    if params.get("objective") == "rank:map":
        labels = train.groupby("date")["relevance"].transform(lambda s: (s >= (s.max() - map_top_k + 1)).astype(int))
    else:
        labels = train["relevance"].astype(int)
    model = XGBRanker(
        **params,
        random_state=seed,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
    )
    model.fit(
        train[feature_cols].to_numpy(),
        labels.to_numpy(),
        group=group_sizes(train),
        verbose=False,
    )
    return model


def score_predictions(preds: pd.DataFrame) -> dict[str, Any]:
    valid = preds.dropna(subset=["y_true", "score"]).copy()
    if valid.empty:
        return {}
    by_window = []
    spread_rows = []
    top1_hits = []
    for _, group in valid.groupby("forecast_date"):
        if group["y_true"].nunique() > 1 and group["score"].nunique() > 1:
            by_window.append(float(group["y_true"].corr(group["score"], method="spearman")))
        ranked = group.sort_values("score", ascending=False)
        actual_ranked = group.sort_values("y_true", ascending=False)
        if len(ranked) >= 2:
            spread_rows.append(float(ranked.iloc[0]["y_true"] - ranked.iloc[-1]["y_true"]))
            top1_hits.append(float(ranked.iloc[0]["asset_id"] == actual_ranked.iloc[0]["asset_id"]))
    return {
        "n": int(len(valid)),
        "mean_window_spearman": float(np.nanmean(by_window)) if by_window else float("nan"),
        "long_short_top1_bottom1_mean": float(np.nanmean(spread_rows)) if spread_rows else float("nan"),
        "top1_hit_rate": float(np.nanmean(top1_hits)) if top1_hits else float("nan"),
    }


def validation_score(preds: pd.DataFrame) -> float:
    metrics = score_predictions(preds)
    score = finite_float(metrics.get("mean_window_spearman"))
    if score is None:
        return -1.0
    return score


def predict_frame(
    model: XGBRanker, frame: pd.DataFrame, model_name: str, train_end: str, feature_cols: list[str]
) -> pd.DataFrame:
    preds = frame[["date", "asset_id", "y_excess_lead"]].rename(columns={"y_excess_lead": "y_true"}).copy()
    preds["score"] = model.predict(frame[feature_cols].to_numpy())
    preds["rank_pred"] = preds["score"].rank(ascending=False, method="first")
    preds["rank_true"] = preds["y_true"].rank(ascending=False, method="first")
    preds["model"] = model_name
    preds["train_end_date"] = train_end
    preds["forecast_date"] = preds["date"].dt.date.astype(str)
    return preds


def suggest_params(trial: optuna.Trial, objectives: list[str]) -> dict[str, Any]:
    objective = objectives[0] if len(objectives) == 1 else trial.suggest_categorical("objective", objectives)
    params = {
        "objective": objective,
        "eval_metric": "ndcg",
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 1, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 8.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.70, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
        "reg_alpha": 0.0,
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }
    if objective == "rank:map":
        params["map_top_k"] = trial.suggest_categorical("map_top_k", [1, 2])
    return params


def tune_for_window(
    train_df: pd.DataFrame,
    n_trials: int,
    inner_val_months: int,
    seed: int,
    objectives: list[str],
    feature_cols: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    dates = sorted(pd.Timestamp(d) for d in train_df["date"].dropna().unique())
    if len(dates) <= inner_val_months + 3:
        inner_val_months = max(1, min(3, len(dates) // 4))
    val_dates = set(dates[-inner_val_months:])
    fit_df = train_df[~train_df["date"].isin(val_dates)].copy()
    val_df = train_df[train_df["date"].isin(val_dates)].copy()
    fit_df, val_df, _ = impute_features(fit_df, val_df, feature_cols)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, objectives)
        trial.set_user_attr("full_params", params)
        try:
            model = fit_ranker(fit_df, params, seed, feature_cols)
            preds = predict_frame(
                model, val_df, "xgb_ranker", str(pd.Timestamp(fit_df["date"].max()).date()), feature_cols
            )
            return validation_score(preds)
        except Exception as exc:  # noqa: BLE001
            trial.set_user_attr("error", str(exc))
            return -1.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    best_params = study.best_trial.user_attrs.get("full_params")
    if best_params is None:
        best_params = {
            "objective": objectives[0] if len(objectives) == 1 else study.best_trial.params["objective"],
            "eval_metric": "ndcg",
            "n_estimators": study.best_trial.params["n_estimators"],
            "max_depth": study.best_trial.params["max_depth"],
            "learning_rate": study.best_trial.params["learning_rate"],
            "min_child_weight": study.best_trial.params["min_child_weight"],
            "subsample": study.best_trial.params["subsample"],
            "colsample_bytree": study.best_trial.params["colsample_bytree"],
            "reg_alpha": 0.0,
            "reg_lambda": study.best_trial.params["reg_lambda"],
            "gamma": study.best_trial.params["gamma"],
        }
        if best_params["objective"] == "rank:map":
            best_params["map_top_k"] = study.best_trial.params["map_top_k"]
    return best_params, trials_df


def parse_fixed_params(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    params = json.loads(raw)
    if not isinstance(params, dict):
        raise ValueError("--fixed-params-json must decode to a JSON object")
    return params


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or now_run_id()
    output_dir = args.output_root / run_id
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    etf_universe = list(dict.fromkeys(args.etf_universe))
    fixed_params = parse_fixed_params(args.fixed_params_json)
    panel_artifact = load_panel(args.artifact_path, etf_universe)
    panel_raw = panel_artifact.copy()
    if args.train_date_start is not None:
        panel_raw = panel_raw.loc[panel_raw["date"] >= pd.Timestamp(args.train_date_start)].copy()
    time_scale_reference = panel_artifact if args.time_scale_reference == "artifact" else panel_raw
    panel, time_scale_params = apply_global_time_minmax(panel_raw, time_scale_reference)
    panel, feature_cols = apply_feature_set(panel, args.feature_set)
    panel = add_relevance_labels(panel)
    windows = infer_windows(panel, args.last_n_windows, args.forecast_date_start, args.forecast_date_end)

    manifest = {
        "schema": "gparchitect.xgboost_ranker_optuna.v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_path": str(args.artifact_path),
        "baseline_run": str(args.baseline_run),
        "output_dir": str(output_dir),
        "etf_universe": etf_universe,
        "feature_set": args.feature_set,
        "feature_cols": feature_cols,
        "time_scale_params": time_scale_params,
        "windows": [asdict(w) for w in windows],
        "last_n_windows": args.last_n_windows,
        "train_date_start": args.train_date_start,
        "train_window_months": args.train_window_months,
        "forecast_date_start": args.forecast_date_start,
        "forecast_date_end": args.forecast_date_end,
        "time_scale_reference": args.time_scale_reference,
        "n_trials": args.n_trials,
        "inner_val_months": args.inner_val_months,
        "objectives": args.objectives,
        "fixed_params": fixed_params,
        "skip_baseline": args.skip_baseline,
        "seed": args.seed,
        "git": git_info(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    trial_rows: list[pd.DataFrame] = []
    best_param_rows: list[dict[str, Any]] = []

    for index, window in enumerate(windows):
        train_end = pd.Timestamp(window.train_end_date)
        forecast_date = pd.Timestamp(window.forecast_date)
        train_df = panel.loc[(panel["date"] <= train_end) & panel["y_excess_lead"].notna()].copy()
        if args.train_window_months is not None:
            train_start = train_end - pd.DateOffset(months=args.train_window_months)
            train_df = train_df.loc[train_df["date"] >= train_start].copy()
        test_df = panel.loc[(panel["date"] == forecast_date) & panel["y_excess_lead"].notna()].copy()
        if test_df.empty:
            continue
        try:
            if fixed_params is None:
                best_params, trials_df = tune_for_window(
                    train_df, args.n_trials, args.inner_val_months, args.seed + index, args.objectives, feature_cols
                )
            else:
                best_params = fixed_params.copy()
                trials_df = pd.DataFrame()
            train_imp, test_imp, medians = impute_features(train_df, test_df, feature_cols)
            model = fit_ranker(train_imp, best_params, args.seed + index, feature_cols)
            model_name = "xgb_ranker_fixed" if fixed_params is not None else "xgb_ranker_optuna"
            preds = predict_frame(model, test_imp, model_name, window.train_end_date, feature_cols)
            metrics = {
                "scope": "multitask",
                "model": model_name,
                "status": "ok",
                "message": "",
                "train_end_date": window.train_end_date,
                "forecast_date": window.forecast_date,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "task_count": int(test_df["asset_id"].nunique()),
                **score_predictions(preds),
            }
            if not trials_df.empty:
                trials_df["train_end_date"] = window.train_end_date
                trials_df["forecast_date"] = window.forecast_date
                trial_rows.append(trials_df)
            best_param_rows.append(
                {
                    "train_end_date": window.train_end_date,
                    "forecast_date": window.forecast_date,
                    **best_params,
                    "feature_medians": json.dumps(medians),
                }
            )
        except Exception as exc:  # noqa: BLE001
            preds = test_df[["date", "asset_id", "y_excess_lead"]].rename(columns={"y_excess_lead": "y_true"}).copy()
            preds["score"] = np.nan
            preds["rank_pred"] = np.nan
            preds["rank_true"] = preds["y_true"].rank(ascending=False, method="first")
            preds["model"] = "xgb_ranker_fixed" if fixed_params is not None else "xgb_ranker_optuna"
            preds["train_end_date"] = window.train_end_date
            preds["forecast_date"] = window.forecast_date
            metrics = {
                "scope": "multitask",
                "model": "xgb_ranker_fixed" if fixed_params is not None else "xgb_ranker_optuna",
                "status": "failed",
                "message": str(exc),
                "train_end_date": window.train_end_date,
                "forecast_date": window.forecast_date,
            }
        pred_rows.append(preds)
        metric_rows.append(metrics)

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    trials = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()
    best_params = pd.DataFrame(best_param_rows)
    predictions.to_csv(output_dir / "xgb_ranker_predictions.csv", index=False)
    metrics.to_csv(output_dir / "xgb_ranker_window_metrics.csv", index=False)
    trials.to_csv(output_dir / "optuna_trials.csv", index=False)
    best_params.to_csv(output_dir / "best_params_by_window.csv", index=False)

    summary = summarize(metrics)
    if args.skip_baseline:
        baseline_metrics = pd.DataFrame()
        baseline_summary = pd.DataFrame()
        comparison = pd.DataFrame()
    else:
        baseline_metrics, baseline_summary = load_baseline(args.baseline_run)
        comparison = compare_to_baseline(metrics, baseline_metrics)
    summary.to_csv(output_dir / "summary.csv", index=False)
    baseline_summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2))

    write_plots(output_dir, metrics, baseline_metrics, comparison)
    write_report(output_dir, manifest, summary, baseline_summary, comparison)
    return {
        "output_dir": str(output_dir),
        "summary": summary.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    ok = metrics[metrics["status"] == "ok"].copy()
    row: dict[str, Any] = {
        "scope": "multitask",
        "model": str(ok["model"].iloc[0]) if not ok.empty and "model" in ok else "xgb_ranker",
        "ok_windows": int(len(ok)),
        "failed_windows": int((metrics["status"] != "ok").sum()),
    }
    for col in ["mean_window_spearman", "long_short_top1_bottom1_mean", "top1_hit_rate"]:
        if col in ok:
            row[col] = float(pd.to_numeric(ok[col], errors="coerce").mean())
    return pd.DataFrame([row])


def load_baseline(baseline_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(baseline_run / "multitask_window_metrics.csv")
    metrics = metrics[(metrics["model"] == "plain") & (metrics["status"] == "ok")].copy()
    preds = pd.read_csv(baseline_run / "multitask_predictions.csv")
    preds = preds[preds["model"] == "plain"].copy()
    top1_rows = []
    for forecast_date, group in preds.groupby("forecast_date"):
        ranked = group.sort_values("y_pred", ascending=False)
        actual_ranked = group.sort_values("y_true", ascending=False)
        top1_rows.append(
            {
                "forecast_date": forecast_date,
                "top1_hit_rate": float(ranked.iloc[0]["asset_id"] == actual_ranked.iloc[0]["asset_id"]),
            }
        )
    top1_df = pd.DataFrame(top1_rows)
    metrics = metrics.merge(top1_df, on="forecast_date", how="left")
    summary = pd.read_csv(baseline_run / "summary.csv")
    summary = summary[(summary["scope"] == "multitask") & (summary["model"] == "plain")].copy()
    if not top1_df.empty:
        summary["top1_hit_rate"] = float(top1_df["top1_hit_rate"].mean())
    return metrics, summary


def compare_to_baseline(xgb_metrics: pd.DataFrame, baseline_metrics: pd.DataFrame) -> pd.DataFrame:
    xgb = xgb_metrics[xgb_metrics["status"] == "ok"].copy()
    gp = baseline_metrics.copy()
    rows = []
    merged = xgb.merge(gp, on="forecast_date", suffixes=("_xgb", "_gp"))
    for metric in ["mean_window_spearman", "long_short_top1_bottom1_mean", "top1_hit_rate"]:
        x_col = f"{metric}_xgb"
        g_col = f"{metric}_gp"
        if x_col not in merged or g_col not in merged:
            continue
        delta = pd.to_numeric(merged[x_col], errors="coerce") - pd.to_numeric(merged[g_col], errors="coerce")
        rows.append(
            {
                "metric": metric,
                "xgb_mean": float(pd.to_numeric(merged[x_col], errors="coerce").mean()),
                "gp_plain_mean": float(pd.to_numeric(merged[g_col], errors="coerce").mean()),
                "delta_xgb_minus_gp": float(delta.mean()),
                "xgb_win_rate": float((delta > 0).mean()),
                "paired_windows": int(delta.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def write_plots(
    output_dir: Path, metrics: pd.DataFrame, baseline_metrics: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    ok = metrics[metrics["status"] == "ok"].copy()
    gp = baseline_metrics.copy()
    if not ok.empty:
        ok["forecast_date"] = pd.to_datetime(ok["forecast_date"])
    if not gp.empty:
        gp["forecast_date"] = pd.to_datetime(gp["forecast_date"])

    for metric in ["mean_window_spearman", "long_short_top1_bottom1_mean"]:
        fig, ax = plt.subplots(figsize=(11, 5))
        if metric in gp:
            ax.plot(gp["forecast_date"], gp[metric], marker="o", label="plain_gp")
        if metric in ok:
            ax.plot(ok["forecast_date"], ok[metric], marker="o", label="xgb_ranker_optuna")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"Window {metric}: XGBRanker vs plain GP")
        ax.set_xlabel("forecast date")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"window_{metric}.png", dpi=160)
        plt.close(fig)

    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(comparison["metric"], comparison["delta_xgb_minus_gp"])
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title("Mean metric delta: XGBRanker minus plain GP")
        ax.set_xlabel("delta")
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / "comparison_delta.png", dpi=160)
        plt.close(fig)


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    if "rank:map" in manifest.get("objectives", []):
        relevance_note = "Within-month realized `y_excess_lead` rank, 0=worst and highest=best. `rank:map` trials convert that to top-k binary relevance because XGBoost MAP requires binary labels."
    else:
        relevance_note = "Within-month realized `y_excess_lead` rank, 0=worst and highest=best."
    etf_universe = manifest.get("etf_universe", ETF_UNIVERSE)
    model_style = "fixed-parameter" if manifest.get("fixed_params") else "Optuna-tuned"
    lines = [
        f"# XGBRanker {model_style} Ranking Experiment",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Created: `{manifest['created_at_utc']}`",
        "",
        "## Question",
        "",
        "Can a directly optimized monthly ETF ranker produce useful cross-sectional rankings on the configured ETF universe?",
        "",
        "## Design",
        "",
        f"- ETF universe: `{', '.join(etf_universe)}`",
        f"- Windows: `{manifest['last_n_windows']}`",
        f"- Train date start: `{manifest.get('train_date_start')}`",
        f"- Train window months: `{manifest.get('train_window_months')}`",
        f"- Time scale reference: `{manifest.get('time_scale_reference')}`; params `{manifest.get('time_scale_params')}`",
        f"- Feature set: `{manifest.get('feature_set', 'base')}` with `{len(manifest.get('feature_cols', []))}` features",
        f"- Optuna trials per outer window: `{manifest['n_trials']}`",
        f"- Inner validation months per window: `{manifest['inner_val_months']}`",
        f"- Objectives: `{', '.join(manifest.get('objectives', []))}`",
        f"- Fixed params: `{manifest.get('fixed_params')}`",
        "- Ranking groups: one month is one query group; ETFs are items.",
        f"- Relevance target: {relevance_note}",
        "",
        "## XGBRanker Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Plain GP Baseline Summary",
        "",
        "```text",
        "Skipped" if baseline_summary.empty else baseline_summary.to_string(index=False),
        "```",
        "",
        "## Paired Comparison",
        "",
        "```text",
        "Skipped" if comparison.empty else comparison.to_string(index=False),
        "```",
        "",
        "## Visuals",
        "",
        "- `plots/window_mean_window_spearman.png`",
        "- `plots/window_long_short_top1_bottom1_mean.png`",
        "- `plots/comparison_delta.png`",
        "",
        "## Critic Pass",
        "",
        f"This is a stronger ranking-aligned baseline than RMSE-style regression, but the evidence is still limited by only {len(etf_universe)} ETFs per month and 24 OOS windows. If fixed params are used, the evaluation avoids per-window Optuna variance but still inherits any bias in how those params were chosen.",
        "",
        "## Lineage",
        "",
        f"- Manifest: `{output_dir / 'manifest.json'}`",
        f"- Feature artifact: `{manifest['artifact_path']}`",
        f"- Baseline run: `{manifest['baseline_run']}`",
        f"- Git commit: `{manifest['git']['commit']}`",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
