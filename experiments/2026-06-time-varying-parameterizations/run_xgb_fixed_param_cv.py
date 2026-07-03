from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from run_xgboost_ranker import (
    ARTIFACT_PATH,
    OUTPUT_ROOT,
    add_relevance_labels,
    apply_feature_set,
    apply_global_time_minmax,
    finite_float,
    fit_ranker,
    git_info,
    impute_features,
    infer_windows,
    load_panel,
    predict_frame,
    score_predictions,
)

ALL_18_ETFS = [
    "BND",
    "BNDX",
    "EWX",
    "HYEM",
    "HYG",
    "IEF",
    "IJR",
    "IWM",
    "LQD",
    "MGK",
    "SPY",
    "VEA",
    "VNQ",
    "VNQI",
    "VSS",
    "VTV",
    "VWO",
    "VWOB",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-aware multiobjective CV for fixed XGBRanker params.")
    parser.add_argument("--artifact-path", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--etf-universe", nargs="+", default=ALL_18_ETFS)
    parser.add_argument("--feature-set", type=str, default="full_tree")
    parser.add_argument("--train-date-start", type=str, default="2016-07-29")
    parser.add_argument("--train-window-months", type=int, default=60)
    parser.add_argument("--validation-date-start", type=str, default="2021-08-31")
    parser.add_argument("--validation-date-end", type=str, default="2024-03-29")
    parser.add_argument("--time-scale-reference", choices=["artifact", "post-train-filter"], default="artifact")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--sampler", choices=["tpe", "nsga2"], default="tpe")
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--selection-strategy",
        choices=["feasible_lexicographic", "hypervolume"],
        default="feasible_lexicographic",
    )
    return parser.parse_args()


def now_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def suggest_fixed_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "n_estimators": trial.suggest_int("n_estimators", 25, 400, step=25),
        "max_depth": trial.suggest_int("max_depth", 1, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 0.98),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.98),
        "reg_alpha": 0.0,
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
    }


def top3_metrics(preds: pd.DataFrame) -> dict[str, float]:
    contains = []
    overlaps = []
    for _, group in preds.groupby("forecast_date"):
        ranked = group.sort_values("rank_pred", ascending=True)
        actual = group.sort_values("rank_true", ascending=True)
        if len(ranked) < 3 or len(actual) < 3:
            continue
        pred_top3 = set(ranked.head(3)["asset_id"])
        true_top3 = set(actual.head(3)["asset_id"])
        true_top1 = actual.iloc[0]["asset_id"]
        contains.append(float(true_top1 in pred_top3))
        overlaps.append(float(len(pred_top3 & true_top3) / 3.0))
    return {
        "top3_contains_true_top1": float(pd.Series(contains).mean()) if contains else float("nan"),
        "top3_overlap": float(pd.Series(overlaps).mean()) if overlaps else float("nan"),
    }


def evaluate_params(
    panel: pd.DataFrame,
    windows: list[Any],
    feature_cols: list[str],
    params: dict[str, Any],
    model_seeds: list[int],
    train_window_months: int,
) -> dict[str, float]:
    seed_rows: list[dict[str, float]] = []
    for seed in model_seeds:
        pred_rows = []
        for window in windows:
            train_end = pd.Timestamp(window.train_end_date)
            forecast_date = pd.Timestamp(window.forecast_date)
            train_start = train_end - pd.DateOffset(months=train_window_months)
            train_df = panel.loc[
                (panel["date"] <= train_end) & (panel["date"] >= train_start) & panel["y_excess_lead"].notna()
            ].copy()
            test_df = panel.loc[(panel["date"] == forecast_date) & panel["y_excess_lead"].notna()].copy()
            train_imp, test_imp, _ = impute_features(train_df, test_df, feature_cols)
            model = fit_ranker(train_imp, params, seed, feature_cols)
            pred_rows.append(predict_frame(model, test_imp, "xgb_ranker_fixed_cv", window.train_end_date, feature_cols))
        preds = pd.concat(pred_rows, ignore_index=True)
        metrics = {**score_predictions(preds), **top3_metrics(preds)}
        seed_rows.append(metrics)

    metrics_df = pd.DataFrame(seed_rows)
    out = {
        "mean_window_spearman": float(metrics_df["mean_window_spearman"].mean()),
        "long_short_top1_bottom1_mean": float(metrics_df["long_short_top1_bottom1_mean"].mean()),
        "top3_contains_true_top1": float(metrics_df["top3_contains_true_top1"].mean()),
        "top3_overlap": float(metrics_df["top3_overlap"].mean()),
    }
    return {
        "mean_window_spearman": out["mean_window_spearman"]
        if finite_float(out["mean_window_spearman"]) is not None
        else -1.0,
        "long_short_top1_bottom1_mean": out["long_short_top1_bottom1_mean"]
        if finite_float(out["long_short_top1_bottom1_mean"]) is not None
        else -1.0,
        "top3_contains_true_top1": out["top3_contains_true_top1"]
        if finite_float(out["top3_contains_true_top1"]) is not None
        else 0.0,
        "top3_overlap": out["top3_overlap"] if finite_float(out["top3_overlap"]) is not None else 0.0,
    }


def select_trial(best_trials: list[optuna.trial.FrozenTrial], etf_count: int) -> optuna.trial.FrozenTrial:
    random_top3 = min(3.0 / etf_count, 1.0)
    feasible = [
        trial
        for trial in best_trials
        if finite_float(trial.values[1]) is not None
        and trial.values[1] > 0.0
        and finite_float(trial.values[2]) is not None
        and trial.values[2] >= random_top3
    ]
    candidates = feasible or best_trials
    return max(candidates, key=lambda trial: (trial.values[0], trial.values[1], trial.values[2]))


def select_trial_by_hypervolume(
    best_trials: list[optuna.trial.FrozenTrial],
    etf_count: int,
) -> optuna.trial.FrozenTrial:
    reference = (0.0, 0.0, min(3.0 / etf_count, 1.0))

    def dominated_box_volume(trial: optuna.trial.FrozenTrial) -> tuple[float, float, float, float]:
        assert trial.values is not None
        gains = tuple(max(float(value) - ref, 0.0) for value, ref in zip(trial.values, reference, strict=True))
        volume = gains[0] * gains[1] * gains[2]
        return (volume, gains[0], gains[1], gains[2])

    selected = max(best_trials, key=dominated_box_volume)
    selected.set_user_attr("hypervolume_reference", reference)
    selected.set_user_attr("hypervolume_box_volume", dominated_box_volume(selected)[0])
    return selected


def selected_params_for_trial(selected: optuna.trial.FrozenTrial) -> dict[str, Any]:
    full_params = selected.user_attrs.get("full_params")
    if full_params is not None:
        return dict(full_params)
    return {"objective": "rank:pairwise", "eval_metric": "ndcg", "reg_alpha": 0.0, **selected.params}


def write_outputs(
    output_dir: Path,
    manifest: dict[str, Any],
    study: optuna.Study,
    selected: optuna.trial.FrozenTrial,
) -> None:
    rows = []
    for trial in study.trials:
        values = trial.values or [None, None, None]
        rows.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "mean_window_spearman": values[0],
                "long_short_top1_bottom1_mean": values[1],
                "top3_contains_true_top1": values[2],
                "top3_overlap": trial.user_attrs.get("top3_overlap"),
                **(trial.user_attrs.get("full_params") or trial.params),
            }
        )
    trials_df = pd.DataFrame(rows)
    trials_df.to_csv(output_dir / "cv_trials.csv", index=False)

    pareto_rows = []
    for trial in study.best_trials:
        pareto_rows.append(
            {
                "number": trial.number,
                "mean_window_spearman": trial.values[0],
                "long_short_top1_bottom1_mean": trial.values[1],
                "top3_contains_true_top1": trial.values[2],
                "top3_overlap": trial.user_attrs.get("top3_overlap"),
                **(trial.user_attrs.get("full_params") or trial.params),
            }
        )
    pareto_df = pd.DataFrame(pareto_rows).sort_values("mean_window_spearman", ascending=False)
    pareto_df.to_csv(output_dir / "pareto_front.csv", index=False)

    selected_params = selected_params_for_trial(selected)
    (output_dir / "selected_params.json").write_text(json.dumps(selected_params, indent=2))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ok = trials_df.dropna(subset=["mean_window_spearman", "long_short_top1_bottom1_mean"]).copy()
    scatter = ax.scatter(
        ok["mean_window_spearman"],
        ok["long_short_top1_bottom1_mean"],
        c=ok["top3_contains_true_top1"],
        cmap="viridis",
        alpha=0.75,
    )
    ax.scatter(
        [selected.values[0]],
        [selected.values[1]],
        marker="*",
        s=220,
        color="red",
        label="selected",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("CV mean IC")
    ax.set_ylabel("CV long-short spread")
    ax.set_title("Multiobjective CV trials")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="top3 contains true top1")
    fig.tight_layout()
    fig.savefig(output_dir / "multiobjective_cv_trials.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Multiobjective Fixed-Parameter XGBRanker CV",
        "",
        "## Design",
        "",
        f"- Artifact: `{manifest['artifact_path']}`",
        f"- Universe: `{', '.join(manifest['etf_universe'])}`",
        f"- Feature set: `{manifest['feature_set']}`",
        f"- Validation windows: `{manifest['validation_date_start']}` through `{manifest['validation_date_end']}`",
        f"- Train date floor: `{manifest['train_date_start']}`",
        f"- Rolling train window months: `{manifest['train_window_months']}`",
        f"- Model seeds per candidate: `{manifest['model_seeds']}`",
        f"- Trials: `{manifest['n_trials']}`",
        f"- Selection strategy: `{manifest['selection_strategy']}`",
        "",
        "## Objectives",
        "",
        "- Maximize validation mean IC.",
        "- Maximize validation top-bottom spread.",
        "- Maximize probability that predicted top3 contains the true top1 ETF.",
        "",
        "## Selected Trial",
        "",
        f"- Trial: `{selected.number}`",
        f"- Mean IC: `{selected.values[0]:.4f}`",
        f"- Spread: `{selected.values[1]:.4f}`",
        f"- Top3 contains true top1: `{selected.values[2]:.4f}`",
        f"- Top3 overlap: `{selected.user_attrs.get('top3_overlap'):.4f}`",
        f"- Hypervolume reference: `{selected.user_attrs.get('hypervolume_reference')}`",
        f"- Hypervolume box volume: `{selected.user_attrs.get('hypervolume_box_volume')}`",
        "",
        "```json",
        json.dumps(selected_params, indent=2),
        "```",
        "",
        "## Artifacts",
        "",
        "- `cv_trials.csv`",
        "- `pareto_front.csv`",
        "- `selected_params.json`",
        "- `multiobjective_cv_trials.png`",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def run_cv(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.run_id or now_run_id()
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    etf_universe = list(dict.fromkeys(args.etf_universe))
    panel_artifact = load_panel(args.artifact_path, etf_universe)
    panel_raw = panel_artifact.copy()
    if args.train_date_start is not None:
        panel_raw = panel_raw.loc[panel_raw["date"] >= pd.Timestamp(args.train_date_start)].copy()
    time_scale_reference = panel_artifact if args.time_scale_reference == "artifact" else panel_raw
    panel, time_scale_params = apply_global_time_minmax(panel_raw, time_scale_reference)
    panel, feature_cols = apply_feature_set(panel, args.feature_set)
    panel = add_relevance_labels(panel)
    windows = infer_windows(
        panel,
        10_000,
        forecast_date_start=args.validation_date_start,
        forecast_date_end=args.validation_date_end,
    )

    manifest = {
        "schema": "gparchitect.xgboost_ranker_fixed_param_cv.v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_path": str(args.artifact_path),
        "output_dir": str(output_dir),
        "etf_universe": etf_universe,
        "feature_set": args.feature_set,
        "feature_cols": feature_cols,
        "train_date_start": args.train_date_start,
        "train_window_months": args.train_window_months,
        "validation_date_start": args.validation_date_start,
        "validation_date_end": args.validation_date_end,
        "validation_windows": [asdict(window) for window in windows],
        "time_scale_reference": args.time_scale_reference,
        "time_scale_params": time_scale_params,
        "n_trials": args.n_trials,
        "sampler": args.sampler,
        "sampler_seed": args.sampler_seed,
        "model_seeds": args.model_seeds,
        "selection_strategy": args.selection_strategy,
        "search_space": {
            "n_estimators": "25..400 step 25",
            "max_depth": "1..8",
            "learning_rate": "0.01..0.30 log",
            "min_child_weight": "1.0..12.0 log",
            "subsample": "0.60..0.98",
            "colsample_bytree": "0.60..0.98",
            "reg_alpha": "fixed 0.0",
            "reg_lambda": "1e-4..100.0 log",
            "gamma": "1e-8..10.0 log",
        },
        "objectives": [
            "mean_window_spearman",
            "long_short_top1_bottom1_mean",
            "top3_contains_true_top1",
        ],
        "git": git_info(),
    }

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    else:
        sampler = optuna.samplers.NSGAIISampler(seed=args.sampler_seed)
    study = optuna.create_study(directions=["maximize", "maximize", "maximize"], sampler=sampler)

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = suggest_fixed_params(trial)
        metrics = evaluate_params(panel, windows, feature_cols, params, args.model_seeds, args.train_window_months)
        trial.set_user_attr("full_params", params)
        trial.set_user_attr("top3_overlap", metrics["top3_overlap"])
        return (
            metrics["mean_window_spearman"],
            metrics["long_short_top1_bottom1_mean"],
            metrics["top3_contains_true_top1"],
        )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    if args.selection_strategy == "hypervolume":
        selected = select_trial_by_hypervolume(study.best_trials, len(etf_universe))
    else:
        selected = select_trial(study.best_trials, len(etf_universe))
    write_outputs(output_dir, manifest, study, selected)
    return {
        "output_dir": str(output_dir),
        "selected_trial": selected.number,
        "selected_values": selected.values,
        "selected_params": selected_params_for_trial(selected),
        "pareto_trials": [trial.number for trial in study.best_trials],
    }


def main() -> None:
    result = run_cv(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
