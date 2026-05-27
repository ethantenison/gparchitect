from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

TARGET_ETFS: tuple[str, ...] = ("SPY", "VTV", "IEF", "IWM", "VEA")
TIME_FEATURES: tuple[str, ...] = ("t_index",)
ETF_FEATURES: tuple[str, ...] = (
    "lag_y_excess_lead",
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
)
MACRO_FEATURES: tuple[str, ...] = (
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
)
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "asset_id", "y_excess_lead")
LONG_SHORT_TOP_K = 1


@dataclass(frozen=True)
class FeatureSelectionConfig:
    bayesfolio_repo: Path
    artifact_path: Path | None = None
    output_dir: Path = Path("results/feature_selection")
    target_etfs: tuple[str, ...] = TARGET_ETFS
    time_features: tuple[str, ...] = TIME_FEATURES
    etf_features: tuple[str, ...] = ETF_FEATURES
    macro_features: tuple[str, ...] = MACRO_FEATURES
    min_train_periods: int = 24
    ridge_alpha: float = 1.0

    @property
    def all_features(self) -> tuple[str, ...]:
        return (*self.time_features, *self.etf_features, *self.macro_features)

    @property
    def keep_columns(self) -> tuple[str, ...]:
        return (*REQUIRED_COLUMNS, *self.all_features)


@dataclass
class FeatureSelectionArtifacts:
    panel: pd.DataFrame
    feature_inventory: pd.DataFrame
    hygiene_report: pd.DataFrame
    rolling_windows: list["RollingWindow"]
    univariate_scores: pd.DataFrame
    family_ablation: pd.DataFrame
    subset_comparison: pd.DataFrame
    recommended_features: pd.DataFrame


@dataclass(frozen=True)
class RollingWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_date: pd.Timestamp


FAMILY_MAP = {
    "time": TIME_FEATURES,
    "etf": ETF_FEATURES,
    "macro": MACRO_FEATURES,
}


MACRO_GROUPS = {
    "credit": ("hy_spread", "hy_spread_chg_1m", "hy_spread_z_12m"),
    "vol": ("vix_slope", "vix_ts_z_12m", "vix"),
    "flow_trend": ("spy_flow_z_12m", "spy_ret", "pct_above_50dma"),
    "inflation": ("cpi_yoy", "cpi_mom"),
    "commodities_fx": ("copper_ret", "oil_ret", "gold_crude_ratio", "em_fx_ret"),
    "valuation": ("erp",),
}


def load_feature_panel(config: FeatureSelectionConfig) -> pd.DataFrame:
    artifact = config.artifact_path
    if artifact is None:
        raise ValueError("artifact_path is required for this analysis run")
    df = pd.read_parquet(artifact)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["asset_id"].isin(config.target_etfs)]
    missing = [col for col in config.keep_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    keep_cols = [col for col in df.columns if col in config.keep_columns]
    return df.loc[:, keep_cols].sort_values(["date", "asset_id"]).reset_index(drop=True)


def build_feature_inventory(config: FeatureSelectionConfig, panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, features in FAMILY_MAP.items():
        for feature in features:
            series = panel[feature]
            rows.append(
                {
                    "feature": feature,
                    "family": family,
                    "missing_frac": float(series.isna().mean()),
                    "n_unique": int(series.nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "feature"]).reset_index(drop=True)


def build_hygiene_report(panel: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        series = panel[feature]
        rows.append(
            {
                "feature": feature,
                "missing_frac": float(series.isna().mean()),
                "zero_variance": bool(series.nunique(dropna=True) <= 1),
                "min": float(series.min()) if not series.dropna().empty else None,
                "max": float(series.max()) if not series.dropna().empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def build_rolling_windows(panel: pd.DataFrame, min_train_periods: int = 24) -> list[RollingWindow]:
    monthly_dates = sorted(pd.to_datetime(panel["date"]).dropna().unique())
    return [
        RollingWindow(
            train_start=pd.Timestamp(monthly_dates[0]),
            train_end=pd.Timestamp(monthly_dates[idx - 1]),
            test_date=pd.Timestamp(monthly_dates[idx]),
        )
        for idx in range(min_train_periods, len(monthly_dates))
    ]


def _safe_rank_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return float("nan")
    return float(valid["x"].corr(valid["y"], method=method))


def _window_slices(panel: pd.DataFrame, window: RollingWindow) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = (panel["date"] >= window.train_start) & (panel["date"] <= window.train_end)
    test_mask = panel["date"] == window.test_date
    return panel.loc[train_mask].copy(), panel.loc[test_mask].copy()


def _fit_predict_ridge(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], alpha: float
) -> pd.DataFrame:
    train = train_df[[*features, "y_excess_lead"]].dropna().copy()
    test = test_df[["date", "asset_id", *features, "y_excess_lead"]].dropna().copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=["date", "asset_id", "y_true", "y_pred"])
    x_train = train[features].to_numpy(dtype=float)
    y_train = train["y_excess_lead"].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "asset_id": test["asset_id"].to_numpy(),
            "y_true": test["y_excess_lead"].to_numpy(dtype=float),
            "y_pred": preds,
        }
    )


def _metrics_from_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    if pred_df.empty:
        return {"mae": np.nan, "rmse": np.nan, "spearman": np.nan, "kendall": np.nan, "long_short": np.nan}
    err = pred_df["y_pred"] - pred_df["y_true"]
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(np.square(err))))
    spearman = _safe_rank_corr(pred_df["y_pred"], pred_df["y_true"], method="spearman")
    kendall = _safe_rank_corr(pred_df["y_pred"], pred_df["y_true"], method="kendall")
    ordered = pred_df.sort_values("y_pred", ascending=False)
    if len(ordered) < 2 * LONG_SHORT_TOP_K:
        long_short = float("nan")
    else:
        long_bucket = ordered.head(LONG_SHORT_TOP_K)["y_true"].mean()
        short_bucket = ordered.tail(LONG_SHORT_TOP_K)["y_true"].mean()
        long_short = float(long_bucket - short_bucket)
    return {"mae": mae, "rmse": rmse, "spearman": spearman, "kendall": kendall, "long_short": long_short}


def run_univariate_screen(
    panel: pd.DataFrame, windows: list[RollingWindow], features: list[str], alpha: float
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        window_metrics: list[dict[str, float]] = []
        for window in windows:
            train_df, test_df = _window_slices(panel, window)
            pred_df = _fit_predict_ridge(train_df, test_df, [feature], alpha=alpha)
            metrics = _metrics_from_predictions(pred_df)
            window_metrics.append(metrics)
        metrics_df = pd.DataFrame(window_metrics)
        rows.append(
            {
                "feature": feature,
                "family": next(name for name, cols in FAMILY_MAP.items() if feature in cols),
                "mean_mae": float(metrics_df["mae"].mean()),
                "mean_rmse": float(metrics_df["rmse"].mean()),
                "mean_spearman": float(metrics_df["spearman"].mean()),
                "mean_kendall": float(metrics_df["kendall"].mean()),
                "mean_long_short": float(metrics_df["long_short"].mean()),
                "positive_long_short_frac": float((metrics_df["long_short"] > 0).mean()),
            }
        )
    result = pd.DataFrame(rows)
    result["screen_score"] = result["mean_spearman"].fillna(-1) + 0.5 * result["mean_long_short"].fillna(0)
    return result.sort_values(["family", "screen_score"], ascending=[True, False]).reset_index(drop=True)


def _evaluate_feature_set(
    panel: pd.DataFrame, windows: list[RollingWindow], features: list[str], alpha: float
) -> dict[str, float]:
    window_metrics: list[dict[str, float]] = []
    for window in windows:
        train_df, test_df = _window_slices(panel, window)
        pred_df = _fit_predict_ridge(train_df, test_df, features, alpha=alpha)
        window_metrics.append(_metrics_from_predictions(pred_df))
    metrics_df = pd.DataFrame(window_metrics)
    return {
        "mean_mae": float(metrics_df["mae"].mean()),
        "mean_rmse": float(metrics_df["rmse"].mean()),
        "mean_spearman": float(metrics_df["spearman"].mean()),
        "mean_kendall": float(metrics_df["kendall"].mean()),
        "mean_long_short": float(metrics_df["long_short"].mean()),
    }


def run_family_ablation(panel: pd.DataFrame, windows: list[RollingWindow], alpha: float) -> pd.DataFrame:
    baseline_features = list(TIME_FEATURES + ETF_FEATURES + MACRO_FEATURES)
    baseline = _evaluate_feature_set(panel, windows, baseline_features, alpha)
    rows = [{"family": "all", **baseline, "n_features": len(baseline_features)}]
    for family, family_features in FAMILY_MAP.items():
        kept = [feature for feature in baseline_features if feature not in family_features]
        rows.append(
            {"family": f"drop_{family}", **_evaluate_feature_set(panel, windows, kept, alpha), "n_features": len(kept)}
        )
    return pd.DataFrame(rows)


def select_candidate_subsets(univariate_scores: pd.DataFrame) -> dict[str, list[str]]:
    top_by_family = {
        family: group.sort_values("screen_score", ascending=False)["feature"].tolist()
        for family, group in univariate_scores.groupby("family")
    }
    small = [*TIME_FEATURES]
    medium = [*TIME_FEATURES]
    large = [*TIME_FEATURES]
    for family, ranked in top_by_family.items():
        if family == "time":
            continue
        small.extend(ranked[:2])
        medium.extend(ranked[:4])
        large.extend(ranked[:6])
    return {"small": small, "medium": medium, "large": large}


def run_subset_comparison(
    panel: pd.DataFrame, windows: list[RollingWindow], alpha: float, univariate_scores: pd.DataFrame
) -> pd.DataFrame:
    subsets = select_candidate_subsets(univariate_scores)
    rows: list[dict[str, object]] = []
    for name, features in subsets.items():
        rows.append(
            {"subset": name, "n_features": len(features), **_evaluate_feature_set(panel, windows, features, alpha)}
        )
    return pd.DataFrame(rows).sort_values("n_features").reset_index(drop=True)


def refine_macro_features(
    panel: pd.DataFrame, windows: list[RollingWindow], alpha: float, etf_core: list[str]
) -> list[str]:
    chosen: list[str] = []
    baseline = _evaluate_feature_set(panel, windows, etf_core, alpha)
    baseline_score = baseline["mean_spearman"] + 0.5 * baseline["mean_long_short"]
    for _, group_features in MACRO_GROUPS.items():
        best_feature = None
        best_score = baseline_score
        for feature in group_features:
            score = _evaluate_feature_set(panel, windows, [*etf_core, *chosen, feature], alpha)
            composite = score["mean_spearman"] + 0.5 * score["mean_long_short"]
            if composite > best_score:
                best_score = composite
                best_feature = feature
        if best_feature is not None:
            chosen.append(best_feature)
            baseline_score = best_score
    return chosen


def recommend_feature_set(
    panel: pd.DataFrame,
    windows: list[RollingWindow],
    univariate_scores: pd.DataFrame,
    subset_comparison: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    best_subset = subset_comparison.sort_values(["mean_spearman", "mean_long_short"], ascending=[False, False]).iloc[0][
        "subset"
    ]
    conservative_etf = ["t_index", "ret_autocorr", "chmom", "baspread", "max_dd_6m"]
    conservative_macro = ["hy_spread_z_12m", "vix_ts_z_12m", "cpi_yoy", "pct_above_50dma", "erp", "oil_ret"]
    features = [*conservative_etf, *conservative_macro]
    rows = []
    for feature in features:
        family = next(name for name, cols in FAMILY_MAP.items() if feature in cols)
        score_row = univariate_scores.loc[univariate_scores["feature"] == feature].head(1)
        rows.append(
            {
                "feature": feature,
                "family": family,
                "selected_subset": best_subset,
                "screen_score": float(score_row["screen_score"].iloc[0]) if not score_row.empty else np.nan,
                "selection_rationale": "interaction-aware conservative freeze"
                if feature in conservative_macro
                else "screened ETF core",
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "feature"]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_analysis(config: FeatureSelectionConfig) -> FeatureSelectionArtifacts:
    panel = load_feature_panel(config)
    inventory = build_feature_inventory(config, panel)
    hygiene = build_hygiene_report(panel, config.all_features)
    windows = build_rolling_windows(panel, min_train_periods=config.min_train_periods)
    univariate = run_univariate_screen(panel, windows, list(config.all_features), alpha=config.ridge_alpha)
    ablation = run_family_ablation(panel, windows, alpha=config.ridge_alpha)
    subset = run_subset_comparison(panel, windows, alpha=config.ridge_alpha, univariate_scores=univariate)
    recommended = recommend_feature_set(panel, windows, univariate, subset, config.ridge_alpha)
    return FeatureSelectionArtifacts(
        panel=panel,
        feature_inventory=inventory,
        hygiene_report=hygiene,
        rolling_windows=windows,
        univariate_scores=univariate,
        family_ablation=ablation,
        subset_comparison=subset,
        recommended_features=recommended,
    )
