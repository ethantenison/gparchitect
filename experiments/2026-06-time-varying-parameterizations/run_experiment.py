from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import types
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize, StratifiedStandardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
BAYESFOLIO_REPO = Path("/Users/et/Documents/BayesFolio")
ARTIFACT_PATH = Path("/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet")
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

if str(BAYESFOLIO_REPO) not in sys.path:
    sys.path.insert(0, str(BAYESFOLIO_REPO))

ETF_UNIVERSE = ["SPY", "VTV", "IWM", "VEA", "IEF"]
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
SPY_FEATURE_COLS = [
    "lag_y_excess_lead",
    "ret_autocorr",
    "chmom",
    "baspread",
    "max_dd_6m",
    "hy_spread_z_12m",
    "vix_ts_z_12m",
    "cpi_yoy",
    "pct_above_50dma",
    "erp",
    "oil_ret",
]
MULTITASK_INPUT_COLUMNS = [*TIME_COLS, *ETF_COLS, *MACRO_COLS]
SPY_INPUT_COLUMNS = [*TIME_COLS, *SPY_FEATURE_COLS]


@dataclass(frozen=True)
class Variant:
    name: str
    parameterization: str
    target: str


VARIANTS = [
    Variant("plain", "none", "none"),
    Variant("linear_tvls", "linear", "lengthscale"),
    Variant("linear_tvos", "linear", "outputscale"),
    Variant("linear_tvls_tvos", "linear", "both"),
    Variant("spline3_tvls", "spline3", "lengthscale"),
    Variant("spline3_tvos", "spline3", "outputscale"),
    Variant("spline3_tvls_tvos", "spline3", "both"),
    Variant("piecewise_linear_tvls", "piecewise_linear", "lengthscale"),
    Variant("piecewise_linear_tvos", "piecewise_linear", "outputscale"),
    Variant("piecewise_linear_tvls_tvos", "piecewise_linear", "both"),
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_bayesfolio_gp_shim() -> None:
    for name in ["bayesfolio", "bayesfolio.engine", "bayesfolio.engine.forecast", "bayesfolio.engine.forecast.gp"]:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = package
    _load_module(
        "bayesfolio.engine.forecast.gp.time_varying_kernel",
        BAYESFOLIO_REPO / "bayesfolio/engine/forecast/gp/time_varying_kernel.py",
    )


_install_bayesfolio_gp_shim()
gp_data_prep = _load_module("bf_gp_data_prep", BAYESFOLIO_REPO / "bayesfolio/engine/features/gp_data_prep.py")
multitask_builder = _load_module(
    "bf_multitask_builder", BAYESFOLIO_REPO / "bayesfolio/engine/forecast/gp/multitask_builder.py"
)

prepare_multitask_gp_data_with_task_feature = gp_data_prep.prepare_multitask_gp_data_with_task_feature
BlockStructure = multitask_builder.BlockStructure
CovarModuleConfig = multitask_builder.CovarModuleConfig
GlobalStructure = multitask_builder.GlobalStructure
InteractionPolicy = multitask_builder.InteractionPolicy
KernelBlockConfig = multitask_builder.KernelBlockConfig
KernelBlockRole = multitask_builder.KernelBlockRole
KernelInteractionConfig = multitask_builder.KernelInteractionConfig
LengthscalePolicy = multitask_builder.LengthscalePolicy
LengthscalePolicyConfig = multitask_builder.LengthscalePolicyConfig
LinearKernelComponentConfig = multitask_builder.LinearKernelComponentConfig
MaternKernelComponentConfig = multitask_builder.MaternKernelComponentConfig
MeanKind = multitask_builder.MeanKind
MeanModuleConfig = multitask_builder.MeanModuleConfig
RQKernelComponentConfig = multitask_builder.RQKernelComponentConfig
build_covar_module = multitask_builder.build_covar_module
build_multitask_gp = multitask_builder.build_multitask_gp

for _name in [
    "LengthscalePolicyConfig",
    "MaternKernelComponentConfig",
    "RQKernelComponentConfig",
    "LinearKernelComponentConfig",
    "KernelInteractionConfig",
    "KernelBlockConfig",
    "CovarModuleConfig",
    "MeanModuleConfig",
]:
    cls = getattr(multitask_builder, _name, None)
    if cls is not None and hasattr(cls, "model_rebuild"):
        cls.model_rebuild(_types_namespace=multitask_builder.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded TVOS/TVLS parameterization experiment")
    parser.add_argument("--artifact-path", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--last-n-windows", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--etf-universe", nargs="+", default=ETF_UNIVERSE)
    parser.add_argument("--train-window-months", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=[v.name for v in VARIANTS], choices=[v.name for v in VARIANTS])
    parser.add_argument("--skip-fake", action="store_true")
    parser.add_argument("--skip-spy", action="store_true")
    parser.add_argument("--skip-multitask", action="store_true")
    parser.add_argument(
        "--write-qmd", action="store_true", help="Write a Quarto report source into the output directory."
    )
    return parser.parse_args()


def now_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_info() -> dict[str, str]:
    def run(cmd: list[str]) -> str:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status_short": run(["git", "status", "--short"]),
    }


def set_seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class TimeVaryingKernel(torch.nn.Module):
    def __new__(
        cls,
        base_kernel: Any,
        *,
        time_feature_index: int,
        target: str,
        parameterization: str,
        coefficient_limit: float = 3.0,
    ) -> Any:
        import gpytorch

        class _Module(gpytorch.kernels.Kernel):
            has_lengthscale = False

            def __init__(self) -> None:
                super().__init__()
                self.base_kernel = base_kernel
                self.time_feature_index = time_feature_index
                self.target = target
                self.parameterization = parameterization
                self.coefficient_limit = coefficient_limit
                self.register_parameter("raw_tv_weights", torch.nn.Parameter(torch.zeros(self._basis_dim())))

            def _basis_dim(self) -> int:
                if self.parameterization == "linear":
                    return 2
                if self.parameterization == "piecewise_linear":
                    return 4
                if self.parameterization == "spline3":
                    return 5
                raise ValueError(f"Unsupported parameterization: {self.parameterization}")

            def _bspline3_basis(self, t: torch.Tensor) -> torch.Tensor:
                # Open-uniform cubic B-spline basis with one interior knot at 0.5.
                knots = t.new_tensor([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
                x = t.clamp(0.0, 1.0)
                bases = []
                for i in range(len(knots) - 1):
                    right_edge = (i == len(knots) - 2) & (x == knots[-1])
                    bases.append(((x >= knots[i]) & (x < knots[i + 1]) | right_edge).to(x.dtype))
                basis = torch.stack(bases, dim=-1)
                for degree in range(1, 4):
                    next_basis = []
                    for i in range(len(knots) - degree - 1):
                        left_den = knots[i + degree] - knots[i]
                        right_den = knots[i + degree + 1] - knots[i + 1]
                        left = (
                            torch.zeros_like(x)
                            if float(left_den) == 0.0
                            else ((x - knots[i]) / left_den) * basis[..., i]
                        )
                        right = (
                            torch.zeros_like(x)
                            if float(right_den) == 0.0
                            else ((knots[i + degree + 1] - x) / right_den) * basis[..., i + 1]
                        )
                        next_basis.append(left + right)
                    basis = torch.stack(next_basis, dim=-1)
                return basis

            def _basis(self, x: torch.Tensor) -> torch.Tensor:
                t = x[..., self.time_feature_index].clamp(0.0, 1.0)
                if self.parameterization == "linear":
                    return torch.stack([torch.ones_like(t), t], dim=-1)
                if self.parameterization == "piecewise_linear":
                    return torch.stack(
                        [torch.ones_like(t), t, torch.relu(t - 1.0 / 3.0), torch.relu(t - 2.0 / 3.0)], dim=-1
                    )
                if self.parameterization == "spline3":
                    return self._bspline3_basis(t)
                raise ValueError(f"Unsupported parameterization: {self.parameterization}")

            def _modulation(self, x: torch.Tensor) -> torch.Tensor:
                weights = self.coefficient_limit * torch.tanh(self.raw_tv_weights).to(dtype=x.dtype, device=x.device)
                eta = self._basis(x) @ weights
                return torch.nn.functional.softplus(eta) + 1e-4

            def modulation_curve(self, t: torch.Tensor) -> torch.Tensor:
                if t.ndim == 1:
                    x = torch.zeros(t.shape[0], max(self.time_feature_index + 1, 1), dtype=t.dtype, device=t.device)
                    x[:, self.time_feature_index] = t
                else:
                    x = t
                return self._modulation(x)

            def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: object) -> torch.Tensor:
                if self.target == "outputscale":
                    s1 = self._modulation(x1)
                    s2 = self._modulation(x2)
                    k_base = self.base_kernel(x1, x2, **kwargs).to_dense()
                    return s1.unsqueeze(-1) * k_base * s2.unsqueeze(-2)
                if self.target == "lengthscale":
                    x1_warped = x1.clone()
                    x2_warped = x2.clone()
                    x1_warped[..., self.time_feature_index] = x1[..., self.time_feature_index] / self._modulation(x1)
                    x2_warped[..., self.time_feature_index] = x2[..., self.time_feature_index] / self._modulation(x2)
                    return self.base_kernel(x1_warped, x2_warped, **kwargs).to_dense()
                raise ValueError(f"Unsupported target: {self.target}")

        return _Module()


def wrap_kernel(base_kernel: Any, variant: Variant, *, time_feature_index: int = 0) -> Any:
    if variant.target == "none":
        return base_kernel
    if variant.target == "both":
        wrapped = TimeVaryingKernel(
            base_kernel,
            time_feature_index=time_feature_index,
            target="lengthscale",
            parameterization=variant.parameterization,
        )
        return TimeVaryingKernel(
            wrapped,
            time_feature_index=time_feature_index,
            target="outputscale",
            parameterization=variant.parameterization,
        )
    return TimeVaryingKernel(
        base_kernel,
        time_feature_index=time_feature_index,
        target=variant.target,
        parameterization=variant.parameterization,
    )


def iter_tv_modules(module: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in module.modules() if hasattr(m, "modulation_curve") and hasattr(m, "target")]


def modulation_rows(
    model: torch.nn.Module, variant: Variant, scope: str, window: str | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t_grid = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    for index, module in enumerate(iter_tv_modules(model)):
        with torch.no_grad():
            values = module.modulation_curve(t_grid).detach().cpu().double().numpy()
        for t, value in zip(t_grid.numpy(), values, strict=True):
            rows.append(
                {
                    "scope": scope,
                    "model": variant.name,
                    "parameterization": variant.parameterization,
                    "target": getattr(module, "target", "unknown"),
                    "module_index": index,
                    "window": window,
                    "t": float(t),
                    "modulation": float(value),
                }
            )
    return rows


def standard_scale(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    train2 = train.copy()
    test2 = test.copy()
    params: dict[str, dict[str, float]] = {}
    for col in cols:
        mean = float(train2[col].mean())
        std = float(train2[col].std())
        if not math.isfinite(std) or std <= 1e-12:
            std = 1.0
        train2[col] = (train2[col] - mean) / std
        test2[col] = (test2[col] - mean) / std
        params[col] = {"mean": mean, "std": std}
    return train2, test2, params


def unscale(values: pd.Series | np.ndarray, mean: float, std: float) -> pd.Series | np.ndarray:
    return values * std + mean


def load_panel(path: Path, universe: list[str]) -> pd.DataFrame:
    cols = sorted(set(["date", "asset_id", "y_excess_lead", *MULTITASK_INPUT_COLUMNS, *SPY_INPUT_COLUMNS]))
    df = pd.read_parquet(path)
    df = df[df["asset_id"].isin(universe)].copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in [c for c in cols if c not in {"date", "asset_id"} and c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df[[c for c in cols if c in df.columns]].sort_values(["date", "asset_id"]).reset_index(drop=True)


def apply_global_time_minmax(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = panel.copy()
    t_min = float(out["t_index"].min())
    t_max = float(out["t_index"].max())
    denom = t_max - t_min
    if denom <= 0:
        raise ValueError("Cannot scale t_index with zero range")
    out["t_index"] = (out["t_index"] - t_min) / denom
    return out, {"min": t_min, "max": t_max, "denom": denom}


def infer_windows(panel: pd.DataFrame, last_n: int) -> list[tuple[str, str]]:
    labeled_dates = sorted(panel.loc[panel["y_excess_lead"].notna(), "date"].dropna().unique())
    dates = [pd.Timestamp(d) for d in labeled_dates]
    pairs = [(str(dates[i - 1].date()), str(dates[i].date())) for i in range(1, len(dates))]
    return pairs[-last_n:]


def predictive_log_likelihood(y_true: pd.Series, y_pred: pd.Series, pred_std: pd.Series) -> float:
    std = pred_std.clip(lower=1e-12)
    ll = -0.5 * (((y_true - y_pred) / std) ** 2 + 2.0 * np.log(std) + math.log(2.0 * math.pi))
    return float(ll.mean())


def score_predictions(preds: pd.DataFrame, *, multitask: bool) -> dict[str, Any]:
    valid = preds.dropna(subset=["y_true", "y_pred", "pred_std"]).copy()
    if valid.empty:
        return {}
    err = valid["y_pred"] - valid["y_true"]
    out: dict[str, Any] = {
        "n": int(len(valid)),
        "mae": float(err.abs().mean()),
        "rmse": float((err.pow(2).mean()) ** 0.5),
        "predictive_log_likelihood": predictive_log_likelihood(valid["y_true"], valid["y_pred"], valid["pred_std"]),
        "interval_coverage_95": float(
            (
                (valid["y_true"] >= valid["y_pred"] - 1.96 * valid["pred_std"])
                & (valid["y_true"] <= valid["y_pred"] + 1.96 * valid["pred_std"])
            ).mean()
        ),
        "directional_accuracy": float(((valid["y_true"] > 0) == (valid["y_pred"] > 0)).mean()),
        "mean_pred_std": float(valid["pred_std"].mean()),
    }
    if multitask and valid["asset_id"].nunique() > 1:
        by_window = []
        for _, group in valid.groupby("forecast_date"):
            if group["y_true"].nunique() > 1 and group["y_pred"].nunique() > 1:
                by_window.append(float(group["y_true"].corr(group["y_pred"], method="spearman")))
        out["mean_window_spearman"] = float(np.nanmean(by_window)) if by_window else float("nan")
        ranked = valid.sort_values(["forecast_date", "y_pred"], ascending=[True, False])
        spread_rows = []
        for _, group in ranked.groupby("forecast_date"):
            if len(group) >= 2:
                spread_rows.append(float(group.iloc[0]["y_true"] - group.iloc[-1]["y_true"]))
        out["long_short_top1_bottom1_mean"] = float(np.nanmean(spread_rows)) if spread_rows else float("nan")
    return out


def build_single_task_kernel(variant: Variant, input_dim: int) -> Any:
    base = MaternKernel(nu=0.5, ard_num_dims=input_dim)
    return wrap_kernel(base, variant, time_feature_index=0)


def fit_single_task(
    train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, variant: Variant, max_iter: int
) -> tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    set_seed()
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior, noise_constraint=GreaterThan(5e-3, initial_value=noise_prior.mode)
    )
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=build_single_task_kernel(variant, train_x.shape[-1]),
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": max_iter})
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        posterior = model.posterior(test_x, observation_noise=True)
        pred_mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
        pred_std = posterior.variance.squeeze(-1).clamp_min(0.0).sqrt().detach().cpu().numpy()
    return pred_mean, pred_std, model


def run_fake_data(variants: list[Variant], output_dir: Path, max_iter: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    t = np.linspace(0.0, 1.0, 72)
    x = torch.tensor(t.reshape(-1, 1), dtype=torch.float64)
    split = 54
    scenarios = {
        "amplitude_bump": {
            "y": (0.30 + 1.30 * np.exp(-0.5 * ((t - 0.62) / 0.16) ** 2)) * np.sin(2.0 * np.pi * 4.0 * t),
            "true_curve": 0.30 + 1.30 * np.exp(-0.5 * ((t - 0.62) / 0.16) ** 2),
        },
        "frequency_ramp": {
            "y": np.sin(2.0 * np.pi * (1.5 + 4.0 * t**2) * t),
            "true_curve": 1.0 / (1.5 + 4.0 * t**2),
        },
    }
    rng = np.random.default_rng(0)
    for scenario, payload in scenarios.items():
        y = payload["y"] + rng.normal(0.0, 0.08, size=t.shape[0])
        train_x = x[:split]
        test_x = x[split:]
        train_y = torch.tensor(y[:split].reshape(-1, 1), dtype=torch.float64)
        test_y = y[split:]
        for variant in variants:
            if variant.name == "plain":
                applicable = True
            elif scenario == "amplitude_bump":
                applicable = variant.target in {"outputscale", "both"}
            else:
                applicable = variant.target in {"lengthscale", "both"}
            if not applicable:
                continue
            try:
                pred_mean, pred_std, model = fit_single_task(train_x, train_y, test_x, variant, max_iter)
                err = pred_mean - test_y
                status = "ok"
                message = ""
                for curve_row in modulation_rows(model, variant, "fake", scenario):
                    curves.append(curve_row)
                learned = [
                    r["modulation"]
                    for r in curves
                    if r["scope"] == "fake" and r["model"] == variant.name and r["window"] == scenario
                ]
                true_curve = np.interp(np.linspace(0, 1, len(learned)), t, payload["true_curve"]) if learned else []
                curve_spearman = (
                    float(spearmanr(true_curve, learned).statistic)
                    if len(learned) > 2 and np.std(learned) > 0
                    else float("nan")
                )
            except Exception as exc:  # noqa: BLE001
                pred_std = np.array([])
                err = np.array([np.nan])
                status = "failed"
                message = str(exc)
                curve_spearman = float("nan")
            rmse = float(np.sqrt(np.nanmean(err**2))) if np.isfinite(err).any() else float("nan")
            mae = float(np.nanmean(np.abs(err))) if np.isfinite(err).any() else float("nan")
            rows.append(
                {
                    "scope": "fake",
                    "scenario": scenario,
                    "model": variant.name,
                    "parameterization": variant.parameterization,
                    "target": variant.target,
                    "status": status,
                    "message": message,
                    "rmse": rmse,
                    "mae": mae,
                    "mean_pred_std": float(np.mean(pred_std)) if len(pred_std) else float("nan"),
                    "curve_spearman_proxy": curve_spearman,
                }
            )
    fake_df = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curves)
    fake_df.to_csv(output_dir / "fake_recovery_metrics.csv", index=False)
    curve_df.to_csv(output_dir / "fake_modulation_curves.csv", index=False)
    return fake_df, curve_df


def training_frame(
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    *,
    train_window_months: int | None,
) -> pd.DataFrame:
    mask = (panel["date"] <= train_end) & panel["y_excess_lead"].notna()
    if train_window_months is not None:
        train_start = train_end - pd.DateOffset(months=train_window_months)
        mask &= panel["date"] >= train_start
    return panel.loc[mask].copy()


def run_spy(
    panel: pd.DataFrame,
    variants: list[Variant],
    windows: list[tuple[str, str]],
    output_dir: Path,
    max_iter: int,
    *,
    train_window_months: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    spy_panel = panel[panel["asset_id"] == "SPY"].copy()
    for variant in variants:
        for train_end_s, forecast_date_s in windows:
            train_end = pd.Timestamp(train_end_s)
            forecast_date = pd.Timestamp(forecast_date_s)
            train_df = training_frame(spy_panel, train_end, train_window_months=train_window_months)
            test_df = spy_panel.loc[(spy_panel["date"] == forecast_date) & spy_panel["y_excess_lead"].notna()].copy()
            if test_df.empty:
                continue
            train_scaled, test_scaled, scale_params = standard_scale(
                train_df, test_df, [*SPY_FEATURE_COLS, "y_excess_lead"]
            )
            train_x = torch.tensor(train_scaled[SPY_INPUT_COLUMNS].to_numpy(), dtype=torch.float64)
            train_y = torch.tensor(train_scaled["y_excess_lead"].to_numpy(), dtype=torch.float64).unsqueeze(-1)
            test_x = torch.tensor(test_scaled[SPY_INPUT_COLUMNS].to_numpy(), dtype=torch.float64)
            try:
                pred_mean, pred_std, model = fit_single_task(train_x, train_y, test_x, variant, max_iter)
                y_params = scale_params["y_excess_lead"]
                preds = (
                    test_df[["date", "asset_id", "y_excess_lead"]].rename(columns={"y_excess_lead": "y_true"}).copy()
                )
                preds["y_pred"] = unscale(pred_mean, y_params["mean"], y_params["std"])
                preds["pred_std"] = pred_std * y_params["std"]
                preds["model"] = variant.name
                preds["train_end_date"] = str(train_end.date())
                preds["forecast_date"] = str(forecast_date.date())
                metrics = {
                    "scope": "spy",
                    "status": "ok",
                    "message": "",
                    "model": variant.name,
                    "train_end_date": str(train_end.date()),
                    "forecast_date": str(forecast_date.date()),
                    **score_predictions(preds, multitask=False),
                }
                curve_rows.extend(modulation_rows(model, variant, "spy", str(forecast_date.date())))
            except Exception as exc:  # noqa: BLE001
                preds = (
                    test_df[["date", "asset_id", "y_excess_lead"]].rename(columns={"y_excess_lead": "y_true"}).copy()
                )
                preds["y_pred"] = np.nan
                preds["pred_std"] = np.nan
                preds["model"] = variant.name
                preds["train_end_date"] = str(train_end.date())
                preds["forecast_date"] = str(forecast_date.date())
                metrics = {
                    "scope": "spy",
                    "status": "failed",
                    "message": str(exc),
                    "model": variant.name,
                    "train_end_date": str(train_end.date()),
                    "forecast_date": str(forecast_date.date()),
                }
            pred_rows.append(preds)
            metric_rows.append(metrics)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    curves_df = pd.DataFrame(curve_rows)
    preds_df.to_csv(output_dir / "spy_predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "spy_window_metrics.csv", index=False)
    curves_df.to_csv(output_dir / "spy_modulation_curves.csv", index=False)
    return preds_df, metrics_df, curves_df


def build_multitask_config() -> Any:
    idx = {name: i for i, name in enumerate(MULTITASK_INPUT_COLUMNS)}
    time_dims = [idx[c] for c in TIME_COLS]
    etf_dims = [idx[c] for c in ETF_COLS]
    macro_dims = [idx[c] for c in MACRO_COLS]
    macro_components = [
        MaternKernelComponentConfig(
            dims=macro_dims,
            matern_nu=0.5,
            ard=True,
            use_outputscale=True,
            lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
        ),
        RQKernelComponentConfig(
            dims=macro_dims,
            ard=True,
            use_outputscale=True,
            lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
        ),
        LinearKernelComponentConfig(dims=macro_dims, use_outputscale=True),
    ]
    return CovarModuleConfig(
        blocks=[
            KernelBlockConfig(
                name="time",
                variable_type=KernelBlockRole.TIME,
                components=[
                    MaternKernelComponentConfig(
                        dims=time_dims,
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                    )
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=False,
            ),
            KernelBlockConfig(
                name="etf",
                variable_type=KernelBlockRole.ETF,
                components=[
                    MaternKernelComponentConfig(
                        dims=etf_dims,
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                    )
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=False,
            ),
            KernelBlockConfig(
                name="macro",
                variable_type=KernelBlockRole.MACRO,
                components=macro_components,
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=False,
            ),
        ],
        global_structure=GlobalStructure.HIERARCHICAL,
        interaction_policy=InteractionPolicy.CUSTOM,
        custom_interactions=[
            KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            KernelInteractionConfig(blocks=["etf", "macro"], name="etf_x_macro", use_outputscale=True),
        ],
    )


def build_outcome_transform(train_x: torch.Tensor, train_y: torch.Tensor) -> Any:
    task_feature_idx = train_x.shape[-1] - 1
    task_values = train_x[:, task_feature_idx].to(torch.long)
    try:
        return StratifiedStandardize(
            stratification_idx=task_feature_idx,
            all_task_values=task_values.unique(sorted=True),
            observed_task_values=task_values,
            batch_shape=train_y.shape[:-2],
        )
    except TypeError:
        return StratifiedStandardize(
            stratification_idx=task_feature_idx,
            all_task_values=task_values.unique(sorted=True),
            batch_shape=train_y.shape[:-2],
        )


def fit_multitask(
    train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, variant: Variant, rank: int, max_iter: int
) -> tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    set_seed()
    covar_config = build_multitask_config()
    covar_module = build_covar_module(covar_config, batch_shape=train_x.shape[:-2])
    covar_module = wrap_kernel(covar_module, variant, time_feature_index=0)
    model = build_multitask_gp(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
        covar_config=covar_config,
        mean_config=MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT),
        rank=rank,
        min_inferred_noise_level=5e-3,
        outcome_transform=build_outcome_transform(train_x, train_y),
        input_transform=None,
        task_covar_prior=None,
    )
    model.covar_module = covar_module
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={"maxiter": max_iter})
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        posterior = model.posterior(test_x, observation_noise=True)
        pred_mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
        pred_std = posterior.variance.squeeze(-1).clamp_min(0.0).sqrt().detach().cpu().numpy()
    return pred_mean, pred_std, model


def run_multitask(
    panel: pd.DataFrame,
    variants: list[Variant],
    windows: list[tuple[str, str]],
    output_dir: Path,
    rank: int,
    max_iter: int,
    *,
    train_window_months: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for variant in variants:
        for train_end_s, forecast_date_s in windows:
            train_end = pd.Timestamp(train_end_s)
            forecast_date = pd.Timestamp(forecast_date_s)
            train_df = training_frame(panel, train_end, train_window_months=train_window_months)
            test_df = panel.loc[(panel["date"] == forecast_date) & panel["y_excess_lead"].notna()].copy()
            if test_df.empty:
                continue
            train_scaled, test_scaled, scale_params = standard_scale(
                train_df, test_df, [*ETF_COLS, *MACRO_COLS, "y_excess_lead"]
            )
            train_x, train_y, task_map = prepare_multitask_gp_data_with_task_feature(
                train_scaled[[*MULTITASK_INPUT_COLUMNS, "asset_id", "y_excess_lead"]],
                target_col="y_excess_lead",
                asset_col="asset_id",
                drop_cols=[],
                dtype=torch.float64,
            )
            test_x, _, _ = prepare_multitask_gp_data_with_task_feature(
                test_scaled[[*MULTITASK_INPUT_COLUMNS, "asset_id", "y_excess_lead"]],
                target_col="y_excess_lead",
                asset_col="asset_id",
                drop_cols=[],
                dtype=torch.float64,
            )
            try:
                pred_mean, pred_std, model = fit_multitask(train_x, train_y, test_x, variant, rank, max_iter)
                y_params = scale_params["y_excess_lead"]
                preds = (
                    test_df[["date", "asset_id", "y_excess_lead"]].rename(columns={"y_excess_lead": "y_true"}).copy()
                )
                preds["y_pred"] = unscale(pred_mean, y_params["mean"], y_params["std"])
                preds["pred_std"] = pred_std * y_params["std"]
                preds["rank_pred"] = preds["y_pred"].rank(ascending=False, method="first")
                preds["rank_true"] = preds["y_true"].rank(ascending=False, method="first")
                preds["model"] = variant.name
                preds["train_end_date"] = str(train_end.date())
                preds["forecast_date"] = str(forecast_date.date())
                metrics = {
                    "scope": "multitask",
                    "status": "ok",
                    "message": "",
                    "model": variant.name,
                    "train_end_date": str(train_end.date()),
                    "forecast_date": str(forecast_date.date()),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "task_count": int(len(task_map)),
                    **score_predictions(preds, multitask=True),
                }
                curve_rows.extend(modulation_rows(model, variant, "multitask", str(forecast_date.date())))
            except Exception as exc:  # noqa: BLE001
                preds = (
                    test_df[["date", "asset_id", "y_excess_lead"]].rename(columns={"y_excess_lead": "y_true"}).copy()
                )
                preds["y_pred"] = np.nan
                preds["pred_std"] = np.nan
                preds["rank_pred"] = np.nan
                preds["rank_true"] = preds["y_true"].rank(ascending=False, method="first")
                preds["model"] = variant.name
                preds["train_end_date"] = str(train_end.date())
                preds["forecast_date"] = str(forecast_date.date())
                metrics = {
                    "scope": "multitask",
                    "status": "failed",
                    "message": str(exc),
                    "model": variant.name,
                    "train_end_date": str(train_end.date()),
                    "forecast_date": str(forecast_date.date()),
                }
            pred_rows.append(preds)
            metric_rows.append(metrics)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    curves_df = pd.DataFrame(curve_rows)
    preds_df.to_csv(output_dir / "multitask_predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "multitask_window_metrics.csv", index=False)
    curves_df.to_csv(output_dir / "multitask_modulation_curves.csv", index=False)
    return preds_df, metrics_df, curves_df


def aggregate_window_metrics(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for model, group in metrics.groupby("model"):
        ok = group[group["status"] == "ok"].copy()
        row: dict[str, Any] = {
            "scope": scope,
            "model": model,
            "ok_windows": int(len(ok)),
            "failed_windows": int((group["status"] != "ok").sum()),
        }
        for col in [
            "mae",
            "rmse",
            "predictive_log_likelihood",
            "interval_coverage_95",
            "directional_accuracy",
            "mean_pred_std",
            "mean_window_spearman",
            "long_short_top1_bottom1_mean",
        ]:
            if col in ok:
                row[col] = float(pd.to_numeric(ok[col], errors="coerce").mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if "rmse" not in out:
        out["rmse"] = float("nan")
    return out.sort_values(["scope", "rmse", "model"], na_position="last")


def save_summary(
    fake_metrics: pd.DataFrame | None,
    spy_metrics: pd.DataFrame | None,
    multitask_metrics: pd.DataFrame | None,
    output_dir: Path,
) -> pd.DataFrame:
    frames = []
    if fake_metrics is not None and not fake_metrics.empty:
        frames.append(aggregate_window_metrics(fake_metrics.rename(columns={"scenario": "forecast_date"}), "fake"))
    if spy_metrics is not None and not spy_metrics.empty:
        frames.append(aggregate_window_metrics(spy_metrics, "spy"))
    if multitask_metrics is not None and not multitask_metrics.empty:
        frames.append(aggregate_window_metrics(multitask_metrics, "multitask"))
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    return summary


def plot_metric_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    for metric in ["rmse", "predictive_log_likelihood", "mean_window_spearman", "long_short_top1_bottom1_mean"]:
        if metric not in summary:
            continue
        df = summary.dropna(subset=[metric])
        if df.empty:
            continue
        fig, axes = plt.subplots(
            len(df["scope"].unique()), 1, figsize=(11, max(4, 3 * len(df["scope"].unique()))), squeeze=False
        )
        for ax, (scope, group) in zip(axes.ravel(), df.groupby("scope"), strict=False):
            group = group.sort_values(metric)
            ax.barh(group["model"], group[metric])
            ax.set_title(f"{scope}: {metric}")
            ax.set_xlabel(metric)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"summary_{metric}.png", dpi=160)
        plt.close(fig)


def plot_predictions(preds: pd.DataFrame, scope: str, output_dir: Path) -> None:
    if preds.empty or "y_pred" not in preds:
        return
    ok = preds.dropna(subset=["y_true", "y_pred"]).copy()
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, group in ok.groupby("model"):
        ax.scatter(group["y_true"], group["y_pred"], s=24, alpha=0.65, label=model)
    lo = min(ok["y_true"].min(), ok["y_pred"].min())
    hi = max(ok["y_true"].max(), ok["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title(f"{scope}: predicted vs actual")
    ax.set_xlabel("actual excess return")
    ax.set_ylabel("predicted excess return")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / f"{scope}_predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    ok["abs_error"] = (ok["y_pred"] - ok["y_true"]).abs()
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, group in ok.groupby("model"):
        ax.scatter(group["pred_std"], group["abs_error"], s=24, alpha=0.65, label=model)
    ax.set_title(f"{scope}: predicted std vs absolute error")
    ax.set_xlabel("predicted std")
    ax.set_ylabel("absolute error")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / f"{scope}_uncertainty_vs_error.png", dpi=160)
    plt.close(fig)


def plot_modulation(curves: pd.DataFrame, scope: str, output_dir: Path) -> None:
    if curves.empty:
        return
    latest_window = sorted(curves["window"].dropna().unique())[-1] if curves["window"].notna().any() else None
    df = curves[curves["window"] == latest_window].copy() if latest_window is not None else curves.copy()
    if df.empty:
        return
    for target, group_target in df.groupby("target"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for (model, module_index), group in group_target.groupby(["model", "module_index"]):
            ax.plot(group["t"], group["modulation"], label=f"{model}:{module_index}", linewidth=1.4)
        ax.set_title(f"{scope}: learned {target} modulation, window={latest_window}")
        ax.set_xlabel("normalized time")
        ax.set_ylabel("positive modulation")
        ax.legend(fontsize=6, ncols=2)
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / f"{scope}_modulation_{target}.png", dpi=160)
        plt.close(fig)


def plot_multitask_windows(metrics: pd.DataFrame, output_dir: Path, *, universe_size: int) -> None:
    if metrics.empty or "mean_window_spearman" not in metrics:
        return
    ok = metrics[metrics["status"] == "ok"].copy()
    ok["forecast_date"] = pd.to_datetime(ok["forecast_date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for model, group in ok.groupby("model"):
        ax.plot(group["forecast_date"], group["mean_window_spearman"], marker="o", label=model)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"Multitask {universe_size}-ETF window Spearman IC")
    ax.set_xlabel("forecast date")
    ax.set_ylabel("Spearman IC")
    ax.legend(fontsize=6, ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "multitask_window_spearman.png", dpi=160)
    plt.close(fig)


def write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    summary: pd.DataFrame,
    fake_metrics: pd.DataFrame | None,
    spy_metrics: pd.DataFrame | None,
    multitask_metrics: pd.DataFrame | None,
) -> None:
    lines = [
        "# Time-Varying Parameterization Experiment",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Created: `{manifest['created_at_utc']}`",
        "",
        "## Question",
        "",
        "Can bounded nonlinear time-varying lengthscale/outputscale parameterizations improve GPArchitect forecasts versus the current linear TVOS/TVLS hook, without using changepoint kernels?",
        "",
        "## Design",
        "",
        f"- ETF universe: `{', '.join(manifest['etf_universe'])}`",
        f"- Windows: `{manifest['windows']}`",
        f"- Max optimizer iterations: `{manifest['max_iter']}`",
        f"- Multitask rank: `{manifest['rank']}`",
        f"- Train window months: `{manifest['train_window_months']}`",
        "- Changepoint kernels: excluded by design.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No completed summary rows were produced.")
    else:
        cols = [
            c
            for c in [
                "scope",
                "model",
                "ok_windows",
                "failed_windows",
                "rmse",
                "predictive_log_likelihood",
                "directional_accuracy",
                "mean_window_spearman",
                "long_short_top1_bottom1_mean",
            ]
            if c in summary.columns
        ]
        lines.append("```text")
        lines.append(summary[cols].to_string(index=False))
        lines.append("```")
    lines.extend(
        [
            "",
            "## Visuals",
            "",
            "- `plots/summary_rmse.png`",
            "- `plots/summary_predictive_log_likelihood.png`",
            "- `plots/spy_predicted_vs_actual.png`",
            "- `plots/spy_uncertainty_vs_error.png`",
            "- `plots/multitask_predicted_vs_actual.png`",
            "- `plots/multitask_uncertainty_vs_error.png`",
            "- `plots/multitask_window_spearman.png`",
            "- `plots/*_modulation_*.png`",
            "",
            "## Critic Pass",
            "",
            "This is a smoke experiment, not a promotion run. The strongest reason the result could be wrong is optimizer noise and very small window count. Treat any win as a candidate for the next larger run, not as a final modeling decision.",
            "",
            "## Lineage",
            "",
            f"- Manifest: `{output_dir / 'manifest.json'}`",
            f"- Artifact path: `{manifest['artifact_path']}`",
            f"- Git commit: `{manifest['git']['commit']}`",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines))


def write_quarto_report(output_dir: Path, manifest: dict[str, Any], summary: pd.DataFrame) -> Path:
    qmd = output_dir / "tvls_24_window_report.qmd"
    variants = ", ".join(v["name"] for v in manifest["variants"])
    best_rows = []
    if not summary.empty:
        for scope, metric, ascending in [
            ("spy", "rmse", True),
            ("spy", "predictive_log_likelihood", False),
            ("multitask", "rmse", True),
            ("multitask", "mean_window_spearman", False),
            ("multitask", "long_short_top1_bottom1_mean", False),
        ]:
            scope_df = (
                summary[(summary["scope"] == scope) & summary[metric].notna()] if metric in summary else pd.DataFrame()
            )
            if not scope_df.empty:
                row = scope_df.sort_values(metric, ascending=ascending).iloc[0]
                best_rows.append(f"- `{scope}` best by `{metric}`: `{row['model']}` ({metric}={row[metric]:.6g})")
    best_text = "\n".join(best_rows) if best_rows else "- No completed best-model rows were available."
    qmd.write_text(
        f"""---
title: "TVLS 24-Window GPArchitect Experiment"
format:
  html:
    embed-resources: true
    toc: true
    code-fold: true
jupyter: python3
---

```{{python}}
#| echo: false
from pathlib import Path
import pandas as pd
from IPython.display import Markdown, display

RUN_DIR = Path(r"{output_dir}")
summary = pd.read_csv(RUN_DIR / "summary.csv")
spy_windows = pd.read_csv(RUN_DIR / "spy_window_metrics.csv")
multitask_windows = pd.read_csv(RUN_DIR / "multitask_window_metrics.csv")
spy_preds = pd.read_csv(RUN_DIR / "spy_predictions.csv")
multitask_preds = pd.read_csv(RUN_DIR / "multitask_predictions.csv")
```

## Decision Readout

This run narrows the experiment to time-varying lengthscale only. It compares `plain`, `linear_tvls`, `piecewise_linear_tvls`, and `spline3_tvls` over 24 rolling windows for both single-task SPY and the 5-ETF Hadamard-style multitask GP.

{best_text}

Interpretation rule: treat this as an evidence-gathering run, not an automatic promotion. A TVLS variant needs to beat `plain` on the decision-relevant metric without concentrating the gain in only one or two windows.

## Experiment Design

- ETF universe: `{", ".join(ETF_UNIVERSE)}`
- Variants: `{variants}`
- Windows: `{manifest["last_n_windows"]}`
- Max optimizer iterations: `{manifest["max_iter"]}`
- Multitask rank: `{manifest["rank"]}`
- Feature artifact: `{manifest["artifact_path"]}`
- Changepoint and TVOS variants: excluded by design for this run

```{{python}}
#| echo: false
display(summary)
```

## SPY Single-Task Results

```{{python}}
#| echo: false
display(
    summary[summary["scope"].eq("spy")]
    .sort_values("rmse")
    [["model", "ok_windows", "failed_windows", "rmse", "predictive_log_likelihood", "directional_accuracy", "mean_pred_std"]]
)
```

![](plots/summary_rmse.png)

![](plots/summary_predictive_log_likelihood.png)

![](plots/spy_predicted_vs_actual.png)

![](plots/spy_uncertainty_vs_error.png)

![](plots/spy_modulation_lengthscale.png)

## 5-ETF Multitask Results

```{{python}}
#| echo: false
display(
    summary[summary["scope"].eq("multitask")]
    .sort_values("rmse")
    [["model", "ok_windows", "failed_windows", "rmse", "predictive_log_likelihood", "directional_accuracy", "mean_window_spearman", "long_short_top1_bottom1_mean", "mean_pred_std"]]
)
```

![](plots/multitask_window_spearman.png)

![](plots/summary_mean_window_spearman.png)

![](plots/summary_long_short_top1_bottom1_mean.png)

![](plots/multitask_predicted_vs_actual.png)

![](plots/multitask_uncertainty_vs_error.png)

![](plots/multitask_modulation_lengthscale.png)

## Window-Level Diagnostics

```{{python}}
#| echo: false
cols = ["model", "forecast_date", "status", "rmse", "predictive_log_likelihood", "directional_accuracy"]
display(spy_windows[[c for c in cols if c in spy_windows.columns]].sort_values(["forecast_date", "model"]).tail(96))
```

```{{python}}
#| echo: false
cols = ["model", "forecast_date", "status", "rmse", "predictive_log_likelihood", "mean_window_spearman", "long_short_top1_bottom1_mean"]
display(multitask_windows[[c for c in cols if c in multitask_windows.columns]].sort_values(["forecast_date", "model"]).tail(96))
```

## Critic Pass

The strongest reason this conclusion could be wrong is optimizer noise interacting with a flexible kernel wrapper, especially under short `max_iter` settings and rolling-window retraining. A second risk is aggregation: average RMSE or average IC can hide a model that only wins in a few recent windows. The window-level tables and IC plot should therefore carry more weight than a single headline metric.

## Lineage

- Manifest: `{output_dir / "manifest.json"}`
- Summary CSV: `{output_dir / "summary.csv"}`
- SPY metrics: `{output_dir / "spy_window_metrics.csv"}`
- Multitask metrics: `{output_dir / "multitask_window_metrics.csv"}`
- Git commit: `{manifest["git"]["commit"]}`
- Git status at run start:

```text
{manifest["git"]["status_short"]}
```
"""
    )
    return qmd


def main() -> None:
    args = parse_args()
    run_id = args.run_id or now_run_id()
    output_dir = args.output_root / run_id
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    selected = [v for v in VARIANTS if v.name in set(args.variants)]

    panel_raw = load_panel(args.artifact_path, args.etf_universe)
    panel, time_scale_params = apply_global_time_minmax(panel_raw)
    windows = infer_windows(panel, args.last_n_windows)
    manifest = {
        "schema": "gparchitect.time_varying_parameterizations.v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_path": str(args.artifact_path),
        "output_dir": str(output_dir),
        "variants": [asdict(v) for v in selected],
        "etf_universe": args.etf_universe,
        "time_scale_params": time_scale_params,
        "windows": [{"train_end_date": t, "forecast_date": f} for t, f in windows],
        "last_n_windows": args.last_n_windows,
        "max_iter": args.max_iter,
        "rank": args.rank,
        "train_window_months": args.train_window_months,
        "git": git_info(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    fake_metrics = fake_curves = None
    spy_preds = spy_metrics = spy_curves = None
    multitask_preds = multitask_metrics = multitask_curves = None

    if not args.skip_fake:
        fake_metrics, fake_curves = run_fake_data(selected, output_dir, args.max_iter)
        plot_modulation(fake_curves, "fake", output_dir)
    if not args.skip_spy:
        spy_preds, spy_metrics, spy_curves = run_spy(
            panel,
            selected,
            windows,
            output_dir,
            args.max_iter,
            train_window_months=args.train_window_months,
        )
        plot_predictions(spy_preds, "spy", output_dir)
        plot_modulation(spy_curves, "spy", output_dir)
    if not args.skip_multitask:
        multitask_preds, multitask_metrics, multitask_curves = run_multitask(
            panel,
            selected,
            windows,
            output_dir,
            args.rank,
            args.max_iter,
            train_window_months=args.train_window_months,
        )
        plot_predictions(multitask_preds, "multitask", output_dir)
        plot_multitask_windows(multitask_metrics, output_dir, universe_size=len(args.etf_universe))
        plot_modulation(multitask_curves, "multitask", output_dir)

    summary = save_summary(fake_metrics, spy_metrics, multitask_metrics, output_dir)
    plot_metric_bars(summary, output_dir)
    write_report(output_dir, manifest, summary, fake_metrics, spy_metrics, multitask_metrics)
    if args.write_qmd:
        write_quarto_report(output_dir, manifest, summary)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
