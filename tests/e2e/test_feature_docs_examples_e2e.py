"""CI-checked worked examples that back feature docs."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from gparchitect import run_gparchitect

pytestmark = pytest.mark.e2e


def _require_runtime_dependencies() -> None:
    try:
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch, gpytorch, or botorch not installed")


def test_time_varying_outputscale_worked_example_runs() -> None:
    """Worked example from docs/time_varying_outputscale.md."""
    _require_runtime_dependencies()

    x_values = [index / 17 for index in range(18)]
    dataframe = pd.DataFrame(
        {
            "time": x_values,
            "signal": [(0.2 + 1.4 * x) * math.sin(2.0 * math.pi * x * 2.0) for x in x_values],
        }
    )

    baseline_model, baseline_log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use a Matern 5/2 kernel on time.",
        input_columns=["time"],
        output_columns=["signal"],
        max_retries=0,
    )
    adaptive_model, adaptive_log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use a Matern 5/2 kernel with time-varying outputscale on time.",
        input_columns=["time"],
        output_columns=["signal"],
        max_retries=0,
    )

    assert baseline_model is not None
    assert adaptive_model is not None
    assert baseline_log.final_success is True
    assert adaptive_log.final_success is True

    tv_snapshot = adaptive_log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]["time_varying"]
    assert tv_snapshot["target"] == "outputscale"


def test_time_varying_lengthscale_worked_example_runs() -> None:
    """Worked example from docs/time_varying_lengthscale.md."""
    _require_runtime_dependencies()

    x_values = [index / 23 for index in range(24)]
    dataframe = pd.DataFrame(
        {
            "time": x_values,
            "signal": [math.sin(2.0 * math.pi * (1.0 + (2.0 * x)) * x) for x in x_values],
        }
    )

    baseline_model, baseline_log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use a Matern 5/2 kernel on time.",
        input_columns=["time"],
        output_columns=["signal"],
        max_retries=0,
    )
    adaptive_model, adaptive_log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use a Matern 5/2 kernel with time-varying lengthscale on time.",
        input_columns=["time"],
        output_columns=["signal"],
        max_retries=0,
    )

    assert baseline_model is not None
    assert adaptive_model is not None
    assert baseline_log.final_success is True
    assert adaptive_log.final_success is True

    tv_snapshot = adaptive_log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]["time_varying"]
    assert tv_snapshot["target"] == "lengthscale"


def test_changepoint_kernel_worked_example_runs() -> None:
    """Worked example from docs/changepoint_kernel.md."""
    _require_runtime_dependencies()

    x_values = [index / 19 for index in range(20)]
    dataframe = pd.DataFrame(
        {
            "time": x_values,
            "signal": [
                (0.7 * math.sin(2.0 * math.pi * x * 1.2)) if x < 0.45 else (0.2 + 1.5 * (x - 0.45))
                for x in x_values
            ],
        }
    )

    baseline_model, baseline_log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use a Matern 5/2 kernel on time.",
        input_columns=["time"],
        output_columns=["signal"],
        max_retries=0,
    )
    changepoint_model, changepoint_log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use a changepoint kernel at 0.45 with steepness 8.0 on time.",
        input_columns=["time"],
        output_columns=["signal"],
        max_retries=0,
    )

    assert baseline_model is not None
    assert changepoint_model is not None
    assert baseline_log.final_success is True
    assert changepoint_log.final_success is True

    kernel_snapshot = changepoint_log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]
    assert kernel_snapshot["kind"] == "changepoint"
    assert kernel_snapshot["kernel_before"]["kind"] == "leaf"
    assert kernel_snapshot["kernel_after"]["kind"] == "leaf"
