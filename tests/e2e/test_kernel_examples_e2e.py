"""End-to-end user-facing kernel examples.

These tests intentionally exercise the public API with realistic natural-language
instructions and small synthetic datasets so the examples remain CI-checked.
"""

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


def _assert_single_attempt_success(log, expected_kernel: str) -> None:  # noqa: ANN001
    assert log.final_success is True
    assert len(log.attempts) == 1
    assert log.attempts[0].fit_success is True
    kernel_snapshot = log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]
    assert kernel_snapshot["kernel_type"] == expected_kernel


def test_rq_kernel_example_runs_end_to_end() -> None:
    """Example: rational quadratic kernel with an explicit alpha."""
    _require_runtime_dependencies()

    x_values = [index / 11 for index in range(12)]
    dataframe = pd.DataFrame(
        {
            "x": x_values,
            "y": [math.sin(2.5 * math.pi * value) / (1.0 + 1.5 * value) + 0.15 * value for value in x_values],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use an RQ kernel with alpha 0.75 on x.",
        input_columns=["x"],
        output_columns=["y"],
        max_retries=0,
    )

    assert model is not None
    _assert_single_attempt_success(log, "RQ")
    kernel_snapshot = log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]
    assert kernel_snapshot["rq_alpha"] == pytest.approx(0.75)


def test_spectral_mixture_example_runs_end_to_end() -> None:
    """Example: spectral mixture kernel with evenly spaced data and empirical-spectrum init."""
    _require_runtime_dependencies()

    steps = list(range(16))
    dataframe = pd.DataFrame(
        {
            "time_index": steps,
            "signal": [
                math.sin(2.0 * math.pi * step / 4.0) + 0.35 * math.cos(2.0 * math.pi * step / 7.0)
                for step in steps
            ],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction=(
            "Use a 3 component spectral mixture kernel on time_index initialized from the empirical spectrum."
        ),
        input_columns=["time_index"],
        output_columns=["signal"],
        max_retries=0,
    )

    assert model is not None
    _assert_single_attempt_success(log, "SpectralMixture")
    kernel_snapshot = log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]
    assert kernel_snapshot["num_mixtures"] == 3
    assert kernel_snapshot["spectral_init"] == "from_empirical_spectrum"


def test_polynomial_kernel_example_runs_end_to_end() -> None:
    """Example: polynomial kernel with explicit degree and offset."""
    _require_runtime_dependencies()

    x1_values = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]
    x2_values = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0]
    dataframe = pd.DataFrame(
        {
            "x1": x1_values,
            "x2": x2_values,
            "target": [
                1.0 + 0.8 * x1 + 0.4 * x2 + 0.9 * x1 * x2 + 0.6 * x1**2 for x1, x2 in zip(x1_values, x2_values)
            ],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use a polynomial kernel with degree 3 and offset 1.5.",
        input_columns=["x1", "x2"],
        output_columns=["target"],
        max_retries=0,
    )

    assert model is not None
    _assert_single_attempt_success(log, "Polynomial")
    kernel_snapshot = log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]
    assert kernel_snapshot["polynomial_power"] == 3
    assert kernel_snapshot["polynomial_offset"] == pytest.approx(1.5)


def test_infinite_width_bnn_example_runs_end_to_end() -> None:
    """Example: infinite-width BNN kernel with explicit depth."""
    _require_runtime_dependencies()

    feature_one = [0.05, 0.15, 0.2, 0.35, 0.45, 0.6, 0.75, 0.9]
    feature_two = [0.9, 0.75, 0.6, 0.55, 0.4, 0.35, 0.2, 0.1]
    dataframe = pd.DataFrame(
        {
            "feature_one": feature_one,
            "feature_two": feature_two,
            "response": [
                max(0.0, first - 0.35) + 0.8 * max(0.0, 0.65 - second)
                for first, second in zip(feature_one, feature_two)
            ],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use an infinite width bnn kernel with depth 5.",
        input_columns=["feature_one", "feature_two"],
        output_columns=["response"],
        max_retries=0,
    )

    assert model is not None
    _assert_single_attempt_success(log, "InfiniteWidthBNN")
    kernel_snapshot = log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]
    assert kernel_snapshot["bnn_depth"] == 5


def test_exponential_decay_example_runs_end_to_end() -> None:
    """Example: exponential decay kernel with explicit power and offset."""
    _require_runtime_dependencies()

    iterations = list(range(1, 13))
    dataframe = pd.DataFrame(
        {
            "iteration": iterations,
            "score": [1.4 / ((step + 0.8) ** 1.2) + 0.03 for step in iterations],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use an exponential decay kernel with power 2.5 and offset 0.2 on iteration.",
        input_columns=["iteration"],
        output_columns=["score"],
        max_retries=0,
    )

    assert model is not None
    _assert_single_attempt_success(log, "ExponentialDecay")
    kernel_snapshot = log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]
    assert kernel_snapshot["exponential_decay_power"] == pytest.approx(2.5)
    assert kernel_snapshot["exponential_decay_offset"] == pytest.approx(0.2)


def test_multi_feature_group_example_runs_end_to_end() -> None:
    """Example: multiple feature groups with distinct kernels and local parameters."""
    _require_runtime_dependencies()

    seasonality_index = [index / 15 for index in range(16)]
    system_age = [0.05, 0.12, 0.18, 0.25, 0.31, 0.37, 0.44, 0.5, 0.57, 0.63, 0.69, 0.76, 0.82, 0.88, 0.94, 1.0]
    dataframe = pd.DataFrame(
        {
            "seasonality_index": seasonality_index,
            "system_age": system_age,
            "target": [
                0.55 * math.sin(2.0 * math.pi * seasonal * 2.0) + 0.9 / ((4.0 * age) + 1.2) + 0.05
                for seasonal, age in zip(seasonality_index, system_age)
            ],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction=(
            "Use a periodic kernel with period length 0.5 on seasonality_index and an exponential decay "
            "kernel with power 2.0 and offset 0.15 on system_age."
        ),
        input_columns=["seasonality_index", "system_age"],
        output_columns=["target"],
        max_retries=0,
    )

    assert model is not None
    assert log.final_success is True
    assert len(log.attempts) == 1
    assert log.attempts[0].fit_success is True
    assert log.attempts[0].spec_snapshot["group_composition"] == "hierarchical"

    feature_groups = log.attempts[0].spec_snapshot["feature_groups"]
    assert len(feature_groups) == 2

    periodic_group = feature_groups[0]
    decay_group = feature_groups[1]

    assert periodic_group["feature_indices"] == [0]
    assert periodic_group["kernel"]["kernel_type"] == "Periodic"
    assert periodic_group["kernel"]["period_length"] == pytest.approx(0.5)

    assert decay_group["feature_indices"] == [1]
    assert decay_group["kernel"]["kernel_type"] == "ExponentialDecay"
    assert decay_group["kernel"]["exponential_decay_power"] == pytest.approx(2.0)
    assert decay_group["kernel"]["exponential_decay_offset"] == pytest.approx(0.15)


def test_three_group_mixed_kernel_example_runs_end_to_end() -> None:
    """Example: three feature groups mixing existing and newly added kernels."""
    _require_runtime_dependencies()

    trend_feature = [index / 17 for index in range(18)]
    seasonal_feature = [(index % 6) / 5 for index in range(18)]
    decay_feature = [0.08 + (0.92 * index / 17) for index in range(18)]

    dataframe = pd.DataFrame(
        {
            "trend_feature": trend_feature,
            "seasonal_feature": seasonal_feature,
            "decay_feature": decay_feature,
            "target": [
                0.4 * trend
                + 0.35 * math.sin(2.0 * math.pi * seasonal)
                + 0.75 / ((3.5 * decay) + 1.0)
                + 0.03
                for trend, seasonal, decay in zip(trend_feature, seasonal_feature, decay_feature)
            ],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction=(
            "Use an rbf kernel on trend_feature, a periodic kernel with period length 0.4 on seasonal_feature, "
            "and an exponential decay kernel with power 1.8 and offset 0.1 on decay_feature."
        ),
        input_columns=["trend_feature", "seasonal_feature", "decay_feature"],
        output_columns=["target"],
        max_retries=0,
    )

    assert model is not None
    assert log.final_success is True
    assert len(log.attempts) == 1
    assert log.attempts[0].fit_success is True
    assert log.attempts[0].spec_snapshot["group_composition"] == "hierarchical"

    feature_groups = log.attempts[0].spec_snapshot["feature_groups"]
    assert len(feature_groups) == 3

    first_group = feature_groups[0]
    second_group = feature_groups[1]
    third_group = feature_groups[2]

    assert first_group["feature_indices"] == [0]
    assert first_group["kernel"]["kernel_type"] == "RBF"

    assert second_group["feature_indices"] == [1]
    assert second_group["kernel"]["kernel_type"] == "Periodic"
    assert second_group["kernel"]["period_length"] == pytest.approx(0.4)

    assert third_group["feature_indices"] == [2]
    assert third_group["kernel"]["kernel_type"] == "ExponentialDecay"
    assert third_group["kernel"]["exponential_decay_power"] == pytest.approx(1.8)
    assert third_group["kernel"]["exponential_decay_offset"] == pytest.approx(0.1)