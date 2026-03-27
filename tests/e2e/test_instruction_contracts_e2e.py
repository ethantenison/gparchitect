"""End-to-end instruction contract tests for the public API.

Purpose:
    Verify that representative natural-language instructions survive the full
    GPArchitect pipeline and produce the expected DSL snapshot plus the expected
    high-level BoTorch model structure.

Role in pipeline:
    These tests sit at the public API boundary and exercise
    natural language -> DSL -> validation -> builder -> fit.

Inputs and outputs:
    Each test provides a small deterministic pandas DataFrame and an instruction
    string, then asserts against the returned model and experiment log.

Non-obvious design decisions:
    - The cases are intentionally few and canonical to avoid combinatorial growth.
    - Assertions focus on contract-level behavior: model class, feature-group
      mapping, explicit mean choices, and explicit-vs-default parameter handling.
    - Optional parameter default behavior is verified via the DSL snapshot staying
      unset rather than by asserting version-sensitive library default values.

What this module does not do:
    - It does not exhaustively test every kernel/mean/model-class combination.
    - It does not duplicate translator or builder unit coverage for every branch.
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


def _assert_single_attempt_success(log) -> dict[str, object]:  # noqa: ANN001
    assert log.final_success is True
    assert len(log.attempts) == 1
    assert log.attempts[0].fit_success is True
    return log.attempts[0].spec_snapshot


def _assert_feature_groups(
    spec_snapshot: dict[str, object],
    expected_groups: list[tuple[list[int], str]],
) -> None:
    feature_groups = spec_snapshot["feature_groups"]
    assert isinstance(feature_groups, list)
    assert len(feature_groups) == len(expected_groups)

    for group_snapshot, (expected_indices, expected_kernel_type) in zip(feature_groups, expected_groups):
        assert group_snapshot["feature_indices"] == expected_indices
        assert group_snapshot["kernel"]["kernel_type"] == expected_kernel_type


def test_single_task_instruction_contract_preserves_groups_mean_and_rq_alpha() -> None:
    """SingleTaskGP instruction should preserve groups, mean, and explicit RQ alpha."""
    _require_runtime_dependencies()
    import gpytorch
    from botorch.models import SingleTaskGP

    dataframe = pd.DataFrame(
        {
            "seasonality": [index / 11 for index in range(12)],
            "trend": [0.1 + (0.9 * index / 11) for index in range(12)],
            "target": [
                0.45 * math.sin(2.0 * math.pi * seasonal) + 0.35 * trend + 0.05
                for seasonal, trend in zip(
                    [index / 11 for index in range(12)],
                    [0.1 + (0.9 * index / 11) for index in range(12)],
                )
            ],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction=(
            "Use a single task gp with a constant mean, an rq kernel with alpha 0.75 on seasonality, "
            "and an rbf kernel on trend."
        ),
        input_columns=["seasonality", "trend"],
        output_columns=["target"],
        max_retries=0,
    )

    assert isinstance(model, SingleTaskGP)
    assert isinstance(model.mean_module, gpytorch.means.ConstantMean)

    spec_snapshot = _assert_single_attempt_success(log)
    assert spec_snapshot["model_class"] == "SingleTaskGP"
    assert spec_snapshot["mean"]["mean_type"] == "Constant"
    _assert_feature_groups(spec_snapshot, [([0], "RQ"), ([1], "RBF")])
    assert spec_snapshot["feature_groups"][0]["kernel"]["rq_alpha"] == pytest.approx(0.75)


def test_instruction_contract_leaves_optional_rq_alpha_unset_when_omitted() -> None:
    """Omitted optional kernel parameters should remain unset in the DSL snapshot."""
    _require_runtime_dependencies()
    import gpytorch
    from botorch.models import SingleTaskGP

    x_values = [index / 11 for index in range(12)]
    dataframe = pd.DataFrame(
        {
            "x": x_values,
            "y": [math.sin(2.0 * math.pi * value) / (1.0 + value) for value in x_values],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction="Use an rq kernel on x.",
        input_columns=["x"],
        output_columns=["y"],
        max_retries=0,
    )

    assert isinstance(model, SingleTaskGP)
    assert isinstance(model.covar_module.base_kernel, gpytorch.kernels.RQKernel)

    spec_snapshot = _assert_single_attempt_success(log)
    assert spec_snapshot["model_class"] == "SingleTaskGP"
    _assert_feature_groups(spec_snapshot, [([0], "RQ")])
    assert spec_snapshot["feature_groups"][0]["kernel"]["rq_alpha"] is None


def test_model_list_instruction_contract_runs_end_to_end() -> None:
    """ModelListGP instruction should fit end to end with per-output means."""
    _require_runtime_dependencies()
    import gpytorch
    from botorch.models.model_list_gp_regression import ModelListGP

    x1_values = [index / 9 for index in range(10)]
    x2_values = [1.0 - (0.8 * index / 9) for index in range(10)]
    dataframe = pd.DataFrame(
        {
            "x1": x1_values,
            "x2": x2_values,
            "y_one": [0.4 * value + 0.1 for value in x1_values],
            "y_two": [0.25 + (0.6 * first) - (0.2 * second) for first, second in zip(x1_values, x2_values)],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction=(
            "Use a model list gp with an rbf kernel on x1 and a matern1/2 kernel on x2, "
            "where output 1 uses zero mean and output 2 uses linear mean."
        ),
        input_columns=["x1", "x2"],
        output_columns=["y_one", "y_two"],
        max_retries=0,
    )

    assert isinstance(model, ModelListGP)
    assert isinstance(model.models[0].mean_module, gpytorch.means.ZeroMean)
    assert isinstance(model.models[1].mean_module, gpytorch.means.LinearMean)

    spec_snapshot = _assert_single_attempt_success(log)
    assert spec_snapshot["model_class"] == "ModelListGP"
    assert spec_snapshot["output_means"]["0"]["mean_type"] == "Zero"
    assert spec_snapshot["output_means"]["1"]["mean_type"] == "Linear"
    _assert_feature_groups(spec_snapshot, [([0], "RBF"), ([1], "Matern12")])


def test_multitask_instruction_contract_runs_end_to_end() -> None:
    """MultiTaskGP instruction should fit end to end with per-task means."""
    _require_runtime_dependencies()
    import gpytorch
    import torch
    from botorch.models import MultiTaskGP

    torch.manual_seed(0)

    rows = [(index / 10, (index % 5) / 4, task) for task in (0, 1) for index in range(10)]
    dataframe = pd.DataFrame(
        {
            "x1": [first for first, _, _ in rows],
            "x2": [second for _, second, _ in rows],
            "task": [task for _, _, task in rows],
            "y": [(0.25 * first * first) + (0.15 * second) + (0.3 * task) + 0.05 for first, second, task in rows],
        }
    )

    model, log = run_gparchitect(
        dataframe=dataframe,
        instruction=(
            "Use a multi-task gp with an rbf kernel on x1 and a matern3/2 kernel on x2, "
            "zero mean for task 0, and constant mean for task 1."
        ),
        input_columns=["x1", "x2"],
        output_columns=["y"],
        task_column="task",
        max_retries=0,
    )

    assert isinstance(model, MultiTaskGP)
    assert hasattr(model.mean_module, "base_means")
    assert isinstance(model.mean_module.base_means[0], gpytorch.means.ZeroMean)
    assert isinstance(model.mean_module.base_means[1], gpytorch.means.ConstantMean)

    spec_snapshot = _assert_single_attempt_success(log)
    assert spec_snapshot["model_class"] == "MultiTaskGP"
    assert spec_snapshot["output_means"]["0"]["mean_type"] == "Zero"
    assert spec_snapshot["output_means"]["1"]["mean_type"] == "Constant"
    _assert_feature_groups(spec_snapshot, [([0], "RBF"), ([1], "Matern32")])
