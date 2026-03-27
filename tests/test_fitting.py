"""
Tests for the GPArchitect fitting module.

Tests for FitResult structure work without torch.
Tests for fit_and_validate are skipped when torch/botorch/gpytorch are not installed.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from gparchitect.fitting.fitter import FitResult


class TestFitResult:
    def test_success_fields(self) -> None:
        model = MagicMock()
        result = FitResult(success=True, model=model, mll_value=-1.5)
        assert result.success is True
        assert result.mll_value == pytest.approx(-1.5)
        assert result.error_message == ""

    def test_failure_fields(self) -> None:
        result = FitResult(
            success=False,
            model=None,
            mll_value=None,
            error_message="RuntimeError: Cholesky failed",
        )
        assert result.success is False
        assert result.mll_value is None
        assert "Cholesky" in result.error_message

    def test_dataclass_fields_present(self) -> None:
        field_names = {f.name for f in fields(FitResult)}
        assert "success" in field_names
        assert "model" in field_names
        assert "mll_value" in field_names
        assert "error_message" in field_names
        assert "train_X" in field_names
        assert "train_Y" in field_names


class TestFitAndValidateMocked:
    """Tests for fit_and_validate — skipped when torch/botorch are not installed."""

    def _skip_if_no_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed")

    def _skip_if_no_botorch(self) -> None:
        try:
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch/botorch/gpytorch not installed")

    def test_successful_fit_returns_success_true(self) -> None:
        self._skip_if_no_botorch()
        import torch

        from gparchitect.builders.builder import build_model_from_dsl
        from gparchitect.builders.data import prepare_data
        from gparchitect.dsl.schema import FeatureGroupSpec, GPSpec, KernelSpec, KernelType, ModelClass
        from gparchitect.fitting.fitter import fit_and_validate

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=2,
            output_dim=1,
        )
        train_X = torch.randn(10, 2, dtype=torch.double)
        train_Y = torch.randn(10, 1, dtype=torch.double)
        model = build_model_from_dsl(spec, train_X, train_Y)
        result = fit_and_validate(model, train_X, train_Y)
        assert result.success is True

    def test_fit_failure_returns_success_false(self) -> None:
        self._skip_if_no_botorch()
        import torch

        from gparchitect.fitting.fitter import fit_and_validate

        # Pass a non-model object to trigger a failure
        broken_model = object()
        train_X = torch.zeros(5, 2, dtype=torch.double)
        train_Y = torch.zeros(5, 1, dtype=torch.double)

        result = fit_and_validate(broken_model, train_X, train_Y)
        assert result.success is False
        assert result.error_message != ""

    def test_fit_result_stores_train_tensors(self) -> None:
        self._skip_if_no_torch()
        import torch

        train_X = torch.zeros(5, 2)
        train_Y = torch.zeros(5, 1)
        result = FitResult(success=True, model=MagicMock(), mll_value=None, train_X=train_X, train_Y=train_Y)
        assert result.train_X is train_X
        assert result.train_Y is train_Y

    def test_model_list_gp_fit_succeeds(self) -> None:
        self._skip_if_no_botorch()
        import torch

        from gparchitect.builders.builder import build_model_from_dsl
        from gparchitect.dsl.schema import FeatureGroupSpec, GPSpec, KernelSpec, KernelType, ModelClass
        from gparchitect.fitting.fitter import fit_and_validate

        spec = GPSpec(
            model_class=ModelClass.MODEL_LIST_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=2,
            output_dim=2,
        )
        train_X = torch.tensor(
            [[0.0, 0.0], [0.2, 0.1], [0.4, 0.3], [0.6, 0.5], [0.8, 0.7], [1.0, 0.9]],
            dtype=torch.double,
        )
        train_Y = torch.tensor(
            [[0.1, 0.4], [0.2, 0.45], [0.35, 0.55], [0.5, 0.65], [0.7, 0.8], [0.9, 0.95]],
            dtype=torch.double,
        )

        model = build_model_from_dsl(spec, train_X, train_Y)
        result = fit_and_validate(model, train_X, train_Y)

        assert result.success is True

    def test_multitask_gp_with_per_task_means_fit_succeeds(self) -> None:
        self._skip_if_no_botorch()
        import torch

        from gparchitect.builders.builder import build_model_from_dsl
        from gparchitect.dsl.schema import (
            FeatureGroupSpec,
            GPSpec,
            KernelSpec,
            KernelType,
            MeanFunctionType,
            MeanSpec,
            ModelClass,
        )
        from gparchitect.fitting.fitter import fit_and_validate

        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=3,
            output_dim=1,
            task_feature_index=2,
            multitask_rank=1,
            output_means={
                0: MeanSpec(mean_type=MeanFunctionType.ZERO),
                1: MeanSpec(mean_type=MeanFunctionType.CONSTANT),
            },
        )
        train_X = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.1, 0.0],
                [0.4, 0.3, 0.0],
                [0.6, 0.5, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.1, 1.0],
                [0.4, 0.3, 1.0],
                [0.6, 0.5, 1.0],
            ],
            dtype=torch.double,
        )
        train_Y = torch.tensor(
            [[0.05], [0.15], [0.3], [0.45], [0.35], [0.45], [0.6], [0.75]],
            dtype=torch.double,
        )

        model = build_model_from_dsl(spec, train_X, train_Y)
        result = fit_and_validate(model, train_X, train_Y)

        assert result.success is True
