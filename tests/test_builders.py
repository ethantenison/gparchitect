"""
Tests for GPArchitect model builders.

Tests for the data preparation helpers run with standard dependencies (pandas, torch).
Tests for build_model_from_dsl are skipped when botorch/torch are not installed.
"""

from __future__ import annotations

import pytest

from gparchitect.dsl.schema import (
    CompositionType,
    FeatureGroupSpec,
    GPSpec,
    KernelSpec,
    KernelType,
    ModelClass,
    NoiseSpec,
)


def _make_continuous_spec(input_dim: int = 3) -> GPSpec:
    return GPSpec(
        model_class=ModelClass.SINGLE_TASK_GP,
        feature_groups=[
            FeatureGroupSpec(
                name="all",
                feature_indices=list(range(input_dim)),
                kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
            )
        ],
        noise=NoiseSpec(),
        input_dim=input_dim,
        output_dim=1,
    )


class TestPrepareInputs:
    """Tests for the _prepare_inputs helper."""

    def test_prepare_no_task_column(self) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from gparchitect.builders.builder import _prepare_inputs

        spec = _make_continuous_spec(input_dim=3)
        train_X = torch.zeros(5, 3, dtype=torch.double)
        train_Y = torch.zeros(5, 1, dtype=torch.double)

        full_X, full_Y, feature_index_map = _prepare_inputs(spec, train_X, train_Y)
        assert full_X.shape == (5, 3)
        assert full_Y.shape == (5, 1)
        assert feature_index_map == {0: 0, 1: 1, 2: 2}

    def test_prepare_with_task_column(self) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from gparchitect.builders.builder import _prepare_inputs

        spec = _make_continuous_spec(input_dim=3)
        spec.task_feature_index = 2
        spec.feature_groups[0].feature_indices = [0, 1]  # exclude task column

        train_X = torch.zeros(5, 3, dtype=torch.double)
        train_Y = torch.zeros(5, 1, dtype=torch.double)

        full_X, full_Y, feature_index_map = _prepare_inputs(spec, train_X, train_Y)
        # Should concatenate continuous (2) + task (1) = 3 columns
        assert full_X.shape[1] == 3
        assert feature_index_map == {0: 0, 1: 1}


class TestCovarianceBuilder:
    def test_hierarchical_covariance_uses_active_dims(self) -> None:
        try:
            import gpytorch
            import torch
        except ImportError:
            pytest.skip("gpytorch or torch not installed")

        from gparchitect.builders.builder import _build_covariance_module, _prepare_inputs

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="x1",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.RBF),
                ),
                FeatureGroupSpec(
                    name="x2",
                    feature_indices=[1],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_12),
                ),
            ],
            noise=NoiseSpec(),
            input_dim=2,
            output_dim=1,
            group_composition=CompositionType.HIERARCHICAL,
        )

        train_X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        _, _, feature_index_map = _prepare_inputs(spec, train_X, train_Y)
        covar_module = _build_covariance_module(spec, feature_index_map)

        assert isinstance(covar_module, gpytorch.kernels.AdditiveKernel)
        assert len(covar_module.kernels) == 3
        assert tuple(int(value) for value in covar_module.kernels[0].base_kernel.active_dims) == (0,)
        assert tuple(int(value) for value in covar_module.kernels[1].base_kernel.active_dims) == (1,)
        interaction_kernel = covar_module.kernels[2]
        assert isinstance(interaction_kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(interaction_kernel.base_kernel, gpytorch.kernels.ProductKernel)
        assert all(
            not isinstance(component, gpytorch.kernels.ScaleKernel)
            for component in interaction_kernel.base_kernel.kernels
        )

    def test_multiplicative_group_composition_wraps_only_outer_product(self) -> None:
        try:
            import gpytorch
            import torch
        except ImportError:
            pytest.skip("gpytorch or torch not installed")

        from gparchitect.builders.builder import _build_covariance_module, _prepare_inputs

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="x1",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.RBF),
                ),
                FeatureGroupSpec(
                    name="x2",
                    feature_indices=[1],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_12),
                ),
            ],
            noise=NoiseSpec(),
            input_dim=2,
            output_dim=1,
            group_composition=CompositionType.MULTIPLICATIVE,
        )

        train_X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        _, _, feature_index_map = _prepare_inputs(spec, train_X, train_Y)
        covar_module = _build_covariance_module(spec, feature_index_map)

        assert isinstance(covar_module, gpytorch.kernels.ScaleKernel)
        assert isinstance(covar_module.base_kernel, gpytorch.kernels.ProductKernel)
        assert all(
            not isinstance(component, gpytorch.kernels.ScaleKernel)
            for component in covar_module.base_kernel.kernels
        )

    def test_nested_product_kernel_wraps_only_outer_product(self) -> None:
        try:
            import gpytorch
        except ImportError:
            pytest.skip("gpytorch not installed")

        from gparchitect.builders.builder import _build_gpytorch_kernel

        kernel_spec = KernelSpec(
            kernel_type=KernelType.RBF,
            composition=CompositionType.MULTIPLICATIVE,
            children=[
                KernelSpec(kernel_type=KernelType.RBF),
                KernelSpec(kernel_type=KernelType.MATERN_12),
            ],
        )

        kernel = _build_gpytorch_kernel(kernel_spec, num_features=1)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, gpytorch.kernels.ProductKernel)
        assert all(
            not isinstance(component, gpytorch.kernels.ScaleKernel)
            for component in kernel.base_kernel.kernels
        )


class TestBuildModelMocked:
    """Tests for build_model_from_dsl — skipped when botorch/torch are not installed."""

    def _skip_if_no_torch_botorch(self) -> None:
        try:
            import botorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch/botorch not installed")

    def test_single_task_gp_created(self) -> None:
        self._skip_if_no_torch_botorch()
        import torch
        from botorch.models import SingleTaskGP

        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_continuous_spec(input_dim=3)
        train_X = torch.zeros(5, 3, dtype=torch.double)
        train_Y = torch.zeros(5, 1, dtype=torch.double)
        model = build_model_from_dsl(spec, train_X, train_Y)
        assert isinstance(model, SingleTaskGP)
        assert model.outcome_transform is not None

    def test_invalid_model_class_raises(self) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_continuous_spec(input_dim=2)
        spec.model_class = "InvalidClass"  # type: ignore[assignment]
        train_X = torch.zeros(5, 2, dtype=torch.double)
        train_Y = torch.zeros(5, 1, dtype=torch.double)

        with pytest.raises((ValueError, AttributeError, Exception)):
            build_model_from_dsl(spec, train_X, train_Y)


class TestDataPrepare:
    """Tests for prepare_data (mocked pandas/torch)."""

    def test_prepare_data_basic(self) -> None:
        try:
            import pandas as pd
            import torch
        except ImportError:
            pytest.skip("pandas or torch not installed")

        from gparchitect.builders.data import prepare_data

        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [0.1, 0.2, 0.3]})
        bundle = prepare_data(df, input_columns=["x1", "x2"], output_columns=["y"])

        assert bundle.train_X.shape == (3, 2)
        assert bundle.train_Y.shape == (3, 1)
        assert bundle.input_dim == 2
        assert bundle.output_dim == 1
        assert bundle.task_feature_index is None
        assert bundle.input_scaling_applied is True
        assert bundle.input_feature_ranges == {"x1": (1.0, 3.0), "x2": (4.0, 6.0)}
        assert bundle.train_X[:, 0].tolist() == [0.0, 0.5, 1.0]
        assert bundle.train_X[:, 1].tolist() == [0.0, 0.5, 1.0]

    def test_prepare_data_with_task_column(self) -> None:
        try:
            import pandas as pd
            import torch
        except ImportError:
            pytest.skip("pandas or torch not installed")

        from gparchitect.builders.data import prepare_data

        df = pd.DataFrame(
            {"x1": [1.0, 2.0], "x2": [3.0, 4.0], "task": [0, 1], "y": [0.1, 0.2]}
        )
        bundle = prepare_data(df, input_columns=["x1", "x2"], output_columns=["y"], task_column="task")

        assert bundle.task_feature_index == 2
        assert bundle.train_X.shape[1] == 3  # x1, x2, task
        assert bundle.train_X[:, 2].tolist() == [0.0, 1.0]

    def test_prepare_data_constant_column_scales_to_zero(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        from gparchitect.builders.data import prepare_data

        df = pd.DataFrame({"x1": [2.0, 2.0, 2.0], "y": [0.1, 0.2, 0.3]})
        bundle = prepare_data(df, input_columns=["x1"], output_columns=["y"])

        assert bundle.train_X[:, 0].tolist() == [0.0, 0.0, 0.0]

    def test_prepare_data_missing_column_raises(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        from gparchitect.builders.data import prepare_data

        df = pd.DataFrame({"x1": [1.0], "y": [0.1]})
        with pytest.raises(ValueError, match="not found"):
            prepare_data(df, input_columns=["x1", "x_missing"], output_columns=["y"])

    def test_prepare_data_nan_raises(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        import math

        from gparchitect.builders.data import prepare_data

        df = pd.DataFrame({"x1": [1.0, math.nan], "y": [0.1, 0.2]})
        with pytest.raises(ValueError, match="NaN"):
            prepare_data(df, input_columns=["x1"], output_columns=["y"])
