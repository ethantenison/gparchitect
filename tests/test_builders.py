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
    MeanFunctionType,
    MeanSpec,
    ModelClass,
    NoiseSpec,
    PriorSpec,
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
    def test_rq_kernel_builds_with_configured_alpha(self) -> None:
        try:
            import gpytorch
        except ImportError:
            pytest.skip("gpytorch not installed")

        from gparchitect.builders.builder import _build_gpytorch_kernel

        kernel_spec = KernelSpec(kernel_type=KernelType.RQ, ard=True, rq_alpha=0.75)
        kernel = _build_gpytorch_kernel(kernel_spec, num_features=2)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, gpytorch.kernels.RQKernel)
        assert kernel.base_kernel.alpha.item() == pytest.approx(0.75, rel=1e-5)

    def test_spectral_mixture_kernel_uses_requested_num_mixtures_without_scale(self) -> None:
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
                    name="all",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.SPECTRAL_MIXTURE, num_mixtures=3),
                )
            ],
            noise=NoiseSpec(),
            input_dim=2,
            output_dim=1,
        )

        train_X = torch.tensor([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]], dtype=torch.double)
        train_Y = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
        full_X, full_Y, feature_index_map = _prepare_inputs(spec, train_X, train_Y)
        covar_module = _build_covariance_module(spec, feature_index_map, full_X, full_Y)

        assert isinstance(covar_module, gpytorch.kernels.SpectralMixtureKernel)
        assert covar_module.num_mixtures == 3
        assert tuple(int(value) for value in covar_module.active_dims) == (0, 1)

    def test_spectral_mixture_empirical_spectrum_initializer_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            import gpytorch
            import torch
        except ImportError:
            pytest.skip("gpytorch or torch not installed")

        from gparchitect.builders.builder import _build_covariance_module, _prepare_inputs

        called = {"empspect": False}
        def _record_initialize(self, train_x, train_y):  # noqa: ANN001, ANN202
            called["empspect"] = True
            raise ImportError("optional empirical-spectrum dependencies unavailable in test")

        monkeypatch.setattr(
            gpytorch.kernels.SpectralMixtureKernel,
            "initialize_from_data_empspect",
            _record_initialize,
        )

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.SPECTRAL_MIXTURE,
                        num_mixtures=2,
                        spectral_init="from_empirical_spectrum",
                    ),
                )
            ],
            noise=NoiseSpec(),
            input_dim=1,
            output_dim=1,
        )

        train_X = torch.tensor([[0.0], [0.5], [1.0], [1.5]], dtype=torch.double)
        train_Y = torch.tensor([[0.0], [1.0], [0.0], [-1.0]], dtype=torch.double)
        full_X, full_Y, feature_index_map = _prepare_inputs(spec, train_X, train_Y)

        _build_covariance_module(spec, feature_index_map, full_X, full_Y)

        assert called["empspect"] is True

    def test_polynomial_kernel_builds_with_configured_power_and_offset(self) -> None:
        try:
            import gpytorch
        except ImportError:
            pytest.skip("gpytorch not installed")

        from gparchitect.builders.builder import _build_gpytorch_kernel

        kernel_spec = KernelSpec(
            kernel_type=KernelType.POLYNOMIAL,
            polynomial_power=3,
            polynomial_offset=1.5,
        )
        kernel = _build_gpytorch_kernel(kernel_spec, num_features=2)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, gpytorch.kernels.PolynomialKernel)
        assert kernel.base_kernel.power == 3
        assert kernel.base_kernel.offset.item() == pytest.approx(1.5, rel=1e-5)

    def test_infinite_width_bnn_kernel_builds_with_depth(self) -> None:
        try:
            import gpytorch
            from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel
        except ImportError:
            pytest.skip("gpytorch or botorch not installed")

        from gparchitect.builders.builder import _build_gpytorch_kernel

        kernel_spec = KernelSpec(kernel_type=KernelType.INFINITE_WIDTH_BNN, bnn_depth=5)
        kernel = _build_gpytorch_kernel(kernel_spec, num_features=3)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, InfiniteWidthBNNKernel)
        assert kernel.base_kernel.depth == 5

    def test_exponential_decay_kernel_builds_with_configured_parameters(self) -> None:
        try:
            import gpytorch
            from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
        except ImportError:
            pytest.skip("gpytorch or botorch not installed")

        from gparchitect.builders.builder import _build_gpytorch_kernel

        kernel_spec = KernelSpec(
            kernel_type=KernelType.EXPONENTIAL_DECAY,
            exponential_decay_power=2.5,
            exponential_decay_offset=0.3,
        )
        kernel = _build_gpytorch_kernel(kernel_spec, num_features=1)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, ExponentialDecayKernel)
        assert kernel.base_kernel.power.item() == pytest.approx(2.5, rel=1e-5)
        assert kernel.base_kernel.offset.item() == pytest.approx(0.3, rel=1e-5)

    def test_additive_group_composition_keeps_per_term_scales(self) -> None:
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
            group_composition=CompositionType.ADDITIVE,
        )

        train_X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        _, _, feature_index_map = _prepare_inputs(spec, train_X, train_Y)
        covar_module = _build_covariance_module(spec, feature_index_map)

        assert isinstance(covar_module, gpytorch.kernels.AdditiveKernel)
        assert all(isinstance(component, gpytorch.kernels.ScaleKernel) for component in covar_module.kernels)

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

    def test_nested_additive_kernel_scales_each_term_without_outer_scale(self) -> None:
        try:
            import gpytorch
        except ImportError:
            pytest.skip("gpytorch not installed")

        from gparchitect.builders.builder import _build_gpytorch_kernel

        kernel_spec = KernelSpec(
            kernel_type=KernelType.RBF,
            composition=CompositionType.ADDITIVE,
            children=[
                KernelSpec(kernel_type=KernelType.RBF),
                KernelSpec(kernel_type=KernelType.MATERN_12),
            ],
        )

        kernel = _build_gpytorch_kernel(kernel_spec, num_features=1)

        assert isinstance(kernel, gpytorch.kernels.AdditiveKernel)
        assert all(isinstance(component, gpytorch.kernels.ScaleKernel) for component in kernel.kernels)

    def test_product_of_additive_factors_has_single_outer_scale(self) -> None:
        try:
            import gpytorch
        except ImportError:
            pytest.skip("gpytorch not installed")

        from gparchitect.builders.builder import _build_gpytorch_kernel

        additive_factor = KernelSpec(
            kernel_type=KernelType.RBF,
            composition=CompositionType.ADDITIVE,
            children=[
                KernelSpec(kernel_type=KernelType.RBF),
                KernelSpec(kernel_type=KernelType.LINEAR),
            ],
        )
        kernel_spec = KernelSpec(
            kernel_type=KernelType.RBF,
            composition=CompositionType.MULTIPLICATIVE,
            children=[
                additive_factor,
                KernelSpec(kernel_type=KernelType.MATERN_12),
            ],
        )

        kernel = _build_gpytorch_kernel(kernel_spec, num_features=1)

        assert isinstance(kernel, gpytorch.kernels.ScaleKernel)
        assert isinstance(kernel.base_kernel, gpytorch.kernels.ProductKernel)
        assert isinstance(kernel.base_kernel.kernels[0], gpytorch.kernels.AdditiveKernel)
        assert all(
            isinstance(component, gpytorch.kernels.ScaleKernel)
            for component in kernel.base_kernel.kernels[0].kernels
        )
        assert not isinstance(kernel.base_kernel.kernels[1], gpytorch.kernels.ScaleKernel)


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

    def test_single_task_gp_uses_requested_mean_and_fixed_noise_likelihood(self) -> None:
        self._skip_if_no_torch_botorch()
        import gpytorch
        import torch

        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_continuous_spec(input_dim=2)
        spec.mean = MeanSpec(mean_type=MeanFunctionType.ZERO)
        spec.noise = NoiseSpec(fixed=True, noise_value=1e-4)

        train_X = torch.zeros(5, 2, dtype=torch.double)
        train_Y = torch.zeros(5, 1, dtype=torch.double)
        model = build_model_from_dsl(spec, train_X, train_Y)

        assert isinstance(model.mean_module, gpytorch.means.ZeroMean)
        assert isinstance(model.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood)

    def test_single_task_gp_applies_supported_kernel_and_noise_priors(self) -> None:
        self._skip_if_no_torch_botorch()
        import gpytorch
        import torch

        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_continuous_spec(input_dim=2)
        spec.feature_groups[0].kernel = KernelSpec(
            kernel_type=KernelType.PERIODIC,
            ard=True,
            lengthscale_prior=PriorSpec(distribution="Normal", params={"loc": 0.0, "scale": 1.0}),
            outputscale_prior=PriorSpec(distribution="Gamma", params={"concentration": 2.0, "rate": 0.5}),
            period_prior=PriorSpec(distribution="LogNormal", params={"loc": 0.0, "scale": 1.0}),
        )
        spec.noise = NoiseSpec(
            fixed=False,
            prior=PriorSpec(distribution="Gamma", params={"concentration": 3.0, "rate": 1.5}),
        )

        train_X = torch.zeros(5, 2, dtype=torch.double)
        train_Y = torch.zeros(5, 1, dtype=torch.double)
        model = build_model_from_dsl(spec, train_X, train_Y)

        assert isinstance(model.covar_module, gpytorch.kernels.ScaleKernel)
        assert isinstance(model.covar_module.base_kernel, gpytorch.kernels.PeriodicKernel)
        assert model.covar_module.base_kernel.lengthscale_prior is not None
        assert model.covar_module.base_kernel.period_length_prior is not None
        assert model.covar_module.outputscale_prior is not None
        assert model.likelihood.noise_covar.noise_prior is not None

    def test_single_task_gp_applies_half_cauchy_and_uniform_priors(self) -> None:
        self._skip_if_no_torch_botorch()
        import torch
        from gpytorch.priors.torch_priors import HalfCauchyPrior, UniformPrior

        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_continuous_spec(input_dim=2)
        spec.feature_groups[0].kernel = KernelSpec(
            kernel_type=KernelType.PERIODIC,
            ard=True,
            lengthscale_prior=PriorSpec(distribution="HalfCauchy", params={"scale": 0.75}),
            period_prior=PriorSpec(distribution="Uniform", params={"a": 0.1, "b": 2.0}),
        )
        spec.noise = NoiseSpec(
            fixed=False,
            prior=PriorSpec(distribution="HalfCauchy", params={"scale": 0.2}),
        )

        train_X = torch.zeros(5, 2, dtype=torch.double)
        train_Y = torch.zeros(5, 1, dtype=torch.double)
        model = build_model_from_dsl(spec, train_X, train_Y)

        assert isinstance(model.covar_module.base_kernel.lengthscale_prior, HalfCauchyPrior)
        assert isinstance(model.covar_module.base_kernel.period_length_prior, UniformPrior)
        assert isinstance(model.likelihood.noise_covar.noise_prior, HalfCauchyPrior)

    def test_model_list_gp_supports_per_output_means(self) -> None:
        self._skip_if_no_torch_botorch()
        import gpytorch
        import torch

        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_continuous_spec(input_dim=2)
        spec.model_class = ModelClass.MODEL_LIST_GP
        spec.output_dim = 2
        spec.output_means = {
            0: MeanSpec(mean_type=MeanFunctionType.ZERO),
            1: MeanSpec(mean_type=MeanFunctionType.LINEAR),
        }

        train_X = torch.zeros(5, 2, dtype=torch.double)
        train_Y = torch.zeros(5, 2, dtype=torch.double)
        model = build_model_from_dsl(spec, train_X, train_Y)

        assert isinstance(model.models[0].mean_module, gpytorch.means.ZeroMean)
        assert isinstance(model.models[1].mean_module, gpytorch.means.LinearMean)

    def test_multitask_gp_receives_custom_covar_mean_and_likelihood(self) -> None:
        self._skip_if_no_torch_botorch()
        import gpytorch
        import torch

        from gparchitect.builders.builder import build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="continuous",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.RBF),
                )
            ],
            noise=NoiseSpec(fixed=True, noise_value=1e-4),
            input_dim=3,
            output_dim=1,
            task_feature_index=2,
            task_values=[0, 1],
            multitask_rank=1,
            output_means={0: MeanSpec(mean_type=MeanFunctionType.ZERO), 1: MeanSpec(mean_type=MeanFunctionType.CONSTANT)},
        )

        train_X = torch.tensor(
            [[0.0, 0.1, 0.0], [1.0, 0.2, 0.0], [0.0, 0.1, 1.0], [1.0, 0.2, 1.0]],
            dtype=torch.double,
        )
        train_Y = torch.tensor([[0.0], [1.0], [0.5], [1.5]], dtype=torch.double)
        model = build_model_from_dsl(spec, train_X, train_Y)

        assert hasattr(model.mean_module, "base_means")
        assert isinstance(model.mean_module.base_means[0], gpytorch.means.ZeroMean)
        assert isinstance(model.mean_module.base_means[1], gpytorch.means.ConstantMean)
        assert isinstance(model.likelihood, gpytorch.likelihoods.FixedNoiseGaussianLikelihood)
        assert isinstance(model.covar_module, gpytorch.kernels.ProductKernel)
        assert isinstance(model.covar_module.kernels[0], gpytorch.kernels.ScaleKernel)
        assert isinstance(model.covar_module.kernels[0].base_kernel, gpytorch.kernels.RBFKernel)
        assert type(model.covar_module.kernels[1]).__name__ == "PositiveIndexKernel"

    def test_multitask_gp_uses_documented_default_likelihood_when_available(self) -> None:
        self._skip_if_no_torch_botorch()
        import torch
        from botorch.models import MultiTaskGP

        from gparchitect.builders.builder import _prepare_inputs, build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="continuous",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.RBF),
                )
            ],
            noise=NoiseSpec(),
            input_dim=3,
            output_dim=1,
            task_feature_index=2,
            task_values=[0, 1],
            multitask_rank=1,
        )

        train_X = torch.tensor(
            [[0.0, 0.1, 0.0], [1.0, 0.2, 0.0], [0.0, 0.1, 1.0], [1.0, 0.2, 1.0]],
            dtype=torch.double,
        )
        train_Y = torch.tensor([[0.0], [1.0], [0.5], [1.5]], dtype=torch.double)

        model = build_model_from_dsl(spec, train_X, train_Y)
        full_X, full_Y, _ = _prepare_inputs(spec, train_X, train_Y)
        reference_model = MultiTaskGP(train_X=full_X, train_Y=full_Y, task_feature=-1)

        assert type(model.likelihood) is type(reference_model.likelihood)

    def test_multitask_gp_rejects_observed_task_values_outside_declared_domain(self) -> None:
        self._skip_if_no_torch_botorch()
        import torch

        from gparchitect.builders.builder import build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="continuous",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.RBF),
                )
            ],
            noise=NoiseSpec(),
            input_dim=3,
            output_dim=1,
            task_feature_index=2,
            task_values=[0],
            multitask_rank=1,
        )

        train_X = torch.tensor(
            [[0.0, 0.1, 0.0], [1.0, 0.2, 1.0]],
            dtype=torch.double,
        )
        train_Y = torch.tensor([[0.0], [1.0]], dtype=torch.double)

        with pytest.raises(ValueError, match="task_values"):
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
