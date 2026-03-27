"""Tests for the GPArchitect DSL validation module."""

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
from gparchitect.validation.validator import ValidationResult, validate_dsl, validate_or_raise


def _make_simple_spec(input_dim: int = 3, output_dim: int = 1) -> GPSpec:
    """Helper: minimal valid SingleTaskGP spec."""
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
        output_dim=output_dim,
    )


class TestValidationResult:
    def test_is_valid_when_no_errors(self) -> None:
        result = ValidationResult()
        assert result.is_valid is True

    def test_is_invalid_when_errors_present(self) -> None:
        result = ValidationResult(errors=["something went wrong"])
        assert result.is_valid is False

    def test_str_valid(self) -> None:
        result = ValidationResult()
        assert str(result) == "Valid"

    def test_str_shows_errors(self) -> None:
        result = ValidationResult(errors=["bad kernel"])
        assert "bad kernel" in str(result)


class TestValidateDimensions:
    def test_valid_spec_passes(self) -> None:
        spec = _make_simple_spec()
        result = validate_dsl(spec)
        assert result.is_valid

    def test_zero_input_dim_fails(self) -> None:
        spec = _make_simple_spec()
        spec.input_dim = 0
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("input_dim" in e for e in result.errors)

    def test_zero_output_dim_fails(self) -> None:
        spec = _make_simple_spec()
        spec.output_dim = 0
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("output_dim" in e for e in result.errors)


class TestValidateFeatureGroups:
    def test_no_feature_groups_fails(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups = []
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("feature group" in e.lower() for e in result.errors)

    def test_out_of_range_index_fails(self) -> None:
        spec = _make_simple_spec(input_dim=2)
        spec.feature_groups[0].feature_indices = [0, 5]  # 5 >= input_dim=2
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("out of range" in e for e in result.errors)

    def test_negative_index_fails(self) -> None:
        spec = _make_simple_spec(input_dim=3)
        spec.feature_groups[0].feature_indices = [-1, 0]
        result = validate_dsl(spec)
        assert not result.is_valid

    def test_task_index_in_continuous_group_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0, 1, 2],  # 2 is task index
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=3,
            output_dim=1,
            task_feature_index=2,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("task_feature_index" in e for e in result.errors)

    def test_empty_feature_indices_fails(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].feature_indices = []
        result = validate_dsl(spec)
        assert not result.is_valid


class TestValidateModelClass:
    def test_multitask_without_task_index_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=2,
            output_dim=1,
            task_feature_index=None,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("task_feature_index" in e for e in result.errors)

    def test_multitask_requires_long_format_output_dim(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="features",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=2,
            output_dim=2,
            task_feature_index=1,
            task_values=[0, 1],
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("output_dim=1" in error for error in result.errors)

    def test_multitask_invalid_rank_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="features",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=2,
            output_dim=1,
            task_feature_index=1,
            multitask_rank=0,
        )
        result = validate_dsl(spec)
        assert not result.is_valid

    def test_single_task_with_task_index_warns(self) -> None:
        spec = _make_simple_spec(input_dim=4)
        # task_feature_index=3 is NOT in the feature_groups [0,1,2,3] yet — remove it
        spec.feature_groups[0].feature_indices = [0, 1, 2]
        spec.task_feature_index = 3
        result = validate_dsl(spec)
        assert result.is_valid  # warning, not error
        assert result.warnings

    def test_model_list_single_output_warns(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MODEL_LIST_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.RBF),
                )
            ],
            input_dim=2,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert result.is_valid
        assert result.warnings


class TestValidateNoise:
    def test_fixed_noise_without_value_fails(self) -> None:
        spec = _make_simple_spec()
        spec.noise.fixed = True
        spec.noise.noise_value = None
        result = validate_dsl(spec)
        assert not result.is_valid

    def test_negative_noise_value_fails(self) -> None:
        spec = _make_simple_spec()
        spec.noise.noise_value = -0.1
        result = validate_dsl(spec)
        assert not result.is_valid

    def test_valid_fixed_noise(self) -> None:
        spec = _make_simple_spec()
        spec.noise.fixed = True
        spec.noise.noise_value = 1e-4
        result = validate_dsl(spec)
        assert result.is_valid

    def test_fixed_noise_rejects_noise_prior(self) -> None:
        spec = _make_simple_spec()
        spec.noise.fixed = True
        spec.noise.noise_value = 1e-4
        spec.noise.prior = PriorSpec(distribution="Gamma", params={"concentration": 2.0, "rate": 0.5})
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("noise.prior" in error for error in result.errors)


class TestValidateMeans:
    def test_single_task_rejects_output_means(self) -> None:
        spec = _make_simple_spec()
        spec.output_means = {0: MeanSpec(mean_type=MeanFunctionType.ZERO)}
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("output_means" in error for error in result.errors)

    def test_model_list_rejects_out_of_range_output_mean_index(self) -> None:
        spec = _make_simple_spec(output_dim=2)
        spec.model_class = ModelClass.MODEL_LIST_GP
        spec.output_means = {2: MeanSpec(mean_type=MeanFunctionType.LINEAR)}
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("out of range" in error for error in result.errors)

    def test_multitask_rejects_negative_task_mean_index(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="features",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=2,
            output_dim=1,
            task_feature_index=1,
            output_means={-1: MeanSpec(mean_type=MeanFunctionType.CONSTANT)},
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("task index" in error for error in result.errors)

    def test_multitask_targeted_means_require_task_values(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="features",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=2,
            output_dim=1,
            task_feature_index=1,
            output_means={0: MeanSpec(mean_type=MeanFunctionType.CONSTANT)},
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("task_values" in error for error in result.errors)

    def test_multitask_targeted_means_must_match_declared_task_values(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="features",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=2,
            output_dim=1,
            task_feature_index=1,
            task_values=[0],
            output_means={1: MeanSpec(mean_type=MeanFunctionType.CONSTANT)},
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("task_values" in error for error in result.errors)


class TestValidatePriors:
    def test_prior_without_distribution_fails(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].kernel.lengthscale_prior = PriorSpec(distribution="")
        result = validate_dsl(spec)
        assert not result.is_valid

    def test_valid_prior_passes(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].kernel.lengthscale_prior = PriorSpec(
            distribution="Normal", params={"loc": 0.0, "scale": 1.0}
        )
        result = validate_dsl(spec)
        assert result.is_valid

    def test_unsupported_prior_distribution_fails(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].kernel.lengthscale_prior = PriorSpec(
            distribution="Uniform", params={"low": 0.0, "high": 1.0}
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("unsupported distribution" in error for error in result.errors)

    def test_period_prior_requires_periodic_kernel(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].kernel.period_prior = PriorSpec(
            distribution="LogNormal", params={"loc": 0.0, "scale": 1.0}
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("period_prior" in error for error in result.errors)

    def test_outputscale_prior_rejects_composed_kernel(self) -> None:
        spec = _make_simple_spec(input_dim=1)
        spec.feature_groups[0].kernel = KernelSpec(
            kernel_type=KernelType.RBF,
            composition=CompositionType.ADDITIVE,
            children=[KernelSpec(kernel_type=KernelType.RBF)],
            outputscale_prior=PriorSpec(distribution="Gamma", params={"concentration": 2.0, "rate": 0.5}),
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("outputscale_prior" in error for error in result.errors)


class TestValidateComposition:
    def test_multiple_groups_reject_composition_none(self) -> None:
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
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                ),
            ],
            input_dim=2,
            output_dim=1,
            group_composition=CompositionType.NONE,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("composition=none" in error for error in result.errors)


class TestValidateKernelSpecificParameters:
    def test_invalid_polynomial_power_fails(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].kernel = KernelSpec(kernel_type=KernelType.POLYNOMIAL, polynomial_power=0)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("Polynomial power" in error for error in result.errors)

    def test_invalid_periodic_period_length_fails(self) -> None:
        spec = _make_simple_spec(input_dim=1)
        spec.feature_groups[0].kernel = KernelSpec(kernel_type=KernelType.PERIODIC, period_length=0.0)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("period_length" in error for error in result.errors)

    def test_invalid_bnn_depth_fails(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].kernel = KernelSpec(kernel_type=KernelType.INFINITE_WIDTH_BNN, bnn_depth=0)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("InfiniteWidthBNN depth" in error for error in result.errors)

    def test_exponential_decay_requires_single_feature(self) -> None:
        spec = _make_simple_spec(input_dim=2)
        spec.feature_groups[0].kernel = KernelSpec(kernel_type=KernelType.EXPONENTIAL_DECAY)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("ExponentialDecayKernel requires exactly one active feature" in error for error in result.errors)


class TestValidateKernelSpecificParameters:
    def test_rq_alpha_must_be_positive(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].kernel = KernelSpec(kernel_type=KernelType.RQ, rq_alpha=0.0)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("RQ alpha" in error for error in result.errors)

    def test_spectral_mixture_num_mixtures_must_be_positive(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups[0].kernel = KernelSpec(kernel_type=KernelType.SPECTRAL_MIXTURE, num_mixtures=0)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("num_mixtures" in error for error in result.errors)


class TestValidateOrRaise:
    def test_valid_spec_does_not_raise(self) -> None:
        spec = _make_simple_spec()
        validate_or_raise(spec)  # should not raise

    def test_invalid_spec_raises_value_error(self) -> None:
        spec = _make_simple_spec()
        spec.feature_groups = []
        with pytest.raises(ValueError, match="Invalid GPSpec"):
            validate_or_raise(spec)
