"""Tests for the GPArchitect DSL schema module."""

from __future__ import annotations

import json

import pytest

from gparchitect.dsl.compat import normalize_kernel_spec
from gparchitect.dsl.schema import (
    ChangepointKernelSpec,
    CompositeKernelSpec,
    CompositionType,
    ExecutionSpec,
    FeatureGroupSpec,
    GPSpec,
    KernelSpec,
    KernelType,
    LeafKernelSpec,
    MeanFunctionType,
    MeanSpec,
    ModelClass,
    NoiseSpec,
    PriorDistribution,
    PriorSpec,
    SpectralMixtureInitialization,
)


class TestPriorSpec:
    def test_distribution_is_typed_enum(self) -> None:
        prior = PriorSpec(distribution=PriorDistribution.NORMAL)
        assert prior.distribution == PriorDistribution.NORMAL

    def test_additional_supported_distributions_are_typed(self) -> None:
        half_cauchy = PriorSpec(distribution=PriorDistribution.HALF_CAUCHY, params={"scale": 1.0})
        uniform = PriorSpec(distribution=PriorDistribution.UNIFORM, params={"a": 0.0, "b": 1.0})
        assert half_cauchy.distribution == PriorDistribution.HALF_CAUCHY
        assert uniform.distribution == PriorDistribution.UNIFORM

    def test_default_params_is_empty_dict(self) -> None:
        prior = PriorSpec(distribution="Normal")
        assert prior.params == {}

    def test_params_stored_correctly(self) -> None:
        prior = PriorSpec(distribution="Gamma", params={"concentration": 2.0, "rate": 0.5})
        assert prior.params["concentration"] == 2.0

    def test_json_serializable(self) -> None:
        prior = PriorSpec(distribution="LogNormal", params={"loc": 0.0, "scale": 1.0})
        data = json.loads(prior.model_dump_json())
        assert data["distribution"] == "LogNormal"


class TestNoiseSpec:
    def test_defaults(self) -> None:
        noise = NoiseSpec()
        assert noise.fixed is False
        assert noise.noise_value is None
        assert noise.prior is None

    def test_fixed_noise(self) -> None:
        noise = NoiseSpec(fixed=True, noise_value=1e-4)
        assert noise.fixed is True
        assert noise.noise_value == pytest.approx(1e-4)

    def test_json_round_trip(self) -> None:
        noise = NoiseSpec(fixed=True, noise_value=0.01)
        data = json.loads(noise.model_dump_json())
        assert data["fixed"] is True
        assert data["noise_value"] == pytest.approx(0.01)


class TestMeanSpec:
    def test_mean_spec_serializes(self) -> None:
        mean = MeanSpec(mean_type=MeanFunctionType.LINEAR)
        data = json.loads(mean.model_dump_json())
        assert data["mean_type"] == "Linear"


class TestKernelSpec:
    def test_default_kernel(self) -> None:
        kernel = KernelSpec(kernel_type=KernelType.MATERN_52)
        assert kernel.ard is False
        assert kernel.children == []
        assert kernel.composition == CompositionType.NONE

    def test_ard_flag(self) -> None:
        kernel = KernelSpec(kernel_type=KernelType.RBF, ard=True)
        assert kernel.ard is True

    def test_nested_children(self) -> None:
        child = KernelSpec(kernel_type=KernelType.PERIODIC)
        parent = KernelSpec(
            kernel_type=KernelType.RBF,
            children=[child],
            composition=CompositionType.ADDITIVE,
        )
        assert len(parent.children) == 1
        assert parent.children[0].kernel_type == KernelType.PERIODIC

    def test_json_serializable(self) -> None:
        kernel = KernelSpec(
            kernel_type=KernelType.MATERN_32,
            ard=True,
            lengthscale_prior=PriorSpec(distribution="Normal", params={"loc": 0.0, "scale": 1.0}),
        )
        data = json.loads(kernel.model_dump_json())
        assert data["kernel_type"] == "Matern32"
        assert data["ard"] is True
        assert data["lengthscale_prior"]["distribution"] == "Normal"

    def test_rq_alpha_serializes(self) -> None:
        kernel = KernelSpec(kernel_type=KernelType.RQ, rq_alpha=0.75)
        data = json.loads(kernel.model_dump_json())
        assert data["kernel_type"] == "RQ"
        assert data["rq_alpha"] == pytest.approx(0.75)

    def test_spectral_mixture_fields_serialize(self) -> None:
        kernel = KernelSpec(
            kernel_type=KernelType.SPECTRAL_MIXTURE,
            num_mixtures=4,
            spectral_init=SpectralMixtureInitialization.FROM_EMPIRICAL_SPECTRUM,
        )
        data = json.loads(kernel.model_dump_json())
        assert data["kernel_type"] == "SpectralMixture"
        assert data["num_mixtures"] == 4
        assert data["spectral_init"] == "from_empirical_spectrum"

    def test_polynomial_fields_serialize(self) -> None:
        kernel = KernelSpec(
            kernel_type=KernelType.POLYNOMIAL,
            polynomial_power=3,
            polynomial_offset=1.25,
        )
        data = json.loads(kernel.model_dump_json())
        assert data["kernel_type"] == "Polynomial"
        assert data["polynomial_power"] == 3
        assert data["polynomial_offset"] == pytest.approx(1.25)

    def test_botorch_kernel_fields_serialize(self) -> None:
        kernel = KernelSpec(kernel_type=KernelType.INFINITE_WIDTH_BNN, bnn_depth=5)
        data = json.loads(kernel.model_dump_json())
        assert data["kernel_type"] == "InfiniteWidthBNN"
        assert data["bnn_depth"] == 5


class TestFeatureGroupSpec:
    def test_basic_group(self) -> None:
        group = FeatureGroupSpec(
            name="spatial",
            feature_indices=[0, 1],
            kernel=KernelSpec(kernel_type=KernelType.RBF),
        )
        assert group.name == "spatial"
        assert group.feature_indices == [0, 1]

    def test_empty_indices_allowed(self) -> None:
        group = FeatureGroupSpec(
            name="empty",
            feature_indices=[],
            kernel=KernelSpec(kernel_type=KernelType.LINEAR),
        )
        assert group.feature_indices == []


class TestGPSpec:
    def test_defaults(self) -> None:
        spec = GPSpec()
        assert spec.model_class == ModelClass.SINGLE_TASK_GP
        assert spec.input_dim == 1
        assert spec.output_dim == 1
        assert spec.feature_groups == []
        assert spec.mean is None
        assert spec.output_means == {}
        assert spec.task_feature_index is None
        assert spec.task_values is None
        assert spec.multitask_rank is None
        assert spec.execution == ExecutionSpec()

    def test_full_spec(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            noise=NoiseSpec(fixed=False),
            input_dim=3,
            output_dim=1,
            task_feature_index=2,
            task_values=[0, 1],
            multitask_rank=1,
        )
        assert spec.model_class == ModelClass.MULTI_TASK_GP
        assert spec.task_feature_index == 2
        assert spec.task_values == [0, 1]
        assert len(spec.feature_groups) == 1

    def test_json_round_trip(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="inputs",
                    feature_indices=[0, 1, 2],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52, ard=True),
                )
            ],
            mean=MeanSpec(mean_type=MeanFunctionType.ZERO),
            input_dim=3,
            output_dim=1,
        )
        data = json.loads(spec.model_dump_json())
        assert data["model_class"] == "SingleTaskGP"
        assert data["input_dim"] == 3
        assert data["mean"]["mean_type"] == "Zero"
        assert data["feature_groups"][0]["kernel"]["ard"] is True

    def test_output_means_serialize(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MODEL_LIST_GP,
            output_dim=2,
            output_means={
                0: MeanSpec(mean_type=MeanFunctionType.CONSTANT),
                1: MeanSpec(mean_type=MeanFunctionType.LINEAR),
            },
        )
        data = json.loads(spec.model_dump_json())
        assert data["output_means"]["0"]["mean_type"] == "Constant"
        assert data["output_means"]["1"]["mean_type"] == "Linear"

    def test_task_values_serialize(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.MULTI_TASK_GP,
            output_dim=1,
            task_feature_index=2,
            task_values=[0, 1],
        )
        data = json.loads(spec.model_dump_json())
        assert data["task_values"] == [0, 1]

    def test_execution_spec_serializes(self) -> None:
        spec = GPSpec(execution=ExecutionSpec(input_scaling=False, outcome_standardization=False))
        data = json.loads(spec.model_dump_json())
        assert data["execution"]["input_scaling"] is False
        assert data["execution"]["outcome_standardization"] is False

    def test_description_field(self) -> None:
        spec = GPSpec(description="Test spec")
        assert spec.description == "Test spec"

    def test_enum_values_are_strings(self) -> None:
        assert ModelClass.SINGLE_TASK_GP.value == "SingleTaskGP"
        assert KernelType.MATERN_52.value == "Matern52"
        assert CompositionType.ADDITIVE.value == "additive"
        assert CompositionType.HIERARCHICAL.value == "hierarchical"


class TestLeafKernelSpec:
    def test_default_kind_is_leaf(self) -> None:
        kernel = LeafKernelSpec(kernel_type=KernelType.MATERN_52)
        assert kernel.kind == "leaf"

    def test_json_round_trip(self) -> None:
        kernel = LeafKernelSpec(
            kernel_type=KernelType.RBF,
            ard=True,
            rq_alpha=None,
        )
        data = json.loads(kernel.model_dump_json())
        assert data["kind"] == "leaf"
        assert data["kernel_type"] == "RBF"
        assert data["ard"] is True


class TestCompositeKernelSpec:
    def test_default_kind_is_composite(self) -> None:
        kernel = CompositeKernelSpec(
            composition=CompositionType.ADDITIVE,
            children=[
                LeafKernelSpec(kernel_type=KernelType.RBF),
                LeafKernelSpec(kernel_type=KernelType.MATERN_52),
            ],
        )
        assert kernel.kind == "composite"

    def test_json_round_trip(self) -> None:
        kernel = CompositeKernelSpec(
            composition=CompositionType.ADDITIVE,
            children=[
                LeafKernelSpec(kernel_type=KernelType.RBF),
                LeafKernelSpec(kernel_type=KernelType.MATERN_52),
            ],
        )
        data = json.loads(kernel.model_dump_json())
        assert data["kind"] == "composite"
        assert data["composition"] == "additive"
        assert len(data["children"]) == 2
        assert data["children"][0]["kind"] == "leaf"

    def test_invalid_composition_raises(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CompositeKernelSpec(
                composition=CompositionType.NONE,
                children=[
                    LeafKernelSpec(kernel_type=KernelType.RBF),
                    LeafKernelSpec(kernel_type=KernelType.MATERN_52),
                ],
            )

    def test_fewer_than_two_children_raises(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CompositeKernelSpec(
                composition=CompositionType.ADDITIVE,
                children=[LeafKernelSpec(kernel_type=KernelType.RBF)],
            )


class TestChangepointKernelSpec:
    def test_default_kind_is_changepoint(self) -> None:
        kernel = ChangepointKernelSpec(
            kernel_before=LeafKernelSpec(kernel_type=KernelType.MATERN_52),
            kernel_after=LeafKernelSpec(kernel_type=KernelType.RBF),
        )
        assert kernel.kind == "changepoint"

    def test_json_round_trip(self) -> None:
        kernel = ChangepointKernelSpec(
            kernel_before=LeafKernelSpec(kernel_type=KernelType.MATERN_52),
            kernel_after=LeafKernelSpec(kernel_type=KernelType.RBF),
            changepoint_location=0.5,
            changepoint_steepness=2.0,
        )
        data = json.loads(kernel.model_dump_json())
        assert data["kind"] == "changepoint"
        assert data["kernel_before"]["kind"] == "leaf"
        assert data["kernel_before"]["kernel_type"] == "Matern52"
        assert data["kernel_after"]["kind"] == "leaf"
        assert data["changepoint_location"] == pytest.approx(0.5)


class TestNormalizeKernelSpec:
    def test_leaf_kernel_normalizes_to_leaf_spec(self) -> None:
        old = KernelSpec(kernel_type=KernelType.MATERN_52, ard=True)
        result = normalize_kernel_spec(old)
        assert isinstance(result, LeafKernelSpec)
        assert result.kernel_type == KernelType.MATERN_52
        assert result.ard is True

    def test_additive_composite_normalizes(self) -> None:
        old = KernelSpec(
            kernel_type=KernelType.RBF,
            composition=CompositionType.ADDITIVE,
            children=[
                KernelSpec(kernel_type=KernelType.RBF),
                KernelSpec(kernel_type=KernelType.MATERN_52),
            ],
        )
        result = normalize_kernel_spec(old)
        assert isinstance(result, CompositeKernelSpec)
        assert result.composition == CompositionType.ADDITIVE
        assert len(result.children) == 2

    def test_changepoint_normalizes(self) -> None:
        old = KernelSpec(
            kernel_type=KernelType.CHANGEPOINT,
            children=[
                KernelSpec(kernel_type=KernelType.MATERN_52),
                KernelSpec(kernel_type=KernelType.RBF),
            ],
            changepoint_location=0.5,
        )
        result = normalize_kernel_spec(old)
        assert isinstance(result, ChangepointKernelSpec)
        assert isinstance(result.kernel_before, LeafKernelSpec)
        assert isinstance(result.kernel_after, LeafKernelSpec)
        assert result.changepoint_location == pytest.approx(0.5)

    def test_ambiguous_children_with_none_composition_raises(self) -> None:
        old = KernelSpec(
            kernel_type=KernelType.RBF,
            composition=CompositionType.NONE,
            children=[
                KernelSpec(kernel_type=KernelType.RBF),
                KernelSpec(kernel_type=KernelType.MATERN_52),
            ],
        )
        with pytest.raises(ValueError, match="ADDITIVE or MULTIPLICATIVE"):
            normalize_kernel_spec(old)


class TestFeatureGroupKernelNormalization:
    def test_legacy_kernel_spec_normalized_to_leaf(self) -> None:
        group = FeatureGroupSpec(
            name="test",
            feature_indices=[0],
            kernel=KernelSpec(kernel_type=KernelType.RBF, ard=True),
        )
        assert isinstance(group.kernel, LeafKernelSpec)
        assert group.kernel.kernel_type == KernelType.RBF
        assert group.kernel.ard is True

    def test_new_leaf_kernel_spec_accepted(self) -> None:
        group = FeatureGroupSpec(
            name="test",
            feature_indices=[0],
            kernel=LeafKernelSpec(kernel_type=KernelType.MATERN_52),
        )
        assert isinstance(group.kernel, LeafKernelSpec)
