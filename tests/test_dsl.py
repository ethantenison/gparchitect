"""Tests for the GPArchitect DSL schema module."""

from __future__ import annotations

import json

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
    SpectralMixtureInitialization,
)


class TestPriorSpec:
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
        assert spec.multitask_rank is None

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
            output_dim=2,
            task_feature_index=2,
            multitask_rank=1,
        )
        assert spec.model_class == ModelClass.MULTI_TASK_GP
        assert spec.task_feature_index == 2
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
            output_means={0: MeanSpec(mean_type=MeanFunctionType.CONSTANT), 1: MeanSpec(mean_type=MeanFunctionType.LINEAR)},
        )
        data = json.loads(spec.model_dump_json())
        assert data["output_means"]["0"]["mean_type"] == "Constant"
        assert data["output_means"]["1"]["mean_type"] == "Linear"

    def test_description_field(self) -> None:
        spec = GPSpec(description="Test spec")
        assert spec.description == "Test spec"

    def test_enum_values_are_strings(self) -> None:
        assert ModelClass.SINGLE_TASK_GP.value == "SingleTaskGP"
        assert KernelType.MATERN_52.value == "Matern52"
        assert CompositionType.ADDITIVE.value == "additive"
        assert CompositionType.HIERARCHICAL.value == "hierarchical"
