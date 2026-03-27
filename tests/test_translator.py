"""Tests for the GPArchitect translator module."""

from __future__ import annotations

import pytest

from gparchitect.dsl.schema import (
    CompositionType,
    KernelType,
    MeanFunctionType,
    ModelClass,
    PriorDistribution,
)
from gparchitect.translator.translator import translate_to_dsl


class TestTranslateKernelDetection:
    def test_default_gives_matern52(self) -> None:
        spec = translate_to_dsl("Build a GP model", input_dim=3)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.MATERN_52

    def test_rbf_keyword(self) -> None:
        spec = translate_to_dsl("Use an RBF kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.RBF

    def test_squared_exponential_maps_to_rbf(self) -> None:
        spec = translate_to_dsl("Use a squared exponential kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.RBF

    def test_rq_keyword(self) -> None:
        spec = translate_to_dsl("Use a rational quadratic kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.RQ

    def test_spectral_mixture_keyword(self) -> None:
        spec = translate_to_dsl("Use a spectral mixture kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.SPECTRAL_MIXTURE

    def test_infinite_width_bnn_keyword(self) -> None:
        spec = translate_to_dsl("Use an infinite width bnn kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.INFINITE_WIDTH_BNN

    def test_exponential_decay_keyword(self) -> None:
        spec = translate_to_dsl("Use an exponential decay kernel", input_dim=1)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.EXPONENTIAL_DECAY

    def test_matern_52_keyword(self) -> None:
        spec = translate_to_dsl("Use Matern 5/2 kernel", input_dim=4)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.MATERN_52

    def test_matern_32_keyword(self) -> None:
        spec = translate_to_dsl("Matern 3/2 kernel please", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.MATERN_32

    def test_matern_12_keyword(self) -> None:
        spec = translate_to_dsl("Matern 1/2 kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.MATERN_12

    def test_matern_no_nu_defaults_to_52(self) -> None:
        spec = translate_to_dsl("Use a Matern kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.MATERN_52

    def test_periodic_kernel(self) -> None:
        spec = translate_to_dsl("Use a periodic kernel", input_dim=1)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.PERIODIC

    def test_linear_kernel(self) -> None:
        spec = translate_to_dsl("Use a linear kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.LINEAR


class TestTranslateModelClassDetection:
    def test_default_single_task(self) -> None:
        spec = translate_to_dsl("A simple GP", input_dim=3)
        assert spec.model_class == ModelClass.SINGLE_TASK_GP

    def test_multitask_keyword(self) -> None:
        spec = translate_to_dsl(
            "Use a multi-task GP",
            input_dim=4,
            output_dim=1,
            task_feature_index=3,
            task_values=[0, 1],
        )
        assert spec.model_class == ModelClass.MULTI_TASK_GP

    def test_task_feature_index_forces_multitask(self) -> None:
        spec = translate_to_dsl("Just a GP", input_dim=4, output_dim=1, task_feature_index=3, task_values=[0, 1])
        assert spec.model_class == ModelClass.MULTI_TASK_GP

    def test_model_list_keyword(self) -> None:
        spec = translate_to_dsl("Use a ModelListGP with independent outputs", input_dim=3, output_dim=2)
        assert spec.model_class == ModelClass.MODEL_LIST_GP

    def test_single_task_explicit(self) -> None:
        spec = translate_to_dsl("SingleTaskGP with RBF", input_dim=2)
        assert spec.model_class == ModelClass.SINGLE_TASK_GP


class TestTranslateARD:
    def test_ard_enabled_by_default(self) -> None:
        spec = translate_to_dsl("GP model", input_dim=5)
        assert spec.feature_groups[0].kernel.ard is True

    def test_ard_keyword_keeps_ard_enabled(self) -> None:
        spec = translate_to_dsl("Use ARD on all inputs", input_dim=5)
        assert spec.feature_groups[0].kernel.ard is True

    def test_automatic_relevance_keyword(self) -> None:
        spec = translate_to_dsl("automatic relevance determination", input_dim=3)
        assert spec.feature_groups[0].kernel.ard is True

    def test_disable_ard_keyword_turns_ard_off(self) -> None:
        spec = translate_to_dsl("Use an RBF kernel without ARD", input_dim=3)
        assert spec.feature_groups[0].kernel.ard is False

    def test_shared_lengthscale_turns_ard_off(self) -> None:
        spec = translate_to_dsl("Use a Matern kernel with shared lengthscale", input_dim=3)
        assert spec.feature_groups[0].kernel.ard is False

    def test_linear_kernel_does_not_enable_ard(self) -> None:
        spec = translate_to_dsl("Use a linear kernel", input_dim=3)
        assert spec.feature_groups[0].kernel.ard is False

    def test_spectral_mixture_keeps_required_ard(self) -> None:
        spec = translate_to_dsl("Use a spectral mixture kernel without ARD", input_dim=3)
        assert spec.feature_groups[0].kernel.ard is True


class TestTranslateNoise:
    def test_default_learnable_noise(self) -> None:
        spec = translate_to_dsl("GP model", input_dim=2)
        assert spec.noise.fixed is False

    def test_fixed_noise_keyword(self) -> None:
        spec = translate_to_dsl("Use fixed noise", input_dim=2)
        assert spec.noise.fixed is True
        assert spec.noise.noise_value is not None

    def test_noiseless_keyword(self) -> None:
        spec = translate_to_dsl("Noiseless GP", input_dim=2)
        assert spec.noise.fixed is True

    def test_gamma_noise_prior_phrase_is_parsed(self) -> None:
        spec = translate_to_dsl(
            "Use an rbf kernel with gamma prior on noise concentration 2.0 rate 0.5",
            input_dim=2,
        )
        assert spec.noise.prior is not None
        assert spec.noise.prior.distribution == PriorDistribution.GAMMA
        assert spec.noise.prior.params == {"concentration": 2.0, "rate": 0.5}

    def test_noise_target_first_gamma_prior_phrase_is_parsed(self) -> None:
        spec = translate_to_dsl(
            "Use an rbf kernel where observation noise has a gamma prior with shape 2.0 beta 0.5",
            input_dim=2,
        )
        assert spec.noise.prior is not None
        assert spec.noise.prior.distribution == PriorDistribution.GAMMA
        assert spec.noise.prior.params == {"concentration": 2.0, "rate": 0.5}


class TestTranslateMeans:
    def test_default_mean_is_unset(self) -> None:
        spec = translate_to_dsl("GP model", input_dim=2)
        assert spec.mean is None
        assert spec.output_means == {}

    def test_constant_mean_keyword(self) -> None:
        spec = translate_to_dsl("Use a constant mean with an RBF kernel", input_dim=2)
        assert spec.mean is not None
        assert spec.mean.mean_type == MeanFunctionType.CONSTANT

    def test_zero_mean_keyword(self) -> None:
        spec = translate_to_dsl("Use a zero mean", input_dim=2)
        assert spec.mean is not None
        assert spec.mean.mean_type == MeanFunctionType.ZERO

    def test_linear_mean_keyword(self) -> None:
        spec = translate_to_dsl("Use a linear mean", input_dim=3)
        assert spec.mean is not None
        assert spec.mean.mean_type == MeanFunctionType.LINEAR

    def test_model_list_output_mean_keyword(self) -> None:
        spec = translate_to_dsl(
            "Use a ModelListGP where output 1 uses zero mean and output 2 uses linear mean",
            input_dim=2,
            output_dim=2,
        )
        assert spec.output_means[0].mean_type == MeanFunctionType.ZERO
        assert spec.output_means[1].mean_type == MeanFunctionType.LINEAR

    def test_multitask_task_mean_keyword(self) -> None:
        spec = translate_to_dsl(
            "Use a multitask GP with zero mean for task 0 and constant mean for task 1",
            input_dim=3,
            output_dim=1,
            task_feature_index=2,
        )
        assert spec.output_means[0].mean_type == MeanFunctionType.ZERO
        assert spec.output_means[1].mean_type == MeanFunctionType.CONSTANT
        assert spec.task_values == [0, 1]


class TestTranslateComposition:
    def test_default_additive(self) -> None:
        spec = translate_to_dsl("GP model", input_dim=3)
        assert spec.group_composition == CompositionType.ADDITIVE

    def test_multiplicative_keyword(self) -> None:
        spec = translate_to_dsl("Use multiplicative kernel combination", input_dim=3)
        assert spec.group_composition == CompositionType.MULTIPLICATIVE

    def test_feature_specific_kernels_default_to_hierarchical(self) -> None:
        spec = translate_to_dsl(
            "Use a rbf kernel on x1 and a matern1/2 kernel on x2.",
            input_dim=2,
            input_feature_names=["x1", "x2"],
        )

        assert spec.group_composition == CompositionType.HIERARCHICAL
        assert len(spec.feature_groups) == 2
        assert spec.feature_groups[0].feature_indices == [0]
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.RBF
        assert spec.feature_groups[0].kernel.ard is True
        assert spec.feature_groups[1].feature_indices == [1]
        assert spec.feature_groups[1].kernel.kernel_type == KernelType.MATERN_12
        assert spec.feature_groups[1].kernel.ard is True

    def test_feature_specific_kernels_respect_additive_override(self) -> None:
        spec = translate_to_dsl(
            "Use an additive rbf kernel on x1 and a matern1/2 kernel on x2.",
            input_dim=2,
            input_feature_names=["x1", "x2"],
        )

        assert spec.group_composition == CompositionType.ADDITIVE

    def test_kernel_mentions_preserve_time_and_grouped_features(self) -> None:
        spec = translate_to_dsl(
            (
                "Use an rbf kernel on month_index, a matern3/2 kernel on credit_spread_bps "
                "and vix_level, and an rbf kernel on net_flow_pct and momentum_3m_pct."
            ),
            input_dim=5,
            input_feature_names=[
                "month_index",
                "credit_spread_bps",
                "vix_level",
                "net_flow_pct",
                "momentum_3m_pct",
            ],
        )

        assert spec.group_composition == CompositionType.HIERARCHICAL
        assert len(spec.feature_groups) == 3

        assert spec.feature_groups[0].feature_indices == [0]
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.RBF
        assert spec.feature_groups[0].kernel.ard is True

        assert spec.feature_groups[1].feature_indices == [1, 2]
        assert spec.feature_groups[1].kernel.kernel_type == KernelType.MATERN_32
        assert spec.feature_groups[1].kernel.ard is True

        assert spec.feature_groups[2].feature_indices == [3, 4]
        assert spec.feature_groups[2].kernel.kernel_type == KernelType.RBF
        assert spec.feature_groups[2].kernel.ard is True

    def test_feature_specific_kernels_can_disable_ard_globally(self) -> None:
        spec = translate_to_dsl(
            "Use an rbf kernel on x1 and a matern1/2 kernel on x2 without ARD.",
            input_dim=2,
            input_feature_names=["x1", "x2"],
        )

        assert all(group.kernel.ard is False for group in spec.feature_groups)


class TestTranslateStructure:
    def test_input_dim_stored(self) -> None:
        spec = translate_to_dsl("GP model", input_dim=7)
        assert spec.input_dim == 7

    def test_output_dim_stored(self) -> None:
        spec = translate_to_dsl("GP model", input_dim=3, output_dim=2)
        assert spec.output_dim == 2

    def test_feature_indices_exclude_task_column(self) -> None:
        spec = translate_to_dsl("GP model", input_dim=4, output_dim=1, task_feature_index=3, task_values=[0, 1])
        for group in spec.feature_groups:
            assert 3 not in group.feature_indices

    def test_single_task_defaults_to_explicit_execution_spec(self) -> None:
        spec = translate_to_dsl("GP model", input_dim=2)
        assert spec.execution.input_scaling is True
        assert spec.execution.outcome_standardization is True

    def test_multitask_defaults_disable_outcome_standardization(self) -> None:
        spec = translate_to_dsl(
            "multitask GP",
            input_dim=3,
            output_dim=1,
            task_feature_index=2,
            task_values=[0, 1],
        )
        assert spec.execution.input_scaling is True
        assert spec.execution.outcome_standardization is False

    def test_model_list_uses_single_shared_group_when_no_feature_mapping_exists(self) -> None:
        spec = translate_to_dsl("ModelListGP", input_dim=3, output_dim=3)
        assert len(spec.feature_groups) == 1
        assert spec.feature_groups[0].feature_indices == [0, 1, 2]

    def test_description_contains_instruction(self) -> None:
        spec = translate_to_dsl("Use Matern52 with ARD", input_dim=2)
        assert "Matern52" in spec.description or "ARD" in spec.description.upper()

    def test_invalid_input_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="input_dim"):
            translate_to_dsl("GP model", input_dim=0)

    def test_invalid_output_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="output_dim"):
            translate_to_dsl("GP model", input_dim=2, output_dim=0)

    def test_deterministic_output(self) -> None:
        """Same instruction always produces the same spec."""
        spec1 = translate_to_dsl("Use RBF with ARD", input_dim=4)
        spec2 = translate_to_dsl("Use RBF with ARD", input_dim=4)
        assert spec1.model_dump() == spec2.model_dump()

    def test_multitask_rank_set(self) -> None:
        spec = translate_to_dsl("multitask GP", input_dim=5, output_dim=1, task_feature_index=4, task_values=[0, 1])
        assert spec.multitask_rank is not None
        assert spec.multitask_rank >= 1


class TestTranslateKernelSpecificParameters:
    def test_rq_alpha_parsed(self) -> None:
        spec = translate_to_dsl("Use an RQ kernel with alpha 0.75", input_dim=2)
        assert spec.feature_groups[0].kernel.rq_alpha == pytest.approx(0.75)

    def test_spectral_mixture_num_mixtures_parsed(self) -> None:
        spec = translate_to_dsl("Use a 4 component spectral mixture kernel", input_dim=2)
        assert spec.feature_groups[0].kernel.num_mixtures == 4

    def test_spectral_mixture_empirical_spectrum_init_parsed(self) -> None:
        spec = translate_to_dsl(
            "Use a spectral mixture kernel with 3 mixtures initialized from the empirical spectrum",
            input_dim=2,
        )
        assert spec.feature_groups[0].kernel.num_mixtures == 3
        assert spec.feature_groups[0].kernel.spectral_init.value == "from_empirical_spectrum"

    def test_feature_specific_kernel_params_stay_local(self) -> None:
        spec = translate_to_dsl(
            "Use an RQ kernel with alpha 1.5 on x1 and a 3 component spectral mixture kernel on x2.",
            input_dim=2,
            input_feature_names=["x1", "x2"],
        )

        assert spec.feature_groups[0].kernel.kernel_type == KernelType.RQ
        assert spec.feature_groups[0].kernel.rq_alpha == pytest.approx(1.5)
        assert spec.feature_groups[1].kernel.kernel_type == KernelType.SPECTRAL_MIXTURE
        assert spec.feature_groups[1].kernel.num_mixtures == 3

    def test_polynomial_power_and_offset_parsed(self) -> None:
        spec = translate_to_dsl("Use a polynomial kernel with degree 3 and offset 1.5", input_dim=2)
        kernel = spec.feature_groups[0].kernel
        assert kernel.kernel_type == KernelType.POLYNOMIAL
        assert kernel.polynomial_power == 3
        assert kernel.polynomial_offset == pytest.approx(1.5)

    def test_periodic_period_length_parsed(self) -> None:
        spec = translate_to_dsl("Use a periodic kernel with period length 12", input_dim=1)
        kernel = spec.feature_groups[0].kernel
        assert kernel.kernel_type == KernelType.PERIODIC
        assert kernel.period_length == pytest.approx(12.0)

    def test_infinite_width_bnn_depth_parsed(self) -> None:
        spec = translate_to_dsl("Use an infinite width bnn kernel with depth 5", input_dim=3)
        kernel = spec.feature_groups[0].kernel
        assert kernel.kernel_type == KernelType.INFINITE_WIDTH_BNN
        assert kernel.bnn_depth == 5

    def test_exponential_decay_parameters_parsed(self) -> None:
        spec = translate_to_dsl("Use an exponential decay kernel with power 2.5 and offset 0.2", input_dim=1)
        kernel = spec.feature_groups[0].kernel
        assert kernel.kernel_type == KernelType.EXPONENTIAL_DECAY
        assert kernel.exponential_decay_power == pytest.approx(2.5)
        assert kernel.exponential_decay_offset == pytest.approx(0.2)

    def test_prior_phrases_are_parsed_into_kernel_specs(self) -> None:
        spec = translate_to_dsl(
            (
                "Use a periodic kernel with normal prior on lengthscale loc 0.0 scale 1.0, "
                "uniform prior on period a 0.5 b 3.0, and halfcauchy prior on outputscale scale 0.75"
            ),
            input_dim=1,
        )
        kernel = spec.feature_groups[0].kernel
        assert kernel.lengthscale_prior is not None
        assert kernel.lengthscale_prior.distribution == PriorDistribution.NORMAL
        assert kernel.period_prior is not None
        assert kernel.period_prior.distribution == PriorDistribution.UNIFORM
        assert kernel.outputscale_prior is not None
        assert kernel.outputscale_prior.distribution == PriorDistribution.HALF_CAUCHY

    def test_target_first_prior_phrases_with_synonyms_are_parsed(self) -> None:
        spec = translate_to_dsl(
            (
                "Use a periodic kernel where length scale has a normal prior with mean 0.0 std 1.0, "
                "period length has a uniform prior between 0.5 and 3.0, and output scale has a "
                "half-cauchy prior with beta 0.75"
            ),
            input_dim=1,
        )
        kernel = spec.feature_groups[0].kernel
        assert kernel.lengthscale_prior is not None
        assert kernel.lengthscale_prior.distribution == PriorDistribution.NORMAL
        assert kernel.lengthscale_prior.params == {"loc": 0.0, "scale": 1.0}
        assert kernel.period_prior is not None
        assert kernel.period_prior.distribution == PriorDistribution.UNIFORM
        assert kernel.period_prior.params == {"a": 0.5, "b": 3.0}
        assert kernel.outputscale_prior is not None
        assert kernel.outputscale_prior.distribution == PriorDistribution.HALF_CAUCHY
        assert kernel.outputscale_prior.params == {"scale": 0.75}
