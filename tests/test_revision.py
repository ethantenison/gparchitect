"""Tests for the GPArchitect DSL revision/recovery module."""

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
from gparchitect.revision.revision import RevisionResult, _STRATEGIES, revise_dsl


def _make_spec() -> GPSpec:
    """Helper: a spec with ARD, priors, and MultiTaskGP for testing recovery."""
    return GPSpec(
        model_class=ModelClass.MULTI_TASK_GP,
        feature_groups=[
            FeatureGroupSpec(
                name="features",
                feature_indices=[0, 1, 2],
                kernel=KernelSpec(
                    kernel_type=KernelType.RBF,
                    ard=True,
                    lengthscale_prior=PriorSpec(distribution="Normal", params={"loc": 0.0, "scale": 1.0}),
                ),
            )
        ],
        noise=NoiseSpec(
            fixed=True,
            noise_value=1e-4,
            prior=PriorSpec(distribution="Gamma"),
        ),
        input_dim=4,
        output_dim=2,
        task_feature_index=3,
        multitask_rank=1,
    )


class TestReviseDisableARD:
    def test_disables_ard(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "singular matrix", attempt=0)
        assert result is not None
        for group in result.revised_spec.feature_groups:
            assert group.kernel.ard is False

    def test_strategy_name(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=0)
        assert result is not None
        assert result.strategy == "disable_ard"

    def test_rationale_contains_error(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "singular matrix", attempt=0)
        assert result is not None
        assert "singular" in result.rationale


class TestReviseMeanConfiguration:
    def test_mean_error_prefers_mean_simplification(self) -> None:
        spec = _make_spec()
        spec.mean = MeanSpec(mean_type=MeanFunctionType.LINEAR)
        spec.output_means = {0: MeanSpec(mean_type=MeanFunctionType.ZERO)}

        result = revise_dsl(spec, "RuntimeError: mean_module shape mismatch", attempt=0)

        assert result is not None
        assert result.strategy == "simplify_mean_to_default"
        assert result.revised_spec.mean is None
        assert result.revised_spec.output_means == {}

    def test_non_mean_error_keeps_standard_strategy_order(self) -> None:
        spec = _make_spec()
        spec.mean = MeanSpec(mean_type=MeanFunctionType.LINEAR)

        result = revise_dsl(spec, "RuntimeError: cholesky failed", attempt=0)

        assert result is not None
        assert result.strategy == "disable_ard"
        assert result.revised_spec.mean is not None

    def test_mean_error_without_mean_config_uses_standard_strategy(self) -> None:
        spec = _make_spec()

        result = revise_dsl(spec, "RuntimeError: mean_module shape mismatch", attempt=0)

        assert result is not None
        assert result.strategy == "disable_ard"


class TestReviseSimplifyKernels:
    def test_simplifies_to_matern52(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=1)
        assert result is not None
        for group in result.revised_spec.feature_groups:
            assert group.kernel.kernel_type == KernelType.MATERN_52
            assert group.kernel.ard is False

    def test_single_feature_group_after_simplification(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=1)
        assert result is not None
        assert len(result.revised_spec.feature_groups) == 1

    def test_composition_reset_to_additive(self) -> None:
        spec = _make_spec()
        spec.group_composition = CompositionType.MULTIPLICATIVE
        result = revise_dsl(spec, "error", attempt=1)
        assert result is not None
        assert result.revised_spec.group_composition == CompositionType.ADDITIVE


class TestReviseRemovePriors:
    def test_removes_lengthscale_prior(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=2)
        assert result is not None
        for group in result.revised_spec.feature_groups:
            assert group.kernel.lengthscale_prior is None

    def test_removes_noise_prior(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=2)
        assert result is not None
        assert result.revised_spec.noise.prior is None


class TestReviseSwitchToSingleTask:
    def test_switches_model_class(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=3)
        assert result is not None
        assert result.revised_spec.model_class == ModelClass.SINGLE_TASK_GP

    def test_clears_task_feature_index(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=3)
        assert result is not None
        assert result.revised_spec.task_feature_index is None

    def test_clears_multitask_rank(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=3)
        assert result is not None
        assert result.revised_spec.multitask_rank is None

    def test_single_feature_group_with_matern52(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=3)
        assert result is not None
        assert len(result.revised_spec.feature_groups) == 1
        assert result.revised_spec.feature_groups[0].kernel.kernel_type == KernelType.MATERN_52


class TestReviseDefaultNoise:
    def test_resets_noise_to_default(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=4)
        assert result is not None
        assert result.revised_spec.noise.fixed is False
        assert result.revised_spec.noise.noise_value is None
        assert result.revised_spec.noise.prior is None


class TestReviseExhaustion:
    def test_returns_none_when_all_strategies_exhausted(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "error", attempt=len(_STRATEGIES))
        assert result is None

    def test_strategies_count_matches_constant(self) -> None:
        assert len(_STRATEGIES) == 5

    def test_original_spec_not_mutated(self) -> None:
        """revise_dsl should not mutate the original spec."""
        spec = _make_spec()
        original_model_class = spec.model_class
        revise_dsl(spec, "error", attempt=3)  # switch_to_single_task
        assert spec.model_class == original_model_class


class TestRevisionResult:
    def test_has_required_fields(self) -> None:
        spec = _make_spec()
        result = revise_dsl(spec, "some error", attempt=0)
        assert result is not None
        assert isinstance(result, RevisionResult)
        assert result.revised_spec is not None
        assert result.rationale != ""
        assert result.strategy != ""
