"""Tests for Tier 1 non-stationary GP features: changepoint kernel, recency weighting,
and heteroskedastic noise guardrail."""

from __future__ import annotations

import json

import pytest

from gparchitect.dsl.schema import (
    CompositionType,
    ExecutionSpec,
    FeatureGroupSpec,
    GPSpec,
    KernelSpec,
    KernelType,
    ModelClass,
    NoiseSpec,
    RecencyWeightingMode,
    RecencyWeightingSpec,
)
from gparchitect.validation.validator import validate_dsl


def _make_changepoint_spec(input_dim: int = 1) -> GPSpec:
    """Minimal valid single-time-feature changepoint spec."""
    return GPSpec(
        model_class=ModelClass.SINGLE_TASK_GP,
        feature_groups=[
            FeatureGroupSpec(
                name="time",
                feature_indices=[0],
                kernel=KernelSpec(
                    kernel_type=KernelType.CHANGEPOINT,
                    changepoint_location=0.5,
                    changepoint_steepness=1.0,
                    children=[
                        KernelSpec(kernel_type=KernelType.MATERN_52),
                        KernelSpec(kernel_type=KernelType.RBF),
                    ],
                ),
            )
        ],
        input_dim=input_dim,
        output_dim=1,
    )


def _make_recency_spec(mode: RecencyWeightingMode, **kwargs: object) -> GPSpec:
    """Minimal valid spec with recency weighting on feature index 0."""
    return GPSpec(
        model_class=ModelClass.SINGLE_TASK_GP,
        feature_groups=[
            FeatureGroupSpec(
                name="all",
                feature_indices=[0],
                kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
            )
        ],
        execution=ExecutionSpec(
            recency_weighting=RecencyWeightingSpec(
                mode=mode,
                time_feature_index=0,
                **kwargs,  # type: ignore[arg-type]
            )
        ),
        input_dim=1,
        output_dim=1,
    )


# ---------------------------------------------------------------------------
# DSL schema tests
# ---------------------------------------------------------------------------


class TestChangepointKernelDSL:
    def test_changepoint_kernel_type_exists(self) -> None:
        assert KernelType.CHANGEPOINT.value == "Changepoint"

    def test_kernel_spec_accepts_changepoint_fields(self) -> None:
        kernel = KernelSpec(
            kernel_type=KernelType.CHANGEPOINT,
            changepoint_location=0.3,
            changepoint_steepness=2.0,
            children=[
                KernelSpec(kernel_type=KernelType.MATERN_52),
                KernelSpec(kernel_type=KernelType.RBF),
            ],
        )
        assert kernel.changepoint_location == pytest.approx(0.3)
        assert kernel.changepoint_steepness == pytest.approx(2.0)
        assert len(kernel.children) == 2

    def test_changepoint_fields_default_to_none(self) -> None:
        kernel = KernelSpec(kernel_type=KernelType.MATERN_52)
        assert kernel.changepoint_location is None
        assert kernel.changepoint_steepness is None

    def test_changepoint_spec_round_trips_json(self) -> None:
        spec = _make_changepoint_spec()
        data = json.loads(spec.model_dump_json())
        kernel = data["feature_groups"][0]["kernel"]
        assert kernel["kernel_type"] == "Changepoint"
        assert kernel["changepoint_location"] == pytest.approx(0.5)
        assert len(kernel["children"]) == 2


class TestRecencyWeightingDSL:
    def test_recency_weighting_mode_values(self) -> None:
        assert RecencyWeightingMode.SLIDING_WINDOW.value == "sliding_window"
        assert RecencyWeightingMode.EXPONENTIAL_DISCOUNT.value == "exponential_discount"

    def test_recency_weighting_spec_defaults(self) -> None:
        rw = RecencyWeightingSpec(
            mode=RecencyWeightingMode.SLIDING_WINDOW,
            time_feature_index=0,
            window_size=0.5,
        )
        assert rw.min_weight == pytest.approx(0.01)
        assert rw.discount_rate is None

    def test_execution_spec_accepts_recency_weighting(self) -> None:
        ex = ExecutionSpec(
            recency_weighting=RecencyWeightingSpec(
                mode=RecencyWeightingMode.EXPONENTIAL_DISCOUNT,
                time_feature_index=1,
                discount_rate=2.0,
            )
        )
        assert ex.recency_weighting is not None
        assert ex.recency_weighting.discount_rate == pytest.approx(2.0)

    def test_execution_spec_recency_defaults_to_none(self) -> None:
        ex = ExecutionSpec()
        assert ex.recency_weighting is None

    def test_recency_spec_serializes(self) -> None:
        spec = _make_recency_spec(RecencyWeightingMode.SLIDING_WINDOW, window_size=0.3)
        data = json.loads(spec.model_dump_json())
        rw = data["execution"]["recency_weighting"]
        assert rw["mode"] == "sliding_window"
        assert rw["window_size"] == pytest.approx(0.3)


class TestHeteroskedasticNoiseDSL:
    def test_heteroskedastic_noise_defaults_false(self) -> None:
        noise = NoiseSpec()
        assert noise.heteroskedastic_noise is False

    def test_heteroskedastic_noise_serializes(self) -> None:
        noise = NoiseSpec(heteroskedastic_noise=False)
        data = json.loads(noise.model_dump_json())
        assert data["heteroskedastic_noise"] is False


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestChangepointValidation:
    def test_valid_changepoint_spec_passes(self) -> None:
        spec = _make_changepoint_spec()
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_changepoint_with_wrong_feature_count_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0, 1],  # 2 features — invalid for changepoint
                    kernel=KernelSpec(
                        kernel_type=KernelType.CHANGEPOINT,
                        children=[
                            KernelSpec(kernel_type=KernelType.MATERN_52),
                            KernelSpec(kernel_type=KernelType.RBF),
                        ],
                    ),
                )
            ],
            input_dim=2,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("Changepoint" in e and "one active feature" in e for e in result.errors)

    def test_changepoint_with_wrong_child_count_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.CHANGEPOINT,
                        children=[KernelSpec(kernel_type=KernelType.MATERN_52)],  # only 1 child
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("Changepoint" in e and "2 children" in e for e in result.errors)

    def test_changepoint_negative_steepness_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.CHANGEPOINT,
                        changepoint_steepness=-1.0,
                        children=[
                            KernelSpec(kernel_type=KernelType.MATERN_52),
                            KernelSpec(kernel_type=KernelType.RBF),
                        ],
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("steepness" in e for e in result.errors)


class TestRecencyWeightingValidation:
    def test_valid_sliding_window_passes(self) -> None:
        spec = _make_recency_spec(RecencyWeightingMode.SLIDING_WINDOW, window_size=0.5)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_valid_exponential_discount_passes(self) -> None:
        spec = _make_recency_spec(RecencyWeightingMode.EXPONENTIAL_DISCOUNT, discount_rate=1.0)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_sliding_window_missing_window_size_fails(self) -> None:
        spec = _make_recency_spec(RecencyWeightingMode.SLIDING_WINDOW)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("window_size" in e for e in result.errors)

    def test_exponential_discount_missing_rate_fails(self) -> None:
        spec = _make_recency_spec(RecencyWeightingMode.EXPONENTIAL_DISCOUNT)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("discount_rate" in e for e in result.errors)

    def test_negative_window_size_fails(self) -> None:
        spec = _make_recency_spec(RecencyWeightingMode.SLIDING_WINDOW, window_size=-0.1)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("window_size" in e for e in result.errors)

    def test_negative_discount_rate_fails(self) -> None:
        spec = _make_recency_spec(RecencyWeightingMode.EXPONENTIAL_DISCOUNT, discount_rate=-1.0)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("discount_rate" in e for e in result.errors)

    def test_time_feature_index_out_of_range_fails(self) -> None:
        spec = _make_recency_spec(RecencyWeightingMode.SLIDING_WINDOW, window_size=0.5)
        assert spec.execution.recency_weighting is not None
        spec.execution.recency_weighting.time_feature_index = 99
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("time_feature_index" in e for e in result.errors)

    def test_invalid_min_weight_fails(self) -> None:
        spec = _make_recency_spec(
            RecencyWeightingMode.EXPONENTIAL_DISCOUNT,
            discount_rate=1.0,
            min_weight=1.5,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("min_weight" in e for e in result.errors)


class TestHeteroskedasticNoiseValidation:
    def test_heteroskedastic_noise_true_rejected(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            noise=NoiseSpec(heteroskedastic_noise=True),
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("heteroskedastic_noise" in e for e in result.errors)

    def test_heteroskedastic_noise_false_accepted(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            noise=NoiseSpec(heteroskedastic_noise=False),
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Translator tests
# ---------------------------------------------------------------------------


class TestTranslatorChangepoint:
    def test_changepoint_keyword_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a changepoint kernel", input_dim=1)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.CHANGEPOINT

    def test_change_point_two_words_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a change point kernel", input_dim=1)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.CHANGEPOINT

    def test_regime_shift_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Model a regime shift at time 0.5", input_dim=1)
        assert spec.feature_groups[0].kernel.kernel_type == KernelType.CHANGEPOINT

    def test_changepoint_location_extracted(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Changepoint at 0.3", input_dim=1)
        assert spec.feature_groups[0].kernel.changepoint_location == pytest.approx(0.3)

    def test_changepoint_steepness_extracted(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Changepoint with steepness 2.0", input_dim=1)
        assert spec.feature_groups[0].kernel.changepoint_steepness == pytest.approx(2.0)

    def test_changepoint_has_two_children(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a changepoint kernel", input_dim=1)
        assert len(spec.feature_groups[0].kernel.children) == 2

    def test_translated_changepoint_spec_is_valid(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Changepoint at 0.5 with steepness 1.0", input_dim=1)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors


class TestTranslatorRecencyWeighting:
    def test_sliding_window_keyword_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a sliding window of 0.4", input_dim=2)
        assert spec.execution.recency_weighting is not None
        assert spec.execution.recency_weighting.mode == RecencyWeightingMode.SLIDING_WINDOW

    def test_exponential_forgetting_keyword_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use exponential forgetting with rate 2.0", input_dim=2)
        assert spec.execution.recency_weighting is not None
        assert spec.execution.recency_weighting.mode == RecencyWeightingMode.EXPONENTIAL_DISCOUNT

    def test_discount_rate_extracted(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Apply exponential discount with rate 3.0", input_dim=2)
        rw = spec.execution.recency_weighting
        assert rw is not None
        assert rw.discount_rate == pytest.approx(3.0)

    def test_window_size_extracted(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Sliding window size 0.6", input_dim=2)
        rw = spec.execution.recency_weighting
        assert rw is not None
        assert rw.window_size == pytest.approx(0.6)

    def test_time_column_name_sets_feature_index(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl(
            "Use a sliding window",
            input_dim=3,
            input_feature_names=["x1", "time", "x3"],
        )
        rw = spec.execution.recency_weighting
        assert rw is not None
        assert rw.time_feature_index == 1

    def test_no_recency_keyword_gives_none(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a Matern52 kernel", input_dim=2)
        assert spec.execution.recency_weighting is None


# ---------------------------------------------------------------------------
# Recency weighting functional tests
# ---------------------------------------------------------------------------


class TestRecencyWeightingFunctional:
    """Verify that the recency weighting filter produces the expected subsets."""

    @pytest.fixture()
    def tensors(self):  # type: ignore[no-untyped-def]
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        # 10 observations with time in [0.0, 0.9]
        t = torch.linspace(0.0, 0.9, 10, dtype=torch.double)
        train_X = t.unsqueeze(-1)
        train_Y = torch.randn(10, 1, dtype=torch.double)
        return train_X, train_Y

    def test_sliding_window_keeps_recent(self, tensors):  # type: ignore[no-untyped-def]
        from gparchitect.builders.recency import apply_recency_weighting

        train_X, train_Y = tensors
        rw = RecencyWeightingSpec(
            mode=RecencyWeightingMode.SLIDING_WINDOW,
            time_feature_index=0,
            window_size=0.3,
        )
        filtered_X, filtered_Y = apply_recency_weighting(train_X, train_Y, rw)
        # max time is 0.9; cutoff is 0.6; observations with t >= 0.6
        assert filtered_X[:, 0].min().item() >= 0.6 - 1e-9
        assert len(filtered_X) < len(train_X)

    def test_exponential_discount_keeps_recent(self, tensors):  # type: ignore[no-untyped-def]
        from gparchitect.builders.recency import apply_recency_weighting

        train_X, train_Y = tensors
        rw = RecencyWeightingSpec(
            mode=RecencyWeightingMode.EXPONENTIAL_DISCOUNT,
            time_feature_index=0,
            discount_rate=10.0,
            min_weight=0.1,
        )
        filtered_X, filtered_Y = apply_recency_weighting(train_X, train_Y, rw)
        # With rate=10 and min_weight=0.1, cutoff delta_t = -ln(0.1)/10 ≈ 0.23
        # so only the last few observations are kept
        assert len(filtered_X) < len(train_X)
        # The most recent observation must always be included
        import torch
        assert float(filtered_X[:, 0].max().item()) == pytest.approx(float(train_X[:, 0].max().item()))

    def test_sliding_window_all_too_old_retains_one(self, tensors):  # type: ignore[no-untyped-def]
        from gparchitect.builders.recency import apply_recency_weighting

        train_X, train_Y = tensors
        rw = RecencyWeightingSpec(
            mode=RecencyWeightingMode.SLIDING_WINDOW,
            time_feature_index=0,
            window_size=1e-10,  # essentially zero window
        )
        filtered_X, filtered_Y = apply_recency_weighting(train_X, train_Y, rw)
        # Should retain at least 1 observation (the most recent)
        assert len(filtered_X) >= 1

    def test_out_of_range_feature_index_raises(self, tensors):  # type: ignore[no-untyped-def]
        from gparchitect.builders.recency import apply_recency_weighting

        train_X, train_Y = tensors
        rw = RecencyWeightingSpec(
            mode=RecencyWeightingMode.SLIDING_WINDOW,
            time_feature_index=5,  # only 1 feature column
            window_size=0.5,
        )
        with pytest.raises(ValueError, match="time_feature_index"):
            apply_recency_weighting(train_X, train_Y, rw)


# ---------------------------------------------------------------------------
# Changepoint builder tests
# ---------------------------------------------------------------------------


class TestChangepointBuilder:
    def test_changepoint_kernel_constructed(self) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_changepoint_spec(input_dim=1)
        spec.input_dim = 1
        train_X = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
        train_Y = torch.randn(10, 1, dtype=torch.double)

        model = build_model_from_dsl(spec, train_X, train_Y)
        assert model is not None

    def test_changepoint_builder_wrong_children_raises(self) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from gparchitect.builders.builder import build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.CHANGEPOINT,
                        children=[KernelSpec(kernel_type=KernelType.MATERN_52)],  # only 1
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )
        train_X = torch.ones(5, 1, dtype=torch.double)
        train_Y = torch.ones(5, 1, dtype=torch.double)

        with pytest.raises((ValueError, Exception)):
            build_model_from_dsl(spec, train_X, train_Y)
