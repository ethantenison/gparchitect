"""Tests for Tier 1 and Tier 2 non-stationary GP features.

Tier 1 (changepoint kernel, recency filtering, heteroskedastic noise guardrail).
Tier 2 (time-varying hyperparameters, input warping).

Recency filtering (formerly mislabeled "recency weighting") is dataset truncation —
it removes old observations before fitting.  It does NOT perform likelihood-weighted
GP inference.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gparchitect.dsl.schema import (
    ChangepointKernelSpec,
    CompositionType,
    ExecutionSpec,
    FeatureGroupSpec,
    GPSpec,
    InputWarpingSpec,
    KernelSpec,
    KernelType,
    LeafKernelSpec,
    ModelClass,
    NoiseSpec,
    RecencyFilteringMode,
    RecencyFilteringSpec,
    TimeVaryingSpec,
    TimeVaryingTarget,
    WarpType,
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
                kernel=ChangepointKernelSpec(
                    changepoint_location=0.5,
                    changepoint_steepness=1.0,
                    kernel_before=LeafKernelSpec(kernel_type=KernelType.MATERN_52),
                    kernel_after=LeafKernelSpec(kernel_type=KernelType.RBF),
                ),
            )
        ],
        input_dim=input_dim,
        output_dim=1,
    )


def _make_recency_spec(mode: RecencyFilteringMode, **kwargs: object) -> GPSpec:
    """Minimal valid spec with recency filtering on feature index 0."""
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
            recency_filtering=RecencyFilteringSpec(
                mode=mode,
                time_feature_index=0,
                **kwargs,  # type: ignore[arg-type]
            )
        ),
        input_dim=1,
        output_dim=1,
    )


def _make_time_varying_spec(target: TimeVaryingTarget, time_feature_index: int = 0) -> GPSpec:
    """Minimal valid spec with a time-varying hyperparameter kernel."""
    return GPSpec(
        model_class=ModelClass.SINGLE_TASK_GP,
        feature_groups=[
            FeatureGroupSpec(
                name="time",
                feature_indices=[0],
                kernel=KernelSpec(
                    kernel_type=KernelType.MATERN_52,
                    time_varying=TimeVaryingSpec(
                        target=target,
                        time_feature_index=time_feature_index,
                    ),
                ),
            )
        ],
        input_dim=1,
        output_dim=1,
    )


def _make_input_warping_spec(time_feature_index: int = 0) -> GPSpec:
    """Minimal valid spec with input warping on a time feature."""
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
            input_warping=InputWarpingSpec(
                warp_type=WarpType.KUMARASWAMY,
                time_feature_index=time_feature_index,
            )
        ),
        input_dim=1,
        output_dim=1,
    )


# ---------------------------------------------------------------------------
# DSL schema tests — Tier 1
# ---------------------------------------------------------------------------


class TestChangepointKernelDSL:
    def test_changepoint_kernel_type_exists(self) -> None:
        assert KernelType.CHANGEPOINT.value == "Changepoint"

    def test_kernel_spec_accepts_changepoint_fields(self) -> None:
        kernel = ChangepointKernelSpec(
            changepoint_location=0.3,
            changepoint_steepness=2.0,
            kernel_before=LeafKernelSpec(kernel_type=KernelType.MATERN_52),
            kernel_after=LeafKernelSpec(kernel_type=KernelType.RBF),
        )
        assert kernel.changepoint_location == pytest.approx(0.3)
        assert kernel.changepoint_steepness == pytest.approx(2.0)
        assert kernel.kernel_before.kernel_type == KernelType.MATERN_52
        assert kernel.kernel_after.kernel_type == KernelType.RBF

    def test_changepoint_fields_default_to_none(self) -> None:
        kernel = KernelSpec(kernel_type=KernelType.MATERN_52)
        assert kernel.changepoint_location is None
        assert kernel.changepoint_steepness is None

    def test_changepoint_spec_round_trips_json(self) -> None:
        spec = _make_changepoint_spec()
        data = json.loads(spec.model_dump_json())
        kernel = data["feature_groups"][0]["kernel"]
        assert kernel["kind"] == "changepoint"
        assert kernel["changepoint_location"] == pytest.approx(0.5)
        assert kernel["kernel_before"]["kind"] == "leaf"
        assert kernel["kernel_after"]["kind"] == "leaf"


class TestRecencyFilteringDSL:
    """Tests for the Tier 1 recency filtering DSL (formerly 'recency weighting').

    Recency filtering is dataset truncation — old observations are removed
    before fitting.  It is NOT true likelihood-weighted GP inference.
    """

    def test_recency_filtering_mode_values(self) -> None:
        assert RecencyFilteringMode.SLIDING_WINDOW.value == "sliding_window"
        assert RecencyFilteringMode.EXPONENTIAL_DISCOUNT.value == "exponential_discount"

    def test_recency_filtering_spec_defaults(self) -> None:
        rf = RecencyFilteringSpec(
            mode=RecencyFilteringMode.SLIDING_WINDOW,
            time_feature_index=0,
            window_size=0.5,
        )
        assert rf.min_weight == pytest.approx(0.01)
        assert rf.discount_rate is None

    def test_execution_spec_accepts_recency_filtering(self) -> None:
        ex = ExecutionSpec(
            recency_filtering=RecencyFilteringSpec(
                mode=RecencyFilteringMode.EXPONENTIAL_DISCOUNT,
                time_feature_index=1,
                discount_rate=2.0,
            )
        )
        assert ex.recency_filtering is not None
        assert ex.recency_filtering.discount_rate == pytest.approx(2.0)

    def test_execution_spec_recency_defaults_to_none(self) -> None:
        ex = ExecutionSpec()
        assert ex.recency_filtering is None

    def test_recency_spec_serializes(self) -> None:
        spec = _make_recency_spec(RecencyFilteringMode.SLIDING_WINDOW, window_size=0.3)
        data = json.loads(spec.model_dump_json())
        rf = data["execution"]["recency_filtering"]
        assert rf["mode"] == "sliding_window"
        assert rf["window_size"] == pytest.approx(0.3)

    def test_canonical_field_name_is_recency_filtering(self) -> None:
        """The canonical DSL field is recency_filtering, not recency_weighting."""
        ex = ExecutionSpec(
            recency_filtering=RecencyFilteringSpec(
                mode=RecencyFilteringMode.SLIDING_WINDOW,
                time_feature_index=0,
                window_size=0.5,
            )
        )
        data = json.loads(ex.model_dump_json())
        assert "recency_filtering" in data
        assert "recency_weighting" not in data


class TestHeteroskedasticNoiseDSL:
    def test_heteroskedastic_noise_defaults_false(self) -> None:
        noise = NoiseSpec()
        assert noise.heteroskedastic_noise is False

    def test_heteroskedastic_noise_serializes(self) -> None:
        noise = NoiseSpec(heteroskedastic_noise=False)
        data = json.loads(noise.model_dump_json())
        assert data["heteroskedastic_noise"] is False


# ---------------------------------------------------------------------------
# DSL schema tests — Tier 2: time-varying hyperparameters
# ---------------------------------------------------------------------------


class TestTimeVaryingDSL:
    def test_time_varying_target_values(self) -> None:
        assert TimeVaryingTarget.OUTPUTSCALE.value == "outputscale"
        assert TimeVaryingTarget.LENGTHSCALE.value == "lengthscale"

    def test_time_varying_spec_defaults(self) -> None:
        tv = TimeVaryingSpec(target=TimeVaryingTarget.OUTPUTSCALE, time_feature_index=0)
        assert tv.parameterization == "linear"
        assert tv.time_feature_index == 0
        assert tv.outputscale_bias_limit == pytest.approx(4.0)
        assert tv.outputscale_slope_limit == pytest.approx(4.0)

    def test_kernel_spec_accepts_time_varying(self) -> None:
        kernel = KernelSpec(
            kernel_type=KernelType.MATERN_52,
            time_varying=TimeVaryingSpec(target=TimeVaryingTarget.OUTPUTSCALE, time_feature_index=0),
        )
        assert kernel.time_varying is not None
        assert kernel.time_varying.target == TimeVaryingTarget.OUTPUTSCALE

    def test_kernel_spec_time_varying_defaults_to_none(self) -> None:
        kernel = KernelSpec(kernel_type=KernelType.MATERN_52)
        assert kernel.time_varying is None

    def test_time_varying_spec_serializes_round_trip(self) -> None:
        spec = _make_time_varying_spec(TimeVaryingTarget.OUTPUTSCALE)
        data = json.loads(spec.model_dump_json())
        tv = data["feature_groups"][0]["kernel"]["time_varying"]
        assert tv["target"] == "outputscale"
        assert tv["time_feature_index"] == 0
        assert tv["parameterization"] == "linear"
        assert tv["outputscale_bias_limit"] == pytest.approx(4.0)
        assert tv["outputscale_slope_limit"] == pytest.approx(4.0)

    def test_time_varying_lengthscale_serializes_round_trip(self) -> None:
        spec = _make_time_varying_spec(TimeVaryingTarget.LENGTHSCALE)
        data = json.loads(spec.model_dump_json())
        tv = data["feature_groups"][0]["kernel"]["time_varying"]
        assert tv["target"] == "lengthscale"


# ---------------------------------------------------------------------------
# DSL schema tests — Tier 2: input warping
# ---------------------------------------------------------------------------


class TestInputWarpingDSL:
    def test_warp_type_values(self) -> None:
        assert WarpType.KUMARASWAMY.value == "kumaraswamy"

    def test_input_warping_spec_defaults(self) -> None:
        iw = InputWarpingSpec(warp_type=WarpType.KUMARASWAMY, time_feature_index=0)
        assert iw.concentration0 is None
        assert iw.concentration1 is None

    def test_input_warping_spec_with_concentrations(self) -> None:
        iw = InputWarpingSpec(
            warp_type=WarpType.KUMARASWAMY,
            time_feature_index=1,
            concentration0=2.0,
            concentration1=0.5,
        )
        assert iw.concentration0 == pytest.approx(2.0)
        assert iw.concentration1 == pytest.approx(0.5)

    def test_execution_spec_accepts_input_warping(self) -> None:
        ex = ExecutionSpec(
            input_warping=InputWarpingSpec(
                warp_type=WarpType.KUMARASWAMY,
                time_feature_index=0,
            )
        )
        assert ex.input_warping is not None
        assert ex.input_warping.warp_type == WarpType.KUMARASWAMY

    def test_execution_spec_input_warping_defaults_to_none(self) -> None:
        ex = ExecutionSpec()
        assert ex.input_warping is None

    def test_input_warping_spec_serializes(self) -> None:
        spec = _make_input_warping_spec()
        data = json.loads(spec.model_dump_json())
        iw = data["execution"]["input_warping"]
        assert iw["warp_type"] == "kumaraswamy"
        assert iw["time_feature_index"] == 0


# ---------------------------------------------------------------------------
# Validation tests — Tier 1
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
        with pytest.raises(ValidationError):
            FeatureGroupSpec(
                name="time",
                feature_indices=[0],
                kernel=ChangepointKernelSpec(
                    kernel_before=LeafKernelSpec(kernel_type=KernelType.MATERN_52),
                    kernel_after=None,  # type: ignore[arg-type]
                ),
            )

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


class TestRecencyFilteringValidation:
    def test_valid_sliding_window_passes(self) -> None:
        spec = _make_recency_spec(RecencyFilteringMode.SLIDING_WINDOW, window_size=0.5)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_valid_exponential_discount_passes(self) -> None:
        spec = _make_recency_spec(RecencyFilteringMode.EXPONENTIAL_DISCOUNT, discount_rate=1.0)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_sliding_window_missing_window_size_fails(self) -> None:
        spec = _make_recency_spec(RecencyFilteringMode.SLIDING_WINDOW)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("window_size" in e for e in result.errors)

    def test_exponential_discount_missing_rate_fails(self) -> None:
        spec = _make_recency_spec(RecencyFilteringMode.EXPONENTIAL_DISCOUNT)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("discount_rate" in e for e in result.errors)

    def test_negative_window_size_fails(self) -> None:
        spec = _make_recency_spec(RecencyFilteringMode.SLIDING_WINDOW, window_size=-0.1)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("window_size" in e for e in result.errors)

    def test_negative_discount_rate_fails(self) -> None:
        spec = _make_recency_spec(RecencyFilteringMode.EXPONENTIAL_DISCOUNT, discount_rate=-1.0)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("discount_rate" in e for e in result.errors)

    def test_time_feature_index_out_of_range_fails(self) -> None:
        spec = _make_recency_spec(RecencyFilteringMode.SLIDING_WINDOW, window_size=0.5)
        assert spec.execution.recency_filtering is not None
        spec.execution.recency_filtering.time_feature_index = 99
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("time_feature_index" in e for e in result.errors)

    def test_invalid_min_weight_fails(self) -> None:
        spec = _make_recency_spec(
            RecencyFilteringMode.EXPONENTIAL_DISCOUNT,
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
# Validation tests — Tier 2: time-varying hyperparameters
# ---------------------------------------------------------------------------


class TestTimeVaryingValidation:
    def test_valid_outputscale_spec_passes(self) -> None:
        spec = _make_time_varying_spec(TimeVaryingTarget.OUTPUTSCALE)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_valid_lengthscale_spec_passes(self) -> None:
        spec = _make_time_varying_spec(TimeVaryingTarget.LENGTHSCALE)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_unsupported_parameterization_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.OUTPUTSCALE,
                            time_feature_index=0,
                            parameterization="neural",  # unsupported
                        ),
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("parameterization" in e for e in result.errors)

    def test_time_varying_negative_feature_index_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.OUTPUTSCALE,
                            time_feature_index=-1,  # invalid
                        ),
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("time_feature_index" in e for e in result.errors)

    def test_time_varying_nonpositive_outputscale_limits_fail(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.OUTPUTSCALE,
                            time_feature_index=0,
                            outputscale_bias_limit=0.0,
                            outputscale_slope_limit=-1.0,
                        ),
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("outputscale_bias_limit" in e for e in result.errors)
        assert any("outputscale_slope_limit" in e for e in result.errors)

    def test_time_varying_on_composed_kernel_passes(self) -> None:
        """time_varying is supported on composed kernels in Tier 2."""
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        composition=CompositionType.ADDITIVE,
                        children=[
                            KernelSpec(kernel_type=KernelType.MATERN_52),
                            KernelSpec(kernel_type=KernelType.RBF),
                        ],
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.OUTPUTSCALE,
                            time_feature_index=0,
                        ),
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert result.is_valid, result.errors


class TestTimeVaryingRuntimeGuards:
    def test_outputscale_modulation_is_bounded_under_extreme_raw_parameters(self) -> None:
        torch = pytest.importorskip("torch")
        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_time_varying_spec(TimeVaryingTarget.OUTPUTSCALE)

        train_x = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
        train_y = torch.sin(2.0 * torch.pi * train_x)
        model = build_model_from_dsl(spec, train_x, train_y)

        # Drive raw params to extreme values; bounded mapping should still keep
        # effective modulation in a numerically stable range.
        with torch.no_grad():
            model.covar_module.raw_tv_bias.fill_(1_000.0)
            model.covar_module.raw_tv_slope.fill_(1_000.0)

            probe_x = torch.tensor([[0.0], [1.0]], dtype=torch.double)
            modulation = model.covar_module._modulation(probe_x)

        assert torch.isfinite(modulation).all()
        assert float(modulation.max()) < 10.0
        assert float(modulation.min()) > 0.0

    def test_outputscale_custom_limits_are_applied(self) -> None:
        torch = pytest.importorskip("torch")
        from gparchitect.builders.builder import build_model_from_dsl

        spec = _make_time_varying_spec(TimeVaryingTarget.OUTPUTSCALE)
        tv_spec = spec.feature_groups[0].kernel.time_varying
        assert tv_spec is not None
        tv_spec.outputscale_bias_limit = 1.5
        tv_spec.outputscale_slope_limit = 2.5

        train_x = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
        train_y = torch.sin(2.0 * torch.pi * train_x)
        model = build_model_from_dsl(spec, train_x, train_y)

        with torch.no_grad():
            model.covar_module.raw_tv_bias.fill_(1_000.0)
            model.covar_module.raw_tv_slope.fill_(1_000.0)
            bias, slope = model.covar_module._bounded_params()

        assert float(bias) == pytest.approx(1.5, abs=1e-6)
        assert float(slope) == pytest.approx(2.5, abs=1e-6)

    def test_outputscale_time_varying_wraps_composed_kernel(self) -> None:
        torch = pytest.importorskip("torch")
        from gparchitect.builders.builder import build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        composition=CompositionType.ADDITIVE,
                        children=[
                            KernelSpec(kernel_type=KernelType.MATERN_52),
                            KernelSpec(kernel_type=KernelType.RBF),
                        ],
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.OUTPUTSCALE,
                            time_feature_index=0,
                        ),
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )

        train_x = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
        train_y = torch.sin(2.0 * torch.pi * train_x)
        model = build_model_from_dsl(spec, train_x, train_y)

        assert hasattr(model.covar_module, "raw_tv_bias")
        assert hasattr(model.covar_module, "raw_tv_slope")
        assert hasattr(model.covar_module, "_modulation")

    def test_lengthscale_time_varying_wraps_composed_kernel(self) -> None:
        torch = pytest.importorskip("torch")
        from gparchitect.builders.builder import build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        composition=CompositionType.MULTIPLICATIVE,
                        children=[
                            KernelSpec(kernel_type=KernelType.MATERN_52),
                            KernelSpec(kernel_type=KernelType.RBF),
                        ],
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.LENGTHSCALE,
                            time_feature_index=0,
                        ),
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )

        train_x = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
        train_y = torch.sin(2.0 * torch.pi * train_x)
        model = build_model_from_dsl(spec, train_x, train_y)

        assert hasattr(model.covar_module, "raw_tv_bias")
        assert hasattr(model.covar_module, "raw_tv_slope")
        assert hasattr(model.covar_module, "_effective_lengthscale")

    def test_outputscale_time_varying_changes_with_time_for_composed_kernel(self) -> None:
        torch = pytest.importorskip("torch")
        from gparchitect.builders.builder import build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        composition=CompositionType.ADDITIVE,
                        children=[
                            KernelSpec(kernel_type=KernelType.MATERN_52),
                            KernelSpec(kernel_type=KernelType.RBF),
                        ],
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.OUTPUTSCALE,
                            time_feature_index=0,
                        ),
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )

        train_x = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
        train_y = torch.sin(2.0 * torch.pi * train_x)
        model = build_model_from_dsl(spec, train_x, train_y)

        with torch.no_grad():
            model.covar_module.raw_tv_bias.fill_(0.0)
            model.covar_module.raw_tv_slope.fill_(1.0)

            x_early = torch.tensor([[0.05]], dtype=torch.double)
            x_late = torch.tensor([[0.95]], dtype=torch.double)
            k_early = float(model.covar_module(x_early, x_early).to_dense().item())
            k_late = float(model.covar_module(x_late, x_late).to_dense().item())

        assert k_early != pytest.approx(k_late, rel=1e-3)
        assert k_late > k_early

    def test_lengthscale_time_varying_changes_with_time_for_composed_kernel(self) -> None:
        torch = pytest.importorskip("torch")
        from gparchitect.builders.builder import build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time",
                    feature_indices=[0],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        composition=CompositionType.MULTIPLICATIVE,
                        children=[
                            KernelSpec(kernel_type=KernelType.MATERN_52),
                            KernelSpec(kernel_type=KernelType.RBF),
                        ],
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.LENGTHSCALE,
                            time_feature_index=0,
                        ),
                    ),
                )
            ],
            input_dim=1,
            output_dim=1,
        )

        train_x = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
        train_y = torch.sin(2.0 * torch.pi * train_x)
        model = build_model_from_dsl(spec, train_x, train_y)

        with torch.no_grad():
            model.covar_module.raw_tv_bias.fill_(0.0)
            model.covar_module.raw_tv_slope.fill_(1.0)

            x_early = torch.tensor([[0.10], [0.15]], dtype=torch.double)
            x_late = torch.tensor([[0.80], [0.85]], dtype=torch.double)
            k_early = model.covar_module(x_early, x_early).to_dense()[0, 1].item()
            k_late = model.covar_module(x_late, x_late).to_dense()[0, 1].item()

        # With positive slope, late-time effective lengthscale is larger,
        # so equal temporal gaps produce stronger covariance at late time.
        assert float(k_late) > float(k_early)

    def test_time_varying_composed_kernel_supports_ard(self) -> None:
        torch = pytest.importorskip("torch")
        from gparchitect.builders.builder import build_model_from_dsl

        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="time_plus_aux",
                    feature_indices=[0, 1],
                    kernel=KernelSpec(
                        kernel_type=KernelType.MATERN_52,
                        composition=CompositionType.ADDITIVE,
                        children=[
                            KernelSpec(kernel_type=KernelType.MATERN_52, ard=True),
                            KernelSpec(kernel_type=KernelType.RBF, ard=True),
                        ],
                        time_varying=TimeVaryingSpec(
                            target=TimeVaryingTarget.OUTPUTSCALE,
                            time_feature_index=0,
                        ),
                    ),
                )
            ],
            input_dim=2,
            output_dim=1,
        )

        t = torch.linspace(0.0, 1.0, 24, dtype=torch.double)
        x_aux = torch.linspace(-1.0, 1.0, 24, dtype=torch.double)
        train_x = torch.stack([t, x_aux], dim=-1)
        train_y = (torch.sin(2.0 * torch.pi * t) + 0.2 * x_aux).unsqueeze(-1)

        model = build_model_from_dsl(spec, train_x, train_y)

        lengthscale_params = [param for name, param in model.named_parameters() if "lengthscale" in name]
        assert lengthscale_params
        assert any(param.shape[-1] == 2 for param in lengthscale_params if param.ndim >= 1)

        with torch.no_grad():
            k = model.covar_module(train_x[:3], train_x[:3]).to_dense()
        assert torch.isfinite(k).all()


# ---------------------------------------------------------------------------
# Validation tests — Tier 2: input warping
# ---------------------------------------------------------------------------


class TestInputWarpingValidation:
    def test_valid_input_warping_spec_passes(self) -> None:
        spec = _make_input_warping_spec()
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_input_warping_with_concentrations_passes(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            execution=ExecutionSpec(
                input_warping=InputWarpingSpec(
                    warp_type=WarpType.KUMARASWAMY,
                    time_feature_index=0,
                    concentration0=1.5,
                    concentration1=2.0,
                )
            ),
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert result.is_valid, result.errors

    def test_input_warping_index_out_of_range_fails(self) -> None:
        spec = _make_input_warping_spec(time_feature_index=99)
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("time_feature_index" in e for e in result.errors)

    def test_input_warping_zero_concentration_fails(self) -> None:
        spec = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            execution=ExecutionSpec(
                input_warping=InputWarpingSpec(
                    warp_type=WarpType.KUMARASWAMY,
                    time_feature_index=0,
                    concentration0=0.0,  # invalid
                )
            ),
            input_dim=1,
            output_dim=1,
        )
        result = validate_dsl(spec)
        assert not result.is_valid
        assert any("concentration0" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Translator tests — Tier 1
# ---------------------------------------------------------------------------


class TestTranslatorChangepoint:
    def test_changepoint_keyword_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a changepoint kernel", input_dim=1)
        assert isinstance(spec.feature_groups[0].kernel, ChangepointKernelSpec)

    def test_change_point_two_words_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a change point kernel", input_dim=1)
        assert isinstance(spec.feature_groups[0].kernel, ChangepointKernelSpec)

    def test_regime_shift_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Model a regime shift at time 0.5", input_dim=1)
        assert isinstance(spec.feature_groups[0].kernel, ChangepointKernelSpec)

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
        kernel = spec.feature_groups[0].kernel
        assert isinstance(kernel, ChangepointKernelSpec)
        assert kernel.kernel_before is not None
        assert kernel.kernel_after is not None

    def test_translated_changepoint_spec_is_valid(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Changepoint at 0.5 with steepness 1.0", input_dim=1)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors


class TestTranslatorRecencyFiltering:
    """Translator tests for Tier 1 recency filtering (formerly 'recency weighting')."""

    def test_sliding_window_keyword_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a sliding window of 0.4", input_dim=2)
        assert spec.execution.recency_filtering is not None
        assert spec.execution.recency_filtering.mode == RecencyFilteringMode.SLIDING_WINDOW

    def test_exponential_forgetting_keyword_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use exponential forgetting with rate 2.0", input_dim=2)
        assert spec.execution.recency_filtering is not None
        assert spec.execution.recency_filtering.mode == RecencyFilteringMode.EXPONENTIAL_DISCOUNT

    def test_discount_rate_extracted(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Apply exponential discount with rate 3.0", input_dim=2)
        rf = spec.execution.recency_filtering
        assert rf is not None
        assert rf.discount_rate == pytest.approx(3.0)

    def test_window_size_extracted(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Sliding window size 0.6", input_dim=2)
        rf = spec.execution.recency_filtering
        assert rf is not None
        assert rf.window_size == pytest.approx(0.6)

    def test_time_column_name_sets_feature_index(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl(
            "Use a sliding window",
            input_dim=3,
            input_feature_names=["x1", "time", "x3"],
        )
        rf = spec.execution.recency_filtering
        assert rf is not None
        assert rf.time_feature_index == 1

    def test_no_recency_keyword_gives_none(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a Matern52 kernel", input_dim=2)
        assert spec.execution.recency_filtering is None

    def test_translated_output_uses_recency_filtering_field(self) -> None:
        """The translator must populate recency_filtering, not recency_weighting."""
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a sliding window of 0.5", input_dim=2)
        data = json.loads(spec.model_dump_json())
        assert "recency_filtering" in data["execution"]
        assert "recency_weighting" not in data["execution"]


# ---------------------------------------------------------------------------
# Translator tests — Tier 2: time-varying hyperparameters
# ---------------------------------------------------------------------------


class TestTranslatorTimeVarying:
    def test_time_varying_outputscale_phrase_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use time-varying outputscale", input_dim=1)
        kernel = spec.feature_groups[0].kernel
        assert kernel.time_varying is not None
        assert kernel.time_varying.target == TimeVaryingTarget.OUTPUTSCALE

    def test_amplitude_changes_over_time_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("amplitude changes over time", input_dim=1)
        kernel = spec.feature_groups[0].kernel
        assert kernel.time_varying is not None
        assert kernel.time_varying.target == TimeVaryingTarget.OUTPUTSCALE

    def test_time_varying_lengthscale_phrase_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use time-varying lengthscale", input_dim=1)
        kernel = spec.feature_groups[0].kernel
        assert kernel.time_varying is not None
        assert kernel.time_varying.target == TimeVaryingTarget.LENGTHSCALE

    def test_lengthscale_changes_over_time_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("lengthscale changes over time", input_dim=1)
        kernel = spec.feature_groups[0].kernel
        assert kernel.time_varying is not None
        assert kernel.time_varying.target == TimeVaryingTarget.LENGTHSCALE

    def test_nonstationary_smoothness_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("non-stationary smoothness over time", input_dim=1)
        kernel = spec.feature_groups[0].kernel
        assert kernel.time_varying is not None
        assert kernel.time_varying.target == TimeVaryingTarget.LENGTHSCALE

    def test_no_time_varying_keyword_gives_none(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a Matern52 kernel", input_dim=1)
        assert spec.feature_groups[0].kernel.time_varying is None

    def test_time_feature_index_uses_time_column_name(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl(
            "time-varying outputscale",
            input_dim=2,
            input_feature_names=["x1", "timestamp"],
        )
        kernel = spec.feature_groups[0].kernel
        assert kernel.time_varying is not None
        assert kernel.time_varying.time_feature_index == 1

    def test_translated_time_varying_spec_is_valid(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("time-varying outputscale", input_dim=1)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors


# ---------------------------------------------------------------------------
# Translator tests — Tier 2: input warping
# ---------------------------------------------------------------------------


class TestTranslatorInputWarping:
    def test_warp_time_phrase_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("warp time", input_dim=1)
        assert spec.execution.input_warping is not None
        assert spec.execution.input_warping.warp_type == WarpType.KUMARASWAMY

    def test_warped_time_axis_phrase_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a warped time axis", input_dim=1)
        assert spec.execution.input_warping is not None

    def test_nonlinear_time_warp_phrase_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Apply nonlinear time warp", input_dim=1)
        assert spec.execution.input_warping is not None

    def test_input_warping_phrase_detected(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use input warping on time", input_dim=1)
        assert spec.execution.input_warping is not None

    def test_no_warp_keyword_gives_none(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("Use a Matern52 kernel", input_dim=1)
        assert spec.execution.input_warping is None

    def test_time_feature_index_uses_time_column(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl(
            "warp time",
            input_dim=2,
            input_feature_names=["timestamp", "x2"],
        )
        assert spec.execution.input_warping is not None
        assert spec.execution.input_warping.time_feature_index == 0

    def test_translated_input_warping_spec_is_valid(self) -> None:
        from gparchitect.translator.translator import translate_to_dsl

        spec = translate_to_dsl("warp time", input_dim=1)
        result = validate_dsl(spec)
        assert result.is_valid, result.errors


# ---------------------------------------------------------------------------
# Recency filtering functional tests — Tier 1
# ---------------------------------------------------------------------------


class TestRecencyFilteringFunctional:
    """Verify that the recency filtering function produces the expected subsets.

    Recency filtering removes old observations.  It is dataset truncation,
    not true likelihood-weighted GP inference.
    """

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
        from gparchitect.builders.recency import apply_recency_filtering

        train_X, train_Y = tensors
        rf = RecencyFilteringSpec(
            mode=RecencyFilteringMode.SLIDING_WINDOW,
            time_feature_index=0,
            window_size=0.3,
        )
        filtered_X, filtered_Y = apply_recency_filtering(train_X, train_Y, rf)
        # max time is 0.9; cutoff is 0.6; observations with t >= 0.6
        assert filtered_X[:, 0].min().item() >= 0.6 - 1e-9
        assert len(filtered_X) < len(train_X)

    def test_exponential_discount_keeps_recent(self, tensors):  # type: ignore[no-untyped-def]
        from gparchitect.builders.recency import apply_recency_filtering

        train_X, train_Y = tensors
        rf = RecencyFilteringSpec(
            mode=RecencyFilteringMode.EXPONENTIAL_DISCOUNT,
            time_feature_index=0,
            discount_rate=10.0,
            min_weight=0.1,
        )
        filtered_X, filtered_Y = apply_recency_filtering(train_X, train_Y, rf)
        # With rate=10 and min_weight=0.1, cutoff delta_t = -ln(0.1)/10 ≈ 0.23
        # so only the last few observations are kept
        assert len(filtered_X) < len(train_X)
        # The most recent observation must always be included
        assert float(filtered_X[:, 0].max().item()) == pytest.approx(float(train_X[:, 0].max().item()))

    def test_sliding_window_all_too_old_retains_one(self, tensors):  # type: ignore[no-untyped-def]
        from gparchitect.builders.recency import apply_recency_filtering

        train_X, train_Y = tensors
        rf = RecencyFilteringSpec(
            mode=RecencyFilteringMode.SLIDING_WINDOW,
            time_feature_index=0,
            window_size=1e-10,  # essentially zero window
        )
        filtered_X, filtered_Y = apply_recency_filtering(train_X, train_Y, rf)
        # Should retain at least 1 observation (the most recent)
        assert len(filtered_X) >= 1

    def test_out_of_range_feature_index_raises(self, tensors):  # type: ignore[no-untyped-def]
        from gparchitect.builders.recency import apply_recency_filtering

        train_X, train_Y = tensors
        rf = RecencyFilteringSpec(
            mode=RecencyFilteringMode.SLIDING_WINDOW,
            time_feature_index=5,  # only 1 feature column
            window_size=0.5,
        )
        with pytest.raises(ValueError, match="time_feature_index"):
            apply_recency_filtering(train_X, train_Y, rf)

    def test_filtering_does_not_weight_remaining_observations(self, tensors):  # type: ignore[no-untyped-def]
        """Filtering removes observations; the remaining ones have equal weight in the GP.

        This documents the key difference vs true likelihood weighting: no
        per-observation weight is passed to the GP kernel or likelihood.
        """
        from gparchitect.builders.recency import apply_recency_filtering

        train_X, train_Y = tensors
        rf = RecencyFilteringSpec(
            mode=RecencyFilteringMode.SLIDING_WINDOW,
            time_feature_index=0,
            window_size=0.5,
        )
        filtered_X, filtered_Y = apply_recency_filtering(train_X, train_Y, rf)
        # The result is just a smaller dataset — same dtype, no attached weights
        assert filtered_X.dtype == train_X.dtype
        assert filtered_Y.shape[-1] == train_Y.shape[-1]


# ---------------------------------------------------------------------------
# Changepoint builder tests — Tier 1
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
        with pytest.raises((ValueError, Exception, ValidationError)):
            GPSpec(
                model_class=ModelClass.SINGLE_TASK_GP,
                feature_groups=[
                    FeatureGroupSpec(
                        name="time",
                        feature_indices=[0],
                        kernel=ChangepointKernelSpec(
                            kernel_before=LeafKernelSpec(kernel_type=KernelType.MATERN_52),
                            kernel_after=None,  # type: ignore[arg-type]
                        ),
                    )
                ],
                input_dim=1,
                output_dim=1,
            )


# ---------------------------------------------------------------------------
# Time-varying hyperparameter builder tests — Tier 2
# ---------------------------------------------------------------------------


class TestTimeVaryingBuilder:
    """Verify that time-varying kernels can be constructed and produce
    observably different outputs for different time values."""

    @pytest.fixture()
    def small_dataset(self):  # type: ignore[no-untyped-def]
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        torch.manual_seed(42)
        train_X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
        train_Y = torch.sin(train_X * 3.14) + 0.1 * torch.randn(12, 1, dtype=torch.double)
        return train_X, train_Y

    def test_outputscale_model_constructed(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        from gparchitect.builders.builder import build_model_from_dsl

        train_X, train_Y = small_dataset
        spec = _make_time_varying_spec(TimeVaryingTarget.OUTPUTSCALE)
        model = build_model_from_dsl(spec, train_X, train_Y)
        assert model is not None

    def test_lengthscale_model_constructed(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        from gparchitect.builders.builder import build_model_from_dsl

        train_X, train_Y = small_dataset
        spec = _make_time_varying_spec(TimeVaryingTarget.LENGTHSCALE)
        model = build_model_from_dsl(spec, train_X, train_Y)
        assert model is not None

    def test_time_varying_model_has_tv_parameters(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        """The wrapped kernel must expose the learnable modulation parameters."""
        from gparchitect.builders.builder import build_model_from_dsl

        train_X, train_Y = small_dataset
        spec = _make_time_varying_spec(TimeVaryingTarget.OUTPUTSCALE)
        model = build_model_from_dsl(spec, train_X, train_Y)
        # Check that raw_tv_bias and raw_tv_slope are present in named parameters
        param_names = {name for name, _ in model.named_parameters()}
        assert any("raw_tv_bias" in n for n in param_names)
        assert any("raw_tv_slope" in n for n in param_names)

    def test_outputscale_modulation_produces_different_covariance(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        """With a non-zero slope, kernel outputs at early and late time differ."""
        import gpytorch
        import torch

        from gparchitect.builders.time_varying_kernel import build_time_varying_kernel

        train_X, train_Y = small_dataset

        base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        tv_kernel = build_time_varying_kernel(base_kernel, time_feature_index=0, target="outputscale")

        # Manually set a non-trivial slope so early vs late time differ
        with torch.no_grad():
            tv_kernel.raw_tv_slope.fill_(1.0)

        x_early = torch.tensor([[0.0]], dtype=torch.double)
        x_late = torch.tensor([[1.0]], dtype=torch.double)

        # Self-covariance at early vs late time should differ (different amplitude)
        k_early = tv_kernel(x_early, x_early).to_dense().item()
        k_late = tv_kernel(x_late, x_late).to_dense().item()
        assert k_early != pytest.approx(k_late, rel=1e-3)

    def test_time_varying_model_fits(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        """The time-varying model should be fittable with the standard fitting path."""
        from gparchitect.builders.builder import build_model_from_dsl
        from gparchitect.fitting.fitter import fit_and_validate

        train_X, train_Y = small_dataset
        spec = _make_time_varying_spec(TimeVaryingTarget.OUTPUTSCALE)
        model = build_model_from_dsl(spec, train_X, train_Y)
        result = fit_and_validate(model, train_X, train_Y)
        assert result.success, result.error_message


# ---------------------------------------------------------------------------
# Input warping builder tests — Tier 2
# ---------------------------------------------------------------------------


class TestInputWarpingBuilder:
    """Verify that input warping transforms are applied and affect the model."""

    @pytest.fixture()
    def small_dataset(self):  # type: ignore[no-untyped-def]
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        torch.manual_seed(7)
        train_X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
        train_Y = train_X**2 + 0.05 * torch.randn(12, 1, dtype=torch.double)
        return train_X, train_Y

    def test_input_warping_model_constructed(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        from gparchitect.builders.builder import build_model_from_dsl

        train_X, train_Y = small_dataset
        spec = _make_input_warping_spec()
        model = build_model_from_dsl(spec, train_X, train_Y)
        assert model is not None

    def test_input_warping_model_has_input_transform(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        """Model built with input warping must have an input_transform attached."""
        from gparchitect.builders.builder import build_model_from_dsl

        train_X, train_Y = small_dataset
        spec = _make_input_warping_spec()
        model = build_model_from_dsl(spec, train_X, train_Y)
        assert hasattr(model, "input_transform")
        assert model.input_transform is not None

    def test_input_warping_transform_is_warp(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        """The input transform must be a BoTorch Warp instance."""
        from botorch.models.transforms.input import Warp

        from gparchitect.builders.builder import build_model_from_dsl

        train_X, train_Y = small_dataset
        spec = _make_input_warping_spec()
        model = build_model_from_dsl(spec, train_X, train_Y)
        assert isinstance(model.input_transform, Warp)

    def test_warp_transform_keeps_extrapolation_resolution_when_scaled(self) -> None:
        """Scaled extrapolation inputs should not collapse to one warped coordinate."""
        import torch

        from gparchitect.builders.builder import build_model_from_dsl

        train_X = torch.linspace(0.0, 0.7, 12, dtype=torch.double).unsqueeze(-1)
        train_Y = train_X**2
        test_X = torch.linspace(0.71, 1.0, 8, dtype=torch.double).unsqueeze(-1)

        spec = _make_input_warping_spec()
        model = build_model_from_dsl(spec, train_X, train_Y)
        model.eval()

        with torch.no_grad():
            transformed_test = model.input_transform(test_X).squeeze(-1)

        assert transformed_test.min() < transformed_test.max()

    def test_warp_transform_changes_predictions(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        """A model with non-identity input warping should produce different predictions than without.

        We use concentration0=3.0, concentration1=0.5 to force a strongly non-linear warp
        that deviates from the identity at initialization.
        """
        import torch

        from gparchitect.builders.builder import build_model_from_dsl

        train_X, train_Y = small_dataset

        spec_plain = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            input_dim=1,
            output_dim=1,
        )
        # Use concentration0=3, concentration1=0.5 → strongly non-linear warp ≠ identity
        spec_warped = GPSpec(
            model_class=ModelClass.SINGLE_TASK_GP,
            feature_groups=[
                FeatureGroupSpec(
                    name="all",
                    feature_indices=[0],
                    kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
                )
            ],
            execution=ExecutionSpec(
                input_warping=InputWarpingSpec(
                    warp_type=WarpType.KUMARASWAMY,
                    time_feature_index=0,
                    concentration0=3.0,
                    concentration1=0.5,
                )
            ),
            input_dim=1,
            output_dim=1,
        )

        model_plain = build_model_from_dsl(spec_plain, train_X, train_Y)
        model_warped = build_model_from_dsl(spec_warped, train_X, train_Y)

        model_plain.eval()
        model_warped.eval()

        test_X = torch.linspace(0.1, 0.9, 5, dtype=torch.double).unsqueeze(-1)
        with torch.no_grad():
            pred_plain = model_plain(test_X).mean
            pred_warped = model_warped(test_X).mean

        # With a strongly non-identity warp, predictions should differ
        assert not torch.allclose(pred_plain, pred_warped, atol=1e-4)

    def test_input_warping_model_fits(self, small_dataset) -> None:  # type: ignore[no-untyped-def]
        """The warped model should be fittable with the standard fitting path."""
        from gparchitect.builders.builder import build_model_from_dsl
        from gparchitect.fitting.fitter import fit_and_validate

        train_X, train_Y = small_dataset
        spec = _make_input_warping_spec()
        model = build_model_from_dsl(spec, train_X, train_Y)
        result = fit_and_validate(model, train_X, train_Y)
        assert result.success, result.error_message
