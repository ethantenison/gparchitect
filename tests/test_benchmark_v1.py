"""
Tests for benchmark_v1 — dataset generators, registry, prompts, baselines, and metrics.

These tests are unit-level and do not require fitting real GP models (except for
the lightweight smoke tests that exercise the full runner on tiny datasets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Tier 2 synthetic dataset generators
# ---------------------------------------------------------------------------


class TestSyntheticDatasets:
    """Tests for benchmark_v1.datasets.synthetic generators."""

    def test_additive_dataset_shape(self) -> None:
        from benchmark_v1.datasets.synthetic import make_additive_dataset

        split = make_additive_dataset(seed=0, n_train=20, n_test=10)
        assert split.train.shape == (20, 6)  # 5 inputs + 1 output
        assert split.test.shape == (10, 6)
        assert split.output_column in split.train.columns

    def test_additive_dataset_input_columns(self) -> None:
        from benchmark_v1.datasets.synthetic import make_additive_dataset

        split = make_additive_dataset(seed=0)
        expected = {"x_smooth", "x_trend", "x_scale", "x_irrelevant_1", "x_irrelevant_2"}
        assert set(split.input_columns) == expected

    def test_periodic_decay_dataset_shape(self) -> None:
        from benchmark_v1.datasets.synthetic import make_periodic_decay_dataset

        split = make_periodic_decay_dataset(seed=0, n_train=30, n_test=15)
        assert len(split.train) == 30
        assert len(split.test) == 15

    def test_periodic_decay_columns(self) -> None:
        from benchmark_v1.datasets.synthetic import make_periodic_decay_dataset

        split = make_periodic_decay_dataset(seed=0)
        assert "seasonality_index" in split.input_columns
        assert "system_age" in split.input_columns

    def test_interaction_dataset_columns(self) -> None:
        from benchmark_v1.datasets.synthetic import make_interaction_dataset

        split = make_interaction_dataset(seed=0)
        assert "material_hardness" in split.input_columns
        assert "process_temperature" in split.input_columns
        assert "cooldown_rate" in split.input_columns

    def test_ard_stress_dataset_columns(self) -> None:
        from benchmark_v1.datasets.synthetic import make_ard_stress_dataset

        split = make_ard_stress_dataset(seed=0)
        assert "x_signal_1" in split.input_columns
        assert "x_signal_2" in split.input_columns
        assert "x_irrelevant" in split.input_columns
        assert len(split.input_columns) == 7

    def test_datasets_are_deterministic(self) -> None:
        from benchmark_v1.datasets.synthetic import make_additive_dataset

        s1 = make_additive_dataset(seed=7)
        s2 = make_additive_dataset(seed=7)
        pd.testing.assert_frame_equal(s1.train, s2.train)

    def test_different_seeds_differ(self) -> None:
        from benchmark_v1.datasets.synthetic import make_additive_dataset

        s1 = make_additive_dataset(seed=0)
        s2 = make_additive_dataset(seed=1)
        assert not s1.train["y"].equals(s2.train["y"])

    def test_noise_std_zero_gives_lower_variance(self) -> None:
        from benchmark_v1.datasets.synthetic import make_additive_dataset

        s_clean = make_additive_dataset(seed=0, noise_std=0.0)
        s_noisy = make_additive_dataset(seed=0, noise_std=0.3)
        # Noisy version should have higher variance in y
        assert s_noisy.train["y"].std() >= s_clean.train["y"].std() - 0.01

    def test_no_nan_in_generated_data(self) -> None:
        from benchmark_v1.datasets.synthetic import (
            make_additive_dataset,
            make_ard_stress_dataset,
            make_interaction_dataset,
            make_periodic_decay_dataset,
        )

        for gen in [
            make_additive_dataset,
            make_periodic_decay_dataset,
            make_interaction_dataset,
            make_ard_stress_dataset,
        ]:
            split = gen(seed=42)
            assert not split.train.isnull().any().any(), f"{split.name} train has NaN"
            assert not split.test.isnull().any().any(), f"{split.name} test has NaN"

    def test_list_synthetic_datasets(self) -> None:
        from benchmark_v1.datasets.synthetic import list_synthetic_datasets

        names = list_synthetic_datasets()
        assert "additive" in names
        assert "periodic_decay" in names
        assert "interaction" in names
        assert "ard_stress" in names


# ---------------------------------------------------------------------------
# Tier 1 BoTorch function adapters
# ---------------------------------------------------------------------------


class TestBoTorchDatasets:
    """Tests for benchmark_v1.datasets.botorch_functions generators."""

    def _skip_if_no_botorch(self) -> None:
        try:
            import botorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("botorch/torch not installed")

    def test_branin_shape(self) -> None:
        self._skip_if_no_botorch()
        from benchmark_v1.datasets.botorch_functions import make_branin_dataset

        split = make_branin_dataset(seed=0, n_train=20, n_test=10)
        assert split.train.shape == (20, 3)  # x0, x1, y
        assert split.test.shape == (10, 3)
        assert split.input_columns == ["x0", "x1"]

    def test_branin_deterministic(self) -> None:
        self._skip_if_no_botorch()
        from benchmark_v1.datasets.botorch_functions import make_branin_dataset

        s1 = make_branin_dataset(seed=5, n_train=20, n_test=10)
        s2 = make_branin_dataset(seed=5, n_train=20, n_test=10)
        pd.testing.assert_frame_equal(s1.train, s2.train)

    def test_hartmann6_shape(self) -> None:
        self._skip_if_no_botorch()
        from benchmark_v1.datasets.botorch_functions import make_hartmann6_dataset

        split = make_hartmann6_dataset(seed=0, n_train=20, n_test=10)
        assert len(split.input_columns) == 6
        assert split.train.shape == (20, 7)  # 6 inputs + y

    def test_rosenbrock_shape(self) -> None:
        self._skip_if_no_botorch()
        from benchmark_v1.datasets.botorch_functions import make_rosenbrock_dataset

        split = make_rosenbrock_dataset(seed=0, n_train=20, n_test=10)
        assert len(split.input_columns) == 4
        assert split.train.shape == (20, 5)

    def test_noiseless_has_no_stochastic_shift(self) -> None:
        self._skip_if_no_botorch()
        from benchmark_v1.datasets.botorch_functions import make_branin_dataset

        # Two noiseless runs with same seed should produce identical y
        s1 = make_branin_dataset(seed=0, n_train=20, n_test=5, noise_std=0.0)
        s2 = make_branin_dataset(seed=0, n_train=20, n_test=5, noise_std=0.0)
        np.testing.assert_array_equal(s1.train["y"].to_numpy(), s2.train["y"].to_numpy())

    def test_list_botorch_datasets(self) -> None:
        from benchmark_v1.datasets.botorch_functions import list_botorch_datasets

        names = list_botorch_datasets()
        assert "branin" in names
        assert "hartmann6" in names
        assert "rosenbrock" in names


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for benchmark_v1.registry."""

    def test_registry_has_seven_entries(self) -> None:
        from benchmark_v1.registry import REGISTRY

        assert len(REGISTRY) == 7

    def test_all_datasets_present(self) -> None:
        from benchmark_v1.registry import list_datasets

        names = list_datasets()
        for expected in [
            "additive",
            "periodic_decay",
            "interaction",
            "ard_stress",
            "branin",
            "hartmann6",
            "rosenbrock",
        ]:
            assert expected in names

    def test_tier_values(self) -> None:
        from benchmark_v1.registry import REGISTRY

        for entry in REGISTRY:
            assert entry.tier in (1, 2)

    def test_get_entry_by_name(self) -> None:
        from benchmark_v1.registry import get_entry

        entry = get_entry("additive")
        assert entry.dataset_name == "additive"
        assert entry.tier == 2

    def test_get_entry_missing_raises(self) -> None:
        from benchmark_v1.registry import get_entry

        with pytest.raises(KeyError, match="not found"):
            get_entry("nonexistent_dataset")

    def test_all_entries_have_three_prompt_variants(self) -> None:
        from benchmark_v1.registry import REGISTRY

        for entry in REGISTRY:
            assert "aligned" in entry.prompt_variants
            assert "vague" in entry.prompt_variants
            assert "misleading" in entry.prompt_variants

    def test_all_entries_have_baselines(self) -> None:
        from benchmark_v1.registry import REGISTRY

        for entry in REGISTRY:
            assert len(entry.baselines) >= 1

    def test_all_entries_have_seeds(self) -> None:
        from benchmark_v1.registry import REGISTRY

        for entry in REGISTRY:
            assert len(entry.seeds) >= 1

    def test_noise_levels_include_zero(self) -> None:
        from benchmark_v1.registry import REGISTRY

        for entry in REGISTRY:
            assert 0.0 in entry.noise_levels


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPromptVariants:
    """Tests for benchmark_v1.prompts.variants."""

    def test_all_datasets_have_prompts(self) -> None:
        from benchmark_v1.prompts.variants import ALL_PROMPT_VARIANTS

        expected = ["additive", "periodic_decay", "interaction", "ard_stress", "branin", "hartmann6", "rosenbrock"]
        for name in expected:
            assert name in ALL_PROMPT_VARIANTS

    def test_get_prompts_returns_variants(self) -> None:
        from benchmark_v1.prompts.variants import get_prompts

        pv = get_prompts("additive")
        assert pv.aligned
        assert pv.vague
        assert pv.misleading

    def test_as_dict_has_three_keys(self) -> None:
        from benchmark_v1.prompts.variants import get_prompts

        d = get_prompts("branin").as_dict()
        assert set(d.keys()) == {"aligned", "vague", "misleading"}

    def test_all_prompts_nonempty(self) -> None:
        from benchmark_v1.prompts.variants import ALL_PROMPT_VARIANTS

        for name, pv in ALL_PROMPT_VARIANTS.items():
            assert pv.aligned.strip(), f"{name} aligned prompt is empty"
            assert pv.vague.strip(), f"{name} vague prompt is empty"
            assert pv.misleading.strip(), f"{name} misleading prompt is empty"

    def test_get_prompts_missing_raises(self) -> None:
        from benchmark_v1.prompts.variants import get_prompts

        with pytest.raises(KeyError):
            get_prompts("nonexistent_dataset")

    def test_aligned_not_equal_to_vague(self) -> None:
        from benchmark_v1.prompts.variants import get_prompts

        for name in ["additive", "periodic_decay", "interaction", "ard_stress"]:
            pv = get_prompts(name)
            assert pv.aligned != pv.vague, f"{name}: aligned and vague are identical"


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class TestBaselines:
    """Tests for benchmark_v1.baselines.configs."""

    def test_default_singletask_spec_shape(self) -> None:
        from benchmark_v1.baselines.configs import make_default_singletask_spec
        from gparchitect.dsl.schema import ModelClass

        spec = make_default_singletask_spec(input_dim=5)
        assert spec.model_class == ModelClass.SINGLE_TASK_GP
        assert spec.input_dim == 5
        assert spec.output_dim == 1
        assert not spec.feature_groups[0].kernel.ard

    def test_matern52_ard_spec_has_ard(self) -> None:
        from benchmark_v1.baselines.configs import make_matern52_ard_spec

        spec = make_matern52_ard_spec(input_dim=3)
        assert spec.feature_groups[0].kernel.ard is True

    def test_baseline_feature_indices_cover_all_inputs(self) -> None:
        from benchmark_v1.baselines.configs import make_default_singletask_spec, make_matern52_ard_spec

        for factory in [make_default_singletask_spec, make_matern52_ard_spec]:
            spec = factory(input_dim=7)
            indices = spec.feature_groups[0].feature_indices
            assert indices == list(range(7))

    def test_list_baselines(self) -> None:
        from benchmark_v1.baselines.configs import list_baselines

        names = list_baselines()
        assert "default_singletask" in names
        assert "matern52_ard" in names


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestMetricHelpers:
    """Tests for benchmark_v1.run_benchmark._compute_metrics."""

    def test_perfect_predictions_give_zero_rmse(self) -> None:
        from benchmark_v1.run_benchmark import _compute_metrics

        y = np.array([1.0, 2.0, 3.0])
        metrics = _compute_metrics(y_mean=y, y_std=np.ones(3) * 0.1, y_true=y)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_coverage_100_when_wide_intervals(self) -> None:
        from benchmark_v1.run_benchmark import _compute_metrics

        y = np.array([1.0, 2.0, 3.0])
        metrics = _compute_metrics(y_mean=y, y_std=np.ones(3) * 1000.0, y_true=y + 1.0)
        assert metrics["coverage_95"] == pytest.approx(1.0)

    def test_coverage_zero_when_no_intervals_contain_truth(self) -> None:
        from benchmark_v1.run_benchmark import _compute_metrics

        y_mean = np.zeros(100)
        y_std = np.ones(100) * 0.001
        y_true = np.ones(100) * 100.0  # far outside
        metrics = _compute_metrics(y_mean=y_mean, y_std=y_std, y_true=y_true)
        assert metrics["coverage_95"] == pytest.approx(0.0)

    def test_interval_width_scales_with_std(self) -> None:
        from benchmark_v1.run_benchmark import _compute_metrics

        y = np.zeros(10)
        m1 = _compute_metrics(y_mean=y, y_std=np.ones(10) * 1.0, y_true=y)
        m2 = _compute_metrics(y_mean=y, y_std=np.ones(10) * 2.0, y_true=y)
        assert m1["interval_width_95"] is not None
        assert m2["interval_width_95"] is not None
        assert m2["interval_width_95"] > m1["interval_width_95"]

    def test_nll_lower_for_better_calibrated_predictions(self) -> None:
        from benchmark_v1.run_benchmark import _compute_metrics

        y = np.array([0.0, 0.0, 0.0])
        # Well-calibrated: residuals match std
        m_good = _compute_metrics(y_mean=y, y_std=np.ones(3) * 0.1, y_true=np.ones(3) * 0.1)
        # Overconfident: tiny std but large residuals
        m_bad = _compute_metrics(y_mean=y, y_std=np.ones(3) * 0.001, y_true=np.ones(3) * 1.0)
        assert m_good["nll"] is not None
        assert m_bad["nll"] is not None
        assert m_bad["nll"] > m_good["nll"]


# ---------------------------------------------------------------------------
# Run plan builder
# ---------------------------------------------------------------------------


class TestRunPlanBuilder:
    """Tests for benchmark_v1.run_benchmark.build_run_plan."""

    def test_unfiltered_plan_covers_all_entries(self) -> None:
        from benchmark_v1.registry import REGISTRY
        from benchmark_v1.run_benchmark import build_run_plan

        plan = build_run_plan(None, None, None, None)
        # Should have entries for all datasets × seeds × noise levels × (variants + baselines)
        assert len(plan) > 0
        dataset_names = {item["entry"].dataset_name for item in plan}
        expected = {e.dataset_name for e in REGISTRY}
        assert dataset_names == expected

    def test_dataset_filter(self) -> None:
        from benchmark_v1.run_benchmark import build_run_plan

        plan = build_run_plan(["additive"], None, None, None)
        assert all(item["entry"].dataset_name == "additive" for item in plan)

    def test_seed_filter(self) -> None:
        from benchmark_v1.run_benchmark import build_run_plan

        plan = build_run_plan(None, None, [0], None)
        assert all(item["seed"] == 0 for item in plan)

    def test_noise_filter(self) -> None:
        from benchmark_v1.run_benchmark import build_run_plan

        plan = build_run_plan(None, None, None, [0.0])
        assert all(abs(item["noise_std"] - 0.0) < 1e-9 for item in plan)

    def test_tier_filter(self) -> None:
        from benchmark_v1.run_benchmark import build_run_plan

        plan_t2 = build_run_plan(None, 2, None, None)
        assert all(item["entry"].tier == 2 for item in plan_t2)

    def test_plan_includes_both_model_types(self) -> None:
        from benchmark_v1.run_benchmark import build_run_plan

        plan = build_run_plan(["additive"], None, [0], [0.0])
        types = {item["model_type"] for item in plan}
        assert "gparchitect" in types
        assert "baseline" in types


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    """Tests for benchmark_v1.analyze_results.aggregate_metrics."""

    def _make_records_df(self) -> pd.DataFrame:
        from benchmark_v1.run_benchmark import RunRecord

        records = [
            RunRecord(
                dataset_name="additive",
                tier=2,
                seed=s,
                noise_std=0.05,
                model_type="baseline",
                model_id="matern52_ard",
                fit_success=True,
                retry_count=0,
                rmse=0.1 + 0.01 * s,
                mae=0.08,
                nll=-1.0,
                coverage_95=0.93,
                interval_width_95=0.5,
                wall_time_s=1.0,
            )
            for s in range(3)
        ]
        return pd.DataFrame([r.to_dict() for r in records])

    def test_aggregates_over_seeds(self) -> None:
        from benchmark_v1.analyze_results import aggregate_metrics

        df = self._make_records_df()
        agg = aggregate_metrics(df)
        assert len(agg) == 1  # one group
        assert agg.iloc[0]["n_runs"] == 3

    def test_success_rate_all_success(self) -> None:
        from benchmark_v1.analyze_results import aggregate_metrics

        df = self._make_records_df()
        agg = aggregate_metrics(df)
        assert agg.iloc[0]["success_rate"] == pytest.approx(1.0)

    def test_none_metrics_excluded_from_mean(self) -> None:
        from benchmark_v1.analyze_results import aggregate_metrics
        from benchmark_v1.run_benchmark import RunRecord

        records = [
            RunRecord(
                dataset_name="test_ds",
                tier=2,
                seed=0,
                noise_std=0.0,
                model_type="baseline",
                model_id="x",
                fit_success=False,
                retry_count=0,
                rmse=None,
                mae=None,
                nll=None,
                coverage_95=None,
                interval_width_95=None,
                wall_time_s=0.1,
            ),
            RunRecord(
                dataset_name="test_ds",
                tier=2,
                seed=1,
                noise_std=0.0,
                model_type="baseline",
                model_id="x",
                fit_success=True,
                retry_count=0,
                rmse=0.2,
                mae=0.15,
                nll=-0.5,
                coverage_95=0.90,
                interval_width_95=0.4,
                wall_time_s=0.2,
            ),
        ]
        df = pd.DataFrame([r.to_dict() for r in records])
        agg = aggregate_metrics(df)
        assert agg.iloc[0]["rmse_mean"] == pytest.approx(0.2)
        assert agg.iloc[0]["success_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RunRecord serialization
# ---------------------------------------------------------------------------


class TestRunRecord:
    """Tests for benchmark_v1.run_benchmark.RunRecord."""

    def test_to_dict_roundtrip(self) -> None:
        from benchmark_v1.run_benchmark import RunRecord

        record = RunRecord(
            dataset_name="additive",
            tier=2,
            seed=0,
            noise_std=0.05,
            model_type="gparchitect",
            model_id="aligned",
            fit_success=True,
            retry_count=1,
            rmse=0.12,
            mae=0.09,
            nll=-1.5,
            coverage_95=0.94,
            interval_width_95=0.6,
            wall_time_s=3.5,
        )
        d = record.to_dict()
        assert d["dataset_name"] == "additive"
        assert d["rmse"] == pytest.approx(0.12)
        assert d["fit_success"] is True

    def test_failed_record_has_none_metrics(self) -> None:
        from benchmark_v1.run_benchmark import RunRecord

        record = RunRecord(
            dataset_name="branin",
            tier=1,
            seed=2,
            noise_std=0.15,
            model_type="baseline",
            model_id="default_singletask",
            fit_success=False,
            retry_count=0,
            rmse=None,
            mae=None,
            nll=None,
            coverage_95=None,
            interval_width_95=None,
            wall_time_s=0.5,
            error_message="Cholesky failed",
        )
        assert record.rmse is None
        assert record.fit_success is False
        assert "Cholesky" in record.error_message
