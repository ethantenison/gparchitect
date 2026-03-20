"""
Tests for the GPArchitect public API module.

These tests validate the pipeline orchestration logic using mocked components.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestRunGPArchitectMocked:
    """Integration-style tests for run_gparchitect using mocked fitting."""

    def _skip_if_no_pandas_torch(self) -> None:
        try:
            import pandas  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("pandas or torch not installed")

    @patch("gparchitect.api.fit_and_validate")
    @patch("gparchitect.api.build_model_from_dsl")
    def test_successful_run_returns_model_and_log(self, mock_build, mock_fit) -> None:
        self._skip_if_no_pandas_torch()
        import pandas as pd

        from gparchitect.api import run_gparchitect
        from gparchitect.fitting.fitter import FitResult

        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [0.1, 0.2, 0.3]})
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        mock_fit.return_value = FitResult(success=True, model=mock_model, mll_value=-1.0)

        model, log = run_gparchitect(
            dataframe=df,
            instruction="Use Matern52",
            input_columns=["x1", "x2"],
            output_columns=["y"],
        )

        assert model is mock_model
        assert log.final_success is True
        assert len(log.attempts) == 1
        assert log.input_scaling_applied is True
        assert log.input_feature_ranges == {"x1": (1.0, 3.0), "x2": (4.0, 6.0)}

    @patch("gparchitect.api.fit_and_validate")
    @patch("gparchitect.api.build_model_from_dsl")
    def test_failed_run_returns_none_and_log(self, mock_build, mock_fit) -> None:
        self._skip_if_no_pandas_torch()
        import pandas as pd

        from gparchitect.api import run_gparchitect
        from gparchitect.fitting.fitter import FitResult

        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [0.1, 0.2, 0.3]})
        mock_build.return_value = MagicMock()
        mock_fit.return_value = FitResult(
            success=False, model=None, mll_value=None, error_message="Cholesky failed"
        )

        model, log = run_gparchitect(
            dataframe=df,
            instruction="Use RBF kernel",
            input_columns=["x1", "x2"],
            output_columns=["y"],
            max_retries=0,
        )

        assert model is None
        assert log.final_success is False

    @patch("gparchitect.api.fit_and_validate")
    @patch("gparchitect.api.build_model_from_dsl")
    def test_recovery_retry_on_failure(self, mock_build, mock_fit) -> None:
        self._skip_if_no_pandas_torch()
        import pandas as pd

        from gparchitect.api import run_gparchitect
        from gparchitect.fitting.fitter import FitResult

        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [0.1, 0.2, 0.3]})
        mock_model = MagicMock()
        mock_build.return_value = mock_model

        # Fail on first attempt, succeed on second
        mock_fit.side_effect = [
            FitResult(success=False, model=None, mll_value=None, error_message="First fail"),
            FitResult(success=True, model=mock_model, mll_value=-0.5),
        ]

        model, log = run_gparchitect(
            dataframe=df,
            instruction="Use Matern52",
            input_columns=["x1", "x2"],
            output_columns=["y"],
            max_retries=3,
        )

        assert model is mock_model
        assert log.final_success is True
        assert len(log.attempts) >= 2

    def test_run_with_missing_columns_raises(self) -> None:
        self._skip_if_no_pandas_torch()
        import pandas as pd

        from gparchitect.api import run_gparchitect

        df = pd.DataFrame({"x1": [1.0, 2.0], "y": [0.1, 0.2]})
        with pytest.raises(ValueError, match="not found"):
            run_gparchitect(
                dataframe=df,
                instruction="Use Matern52",
                input_columns=["x1", "x_missing"],
                output_columns=["y"],
            )


class TestPublicAPIImports:
    """Verify that all required public API functions are importable."""

    def test_translate_to_dsl_importable(self) -> None:
        from gparchitect.api import translate_to_dsl  # noqa: F401

    def test_validate_dsl_importable(self) -> None:
        from gparchitect.api import validate_dsl  # noqa: F401

    def test_build_model_from_dsl_importable(self) -> None:
        from gparchitect.api import build_model_from_dsl  # noqa: F401

    def test_fit_and_validate_importable(self) -> None:
        from gparchitect.api import fit_and_validate  # noqa: F401

    def test_revise_dsl_importable(self) -> None:
        from gparchitect.api import revise_dsl  # noqa: F401

    def test_run_gparchitect_importable(self) -> None:
        from gparchitect.api import run_gparchitect  # noqa: F401

    def test_summarize_attempts_importable(self) -> None:
        from gparchitect.api import summarize_attempts  # noqa: F401

    def test_prepare_data_importable(self) -> None:
        from gparchitect.api import prepare_data  # noqa: F401
