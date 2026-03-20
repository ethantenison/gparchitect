"""Tests for the GPArchitect experiment logging module."""

from __future__ import annotations

import json

import pytest

from gparchitect.dsl.schema import FeatureGroupSpec, GPSpec, KernelSpec, KernelType, ModelClass
from gparchitect.logging.experiment_log import AttemptRecord, ExperimentLog, summarize_attempts


def _make_attempt(attempt_number: int = 0, success: bool = True) -> AttemptRecord:
    spec = GPSpec(
        model_class=ModelClass.SINGLE_TASK_GP,
        feature_groups=[
            FeatureGroupSpec(
                name="all",
                feature_indices=[0, 1],
                kernel=KernelSpec(kernel_type=KernelType.MATERN_52),
            )
        ],
        input_dim=2,
        output_dim=1,
    )
    return AttemptRecord(
        attempt_number=attempt_number,
        spec_snapshot=spec.model_dump(mode="json"),
        fit_success=success,
        mll_value=-1.23 if success else None,
        error_message="" if success else "Something failed",
    )


class TestAttemptRecord:
    def test_default_values(self) -> None:
        record = AttemptRecord(
            attempt_number=0,
            spec_snapshot={},
        )
        assert record.fit_success is False
        assert record.mll_value is None
        assert record.error_message == ""
        assert record.revision_strategy is None
        assert record.revision_rationale is None

    def test_timestamp_is_set(self) -> None:
        record = AttemptRecord(attempt_number=0, spec_snapshot={})
        assert record.timestamp != ""

    def test_spec_snapshot_stored(self) -> None:
        record = _make_attempt(0)
        assert record.spec_snapshot["model_class"] == "SingleTaskGP"


class TestExperimentLog:
    def test_initial_state(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=3, output_dim=1)
        assert log.attempts == []
        assert log.final_success is False
        assert log.instruction == "Test"

    def test_add_attempt_appends(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=3, output_dim=1)
        record = _make_attempt(0)
        log.add_attempt(record)
        assert len(log.attempts) == 1
        assert log.attempts[0] is record

    def test_add_multiple_attempts(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        for i in range(3):
            log.add_attempt(_make_attempt(i))
        assert len(log.attempts) == 3

    def test_to_dict_structure(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        log.add_attempt(_make_attempt(0, success=True))
        data = log.to_dict()
        assert data["instruction"] == "Test"
        assert data["input_dim"] == 2
        assert data["final_success"] is False  # not set manually
        assert len(data["attempts"]) == 1

    def test_to_json_is_valid_json(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        log.add_attempt(_make_attempt(0))
        json_str = log.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)
        assert "attempts" in data

    def test_to_dict_is_json_serializable(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        log.final_success = True
        log.add_attempt(_make_attempt(0, success=True))
        data = log.to_dict()
        json.dumps(data)  # Should not raise

    def test_final_success_can_be_set(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        log.final_success = True
        assert log.final_success is True

    def test_created_at_is_set(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        assert log.created_at != ""


class TestSummarizeAttempts:
    def test_summary_contains_instruction(self) -> None:
        log = ExperimentLog(instruction="Use Matern52", input_dim=2, output_dim=1)
        log.add_attempt(_make_attempt(0, success=True))
        log.final_success = True
        summary = summarize_attempts(log)
        assert "Matern52" in summary

    def test_summary_contains_attempt_info(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        log.add_attempt(_make_attempt(0, success=False))
        log.add_attempt(_make_attempt(1, success=True))
        summary = summarize_attempts(log)
        assert "Attempt 0" in summary
        assert "Attempt 1" in summary

    def test_summary_shows_success(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        log.add_attempt(_make_attempt(0, success=True))
        log.final_success = True
        summary = summarize_attempts(log)
        assert "SUCCESS" in summary

    def test_summary_shows_failure(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        log.add_attempt(_make_attempt(0, success=False))
        summary = summarize_attempts(log)
        assert "FAILED" in summary

    def test_summary_shows_mll_on_success(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        log.add_attempt(_make_attempt(0, success=True))
        summary = summarize_attempts(log)
        assert "MLL" in summary

    def test_summary_shows_revision_strategy(self) -> None:
        log = ExperimentLog(instruction="Test", input_dim=2, output_dim=1)
        record = _make_attempt(0, success=False)
        record.revision_strategy = "disable_ard"
        record.revision_rationale = "Disabled ARD to reduce complexity."
        log.add_attempt(record)
        summary = summarize_attempts(log)
        assert "disable_ard" in summary
