"""
Public API for GPArchitect.

Purpose:
    Provides the user-facing entry points for the full GPArchitect pipeline.
    Each function corresponds to a single pipeline stage, plus a top-level
    orchestration function (run_gparchitect) that drives the complete pipeline.

Role in pipeline:
    This module is the outermost layer; all other modules are implementation details.

Inputs / Outputs:
    See individual function docstrings below.

Non-obvious design decisions:
    - run_gparchitect encapsulates the retry/recovery loop, keeping callers simple.
    - All public functions accept and return plain Python objects or typed dataclasses
      so that callers do not need to import internal types.
    - max_retries limits the recovery loop to a fixed number of revisions.

What this module does NOT do:
    - It does not accept file paths or raw CSV input — see cli.py for that.
    - It does not configure logging — callers should set up logging themselves.
"""

from __future__ import annotations

import logging

from gparchitect.builders.builder import build_model_from_dsl
from gparchitect.builders.data import DataBundle, prepare_data
from gparchitect.dsl.schema import GPSpec
from gparchitect.fitting.fitter import FitResult, fit_and_validate
from gparchitect.logging.experiment_log import (
    AttemptRecord,
    ExperimentLog,
    summarize_attempts,
)
from gparchitect.revision.revision import RevisionResult, revise_dsl
from gparchitect.translator.translator import translate_to_dsl
from gparchitect.validation.validator import ValidationResult, validate_dsl

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "translate_to_dsl",
    "validate_dsl",
    "build_model_from_dsl",
    "fit_and_validate",
    "revise_dsl",
    "run_gparchitect",
    "summarize_attempts",
    "prepare_data",
]


def run_gparchitect(
    dataframe: pd.DataFrame,
    instruction: str,
    input_columns: list[str],
    output_columns: list[str],
    task_column: str | None = None,
    max_retries: int = 5,
) -> tuple[object | None, ExperimentLog]:
    """Run the complete GPArchitect pipeline.

    Translates the natural-language instruction into a GPSpec, validates it,
    builds and fits a BoTorch model, and retries with DSL revisions on failure.

    Args:
        dataframe: pandas DataFrame with input/output data.
        instruction: Natural-language description of the desired GP architecture.
        input_columns: List of DataFrame column names to use as GP inputs.
        output_columns: List of DataFrame column names to use as GP outputs.
        task_column: Optional column name for the task indicator (MultiTaskGP only).
        max_retries: Maximum number of DSL revision attempts before giving up.

    Returns:
        A tuple of (model_or_None, ExperimentLog). The model is None if all attempts fail.
    """
    data_bundle: DataBundle = prepare_data(dataframe, input_columns, output_columns, task_column)

    spec: GPSpec = translate_to_dsl(
        instruction=instruction,
        input_dim=data_bundle.input_dim,
        output_dim=data_bundle.output_dim,
        task_feature_index=data_bundle.task_feature_index,
        input_feature_names=input_columns,
    )

    experiment_log = ExperimentLog(
        instruction=instruction,
        input_dim=data_bundle.input_dim,
        output_dim=data_bundle.output_dim,
        input_scaling_applied=data_bundle.input_scaling_applied,
        input_feature_ranges=data_bundle.input_feature_ranges,
    )

    for attempt in range(max_retries + 1):
        validation_result: ValidationResult = validate_dsl(spec)

        if not validation_result.is_valid:
            attempt_record = AttemptRecord(
                attempt_number=attempt,
                spec_snapshot=spec.model_dump(mode="json"),
                validation_errors=validation_result.errors,
                validation_warnings=validation_result.warnings,
                fit_success=False,
                error_message=f"Validation failed: {validation_result.errors}",
            )
            experiment_log.add_attempt(attempt_record)

            revision: RevisionResult | None = revise_dsl(spec, str(validation_result.errors), attempt)
            if revision is None:
                logger.error("All revision strategies exhausted; giving up.")
                break
            attempt_record.revision_strategy = revision.strategy
            attempt_record.revision_rationale = revision.rationale
            spec = revision.revised_spec
            continue

        try:
            model = build_model_from_dsl(spec, data_bundle.train_X, data_bundle.train_Y)
            fit_result: FitResult = fit_and_validate(model, data_bundle.train_X, data_bundle.train_Y)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            fit_result = FitResult(success=False, model=None, mll_value=None, error_message=error_message)

        attempt_record = AttemptRecord(
            attempt_number=attempt,
            spec_snapshot=spec.model_dump(mode="json"),
            validation_errors=validation_result.errors,
            validation_warnings=validation_result.warnings,
            fit_success=fit_result.success,
            mll_value=fit_result.mll_value,
            error_message=fit_result.error_message,
        )
        experiment_log.add_attempt(attempt_record)

        if fit_result.success:
            experiment_log.final_success = True
            logger.info("GPArchitect pipeline succeeded on attempt %d.", attempt)
            return fit_result.model, experiment_log

        revision = revise_dsl(spec, fit_result.error_message, attempt)
        if revision is None:
            logger.error("All revision strategies exhausted after %d attempts.", attempt + 1)
            break

        attempt_record.revision_strategy = revision.strategy
        attempt_record.revision_rationale = revision.rationale
        spec = revision.revised_spec

    logger.error("GPArchitect pipeline failed after %d attempt(s).", len(experiment_log.attempts))
    return None, experiment_log
