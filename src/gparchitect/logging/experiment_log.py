"""
Experiment logging for GPArchitect.

Purpose:
    Records all stages of the GPArchitect pipeline — original instructions, generated
    DSL, validation results, model construction, fitting, DSL revisions, and final outcomes.

Role in pipeline:
    Logging is a cross-cutting concern used by every stage of the pipeline.

Inputs:
    Structured dictionaries and dataclasses from each pipeline stage.

Outputs:
    ExperimentLog — a JSON-serializable dataclass capturing the full experiment history.

Non-obvious design decisions:
    - All log entries are JSON-serializable to enable export and inspection.
    - The ExperimentLog is append-only; entries are never modified after being added.
    - Timestamps are ISO 8601 strings to avoid timezone ambiguity.
    - GPSpec objects are serialised via Pydantic's model_dump(mode="json").

What this module does NOT do:
    - It does not write to disk by default — callers must explicitly call to_json().
    - It does not handle log rotation or file management.
    - It does not configure the Python logging module (that is done in the main API/CLI).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class AttemptRecord:
    """Record of a single pipeline attempt (including any revision).

    Attributes:
        attempt_number: Zero-indexed attempt counter.
        spec_snapshot: JSON-serializable snapshot of the GPSpec used in this attempt.
        validation_errors: List of validation errors, empty if valid.
        validation_warnings: List of validation warnings.
        fit_success: Whether fitting succeeded.
        mll_value: Marginal log-likelihood value, or None.
        error_message: Error message on failure, otherwise empty.
        revision_strategy: Strategy applied to generate this spec, or None for attempt 0.
        revision_rationale: Rationale for the revision, or None for attempt 0.
        timestamp: ISO 8601 UTC timestamp.
    """

    attempt_number: int
    spec_snapshot: dict[str, Any]
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    fit_success: bool = False
    mll_value: float | None = None
    error_message: str = ""
    revision_strategy: str | None = None
    revision_rationale: str | None = None
    timestamp: str = field(default_factory=_now_iso)


@dataclass
class ExperimentLog:
    """Full experiment log for a single GPArchitect run.

    Attributes:
        instruction: The original natural-language instruction.
        input_dim: Number of input features.
        output_dim: Number of output features.
        input_scaling_applied: Whether continuous inputs were min-max scaled.
        input_feature_ranges: Original min/max ranges for each continuous input feature.
        attempts: List of AttemptRecord objects, one per pipeline attempt.
        final_success: Whether any attempt succeeded.
        created_at: ISO 8601 UTC timestamp for when the run started.
    """

    instruction: str
    input_dim: int
    output_dim: int
    input_scaling_applied: bool = False
    input_feature_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    attempts: list[AttemptRecord] = field(default_factory=list)
    final_success: bool = False
    created_at: str = field(default_factory=_now_iso)

    def add_attempt(self, record: AttemptRecord) -> None:
        """Append a new attempt record to the log.

        Args:
            record: The AttemptRecord to add.
        """
        self.attempts.append(record)
        logger.debug("Logged attempt %d: success=%s", record.attempt_number, record.fit_success)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full experiment log to a JSON-compatible dict.

        Returns:
            A dictionary representation of the log.
        """
        return {
            "instruction": self.instruction,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scaling_applied": self.input_scaling_applied,
            "input_feature_ranges": self.input_feature_ranges,
            "final_success": self.final_success,
            "created_at": self.created_at,
            "attempts": [
                {
                    "attempt_number": rec.attempt_number,
                    "spec_snapshot": rec.spec_snapshot,
                    "validation_errors": rec.validation_errors,
                    "validation_warnings": rec.validation_warnings,
                    "fit_success": rec.fit_success,
                    "mll_value": rec.mll_value,
                    "error_message": rec.error_message,
                    "revision_strategy": rec.revision_strategy,
                    "revision_rationale": rec.revision_rationale,
                    "timestamp": rec.timestamp,
                }
                for rec in self.attempts
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the experiment log to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            A formatted JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)


def summarize_attempts(experiment_log: ExperimentLog) -> str:
    """Produce a human-readable summary of all pipeline attempts.

    Args:
        experiment_log: The ExperimentLog to summarise.

    Returns:
        A multi-line string summarising each attempt and the final outcome.
    """
    lines: list[str] = [
        "GPArchitect Experiment Summary",
        f"  Instruction : {experiment_log.instruction[:100]}",
        f"  Input dim   : {experiment_log.input_dim}",
        f"  Output dim  : {experiment_log.output_dim}",
        f"  Input scaled: {experiment_log.input_scaling_applied}",
        f"  Total attempts: {len(experiment_log.attempts)}",
        f"  Final success : {experiment_log.final_success}",
        "",
    ]

    if experiment_log.input_feature_ranges:
        lines.append(f"  Input ranges: {experiment_log.input_feature_ranges}")
        lines.append("")

    for rec in experiment_log.attempts:
        status = "SUCCESS" if rec.fit_success else "FAILED"
        lines.append(f"  Attempt {rec.attempt_number}: [{status}]")
        if rec.revision_strategy:
            lines.append(f"    Strategy : {rec.revision_strategy}")
            lines.append(f"    Rationale: {rec.revision_rationale}")
        if rec.validation_errors:
            lines.append(f"    Validation errors: {rec.validation_errors}")
        if rec.error_message:
            lines.append(f"    Error: {rec.error_message[:100]}")
        if rec.mll_value is not None:
            lines.append(f"    MLL: {rec.mll_value:.4f}")

    return "\n".join(lines)
