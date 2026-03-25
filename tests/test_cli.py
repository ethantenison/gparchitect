"""Tests for the GPArchitect CLI planning commands."""

from __future__ import annotations

import json

from click.testing import CliRunner
from gparchitect.cli import main
from gparchitect.planning import run_prior_knowledge


class TestPlanningCLI:
    """Validate the planning CLI group and outputs."""

    def test_plan_prior_json_output(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "prior",
                "--text",
                "Temperature and pressure interact. Noise increases near the upper pressure limit.",
                "--output-format",
                "json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert any("temperature" in item.lower() for item in payload["feature_grouping_signals"])
        assert payload["metadata"]["source_kind"] == "raw_input"

    def test_plan_architecture_text_output(self) -> None:
        runner = CliRunner()
        prior_handoff = run_prior_knowledge(
            "Demand forecasting has weekly seasonality. Labels are delayed by one week."
        ).to_handoff_text()

        result = runner.invoke(
            main,
            [
                "plan",
                "architecture",
                "--text",
                prior_handoff,
                "--output-format",
                "text",
            ],
        )

        assert result.exit_code == 0
        assert "BEGIN GPARCHITECT ARCHITECTURE HANDOFF" in result.output
        assert "Periodic kernel candidate" in result.output

    def test_plan_auto_json_routes_to_prior_only_when_requested(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "auto",
                "--text",
                "I only want prior-knowledge elicitation for a demand forecasting system with delayed labels. "
                "Do not do architecture planning yet.",
                "--output-format",
                "json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["chosen_path"] == "Prior Knowledge only"
        assert payload["architecture"] is None

    def test_plan_architecture_rejects_malformed_handoff(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "plan",
                "architecture",
                "--text",
                "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF\nSystem Summary:\n- Broken block",
                "--output-format",
                "json",
            ],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
