"""Tests for the tiny planning bridge command."""

from __future__ import annotations

import json

from click.testing import CliRunner

from gparchitect.planning.bridge import main as bridge_main
from gparchitect.planning.runtime import run_prior_knowledge


class TestPlanningBridgeCLI:
    """Validate the agent-friendly planning wrapper command."""

    def test_bridge_auto_reads_prompt_from_file(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("prompt.txt", "w", encoding="utf-8") as fh:
                fh.write(
                    "Temperature and pressure interact. Noise increases near the upper pressure limit. "
                    "Operators want downstream planning."
                )

            result = runner.invoke(bridge_main, ["auto", "prompt.txt"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["chosen_path"] == "Prior Knowledge -> Architecture Focus"

    def test_bridge_prior_reads_prompt_from_stdin(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            bridge_main,
            ["prior", "--stdin"],
            input="Demand forecasting has weekly seasonality and delayed labels.",
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert any("weekly seasonality" in item.lower() for item in payload["temporal_or_multiscale_signals"])

    def test_bridge_architecture_text_output(self) -> None:
        runner = CliRunner()
        handoff = run_prior_knowledge(
            "Battery degradation forecasting depends on temperature and cycle count."
        ).to_handoff_text()

        result = runner.invoke(
            bridge_main,
            ["architecture", "--text", handoff, "--output-format", "text"],
        )

        assert result.exit_code == 0
        assert "BEGIN GPARCHITECT ARCHITECTURE HANDOFF" in result.output

    def test_bridge_requires_exactly_one_input_source(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            bridge_main,
            ["auto", "--stdin", "--text", "prompt"],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, SystemExit)
