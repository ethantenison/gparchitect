"""
Command wrapper for the GPArchitect planning runtime.

Purpose:
    Provides a small, quote-resistant command surface for VS Code custom agents
    and shell tooling that need to invoke the planning runtime.

Role in pipeline:
    This module is a thin CLI bridge over the planning subsystem. It does not
    construct GP DSL objects or build models.

Inputs / Outputs:
    Accepts planning input from a file path, stdin, or inline text and emits
    either JSON or canonical handoff text.

Non-obvious design decisions:
    - JSON output defaults to make agent/tool consumption straightforward.
    - File input is positional so agents can avoid nested subcommands and shell
      quoting of large multiline prompts.

What this module does NOT do:
    - It does not replace the main gparchitect CLI.
    - It does not route into model construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from gparchitect.planning import run_architect, run_architecture_focus, run_prior_knowledge


def _resolve_bridge_input(input_file: Path | None, use_stdin: bool, text: str | None) -> str:
    provided_sources = int(input_file is not None) + int(use_stdin) + int(text is not None)
    if provided_sources != 1:
        raise click.UsageError("Provide exactly one of INPUT_FILE, --stdin, or --text.")

    if input_file is not None:
        return input_file.read_text(encoding="utf-8")
    if use_stdin:
        return sys.stdin.read()
    assert text is not None
    return text


@click.command(name="gparchitect-plan")
@click.argument("mode", type=click.Choice(["prior", "architecture", "auto"], case_sensitive=False))
@click.argument("input_file", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--stdin", "use_stdin", is_flag=True, default=False, help="Read planning input from stdin.")
@click.option("--text", type=str, default=None, help="Inline planning input text.")
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Render output as JSON or canonical handoff text.",
)
@click.option(
    "--planning-requested",
    "planning_requested",
    flag_value=True,
    default=None,
    help="Force auto mode to continue into architecture planning.",
)
@click.option(
    "--elicitation-only",
    "planning_requested",
    flag_value=False,
    help="Force auto mode to stop after prior-knowledge elicitation.",
)
def main(
    mode: str,
    input_file: Path | None,
    use_stdin: bool,
    text: str | None,
    output_format: str,
    planning_requested: bool | None,
) -> None:
    """Run the planning runtime through a small agent-friendly wrapper."""

    input_text = _resolve_bridge_input(input_file, use_stdin, text)
    normalized_mode = mode.lower()

    if normalized_mode == "prior":
        result = run_prior_knowledge(input_text)
        if output_format.lower() == "json":
            click.echo(result.model_dump_json(indent=2))
        else:
            click.echo(result.to_handoff_text())
        return

    if normalized_mode == "architecture":
        result = run_architecture_focus(input_text)
        if output_format.lower() == "json":
            click.echo(result.model_dump_json(indent=2))
        else:
            click.echo(result.to_handoff_text())
        return

    result = run_architect(input_text, mode="auto", planning_requested=planning_requested)
    if output_format.lower() == "json":
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(result.to_text())
