"""
Command-line interface for GPArchitect.

Purpose:
    Provides a CLI entry point for both the executable planning runtime and the
    existing model-building pipeline.

Role in pipeline:
    This is the outermost user-facing layer. It delegates model construction to
    gparchitect.api and planning work to gparchitect.planning.

Inputs:
    - Model pipeline mode accepts CSV, instruction, and column-selection options.
    - Planning mode accepts inline text, file input, or stdin plus an output format.

Outputs:
    - Model pipeline mode prints DSL JSON, summary, and revision history.
    - Planning mode prints structured JSON or canonical handoff text.

What this module does NOT do:
    - It does not implement a web server or interactive REPL.
    - It does not directly build models from planning text.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from gparchitect.api import run_gparchitect, summarize_attempts
from gparchitect.planning import run_architect, run_architecture_focus, run_prior_knowledge
from gparchitect.translator.translator import translate_to_dsl
from gparchitect.validation.validator import validate_dsl


def _configure_logging(verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)


def _resolve_text_input(text: str | None, input_file: Path | None, use_stdin: bool) -> str:
    provided_sources = int(text is not None) + int(input_file is not None) + int(use_stdin)
    if provided_sources != 1:
        raise click.UsageError("Provide exactly one of --text, --input-file, or --stdin.")

    if text is not None:
        return text
    if input_file is not None:
        return input_file.read_text(encoding="utf-8")
    return sys.stdin.read()


def _emit_planning_output(result, output_format: str, *, text_renderer) -> None:  # noqa: ANN001
    if output_format == "json":
        click.echo(result.model_dump_json(indent=2))
        return
    click.echo(text_renderer())


def _planning_input_options(command):  # noqa: ANN001
    command = click.option("--stdin", "use_stdin", is_flag=True, default=False, help="Read planning input from stdin.")(
        command
    )
    command = click.option(
        "--input-file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Path to a text file containing planning input.",
    )(command)
    command = click.option("--text", type=str, default=None, help="Inline planning input text.")(command)
    return command


@click.group(invoke_without_command=True)
@click.option("--csv", "csv_path", type=click.Path(exists=True), default=None, help="Path to the CSV data file.")
@click.option("--instruction", "-i", type=str, default=None, help="Natural-language GP architecture instruction.")
@click.option("--inputs", type=str, default=None, help="Comma-separated input column names.")
@click.option("--outputs", type=str, default=None, help="Comma-separated output column names.")
@click.option("--task-column", default=None, type=str, help="Task indicator column name (MultiTaskGP only).")
@click.option("--max-retries", default=5, show_default=True, type=int, help="Maximum DSL revision attempts.")
@click.option("--log-json", default=None, type=click.Path(), help="Path to write the experiment log as JSON.")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging output.")
@click.pass_context
def main(
    ctx: click.Context,
    csv_path: str,
    instruction: str,
    inputs: str,
    outputs: str,
    task_column: str | None,
    max_retries: int,
    log_json: str | None,
    verbose: bool,
) -> None:
    """GPArchitect CLI.

    Example:\n
        gparchitect --csv data.csv -i "Use a Matern52 kernel with ARD" \\
            --inputs x1,x2,x3 --outputs y
    """
    _configure_logging(verbose)

    if ctx.invoked_subcommand is not None:
        return

    missing_options = [
        option_name
        for option_name, option_value in (
            ("--csv", csv_path),
            ("--instruction", instruction),
            ("--inputs", inputs),
            ("--outputs", outputs),
        )
        if not option_value
    ]
    if missing_options:
        raise click.UsageError(
            "Missing options for model-building mode: " + ", ".join(missing_options) + ". "
            "Use `gparchitect plan --help` for planning commands."
        )

    _run_model_pipeline(
        csv_path=csv_path,
        instruction=instruction,
        inputs=inputs,
        outputs=outputs,
        task_column=task_column,
        max_retries=max_retries,
        log_json=log_json,
    )


def _run_model_pipeline(
    *,
    csv_path: str,
    instruction: str,
    inputs: str,
    outputs: str,
    task_column: str | None,
    max_retries: int,
    log_json: str | None,
) -> None:
    """Run the existing GPArchitect model-building CLI flow."""

    try:
        import pandas as pd
    except ImportError:
        click.echo("Error: pandas is required. Install with: pip install pandas", err=True)
        sys.exit(1)

    input_columns = [col.strip() for col in inputs.split(",") if col.strip()]
    output_columns = [col.strip() for col in outputs.split(",") if col.strip()]
    task_col = task_column.strip() if task_column else None

    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as exc:
        click.echo(f"Error reading CSV: {exc}", err=True)
        sys.exit(1)

    # Show the generated DSL before fitting
    try:
        from gparchitect.builders.data import prepare_data

        data_bundle = prepare_data(dataframe, input_columns, output_columns, task_col)
        spec = translate_to_dsl(
            instruction=instruction,
            input_dim=data_bundle.input_dim,
            output_dim=data_bundle.output_dim,
            task_feature_index=data_bundle.task_feature_index,
            input_feature_names=input_columns,
        )
        validation_result = validate_dsl(spec)

        click.echo("\n=== Generated GP DSL ===")
        click.echo(spec.model_dump_json(indent=2))

        if not validation_result.is_valid:
            click.echo("\n=== Validation Errors ===")
            for error in validation_result.errors:
                click.echo(f"  ERROR: {error}")
        if validation_result.warnings:
            click.echo("\n=== Validation Warnings ===")
            for warning in validation_result.warnings:
                click.echo(f"  WARNING: {warning}")
    except Exception as exc:
        click.echo(f"Warning: could not display initial DSL: {exc}", err=True)

    click.echo("\n=== Running GPArchitect Pipeline ===")
    model, experiment_log = run_gparchitect(
        dataframe=dataframe,
        instruction=instruction,
        input_columns=input_columns,
        output_columns=output_columns,
        task_column=task_col,
        max_retries=max_retries,
    )

    click.echo("\n" + summarize_attempts(experiment_log))

    if log_json:
        try:
            with open(log_json, "w", encoding="utf-8") as fh:
                fh.write(experiment_log.to_json())
            click.echo(f"\nExperiment log written to: {log_json}")
        except OSError as exc:
            click.echo(f"Warning: could not write log file: {exc}", err=True)

    if model is None:
        click.echo("\nPipeline FAILED: no model could be built. See log for details.", err=True)
        sys.exit(1)
    else:
        click.echo("\nPipeline SUCCEEDED.")


@main.group()
def plan() -> None:
    """Run the executable planning runtime without constructing a model."""


@plan.command("prior")
@_planning_input_options
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Render the prior-knowledge handoff as text or JSON.",
)
def plan_prior(text: str | None, input_file: Path | None, use_stdin: bool, output_format: str) -> None:
    """Generate a GPArchitect prior-knowledge handoff."""

    input_text = _resolve_text_input(text, input_file, use_stdin)
    result = run_prior_knowledge(input_text)
    _emit_planning_output(result, output_format.lower(), text_renderer=result.to_handoff_text)


@plan.command("architecture")
@_planning_input_options
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Render the architecture handoff as text or JSON.",
)
def plan_architecture(text: str | None, input_file: Path | None, use_stdin: bool, output_format: str) -> None:
    """Generate a GPArchitect architecture handoff from prior knowledge."""

    input_text = _resolve_text_input(text, input_file, use_stdin)
    result = run_architecture_focus(input_text)
    _emit_planning_output(result, output_format.lower(), text_renderer=result.to_handoff_text)


@plan.command("auto")
@_planning_input_options
@click.option(
    "--planning-requested",
    "planning_requested",
    flag_value=True,
    default=None,
    help="Force the auto router to continue into architecture planning.",
)
@click.option(
    "--elicitation-only",
    "planning_requested",
    flag_value=False,
    help="Force the auto router to stop after prior-knowledge elicitation.",
)
@click.option(
    "--output-format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Render the full planning result as text or JSON.",
)
def plan_auto(
    text: str | None,
    input_file: Path | None,
    use_stdin: bool,
    planning_requested: bool | None,
    output_format: str,
) -> None:
    """Route planning input through the executable planner."""

    input_text = _resolve_text_input(text, input_file, use_stdin)
    result = run_architect(input_text, mode="auto", planning_requested=planning_requested)
    _emit_planning_output(result, output_format.lower(), text_renderer=result.to_text)


if __name__ == "__main__":
    main()
