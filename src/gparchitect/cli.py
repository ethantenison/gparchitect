"""
Command-line interface for GPArchitect.

Purpose:
    Provides a CLI entry point that loads a CSV file, accepts a GP architecture
    instruction string, runs the full pipeline, and prints the generated DSL,
    summary, and revision history.

Role in pipeline:
    This is the outermost user-facing layer, delegating all work to the public API.

Inputs:
    - --csv: Path to a CSV data file.
    - --instruction / -i: Natural-language GP instruction string.
    - --inputs: Comma-separated input column names.
    - --outputs: Comma-separated output column names.
    - --task-column: Optional task indicator column name.
    - --max-retries: Maximum number of DSL revision attempts (default 5).
    - --log-json: Path to write the experiment log as JSON (optional).

Outputs:
    Prints DSL JSON, summary, and revision history to stdout.
    Optionally writes the experiment log to a JSON file.

What this module does NOT do:
    - It does not implement a web server or interactive REPL.
    - It does not train models in the background or schedule jobs.
"""

from __future__ import annotations

import json
import logging
import sys

import click

from gparchitect.api import run_gparchitect, summarize_attempts
from gparchitect.translator.translator import translate_to_dsl
from gparchitect.validation.validator import validate_dsl


@click.command()
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True), help="Path to the CSV data file.")
@click.option("--instruction", "-i", required=True, type=str, help="Natural-language GP architecture instruction.")
@click.option("--inputs", required=True, type=str, help="Comma-separated input column names.")
@click.option("--outputs", required=True, type=str, help="Comma-separated output column names.")
@click.option("--task-column", default=None, type=str, help="Task indicator column name (MultiTaskGP only).")
@click.option("--max-retries", default=5, show_default=True, type=int, help="Maximum DSL revision attempts.")
@click.option("--log-json", default=None, type=click.Path(), help="Path to write the experiment log as JSON.")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging output.")
def main(
    csv_path: str,
    instruction: str,
    inputs: str,
    outputs: str,
    task_column: str | None,
    max_retries: int,
    log_json: str | None,
    verbose: bool,
) -> None:
    """GPArchitect: build BoTorch GP models from natural-language instructions.

    Example:\n
        gparchitect --csv data.csv -i "Use a Matern52 kernel with ARD" \\
            --inputs x1,x2,x3 --outputs y
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

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


if __name__ == "__main__":
    main()
