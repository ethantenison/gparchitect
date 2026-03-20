"""
Data interface for GPArchitect.

Purpose:
    Accepts a pandas DataFrame and converts it to torch tensors suitable for BoTorch.
    Validates column names and data types before conversion.

Role in pipeline:
    This module is called before model building to prepare tensor inputs.

Inputs:
    - dataframe: pd.DataFrame — input/output data.
    - input_columns: list[str] — column names for input features.
    - output_columns: list[str] — column names for model outputs.
    - task_column: str | None — column name for the task indicator.

Outputs:
    DataBundle — a dataclass holding train_X, train_Y, input_dim, output_dim,
    task_feature_index, and column metadata.

Non-obvious design decisions:
    - Tensors are always float64 (torch.double) for numerical stability with BoTorch.
        - Continuous input columns are min-max scaled into [0, 1] before model building.
    - The task column is kept in train_X (as the last column when specified) so that
      the builder module can extract it by index.
    - Missing values are rejected with a clear error.

What this module does NOT do:
    - It does not fit the data.
    - It does not infer column roles — callers must specify them explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    """Holds tensors and metadata prepared from a pandas DataFrame.

    Attributes:
        train_X: Input tensor of shape (N, D).
        train_Y: Output tensor of shape (N, M).
        input_dim: Number of input columns (D).
        output_dim: Number of output columns (M).
        task_feature_index: Column index of the task indicator in train_X, or None.
        input_columns: Names of the input columns.
        output_columns: Names of the output columns.
        task_column: Name of the task column, or None.
    """

    train_X: "torch.Tensor"  # noqa: F821
    train_Y: "torch.Tensor"  # noqa: F821
    input_dim: int
    output_dim: int
    task_feature_index: int | None
    input_columns: list[str]
    output_columns: list[str]
    task_column: str | None
    input_scaling_applied: bool
    input_feature_ranges: dict[str, tuple[float, float]]


def prepare_data(
    dataframe,  # noqa: ANN001
    input_columns: list[str],
    output_columns: list[str],
    task_column: str | None = None,
) -> DataBundle:
    """Convert a pandas DataFrame into a DataBundle of scaled torch tensors.

    Args:
        dataframe: A pandas DataFrame with numeric columns.
        input_columns: Column names to use as model inputs.
        output_columns: Column names to use as model outputs.
        task_column: Optional column name for the task indicator (MultiTaskGP).

    Returns:
        DataBundle containing train_X, train_Y, and associated metadata.

    Raises:
        ValueError: If any specified columns are missing or contain NaN values.
        ImportError: If pandas or torch are not installed.
    """
    import torch

    all_required = set(input_columns) | set(output_columns)
    if task_column:
        all_required.add(task_column)

    missing = all_required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {sorted(missing)}")

    if dataframe[list(all_required)].isnull().any().any():
        raise ValueError("DataFrame contains NaN values in the specified columns.")

    input_cols_full = list(input_columns)
    task_feature_index: int | None = None
    if task_column is not None:
        input_cols_full.append(task_column)
        task_feature_index = len(input_cols_full) - 1

    scaled_inputs = dataframe[input_columns].copy()
    input_feature_ranges: dict[str, tuple[float, float]] = {}
    for column in input_columns:
        column_min = float(scaled_inputs[column].min())
        column_max = float(scaled_inputs[column].max())
        input_feature_ranges[column] = (column_min, column_max)
        scale = column_max - column_min
        if scale > 0:
            scaled_inputs[column] = (scaled_inputs[column] - column_min) / scale
        else:
            scaled_inputs[column] = 0.0

    train_X_frame = scaled_inputs.copy()
    if task_column is not None:
        train_X_frame[task_column] = dataframe[task_column].astype(float)

    train_X = torch.tensor(
        train_X_frame[input_cols_full].to_numpy(dtype=float),
        dtype=torch.double,
    )
    train_Y = torch.tensor(
        dataframe[output_columns].to_numpy(dtype=float),
        dtype=torch.double,
    )

    logger.info(
        "Prepared data: train_X=%s, train_Y=%s, task_feature_index=%s",
        tuple(train_X.shape),
        tuple(train_Y.shape),
        task_feature_index,
    )

    return DataBundle(
        train_X=train_X,
        train_Y=train_Y,
        input_dim=len(input_cols_full),
        output_dim=len(output_columns),
        task_feature_index=task_feature_index,
        input_columns=input_cols_full,
        output_columns=output_columns,
        task_column=task_column,
        input_scaling_applied=True,
        input_feature_ranges=input_feature_ranges,
    )
