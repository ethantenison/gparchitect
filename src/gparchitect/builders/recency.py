"""
Recency-filtering utilities for GPArchitect — time-driven non-stationarity support.

Purpose:
    Provides functions to filter training observations based on their position along a
    time-like input dimension.  This implements Tier 1 forgetting strategies:
    sliding-window filtering and exponential-discount filtering.

Role in pipeline:
    Natural language → GP DSL → Validation → **Recency Filtering** → Model Builder → Fit

Inputs:
    - train_X: torch.Tensor of shape (N, D) — scaled input features.
    - train_Y: torch.Tensor of shape (N, M) — output targets.
    - spec: RecencyFilteringSpec — the filtering configuration from ExecutionSpec.

Outputs:
    A (filtered_train_X, filtered_train_Y) pair with the same dtype and device as the
    inputs but potentially fewer rows.

Non-obvious design decisions:
    - Both strategies reduce the dataset to a subset of the original observations.
      Neither performs likelihood-weighted GP inference — this is dataset truncation,
      not true observation weighting.
    - SLIDING_WINDOW removes observations older than max_time − window_size.
    - EXPONENTIAL_DISCOUNT computes weights w_i = exp(−rate · (t_max − t_i)) and
      removes observations with w_i < min_weight.  This is a deterministic threshold,
      not a probabilistic sampling, and is equivalent to a soft sliding window.
    - Neither strategy rescales inputs after filtering; the caller is responsible for
      input scaling (which should happen before this step so that window_size and
      discount_rate operate in the normalised [0, 1] feature space when input_scaling
      is active).
    - If the filter would produce an empty dataset, the most recent single observation
      is retained to avoid raising an error during model construction.

What this module does NOT do:
    - It does not perform weighted maximum-likelihood estimation; filtering is a
      simpler approximation.
    - It does not modify the DSL or the spec.
    - It does not re-scale inputs after filtering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gparchitect.dsl.schema import RecencyFilteringMode, RecencyFilteringSpec

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def apply_recency_filtering(
    train_X: "torch.Tensor",
    train_Y: "torch.Tensor",
    spec: RecencyFilteringSpec,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Filter training data according to a recency-filtering specification.

    This implements dataset truncation before GP fitting — it is NOT likelihood-
    weighted inference.  Old observations are removed from the training set;
    the remaining observations are treated with equal weight by the GP.

    Args:
        train_X: Input tensor of shape (N, D).
        train_Y: Output tensor of shape (N, M).
        spec: RecencyFilteringSpec describing the filtering strategy.

    Returns:
        A (filtered_train_X, filtered_train_Y) tuple.  If no observations pass the
        filter, the single most recent observation is retained.

    Raises:
        ValueError: If spec.time_feature_index is out of bounds for train_X.
    """
    import torch

    n_obs, n_features = train_X.shape[:2] if train_X.ndim >= 2 else (train_X.shape[0], 1)
    if spec.time_feature_index < 0 or spec.time_feature_index >= n_features:
        raise ValueError(
            f"RecencyFilteringSpec.time_feature_index={spec.time_feature_index} is out of range "
            f"for train_X with {n_features} features."
        )

    time_col = train_X[:, spec.time_feature_index]
    t_max = time_col.max()

    if spec.mode == RecencyFilteringMode.SLIDING_WINDOW:
        mask = _sliding_window_mask(time_col, t_max, spec)
    else:
        mask = _exponential_discount_mask(time_col, t_max, spec)

    n_kept = int(mask.sum().item())
    if n_kept == 0:
        logger.warning(
            "RecencyFiltering: filter left 0 observations; retaining the single most recent observation."
        )
        most_recent_idx = int(torch.argmax(time_col).item())
        mask = torch.zeros(n_obs, dtype=torch.bool, device=train_X.device)
        mask[most_recent_idx] = True
        n_kept = 1

    logger.info(
        "RecencyFiltering (%s): kept %d / %d observations.",
        spec.mode.value,
        n_kept,
        n_obs,
    )
    return train_X[mask], train_Y[mask]


def _sliding_window_mask(
    time_col: "torch.Tensor",
    t_max: "torch.Tensor",
    spec: RecencyFilteringSpec,
) -> "torch.Tensor":
    """Return a boolean mask for observations within the sliding window.

    Args:
        time_col: 1-D tensor of time values for each observation.
        t_max: Maximum time value in the dataset.
        spec: RecencyFilteringSpec with window_size set.

    Returns:
        Boolean mask tensor of shape (N,).

    Raises:
        ValueError: If window_size is not set or is not positive.
    """
    if spec.window_size is None:
        raise ValueError("RecencyFilteringSpec.window_size must be set for SLIDING_WINDOW mode.")
    if spec.window_size <= 0:
        raise ValueError(f"RecencyFilteringSpec.window_size must be > 0, got {spec.window_size}.")

    cutoff = t_max - spec.window_size
    return time_col >= cutoff


def _exponential_discount_mask(
    time_col: "torch.Tensor",
    t_max: "torch.Tensor",
    spec: RecencyFilteringSpec,
) -> "torch.Tensor":
    """Return a boolean mask for observations with exponential weight >= min_weight.

    The weight w_i = exp(−rate · (t_max − t_i)) is a thresholded-retention rule.
    Observations with w_i < min_weight are removed.  This is NOT true exponential
    likelihood weighting — it is a dataset truncation approximation.

    Args:
        time_col: 1-D tensor of time values for each observation.
        t_max: Maximum time value in the dataset.
        spec: RecencyFilteringSpec with discount_rate set.

    Returns:
        Boolean mask tensor of shape (N,).

    Raises:
        ValueError: If discount_rate is not set or is not positive.
    """
    import math

    if spec.discount_rate is None:
        raise ValueError("RecencyFilteringSpec.discount_rate must be set for EXPONENTIAL_DISCOUNT mode.")
    if spec.discount_rate <= 0:
        raise ValueError(f"RecencyFilteringSpec.discount_rate must be > 0, got {spec.discount_rate}.")

    delta_t = t_max - time_col
    weights = (-spec.discount_rate * delta_t).exp()
    min_weight = spec.min_weight if spec.min_weight > 0 else 0.01

    # Guard against log(0) in the threshold; ensure the threshold is reachable
    max_delta_threshold = -math.log(min_weight) / spec.discount_rate
    if float(delta_t.max().item()) > max_delta_threshold * 1e3:
        logger.debug(
            "RecencyFiltering: very large time range detected; min_weight threshold may discard most data."
        )

    return weights >= min_weight

