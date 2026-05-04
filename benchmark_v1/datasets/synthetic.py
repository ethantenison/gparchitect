"""
benchmark_v1.datasets.synthetic — Tier 2 named-column synthetic tabular datasets.

Purpose:
    Generates reproducible synthetic tabular datasets with realistic named columns
    and known generating structure.  These are the primary benchmark datasets for
    benchmark_v1 because they directly exercise GPArchitect's named-column parsing,
    feature-group resolution, and kernel selection claims.

Role in benchmark pipeline:
    Dataset generators → Registry → Runner → Evaluation

Inputs:
    seed: int — controls all randomness deterministically.
    n_train / n_test: int — number of training and test points.
    noise_std: float — observation noise level.

Outputs:
    DatasetSplit — a dataclass with train/test pandas DataFrames plus metadata.

Non-obvious design decisions:
    - All generating equations are defined as module-level constants so reviewers
      can verify the ground truth without running code.
    - Noise is Gaussian and added after the deterministic signal to keep RMSE
      interpretable.
    - Column names are chosen to be human-readable and plausible for prompting.

What this module does NOT do:
    - It does not call any external data source.
    - It does not cache results to disk (the runner handles persistence).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DatasetSplit — shared output type for all generators
# ---------------------------------------------------------------------------


@dataclass
class DatasetSplit:
    """A deterministic train/test split for a named benchmark dataset.

    Attributes:
        name: Short identifier for the dataset (e.g. ``"additive"``).
        description: Human-readable description of the generating equation.
        train: Training DataFrame.
        test: Test DataFrame.
        input_columns: Names of the input feature columns.
        output_column: Name of the target column.
        noise_std: Noise standard deviation used to generate the data.
        seed: Random seed used for generation and splitting.
        generating_equation: String description of the latent function.
        tags: Descriptive tags (e.g. ``["additive", "ard_stress"]``).
    """

    name: str
    description: str
    train: pd.DataFrame
    test: pd.DataFrame
    input_columns: list[str]
    output_column: str
    noise_std: float
    seed: int
    generating_equation: str
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Additive dataset
# ---------------------------------------------------------------------------

_ADDITIVE_EQ = (
    "y = 0.5 * sin(2*pi*x_smooth) + 0.3 * x_trend + 0.2 * x_scale + eps"
    "  (x_irrelevant_1, x_irrelevant_2 are pure noise)"
)


def make_additive_dataset(
    seed: int = 42,
    n_train: int = 60,
    n_test: int = 40,
    noise_std: float = 0.05,
) -> DatasetSplit:
    """Generate an additive synthetic dataset with named columns.

    The latent function is a sum of three smooth terms.  Two irrelevant
    features are added to stress-test ARD and feature-selection behaviour.

    Args:
        seed: Random seed for reproducibility.
        n_train: Number of training samples.
        n_test: Number of test samples.
        noise_std: Standard deviation of Gaussian observation noise.

    Returns:
        DatasetSplit with train/test DataFrames.
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    x_smooth = rng.uniform(0.0, 1.0, n_total)
    x_trend = rng.uniform(0.0, 1.0, n_total)
    x_scale = rng.uniform(0.0, 1.0, n_total)
    x_irrelevant_1 = rng.uniform(0.0, 1.0, n_total)
    x_irrelevant_2 = rng.uniform(0.0, 1.0, n_total)

    signal = 0.5 * np.sin(2.0 * math.pi * x_smooth) + 0.3 * x_trend + 0.2 * x_scale
    noise = rng.normal(0.0, noise_std, n_total)
    y = signal + noise

    df = pd.DataFrame(
        {
            "x_smooth": x_smooth,
            "x_trend": x_trend,
            "x_scale": x_scale,
            "x_irrelevant_1": x_irrelevant_1,
            "x_irrelevant_2": x_irrelevant_2,
            "y": y,
        }
    )

    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)

    return DatasetSplit(
        name="additive",
        description="Additive GP signal with irrelevant features.",
        train=train,
        test=test,
        input_columns=["x_smooth", "x_trend", "x_scale", "x_irrelevant_1", "x_irrelevant_2"],
        output_column="y",
        noise_std=noise_std,
        seed=seed,
        generating_equation=_ADDITIVE_EQ,
        tags=["additive", "irrelevant_features"],
    )


# ---------------------------------------------------------------------------
# 2. Periodic + decay dataset
# ---------------------------------------------------------------------------

_PERIODIC_DECAY_EQ = (
    "y = 0.6 * sin(2*pi * seasonality_index / 0.25) * exp(-0.5 * system_age) + 0.4 * exp(-1.5 * system_age) + eps"
)


def make_periodic_decay_dataset(
    seed: int = 42,
    n_train: int = 60,
    n_test: int = 40,
    noise_std: float = 0.05,
) -> DatasetSplit:
    """Generate a periodic-plus-decay synthetic dataset.

    The latent function combines a periodic seasonal component with a
    monotone exponential decay that depends on ``system_age``.

    Args:
        seed: Random seed for reproducibility.
        n_train: Number of training samples.
        n_test: Number of test samples.
        noise_std: Standard deviation of Gaussian observation noise.

    Returns:
        DatasetSplit with train/test DataFrames.
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    seasonality_index = rng.uniform(0.0, 1.0, n_total)
    system_age = rng.uniform(0.0, 1.0, n_total)

    periodic_component = 0.6 * np.sin(2.0 * math.pi * seasonality_index / 0.25)
    decay_component = np.exp(-1.5 * system_age)
    signal = periodic_component * decay_component + 0.4 * decay_component

    noise = rng.normal(0.0, noise_std, n_total)
    y = signal + noise

    df = pd.DataFrame(
        {
            "seasonality_index": seasonality_index,
            "system_age": system_age,
            "y": y,
        }
    )

    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)

    return DatasetSplit(
        name="periodic_decay",
        description="Periodic seasonal effect multiplied by exponential decay.",
        train=train,
        test=test,
        input_columns=["seasonality_index", "system_age"],
        output_column="y",
        noise_std=noise_std,
        seed=seed,
        generating_equation=_PERIODIC_DECAY_EQ,
        tags=["periodic", "decay", "multiplicative_structure"],
    )


# ---------------------------------------------------------------------------
# 3. Interaction-dominant dataset
# ---------------------------------------------------------------------------

_INTERACTION_EQ = (
    "y = 0.8 * (material_hardness * process_temperature)"
    " + 0.2 * material_hardness + eps"
    "  (pure cross-feature interaction dominates)"
)


def make_interaction_dataset(
    seed: int = 42,
    n_train: int = 60,
    n_test: int = 40,
    noise_std: float = 0.05,
) -> DatasetSplit:
    """Generate an interaction-dominant synthetic dataset.

    The latent function is dominated by the product of two input features,
    so kernels that capture multiplicative cross-feature interactions (e.g.,
    product kernels) should outperform purely additive choices.

    Args:
        seed: Random seed for reproducibility.
        n_train: Number of training samples.
        n_test: Number of test samples.
        noise_std: Standard deviation of Gaussian observation noise.

    Returns:
        DatasetSplit with train/test DataFrames.
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    material_hardness = rng.uniform(0.0, 1.0, n_total)
    process_temperature = rng.uniform(0.0, 1.0, n_total)
    cooldown_rate = rng.uniform(0.0, 1.0, n_total)

    signal = 0.8 * material_hardness * process_temperature + 0.2 * material_hardness
    noise = rng.normal(0.0, noise_std, n_total)
    y = signal + noise

    df = pd.DataFrame(
        {
            "material_hardness": material_hardness,
            "process_temperature": process_temperature,
            "cooldown_rate": cooldown_rate,
            "y": y,
        }
    )

    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)

    return DatasetSplit(
        name="interaction",
        description="Cross-feature interaction dominates; additive kernels are misspecified.",
        train=train,
        test=test,
        input_columns=["material_hardness", "process_temperature", "cooldown_rate"],
        output_column="y",
        noise_std=noise_std,
        seed=seed,
        generating_equation=_INTERACTION_EQ,
        tags=["interaction", "multiplicative_structure", "grouped_kernels"],
    )


# ---------------------------------------------------------------------------
# 4. ARD stress dataset
# ---------------------------------------------------------------------------

_ARD_EQ = (
    "y = sin(pi * x_signal_1) + 0.5 * x_signal_2^2 + eps"
    "  (x_weak_1..x_weak_4 have near-zero coefficients;"
    "   x_irrelevant is pure noise)"
)


def make_ard_stress_dataset(
    seed: int = 42,
    n_train: int = 60,
    n_test: int = 40,
    noise_std: float = 0.05,
) -> DatasetSplit:
    """Generate an ARD stress dataset with relevant and irrelevant features.

    The signal depends only on ``x_signal_1`` and ``x_signal_2``.  Four
    weak features contribute negligibly, and one feature is pure noise.
    This tests whether ARD instructions help (by shrinking irrelevant
    lengthscales) or hurt (by over-fitting).

    Args:
        seed: Random seed for reproducibility.
        n_train: Number of training samples.
        n_test: Number of test samples.
        noise_std: Standard deviation of Gaussian observation noise.

    Returns:
        DatasetSplit with train/test DataFrames.
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    x_signal_1 = rng.uniform(0.0, 1.0, n_total)
    x_signal_2 = rng.uniform(0.0, 1.0, n_total)
    x_weak_1 = rng.uniform(0.0, 1.0, n_total)
    x_weak_2 = rng.uniform(0.0, 1.0, n_total)
    x_weak_3 = rng.uniform(0.0, 1.0, n_total)
    x_weak_4 = rng.uniform(0.0, 1.0, n_total)
    x_irrelevant = rng.uniform(0.0, 1.0, n_total)

    signal = (
        np.sin(math.pi * x_signal_1)
        + 0.5 * x_signal_2**2
        + 0.03 * x_weak_1
        + 0.02 * x_weak_2
        + 0.01 * x_weak_3
        + 0.01 * x_weak_4
    )
    noise = rng.normal(0.0, noise_std, n_total)
    y = signal + noise

    df = pd.DataFrame(
        {
            "x_signal_1": x_signal_1,
            "x_signal_2": x_signal_2,
            "x_weak_1": x_weak_1,
            "x_weak_2": x_weak_2,
            "x_weak_3": x_weak_3,
            "x_weak_4": x_weak_4,
            "x_irrelevant": x_irrelevant,
            "y": y,
        }
    )

    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)

    return DatasetSplit(
        name="ard_stress",
        description="Two relevant features, four weak features, one pure noise feature.",
        train=train,
        test=test,
        input_columns=["x_signal_1", "x_signal_2", "x_weak_1", "x_weak_2", "x_weak_3", "x_weak_4", "x_irrelevant"],
        output_column="y",
        noise_std=noise_std,
        seed=seed,
        generating_equation=_ARD_EQ,
        tags=["ard", "irrelevant_features", "feature_selection"],
    )


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

#: Type alias for dataset generator functions.
DatasetGeneratorFn = Callable[..., DatasetSplit]

#: Map of dataset name → generator function.
SYNTHETIC_GENERATORS: dict[str, DatasetGeneratorFn] = {
    "additive": make_additive_dataset,
    "periodic_decay": make_periodic_decay_dataset,
    "interaction": make_interaction_dataset,
    "ard_stress": make_ard_stress_dataset,
}


def list_synthetic_datasets() -> list[str]:
    """Return the names of all available synthetic datasets.

    Returns:
        Sorted list of dataset name strings.
    """
    return sorted(SYNTHETIC_GENERATORS.keys())
