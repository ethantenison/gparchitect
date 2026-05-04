"""
benchmark_v1.datasets.botorch_functions — Tier 1 BoTorch test-function adapters.

Purpose:
    Generates reproducible tabular benchmark datasets by sampling from standard
    BoTorch analytic test functions (Branin, Hartmann6, Rosenbrock).  These
    provide a controlled function layer with known optimal structure.

Role in benchmark pipeline:
    Dataset generators → Registry → Runner → Evaluation

Inputs:
    seed: int — controls point sampling deterministically.
    n_train / n_test: int — number of training and test samples.
    noise_std: float — observation noise level (0.0 = noiseless).

Outputs:
    DatasetSplit — same shape as Tier 2 datasets, with named feature columns.

Non-obvious design decisions:
    - Feature columns are given stable human-readable names (e.g. ``x0``, ``x1``)
      so that prompts can reference them consistently.
    - BoTorch functions are evaluated in their native [0,1]^d or [-5,10]^d domains,
      then the raw inputs are stored.  The runner/builder handles re-scaling.
    - Noiseless evaluation uses noise_std=0.0; the generator still returns the
      same DatasetSplit type for uniform handling downstream.

What this module does NOT do:
    - It does not import BoTorch at module-load time — imports are lazy so that
      the module is importable even in environments without BoTorch installed.
    - It does not perform any model fitting.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from benchmark_v1.datasets.synthetic import DatasetSplit

logger = logging.getLogger(__name__)


def _require_botorch() -> None:
    """Raise ImportError if botorch is not available."""
    try:
        import botorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "BoTorch and torch must be installed to use Tier 1 benchmark functions. "
            "Install with: pip install botorch torch"
        ) from exc


# ---------------------------------------------------------------------------
# Branin (2-dimensional)
# ---------------------------------------------------------------------------

_BRANIN_EQ = "Branin(x0, x1) — standard 2D benchmark function with 3 global optima. x0 in [-5, 10], x1 in [0, 15]."


def make_branin_dataset(
    seed: int = 42,
    n_train: int = 60,
    n_test: int = 40,
    noise_std: float = 0.0,
) -> DatasetSplit:
    """Generate a tabular dataset by sampling from the Branin test function.

    Args:
        seed: Random seed for reproducibility.
        n_train: Number of training samples.
        n_test: Number of test samples.
        noise_std: Gaussian noise standard deviation (0.0 = noiseless).

    Returns:
        DatasetSplit with named columns ``x0``, ``x1``, and ``y``.
    """
    _require_botorch()
    import torch
    from botorch.test_functions import Branin

    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    func = Branin(noise_std=0.0, negate=False)

    # Sample in [0, 1]^2 and map to domain
    unit_samples = rng.uniform(0.0, 1.0, (n_total, 2))
    bounds = func.bounds  # shape (2, 2): [[lb0, lb1], [ub0, ub1]]
    lb = bounds[0].numpy()
    ub = bounds[1].numpy()
    raw_x = unit_samples * (ub - lb) + lb

    x_tensor = torch.tensor(raw_x, dtype=torch.double)
    y_noiseless = func(x_tensor).numpy()

    noise = rng.normal(0.0, noise_std, n_total) if noise_std > 0.0 else np.zeros(n_total)
    y = y_noiseless + noise

    df = pd.DataFrame({"x0": raw_x[:, 0], "x1": raw_x[:, 1], "y": y})

    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)

    return DatasetSplit(
        name="branin",
        description="Branin 2D test function; 3 global optima.",
        train=train,
        test=test,
        input_columns=["x0", "x1"],
        output_column="y",
        noise_std=noise_std,
        seed=seed,
        generating_equation=_BRANIN_EQ,
        tags=["botorch_function", "2d", "multimodal"],
    )


# ---------------------------------------------------------------------------
# Hartmann6 (6-dimensional)
# ---------------------------------------------------------------------------

_HARTMANN6_EQ = "Hartmann6(x0..x5) — 6D benchmark function with one global optimum. All inputs in [0, 1]^6."


def make_hartmann6_dataset(
    seed: int = 42,
    n_train: int = 80,
    n_test: int = 40,
    noise_std: float = 0.0,
) -> DatasetSplit:
    """Generate a tabular dataset by sampling from the Hartmann6 test function.

    Args:
        seed: Random seed for reproducibility.
        n_train: Number of training samples.
        n_test: Number of test samples.
        noise_std: Gaussian noise standard deviation (0.0 = noiseless).

    Returns:
        DatasetSplit with named columns ``x0``–``x5`` and ``y``.
    """
    _require_botorch()
    import torch
    from botorch.test_functions import Hartmann

    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    dim = 6

    func = Hartmann(dim=dim, noise_std=0.0, negate=False)

    unit_samples = rng.uniform(0.0, 1.0, (n_total, dim))
    x_tensor = torch.tensor(unit_samples, dtype=torch.double)
    y_noiseless = func(x_tensor).numpy()

    noise = rng.normal(0.0, noise_std, n_total) if noise_std > 0.0 else np.zeros(n_total)
    y = y_noiseless + noise

    input_cols = [f"x{i}" for i in range(dim)]
    df = pd.DataFrame(dict(zip(input_cols, unit_samples.T)))
    df["y"] = y

    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)

    return DatasetSplit(
        name="hartmann6",
        description="Hartmann6 6D test function; one global optimum.",
        train=train,
        test=test,
        input_columns=input_cols,
        output_column="y",
        noise_std=noise_std,
        seed=seed,
        generating_equation=_HARTMANN6_EQ,
        tags=["botorch_function", "6d", "single_optimum"],
    )


# ---------------------------------------------------------------------------
# Rosenbrock (4-dimensional slice)
# ---------------------------------------------------------------------------

_ROSENBROCK_EQ = "Rosenbrock(x0..x3) — 4D slice of the Rosenbrock function. Each input in [-2, 2]."


def make_rosenbrock_dataset(
    seed: int = 42,
    n_train: int = 80,
    n_test: int = 40,
    noise_std: float = 0.0,
) -> DatasetSplit:
    """Generate a tabular dataset by sampling from the Rosenbrock test function.

    Uses a 4-dimensional slice for a manageable number of features while
    retaining the challenging banana-shaped valley structure.

    Args:
        seed: Random seed for reproducibility.
        n_train: Number of training samples.
        n_test: Number of test samples.
        noise_std: Gaussian noise standard deviation (0.0 = noiseless).

    Returns:
        DatasetSplit with named columns ``x0``–``x3`` and ``y``.
    """
    _require_botorch()
    import torch
    from botorch.test_functions import Rosenbrock

    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    dim = 4

    func = Rosenbrock(dim=dim, noise_std=0.0, negate=False)

    bounds = func.bounds  # shape (2, dim)
    lb = bounds[0].numpy()
    ub = bounds[1].numpy()
    unit_samples = rng.uniform(0.0, 1.0, (n_total, dim))
    raw_x = unit_samples * (ub - lb) + lb

    x_tensor = torch.tensor(raw_x, dtype=torch.double)
    y_noiseless = func(x_tensor).numpy()

    # Rosenbrock outputs are large; log-scale for numeric stability
    y_log = np.log1p(np.abs(y_noiseless)) * np.sign(y_noiseless)
    noise = rng.normal(0.0, noise_std, n_total) if noise_std > 0.0 else np.zeros(n_total)
    y = y_log + noise

    input_cols = [f"x{i}" for i in range(dim)]
    df = pd.DataFrame(dict(zip(input_cols, raw_x.T)))
    df["y"] = y

    train = df.iloc[:n_train].reset_index(drop=True)
    test = df.iloc[n_train:].reset_index(drop=True)

    return DatasetSplit(
        name="rosenbrock",
        description="Rosenbrock 4D test function (log-scaled output); banana-shaped valley.",
        train=train,
        test=test,
        input_columns=input_cols,
        output_column="y",
        noise_std=noise_std,
        seed=seed,
        generating_equation=_ROSENBROCK_EQ,
        tags=["botorch_function", "4d", "rosenbrock"],
    )


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

#: Map of function name → generator callable.
BOTORCH_GENERATORS: dict[str, Callable[..., DatasetSplit]] = {
    "branin": make_branin_dataset,
    "hartmann6": make_hartmann6_dataset,
    "rosenbrock": make_rosenbrock_dataset,
}


def list_botorch_datasets() -> list[str]:
    """Return the names of all available BoTorch benchmark datasets.

    Returns:
        Sorted list of dataset name strings.
    """
    return sorted(BOTORCH_GENERATORS.keys())
