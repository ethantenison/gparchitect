"""
benchmark_v1.registry — benchmark suite registry for benchmark_v1.

Purpose:
    Defines the complete, fixed set of benchmark entries: which datasets to run,
    which noise levels to use, which seeds to sweep over, and which baselines to
    include.  This is the single source of truth for ``run_benchmark.py``.

Role in benchmark pipeline:
    Registry → Runner → Evaluation → Report

Non-obvious design decisions:
    - Seeds are fixed at the registry level so that results from different runs
      are directly comparable.
    - Noise levels are expressed as absolute standard deviations (not SNR) so
      that dataset-specific signal amplitudes do not confuse interpretation.
    - The registry is a plain Python data structure (list of dataclasses) so
      that it can be serialised to JSON for archival without custom logic.

What this module does NOT do:
    - It does not generate data (see datasets/ for that).
    - It does not run models (see run_benchmark.py for that).
    - It does not define prompts (see prompts/variants.py for that).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BenchmarkEntry — one row in the registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkEntry:
    """A single benchmark configuration entry.

    Attributes:
        dataset_name: Identifier matching a dataset generator.
        tier: ``1`` (BoTorch function) or ``2`` (synthetic tabular).
        noise_levels: Noise standard deviations to evaluate.
        seeds: Random seeds for train/test split generation.
        baselines: Names of baseline configurations to compare against.
        prompt_variants: Prompt variant names to run (subset of aligned/vague/misleading).
        n_train: Number of training samples.
        n_test: Number of test samples.
        tags: Descriptive tags for filtering.
    """

    dataset_name: str
    tier: int
    noise_levels: tuple[float, ...]
    seeds: tuple[int, ...]
    baselines: tuple[str, ...]
    prompt_variants: tuple[str, ...]
    n_train: int
    n_test: int
    tags: tuple[str, ...] = field(default=())


# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

#: Seeds used for all benchmark entries.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Baseline names applied to all entries.
DEFAULT_BASELINES: tuple[str, ...] = ("default_singletask", "matern52_ard")

#: All three prompt variants.
ALL_VARIANTS: tuple[str, ...] = ("aligned", "vague", "misleading")

#: The full benchmark registry.
REGISTRY: list[BenchmarkEntry] = [
    # ------------------------------------------------------------------
    # Tier 2 — named-column synthetic tabular datasets
    # ------------------------------------------------------------------
    BenchmarkEntry(
        dataset_name="additive",
        tier=2,
        noise_levels=(0.0, 0.05, 0.15),
        seeds=DEFAULT_SEEDS,
        baselines=DEFAULT_BASELINES,
        prompt_variants=ALL_VARIANTS,
        n_train=60,
        n_test=40,
        tags=("additive", "irrelevant_features"),
    ),
    BenchmarkEntry(
        dataset_name="periodic_decay",
        tier=2,
        noise_levels=(0.0, 0.05, 0.15),
        seeds=DEFAULT_SEEDS,
        baselines=DEFAULT_BASELINES,
        prompt_variants=ALL_VARIANTS,
        n_train=60,
        n_test=40,
        tags=("periodic", "decay", "multiplicative_structure"),
    ),
    BenchmarkEntry(
        dataset_name="interaction",
        tier=2,
        noise_levels=(0.0, 0.05, 0.15),
        seeds=DEFAULT_SEEDS,
        baselines=DEFAULT_BASELINES,
        prompt_variants=ALL_VARIANTS,
        n_train=60,
        n_test=40,
        tags=("interaction", "grouped_kernels"),
    ),
    BenchmarkEntry(
        dataset_name="ard_stress",
        tier=2,
        noise_levels=(0.0, 0.05, 0.15),
        seeds=DEFAULT_SEEDS,
        baselines=DEFAULT_BASELINES,
        prompt_variants=ALL_VARIANTS,
        n_train=60,
        n_test=40,
        tags=("ard", "feature_selection"),
    ),
    # ------------------------------------------------------------------
    # Tier 1 — BoTorch test functions
    # ------------------------------------------------------------------
    BenchmarkEntry(
        dataset_name="branin",
        tier=1,
        noise_levels=(0.0, 0.05, 0.15),
        seeds=DEFAULT_SEEDS,
        baselines=DEFAULT_BASELINES,
        prompt_variants=ALL_VARIANTS,
        n_train=60,
        n_test=40,
        tags=("botorch_function", "2d"),
    ),
    BenchmarkEntry(
        dataset_name="hartmann6",
        tier=1,
        noise_levels=(0.0, 0.05, 0.15),
        seeds=DEFAULT_SEEDS,
        baselines=DEFAULT_BASELINES,
        prompt_variants=ALL_VARIANTS,
        n_train=80,
        n_test=40,
        tags=("botorch_function", "6d"),
    ),
    BenchmarkEntry(
        dataset_name="rosenbrock",
        tier=1,
        noise_levels=(0.0, 0.05, 0.15),
        seeds=DEFAULT_SEEDS,
        baselines=DEFAULT_BASELINES,
        prompt_variants=ALL_VARIANTS,
        n_train=80,
        n_test=40,
        tags=("botorch_function", "4d"),
    ),
]

#: Quick lookup by dataset name.
REGISTRY_BY_NAME: dict[str, BenchmarkEntry] = {entry.dataset_name: entry for entry in REGISTRY}


def list_datasets() -> list[str]:
    """Return the names of all registered benchmark datasets.

    Returns:
        Sorted list of dataset name strings.
    """
    return sorted(entry.dataset_name for entry in REGISTRY)


def get_entry(dataset_name: str) -> BenchmarkEntry:
    """Retrieve a registry entry by dataset name.

    Args:
        dataset_name: The benchmark dataset identifier.

    Returns:
        The matching BenchmarkEntry.

    Raises:
        KeyError: If ``dataset_name`` is not registered.
    """
    if dataset_name not in REGISTRY_BY_NAME:
        raise KeyError(f"Dataset '{dataset_name}' not found in registry. Available: {list_datasets()}")
    return REGISTRY_BY_NAME[dataset_name]
