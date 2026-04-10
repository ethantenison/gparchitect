"""
benchmark_v1.baselines.configs — baseline GP model configurations for benchmark_v1.

Purpose:
    Defines a small set of strong GP baselines that GPArchitect results are
    compared against.  Each baseline is a fixed DSL specification (GPSpec) that
    can be built and fit using the standard GPArchitect pipeline without going
    through natural-language translation.

Role in benchmark pipeline:
    Baselines → Runner → Evaluation → Report

Inputs:
    input_dim: int — number of input features (used to set feature_indices).
    output_dim: int — number of output features.

Outputs:
    GPSpec — a validated DSL specification ready for build_model_from_dsl.

Baselines defined:
    1. default_singletask — SingleTaskGP with Matern52, no ARD, default settings.
    2. matern52_ard      — SingleTaskGP with Matern52 and full ARD.

Non-obvious design decisions:
    - Baselines do NOT go through the translator; they are constructed directly
      as GPSpec objects to avoid any dependency on translation quality.
    - All baselines use the same execution settings as GPArchitect runs
      (input_scaling=True, outcome_standardization=True) for a fair comparison.

What this module does NOT do:
    - It does not fit models (the runner handles that).
    - It does not define oracle specs (those are dataset-specific and defined in
      the registry).
"""

from __future__ import annotations

import logging
from typing import Callable

from gparchitect.dsl.schema import (
    CompositionType,
    ExecutionSpec,
    FeatureGroupSpec,
    GPSpec,
    KernelSpec,
    KernelType,
    ModelClass,
    NoiseSpec,
)

logger = logging.getLogger(__name__)


def make_default_singletask_spec(input_dim: int, output_dim: int = 1) -> GPSpec:
    """Build a default SingleTaskGP spec with Matern52 (no ARD).

    This is the primary baseline: a standard BoTorch SingleTaskGP with
    a single Matern52 kernel over all inputs, no ARD, and default settings.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output dimensions (must be 1 for SingleTaskGP).

    Returns:
        GPSpec for a default SingleTaskGP.
    """
    return GPSpec(
        model_class=ModelClass.SINGLE_TASK_GP,
        feature_groups=[
            FeatureGroupSpec(
                name="all_inputs",
                feature_indices=list(range(input_dim)),
                kernel=KernelSpec(
                    kernel_type=KernelType.MATERN_52,
                    ard=False,
                ),
            )
        ],
        noise=NoiseSpec(fixed=False),
        execution=ExecutionSpec(input_scaling=True, outcome_standardization=True),
        input_dim=input_dim,
        output_dim=output_dim,
        group_composition=CompositionType.ADDITIVE,
        description="Baseline: default SingleTaskGP with Matern52 (no ARD).",
    )


def make_matern52_ard_spec(input_dim: int, output_dim: int = 1) -> GPSpec:
    """Build a SingleTaskGP spec with Matern52 and full ARD.

    This is the ``simple kernel heuristic'' baseline: Matern52 with
    a separate lengthscale per input dimension.  This is a strong but
    conventional choice for continuous regression problems.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output dimensions (must be 1 for SingleTaskGP).

    Returns:
        GPSpec for a Matern52 ARD SingleTaskGP.
    """
    return GPSpec(
        model_class=ModelClass.SINGLE_TASK_GP,
        feature_groups=[
            FeatureGroupSpec(
                name="all_inputs_ard",
                feature_indices=list(range(input_dim)),
                kernel=KernelSpec(
                    kernel_type=KernelType.MATERN_52,
                    ard=True,
                ),
            )
        ],
        noise=NoiseSpec(fixed=False),
        execution=ExecutionSpec(input_scaling=True, outcome_standardization=True),
        input_dim=input_dim,
        output_dim=output_dim,
        group_composition=CompositionType.ADDITIVE,
        description="Baseline: Matern52 with full ARD over all inputs.",
    )


#: Available baseline names → factory functions.
BASELINE_FACTORIES: dict[str, Callable[[int, int], GPSpec]] = {
    "default_singletask": make_default_singletask_spec,
    "matern52_ard": make_matern52_ard_spec,
}


def list_baselines() -> list[str]:
    """Return the names of all available baselines.

    Returns:
        Sorted list of baseline name strings.
    """
    return sorted(BASELINE_FACTORIES.keys())
