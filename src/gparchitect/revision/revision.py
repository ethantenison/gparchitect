"""
DSL revision and failure recovery for GPArchitect.

Purpose:
    Revises a GPSpec DSL object when model construction, fitting, or prediction fails.
    Applies a sequence of conservative recovery strategies, each producing a simpler
    or more robust DSL specification.

Role in pipeline:
    Natural language → GP DSL → Validation → Model Builder → Fit → Validation → **Recovery**

Inputs:
    - spec: GPSpec — the DSL that caused a failure.
    - error_message: str — the error message from the failed step.
    - attempt: int — the revision attempt number (0-indexed), used to select strategy.

Outputs:
    RevisionResult — a dataclass containing the revised GPSpec, a human-readable
    rationale string, and the strategy applied. Returns None if no further revision
    is possible (all strategies exhausted).

Non-obvious design decisions:
    - Revisions operate on the DSL only, never on the original natural-language string.
    - Each strategy is tried in order of decreasing model complexity, ensuring the
      system always moves toward a simpler, more robust configuration.
    - All revisions are recorded with a rationale string for logging purposes.
    - Strategies are exhausted in a fixed order to keep recovery deterministic.

What this module does NOT do:
    - It does not call the translator to re-parse natural language.
    - It does not fit models — that is the fitting module's responsibility.
    - It does not log experiment history (see logging module).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

from gparchitect.dsl.schema import (
    CompositionType,
    FeatureGroupSpec,
    GPSpec,
    KernelSpec,
    KernelType,
    ModelClass,
    NoiseSpec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recovery strategies in decreasing order of complexity
# ---------------------------------------------------------------------------
_STRATEGIES = [
    "disable_ard",
    "simplify_kernels_to_matern52",
    "remove_priors",
    "switch_to_single_task_gp",
    "use_default_noise",
]


@dataclass
class RevisionResult:
    """Result of a DSL revision attempt.

    Attributes:
        revised_spec: The new, revised GPSpec.
        rationale: Human-readable explanation of what was changed and why.
        strategy: The strategy name that was applied.
    """

    revised_spec: GPSpec
    rationale: str
    strategy: str


def revise_dsl(spec: GPSpec, error_message: str, attempt: int) -> RevisionResult | None:
    """Revise a GPSpec to recover from a fitting or construction failure.

    Applies recovery strategies in order of increasing simplification. Returns None
    when all strategies have been exhausted.

    Args:
        spec: The GPSpec that caused the failure.
        error_message: The error string from the failed step.
        attempt: Zero-indexed attempt number (determines which strategy to apply).

    Returns:
        RevisionResult with the revised spec and rationale, or None if exhausted.
    """
    if attempt >= len(_STRATEGIES):
        logger.warning("All revision strategies exhausted after %d attempts.", attempt)
        return None

    strategy = _STRATEGIES[attempt]
    revised = copy.deepcopy(spec)

    strategy_fn = {
        "disable_ard": _disable_ard,
        "simplify_kernels_to_matern52": _simplify_kernels,
        "remove_priors": _remove_priors,
        "switch_to_single_task_gp": _switch_to_single_task,
        "use_default_noise": _use_default_noise,
    }[strategy]

    rationale = strategy_fn(revised, error_message)

    logger.info("Revision attempt %d: strategy=%s, rationale=%s", attempt, strategy, rationale)

    return RevisionResult(revised_spec=revised, rationale=rationale, strategy=strategy)


# ---------------------------------------------------------------------------
# Individual recovery strategies
# ---------------------------------------------------------------------------


def _disable_ard(spec: GPSpec, error_message: str) -> str:
    """Disable ARD on all feature group kernels."""
    changed = False
    for group in spec.feature_groups:
        if group.kernel.ard:
            group.kernel.ard = False
            changed = True
        for child in group.kernel.children:
            if child.ard:
                child.ard = False
                changed = True

    if changed:
        return f"Disabled ARD on all kernels to reduce hyperparameter count. Error was: {error_message[:120]}"
    return f"ARD was already disabled; no change made. Error was: {error_message[:120]}"


def _simplify_kernels(spec: GPSpec, error_message: str) -> str:
    """Replace all kernels with a simple Matern52, removing compositions."""
    num_features = spec.input_dim
    if spec.task_feature_index is not None:
        num_features -= 1

    simple_kernel = KernelSpec(kernel_type=KernelType.MATERN_52, ard=False)
    continuous_indices = [idx for idx in range(spec.input_dim) if idx != spec.task_feature_index]

    spec.feature_groups = [
        FeatureGroupSpec(
            name="all_features",
            feature_indices=continuous_indices,
            kernel=simple_kernel,
        )
    ]
    spec.group_composition = CompositionType.ADDITIVE
    return f"Simplified all kernels to Matern52 with no compositions. Error was: {error_message[:120]}"


def _remove_priors(spec: GPSpec, error_message: str) -> str:
    """Remove all priors from kernels and noise."""
    for group in spec.feature_groups:
        group.kernel.lengthscale_prior = None
        group.kernel.outputscale_prior = None
        group.kernel.period_prior = None
        for child in group.kernel.children:
            child.lengthscale_prior = None
            child.outputscale_prior = None
    spec.noise.prior = None
    return f"Removed all priors from kernels and noise. Error was: {error_message[:120]}"


def _switch_to_single_task(spec: GPSpec, error_message: str) -> str:
    """Switch model class to SingleTaskGP with safe defaults."""
    spec.model_class = ModelClass.SINGLE_TASK_GP
    spec.task_feature_index = None
    spec.multitask_rank = None
    spec.output_dim = 1

    continuous_indices = list(range(spec.input_dim))
    spec.feature_groups = [
        FeatureGroupSpec(
            name="all_features",
            feature_indices=continuous_indices,
            kernel=KernelSpec(kernel_type=KernelType.MATERN_52, ard=False),
        )
    ]
    return f"Switched to SingleTaskGP with Matern52 kernel. Error was: {error_message[:120]}"


def _use_default_noise(spec: GPSpec, error_message: str) -> str:
    """Reset noise to learnable BoTorch default."""
    spec.noise = NoiseSpec(fixed=False, noise_value=None, prior=None)
    return f"Reset noise to learnable BoTorch default. Error was: {error_message[:120]}"
