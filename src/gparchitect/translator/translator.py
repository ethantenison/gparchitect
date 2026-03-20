"""
Natural language to GP DSL translator for GPArchitect.

Purpose:
    Converts a natural-language instruction string into a structured GPSpec (DSL object).
    Missing or ambiguous details are filled with BoTorch-compatible defaults.

Role in pipeline:
    **Natural language** → GP DSL → Validation → Model Builder → Fit → Validation → Recovery

Inputs:
    - instruction: str — a free-text description of the desired GP model architecture.
    - input_dim: int — total number of input features (from the supplied DataFrame).
    - output_dim: int — number of output dimensions.
    - task_feature_index: int | None — column index of the task indicator, if any.

Outputs:
    GPSpec — a fully-specified DSL object suitable for validation and model construction.

Non-obvious design decisions:
    - Translation is rule-based (keyword matching) rather than LLM-driven in v1, ensuring
      deterministic output without external API calls.
    - Unrecognised instructions fall back to a safe SingleTaskGP + Matern52 default.
    - Kernel keywords are matched case-insensitively against the instruction string.
    - ARD is enabled when the instruction mentions "ard" or "automatic relevance".

What this module does NOT do:
    - It does not call any external language model or API.
    - It does not validate the resulting DSL (see validation module).
    - It does not handle DataFrame column names — only integer indices are used.
"""

from __future__ import annotations

import logging
import re

from gparchitect.dsl.schema import (
    CompositionType,
    FeatureGroupSpec,
    GPSpec,
    KernelSpec,
    KernelType,
    ModelClass,
    NoiseSpec,
    PriorSpec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword → kernel type mapping (ordered by specificity)
# ---------------------------------------------------------------------------
_KERNEL_KEYWORDS: list[tuple[re.Pattern[str], KernelType]] = [
    (re.compile(r"\bmatern.?1[/\-_]?2\b", re.IGNORECASE), KernelType.MATERN_12),
    (re.compile(r"\bmatern.?3[/\-_]?2\b", re.IGNORECASE), KernelType.MATERN_32),
    (re.compile(r"\bmatern.?5[/\-_]?2\b", re.IGNORECASE), KernelType.MATERN_52),
    (re.compile(r"\bmatern\b", re.IGNORECASE), KernelType.MATERN_52),  # default Matern
    (re.compile(r"\brbf\b|\bsquared.?exponential\b|\bgaussian.?kernel\b", re.IGNORECASE), KernelType.RBF),
    (re.compile(r"\bperiodic\b", re.IGNORECASE), KernelType.PERIODIC),
    (re.compile(r"\blinear\b", re.IGNORECASE), KernelType.LINEAR),
    (re.compile(r"\bpolynomial\b", re.IGNORECASE), KernelType.POLYNOMIAL),
]

# ---------------------------------------------------------------------------
# Keyword → model class mapping
# ---------------------------------------------------------------------------
_MODEL_KEYWORDS: list[tuple[re.Pattern[str], ModelClass]] = [
    (re.compile(r"\bmulti.?task\b|\bmultitask\b|\btask.?aware\b", re.IGNORECASE), ModelClass.MULTI_TASK_GP),
    (re.compile(r"\bmodel.?list|\bmodellist|\bindependent.?output", re.IGNORECASE), ModelClass.MODEL_LIST_GP),
    (re.compile(r"\bsingle.?task\b|\bsingletask\b", re.IGNORECASE), ModelClass.SINGLE_TASK_GP),
]

_ARD_PATTERN = re.compile(r"\bard\b|\bautomatic.?relevance\b", re.IGNORECASE)
_FIXED_NOISE_PATTERN = re.compile(r"\bfixed.?noise\b|\bnoise.?free\b|\bnoiseless\b", re.IGNORECASE)
_ADDITIVE_PATTERN = re.compile(r"\badditive\b|\bsum.?of.?kernel\b", re.IGNORECASE)
_MULTIPLICATIVE_PATTERN = re.compile(r"\bmultiplicative\b|\bproduct.?of.?kernel\b", re.IGNORECASE)


def _detect_kernel_type(instruction: str) -> KernelType:
    """Return the first matching kernel type from the instruction, defaulting to Matern52."""
    for pattern, kernel_type in _KERNEL_KEYWORDS:
        if pattern.search(instruction):
            return kernel_type
    return KernelType.MATERN_52


def _detect_model_class(instruction: str, task_feature_index: int | None) -> ModelClass:
    """Return the detected model class from the instruction."""
    if task_feature_index is not None:
        return ModelClass.MULTI_TASK_GP
    for pattern, model_class in _MODEL_KEYWORDS:
        if pattern.search(instruction):
            return model_class
    return ModelClass.SINGLE_TASK_GP


def _detect_composition(instruction: str) -> CompositionType:
    """Detect the inter-group kernel composition from the instruction."""
    if _MULTIPLICATIVE_PATTERN.search(instruction):
        return CompositionType.MULTIPLICATIVE
    return CompositionType.ADDITIVE


def _detect_noise(instruction: str) -> NoiseSpec:
    """Build a NoiseSpec from the instruction, defaulting to learnable noise."""
    if _FIXED_NOISE_PATTERN.search(instruction):
        return NoiseSpec(fixed=True, noise_value=1e-4)
    return NoiseSpec(fixed=False)


def translate_to_dsl(
    instruction: str,
    input_dim: int,
    output_dim: int = 1,
    task_feature_index: int | None = None,
) -> GPSpec:
    """Translate a natural-language GP instruction into a GPSpec DSL object.

    All unspecified aspects are filled with safe BoTorch defaults.

    Args:
        instruction: Free-text description of the desired GP architecture.
        input_dim: Total number of continuous input features.
        output_dim: Number of model outputs (default 1).
        task_feature_index: Column index of the task indicator for MultiTaskGP.

    Returns:
        GPSpec: A fully-specified, deterministic DSL object.
    """
    if input_dim < 1:
        raise ValueError(f"input_dim must be at least 1, got {input_dim}")
    if output_dim < 1:
        raise ValueError(f"output_dim must be at least 1, got {output_dim}")

    kernel_type = _detect_kernel_type(instruction)
    model_class = _detect_model_class(instruction, task_feature_index)
    composition = _detect_composition(instruction)
    noise = _detect_noise(instruction)
    use_ard = bool(_ARD_PATTERN.search(instruction))

    # For ModelListGP, create one feature group per output; otherwise one group for all inputs.
    if model_class == ModelClass.MODEL_LIST_GP:
        continuous_indices = [idx for idx in range(input_dim) if idx != task_feature_index]
        feature_groups = [
            FeatureGroupSpec(
                name=f"output_{output_idx}_features",
                feature_indices=continuous_indices,
                kernel=KernelSpec(kernel_type=kernel_type, ard=use_ard),
            )
            for output_idx in range(output_dim)
        ]
    else:
        continuous_indices = [idx for idx in range(input_dim) if idx != task_feature_index]
        feature_groups = [
            FeatureGroupSpec(
                name="all_features",
                feature_indices=continuous_indices,
                kernel=KernelSpec(kernel_type=kernel_type, ard=use_ard),
            )
        ]

    multitask_rank: int | None = None
    if model_class == ModelClass.MULTI_TASK_GP:
        multitask_rank = min(output_dim, 2)

    description = (
        f"Translated from: '{instruction[:80]}{'...' if len(instruction) > 80 else ''}' | "
        f"model={model_class.value}, kernel={kernel_type.value}, ard={use_ard}"
    )

    spec = GPSpec(
        model_class=model_class,
        feature_groups=feature_groups,
        noise=noise,
        input_dim=input_dim,
        output_dim=output_dim,
        task_feature_index=task_feature_index,
        multitask_rank=multitask_rank,
        group_composition=composition,
        description=description,
    )

    logger.info("Translated instruction to DSL: model=%s, kernel=%s", model_class.value, kernel_type.value)
    return spec
