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
    - input_feature_names: list[str] | None — optional continuous input column names.

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
    - It does not inspect the actual data values.
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
_FEATURE_PREPOSITION_PATTERN = re.compile(r"\b(?:on|for|across|over|applied to|using)\b", re.IGNORECASE)


def _match_kernel_type(instruction: str) -> KernelType | None:
    """Return the first matching kernel type from the instruction, if any."""
    for pattern, kernel_type in _KERNEL_KEYWORDS:
        if pattern.search(instruction):
            return kernel_type
    return None


def _detect_kernel_type(instruction: str) -> KernelType:
    """Return the first matching kernel type from the instruction, defaulting to Matern52."""
    return _match_kernel_type(instruction) or KernelType.MATERN_52


def _detect_model_class(instruction: str, task_feature_index: int | None) -> ModelClass:
    """Return the detected model class from the instruction."""
    if task_feature_index is not None:
        return ModelClass.MULTI_TASK_GP
    for pattern, model_class in _MODEL_KEYWORDS:
        if pattern.search(instruction):
            return model_class
    return ModelClass.SINGLE_TASK_GP


def _detect_explicit_composition(instruction: str) -> CompositionType | None:
    """Detect an explicit inter-group kernel composition directive from the instruction."""
    if _MULTIPLICATIVE_PATTERN.search(instruction):
        return CompositionType.MULTIPLICATIVE
    if _ADDITIVE_PATTERN.search(instruction):
        return CompositionType.ADDITIVE
    return None


def _detect_noise(instruction: str) -> NoiseSpec:
    """Build a NoiseSpec from the instruction, defaulting to learnable noise."""
    if _FIXED_NOISE_PATTERN.search(instruction):
        return NoiseSpec(fixed=True, noise_value=1e-4)
    return NoiseSpec(fixed=False)


def _find_referenced_features(clause: str, input_feature_names: list[str]) -> list[int]:
    """Return continuous feature indices referenced in a clause by exact column name."""
    clause_lower = clause.lower()
    referenced_indices: list[int] = []
    for index, feature_name in enumerate(input_feature_names):
        feature_pattern = re.compile(rf"(?<!\w){re.escape(feature_name.lower())}(?!\w)")
        if feature_pattern.search(clause_lower):
            referenced_indices.append(index)
    return referenced_indices


def _find_kernel_mentions(instruction: str) -> list[tuple[int, int, KernelType]]:
    """Return non-overlapping kernel mentions in instruction order."""
    matches: list[tuple[int, int, int, KernelType]] = []
    for priority, (pattern, kernel_type) in enumerate(_KERNEL_KEYWORDS):
        for match in pattern.finditer(instruction):
            matches.append((match.start(), match.end(), priority, kernel_type))

    matches.sort(key=lambda item: (item[0], item[2], -(item[1] - item[0])))

    resolved_matches: list[tuple[int, int, KernelType]] = []
    last_end = -1
    for start, end, _, kernel_type in matches:
        if start < last_end:
            continue
        resolved_matches.append((start, end, kernel_type))
        last_end = end

    return resolved_matches


def _extract_feature_groups(
    instruction: str,
    input_feature_names: list[str] | None,
    use_ard: bool,
) -> list[FeatureGroupSpec]:
    """Extract feature-specific kernel groups from natural language when column names are available."""
    if not input_feature_names:
        return []

    feature_groups: list[FeatureGroupSpec] = []
    kernel_mentions = _find_kernel_mentions(instruction)
    for index, (kernel_start, kernel_end, kernel_type) in enumerate(kernel_mentions):
        clause_end = len(instruction)
        if index + 1 < len(kernel_mentions):
            clause_end = kernel_mentions[index + 1][0]

        clause_tail = instruction[kernel_end:clause_end]
        preposition_match = _FEATURE_PREPOSITION_PATTERN.search(clause_tail)
        if preposition_match is None:
            continue

        feature_text = clause_tail[preposition_match.end() :]
        feature_indices = _find_referenced_features(feature_text, input_feature_names)
        if not feature_indices:
            continue

        feature_groups.append(
            FeatureGroupSpec(
                name="_".join(input_feature_names[index] for index in feature_indices),
                feature_indices=feature_indices,
                kernel=KernelSpec(kernel_type=kernel_type, ard=use_ard),
            )
        )

    return feature_groups


def translate_to_dsl(
    instruction: str,
    input_dim: int,
    output_dim: int = 1,
    task_feature_index: int | None = None,
    input_feature_names: list[str] | None = None,
) -> GPSpec:
    """Translate a natural-language GP instruction into a GPSpec DSL object.

    All unspecified aspects are filled with safe BoTorch defaults.

    Args:
        instruction: Free-text description of the desired GP architecture.
        input_dim: Total number of continuous input features.
        output_dim: Number of model outputs (default 1).
        task_feature_index: Column index of the task indicator for MultiTaskGP.
        input_feature_names: Optional continuous feature names used for column-aware parsing.

    Returns:
        GPSpec: A fully-specified, deterministic DSL object.
    """
    if input_dim < 1:
        raise ValueError(f"input_dim must be at least 1, got {input_dim}")
    if output_dim < 1:
        raise ValueError(f"output_dim must be at least 1, got {output_dim}")

    kernel_type = _detect_kernel_type(instruction)
    model_class = _detect_model_class(instruction, task_feature_index)
    explicit_composition = _detect_explicit_composition(instruction)
    noise = _detect_noise(instruction)
    use_ard = bool(_ARD_PATTERN.search(instruction))
    feature_groups = _extract_feature_groups(instruction, input_feature_names, use_ard)

    if feature_groups:
        composition = explicit_composition
        if composition is None:
            composition = CompositionType.HIERARCHICAL if len(feature_groups) > 1 else CompositionType.ADDITIVE

    # For ModelListGP, create one feature group per output when no column-specific mapping is provided.
    elif model_class == ModelClass.MODEL_LIST_GP:
        composition = explicit_composition or CompositionType.ADDITIVE
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
        composition = explicit_composition or CompositionType.ADDITIVE
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
        f"model={model_class.value}, kernel={kernel_type.value}, ard={use_ard}, groups={len(feature_groups)}"
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
