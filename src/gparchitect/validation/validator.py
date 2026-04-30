"""
DSL validation for GPArchitect.

Purpose:
    Validates a GPSpec DSL object for dimensional consistency, kernel compatibility,
    model-class compatibility, valid priors, and ARD usage before model construction.

Role in pipeline:
    Natural language → GP DSL → **Validation** → Model Builder → Fit → Validation → Recovery

Inputs:
    GPSpec — a DSL object produced by the translator or the revision module.

Outputs:
    ValidationResult — a dataclass containing a list of error messages and a boolean
    indicating whether the spec is valid. Raises ValueError on fatal errors when
    validate_or_raise() is used.

Non-obvious design decisions:
    - Validation is purely structural and does not require torch or BoTorch.
    - Errors are collected (not raised immediately) so that callers receive a full
      list of problems rather than failing on the first one.
    - Warnings (non-fatal issues) are distinct from errors.

What this module does NOT do:
    - It does not modify the DSL — that is the revision module's responsibility.
    - It does not construct or fit models.
    - It does not check numerical properties of prior parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gparchitect.dsl.schema import (
    CompositionType,
    GPSpec,
    KernelType,
    ModelClass,
    PriorDistribution,
    PriorSpec,
    RecencyWeightingMode,
    SpectralMixtureInitialization,
)

logger = logging.getLogger(__name__)

_PERIODIC_ONLY_KERNELS = {KernelType.PERIODIC}
_MULTITASK_MODELS = {ModelClass.MULTI_TASK_GP}
_MODEL_LIST_MODELS = {ModelClass.MODEL_LIST_GP}
_SUPPORTED_PRIOR_DISTRIBUTIONS = {
    PriorDistribution.NORMAL,
    PriorDistribution.LOG_NORMAL,
    PriorDistribution.GAMMA,
    PriorDistribution.HALF_CAUCHY,
    PriorDistribution.UNIFORM,
}
_REQUIRED_PRIOR_PARAMS = {
    PriorDistribution.NORMAL: {"loc", "scale"},
    PriorDistribution.LOG_NORMAL: {"loc", "scale"},
    PriorDistribution.GAMMA: {"concentration", "rate"},
    PriorDistribution.HALF_CAUCHY: {"scale"},
    PriorDistribution.UNIFORM: {"a", "b"},
}
_LENGTHSCALE_PRIOR_KERNELS = {
    KernelType.RBF,
    KernelType.RQ,
    KernelType.MATERN_12,
    KernelType.MATERN_32,
    KernelType.MATERN_52,
    KernelType.PERIODIC,
}


@dataclass
class ValidationResult:
    """Result of validating a GPSpec.

    Attributes:
        errors: List of error messages that make the spec invalid.
        warnings: List of non-fatal warning messages.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if there are no errors."""
        return len(self.errors) == 0

    def __str__(self) -> str:
        parts: list[str] = []
        if self.errors:
            parts.append("Errors:\n" + "\n".join(f"  - {e}" for e in self.errors))
        if self.warnings:
            parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in self.warnings))
        return "\n".join(parts) if parts else "Valid"


def validate_dsl(spec: GPSpec) -> ValidationResult:
    """Validate a GPSpec for structural and semantic correctness.

    Args:
        spec: The GPSpec DSL object to validate.

    Returns:
        ValidationResult with lists of errors and warnings.
    """
    result = ValidationResult()

    _check_dimensions(spec, result)
    _check_feature_groups(spec, result)
    _check_model_class_consistency(spec, result)
    _check_mean_spec(spec, result)
    _check_noise(spec, result)
    _check_execution(spec, result)
    _check_priors(spec, result)

    if result.is_valid:
        logger.info("DSL validation passed for model_class=%s", spec.model_class.value)
    else:
        logger.warning("DSL validation failed: %d error(s)", len(result.errors))

    return result


def validate_or_raise(spec: GPSpec) -> None:
    """Validate a GPSpec and raise ValueError if invalid.

    Args:
        spec: The GPSpec DSL object to validate.

    Raises:
        ValueError: If the spec contains any validation errors.
    """
    result = validate_dsl(spec)
    if not result.is_valid:
        raise ValueError(f"Invalid GPSpec:\n{result}")


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _check_dimensions(spec: GPSpec, result: ValidationResult) -> None:
    if spec.input_dim < 1:
        result.errors.append(f"input_dim must be >= 1, got {spec.input_dim}")
    if spec.output_dim < 1:
        result.errors.append(f"output_dim must be >= 1, got {spec.output_dim}")


def _check_feature_groups(spec: GPSpec, result: ValidationResult) -> None:
    if not spec.feature_groups:
        result.errors.append("At least one feature group must be specified.")
        return

    all_indices: list[int] = []
    for group_idx, group in enumerate(spec.feature_groups):
        if not group.feature_indices:
            result.errors.append(f"Feature group '{group.name}' (index {group_idx}) has no feature indices.")
        for idx in group.feature_indices:
            if idx < 0 or idx >= spec.input_dim:
                result.errors.append(
                    f"Feature group '{group.name}': index {idx} is out of range "
                    f"for input_dim={spec.input_dim}."
                )
            if spec.task_feature_index is not None and idx == spec.task_feature_index:
                result.errors.append(
                    f"Feature group '{group.name}': task_feature_index {idx} "
                    "must not appear in continuous feature groups."
                )
        all_indices.extend(group.feature_indices)

        _check_kernel_spec(group.name, group.kernel, len(group.feature_indices), result)

    if len(spec.feature_groups) > 1 and spec.group_composition == CompositionType.NONE:
        result.errors.append(
            "Multiple feature groups require additive, multiplicative, or hierarchical composition; "
            "composition=none is only valid for a single feature group."
        )


def _check_kernel_spec(group_name: str, kernel, feature_count: int, result: ValidationResult) -> None:  # noqa: ANN001
    """Recursively validate a KernelSpec."""
    if kernel.kernel_type in _PERIODIC_ONLY_KERNELS:
        # Periodic kernels require a single feature dimension
        pass  # Not enforced at DSL level; handled by builder

    if kernel.ard and not kernel.children:
        # ARD on composed kernels is not directly supported at the top level
        pass

    if kernel.kernel_type == KernelType.RQ and kernel.rq_alpha is not None and kernel.rq_alpha <= 0:
        result.errors.append(
            f"Feature group '{group_name}': RQ alpha must be > 0, got {kernel.rq_alpha}."
        )

    if kernel.kernel_type == KernelType.PERIODIC and kernel.period_length is not None and kernel.period_length <= 0:
        result.errors.append(
            f"Feature group '{group_name}': Periodic period_length must be > 0, got {kernel.period_length}."
        )

    if kernel.kernel_type == KernelType.POLYNOMIAL:
        if kernel.polynomial_power is not None and kernel.polynomial_power < 1:
            result.errors.append(
                f"Feature group '{group_name}': Polynomial power must be >= 1, got {kernel.polynomial_power}."
            )
        if kernel.polynomial_offset is not None and kernel.polynomial_offset <= 0:
            result.errors.append(
                f"Feature group '{group_name}': Polynomial offset must be > 0, got {kernel.polynomial_offset}."
            )

    if kernel.kernel_type == KernelType.SPECTRAL_MIXTURE:
        if kernel.num_mixtures is not None and kernel.num_mixtures < 1:
            result.errors.append(
                f"Feature group '{group_name}': SpectralMixture num_mixtures must be >= 1, "
                f"got {kernel.num_mixtures}."
            )
        if kernel.spectral_init not in {
            SpectralMixtureInitialization.FROM_DATA,
            SpectralMixtureInitialization.FROM_EMPIRICAL_SPECTRUM,
        }:
            result.errors.append(
                f"Feature group '{group_name}': unsupported spectral_init '{kernel.spectral_init}'."
            )
        if not kernel.ard:
            result.warnings.append(
                f"Feature group '{group_name}': SpectralMixtureKernel requires ard_num_dims to match the "
                "active dimensionality; the builder will enforce this regardless of the DSL ard flag."
            )

    if kernel.kernel_type == KernelType.INFINITE_WIDTH_BNN:
        if kernel.bnn_depth is not None and kernel.bnn_depth < 1:
            result.errors.append(
                f"Feature group '{group_name}': InfiniteWidthBNN depth must be >= 1, got {kernel.bnn_depth}."
            )

    if kernel.kernel_type == KernelType.EXPONENTIAL_DECAY:
        if feature_count != 1:
            result.errors.append(
                f"Feature group '{group_name}': ExponentialDecayKernel requires exactly one active feature, "
                f"got {feature_count}."
            )
        if kernel.exponential_decay_power is not None and kernel.exponential_decay_power <= 0:
            result.errors.append(
                f"Feature group '{group_name}': ExponentialDecay power must be > 0, "
                f"got {kernel.exponential_decay_power}."
            )
        if kernel.exponential_decay_offset is not None and kernel.exponential_decay_offset <= 0:
            result.errors.append(
                f"Feature group '{group_name}': ExponentialDecay offset must be > 0, "
                f"got {kernel.exponential_decay_offset}."
            )

    if kernel.kernel_type == KernelType.CHANGEPOINT:
        if feature_count != 1:
            result.errors.append(
                f"Feature group '{group_name}': Changepoint kernel requires exactly one active feature "
                f"(a time-like input), got {feature_count}."
            )
        if len(kernel.children) != 2:  # noqa: PLR2004
            result.errors.append(
                f"Feature group '{group_name}': Changepoint kernel requires exactly 2 children "
                f"(before and after kernels), got {len(kernel.children)}."
            )
        if kernel.changepoint_steepness is not None and kernel.changepoint_steepness <= 0:
            result.errors.append(
                f"Feature group '{group_name}': Changepoint steepness must be > 0, "
                f"got {kernel.changepoint_steepness}."
            )

    for child in kernel.children:
        _check_kernel_spec(group_name, child, feature_count, result)


def _check_model_class_consistency(spec: GPSpec, result: ValidationResult) -> None:
    if spec.model_class in _MULTITASK_MODELS:
        if spec.task_feature_index is None:
            result.errors.append("MultiTaskGP requires task_feature_index to be set.")
        if spec.task_values is None:
            result.errors.append("MultiTaskGP requires explicit task_values.")
        if spec.output_dim != 1:
            result.errors.append(
                f"MultiTaskGP currently supports long-format training data with output_dim=1, got {spec.output_dim}."
            )
        if spec.multitask_rank is not None and spec.multitask_rank < 1:
            result.errors.append(f"multitask_rank must be >= 1, got {spec.multitask_rank}.")
        if spec.task_values is not None:
            if not spec.task_values:
                result.errors.append("MultiTaskGP task_values must not be empty when provided.")
            if any(task_value < 0 for task_value in spec.task_values):
                result.errors.append("MultiTaskGP task_values must all be >= 0.")
            if len(set(spec.task_values)) != len(spec.task_values):
                result.errors.append("MultiTaskGP task_values must be unique.")

    elif spec.model_class in _MODEL_LIST_MODELS:
        if spec.output_dim < 2:
            result.warnings.append("ModelListGP with output_dim=1 is unusual; consider SingleTaskGP.")

    elif spec.model_class == ModelClass.SINGLE_TASK_GP:
        if spec.task_feature_index is not None:
            result.warnings.append(
                "task_feature_index is set but model_class is SingleTaskGP; "
                "the task column will be ignored."
            )

    if spec.model_class != ModelClass.MULTI_TASK_GP and spec.task_values is not None:
        result.errors.append("task_values may only be set for MultiTaskGP.")


def _check_mean_spec(spec: GPSpec, result: ValidationResult) -> None:
    if spec.model_class == ModelClass.SINGLE_TASK_GP and spec.output_means:
        result.errors.append("SingleTaskGP does not support output_means; use mean for a shared mean function.")

    if spec.model_class == ModelClass.MODEL_LIST_GP:
        for output_index in spec.output_means:
            if output_index < 0 or output_index >= spec.output_dim:
                result.errors.append(
                    f"ModelListGP output_means index {output_index} is out of range for output_dim={spec.output_dim}."
                )

    if spec.model_class == ModelClass.MULTI_TASK_GP:
        if spec.output_means and spec.task_values is None:
            result.errors.append(
                "MultiTaskGP targeted output_means require explicit task_values so the validator can check task scope."
            )
        for task_index in spec.output_means:
            if task_index < 0:
                result.errors.append(f"MultiTaskGP output_means task index must be >= 0, got {task_index}.")
            elif spec.task_values is not None and task_index not in spec.task_values:
                result.errors.append(
                    f"MultiTaskGP output_means task index {task_index} is not declared in task_values."
                )


def _check_noise(spec: GPSpec, result: ValidationResult) -> None:
    if spec.noise.fixed and spec.noise.noise_value is None:
        result.errors.append("noise.fixed=True but noise.noise_value is None; a value must be provided.")
    if spec.noise.noise_value is not None and spec.noise.noise_value < 0:
        result.errors.append(f"noise.noise_value must be non-negative, got {spec.noise.noise_value}.")
    if spec.noise.fixed and spec.noise.prior is not None:
        result.errors.append("noise.prior is not supported when noise.fixed=True.")
    if spec.noise.heteroskedastic_noise:
        result.errors.append(
            "noise.heteroskedastic_noise=True is not yet supported. "
            "Heteroskedastic noise is a planned Tier 2 feature. "
            "Set noise.heteroskedastic_noise=False (default) to use the standard homoskedastic noise model."
        )


def _check_execution(spec: GPSpec, result: ValidationResult) -> None:
    if spec.model_class == ModelClass.MULTI_TASK_GP and spec.execution.outcome_standardization:
        result.errors.append("MultiTaskGP does not support outcome_standardization in the current contract.")

    rw = spec.execution.recency_weighting
    if rw is not None:
        if rw.time_feature_index < 0 or rw.time_feature_index >= spec.input_dim:
            result.errors.append(
                f"recency_weighting.time_feature_index={rw.time_feature_index} is out of range "
                f"for input_dim={spec.input_dim}."
            )
        if spec.task_feature_index is not None and rw.time_feature_index == spec.task_feature_index:
            result.errors.append(
                f"recency_weighting.time_feature_index={rw.time_feature_index} must not be the "
                "task feature index."
            )
        if rw.mode == RecencyWeightingMode.SLIDING_WINDOW:
            if rw.window_size is None:
                result.errors.append(
                    "recency_weighting.window_size must be set when mode=sliding_window."
                )
            elif rw.window_size <= 0:
                result.errors.append(
                    f"recency_weighting.window_size must be > 0, got {rw.window_size}."
                )
        if rw.mode == RecencyWeightingMode.EXPONENTIAL_DISCOUNT:
            if rw.discount_rate is None:
                result.errors.append(
                    "recency_weighting.discount_rate must be set when mode=exponential_discount."
                )
            elif rw.discount_rate <= 0:
                result.errors.append(
                    f"recency_weighting.discount_rate must be > 0, got {rw.discount_rate}."
                )
        if rw.min_weight is not None and not (0 < rw.min_weight < 1):
            result.errors.append(
                f"recency_weighting.min_weight must be in (0, 1), got {rw.min_weight}."
            )


def _check_priors(spec: GPSpec, result: ValidationResult) -> None:
    for group in spec.feature_groups:
        _check_kernel_priors(group.name, group.kernel, result)

    if spec.noise.prior is not None:
        _check_prior_spec("noise.prior", spec.noise.prior, result)


def _check_prior_spec(location: str, prior: PriorSpec, result: ValidationResult) -> None:
    if not prior.distribution:
        result.errors.append(f"{location} has no distribution name.")
        return
    if prior.distribution not in _SUPPORTED_PRIOR_DISTRIBUTIONS:
        supported = ", ".join(sorted(_SUPPORTED_PRIOR_DISTRIBUTIONS))
        result.errors.append(
            f"{location} uses unsupported distribution '{prior.distribution}'. Supported distributions: {supported}."
        )
        return

    missing_params = sorted(_REQUIRED_PRIOR_PARAMS[prior.distribution] - set(prior.params))
    if missing_params:
        result.errors.append(
            f"{location} is missing required parameters for {prior.distribution}: {', '.join(missing_params)}."
        )


def _check_kernel_priors(group_name: str, kernel, result: ValidationResult) -> None:  # noqa: ANN001
    if kernel.lengthscale_prior is not None:
        _check_prior_spec(f"Feature group '{group_name}': lengthscale_prior", kernel.lengthscale_prior, result)
        if kernel.children:
            result.errors.append(
                f"Feature group '{group_name}': lengthscale_prior is only supported on leaf kernels."
            )
        elif kernel.kernel_type not in _LENGTHSCALE_PRIOR_KERNELS:
            result.errors.append(
                f"Feature group '{group_name}': lengthscale_prior is not supported for {kernel.kernel_type.value}."
            )

    if kernel.outputscale_prior is not None:
        _check_prior_spec(f"Feature group '{group_name}': outputscale_prior", kernel.outputscale_prior, result)
        if kernel.children:
            result.errors.append(
                f"Feature group '{group_name}': outputscale_prior is only supported on leaf kernels."
            )
        elif kernel.kernel_type == KernelType.SPECTRAL_MIXTURE:
            result.errors.append(
                f"Feature group '{group_name}': outputscale_prior is not supported for SpectralMixture."
            )

    if kernel.period_prior is not None:
        _check_prior_spec(f"Feature group '{group_name}': period_prior", kernel.period_prior, result)
        if kernel.children:
            result.errors.append(
                f"Feature group '{group_name}': period_prior is only supported on leaf kernels."
            )
        elif kernel.kernel_type != KernelType.PERIODIC:
            result.errors.append(
                f"Feature group '{group_name}': period_prior is only supported for Periodic kernels."
            )

    for child in kernel.children:
        _check_kernel_priors(group_name, child, result)
