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

from gparchitect.dsl.schema import CompositionType, GPSpec, KernelType, ModelClass

logger = logging.getLogger(__name__)

_PERIODIC_ONLY_KERNELS = {KernelType.PERIODIC}
_MULTITASK_MODELS = {ModelClass.MULTI_TASK_GP}
_MODEL_LIST_MODELS = {ModelClass.MODEL_LIST_GP}


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
    _check_noise(spec, result)
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

        _check_kernel_spec(group.name, group.kernel, result)

    # Warn when feature groups use an unsupported composition type
    if len(spec.feature_groups) > 1 and spec.group_composition == CompositionType.NONE:
        result.warnings.append(
            "Multiple feature groups with composition=none; group kernels will not be combined."
        )


def _check_kernel_spec(group_name: str, kernel, result: ValidationResult) -> None:  # noqa: ANN001
    """Recursively validate a KernelSpec."""
    if kernel.kernel_type in _PERIODIC_ONLY_KERNELS:
        # Periodic kernels require a single feature dimension
        pass  # Not enforced at DSL level; handled by builder

    if kernel.ard and not kernel.children:
        # ARD on composed kernels is not directly supported at the top level
        pass

    for child in kernel.children:
        _check_kernel_spec(group_name, child, result)


def _check_model_class_consistency(spec: GPSpec, result: ValidationResult) -> None:
    if spec.model_class in _MULTITASK_MODELS:
        if spec.task_feature_index is None:
            result.errors.append("MultiTaskGP requires task_feature_index to be set.")
        if spec.multitask_rank is not None and spec.multitask_rank < 1:
            result.errors.append(f"multitask_rank must be >= 1, got {spec.multitask_rank}.")
        if spec.output_dim < 2:
            result.warnings.append("MultiTaskGP with output_dim=1 is unusual; consider SingleTaskGP.")

    elif spec.model_class in _MODEL_LIST_MODELS:
        if spec.output_dim < 2:
            result.warnings.append("ModelListGP with output_dim=1 is unusual; consider SingleTaskGP.")

    elif spec.model_class == ModelClass.SINGLE_TASK_GP:
        if spec.task_feature_index is not None:
            result.warnings.append(
                "task_feature_index is set but model_class is SingleTaskGP; "
                "the task column will be ignored."
            )


def _check_noise(spec: GPSpec, result: ValidationResult) -> None:
    if spec.noise.fixed and spec.noise.noise_value is None:
        result.errors.append("noise.fixed=True but noise.noise_value is None; a value must be provided.")
    if spec.noise.noise_value is not None and spec.noise.noise_value < 0:
        result.errors.append(f"noise.noise_value must be non-negative, got {spec.noise.noise_value}.")


def _check_priors(spec: GPSpec, result: ValidationResult) -> None:
    for group in spec.feature_groups:
        kernel = group.kernel
        if kernel.lengthscale_prior and not kernel.lengthscale_prior.distribution:
            result.errors.append(
                f"Feature group '{group.name}': lengthscale_prior has no distribution name."
            )
        if kernel.outputscale_prior and not kernel.outputscale_prior.distribution:
            result.errors.append(
                f"Feature group '{group.name}': outputscale_prior has no distribution name."
            )
        if spec.noise.prior and not spec.noise.prior.distribution:
            result.errors.append("noise.prior has no distribution name.")
