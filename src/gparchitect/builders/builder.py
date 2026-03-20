"""
Model builder for GPArchitect — constructs BoTorch GP models from a validated GPSpec.

Purpose:
    Translates a validated GPSpec DSL object into a BoTorch-compatible GP model
    (SingleTaskGP, MultiTaskGP, or ModelListGP) with the appropriate GPyTorch kernels.

Role in pipeline:
    Natural language → GP DSL → Validation → **Model Builder** → Fit → Validation → Recovery

Inputs:
    - spec: GPSpec — a validated DSL object.
    - train_X: torch.Tensor — input training data of shape (N, D).
    - train_Y: torch.Tensor — output training data of shape (N, M).

Outputs:
    A BoTorch GP model instance (SingleTaskGP, MultiTaskGP, or ModelListGP).

Non-obvious design decisions:
    - BoTorch and torch are imported lazily (inside functions) to allow the rest of
      GPArchitect to import without requiring them to be installed.
    - Kernel construction follows the feature-group spec: each group gets its own
      kernel; groups are combined via AdditiveKernel or ProductKernel as specified.
    - ARD is implemented by passing `ard_num_dims` equal to the number of features
      in the group.
    - Custom ExactGP subclasses are avoided; BoTorch's SingleTaskGP is used directly
      with a custom covariance_module argument.

What this module does NOT do:
    - It does not validate the DSL (see validation module).
    - It does not fit the model (see fitting module).
    - It does not handle data loading or DataFrame conversion (see data module).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gparchitect.dsl.schema import (
    CompositionType,
    GPSpec,
    KernelSpec,
    KernelType,
    ModelClass,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _build_gpytorch_kernel(kernel_spec: KernelSpec, num_features: int):  # noqa: ANN201
    """Construct a GPyTorch kernel from a KernelSpec.

    Args:
        kernel_spec: The kernel specification to build.
        num_features: Number of input features this kernel acts on.

    Returns:
        A gpytorch.kernels.Kernel instance.
    """
    import gpytorch

    ard_num_dims = num_features if kernel_spec.ard else None

    kernel_map = {
        KernelType.RBF: lambda: gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims),
        KernelType.MATERN_12: lambda: gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=ard_num_dims),
        KernelType.MATERN_32: lambda: gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims),
        KernelType.MATERN_52: lambda: gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims),
        KernelType.LINEAR: lambda: gpytorch.kernels.LinearKernel(),
        KernelType.PERIODIC: lambda: gpytorch.kernels.PeriodicKernel(),
        KernelType.POLYNOMIAL: lambda: gpytorch.kernels.PolynomialKernel(power=2),
    }

    if kernel_spec.kernel_type not in kernel_map:
        logger.warning("Unknown kernel type %s; falling back to Matern52.", kernel_spec.kernel_type)
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    else:
        base_kernel = kernel_map[kernel_spec.kernel_type]()

    if kernel_spec.children:
        child_kernels = [_build_gpytorch_kernel(child, num_features) for child in kernel_spec.children]
        if kernel_spec.composition == CompositionType.MULTIPLICATIVE:
            composed = gpytorch.kernels.ProductKernel(*child_kernels)
        else:
            composed = gpytorch.kernels.AdditiveKernel(*child_kernels)
        return gpytorch.kernels.ScaleKernel(composed)

    return gpytorch.kernels.ScaleKernel(base_kernel)


def _build_covariance_module(spec: GPSpec):  # noqa: ANN201
    """Build the combined covariance module from all feature groups.

    Args:
        spec: The validated GPSpec.

    Returns:
        A gpytorch.kernels.Kernel combining all feature groups.
    """
    import gpytorch

    group_kernels = [
        _build_gpytorch_kernel(group.kernel, len(group.feature_indices))
        for group in spec.feature_groups
    ]

    if len(group_kernels) == 1:
        return group_kernels[0]

    if spec.group_composition == CompositionType.MULTIPLICATIVE:
        return gpytorch.kernels.ProductKernel(*group_kernels)
    return gpytorch.kernels.AdditiveKernel(*group_kernels)


def _prepare_inputs(spec: GPSpec, train_X: "torch.Tensor", train_Y: "torch.Tensor"):  # noqa: ANN201
    """Extract the continuous feature columns from train_X.

    Args:
        spec: The validated GPSpec.
        train_X: Raw input tensor of shape (N, D).
        train_Y: Output tensor of shape (N, M).

    Returns:
        Tuple of (continuous_X, task_X, train_Y) where task_X may be None.
    """
    import torch

    continuous_indices = sorted(
        {idx for group in spec.feature_groups for idx in group.feature_indices}
    )
    continuous_X = train_X[:, continuous_indices]

    if spec.task_feature_index is not None:
        task_X = train_X[:, spec.task_feature_index].long().unsqueeze(-1)
        full_X = torch.cat([continuous_X, task_X.float()], dim=-1)
    else:
        full_X = continuous_X

    return full_X, train_Y


def build_model_from_dsl(spec: GPSpec, train_X: "torch.Tensor", train_Y: "torch.Tensor"):  # noqa: ANN201
    """Construct a BoTorch GP model from a validated GPSpec.

    Args:
        spec: A validated GPSpec DSL object.
        train_X: Input training tensor of shape (N, D).
        train_Y: Output training tensor of shape (N, M).

    Returns:
        A BoTorch GP model instance ready for fitting.

    Raises:
        ValueError: If the model class in the spec is not supported.
        RuntimeError: If model construction fails for any reason.
    """
    import botorch.models
    import torch

    full_X, full_Y = _prepare_inputs(spec, train_X, train_Y)

    logger.info("Building model: class=%s, input_shape=%s", spec.model_class.value, tuple(full_X.shape))

    if spec.model_class == ModelClass.SINGLE_TASK_GP:
        covar_module = _build_covariance_module(spec)
        likelihood = _build_likelihood(spec)
        model = botorch.models.SingleTaskGP(
            train_X=full_X,
            train_Y=full_Y,
            covar_module=covar_module,
            likelihood=likelihood,
        )

    elif spec.model_class == ModelClass.MULTI_TASK_GP:
        if spec.task_feature_index is None:
            raise ValueError("MultiTaskGP requires task_feature_index.")
        rank = spec.multitask_rank if spec.multitask_rank is not None else 1
        # Append task column back for MultiTaskGP
        task_col = train_X[:, spec.task_feature_index].long().unsqueeze(-1)
        continuous_indices = sorted(
            {idx for group in spec.feature_groups for idx in group.feature_indices}
        )
        continuous_X = train_X[:, continuous_indices]
        combined_X = torch.cat([continuous_X, task_col.float()], dim=-1)
        model = botorch.models.MultiTaskGP(
            train_X=combined_X,
            train_Y=full_Y,
            task_feature=-1,
            rank=rank,
        )

    elif spec.model_class == ModelClass.MODEL_LIST_GP:
        individual_models = []
        for output_idx in range(spec.output_dim):
            covar_module = _build_covariance_module(spec)
            likelihood = _build_likelihood(spec)
            single_model = botorch.models.SingleTaskGP(
                train_X=full_X,
                train_Y=full_Y[:, output_idx : output_idx + 1],
                covar_module=covar_module,
                likelihood=likelihood,
            )
            individual_models.append(single_model)
        model = botorch.models.ModelListGP(*individual_models)

    else:
        raise ValueError(f"Unsupported model class: {spec.model_class}")

    logger.info("Model built successfully: %s", type(model).__name__)
    return model


def _build_likelihood(spec: GPSpec):  # noqa: ANN201
    """Build a GPyTorch likelihood from the noise specification.

    Args:
        spec: The validated GPSpec.

    Returns:
        A gpytorch.likelihoods.Likelihood instance.
    """
    import gpytorch

    if spec.noise.fixed and spec.noise.noise_value is not None:
        noise_val = spec.noise.noise_value
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_val
        likelihood.noise_covar.raw_noise.requires_grad_(False)
        return likelihood

    return gpytorch.likelihoods.GaussianLikelihood()
