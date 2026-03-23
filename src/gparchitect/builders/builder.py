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
        - Additive compositions keep per-term ScaleKernel wrappers and do not add an
            extra outer scale around the sum.
        - Multiplicative compositions keep a single outer ScaleKernel around the full
            product and avoid scaling each factor independently.
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
from typing import TYPE_CHECKING, Any, cast

from gparchitect.dsl.schema import (
    CompositionType,
    GPSpec,
    KernelSpec,
    KernelType,
    ModelClass,
    SpectralMixtureInitialization,
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

    active_dims = tuple(range(num_features))
    ard_num_dims = num_features if kernel_spec.ard else None

    return _build_gpytorch_kernel_with_active_dims(kernel_spec, active_dims, ard_num_dims)


def _build_gpytorch_kernel_with_active_dims(
    kernel_spec: KernelSpec,
    active_dims: tuple[int, ...],
    ard_num_dims: int | None = None,
    *,
    wrap_in_scale: bool = True,
    train_X=None,  # noqa: ANN001
    train_Y=None,  # noqa: ANN001
):  # noqa: ANN201
    """Construct a GPyTorch kernel from a KernelSpec and explicit active dimensions."""
    from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
    from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel
    import gpytorch

    resolved_ard_num_dims = len(active_dims) if kernel_spec.ard else ard_num_dims

    if kernel_spec.kernel_type == KernelType.SPECTRAL_MIXTURE:
        num_mixtures = kernel_spec.num_mixtures or 4
        base_kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
        )
        if train_X is not None and train_Y is not None:
            if kernel_spec.spectral_init == SpectralMixtureInitialization.FROM_EMPIRICAL_SPECTRUM:
                try:
                    base_kernel.initialize_from_data_empspect(train_X, _kernel_initialization_target(train_Y))
                except ImportError as exc:
                    logger.warning(
                        "Empirical-spectrum SpectralMixture initialization is unavailable (%s); "
                        "falling back to initialize_from_data.",
                        exc,
                    )
                    base_kernel.initialize_from_data(train_X, _kernel_initialization_target(train_Y))
            else:
                base_kernel.initialize_from_data(train_X, _kernel_initialization_target(train_Y))
        return base_kernel

    kernel_map = {
        KernelType.RBF: lambda: gpytorch.kernels.RBFKernel(
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
        ),
        KernelType.RQ: lambda: gpytorch.kernels.RQKernel(
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
        ),
        KernelType.MATERN_12: lambda: gpytorch.kernels.MaternKernel(
            nu=0.5,
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
        ),
        KernelType.MATERN_32: lambda: gpytorch.kernels.MaternKernel(
            nu=1.5,
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
        ),
        KernelType.MATERN_52: lambda: gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
        ),
        KernelType.LINEAR: lambda: gpytorch.kernels.LinearKernel(active_dims=active_dims),
        KernelType.PERIODIC: lambda: gpytorch.kernels.PeriodicKernel(
            active_dims=active_dims,
            ard_num_dims=resolved_ard_num_dims,
        ),
        KernelType.POLYNOMIAL: lambda: gpytorch.kernels.PolynomialKernel(
            power=kernel_spec.polynomial_power or 2,
            active_dims=active_dims,
        ),
        KernelType.INFINITE_WIDTH_BNN: lambda: InfiniteWidthBNNKernel(
            depth=kernel_spec.bnn_depth or 3,
            active_dims=active_dims,
        ),
        KernelType.EXPONENTIAL_DECAY: lambda: ExponentialDecayKernel(active_dims=active_dims),
    }

    if kernel_spec.kernel_type not in kernel_map:
        logger.warning("Unknown kernel type %s; falling back to Matern52.", kernel_spec.kernel_type)
        base_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
        )
    else:
        base_kernel = kernel_map[kernel_spec.kernel_type]()

    if kernel_spec.kernel_type == KernelType.RQ and kernel_spec.rq_alpha is not None:
        base_kernel.initialize(alpha=kernel_spec.rq_alpha)
    if kernel_spec.kernel_type == KernelType.PERIODIC and kernel_spec.period_length is not None:
        base_kernel.initialize(period_length=kernel_spec.period_length)
    if kernel_spec.kernel_type == KernelType.POLYNOMIAL and kernel_spec.polynomial_offset is not None:
        base_kernel.initialize(offset=kernel_spec.polynomial_offset)
    if kernel_spec.kernel_type == KernelType.EXPONENTIAL_DECAY:
        init_kwargs: dict[str, float] = {}
        if kernel_spec.exponential_decay_power is not None:
            init_kwargs["power"] = kernel_spec.exponential_decay_power
        if kernel_spec.exponential_decay_offset is not None:
            init_kwargs["offset"] = kernel_spec.exponential_decay_offset
        if init_kwargs:
            base_kernel.initialize(**init_kwargs)

    if kernel_spec.children:
        if kernel_spec.composition == CompositionType.ADDITIVE:
            child_kernels = [
                _build_gpytorch_kernel_with_active_dims(
                    child,
                    active_dims,
                    train_X=train_X,
                    train_Y=train_Y,
                )
                for child in kernel_spec.children
            ]
            return gpytorch.kernels.AdditiveKernel(*cast(list[Any], child_kernels))

        child_kernels = [
            _build_gpytorch_kernel_with_active_dims(
                child,
                active_dims,
                wrap_in_scale=kernel_spec.composition != CompositionType.MULTIPLICATIVE,
                train_X=train_X,
                train_Y=train_Y,
            )
            for child in kernel_spec.children
        ]
        if kernel_spec.composition == CompositionType.MULTIPLICATIVE:
            composed = gpytorch.kernels.ProductKernel(*cast(list[Any], child_kernels))
        else:
            composed = gpytorch.kernels.AdditiveKernel(*cast(list[Any], child_kernels))
        return gpytorch.kernels.ScaleKernel(composed) if wrap_in_scale else composed

    should_wrap = wrap_in_scale and kernel_spec.kernel_type != KernelType.SPECTRAL_MIXTURE
    return gpytorch.kernels.ScaleKernel(base_kernel) if should_wrap else base_kernel


def _kernel_initialization_target(train_Y):  # noqa: ANN201, ANN001
    """Reduce training targets to the shape expected by kernel initializers."""
    if train_Y.ndim == 1:
        return train_Y
    if train_Y.shape[-1] == 1:
        return train_Y.squeeze(-1)
    return train_Y[:, 0]


def _build_group_kernel(
    group,
    feature_index_map: dict[int, int],
    *,
    wrap_in_scale: bool = True,
    train_X=None,  # noqa: ANN001
    train_Y=None,  # noqa: ANN001
):  # noqa: ANN001, ANN201
    """Build a covariance kernel for a single feature group."""
    active_dims = tuple(feature_index_map[index] for index in group.feature_indices)
    return _build_gpytorch_kernel_with_active_dims(
        group.kernel,
        active_dims,
        wrap_in_scale=wrap_in_scale,
        train_X=train_X,
        train_Y=train_Y,
    )


def _build_covariance_module(
    spec: GPSpec,
    feature_index_map: dict[int, int],
    train_X=None,
    train_Y=None,
):  # noqa: ANN001, ANN201
    """Build the combined covariance module from all feature groups.

    Args:
        spec: The validated GPSpec.

    Returns:
        A gpytorch.kernels.Kernel combining all feature groups.
    """
    import gpytorch

    group_kernels = [
        _build_group_kernel(group, feature_index_map, train_X=train_X, train_Y=train_Y)
        for group in spec.feature_groups
    ]

    if len(group_kernels) == 1:
        return group_kernels[0]

    if spec.group_composition == CompositionType.HIERARCHICAL:
        interaction_terms = []
        for left_index, left_group in enumerate(spec.feature_groups):
            for right_group in spec.feature_groups[left_index + 1 :]:
                interaction_components = cast(
                    list[Any],
                    [
                        _build_group_kernel(
                            left_group,
                            feature_index_map,
                            wrap_in_scale=False,
                            train_X=train_X,
                            train_Y=train_Y,
                        ),
                        _build_group_kernel(
                            right_group,
                            feature_index_map,
                            wrap_in_scale=False,
                            train_X=train_X,
                            train_Y=train_Y,
                        ),
                    ],
                )
                interaction_terms.append(
                    gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.ProductKernel(*interaction_components)
                    )
                )
        combined_kernels = cast(list[Any], group_kernels + interaction_terms)
        return gpytorch.kernels.AdditiveKernel(*combined_kernels)

    if spec.group_composition == CompositionType.MULTIPLICATIVE:
        product_kernels = cast(
            list[Any],
            [
                _build_group_kernel(
                    group,
                    feature_index_map,
                    wrap_in_scale=False,
                    train_X=train_X,
                    train_Y=train_Y,
                )
                for group in spec.feature_groups
            ],
        )
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ProductKernel(*product_kernels)
        )
    return gpytorch.kernels.AdditiveKernel(*cast(list[Any], group_kernels))


def _prepare_inputs(spec: GPSpec, train_X: "torch.Tensor", train_Y: "torch.Tensor"):  # noqa: ANN201
    """Extract the continuous feature columns from train_X.

    Args:
        spec: The validated GPSpec.
        train_X: Raw input tensor of shape (N, D).
        train_Y: Output tensor of shape (N, M).

    Returns:
        Tuple of (continuous_X, train_Y, feature_index_map).
    """
    import torch

    continuous_indices = sorted(
        {idx for group in spec.feature_groups for idx in group.feature_indices}
    )
    feature_index_map = {index: position for position, index in enumerate(continuous_indices)}
    continuous_X = train_X[:, continuous_indices]

    if spec.task_feature_index is not None:
        task_X = train_X[:, spec.task_feature_index].long().unsqueeze(-1)
        full_X = torch.cat([continuous_X, task_X.float()], dim=-1)
    else:
        full_X = continuous_X

    return full_X, train_Y, feature_index_map


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
    from botorch.models.transforms.outcome import Standardize

    full_X, full_Y, feature_index_map = _prepare_inputs(spec, train_X, train_Y)

    logger.info("Building model: class=%s, input_shape=%s", spec.model_class.value, tuple(full_X.shape))

    if spec.model_class == ModelClass.SINGLE_TASK_GP:
        covar_module = _build_covariance_module(spec, feature_index_map, full_X, full_Y)
        likelihood = _build_likelihood(spec)
        model = botorch.models.SingleTaskGP(
            train_X=full_X,
            train_Y=full_Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=full_Y.shape[-1]),
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
            output_train_Y = full_Y[:, output_idx : output_idx + 1]
            covar_module = _build_covariance_module(spec, feature_index_map, full_X, output_train_Y)
            likelihood = _build_likelihood(spec)
            single_model = botorch.models.SingleTaskGP(
                train_X=full_X,
                train_Y=output_train_Y,
                covar_module=covar_module,
                likelihood=likelihood,
                outcome_transform=Standardize(m=1),
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
    import torch

    if spec.noise.fixed and spec.noise.noise_value is not None:
        noise_val = spec.noise.noise_value
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(noise_val, dtype=torch.double)
        likelihood.noise_covar.raw_noise.requires_grad_(False)
        return likelihood

    return gpytorch.likelihoods.GaussianLikelihood()
