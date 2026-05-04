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
    MeanFunctionType,
    MeanSpec,
    ModelClass,
    PriorDistribution,
    PriorSpec,
    SpectralMixtureInitialization,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _build_gpytorch_prior(prior_spec: PriorSpec | None):  # noqa: ANN201
    """Build a supported GPyTorch prior from a PriorSpec."""
    if prior_spec is None:
        return None

    from gpytorch.priors.torch_priors import GammaPrior, HalfCauchyPrior, LogNormalPrior, NormalPrior, UniformPrior

    if prior_spec.distribution == PriorDistribution.NORMAL:
        return NormalPrior(
            loc=prior_spec.params["loc"],
            scale=prior_spec.params["scale"],
        )
    if prior_spec.distribution == PriorDistribution.LOG_NORMAL:
        return LogNormalPrior(
            loc=prior_spec.params["loc"],
            scale=prior_spec.params["scale"],
        )
    if prior_spec.distribution == PriorDistribution.GAMMA:
        return GammaPrior(
            concentration=prior_spec.params["concentration"],
            rate=prior_spec.params["rate"],
        )
    if prior_spec.distribution == PriorDistribution.HALF_CAUCHY:
        return HalfCauchyPrior(scale=prior_spec.params["scale"])
    if prior_spec.distribution == PriorDistribution.UNIFORM:
        return UniformPrior(a=prior_spec.params["a"], b=prior_spec.params["b"])

    raise ValueError(f"Unsupported prior distribution: {prior_spec.distribution}")


def _build_gpytorch_kernel(kernel_spec: KernelSpec, num_features: int):  # noqa: ANN201
    """Construct a GPyTorch kernel from a KernelSpec.

    Args:
        kernel_spec: The kernel specification to build.
        num_features: Number of input features this kernel acts on.

    Returns:
        A gpytorch.kernels.Kernel instance.
    """

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
    import gpytorch
    from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
    from botorch.models.kernels.infinite_width_bnn import InfiniteWidthBNNKernel

    resolved_ard_num_dims = len(active_dims) if kernel_spec.ard else ard_num_dims
    lengthscale_prior = _build_gpytorch_prior(kernel_spec.lengthscale_prior)
    outputscale_prior = _build_gpytorch_prior(kernel_spec.outputscale_prior)
    period_prior = _build_gpytorch_prior(kernel_spec.period_prior)

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
            lengthscale_prior=lengthscale_prior,
        ),
        KernelType.RQ: lambda: gpytorch.kernels.RQKernel(
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
        ),
        KernelType.MATERN_12: lambda: gpytorch.kernels.MaternKernel(
            nu=0.5,
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
        ),
        KernelType.MATERN_32: lambda: gpytorch.kernels.MaternKernel(
            nu=1.5,
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
        ),
        KernelType.MATERN_52: lambda: gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=resolved_ard_num_dims,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
        ),
        KernelType.LINEAR: lambda: gpytorch.kernels.LinearKernel(active_dims=active_dims),
        KernelType.PERIODIC: lambda: gpytorch.kernels.PeriodicKernel(
            active_dims=active_dims,
            ard_num_dims=resolved_ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            period_length_prior=period_prior,
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

    # Changepoint kernel is handled separately because it requires two child kernels
    # built recursively and passed to the ChangepointKernel constructor.
    if kernel_spec.kernel_type == KernelType.CHANGEPOINT:
        from gparchitect.builders.changepoint_kernel import ChangepointKernel

        if len(kernel_spec.children) != 2:  # noqa: PLR2004
            raise ValueError(
                f"Changepoint kernel requires exactly 2 children (before/after kernels), "
                f"got {len(kernel_spec.children)}."
            )
        k_before = _build_gpytorch_kernel_with_active_dims(
            kernel_spec.children[0],
            active_dims,
            wrap_in_scale=False,
            train_X=train_X,
            train_Y=train_Y,
        )
        k_after = _build_gpytorch_kernel_with_active_dims(
            kernel_spec.children[1],
            active_dims,
            wrap_in_scale=False,
            train_X=train_X,
            train_Y=train_Y,
        )
        location = kernel_spec.changepoint_location if kernel_spec.changepoint_location is not None else 0.5
        steepness = kernel_spec.changepoint_steepness if kernel_spec.changepoint_steepness is not None else 1.0
        base_kernel = ChangepointKernel(
            kernel_before=k_before,
            kernel_after=k_after,
            changepoint_location=location,
            changepoint_steepness=steepness,
            active_dims=active_dims,
        )
        if wrap_in_scale:
            return gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior=outputscale_prior)
        return base_kernel

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
            if outputscale_prior is not None:
                raise ValueError("outputscale_prior is only supported on leaf kernels or multiplicative composites.")
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
        if not wrap_in_scale:
            if outputscale_prior is not None:
                raise ValueError("outputscale_prior requires an outer ScaleKernel.")
            return composed
        return gpytorch.kernels.ScaleKernel(composed, outputscale_prior=outputscale_prior)

    should_wrap = wrap_in_scale and kernel_spec.kernel_type != KernelType.SPECTRAL_MIXTURE
    if should_wrap:
        scaled = gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior=outputscale_prior)
    else:
        scaled = base_kernel

    # Apply Tier 2 time-varying hyperparameter wrapper if requested.
    # The wrapper is applied after the base kernel (and optional ScaleKernel) is built,
    # so that the time-varying modulation acts on the full scaled kernel output or
    # on the kernel's time input, depending on the target.
    if kernel_spec.time_varying is not None:
        from gparchitect.builders.time_varying_kernel import build_time_varying_kernel

        tv_spec = kernel_spec.time_varying
        scaled = build_time_varying_kernel(
            base_kernel=scaled,
            time_feature_index=tv_spec.time_feature_index,
            target=tv_spec.target.value,
        )

    return scaled


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
        _build_group_kernel(group, feature_index_map, train_X=train_X, train_Y=train_Y) for group in spec.feature_groups
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
                    gpytorch.kernels.ScaleKernel(gpytorch.kernels.ProductKernel(*interaction_components))
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
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.ProductKernel(*product_kernels))
    if spec.group_composition == CompositionType.NONE:
        raise ValueError("Multiple feature groups require explicit combination semantics; composition=none is invalid.")
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

    continuous_indices = sorted({idx for group in spec.feature_groups for idx in group.feature_indices})
    feature_index_map = {index: position for position, index in enumerate(continuous_indices)}
    continuous_X = train_X[:, continuous_indices]

    if spec.task_feature_index is not None:
        task_X = train_X[:, spec.task_feature_index].long().unsqueeze(-1)
        full_X = torch.cat([continuous_X, task_X.float()], dim=-1)
    else:
        full_X = continuous_X

    return full_X, train_Y, feature_index_map


def _build_mean_module_from_spec(mean_spec: MeanSpec, input_size: int):  # noqa: ANN201
    """Build a GPyTorch mean module from a mean specification."""
    import gpytorch

    if mean_spec.mean_type == MeanFunctionType.CONSTANT:
        return gpytorch.means.ConstantMean()
    if mean_spec.mean_type == MeanFunctionType.ZERO:
        return gpytorch.means.ZeroMean()
    return gpytorch.means.LinearMean(input_size=input_size)


def _build_mean_module(
    spec: GPSpec,
    input_size: int,
    *,
    output_index: int | None = None,
):  # noqa: ANN201
    """Build a mean module for single-task or per-output independent models."""
    mean_spec = spec.output_means.get(output_index) if output_index is not None else None
    if mean_spec is None:
        mean_spec = spec.mean
    if mean_spec is None:
        return None
    return _build_mean_module_from_spec(mean_spec, input_size)


def _build_multitask_mean_module(
    spec: GPSpec,
    task_values: list[int],
    num_non_task_features: int,
    combined_input_size: int,
):  # noqa: ANN201
    """Build a mean module for MultiTaskGP, including optional per-task overrides."""
    import gpytorch
    import torch

    class _TaskIndexedMean(gpytorch.means.Mean):
        """Dispatch per-task scalar means for the long-format MultiTaskGP input layout."""

        def __init__(self, base_means, resolved_task_values: list[int], task_feature_index: int) -> None:  # noqa: ANN001
            super().__init__()
            self.base_means = torch.nn.ModuleList(base_means)
            self.task_values = tuple(int(value) for value in resolved_task_values)
            self.task_feature_index = task_feature_index

        def forward(self, x):  # noqa: ANN001, ANN201
            resolved_task_feature_index = self.task_feature_index
            if resolved_task_feature_index < 0:
                resolved_task_feature_index = x.shape[-1] + resolved_task_feature_index

            continuous_parts = [x[..., :resolved_task_feature_index], x[..., resolved_task_feature_index + 1 :]]
            continuous_x = torch.cat([part for part in continuous_parts if part.shape[-1] > 0], dim=-1)
            task_column = x[..., resolved_task_feature_index].long()

            mean_values = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
            for task_value, base_mean in zip(self.task_values, self.base_means):
                task_mask = task_column == task_value
                if bool(task_mask.any()):
                    mean_values[task_mask] = base_mean(continuous_x[task_mask])
            return mean_values

    if not spec.output_means and spec.mean is None:
        return None

    if not spec.output_means:
        return _build_mean_module_from_spec(spec.mean, combined_input_size) if spec.mean is not None else None

    base_means = []
    for task_value in task_values:
        task_mean_spec = spec.output_means.get(task_value)
        if task_mean_spec is None:
            task_mean_spec = spec.mean or MeanSpec(mean_type=MeanFunctionType.CONSTANT)
        base_means.append(_build_mean_module_from_spec(task_mean_spec, num_non_task_features))

    return _TaskIndexedMean(base_means=base_means, resolved_task_values=task_values, task_feature_index=-1)


def _build_input_transform(
    spec: GPSpec,
    feature_index_map: dict[int, int],
    full_X: "torch.Tensor",
) -> "Any | None":
    """Build an optional BoTorch input transform from the ExecutionSpec.

    Currently only supports the Tier 2 Kumaraswamy input warp.  Returns None when
    no input transform is requested.

    Args:
        spec: The validated GPSpec.
        feature_index_map: Mapping from original feature indices to contiguous indices
            in the continuous-feature tensor (used to translate the DSL index).
        full_X: The continuous feature tensor (used only for shape information).

    Returns:
        A botorch.models.transforms.input.InputTransform instance, or None.
    """
    warping_spec = spec.execution.input_warping
    if warping_spec is None:
        return None

    from botorch.models.transforms.input import Warp

    # Map the DSL-level time_feature_index to the contiguous input index.
    # If the index is not in feature_index_map (e.g. it was removed from the
    # continuous set), fall back to the raw index clamped to valid range.
    mapped_index = feature_index_map.get(warping_spec.time_feature_index, warping_spec.time_feature_index)
    n_cols = full_X.shape[-1]
    mapped_index = max(0, min(mapped_index, n_cols - 1))

    # When input_scaling is enabled, the DSL contract expects continuous inputs in [0, 1].
    # Provide explicit unit-cube bounds so Warp does not learn bounds from only the
    # training subset (which can collapse extrapolation points to a constant value).
    warp_bounds = None
    if spec.execution.input_scaling:
        warp_bounds = full_X.new_tensor([[0.0], [1.0]])

    warp_transform = Warp(indices=[mapped_index], d=n_cols, bounds=warp_bounds)

    if warping_spec.concentration0 is not None or warping_spec.concentration1 is not None:
        # Use the Warp's built-in initialize() method which respects constraints.
        init_kwargs: dict[str, float] = {}
        if warping_spec.concentration0 is not None:
            init_kwargs["concentration0"] = warping_spec.concentration0
        if warping_spec.concentration1 is not None:
            init_kwargs["concentration1"] = warping_spec.concentration1
        warp_transform.initialize(**init_kwargs)

    logger.info(
        "Input warping enabled: type=%s, time_feature_index=%d (mapped→%d)",
        warping_spec.warp_type.value,
        warping_spec.time_feature_index,
        mapped_index,
    )
    return warp_transform


def build_model_from_dsl(spec: GPSpec, train_X: "torch.Tensor", train_Y: "torch.Tensor"):  # noqa: ANN201
    """Construct a BoTorch GP model from a validated GPSpec.

    When spec.execution.input_warping is set, a Kumaraswamy input warp transform is
    applied to the designated time-like feature dimension before model construction.
    The warp is a BoTorch-native input transform (botorch.models.transforms.input.Warp)
    that is optimized jointly with the kernel hyperparameters.

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
    from botorch.models.transforms.outcome import Standardize

    full_X, full_Y, feature_index_map = _prepare_inputs(spec, train_X, train_Y)

    logger.info("Building model: class=%s, input_shape=%s", spec.model_class.value, tuple(full_X.shape))

    input_transform = _build_input_transform(spec, feature_index_map, full_X)

    if spec.model_class == ModelClass.SINGLE_TASK_GP:
        covar_module = _build_covariance_module(spec, feature_index_map, full_X, full_Y)
        mean_module = _build_mean_module(spec, input_size=full_X.shape[-1])
        likelihood = _build_likelihood(spec, train_Y=full_Y, model_class=spec.model_class)
        outcome_transform = Standardize(m=full_Y.shape[-1]) if spec.execution.outcome_standardization else None
        model = botorch.models.SingleTaskGP(
            train_X=full_X,
            train_Y=full_Y,
            covar_module=covar_module,
            likelihood=likelihood,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

    elif spec.model_class == ModelClass.MULTI_TASK_GP:
        if spec.task_feature_index is None:
            raise ValueError("MultiTaskGP requires task_feature_index.")
        if spec.task_values is None:
            raise ValueError("MultiTaskGP requires explicit task_values.")
        rank = spec.multitask_rank if spec.multitask_rank is not None else 1
        observed_task_values = sorted({int(value) for value in train_X[:, spec.task_feature_index].long().tolist()})
        task_values = sorted(spec.task_values)
        if observed_task_values != task_values:
            raise ValueError(
                f"Observed task values {observed_task_values} do not match declared task_values {task_values}."
            )
        covar_module = _build_covariance_module(spec, feature_index_map, full_X, full_Y)
        mean_module = _build_multitask_mean_module(
            spec,
            task_values=task_values,
            num_non_task_features=len(feature_index_map),
            combined_input_size=full_X.shape[-1],
        )
        likelihood = _build_likelihood(
            spec,
            train_Y=full_Y,
            model_class=spec.model_class,
            num_tasks=len(task_values),
            task_feature_index=-1,
        )
        model = botorch.models.MultiTaskGP(
            train_X=full_X,
            train_Y=full_Y,
            task_feature=-1,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            rank=rank,
            input_transform=input_transform,
        )

    elif spec.model_class == ModelClass.MODEL_LIST_GP:
        individual_models = []
        for output_idx in range(spec.output_dim):
            output_train_Y = full_Y[:, output_idx : output_idx + 1]
            covar_module = _build_covariance_module(spec, feature_index_map, full_X, output_train_Y)
            mean_module = _build_mean_module(spec, input_size=full_X.shape[-1], output_index=output_idx)
            likelihood = _build_likelihood(spec, train_Y=output_train_Y, model_class=ModelClass.SINGLE_TASK_GP)
            outcome_transform = Standardize(m=1) if spec.execution.outcome_standardization else None
            single_model = botorch.models.SingleTaskGP(
                train_X=full_X,
                train_Y=output_train_Y,
                covar_module=covar_module,
                likelihood=likelihood,
                mean_module=mean_module,
                outcome_transform=outcome_transform,
                input_transform=input_transform,
            )
            individual_models.append(single_model)
        model = botorch.models.ModelListGP(*individual_models)

    else:
        raise ValueError(f"Unsupported model class: {spec.model_class}")

    logger.info("Model built successfully: %s", type(model).__name__)
    return model


def _build_likelihood(
    spec: GPSpec,
    *,
    train_Y,  # noqa: ANN001
    model_class: ModelClass,
    num_tasks: int | None = None,
    task_feature_index: int | None = None,
):  # noqa: ANN201
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
        if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
            fixed_noise = torch.full(train_Y.shape[:-1], noise_val, dtype=train_Y.dtype, device=train_Y.device)
        else:
            fixed_noise = torch.full_like(train_Y, noise_val)
        return gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed_noise)

    if model_class == ModelClass.MULTI_TASK_GP:
        multitask_likelihood = _build_multitask_default_likelihood(
            num_tasks=num_tasks,
            task_feature_index=task_feature_index,
            noise_prior=_build_gpytorch_prior(spec.noise.prior),
        )
        if multitask_likelihood is not None:
            return multitask_likelihood
        return None

    return gpytorch.likelihoods.GaussianLikelihood(noise_prior=_build_gpytorch_prior(spec.noise.prior))


def _build_multitask_default_likelihood(
    *,
    num_tasks: int | None,
    task_feature_index: int | None,
    noise_prior=None,  # noqa: ANN001
):  # noqa: ANN201
    """Build the documented default multitask likelihood when available.

    Falls back to ``None`` when the installed dependency stack does not expose
    ``HadamardGaussianLikelihood`` so BoTorch can apply its own default.
    """
    if num_tasks is None or task_feature_index is None:
        return None

    import torch
    from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
    from gpytorch.constraints import GreaterThan
    from gpytorch.likelihoods.hadamard_gaussian_likelihood import HadamardGaussianLikelihood
    from gpytorch.priors.torch_priors import LogNormalPrior

    resolved_noise_prior = noise_prior or LogNormalPrior(loc=-4.0, scale=1.0)
    return HadamardGaussianLikelihood(
        num_tasks=num_tasks,
        batch_shape=torch.Size(),
        noise_prior=resolved_noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            initial_value=resolved_noise_prior.mode,
        ),
        task_feature_index=task_feature_index,
    )
