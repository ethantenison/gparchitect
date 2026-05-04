"""
DSL schema and representation for GPArchitect.

Purpose:
    Defines the GP Domain-Specific Language (DSL) as structured, typed Pydantic models.
    This module is the single source of truth for GP model specifications within GPArchitect.

Role in pipeline:
    Natural language → **GP DSL** → Validation → Model Builder → Fit → Validation → Recovery

Inputs:
    None — this module defines data structures only.

Outputs:
    Pydantic model classes: GPSpec, KernelSpec, LeafKernelSpec, CompositeKernelSpec,
    ChangepointKernelSpec, KernelExpr, FeatureGroupSpec, PriorSpec, NoiseSpec,
    MeanSpec, ExecutionSpec, RecencyFilteringSpec, TimeVaryingSpec, InputWarpingSpec.

Non-obvious design decisions:
    - All fields use native Python typing (list, dict, X | None) per project style.
    - JSON serialization is guaranteed via Pydantic's model_dump(mode="json").
    - Enums for kernel types and model classes ensure stability across versions.
    - DSL is independent of natural-language phrasing; identical GP architectures
      must always produce identical DSL regardless of how they are described.
    - RecencyFilteringSpec is nested inside ExecutionSpec to keep data/fitting
      concerns separate from kernel/model concerns.  The canonical name is
      "recency filtering" — earlier versions used "recency weighting", which was
      misleading because the implementation is dataset truncation (filtering), not
      true observation-weighted GP inference.
    - heteroskedastic_noise is a forward-compatibility placeholder in NoiseSpec:
      it is currently rejected by the validator, but its presence in the schema
      prevents DSL churn when heteroskedastic support is added in a future tier.
    - TimeVaryingSpec in KernelSpec represents Tier 2 time-varying hyperparameters:
      the kernel's outputscale or lengthscale varies smoothly as a function of a
      designated time-like input.  The builder wraps the base kernel in a custom
      GPyTorch module that adds learnable linear-modulation parameters.
    - InputWarpingSpec in ExecutionSpec represents Tier 2 input warping: a
      monotone Kumaraswamy CDF warp is applied to the designated time-like input
      dimension before the kernel is evaluated.  BoTorch's Warp input transform
      is used directly for compatibility with the existing fitting path.

What this module does NOT do:
    - It does not validate dimensional or semantic consistency (see validation module).
    - It does not construct BoTorch models (see builders module).
    - It does not interpret natural language (see translator module).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


class ModelClass(str, Enum):
    """Supported BoTorch GP model classes."""

    SINGLE_TASK_GP = "SingleTaskGP"
    MULTI_TASK_GP = "MultiTaskGP"
    MODEL_LIST_GP = "ModelListGP"


class KernelType(str, Enum):
    """Supported GPyTorch kernel types."""

    RBF = "RBF"
    RQ = "RQ"
    MATERN_12 = "Matern12"
    MATERN_32 = "Matern32"
    MATERN_52 = "Matern52"
    LINEAR = "Linear"
    PERIODIC = "Periodic"
    POLYNOMIAL = "Polynomial"
    SPECTRAL_MIXTURE = "SpectralMixture"
    INFINITE_WIDTH_BNN = "InfiniteWidthBNN"
    EXPONENTIAL_DECAY = "ExponentialDecay"
    CHANGEPOINT = "Changepoint"


class SpectralMixtureInitialization(str, Enum):
    """Initialization modes for the Spectral Mixture kernel."""

    FROM_DATA = "from_data"
    FROM_EMPIRICAL_SPECTRUM = "from_empirical_spectrum"


class CompositionType(str, Enum):
    """Kernel composition operators."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    HIERARCHICAL = "hierarchical"
    NONE = "none"


class MeanFunctionType(str, Enum):
    """Supported GPyTorch mean function types."""

    CONSTANT = "Constant"
    ZERO = "Zero"
    LINEAR = "Linear"


class PriorDistribution(str, Enum):
    """Supported GP prior distributions in the DSL contract."""

    NORMAL = "Normal"
    LOG_NORMAL = "LogNormal"
    GAMMA = "Gamma"
    HALF_CAUCHY = "HalfCauchy"
    UNIFORM = "Uniform"


class PriorSpec(BaseModel):
    """Specification for a GP hyperparameter prior.

    Attributes:
        distribution: Name of the prior distribution.
        params: Distribution parameters as a name→value mapping.
    """

    distribution: PriorDistribution
    params: dict[str, float] = Field(default_factory=dict)


class RecencyFilteringMode(str, Enum):
    """Supported recency-filtering strategies for time-driven non-stationarity.

    Both strategies reduce the training dataset to a subset of recent observations
    before model fitting.  Neither performs true likelihood-weighted GP inference —
    they are dataset-truncation approximations.

    SLIDING_WINDOW: Keep only observations within a fixed time window ending at the
        most recent observation.  Observations outside the window are removed entirely.
    EXPONENTIAL_DISCOUNT: Compute a weight w_i = exp(−λ · Δt_i) per observation.
        Observations with w_i < min_weight are removed.  This is a thresholded-
        retention rule, not true exponential likelihood weighting.
    """

    SLIDING_WINDOW = "sliding_window"
    EXPONENTIAL_DISCOUNT = "exponential_discount"


class RecencyFilteringSpec(BaseModel):
    """Specification for recency-based filtering of stale observations.

    Implements Tier 1 time-driven non-stationarity by discarding old observations
    before fitting.  This is dataset truncation — it is NOT true observation-
    weighted GP inference.

    - SLIDING_WINDOW mode removes observations older than max_time − window_size.
    - EXPONENTIAL_DISCOUNT mode removes observations whose weight exp(−λ·Δt) falls
      below min_weight.  This is equivalent to a soft sliding window with a smooth
      boundary, not a probabilistic weighting of the likelihood.

    Attributes:
        mode: The filtering strategy to apply.
        time_feature_index: Zero-based column index of the time-like feature in train_X
            that determines observation recency.
        window_size: For SLIDING_WINDOW — width of the time window in the (possibly
            scaled) feature space.  Observations older than max_time − window_size
            are dropped.  Must be > 0.
        discount_rate: For EXPONENTIAL_DISCOUNT — rate λ in w_i = exp(−λ · Δt_i).
            Observations with w_i < min_weight are dropped.  Must be > 0.
        min_weight: Minimum retention weight threshold for EXPONENTIAL_DISCOUNT mode.
            Defaults to 0.01.  Must be in (0, 1).
    """

    mode: RecencyFilteringMode
    time_feature_index: int
    window_size: float | None = None
    discount_rate: float | None = None
    min_weight: float = 0.01


class TimeVaryingTarget(str, Enum):
    """Which kernel hyperparameter varies with the designated time-like input.

    OUTPUTSCALE: The kernel amplitude (outputscale) changes smoothly over time.
        The builder wraps the base kernel so that its effective outputscale is
        s(t) = softplus(bias + slope · t), with learnable bias and slope.
    LENGTHSCALE: The effective lengthscale along the time dimension changes over
        time.  The builder rescales the time feature by 1 / l(t) before passing
        it to the base kernel, where l(t) = softplus(bias + slope · t).
    """

    OUTPUTSCALE = "outputscale"
    LENGTHSCALE = "lengthscale"


class TimeVaryingSpec(BaseModel):
    """Tier 2 specification for a kernel with time-varying hyperparameters.

    Attaches a learned linear modulation to a kernel's outputscale or lengthscale
    as a function of a designated time-like input dimension.  Parameterization is
    a softplus-transformed linear function of time to guarantee positivity.

    The builder wraps the base kernel (specified by the parent KernelSpec) in a
    TimeVaryingKernel module.  The two learnable parameters (bias, slope) are
    optimized alongside the rest of the GP hyperparameters.

    This is intentionally a conservative first implementation:
    - Only one smooth parametric modulation per kernel is supported.
    - Only "linear" parameterization is supported in v1.
    - Multi-output or multi-task time variation is not yet supported.

    Attributes:
        target: Whether the outputscale or lengthscale varies with time.
        time_feature_index: Zero-based column index of the time-like feature.
        parameterization: The functional form of the time dependence.
            Currently only "linear" is supported.
        outputscale_bias_limit: Optional max absolute value for the effective
            outputscale modulation bias after tanh squashing.  Used only when
            target=OUTPUTSCALE.  Smaller values constrain amplitude variation
            more strongly.
        outputscale_slope_limit: Optional max absolute value for the effective
            outputscale modulation slope after tanh squashing.  Used only when
            target=OUTPUTSCALE.  Smaller values constrain temporal growth/decay
            of amplitude more strongly.
    """

    target: TimeVaryingTarget
    time_feature_index: int
    parameterization: str = "linear"
    outputscale_bias_limit: float = 4.0
    outputscale_slope_limit: float = 4.0


class WarpType(str, Enum):
    """Supported input-warping transformations for time-like inputs.

    KUMARASWAMY: Applies the Kumaraswamy CDF as a monotone warp to map
        inputs in [0, 1] → [0, 1] with learnable concentration parameters.
        This is the BoTorch-native warp via botorch.models.transforms.input.Warp.
    """

    KUMARASWAMY = "kumaraswamy"


class InputWarpingSpec(BaseModel):
    """Tier 2 specification for input warping on a designated time-like feature.

    Applies a monotone parametric transformation to the time-like input dimension
    before kernel evaluation.  The warping is explicit in the DSL and applied as a
    BoTorch input transform.  Both the warp parameters and the kernel hyperparameters
    are optimized jointly.

    Requirements on the input:
    - The time-like feature should be normalized to [0, 1] (i.e., input scaling
      should be enabled in ExecutionSpec) for the Kumaraswamy warp to map correctly.

    Attributes:
        warp_type: The warping family.  Currently only KUMARASWAMY is supported.
        time_feature_index: Zero-based column index of the time-like feature to warp.
        concentration0: Optional fixed initialization for the first Kumaraswamy
            concentration parameter (a > 0).  When None, BoTorch initializes
            the parameter to a learnable default.
        concentration1: Optional fixed initialization for the second Kumaraswamy
            concentration parameter (b > 0).  When None, BoTorch initializes
            the parameter to a learnable default.
    """

    warp_type: WarpType = WarpType.KUMARASWAMY
    time_feature_index: int
    concentration0: float | None = None
    concentration1: float | None = None


class ExecutionSpec(BaseModel):
    """Execution semantics that affect how a validated DSL is run.

    Attributes:
        input_scaling: Whether continuous inputs are min-max scaled before model building.
        outcome_standardization: Whether BoTorch outcome transforms standardize outputs where supported.
        recency_filtering: Optional recency-filtering configuration for time-driven
            non-stationarity.  When set, old observations are removed from the training
            set before fitting.  This is dataset truncation, not likelihood weighting.
        input_warping: Optional Tier 2 input-warping configuration.  When set, the
            designated time-like feature dimension is transformed by a monotone warp
            before kernel evaluation.
    """

    input_scaling: bool = True
    outcome_standardization: bool = True
    recency_filtering: RecencyFilteringSpec | None = None
    input_warping: InputWarpingSpec | None = None


class NoiseSpec(BaseModel):
    """Specification for observation noise assumptions.

    Attributes:
        fixed: Whether the noise level is fixed (not optimised).
        noise_value: Fixed noise variance value; None means learnable.
        prior: Optional prior on the noise variance.
        heteroskedastic_noise: Design-guardrail placeholder for future heteroskedastic
            (input- or time-dependent) noise support.  Currently unsupported — the
            validator rejects specs with this flag set to True.  Set to False (default)
            to use the standard homoskedastic noise model.
    """

    fixed: bool = False
    noise_value: float | None = None
    prior: PriorSpec | None = None
    heteroskedastic_noise: bool = False


class MeanSpec(BaseModel):
    """Specification for a GP mean function.

    Attributes:
        mean_type: The mean function family to use.
    """

    mean_type: MeanFunctionType


class KernelSpec(BaseModel):
    """Specification for a single kernel or composed kernel.

    Attributes:
        kernel_type: The kernel family to use.
        ard: Whether to use Automatic Relevance Determination (separate lengthscale per feature).
        lengthscale_prior: Optional prior on the lengthscale hyperparameter.
        outputscale_prior: Optional prior on the outputscale hyperparameter.
        composition: How this kernel is combined with others in the same feature group.
        children: Sub-kernels for composed (additive/multiplicative) kernels.
        period_prior: Optional prior on the period (for Periodic kernel only).
        period_length: Optional fixed initialization value for the Periodic period length.
        rq_alpha: Optional fixed initialization value for the RQ alpha parameter.
        polynomial_power: Optional degree for the Polynomial kernel.
        polynomial_offset: Optional fixed initialization value for the Polynomial offset.
        num_mixtures: Number of components for the Spectral Mixture kernel.
        spectral_init: Initialization strategy for the Spectral Mixture kernel.
        bnn_depth: Optional depth for the Infinite Width BNN kernel.
        exponential_decay_power: Optional fixed initialization value for the ExponentialDecay power.
        exponential_decay_offset: Optional fixed initialization value for the ExponentialDecay offset.
        changepoint_location: Initial value for the changepoint location parameter (for
            Changepoint kernel only).  This is a value in the (possibly scaled) feature
            space of the designated time-like input.  Must be finite.
        changepoint_steepness: Initial value for the sigmoid steepness at the changepoint
            (for Changepoint kernel only).  Controls how sharply the kernel transitions
            between the before and after regimes.  Must be > 0.  Defaults to 1.0.
        time_varying: Optional Tier 2 time-varying hyperparameter specification.  When
            set, the builder wraps this kernel in a TimeVaryingKernel module that adds
            a learned smooth modulation over the designated time-like input.  The base
            kernel is built from the other fields of this KernelSpec (kernel_type,
            ard, etc.) and then wrapped.  Cannot be combined with composed kernels
            (children must be empty).
    """

    kernel_type: KernelType
    ard: bool = False
    lengthscale_prior: PriorSpec | None = None
    outputscale_prior: PriorSpec | None = None
    composition: CompositionType = CompositionType.NONE
    children: list[KernelSpec] = Field(default_factory=list)
    period_prior: PriorSpec | None = None
    period_length: float | None = None
    rq_alpha: float | None = None
    polynomial_power: int | None = None
    polynomial_offset: float | None = None
    num_mixtures: int | None = None
    spectral_init: SpectralMixtureInitialization = SpectralMixtureInitialization.FROM_DATA
    bnn_depth: int | None = None
    exponential_decay_power: float | None = None
    exponential_decay_offset: float | None = None
    changepoint_location: float | None = None
    changepoint_steepness: float | None = None
    time_varying: TimeVaryingSpec | None = None


class LeafKernelSpec(BaseModel):
    """A leaf (base) kernel node with no sub-kernels.

    Attributes:
        kind: Discriminator field, always "leaf".
        kernel_type: The kernel family.
        ard: Whether to use Automatic Relevance Determination.
        lengthscale_prior: Optional prior on the lengthscale.
        outputscale_prior: Optional prior on the outputscale.
        period_prior: Optional prior on the period (Periodic kernel only).
        period_length: Optional fixed initialization for the Periodic period.
        rq_alpha: Optional fixed initialization for the RQ alpha parameter.
        polynomial_power: Optional degree for the Polynomial kernel.
        polynomial_offset: Optional fixed initialization for the Polynomial offset.
        num_mixtures: Number of components for the Spectral Mixture kernel.
        spectral_init: Initialization strategy for the Spectral Mixture kernel.
        bnn_depth: Optional depth for the Infinite Width BNN kernel.
        exponential_decay_power: Optional fixed initialization for ExponentialDecay power.
        exponential_decay_offset: Optional fixed initialization for ExponentialDecay offset.
        time_varying: Optional Tier 2 time-varying hyperparameter specification.
    """

    kind: Literal["leaf"] = "leaf"
    kernel_type: KernelType
    ard: bool = False
    lengthscale_prior: PriorSpec | None = None
    outputscale_prior: PriorSpec | None = None
    period_prior: PriorSpec | None = None
    period_length: float | None = None
    rq_alpha: float | None = None
    polynomial_power: int | None = None
    polynomial_offset: float | None = None
    num_mixtures: int | None = None
    spectral_init: SpectralMixtureInitialization = SpectralMixtureInitialization.FROM_DATA
    bnn_depth: int | None = None
    exponential_decay_power: float | None = None
    exponential_decay_offset: float | None = None
    time_varying: TimeVaryingSpec | None = None


class CompositeKernelSpec(BaseModel):
    """An additive or multiplicative composition over child KernelExprs.

    Attributes:
        kind: Discriminator field, always "composite".
        composition: Must be ADDITIVE or MULTIPLICATIVE.
        children: Sub-kernels; must contain >= 2 items.
        outputscale_prior: Optional prior on the outputscale (multiplicative only).
        time_varying: Optional Tier 2 time-varying hyperparameter specification.
    """

    kind: Literal["composite"] = "composite"
    composition: CompositionType
    children: list[KernelExpr] = Field(default_factory=list)  # type: ignore[type-arg]
    outputscale_prior: PriorSpec | None = None
    time_varying: TimeVaryingSpec | None = None

    @field_validator("composition")
    @classmethod
    def _validate_composition(cls, value: CompositionType) -> CompositionType:
        if value not in {CompositionType.ADDITIVE, CompositionType.MULTIPLICATIVE}:
            raise ValueError(
                f"CompositeKernelSpec.composition must be ADDITIVE or MULTIPLICATIVE, got {value!r}."
            )
        return value

    @field_validator("children")
    @classmethod
    def _validate_children(cls, value: list) -> list:  # type: ignore[type-arg]
        if len(value) < 2:  # noqa: PLR2004
            raise ValueError(
                f"CompositeKernelSpec must have at least 2 children, got {len(value)}."
            )
        return value


class ChangepointKernelSpec(BaseModel):
    """A changepoint kernel transitioning between two child kernel regimes.

    Attributes:
        kind: Discriminator field, always "changepoint".
        kernel_before: Kernel active before the changepoint.
        kernel_after: Kernel active after the changepoint.
        changepoint_location: Initial value for the changepoint location parameter.
        changepoint_steepness: Initial steepness; controls sharpness of transition.
        outputscale_prior: Optional prior on the outputscale.
    """

    kind: Literal["changepoint"] = "changepoint"
    kernel_before: KernelExpr  # type: ignore[type-arg]
    kernel_after: KernelExpr  # type: ignore[type-arg]
    changepoint_location: float | None = None
    changepoint_steepness: float | None = None
    outputscale_prior: PriorSpec | None = None


KernelExpr = Annotated[
    Union[LeafKernelSpec, CompositeKernelSpec, ChangepointKernelSpec],
    Field(discriminator="kind"),
]

# Resolve forward references in models with recursive KernelExpr fields.
CompositeKernelSpec.model_rebuild()
ChangepointKernelSpec.model_rebuild()


def _kernel_spec_to_expr(spec: KernelSpec) -> KernelExpr:  # type: ignore[type-arg]
    """Convert a legacy KernelSpec to the canonical KernelExpr discriminated union.

    Args:
        spec: A KernelSpec instance to normalize.

    Returns:
        A LeafKernelSpec, CompositeKernelSpec, or ChangepointKernelSpec.

    Raises:
        ValueError: If the spec has an ambiguous or invalid shape.
    """
    if spec.kernel_type == KernelType.CHANGEPOINT:
        if len(spec.children) != 2:  # noqa: PLR2004
            raise ValueError(
                f"Changepoint KernelSpec must have exactly 2 children, got {len(spec.children)}."
            )
        return ChangepointKernelSpec(
            kernel_before=_kernel_spec_to_expr(spec.children[0]),
            kernel_after=_kernel_spec_to_expr(spec.children[1]),
            changepoint_location=spec.changepoint_location,
            changepoint_steepness=spec.changepoint_steepness,
            outputscale_prior=spec.outputscale_prior,
        )

    if spec.children:
        if spec.composition not in {CompositionType.ADDITIVE, CompositionType.MULTIPLICATIVE}:
            raise ValueError(
                f"KernelSpec with children must have ADDITIVE or MULTIPLICATIVE composition, "
                f"got {spec.composition!r}."
            )
        return CompositeKernelSpec(
            composition=spec.composition,
            children=[_kernel_spec_to_expr(child) for child in spec.children],
            outputscale_prior=spec.outputscale_prior,
            time_varying=spec.time_varying,
        )

    return LeafKernelSpec(
        kernel_type=spec.kernel_type,
        ard=spec.ard,
        lengthscale_prior=spec.lengthscale_prior,
        outputscale_prior=spec.outputscale_prior,
        period_prior=spec.period_prior,
        period_length=spec.period_length,
        rq_alpha=spec.rq_alpha,
        polynomial_power=spec.polynomial_power,
        polynomial_offset=spec.polynomial_offset,
        num_mixtures=spec.num_mixtures,
        spectral_init=spec.spectral_init,
        bnn_depth=spec.bnn_depth,
        exponential_decay_power=spec.exponential_decay_power,
        exponential_decay_offset=spec.exponential_decay_offset,
        time_varying=spec.time_varying,
    )


def _legacy_dict_to_kernel_expr(value: dict[str, Any]) -> dict[str, Any]:
    """Add a 'kind' discriminator to a legacy KernelSpec-shaped dict.

    Args:
        value: A dict without 'kind', shaped like a KernelSpec.

    Returns:
        A dict with 'kind' inserted for Pydantic discriminated-union parsing.

    Raises:
        ValueError: If the shape is ambiguous (children present but composition=none).
    """
    children = value.get("children", [])
    kernel_type = value.get("kernel_type", "")
    composition = value.get("composition", "none")

    if kernel_type == KernelType.CHANGEPOINT.value or kernel_type == "Changepoint":
        raw_children = children
        tagged_children: list[dict[str, Any]] = []
        for child in raw_children:
            if isinstance(child, dict) and "kind" not in child:
                tagged_children.append(_legacy_dict_to_kernel_expr(child))
            else:
                tagged_children.append(child)
        return {
            **value,
            "kind": "changepoint",
            "kernel_before": tagged_children[0] if len(tagged_children) > 0 else value.get("kernel_before"),
            "kernel_after": tagged_children[1] if len(tagged_children) > 1 else value.get("kernel_after"),
        }

    if children:
        if composition not in {"additive", "multiplicative"}:
            raise ValueError(
                f"Legacy KernelSpec dict has children but composition={composition!r}; "
                "cannot determine kind. Use ADDITIVE or MULTIPLICATIVE."
            )
        tagged: list[dict[str, Any]] = []
        for child in children:
            if isinstance(child, dict) and "kind" not in child:
                tagged.append(_legacy_dict_to_kernel_expr(child))
            else:
                tagged.append(child)
        return {**value, "kind": "composite", "children": tagged}

    return {**value, "kind": "leaf"}


class FeatureGroupSpec(BaseModel):
    """Specification for a group of input features sharing a common kernel.

    Attributes:
        name: Human-readable name for this feature group.
        feature_indices: Zero-based column indices of the features in this group.
        kernel: The kernel specification applied to this group.
    """

    name: str
    feature_indices: list[int] = Field(default_factory=list)
    kernel: KernelExpr  # type: ignore[type-arg]

    @field_validator("kernel", mode="before")
    @classmethod
    def _normalize_legacy_kernel(cls, value: Any) -> Any:
        """Normalize a legacy KernelSpec payload to the new KernelExpr discriminated union."""
        if isinstance(value, (LeafKernelSpec, CompositeKernelSpec, ChangepointKernelSpec)):
            return value
        if isinstance(value, KernelSpec):
            return _kernel_spec_to_expr(value)
        if isinstance(value, dict) and "kind" not in value:
            return _legacy_dict_to_kernel_expr(value)
        return value


class GPSpec(BaseModel):
    """Top-level GP model specification (the DSL root object).

    This is the authoritative representation of a GP model architecture. All
    model construction is derived from this specification.

    Attributes:
        model_class: The BoTorch model class to instantiate.
        feature_groups: One or more feature groups, each with their own kernel.
        mean: Optional shared mean-function specification.
        output_means: Optional per-output or per-task mean overrides.
        noise: Noise model specification.
        execution: Execution semantics that affect preprocessing and model transforms.
        input_dim: Total number of input features.
        output_dim: Number of output dimensions.
        task_feature_index: Column index of the task indicator (MultiTaskGP only).
        task_values: Optional explicit task domain for MultiTaskGP targeted overrides.
        multitask_rank: Rank of the inter-task covariance (MultiTaskGP only).
        group_composition: How feature-group kernels are combined.
        description: Optional human-readable summary of the specification.
    """

    model_class: ModelClass = ModelClass.SINGLE_TASK_GP
    feature_groups: list[FeatureGroupSpec] = Field(default_factory=list)
    mean: MeanSpec | None = None
    output_means: dict[int, MeanSpec] = Field(default_factory=dict)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    input_dim: int = 1
    output_dim: int = 1
    task_feature_index: int | None = None
    task_values: list[int] | None = None
    multitask_rank: int | None = None
    group_composition: CompositionType = CompositionType.ADDITIVE
    description: str = ""
