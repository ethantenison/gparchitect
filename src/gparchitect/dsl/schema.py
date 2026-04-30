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
    Pydantic model classes: GPSpec, KernelSpec, FeatureGroupSpec, PriorSpec, NoiseSpec,
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

from pydantic import BaseModel, Field


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
    """

    target: TimeVaryingTarget
    time_feature_index: int
    parameterization: str = "linear"


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


class FeatureGroupSpec(BaseModel):
    """Specification for a group of input features sharing a common kernel.

    Attributes:
        name: Human-readable name for this feature group.
        feature_indices: Zero-based column indices of the features in this group.
        kernel: The kernel specification applied to this group.
    """

    name: str
    feature_indices: list[int] = Field(default_factory=list)
    kernel: KernelSpec


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
