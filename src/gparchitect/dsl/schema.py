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
    MeanSpec, ExecutionSpec, RecencyWeightingSpec.

Non-obvious design decisions:
    - All fields use native Python typing (list, dict, X | None) per project style.
    - JSON serialization is guaranteed via Pydantic's model_dump(mode="json").
    - Enums for kernel types and model classes ensure stability across versions.
    - DSL is independent of natural-language phrasing; identical GP architectures
      must always produce identical DSL regardless of how they are described.
    - RecencyWeightingSpec is nested inside ExecutionSpec to keep data/fitting
      concerns separate from kernel/model concerns.
    - heteroskedastic_noise is a forward-compatibility placeholder in NoiseSpec:
      it is currently rejected by the validator, but its presence in the schema
      prevents DSL churn when Tier 2 heteroskedastic support is added.

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


class RecencyWeightingMode(str, Enum):
    """Supported recency-weighting strategies for time-driven non-stationarity.

    SLIDING_WINDOW: Keep only observations within a fixed time window ending at the
        most recent observation.  Older observations are discarded entirely.
    EXPONENTIAL_DISCOUNT: Keep all observations but discard those whose exponential
        weight falls below a minimum threshold.  This is equivalent to a soft
        sliding window with a smooth boundary.
    """

    SLIDING_WINDOW = "sliding_window"
    EXPONENTIAL_DISCOUNT = "exponential_discount"


class RecencyWeightingSpec(BaseModel):
    """Specification for recency-based downweighting of stale observations.

    Used to model time-driven non-stationarity by reducing the effective influence
    of older data points.

    Attributes:
        mode: The weighting strategy to apply.
        time_feature_index: Zero-based column index of the time-like feature in train_X
            that determines observation recency.
        window_size: For SLIDING_WINDOW — width of the time window in the (possibly
            scaled) feature space.  Observations older than max_time − window_size
            are dropped.  Must be > 0.
        discount_rate: For EXPONENTIAL_DISCOUNT — rate λ in w_i = exp(−λ · Δt_i).
            Observations with w_i < min_weight are dropped.  Must be > 0.
        min_weight: Minimum observation weight below which observations are discarded
            in EXPONENTIAL_DISCOUNT mode.  Defaults to 0.01.  Must be in (0, 1).
    """

    mode: RecencyWeightingMode
    time_feature_index: int
    window_size: float | None = None
    discount_rate: float | None = None
    min_weight: float = 0.01


class ExecutionSpec(BaseModel):
    """Execution semantics that affect how a validated DSL is run.

    Attributes:
        input_scaling: Whether continuous inputs are min-max scaled before model building.
        outcome_standardization: Whether BoTorch outcome transforms standardize outputs where supported.
        recency_weighting: Optional recency-weighting configuration for time-driven
            non-stationarity.  When set, old observations are filtered or downweighted
            before fitting.
    """

    input_scaling: bool = True
    outcome_standardization: bool = True
    recency_weighting: RecencyWeightingSpec | None = None


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
