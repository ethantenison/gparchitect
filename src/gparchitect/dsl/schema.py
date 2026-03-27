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
    MeanSpec.

Non-obvious design decisions:
    - All fields use native Python typing (list, dict, X | None) per project style.
    - JSON serialization is guaranteed via Pydantic's model_dump(mode="json").
    - Enums for kernel types and model classes ensure stability across versions.
    - DSL is independent of natural-language phrasing; identical GP architectures
      must always produce identical DSL regardless of how they are described.

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


class PriorSpec(BaseModel):
    """Specification for a GP hyperparameter prior.

    Attributes:
        distribution: Name of the prior distribution (e.g. "Normal", "LogNormal", "Gamma").
        params: Distribution parameters as a name→value mapping.
    """

    distribution: str
    params: dict[str, float] = Field(default_factory=dict)


class NoiseSpec(BaseModel):
    """Specification for observation noise assumptions.

    Attributes:
        fixed: Whether the noise level is fixed (not optimised).
        noise_value: Fixed noise variance value; None means learnable.
        prior: Optional prior on the noise variance.
    """

    fixed: bool = False
    noise_value: float | None = None
    prior: PriorSpec | None = None


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
        input_dim: Total number of input features.
        output_dim: Number of output dimensions (1 for single-task).
        task_feature_index: Column index of the task indicator (MultiTaskGP only).
        multitask_rank: Rank of the inter-task covariance (MultiTaskGP only).
        group_composition: How feature-group kernels are combined.
        description: Optional human-readable summary of the specification.
    """

    model_class: ModelClass = ModelClass.SINGLE_TASK_GP
    feature_groups: list[FeatureGroupSpec] = Field(default_factory=list)
    mean: MeanSpec | None = None
    output_means: dict[int, MeanSpec] = Field(default_factory=dict)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    input_dim: int = 1
    output_dim: int = 1
    task_feature_index: int | None = None
    multitask_rank: int | None = None
    group_composition: CompositionType = CompositionType.ADDITIVE
    description: str = ""
