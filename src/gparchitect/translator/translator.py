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
        - ARD is enabled by default for kernels that support it and can be disabled explicitly.

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
    ExecutionSpec,
    FeatureGroupSpec,
    GPSpec,
    InputWarpingSpec,
    KernelSpec,
    KernelType,
    MeanFunctionType,
    MeanSpec,
    ModelClass,
    NoiseSpec,
    PriorDistribution,
    PriorSpec,
    RecencyFilteringMode,
    RecencyFilteringSpec,
    SpectralMixtureInitialization,
    TimeVaryingSpec,
    TimeVaryingTarget,
    WarpType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword → kernel type mapping (ordered by specificity)
# ---------------------------------------------------------------------------
_KERNEL_KEYWORDS: list[tuple[re.Pattern[str], KernelType]] = [
    (re.compile(r"\binfinite[\s\-_]?width\s+bnn\b|\biwbnn\b", re.IGNORECASE), KernelType.INFINITE_WIDTH_BNN),
    (re.compile(r"\bexponential[\s\-_]?decay\b", re.IGNORECASE), KernelType.EXPONENTIAL_DECAY),
    (re.compile(r"\bspectral\s+mixture\b", re.IGNORECASE), KernelType.SPECTRAL_MIXTURE),
    (re.compile(r"\brational\s+quadratic\b|\brq\b", re.IGNORECASE), KernelType.RQ),
    (re.compile(r"\bmatern.?1[/\-_]?2\b", re.IGNORECASE), KernelType.MATERN_12),
    (re.compile(r"\bmatern.?3[/\-_]?2\b", re.IGNORECASE), KernelType.MATERN_32),
    (re.compile(r"\bmatern.?5[/\-_]?2\b", re.IGNORECASE), KernelType.MATERN_52),
    (re.compile(r"\bmatern\b", re.IGNORECASE), KernelType.MATERN_52),  # default Matern
    (re.compile(r"\brbf\b|\bsquared.?exponential\b|\bgaussian.?kernel\b", re.IGNORECASE), KernelType.RBF),
    (re.compile(r"\bperiodic\b", re.IGNORECASE), KernelType.PERIODIC),
    (re.compile(r"\blinear\b", re.IGNORECASE), KernelType.LINEAR),
    (re.compile(r"\bpolynomial\b", re.IGNORECASE), KernelType.POLYNOMIAL),
    (
        re.compile(
            r"\bchangepoint\b|\bchange[\s\-_]?point\b|\bregime[\s\-_]?shift\b|\bregime[\s\-_]?change\b",
            re.IGNORECASE,
        ),
        KernelType.CHANGEPOINT,
    ),
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
_DISABLE_ARD_PATTERN = re.compile(
    r"\b(?:no|without|disable|disabled|turn off|off)\s+ard\b|"
    r"\bshared\s+lengthscale\b|\bsingle\s+lengthscale\b|\bsame\s+lengthscale\b",
    re.IGNORECASE,
)
_FIXED_NOISE_PATTERN = re.compile(r"\bfixed.?noise\b|\bnoise.?free\b|\bnoiseless\b", re.IGNORECASE)
_ADDITIVE_PATTERN = re.compile(r"\badditive\b|\bsum.?of.?kernel\b", re.IGNORECASE)
_MULTIPLICATIVE_PATTERN = re.compile(r"\bmultiplicative\b|\bproduct.?of.?kernel\b", re.IGNORECASE)
_FEATURE_PREPOSITION_PATTERN = re.compile(r"\b(?:on|for|across|over|applied to|using)\b", re.IGNORECASE)
_RQ_ALPHA_PATTERN = re.compile(r"\balpha\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE)
_NUM_MIXTURES_PATTERN = re.compile(
    r"\b(?:num(?:ber)?\s+of\s+mixtures?|n[_\- ]?mixtures?)\s*(?:=|of)?\s*(\d+)\b|"
    r"\b(\d+)\s*(?:component|components|mixture|mixtures)\b",
    re.IGNORECASE,
)
_EMPIRICAL_SPECTRUM_PATTERN = re.compile(
    r"\b(?:emp(?:irical)?\s+spectrum|empspect|evenly\s+spaced)\b",
    re.IGNORECASE,
)
_FROM_DATA_INIT_PATTERN = re.compile(
    r"\b(?:initialize|initialized|initialise|initialised|init)\w*\s+(?:it\s+)?from\s+data\b|"
    r"\bunevenly\s+spaced\b",
    re.IGNORECASE,
)
_PERIOD_LENGTH_PATTERN = re.compile(
    r"\bperiod(?:\s+length)?\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b",
    re.IGNORECASE,
)
_POLYNOMIAL_POWER_PATTERN = re.compile(
    r"\b(?:degree|order|power)\s*(?:=|of)?\s*(\d+)\b",
    re.IGNORECASE,
)
_OFFSET_PATTERN = re.compile(r"\boffset\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE)
_DEPTH_PATTERN = re.compile(
    r"\bdepth\s*(?:=|of)?\s*(\d+)\b|\b(\d+)\s*(?:layer|layers)\b",
    re.IGNORECASE,
)
_POWER_PATTERN = re.compile(r"\bpower\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE)
_MEAN_PATTERN = re.compile(
    r"\b(?:(output|task)\s+(\d+)\s+(?:uses?\s+(?:(?:a|an)\s+)?)?)?"
    r"(constant|zero|linear)\s+mean\b(?:\s+for\s+(output|task)\s+(\d+))?",
    re.IGNORECASE,
)
_PRIOR_DISTRIBUTION_ALIASES = {
    "lognormal": PriorDistribution.LOG_NORMAL,
    "log-normal": PriorDistribution.LOG_NORMAL,
    "halfcauchy": PriorDistribution.HALF_CAUCHY,
    "half-cauchy": PriorDistribution.HALF_CAUCHY,
    "half cauchy": PriorDistribution.HALF_CAUCHY,
    "normal": PriorDistribution.NORMAL,
    "gamma": PriorDistribution.GAMMA,
    "uniform": PriorDistribution.UNIFORM,
}
_PRIOR_TARGET_ALIASES = {
    "lengthscale": "lengthscale",
    "length scale": "lengthscale",
    "outputscale": "outputscale",
    "output scale": "outputscale",
    "period": "period",
    "period length": "period",
    "noise": "noise",
    "observation noise": "noise",
    "noise variance": "noise",
}
_PRIOR_DISTRIBUTION_PATTERN = (
    r"lognormal|log-normal|halfcauchy|half-cauchy|half\s+cauchy|normal|gamma|uniform"
)
_PRIOR_TARGET_PATTERN = (
    r"lengthscale|length\s+scale|outputscale|output\s+scale|period(?:\s+length)?|"
    r"observation\s+noise|noise\s+variance|noise"
)
_PRIOR_END_PATTERN = (
    r"(?=(?:\b(?:lognormal|log-normal|halfcauchy|half-cauchy|half\s+cauchy|normal|gamma|uniform)"
    r"\s+prior\s+(?:on|for)\b)|(?:\b(?:lengthscale|length\s+scale|outputscale|output\s+scale|"
    r"period(?:\s+length)?|observation\s+noise|noise\s+variance|noise)\b\s+"
    r"(?:has|have|uses?|using|with|follows?)\s+(?:a|an)?\s*"
    r"(?:lognormal|log-normal|halfcauchy|half-cauchy|half\s+cauchy|normal|gamma|uniform)"
    r"\s+prior\b)|(?:[,;]|(?<!\d)\.(?!\d))|$)"
)
_PRIOR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        rf"\b(?P<distribution>{_PRIOR_DISTRIBUTION_PATTERN})\s+prior\s+(?:on|for)\s+"
        rf"(?P<target>{_PRIOR_TARGET_PATTERN})\b(?P<params>.*?){_PRIOR_END_PATTERN}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<target>{_PRIOR_TARGET_PATTERN})\b\s+"
        rf"(?:has|have|uses?|using|with|follows?)\s+(?:a|an)?\s*"
        rf"(?P<distribution>{_PRIOR_DISTRIBUTION_PATTERN})\s+prior\b"
        rf"(?P<params>.*?){_PRIOR_END_PATTERN}",
        re.IGNORECASE,
    ),
]

_ARD_SUPPORTED_KERNELS = {
    KernelType.RBF,
    KernelType.RQ,
    KernelType.MATERN_12,
    KernelType.MATERN_32,
    KernelType.MATERN_52,
    KernelType.PERIODIC,
}
_PRIOR_PARAM_PATTERNS: dict[PriorDistribution, dict[str, re.Pattern[str]]] = {
    PriorDistribution.NORMAL: {
        "loc": re.compile(r"\b(?:loc|mean|mu)\s*(?:=|of)?\s*(-?[0-9]*\.?[0-9]+)\b", re.IGNORECASE),
        "scale": re.compile(r"\b(?:scale|std(?:dev)?|sigma)\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE),
    },
    PriorDistribution.LOG_NORMAL: {
        "loc": re.compile(r"\b(?:loc|mean|mu)\s*(?:=|of)?\s*(-?[0-9]*\.?[0-9]+)\b", re.IGNORECASE),
        "scale": re.compile(r"\b(?:scale|std(?:dev)?|sigma)\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE),
    },
    PriorDistribution.GAMMA: {
        "concentration": re.compile(
            r"\b(?:concentration|shape|alpha)\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b",
            re.IGNORECASE,
        ),
        "rate": re.compile(r"\b(?:rate|beta)\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE),
    },
    PriorDistribution.HALF_CAUCHY: {
        "scale": re.compile(r"\b(?:scale|beta)\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE),
    },
    PriorDistribution.UNIFORM: {
        "a": re.compile(r"\b(?:a|low|lower|min(?:imum)?)\s*(?:=|of)?\s*(-?[0-9]*\.?[0-9]+)\b", re.IGNORECASE),
        "b": re.compile(r"\b(?:b|high|upper|max(?:imum)?)\s*(?:=|of)?\s*(-?[0-9]*\.?[0-9]+)\b", re.IGNORECASE),
    },
}
_UNIFORM_RANGE_PATTERN = re.compile(
    r"\b(?:between\s*(-?[0-9]*\.?[0-9]+)\s+and\s*(-?[0-9]*\.?[0-9]+)|"
    r"from\s*(-?[0-9]*\.?[0-9]+)\s+to\s*(-?[0-9]*\.?[0-9]+))\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Changepoint-specific patterns
# ---------------------------------------------------------------------------
_CHANGEPOINT_LOCATION_PATTERN = re.compile(
    r"\b(?:at|location|loc|switches?\s+at|changes?\s+at|changepoint\s+at|before[/\s]+after)\s+"
    r"(?:t(?:ime)?\s*=?\s*)?(-?[0-9]*\.?[0-9]+)\b",
    re.IGNORECASE,
)
_CHANGEPOINT_STEEPNESS_PATTERN = re.compile(
    r"\bsteepness\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Recency filtering patterns (Tier 1)
# ---------------------------------------------------------------------------
_SLIDING_WINDOW_PATTERN = re.compile(
    r"\b(?:sliding[\s\-_]?window|time[\s\-_]?window|recent\s+window|window[\s\-_]?size)\b",
    re.IGNORECASE,
)
_EXPONENTIAL_DISCOUNT_PATTERN = re.compile(
    r"\b(?:exponential[\s\-_]?(?:discount|decay|forget(?:ting)?)|forgetting|"
    r"down[\s\-_]?weight(?:ing)?|recency[\s\-_]?(?:weight(?:ing)?|filter(?:ing)?)|forget(?:ting)?\s+old)\b",
    re.IGNORECASE,
)
_WINDOW_SIZE_PATTERN = re.compile(
    r"\b(?:window(?:\s+size)?|last)\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b",
    re.IGNORECASE,
)
_DISCOUNT_RATE_PATTERN = re.compile(
    r"\b(?:discount[\s\-_]?rate|decay[\s\-_]?rate|lambda|rate)\s*(?:=|of)?\s*([0-9]*\.?[0-9]+)\b",
    re.IGNORECASE,
)
_TIME_FEATURE_PATTERN = re.compile(
    r"\b(?:time[\s\-_]?(?:feature|column|index|col)|time)\s*(?:=|index|col(?:umn)?)?\s*(\d+)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tier 2: time-varying hyperparameter patterns
# ---------------------------------------------------------------------------
_TIME_VARYING_OUTPUTSCALE_PATTERN = re.compile(
    r"\b(?:time[\s\-_]?vary(?:ing)?\s+(?:output[\s\-_]?scale|amplitude|signal|variance)|"
    r"(?:output[\s\-_]?scale|amplitude|signal|variance)\s+(?:change|changes|vary|varies|varies\s+over)\s+(?:over\s+)?time)\b",
    re.IGNORECASE,
)
_TIME_VARYING_LENGTHSCALE_PATTERN = re.compile(
    r"\b(?:time[\s\-_]?vary(?:ing)?\s+(?:length[\s\-_]?scale|smoothness)|"
    r"(?:length[\s\-_]?scale|smoothness)\s+(?:change|changes|vary|varies)\s+(?:over\s+)?time|"
    r"non[\s\-_]?stationary\s+smoothness)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tier 2: input warping patterns
# ---------------------------------------------------------------------------
_INPUT_WARP_PATTERN = re.compile(
    r"\b(?:warp\s+time|warped\s+time(?:\s+axis)?|non[\s\-_]?linear\s+time\s+warp|"
    r"input[\s\-_]?warp(?:ing)?(?:\s+(?:on\s+)?time)?|kumaraswamy\s+warp)\b",
    re.IGNORECASE,
)


def _extract_rq_alpha(instruction: str) -> float | None:
    """Extract an optional RQ alpha value from an instruction fragment."""
    match = _RQ_ALPHA_PATTERN.search(instruction)
    if match is None:
        return None
    return float(match.group(1))


def _extract_changepoint_location(instruction: str) -> float | None:
    """Extract an optional changepoint location from an instruction fragment."""
    match = _CHANGEPOINT_LOCATION_PATTERN.search(instruction)
    if match is None:
        return None
    return float(match.group(1))


def _extract_changepoint_steepness(instruction: str) -> float | None:
    """Extract an optional changepoint steepness from an instruction fragment."""
    match = _CHANGEPOINT_STEEPNESS_PATTERN.search(instruction)
    if match is None:
        return None
    return float(match.group(1))


def _extract_num_mixtures(instruction: str) -> int | None:
    """Extract an optional Spectral Mixture component count."""
    match = _NUM_MIXTURES_PATTERN.search(instruction)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def _detect_spectral_init(instruction: str) -> SpectralMixtureInitialization:
    """Detect the preferred Spectral Mixture initialization strategy."""
    if _EMPIRICAL_SPECTRUM_PATTERN.search(instruction):
        return SpectralMixtureInitialization.FROM_EMPIRICAL_SPECTRUM
    if _FROM_DATA_INIT_PATTERN.search(instruction):
        return SpectralMixtureInitialization.FROM_DATA
    return SpectralMixtureInitialization.FROM_DATA


def _extract_period_length(instruction: str) -> float | None:
    """Extract an optional Periodic period length from an instruction fragment."""
    match = _PERIOD_LENGTH_PATTERN.search(instruction)
    if match is None:
        return None
    return float(match.group(1))


def _extract_polynomial_power(instruction: str) -> int | None:
    """Extract an optional Polynomial degree from an instruction fragment."""
    match = _POLYNOMIAL_POWER_PATTERN.search(instruction)
    if match is None:
        return None
    return int(match.group(1))


def _extract_offset(instruction: str) -> float | None:
    """Extract an optional offset parameter from an instruction fragment."""
    match = _OFFSET_PATTERN.search(instruction)
    if match is None:
        return None
    return float(match.group(1))


def _extract_depth(instruction: str) -> int | None:
    """Extract an optional network depth from an instruction fragment."""
    match = _DEPTH_PATTERN.search(instruction)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def _extract_power(instruction: str) -> float | None:
    """Extract an optional power parameter from an instruction fragment."""
    match = _POWER_PATTERN.search(instruction)
    if match is None:
        return None
    return float(match.group(1))


def _detect_recency_filtering(
    instruction: str,
    input_dim: int,
    input_feature_names: list[str] | None,
) -> RecencyFilteringSpec | None:
    """Detect an optional recency-filtering specification from an instruction.

    Returns a RecencyFilteringSpec (dataset truncation before fitting) if the
    instruction mentions sliding-window or exponential-discount filtering.
    This is NOT true likelihood-weighted GP inference.

    Args:
        instruction: Free-text instruction string.
        input_dim: Total number of input features (excluding task column).
        input_feature_names: Optional list of continuous feature column names.

    Returns:
        A RecencyFilteringSpec if recency filtering was requested, otherwise None.
    """
    has_sliding = _SLIDING_WINDOW_PATTERN.search(instruction) is not None
    has_discount = _EXPONENTIAL_DISCOUNT_PATTERN.search(instruction) is not None

    if not has_sliding and not has_discount:
        return None

    # Determine the time feature index: explicit number, or "time" column name, or default to 0.
    time_feature_index = 0
    tf_match = _TIME_FEATURE_PATTERN.search(instruction)
    if tf_match is not None:
        time_feature_index = int(tf_match.group(1))
    elif input_feature_names:
        for idx, name in enumerate(input_feature_names):
            if re.search(r"\btime\b|\btimestamp\b|\bt\b|\bdate\b|\bday\b|\bweek\b|\bmonth\b|\byear\b",
                         name, re.IGNORECASE):
                time_feature_index = idx
                break

    # Clamp to valid range
    time_feature_index = max(0, min(time_feature_index, input_dim - 1))

    if has_sliding:
        window_size: float | None = None
        ws_match = _WINDOW_SIZE_PATTERN.search(instruction)
        if ws_match is not None:
            window_size = float(ws_match.group(1))
        else:
            window_size = 0.5  # default: half of the (scaled) feature range
        return RecencyFilteringSpec(
            mode=RecencyFilteringMode.SLIDING_WINDOW,
            time_feature_index=time_feature_index,
            window_size=window_size,
        )

    # Exponential discount
    discount_rate: float | None = None
    dr_match = _DISCOUNT_RATE_PATTERN.search(instruction)
    if dr_match is not None:
        discount_rate = float(dr_match.group(1))
    else:
        discount_rate = 1.0  # default rate
    return RecencyFilteringSpec(
        mode=RecencyFilteringMode.EXPONENTIAL_DISCOUNT,
        time_feature_index=time_feature_index,
        discount_rate=discount_rate,
    )


def _detect_time_varying_spec(
    instruction: str,
    input_dim: int,
    input_feature_names: list[str] | None,
) -> tuple[TimeVaryingTarget, int] | None:
    """Detect a Tier 2 time-varying hyperparameter request from an instruction.

    Returns a (target, time_feature_index) pair if a time-varying request is found,
    otherwise None.  Outputscale requests take priority over lengthscale requests
    when both patterns are present.

    Args:
        instruction: Free-text instruction string.
        input_dim: Total number of input features.
        input_feature_names: Optional list of continuous feature column names.

    Returns:
        A (TimeVaryingTarget, time_feature_index) tuple or None.
    """
    has_outputscale = _TIME_VARYING_OUTPUTSCALE_PATTERN.search(instruction) is not None
    has_lengthscale = _TIME_VARYING_LENGTHSCALE_PATTERN.search(instruction) is not None

    if not has_outputscale and not has_lengthscale:
        return None

    # Determine the time feature index using the same heuristic as recency filtering.
    time_feature_index = 0
    tf_match = _TIME_FEATURE_PATTERN.search(instruction)
    if tf_match is not None:
        time_feature_index = int(tf_match.group(1))
    elif input_feature_names:
        for idx, name in enumerate(input_feature_names):
            if re.search(r"\btime\b|\btimestamp\b|\bt\b|\bdate\b|\bday\b|\bweek\b|\bmonth\b|\byear\b",
                         name, re.IGNORECASE):
                time_feature_index = idx
                break

    time_feature_index = max(0, min(time_feature_index, input_dim - 1))

    target = TimeVaryingTarget.OUTPUTSCALE if has_outputscale else TimeVaryingTarget.LENGTHSCALE
    return target, time_feature_index


def _detect_input_warping(
    instruction: str,
    input_dim: int,
    input_feature_names: list[str] | None,
) -> InputWarpingSpec | None:
    """Detect a Tier 2 input warping request from an instruction.

    Args:
        instruction: Free-text instruction string.
        input_dim: Total number of input features.
        input_feature_names: Optional list of continuous feature column names.

    Returns:
        An InputWarpingSpec if warping was requested, otherwise None.
    """
    if _INPUT_WARP_PATTERN.search(instruction) is None:
        return None

    # Determine the time feature index.
    time_feature_index = 0
    tf_match = _TIME_FEATURE_PATTERN.search(instruction)
    if tf_match is not None:
        time_feature_index = int(tf_match.group(1))
    elif input_feature_names:
        for idx, name in enumerate(input_feature_names):
            if re.search(r"\btime\b|\btimestamp\b|\bt\b|\bdate\b|\bday\b|\bweek\b|\bmonth\b|\byear\b",
                         name, re.IGNORECASE):
                time_feature_index = idx
                break

    time_feature_index = max(0, min(time_feature_index, input_dim - 1))
    return InputWarpingSpec(warp_type=WarpType.KUMARASWAMY, time_feature_index=time_feature_index)


def _build_changepoint_kernel_spec(
    instruction: str,
    *,
    ard_instruction: str | None = None,
) -> KernelSpec:
    """Build a Changepoint KernelSpec with two default sub-kernels.

    The translator generates a Matern52 before-kernel and an RBF after-kernel
    as a neutral default.  Users can customise by specifying 'before' and 'after'
    kernels explicitly (not yet parsed in v1 — fixed defaults are used instead).

    Args:
        instruction: The instruction fragment surrounding the changepoint mention.
        ard_instruction: Source text used to determine ARD for child kernels.

    Returns:
        A KernelSpec with kernel_type=CHANGEPOINT and 2 children.
    """
    location = _extract_changepoint_location(instruction)
    steepness = _extract_changepoint_steepness(instruction)

    child_ard_source = ard_instruction if ard_instruction is not None else instruction
    before_kernel = KernelSpec(
        kernel_type=KernelType.MATERN_52,
        ard=_resolve_ard_setting(child_ard_source, KernelType.MATERN_52),
    )
    after_kernel = KernelSpec(
        kernel_type=KernelType.RBF,
        ard=_resolve_ard_setting(child_ard_source, KernelType.RBF),
    )
    return KernelSpec(
        kernel_type=KernelType.CHANGEPOINT,
        changepoint_location=location,
        changepoint_steepness=steepness,
        children=[before_kernel, after_kernel],
    )


def _build_kernel_spec(
    instruction: str,
    kernel_type: KernelType,
    *,
    ard_instruction: str | None = None,
) -> KernelSpec:
    """Build a kernel spec with any kernel-specific parameters parsed from text."""
    if kernel_type == KernelType.CHANGEPOINT:
        return _build_changepoint_kernel_spec(instruction, ard_instruction=ard_instruction)

    ard_source = instruction if ard_instruction is None else ard_instruction
    ard = True if kernel_type == KernelType.SPECTRAL_MIXTURE else _resolve_ard_setting(ard_source, kernel_type)
    kernel_spec = KernelSpec(kernel_type=kernel_type, ard=ard)

    if kernel_type == KernelType.RQ:
        kernel_spec.rq_alpha = _extract_rq_alpha(instruction)
    elif kernel_type == KernelType.PERIODIC:
        kernel_spec.period_length = _extract_period_length(instruction)
    elif kernel_type == KernelType.POLYNOMIAL:
        kernel_spec.polynomial_power = _extract_polynomial_power(instruction)
        kernel_spec.polynomial_offset = _extract_offset(instruction)
    elif kernel_type == KernelType.SPECTRAL_MIXTURE:
        kernel_spec.num_mixtures = _extract_num_mixtures(instruction)
        kernel_spec.spectral_init = _detect_spectral_init(instruction)
    elif kernel_type == KernelType.INFINITE_WIDTH_BNN:
        kernel_spec.bnn_depth = _extract_depth(instruction)
    elif kernel_type == KernelType.EXPONENTIAL_DECAY:
        kernel_spec.exponential_decay_power = _extract_power(instruction)
        kernel_spec.exponential_decay_offset = _extract_offset(instruction)

    return kernel_spec


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


def _parse_prior_params(distribution: PriorDistribution, params_text: str) -> dict[str, float]:
    """Extract prior parameters from the trailing text of a prior phrase."""
    extracted: dict[str, float] = {}
    for param_name, pattern in _PRIOR_PARAM_PATTERNS[distribution].items():
        match = pattern.search(params_text)
        if match is not None:
            extracted[param_name] = float(match.group(1))

    if distribution == PriorDistribution.UNIFORM and ("a" not in extracted or "b" not in extracted):
        range_match = _UNIFORM_RANGE_PATTERN.search(params_text)
        if range_match is not None:
            lower = range_match.group(1) or range_match.group(3)
            upper = range_match.group(2) or range_match.group(4)
            if lower is not None and upper is not None:
                extracted.setdefault("a", float(lower))
                extracted.setdefault("b", float(upper))

    return extracted


def _normalize_prior_distribution(raw_distribution: str) -> PriorDistribution:
    """Normalize a parsed natural-language prior distribution name."""
    return _PRIOR_DISTRIBUTION_ALIASES[raw_distribution.lower()]


def _normalize_prior_target(raw_target: str) -> str:
    """Normalize a parsed natural-language prior target name."""
    return _PRIOR_TARGET_ALIASES[raw_target.lower()]


def _apply_detected_priors(instruction: str, kernel_spec: KernelSpec, noise: NoiseSpec) -> None:
    """Apply parsed prior phrases to a kernel or noise spec in place."""
    for pattern in _PRIOR_PATTERNS:
        for match in pattern.finditer(instruction):
            distribution = _normalize_prior_distribution(match.group("distribution"))
            prior = PriorSpec(
                distribution=distribution,
                params=_parse_prior_params(distribution, match.group("params")),
            )
            target = _normalize_prior_target(match.group("target"))
            if target == "lengthscale":
                kernel_spec.lengthscale_prior = prior
            elif target == "outputscale":
                kernel_spec.outputscale_prior = prior
            elif target == "period":
                kernel_spec.period_prior = prior
            elif target == "noise":
                noise.prior = prior


def _default_execution_spec(model_class: ModelClass) -> ExecutionSpec:
    """Return the default execution semantics for a translated spec."""
    return ExecutionSpec(
        input_scaling=True,
        outcome_standardization=model_class != ModelClass.MULTI_TASK_GP,
    )


def _parse_mean_type(mean_name: str) -> MeanFunctionType:
    """Map a parsed natural-language mean name to a DSL mean enum."""
    normalized_name = mean_name.lower()
    if normalized_name == "constant":
        return MeanFunctionType.CONSTANT
    if normalized_name == "zero":
        return MeanFunctionType.ZERO
    return MeanFunctionType.LINEAR


def _normalize_mean_target(
    target_kind: str,
    target_index: int,
    model_class: ModelClass,
    output_dim: int,
) -> int | None:
    """Normalize a parsed output/task target into a spec output_means key."""
    if target_kind == "task":
        if model_class != ModelClass.MULTI_TASK_GP or target_index < 0:
            return None
        return target_index

    if target_index < 1 or target_index > output_dim:
        return None

    if model_class == ModelClass.MULTI_TASK_GP:
        return target_index - 1
    if model_class == ModelClass.MODEL_LIST_GP:
        return target_index - 1
    return None


def _detect_means(
    instruction: str,
    model_class: ModelClass,
    output_dim: int,
) -> tuple[MeanSpec | None, dict[int, MeanSpec]]:
    """Detect shared and targeted mean-function requests from an instruction."""
    mean: MeanSpec | None = None
    output_means: dict[int, MeanSpec] = {}

    for match in _MEAN_PATTERN.finditer(instruction):
        prefix_kind = match.group(1)
        prefix_index = match.group(2)
        mean_type = MeanSpec(mean_type=_parse_mean_type(match.group(3)))
        suffix_kind = match.group(4)
        suffix_index = match.group(5)

        target_kind = prefix_kind or suffix_kind
        target_index_text = prefix_index or suffix_index
        if target_kind is None or target_index_text is None:
            if mean is None:
                mean = mean_type
            continue

        target_key = _normalize_mean_target(
            target_kind=target_kind.lower(),
            target_index=int(target_index_text),
            model_class=model_class,
            output_dim=output_dim,
        )
        if target_key is not None:
            output_means[target_key] = mean_type

    return mean, output_means


def _resolve_ard_setting(instruction: str, kernel_type: KernelType) -> bool:
    """Return whether ARD should be enabled for a kernel type."""
    if kernel_type == KernelType.SPECTRAL_MIXTURE:
        return True
    if kernel_type not in _ARD_SUPPORTED_KERNELS:
        return False
    if _DISABLE_ARD_PATTERN.search(instruction):
        return False
    return True


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
) -> list[FeatureGroupSpec]:
    """Extract feature-specific kernel groups from natural language when column names are available."""
    if not input_feature_names:
        return []

    feature_groups: list[FeatureGroupSpec] = []
    kernel_mentions = _find_kernel_mentions(instruction)
    for index, (kernel_start, kernel_end, kernel_type) in enumerate(kernel_mentions):
        clause_start = 0
        if index > 0:
            clause_start = kernel_mentions[index - 1][1]
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

        clause_text = instruction[clause_start:clause_end]
        feature_groups.append(
            FeatureGroupSpec(
                name="_".join(input_feature_names[index] for index in feature_indices),
                feature_indices=feature_indices,
                kernel=_build_kernel_spec(clause_text, kernel_type, ard_instruction=instruction),
            )
        )

    return feature_groups


def translate_to_dsl(
    instruction: str,
    input_dim: int,
    output_dim: int = 1,
    task_feature_index: int | None = None,
    task_values: list[int] | None = None,
    input_feature_names: list[str] | None = None,
) -> GPSpec:
    """Translate a natural-language GP instruction into a GPSpec DSL object.

    All unspecified aspects are filled with safe BoTorch defaults.

    Args:
        instruction: Free-text description of the desired GP architecture.
        input_dim: Total number of continuous input features.
        output_dim: Number of model outputs (default 1).
        task_feature_index: Column index of the task indicator for MultiTaskGP.
        task_values: Optional explicit task domain for MultiTaskGP.
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
    use_ard = _resolve_ard_setting(instruction, kernel_type)
    default_kernel_spec = _build_kernel_spec(instruction, kernel_type)
    _apply_detected_priors(instruction, default_kernel_spec, noise)
    feature_groups = _extract_feature_groups(instruction, input_feature_names)
    mean, output_means = _detect_means(instruction, model_class, output_dim)
    recency_filtering = _detect_recency_filtering(instruction, input_dim, input_feature_names)
    time_varying_result = _detect_time_varying_spec(instruction, input_dim, input_feature_names)
    input_warping = _detect_input_warping(instruction, input_dim, input_feature_names)

    # Apply time-varying modulation to the default kernel spec if detected.
    if time_varying_result is not None:
        tv_target, tv_time_index = time_varying_result
        default_kernel_spec.time_varying = TimeVaryingSpec(
            target=tv_target,
            time_feature_index=tv_time_index,
        )

    resolved_task_values = task_values

    if feature_groups:
        composition = explicit_composition
        if composition is None:
            composition = CompositionType.HIERARCHICAL if len(feature_groups) > 1 else CompositionType.ADDITIVE
        for group in feature_groups:
            _apply_detected_priors(instruction, group.kernel, noise)
            # Propagate time-varying to per-group kernels as well, if detected.
            if time_varying_result is not None and group.kernel.time_varying is None:
                tv_target, tv_time_index = time_varying_result
                group.kernel.time_varying = TimeVaryingSpec(
                    target=tv_target,
                    time_feature_index=tv_time_index,
                )

    elif model_class == ModelClass.MODEL_LIST_GP:
        composition = explicit_composition or CompositionType.ADDITIVE
        continuous_indices = [idx for idx in range(input_dim) if idx != task_feature_index]
        feature_groups = [
            FeatureGroupSpec(
                name="all_features",
                feature_indices=continuous_indices,
                kernel=default_kernel_spec.model_copy(deep=True),
            )
        ]
    else:
        composition = explicit_composition or CompositionType.ADDITIVE
        continuous_indices = [idx for idx in range(input_dim) if idx != task_feature_index]
        feature_groups = [
            FeatureGroupSpec(
                name="all_features",
                feature_indices=continuous_indices,
                kernel=default_kernel_spec.model_copy(deep=True),
            )
        ]

    multitask_rank: int | None = None
    if model_class == ModelClass.MULTI_TASK_GP:
        multitask_rank = min(output_dim, 2)
        if resolved_task_values is None and output_means:
            resolved_task_values = sorted(output_means)

    description = (
        f"Translated from: '{instruction[:80]}{'...' if len(instruction) > 80 else ''}' | "
        f"model={model_class.value}, kernel={kernel_type.value}, ard={use_ard}, groups={len(feature_groups)}"
    )

    base_execution = _default_execution_spec(model_class)
    execution = ExecutionSpec(
        input_scaling=base_execution.input_scaling,
        outcome_standardization=base_execution.outcome_standardization,
        recency_filtering=recency_filtering,
        input_warping=input_warping,
    )

    spec = GPSpec(
        model_class=model_class,
        feature_groups=feature_groups,
        mean=mean,
        output_means=output_means,
        noise=noise,
        execution=execution,
        input_dim=input_dim,
        output_dim=output_dim,
        task_feature_index=task_feature_index,
        task_values=resolved_task_values,
        multitask_rank=multitask_rank,
        group_composition=composition,
        description=description,
    )

    logger.info("Translated instruction to DSL: model=%s, kernel=%s", model_class.value, kernel_type.value)
    return spec
