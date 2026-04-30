# GPArchitect

GPArchitect builds Gaussian Process models from natural-language instructions
and tabular data using [BoTorch](https://botorch.org/) and [GPyTorch](https://gpytorch.ai/).

Provide a pandas DataFrame and a text specification, and it constructs, fits,
and validates `SingleTaskGP`, `MultiTaskGP`, or `ModelListGP` models. If fitting
fails, it revises the DSL specification, retries, and logs all changes.

Continuous input columns are min-max scaled before model construction, and single-task
outputs are standardized through BoTorch outcome transforms during fitting.
Kernels that support ARD use it by default unless the instruction explicitly disables it.
Spectral mixture kernels are initialized from training data, with optional empirical-spectrum
initialization when requested in the instruction.
Polynomial kernels can also take a degree and offset, periodic kernels can take a period
length, infinite-width BNN kernels can take a depth, and exponential-decay kernels can take
power and offset values.
Mean functions can be selected with natural-language phrases such as `constant mean`,
`zero mean`, or `linear mean`. If no mean function is specified, GPArchitect defers to
the default mean used by the underlying BoTorch model.
For `MultiTaskGP`, GPArchitect currently supports long-format data with one observed
output column plus a task indicator column. When targeted multitask mean overrides are
parsed from natural language, the translated DSL records the explicit task values that
those overrides apply to.
These execution semantics are now represented explicitly in the DSL through an
execution specification rather than living only as implicit pipeline behavior.

## Architecture

GPArchitect follows a **compiler-style pipeline**:

```
Natural language → GP DSL → Validation → Model Builder → Fit → Validation → Recovery
```

The DSL (`GPSpec`) is the single source of truth. Natural language is an interface only.

## Quick Start

```python
import pandas as pd
from gparchitect import run_gparchitect

df = pd.DataFrame({
    "month_index": [1, 2, 3, 4, 5, 6, 7, 8],
    "credit_spread_bps": [128, 122, 135, 148, 141, 130, 118, 112],
    "vix_level": [14.8, 13.9, 16.4, 18.7, 17.1, 15.2, 13.6, 12.9],
    "net_flow_pct": [0.6, 0.9, 0.2, -0.4, 0.1, 0.8, 1.1, 1.4],
    "momentum_3m_pct": [1.8, 2.4, 1.1, -0.6, 0.4, 1.7, 2.9, 3.6],
    "etf_return_pct": [0.012, 0.018, -0.007, -0.024, -0.003, 0.016, 0.028, 0.035],
})

model, log = run_gparchitect(
    dataframe=df,
    instruction=(
        "Use an rbf kernel on month_index, a matern3/2 kernel on credit_spread_bps "
        "and vix_level, and an rbf kernel on net_flow_pct and momentum_3m_pct."
    ),
    input_columns=[
        "month_index",
        "credit_spread_bps",
        "vix_level",
        "net_flow_pct",
        "momentum_3m_pct",
    ],
    output_columns=["etf_return_pct"],
)

print("Success:", log.final_success)
print("Model:", model)
```

Feature-specific kernel instructions are resolved against the provided `input_columns`.
When multiple per-feature kernels are specified without an explicit additive or
multiplicative directive, GPArchitect uses a hierarchical default that includes the
main effects and their interaction.
If no feature groups are specified, GPArchitect falls back to a single kernel across all
continuous inputs with ARD enabled when the kernel supports it. Use phrases such as
`without ARD` or `shared lengthscale` to disable ARD explicitly.
RQ kernels accept an optional `alpha` value, periodic kernels accept `period length 12`,
polynomial kernels accept `degree 3` and `offset 1.5`, infinite-width BNN kernels accept
`depth 5`, spectral mixture kernels accept an optional number of mixtures plus initialization
hints such as `initialized from data` or `initialized from the empirical spectrum`, and
exponential-decay kernels accept `power 2.5` and `offset 0.2`.
Supported prior phrases can also be translated into the DSL for supported targets, for
example `normal prior on lengthscale loc 0 scale 1`, `gamma prior on noise concentration 2 rate 0.5`,
or target-first variants such as `length scale has a normal prior with mean 0 std 1`.
Mean functions accept `constant mean`, `zero mean`, or `linear mean`. For independent-output
ModelListGP specifications you can target individual outputs with phrases such as `output 1 uses
zero mean` and `output 2 uses linear mean`. For MultiTaskGP you can target individual task values
with phrases such as `zero mean for task 0` and `constant mean for task 1`.
The current validated prior subset in the DSL is `Normal`, `LogNormal`, `Gamma`,
`HalfCauchy`, and `Uniform` for supported parameter placements.
When a `ModelListGP` instruction does not specify output-specific feature ownership,
GPArchitect falls back to one shared feature-group specification that is reused across
the independent output models rather than fabricating separate per-output covariance specs.
For `MultiTaskGP`, the DSL now requires an explicit `task_values` domain before validation
and model building proceed.

## Time-driven non-stationarity

GPArchitect supports several mechanisms for handling data where the GP behavior changes
over time.  Each has a distinct scope and limitation.

### Changepoint kernel (Tier 1)

Use a `changepoint kernel` or `regime shift` phrase to request a kernel that transitions
between two sub-kernels at a time location.  The transition is parameterized by a sigmoid
with learnable location and steepness.  This is a full GP kernel — it participates in
the likelihood.

### Recency filtering (Tier 1)

Use `sliding window`, `exponential forgetting`, or `exponential discount` phrases to
discard stale observations before fitting.

**Important**: this is dataset truncation, not true observation-weighted GP inference.
Old observations are removed; the remaining observations are treated with equal weight.

- `sliding window of 0.4`: removes observations older than `max_time − 0.4`.
- `exponential forgetting with rate 2.0`: removes observations whose
  `exp(−rate · Δt) < min_weight` (default `min_weight=0.01`).

Both modes operate in the (possibly scaled) feature space.  Enable input scaling
(the default) for the window sizes and rates to be interpretable in a `[0, 1]` range.

### Time-varying hyperparameters (Tier 2)

Use `time-varying outputscale`, `amplitude changes over time`, `time-varying lengthscale`,
or `lengthscale changes over time` to request a kernel whose outputscale or lengthscale
varies smoothly with a time-like input.

The implementation wraps any base kernel with a learned linear modulation:
`h(t) = softplus(bias + slope · t)`.  Two additional learnable parameters per kernel
(bias, slope) are optimized jointly with the kernel hyperparameters.

- `time-varying outputscale`: amplitude varies as `s(t_i) · k(x_i, x_j) · s(t_j)`.
- `time-varying lengthscale`: the time dimension is rescaled by `1 / l(t)` before
  kernel evaluation.

### Input warping (Tier 2)

Use `warp time`, `warped time axis`, `nonlinear time warp`, or `input warping` to
apply a Kumaraswamy CDF warp to the time-like input dimension before kernel evaluation.
The warp is monotone and maps the input through a learnable smooth nonlinearity.

The warp uses BoTorch's built-in `Warp` input transform and is optimized jointly with
the kernel hyperparameters.  Concentration parameters can be fixed at construction by
passing `concentration0` and `concentration1` to `InputWarpingSpec`.

Input scaling should be enabled (the default) for the Kumaraswamy warp to operate
correctly in the `[0, 1]` range.

### Heteroskedastic noise (planned, not yet supported)

The `heteroskedastic_noise` field exists in the DSL for forward compatibility but is
rejected by the validator.  Support for input-dependent noise is a planned future tier.

## CLI

```bash
gparchitect --csv data.csv \
    --instruction "Use Matern52" \
    --inputs x1,x2,x3 \
    --outputs y

gparchitect plan prior \
    --text "Temperature and pressure interact, and noise increases near the upper pressure limit." \
    --output-format text

gparchitect plan auto \
    --text "BEGIN GPARCHITECT PRIOR KNOWLEDGE HANDOFF\nSystem Summary:\n- Battery degradation forecasting.\nEND GPARCHITECT PRIOR KNOWLEDGE HANDOFF" \
    --output-format json

gparchitect-plan auto prompt.txt
```

## Planning API

```python
from gparchitect import run_architect, run_architecture_focus, run_prior_knowledge

prior = run_prior_knowledge(
    "Temperature and pressure interact, and noise increases near the upper pressure limit."
)
architecture = run_architecture_focus(prior)
planning_run = run_architect(
    "Weekly seasonality and delayed labels matter, and I want downstream planning.",
    mode="auto",
)

print(prior.to_handoff_text())
print(architecture.to_handoff_text())
print(planning_run.chosen_path)
```

The planning subsystem is upstream of DSL construction. It produces structured
handoffs only and does not build or fit models.

For tool or agent bridges, prefer `gparchitect-plan <mode> prompt.txt` or
`... --stdin` over long quoted inline prompts.

## Installation

```bash
pip install gparchitect
```

Requires Python 3.11–3.13, BoTorch ≥ 0.11, GPyTorch ≥ 1.11, pandas ≥ 2.2, and Click ≥ 8.1.

For local development from a checkout:

```bash
poetry install
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture Module Map](docs/architecture_module_map.md)
- [Testing Strategy](docs/testing_strategy.md)
- [Copilot Instructions](.github/copilot-instructions.md)

## License

Apache-2.0
