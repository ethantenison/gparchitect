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
RQ kernels accept an optional `alpha` value, and spectral mixture kernels accept an optional
number of mixtures plus initialization hints such as `initialized from data` or
`initialized from the empirical spectrum`.

## CLI

```bash
gparchitect --csv data.csv \
    --instruction "Use Matern52" \
    --inputs x1,x2,x3 \
    --outputs y
```

## Installation

```bash
pip install gparchitect
```

Requires Python 3.13.12, BoTorch ≥ 0.11, GPyTorch ≥ 1.11, and pandas ≥ 2.0.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture Module Map](docs/architecture_module_map.md)
- [Copilot Instructions](.github/copilot-instructions.md)

## License

MIT
