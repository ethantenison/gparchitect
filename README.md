# GPArchitect

GPArchitect builds Gaussian Process models from natural-language instructions
and tabular data using [BoTorch](https://botorch.org/) and [GPyTorch](https://gpytorch.ai/).

Provide a pandas DataFrame and a text specification, and it constructs, fits,
and validates `SingleTaskGP`, `MultiTaskGP`, or `ModelListGP` models. If fitting
fails, it revises the DSL specification, retries, and logs all changes.

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
    "x1": [0.1, 0.5, 0.9, 0.2, 0.8],
    "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y":  [0.3, 0.7, 0.4, 0.6, 0.5],
})

model, log = run_gparchitect(
    dataframe=df,
    instruction="Use a Matern 5/2 kernel with ARD on all inputs.",
    input_columns=["x1", "x2"],
    output_columns=["y"],
)

print("Success:", log.final_success)
```

## CLI

```bash
gparchitect --csv data.csv \
    --instruction "Use Matern52 with ARD" \
    --inputs x1,x2,x3 \
    --outputs y
```

## Installation

```bash
pip install gparchitect
```

Requires Python 3.13+, BoTorch ≥ 0.11, GPyTorch ≥ 1.11, and pandas ≥ 2.0.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture Module Map](docs/architecture_module_map.md)
- [Copilot Instructions](.github/copilot-instructions.md)

## License

MIT
