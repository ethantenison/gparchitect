# GPArchitect

## 1. Project Overview

GPArchitect builds Gaussian Process (GP) models from natural-language instructions and tabular data using BoTorch and GPyTorch.

It compiles text into a typed GP DSL, validates the specification, builds/fits a model, and records all attempts in an experiment log.

## 2. Key Features

- **Natural-language to GP DSL compiler** for `SingleTaskGP`, `MultiTaskGP`, and `ModelListGP`
- **Feature-level kernel and mean control** (including kernel-specific parameters)
- **Non-stationary modeling options**, including:
  - [Time-varying outputscale](docs/time_varying_outputscale.md)
  - [Time-varying lengthscale](docs/time_varying_lengthscale.md)
- **Automatic recovery pipeline** when fitting fails (revisions + retry)
- **CI-checked worked examples** in `tests/e2e/` that run on pull requests

## 3. Installation

```bash
pip install gparchitect
```

For local development:

```bash
poetry install
```

## 4. Quick Start

```python
import pandas as pd
from gparchitect import run_gparchitect

df = pd.DataFrame(
    {
        "x1": [0.1, 0.5, 0.9, 0.2, 0.8],
        "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.3, 0.7, 0.4, 0.6, 0.5],
    }
)

model, log = run_gparchitect(
    dataframe=df,
    instruction="Use an RBF kernel on x1 and a Matern 3/2 kernel on x2.",
    input_columns=["x1", "x2"],
    output_columns=["y"],
)

print("Success:", log.final_success)
```

## 5. Architecture Snapshot

Compiler-style pipeline:

```text
Natural language -> GP DSL -> Validation -> Model Builder -> Fit -> Validation -> Recovery
```

The DSL (`GPSpec`) is the single source of truth for model intent.

## 6. Links to Documentation

- [Quickstart](docs/quickstart.md)
- [Time-varying outputscale feature guide](docs/time_varying_outputscale.md)
- [Time-varying lengthscale feature guide](docs/time_varying_lengthscale.md)
- [Architecture module map](docs/architecture_module_map.md)
- [Testing strategy](docs/testing_strategy.md)
- [Kernel worked examples in CI](tests/e2e/test_kernel_examples_e2e.py)
- [Feature-doc worked examples in CI](tests/e2e/test_feature_docs_examples_e2e.py)

## License

Apache-2.0
