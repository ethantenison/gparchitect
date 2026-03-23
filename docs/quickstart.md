# GPArchitect Quickstart

This guide shows how to get started with GPArchitect in under 5 minutes.

---

## Installation

```bash
pip install gparchitect
```

Or, if using Poetry:

```bash
poetry add gparchitect
```

---

## Requirements

- Python 3.13.12
- BoTorch ≥ 0.11
- GPyTorch ≥ 1.11
- pandas ≥ 2.0

---

## Basic Usage

### 1. Prepare your data

GPArchitect accepts a **pandas DataFrame** with labelled input and output columns.
Continuous input columns are min-max scaled into the unit cube before model building.

```python
import pandas as pd

df = pd.DataFrame({
    "x1": [0.1, 0.5, 0.9, 0.2, 0.8],
    "x2": [1.0, 2.0, 3.0, 4.0, 5.0],
    "y":  [0.3, 0.7, 0.4, 0.6, 0.5],
})
```

### 2. Run the pipeline

```python
from gparchitect import run_gparchitect

model, experiment_log = run_gparchitect(
    dataframe=df,
    instruction="Use a rbf kernel on x1 and a matern1/2 kernel on x2.",
    input_columns=["x1", "x2"],
    output_columns=["y"],
)

print("Success:", experiment_log.final_success)
```

If you name different kernels for different input columns, GPArchitect maps those
names onto the provided `input_columns`. Without an explicit additive or multiplicative
directive, the default composition is hierarchical: main effects plus interactions.
If you do not specify feature groups, GPArchitect uses a single kernel across all
continuous inputs and enables ARD by default for kernels that support it.

### 3. Inspect the generated DSL

```python
from gparchitect import translate_to_dsl, validate_dsl

spec = translate_to_dsl(
    instruction="Use a Matern 5/2 kernel",
    input_dim=2,
)
result = validate_dsl(spec)

print(spec.model_dump_json(indent=2))
print("Valid:", result.is_valid)
```

### 4. View the experiment log

```python
from gparchitect import summarize_attempts

print(summarize_attempts(experiment_log))

# Export to JSON
with open("experiment.json", "w") as f:
    f.write(experiment_log.to_json())
```

---

## Multi-Task GP

To build a MultiTaskGP, include a task indicator column:

```python
import pandas as pd
from gparchitect import run_gparchitect

df = pd.DataFrame({
    "x1": [0.1, 0.5, 0.9, 0.2],
    "x2": [1.0, 2.0, 3.0, 4.0],
    "task": [0, 0, 1, 1],
    "y": [0.3, 0.7, 0.4, 0.6],
})

model, log = run_gparchitect(
    dataframe=df,
    instruction="Use a multi-task GP with Matern52 kernel",
    input_columns=["x1", "x2"],
    output_columns=["y"],
    task_column="task",
)
```

---

## CLI Usage

```bash
# Basic usage
gparchitect --csv data.csv \
    --instruction "Use Matern52" \
    --inputs x1,x2,x3 \
    --outputs y

# With task column and log export
gparchitect --csv data.csv \
    --instruction "Multi-task GP with RBF kernel" \
    --inputs x1,x2 \
    --outputs y \
    --task-column task \
    --log-json experiment.json \
    --verbose
```

---

## Failure Recovery

GPArchitect automatically retries with a simplified DSL if fitting fails:

1. Disables ARD when it is currently enabled
2. Simplifies kernels to Matern52
3. Removes priors
4. Switches to SingleTaskGP
5. Uses default noise

All revisions are recorded in the `ExperimentLog`:

```python
for attempt in experiment_log.attempts:
    print(f"Attempt {attempt.attempt_number}: success={attempt.fit_success}")
    if attempt.revision_strategy:
        print(f"  Strategy: {attempt.revision_strategy}")
        print(f"  Rationale: {attempt.revision_rationale}")
```

---

## Supported Models

| Model Class  | Use case                                  |
|:------------ |:----------------------------------------- |
| SingleTaskGP | Single output, standard regression        |
| MultiTaskGP  | Multiple correlated outputs or tasks      |
| ModelListGP  | Multiple independent output models        |

---

## Supported Kernels

| Keyword in instruction            | Kernel type   |
|:--------------------------------- |:------------- |
| `rbf`, `squared exponential`      | RBF           |
| `rq`, `rational quadratic`        | Rational Quadratic |
| `matern`, `matern 5/2`            | Matern 5/2    |
| `matern 3/2`                      | Matern 3/2    |
| `matern 1/2`                      | Matern 1/2    |
| `periodic`                        | Periodic      |
| `linear`                          | Linear        |
| `polynomial`                      | Polynomial    |
| `spectral mixture`                | Spectral Mixture |
| `infinite width bnn`, `iwbnn`     | Infinite Width BNN |
| `exponential decay`               | Exponential Decay |

ARD is enabled by default for kernels that support it. Use `without ard`, `disable ard`,
or `shared lengthscale` to disable ARD explicitly when you want one shared lengthscale
across the input dimensions in a group.
For kernel-specific parameters, phrases like `alpha 0.75` are applied to RQ kernels,
`period length 12` configures Periodic kernels, `degree 3` and `offset 1.5` configure
Polynomial kernels, `depth 5` configures Infinite Width BNN kernels, `4 component spectral
mixture kernel` or `initialized from the empirical spectrum` configure Spectral Mixture
kernels, and `power 2.5` plus `offset 0.2` configure Exponential Decay kernels.

Outputs for `SingleTaskGP` and `ModelListGP` are standardized through BoTorch
outcome transforms during model construction.

---

## Next Steps

- See `docs/architecture_module_map.md` for architecture boundaries.
- See `.github/copilot-instructions.md` for project-wide design rules.
- See the `src/gparchitect/examples/` directory for example scripts.
