# Changepoint kernel

## What it does

`changepoint kernel` lets GPArchitect transition between two kernel regimes around a learned time location.

Conceptually, covariance blends two kernels with a sigmoid gate centered at the changepoint.

## Worked example

```python
import math
import pandas as pd
from gparchitect import run_gparchitect

x_values = [index / 19 for index in range(20)]
df = pd.DataFrame(
    {
        "time": x_values,
        "signal": [
            (0.7 * math.sin(2.0 * math.pi * x * 1.2)) if x < 0.45 else (0.2 + 1.5 * (x - 0.45))
            for x in x_values
        ],
    }
)

# Baseline stationary GP
baseline_model, baseline_log = run_gparchitect(
    dataframe=df,
    instruction="Use a Matern 5/2 kernel on time.",
    input_columns=["time"],
    output_columns=["signal"],
    max_retries=0,
)

# Changepoint kernel GP
changepoint_model, changepoint_log = run_gparchitect(
    dataframe=df,
    instruction="Use a changepoint kernel at 0.45 with steepness 8.0 on time.",
    input_columns=["time"],
    output_columns=["signal"],
    max_retries=0,
)

print("Baseline success:", baseline_log.final_success)
print("Changepoint success:", changepoint_log.final_success)
print(
    "Kernel kind:",
    changepoint_log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]["kind"],
)
```

## Basic visualization vs generic GP

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(df["time"], df["signal"], "o-", label="Observed signal")
plt.axvline(0.45, color="crimson", linestyle="--", label="Regime-change location")
plt.title("Piecewise behavior where changepoint kernels help")
plt.xlabel("time")
plt.ylabel("signal")
plt.legend()
plt.tight_layout()
plt.show()
```

When behavior changes regime around a specific time location, a changepoint kernel is often a better fit than one stationary kernel over the full domain.
