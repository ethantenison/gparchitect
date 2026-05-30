# Time-varying lengthscale

## What it does

`time-varying lengthscale` lets GPArchitect adjust local smoothness over a time-like axis.

This allows the model to stay smooth in stable regions and become more flexible where behavior changes quickly.

## Worked example

```python
import math
import pandas as pd
from gparchitect import run_gparchitect

x_values = [index / 23 for index in range(24)]
df = pd.DataFrame(
    {
        "time": x_values,
        "signal": [
            math.sin(2.0 * math.pi * (1.0 + (2.0 * x)) * x)
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

# Time-varying lengthscale GP
adaptive_model, adaptive_log = run_gparchitect(
    dataframe=df,
    instruction="Use a Matern 5/2 kernel with time-varying lengthscale on time.",
    input_columns=["time"],
    output_columns=["signal"],
    max_retries=0,
)

print("Baseline success:", baseline_log.final_success)
print("Adaptive success:", adaptive_log.final_success)
print(
    "Adaptive target:",
    adaptive_log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]["time_varying"]["target"],
)
```

## Basic visualization vs generic GP

```python
import matplotlib.pyplot as plt

left_half = df[df["time"] <= 0.5]
right_half = df[df["time"] > 0.5]

left_slope_proxy = left_half["signal"].diff().abs().mean()
right_slope_proxy = right_half["signal"].diff().abs().mean()

plt.figure(figsize=(7, 4))
plt.bar(["early time", "late time"], [left_slope_proxy, right_slope_proxy])
plt.title("Local change-rate comparison")
plt.ylabel("mean |delta signal|")
plt.tight_layout()
plt.show()
```

A clear change in local slope proxy suggests different local smoothness, which is the scenario where a time-varying lengthscale is useful.
