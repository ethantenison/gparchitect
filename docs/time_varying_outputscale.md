# Time-varying outputscale

## What it does

`time-varying outputscale` lets GPArchitect use a kernel amplitude that changes over time-like input values.

Conceptually, it scales covariance as:

`k_tv(x_i, x_j) = s(t_i) * k_base(x_i, x_j) * s(t_j)`

where `s(t) = softplus(bias + slope * t)` is learned during fitting.

## Worked example

```python
import math
import pandas as pd
from gparchitect import run_gparchitect

x_values = [index / 17 for index in range(18)]
df = pd.DataFrame(
    {
        "time": x_values,
        "signal": [
            (0.2 + 1.4 * x) * math.sin(2.0 * math.pi * x * 2.0)
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

# Time-varying outputscale GP
adaptive_model, adaptive_log = run_gparchitect(
    dataframe=df,
    instruction="Use a Matern 5/2 kernel with time-varying outputscale on time.",
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

plt.figure(figsize=(8, 4))
plt.plot(df["time"], df["signal"], "o", label="Observed signal")
plt.plot(df["time"], df["signal"].rolling(3, min_periods=1).std(), label="Local variability (proxy)")
plt.hlines(df["signal"].std(), xmin=df["time"].min(), xmax=df["time"].max(), label="Global variability (stationary proxy)")
plt.title("Why time-varying outputscale helps")
plt.xlabel("time")
plt.ylabel("signal / variability")
plt.legend()
plt.tight_layout()
plt.show()
```

The rolling variability curve highlights changing amplitude over time, which a stationary outputscale cannot represent directly.
