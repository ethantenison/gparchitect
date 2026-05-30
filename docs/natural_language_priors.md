# Natural-language priors

## What it does

GPArchitect can parse prior phrases directly from instructions and place them onto supported kernel/noise parameters in the DSL.

## Worked example

```python
import math
import pandas as pd
from gparchitect import run_gparchitect

x_values = [index / 15 for index in range(16)]
df = pd.DataFrame(
    {
        "time": x_values,
        "signal": [math.sin(2.0 * math.pi * value) + (0.2 * value) for value in x_values],
    }
)

model, log = run_gparchitect(
    dataframe=df,
    instruction=(
        "Use an rbf kernel with normal prior on lengthscale loc 0.0 scale 1.0 "
        "and halfcauchy prior on outputscale scale 0.75."
    ),
    input_columns=["time"],
    output_columns=["signal"],
    max_retries=0,
)

kernel = log.attempts[0].spec_snapshot["feature_groups"][0]["kernel"]
print("Success:", log.final_success)
print("Lengthscale prior:", kernel["lengthscale_prior"])
print("Outputscale prior:", kernel["outputscale_prior"])
```
