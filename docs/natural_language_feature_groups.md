# Natural-language feature groups

## What it does

GPArchitect can map different kernels to different input columns from a single natural-language instruction.

## Worked example

```python
import math
import pandas as pd
from gparchitect import run_gparchitect

x1_values = [index / 11 for index in range(12)]
x2_values = [1.0 - (0.9 * index / 11) for index in range(12)]
df = pd.DataFrame(
    {
        "seasonality": x1_values,
        "trend": x2_values,
        "target": [
            0.4 * math.sin(2.0 * math.pi * seasonal) + 0.3 * trend + 0.05
            for seasonal, trend in zip(x1_values, x2_values)
        ],
    }
)

model, log = run_gparchitect(
    dataframe=df,
    instruction="Use an rq kernel with alpha 0.75 on seasonality, and an rbf kernel on trend.",
    input_columns=["seasonality", "trend"],
    output_columns=["target"],
    max_retries=0,
)

spec = log.attempts[0].spec_snapshot
print("Success:", log.final_success)
print("Group composition:", spec["group_composition"])
print("Feature groups:", spec["feature_groups"])
```
