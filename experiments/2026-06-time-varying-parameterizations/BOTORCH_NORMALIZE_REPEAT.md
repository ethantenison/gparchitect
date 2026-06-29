# BoTorch Normalize Scaling Repeat

Purpose: repeat the sealed rank-5 18-ETF GP OOS runs using the BayesFolio-style transform policy:

- BoTorch `Normalize` on all non-task input columns.
- Categorical task feature excluded from input normalization.
- No manual target z-score before fitting.
- `StratifiedStandardize` by ETF task handles the target scale.

Positive task-kernel command:

```bash
uv run python experiments/2026-06-time-varying-parameterizations/run_experiment.py \
  --run-id 20260629_gp_18etf_plain_roll60m_rank5_positive_bfnorm_24w_botorchdefault \
  --last-n-windows 24 \
  --botorch-default-optimizer \
  --rank 5 \
  --task-kernel positive \
  --scaling botorch_normalize \
  --etf-universe BND BNDX EWX HYEM HYG IEF IJR IWM LQD MGK SPY VEA VNQ VNQI VSS VTV VWO VWOB \
  --train-window-months 60 \
  --variants plain \
  --skip-fake \
  --skip-spy
```

Signed no-prior task-kernel command:

```bash
uv run python experiments/2026-06-time-varying-parameterizations/run_experiment.py \
  --run-id 20260629_gp_18etf_plain_roll60m_rank5_signednoprior_bfnorm_24w_botorchdefault \
  --last-n-windows 24 \
  --botorch-default-optimizer \
  --rank 5 \
  --task-kernel signed_no_prior \
  --scaling botorch_normalize \
  --etf-universe BND BNDX EWX HYEM HYG IEF IJR IWM LQD MGK SPY VEA VNQ VNQI VSS VTV VWO VWOB \
  --train-window-months 60 \
  --variants plain \
  --skip-fake \
  --skip-spy
```

One-window smoke checks:

- `outputs/20260629_gp_18etf_plain_roll60m_rank5_positive_bfnorm_smoke_1w_iter10/`: 1 window, 0 failures.
- `outputs/20260629_gp_18etf_plain_roll60m_rank5_signednoprior_bfnorm_smoke_1w_iter10/`: 1 window, 0 failures.

Interpret the smokes as construction checks only. The full 24-window runs are the comparison evidence.
