# Signed LKJ Task-Kernel Repeat

Purpose: repeat the true-task rank-5 18-ETF GP OOS run while changing only the task covariance kernel from BoTorch `PositiveIndexKernel` to the BayesFolio-style signed GPyTorch `IndexKernel` with `LKJCovariancePrior(eta=2.0)`.

Baseline run:

- `outputs/20260628_gp_18etf_plain_roll60m_rank5_fixedtask_24w_botorchdefault/`
- Task kernel: `PositiveIndexKernel`
- Rank: `5`
- Universe: 18 ETFs
- Windows: 24 sealed OOS windows, `2024-04-30` through `2026-03-31`
- Train window: rolling 60 months
- Optimizer: BoTorch default
- Variant: `plain`

Full repeat command:

```bash
uv run python experiments/2026-06-time-varying-parameterizations/run_experiment.py \
  --run-id 20260629_gp_18etf_plain_roll60m_rank5_signedlkj_24w_botorchdefault \
  --last-n-windows 24 \
  --botorch-default-optimizer \
  --rank 5 \
  --task-kernel signed_lkj_eta_2 \
  --etf-universe BND BNDX EWX HYEM HYG IEF IJR IWM LQD MGK SPY VEA VNQ VNQI VSS VTV VWO VWOB \
  --train-window-months 60 \
  --variants plain \
  --skip-fake \
  --skip-spy
```

Bounded smoke already run:

```bash
uv run python experiments/2026-06-time-varying-parameterizations/run_experiment.py \
  --run-id 20260629_gp_18etf_plain_roll60m_rank5_signedlkj_smoke_1w_iter10 \
  --last-n-windows 1 \
  --max-iter 10 \
  --rank 5 \
  --task-kernel signed_lkj_eta_2 \
  --etf-universe BND BNDX EWX HYEM HYG IEF IJR IWM LQD MGK SPY VEA VNQ VNQI VSS VTV VWO VWOB \
  --train-window-months 60 \
  --variants plain \
  --skip-fake \
  --skip-spy
```

Smoke result:

- Output: `outputs/20260629_gp_18etf_plain_roll60m_rank5_signedlkj_smoke_1w_iter10/`
- Windows: 1
- Failures: 0
- Mean window Spearman: `0.5955`
- Top1-bottom1 spread: `-0.0123`
- RMSE: `0.0702`

Interpret the smoke as a construction and fit-path check only. It is not evidence that signed LKJ is better or worse than the positive task kernel.

## Signed No-Prior Follow-Up

Purpose: repeat the same sealed run with the BayesFolio `signed_no_prior` task-kernel variant. This uses the same signed GPyTorch `IndexKernel` as the LKJ variant, but with `prior=None`.

Full repeat command:

```bash
uv run python experiments/2026-06-time-varying-parameterizations/run_experiment.py \
  --run-id 20260629_gp_18etf_plain_roll60m_rank5_signednoprior_24w_botorchdefault \
  --last-n-windows 24 \
  --botorch-default-optimizer \
  --rank 5 \
  --task-kernel signed_no_prior \
  --etf-universe BND BNDX EWX HYEM HYG IEF IJR IWM LQD MGK SPY VEA VNQ VNQI VSS VTV VWO VWOB \
  --train-window-months 60 \
  --variants plain \
  --skip-fake \
  --skip-spy
```
