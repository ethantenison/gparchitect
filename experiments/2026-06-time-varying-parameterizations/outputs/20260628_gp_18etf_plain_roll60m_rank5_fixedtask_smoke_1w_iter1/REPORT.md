# Time-Varying Parameterization Experiment

Run ID: `20260628_gp_18etf_plain_roll60m_rank5_fixedtask_smoke_1w_iter1`
Created: `2026-06-28T15:12:59.312088+00:00`

## Question

Can the 18-ETF sealed multitask GP run with the original BayesFolio-style latent rank of `5` fit successfully once the runner preserves the BoTorch task covariance kernel?

This smoke was triggered after discovering that prior `gparchitect` GP comparison runs built a BoTorch `ProductKernel(data, task)` and then replaced it with the data kernel alone. That made the recorded `rank` parameter ineffective in those artifacts.

## Design

- ETF universe: `BND, BNDX, EWX, HYEM, HYG, IEF, IJR, IWM, LQD, MGK, SPY, VEA, VNQ, VNQI, VSS, VTV, VWO, VWOB`
- Windows: `[{'train_end_date': '2026-02-27', 'forecast_date': '2026-03-31'}]`
- Max optimizer iterations: `1`
- Multitask rank: `5`
- Train window months: `60`
- Changepoint kernels: excluded by design.

## Summary

```text
    scope model  ok_windows  failed_windows     rmse  predictive_log_likelihood  directional_accuracy  mean_window_spearman  long_short_top1_bottom1_mean
multitask plain           1               0 0.070373                   1.340815              0.944444              0.783282                      0.140266
```

## Visuals

- `plots/summary_rmse.png`
- `plots/summary_predictive_log_likelihood.png`
- `plots/spy_predicted_vs_actual.png`
- `plots/spy_uncertainty_vs_error.png`
- `plots/multitask_predicted_vs_actual.png`
- `plots/multitask_uncertainty_vs_error.png`
- `plots/multitask_window_spearman.png`
- `plots/*_modulation_*.png`

## Critic Pass

This is a one-window smoke with `max_iter=1`, not a promotion run. It confirms the fixed-task-kernel path can fit one sealed 18-ETF rank-5 window, but the runtime was much higher than the earlier rank-insensitive run. A full 24-window, `max_iter=15` version needs an explicit larger runtime budget or a cheaper approximation.

## Lineage

- Manifest: `/Users/et/Desktop/Data_Projects/gparchitect/experiments/2026-06-time-varying-parameterizations/outputs/20260628_gp_18etf_plain_roll60m_rank5_fixedtask_smoke_1w_iter1/manifest.json`
- Artifact path: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet`
- Git commit: `6887fc374c2ecf4e027b2695e8f5de9719e85028`
