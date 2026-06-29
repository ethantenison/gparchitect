# Time-Varying Parameterization Experiment

Run ID: `20260629_gp_18etf_plain_roll60m_rank5_signednoprior_24w_botorchdefault`
Created: `2026-06-29T14:57:24.391633+00:00`

## Question

Can bounded nonlinear time-varying lengthscale/outputscale parameterizations improve GPArchitect forecasts versus the current linear TVOS/TVLS hook, without using changepoint kernels?

## Design

- ETF universe: `BND, BNDX, EWX, HYEM, HYG, IEF, IJR, IWM, LQD, MGK, SPY, VEA, VNQ, VNQI, VSS, VTV, VWO, VWOB`
- Windows: `[{'train_end_date': '2024-03-29', 'forecast_date': '2024-04-30'}, {'train_end_date': '2024-04-30', 'forecast_date': '2024-05-31'}, {'train_end_date': '2024-05-31', 'forecast_date': '2024-06-28'}, {'train_end_date': '2024-06-28', 'forecast_date': '2024-07-31'}, {'train_end_date': '2024-07-31', 'forecast_date': '2024-08-30'}, {'train_end_date': '2024-08-30', 'forecast_date': '2024-09-30'}, {'train_end_date': '2024-09-30', 'forecast_date': '2024-10-31'}, {'train_end_date': '2024-10-31', 'forecast_date': '2024-11-29'}, {'train_end_date': '2024-11-29', 'forecast_date': '2024-12-31'}, {'train_end_date': '2024-12-31', 'forecast_date': '2025-01-31'}, {'train_end_date': '2025-01-31', 'forecast_date': '2025-02-28'}, {'train_end_date': '2025-02-28', 'forecast_date': '2025-03-31'}, {'train_end_date': '2025-03-31', 'forecast_date': '2025-04-30'}, {'train_end_date': '2025-04-30', 'forecast_date': '2025-05-30'}, {'train_end_date': '2025-05-30', 'forecast_date': '2025-06-30'}, {'train_end_date': '2025-06-30', 'forecast_date': '2025-07-31'}, {'train_end_date': '2025-07-31', 'forecast_date': '2025-08-29'}, {'train_end_date': '2025-08-29', 'forecast_date': '2025-09-30'}, {'train_end_date': '2025-09-30', 'forecast_date': '2025-10-31'}, {'train_end_date': '2025-10-31', 'forecast_date': '2025-11-28'}, {'train_end_date': '2025-11-28', 'forecast_date': '2025-12-31'}, {'train_end_date': '2025-12-31', 'forecast_date': '2026-01-30'}, {'train_end_date': '2026-01-30', 'forecast_date': '2026-02-27'}, {'train_end_date': '2026-02-27', 'forecast_date': '2026-03-31'}]`
- Max optimizer iterations: `BoTorch default`
- Multitask rank: `5`
- Task kernel: `signed_no_prior`
- Task kernel details: `GPyTorch IndexKernel with prior=None`
- Train window months: `60`
- Changepoint kernels: excluded by design.

## Summary

```text
    scope model  ok_windows  failed_windows     rmse  predictive_log_likelihood  directional_accuracy  mean_window_spearman  long_short_top1_bottom1_mean
multitask plain          24               0 0.031426                    2.08424              0.497685              0.157293                      0.011247
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

This is a smoke experiment, not a promotion run. The strongest reason the result could be wrong is optimizer noise and very small window count. Treat any win as a candidate for the next larger run, not as a final modeling decision.

## Lineage

- Manifest: `/Users/et/Desktop/Data_Projects/gparchitect/experiments/2026-06-time-varying-parameterizations/outputs/20260629_gp_18etf_plain_roll60m_rank5_signednoprior_24w_botorchdefault/manifest.json`
- Artifact path: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet`
- Git commit: `83015946c042c4a1452838b2e1f7a406b638750c`
