# 18-ETF True-Task Rank-5 Multitask GP Sealed OOS

Run ID: `20260628_gp_18etf_plain_roll60m_rank5_fixedtask_24w_botorchdefault`
Created: `2026-06-28T16:05:34.621163+00:00`

## Question

Does the BayesFolio-style plain multitask GP still compare favorably to the XGB ranker family when the `gparchitect` runner preserves BoTorch's task covariance kernel and uses the same 18-ETF sealed OOS windows?

## Decision Readout

This is the first clean same-window result in this thread for the true task-covariance GP path. It improves materially over the earlier `gparchitect` GP artifact where the task kernel was accidentally dropped, and it beats the tracked XGB profiles on mean IC, top1 hit rate, spread, and top3 containment. XGB lexicographic still has slightly higher top3 overlap.

The result supports prioritizing the true multitask GP family over more XGB ranker search. It does not yet prove this specific GP is production-ready; the evidence is only 24 monthly windows and exact GP runtime is high enough that future experiments need bounded designs.

## Design

- ETF universe: `BND, BNDX, EWX, HYEM, HYG, IEF, IJR, IWM, LQD, MGK, SPY, VEA, VNQ, VNQI, VSS, VTV, VWO, VWOB`
- Forecast window: `2024-04-30` through `2026-03-31`
- Windows: `24`
- Failures: `0`
- Train window: rolling `60` months
- Optimizer: BoTorch default `fit_gpytorch_mll(mll)`, matching BayesFolio monthly/notebook usage
- Max optimizer iterations: `BoTorch default`
- Multitask rank: `5`
- Variant: `plain`
- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet`
- Task covariance: preserved `PositiveIndexKernel(task)` inside BoTorch `ProductKernel(data, task)`

## GP Metrics

| metric | value |
|---|---:|
| mean IC / window Spearman | 0.1815 |
| top1-bottom1 spread | 0.0168 |
| top1 hit rate | 0.3750 |
| top3 contains true top1 | 0.3750 |
| top3 overlap | 0.2917 |
| RMSE | 0.0313 |
| predictive log likelihood | 2.0910 |
| 95% interval coverage | 0.9838 |
| directional accuracy | 0.5255 |

## Same-Window Comparison

| profile | IC | spread | top1 | top3 contains true top1 | top3 overlap |
|---|---:|---:|---:|---:|---:|
| 18-ETF true-task multitask GP rank5 | 0.1815 | 0.0168 | 0.3750 | 0.3750 | 0.2917 |
| 18-ETF prior GP artifact, task kernel dropped | 0.1395 | 0.0053 | 0.1667 | 0.2917 | 0.2500 |
| XGB reg-alpha-zero lexicographic | 0.1657 | -0.0027 | 0.1250 | 0.2917 | 0.3056 |
| XGB reg-alpha-zero hypervolume | 0.1256 | 0.0066 | 0.0000 | 0.2500 | 0.2222 |
| XGB prior TPE500 reg-alpha tuned | 0.1658 | 0.0142 | 0.1250 | 0.3333 | 0.2361 |

## Visuals

- `plots/multitask_window_spearman.png`
- `plots/multitask_predicted_vs_actual.png`
- `plots/multitask_uncertainty_vs_error.png`
- `plots/summary_mean_window_spearman.png`
- `plots/summary_long_short_top1_bottom1_mean.png`

## Critic Pass

Strongest reason the GP-positive story could be wrong: 24 monthly windows is still a small sealed sample, and the rank-5 true-task model is much more expensive than the accidentally task-kernel-free artifact. The result should be treated as evidence to continue GP-side work, not as a final allocation rule.

Strongest reason not to reopen XGB exploration: the clean true-task rank-5 GP now beats the XGB candidates on the main decision metrics except top3 overlap, while the XGB gains remain inconsistent across selection strategies.

## Lineage

- Manifest: `/Users/et/Desktop/Data_Projects/gparchitect/experiments/2026-06-time-varying-parameterizations/outputs/20260628_gp_18etf_plain_roll60m_rank5_fixedtask_24w_botorchdefault/manifest.json`
- Predictions: `multitask_predictions.csv`
- Window metrics: `multitask_window_metrics.csv`
- Ranking summary: `ranking_summary.csv`
- Comparison summary: `comparison_summary.csv`
- Git commit at run time: `08f5bbfcc5d610e093d1d33ab8c2cb2a94d50385`
