# 18-ETF Plain Multitask GP Sealed OOS

Run ID: `20260628_gp_18etf_plain_roll60m_24w_iter15`
Created: `2026-06-28T13:55:32.557076+00:00`

## Question

Does the plain multitask GP still look better than the XGB ranker family when evaluated cleanly on the same 18-ETF universe and sealed OOS window?

## Decision Readout

Post-run caveat: this artifact should be treated as a rank-insensitive GP-style baseline, not a valid task-covariance-rank-3 multitask GP. A later inspection found the runner replaced BoTorch's `ProductKernel(data, task)` with the data kernel alone after model construction, dropping the `PositiveIndexKernel` task covariance and making the recorded `rank` ineffective.

The clean same-universe result is mixed, not a GP victory lap.

The 18-ETF plain multitask GP beats the newest XGB runs on top1 hit rate, ties the reg-alpha-zero lexicographic XGB on top3 containment, and has positive top-bottom spread. But it does not beat the strongest XGB readouts on mean IC/spread: the lexicographic XGB has higher IC, and the prior reg-alpha-tuned XGB has higher spread and top3 containment.

I would still stop wide XGB exploration for now, but the reason is weaker after this check: not "plain GP clearly wins on 18 ETFs," but "XGB tuning has not produced a decisive enough advantage to justify more search, and the GP family still deserves the next modeling cycles."

## Design

- ETF universe: `BND, BNDX, EWX, HYEM, HYG, IEF, IJR, IWM, LQD, MGK, SPY, VEA, VNQ, VNQI, VSS, VTV, VWO, VWOB`
- Forecast window: `2024-04-30` through `2026-03-31`
- Windows: `24`
- Failures: `0`
- Train window: rolling `60` months
- Max optimizer iterations: `15`
- Multitask rank: `3` recorded, but ineffective in this artifact because the task covariance kernel was dropped after model construction.
- Variant: `plain`
- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet`

## GP Metrics

| metric | value |
|---|---:|
| mean IC / window Spearman | 0.1395 |
| top1-bottom1 spread | 0.0053 |
| top1 hit rate | 0.1667 |
| top3 contains true top1 | 0.2917 |
| top3 overlap | 0.2500 |
| RMSE | 0.0316 |
| predictive log likelihood | 2.0666 |
| 95% interval coverage | 0.9745 |
| directional accuracy | 0.5046 |

## Same-Universe Comparison

| profile | IC | spread | top1 | top3 contains true top1 | top3 overlap |
|---|---:|---:|---:|---:|---:|
| 18-ETF plain multitask GP | 0.1395 | 0.0053 | 0.1667 | 0.2917 | 0.2500 |
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

Strongest reason the GP-positive story could be wrong: this artifact did not preserve the task covariance kernel, so it is not a clean true multitask GP rank comparison. Even on its own terms, the clean 18-ETF plain GP run loses mean IC to the lexicographic XGB and loses spread/top3 containment to the prior reg-alpha-tuned XGB. The evidence is not strong enough to claim plain multitask GP dominates XGB on the same sealed 18-ETF ranking task.

Strongest reason not to reopen XGB exploration anyway: the XGB edge is not robust across selection strategies, top1 behavior remains weak, and the gains are not decisive over only 24 monthly windows. A better next GP-side check is a stronger multitask GP variant, not more XGB search.

## Lineage

- Manifest: `/Users/et/Desktop/Data_Projects/gparchitect/experiments/2026-06-time-varying-parameterizations/outputs/20260628_gp_18etf_plain_roll60m_24w_iter15/manifest.json`
- Predictions: `multitask_predictions.csv`
- Window metrics: `multitask_window_metrics.csv`
- Summary: `summary.csv`
- Git commit at run time: `539e1040a059a098e4a84abdf15f4280f13f5d3f`
