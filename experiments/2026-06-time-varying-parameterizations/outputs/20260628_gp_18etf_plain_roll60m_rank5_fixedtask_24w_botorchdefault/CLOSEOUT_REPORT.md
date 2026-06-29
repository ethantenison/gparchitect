# Close-Out Report: True-Task Rank-5 Multitask GP

Date: `2026-06-29`

Run: `20260628_gp_18etf_plain_roll60m_rank5_fixedtask_24w_botorchdefault`

## Decision

This experiment supports moving the main ranking-modeling effort back toward the true-task multitask GP family and away from additional XGB ranker search for now.

The result is best interpreted as a useful out-of-sample ranking signal, not as a production-ready allocation rule. The model shows enough cross-sectional skill to justify the next GP-side validation runs, but the evidence is still only 24 sealed monthly windows and the exact GP path is computationally expensive.

## Experiment Question

Does the BayesFolio-style plain multitask GP still compare favorably to the XGB ranker family when the `gparchitect` runner preserves BoTorch's task covariance kernel and uses the same 18-ETF sealed OOS windows?

## Setup

- Universe: 18 ETFs: `BND`, `BNDX`, `EWX`, `HYEM`, `HYG`, `IEF`, `IJR`, `IWM`, `LQD`, `MGK`, `SPY`, `VEA`, `VNQ`, `VNQI`, `VSS`, `VTV`, `VWO`, `VWOB`
- Forecast period: `2024-04-30` through `2026-03-31`
- OOS windows: 24 monthly windows
- Failures: 0
- Training window: rolling 60 months
- Candidate: plain multitask GP, rank 5, BoTorch default optimizer
- Task covariance: preserved `PositiveIndexKernel(task)` inside BoTorch `ProductKernel(data, task)`
- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet`

## Headline Results

| metric | result | interpretation |
|---|---:|---|
| mean IC / window Spearman | 0.1815 | Modest positive ranking signal across ETFs |
| top1-bottom1 spread | 0.0168 | The predicted top ETF beat the predicted bottom ETF by about 1.7 percentage points per window on average |
| top1 hit rate | 0.3750 | Exact top ETF was selected in 9 of 24 windows |
| top3 contains true top1 | 0.3750 | The true winner appeared in the predicted top 3 in 9 of 24 windows |
| top3 overlap | 0.2917 | The predicted and realized top-3 sets overlapped by 0.875 ETFs on average |
| RMSE | 0.0313 | Magnitude forecasts remain noisy |
| predictive log likelihood | 2.0910 | Probabilistic fit is usable, but should not be read alone |
| 95% interval coverage | 0.9838 | Intervals are conservative |
| directional accuracy | 0.5255 | Directional signal is weakly above chance |

## IC Interpretation

The `0.1815` IC is a modest but real cross-sectional ranking signal. It means the model tends to rank stronger future ETF returns above weaker future ETF returns, but the rank ordering remains noisy.

For an 18-ETF universe, this does not mean the model usually gets the full rank order correct. It more likely means the model is better at separating broad winners from broad losers than at placing every ETF in the exact right order. That readout is consistent with the other ranking metrics: top1 hit rate is meaningfully above the random baseline of about `1 / 18 = 0.0556`, but the top3 containment rate of `0.3750` still leaves many missed winners.

The positive spread is important. The IC is not just a cosmetic correlation; the predicted top-minus-bottom pair produced positive realized separation on average. That makes the ranking signal potentially decision-relevant.

## Same-Window Comparison

| profile | IC | spread | top1 | top3 contains true top1 | top3 overlap |
|---|---:|---:|---:|---:|---:|
| 18-ETF true-task multitask GP rank5 | 0.1815 | 0.0168 | 0.3750 | 0.3750 | 0.2917 |
| 18-ETF prior GP artifact, task kernel dropped | 0.1395 | 0.0053 | 0.1667 | 0.2917 | 0.2500 |
| XGB reg-alpha-zero lexicographic | 0.1657 | -0.0027 | 0.1250 | 0.2917 | 0.3056 |
| XGB reg-alpha-zero hypervolume | 0.1256 | 0.0066 | 0.0000 | 0.2500 | 0.2222 |
| XGB prior TPE500 reg-alpha tuned | 0.1658 | 0.0142 | 0.1250 | 0.3333 | 0.2361 |

The corrected GP beats the tracked XGB profiles on IC, spread, top1 hit rate, and top3 containment. The lexicographic XGB profile remains slightly better on top3 overlap, so the GP result is not a clean sweep. But the main decision metrics point toward GP-side work.

The corrected GP also improves over the earlier `gparchitect` GP artifact where the task covariance was accidentally dropped. That is the most important engineering finding: preserving the true task kernel materially changes the ranking readout.

## Window-Level Behavior

The window IC plot shows useful but unstable signal. The model has strong positive windows, including `2026-03-31` (`0.7833`), `2025-04-30` (`0.6512`), and `2024-04-30` (`0.5810`), but it also has materially negative windows, including `2025-02-28` (`-0.7110`), `2025-01-31` (`-0.4469`), and `2026-02-27` (`-0.4345`).

This is the right shape for a promising research result and the wrong shape for a confident live allocation rule. The model has signal, but it can be wrong in bursts.

## Visual Readout

- `plots/multitask_window_spearman.png`: IC is path-dependent, not uniformly positive. The positive mean comes from enough strong ranking windows, not from stable month-after-month dominance.
- `plots/multitask_predicted_vs_actual.png`: point forecasts are compressed near zero relative to realized excess returns. This supports treating the model primarily as a ranker/uncertainty model, not as a precise return-level forecaster.
- `plots/multitask_uncertainty_vs_error.png`: larger predicted standard deviations generally align with larger absolute errors, but calibration is rough and should be checked before using posterior uncertainty directly for sizing.
- `plots/summary_mean_window_spearman.png` and `plots/summary_long_short_top1_bottom1_mean.png`: aggregate IC and spread are both positive for the corrected GP run.

## Critic Pass

Strongest reason this conclusion could be wrong: the sample has only 24 monthly OOS windows. A few strong windows can move the average IC materially, and the negative windows show that the model can invert the ranking in some regimes.

Second concern: the predicted-vs-actual plot shows compressed predictions. That is not necessarily a fatal flaw for ranking, but it warns against interpreting predicted returns as calibrated expected excess returns without more calibration work.

Third concern: exact GP runtime is high. The model may be statistically promising but operationally awkward unless future runs are bounded, profiled, or approximated.

The critic pass does not overturn the conclusion. It narrows the claim: prioritize true-task GP research, but do not treat this run as final trading evidence.

## Close-Out Claim

The corrected true-task rank-5 multitask GP produced the best same-window evidence so far for the 18-ETF sealed OOS ranking task. The model has a modest positive IC (`0.1815`), economically relevant top-minus-bottom separation (`0.0168`), and substantially better top1 hit rate than the tracked XGB profiles.

The result indicates that the model can rank ETFs better than chance on average, especially at the broad winner-versus-loser level. It does not indicate reliable exact rank ordering each month.

## Recommended Next Step

Run one bounded GP-side robustness check before making allocation decisions:

1. Repeat the corrected true-task GP on the same sealed setup with a nearby rank or optimizer setting.
2. Add a simple portfolio-policy backtest using the predictions with explicit turnover and transaction-cost assumptions.
3. Report window-level IC, top-minus-bottom spread, drawdown/path behavior, turnover, and calibration together.

Until that exists, the decision is: keep XGB as a benchmark, focus new modeling cycles on the true-task multitask GP path.

## Lineage

- Manifest: `manifest.json`
- Predictions: `multitask_predictions.csv`
- Window metrics: `multitask_window_metrics.csv`
- Ranking summary: `ranking_summary.csv`
- Comparison summary: `comparison_summary.csv`
- Existing run report: `REPORT.md`
- Evidence note: `../20260628_gp_vs_xgb_evidence_note/REPORT.md`
- Run-time git commit: `08f5bbfcc5d610e093d1d33ab8c2cb2a94d50385`
- Close-out report commit: recorded by the commit that adds this file
