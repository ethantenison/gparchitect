# Signed No-Prior Task-Kernel Closeout

## Decision

The signed no-prior task kernel does not beat the prior positive-task-kernel GP on this sealed 18-ETF ranking experiment.

It is the better signed-kernel ablation versus signed LKJ on mean window Spearman IC, but the positive task kernel remains the best current GPArchitect ranking configuration.

## Experiment

- Candidate: GPyTorch `IndexKernel` task covariance with `prior=None`
- Baseline: BoTorch `PositiveIndexKernel` task covariance with `task_covar_prior=None`
- Same setup as prior true-task GP run: 18 ETFs, `plain` variant, rank 5, rolling 60-month training window, 24 sealed OOS windows, BoTorch default optimizer
- Forecast windows: `2024-04-30` through `2026-03-31`
- Run ID: `20260629_gp_18etf_plain_roll60m_rank5_signednoprior_24w_botorchdefault`
- Manifest: `manifest.json`
- Code commit in manifest: `83015946c042c4a1452838b2e1f7a406b638750c`

## Headline Metrics

| task kernel | windows OK | IC / mean Spearman | spread | RMSE | PLL | top1 hit | top3 contains true top1 | top3 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| positive | 24 / 24 | 0.1815 | 0.01675 | 0.03128 | 2.0910 | 0.3750 | 0.3750 | 0.2917 |
| signed no prior | 24 / 24 | 0.1573 | 0.01125 | 0.03143 | 2.0842 | 0.3333 | 0.3333 | 0.2639 |
| signed LKJ eta=2 | 24 / 24 | 0.1176 | 0.01535 | 0.03190 | 2.0758 | 0.4167 | 0.4167 | 0.2917 |

## Readout

Signed no-prior moves in the expected direction relative to signed LKJ on the primary ranking metric: IC improves from `0.1176` to `0.1573`.

That does not recover the positive-kernel result. The positive task kernel still has the best IC, best spread, best RMSE, best predictive log likelihood, and better top-k containment than signed no-prior. Signed LKJ retains a narrow top1-hit advantage, but it is not enough to offset its weaker full cross-sectional ranking.

This means the BayesFolio lesson does not transfer cleanly as a ranking-only claim. The earlier signed-kernel benefit was conditional on that workflow and, at least partly, on portfolio covariance/rotation behavior. In this GPArchitect sealed OOS ranking setup, relaxing the task covariance away from positive pooling has not improved the main ranking objective.

## Visual Check

- `plots/multitask_window_spearman.png`: IC is highly path-dependent, with strong positive windows in mid-2025 and March 2026, but sharp failures around February/March 2025 and February 2026.
- `plots/multitask_predicted_vs_actual.png`: predictions are compressed near zero, so the model mostly ranks by small relative differences rather than calibrated return magnitude.
- `plots/multitask_uncertainty_vs_error.png`: higher predicted standard deviation corresponds to a wider error cloud, but the relationship is noisy.

The visual readout supports treating `0.1573` as a modest ranking signal, not a dominant kernel improvement.

## Caveats

- This is still one sealed 24-window comparison, not a full sensitivity grid.
- The signed no-prior run used the same GPArchitect feature artifact and runner as the positive-kernel run, so it is a clean local ablation, but it is not identical to the older BayesFolio monthly portfolio workflow.
- Current metrics evaluate ranking and prediction quality, not downstream portfolio utility. A signed kernel could still matter for an allocation workflow if covariance geometry changes portfolio construction.
- Logs contain recurring Cholesky jitter warnings, but the run completed with 24 successful windows and produced the expected 432 predictions.

## Next Action

Keep `PositiveIndexKernel` as the current GPArchitect ranking baseline. If we want to reconcile BayesFolio fully, the next fair test is not another task-kernel tweak alone; it is a portfolio-utility rerun using the same forecast outputs or a GPArchitect run aligned to the BayesFolio feature artifact and allocation objective.
