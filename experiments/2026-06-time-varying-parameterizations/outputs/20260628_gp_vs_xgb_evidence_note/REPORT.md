# GP vs XGB ICC Evidence Note

Run ID: `20260628_gp_vs_xgb_evidence_note`

## Decision

Stop active XGBoost ranker exploration for now. Keep XGBoost as a benchmark family, but move the main modeling effort back to the multitask GP.

The practical reason is that the current XGB evidence does not beat the GP-style ranking signal on the decision metrics we care about. The modeling reason is also coherent: ICC-style monthly ETF ranking is about the joint cross-sectional distribution, and the multitask GP models that dependency directly instead of forcing each month into an independent tree-ranking problem.

Post-run caveat: later runner inspection found the `gparchitect` GP comparison path dropped BoTorch's task covariance kernel after model construction. The committed 18-ETF GP artifact is therefore a GP-style baseline with the recorded rank ineffective, not a valid task-covariance-rank multitask GP benchmark. A fixed one-window rank-5 smoke is tracked separately and the full fixed-task run needs a larger runtime budget.

## Evidence

| profile | universe | windows | IC / mean Spearman | spread / top1-bottom1 | top1 hit |
|---|---:|---:|---:|---:|---:|
| plain multitask GP | 5 ETFs | 24 | 0.2583 | 0.0201 | 0.3750 |
| XGB reg-alpha-zero lexicographic | 18 ETFs | 24 | 0.1657 | -0.0027 | 0.1250 |
| XGB reg-alpha-zero hypervolume | 18 ETFs | 24 | 0.1256 | 0.0066 | 0.0000 |
| XGB prior TPE500 reg-alpha tuned | 18 ETFs | 24 | 0.1658 | 0.0142 | 0.1250 |

The hypervolume-selected XGB profile did not validate as the better balanced profile. It improved spread relative to the reg-alpha-zero lexicographic profile, but lost IC, top1 hit rate, top3 containment, and top3 overlap. The older reg-alpha-tuned XGB profile remains the strongest XGB reference, but it still does not make the XGB exploration track look more promising than the GP track.

## Visual Readout

- `plots/lex_vs_gp_comparison_delta.png`: lexicographic XGB trails the GP baseline on IC, spread, and top1 hit rate.
- `plots/hypervolume_vs_gp_comparison_delta.png`: hypervolume XGB trails the GP baseline more strongly on IC and has zero top1 hits.
- `plots/gp_multitask_window_spearman.png`: GP window-level IC is not uniformly positive, so the GP is not magic; the argument is relative evidence plus a better-matched joint-distribution model.

## Caveats

This is not a clean final same-universe promotion claim. The GP evidence summarized here is the existing 5-ETF multitask GP run, while the latest XGB sealed-OOS runs use the 18-ETF universe. That mismatch means the conclusion should be framed as a stop-work decision for XGB exploration, not as a definitive published GP-vs-XGB benchmark.

The strongest next check, if we need publication-quality evidence, is a sealed 18-ETF multitask GP run over the same `2024-04-30` through `2026-03-31` OOS window and the same feature artifact. Until then, I would not spend more cycles tuning XGB unless it is only to maintain a baseline.

Update: that clean 18-ETF plain multitask GP run now exists at `outputs/20260628_gp_18etf_plain_roll60m_24w_iter15/`. It weakens the GP-positive claim: the 18-ETF plain GP has better top1 hit rate than the XGB profiles, but the lexicographic XGB has higher mean IC and the prior reg-alpha-tuned XGB has higher spread/top3 containment. Treat this note as pre-check context; use the 18-ETF run report for the same-universe readout.

## Critic Pass

Strongest reason this conclusion could be wrong: the GP and XGB evidence are not perfectly matched by universe size, a 5-ETF top1 hit rate is easier to interpret than an 18-ETF top1 hit rate, and the latest `gparchitect` GP artifact dropped the task covariance kernel. The conclusion survives only as a weak resource-allocation decision because multiple XGB variants have failed to produce compelling sealed-OOS ranking behavior; it should not be oversold as a same-universe true multitask GP benchmark.

## Lineage

- GP source run: `experiments/2026-06-time-varying-parameterizations/outputs/20260625_tvls_24w_iter15/`
- XGB lexicographic OOS run: `experiments/2026-06-time-varying-parameterizations/outputs/20260627_xgb_regalpha0_lex_selected_full_tree_18etf_roll60m_24w_oos/`
- XGB hypervolume OOS run: `experiments/2026-06-time-varying-parameterizations/outputs/20260627_xgb_regalpha0_hypervolume_selected_full_tree_18etf_roll60m_24w_oos/`
- XGB family comparison: `experiments/2026-06-time-varying-parameterizations/outputs/20260627_regalpha0_lex_vs_hypervolume_oos_comparison.md`
- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet`
- Manifest: `experiments/2026-06-time-varying-parameterizations/outputs/20260628_gp_vs_xgb_evidence_note/manifest.json`
- Base git commit: `3f40e64760a6e240eac69f480bb4fd9cdcf2b26e`
