# GP vs XGB ICC Evidence Note

Run ID: `20260628_gp_vs_xgb_evidence_note`

## Decision

Stop active XGBoost ranker exploration for now. Keep XGBoost as a benchmark family, but move the main modeling effort back to the multitask GP.

The practical reason is that the current XGB evidence does not beat the GP-style ranking signal on the decision metrics we care about. The modeling reason is also coherent: ICC-style monthly ETF ranking is about the joint cross-sectional distribution, and the multitask GP models that dependency directly instead of forcing each month into an independent tree-ranking problem.

Post-run update: later runner inspection found the first `gparchitect` 18-ETF GP comparison path dropped BoTorch's task covariance kernel after model construction. A corrected rank-5 true-task run now exists at `outputs/20260628_gp_18etf_plain_roll60m_rank5_fixedtask_24w_botorchdefault/` and restores the GP-positive same-window result.

## Evidence

| profile | universe | windows | IC / mean Spearman | spread / top1-bottom1 | top1 hit |
|---|---:|---:|---:|---:|---:|
| true-task rank-5 multitask GP | 18 ETFs | 24 | 0.1815 | 0.0168 | 0.3750 |
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

The original note was not a clean final same-universe promotion claim because the GP evidence summarized there was the existing 5-ETF multitask GP run, while the XGB sealed-OOS runs used the 18-ETF universe. That mismatch is now addressed by the corrected true-task rank-5 18-ETF run.

The strongest next check, if we need publication-quality evidence, is a sealed 18-ETF multitask GP run over the same `2024-04-30` through `2026-03-31` OOS window and the same feature artifact. Until then, I would not spend more cycles tuning XGB unless it is only to maintain a baseline.

Update: the earlier 18-ETF artifact at `outputs/20260628_gp_18etf_plain_roll60m_24w_iter15/` should be treated as a rank-insensitive GP-style baseline because the task kernel was dropped. The corrected same-window readout is `outputs/20260628_gp_18etf_plain_roll60m_rank5_fixedtask_24w_botorchdefault/`; it beats the tracked XGB profiles on IC, spread, top1 hit rate, and top3 containment, while lexicographic XGB retains a slightly higher top3 overlap.

## Critic Pass

Strongest reason this conclusion could be wrong: the corrected true-task rank-5 result still covers only 24 monthly OOS windows, and exact GP runtime is high enough that replication/sensitivity checks are expensive. The conclusion is stronger than the earlier note, but it should still be framed as "prioritize GP-side modeling" rather than a final production allocation rule.

## Lineage

- GP source run: `experiments/2026-06-time-varying-parameterizations/outputs/20260625_tvls_24w_iter15/`
- Corrected true-task rank-5 GP run: `experiments/2026-06-time-varying-parameterizations/outputs/20260628_gp_18etf_plain_roll60m_rank5_fixedtask_24w_botorchdefault/`
- XGB lexicographic OOS run: `experiments/2026-06-time-varying-parameterizations/outputs/20260627_xgb_regalpha0_lex_selected_full_tree_18etf_roll60m_24w_oos/`
- XGB hypervolume OOS run: `experiments/2026-06-time-varying-parameterizations/outputs/20260627_xgb_regalpha0_hypervolume_selected_full_tree_18etf_roll60m_24w_oos/`
- XGB family comparison: `experiments/2026-06-time-varying-parameterizations/outputs/20260627_regalpha0_lex_vs_hypervolume_oos_comparison.md`
- Feature artifact: `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_05.parquet`
- Manifest: `experiments/2026-06-time-varying-parameterizations/outputs/20260628_gp_vs_xgb_evidence_note/manifest.json`
- Base git commit: `3f40e64760a6e240eac69f480bb4fd9cdcf2b26e`
