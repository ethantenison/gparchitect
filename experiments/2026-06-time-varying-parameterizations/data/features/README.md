# Feature Data Notes

## `portfolio_etf_macro_features_18etf_common_history_201306_202605.parquet`

Created on 2026-06-26 from BayesFolio's `build_features_dataset` pipeline using
the same feature-generation path as the monthly portfolio artifacts.

Canonical BayesFolio artifact:

`/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_18etf_common_history_201306_202605.parquet`

Copied into this GPArchitect experiment for local experiment lineage:

`experiments/2026-06-time-varying-parameterizations/data/features/portfolio_etf_macro_features_18etf_common_history_201306_202605.parquet`

Dataset summary:

- ETF universe: `SPY`, `MGK`, `VTV`, `IJR`, `IWM`, `VNQ`, `VNQI`, `VEA`, `VWO`,
  `VSS`, `BND`, `IEF`, `BNDX`, `LQD`, `HYG`, `EWX`, `VWOB`, `HYEM`.
- Earliest common all-ETF month-end: 2013-06-28.
- Full date range: 2013-06-28 through 2026-05-29.
- Labeled target range: 2013-06-28 through 2026-04-30.
- May 2026 is included as the unlabeled forecasting tail.
- Shape: 2,808 rows x 32 columns.
- Labeled rows: 2,790.
- SHA256: `62f696cede8356d68f0e6baf292d964c93c157534ac2c7be0d0b9d13e12bbbf9`.

BayesFolio construction notes:

- ETF and macro predictors are shifted by one monthly period before model use to
  reduce lookahead risk.
- ETF-local rolling features are computed from daily prices, resampled to month
  end, and missing ETF feature values are filled with `0` by the BayesFolio
  feature builder before the one-period predictor shift.
- This means the earliest rows include neutral-imputed rolling-feature warmup,
  not fully informed rolling signals.

Momentum warmup caveat:

- `mom12m` and `chmom` first become genuinely nonzero in the shifted dataset on
  2014-07-31.
- `mom36m` first becomes genuinely nonzero in the shifted dataset on 2016-07-29.
- `cs_mom_rank` is present earlier, but before `mom12m` warms up it is ranking
  zero-imputed momentum and should not be treated as meaningful.

Recommended experiment views:

- Max-history view: use the full 2013-06-28 onward artifact and accept early
  neutral-imputed momentum warmup.
- Strict full-tree view: start at 2016-07-29 when `mom36m` is informed, then
  compare against the max-history view with the same out-of-sample windows.
