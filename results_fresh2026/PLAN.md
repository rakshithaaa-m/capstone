# Fresh out-of-time evaluation on 2026 data — plan

Written 2026-10-30, **before** any 2026 row was scored. Nothing in this plan is
changed after results are seen; deviations, if any, are listed at the end.

## Data
- Backblaze Drive Stats Q1 and Q2 2026 (`data/raw/data_Q1_2026`, `data/raw/data_Q2_2026`),
  2026-01-01 to 2026-06-30. Published after the model, threshold and tiers were frozen.
- Header identical to 2025-12-31 (197 columns). Cleaned to the **same column set** as
  `cleaned_2025_Q4.parquet`; the global near-empty-column decision is not re-made.
- Q4 2025 cleaned rows are prepended as history so rolling windows at the start of
  January use real readings. `days_since_install` uses each drive's first date in the
  **full** 2013–2026 history, as in the original pipeline.
- Features, imputation (train medians) and label definition are those of Stage 3.

## Evaluation window
- Rows dated **2026-01-01 to 2026-06-16**. Later rows cannot be labelled: a failure up
  to 14 days ahead would fall in Q3 2026, which is not published.
- A drive that fails between 2026-06-17 and 2026-06-30 still labels its rows inside
  the window correctly (its failure date is observed).

## Frozen
- The five LightGBM boosters (`models/lgbm_model_seed{42,7,13,99,23}.txt`), averaged.
- Alert threshold 0.62; tier cutoffs HIGH ≥ 0.76406634, MEDIUM ≥ 0.06907166.
- No retraining, recalibration, threshold or cutoff change.

## Reported (same definitions as the 2024-10 → 2025-12 test split)
1. Row-level AUPRC with a drive-clustered bootstrap 95% CI (B = 1000, exact weighted AP),
   ROC-AUC, prevalence and lift.
2. Precision and recall at 0.62.
3. Drive-level AUPRC / ROC-AUC (latest record per drive vs ever positive in the window).
4. Tier table (drive level, latest record) and row-level tier capture.
5. Lead time: share of failing drives at or above 0.62 at 30 / 14 / 7 / 1 days before
   failure (method of `18_leadtime_curve.py`).

## Reading (stated in advance)
- **Stable** if the 2026 row AUPRC CI contains the test point estimate 0.0839;
  **degraded** if the CI lies entirely below it; **improved** if entirely above.
- Because AUPRC depends on prevalence, lift over prevalence is reported alongside.
- The result is reported whichever way it falls.

## Deviations
None to the plan. Post hoc diagnostics are reported separately in FINDINGS_2026.md.

## Pipeline check (before scoring)
`28_fresh_2026.py validate` rebuilt December 2025 features for two of the 60 drive
partitions (351,584 rows) using only the Q4 2025 history, and compared them with
the original `features_test.parquet` rows for the same serial and date. All rows
matched. 205 of 212 features were identical (rtol/atol 1e-4). Seven rolling standard
deviations (SMART 5, 187, 188) differed in at most 0.13% of rows. pandas' rolling
variance depends on where the series starts, and the original series started years
earlier. Output: `validate_features.csv`.
