# Pre-registration: retraining on recent data (evaluated on 2026)

Written 2026-11-02, before any model in this study was trained or any validation or
test score was computed. It is not changed after results are seen; deviations go in
the log at the end with the date and reason.

## Motivation (from FINDINGS_2026.md, already observed)
The frozen model (trained on data to 2024-03-31) degrades on 2026 (row AUPRC 0.0608
against 0.0839). Monthly AUPRC declines from mid-2025, driven by growing daily
HIGH alerts. Question: **does retraining on recent data recover performance, and do
fleet-relative features add to it?**

## Split (temporal, 14-day purge gaps)
| split | dates | drive-days | positive rows | failing drives |
|---|---|---|---|---|
| train | 2013-04-10 → 2025-06-16 | 616,441,524 | 388,417 | 28,672 |
| gap | 2025-06-17 → 2025-06-30 | not used | | |
| validation | 2025-07-01 → 2025-12-17 | 56,039,738 | 25,541 | 2,099 |
| gap | 2025-12-18 → 2025-12-31 | not used (also under-labelled) | | |
| test | 2026-01-01 → 2026-06-16 | 57,594,607 | 27,639 | 2,451 |

Features and labels are exactly those of Stages 3–4 and `28_fresh_2026.py` (212
features; `failure_within_14d`, positive at 1–14 days before failure).

## Candidates
- **F (frozen):** the shipped 5-seed ensemble, unchanged.
- **A (retrained):** same 212 features and the same hyperparameters
  (`models/lgbm_best_params.json`, no re-tuning), five seeds 42, 7, 13, 99 and 23,
  averaged. Training set: all positive rows of the train split plus a Bernoulli
  sample of negatives at 10 per positive (seed 42), as in Stage 5.
- **B (retrained + fleet-relative):** A plus 10 features. For each of
  smart_1_raw, smart_4_raw, smart_5_raw, smart_9_raw, smart_187_raw, smart_188_raw,
  smart_193_raw, smart_194_raw, smart_197_raw and smart_198_raw, the drive's
  `percent_rank` among drives of the **same model on the same day**. The value is
  missing where the attribute is missing, or where fewer than 50 drives of that model
  report that day. Same hyperparameters, seeds and sampling as A.

## Fitted on validation only
For A and B: the alert threshold (cost-optimal at FN:FP = 100:1, grid 0.01–0.99, as in
Stage 6) and the tier cutoffs (HIGH captures 50% and HIGH+MEDIUM 80% of positive
validation rows, as in Stage 9). F keeps its shipped threshold and cutoffs.

## Decisions
1. **B against A (decided on validation).** B is adopted only if its validation AUPRC
   exceeds A's by ≥ +0.003 and the paired drive-clustered bootstrap 95% CI
   (B = 1000) lies entirely above 0. Otherwise A is the retrained candidate (R).
2. **Confirmatory test: R against F on 2026 (one look).** ΔAUPRC = R − F, with a
   paired drive-clustered bootstrap (B = 1000, exact weighted AP):
   - **retraining helps:** Δ ≥ +0.003 and CI lower bound > 0. R replaces F as the
     shipped model, with its refitted threshold and tiers.
   - **no material change:** CI entirely within (−0.003, +0.003). F stays.
   - **worse:** CI upper bound < 0. F stays.
   - otherwise **inconclusive.** F stays.
   The margin of 0.003 is the one used in the earlier pre-registered studies.
3. All candidates are reported on test whatever the decisions, labelled descriptive
   where not confirmatory.

## Secondary (descriptive)
Precision and recall at each model's own threshold; HIGH-tier rows per day and HIGH
row precision; monthly AUPRC; drive-level AUPRC; lead time (≥ threshold at 30/14/7/1
days before failure).

## Order of work
A is trained and scored first (validation and test predictions are written but the
test file is not summarised until the B decision is made on validation). Then B is
built and scored, B is decided on validation, and the confirmatory test is run.

## Deviations
- 2026-11-02, implementation only: the join that attaches percentiles to the validation
  and test rows ran out of memory as a single query, so it is now done one month at a
  time. The rows, features and values are the same. No results had been seen.
