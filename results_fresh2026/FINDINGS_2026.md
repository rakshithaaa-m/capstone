# Fresh 2026 evaluation: findings

Plan: `PLAN.md`, written before scoring. Script: `28_fresh_2026.py`.

## Pre-registered result (`evaluate`)
Frozen 5-seed LightGBM, threshold 0.62, frozen tiers. Window 2026-01-01 to 2026-06-16:
57.6M drive-days, 363,405 drives, 27,639 positive rows, 2,451 failing drives.

| metric | 2024-10 → 2025-12 test | 2026 H1 |
|---|---|---|
| Row AUPRC [95% CI, B = 1000] | 0.0839 [0.0787, 0.0897] | **0.0608 [0.0551, 0.0673]** |
| Lift over prevalence | 179× | 127× |
| ROC-AUC | 0.889 | 0.870 |
| Precision / recall at 0.62 | 2.7% / 59.9% | 2.2% / 63.8% |
| Drive-level AUPRC | 0.598 | 0.422 |
| HIGH tier: share of failing drives caught | 66.9% | 66.4% |
| Flagged at ≥ 0.62, 14 days before failure (share of all failing) | 50.7% | 50.9% |

**Reading, as stated in the plan: degraded.** The 2026 CI lies entirely below 0.0839.
Recall and lead time held. Ranking quality and precision fell.

## Diagnostics (post hoc, not in the plan)
1. **New drive models are not the cause.** Models absent from training are 7.6% of the
   2026 fleet (27,688 drives). Their HIGH-tier precision is 32%, against 34% for known
   models. (`high_by_model_2026.csv`)
2. **The decline is gradual and began inside the test period.** Monthly AUPRC was
   0.10–0.13 from Oct 2024 to Apr 2025, then 0.07–0.09 from May to Sep 2025, and
   0.04–0.10 in 2026 (median 0.054). The model has been ageing since its training data
   ended (March 2024); 2026 continues that trend. (`monthly_test_vs_2026.csv`)
3. **The cause is false alarms growing over time, not missed failures.**
   - HIGH-tier rows per day rose steadily: 1,393 (Oct 2024) → 2,405 (Dec 2025) → 2,960 (Jun 2026).
   - Positive rows per day stayed roughly flat, at about 140–180.
   - Precision fell for every major drive model, not for a few.
   Hypothesis to test: cumulative counters (power-on hours, start/stop count, raw
   read-error count) grow as the fleet ages, which pushes healthy drives' scores up.
   Retraining on recent data and adding fleet-relative (same-model percentile) features
   both target this.
4. **The drive-level comparison is affected by window length.** Drive-level precision
   compares failing drives, which accumulate over the window, against a one-day
   snapshot of flagged survivors. The 15-month and 5.5-month windows are therefore not
   comparable. The HIGH tier caught more failures per month in 2026 (296) than in the
   test period (234). Row-level and monthly metrics are the fair comparison.
5. **The original test set under-labels its final two weeks.** Positive rows fall from
   about 80 a day in early December 2025 to 0 on 31 December. Failures in January 2026
   were not visible when the test labels were built. Late-December failures were also
   genuinely low (10 in the week of 22 Dec, holidays). Rows from 18 to 31 December 2025
   therefore contain unlabelled positives, which slightly depresses the test AUPRC
   (December AUPRC 0.032). The fix is to relabel those rows using the 2026 failures now
   available, or to end the test window on 2025-12-17; either needs a decision before it
   is applied. The 2026 window already excludes its last 14 days for this reason.
