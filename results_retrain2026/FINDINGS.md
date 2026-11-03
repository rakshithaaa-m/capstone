# Retraining study: findings

Pre-registration: `PREREGISTRATION.md` (one implementation-only deviation, logged there).
Script: `29_retrain_2026.py`. Full numbers: `results.json`. Logs: `run_A.log`, `run_B.log`.

## Decision 1: fleet-relative features (validation, 2025-07-01 to 2025-12-17)
| | AUPRC |
|---|---|
| F (frozen) | 0.0663 |
| A (retrained) | 0.0842 |
| B (retrained + same-model percentiles) | 0.0864 |

B − A = +0.0022 [95% CI +0.0005, +0.0041]. The gain is real (CI above 0) but below the
+0.003 margin, so **B is not adopted; R = A.**

## Decision 2: confirmatory test on 2026 (2026-01-01 to 2026-06-16, one look)
A − F = **+0.0171 [95% CI +0.0132, +0.0209]**, B = 1000 drive-clustered bootstrap.
**Verdict: retraining helps.** A replaces F as the shipped model, with threshold 0.73 and
tiers HIGH ≥ 0.7211, MEDIUM ≥ 0.0325 (fitted on validation).

## Descriptive (test, each model at its own validation-fitted threshold and tiers)
| | F (frozen) | A (retrained) | B (descriptive) |
|---|---|---|---|
| Row AUPRC | 0.0608 | **0.0779** | 0.0784 |
| ROC-AUC | 0.870 | 0.886 | 0.892 |
| Threshold | 0.62 | 0.73 | 0.75 |
| Precision / recall at threshold | 2.2% / 63.8% | 3.6% / 57.4% | 4.1% / 54.7% |
| Drive-level AUPRC | 0.422 | 0.459 | 0.469 |
| HIGH rows per day | 2,738 | 2,747 | 2,653 |
| Flagged 14 days before failure (share of failing drives) | 50.9% | 44.1% | 41.3% |

Monthly AUPRC improves in every 2026 month (A vs F: Jan 0.053 vs 0.037, Feb 0.087 vs
0.064, Mar 0.060 vs 0.048, Apr 0.118 vs 0.103, May 0.081 vs 0.059, Jun 0.058 vs 0.038).

## Reading
- Retraining on data to mid-2025 recovers most of the 2026 loss: AUPRC 0.061 → 0.078
  (+28%), against 0.084 on the original 2024-25 test. The decline was mostly model
  ageing, which periodic retraining fixes.
- The cost-optimal threshold rose (0.62 → 0.73), trading recall and lead time for
  precision (2.2% → 3.6%, +60% relative). At a matched recall the comparison is by AUPRC.
- HIGH-tier volume did not fall, because the tier cutoffs are set to capture 50% of
  validation positives; HIGH precision rose only slightly (3.4% → 3.5%).
- Fleet-relative percentiles add a small, consistent gain (validation +0.002, test
  +0.0005, better ROC and precision) that did not meet the pre-registered margin.
