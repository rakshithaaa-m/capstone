# Findings — infrastructure-identifier ablation (2026-10-26)

Complete, post-Amendment-02 results for the pre-registered ablation.

## Headline

**EQUIVALENCE.** Under the matched protocol, the five infrastructure identifiers
(`vault_id`, `pod_id`, `cluster_id`, `pod_slot_num`, `is_legacy_format`) are not
detected as an effect on row-level AUPRC.

| quantity | arm A (212 feat, no infra) | arm B (217 feat, infra) | delta (A - B) |
|---|---|---|---|
| AUPRC (full test) | 0.08393801402281403 | 0.0840758288095627 | **-0.0001378 (-0.16%)** |
| ROC-AUC (full test) | 0.8886709964888508 | 0.8932872413460735 | +0.0046 |

| statistic | value |
|---|---|
| 95% bootstrap CI (B=1000, drive-clustered, paired) | [-0.002470, +0.002150] |
| point estimate relative to pre-registered ±0.003 margin | inside the margin |
| verdict (pre-registered decision table) | EQUIVALENCE |

## Gates

1. **Primary gate.** CI spans zero (`[-0.00247, +0.00215]`); point estimate lies
   well inside the pre-registered ±0.003 "immaterial" margin. SUPERIORITY requires
   CI lower bound > 0 and is therefore **not** satisfied; the result is an
   equivalence-class null.
2. **Per-seed gate.** A must exceed B in >=4 of 5 paired seeds. Achieved: **3/5**
   (gains +0.0053, +0.0073, +0.0045 for A in seeds 42, 7, 13; B marginally ahead
   in seeds 99 and 23). The gate is therefore **not** satisfied, consistent with
   a null rather than with a material effect.

## Cross-check that validates the computation path

Arm B's seed-42 AUPRC on the full split = **0.07654906796445234**, which matches
the pre-registration's independently-computed "with infra, 1 seed" value
**0.0765490679644523** to 17 significant figures.

This is the strongest evidence that `17c_per_seed_full.py` implements the same
full-split AUPRC statistic as the earlier Stage-6 runs, and that the earlier
"with infra" number was a full-split single-seed result.

## 2026-10-27 addendum — lead time and calibration

### Lead-time curve (new, `18_leadtime_curve.py`)

Share of failing drives flagged HIGH at `failure_date - L >= ...` horizons:

| horizon (days before failure) | n_evaluated | n_flagged_HIGH | share flagged HIGH |
|---|---|---|---|
| 30 | 4,764 | 2,257 | 0.4296 |
| 14 | 4,915 | 2,664 | 0.5070 |
| 7 | 4,939 | 2,884 | 0.5489 |
| 1 | 5,009 | 3,337 | 0.6351 |

Interpretation: as the lead time shrinks toward failure, the share of failing
drives that the model flags HIGH rises (0.43 at 30 d -> 0.64 at 1 d), which is the
expected qualitative behaviour for a near-failure alert.

### Calibration (new, `19_calibration.py`)

| metric | value |
|---|---|
| Brier score, all rows | 0.010685 |
| Brier score, positives only | 0.315892 |
| ECE, 10 equal-width bins, test set | 0.040070 |

The decision thresholds (Stage 09) are recall-calibrated and therefore quantile-like
decision thresholds, not probability-calibration outputs; the Brier/ECE numbers are
reported without retroactively redefining them.

### RUL control (settled)

The stacking study (`16_stacking_study.py`) already settles the LightGBM-RUL scope:
the best combiner (logistic on classifier prob + RUL) loses to LightGBM-only
(Δ = -0.0165, 95% CI [-0.0246, -0.0091], wins 0% of resamples); and RUL AUROC within
classifier deciles is ~0.49 (median), i.e. RUL carries no information the classifier
lacks. No new LightGBM-RUL control run was needed.

## Cohort diagnostics

| cohort | rows | AUPRC A | AUPRC B | delta |
|---|---|---|---|---|
| vault_seen | 130,838,318 | 0.083424 | 0.083843 | -0.000418 |
| vault_unseen | 13,901,161 | 0.145514 | 0.143423 | +0.002091 |
| drive_seen | 106,951,884 | 0.084964 | 0.085317 | -0.000353 |
| drive_unseen | 37,787,595 | 0.089400 | 0.089686 | -0.000286 |

Unseen-vault rows carry much higher AUPRC (~0.145) with only 2,795 positives —
a sparsity artifact to be handled carefully in prose.

## Amendment 02 (drive-holdout) — retained as diagnostic

On the drive-disjoint subsample (Amendment 02), A beat B by ~+0.009 on average,
consistent with identifiers failing to generalise to drives absent from training.
Reported as a portability diagnostic, not as the primary row-level result.
