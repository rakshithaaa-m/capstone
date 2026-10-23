# Pre-registration — infrastructure-identifier ablation

**Written:** 2026-10-23, before any training run in this protocol.
**Author:** Hemanth (UE23AM441A)
**Status:** FROZEN. Do not edit after the first fit starts; deviations go in `DEVIATIONS.md`.

---

## 1. Why this run exists

The previously quoted claim — "removing the five infrastructure identifiers raised
AUPRC by 9.7% (0.07655 → 0.08394)" — compared a **single-seed** model against a
**five-seed ensemble**. Decomposing the existing artefacts:

| quantity | value |
|---|---|
| with infra, 1 seed | 0.0765490679644523 |
| without infra, 1 seed | 0.0790908178208553 |
| without infra, 5-seed ensemble | 0.08393801402281403 |

- identifier effect, single seed: **+0.0025417** (+3.32%)
- ensembling effect: **+0.0048472** (+6.13%)
- originally claimed gap: **+0.0073889** (+9.65%)
- share attributable to ensembling: **65.6%**

The identifier effect has therefore **never been measured under a matched protocol**.
The 9.7% figure is withdrawn and must not appear in any draft, slide, or paper.

## 2. Hypotheses

- **H1 (superiority).** The 212-feature model (identifiers removed) achieves higher
  row-level AUPRC than the 217-feature model (identifiers restored).
- **H0 (negligible difference).** |ΔAUPRC| < 0.003; the identifiers are immaterial
  to ranking quality.

## 3. Features under test

Removed in the 212-feature arm, restored in the 217-feature arm:

`vault_id`, `pod_id`, `cluster_id`, `pod_slot_num`, `is_legacy_format`

## 4. Design

| item | value |
|---|---|
| Arm A | 212 features (identifiers removed) |
| Arm B | 217 features (identifiers restored) |
| Tuning | Optuna, `n_trials=30`, tuning seed 42, same validation sample, **independent per arm** |
| Training seeds | `42,7,13,99,23` (identical across arms) |
| Arm A output | `models_ablation/without_infra`, `results_ablation/without_infra` |
| Arm B output | `models_ablation/with_infra`, `results_ablation/with_infra` |
| Target | `failure_within_14d` |
| Shipped `models/`, `results/` | **read-only inputs; never written** |

The shipped 212-feature arm is **re-run inside this protocol** because its original
tuning budget and seed are not recorded anywhere (`lgbm_best_params.json` stores only
the fitted values). Without the re-run there is no matched comparison.

## 5. Metrics

- **Primary:** row-level (drive-day) AUPRC over the full test split.
- **Secondary:** row-level ROC-AUC; drive-level AUPRC; row-level F1 / precision /
  recall at the cost-optimal threshold.
- **Per-seed:** AUPRC for all five seeds in each arm (mean, min, max, spread).

Drive-level metrics are **declared as last-observation scoring** and are *not* an
early-warning measure. They may only be reported alongside a lead-time curve.

## 6. Statistics

1. **Paired drive-clustered bootstrap** on ΔAUPRC = AUPRC(A) − AUPRC(B).
   Resample **physical drives** with replacement; B_boot = 1000; 95% percentile CI.
2. **Sign consistency.** A must exceed B in **≥ 4 of 5 paired seeds** (matched by
   seed index), in addition to the CI condition.
3. **Determinism.** Every evaluation query carries an explicit
   `ORDER BY serial_number, date`. Any comparison below 1% relative is invalid otherwise.

## 7. Decision rules (fixed in advance, mutually exclusive)

| outcome | condition | interpretation | action |
|---|---|---|---|
| **SUPERIORITY** | CI lower bound > 0 **and** A wins ≥ 4/5 seeds | removing identifiers improves ranking | report the measured Δ with CI; keep 212 |
| **EQUIVALENCE** | entire CI inside (−0.003, +0.003) **and** no sign-consistency requirement met | identifiers are immaterial | keep 212 for simplicity and portability; report as a null |
| **NON-INFERIORITY** | CI lower bound > −0.003 but not equivalence | removal does not materially harm | keep 212; report as "no material cost" |
| **REVERSAL** | CI upper bound < 0 | identifiers help | restore them; **retract the leakage framing** |

A **null with a tight CI is a publishable result** and strengthens the portability
argument: the shipped model does not depend on site identity.

## 8. Pre-declared limitations

- Identifier provenance is unverified. "Deployment cohort" is a **hypothesis**, tested
  separately (share of test rows in vaults never seen in training).
- Only the three largest datacenters are large enough to hold out; 32% of training
  rows are unattributable and dropped from every fold.
- Row-level AUPRC at 0.0468% prevalence is the headline metric. Accuracy is not reported.
- The LSTM RUL estimator is small (≈32.7k parameters). Any "RUL adds nothing" claim is
  scoped to this estimator unless the LightGBM-regressor control also adds nothing.

## 9. Output discipline

Every result is written to `results_ablation/` **as it completes**, never only to chat:

- `<arm>/lgbm_metrics.csv`
- `<arm>/lgbm_best_params.json`
- `<arm>/per_seed_auprc.csv`
- `ablation_summary.csv`
- `ablation_bootstrap.csv`
- `cohort_test.csv`
