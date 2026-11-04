# Pre-registration: infrastructure-identifier ablation for Model 2

Written 2026-11-04, before any model in this protocol was trained. Deviations go in the
log at the end.

## Question
Does the shipped Model 2 (`models_retrain2026/A`, 212 features, identifiers excluded)
lose ranking quality compared with the same model with the five infrastructure
identifiers restored? Same question and decision rules as Model 1's pre-registered
ablation (`results_ablation/PREREGISTRATION_infra_ablation.md`, section 7).

## Arms
- **A (shipped):** Model 2 as trained, 212 features.
- **B:** 217 features = A + `vault_id`, `pod_id`, `cluster_id`, `pod_slot_num`,
  `is_legacy_format` (boolean as 0/1). Same training rows (`train_resampled_A`), same
  hyperparameters, same seeds 42, 7, 13, 99, 23, five-seed average.

**Difference from Model 1's protocol:** Model 1's arms were each re-tuned with Optuna
(30 trials). Here both arms use Model 2's hyperparameters (`models/lgbm_best_params.json`)
without re-tuning, so the comparison isolates the features at fixed settings.

## Test
2026-01-01 → 2026-06-16 (57.6M drive-days, one look). ΔAUPRC = AUPRC(A) − AUPRC(B),
paired drive-clustered bootstrap, B = 1000. Sign consistency: A beats B in ≥ 4 of 5
seeds (matched by seed).

## Decision rules (Model 1's, unchanged)
| outcome | condition |
|---|---|
| SUPERIORITY | CI lower bound > 0 and A wins ≥ 4/5 seeds |
| EQUIVALENCE | entire CI inside (−0.003, +0.003) |
| NON-INFERIORITY | CI lower bound > −0.003 but not equivalence |
| REVERSAL | CI upper bound < 0 (identifiers help) |
Otherwise: inconclusive.

## Deviations
(none)

## Result (2026-11-04, appended after the single look)
AUPRC A (shipped, no identifiers) 0.07789; B (identifiers restored) 0.07972.
ΔAUPRC = A − B = −0.00182, 95% CI [−0.00339, −0.00027] (B = 1000, 363,405 drives).
B beats A in all 5 matched seeds. **Verdict: REVERSAL** — the identifiers help Model 2
slightly. Model 1's verdict was equivalence. Full numbers: `infra_ablation.json`.
