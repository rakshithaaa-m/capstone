# 3. Methods

> DRAFT. Algorithms are described as implemented; exact hyper-parameters are read
> from `models/lgbm_best_params.json` and the per-arm JSONs written by the
> ablation, not restated here from memory.

## 3.1 Reference model

The reference is LightGBM [CITE: Ke et al., 2017] trained on the 212-feature
set. Hyper-parameters are selected by Optuna [CITE: Akiba et al., 2019] with 30
trials maximising validation AUPRC on a fixed 5,000,000-row tuning sample drawn
from the validation split. The objective is AUPRC rather than log-loss because
the decision problem is ranking, not calibration.

The shipped model is an ensemble of five boosters trained with seeds
{42, 7, 13, 99, 23}; prediction is the mean of the five probabilities. Ensembling
is not cosmetic: on this split it moves AUPRC from 0.0791 to 0.0839, which is
larger than the effect of any single feature-set change we tested. Any comparison
between feature sets must therefore hold the ensemble fixed, and Section 3.6 does.

The decision threshold is chosen on the full validation split to minimise
expected cost under a 100:1 false-negative-to-false-positive cost ratio, and is
then applied unchanged to test.

## 3.2 RUL estimator

The second model is an LSTM [CITE: Hochreiter & Schmidhuber, 1997] over per-drive
SMART sequences, regressing remaining useful life in days with a Softplus output
head so that predictions cannot be negative. Training uses a censored hinge loss
so that drives still alive at the end of the observation window contribute a
lower bound rather than a point target.

The network is small (≈32.7k parameters). This is a deliberate scope choice, and
it bounds the claim: Section 3.4 tests whether *this* estimator adds information,
not whether any RUL model could. A LightGBM regressor on the same inputs is used
as a control to separate "the fusion idea fails" from "the estimator is weak"
(Section 3.7).

## 3.3 Fusion candidates

Five combiners are evaluated against the reference on the same test split:

| id | combiner |
|---|---|
| C0 | probability-only control (RUL excluded) |
| C1 | logistic regression stacker on classifier probability + RUL |
| C2 | gradient-boosted stacker on the same two inputs |
| C3 | gradient-boosted stacker with engineered interaction terms |
| C4 | shipped meta-learner (horizon-bounded RUL encoding) |

C0 is the important one. It isolates the effect of *adding the stacking machinery*
from the effect of *adding the RUL signal*: if C1–C4 beat the reference but C0
does not, the gain is the combiner; if C0 also beats it, the comparison is
confounded by model class.

All combiners are fitted on validation predictions and evaluated on test.

## 3.4 Conditional test

Fusion results depend on the combiner, so we also run a model-free test. Partition
test drive-days into deciles by the reference classifier's probability. Within
each decile, compute the AUROC of the RUL score for the failure label.

The logic: inside a band where the classifier has already assigned an
approximately equal probability, a genuinely informative second model should still
order failures above non-failures. AUROC of 0.5 means it does not.

The reported statistic is the median across the ten deciles, with the number of
deciles above 0.5 stated.

## 3.5 Evaluation

**Primary metric.** Row-level (drive-day) AUPRC on the full test split, reported
with lift over prevalence (AUPRC / 0.00046766). Accuracy is not reported: at this
prevalence it is uninformative.

**Uncertainty.** Rows from one drive are not independent. All intervals resample
**physical drives**, never rows. Two procedures are used:

- drive-day AUPRC: subsampled drive bootstrap (40,000 drives per replicate)
  because a full 144.7M-row resample does not fit in memory;
- drive-level AUPRC: exact full bootstrap over all 352,383 drives.

The number of replicates is reported with every interval. `[[VERIFY]]` the
existing headline interval was produced with only 25 replicates; this is too few
for a stable 95% interval and must be increased before publication.

**Row-level versus drive-level.** Drive-level AUPRC (0.598) is computed by taking
each drive's score on its **last observed day** and asking whether that drive ever
fails. This is a legitimate but different question: for a failing drive the last
observation is at most 14 days before failure and typically 0–1 days. It measures
contemporaneous detection, not early warning, and is reported only with that
qualifier.

**Lead time.** The operational metric is the share of failing drives flagged HIGH
at least 7, 14 and 30 days before failure. This is computed from daily
probabilities and is reported separately. `[[VERIFY]]` pending.

**Determinism.** Every evaluation query carries an explicit
`ORDER BY serial_number, date`. A previous comparison in this project reversed
sign (+0.0199) on re-run because of non-deterministic row order from DuckDB.

## 3.6 Identifier ablation (pre-registered)

Five identifiers are removed: `vault_id`, `pod_id`, `cluster_id`, `pod_slot_num`,
`is_legacy_format`. Both arms are trained by the same script with the same seeds,
the same Optuna budget (30 trials, seed 42) and independent tuning, so neither arm
is handicapped by borrowed hyper-parameters.

The decision rule was fixed before the run:

- **superiority** — 95% CI lower bound > 0 and the 212-feature arm wins ≥4/5 paired seeds;
- **equivalence** — CI entirely within ±0.003 AUPRC;
- **non-inferiority** — CI lower bound > −0.003;
- **reversal** — CI upper bound < 0, in which case the identifiers are restored.

The protocol, including this rule, is archived at
`results_ablation/PREREGISTRATION_infra_ablation.md` with the timestamp of the
run log. The 9.7% figure quoted in earlier internal documents is withdrawn: it
compared a single-seed model against a five-seed ensemble, and 65.6% of the gap
was ensembling.

## 3.7 RUL control

To separate "fusion fails" from "the LSTM is weak," a LightGBM regressor is
trained on the same inputs to predict RUL, and its prediction is used as a
stacking feature in place of the LSTM output. If the result is unchanged, the
negative finding is about the information content of RUL, not about the
estimator. If the result changes, the claim is scoped to the LSTM.

## 3.8 Portability protocol

Three folds hold out one datacenter entirely. Each fold is paired with a
size-matched control that holds out a random sample of the same number of drives
from the remaining sites. The control is what makes the result interpretable:
without it, a drop on a held-out site is confounded with having less training
data.

Because each fold scores a 3,000,000-row sample, lift is the stable metric across
folds; raw AUPRC is reported for completeness.

## 3.9 Reproducibility

All stages are scripted and runnable from `RUNBOOK.md`. Large artefacts are
excluded from version control and regenerated. A dated snapshot of models and
result tables is taken before any re-run (`snapshot_2026-10-23/`), because a
previous baseline was lost when model weights were overwritten.
