# Amendment 01 — per-seed statistic on a fixed subsample

**Written:** 2026-10-23, after arm A (`without_infra`) completed and **before**
arm B (`with_infra`) was trained or scored.

## Why this amendment exists

`PREREGISTRATION_infra_ablation.md` requires the 212-feature arm to beat the
217-feature arm in **≥4 of 5 paired seeds**, in addition to the ΔAUPRC margin
and the bootstrap interval.

The pipeline as built cannot satisfy that requirement from its own artifacts.
`stream_predict_ensemble()` accumulates `acc += probs` over the five boosters
and stores only `acc / 5` in `lgbm_test_probs.csv`; the individual seed
probabilities are discarded after each pass. Nothing on disk therefore
identifies how large the seed-to-seed spread is, and the sign-consistency rule
cannot be evaluated as written.

This is a defect in the pre-registration, not a finding. It is corrected here,
before arm B exists, so that the correction cannot be chosen to fit a result.

## The rule

Per-seed scores are computed by `17b_per_seed_auprc.py` for **both** arms with
one fixed sampling rule:

- split: `data/processed/features_test.parquet` (144,739,479 rows, 1,334 row groups);
- selected row groups: indices `0, 14, 28, ...` — every 14th row group;
- actual sample: 96 row groups, **10,985,502 rows = 7.59% of test**. (Row groups
  are not uniformly sized, so the earlier estimate of 10,416,000 rows was wrong;
  the *rule* is the group stride and is unchanged — only the row count is
  corrected here, after running arm A but before arm B exists.)
- observed sample prevalence **0.044231%** (4,859 positives) against 0.046766%
  for the full test split, i.e. −5.4% relative. Within the 20% tolerance
  declared below, so the subsample is used as planned;
- seed order taken from `lgbm_ensemble_seeds.json` in each arm's model directory;
- statistic: AUPRC (average precision) and ROC-AUC per seed, plus the mean of
  the five seed probabilities, which is reported for reference only.

Both arms are scored on the *same* rows, so the per-seed comparison is paired by
construction: seed *s* of arm A and seed *s* of arm B see identical data and were
tuned with identical seeds, identical trial count and identical validation sample.

## What this changes and what it does not

- The **headline comparison is unchanged**: ΔAUPRC between the two ensemble
  probabilities over the **full** test set, with the drive-clustered paired
  bootstrap (B=1000) described in the pre-registration.
- The sign-consistency test is evaluated on the subsample defined above, not the
  full split. It is a robustness check on the seed dimension; the effect size
  and its interval come from the full split.
- The ±0.003 ΔAUPRC margin still applies to the full-split ensemble difference.

## Known limitation, declared in advance

Row groups are a fixed, order-preserving 7.59% sample, not a random one. If the
test parquet is time-ordered, the sample is time-stratified but not uniform.
Failure prevalence on the subsample is reported next to the full-test
prevalence; the observed values are 0.044231% (subsample) and 0.046766% (full
test), a −5.4% relative deviation, so no caveat is triggered.

No switching to a different sampling rule after seeing arm B. If this rule
proves unusable for a reason that is independent of the result (e.g. the
subsample turns out to contain zero positives), the reason and the replacement
rule will be recorded here as Amendment 02 **before** any arm B per-seed number
is computed.
