# Amendment 02 — per-seed statistic must use the full test split

**Written:** 2026-10-24, after both arms finished, before the corrected per-seed
measurement was run.
**Supersedes:** the sampling rule in `AMENDMENT_01.md` sections "The rule" and
"Known limitation".

## What Amendment 01 got wrong

Amendment 01 assumed that "every 14th row group" of
`features_test.parquet` yields a ~7.6% **row** sample. It does not.

The file is sorted by `serial_number`, and its row groups are **drive-disjoint**.
Verified directly:

| row group | rows | distinct drives | overlap with previous |
|---|---|---|---|
| 0 | 280,655 | 753 | — |
| 1 | 178,077 | 453 | 0 |
| 14 | 65,683 | 149 | 0 |
| 15 | 29,467 | 65 | 0 |
| 28 | 153,046 | 347 | 0 |

So the rule selects about 1/14 of the **drives**, each with all of its rows. That
is a drive-holdout sample, not a row sample. It answers "how well does the model
rank rows on drives it was not trained on", which is a genuinely interesting
question — but it is not the pre-registered one, which is row-level AUPRC on the
full test split.

## Why this matters numerically

The two quantities disagree, and not slightly:

| quantity | arm A | arm B | delta (A−B) |
|---|---|---|---|
| full test, ensemble (pre-registered headline) | 0.083938 | 0.084076 | **−0.000138** |
| drive-disjoint sample, ensemble | 0.085645 | 0.078981 | +0.006664 |
| drive-disjoint sample, per seed | 0.0815–0.0874 | 0.0684–0.0837 | A wins **5/5** |

A 5/5 seed sweep with a +0.009 mean on one sample and a −0.0001 difference on
the other cannot both be estimates of the same parameter. The drive-disjoint
sample is a different parameter.

## Disclosure of sequence

These drive-disjoint numbers were computed **before** this amendment was
written. The reason for the correction is structural — drive-disjoint row groups
are a property of how the file was written — and does not depend on the values
observed. But the sequence is stated plainly: the corrected statistic is
specified after seeing numbers that the corrected statistic is not the same as.

Consequently the corrected per-seed result is reported **alongside** the
drive-disjoint result, not in place of it. Either may be checked by anyone with
the snapshot; neither is being selected because it is convenient.

## The corrected rule (implemented in `17c_per_seed_full.py`)

- split: full `features_test.parquet`, all 144,739,479 rows;
- one streaming pass; all five boosters predict the same chunk before it is
  released, so the parquet is read once and every seed sees identical rows;
- statistic: AUPRC and ROC-AUC per seed, plus the mean of the five seed
  probabilities (which must reproduce `lgbm_metrics.csv` — a built-in check);
- run for both arms; this is the per-seed basis for the ≥4/5 sign rule.

Cost is roughly 60 minutes per arm at the observed prediction rate.

## The drive-disjoint result is retained as a diagnostic

`17b_per_seed_auprc.py` stays in the repository and its output stays in
`results_ablation/per_seed_*_driveholdout.csv`. It is reported as a portability
diagnostic: on drives absent from training, the 212-feature arm is clearly
better, which is consistent with the hypothesis that infrastructure identifiers
do not transfer to new drives. That claim, if made, must be supported by
`17c`-based cohort analysis on the full split, not by this sample alone.
