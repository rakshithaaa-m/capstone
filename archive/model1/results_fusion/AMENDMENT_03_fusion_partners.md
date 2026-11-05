# Amendment 03 — fusion-partner study

**Written:** 2026-10-27, before any partner model in this protocol is trained.
**Author:** Hemanth (UE23AM441A)
**Status:** DRAFT. Becomes FROZEN when the author adds the freeze line in §11.
After that, do not edit; deviations go in `results_fusion/DEVIATIONS.md`.

---

## 1. Why this run exists

The stacking study (`16_stacking_study.py`, outputs `results/stacking_study*.csv`)
found that no combiner of LightGBM + LSTM-RUL beats LightGBM alone:

| quantity | value |
|---|---|
| LightGBM alone, drive-level AUPRC, selection half | 0.60526 |
| best combiner (logistic [prob, risk]) | 0.58878 |
| gap, drive-clustered bootstrap, B = 400 | −0.01648, 95% CI [−0.02457, −0.00915], wins 0.0% |
| RUL AUROC within LightGBM-probability deciles | median 0.4928, 4/10 bands > 0.5 |

That null is **scoped to one estimator**: a ≈32.7k-parameter LSTM with test
MAE ≈ 96 days. The original pre-registration (§8) already flagged this:
*"Any 'RUL adds nothing' claim is scoped to this estimator unless the
LightGBM-regressor control also adds nothing."* The control described in paper
§3.7 was never run.

This amendment asks one question: **is the fusion null a property of the LSTM,
or of the information available to any partner model on these features?**

## 2. What is already known (disclosure)

Before this amendment was written the author had seen: every number in §1; the
LightGBM test metrics (row AUPRC 0.08394, drive AUPRC 0.598); the identifier
ablation (EQUIVALENCE); and the per-band positive counts of the LSTM conditional
test (1,178 of 1,353 selection-half positives fall in the top probability decile).

No partner model listed in §3 has been trained or scored. Nothing about their
performance is known.

## 3. Candidates (fixed list, run in this order)

| id | model | output | `--partner_kind` | compute |
|---|---|---|---|---|
| **P-A** | XGBoost survival, `objective="survival:aft"`, normal distribution | predicted time to failure, days | `rul` | CPU |
| **P-B** | XGBoost binary classifier, tuned (replaces the 0.0020 baseline) | P(failure within 14 d) | `prob` | CPU |
| **P-C** | TCN classifier on raw daily SMART sequences, lookback 30 d | P(failure within 14 d) | `prob` | GPU (6 GB) |

Common to all three:

- **Inputs.** P-A and P-B use the 212-feature set (`models/lgbm_feature_cols.json`).
  P-C uses raw SMART columns only, no engineered features. That is its point:
  it tests whether timing information exists that the engineered features miss.
- **Training rows.** P-A and P-B train on `data/processed/train_resampled.parquet`,
  the same rows as the LightGBM reference. P-C trains on a drive-level sample of
  `features_train.parquet`: **every** drive with a failure, plus non-failing drives
  sampled at a rate fixed before training (seed 20261027). The rate is the
  largest that fits in 6 GB and is recorded in the run log. Sampling is by
  drive, never by row.
- **P-A labels.** Time is `rul_days + 1`, so the failure day is 1 (AFT works on
  log time). Failing rows: `lower = upper = rul_days + 1`. Censored rows:
  `lower = (last day the drive is observed in the same split − row date) + 1`,
  `upper = +inf`. The last-observed date comes from the full split file, not
  from the undersampled training file. This is **not** the LSTM's bound. The LSTM
  uses `days_since_install`, which bounds total lifetime, not remaining life
  (see §10).
- **Tuning.** Optuna, `n_trials = 30`, tuning seed 42, on a fixed 5,000,000-row
  prefix of the validation split, the same sampling rule as the reference.
  Objective: validation AUPRC (P-B, P-C), or validation AFT negative
  log-likelihood (P-A). P-A has early stopping (50 rounds) inside each trial.
  P-C has a fixed budget of 30 epochs with early stopping (patience 5) on
  validation AUPRC.
- **Seeds.** One training seed (42) per candidate. A candidate that passes §6
  gets the full five-seed ensemble `{42, 7, 13, 99, 23}` before the test
  confirmation in §7.
- **LightGBM is untouched.** `models/` and `results/` are read-only inputs.

**P-C is optional.** Whether it runs depends on time remaining, not on the
P-A/P-B results. The decision, its date and its reason go in `DEVIATIONS.md`
**before** P-B's results are opened.

**Excluded and why.** Informer: 30–90 day sequences are too short to need sparse
attention. GNN over rack topology: the identifier ablation already found rack and
pod position immaterial (Δ −0.000138, EQUIVALENCE), and Backblaze has no
vibration, power or rack-temperature telemetry to build edges from. CatBoost:
not installed, and it would add a fourth near-duplicate gradient-boosting model.

## 4. Hypotheses (per candidate)

- **H1.** The best combiner of LightGBM + partner reaches higher drive-level
  AUPRC than LightGBM alone.
- **H0.** The partner adds no ranking information beyond the LightGBM probability.

## 5. Evaluation protocol: Stage 16, unchanged

Each candidate writes `results_fusion/<id>_{val,test}_preds.csv` with columns
`serial_number, date, <score column>`, every row of the split. P-A:
`pa_xgb_aft_*_preds.csv`, column `aft_rul_pred`, in days. Its script is
`20_train_xgb_aft.py`. It is then run through
`16_stacking_study.py --partner_csv … --partner_col … --partner_kind … --boot 400`.
The code was generalised for this amendment. Its defaults must first reproduce
`results/stacking_study*.csv` exactly. That check is logged in §10 before any
partner run.

Inherited without change:

- one row per drive (latest-known assessment), label = `MAX(target)` over the drive;
- input sorted by `serial_number`; `GroupShuffleSplit` seed 20260926, 50/50
  drive-level fit/select split of the **validation** split;
- combiners A–E as in the script; spec D's engineered columns depend on
  `partner_kind`, and are fixed in code before this amendment;
- paired drive-clustered bootstrap on the best combiner, B = 400.

A partner must score every validation drive. The script reports drives with no
partner score. If any are missing, that is a deviation and is logged.

## 6. Screening rule (validation, fixed in advance)

The conditional test runs first. Its statistic is the **positive-weighted mean**
of within-decile AUROC, not the median. The median gives equal weight to nine
deciles holding 8–38 positives each. The LSTM reference value is 0.5202
(see §10).

| outcome | condition | action |
|---|---|---|
| **SCREEN PASS** | best combiner's bootstrap CI lower bound > 0 | proceed to §7 |
| **SCREEN FAIL** | CI includes 0 or lies below it | report the null; candidate stops |

The positive-weighted conditional AUROC is reported for every candidate but is
**not** a gate. It explains a pass or a fail; it does not decide one.

**Known optimism.** The combiner is selected and bootstrapped on the same selection
half. A pass is therefore slightly optimistic, which is why §7 exists.

## 7. Test confirmation (only for candidates that pass §6)

1. Five-seed ensemble of the partner, as for LightGBM.
2. The **single** combiner that won in §6 is refitted on the **full** validation
   split. No re-selection.
3. Evaluate on the full test split: row-level AUPRC (primary, comparable to
   0.08394) and drive-level AUPRC (secondary).
4. Paired drive-clustered bootstrap, B = 1000, on ΔAUPRC = fused − LightGBM.

| outcome | condition | action |
|---|---|---|
| **SHIP** | CI lower bound > 0 **and** Δ ≥ +0.003 | fused score replaces LightGBM; re-run Stages 9–13 on it |
| **REAL BUT IMMATERIAL** | CI lower bound > 0, Δ < +0.003 | report it; keep LightGBM |
| **NULL** | CI includes 0 | report it; keep LightGBM |
| **HARM** | CI upper bound < 0 | report it; keep LightGBM |

The ±0.003 margin is the same one the identifier ablation used.

## 7b. RUL output selection (P-A vs LSTM, independent of fusion)

Fusion asks whether RUL improves the failure prediction. This section asks a
separate question: **which model supplies the days-left estimate shown in the
dashboard and passed to the agent?** It runs whether P-A passes §6 or not.

- **Rows.** Test split, rows of drives that fail (`is_censored = 0`), matched by
  `serial_number, date` across both prediction files. Rows either model lacks
  are reported and excluded from both.
- **Primary metric.** MAE in days on rows with true `rul_days ≤ 90`, the range
  where a days-left number changes what an engineer does.
- **Secondary metrics.** MAE over all failing-drive rows, comparable to the
  LSTM's 96.3; MAE on `rul_days ≤ 30`; Harrell's C-index on one row per drive
  (latest), all failing test drives plus censored drives, using the §3
  censoring bound.
- **Statistic.** Paired drive-clustered bootstrap, B = 1000, on
  ΔMAE = MAE(LSTM) − MAE(P-A).

| outcome | condition | action |
|---|---|---|
| **REPLACE** | ΔMAE CI lower bound > 0 | P-A becomes the displayed RUL |
| **KEEP** | otherwise | the LSTM stays as the displayed RUL |

Either way, the RUL stays display-only, marked approximate, unless §7 ships a
fused score.

## 8. Multiplicity

Up to three candidates × four non-trivial combiners. Only §7 can change what
ships, and §7 tests exactly one pre-selected combiner per candidate on data that
played no part in selecting it. Results of every candidate are reported,
including failures. None are dropped.

## 9. Pre-declared limitations

- **P-C is compute-bounded.** It trains on a drive sample on a 6 GB laptop GPU.
  To separate "deep models do not help" from "this deep model was undertrained",
  report a learning curve at 25 / 50 / 100% of the sample. If validation AUPRC is
  still rising at 100%, the P-C result is stated as compute-limited, not general.
- P-A's concordance-based tuning and the drive-level AUPRC screen measure
  different things. A survival model can rank time-to-failure well and still add
  no fusion value. That is the question being asked, not a flaw.
- The Stage 16 protocol evaluates on a **validation** fit/select split, not on
  test. Paper §3.3 currently says the opposite and must be corrected regardless
  of the outcome here.

## 10. Log (append-only)

- 2026-10-27: `16_stacking_study.py` generalised (`--partner_csv`,
  `--partner_col`, `--partner_kind`). Regression check: defaults with
  `--boot 400` reproduce `results/stacking_study.csv`, `_conditional.csv` and
  `_bootstrap.csv` **byte-identically**. 308,202 validation drives, 0 without a
  partner score, 1,353 selection-half positives.
- 2026-10-27: LSTM reference value of the §6 statistic: positive-weighted mean
  within-decile AUROC = **0.5202** (median 0.4928, 4/10 bands > 0.5; top-decile
  band 0.5230 on 1,178 positives).
- 2026-10-27: Finding while specifying P-A. `07_train_lstm_v2.py:259` uses
  `days_since_install` (drive age) as the censored lower bound on remaining
  life. Age bounds total lifetime, not RUL. The correct bound is days
  observed alive after the row. This affects every surviving drive in the
  LSTM's training loss and may contribute to its ≈96-day MAE. P-A uses the
  correct bound. The LSTM is not retrained under this amendment. If §7b keeps
  the LSTM, this is listed as a known defect.
- 2026-10-27: Data finding from the P-A smoke test. Some failing drives keep
  reporting after their last recorded failure, up to 3,928 days later. These are
  probably returned or serial-reused drives. `rul_days < 0` on those rows:
  train_resampled 2,292 rows / 260 drives; full train 347,695 / 893; val
  17,540 / 221; test 66,926 / 246. None is a positive for `failure_within_14d`,
  so the classifier labels are unaffected. **Rule:** these rows are excluded
  from P-A training and tuning and from the §7b MAE rows. They still receive
  predictions, and they still count in every Stage-16 drive-level analysis
  exactly as before.

## 11. Freeze

_To freeze, the author adds below: "FROZEN <date> <initials>" and commits this
file before training P-A._

FROZEN 2026-10-27 H. Frozen after the full P-A run was launched and before any
of its output was read. The only P-A numbers seen before freezing are from the
200k-row smoke test, which is used solely to check that the pipeline runs.
