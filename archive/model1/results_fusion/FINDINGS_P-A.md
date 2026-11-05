# Findings — P-A (XGBoost AFT survival model), 2026-10-27

## §6 screen (pre-registered): **SCREEN FAIL**

Stage 16 protocol, unchanged. Validation, drive level, 50/50 fit/select split,
1,353 selection-half positives, B = 400.

| combiner | AUPRC | vs LightGBM (0.60526) |
|---|---|---|
| B. logistic [prob, risk] | 0.54541 | −0.05985 |
| C. GBDT stacker [prob, risk] | 0.54983 | −0.05544 |
| D. GBDT + engineered | 0.56582 | **−0.03945** (best) |
| E. GBDT prob-only (control) | 0.53612 | −0.06914 |

Best combiner vs LightGBM: **−0.03929, 95% CI [−0.05025, −0.02793]**, wins 0.0%.
Under §6 the candidate stops here. P-A does not go to §7.

## Conditional test (reported, not a gate)

| | LSTM | P-A |
|---|---|---|
| marginal AUROC of partner risk | 0.5264 | 0.7106 |
| positive-weighted mean within-decile AUROC | 0.5202 | 0.7017 |
| top decile (1,178 positives) | 0.5230 | 0.7316 |
| deciles 1–9 | 0.45–0.55 | exactly 0.5000 |

## Diagnostic (exploratory, run after the screen)

**1. The horizon encoding discards P-A outside the top decile.** The risk is
`clip((14 − rul) / 14, 0, 1)`, which is 0 whenever predicted RUL ≥ 14 days.
Below the top decile P-A always predicts ≥ 14 days, so the risk is constant and
the AUROC is exactly 0.5. Spec D also sees raw `rul_log` and `rul_frac`. It
gains +0.030 over the prob-only control E, but the GBDT stacker itself is weaker
than LightGBM (E alone: −0.069).

**2. Stage 16 scores failing drives on their failure day.** For the 2,770
ever-failing validation drives, the latest-row RUL is:

| latest-row `rul_days` | drives |
|---|---|
| 0 (the failure day) | **2,516 (91%)** |
| 1–14 | 146 |
| < 0 (post-failure rows) | 108 |
| > 14 | 0 |

`failure_within_14d` excludes the failure day (rul 1–14), so LightGBM is trained
to give that day label 0. A survival model, by contrast, should say "≈0 days
left" there. The 0.73 top-decile AUROC is therefore most plausibly P-A
recognising the failure day, which has no operational value because the warning
comes too late. It is not early-warning information LightGBM lacks.

**Consequence for the remaining candidates.** The Stage-16 drive-level design
compares partners mostly on failure-day rows. That affects how its conditional
test should be read for P-B and P-C as well. The pre-registered gate is
unchanged. A fairer early-warning comparison (for example, scoring each drive at
a fixed lead time before failure) would be a new, separately registered
analysis, not a change to Amendment 03.

## §7b RUL output selection

Test split, failing-drive rows with `rul_days ≥ 0`: 1,131,717 rows, 5,341 drives.
No rows were missing from either prediction file. Paired drive-clustered
bootstrap, B = 1000, ΔMAE = MAE(LSTM) − MAE(P-A).

| window | LSTM MAE | P-A MAE | ΔMAE | 95% CI |
|---|---|---|---|---|
| **rul ≤ 90 (primary)** | **42.11** | 1020.96 | −978.85 | [−1014.17, −940.21] |
| rul ≤ 30 | 61.46 | 842.91 | −781.45 | [−815.52, −744.40] |
| all failing rows | 88.32 | 1345.03 | −1256.71 | [−1298.20, −1215.67] |

Harrell's C, first test row per drive (D-01), 352,197 drives (5,341 failing):
**LSTM 0.5380, P-A 0.7491.**

**Verdict (pre-registered): KEEP.** The LSTM stays as the displayed RUL.

**Interpretation.** The two metrics disagree, and each exposes a different defect:

- The LSTM's low MAE is partly structural. It was trained almost entirely on
  drives that fail, so it predicts short lives, and an MAE computed only on
  drives that did fail rewards that. Its C-index of 0.538 shows it barely ranks
  which drive fails sooner.
- P-A ranks well (C = 0.749) but is badly miscalibrated in absolute days. About
  85% of its training rows are censored, so it learned that drives typically
  last years, and its day counts for failing drives are off by about 1,000 days.

Neither model supports a reliable "N days left" number. The displayed RUL stays
marked approximate, as before.
