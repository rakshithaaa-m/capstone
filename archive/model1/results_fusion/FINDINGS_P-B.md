# Findings: P-B (tuned XGBoost classifier), 2026-10-27

P-C was ruled out (D-04) before any P-B output was opened.

## The model

`22_train_xgb_clf.py`. Same training rows, 212 features and target as LightGBM
(`train_resampled.parquet`, 3,440,506 rows, 313,002 positives). 30 Optuna trials,
TPE seed 42, maximising AUPRC on the 5,000,000-row validation prefix
(3,618 positives).

| | value |
|---|---|
| best tuning-sample AUPRC | 0.11100 |
| tuning-sample ROC-AUC | 0.90548 |
| chosen | 717 trees, lr 0.058, depth 5, min_child_weight 8.4, subsample 0.83, colsample 0.61 |

This replaces the 0.0020-AUPRC XGBoost row in `results/baseline_comparison.csv`.
That row was a configuration fault, not a property of XGBoost.

## §6 screen (pre-registered): **PASS**, by a hair

Stage 16 protocol, unchanged. 308,202 validation drives, 0 without a partner
score, 1,353 selection-half positives, B = 400. Log: `stacking_pb.log`.

| combiner | AUPRC | vs LightGBM (0.60526) |
|---|---|---|
| B. logistic [prob, risk] | 0.60553 | **+0.00027** (best) |
| C. GBDT stacker [prob, risk] | 0.56008 | −0.04519 |
| D. GBDT + engineered | 0.56897 | −0.03629 |
| E. GBDT prob-only (control) | 0.53612 | −0.06914 |

Best combiner vs LightGBM: **+0.00027, 95% CI [+0.00003, +0.00051]**, wins in
98.8% of resamples. The lower bound is above 0, so under §6 P-B goes on to §7.
The gain is about a tenth of the ±0.003 materiality margin §7 applies.

## Conditional test (reported, not a gate)

| | LSTM | P-A | P-B |
|---|---|---|---|
| marginal AUROC of partner | 0.5264 | 0.7106 | 0.9372 |
| positive-weighted mean within-decile AUROC | 0.5202 | 0.7017 | 0.8816 |
| top decile (1,178 positives) | 0.5230 | 0.7316 | 0.9373 |
| deciles 1–9 | 0.45–0.55 | exactly 0.5000 | 0.38–0.66, 99 positives in all |

Almost all of P-B's conditional signal is in LightGBM's top decile. There it
re-ranks drives that LightGBM already flags. Below the top decile it is
close to chance. Two strong tree models trained on the same rows mostly agree,
so the room for fusion is small, and the screen result reflects that.

## §7 test confirmation

Partner: the five-seed P-B ensemble (`22_train_xgb_clf.py --ensemble`, LightGBM's
seeds 42, 7, 13, 99, 23). Combiner B was refitted on all 308,202 validation
drives (2,770 ever-fail) with no re-selection:

    fused = sigmoid(−1.962 + 8.335 · lgbm_prob + 0.370 · xgb_prob)

`24_pb_confirm.py`, log `pb_confirm.log`, output `pb_confirm.json`. Join check:
144,739,479 rows, 67,689 positives, both matching `results/lgbm_metrics.csv`.

| test split | LightGBM | fused | Δ |
|---|---|---|---|
| **row-level AUPRC (primary)** | 0.083938 | 0.084340 | **+0.000402** (+0.48%) |
| row-level ROC-AUC | 0.888671 | 0.888851 | +0.000180 |
| drive-level AUPRC (352,383 drives, 5,254 ever-fail) | 0.59796 | 0.59852 | +0.00056 |

Paired drive-clustered bootstrap, B = 1000: **Δ 95% CI [+0.000232, +0.000573]**.
The fused score was ahead in 100% of resamples.

**Verdict (pre-registered): REAL BUT IMMATERIAL.** The CI lower bound is above 0,
but Δ is about 1/7 of the +0.003 margin. LightGBM stays. Stages 9–13 are not
re-run.

The improvement is genuine but operationally negligible. It equals about
+0.48% relative AUPRC, roughly the same failures flagged at the same alert
budget. This is the best of the three partners tested (LSTM, P-A and P-B), and
it is consistent with the conditional test: a second boosted model on the same
rows and features mostly agrees with LightGBM.

## Fusion study closed

None of the partners clears the materiality margin. With P-C ruled out (D-04),
Amendment 03 is complete. The delivered predictor is LightGBM alone.
