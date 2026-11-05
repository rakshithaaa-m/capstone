# Archived: fusion study (LSTM, P-B)

These scripts tested whether combining LightGBM with a second model improves
failure prediction. None did by a material margin, so the delivered system
uses LightGBM alone. They are kept, unchanged, as the evidence behind the
paper's one-paragraph summary of that evaluation. **They are not part of the
running system.**

| file | what it tested | result |
|---|---|---|
| `07_train_lstm_v2.py` | LSTM remaining-useful-life model | MAE 42 d on the last 90 days, but C-index 0.54 (barely ranks) |
| `16_stacking_study.py` | LightGBM + partner combiners, drive-level screen | LSTM −0.016 AUPRC, CI [−0.025, −0.009] |
| `22_train_xgb_clf.py`, `24_pb_confirm.py` | P-B: tuned XGBoost classifier as partner | test +0.0004 AUPRC, CI [+0.0002, +0.0006]: real but immaterial |
| `21_rul_compare.py` | LSTM vs survival model for "days left" | neither supports a day count |
| `14c_rul_window_eval.py`, `diagnose_stage7_scale.py` | LSTM error by window; scale diagnostics | — |
| `models_fusion/` | P-B five-seed models | — |

P-C (a TCN) was planned but never run (deviation D-04).

**Still in the main tree:**
- The survival model P-A (`20_train_xgb_aft.py`, `23_pa_urgency.py`, `models_fusion/pa_*`) supplies the dashboard's fleet-urgency ranking.
- `08_hybrid_fusion.py` (Stage 8) remains the pipeline step whose output files Stage 9 reads. It selected `lgbm_only`, so its score is exactly the LightGBM probability.
- The pre-registered records are unchanged in `results_fusion/`: Amendment 03, `FINDINGS_P-A.md`, `FINDINGS_P-B.md`, `DEVIATIONS.md`.

**Re-running.** The scripts load each other by relative path. To re-run them,
copy them back to the repository root.
