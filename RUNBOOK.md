# RUNBOOK — full re-run after the model fixes

Two model changes are in the working tree and **not** yet reflected in
`results/`:

1. `07_train_lstm_v2.py` — output head is now `Softplus`, so RUL can no
   longer be negative (previously 10,747 negative predictions, min −4,475.9 d).
2. `08_hybrid_fusion.py` — the meta-learner and the weighted blend now share
   one horizon-bounded RUL encoding, and strategy selection is fixed.

Until these are re-run, every number in `results/` describes the *old* model.

**Stage 6 (LightGBM) does NOT need retraining.** It is the strongest
component and is unaffected by either change. That saves hours.

---

## Step 0 — snapshot the current (old-model) results

So you can compare old vs new and report the delta honestly.

```powershell
Copy-Item -Recurse results results_before_rerun
```

## Step 1 — Stage 7: retrain the LSTM  ← the long pole, budget hours

```powershell
.\.venv\Scripts\python.exe 07_train_lstm_v2.py `
  --train data/processed/features_train.parquet `
  --val   data/processed/features_val.parquet `
  --test  data/processed/features_test.parquet `
  --out_dir models results
```

**Acceptance check**

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_csv('results/lstm_metrics.csv'))"
```

- `mae` should be **lower than 163.67**.
- `eval_window_days` should read `full` (a new column; this run is uncapped).
- `results/lstm_metrics_history.csv` should have one appended row.
- `models/lstm_config.json` should now contain `head_activation: softplus`
  and a `train_args` block.

**Then verify no negative RUL survived:**

```powershell
.\.venv\Scripts\python.exe -c "import duckdb; c=duckdb.connect(); print(c.execute(\"select count(*) filter(where lstm_rul_pred<0), min(lstm_rul_pred), max(lstm_rul_pred) from read_csv_auto('results/lstm_full_test_preds.csv')\").fetchall())"
```

Expect the negative count to be **0**. If it is not, the head fix did not
take effect — stop and check before continuing.

## Step 2 — Stage 8: re-run the fusion

```powershell
.\.venv\Scripts\python.exe 08_hybrid_fusion.py `
  --lgbm_val  results/lgbm_val_probs.csv `
  --lgbm_test results/lgbm_test_probs.csv `
  --lstm_val  results/lstm_full_val_preds.csv `
  --lstm_test results/lstm_full_test_preds.csv `
  --target failure_within_14d `
  --horizon_days 14 `
  --out_dir models results
```

**Acceptance check** — read the `[SELECT]` block it prints. It scores every
candidate on the *same* held-out half:

```
[SELECT] candidate AUPRC on the held-out selection half (N positives / M rows):
[SELECT]   lgbm_only    0.xxxxx
[SELECT]   meta         0.xxxxx
[SELECT]   weighted     0.xxxxx
[CHOSEN] strategy = <winner>
```

- If it prints `lgbm_only`, that is a **correct and defensible result**: the
  fusion did not beat its own component, so it was not shipped. Do not
  override this to make the system look more complex.
- `models/hybrid_config.json` should now contain `rul_encoding:
  horizon_bounded` and a `candidate_val_auprc` block.

## Step 3 — Stage 9: re-calibrate the tier cutoffs

The cutoffs are quantile-based, so new scores mean new cutoffs.

```powershell
.\.venv\Scripts\python.exe 09_risk_tier_calibration.py `
  --val_scores  results/hybrid_val_scores.csv `
  --test_scores results/hybrid_test_scores.csv `
  --target target `
  --high_target_recall 0.5 `
  --medium_target_recall 0.8 `
  --out_dir results
```

## Step 4 — Stage 10: evaluation

```powershell
.\.venv\Scripts\python.exe 10_evaluation.py `
  --train_resampled   data/processed/train_resampled.parquet `
  --train_full        data/processed/features_train.parquet `
  --val               data/processed/features_val.parquet `
  --test              data/processed/features_test.parquet `
  --cost_matrix       data/processed/cost_matrix.json `
  --lgbm_metrics      results/lgbm_metrics.csv `
  --lgbm_test_probs   results/lgbm_test_probs.csv `
  --hybrid_metrics    results/hybrid_metrics.csv `
  --hybrid_test_scores results/hybrid_test_scores.csv `
  --target failure_within_14d `
  --out_dir results
```

## Step 5 - Stage 11: SHAP (LightGBM only - it explains the LightGBM component)

`--test` is the **feature parquet**, not the scores CSV. The script samples
whole row groups with pyarrow, so passing a CSV fails with
"Parquet magic bytes not found in footer". `--risk_tiers` IS the Stage 9 CSV.

```powershell
.\.venv\Scripts\python.exe 11_shap_explainability.py `
  --test data/processed/features_test.parquet `
  --lgbm_model models/lgbm_model.txt `
  --lgbm_feature_cols models/lgbm_feature_cols.json `
  --risk_tiers results/risk_tiers_test.csv `
  --n_samples 3000 `
  --max_at_risk_rows 50000 `
  --out_dir results
```

## Step 6 — Stage 12: mitigation mapping

Must use the **at-risk** SHAP file or coverage collapses from 35,647 records
to roughly 113.

```powershell
.\.venv\Scripts\python.exe 12_mitigation_mapping.py `
  --risk_tiers results/risk_tiers_test.csv `
  --shap_values results/shap_values_at_risk.parquet `
  --out results/mitigation_mapping.json
```

## Step 7 — rebuild the serving database

```powershell
.\.venv\Scripts\python.exe build_dashboard_db.py --results_dir results
```

Then confirm it is not stale:

```powershell
.\.venv\Scripts\python.exe -c "import importlib.util,pathlib; s=importlib.util.spec_from_file_location('d','13_dashboard.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.serving_status('results'))"
```

`stale` must be `False`.

## Step 8 — regenerate the paper tables

```powershell
.\.venv\Scripts\python.exe -u 14b_paper_evaluation.py --results_dir results --boot 200 --boot-row 25
```

Takes roughly 20 minutes (it scans the 13 GB export twice). Writes to
`results/paper_tables/`.

## Step 9 — verify the dashboard end to end

```powershell
.\.venv\Scripts\streamlit.exe run 13_dashboard.py --server.headless true --server.port 8501 -- --results_dir results
```

Check, in this order:

1. Fleet overview tier cards match `drive_status` counts.
2. Operational queue shows `Age (days)` and 6-decimal risk scores.
3. Drive explorer: pick a HIGH drive, confirm the RUL caption no longer says
   the coefficient is "negligible" if the retrained fusion changed its weight.
4. Agent workspace: run a simulation, confirm the ticket's priority matches
   the tier and that `root_cause` is present.
5. Browser console must be free of errors.

---

## What to compare against `results_before_rerun`

| Quantity | Old (broken) | Expected direction |
|---|---|---|
| Negative RUL predictions | 10,747 | **0** |
| LSTM MAE (full range) | 163.67 | lower |
| Tier changes from removing RUL | 11.9% of queue | much lower |
| Risk delta for RUL<0 group | −0.1176 | ≈ 0 or positive |
| Mean RUL, failed vs not-failed | 54.9 vs 40.3 d | failed should be **lower** |

The last row is the real physics test. Failing drives must have *shorter*
predicted remaining life than surviving drives. If they still do not, the RUL
branch is not usable and you should ship `lgbm_only` and report the negative
result.
