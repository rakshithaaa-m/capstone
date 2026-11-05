# DiskGuard: explainable, agent-assisted hard-drive failure management

An end-to-end system on the Backblaze SMART dataset. For every drive in a
363,000-drive fleet it answers four questions:

| question | component | output |
|---|---|---|
| **Will this drive fail soon?** | LightGBM, 5-seed ensemble, 212 features | risk score → HIGH / MEDIUM / LOW tier, daily alert queue |
| **How urgent is it, relative to the fleet?** | XGBoost survival model (AFT) | fleet-urgency ranking within each tier |
| **Why?** | Exact TreeSHAP on LightGBM | top driver → plain-language root cause |
| **What should we do?** | Rule mapping + tool-using LLM agent | maintenance plan; a human approves every action |

Everything is shown in a Streamlit control room (`13_dashboard.py`), which also
runs a monthly drift monitor that says when the model needs retraining.

---

## Headline results (held-out test, 1 Jan – 16 Jun 2026)

The positive label is "fails within the next 1–14 days". 0.048% of drive-days
and 0.67% of drives are positive. Full numbers: `results_model2/`.

| metric | value |
|---|---|
| Row-level AUPRC (57.6M drive-days), 95% CI | **0.0779** [0.0701, 0.0859], ~162× the base rate |
| Row-level ROC-AUC | 0.886 |
| Drive-level AUPRC (363,405 drives, 2,451 fail), 95% CI | **0.459** [0.438, 0.479] |
| Drive-level ROC-AUC, 95% CI | 0.908 [0.900, 0.916] |
| Precision / recall at the alert threshold (0.73) | 3.6% / 57.4% |
| HIGH tier: failing drives caught (drive level) | 67.3% of failures, at 34.1% precision (4,841 drives, 1.3% of the fleet) |
| HIGH tier: failing drive-days caught (row level) | 57.9%; HIGH + MEDIUM: 79.6% |
| Failing drives at or above the threshold 30 / 14 / 7 / 1 days before failure | 35% / 44% / 48% / 54% |
| Daily alert queue (top 2,074 drives/day) | 67.2% of failing drives caught at a fixed daily workload |
| Fleet-urgency ranking, C-index | 0.841 |

The threshold and tier cutoffs (HIGH 50%, MEDIUM 80% recall) were fitted on
validation, not on test. Scores are **ranking scores** (Brier 0.011, ECE 0.037),
so do not read them as probabilities.

Baseline on the same test period: logistic regression AUPRC 0.021, ROC-AUC 0.820.
Single seeds reach 0.0765–0.0782; the ensemble 0.0779.

**Evaluation shortcuts inflate the same scores.** Re-scored with balanced 1:1
undersampling, the same predictions give AUPRC 0.917 and 98.7% precision at the
threshold (`results_model2/inflation.csv`); ROC-AUC does not move.

**Identifier ablation (pre-registered).** Adding the vault, pod, cluster, slot
and legacy-format identifiers raises AUPRC by 0.0018, 95% CI [0.0003, 0.0034],
in all 5 seeds (verdict: reversal, `results_model2/infra_ablation.json`). The
gain is small and the identifiers describe one operator's buildings, so they
stay excluded to keep the model portable.

**Why urgency is a ranking, not a countdown.** The survival model ranks drives
well but its day counts are far off, so the system shows "expected to fail
sooner than X% of the fleet". Urgency orders work within a tier and never
changes a drive's tier.

---

## Data and splits

Backblaze daily SMART snapshots, April 2013 to June 2026, split by time with a
gap before each held-out period:

| split | dates | drive-days |
|---|---|---|
| train | 2013-04-10 → 2025-06-16 | 616.4M (undersampled to 4.27M; 28,672 failing drives) |
| validation | 2025-07-01 → 2025-12-17 | 56.0M (2,099 failing drives) |
| test | 2026-01-01 → 2026-06-16 | 57.6M (2,451 failing drives) |

Features are built on each drive's full history before splitting, so a held-out
drive's rolling windows start from its real earlier readings. Training uses only
the training period; the threshold, tiers and alert budget are fitted on
validation; the test period was scored once.

---

## Repository layout

**Pipeline**

| stage | script | role |
|---|---|---|
| 1–2 | `01_acquire_data.py`, `02_clean_data.py` | download and clean Backblaze CSVs |
| 3–5 | `03_build_targets_and_features.py`, `04_temporal_split.py`, `05_imbalance_strategy.py` | targets, 212 features, time split, undersampling |
| 6 | `06_train_lightgbm.py` | Optuna hyperparameter search on 2013–2024 data, inside the training period (settings in `models/lgbm_best_params.json`) |
| | `28_fresh_2026.py` | builds the 2026 features from Backblaze 2026 H1 |
| | `29_retrain_2026.py` | builds the training set to June 2025 and trains the 5-seed ensemble → `models_retrain2026/A`, fits threshold and tiers on validation → `results_retrain2026/` |
| | `31_deploy_2026.py` | scores 2026, tiers, urgency, SHAP actions, dashboard DB → `results_deploy2026/` |
| | `09_risk_tier_calibration.py`, `11_shap_explainability.py`, `12_mitigation_mapping.py` | recall-calibrated tiers, SHAP, root cause → tier-aware action |
| | `13_dashboard.py`, `14_agentic_mitigation.py`, `build_dashboard_db.py` | dashboard, agent, serving database |

**Evaluation and studies**

| script | role |
|---|---|
| `35_model2_eval.py` | full test evaluation (headline + CI, inflation, tiers, lead time, calibration, seeds, logistic regression, urgency, SHAP, agent policy) → `results_model2/` |
| `36_model2_infra_ablation.py` | pre-registered identifier ablation → `results_model2/infra_ablation.json` |
| `17_ablation_compare.py`, `26_row_auprc_ci.py` | exact drive-clustered bootstrap for AUPRC and paired differences |
| `18_leadtime_curve.py`, `19_calibration.py`, `27_prevalence_inflation.py` | lead time, calibration, evaluation-shortcut table |
| `20_train_xgb_aft.py`, `23_pa_urgency.py` | survival model (`models/urgency/`) and fleet-urgency ranking |
| `25_agent_eval.py` | agent evaluation: policy audit, red team, live harness |
| `32_alert_policies.py` | pre-registered alert-policy study (daily top-k adopted) → `results_alerts2026/` |
| `33_drift_monitor.py` | monthly drift monitor (alert volume, score PSI, 3-month AUPRC) with a retrain flag → dashboard panel |
| `34_tier_sizing.py` | pre-registered MEDIUM-tier sizing study (kept the 80%-recall cutoff) → `results_tiers2026/` |

**Folders.**
- `models/` holds shared settings (features, hyperparameters, seeds) and the survival model in `models/urgency/`.
- `models_retrain2026/A` holds the shipped LightGBM ensemble.
- `results_retrain2026/`, `results_deploy2026/`, `results_model2/`, `results_alerts2026/`, `results_tiers2026/` hold the records; multi-GB exports are gitignored.
- `archive/model1/` holds an earlier model (trained to March 2024) and its studies, kept for provenance; nothing in the pipeline above depends on its results.
- `data/` holds the parquet data (gitignored).

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Tested with Python 3.12, LightGBM 4.7, XGBoost 3.4, DuckDB 1.5 and
Streamlit 1.62, on a 16 GB laptop. For the exact tested versions, run
`pip install -r requirements-lock.txt` instead.

**Dashboard.**

```bash
python 31_deploy_2026.py all
streamlit run 13_dashboard.py
```

**Agent, one drive.** Dry run is deterministic and needs no API key. Pick a drive
from the dashboard's queue:

```bash
python 14_agentic_mitigation.py --serial_number <SERIAL> --dry_run
```

For live mode, set `GROQ_API_KEY` (default model `openai/gpt-oss-120b`) or
`ANTHROPIC_API_KEY`. The agent can check spare parts and technician
availability and draft a maintenance ticket. It never executes an action: an
operator approves every ticket. The model's tier is an immutable input, and the
agent cannot raise or lower it. This is enforced in code, not only in the prompt:
- the ticket's identity, tier and root cause are attached by the server;
- the parts lookup is locked to the drive's own model;
- a non-HIGH drive cannot be scheduled as urgent.

The rule policy passes all 13 invariants on all 64,743 evaluated drives
(`results_model2/agent_eval/`). In the live run every invariant held, 4 of 4
prompt injections were resisted and 20 of 20 drives got a ticket
(`results_deploy2026/agent_eval/`).

**Evaluation.** `python 35_model2_eval.py all` (about 3 hours) and
`python 36_model2_infra_ablation.py all` (about 2 hours).

**Full rebuild from raw data.** Stages 1–5 (download, cleaning, features, split)
are documented in `archive/model1/00_setup.md`; then run `28_fresh_2026.py`,
`29_retrain_2026.py` and `31_deploy_2026.py`. Stage 3 takes hours.

---

## Limitations

- **Scores rank; they are not probabilities.** Use the tiers or the daily queue,
  not raw scores, for decisions.
- **SMART has a ceiling.** 16% of failing drives are still in the LOW tier. Many
  failures give no SMART warning, a known limit of this data source.
- **No reliable day count.** Urgency is a ranking; see above.
- **The model ages.** The drift monitor flagged RETRAIN in April 2026 on alert
  volume (ranking quality still at 1.03–1.07× validation); retrain with
  `29_retrain_2026.py` on the newest labelled data when it does.
- **Compute-bound choices.** Undersampled training set; Optuna on a
  validation prefix.
