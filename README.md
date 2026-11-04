# DiskGuard: explainable, agent-assisted hard-drive failure management

An end-to-end system on the Backblaze SMART dataset. For every drive in a
352,000-drive fleet it answers four questions:

| question | component | output |
|---|---|---|
| **Will this drive fail soon?** | LightGBM, 5-seed ensemble, 212 features | risk score → HIGH / MEDIUM / LOW tier |
| **How urgent is it, relative to the fleet?** | XGBoost survival model (AFT) | fleet-urgency ranking within each tier |
| **Why?** | Exact TreeSHAP on LightGBM | top driver → plain-language root cause |
| **What should we do?** | Rule mapping + Claude tool-using agent | maintenance plan; a human approves every action |

Everything is shown in a Streamlit control room (`13_dashboard.py`).

More complex designs, an LSTM and combinations of LightGBM with a second
model, were evaluated and did not beat LightGBM alone. That work is archived
in `archive/fusion_study/`. See [Evaluated and not adopted](#evaluated-and-not-adopted).

---

## Headline results (held-out test, 1 Oct 2024 – 31 Dec 2025)

The positive label is "fails within the next 1–14 days". 0.047% of drive-days
and 1.49% of drives are positive.

| metric | value |
|---|---|
| Row-level AUPRC (144.7M drive-days) | **0.0839**, ~179× the base rate |
| Row-level ROC-AUC | 0.889 |
| Drive-level AUPRC (352,383 drives, 5,254 fail) | **0.598**, 95% CI [0.586, 0.611] |
| Drive-level ROC-AUC | 0.920, 95% CI [0.915, 0.925] |
| HIGH tier: failing drives caught (drive level) | 66.9% of failures, at 56.0% precision (6,282 drives flagged) |
| HIGH tier: failing drive-days caught (row level) | 52.3%; HIGH + MEDIUM: 75.5% |
| Failing drives already HIGH 30 / 14 / 7 / 1 days before failure | 43% / 51% / 55% / 64% |

The tier cutoffs are recall targets (HIGH 0.5, MEDIUM 0.8) calibrated on
validation, not on test. Scores are **ranking scores**: ECE is 0.040, so do not
read them as probabilities.

Baseline on the same test split: logistic regression AUPRC 0.018. The
original XGBoost baseline row (0.002) was a configuration fault.

---

## Evaluated and not adopted

LightGBM was compared, under pre-registered decision rules, with an LSTM
remaining-life model and with combinations of LightGBM and a second model.
None improved on LightGBM by a material margin. The best gained +0.0004
AUPRC, against a ±0.003 threshold. The scripts are in `archive/fusion_study/`
and the records in `results_fusion/`.

**Why urgency is a ranking, not a countdown.** No model supports a trustworthy
"N days left". The survival model ranks drives well (C-index 0.75), but its
day counts are about 1,000 days off. The system therefore shows a **ranking**
("expected to fail sooner than X% of the fleet"). Urgency orders work within a
tier and never raises a drive's tier.

**Identifier ablation.** Removing datacenter, vault, pod and slot IDs changes
AUPRC by −0.00014, 95% CI [−0.0025, +0.0022]: equivalence, so the IDs are
excluded. The records are in `results_ablation/`.

---

## Data and splits

Backblaze daily SMART snapshots, split by time:

| split | dates | drive-days |
|---|---|---|
| train | 2013-04-10 → 2024-03-31 | 484.1M (undersampled to 3.44M for training) |
| validation | 2024-04-01 → 2024-09-30 | 52.8M |
| test | 2024-10-01 → 2025-12-31 | 144.7M |

Features are built on each drive's full history before splitting (Stage 3).
That way, a validation or test drive's rolling windows start from its real
earlier readings. The data is then cut by date (Stage 4). Training uses only
the training period, and the tier cutoffs are fitted on validation.

---

## Repository layout

**Core pipeline** (run in order; `RUNBOOK.md` gives the exact commands)

| stage | script | role |
|---|---|---|
| 1–2 | `01_acquire_data.py`, `02_clean_data.py` | download and clean Backblaze CSVs |
| 3–5 | `03_build_targets_and_features.py`, `04_temporal_split.py`, `05_imbalance_strategy.py` | targets, 212 features, time split, undersampling |
| 6 | `06_train_lightgbm.py` | Optuna-tuned LightGBM, 5-seed ensemble |
| 8–9 | `08_hybrid_fusion.py`, `09_risk_tier_calibration.py` | score export (selected `lgbm_only`, so the score is LightGBM's probability), recall-calibrated tiers |
| 10–12 | `10_evaluation.py`, `11_shap_explainability.py`, `12_mitigation_mapping.py` | evaluation, SHAP, root cause → tier-aware action |
| 13–14 | `13_dashboard.py`, `14_agentic_mitigation.py`, `build_dashboard_db.py` | dashboard, agent, serving database |

**Evaluation and studies**

| script | role |
|---|---|
| `14b_paper_evaluation.py` | bootstrap CIs and tier accuracy → `results/paper_tables/` |
| `15_datacenter_holdout.py` | leave-one-datacenter-out generalisation check |
| `17_ablation_compare.py`, `17b_…`, `17c_…` | identifier ablation and its bootstrap |
| `18_leadtime_curve.py`, `19_calibration.py` | lead-time curve, calibration |
| `20_train_xgb_aft.py` | P-A survival model, the source of fleet urgency |
| `23_pa_urgency.py` | fleet-urgency ranking served by the dashboard and agent |
| `25_agent_eval.py` | agent evaluation: policy audit, red team, live harness → `results/agent_eval/` |
| `26_row_auprc_ci.py` | headline row-level AUPRC CI, full drive bootstrap (B = 1000) → `results/paper_tables/` |
| `27_prevalence_inflation.py` | same scores under undersampled / healthy-only evaluation protocols (paper Table 3) → `results/paper_tables/` |
| `28_fresh_2026.py` | out-of-time evaluation of the frozen model on Backblaze 2026 H1 → `results_fresh2026/` |
| `29_retrain_2026.py` | pre-registered retraining study (retrained and fleet-relative variants vs frozen, tested on 2026) → `results_retrain2026/` |
| `30_relabel_dec2025.py` | completes test labels for 2025-12-18..31 with January 2026 failures (602 rows); rewrites the test label files and bootstrap cache |
| `31_deploy_2026.py` | ships the retrained model: scores 2026, tiers, urgency, SHAP actions, dashboard DB → `results_deploy2026/` |
| `32_alert_policies.py` | pre-registered alert-policy study (persistence, daily top-k) for the shipped model → `results_alerts2026/` |
| `33_drift_monitor.py` | monthly drift monitor (alert volume, score PSI, 3-month AUPRC) with a retrain flag; frozen-model replay → `results_drift/`, dashboard panel |

The `fix*.py`, `check_*.py`, `verify_*.py` and `diagnose_*.py` files are
one-off diagnostics kept for provenance.

**Folders.**
- `models/` holds the LightGBM ensemble (tracked).
- `models_fusion/` holds the P-A survival model.
- `results/` holds metrics and tables; the multi-GB exports are gitignored.
- `results_fusion/` and `results_ablation/` hold the study records.
- `archive/` holds the evaluated-but-not-adopted fusion work and old notes.
- `data/` holds the parquet splits (gitignored).

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Tested with Python 3.12, LightGBM 4.7, XGBoost 3.4, DuckDB 1.5 and
Streamlit 1.62, on a 16 GB laptop. The GPU (RTX 3050 6 GB) was used only by
the archived LSTM.
For the exact tested versions, run `pip install -r requirements-lock.txt`
instead. A fresh install from the loose `requirements.txt` also works: on
2026-10-28 the agent and its evaluation passed on the newer versions it
resolved.

**Dashboard (deployed model).** The shipped model is the retrained ensemble
(`models_retrain2026/A`, chosen by the pre-registered study in `results_retrain2026/`).
`31_deploy_2026.py` scores Jan–Jun 2026 with it and writes everything the dashboard
and agent read to `results_deploy2026/`, which is now their default:

```bash
python 31_deploy_2026.py all
streamlit run 13_dashboard.py
```

The frozen model on the 2024–25 test period (the paper's evaluation) is still
available:

```bash
python 23_pa_urgency.py
python build_dashboard_db.py --results_dir results
streamlit run 13_dashboard.py -- --results_dir results
```

**Agent, one drive.** Dry run is deterministic and needs no API key. Pick a drive
from the dashboard's queue (2026 serials); add `--results_dir results` for the
2024–25 test period:

```bash
python 14_agentic_mitigation.py --serial_number <SERIAL> --dry_run
```

For live mode, set `ANTHROPIC_API_KEY`. `ANTHROPIC_MODEL` is optional; the
default is `claude-sonnet-5-5`.
The agent can check spare parts and technician availability and draft a
maintenance ticket. It never executes an action: an operator approves every
ticket. The model's tier is an immutable input, and the agent cannot raise or
lower it. This is enforced in code, not only in the prompt:
- the ticket's identity, tier and root cause are attached by the server;
- the parts lookup is locked to the drive's own model;
- a non-HIGH drive cannot be scheduled as urgent.

`results/agent_eval/FINDINGS_agent_eval.md` reports the evaluation. The rule
policy passes all 13 invariants on all 32,794 at-risk drives. The server
boundary holds in 10 of 10 red-team scenarios.

**Full rebuild from raw data.** Use `00_setup.md` for stages 1–5. For
stage 6, use the `--mode train` and `--mode predict` usage in the docstring of
`06_train_lightgbm.py`. Use `RUNBOOK.md` for stages 8–14. Stage 8 re-exports
scores from the archived Stage 7 LSTM predictions. Stage 3 takes hours; most
other stages take minutes. The large per-row exports are not in git; `00_setup.md` lists
which stage regenerates each one.

---

## Limitations

- **Scores rank; they are not probabilities.** Use the tiers, not raw scores,
  for decisions.
- **Retrospective test.** The "latest" assessment for a drive can be months
  old. The dashboard shows each row's age.
- **SMART has a ceiling.** 17% of failing drives (25% of failing drive-days)
  are still in the LOW tier. Many failures give no SMART warning, a known limit
  of this data source.
- **No reliable day count.** Urgency is a ranking; see above.
- **Compute-bound choices.** Undersampled training set; Optuna on a
  5M-row validation prefix.
