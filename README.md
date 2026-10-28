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

The project also contains a **pre-registered evaluation of hybrid fusion**. It
tested whether combining LightGBM with a second model (an LSTM, a survival
model, or a second boosted model) improves failure prediction. None did by a
material margin, so the shipped predictor is LightGBM alone.
See [Evaluated and not adopted](#evaluated-and-not-adopted).

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

Baselines on the same test split: logistic regression AUPRC 0.018. The
original XGBoost baseline row (0.002) was a configuration fault. A tuned
XGBoost was later built as fusion partner P-B; see below.

---

## Evaluated and not adopted

Each study was written down, with its decision rule, before its results were
seen. Deviations are logged in the study folder.

| study | question | result | decision |
|---|---|---|---|
| Stage 8 fusion + Stage 16 stacking | Does the LSTM's RUL estimate improve LightGBM? | stacking −0.016 AUPRC, CI [−0.025, −0.009] | `lgbm_only` |
| Amendment 03, P-A | XGBoost AFT survival model as fusion partner | −0.039, CI [−0.050, −0.028] | screen fail |
| Amendment 03, P-B | Tuned XGBoost classifier as fusion partner | test +0.0004, CI [+0.0002, +0.0006] | real but immaterial (margin ±0.003) |
| Amendment 03, P-C | TCN on raw SMART sequences | not run (time; D-04) | — |
| Identifier ablation | Do datacenter / vault / pod IDs inflate results? | Δ −0.00014, CI [−0.0025, +0.0022] | equivalence: IDs excluded |

**Remaining useful life (RUL).** Neither model supports a trustworthy
"N days left":
- The LSTM has a low error on failing drives (MAE 42 d on the last 90 days), but
  it barely ranks drives (C-index 0.54).
- The survival model ranks well (C-index 0.75), but its day counts are about
  1,000 days off.

The system therefore shows a **ranking** ("expected to fail sooner than X% of
the fleet") instead of a countdown. Urgency orders work within a tier and never
raises a drive's tier.

Write-ups:
- `results_fusion/` (Amendment 03, `FINDINGS_P-A.md`, `FINDINGS_P-B.md`,
  `DEVIATIONS.md`)
- `results_ablation/` (pre-registration, Amendments 01–02, findings)

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
| 7 | `07_train_lstm_v2.py` | LSTM RUL model (kept for the fusion evaluation) |
| 8–9 | `08_hybrid_fusion.py`, `09_risk_tier_calibration.py` | fusion selection (chose `lgbm_only`), recall-calibrated tiers |
| 10–12 | `10_evaluation.py`, `11_shap_explainability.py`, `12_mitigation_mapping.py` | evaluation, SHAP, root cause → tier-aware action |
| 13–14 | `13_dashboard.py`, `14_agentic_mitigation.py`, `build_dashboard_db.py` | dashboard, agent, serving database |

**Evaluation and studies**

| script | role |
|---|---|
| `14b_paper_evaluation.py`, `14c_rul_window_eval.py` | bootstrap CIs, tier accuracy, RUL windows → `results/paper_tables/` |
| `15_datacenter_holdout.py` | leave-one-datacenter-out generalisation check |
| `16_stacking_study.py` | drive-level fusion screen with a pluggable partner |
| `17_ablation_compare.py`, `17b_…`, `17c_…` | identifier ablation and its bootstrap |
| `18_leadtime_curve.py`, `19_calibration.py` | lead-time curve, calibration |
| `20_train_xgb_aft.py`, `21_rul_compare.py` | P-A survival model; RUL head-to-head with the LSTM |
| `22_train_xgb_clf.py`, `24_pb_confirm.py` | P-B XGBoost classifier; §7 test confirmation |
| `23_pa_urgency.py` | fleet-urgency ranking served by the dashboard and agent |
| `25_agent_eval.py` | agent evaluation: policy audit, red team, live harness → `results/agent_eval/` |

The `fix*.py`, `check_*.py`, `verify_*.py` and `diagnose_*.py` files are
one-off diagnostics kept for provenance.

**Folders.** `models/` holds the LightGBM ensemble, tracked. `models_fusion/`
holds the P-A and P-B models. `results/` holds metrics and tables (multi-GB
exports are gitignored). `results_fusion/` and `results_ablation/` hold the
study records. `data/` holds the parquet splits (gitignored).

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Tested with Python 3.12, LightGBM 4.7, XGBoost 3.4, DuckDB 1.5 and
Streamlit 1.62, on a 16 GB laptop (RTX 3050 6 GB, used only by the LSTM).

**Dashboard.** It needs the Stage 6–12 exports in `results/` and the fleet
urgency file:

```bash
python 23_pa_urgency.py
python build_dashboard_db.py --results_dir results
streamlit run 13_dashboard.py -- --results_dir results
```

**Agent, one drive.** Dry run is deterministic and needs no API key:

```bash
python 14_agentic_mitigation.py --serial_number ZA1C0JYE --dry_run
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
`06_train_lightgbm.py`. Use `RUNBOOK.md` for stages 7–14. Stage 3 and Stage 7 take hours; most other stages
take minutes. The large per-row exports are not in git; `00_setup.md` lists
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
- **No reliable day count.** See the RUL note above.
- **The fusion screen is dominated by failure-day rows.** 91% of failing
  validation drives are scored on the day they fail. A fixed-lead-time
  comparison would be a separate, new study.
- **Compute-bound choices.** Undersampled training set; Optuna on a
  5M-row validation prefix.
