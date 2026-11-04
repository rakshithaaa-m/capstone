# Drift monitor: findings

Script: `33_drift_monitor.py`. Outputs: `monthly_frozen.csv`, `monthly_deployed.csv`,
`summary.json`, and `results_deploy2026/drift_*` for the dashboard.

Each month the model is compared with its own validation period on three signals:
- alert volume: share of drive-days at or above the alert threshold, as a ratio (label-free);
- score shift: PSI over ten reference deciles (label-free);
- AUPRC over the trailing three labelled months, as a ratio (labels arrive 14 days late).

RETRAIN at alert volume ≥ 1.25×, PSI ≥ 0.25 or AUPRC ≤ 0.75×; WARN at 1.10×, 0.10 or 0.85×.

**Status: illustrative, not pre-registered.** The limits were set after the frozen
model's ageing was known. A first version used single-month AUPRC
(`run_v1_single_month.log`); monthly AUPRC swings 0.6–1.4× without drift, so it raised
RETRAIN on the deployed model's first month (January, a low-failure holiday month).
The three-month pooled window replaced it.

## Frozen model replayed (trained to March 2024, threshold 0.62)
| | Month | Alert volume | 3-month AUPRC |
|---|---|---|---|
| First WARN | Nov 2024 | 1.11× | – |
| First RETRAIN | **Jul 2025** | 1.30× | 0.72× |
| End of data | Jun 2026 | 1.77× | 0.60× |

The monitor would have called for retraining in July 2025, 16 months after the training
data ended and six months before the 2026 degradation was measured. Retraining on data
to June 2025 then recovered 2026 AUPRC from 0.061 to 0.078 (`results_retrain2026`).

## Deployed model (retrained to June 2025, threshold 0.73)
| Month | Alert volume | 3-month AUPRC | Status |
|---|---|---|---|
| Jan 2026 | 1.07× | – | OK |
| Feb 2026 | 1.12× | – | WARN |
| Mar 2026 | 1.23× | 0.77× | WARN |
| Apr 2026 | 1.28× | 1.06× | RETRAIN |
| Jun 2026 | 1.29× | 1.07× | RETRAIN |

The retrained model is ageing the same way: alert volume crossed 1.25× in April 2026,
ten months after its training data ended. Ranking quality has not degraded (three-month
AUPRC 1.03–1.07×), so this is an early warning; the daily top-k queue
(`results_alerts2026`) keeps operator workload fixed meanwhile. The two replays suggest
retraining roughly every 9–12 months.

PSI stays below 0.06 throughout: the drift is in the high-score tail (more drives cross
the threshold), which a whole-distribution measure barely registers. Alert volume is the
earliest and cleanest signal.
