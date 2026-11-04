# Alert-policy study: findings

Pre-registration: `PREREGISTRATION.md` (committed 3447c80 before any metric; one
implementation deviation, logged there). Script: `32_alert_policies.py`. Numbers:
`results.json`, `policies_{val,test}.csv`, `topk_grid_{val,test}.csv`.

Model: the shipped retrained ensemble (A). Baseline B = alert when score ≥ 0.73.
k\* = 2,074 alerts a day, fitted on validation to match B's validation catch rate (62.2%).

## Test (2026-01-01 → 2026-06-16, one look)
| Policy | New-alert precision | Failing drives caught | New alerts/day | Alerted rows/day |
|---|---|---|---|---|
| B: score ≥ 0.73 (shipped) | 4.02% | 69.5% | 193 | 2,654 |
| P2: 2 days in a row | 3.50% | 61.6% | 129 | 2,461 |
| P3: 3 days in a row (descriptive) | 3.27% | 57.9% | 115 | 2,332 |
| **K\*: daily top 2,074** | **4.58%** | 67.2% | 180 | **2,074** |

Paired drive-clustered bootstrap (B = 1000):
- **P2 vs B:** precision −0.52 points [−0.71, −0.32], catch −7.9 points [−9.0, −6.8].
  Persistence is **worse** on both; not adopted.
- **K\* vs B:** precision **+0.55 points [+0.43, +0.68]** (+14% relative), catch −2.2
  points [−2.8, −1.7], within the 5-point limit. **Adopted.**

## Reading
- **A fixed daily budget beats a fixed threshold on 2026.** The fleet's scores drift
  upward over time (the same ageing that degraded the frozen model), so a fixed
  threshold raised 26% more alerts a day in 2026 (2,654) than on validation (2,108).
  The top-k rule holds the workload at 2,074 a day, cuts alerts by 22%, and raises
  precision, for a small loss in coverage.
- **Persistence does not help.** Failing drives' scores do not stay high on
  consecutive days more often than healthy drives' spikes do; requiring two days
  mainly delays and drops true alerts.
- **The budget sets the precision.** Descriptive curve on 2026:

| Alerts a day (top-k) | Precision of alerted rows | Failing drives caught |
|---|---|---|
| 50 | 21.9% | 15.5% |
| 100 | 17.7% | 23.4% |
| 300 | 12.4% | 42.8% |
| 500 | 10.1% | 50.6% |
| 1,000 | 6.9% | 58.8% |
| 2,074 (adopted) | 4.2% | 67.2% |

  A team that can review 100 drives a day sees about one real failure in six alerts,
  against one in 28 under the threshold.
