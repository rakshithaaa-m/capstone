# Pre-registration: alert policies (top-k and persistence) for the shipped model

Written 2026-11-04, before any alert-policy metric was computed on validation or test.
It is not changed after results are seen; deviations go in the log at the end.

## Motivation
The shipped model (retrained, `models_retrain2026/A`) alerts on every drive-day with
score ≥ 0.73. On 2026 only 3.6% of alerted rows are followed by a failure within 14
days. Operators act on **new** alerts and have a fixed daily capacity. Two alert
policies change *which* drive-days raise an alert without changing the model:
- **Persistence:** alert only after the score stays above the threshold on consecutive
  days, which filters one-day spikes.
- **Daily top-k:** alert on the k highest-scoring drives each day, a fixed workload.

## Data
Scores of model A, already computed (`data/processed/retrain2026/preds_{val,test}_A.parquet`):
- validation 2025-07-01 → 2025-12-17 (56.0M drive-days, 2,099 failing drives): used to fit k;
- test 2026-01-01 → 2026-06-16 (57.6M drive-days, 2,451 failing drives): one look.

## Definitions
- **Alert episode (new alert):** a maximal run of alerted rows of one drive on
  consecutive calendar days. Its start is the day an operator would be paged.
- **New-alert precision (primary):** share of episodes whose start row is positive
  (the drive fails 1–14 days after that day).
- **Failing-drive catch rate (primary):** share of drives with ≥ 1 positive row in the
  window that have ≥ 1 alerted positive row.
- Secondary: alerted rows per day, new alerts per day, row precision.

## Policies
- **B (baseline, shipped):** alert when score ≥ 0.73.
- **P2:** score ≥ 0.73 today and on the previous calendar day.
- **P3:** score ≥ 0.73 on three consecutive calendar days (descriptive only).
- **K\*:** alert on the k highest scores of each day, where k is the smallest integer
  whose validation catch rate is at least B's validation catch rate (matched recall).
- Descriptive grid: k ∈ {50, 100, 200, 300, 500, 1000, 2000} (catch rate, row precision).

## Confirmatory comparisons on test (two, paired drive-clustered bootstrap, B = 1000)
For P2 vs B and for K\* vs B:
- Δ new-alert precision, and Δ catch rate (percentage points), each with a 95% CI.
- **Improves precision:** Δ precision CI lower bound > 0.
- **Adoptable:** improves precision and the catch-rate drop is at most 5 percentage
  points (point estimate). If both are adoptable, the one with the higher validation
  new-alert precision is adopted; if neither, B stays.

## Deviations
(none)
