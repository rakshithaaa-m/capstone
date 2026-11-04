# MEDIUM-tier sizing: findings

Pre-registration: `PREREGISTRATION.md` (committed abaf93f before any metric). Script:
`34_tier_sizing.py`. Numbers: `results.json`, `run.log`.

## Decision (validation, 2025-07-01 → 2025-12-17)
| | R80 (current, 80% recall) | S9 (HIGH + MEDIUM = 9.3% of drive-days) |
|---|---|---|
| MEDIUM cutoff | 0.0325 | 0.0442 |
| HIGH + MEDIUM share of drive-days | 16.1% | 9.3% (−42%, criterion 1 met) |
| Failing drives reaching HIGH or MEDIUM | 85.8% | 78.7% (−7.1 points, criterion 2 **not** met: limit 5) |

**R80 stays.** The deployment is unchanged.

## 2026 (descriptive, one look)
| | R80 | S9 |
|---|---|---|
| MEDIUM drives on 2026-06-30 | 56,063 | 33,415 |
| Positive rows in HIGH + MEDIUM | 79.6% | 77.0% |
| Failing drives reaching HIGH or MEDIUM | 84.7% | 82.4% (−2.3 points) |
| MEDIUM row precision | 0.07% | 0.10% |

On 2026 the size-based rule would have cost only 2.3 points of coverage, but the
decision was made on validation and is not revisited.

## Reading
- A smaller MEDIUM tier does lose failing drives: at the validation level the cost was
  7 points of coverage for a 42% shorter list, more than the 5 points allowed.
- MEDIUM rows are barely more likely to fail than the fleet average (0.07–0.10% of rows
  against a base rate of 0.05%). The tier's value is coverage, not precision: it is a
  low-cost "poll more often" list. Prioritised review belongs to HIGH and to the daily
  top-k queue (`results_alerts2026`), which is what operators work from.
