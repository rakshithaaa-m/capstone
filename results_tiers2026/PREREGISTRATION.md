# Pre-registration: MEDIUM-tier sizing for the shipped model

Written 2026-11-04, before any tier metric in this study was computed. Deviations go
in the log at the end.

## Motivation
Tier cutoffs are recall-calibrated on validation (Stage 9): HIGH captures 50% of
positive validation rows and HIGH + MEDIUM 80%. With the frozen model that put 9.3% of
drives in HIGH or MEDIUM (32,794 of 352,383 on its test period). With the retrained
model the same rule needs a much lower MEDIUM cutoff (0.0325 against 0.0691), and 17%
of the 2026 fleet (62,743 drives) is HIGH or MEDIUM. A recall rule lets the size of the
"watch closely" tier move with every retrain.

## Rules compared (HIGH cutoff unchanged at 0.7211 in both)
- **R80 (current):** MEDIUM cutoff such that HIGH + MEDIUM captures 80% of positive
  validation rows (0.0325).
- **S9 (size-based):** MEDIUM cutoff = the score quantile at which HIGH + MEDIUM covers
  9.3% of validation drive-days, the at-risk share of the original design.

## Decision (on validation, model A scores 2025-07-01 → 2025-12-17)
Adopt S9 if both hold:
1. it reduces HIGH + MEDIUM drive-days by at least 30% relative to R80; and
2. the share of failing drives with at least one positive row in HIGH or MEDIUM falls by
   at most 5 percentage points relative to R80.
Otherwise R80 stays.

## Reported on 2026 (descriptive, one look)
For both rules: HIGH + MEDIUM drives on 2026-06-30 (latest row per drive), share of
positive rows in HIGH + MEDIUM, failing-drive capture as in criterion 2, and the
precision of MEDIUM rows. If S9 is adopted, the deployment (`31_deploy_2026.py`) uses
its cutoff.

## Deviations
(none)
