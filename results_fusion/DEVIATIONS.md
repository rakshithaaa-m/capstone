# Deviations from Amendment 03 (frozen 2026-10-27)

Append-only. Each entry: what changed, why, and whether any result had been
seen when the change was made.

## D-01 — §7b C-index uses each drive's FIRST test row, not the latest

**When:** 2026-10-27, while writing `21_rul_compare.py`, before it was run.
No §7b output had been seen.

**Why:** On a surviving drive's latest row, the drive's known follow-up is
zero days (last observed date − row date = 0). A censored drive with time 0
can never be the longer-lived member of a comparable pair, so every censored
drive drops out and the C-index is computed on failing drives only. That is
not what the frozen text intended. On the first test row, survivors carry
their full observed follow-up within the test split.

**Change:** one row per drive = the first test row with `rul_days ≥ 0` (or
any row, for survivors). Event time = `rul_days`. Censoring time = last test
date − first test date.

## D-02 — P-A prediction files rewritten with consistent line endings

**When:** 2026-10-27, after the first Stage-16 screen attempt failed to parse
`pa_xgb_aft_val_preds.csv`. No screen output was produced.

**Why:** `20_train_xgb_aft.py` wrote the header with `\n` and the rows with
pandas' Windows default `\r\n`. DuckDB's CSV sniffer rejects mixed endings.

**Change:** both files were passed through `tr -d '\r'`, and the line counts
were checked as unchanged (52,845,257 and 144,739,480, including the header).
Only line endings changed; no prediction value was touched. The script now
writes `lineterminator="\n"`. The model was not retrained.
