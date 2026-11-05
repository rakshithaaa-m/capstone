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

## D-03: Displayed RUL replaced by the P-A fleet-urgency ranking

**When:** 2026-10-27, after §7b returned KEEP (LSTM) and both §7b metrics had
been seen. This is a **product decision by the author**, made with the
results in view. It is not a statistical result of the protocol.

**What §7b decided:** the LSTM keeps supplying the displayed **day count**
(primary metric, MAE ≤ 90 d: LSTM 42.1 vs P-A 1021.0).

**What changed:** the dashboard and the agent no longer show any day count.
They show `fleet_urgency_pct`, the share of the validation-split fleet that P-A
expects to outlive the drive (`23_pa_urgency.py`).

**Why:** neither model supports a day count. The LSTM barely ranks drives
(C = 0.538), and its MAE advantage comes largely from training mostly on drives
that fail. P-A ranks well (C = 0.749) but its absolute days are about 1,000 days
off. A ranking uses only the property that was shown to hold.

**Guard rails:** urgency orders work within a tier and never escalates a drive
above its tier, in either the agent prompt or the rule policy. The LSTM
estimate stays in the agent context as an audit field only.

## D-04: P-C (TCN) is not run under Amendment 03

**When:** 2026-10-27, after the P-B run had finished and **before** any P-B
output (training log, prediction files or screen) was opened, as §4 requires.

**Decision:** P-C is not run.

**Reason:** time, not results. The author's stated priority is to finish the
project end to end: the dashboard, the agent, the README and the evaluation of
the agent. P-C needs a new raw-sequence data pipeline and GPU training on a
6 GB laptop, which is days of work. If it is attempted later, it goes under a
new amendment and is not reported as part of Amendment 03.
