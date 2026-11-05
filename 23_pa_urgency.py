"""
Fleet urgency from the survival model (XGBoost AFT, models/urgency/pa_xgb_aft.json).

The model ranks drives by time-to-failure well (Model 2 test C-index 0.841,
results_model2/urgency.json) but its absolute day counts are far off, so the
dashboard shows a RANKING rather than a countdown:

    fleet_urgency_pct = share of the reference fleet that the model expects to
                        outlive this drive (ties count half)

The reference fleet is each drive's latest prediction on the validation split
(models/urgency/pa_xgb_aft_val_preds.csv), so test drives are never ranked
against themselves.  Helpers only: 31_deploy_2026.py builds the urgency table
and 35_model2_eval.py measures the C-index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def latest_per_drive(con, csv: Path):
    return con.execute(f"""
        SELECT serial_number,
               MAX(CAST(date AS DATE)) AS date,
               ARG_MAX(aft_rul_pred, CAST(date AS DATE)) AS aft_rul_pred
        FROM read_csv_auto('{csv.resolve().as_posix()}')
        GROUP BY 1 ORDER BY 1
    """).fetchdf()


def harrell_c(time: np.ndarray, event: np.ndarray, pred: np.ndarray) -> float:
    """Harrell's C for predicted survival time (higher pred = longer life).

    A pair (i, j) is comparable when i has an event and time_i < time_j.  It is
    concordant when pred_i < pred_j; tied predictions count 1/2.  O(n log n)
    with a Fenwick tree over prediction ranks, inserting drives in order of
    decreasing time so every query sees exactly the strictly-longer-lived set.
    """
    ranks = np.unique(pred, return_inverse=True)[1] + 1
    m = int(ranks.max())
    tree = np.zeros(m + 1, dtype=np.int64)

    def add(r):
        while r <= m:
            tree[r] += 1
            r += r & -r

    def prefix(r):
        s = 0
        while r > 0:
            s += tree[r]
            r -= r & -r
        return s

    order = np.argsort(-time, kind="stable")
    t_sorted = time[order]
    conc = ties = total = 0.0
    inserted = 0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and t_sorted[j] == t_sorted[i]:
            j += 1
        group = order[i:j]
        for k in group[event[group] == 1]:
            r = ranks[k]
            le = prefix(r)
            eq = le - prefix(r - 1)
            conc += inserted - le
            ties += eq
            total += inserted
        for k in group:
            add(ranks[k])
        inserted += len(group)
        i = j
    return (conc + 0.5 * ties) / total if total else float("nan")

