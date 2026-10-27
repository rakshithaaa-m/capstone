"""
Stage 21 -- Amendment 03, section 7b: which model supplies the displayed RUL?

Compares the LSTM (results/lstm_full_test_preds.csv) with P-A, the XGBoost AFT
survival model (results_fusion/pa_xgb_aft_test_preds.csv), on the test split.
This is independent of fusion: fusion asks whether RUL improves the failure
prediction; this asks whose days-left number is more accurate.

Rows: test rows of drives that fail (is_censored = 0) with rul_days >= 0.
Post-failure rows (rul_days < 0) are excluded, as fixed in Amendment 03 s.10.

Primary   MAE on rul_days <= 90, paired drive-clustered bootstrap B = 1000 on
          dMAE = MAE(LSTM) - MAE(P-A).  REPLACE if the CI lower bound > 0.
Secondary MAE on all rows (the LSTM's 96.3 basis) and on rul_days <= 30;
          Harrell's C on one row per drive -- the FIRST test row, not the
          latest (see results_fusion/DEVIATIONS.md, D-01).

Usage:
    python 21_rul_compare.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261027


def load(test: Path, lstm: Path, pa: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import duckdb

    con = duckdb.connect(config={"memory_limit": "6GB",
                                 "temp_directory": str(Path.home() / "duck_spool")})
    q = lambda p: p.resolve().as_posix()
    try:
        con.execute(f"""
            CREATE TEMP TABLE t AS
            SELECT serial_number, CAST(date AS DATE) AS date, rul_days, is_censored
            FROM read_parquet('{q(test)}')
        """)
        con.execute("""CREATE TEMP TABLE fail AS
                       SELECT DISTINCT serial_number FROM t WHERE is_censored = 0""")
        # Failing-drive rows with both predictions, matched on (serial, date).
        rows = con.execute(f"""
            WITH l AS (SELECT serial_number, CAST(date AS DATE) AS date, lstm_rul_pred
                       FROM read_csv_auto('{q(lstm)}')
                       WHERE serial_number IN (SELECT serial_number FROM fail)),
                 a AS (SELECT serial_number, CAST(date AS DATE) AS date, aft_rul_pred
                       FROM read_csv_auto('{q(pa)}')
                       WHERE serial_number IN (SELECT serial_number FROM fail))
            SELECT t.serial_number, t.date, t.rul_days, l.lstm_rul_pred, a.aft_rul_pred
            FROM t
            LEFT JOIN l USING (serial_number, date)
            LEFT JOIN a USING (serial_number, date)
            WHERE t.is_censored = 0 AND t.rul_days >= 0
            ORDER BY t.serial_number, t.date
        """).fetchdf()
        # One row per drive for the C-index: each drive's first usable test
        # row.  Failing drives: event at rul_days.  Surviving drives: censored
        # at (last test date - first test date), the days they are known alive.
        first = con.execute(f"""
            WITH ok AS (SELECT * FROM t WHERE is_censored = 1 OR rul_days >= 0),
                 f AS (SELECT serial_number, MIN(date) AS d0 FROM ok GROUP BY 1),
                 last AS (SELECT serial_number, MAX(date) AS d1 FROM t GROUP BY 1)
            SELECT ok.serial_number,
                   CASE WHEN ok.is_censored = 0 THEN ok.rul_days
                        ELSE date_diff('day', f.d0, last.d1) END AS time,
                   1 - ok.is_censored AS event
            FROM ok
            JOIN f ON f.serial_number = ok.serial_number AND f.d0 = ok.date
            JOIN last ON last.serial_number = ok.serial_number
            ORDER BY 1
        """).fetchdf()
        preds = con.execute(f"""
            WITH f AS (SELECT serial_number, MIN(date) AS d0 FROM t
                       WHERE is_censored = 1 OR rul_days >= 0 GROUP BY 1),
                 l AS (SELECT serial_number, CAST(date AS DATE) AS date, lstm_rul_pred
                       FROM read_csv_auto('{q(lstm)}')),
                 a AS (SELECT serial_number, CAST(date AS DATE) AS date, aft_rul_pred
                       FROM read_csv_auto('{q(pa)}'))
            SELECT f.serial_number, l.lstm_rul_pred, a.aft_rul_pred
            FROM f
            LEFT JOIN l ON l.serial_number = f.serial_number AND l.date = f.d0
            LEFT JOIN a ON a.serial_number = f.serial_number AND a.date = f.d0
            ORDER BY 1
        """).fetchdf()
    finally:
        con.close()
    first = first.merge(preds, on="serial_number", how="left")
    info = {
        "failing_rows": len(rows),
        "rows_missing_lstm": int(rows["lstm_rul_pred"].isna().sum()),
        "rows_missing_pa": int(rows["aft_rul_pred"].isna().sum()),
    }
    rows = rows.dropna(subset=["lstm_rul_pred", "aft_rul_pred"]).reset_index(drop=True)
    return rows, first, info


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


def mae_bootstrap(rows: pd.DataFrame, mask: np.ndarray, boot: int) -> dict:
    """Paired drive-clustered bootstrap on dMAE = MAE(LSTM) - MAE(P-A)."""
    r = rows[mask]
    drive = pd.factorize(r["serial_number"])[0]
    nd = drive.max() + 1
    el = np.bincount(drive, weights=np.abs(r["lstm_rul_pred"] - r["rul_days"]), minlength=nd)
    ea = np.bincount(drive, weights=np.abs(r["aft_rul_pred"] - r["rul_days"]), minlength=nd)
    n = np.bincount(drive, minlength=nd).astype(np.float64)
    rng = np.random.default_rng(SEED)
    deltas = np.empty(boot)
    for b in range(boot):
        w = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        deltas[b] = (w @ el - w @ ea) / (w @ n)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"rows": int(mask.sum()), "drives": int(nd),
            "mae_lstm": float(el.sum() / n.sum()), "mae_pa": float(ea.sum() / n.sum()),
            "delta": float((el.sum() - ea.sum()) / n.sum()),
            "ci_lo": float(lo), "ci_hi": float(hi), "boot": boot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--lstm", default="results/lstm_full_test_preds.csv")
    ap.add_argument("--pa", default="results_fusion/pa_xgb_aft_test_preds.csv")
    ap.add_argument("--out", default="results_fusion/rul_compare.csv")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    print("[LOAD] failing-drive test rows with both predictions ...")
    rows, first, info = load(Path(args.test), Path(args.lstm), Path(args.pa))
    print(f"[LOAD] {info}")

    rul = rows["rul_days"].to_numpy()
    out = []
    for name, mask in (("rul<=90 (primary)", rul <= 90), ("rul<=30", rul <= 30),
                       ("all failing rows", np.ones(len(rows), bool))):
        res = mae_bootstrap(rows, mask, args.boot)
        out.append({"window": name, **res})
        print(f"[MAE] {name:18s} rows {res['rows']:>9,}  drives {res['drives']:>5,}  "
              f"LSTM {res['mae_lstm']:7.2f}  P-A {res['mae_pa']:7.2f}  "
              f"dMAE {res['delta']:+7.2f}  95% CI [{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]")

    first = first.dropna(subset=["lstm_rul_pred", "aft_rul_pred"])
    t = first["time"].to_numpy(dtype=np.float64)
    e = first["event"].to_numpy(dtype=np.int64)
    c_l = harrell_c(t, e, first["lstm_rul_pred"].to_numpy(dtype=np.float64))
    c_a = harrell_c(t, e, first["aft_rul_pred"].to_numpy(dtype=np.float64))
    print(f"[C-INDEX] {len(first):,} drives ({int(e.sum()):,} failing), first test row: "
          f"LSTM {c_l:.4f}  P-A {c_a:.4f}")

    df = pd.DataFrame(out)
    df["c_index_lstm"] = c_l
    df["c_index_pa"] = c_a
    df["c_index_drives"] = len(first)
    for k, v in info.items():
        df[k] = v
    df.to_csv(args.out, index=False)
    print(f"[SAVE] {args.out}")

    prim = out[0]
    verdict = "REPLACE" if prim["ci_lo"] > 0 else "KEEP"
    print(f"[VERDICT] {verdict}: "
          + ("P-A becomes the displayed RUL." if verdict == "REPLACE"
             else "the LSTM stays as the displayed RUL."))


if __name__ == "__main__":
    main()
