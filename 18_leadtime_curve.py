"""
18 -- Lead-time curve for the drive-failure classifier.

Question: for drives that eventually fail, how far before failure does the
model flag HIGH?

Specification (from the project plan): the lead-time curve is the share of
failing drives flagged HIGH at >= 30 / 14 / 7 / 1 days before failure.

Implementation notes
--------------------
The label `failure_within_14d` is prospective: a row is flagged when a failure
falls within 14 days after that row.  The last row flagged for a failing drive is
therefore the failure date (the failure occurs inside the window starting at that
row).  We take fail_date = max(date) over the failing rows of each drive.

For each failing drive and each horizon L in {30, 14, 7, 1}, we look up the
probability at date = fail_date - L and flag HIGH when it is >= the shipped
cost-optimal threshold (0.62 from models/lgbm_threshold.json).

Outputs:
  results_ablation/leadtime_curve.csv
"""
import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

HORIZONS = [30, 14, 7, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="results/lgbm_test_probs.csv")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default="results_ablation/leadtime_curve.csv")
    args = ap.parse_args()

    thr = args.threshold
    if thr is None:
        thr = float(json.load(open("models/lgbm_threshold.json"))["threshold"])

    con = duckdb.connect()
    t = "read_csv('" + Path(args.test).as_posix() + "', header=true)"

    failing = con.execute(f"""
        SELECT serial_number, max(date) AS fail_date
        FROM {t} WHERE failure_within_14d = 1 GROUP BY serial_number
    """).df()
    n_fail = int(len(failing))
    print(f"[LEAD] {n_fail:,} failing drives")

    rows = []
    for L in HORIZONS:
        q = f"""
        SELECT f.serial_number, f.fail_date, p.lgbm_prob
        FROM failing f
        JOIN read_csv('{Path(args.test).as_posix()}', header=true) p
          ON p.serial_number = f.serial_number AND p.date = f.fail_date - INTERVAL '{L} days'
        """
        got = con.execute(q).df()
        n_found = int(len(got))
        flagged = 0
        serials_flagged = set()
        for _, r in got.iterrows():
            if float(r["lgbm_prob"]) >= thr:
                flagged += 1
                serials_flagged.add(r["serial_number"])
        rows.append({
            "horizon_days": L,
            "n_drives": n_fail,
            "n_evaluated": n_found,
            "n_flagged_high": flagged,
            "share_flagged_high": flagged / n_fail if n_fail else float("nan"),
        })
        print(f"[LEAD] L={L:2d}d  evaluated={n_found:5d}  flagged_HIGH={flagged:5d}  "
              f"share={flagged / n_fail if n_fail else float('nan'):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[SAVE] {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
