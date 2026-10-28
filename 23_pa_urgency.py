"""
Stage 23 -- fleet urgency from the P-A survival model (Amendment 03, D-03).

P-A (XGBoost AFT) ranks drives by time-to-failure well (test C-index 0.749)
but its absolute day counts are ~1,000 days off (FINDINGS_P-A.md), so the
dashboard shows a RANKING rather than a countdown:

    fleet_urgency_pct = share of the reference fleet that P-A expects to
                        outlive this drive (ties count half)

The reference fleet is each drive's latest P-A prediction on the VALIDATION
split, so test drives are never ranked against themselves.  Output is one row
per test drive at its latest date -- the same row the dashboard's
``drive_status`` table holds.

Usage:
    python 23_pa_urgency.py
"""
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    import duckdb

    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="results_fusion/pa_xgb_aft_val_preds.csv")
    ap.add_argument("--test", default="results_fusion/pa_xgb_aft_test_preds.csv")
    ap.add_argument("--out", default="results_fusion/pa_urgency_test.csv")
    args = ap.parse_args()

    con = duckdb.connect(config={"memory_limit": "6GB",
                                 "temp_directory": str(Path.home() / "duck_spool")})
    try:
        print("[LOAD] reference fleet: latest validation prediction per drive ...")
        ref = np.sort(latest_per_drive(con, Path(args.val))["aft_rul_pred"].to_numpy())
        print("[LOAD] test drives: latest prediction per drive ...")
        test = latest_per_drive(con, Path(args.test))
    finally:
        con.close()

    x = test["aft_rul_pred"].to_numpy()
    shorter = np.searchsorted(ref, x, side="left")
    not_longer = np.searchsorted(ref, x, side="right")
    longer = len(ref) - not_longer
    ties = not_longer - shorter
    test["fleet_urgency_pct"] = 100.0 * (longer + 0.5 * ties) / len(ref)
    test["date"] = test["date"].astype(str)
    test.to_csv(args.out, index=False, lineterminator="\n")

    meta = {
        "reference": "latest P-A prediction per drive, validation split",
        "reference_drives": int(len(ref)),
        "test_drives": int(len(test)),
        "definition": "share of the reference fleet P-A expects to outlive the drive",
        "test_c_index": 0.7491,
        "source": "results_fusion/FINDINGS_P-A.md",
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    q = np.percentile(test["fleet_urgency_pct"], [10, 50, 90])
    print(f"[SAVE] {args.out}  {len(test):,} drives vs {len(ref):,} reference; "
          f"urgency p10/p50/p90 = {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}")


if __name__ == "__main__":
    main()
