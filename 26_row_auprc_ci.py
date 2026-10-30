"""
Stage 26 -- headline row-level AUPRC confidence interval for LightGBM (paper).

The earlier interval in results/paper_tables/table_bootstrap_ci.csv used 25
replicates on 40,000-drive subsamples, which is too few for a stable 95%
interval.  This script uses the exact weighted-AP bootstrap from
17_ablation_compare.py: B replicates over ALL 352,383 test drives and ALL
144.7M drive-day rows, resampling physical drives.

Input: the joined LightGBM test cache written by the archived §7 confirmation
(results_fusion/_pb_confirm_joined.parquet: drive_id, prob, y).  Only the
LightGBM column is used.

Usage:
    python 26_row_auprc_ci.py --boot 1000
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results_fusion/_pb_confirm_joined.parquet")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results/paper_tables/table_row_auprc_ci_full.csv")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("s17", "17_ablation_compare.py")
    s17 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s17)

    t = pq.read_table(args.cache, columns=["drive_id", "prob", "y"])
    drive = t["drive_id"].to_numpy()
    prob = t["prob"].to_numpy()
    y = t["y"].to_numpy()
    del t
    point = float(average_precision_score(y, prob))
    print(f"[LOAD] {len(y):,} rows, {int(y.sum()):,} positives, "
          f"{int(drive.max()) + 1:,} drives; AUPRC {point:.6f}")

    order = np.argsort(-prob, kind="stable")
    ys, ss, ds = y[order], prob[order], drive[order]
    del order, prob, y, drive
    n_drives = int(ds.max()) + 1
    w1 = np.ones(n_drives, dtype=np.int32)
    check = s17.weighted_ap(ys, ss, ds, w1)
    print(f"[CHECK] weighted AP at unit weights {check:.10f} vs sklearn {point:.10f}")
    assert abs(check - point) < 1e-9

    rng = np.random.default_rng(SEED)
    reps = np.empty(args.boot)
    t0 = time.time()
    for b in range(args.boot):
        mult = np.bincount(rng.integers(0, n_drives, n_drives), minlength=n_drives).astype(np.int32)
        reps[b] = s17.weighted_ap(ys, ss, ds, mult)
        if (b + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[BOOT] {b + 1}/{args.boot}  eta {(args.boot - b - 1) * el / (b + 1) / 60:.1f} min")
    lo, hi = np.percentile(reps, [2.5, 97.5])
    prevalence = float(ys.mean())
    pd.DataFrame([{
        "level": "drive-day", "model": "LightGBM (5-seed ensemble)", "auprc": point,
        "ci_lo": lo, "ci_hi": hi, "n_boot": args.boot, "lift_over_prevalence": point / prevalence,
        "resampling_unit": "physical drive (full, all 352,383 drives)",
    }]).to_csv(args.out, index=False)
    print(f"[RESULT] AUPRC {point:.5f}  95% CI [{lo:.5f}, {hi:.5f}]  B={args.boot}")
    print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()
