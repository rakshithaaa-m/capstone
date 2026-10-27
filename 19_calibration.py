"""19 -- Probability calibration metrics for the LightGBM classifier (test set).

Computes, on the held-out test split:
  * Brier score (all rows and over positives),
  * Expected Calibration Error (ECE) with NBINS equal-width bins,
  * per-bin reliability table (predicted vs actual).

Outputs:
  results_ablation/calibration_metrics_<n>.csv   (per-bin table)
  results_ablation/calibration_summary.csv      (scalar summary)
"""
import argparse
from pathlib import Path

import duckdb
import pandas as pd

NBINS = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="results/lgbm_test_probs.csv")
    ap.add_argument("--out", default="results_ablation/calibration_metrics.csv")
    ap.add_argument("--bins", type=int, default=NBINS)
    ap.add_argument("--out_summary", default="results_ablation/calibration_summary.csv")
    args = ap.parse_args()

    con = duckdb.connect()
    t = "read_csv('" + str(Path(args.test).resolve()) + "', header=true)"

    brier_all, n_all = con.execute(
        f"SELECT avg(pow(lgbm_prob - failure_within_14d, 2)), count(*) FROM {t}").fetchone()
    brier_pos, n_pos = con.execute(
        f"SELECT avg(pow(lgbm_prob - failure_within_14d, 2)), count(*) FROM {t} WHERE failure_within_14d = 1").fetchone()
    n_pos = int(n_pos)
    brier_all = float(brier_all)

    rows = []
    ece_num = 0.0
    for i in range(args.bins):
        lo = i / args.bins
        hi = (i + 1) / args.bins
        cnt, mp, my, s = con.execute(f"""
            SELECT count(*), avg(lgbm_prob), avg(failure_within_14d), sum(failure_within_14d)
            FROM {t} WHERE lgbm_prob > {lo} AND lgbm_prob <= {hi}
        """).fetchone()
        if cnt == 0:
            continue
        ece_num += int(cnt) * abs(float(mp) - float(my))
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(cnt), "mean_pred": float(mp),
                     "mean_actual": float(my), "abs_err": float(abs(float(mp) - float(my)))})
    ece = ece_num / n_all

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    pd.DataFrame([{
        "brier_all": brier_all, "brier_positives": float(brier_pos),
        "n": int(n_all), "n_positives": n_pos, "ece": float(ece),
        "bins": args.bins, "calibration_scheme": "10 equal-width bins, test set",
        "cutoff_source": "quantile-based decision thresholds from Stage 09 (recall-calibrated), not probability calibration output",
    }]).to_csv(args.out_summary, index=False)

    print(f"[CAL] Brier (all) = {brier_all:.8f}   Brier (positives) = {float(brier_pos):.8f}")
    print(f"[CAL] ECE ({args.bins} equal-width bins) = {float(ece):.8f}")
    print(f"[CAL] n = {int(n_all):,}  positives = {n_pos:,}")
    print(f"[SAVE] {args.out} / {args.out_summary}")


if __name__ == "__main__":
    main()
