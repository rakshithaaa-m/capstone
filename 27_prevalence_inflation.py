"""
Stage 27 -- how much common evaluation shortcuts inflate the reported metrics (paper).

Post-hoc and descriptive: it re-scores the SAME shipped LightGBM test scores
under evaluation protocols often used in the drive-failure literature.  No
model is retrained and no system decision depends on it.

Protocols
  true        all 144.7M test drive-days at the real positive rate (this paper)
  never       the trivial "no drive ever fails" rule, true prevalence
  healthy     negatives taken only from drives that never fail in the test window
              (the "failed vs healthy drives" design); all positives kept
  1:100, 1:10, 1:1
              random negative undersampling of the test set; all positives kept,
              each negative kept with probability k / n_neg; repeated --reps times

Metrics: AUPRC, ROC-AUC, and precision / recall / F1 / accuracy at the shipped
threshold (models/lgbm_threshold.json), plus the best F1 over all thresholds
(what a paper that tunes its threshold on the evaluation data would report).

Input: results_fusion/_pb_confirm_joined.parquet (drive_id, prob, y); only the
LightGBM column is used.

Usage:
    python 27_prevalence_inflation.py --reps 10
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 20261030
RATIOS = [100, 10, 1]   # negatives per positive


def metrics(prob: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pred = prob >= thr
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    pos = int(np.count_nonzero(y))
    fn = pos - tp
    tn = len(y) - pos - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / pos
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # best F1 over every distinct threshold
    order = np.argsort(-prob, kind="stable")
    ys = y[order]
    ps = prob[order]
    del order
    ctp = np.cumsum(ys, dtype=np.int64)
    k = np.arange(1, len(ys) + 1, dtype=np.int64)
    last = np.r_[ps[1:] != ps[:-1], True]      # end of each tied block
    ctp, k = ctp[last], k[last]
    best_f1 = float(np.max(2 * ctp / (k + pos)))
    del ys, ps, ctp, k, last

    return {
        "n_rows": len(y), "n_pos": pos, "prevalence": pos / len(y),
        "auprc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "precision_at_thr": prec, "recall_at_thr": rec, "f1_at_thr": f1,
        "accuracy_at_thr": (tp + tn) / len(y), "best_f1": best_f1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results_fusion/_pb_confirm_joined.parquet")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="results/paper_tables/table_prevalence_inflation.csv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="alert threshold; default: the frozen model's (models/lgbm_threshold.json)")
    args = ap.parse_args()

    thr = args.threshold if args.threshold is not None else float(
        json.load(open("models/lgbm_threshold.json"))["threshold"])
    t = pq.read_table(args.cache, columns=["drive_id", "prob", "y"])
    drive = t["drive_id"].to_numpy()
    prob = t["prob"].to_numpy()
    y = t["y"].to_numpy()
    del t
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"[LOAD] {len(y):,} rows, {n_pos:,} positives; threshold {thr}")

    rows = []

    def add(protocol: str, ratio: str, m: dict, rep: int | None = None) -> None:
        rows.append({"protocol": protocol, "neg_per_pos": ratio, "rep": rep, **m})
        print(f"[{protocol:>8} {ratio:>6} rep={rep}] AUPRC {m['auprc']:.4f}  ROC {m['roc_auc']:.4f}  "
              f"P {m['precision_at_thr']:.4f}  R {m['recall_at_thr']:.4f}  "
              f"acc {m['accuracy_at_thr']:.5f}  bestF1 {m['best_f1']:.4f}", flush=True)

    t0 = time.time()
    add("true", f"{n_neg / n_pos:.0f}", metrics(prob, y, thr))

    never = {"n_rows": len(y), "n_pos": n_pos, "prevalence": n_pos / len(y),
             "auprc": n_pos / len(y), "roc_auc": 0.5, "precision_at_thr": 0.0,
             "recall_at_thr": 0.0, "f1_at_thr": 0.0, "accuracy_at_thr": n_neg / len(y),
             "best_f1": 0.0}
    add("never", f"{n_neg / n_pos:.0f}", never)

    failing = np.zeros(int(drive.max()) + 1, dtype=bool)
    failing[drive[y == 1]] = True
    keep = (y == 1) | ~failing[drive]
    m = metrics(prob[keep], y[keep], thr)
    add("healthy", f"{(m['n_rows'] - m['n_pos']) / m['n_pos']:.0f}", m)
    del keep, failing, drive

    rng = np.random.default_rng(SEED)
    pos_mask = y == 1
    for ratio in RATIOS:
        p_keep = ratio * n_pos / n_neg
        for r in range(args.reps):
            keep = pos_mask | (rng.random(len(y), dtype=np.float32) < p_keep)
            add("undersampled", f"{ratio}", metrics(prob[keep], y[keep], thr), r)
            del keep
    print(f"[TIME] {(time.time() - t0) / 60:.1f} min")

    df = pd.DataFrame(rows)
    df.to_csv(args.out.replace(".csv", "_reps.csv"), index=False)
    cols = ["n_rows", "prevalence", "auprc", "roc_auc", "precision_at_thr", "recall_at_thr",
            "f1_at_thr", "accuracy_at_thr", "best_f1"]
    g = df.groupby(["protocol", "neg_per_pos"], sort=False)
    summary = g[cols].mean()
    for c in ["auprc", "precision_at_thr", "best_f1"]:
        summary[f"{c}_min"] = g[c].min()
        summary[f"{c}_max"] = g[c].max()
    summary["reps"] = g.size()
    summary.reset_index().to_csv(args.out, index=False)
    print(summary.reset_index().to_string(index=False))
    print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()
