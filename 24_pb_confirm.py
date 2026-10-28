"""
Stage 24 -- Amendment 03 §7: test confirmation for P-B (tuned XGBoost classifier).

P-B passed the §6 screen (results_fusion/stacking_pb.log): the winning combiner
was ``B. logistic [prob, risk]``, +0.00027 AUPRC, 95% CI [+0.00003, +0.00051].
This script does exactly what §7 fixes in advance, nothing more:

1. partner = the five-seed P-B ensemble (22_train_xgb_clf.py --ensemble);
2. the SINGLE winning combiner is refitted on the FULL validation split, in the
   same drive-level form Stage 16 used (latest row per drive, ever-fail label,
   ``class_weight="balanced"``) -- no re-selection;
3. it scores every test row; primary metric row-level AUPRC (reference
   LightGBM 0.08394), secondary drive-level AUPRC;
4. paired drive-clustered bootstrap, B = 1000, on dAUPRC = fused - LightGBM,
   using the exact weighted-AP routine of 17_ablation_compare.py.

Verdict table (§7): SHIP needs CI lower bound > 0 AND delta >= +0.003.

Usage:
    python 24_pb_confirm.py
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

MARGIN = 0.003
COMBINER = "B. logistic [prob, risk]"


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_combiner(stage16, results_dir: Path, val_csv: Path):
    """Refit combiner B on every validation drive (§7.2)."""
    from sklearn.linear_model import LogisticRegression

    df = stage16.load_split(results_dir, val_csv, "xgb_prob")
    f = stage16.features(df["prob"].to_numpy(), df["rul"].to_numpy(), 14.0, kind="prob")
    X = np.column_stack([f["prob"], f["risk"]])
    y = df["ever_fail"].to_numpy().astype(int)
    m = LogisticRegression(class_weight="balanced", max_iter=1000)
    m.fit(X, y)
    coef = {"intercept": float(m.intercept_[0]), "prob": float(m.coef_[0, 0]),
            "risk": float(m.coef_[0, 1]), "drives": int(len(y)),
            "positives": int(y.sum())}
    print(f"[COMBINER] {COMBINER} refitted on {len(y):,} validation drives "
          f"({int(y.sum()):,} ever-fail): {coef}")
    return coef


def fused_score(coef: dict, prob: np.ndarray, partner: np.ndarray) -> np.ndarray:
    z = (coef["intercept"] + coef["prob"] * prob.astype(np.float64)
         + coef["risk"] * np.clip(partner.astype(np.float64), 0.0, 1.0))
    return 1.0 / (1.0 + np.exp(-z))


def build_cache(lgbm_csv: Path, pb_csv: Path, out: Path, drive_out: Path,
                expect_rows: int, expect_pos: int) -> None:
    import duckdb

    if out.exists() and drive_out.exists():
        print(f"[CACHE] reusing {out}")
        return
    tmp = Path.home() / "duck_spool"
    con = duckdb.connect()
    con.execute("SET threads=3")
    con.execute("SET memory_limit='5GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    a = (f"read_csv('{lgbm_csv.as_posix()}', header=true, columns="
         "{'serial_number':'VARCHAR','date':'DATE','lgbm_prob':'FLOAT',"
         "'failure_within_14d':'TINYINT'})")
    b = (f"read_csv('{pb_csv.as_posix()}', header=true, columns="
         "{'serial_number':'VARCHAR','date':'DATE','xgb_prob':'FLOAT'})")
    try:
        print("[CACHE] drive dimension ...")
        con.execute(f"""
            CREATE TEMP TABLE dim AS
            SELECT serial_number,
                   (row_number() OVER (ORDER BY serial_number) - 1)::INTEGER AS drive_id
            FROM (SELECT DISTINCT serial_number FROM {a})
        """)
        print("[CACHE] joining LightGBM and P-B test predictions ...")
        con.execute(f"""
            COPY (
                SELECT d.drive_id, a.date, a.lgbm_prob AS prob, b.xgb_prob AS partner,
                       a.failure_within_14d AS y
                FROM {a} a
                JOIN {b} b ON a.serial_number = b.serial_number AND a.date = b.date
                JOIN dim d ON a.serial_number = d.serial_number
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        n, pos = con.execute(
            f"SELECT count(*), sum(y) FROM read_parquet('{out.as_posix()}')").fetchone()
        print(f"[CACHE] {n:,} rows, {int(pos):,} positives "
              f"(expected {expect_rows:,} / {expect_pos:,})")
        if n != expect_rows or int(pos) != expect_pos:
            out.unlink()
            raise RuntimeError("join lost or duplicated rows")
        # Drive level, as Stage 16: latest row's scores, ever-fail label.
        con.execute(f"""
            COPY (
                SELECT drive_id, ARG_MAX(prob, date) AS prob,
                       ARG_MAX(partner, date) AS partner, MAX(y) AS ever_fail
                FROM read_parquet('{out.as_posix()}') GROUP BY 1
            ) TO '{drive_out.as_posix()}' (FORMAT PARQUET)
        """)
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--val_preds", default="results_fusion/pb_xgb_clf_ens_val_preds.csv")
    ap.add_argument("--test_preds", default="results_fusion/pb_xgb_clf_ens_test_preds.csv")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results_fusion/pb_confirm.json")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    stage16 = load_module("16_stacking_study.py", "stage16")
    stage17 = load_module("17_ablation_compare.py", "stage17")

    coef = fit_combiner(stage16, results_dir, Path(args.val_preds))

    ref = pd.read_csv(results_dir / "lgbm_metrics.csv").iloc[0]
    expect_pos = int(ref["tp"] + ref["fn"])
    expect_rows = int(ref["tp"] + ref["fn"] + ref["tn"] + ref["fp"])
    cache = Path("results_fusion/_pb_confirm_joined.parquet")
    drive_cache = cache.with_name("_pb_confirm_drives.parquet")
    build_cache(results_dir / "lgbm_test_probs.csv", Path(args.test_preds),
                cache, drive_cache, expect_rows, expect_pos)

    d = pq.read_table(drive_cache).to_pandas()
    d_fused = fused_score(coef, d["prob"].to_numpy(), d["partner"].to_numpy())
    drive_ap_lgbm = float(average_precision_score(d["ever_fail"], d["prob"]))
    drive_ap_fused = float(average_precision_score(d["ever_fail"], d_fused))
    print(f"[DRIVE] {len(d):,} test drives, {int(d['ever_fail'].sum()):,} ever-fail: "
          f"LightGBM {drive_ap_lgbm:.5f}  fused {drive_ap_fused:.5f}  "
          f"delta {drive_ap_fused - drive_ap_lgbm:+.5f}")
    del d, d_fused

    tbl = pq.read_table(cache, columns=["drive_id", "prob", "partner", "y"])
    drive_id = tbl["drive_id"].to_numpy()
    prob = tbl["prob"].to_numpy()
    y = tbl["y"].to_numpy()
    fused = fused_score(coef, prob, tbl["partner"].to_numpy())
    del tbl
    gc.collect()

    ap_lgbm = float(average_precision_score(y, prob))
    ap_fused = float(average_precision_score(y, fused))
    roc_lgbm = float(roc_auc_score(y, prob))
    roc_fused = float(roc_auc_score(y, fused))
    delta = ap_fused - ap_lgbm
    print(f"[ROW] LightGBM AUPRC {ap_lgbm:.6f}  ROC {roc_lgbm:.6f}")
    print(f"[ROW] fused    AUPRC {ap_fused:.6f}  ROC {roc_fused:.6f}")
    print(f"[ROW] delta {delta:+.6f} ({100 * delta / ap_lgbm:+.2f}% relative)")

    gaps = stage17.bootstrap_delta(fused, prob, y, drive_id, args.boot)
    pd.DataFrame({"delta": gaps}).to_csv(
        Path(args.out).with_name("pb_confirm_bootstrap.csv"), index=False)
    lo, hi = (float(v) for v in np.percentile(gaps, [2.5, 97.5]))
    print(f"[BOOT] delta 95% CI [{lo:+.6f}, {hi:+.6f}]  B={len(gaps)}  "
          f"fused > LightGBM in {100 * float((gaps > 0).mean()):.1f}%")

    if hi < 0:
        verdict = "HARM"
    elif lo <= 0:
        verdict = "NULL"
    elif delta >= MARGIN:
        verdict = "SHIP"
    else:
        verdict = "REAL BUT IMMATERIAL"
    print(f"[VERDICT] {verdict}")

    Path(args.out).write_text(json.dumps({
        "combiner": COMBINER, "coefficients": coef,
        "row_auprc_lgbm": ap_lgbm, "row_auprc_fused": ap_fused,
        "row_roc_lgbm": roc_lgbm, "row_roc_fused": roc_fused,
        "row_delta": delta, "ci95": [lo, hi], "boot": int(len(gaps)),
        "drive_auprc_lgbm": drive_ap_lgbm, "drive_auprc_fused": drive_ap_fused,
        "margin": MARGIN, "verdict": verdict,
    }, indent=2))
    print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()
