"""Full evaluation of Model 2 (the retrained, shipped model: models_retrain2026/A) on its
held-out test period, 2026-01-01 to 2026-06-16 -- every analysis reported for Model 1.

Model 2 was trained on data to 2025-06-16; threshold 0.73 and tier cutoffs were fitted on
2025-07-01..12-17 (results_retrain2026). Scores come from 31_deploy_2026.py.

Stages (or `all`):
  export     test CSV and bootstrap cache in the formats the Model 1 scripts read
  headline   AUPRC/ROC/lift/precision/recall; full drive-bootstrap CI (26); drive-level CI
  inflation  evaluation-protocol table (27, threshold 0.73)
  tiers      drive-level tier table, row-level tier capture
  leadtime   share flagged 30/14/7/1 days before failure (18, threshold 0.73)
  calib      Brier and ECE (19)
  seeds      single-seed AUPRC
  logreg     logistic-regression baseline trained on Model 2's training rows
  urgency    survival-model C-index on each drive's first 2026 row; urgency by tier
  shap       top features by mean |SHAP| on a random 3,000-row sample
  agent      13 agent invariants on every at-risk 2026 drive (25 policy mode)

    python 35_model2_eval.py all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path("results_model2")
WORK = OUT / "_work"
SCORES = "results_deploy2026/_work/scores/*.parquet"
FEATURES = Path("data/processed/fresh2026/features")
MODEL_DIR = Path("models_retrain2026/A")
TEST_END = "2026-06-16"
SEED = 20261104


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(Path.home() / 'duck_spool').as_posix()}'")
    return con


def _cfg() -> dict:
    cfg = json.load(open(MODEL_DIR / "config.json"))
    fit = json.load(open("results_retrain2026/results.json"))["fitted_on_validation"]["A"]
    return {**cfg, "threshold": fit["threshold"], "cut": fit["cut"]}


def _save(name: str, obj) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))
    print(f"[{name.upper()}] {json.dumps(obj, default=float)[:600]}", flush=True)


def _run(args: list[str], log: str) -> None:
    with open(OUT / log, "w") as fh:
        subprocess.run([sys.executable, "-u", *args], check=True, stdout=fh, stderr=subprocess.STDOUT)


# ------------------------------------------------------------------- export
def export() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    con = _duck()
    src = f"(SELECT serial_number, date, p, y FROM read_parquet('{SCORES}') WHERE date <= DATE '{TEST_END}')"
    con.execute(f"""COPY (SELECT serial_number, date, p AS lgbm_prob, y AS failure_within_14d FROM {src}
                          ORDER BY serial_number, date)
                    TO '{(WORK / 'test2026.csv').as_posix()}' (HEADER, DELIMITER ',')""")
    # drive_id = rank of serial_number - 1, as the Model 1 bootstrap cache
    con.execute(f"""COPY (SELECT CAST(dense_rank() OVER (ORDER BY serial_number) - 1 AS INTEGER) AS drive_id,
                                 date, p AS prob, p AS partner, CAST(y AS TINYINT) AS y FROM {src}
                          ORDER BY drive_id, date)
                    TO '{(WORK / 'cache2026.parquet').as_posix()}' (FORMAT parquet)""")
    n, pos, d = con.execute(f"SELECT count(*), sum(y), count(DISTINCT serial_number) FROM {src}").fetchone()
    print(f"[EXPORT] {n:,} rows, {pos:,} positives, {d:,} drives -> {WORK}", flush=True)


# ------------------------------------------------------------------ headline
def headline() -> None:
    thr = _cfg()["threshold"]
    t = pq.read_table(WORK / "cache2026.parquet", columns=["drive_id", "prob", "y"])
    p, y, drive = t["prob"].to_numpy(), t["y"].to_numpy(), t["drive_id"].to_numpy()
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    prev = y.mean()
    res = {"rows": len(y), "positives": int(y.sum()), "drives": int(drive.max()) + 1, "prevalence": prev,
           "auprc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
           "threshold": thr, "precision_at_thr": tp / (tp + fp), "recall_at_thr": tp / y.sum()}
    res["lift"] = res["auprc"] / prev
    # drive level: latest row per drive vs ever positive in the window (as Model 1, 14b)
    df = pd.DataFrame({"d": drive, "p": p, "y": y})
    last = df.groupby("d").tail(1).set_index("d")["p"]   # cache rows are in date order per drive
    ever = df.groupby("d")["y"].max()
    yd, pd_ = ever.to_numpy(), last.reindex(ever.index).to_numpy()
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(200):
        i = rng.integers(0, len(yd), len(yd))
        boots.append((average_precision_score(yd[i], pd_[i]), roc_auc_score(yd[i], pd_[i])))
    boots = np.array(boots)
    res.update({"drive_failing": int(yd.sum()), "drive_auprc": average_precision_score(yd, pd_),
                "drive_auprc_ci": list(np.percentile(boots[:, 0], [2.5, 97.5])),
                "drive_roc_auc": roc_auc_score(yd, pd_),
                "drive_roc_ci": list(np.percentile(boots[:, 1], [2.5, 97.5]))})
    _save("headline", res)
    _run(["26_row_auprc_ci.py", "--cache", str(WORK / "cache2026.parquet"), "--boot", "1000",
          "--out", str(OUT / "row_auprc_ci.csv")], "row_auprc_ci.log")
    print(pd.read_csv(OUT / "row_auprc_ci.csv").to_string(index=False), flush=True)


def inflation() -> None:
    _run(["27_prevalence_inflation.py", "--cache", str(WORK / "cache2026.parquet"), "--threshold",
          str(_cfg()["threshold"]), "--out", str(OUT / "inflation.csv")], "inflation.log")
    print(pd.read_csv(OUT / "inflation.csv").to_string(index=False), flush=True)


# --------------------------------------------------------------------- tiers
def tiers() -> None:
    cut = _cfg()["cut"]
    hi, med = cut["high_cutoff"], cut["medium_cutoff"]
    con = _duck()
    csv = f"read_csv('{(WORK / 'test2026.csv').as_posix()}')"
    drv = con.execute(f"""
        SELECT max(failure_within_14d) AS fail, arg_max(lgbm_prob, date) AS p FROM {csv} GROUP BY serial_number
        """).fetchdf()
    drv["tier"] = np.where(drv.p >= hi, "HIGH", np.where(drv.p >= med, "MEDIUM", "LOW"))
    total_fail = drv["fail"].sum()
    rows = []
    for t in ("HIGH", "MEDIUM", "LOW"):
        g = drv[drv.tier == t]
        rows.append({"tier": t, "drives": len(g), "failing_caught": int(g["fail"].sum()),
                     "share_of_failing": g["fail"].sum() / total_fail, "precision": g["fail"].mean()})
    g = drv[drv.tier != "LOW"]
    rows.insert(2, {"tier": "HIGH+MEDIUM", "drives": len(g), "failing_caught": int(g["fail"].sum()),
                    "share_of_failing": g["fail"].sum() / total_fail, "precision": g["fail"].mean()})
    pd.DataFrame(rows).to_csv(OUT / "tiers_drive.csv", index=False)
    cap = con.execute(f"""SELECT avg(CASE WHEN lgbm_prob >= {hi} THEN 1.0 ELSE 0 END),
                                 avg(CASE WHEN lgbm_prob >= {med} THEN 1.0 ELSE 0 END)
                          FROM {csv} WHERE failure_within_14d = 1""").fetchone()
    _save("tiers", {"drive_table": rows, "row_capture_high": cap[0], "row_capture_high_medium": cap[1],
                    "high_share_of_drives": rows[0]["drives"] / len(drv)})


def leadtime() -> None:
    _run(["18_leadtime_curve.py", "--test", str(WORK / "test2026.csv"), "--threshold", str(_cfg()["threshold"]),
          "--out", str(OUT / "leadtime.csv")], "leadtime.log")
    print(pd.read_csv(OUT / "leadtime.csv").to_string(index=False), flush=True)


def calib() -> None:
    _run(["19_calibration.py", "--test", str(WORK / "test2026.csv"), "--out", str(OUT / "calibration_bins.csv"),
          "--out_summary", str(OUT / "calibration.csv")], "calibration.log")
    print(pd.read_csv(OUT / "calibration.csv").to_string(index=False), flush=True)


# --------------------------------------------------------------------- seeds
def _stream_2026(cols: list[str]):
    """Batches of the 2026 feature rows up to TEST_END (labels complete)."""
    for f in sorted(FEATURES.glob("part_*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=500_000, columns=["date", "failure_within_14d"] + cols):
            df = b.to_pandas()
            df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(TEST_END)]
            if len(df):
                yield df


def seeds() -> None:
    cfg = _cfg()
    feats = cfg["feature_cols"]
    boosters = {s: lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_model_seed{s}.txt")) for s in cfg["seeds"]}
    preds = {s: [] for s in boosters}
    ys = []
    t0 = time.time()
    for df in _stream_2026(feats):
        X = df[feats].astype("float32").to_numpy()
        for s, b in boosters.items():
            preds[s].append(b.predict(X, num_threads=0).astype("float32"))
        ys.append(df["failure_within_14d"].to_numpy(np.int8))
    y = np.concatenate(ys)
    per = {str(s): average_precision_score(y, np.concatenate(v)) for s, v in preds.items()}
    _save("seeds", {"per_seed_auprc": per, "min": min(per.values()), "max": max(per.values()),
                    "mean": float(np.mean(list(per.values()))), "minutes": (time.time() - t0) / 60})


# -------------------------------------------------------------------- logreg
def logreg() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feats = _cfg()["feature_cols"]
    tr = pd.read_parquet("data/processed/retrain2026/train_resampled_A.parquet", columns=feats + ["failure_within_14d"])
    scaler = StandardScaler()
    X = scaler.fit_transform(tr[feats].astype("float32").fillna(0.0))
    model = LogisticRegression(class_weight="balanced", max_iter=1000)   # as Model 1's baseline (fix2)
    model.fit(X, tr["failure_within_14d"].to_numpy())
    del tr, X
    ps, ys = [], []
    for df in _stream_2026(feats):
        ps.append(model.predict_proba(scaler.transform(df[feats].astype("float32").fillna(0.0)))[:, 1])
        ys.append(df["failure_within_14d"].to_numpy(np.int8))
    y, p = np.concatenate(ys), np.concatenate(ps)
    _save("logreg", {"auprc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)})


# ------------------------------------------------------------------- urgency
def urgency() -> None:
    import xgboost as xgb
    s21 = _load("s21", "archive/fusion_study/21_rul_compare.py")
    feats = json.load(open("models/lgbm_feature_cols.json"))["feature_cols"]
    con = _duck()
    cols = ", ".join(f'"{c}"' for c in feats)
    # as Model 1 (21_rul_compare): each drive's first usable row; failing drives have an
    # event at rul_days, surviving drives are censored at the days they are known alive
    first = con.execute(f"""
        WITH ok AS (SELECT * FROM read_parquet('{FEATURES.as_posix()}/*.parquet')
                    WHERE CAST(date AS DATE) <= DATE '{TEST_END}' AND (rul_days IS NULL OR rul_days >= 0)),
             span AS (SELECT serial_number, max(CAST(date AS DATE)) AS last_d FROM ok GROUP BY 1)
        SELECT ok.serial_number,
               CASE WHEN ok.rul_days IS NOT NULL THEN ok.rul_days
                    ELSE date_diff('day', CAST(ok.date AS DATE), span.last_d) END AS t,
               CASE WHEN ok.rul_days IS NOT NULL THEN 1 ELSE 0 END AS event, {cols}
        FROM ok JOIN span USING (serial_number)
        QUALIFY row_number() OVER (PARTITION BY ok.serial_number ORDER BY ok.date) = 1""").fetchdf()
    bst = xgb.Booster(model_file="models_fusion/pa_xgb_aft.json")
    pred = np.clip(bst.inplace_predict(first[feats].astype("float32").to_numpy()) - 1.0, 0.0, None)
    c = s21.harrell_c(first["t"].to_numpy(float), first["event"].to_numpy(int), pred)
    db = duckdb.connect("results_deploy2026/dashboard.duckdb", read_only=True)
    by_tier = dict(db.execute("SELECT risk_tier, avg(fleet_urgency_pct) FROM drive_status GROUP BY 1").fetchall())
    _save("urgency", {"drives": len(first), "failing": int(first["event"].sum()), "c_index": c,
                      "mean_urgency_by_tier": by_tier})


# ---------------------------------------------------------------------- shap
def shap_top() -> None:
    import shap
    feats = _cfg()["feature_cols"]
    con = _duck()
    sample = con.execute(f"""SELECT {", ".join(f'"{c}"' for c in feats)}
                             FROM read_parquet('{FEATURES.as_posix()}/*.parquet')
                             WHERE CAST(date AS DATE) <= DATE '{TEST_END}'
                             USING SAMPLE 3000 ROWS (reservoir, 42)""").fetchdf()
    sv = shap.TreeExplainer(lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model_seed42.txt"))).shap_values(
        sample[feats].astype("float32"))
    sv = sv[1] if isinstance(sv, list) else sv
    top = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False).head(15)
    top.rename_axis("feature").reset_index(name="mean_abs_shap").to_csv(OUT / "shap_top_features.csv", index=False)
    print(top.head(8).to_string(), flush=True)


def agent() -> None:
    _run(["25_agent_eval.py", "policy", "--results_dir", "results_deploy2026", "--out_dir",
          str(OUT / "agent_eval")], "agent_policy.log")
    print(pd.read_csv(OUT / "agent_eval" / "policy_summary.csv").to_string(index=False), flush=True)


STAGES = {"export": export, "headline": headline, "inflation": inflation, "tiers": tiers, "leadtime": leadtime,
          "calib": calib, "seeds": seeds, "logreg": logreg, "urgency": urgency, "shap": shap_top, "agent": agent}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name in (STAGES if args.stage == "all" else [args.stage]):
        t0 = time.time()
        STAGES[name]()
        print(f"[DONE] {name} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
