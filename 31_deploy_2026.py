"""Deploy the retrained model (results_retrain2026 verdict: "retraining helps").

Builds results_deploy2026/, a results directory with the same layout the dashboard
(13), the agent (14) and build_dashboard_db.py read, but populated by the retrained
model A (models_retrain2026/A) on the newest data, 2026-01-01 to 2026-06-30. Model A
was trained on data to 2025-06-16, so these are genuine out-of-sample scores; it is
never applied to the 2024-25 test period, part of which it was trained on.

models/ and results/ are left untouched: they hold the frozen model and the paper's
evaluation, which stay reproducible.

Stages (run in order, or `all`):
  score      ensemble scores for every 2026 row -> risk_tiers_test.csv, hybrid_test_scores.csv
  metrics    lgbm/hybrid metrics, tier accuracy, cutoffs, deployment.json
             (labelled rows only: to 2026-06-16, the last date with 14 days of follow-up)
  policy     adopted alert policy (daily top-k, results_alerts2026) into deployment.json
  urgency    fleet urgency from the P-A survival model (Stage 23 method)
  explain    TreeSHAP for every at-risk drive + Stage 12 mitigation mapping
  dashboard  build_dashboard_db.py on results_deploy2026

    python 31_deploy_2026.py all
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

FEATURES = Path("data/processed/fresh2026/features")
MODEL_DIR = Path("models_retrain2026/A")
DECISION = Path("results_retrain2026/results.json")
OUT = Path("results_deploy2026")
WORK = OUT / "_work"
LABEL_END = "2026-06-16"
TARGET = "failure_within_14d"
BATCH = 500_000


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config() -> dict:
    cfg = json.load(open(MODEL_DIR / "config.json"))
    fit = json.load(open(DECISION))
    assert fit["verdict"] == "retraining helps" and fit["retrained_candidate"] == "A", fit["verdict"]
    cfg["threshold"] = fit["fitted_on_validation"]["A"]["threshold"]
    cfg["cut"] = fit["fitted_on_validation"]["A"]["cut"]
    return cfg


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(Path.home() / 'duck_spool').as_posix()}'")
    return con


# ---------------------------------------------------------------------- score
def score() -> None:
    cfg = _config()
    feats = cfg["feature_cols"]
    boosters = [lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_model_seed{s}.txt")) for s in cfg["seeds"]]
    (WORK / "scores").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for f in sorted(FEATURES.glob("part_*.parquet")):
        dst = WORK / "scores" / f.name
        if dst.exists():
            continue
        parts = []
        for batch in pq.ParquetFile(f).iter_batches(batch_size=BATCH,
                                                   columns=["serial_number", "date", TARGET] + feats):
            df = batch.to_pandas()
            X = df[feats].astype("float32").to_numpy()
            p = np.zeros(len(X))
            for b in boosters:
                p += b.predict(X, num_threads=0)
            parts.append(pd.DataFrame({"serial_number": df["serial_number"].astype(str),
                                       "date": pd.to_datetime(df["date"]).dt.date,
                                       "p": (p / len(boosters)).astype("float32"),
                                       "y": df[TARGET].astype("int8")}))
        pd.concat(parts, ignore_index=True).to_parquet(dst, index=False)
        print(f"[SCORE] {f.name} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    hi, med = cfg["cut"]["high_cutoff"], cfg["cut"]["medium_cutoff"]
    tier = f"CASE WHEN p >= {hi} THEN 'HIGH' WHEN p >= {med} THEN 'MEDIUM' ELSE 'LOW' END"
    src = f"read_parquet('{(WORK / 'scores').as_posix()}/*.parquet')"
    con = _duck()
    con.execute(f"""COPY (SELECT serial_number, date, p AS hybrid_score, {tier} AS risk_tier, y AS target
                          FROM {src} ORDER BY serial_number, date)
                    TO '{(OUT / 'risk_tiers_test.csv').as_posix()}' (HEADER, DELIMITER ',')""")
    # same columns as Stage 8's export; the LSTM and fusion columns are not produced
    # (the shipped strategy is lgbm_only and the LSTM was not retrained)
    con.execute(f"""COPY (SELECT serial_number, date, p AS lgbm_prob, CAST(NULL AS FLOAT) AS lstm_rul_pred,
                                 CAST(NULL AS FLOAT) AS hybrid_weighted, CAST(NULL AS FLOAT) AS hybrid_meta,
                                 p AS hybrid_score, y AS target
                          FROM {src} ORDER BY serial_number, date)
                    TO '{(OUT / 'hybrid_test_scores.csv').as_posix()}' (HEADER, DELIMITER ',')""")
    # drive -> model lookup for build_dashboard_db.py (latest row per drive)
    con.execute(f"""COPY (SELECT serial_number, max(CAST(date AS DATE)) AS date, arg_max(model, date) AS model
                          FROM read_parquet('{FEATURES.as_posix()}/*.parquet') GROUP BY 1)
                    TO '{(WORK / 'drive_meta.parquet').as_posix()}' (FORMAT parquet)""")
    n, d, last = con.execute(f"SELECT count(*), count(DISTINCT serial_number), max(date) FROM {src}").fetchone()
    print(f"[SCORE] {n:,} rows, {d:,} drives, latest {last} -> {OUT}", flush=True)


# -------------------------------------------------------------------- metrics
def metrics() -> None:
    cfg = _config()
    thr, hi, med = cfg["threshold"], cfg["cut"]["high_cutoff"], cfg["cut"]["medium_cutoff"]
    con = _duck()
    r = con.execute(f"""SELECT y, p FROM read_parquet('{(WORK / 'scores').as_posix()}/*.parquet')
                        WHERE date <= DATE '{LABEL_END}'""").fetchnumpy()
    y, p = np.asarray(r["y"]), np.asarray(r["p"])
    auprc, auc = average_precision_score(y, p), roc_auc_score(y, p)
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int(y.sum()) - tp, int((y == 0).sum()) - fp
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    pd.DataFrame([{"roc_auc": auc, "auprc": auprc, "f1": 2 * prec * rec / (prec + rec), "precision": prec,
                   "recall": rec, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": thr}]
                 ).to_csv(OUT / "lgbm_metrics.csv", index=False)
    val = json.load(open(DECISION))["val_auprc"]["A"]
    pd.DataFrame([{"chosen_strategy": "lgbm_only", "test_auprc": auprc, "test_roc_auc": auc,
                   "val_auprc_used_for_selection": val, "val_select_auprc_lgbm_only": val}]
                 ).to_csv(OUT / "hybrid_metrics.csv", index=False)

    # row-level tier capture, same layout as Stage 9's tier_accuracy_test.csv
    tiers = np.where(p >= hi, "HIGH", np.where(p >= med, "MEDIUM", "LOW"))
    rows, cum, total = [], 0, int(y.sum())
    for t in ("HIGH", "MEDIUM", "LOW"):
        m = tiers == t
        k = int(y[m].sum())
        cum += k
        rows.append({"tier": t, "total_drives": int(m.sum()), "true_failures_in_tier_alone": k,
                     "share_of_failures_in_tier_alone": k / total, "cumulative_failures_captured": cum,
                     "cumulative_share_captured": cum / total})
    pd.DataFrame(rows).to_csv(OUT / "tier_accuracy_test.csv", index=False)
    (OUT / "tier_cutoffs.json").write_text(json.dumps(cfg["cut"], indent=2))
    (OUT / "deployment.json").write_text(json.dumps({
        "model": "LightGBM 5-seed ensemble, retrained (models_retrain2026/A)",
        "train_window": cfg["train_window"],
        "threshold": thr, "tier_cutoffs": cfg["cut"],
        "scored_window": ["2026-01-01", "2026-06-30"],
        "metrics_window": ["2026-01-01", LABEL_END],
        "decision": str(DECISION), "replaces": "models/ (frozen, trained to 2024-03-31)",
    }, indent=2))
    print(f"[METRICS] 2026 to {LABEL_END}: AUPRC {auprc:.4f}, ROC {auc:.3f}, P/R at {thr} "
          f"{prec:.3f}/{rec:.3f}, HIGH capture {rows[0]['cumulative_share_captured']:.3f}", flush=True)


# --------------------------------------------------------------------- policy
ALERTS = Path("results_alerts2026")


def policy() -> None:
    """Record the adopted alert policy (32_alert_policies.py) in deployment.json:
    the daily budget k and the budget curve measured on 2026, shown by the dashboard."""
    res = json.loads((ALERTS / "results.json").read_text())
    if res["adopted"] != "K":
        print(f"[POLICY] adopted policy is {res['adopted']}; dashboard keeps the threshold queue")
        return
    curve = pd.read_csv(ALERTS / "topk_grid_test.csv")
    path = OUT / "deployment.json"
    info = json.loads(path.read_text())
    info["alert_policy"] = {
        "rule": "daily_top_k", "k": int(res["k_star"]),
        "decision": str(ALERTS / "results.json"),
        "budget_curve_2026": curve.to_dict(orient="records"),
    }
    path.write_text(json.dumps(info, indent=2))
    print(f"[POLICY] daily top-{res['k_star']} recorded in {path}", flush=True)


# -------------------------------------------------------------------- urgency
def urgency() -> None:
    import xgboost as xgb
    feats = json.load(open("models/lgbm_feature_cols.json"))["feature_cols"]
    con = _duck()
    cols = ", ".join(f'"{c}"' for c in feats)
    latest = con.execute(f"""
        SELECT serial_number, CAST(date AS DATE) AS date, {cols} FROM read_parquet('{FEATURES.as_posix()}/*.parquet')
        QUALIFY row_number() OVER (PARTITION BY serial_number ORDER BY date DESC) = 1""").fetchdf()
    bst = xgb.Booster(model_file="models_fusion/pa_xgb_aft.json")
    x = np.clip(bst.inplace_predict(latest[feats].astype("float32").to_numpy()) - 1.0, 0.0, None)
    # same reference fleet and definition as Stage 23 (the survival model is unchanged)
    s23 = _load("s23", "23_pa_urgency.py")
    ref = np.sort(s23.latest_per_drive(con, Path("results_fusion/pa_xgb_aft_val_preds.csv"))["aft_rul_pred"].to_numpy())
    shorter, not_longer = np.searchsorted(ref, x, side="left"), np.searchsorted(ref, x, side="right")
    out = pd.DataFrame({"serial_number": latest["serial_number"], "date": latest["date"].astype(str),
                        "aft_rul_pred": x.astype("float32"),
                        "fleet_urgency_pct": 100.0 * (len(ref) - not_longer + 0.5 * (not_longer - shorter)) / len(ref)})
    out.to_csv(OUT / "pa_urgency_test.csv", index=False, lineterminator="\n")
    q = np.percentile(out["fleet_urgency_pct"], [10, 50, 90])
    print(f"[URGENCY] {len(out):,} drives; p10/p50/p90 {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}", flush=True)


# -------------------------------------------------------------------- explain
def explain() -> None:
    import shap
    cfg = _config()
    feats = cfg["feature_cols"]
    s11 = _load("s11", "11_shap_explainability.py")
    # Stage 11 explains the seed-42 booster; the same here for the retrained model
    explainer = shap.TreeExplainer(lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model_seed42.txt")))
    at_risk = s11.sample_at_risk_rows(Path(f"{FEATURES.as_posix()}/*.parquet"), OUT / "risk_tiers_test.csv",
                                      feats, max_rows=100_000)
    sv = explainer.shap_values(at_risk[feats].astype("float32"))
    sv = sv[1] if isinstance(sv, list) else sv
    df = pd.DataFrame(sv, columns=[f"shap_{c}" for c in feats])
    df["serial_number"], df["date"] = at_risk["serial_number"].values, at_risk["date"].values
    df.to_parquet(OUT / "shap_values_at_risk.parquet", index=False)
    top = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False).head(15)
    top.rename_axis("feature").reset_index(name="mean_abs_shap").to_csv(OUT / "shap_top_features.csv", index=False)
    print(f"[EXPLAIN] SHAP for {len(df):,} at-risk drives; top: {', '.join(top.index[:5])}", flush=True)
    subprocess.run([sys.executable, "12_mitigation_mapping.py", "--risk_tiers", str(OUT / "risk_tiers_test.csv"),
                    "--shap_values", str(OUT / "shap_values_at_risk.parquet"),
                    "--out", str(OUT / "mitigation_mapping.json")], check=True)


def dashboard() -> None:
    subprocess.run([sys.executable, "build_dashboard_db.py", "--results_dir", str(OUT),
                    "--features_path", str(WORK / "drive_meta.parquet"), "--refresh-serving"], check=True)


def main() -> None:
    stages = {"score": score, "metrics": metrics, "policy": policy, "urgency": urgency, "explain": explain,
              "dashboard": dashboard}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(stages) + ["all"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name in (stages if args.stage == "all" else [args.stage]):
        stages[name]()


if __name__ == "__main__":
    main()
