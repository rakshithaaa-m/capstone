"""Drift monitor with a retraining flag.

Each month the shipped model is compared with its own validation period (reference):
  alert_ratio  share of drive-days at or above the alert threshold / reference share
               (label-free: available the same day)
  psi          population stability index of the score distribution (10 reference deciles;
               label-free)
  auprc_ratio  AUPRC over the trailing three months / reference AUPRC (needs labels, which
               arrive 14 days late, so only rows at least 14 days before the end of the data
               are used; reported once three labelled months exist)

Status per month:
  RETRAIN  alert_ratio >= 1.25, psi >= 0.25 or auprc_ratio <= 0.75
  WARN     alert_ratio >= 1.10, psi >= 0.10 or auprc_ratio <= 0.85
  OK       otherwise
The limits are conventional (PSI 0.10 / 0.25) or round numbers; they were set after
the frozen model's ageing had been seen (FINDINGS_2026.md), so the frozen-model replay
below is illustrative, not a pre-registered test. A first version used single-month
AUPRC; monthly AUPRC swings 0.6-1.4x without drift (e.g. the January holiday month), so
it raised RETRAIN on noise and was replaced by the pooled three-month window.

Two runs:
  frozen    archive/model1/models (threshold 0.62), reference 2024-04..09, Oct 2024..Jun 2026 replayed
  deployed  models_retrain2026/A (threshold 0.73), reference 2025-07..12-17, 2026 monitored;
            written to results_deploy2026/drift_*.json|csv for the dashboard

    python 33_drift_monitor.py
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

OUT = Path("archive/model1/results_drift")
DEPLOY = Path("results_deploy2026")
DATA_END = "2026-06-30"
LABEL_LAG_DAYS = 14
LIMITS = {"RETRAIN": {"alert_ratio": 1.25, "psi": 0.25, "auprc_ratio": 0.75},
          "WARN": {"alert_ratio": 1.10, "psi": 0.10, "auprc_ratio": 0.85}}

RUNS = {
    "frozen": {
        "threshold": 0.62,
        "reference": "SELECT CAST(date AS DATE) d, lgbm_prob p, failure_within_14d y "
                     "FROM read_csv('archive/model1/results/lgbm_val_probs.csv')",
        "monitor": "SELECT CAST(date AS DATE) d, lgbm_prob p, failure_within_14d y "
                   "FROM read_csv('archive/model1/results/lgbm_test_probs.csv') UNION ALL "
                   "SELECT CAST(date AS DATE) d, prob p, failure_within_14d y "
                   "FROM read_parquet('data/processed/fresh2026/scores/*.parquet')",
    },
    "deployed": {
        "threshold": 0.73,
        "reference": "SELECT CAST(date AS DATE) d, p_A p, y "
                     "FROM read_parquet('data/processed/retrain2026/preds_val_A.parquet')",
        "monitor": "SELECT date d, p, y FROM read_parquet('results_deploy2026/_work/scores/*.parquet')",
    },
}


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(Path.home() / 'duck_spool').as_posix()}'")
    return con


def status(row: dict) -> str:
    for level in ("RETRAIN", "WARN"):
        lim = LIMITS[level]
        if (row["alert_ratio"] >= lim["alert_ratio"] or row["psi"] >= lim["psi"]
                or (not np.isnan(row["auprc_ratio"]) and row["auprc_ratio"] <= lim["auprc_ratio"])):
            return level
    return "OK"


def run(name: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    con = _duck()
    thr = cfg["threshold"]
    con.execute(f"CREATE TEMP TABLE ref AS {cfg['reference']}")
    edges = np.unique(con.execute(
        "SELECT quantile_cont(p, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) FROM ref").fetchone()[0])
    cut = " ".join(f"WHEN p < {e} THEN {i}" for i, e in enumerate(edges))
    bin_expr = f"CASE {cut} ELSE {len(edges)} END"
    ref_rate, = con.execute(f"SELECT avg(CASE WHEN p >= {thr} THEN 1.0 ELSE 0 END) FROM ref").fetchone()
    r = con.execute("SELECT y, p FROM ref").fetchnumpy()
    ref_auprc = average_precision_score(r["y"], r["p"])
    ref_bins = con.execute(f"SELECT {bin_expr} b, count(*) FROM ref GROUP BY 1 ORDER BY 1").fetchnumpy()
    ref_share = ref_bins["count_star()"] / ref_bins["count_star()"].sum()

    con.execute(f"CREATE TEMP TABLE mon AS SELECT d, p, y, strftime(d, '%Y-%m') m, {bin_expr} b FROM ({cfg['monitor']})")
    label_end = (pd.Timestamp(DATA_END) - pd.Timedelta(days=LABEL_LAG_DAYS)).date()
    rows, done = [], []
    for (m,) in con.execute("SELECT DISTINCT m FROM mon ORDER BY m").fetchall():
        n, rate = con.execute(f"SELECT count(*), avg(CASE WHEN p >= {thr} THEN 1.0 ELSE 0 END) FROM mon WHERE m = ?",
                              [m]).fetchone()
        b = con.execute("SELECT b, count(*) c FROM mon WHERE m = ? GROUP BY 1", [m]).fetchdf().set_index("b")["c"]
        share = b.reindex(range(len(ref_share)), fill_value=0).to_numpy() / n
        psi = float(np.sum((share - ref_share) * np.log(np.maximum(share, 1e-6) / ref_share)))
        lab = con.execute("SELECT y, p FROM mon WHERE m = ? AND d <= ?", [m, label_end]).fetchnumpy()
        labelled = len(lab["y"]) / n
        auprc = average_precision_score(lab["y"], lab["p"]) if labelled >= 0.5 and lab["y"].sum() > 0 else np.nan
        if labelled >= 0.5:
            done.append(m)
        auprc_3m = np.nan
        if len(done) >= 3 and done[-1] == m:
            w = con.execute(f"SELECT y, p FROM mon WHERE m IN ({', '.join('?' * 3)}) AND d <= ?",
                            done[-3:] + [label_end]).fetchnumpy()
            auprc_3m = average_precision_score(w["y"], w["p"])
        row = {"month": m, "drive_days": n, "alert_rate": rate, "alert_ratio": rate / ref_rate, "psi": psi,
               "labelled_share": labelled, "auprc_month": auprc, "auprc_3m": auprc_3m,
               "auprc_ratio": auprc_3m / ref_auprc}
        row["status"] = status(row)
        rows.append(row)
    df = pd.DataFrame(rows)
    flagged = df.loc[df["status"] == "RETRAIN", "month"]
    summary = {"run": name, "threshold": thr, "reference_alert_rate": ref_rate, "reference_auprc": ref_auprc,
               "limits": LIMITS, "label_lag_days": LABEL_LAG_DAYS,
               "first_warn": df.loc[df["status"] != "OK", "month"].min() if (df["status"] != "OK").any() else None,
               "first_retrain": flagged.min() if len(flagged) else None,
               "latest_month": df["month"].iloc[-1], "latest_status": df["status"].iloc[-1]}
    print(f"\n[{name.upper()}] reference alert rate {ref_rate:.5f}, AUPRC {ref_auprc:.4f}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"[{name.upper()}] first WARN {summary['first_warn']}, first RETRAIN {summary['first_retrain']}, "
          f"latest {summary['latest_month']} = {summary['latest_status']}", flush=True)
    return df, summary


def main() -> None:
    OUT.mkdir(exist_ok=True)
    summaries = {}
    for name, cfg in RUNS.items():
        df, summary = run(name, cfg)
        df.to_csv(OUT / f"monthly_{name}.csv", index=False)
        summaries[name] = summary
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    # the dashboard reads the deployed model's monitor, plus the frozen replay as context
    pd.read_csv(OUT / "monthly_deployed.csv").to_csv(DEPLOY / "drift_monthly.csv", index=False)
    (DEPLOY / "drift.json").write_text(json.dumps(
        {"deployed": summaries["deployed"], "frozen_replay": summaries["frozen"]}, indent=2, default=str))
    print(f"[SAVE] {OUT}/ and {DEPLOY}/drift_*", flush=True)


if __name__ == "__main__":
    main()
