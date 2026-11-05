"""
Stage 28 -- fresh out-of-time evaluation: score Backblaze Q1-Q2 2026 with the
frozen LightGBM ensemble.  Plan (written before scoring): results_fresh2026/PLAN.md

Rebuilding Stages 2-3 over all 2013-2026 data is not needed and would re-make
the global near-empty-column decision.  Instead this script extends the
existing pipeline forward:

  first_seen  each drive's first date in the full cleaned history (for
              days_since_install, which Stage 3 measures from the first row)
  partition   cleaned Q4 2025 rows (history) + raw 2026 CSVs cast to the
              cleaned Q4 2025 schema, hash-partitioned by drive (DuckDB, out of core)
  features    per partition: Stage 2 cleaning (dedup, per-drive within-quarter
              forward fill) and Stage 3 features/labels, imported from those
              scripts; then scored by the five frozen boosters (mean probability)
  validate    recomputes December 2025 features and compares them with the
              original features_test.parquet rows (same serial and date)
  evaluate    the metrics listed in PLAN.md

Usage:
    python 28_fresh_2026.py first_seen
    python 28_fresh_2026.py partition
    python 28_fresh_2026.py features [--worker i --n_workers k]
    python 28_fresh_2026.py validate
    python 28_fresh_2026.py evaluate
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CLEANED = Path("data/processed/cleaned_by_quarter")
HISTORY = CLEANED / "cleaned_2025_Q4.parquet"
RAW_GLOB = "data/raw/data_Q*_2026/**/*.csv"
WORK = Path("data/processed/fresh2026")
OUT = Path("archive/model1/results_fresh2026")
N_PARTS = 60
FIRST_2026 = pd.Timestamp("2026-01-01")
EVAL_END = pd.Timestamp("2026-06-16")   # last day with a full 14-day follow-up
HORIZON = 14
TARGET = f"failure_within_{HORIZON}d"
SEED = 42


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _duck() -> duckdb.DuckDBPyConnection:
    spool = Path.home() / "duck_spool"
    spool.mkdir(exist_ok=True)
    return duckdb.connect(config={"memory_limit": "6GB", "temp_directory": str(spool),
                                  "preserve_insertion_order": False})


# --------------------------------------------------------------------- stages
def first_seen() -> None:
    out = WORK / "first_seen.parquet"
    con = _duck()
    con.execute(f"""
        COPY (
          SELECT serial_number, min(date) AS first_date FROM (
            SELECT serial_number, CAST(date AS DATE) AS date
              FROM read_parquet('{CLEANED.as_posix()}/cleaned_*.parquet', union_by_name=true)
            UNION ALL
            SELECT serial_number, CAST(date AS DATE)
              FROM read_csv('{RAW_GLOB}', all_varchar=true, union_by_name=true)
          ) GROUP BY serial_number
        ) TO '{out.as_posix()}' (FORMAT parquet)""")
    n = con.execute(f"SELECT count(*) FROM '{out.as_posix()}'").fetchone()[0]
    print(f"[FIRST_SEEN] {n:,} drives -> {out}")


def partition() -> None:
    schema = pq.read_schema(HISTORY)
    sel = []
    for f in schema:
        name, t = f.name, str(f.type)
        if name == "date":
            sel.append("CAST(date AS TIMESTAMP) AS date")
        elif name == "failure":
            sel.append("COALESCE(TRY_CAST(failure AS BIGINT), 0) AS failure")
        elif t in ("large_string", "string"):
            sel.append(f'"{name}"')
        else:
            duck = {"int64": "BIGINT", "double": "DOUBLE", "float": "FLOAT", "bool": "BOOLEAN"}[t]
            sel.append(f'TRY_CAST("{name}" AS {duck}) AS "{name}"')
    names = ", ".join(f'"{f.name}"' for f in schema)
    out_dir = WORK / "partitions"
    con = _duck()
    t0 = time.time()
    con.execute(f"""
        COPY (
          SELECT *, CAST(hash(serial_number) % {N_PARTS} AS INTEGER) AS part FROM (
            SELECT {names}, '2025_Q4' AS q FROM read_parquet('{HISTORY.as_posix()}')
            UNION ALL BY NAME
            SELECT {", ".join(sel)},
                   CASE WHEN CAST(date AS DATE) < DATE '2026-04-01' THEN '2026_Q1' ELSE '2026_Q2' END AS q
              FROM read_csv('{RAW_GLOB}', all_varchar=true, union_by_name=true)
          )
        ) TO '{out_dir.as_posix()}' (FORMAT parquet, PARTITION_BY (part), OVERWRITE_OR_IGNORE)""")
    n = con.execute(f"SELECT q, count(*), sum(failure) FROM read_parquet('{out_dir.as_posix()}/*/*.parquet') "
                    f"GROUP BY q ORDER BY q").fetchall()
    print(f"[PARTITION] {n}  ({(time.time() - t0) / 60:.1f} min) -> {out_dir}")


def build_features(df: pd.DataFrame, s03, medians: pd.Series, core: list[str],
                   first: pd.Series) -> pd.DataFrame:
    """Stage 2 cleaning + Stage 3 features for one drive partition."""
    df["date"] = pd.to_datetime(df["date"])
    df["serial_number"] = df["serial_number"].astype(str)
    df["failure"] = df["failure"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["serial_number", "date"], keep="last")
    smart = [c for c in df.columns if c.startswith("smart_")]
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    df[smart] = df.groupby(["serial_number", "q"])[smart].ffill()   # Stage 2: within a quarter
    df = s03.impute_with_train_medians(df, core, medians)
    df = s03.add_failure_date_and_rul(df, HORIZON)
    df = s03.add_rolling_features(df, core)
    df = s03.add_tenure_features(df)
    df["days_since_install"] = (df["date"] - pd.to_datetime(df["serial_number"].map(first))).dt.days
    return df


def _setup():
    s03 = _load("s03", "03_build_targets_and_features.py")
    medians = pd.Series(json.load(open("data/processed/features_by_batch/_train_medians.json")))
    core = list(medians.index)
    first = pd.read_parquet(WORK / "first_seen.parquet").set_index("serial_number")["first_date"]
    feature_cols = json.load(open("models/lgbm_feature_cols.json"))["feature_cols"]
    return s03, medians, core, first, feature_cols


def features(worker: int = 0, n_workers: int = 1) -> None:
    s03, medians, core, first, feature_cols = _setup()
    seeds = json.load(open("models/lgbm_ensemble_seeds.json"))["seeds"]
    boosters = [lgb.Booster(model_file=f"archive/model1/models/lgbm_model_seed{s}.txt") for s in seeds]
    (WORK / "features").mkdir(parents=True, exist_ok=True)
    (WORK / "scores").mkdir(parents=True, exist_ok=True)
    keep = ["serial_number", "date", "model", "datacenter", "failure", "rul_days", TARGET]
    t0 = time.time()
    for p in range(worker, N_PARTS, n_workers):
        score_path = WORK / "scores" / f"part_{p:02d}.parquet"
        if score_path.exists():
            continue
        df = pd.read_parquet(WORK / "partitions" / f"part={p}")
        df = build_features(df, s03, medians, core, first)
        df = df[df["date"] >= FIRST_2026].reset_index(drop=True)
        X = df[feature_cols].astype("float32")
        prob = np.zeros(len(df), dtype=np.float64)
        for b in boosters:
            prob += b.predict(X, num_threads=0)
        df["prob"] = (prob / len(boosters)).astype("float32")
        df[keep + feature_cols].to_parquet(WORK / "features" / f"part_{p:02d}.parquet", index=False)
        tmp = score_path.with_suffix(".tmp")
        df[keep + ["prob"]].to_parquet(tmp, index=False)
        tmp.replace(score_path)
        el = time.time() - t0
        print(f"[FEATURES] part {p + 1}/{N_PARTS}: {len(df):,} rows, {int(df[TARGET].sum())} positives  "
              f"({el / 60:.1f} min)", flush=True)
        del df, X, prob


def validate(parts: int = 2) -> None:
    """December 2025 rows: recomputed features vs the original test features."""
    s03, medians, core, first, feature_cols = _setup()
    rep = []
    for p in range(parts):
        df = pd.read_parquet(WORK / "partitions" / f"part={p}")
        df = df[df["q"] == "2025_Q4"]
        df = build_features(df, s03, medians, core, first)
        df = df[(df["date"] >= "2025-12-01") & (df["date"] <= "2025-12-31")]
        serials = df["serial_number"].unique().tolist()
        con = _duck()
        con.register("s", pd.DataFrame({"serial_number": serials}))
        cols = ", ".join(f'"{c}"' for c in ["serial_number", "date"] + feature_cols)
        orig = con.execute(f"""
            SELECT {cols} FROM read_parquet('data/processed/features_test.parquet')
            WHERE date >= TIMESTAMP '2025-12-01' AND serial_number IN (SELECT serial_number FROM s)
        """).df()
        orig["date"] = pd.to_datetime(orig["date"])
        m = df[["serial_number", "date"] + feature_cols].merge(
            orig, on=["serial_number", "date"], suffixes=("", "_o"))
        for c in feature_cols:
            a = m[c].astype("float64").to_numpy()
            o = m[f"{c}_o"].astype("float64").to_numpy()
            same = np.isclose(a, o, rtol=1e-4, atol=1e-4) | (np.isnan(a) & np.isnan(o))
            rep.append({"part": p, "feature": c, "rows": len(m), "mismatch_share": 1 - same.mean()})
        print(f"[VALIDATE] part {p}: {len(m):,} matched rows of {len(df):,} recomputed")
    r = pd.DataFrame(rep)
    bad = r[r["mismatch_share"] > 0].sort_values("mismatch_share", ascending=False)
    print(f"[VALIDATE] features with any mismatch: {bad['feature'].nunique()} of {len(feature_cols)}")
    print(bad.head(20).to_string(index=False))
    r.to_csv(OUT / "validate_features.csv", index=False)


def evaluate(n_boot: int) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    s17 = _load("s17", "17_ablation_compare.py")
    thr = float(json.load(open("archive/model1/models/lgbm_threshold.json"))["threshold"])
    cut = json.load(open("archive/model1/results/tier_cutoffs.json"))
    con = _duck()
    src = (WORK / "scores").as_posix() + "/*.parquet"
    df = con.execute(f"""
        SELECT serial_number, date, prob, CAST({TARGET} AS TINYINT) AS y
        FROM read_parquet('{src}') WHERE date <= TIMESTAMP '{EVAL_END.date()}'
    """).df()
    df["drive_id"] = df["serial_number"].astype("category").cat.codes.astype(np.int32)
    y = df["y"].to_numpy()
    prob = df["prob"].to_numpy()
    n_pos, n = int(y.sum()), len(y)
    out: dict = {"window": f"2026-01-01..{EVAL_END.date()}", "rows": n, "positives": n_pos,
                 "drives": int(df["drive_id"].max()) + 1, "prevalence": n_pos / n}
    out["auprc"] = float(average_precision_score(y, prob))
    out["roc_auc"] = float(roc_auc_score(y, prob))
    out["lift"] = out["auprc"] / out["prevalence"]
    pred = prob >= thr
    tp = int((pred & (y == 1)).sum())
    out["precision_at_thr"] = tp / max(int(pred.sum()), 1)
    out["recall_at_thr"] = tp / n_pos
    print(f"[EVAL] {out}", flush=True)

    # drive-clustered bootstrap (exact weighted AP)
    order = np.argsort(-prob, kind="stable")
    ys, ss, ds = y[order], prob[order], df["drive_id"].to_numpy()[order]
    nd = int(ds.max()) + 1
    assert abs(s17.weighted_ap(ys, ss, ds, np.ones(nd, dtype=np.int32)) - out["auprc"]) < 1e-9
    rng = np.random.default_rng(SEED)
    reps = np.empty(n_boot)
    t0 = time.time()
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(np.int32)
        reps[b] = s17.weighted_ap(ys, ss, ds, mult)
        if (b + 1) % 100 == 0:
            print(f"[BOOT] {b + 1}/{n_boot} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    out["auprc_ci_lo"], out["auprc_ci_hi"] = (float(v) for v in np.percentile(reps, [2.5, 97.5]))
    out["n_boot"] = n_boot
    del ys, ss, ds, order

    # row-level tier capture
    tier = np.where(prob >= cut["high_cutoff"], "HIGH", np.where(prob >= cut["medium_cutoff"], "MEDIUM", "LOW"))
    rows_t = pd.DataFrame({"tier": tier, "y": y}).groupby("tier")["y"].agg(["size", "sum"])
    rows_t["share_of_positive_rows"] = rows_t["sum"] / n_pos
    rows_t.to_csv(OUT / "tier_rows_2026.csv")

    # drive level: latest record in the window vs ever positive
    last = df.sort_values("date").groupby("drive_id").agg(p=("prob", "last"), ever=("y", "max"))
    out["drive_auprc"] = float(average_precision_score(last["ever"], last["p"]))
    out["drive_roc_auc"] = float(roc_auc_score(last["ever"], last["p"]))
    out["failing_drives"] = int(last["ever"].sum())
    trows = []
    for name, lo, hi in [("HIGH", cut["high_cutoff"], 9), ("MEDIUM", cut["medium_cutoff"], cut["high_cutoff"]),
                         ("HIGH+MEDIUM", cut["medium_cutoff"], 9), ("LOW", -1, cut["medium_cutoff"])]:
        sel = (last["p"] >= lo) & (last["p"] < hi)
        tp_d = int(last.loc[sel, "ever"].sum())
        trows.append({"tier": name, "drives": int(sel.sum()), "failing_caught": tp_d,
                      "share_of_failing": tp_d / out["failing_drives"],
                      "precision": tp_d / max(int(sel.sum()), 1)})
    pd.DataFrame(trows).to_csv(OUT / "tier_drives_2026.csv", index=False)

    # lead time (method of 18_leadtime_curve.py)
    fail = df[df["y"] == 1].groupby("serial_number")["date"].max().rename("fail_date").reset_index()
    idx = df.set_index(["serial_number", "date"])["prob"]
    lead = []
    for L in (30, 14, 7, 1):
        keys = pd.MultiIndex.from_arrays([fail["serial_number"], fail["fail_date"] - pd.Timedelta(days=L)])
        got = idx.reindex(keys).dropna()
        lead.append({"days_before": L, "failing_drives": len(fail), "observed": len(got),
                     "flagged": int((got >= thr).sum()),
                     "share_of_all": float((got >= thr).sum() / len(fail)),
                     "share_of_observed": float((got >= thr).mean()) if len(got) else float("nan")})
    pd.DataFrame(lead).to_csv(OUT / "leadtime_2026.csv", index=False)

    json.dump(out, open(OUT / "metrics_2026.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(pd.DataFrame(trows).to_string(index=False))
    print(pd.DataFrame(lead).to_string(index=False))
    print(rows_t.to_string())


def diagnose() -> None:
    """Post hoc (not in PLAN.md): AUPRC by month, and HIGH-tier precision by drive model,
    marking models absent from the training period."""
    from sklearn.metrics import average_precision_score
    cut = json.load(open("archive/model1/results/tier_cutoffs.json"))["high_cutoff"]
    con = _duck()
    src = (WORK / "scores").as_posix() + "/*.parquet"
    df = con.execute(f"""
        SELECT serial_number, date, model, prob, CAST({TARGET} AS TINYINT) AS y
        FROM read_parquet('{src}') WHERE date <= TIMESTAMP '{EVAL_END.date()}'""").df()
    month = df["date"].dt.to_period("M").astype(str)
    rows = []
    for m, g in df.groupby(month):
        rows.append({"month": m, "rows": len(g), "positives": int(g["y"].sum()),
                     "auprc": average_precision_score(g["y"], g["prob"]),
                     "lift": average_precision_score(g["y"], g["prob"]) / g["y"].mean()})
    pd.DataFrame(rows).to_csv(OUT / "monthly_2026.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    train_models = set(con.execute(
        "SELECT DISTINCT model FROM read_parquet('data/processed/features_train.parquet')").df()["model"])
    last = df.sort_values("date").groupby("serial_number").agg(
        model=("model", "last"), p=("prob", "last"), ever=("y", "max"))
    last["in_train"] = last["model"].isin(train_models)
    high = last[last["p"] >= cut]
    by = high.groupby(["model", "in_train"]).agg(high_drives=("ever", "size"), failing=("ever", "sum"))
    by["precision"] = by["failing"] / by["high_drives"]
    fleet = last.groupby("model").size().rename("fleet_drives")
    by = by.join(fleet, on="model").sort_values("high_drives", ascending=False)
    by.to_csv(OUT / "high_by_model_2026.csv")
    print(by.head(15).to_string())
    for flag, g in last.groupby("in_train"):
        h = g[g["p"] >= cut]
        print(f"[MODEL] in_train={flag}: {len(g):,} drives, {int(g['ever'].sum())} failing, "
              f"{len(h):,} HIGH, HIGH precision {h['ever'].mean() if len(h) else float('nan'):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["first_seen", "partition", "features", "validate", "evaluate", "diagnose"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--worker", type=int, default=0, help="features: this worker's index")
    ap.add_argument("--n_workers", type=int, default=1, help="features: parallel workers")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    {"first_seen": first_seen, "partition": partition, "features": lambda: features(args.worker, args.n_workers),
     "validate": validate, "evaluate": lambda: evaluate(args.boot), "diagnose": diagnose}[args.stage]()


if __name__ == "__main__":
    main()
