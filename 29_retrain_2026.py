"""
Stage 29 -- retrain on recent data and evaluate on 2026 (pre-registered:
results_retrain2026/PREREGISTRATION.md).

Candidates: F (shipped ensemble, frozen), A (same features and hyperparameters,
retrained on 2013-04-10..2025-06-16), B (A + same-model/same-day percentile
features; stage "fleet", added after A).

Stages
  resample   all positive train rows + Bernoulli negatives at 10 per positive (seed 42)
  train      five seeds with models/lgbm_best_params.json (no re-tuning)
  predict    F and A scores on validation (2025-07-01..2025-12-17) and
             test (2026-01-01..2026-06-16)
  fleet_prep, fleet_pct, build_b   arm B inputs (same-model/same-day percentiles)
  decide     (after B) B vs A on validation, then the confirmatory R vs F on test

Usage:
    python 29_retrain_2026.py resample
    python 29_retrain_2026.py train --arm A
    python 29_retrain_2026.py predict --arm A
    python 29_retrain_2026.py fleet_prep
    python 29_retrain_2026.py fleet_pct
    python 29_retrain_2026.py build_b
    python 29_retrain_2026.py train --arm B
    python 29_retrain_2026.py predict --arm B
    python 29_retrain_2026.py decide
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TARGET = "failure_within_14d"
TRAIN_END = pd.Timestamp("2025-06-16")
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-17")
TEST_START, TEST_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-16")
NEG_PER_POS = 10
SEED = 42
KEYS = ["serial_number", "date", "model"]

PROCESSED = Path("data/processed")
TRAIN_SOURCES = [PROCESSED / "features_train.parquet", PROCESSED / "features_val.parquet",
                 PROCESSED / "features_test.parquet"]
FRESH = PROCESSED / "fresh2026"
WORK = PROCESSED / "retrain2026"
MODELS = Path("models_retrain2026")
OUT = Path("results_retrain2026")
BATCH = 500_000


def feature_cols() -> list[str]:
    return json.load(open("models/lgbm_feature_cols.json"))["feature_cols"]


def _frame(batch: pa.RecordBatch, feats: list[str]) -> pd.DataFrame:
    df = batch.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df[feats] = df[feats].astype("float32")
    df[TARGET] = df[TARGET].astype("int8")
    return df


def resample() -> None:
    feats = feature_cols()
    con = duckdb.connect()
    src = " UNION ALL ".join(
        f"SELECT CAST({TARGET} AS TINYINT) y FROM read_parquet('{p.as_posix()}') "
        f"WHERE CAST(date AS DATE) <= DATE '{TRAIN_END.date()}'" for p in TRAIN_SOURCES)
    n_pos, n = con.execute(f"SELECT sum(y), count(*) FROM ({src})").fetchone()
    n_pos, n_neg = int(n_pos), int(n - n_pos)
    keep_prob = NEG_PER_POS * n_pos / n_neg
    print(f"[RESAMPLE] train rows {n:,}, positives {n_pos:,}; negative keep prob {keep_prob:.6f}", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "train_resampled_A.parquet"
    tmp = out.with_suffix(".tmp")
    rng = np.random.default_rng(SEED)
    writer, kept_pos, kept_neg = None, 0, 0
    t0 = time.time()
    for path in TRAIN_SOURCES:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=KEYS + [TARGET] + feats):
            df = _frame(batch, feats)
            df = df[df["date"] <= TRAIN_END]
            if df.empty:
                continue
            pos = df[TARGET].to_numpy() == 1
            keep = pos | (rng.random(len(df)) < keep_prob)
            df = df[keep]
            kept_pos += int(pos.sum())
            kept_neg += int((~pos & keep).sum())
            table = pa.Table.from_pandas(df[KEYS + [TARGET] + feats], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table.cast(writer.schema))
        print(f"[RESAMPLE] {path.name} done: {kept_pos:,} pos + {kept_neg:,} neg "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    writer.close()
    tmp.replace(out)
    print(f"[SAVE] {out}  ({kept_pos:,} positives, {kept_neg:,} negatives)")


def feats_for(arm: str) -> list[str]:
    return feature_cols() + (PCT_COLS if arm == "B" else [])


def train(arm: str) -> None:
    feats = feats_for(arm)
    params = json.load(open("models/lgbm_best_params.json"))
    df = pd.read_parquet(WORK / f"train_resampled_{arm}.parquet", columns=feats + [TARGET])
    X, y = df[feats].astype("float32"), df[TARGET].astype(int)
    del df
    print(f"[TRAIN {arm}] {len(y):,} rows, {int(y.sum()):,} positives, {X.shape[1]} features", flush=True)
    out = MODELS / arm
    out.mkdir(parents=True, exist_ok=True)
    seeds = json.load(open("models/lgbm_ensemble_seeds.json"))["seeds"]
    for s in seeds:
        path = out / f"lgbm_model_seed{s}.txt"
        if path.exists():
            print(f"[TRAIN {arm}] seed {s} exists, skipped")
            continue
        t0 = time.time()
        m = lgb.LGBMClassifier(**params, objective="binary", metric="average_precision",
                               verbosity=-1, random_state=s)
        m.fit(X, y)
        m.booster_.save_model(str(path))
        print(f"[TRAIN {arm}] seed {s} fitted in {(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump({"feature_cols": feats, "target": TARGET, "seeds": seeds, "params": params,
               "train_window": ["2013-04-10", str(TRAIN_END.date())], "neg_per_pos": NEG_PER_POS},
              open(out / "config.json", "w"), indent=2)


def _boosters(model_dir: Path) -> list[lgb.Booster]:
    seeds = json.load(open("models/lgbm_ensemble_seeds.json"))["seeds"]
    return [lgb.Booster(model_file=str(model_dir / f"lgbm_model_seed{s}.txt")) for s in seeds]


def _mean_pred(boosters: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    p = np.zeros(len(X))
    for b in boosters:
        p += b.predict(X, num_threads=0)
    return (p / len(boosters)).astype("float32")


def predict_b() -> None:
    feats = feats_for("B")
    boosters = _boosters(MODELS / "B")
    for split in ("val", "test"):
        out = WORK / f"preds_{split}_B.parquet"
        if out.exists():
            continue
        parts, t0 = [], time.time()
        for f in sorted((WORK / f"features_{split}_B").glob("*.parquet")):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=BATCH, columns=["serial_number", "date"] + feats):
                df = batch.to_pandas()
                parts.append(pd.DataFrame({"serial_number": df["serial_number"].to_numpy(),
                                           "date": pd.to_datetime(df["date"]).to_numpy(),
                                           "p_B": _mean_pred(boosters, df[feats].astype("float32").to_numpy())}))
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        print(f"[PREDICT B] {split} ({(time.time() - t0) / 60:.1f} min) -> {out}", flush=True)


def predict(arm: str) -> None:
    if arm == "B":
        return predict_b()
    feats = feature_cols()
    frozen = _boosters(Path("archive/model1/models"))
    new = _boosters(MODELS / arm)
    WORK.mkdir(parents=True, exist_ok=True)

    # validation: from the original test feature file
    out = WORK / f"preds_val_{arm}.parquet"
    if not out.exists():
        parts, t0 = [], time.time()
        for batch in pq.ParquetFile(PROCESSED / "features_test.parquet").iter_batches(
                batch_size=BATCH, columns=KEYS + [TARGET] + feats):
            df = _frame(batch, feats)
            df = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]
            if df.empty:
                continue
            X = df[feats].to_numpy()
            parts.append(pd.DataFrame({"serial_number": df["serial_number"].to_numpy(),
                                       "date": df["date"].to_numpy(), "model": df["model"].to_numpy(),
                                       "y": df[TARGET].to_numpy(), "p_F": _mean_pred(frozen, X),
                                       f"p_{arm}": _mean_pred(new, X)}))
            n = sum(len(p) for p in parts)
            if len(parts) % 20 == 0:
                print(f"[PREDICT val] {n:,} rows ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        print(f"[SAVE] {out}", flush=True)

    # test: 2026 feature parts; F scores come from the row-aligned score parts
    out = WORK / f"preds_test_{arm}.parquet"
    if not out.exists():
        parts, t0 = [], time.time()
        for fp in sorted((FRESH / "features").glob("part_*.parquet")):
            df = pd.read_parquet(fp)
            sc = pd.read_parquet(FRESH / "scores" / fp.name, columns=["serial_number", "date", "prob"])
            assert (df["serial_number"].to_numpy() == sc["serial_number"].to_numpy()).all()
            assert (pd.to_datetime(df["date"]).to_numpy() == pd.to_datetime(sc["date"]).to_numpy()).all()
            df["p_F"] = sc["prob"].to_numpy()
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] <= TEST_END]
            X = df[feats].astype("float32").to_numpy()
            parts.append(pd.DataFrame({"serial_number": df["serial_number"].to_numpy(),
                                       "date": df["date"].to_numpy(), "model": df["model"].to_numpy(),
                                       "y": df[TARGET].astype("int8").to_numpy(), "p_F": df["p_F"].to_numpy(),
                                       f"p_{arm}": _mean_pred(new, X)}))
            print(f"[PREDICT test] {fp.name} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        print(f"[SAVE] {out}  (test predictions written; not summarised until the B decision)")


# --------------------------------------------------------------------- arm B
FLEET_COLS = ["smart_1_raw", "smart_4_raw", "smart_5_raw", "smart_9_raw", "smart_187_raw",
              "smart_188_raw", "smart_193_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw"]
PCT_COLS = [f"{c}_model_pct" for c in FLEET_COLS]
MIN_GROUP = 50


def _duck() -> duckdb.DuckDBPyConnection:
    spool = Path.home() / "duck_spool"
    spool.mkdir(exist_ok=True)
    return duckdb.connect(config={"memory_limit": "6GB", "temp_directory": str(spool),
                                  "preserve_insertion_order": False})


def fleet_prep() -> None:
    """One pass over every population row (train sources + 2026), keeping only the
    columns the percentiles need, partitioned by month so each window fits in memory."""
    cols = ", ".join(["serial_number", "CAST(date AS DATE) AS date", "model"] + [f'"{c}"' for c in FLEET_COLS])
    srcs = [p.as_posix() for p in TRAIN_SOURCES] + [(FRESH / "features").as_posix() + "/*.parquet"]
    union = " UNION ALL ".join(f"SELECT {cols} FROM read_parquet('{s}')" for s in srcs)
    out = WORK / "fleet_population"
    t0 = time.time()
    _duck().execute(f"""
        COPY (SELECT *, strftime(date, '%Y-%m') AS ym FROM ({union}))
        TO '{out.as_posix()}' (FORMAT parquet, PARTITION_BY (ym), OVERWRITE_OR_IGNORE)""")
    print(f"[FLEET] population written in {(time.time() - t0) / 60:.1f} min -> {out}")


def fleet_pct() -> None:
    """percent_rank of each FLEET_COL among non-missing drives of the same model on the
    same day; missing where the value is missing or fewer than MIN_GROUP drives report."""
    src, out = WORK / "fleet_population", WORK / "fleet_pct"
    out.mkdir(exist_ok=True)
    exprs = []
    for c, p in zip(FLEET_COLS, PCT_COLS):
        part = f'PARTITION BY date, model, ("{c}" IS NULL)'
        exprs.append(f'CASE WHEN "{c}" IS NULL OR count("{c}") OVER (PARTITION BY date, model) < {MIN_GROUP} '
                     f'THEN NULL ELSE CAST(percent_rank() OVER ({part} ORDER BY "{c}") AS FLOAT) END AS {p}')
    con = _duck()
    months = sorted(d.name.split("=")[1] for d in src.iterdir() if d.is_dir())
    t0 = time.time()
    for i, ym in enumerate(months, 1):
        dst = out / f"{ym}.parquet"
        if dst.exists():
            continue
        con.execute(f"""
            COPY (SELECT serial_number, date, {", ".join(exprs)}
                  FROM read_parquet('{(src / f"ym={ym}").as_posix()}/*.parquet'))
            TO '{dst.with_suffix(".tmp").as_posix()}' (FORMAT parquet)""")
        dst.with_suffix(".tmp").replace(dst)
        if i % 12 == 0:
            print(f"[FLEET] {i}/{len(months)} months ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[FLEET] percentiles for {len(months)} months -> {out}")


def build_b_inputs() -> None:
    """B uses exactly A's training rows plus the percentile columns; validation and test
    feature files with the percentiles joined on (serial_number, date)."""
    feats = feature_cols()
    pct = (WORK / "fleet_pct").as_posix() + "/*.parquet"
    fsel = ", ".join(f'f."{c}"' for c in KEYS + [TARGET] + feats)
    psel = ", ".join(f"p.{c}" for c in PCT_COLS)
    con = _duck()
    dst = WORK / "train_resampled_B.parquet"
    if not dst.exists():
        t0 = time.time()
        con.execute(f"""
            COPY (SELECT {fsel}, {psel} FROM read_parquet('{(WORK / 'train_resampled_A.parquet').as_posix()}') f
                  LEFT JOIN read_parquet('{pct}') p
                  ON p.serial_number = f.serial_number AND p.date = CAST(f.date AS DATE))
            TO '{dst.with_suffix(".tmp").as_posix()}' (FORMAT parquet)""")
        dst.with_suffix(".tmp").replace(dst)
        n, nn = con.execute(f"SELECT count(*), count({PCT_COLS[0]}) FROM '{dst.as_posix()}'").fetchone()
        print(f"[B] {dst.name}: {n:,} rows, {nn / n:.1%} with {PCT_COLS[0]} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # validation and test (~57M rows each) are joined one month at a time against that
    # month's percentile file only; a single join over all months exceeds the memory limit
    sources = {"val": (f"read_parquet('{(PROCESSED / 'features_test.parquet').as_posix()}')", VAL_START, VAL_END),
               "test": (f"read_parquet('{(FRESH / 'features').as_posix()}/*.parquet')", TEST_START, TEST_END)}
    for split, (src, start, end) in sources.items():
        out = WORK / f"features_{split}_B"
        out.mkdir(exist_ok=True)
        t0, n, nn = time.time(), 0, 0
        for ym in pd.period_range(start, end, freq="M").astype(str):
            dst = out / f"{ym}.parquet"
            if not dst.exists():
                con.execute(f"""
                    COPY (SELECT {fsel}, {psel} FROM {src} f
                          LEFT JOIN read_parquet('{(WORK / "fleet_pct" / f"{ym}.parquet").as_posix()}') p
                          ON p.serial_number = f.serial_number AND p.date = CAST(f.date AS DATE)
                          WHERE CAST(f.date AS DATE) BETWEEN DATE '{start.date()}' AND DATE '{end.date()}'
                            AND strftime(CAST(f.date AS DATE), '%Y-%m') = '{ym}')
                    TO '{dst.with_suffix(".tmp").as_posix()}' (FORMAT parquet)""")
                dst.with_suffix(".tmp").replace(dst)
            a, b = con.execute(f"SELECT count(*), count({PCT_COLS[0]}) FROM '{dst.as_posix()}'").fetchone()
            n, nn = n + a, nn + b
        print(f"[B] features_{split}_B: {n:,} rows, {nn / n:.1%} with {PCT_COLS[0]} ({(time.time() - t0) / 60:.1f} min)", flush=True)


# --------------------------------------------------------------------- decide
MARGIN = 0.003


def _load_split(split: str) -> dict:
    con = _duck()
    a = (WORK / f"preds_{split}_A.parquet").as_posix()
    b = (WORK / f"preds_{split}_B.parquet").as_posix()
    r = con.execute(f"""
        SELECT CAST(dense_rank() OVER (ORDER BY a.serial_number) - 1 AS INTEGER) AS drive_id,
               CAST(a.date AS DATE) AS date, CAST(a.y AS TINYINT) AS y, a.p_F, a.p_A, b.p_B
        FROM read_parquet('{a}') a JOIN read_parquet('{b}') b
          ON a.serial_number = b.serial_number AND CAST(a.date AS DATE) = CAST(b.date AS DATE)
    """).fetchnumpy()
    n_a = con.execute(f"SELECT count(*) FROM read_parquet('{a}')").fetchone()[0]
    assert len(r["y"]) == n_a, "A and B prediction rows do not match"
    return {k: np.asarray(v) for k, v in r.items()}


def _paired(s17, d: dict, x: str, z: str, n_boot: int) -> dict:
    from sklearn.metrics import average_precision_score
    ap_x = float(average_precision_score(d["y"], d[x]))
    ap_z = float(average_precision_score(d["y"], d[z]))
    gaps = s17.bootstrap_delta(d[x].astype(np.float64), d[z].astype(np.float64), d["y"], d["drive_id"], n_boot)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {f"auprc_{x}": ap_x, f"auprc_{z}": ap_z, "delta": ap_x - ap_z,
            "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": int(len(gaps))}


def _describe(d: dict, p: str, thr: float, cut: dict) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s, date = d["y"], d[p], pd.to_datetime(d["date"])
    pred = s >= thr
    hi = s >= cut["high_cutoff"]
    days = date.nunique()
    out = {"auprc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
           "threshold": thr, "high_cutoff": cut["high_cutoff"], "medium_cutoff": cut["medium_cutoff"],
           "precision_at_thr": float(y[pred].mean()) if pred.any() else 0.0,
           "recall_at_thr": float(pred[y == 1].mean()),
           "high_rows_per_day": float(hi.sum() / days), "high_row_precision": float(y[hi].mean()),
           "high_share_of_pos_rows": float(hi[y == 1].mean())}
    month = date.to_period("M")
    out["monthly_auprc"] = {str(m): float(average_precision_score(y[month == m], s[month == m]))
                            for m in sorted(month.unique())}
    df = pd.DataFrame({"drive": d["drive_id"], "date": date, "s": s, "y": y})
    last = df.sort_values("date").groupby("drive").agg(s=("s", "last"), ever=("y", "max"))
    out["drive_auprc"] = float(average_precision_score(last["ever"], last["s"]))
    fail = df[df["y"] == 1].groupby("drive")["date"].max()
    idx = df.set_index(["drive", "date"])["s"]
    out["lead_time_share_of_all"] = {}
    for L in (30, 14, 7, 1):
        got = idx.reindex(pd.MultiIndex.from_arrays([fail.index, fail.to_numpy() - pd.Timedelta(days=L)])).dropna()
        out["lead_time_share_of_all"][L] = float((got >= thr).sum() / len(fail))
    return out


def decide(n_boot: int) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("s17", "17_ablation_compare.py")
    s17 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s17)
    spec = importlib.util.spec_from_file_location("s09", "09_risk_tier_calibration.py")
    s09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s09)
    from sklearn.metrics import confusion_matrix

    def cost_threshold(y, s):   # Stage 6 rule: FN:FP = 100:1 on a 0.01..0.99 grid
        best, best_cost = 0.5, np.inf
        for t in np.linspace(0.01, 0.99, 99):
            tn, fp, fn, tp = confusion_matrix(y, (s >= t).astype(int), labels=[0, 1]).ravel()
            if fn * 100 + fp < best_cost:
                best, best_cost = float(t), fn * 100 + fp
        return best

    res: dict = {"preregistration": "results_retrain2026/PREREGISTRATION.md", "margin": MARGIN}
    val = _load_split("val")
    print(f"[VAL] {len(val['y']):,} rows, {int(val['y'].sum()):,} positives", flush=True)
    res["val_B_vs_A"] = vb = _paired(s17, val, "p_B", "p_A", n_boot)
    adopt_b = vb["delta"] >= MARGIN and vb["ci_lo"] > 0
    R = "p_B" if adopt_b else "p_A"
    res["retrained_candidate"] = R[2:]
    print(f"[DECISION 1] B vs A on validation: {vb} -> R = {R[2:]}", flush=True)

    fit = {}
    for p in ("p_A", "p_B"):
        fit[p] = {"threshold": cost_threshold(val["y"], val[p]),
                  "cut": {"high_cutoff": s09.find_tier_cutoff(val[p], val["y"], 0.5),
                          "medium_cutoff": s09.find_tier_cutoff(val[p], val["y"], 0.8)}}
    fit["p_F"] = {"threshold": float(json.load(open("archive/model1/models/lgbm_threshold.json"))["threshold"]),
                  "cut": json.load(open("archive/model1/results/tier_cutoffs.json"))}
    res["fitted_on_validation"] = {k[2:]: v for k, v in fit.items()}
    res["val_auprc"] = {k[2:]: float(__import__("sklearn.metrics", fromlist=["x"]).average_precision_score(
        val["y"], val[k])) for k in ("p_F", "p_A", "p_B")}
    del val

    test = _load_split("test")
    print(f"[TEST] {len(test['y']):,} rows, {int(test['y'].sum()):,} positives", flush=True)
    res["test_R_vs_F"] = t = _paired(s17, test, R, "p_F", n_boot)
    if t["delta"] >= MARGIN and t["ci_lo"] > 0:
        verdict = "retraining helps"
    elif t["ci_lo"] > -MARGIN and t["ci_hi"] < MARGIN:
        verdict = "no material change"
    elif t["ci_hi"] < 0:
        verdict = "worse"
    else:
        verdict = "inconclusive"
    res["verdict"] = verdict
    print(f"[DECISION 2] {R[2:]} vs F on test: {t} -> {verdict}", flush=True)
    res["test_describe"] = {p[2:]: _describe(test, p, fit[p]["threshold"], fit[p]["cut"])
                            for p in ("p_F", "p_A", "p_B")}
    json.dump(res, open(OUT / "results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["resample", "train", "predict", "fleet_prep", "fleet_pct", "build_b", "decide"])
    ap.add_argument("--arm", default="A")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"resample": resample, "train": lambda: train(args.arm),
     "predict": lambda: predict(args.arm), "fleet_prep": fleet_prep, "fleet_pct": fleet_pct,
     "build_b": build_b_inputs, "decide": lambda: decide(args.boot)}[args.stage]()


if __name__ == "__main__":
    main()
