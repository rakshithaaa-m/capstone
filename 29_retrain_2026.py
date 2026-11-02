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
  decide     (after B) B vs A on validation, then the confirmatory R vs F on test

Usage:
    python 29_retrain_2026.py resample
    python 29_retrain_2026.py train --arm A
    python 29_retrain_2026.py predict --arm A
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


def train(arm: str) -> None:
    feats = feature_cols()
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


def predict(arm: str) -> None:
    feats = feature_cols()
    frozen = _boosters(Path("models"))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["resample", "train", "predict"])
    ap.add_argument("--arm", default="A")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"resample": resample, "train": lambda: train(args.arm),
     "predict": lambda: predict(args.arm)}[args.stage]()


if __name__ == "__main__":
    main()
