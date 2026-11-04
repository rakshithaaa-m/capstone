"""Infrastructure-identifier ablation for Model 2 (results_model2/PREREGISTRATION_infra_ablation.md).

Arm A = shipped Model 2 (212 features). Arm B = the same training rows, hyperparameters and
seeds with the five identifiers restored (217 features). One look at 2026-01-01..06-16.

    python 36_model2_infra_ablation.py build      # identifiers joined onto train and test rows
    python 36_model2_infra_ablation.py train
    python 36_model2_infra_ablation.py predict
    python 36_model2_infra_ablation.py decide     # paired bootstrap, seed signs, verdict
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

INFRA = ["vault_id", "pod_id", "cluster_id", "pod_slot_num", "is_legacy_format"]
TARGET = "failure_within_14d"
TEST_END = "2026-06-16"
FEATURES = Path("data/processed/fresh2026/features")
PARTS = Path("data/processed/fresh2026/partitions")
PROCESSED = Path("data/processed")
WORK = Path("results_model2/_work/infra")
MODELS = Path("models_model2_ablation/with_infra")
OUT = Path("results_model2")
MARGIN = 0.003


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(Path.home() / 'duck_spool').as_posix()}'")
    return con


def _base_cfg() -> dict:
    return json.load(open("models_retrain2026/A/config.json"))


def _infra_sel(alias: str) -> str:
    return ", ".join([f"CAST({alias}.{c} AS DOUBLE) AS {c}" for c in INFRA[:-1]]
                     + [f"CAST(coalesce({alias}.is_legacy_format, false) AS INTEGER) AS is_legacy_format"])


def build() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    con = _duck()
    dst = WORK / "train_infra.parquet"
    if not dst.exists():
        # the identifiers for Model 2's exact training rows, from the source feature files
        src = " UNION ALL ".join(
            f"SELECT serial_number, CAST(date AS DATE) d, {', '.join(INFRA)} FROM read_parquet('{(PROCESSED / f).as_posix()}')"
            for f in ("features_train.parquet", "features_val.parquet", "features_test.parquet"))
        con.execute(f"""
            COPY (SELECT t.*, {_infra_sel('s')}
                  FROM read_parquet('data/processed/retrain2026/train_resampled_A.parquet') t
                  LEFT JOIN ({src}) s ON s.serial_number = t.serial_number AND s.d = CAST(t.date AS DATE))
            TO '{dst.with_suffix('.tmp').as_posix()}' (FORMAT parquet)""")
        dst.with_suffix(".tmp").replace(dst)
    n, nv = con.execute(f"SELECT count(*), count(vault_id) FROM '{dst.as_posix()}'").fetchone()
    print(f"[BUILD] train: {n:,} rows, {nv / n:.1%} with vault_id", flush=True)
    out = WORK / "test"
    out.mkdir(exist_ok=True)
    feats = _base_cfg()["feature_cols"]
    cols = ", ".join(f'f."{c}"' for c in feats)
    for f in sorted(FEATURES.glob("part_*.parquet")):   # part_k matches partitions/part=k
        k = int(f.stem.split("_")[1])
        d = out / f.name
        if d.exists():
            continue
        con.execute(f"""
            COPY (SELECT f.serial_number, CAST(f.date AS DATE) AS date, f.{TARGET}, {cols}, {_infra_sel('p')}
                  FROM read_parquet('{f.as_posix()}') f
                  LEFT JOIN read_parquet('{(PARTS / f'part={k}').as_posix()}/*.parquet') p
                    ON p.serial_number = f.serial_number AND CAST(p.date AS DATE) = CAST(f.date AS DATE)
                  WHERE CAST(f.date AS DATE) <= DATE '{TEST_END}')
            TO '{d.as_posix()}' (FORMAT parquet)""")
    n, nv = con.execute(f"SELECT count(*), count(vault_id) FROM read_parquet('{out.as_posix()}/*.parquet')").fetchone()
    print(f"[BUILD] test: {n:,} rows, {nv / n:.1%} with vault_id", flush=True)


def train() -> None:
    cfg = _base_cfg()
    feats = cfg["feature_cols"] + INFRA
    df = pd.read_parquet(WORK / "train_infra.parquet", columns=feats + [TARGET])
    X, y = df[feats].astype("float32"), df[TARGET].astype(int)
    del df
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"[TRAIN] {len(y):,} rows, {int(y.sum()):,} positives, {len(feats)} features", flush=True)
    for s in cfg["seeds"]:
        path = MODELS / f"lgbm_model_seed{s}.txt"
        if path.exists():
            continue
        t0 = time.time()
        m = lgb.LGBMClassifier(**cfg["params"], objective="binary", metric="average_precision",
                               verbosity=-1, random_state=s)
        m.fit(X, y)
        m.booster_.save_model(str(path))
        print(f"[TRAIN] seed {s} fitted in {(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump({**cfg, "feature_cols": feats}, open(MODELS / "config.json", "w"), indent=2)


def predict() -> None:
    cfg = json.load(open(MODELS / "config.json"))
    feats = cfg["feature_cols"]
    boosters = {s: lgb.Booster(model_file=str(MODELS / f"lgbm_model_seed{s}.txt")) for s in cfg["seeds"]}
    keys, ys, per = [], [], {s: [] for s in boosters}
    t0 = time.time()
    for f in sorted((WORK / "test").glob("part_*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=500_000, columns=["serial_number", "date", TARGET] + feats):
            df = b.to_pandas()
            X = df[feats].astype("float32").to_numpy()
            for s, bst in boosters.items():
                per[s].append(bst.predict(X, num_threads=0).astype("float32"))
            keys.append(df[["serial_number", "date"]])
            ys.append(df[TARGET].to_numpy(np.int8))
    out = pd.concat(keys, ignore_index=True)
    out["y"] = np.concatenate(ys)
    for s in per:
        out[f"p_seed{s}"] = np.concatenate(per[s])
    out["p_B"] = out[[f"p_seed{s}" for s in per]].mean(axis=1).astype("float32")
    out.to_parquet(WORK / "preds_B.parquet", index=False)
    print(f"[PREDICT] {len(out):,} rows ({(time.time() - t0) / 60:.1f} min)", flush=True)


def decide(n_boot: int) -> None:
    spec = importlib.util.spec_from_file_location("s17", "17_ablation_compare.py")
    s17 = importlib.util.module_from_spec(spec)
    sys.modules["s17"] = s17
    spec.loader.exec_module(s17)
    con = _duck()
    d = con.execute(f"""
        SELECT CAST(dense_rank() OVER (ORDER BY b.serial_number) - 1 AS INTEGER) AS drive_id,
               CAST(b.y AS TINYINT) AS y, a.p AS p_A, b.p_B
        FROM read_parquet('{(WORK / 'preds_B.parquet').as_posix()}') b
        JOIN read_parquet('results_deploy2026/_work/scores/*.parquet') a
          ON a.serial_number = b.serial_number AND a.date = CAST(b.date AS DATE)
        ORDER BY b.serial_number, b.date""").fetchnumpy()
    d = {k: np.asarray(v) for k, v in d.items()}
    ap_a, ap_b = average_precision_score(d["y"], d["p_A"]), average_precision_score(d["y"], d["p_B"])
    gaps = s17.bootstrap_delta(d["p_A"].astype(np.float64), d["p_B"].astype(np.float64), d["y"], d["drive_id"], n_boot)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    seeds_a = json.load(open(OUT / "seeds.json"))["per_seed_auprc"]
    pb = pd.read_parquet(WORK / "preds_B.parquet")
    seeds_b = {s: average_precision_score(pb["y"], pb[f"p_seed{s}"]) for s in seeds_a}
    wins = sum(seeds_a[s] > seeds_b[s] for s in seeds_a)
    if lo > 0 and wins >= 4:
        verdict = "SUPERIORITY"
    elif -MARGIN < lo and hi < MARGIN:
        verdict = "EQUIVALENCE"
    elif hi < 0:
        verdict = "REVERSAL"
    elif lo > -MARGIN:
        verdict = "NON-INFERIORITY"
    else:
        verdict = "INCONCLUSIVE"
    res = {"rows": int(len(d["y"])), "auprc_A_without_infra": ap_a, "auprc_B_with_infra": ap_b,
           "delta": ap_a - ap_b, "ci": [float(lo), float(hi)], "n_boot": n_boot,
           "per_seed_A": seeds_a, "per_seed_B": seeds_b, "A_wins_seeds": wins, "verdict": verdict}
    (OUT / "infra_ablation.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[DECIDE] {json.dumps(res, default=float)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "train", "predict", "decide", "all"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    for st in (["build", "train", "predict", "decide"] if args.stage == "all" else [args.stage]):
        {"build": build, "train": train, "predict": predict, "decide": lambda: decide(args.boot)}[st]()


if __name__ == "__main__":
    main()
