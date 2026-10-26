"""
17c -- Per-seed AUPRC on the FULL test split (Amendment 02)

The pre-registration needs per-seed AUPRC for all five training seeds in each
arm, on the test split.  The ensemble only ever stored the averaged probability
(`stream_predict_ensemble` accumulates and divides), so the per-seed numbers have
to be recomputed from the saved boosters.

Amendment 01 first tried to do this on a row-group subsample.  That was wrong:
`features_test.parquet` is sorted by serial number and its row groups are
drive-disjoint, so every-14th-group is a DRIVE holdout, not a row sample.  It
produced a 5/5 seed sweep for arm A (+0.009 mean) while the full-split ensembles
differed by -0.0001.  See `AMENDMENT_02.md`.

This script does it properly: one pass over all 144,739,479 test rows, all five
boosters predicting each chunk while it is in memory.

Memory: five float32 probability vectors, 144.7M rows each (~579 MB per seed),
plus labels (~145 MB).  ~3 GB, plus one chunk of features.

The ensemble column written here is the mean of the five seed probabilities and
must reproduce the corresponding `lgbm_metrics.csv` AUPRC to ~1e-9; that is the
correctness check for this whole path.

Usage
-----
    python 17c_per_seed_full.py --arm_dir models_ablation/without_infra
    python 17c_per_seed_full.py --arm_dir models_ablation/with_infra
"""
import argparse
import gc
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score


def load_module(name: str, path: str):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_dir", required=True)
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--stage6", default="06_train_lightgbm.py")
    ap.add_argument("--out", default=None,
                    help="default results_ablation/per_seed_full_<arm>.csv")
    args = ap.parse_args()

    arm_dir = Path(args.arm_dir)
    out_path = Path(args.out) if args.out else \
        Path("results_ablation") / f"per_seed_full_{arm_dir.name}.csv"

    mod = load_module("stage6", args.stage6)

    with open(arm_dir / "lgbm_feature_cols.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    target = meta.get("target", "failure_within_14d")
    with open(arm_dir / "lgbm_ensemble_seeds.json") as f:
        seeds = json.load(f)["seeds"]

    boosters = [lgb.Booster(model_file=str(arm_dir / f"lgbm_model_seed{sd}.txt"))
                for sd in seeds]

    pf = pq.ParquetFile(args.test)
    total = pf.metadata.num_rows
    print(f"[ARM] {arm_dir.name}: {len(feature_cols)} features, seeds {seeds}, "
          f"{total:,} test rows")

    y = np.empty(total, dtype="int8")
    P = np.empty((len(boosters), total), dtype="float32")

    read_cols = feature_cols + [target]
    pos = 0
    n_chunks = 0
    for batch in pf.iter_batches(batch_size=mod.PREDICT_CHUNK_SIZE, columns=read_cols):
        df = mod.coerce_legacy_format(batch.to_pandas())
        X = df[feature_cols].astype("float32")
        n = len(df)
        for j, b in enumerate(boosters):
            P[j, pos:pos + n] = b.predict(X)
        y[pos:pos + n] = df[target].to_numpy()
        pos += n
        n_chunks += 1
        del df, X, batch
        if n_chunks % 10 == 0:
            pa.default_memory_pool().release_unused()
        if n_chunks % 20 == 0:
            print(f"[STREAM] {n_chunks} chunks / {pos:,} rows")
            gc.collect()
            pa.default_memory_pool().release_unused()

    if pos != total:
        print(f"[WARN] filled {pos:,} of {total:,} rows")
        y, P = y[:pos], P[:, :pos]

    prev = float(y.mean())
    print(f"[SAMPLE] {pos:,} rows, {int(y.sum()):,} positives, "
          f"prevalence {prev:.6%}")

    rows = []
    for j, sd in enumerate(seeds):
        a = float(average_precision_score(y, P[j]))
        r = float(roc_auc_score(y, P[j]))
        rows.append({"arm": arm_dir.name, "seed": str(sd), "n_rows": int(pos),
                     "n_positives": int(y.sum()), "prevalence": prev,
                     "auprc": a, "roc_auc": r, "lift_over_prevalence": a / prev})
        print(f"[SEED {sd}] auprc={a:.10f} roc_auc={r:.10f}")

    ens = P.mean(axis=0, dtype=np.float32)
    ens_a = float(average_precision_score(y, ens))
    ens_r = float(roc_auc_score(y, ens))
    rows.append({"arm": arm_dir.name, "seed": "ensemble", "n_rows": int(pos),
                 "n_positives": int(y.sum()), "prevalence": prev,
                 "auprc": ens_a, "roc_auc": ens_r,
                 "lift_over_prevalence": ens_a / prev})
    print(f"[ENSEMBLE] auprc={ens_a:.10f} roc_auc={ens_r:.10f}")

    ref = Path(arm_dir.parent.parent) / "results_ablation" / arm_dir.name / "lgbm_metrics.csv"
    if ref.exists():
        refa = float(pd.read_csv(ref)["auprc"].iloc[0])
        print(f"[CHECK] lgbm_metrics.csv auprc={refa:.10f}  diff={ens_a - refa:+.2e}")

    seed_vals = np.array([r["auprc"] for r in rows if r["seed"] != "ensemble"])
    print(f"[SPREAD] min={seed_vals.min():.10f} max={seed_vals.max():.10f} "
          f"range={seed_vals.max() - seed_vals.min():.10f} "
          f"sd={seed_vals.std(ddof=1):.10f} mean={seed_vals.mean():.10f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
