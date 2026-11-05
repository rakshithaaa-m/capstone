"""
17b -- Per-seed AUPRC on a fixed row-group subsample (Amendment 01)

Why this exists
---------------
The pre-registration asks for a paired per-seed comparison (212-feature arm must
win in >=4 of 5 seeds).  `stream_predict_ensemble()` only ever writes the
*average* of the five seed probabilities, so the per-seed spread is not
recoverable from `lgbm_test_probs.csv`.

This script recomputes it from the saved boosters using one fixed, deterministic
sampling rule, documented in results_ablation/AMENDMENT_01.md:

    every 14th row group of features_test.parquet (groups 0, 14, 28, ...)
    -> 96 of 1334 groups, 10,416,000 rows, 7.20% of the test split

Both arms are scored with the identical rule on the identical rows, so the
comparison is paired.  Row-group selection avoids a full 144.7M-row read; the
selected groups are spread evenly across the file, so the subsample spans the
whole test period rather than a prefix.

The ensemble (mean of the five seed probabilities) is also reported, but only
as a cross-check against the full-split number in `lgbm_metrics.csv`; the
authoritative effect size remains the full-split comparison.

Usage
-----
    python 17b_per_seed_auprc.py \
        --arm_dir models_ablation/without_infra \
        --test data/processed/features_test.parquet \
        --out results_ablation/per_seed_without_infra.csv
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

ROW_GROUP_STRIDE = 14


def load_module(name: str, path: str):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_dir", required=True,
                    help="model directory, e.g. models_ablation/without_infra")
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage6", default="06_train_lightgbm.py")
    # The pre-registered rule is every 14th row group.  Any other value is a
    # smoke test only and must never be reported in the paper.
    ap.add_argument("--stride", type=int, default=ROW_GROUP_STRIDE,
                    help="DIAGNOSTIC ONLY; pre-registered value is 14")
    args = ap.parse_args()

    if args.stride != ROW_GROUP_STRIDE:
        print(f"[WARN] stride={args.stride} differs from the pre-registered "
              f"{ROW_GROUP_STRIDE}; this run is a smoke test, not a result")

    arm_dir = Path(args.arm_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mod = load_module("stage6", args.stage6)

    with open(arm_dir / "lgbm_feature_cols.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    target = meta.get("target", "failure_within_14d")

    with open(arm_dir / "lgbm_ensemble_seeds.json") as f:
        seeds = json.load(f)["seeds"]

    boosters = []
    for sd in seeds:
        p = arm_dir / f"lgbm_model_seed{sd}.txt"
        if not p.exists():
            raise FileNotFoundError(p)
        boosters.append(lgb.Booster(model_file=str(p)))

    print(f"[MODEL] {arm_dir} | {len(boosters)} boosters, seeds {seeds}")
    print(f"[MODEL] {len(feature_cols)} features, target={target}")

    pf = pq.ParquetFile(args.test)
    n_groups = pf.metadata.num_row_groups
    sel = list(range(0, n_groups, args.stride))
    total = sum(pf.metadata.row_group(gi).num_rows for gi in sel)
    print(f"[SUBSAMPLE] groups {len(sel)}/{n_groups} (every {args.stride}th), "
          f"{total:,} rows = {100.0 * total / pf.metadata.num_rows:.2f}% of test")

    read_cols = feature_cols + [target]
    y = np.empty(total, dtype="int8")
    P = np.empty((len(boosters), total), dtype="float32")
    pos = 0
    for k, gi in enumerate(sel):
        tbl = pf.read_row_group(gi, columns=read_cols)
        df = mod.coerce_legacy_format(tbl.to_pandas())
        X = df[feature_cols].astype("float32")
        n = len(df)
        for j, b in enumerate(boosters):
            P[j, pos:pos + n] = b.predict(X)
        y[pos:pos + n] = df[target].to_numpy()
        pos += n
        del tbl, df, X
        if k % 10 == 0:
            pa.default_memory_pool().release_unused()
            gc.collect()
        if k % 20 == 0:
            print(f"[STREAM] {k + 1}/{len(sel)} groups, {pos:,} rows")

    if pos != total:
        print(f"[WARN] expected {total:,} rows, filled {pos:,}")
        y, P = y[:pos], P[:, :pos]

    prevalence = float(y.mean())
    print(f"[SAMPLE] rows={pos:,} positives={int(y.sum()):,} prevalence={prevalence:.6%}")

    rows = []
    for j, sd in enumerate(seeds):
        rows.append({
            "arm": arm_dir.name,
            "seed": str(sd),
            "n_rows": int(pos),
            "n_positives": int(y.sum()),
            "prevalence": prevalence,
            "auprc": float(average_precision_score(y, P[j])),
            "roc_auc": float(roc_auc_score(y, P[j])),
            "lift_over_prevalence": float(average_precision_score(y, P[j]) / prevalence),
        })
        print(f"[SEED {sd}] auprc={rows[-1]['auprc']:.10f} roc_auc={rows[-1]['roc_auc']:.10f}")

    ens = P.mean(axis=0)
    ens_auprc = float(average_precision_score(y, ens))
    rows.append({
        "arm": arm_dir.name, "seed": "ensemble", "n_rows": int(pos),
        "n_positives": int(y.sum()), "prevalence": prevalence,
        "auprc": ens_auprc, "roc_auc": float(roc_auc_score(y, ens)),
        "lift_over_prevalence": ens_auprc / prevalence,
    })
    print(f"[ENSEMBLE] auprc={ens_auprc:.10f} (subsample cross-check only)")

    seed_rows = [r for r in rows if r["seed"] != "ensemble"]
    a = np.array([r["auprc"] for r in seed_rows])
    print(f"[SPREAD] min={a.min():.10f} max={a.max():.10f} "
          f"range={a.max() - a.min():.10f} sd={a.std(ddof=1):.10f} "
          f"mean={a.mean():.10f}")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
