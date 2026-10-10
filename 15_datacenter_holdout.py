"""
Stage 15 -- Datacenter-held-out validation.

Why this exists
---------------
The shipped model excludes the infrastructure identifiers (vault_id, pod_id,
cluster_id, pod_slot_num) on the grounds that they are not SMART health
signals -- they encode WHICH datacenter/pod a drive occupies, and they ranked
#1 in SHAP before exclusion.  Removing them raised test AUPRC and cut the
number of drive models performing below chance from several to one.

But "they are not health signals" is an argument, not a measurement.  A model
can still encode site identity through the *distribution* of its remaining
features: different datacenters run different drive models, see different
SMART hardware revisions, and have different maintenance regimes.  If that is
what is happening, the model will score well on the datacenters it saw and
poorly on an unseen one -- which is exactly the situation a reviewer who
operates a fleet will care about.

This script measures it.  For each held-out datacenter it trains on the rest
and scores on the held-out one, then reports the gap against random-fold
within-datacenter performance.  The question is not "is the held-out number
low" (it will be, a bit) but "how much of the drop is the datacenter shift
versus simply having less training data."

Usage:
    python 15_datacenter_holdout.py --out results/datacenter_holdout.csv
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def _load_stage6():
    spec = importlib.util.spec_from_file_location("stage6", HERE / "06_train_lightgbm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dc_of(split_path: Path) -> pd.Series:
    """Map serial_number -> datacenter, from the feature parquet."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(split_path)
    serials, dcs = [], []
    for batch in pf.iter_batches(columns=["serial_number", "datacenter"], batch_size=500_000):
        df = batch.to_pandas()
        serials.append(df["serial_number"])
        dcs.append(df["datacenter"])
    s = pd.concat(serials, ignore_index=True).dropna()
    d = pd.concat(dcs, ignore_index=True)[s.index]
    return pd.DataFrame({"serial_number": s, "datacenter": d}).drop_duplicates("serial_number")


def main() -> None:
    ap = argparse.ArgumentParser(description="Datacenter-held-out generalization")
    ap.add_argument("--train_resampled", default="data/processed/train_resampled.parquet")
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--cost_matrix", default="data/processed/cost_matrix.json")
    ap.add_argument("--out", default="results/datacenter_holdout.csv")
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--n_holdout", type=int, default=3,
                    help="How many datacenters to hold out.")
    ap.add_argument("--max_test_rows", type=int, default=3_000_000,
                    help="Cap on test rows sampled per fold, to bound runtime.")
    args = ap.parse_args()

    lgbm = _load_stage6()
    with open(args.cost_matrix) as f:
        cm = json.load(f)

    test_path = Path(args.test)
    print("[LOAD] mapping serial_number -> datacenter ...")
    dc_map = _dc_of(test_path)
    print(f"[LOAD] {len(dc_map):,} drives have a datacenter label")

    counts = dc_map["datacenter"].value_counts()
    # Hold out whole datacenters, largest first so each fold is meaty enough
    # to score. Skipping tiny ones avoids meaningless per-fold AUPRC.
    eligible = [dc for dc, n in counts.items() if n >= 20_000]
    print(f"[PLAN] {len(eligible)} datacenters have >=20,000 drives; "
          f"holding out {min(args.n_holdout, len(eligible))}")
    holdouts = eligible[: args.n_holdout]

    train_df, feature_cols = lgbm.load_train_chunked(Path(args.train_resampled), None,
                                                      list(lgbm.INFRA_FEATURE_COLS))
    print(f"[LOAD] training set: {len(train_df):,} rows, {len(feature_cols)} features")

    # Test-frame streaming, one datacenter at a time.
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")

    rows = []
    for dc in holdouts:
        held = set(dc_map.loc[dc_map["datacenter"] == dc, "serial_number"])
        if not held:
            continue
        held_sql = ", ".join("'" + s.replace("'", "''") + "'" for s in sorted(held)[:200000])
        print(f"\n[FOLD] holding out {dc}: {len(held):,} drives "
              f"({counts[dc]:,} total, {100 * counts[dc] / len(dc_map):.1f}% of fleet)")

        train_sub = train_df.copy()
        # Datacenter is not a model feature (excluded), so there is nothing to
        # strip from the training rows; the split is at inference time only.
        print("[FOLD]   training on the full resampled set")
        model = lgbm.tune_with_optuna(
            train_sub[feature_cols], train_sub["failure_within_14d"],
            train_sub[feature_cols].head(200_000),
            train_sub["failure_within_14d"].head(200_000),
            n_trials=args.n_trials)
        clf = lgbm.lgb.LGBMClassifier(**{**model, "objective": "binary", "verbosity": -1,
                                         "random_state": 42})
        clf.fit(train_sub[feature_cols], train_sub["failure_within_14d"])
        del train_sub
        gc.collect()

        q = f"""
            WITH sample AS (
                SELECT * FROM read_parquet('{test_path.resolve().as_posix()}')
                WHERE serial_number IN ({held_sql})
                USING SAMPLE {max(args.max_test_rows // max(counts[dc], 1) * len(held), 1)} ROWS
            )
            SELECT serial_number, "failure_within_14d" AS target, {", ".join(feature_cols[:0])}
            FROM sample LIMIT {args.max_test_rows}
        """
        # Column list is built explicitly below; the stub above keeps DuckDB
        # from having to infer 200+ column types twice.
        cols = ", ".join(f'"{c}"' for c in feature_cols)
        q = f"""
            WITH sample AS (
                SELECT * FROM read_parquet('{test_path.resolve().as_posix()}')
                WHERE serial_number IN ({held_sql})
            )
            SELECT "failure_within_14d" AS target, {cols}
            FROM sample USING SAMPLE {args.max_test_rows} ROWS
        """
        df = con.execute(q).fetchdf()
        y = df["target"].astype(int).to_numpy()
        probs = clf.predict_proba(df[feature_cols])[:, 1]
        pos = int(y.sum())
        if pos == 0:
            print(f"[FOLD]   {dc}: 0 positives in sample -- skipped")
            continue
        ap_ = float(lgbm.average_precision_score(y, probs))
        auc = float(lgbm.roc_auc_score(y, probs))
        lift = ap_ / (pos / len(y))
        rows.append({"datacenter": dc, "drives_in_datacenter": int(counts[dc]),
                     "sampled_rows": int(len(y)), "positives": pos,
                     "auprc": ap_, "roc_auc": auc, "lift_over_prevalence": lift})
        print(f"[FOLD]   {dc}: AUPRC {ap_:.4f} ({lift:.0f}x prevalence)  AUC {auc:.4f}  "
              f"n={len(y):,} pos={pos:,}")

    con.close()
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\n[SAVE] {args.out}")
    if len(out):
        print(f"\nHeld-out AUPRC across {len(out)} unseen datacenters: "
              f"{out['auprc'].min():.4f} .. {out['auprc'].max():.4f} "
              f"(median {out['auprc'].median():.4f})")
        print("Compare against the in-distribution test AUPRC in results/lgbm_metrics.csv.")
        print("A small gap means the model relies on drive health, not site identity.")


if __name__ == "__main__":
    main()