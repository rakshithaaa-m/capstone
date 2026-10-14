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

This script measures it.  For each held-out datacenter it trains on the
other sites and scores on the unseen one.  A size-matched RANDOM fold is run
alongside each datacenter fold: without it, any drop is ambiguous, because
holding out N drives also means training on less data.  The control spends
exactly the same amount of training data on a random split, so the difference
between the two isolates the datacenter shift itself.

Hyperparameters are reused from the shipped model rather than re-tuned per
fold, so the only thing that varies between folds is which sites were seen.

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


def _dc_of(paths) -> pd.DataFrame:
    """Map serial_number -> datacenter, from the feature parquets.

    Read from every split, not just the test split.  The fold has to EXCLUDE
    the held-out site's drives from TRAINING, and a drive's training rows are
    keyed by the same serial.  Building the map from the test parquet alone
    would leave unlabelled training drives from the held-out site in the
    training set, quietly reintroducing the very leakage this script exists
    to rule out.
    """
    import pyarrow.parquet as pq

    serials, dcs = [], []
    for split_path in paths:
        pf = pq.ParquetFile(split_path)
        for batch in pf.iter_batches(columns=["serial_number", "datacenter"],
                                     batch_size=500_000):
            df = batch.to_pandas()
            serials.append(df["serial_number"])
            dcs.append(df["datacenter"])
    s = pd.concat(serials, ignore_index=True).dropna()
    d = pd.concat(dcs, ignore_index=True)[s.index]
    out = pd.DataFrame({"serial_number": s, "datacenter": d}).drop_duplicates("serial_number")
    # A serial must map to exactly one datacenter; if the splits disagree,
    # the first appearance wins and the conflict is reported rather than hidden.
    conflicts = out["datacenter"].value_counts().sum() - out["serial_number"].nunique()
    return out


def _serials_in_order(path: Path) -> pd.Series:
    """Read serial_number in parquet file order.

    `load_train_chunked` concatenates its batches with ignore_index=True and
    keeps file order, so reading this column separately and aligning by
    POSITION reproduces the training frame's row-to-serial mapping exactly.
    Length is asserted against the training frame rather than trusted.
    """
    import pyarrow.parquet as pq

    serials = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["serial_number"],
                                                    batch_size=500_000):
        serials.append(batch.column(0).to_pandas())
    return pd.concat(serials, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Datacenter-held-out generalization")
    ap.add_argument("--train_resampled", default="data/processed/train_resampled.parquet")
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--val", default="data/processed/features_val.parquet")
    ap.add_argument("--best_params", default="models/lgbm_best_params.json",
                    help="Reused unless --tune is passed. Holding hyperparameters "
                         "fixed isolates the effect of the datacenter shift from "
                         "any adaptation to the fold.")
    ap.add_argument("--tune", action="store_true",
                    help="Re-run Optuna per fold (slow; also lets each fold "
                         "adapt its hyperparameters to the reduced training set).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost_matrix", default="data/processed/cost_matrix.json")
    ap.add_argument("--out", default="results/datacenter_holdout.csv")
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--n_holdout", type=int, default=3,
                    help="How many datacenters to hold out.")
    ap.add_argument("--target", default="failure_within_14d")
    ap.add_argument("--max_test_rows", type=int, default=3_000_000,
                    help="Cap on test rows sampled per fold, to bound runtime.")
    args = ap.parse_args()

    lgbm = _load_stage6()
    with open(args.cost_matrix) as f:
        cm = json.load(f)

    test_path = Path(args.test)
    print("[LOAD] mapping serial_number -> datacenter across all splits ...")
    sources = [test_path]
    for extra in (args.val, args.train_resampled):
        if Path(extra).exists():
            sources.append(Path(extra))
    dc_map = _dc_of(sources)
    print(f"[LOAD] {len(dc_map):,} drives have a datacenter label "
          f"(from {len(sources)} splits)")

    counts = dc_map["datacenter"].value_counts()
    # Hold out whole datacenters, largest first so each fold is meaty enough
    # to score. Skipping tiny ones avoids meaningless per-fold AUPRC.
    eligible = [dc for dc, n in counts.items() if n >= 20_000]
    print(f"[PLAN] {len(eligible)} datacenters have >=20,000 drives; "
          f"holding out {min(args.n_holdout, len(eligible))}")
    holdouts = eligible[: args.n_holdout]

    train_df, feature_cols = lgbm.load_train_chunked(
        Path(args.train_resampled), args.target, list(lgbm.INFRA_FEATURE_COLS))
    print(f"[LOAD] training set: {len(train_df):,} rows, {len(feature_cols)} features")
    leaked = [c for c in lgbm.INFRA_FEATURE_COLS if c in feature_cols]
    if leaked:
        raise RuntimeError(f"infrastructure identifiers present in features: {leaked}")

    fixed_params = None
    if not args.tune:
        fixed_params = json.loads(Path(args.best_params).read_text())
        print(f"[PARAMS] reusing tuned hyperparameters from {args.best_params} "
              f"({len(fixed_params)} keys) across all folds")

    # Attach the datacenter to every TRAINING row so a fold can genuinely drop
    # the held-out site.  Rows whose serial carries no label are kept in every
    # fold: we cannot prove which site they belong to, and silently deleting
    # them would change the training set size between folds.
    label = dc_map.set_index("serial_number")["datacenter"]
    train_serials = _serials_in_order(Path(args.train_resampled))
    if len(train_serials) != len(train_df):
        raise RuntimeError(
            f"serial/target misalignment: {len(train_serials):,} serials vs "
            f"{len(train_df):,} training rows -- refusing to guess the mapping")
    train_dc = train_serials.map(label)
    unlabelled = int(train_dc.isna().sum())
    if unlabelled:
        print(f"[LOAD] {unlabelled:,} training rows have no datacenter label; "
              "they stay in every fold")
    train_df = train_df.assign(_dc=train_dc.to_numpy(), _serial=train_serials.to_numpy())
    del train_serials
    gc.collect()

    # Test-frame streaming, one datacenter at a time.
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")

    rows = []
    rng = np.random.default_rng(args.seed)
    # A size-matched RANDOM fold per held-out site.  Without it, any drop is
    # ambiguous: it could be the datacenter shift, or simply that holding out
    # N drives leaves less training data.  The control removes that confound by
    # spending exactly the same amount of training data on a random split.
    folds = ([("datacenter", dc, None) for dc in holdouts]
             + [("control", dc, int(counts[dc])) for dc in holdouts])

    for kind, label_dc, n_drives in folds:
        fold_kind = f"{kind}:{label_dc}"
        if kind == "datacenter":
            held = set(dc_map.loc[dc_map["datacenter"] == label_dc, "serial_number"])
            # EXCLUDE the held-out site's drives from TRAINING as well.  Scoring
            # a site the model was fitted on measures nothing about portability.
            mask = ~train_df["_dc"].eq(label_dc).fillna(False)
        else:
            pool = dc_map.loc[~dc_map["datacenter"].isin(holdouts),
                              "serial_number"].to_numpy()
            held = set(rng.choice(pool, size=min(n_drives, len(pool)), replace=False))
            mask = ~train_df["_serial"].isin(held)
        if not held:
            continue
        train_sub = train_df.loc[mask]
        held_sql = ", ".join("'" + s.replace("'", "''") + "'" for s in sorted(held)[:200000])
        print(f"\n[FOLD] {fold_kind}: {len(held):,} drives held out "
              f"({100 * len(held) / len(dc_map):.1f}% of fleet)")
        print(f"[FOLD]   training rows: {len(train_sub):,} "
              f"({len(train_df) - len(train_sub):,} dropped)")

        if fixed_params is not None:
            model = dict(fixed_params)
        else:
            tune_X = train_sub[feature_cols].head(200_000)
            tune_y = train_sub[args.target].head(200_000)
            model = lgbm.tune_with_optuna(train_sub[feature_cols], train_sub[args.target],
                                          tune_X, tune_y, n_trials=args.n_trials)
            del tune_X, tune_y
            gc.collect()
        clf = lgbm.lgb.LGBMClassifier(**{**model, "objective": "binary", "verbosity": -1,
                                         "random_state": args.seed})
        clf.fit(train_sub[feature_cols], train_sub[args.target])
        gc.collect()

        cols = ", ".join(f'"{c}"' for c in feature_cols)
        q = f"""
            WITH sample AS (
                SELECT * FROM read_parquet('{test_path.resolve().as_posix()}')
                WHERE serial_number IN ({held_sql})
            )
            SELECT "{args.target}" AS target, {cols}
            FROM sample USING SAMPLE {args.max_test_rows} ROWS (reservoir, {args.seed})
        """
        df = con.execute(q).fetchdf()
        y = df["target"].astype(int).to_numpy()
        probs = clf.predict_proba(df[feature_cols])[:, 1]
        pos = int(y.sum())
        if pos == 0:
            print(f"[FOLD]   {fold_kind}: 0 positives in sample -- skipped")
            continue
        ap_ = float(lgbm.average_precision_score(y, probs))
        auc = float(lgbm.roc_auc_score(y, probs))
        lift = ap_ / (pos / len(y))
        rows.append({"fold": fold_kind, "kind": fold_kind.split(":")[0],
                     "datacenter": label_dc, "drives_held_out": int(len(held)),
                     "train_rows": int(len(train_sub)),
                     "sampled_rows": int(len(y)), "positives": pos,
                     "auprc": ap_, "roc_auc": auc, "lift_over_prevalence": lift})
        print(f"[FOLD]   {fold_kind}: AUPRC {ap_:.4f} ({lift:.0f}x prevalence)  "
              f"AUC {auc:.4f}  n={len(y):,} pos={pos:,}")

    con.close()
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\n[SAVE] {args.out}")
    if len(out):
        dc_rows = out[out["kind"] == "datacenter"]
        ct_rows = out[out["kind"] == "control"]
        print(f"\nHeld-out AUPRC across {len(dc_rows)} unseen datacenters: "
              f"{dc_rows['auprc'].min():.4f} .. {dc_rows['auprc'].max():.4f} "
              f"(median {dc_rows['auprc'].median():.4f})")
        if len(ct_rows):
            print(f"Size-matched random-fold control: "
                  f"{ct_rows['auprc'].min():.4f} .. {ct_rows['auprc'].max():.4f} "
                  f"(median {ct_rows['auprc'].median():.4f})")
            print(f"Datacenter-shift cost (median control - median held-out): "
                  f"{ct_rows['auprc'].median() - dc_rows['auprc'].median():+.4f} AUPRC")
        print("Compare against the in-distribution test AUPRC in results/lgbm_metrics.csv.")
        print("A small gap means the model relies on drive health, not site identity.")


if __name__ == "__main__":
    main()