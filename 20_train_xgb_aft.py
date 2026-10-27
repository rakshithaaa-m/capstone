"""
Stage 20 -- Amendment 03, candidate P-A: XGBoost survival (AFT) RUL model.

Question
--------
Is the fusion null (Stage 16) a property of the LSTM, or of any RUL estimator
on these features?  P-A is a stronger, censoring-aware RUL model trained on
exactly the rows and features the LightGBM reference uses.  Its predictions
then go through ``16_stacking_study.py --partner_kind rul`` unchanged, and the
test predictions feed the RUL-output comparison (Amendment 03, section 7b).

Labels (accelerated failure time, interval-censored form)
---------------------------------------------------------
Time is measured as ``rul_days + 1`` so that the failure day itself is 1, not
0 -- the AFT objective works on log(time).

* drive fails (``is_censored == 0``):  lower = upper = rul_days + 1
* drive survives (``is_censored == 1``): lower = (last day the drive is
  observed IN THIS SPLIT - row date) + 1, upper = +inf

The censored bound is the number of days the drive is KNOWN to survive after
the row.  It is NOT ``days_since_install``: age bounds total lifetime, not
remaining life (07_train_lstm_v2.py uses age, which is recorded as a finding in
Amendment 03).  "In this split" keeps every bound inside the period the split
covers.  ``train_resampled.parquet`` is undersampled, so its last-observed
dates are taken from the full ``features_train.parquet``.

Usage
-----
Full run (tune, train, predict val + test):
    python 20_train_xgb_aft.py

Smoke test (minutes; writes to a scratch folder):
    python 20_train_xgb_aft.py --out_dir <scratch> --model_dir <scratch> \\
        --max_train_rows 200000 --tune_rows 200000 --n_trials 2 \\
        --predict_row_groups 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 42
TUNE_ROWS = 5_000_000      # same size as Stage 6's VAL_TUNING_SAMPLE_ROWS
PRED_COL = "aft_rul_pred"


def last_observed(path: Path) -> pd.DataFrame:
    """Last date each drive appears in a split, read from the full parquet."""
    import duckdb

    con = duckdb.connect(config={"memory_limit": "6GB",
                                 "temp_directory": str(Path.home() / "duck_spool")})
    try:
        df = con.execute(f"""
            SELECT serial_number, MAX(date) AS last_date
            FROM read_parquet('{path.resolve().as_posix()}')
            GROUP BY 1 ORDER BY 1
        """).fetchdf()
    finally:
        con.close()
    return df


def aft_bounds(df: pd.DataFrame, last: pd.DataFrame
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower/upper AFT label bounds in days (+1), plus a mask of usable rows.

    Some failing drives keep reporting after their (last) recorded failure --
    up to ~3,900 days later, i.e. returned or serial-reused drives.  Their
    ``rul_days`` is negative and has no meaning as remaining life, so those
    rows are masked out of training and tuning.  They still get predictions.
    """
    last_date = df["serial_number"].map(last.set_index("serial_number")["last_date"])
    if last_date.isna().any():
        raise ValueError(f"{int(last_date.isna().sum())} rows have no last-observed date")
    observed = (pd.to_datetime(last_date) - pd.to_datetime(df["date"])).dt.days.to_numpy()
    rul = df["rul_days"].to_numpy(dtype=np.float64)
    censored = df["is_censored"].to_numpy().astype(bool)
    if np.any(~censored & ~np.isfinite(rul)):
        raise ValueError("failing rows with missing rul_days")
    keep = censored | (rul >= 0)
    lower = np.where(censored, observed, rul) + 1.0
    upper = np.where(censored, np.inf, rul + 1.0)
    return lower.astype(np.float32), upper.astype(np.float32), keep


def read_rows(path: Path, feature_cols: list[str], max_rows: int | None):
    """Read features into one preallocated float32 matrix (no pandas copy)."""
    meta_cols = ["serial_number", "date", "rul_days", "is_censored"]
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows if max_rows is None else min(max_rows, pf.metadata.num_rows)
    X = np.empty((n, len(feature_cols)), dtype=np.float32)
    meta = []
    done = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=feature_cols + meta_cols):
        take = min(batch.num_rows, n - done)
        b = batch.slice(0, take)
        for j, c in enumerate(feature_cols):
            X[done:done + take, j] = b.column(c).to_numpy(zero_copy_only=False).astype(
                np.float32, copy=False)
        meta.append(b.select(meta_cols).to_pandas())
        done += take
        if done >= n:
            break
    return X, pd.concat(meta, ignore_index=True)


def dmatrix(X, lower, upper):
    import xgboost as xgb
    d = xgb.DMatrix(X, nthread=-1)
    d.set_float_info("label_lower_bound", lower)
    d.set_float_info("label_upper_bound", upper)
    return d


BASE_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "tree_method": "hist",
    "nthread": -1,
    "verbosity": 0,
}


def tune(d_train, d_val, n_trials: int) -> tuple[dict, int]:
    import optuna
    import xgboost as xgb

    def objective(trial: optuna.Trial) -> float:
        params = {
            **BASE_PARAMS,
            "seed": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "aft_loss_distribution_scale": trial.suggest_float(
                "aft_loss_distribution_scale", 0.5, 2.0),
        }
        bst = xgb.train(params, d_train, num_boost_round=2000,
                        evals=[(d_val, "val")], early_stopping_rounds=50,
                        verbose_eval=False)
        trial.set_user_attr("best_iteration", int(bst.best_iteration))
        print(f"[OPTUNA] trial {trial.number:>2}  val aft-nloglik "
              f"{bst.best_score:.5f}  rounds {bst.best_iteration + 1}")
        return float(bst.best_score)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    return dict(best.params), int(best.user_attrs["best_iteration"]) + 1


def stream_predict(bst, path: Path, feature_cols: list[str], out_csv: Path,
                   max_row_groups: int | None) -> int:
    """Predict every row of a split, writing serial_number,date,aft_rul_pred.

    Written to ``.partial`` and renamed at the end, so an interrupted run never
    leaves a file that looks complete.
    """
    pf = pq.ParquetFile(path)
    groups = range(pf.metadata.num_row_groups if max_row_groups is None
                   else min(max_row_groups, pf.metadata.num_row_groups))
    tmp = out_csv.with_suffix(".csv.partial")
    n = 0
    t0 = time.time()
    with open(tmp, "w", newline="") as fh:
        fh.write(f"serial_number,date,{PRED_COL}\n")
        for k, g in enumerate(groups):
            tbl = pf.read_row_group(g, columns=feature_cols + ["serial_number", "date"])
            X = np.column_stack([tbl.column(c).to_numpy(zero_copy_only=False)
                                 .astype(np.float32, copy=False) for c in feature_cols])
            # Back to days: the model predicts rul_days + 1.
            pred = np.clip(bst.inplace_predict(X) - 1.0, 0.0, None)
            pd.DataFrame({
                "serial_number": tbl.column("serial_number").to_pandas(),
                "date": pd.to_datetime(tbl.column("date").to_pandas()).dt.strftime("%Y-%m-%d"),
                PRED_COL: pred.astype(np.float32),
            }).to_csv(fh, header=False, index=False, lineterminator="\n")
            n += len(pred)
            if k % 100 == 0:
                print(f"[PREDICT] {path.name} group {k + 1}/{len(groups)}  rows {n:,}  "
                      f"{time.time() - t0:,.0f}s")
    tmp.replace(out_csv)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train_resampled.parquet")
    ap.add_argument("--train_full", default="data/processed/features_train.parquet",
                    help="full training split, used only for last-observed dates")
    ap.add_argument("--val", default="data/processed/features_val.parquet")
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--feature_cols", default="models/lgbm_feature_cols.json")
    ap.add_argument("--model_dir", default="models_fusion")
    ap.add_argument("--out_dir", default="results_fusion")
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--max_train_rows", type=int, default=None, help="smoke tests only")
    ap.add_argument("--tune_rows", type=int, default=TUNE_ROWS)
    ap.add_argument("--predict_row_groups", type=int, default=None, help="smoke tests only")
    ap.add_argument("--skip_predict", action="store_true")
    args = ap.parse_args()

    import xgboost as xgb

    model_dir, out_dir = Path(args.model_dir), Path(args.out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = json.loads(Path(args.feature_cols).read_text())["feature_cols"]
    print(f"[SETUP] xgboost {xgb.__version__}, {len(feature_cols)} features, seed {SEED}")

    print("[LOAD] last-observed dates per drive (train, val) ...")
    last_train = last_observed(Path(args.train_full))
    last_val = last_observed(Path(args.val))

    print("[LOAD] training rows ...")
    X_tr, m_tr = read_rows(Path(args.train), feature_cols, args.max_train_rows)
    lo_tr, hi_tr, keep = aft_bounds(m_tr, last_train)
    print(f"[LOAD] train {len(m_tr):,} rows, censored {100 * m_tr['is_censored'].mean():.2f}%, "
          f"dropped {int((~keep).sum()):,} post-failure rows "
          f"({m_tr.loc[~keep, 'serial_number'].nunique():,} drives)")
    X_tr, lo_tr, hi_tr = X_tr[keep], lo_tr[keep], hi_tr[keep]
    n_drop_tr = int((~keep).sum())

    print("[LOAD] validation tuning sample (fixed prefix, as Stage 6) ...")
    X_va, m_va = read_rows(Path(args.val), feature_cols, args.tune_rows)
    lo_va, hi_va, keep = aft_bounds(m_va, last_val)
    print(f"[LOAD] val sample {len(m_va):,} rows, "
          f"censored {100 * m_va['is_censored'].mean():.2f}%, "
          f"dropped {int((~keep).sum()):,} post-failure rows")
    X_va, lo_va, hi_va = X_va[keep], lo_va[keep], hi_va[keep]
    n_drop_va = int((~keep).sum())

    d_tr = dmatrix(X_tr, lo_tr, hi_tr)
    d_va = dmatrix(X_va, lo_va, hi_va)
    del X_tr, X_va

    print(f"[OPTUNA] {args.n_trials} trials minimising validation aft-nloglik ...")
    best_params, rounds = tune(d_tr, d_va, args.n_trials)
    print(f"[OPTUNA] best {best_params}  rounds {rounds}")

    params = {**BASE_PARAMS, **best_params, "seed": SEED}
    bst = xgb.train(params, d_tr, num_boost_round=rounds,
                    evals=[(d_va, "val")], verbose_eval=False)
    model_path = model_dir / "pa_xgb_aft.json"
    bst.save_model(model_path)
    (model_dir / "pa_xgb_aft_params.json").write_text(json.dumps(
        {"params": params, "num_boost_round": rounds, "n_trials": args.n_trials,
         "train_rows": int(d_tr.num_row()), "tune_rows": int(d_va.num_row()),
         "dropped_post_failure_rows": {"train": n_drop_tr, "tune": n_drop_va},
         "time_unit": "rul_days + 1 (prediction file is back in days)"}, indent=2))
    print(f"[SAVE] {model_path}")
    del d_tr, d_va

    if args.skip_predict:
        return
    for split, path in (("val", args.val), ("test", args.test)):
        out = out_dir / f"pa_xgb_aft_{split}_preds.csv"
        n = stream_predict(bst, Path(path), feature_cols, out, args.predict_row_groups)
        print(f"[SAVE] {out}  {n:,} rows")


if __name__ == "__main__":
    main()
