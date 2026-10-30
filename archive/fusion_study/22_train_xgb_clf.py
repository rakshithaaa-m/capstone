"""
Stage 22 -- Amendment 03, candidate P-B: tuned XGBoost failure classifier.

Question
--------
Does a second strong gradient-boosting classifier, trained on the same rows,
features and target as LightGBM, add ranking information when fused with it?
It also replaces the 0.0020-AUPRC XGBoost baseline in
results/baseline_comparison.csv, which looks misconfigured.

The tuning deliberately mirrors Stage 6 (06_train_lightgbm.py):
* training rows: data/processed/train_resampled.parquet
* features: models/lgbm_feature_cols.json (212); target failure_within_14d
* Optuna, 30 trials, TPE seed 42, maximising AUPRC on a fixed 5,000,000-row
  prefix of the validation split
* the number of trees is a tuned parameter (100-800), not early-stopped;
  no class weighting, because the training set is already undersampled
* parameters with an XGBoost equivalent use Stage 6's ranges;
  ``min_child_weight`` stands in for ``min_child_samples``.

Output: results_fusion/pb_xgb_clf_{val,test}_preds.csv with columns
serial_number,date,xgb_prob -- run through 16_stacking_study.py with
``--partner_kind prob``.

``--ensemble`` (Amendment 03 §7.1, only after the §6 screen passed): no new
tuning.  The tuned parameters in pb_xgb_clf_params.json are refitted with
LightGBM's ensemble seeds (models/lgbm_ensemble_seeds.json), and the mean of the
five probabilities is written to pb_xgb_clf_ens_{val,test}_preds.csv.

Usage
-----
    python 22_train_xgb_clf.py
    python 22_train_xgb_clf.py --ensemble
Smoke test:
    python 22_train_xgb_clf.py --out_dir <scratch> --model_dir <scratch> \\
        --max_train_rows 200000 --tune_rows 300000 --n_trials 2 --predict_row_groups 3
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
TUNE_ROWS = 5_000_000
TARGET = "failure_within_14d"
PRED_COL = "xgb_prob"


def read_rows(path: Path, feature_cols: list[str], max_rows: int | None):
    """Features into one preallocated float32 matrix, plus the target."""
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows if max_rows is None else min(max_rows, pf.metadata.num_rows)
    X = np.empty((n, len(feature_cols)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    done = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=feature_cols + [TARGET]):
        take = min(batch.num_rows, n - done)
        b = batch.slice(0, take)
        for j, c in enumerate(feature_cols):
            X[done:done + take, j] = b.column(c).to_numpy(zero_copy_only=False).astype(
                np.float32, copy=False)
        y[done:done + take] = b.column(TARGET).to_numpy(zero_copy_only=False)
        done += take
        if done >= n:
            break
    return X, y


BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "nthread": -1,
    "verbosity": 0,
}


def tune(d_train, X_val, y_val, n_trials: int) -> dict:
    import optuna
    import xgboost as xgb
    from sklearn.metrics import average_precision_score

    if y_val.sum() == 0:
        raise ValueError("validation tuning sample has no positives")

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", 100, 800)
        params = {
            **BASE_PARAMS,
            "seed": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
        bst = xgb.train(params, d_train, num_boost_round=n_estimators, verbose_eval=False)
        ap = float(average_precision_score(y_val, bst.inplace_predict(X_val)))
        print(f"[OPTUNA] trial {trial.number:>2}  val AUPRC {ap:.5f}  trees {n_estimators}")
        return ap

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[OPTUNA] best val AUPRC {study.best_value:.5f}")
    return dict(study.best_params)


def stream_predict(bst, path: Path, feature_cols: list[str], out_csv: Path,
                   max_row_groups: int | None) -> int:
    """Predict every row of a split -> serial_number,date,xgb_prob (LF endings).

    ``bst`` may be a list of boosters; their probabilities are averaged."""
    boosters = bst if isinstance(bst, list) else [bst]
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
            pred = np.mean([b.inplace_predict(X) for b in boosters], axis=0)
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
    ap.add_argument("--val", default="data/processed/features_val.parquet")
    ap.add_argument("--test", default="data/processed/features_test.parquet")
    ap.add_argument("--feature_cols", default="models/lgbm_feature_cols.json")
    ap.add_argument("--model_dir", default="models_fusion")
    ap.add_argument("--out_dir", default="results_fusion")
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--max_train_rows", type=int, default=None, help="smoke tests only")
    ap.add_argument("--tune_rows", type=int, default=TUNE_ROWS)
    ap.add_argument("--predict_row_groups", type=int, default=None, help="smoke tests only")
    ap.add_argument("--ensemble", action="store_true",
                    help="refit the tuned parameters with LightGBM's five seeds (§7.1)")
    ap.add_argument("--seeds_json", default="models/lgbm_ensemble_seeds.json")
    args = ap.parse_args()

    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    model_dir, out_dir = Path(args.model_dir), Path(args.out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = json.loads(Path(args.feature_cols).read_text())["feature_cols"]
    print(f"[SETUP] xgboost {xgb.__version__}, {len(feature_cols)} features, seed {SEED}")

    if args.ensemble:
        run_ensemble(args, feature_cols, model_dir, out_dir)
        return

    X_tr, y_tr = read_rows(Path(args.train), feature_cols, args.max_train_rows)
    print(f"[LOAD] train {len(y_tr):,} rows, positives {int(y_tr.sum()):,}")
    X_va, y_va = read_rows(Path(args.val), feature_cols, args.tune_rows)
    print(f"[LOAD] val tuning sample {len(y_va):,} rows, positives {int(y_va.sum()):,}")

    d_tr = xgb.DMatrix(X_tr, label=y_tr, nthread=-1)
    del X_tr

    print(f"[OPTUNA] {args.n_trials} trials maximising validation AUPRC ...")
    best = tune(d_tr, X_va, y_va, args.n_trials)
    n_estimators = best.pop("n_estimators")
    params = {**BASE_PARAMS, **best, "seed": SEED}
    print(f"[TRAIN] final model: {params}  trees {n_estimators}")
    bst = xgb.train(params, d_tr, num_boost_round=n_estimators, verbose_eval=False)
    p_va = bst.inplace_predict(X_va)
    sample_ap = float(average_precision_score(y_va, p_va))
    print(f"[TRAIN] val tuning-sample AUPRC {sample_ap:.5f}  "
          f"ROC-AUC {roc_auc_score(y_va, p_va):.5f}")

    bst.save_model(model_dir / "pb_xgb_clf.json")
    (model_dir / "pb_xgb_clf_params.json").write_text(json.dumps(
        {"params": params, "num_boost_round": n_estimators, "n_trials": args.n_trials,
         "train_rows": int(d_tr.num_row()), "tune_rows": int(len(y_va)),
         "val_tuning_sample_auprc": sample_ap}, indent=2))
    print(f"[SAVE] {model_dir / 'pb_xgb_clf.json'}")
    del d_tr, X_va

    for split, path in (("val", args.val), ("test", args.test)):
        out = out_dir / f"pb_xgb_clf_{split}_preds.csv"
        n = stream_predict(bst, Path(path), feature_cols, out, args.predict_row_groups)
        print(f"[SAVE] {out}  {n:,} rows")


def run_ensemble(args, feature_cols: list[str], model_dir: Path, out_dir: Path) -> None:
    import xgboost as xgb

    saved = json.loads((model_dir / "pb_xgb_clf_params.json").read_text())
    params, rounds = saved["params"], saved["num_boost_round"]
    seeds = json.loads(Path(args.seeds_json).read_text())["seeds"]
    print(f"[ENSEMBLE] seeds {seeds}, {rounds} trees, tuned params from "
          f"{model_dir / 'pb_xgb_clf_params.json'}")

    paths = [model_dir / f"pb_xgb_clf_seed{seed}.json" for seed in seeds]
    if all(p.exists() for p in paths):
        # Resume after an interrupted prediction pass: the seed models are
        # deterministic outputs of the run that saved them; reload, don't refit.
        boosters = [xgb.Booster(model_file=str(p)) for p in paths]
        print(f"[ENSEMBLE] reloaded {len(boosters)} saved seed models")
    else:
        X_tr, y_tr = read_rows(Path(args.train), feature_cols, args.max_train_rows)
        d_tr = xgb.DMatrix(X_tr, label=y_tr, nthread=-1)
        del X_tr
        boosters = []
        for i, (seed, path) in enumerate(zip(seeds, paths), start=1):
            t0 = time.time()
            bst = xgb.train({**params, "seed": seed}, d_tr, num_boost_round=rounds,
                            verbose_eval=False)
            bst.save_model(path)
            boosters.append(bst)
            print(f"[ENSEMBLE] {i}/{len(seeds)} fitted (seed={seed})  {time.time() - t0:,.0f}s")
        del d_tr

    for split, path in (("val", args.val), ("test", args.test)):
        out = out_dir / f"pb_xgb_clf_ens_{split}_preds.csv"
        if out.exists():
            print(f"[SKIP] {out} already complete")
            continue
        n = stream_predict(boosters, Path(path), feature_cols, out, args.predict_row_groups)
        print(f"[SAVE] {out}  {n:,} rows")


if __name__ == "__main__":
    main()
