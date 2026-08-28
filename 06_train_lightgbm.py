"""
Stage 6 — LightGBM Classifier (MEMORY-SAFE, STREAMED VAL/TEST)

Trains a LightGBM binary classifier on the resampled training set,
predicting `failure_within_{horizon}d`. Tunes hyperparameters with Optuna
optimizing AUPRC, then selects the final decision threshold by minimizing
expected cost on the VALIDATION set (never on test) using the cost matrix
from Stage 5.

The original version called pd.read_parquet() on val (52.8M rows) and test
(144.7M rows) in full. With ~226 mostly-float64 columns per row, either of
those loaded whole is on the order of 100-250+ GB in memory -- not
"tight", but categorically impossible on laptop-class hardware regardless
of what else is closed. This version NEVER loads val or test as a full
feature table:

  - TRAIN (resampled, ~31.6M rows) is loaded once, in full -- this is the
    one dataset Stage 5 deliberately shrank to be laptop-sized.
  - VAL and TEST are streamed in row-chunks for prediction. A trained
    model only needs one small chunk of raw features in memory at a time
    to produce predictions; the output (one probability per row) is tiny
    even for 144M rows, so only that small array is ever kept in full.

One documented tradeoff for speed: Optuna's hyperparameter search needs a
validation score on EVERY trial (default 30). Streaming the full 52.8M-row
validation file 30 times would be very slow. So tuning uses a fixed
~5M-row representative SAMPLE of validation (drawn once, kept in memory,
reused across all trials) purely to compare hyperparameter candidates
quickly. This sample is safe to use for relative comparison between
configs -- it is NOT used for anything reported as a final number. The
FINAL cost-optimal threshold selection and the FINAL test evaluation both
stream the complete, real, unsampled val/test data exactly once each,
so nothing reported as a result is based on a subsample.

Usage:
    python3 06_train_lightgbm.py \
        --train data/processed/train_resampled.parquet \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --cost_matrix data/processed/cost_matrix.json \
        --target failure_within_14d \
        --n_trials 30 \
        --out_dir models results
"""
import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

NON_FEATURE_COLS = [
    "date", "serial_number", "model", "capacity_bytes",
    "failure", "rul_days", "is_censored",
    "datacenter",  # text label, not encoded -- excluded rather than mishandled
]
ROW_CHUNK_SIZE = 200_000
VAL_TUNING_SAMPLE_ROWS = 5_000_000  # size of the fixed sample used ONLY
                                     # for Optuna's repeated trials, not
                                     # for any final reported number.


def get_feature_cols(df: pd.DataFrame, target: str) -> list[str]:
    # is_legacy_format is a real, informative True/False flag -- convert it
    # to 0/1 in place so it's usable as a feature, rather than excluding a
    # column that might actually carry signal.
    if "is_legacy_format" in df.columns and not pd.api.types.is_numeric_dtype(df["is_legacy_format"]):
        df["is_legacy_format"] = df["is_legacy_format"].astype(bool).astype("int8")

    exclude = set(NON_FEATURE_COLS) | {target}
    exclude |= {c for c in df.columns if c.startswith("failure_within_") and c != target}
    cols = [c for c in df.columns if c not in exclude]
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found, exclude or encode them: {non_numeric}")
    return cols


def downcast_floats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """float64 -> float32 halves memory footprint with negligible precision
    loss for gradient-boosted trees. Applied on every chunk read."""
    float_cols = [c for c in cols if pd.api.types.is_float_dtype(df[c])]
    if float_cols:
        df[float_cols] = df[float_cols].astype("float32")
    return df


def load_train_chunked(path: Path, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Reads the resampled training set in chunks, downcasting each chunk
    to float32 immediately -- never holding a full float64 copy of all
    31.6M rows at once, even momentarily. The earlier version's single
    pd.read_parquet() call loaded everything as float64 BEFORE any
    downcasting could help, which is what actually caused the silent
    OOM kill (Windows terminates the process outright with no traceback,
    rather than raising a catchable Python error)."""
    parquet_file = pq.ParquetFile(path)
    feature_cols = None
    chunks = []
    rows_so_far = 0
    for batch in parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE):
        df = batch.to_pandas()
        if feature_cols is None:
            feature_cols = get_feature_cols(df, target)
        df = downcast_floats(df, feature_cols)
        chunks.append(df[feature_cols + [target]])
        rows_so_far += len(df)
        del df
        if len(chunks) % 5 == 0:
            gc.collect()
        if rows_so_far % (ROW_CHUNK_SIZE * 10) == 0:
            print(f"[LOAD] {rows_so_far:,} training rows read so far...")
    print(f"[LOAD] Concatenating {rows_so_far:,} already-downcasted rows...")
    gc.collect()
    full = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return full, feature_cols


def load_val_tuning_sample(path: Path, feature_cols: list[str], target: str,
                            max_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    """Reads a fixed-size PREFIX of validation (first ~max_rows rows) once,
    kept in memory for the whole Optuna run. Val was written by Stage 4 in
    batch order, and Stage 3's batch assignment was a hash of serial_number
    -- i.e. each batch is a random subset of drives, not a date-ordered or
    otherwise biased slice. So a prefix of a few batches' worth of rows is
    a reasonable representative sample of drives for comparing hyperparameter
    candidates, without needing a full extra pass to build a proper random
    sample. This is only for tuning speed -- see module docstring."""
    parquet_file = pq.ParquetFile(path)
    chunks = []
    rows_so_far = 0
    for batch in parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE, columns=feature_cols + [target]):
        df = batch.to_pandas()
        df = downcast_floats(df, feature_cols)
        chunks.append(df)
        rows_so_far += len(df)
        if rows_so_far >= max_rows:
            break
    sample = pd.concat(chunks, ignore_index=True)
    print(f"[VAL SAMPLE] Using {len(sample):,} rows (of the full validation set) for Optuna tuning only")
    return sample[feature_cols], sample[target]


def stream_predict(model, path: Path, feature_cols: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    """Streams a parquet file in chunks, predicting on each small chunk in
    turn. Only the resulting (small) probability and label arrays are kept
    for the whole file -- the raw feature chunks are discarded immediately
    after each prediction."""
    parquet_file = pq.ParquetFile(path)
    all_probs = []
    all_labels = []
    for batch in parquet_file.iter_batches(batch_size=ROW_CHUNK_SIZE, columns=feature_cols + [target]):
        df = batch.to_pandas()
        df = downcast_floats(df, feature_cols)
        probs = model.predict_proba(df[feature_cols])[:, 1]
        all_probs.append(probs)
        all_labels.append(df[target].to_numpy())
        del df, probs
    return np.concatenate(all_labels), np.concatenate(all_probs)


def tune_with_optuna(X_train, y_train, X_val_sample, y_val_sample, n_trials: int, seed: int = 42) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "random_state": seed,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val_sample)[:, 1]
        if y_val_sample.sum() == 0:
            raise ValueError(
                "Validation sample has zero positive examples — cannot tune AUPRC. "
                "This shouldn't happen given the sample size relative to the known "
                "validation positive rate; check VAL_TUNING_SAMPLE_ROWS."
            )
        return average_precision_score(y_val_sample, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def select_cost_optimal_threshold(y_val, probs_val, fn_cost: float, fp_cost: float) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_cost = 0.5, np.inf
    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost = cost
            best_thresh = t
    return float(best_thresh)


def evaluate(y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probs) if y_true.sum() > 0 else float("nan"),
        "auprc": average_precision_score(y_true, probs) if y_true.sum() > 0 else float("nan"),
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the LightGBM classifier (streamed val/test)")
    parser.add_argument("--train", type=str, required=True, help="Resampled training set from Stage 5")
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--cost_matrix", type=str, required=True)
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--out_dir", nargs=2, default=["models", "results"], metavar=("MODELS_DIR", "RESULTS_DIR"))
    args = parser.parse_args()

    models_dir, results_dir = Path(args.out_dir[0]), Path(args.out_dir[1])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] Reading resampled training set (chunked, downcast to float32 as it goes)...")
    train_df, feature_cols = load_train_chunked(Path(args.train), args.target)

    with open(args.cost_matrix) as f:
        cost_matrix = json.load(f)

    print(f"Using {len(feature_cols)} feature columns")

    X_train, y_train = train_df[feature_cols], train_df[args.target]

    print("[LOAD] Sampling validation for tuning (fixed prefix, kept in memory across trials)...")
    X_val_sample, y_val_sample = load_val_tuning_sample(
        Path(args.val), feature_cols, args.target, VAL_TUNING_SAMPLE_ROWS
    )

    print(f"[OPTUNA] Running {args.n_trials} trials optimizing AUPRC on the validation sample...")
    best_params = tune_with_optuna(X_train, y_train, X_val_sample, y_val_sample, n_trials=args.n_trials)
    print(f"[OPTUNA] Best params: {best_params}")
    del X_val_sample, y_val_sample  # done with the sample, free it before the full streaming passes

    final_params = {**best_params, "objective": "binary", "metric": "average_precision", "verbosity": -1, "random_state": 42}
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train, y_train)
    del train_df, X_train, y_train  # free training data before the streaming val/test passes

    print("[STREAM] Predicting on FULL validation set (chunked) for threshold selection...")
    y_val_full, val_probs_full = stream_predict(model, Path(args.val), feature_cols, args.target)
    threshold = select_cost_optimal_threshold(
        y_val_full, val_probs_full, cost_matrix["false_negative_cost"], cost_matrix["false_positive_cost"]
    )
    print(f"[THRESHOLD] Cost-optimal threshold on FULL validation: {threshold:.3f}")
    del y_val_full, val_probs_full

    print("[STREAM] Predicting on FULL test set (chunked) for final evaluation...")
    y_test_full, test_probs_full = stream_predict(model, Path(args.test), feature_cols, args.target)
    metrics = evaluate(y_test_full, test_probs_full, threshold)
    print("[TEST METRICS]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    model.booster_.save_model(str(models_dir / "lgbm_model.txt"))
    with open(models_dir / "lgbm_threshold.json", "w") as f:
        json.dump({"threshold": threshold, "target": args.target, "feature_cols": feature_cols}, f, indent=2)
    pd.Series(test_probs_full, name="lgbm_prob").to_csv(results_dir / "lgbm_test_probs.csv", index=False)
    pd.DataFrame([metrics]).to_csv(results_dir / "lgbm_metrics.csv", index=False)

    print(f"[SAVE] {models_dir/'lgbm_model.txt'}")
    print(f"[SAVE] {models_dir/'lgbm_threshold.json'}")
    print(f"[SAVE] {results_dir/'lgbm_metrics.csv'}")


if __name__ == "__main__":
    main()