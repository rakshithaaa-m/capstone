"""
Fix 2 — Baseline comparison (Logistic Regression + XGBoost)

Addresses the reviewer expectation that a paper proposing LightGBM/LSTM
justify that choice against simpler/alternative models, not just assert
it. Reuses Stage 6's streaming val/test evaluation so thresholds and
metrics are computed identically to your headline LightGBM result.

Logistic Regression needs standardized features (unlike tree models) --
handled with a StandardScaler fit on the training sample only.

XGBoost is trained the same way as your LightGBM ablations: on the
resampled training set, Optuna-tuned, evaluated via chunked streaming
over the full val/test parquet files (never loaded fully into memory).

Usage:
    python3 fix2_baseline_comparison.py \
        --train_resampled data/processed/train_resampled.parquet \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --cost_matrix data/processed/cost_matrix.json \
        --lgbm_metrics results/lgbm_metrics.csv \
        --target failure_within_14d \
        --xgb_n_trials 10 \
        --out_dir results
"""
import argparse
import gc
import json
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stream_predict_sklearn_style(predict_fn, path: Path, feature_cols, target: str, lgbm_mod):
    """Same chunked-streaming pattern as Stage 6's stream_predict_booster,
    generalized to accept any predict_fn(X) -> probs (works for sklearn's
    .predict_proba, XGBoost's .predict_proba, etc), not just a raw
    lgb.Booster. Never holds the full val/test file in memory."""
    total_rows = lgbm_mod.pq.ParquetFile(path).metadata.num_rows
    out_probs = np.empty(total_rows, dtype="float32")
    out_labels = np.empty(total_rows, dtype="int8")
    pos = 0
    pf = lgbm_mod.pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=lgbm_mod.PREDICT_CHUNK_SIZE, columns=feature_cols + [target]):
        df = batch.to_pandas()
        df = lgbm_mod.coerce_legacy_format(df)
        X = df[feature_cols].astype("float32").fillna(0.0)
        probs = predict_fn(X)
        n = len(df)
        out_probs[pos:pos + n] = probs
        out_labels[pos:pos + n] = df[target].to_numpy()
        pos += n
        del df, X, probs
        gc.collect()
    return out_labels[:pos], out_probs[:pos]


def run_logistic_regression(train_df, feature_cols, target, val_path, test_path, cost_matrix, lgbm_mod):
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df[target]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    def predict_fn(X):
        return model.predict_proba(scaler.transform(X.fillna(0.0)))[:, 1]

    y_val, val_probs = stream_predict_sklearn_style(predict_fn, val_path, feature_cols, target, lgbm_mod)
    threshold = lgbm_mod.select_cost_optimal_threshold(
        y_val, val_probs, cost_matrix["false_negative_cost"], cost_matrix["false_positive_cost"]
    )
    y_test, test_probs = stream_predict_sklearn_style(predict_fn, test_path, feature_cols, target, lgbm_mod)
    return lgbm_mod.evaluate(y_test, test_probs, threshold)


def run_xgboost(train_df, feature_cols, target, val_path, test_path, cost_matrix, n_trials, lgbm_mod):
    import optuna
    X_train, y_train = train_df[feature_cols], train_df[target]

    X_val_sample, y_val_sample = lgbm_mod.load_val_tuning_sample(
        val_path, feature_cols, target, lgbm_mod.VAL_TUNING_SAMPLE_ROWS
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train)
        preds = m.predict_proba(X_val_sample)[:, 1]
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_val_sample, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    del X_val_sample, y_val_sample
    gc.collect()

    best_params = {**study.best_params, "objective": "binary:logistic", "eval_metric": "aucpr", "random_state": 42}
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    def predict_fn(X):
        return model.predict_proba(X)[:, 1]

    y_val, val_probs = stream_predict_sklearn_style(predict_fn, val_path, feature_cols, target, lgbm_mod)
    threshold = lgbm_mod.select_cost_optimal_threshold(
        y_val, val_probs, cost_matrix["false_negative_cost"], cost_matrix["false_positive_cost"]
    )
    y_test, test_probs = stream_predict_sklearn_style(predict_fn, test_path, feature_cols, target, lgbm_mod)
    return lgbm_mod.evaluate(y_test, test_probs, threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_resampled", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--cost_matrix", type=str, required=True)
    parser.add_argument("--lgbm_metrics", type=str, required=True)
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--xgb_n_trials", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lgbm_mod = load_module("lgbm_stage6", Path(__file__).parent / "06_train_lightgbm.py")
    with open(args.cost_matrix) as f:
        cost_matrix = json.load(f)
    lgbm_result = pd.read_csv(args.lgbm_metrics).iloc[0].to_dict()

    train_df, feature_cols = lgbm_mod.load_train_chunked(Path(args.train_resampled), args.target)

    print("=== Baseline 1: Logistic Regression ===")
    logreg_result = run_logistic_regression(train_df, feature_cols, args.target, Path(args.val), Path(args.test), cost_matrix, lgbm_mod)
    print(logreg_result)

    print("\n=== Baseline 2: XGBoost ===")
    xgb_result = run_xgboost(train_df, feature_cols, args.target, Path(args.val), Path(args.test), cost_matrix, args.xgb_n_trials, lgbm_mod)
    print(xgb_result)
    del train_df
    gc.collect()

    comparison = pd.DataFrame([
        {"model": "Logistic Regression", **{k: logreg_result[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
        {"model": "XGBoost", **{k: xgb_result[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
        {"model": "LightGBM (your original)", **{k: lgbm_result[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
    ])
    print("\n" + comparison.to_string(index=False))
    comparison.to_csv(out_dir / "baseline_comparison.csv", index=False)
    print(f"\n[SAVE] {out_dir/'baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
