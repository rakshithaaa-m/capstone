"""
Stage 10 — Full Evaluation: Comparison, Ablation, Generalizability
(memory-safe version)

CHANGE FROM THE PREVIOUS VERSION: the original script called
pd.read_parquet() on the FULL val (52M rows), test (145M rows), and
train_full (unresampled, likely 400M+ rows) files -- all with the full
217-column feature set. That's dramatically heavier than anything that
has succeeded so far in this pipeline (the CSVs that worked in Stage 9
were already-joined, ~8-column files; this was asking for the full raw
feature matrix at that same row count, which is roughly 25x more data
per row and would not fit in memory on this machine).

This version instead reuses the exact memory-safe machinery Stage 6
already built and proved out, imported directly rather than
re-implemented:
    - load_train_chunked()      -- chunked+downcast read (used as-is for
                                    the already-small resampled train set)
    - load_val_tuning_sample()  -- bounded prefix sample for Optuna tuning
    - stream_predict_booster()  -- chunked prediction over the FULL val/
                                    test parquet files, never holding more
                                    than one chunk in memory at a time
    - select_cost_optimal_threshold(), evaluate()

For ablation 2b ("without imbalance handling"), training on the entire
unresampled train set (hundreds of millions of rows) is infeasible on a
laptop regardless of memory tricks -- LightGBM's own .fit() needs the
training data materialized at least once. Instead we take a bounded
PREFIX SAMPLE directly from the unresampled parquet (same pattern Stage 6
already uses for its val tuning sample), which preserves the natural
class imbalance ratio. This is a resource-driven simplification, called
out explicitly in the printed output and worth one sentence in the
report -- not something to hide.

For generalizability-by-drive-model, rather than loading the full test
set to get its "model" column, DuckDB pulls the (much smaller) distinct
list of drive models directly from the parquet, then queries each
model's rows one at a time against Stage 6's already-computed
lgbm_test_probs.csv -- bounded memory regardless of how many distinct
drive models exist, since only one model's rows are ever in memory.

Usage:
    python3 10_evaluation.py \
        --train_resampled data/processed/train_resampled.parquet \
        --train_full data/processed/features_train.parquet \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --cost_matrix data/processed/cost_matrix.json \
        --lgbm_metrics results/lgbm_metrics.csv \
        --lgbm_test_probs results/lgbm_test_probs.csv \
        --hybrid_metrics results/hybrid_metrics.csv \
        --hybrid_test_scores results/hybrid_test_scores.csv \
        --target failure_within_14d \
        --horizon_days 14 \
        --ablation_n_trials 10 \
        --ablation_2b_sample_rows 5000000 \
        --out_dir results
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

# A per-model ROC-AUC estimated from fewer positives than this has a
# confidence interval wide enough to be uninformative.  Rows below the
# threshold are still computed and reported, but flagged as low power.
MIN_POSITIVES_FOR_PER_MODEL_METRIC = 30


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_train_sample_chunked(path: Path, target: str, max_rows: int, lgbm_mod):
    """Same prefix-sample pattern as Stage 6's load_val_tuning_sample, but
    for TRAINING data -- reads chunked+downcast, stops once max_rows is
    reached. Used only for ablation 2b, where reading the entire
    unresampled train set is not feasible; taking a prefix preserves the
    natural imbalance ratio (no class filtering), which is the point of
    this ablation."""
    parquet_file = lgbm_mod.pq.ParquetFile(path)
    feature_cols = None
    chunks = []
    rows_so_far = 0
    for batch in parquet_file.iter_batches(batch_size=lgbm_mod.ROW_CHUNK_SIZE):
        df = batch.to_pandas()
        if feature_cols is None:
            feature_cols = lgbm_mod.get_feature_cols(df, target)
        else:
            df = lgbm_mod.coerce_legacy_format(df)
        df = lgbm_mod.downcast_floats(df, feature_cols)
        chunks.append(df[feature_cols + [target]])
        rows_so_far += len(df)
        if rows_so_far >= max_rows:
            break
    sample = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"[ABLATION 2b] Using a {len(sample):,}-row PREFIX SAMPLE of the unresampled training "
          f"set (not the full ~hundreds-of-millions-row file) -- a resource-driven simplification "
          f"that preserves the natural class imbalance ratio since no class filtering is applied.")
    return sample, feature_cols


def train_and_stream_evaluate(train_df, feature_cols, target: str, val_path: Path, test_path: Path,
                               cost_matrix: dict, n_trials: int, lgbm_mod) -> dict:
    """Trains a fresh LightGBM model on the given (already in-memory,
    already small enough) training data, then evaluates it by STREAMING
    through the full val/test parquet files chunk-by-chunk via Stage 6's
    stream_predict_booster -- never loading either as one full DataFrame,
    regardless of how many rows they contain."""
    X_train, y_train = train_df[feature_cols], train_df[target]

    print(f"[ABLATION] Sampling validation for tuning (prefix sample, {len(feature_cols)} features)...")
    X_val_sample, y_val_sample = lgbm_mod.load_val_tuning_sample(
        val_path, feature_cols, target, lgbm_mod.VAL_TUNING_SAMPLE_ROWS
    )
    best_params = lgbm_mod.tune_with_optuna(X_train, y_train, X_val_sample, y_val_sample, n_trials=n_trials)
    del X_val_sample, y_val_sample
    gc.collect()

    final_params = {**best_params, "objective": "binary", "metric": "average_precision",
                     "verbosity": -1, "random_state": 42}
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train, y_train)
    del X_train, y_train
    gc.collect()

    print("[ABLATION] Streaming full validation set for threshold selection...")
    y_val_full, val_probs_full, _, _ = lgbm_mod.stream_predict_booster(model.booster_, val_path, feature_cols, target)
    threshold = lgbm_mod.select_cost_optimal_threshold(
        y_val_full, val_probs_full, cost_matrix["false_negative_cost"], cost_matrix["false_positive_cost"]
    )
    del y_val_full, val_probs_full
    gc.collect()

    print("[ABLATION] Streaming full test set for evaluation...")
    y_test_full, test_probs_full, _, _ = lgbm_mod.stream_predict_booster(model.booster_, test_path, feature_cols, target)
    metrics = lgbm_mod.evaluate(y_test_full, test_probs_full, threshold)
    del model, y_test_full, test_probs_full
    gc.collect()
    return metrics


def lstm_rul_to_classification_metrics(rul_preds: np.ndarray, y_true: np.ndarray,
                                        horizon_days: float, lgbm_mod) -> dict:
    """Converts the LSTM's RUL predictions into the same 14-day binary
    decision LightGBM makes (predicted RUL < horizon -> predicted failure),
    scored with LightGBM's own evaluate() for identical metric logic."""
    predicted_positive = (rul_preds < horizon_days).astype(int)
    return lgbm_mod.evaluate(y_true, predicted_positive.astype(float), threshold=0.5)


def build_comparison_table(lgbm_metrics_path: str, hybrid_metrics_path: str,
                            lstm_class_metrics: dict) -> pd.DataFrame:
    lgbm = pd.read_csv(lgbm_metrics_path).iloc[0]
    hybrid = pd.read_csv(hybrid_metrics_path).iloc[0]
    rows = [
        {"model": "LightGBM only", "roc_auc": lgbm["roc_auc"], "auprc": lgbm["auprc"],
         "precision": lgbm.get("precision"), "recall": lgbm.get("recall"), "f1": lgbm.get("f1")},
        {"model": "LSTM (RUL < horizon -> failure)", "roc_auc": float("nan"), "auprc": float("nan"),
         "precision": lstm_class_metrics["precision"], "recall": lstm_class_metrics["recall"],
         "f1": lstm_class_metrics["f1"]},
        {"model": "Hybrid (LightGBM+LSTM)", "roc_auc": hybrid["test_roc_auc"], "auprc": hybrid["test_auprc"],
         "precision": None, "recall": None, "f1": None},
    ]
    return pd.DataFrame(rows)


def generalizability_report(test_parquet_path: str, lgbm_test_probs_path: str, threshold: float,
                             target: str, lgbm_mod) -> pd.DataFrame:
    """Loops over each distinct drive model ONE AT A TIME, pulling only
    that model's (lgbm_prob, target) pairs via a DuckDB query joined
    against Stage 6's already-computed probabilities CSV -- never touches
    the 217-column feature parquet's full row set, and never holds more
    than one drive model's rows in memory at once.

    Per-model metrics are reported with their positive count because several
    drive models have only a handful of failures in the test split.  A
    ROC-AUC estimated from <30 positives has a confidence interval spanning
    most of the unit interval, so those rows are flagged rather than trusted.
    Fleet-level metrics remain the primary result."""
    con = duckdb.connect()
    models = con.execute(
        f"SELECT DISTINCT model FROM read_parquet('{test_parquet_path}') WHERE model IS NOT NULL"
    ).fetchnumpy()["model"]
    print(f"[GENERALIZABILITY] {len(models)} distinct drive models found")

    rows = []
    for model_name in models:
        safe_name = str(model_name).replace("'", "''")
        query = f"""
            SELECT l.lgbm_prob AS lgbm_prob, l."{target}" AS target
            FROM read_csv_auto('{lgbm_test_probs_path}') AS l
            INNER JOIN (
                SELECT DISTINCT serial_number
                FROM read_parquet('{test_parquet_path}')
                WHERE model = '{safe_name}'
            ) AS p
            ON l.serial_number = p.serial_number
        """
        result = con.execute(query).fetchnumpy()
        y_true = result["target"].astype(np.int64)
        y_probs = result["lgbm_prob"]
        n_pos = int(y_true.sum())

        # A per-model ROC-AUC is only meaningful with enough positives to
        # estimate a ranking.  Emitting a bare NaN (previous behaviour) left 34
        # of 100 rows blank, which reads as sloppiness rather than as an
        # honest statement about statistical power.  Report the positive count
        # and an explicit status instead, and mark the low-power rows so they
        # are never quoted as if they were reliable.
        reliable = n_pos >= MIN_POSITIVES_FOR_PER_MODEL_METRIC
        if n_pos == 0:
            metrics = {"roc_auc": float("nan"), "auprc": float("nan"),
                       "f1": float("nan"), "precision": float("nan"), "recall": float("nan"),
                       "tp": 0, "tn": 0, "fp": 0, "fn": 0, "threshold": threshold}
            status = "no_positives"
        elif not reliable:
            # Still compute the point estimate -- it is informative -- but flag
            # it.  Several models land well below 0.5 purely because of this.
            metrics = lgbm_mod.evaluate(y_true, y_probs, threshold)
            status = f"low_power(<{MIN_POSITIVES_FOR_PER_MODEL_METRIC}_positives)"
        else:
            metrics = lgbm_mod.evaluate(y_true, y_probs, threshold)
            status = "ok"
        rows.append({"drive_model": model_name, "n_rows": len(y_true),
                     "n_positives": n_pos, "metric_status": status,
                     "below_chance_auroc": bool(reliable and metrics.get("roc_auc", float("nan")) < 0.5),
                     **metrics})
        del result, y_true, y_probs
    con.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparison, ablation, and generalizability evaluation")
    parser.add_argument("--train_resampled", type=str, required=True, help="Stage 5 output (imbalance-handled)")
    parser.add_argument("--train_full", type=str, required=True, help="Stage 3/4 output, no resampling")
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--cost_matrix", type=str, required=True)
    parser.add_argument("--lgbm_metrics", type=str, required=True)
    parser.add_argument("--lgbm_test_probs", type=str, required=True, help="Stage 6's results/lgbm_test_probs.csv")
    parser.add_argument("--hybrid_metrics", type=str, required=True)
    parser.add_argument("--hybrid_test_scores", type=str, required=True, help="Stage 8's results/hybrid_test_scores.csv")
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--horizon_days", type=float, default=14.0)
    parser.add_argument("--skip_phases", type=str, default="",
                        help="Comma-separated phases to skip when their output CSV already "
                             "exists: '2a', '2b', '3'. Use after an interrupted run to avoid "
                             "redoing multi-hour retrains that already completed.")
    parser.add_argument("--ablation_n_trials", type=int, default=10)
    parser.add_argument("--ablation_2b_sample_rows", type=int, default=5_000_000,
                         help="Prefix-sample size from the unresampled train set for ablation 2b "
                              "(the full file is too large to load on a laptop)")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lgbm_mod = load_module("lgbm_stage6", Path(__file__).parent / "06_train_lightgbm.py")
    with open(args.cost_matrix) as f:
        cost_matrix = json.load(f)

    print("=== 1. Comparison table (LightGBM vs LSTM-as-classifier vs Hybrid) ===")
    hybrid_scores = pd.read_csv(args.hybrid_test_scores, usecols=["lstm_rul_pred", "target"])
    lstm_class_metrics = lstm_rul_to_classification_metrics(
        hybrid_scores["lstm_rul_pred"].to_numpy(), hybrid_scores["target"].to_numpy(),
        args.horizon_days, lgbm_mod
    )
    del hybrid_scores
    gc.collect()
    comparison = build_comparison_table(args.lgbm_metrics, args.hybrid_metrics, lstm_class_metrics)
    print(comparison.to_string(index=False))
    comparison.to_csv(out_dir / "full_comparison.csv", index=False)

    print("\n=== 2a. Ablation: with vs without rolling/delta features ===")
    with_rolling = pd.read_csv(args.lgbm_metrics).iloc[0].to_dict()

    # --skip_phases lets a resumed run redo only the phases whose outputs are
    # stale. Phases 2a/2b each retrain LightGBM and stream the full 145M-row
    # test set (tens of minutes apiece), so re-running everything after a crash
    # or a machine hiccup is pure waste when 2a already succeeded.
    skip = {p.strip().lower() for p in str(args.skip_phases).split(",") if p.strip()}
    if "2a" in skip and (out_dir / "ablation_features.csv").exists():
        print("[SKIP] phase 2a requested and ablation_features.csv exists -- skipping")
        filtered_cols = None
        train_resampled_df = None
    else:
        train_resampled_df, all_feature_cols = lgbm_mod.load_train_chunked(Path(args.train_resampled), args.target)
    if train_resampled_df is not None:
        filtered_cols = [c for c in all_feature_cols if "_roll" not in c and "_delta" not in c]
        print(f"[ABLATION] Using {len(filtered_cols)}/{len(all_feature_cols)} feature columns")
        without_rolling = train_and_stream_evaluate(
            train_resampled_df, filtered_cols, args.target, Path(args.val), Path(args.test),
            cost_matrix, args.ablation_n_trials, lgbm_mod
        )
        feature_ablation_df = pd.DataFrame([
            {"variant": "full (with rolling+delta)", **{k: with_rolling[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
            {"variant": "without rolling+delta", **{k: without_rolling[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
        ])
        print(feature_ablation_df.to_string(index=False))
        feature_ablation_df.to_csv(out_dir / "ablation_features.csv", index=False)
        del train_resampled_df
        gc.collect()

    print("\n=== 2b. Ablation: with vs without Stage 5 imbalance handling ===")
    if "2b" in skip and (out_dir / "ablation_imbalance.csv").exists():
        print("[SKIP] phase 2b requested and ablation_imbalance.csv exists -- skipping")
    else:
        train_sample_df, full_feature_cols = load_train_sample_chunked(
            Path(args.train_full), args.target, args.ablation_2b_sample_rows, lgbm_mod
        )
        without_imbalance = train_and_stream_evaluate(
            train_sample_df, full_feature_cols, args.target, Path(args.val), Path(args.test),
            cost_matrix, args.ablation_n_trials, lgbm_mod
        )
        del train_sample_df
        gc.collect()
        imbalance_ablation_df = pd.DataFrame([
            {"variant": "with imbalance handling (resampled)", **{k: with_rolling[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
            {"variant": "without imbalance handling (unresampled prefix sample)",
             **{k: without_imbalance[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
        ])
        print(imbalance_ablation_df.to_string(index=False))
        imbalance_ablation_df.to_csv(out_dir / "ablation_imbalance.csv", index=False)

    print("\n=== 3. Generalizability by drive model ===")
    if "3" in skip and (out_dir / "generalizability.csv").exists():
        print("[SKIP] phase 3 requested and generalizability.csv exists -- skipping")
    else:
        gen_report = generalizability_report(
            args.test, args.lgbm_test_probs, with_rolling["threshold"], args.target, lgbm_mod
        )
        print(gen_report.to_string(index=False))
        gen_report.to_csv(out_dir / "generalizability.csv", index=False)

    print(f"\n[SAVE] {out_dir/'full_comparison.csv'}")
    print(f"[SAVE] {out_dir/'ablation_features.csv'}")
    print(f"[SAVE] {out_dir/'ablation_imbalance.csv'}")
    print(f"[SAVE] {out_dir/'generalizability.csv'}")


if __name__ == "__main__":
    main()
