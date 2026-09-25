"""
Stage 8 — Hybrid Fusion (LightGBM + LSTM), memory-safe version

CHANGE FROM THE VERSION BEFORE THIS ONE: joining the two CSVs in DuckDB
fixed the join itself, but .fetchdf() still pulled the ENTIRE joined
result -- including the serial_number/date STRING columns, for up to
~145M rows -- into a pandas DataFrame afterward. Materializing that many
Python string objects is the same memory wall as before, just moved one
step later (it crashed inside fetchdf(), not inside the join).

The fix here is to never bring serial_number/date into Python for the
full-size files at all:
    1. Fetch ONLY the small numeric columns (lgbm_prob, lstm_rul_pred,
       target) as numpy arrays -- three float/int columns for ~145M rows
       is ~1-2GB, trivial to hold in memory, and is all that's needed for
       every score computation and metric below.
    2. For the final output CSVs (which DO need serial_number/date, for
       traceability and for Stage 9/12 downstream), have DuckDB compute
       the hybrid score as a SQL expression and write the result
       DIRECTLY TO DISK via `COPY ... TO`, entirely inside DuckDB's own
       columnar engine. Python never holds the joined string columns for
       the full dataset at any point.

Usage:
    python3 08_hybrid_fusion.py \
        --lgbm_val results/lgbm_val_probs.csv \
        --lgbm_test results/lgbm_test_probs.csv \
        --lstm_val results/lstm_full_val_preds.csv \
        --lstm_test results/lstm_full_test_preds.csv \
        --target failure_within_14d \
        --out_dir models results
"""
import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def fetch_numeric_columns(lgbm_path: str, lstm_path: str, target: str):
    """Joins the two CSVs in DuckDB and fetches ONLY the numeric columns
    needed for score computation -- never selects serial_number/date, so
    there's no string-object memory cost regardless of row count."""
    con = duckdb.connect()
    query = f"""
        SELECT l.lgbm_prob AS lgbm_prob, r.lstm_rul_pred AS lstm_rul_pred, l."{target}" AS target
        FROM read_csv_auto('{lgbm_path}') AS l
        INNER JOIN read_csv_auto('{lstm_path}') AS r
            ON l.serial_number = r.serial_number AND l.date = r.date
    """
    result = con.execute(query).fetchnumpy()
    con.close()
    n = len(result["lgbm_prob"])
    print(f"[JOIN] {lgbm_path} <-> {lstm_path}: {n:,} matched rows (numeric columns only)")
    return result["lgbm_prob"], result["lstm_rul_pred"], result["target"].astype(np.int64)


def export_hybrid_csv(lgbm_path: str, lstm_path: str, target: str, out_path: Path,
                       weight_lgbm: float, horizon_days: float,
                       strategy: str, meta_coef, meta_intercept) -> None:
    """Writes serial_number, date, both candidate scores, the chosen
    hybrid_score, and target DIRECTLY to a CSV on disk via DuckDB's SQL
    engine -- no pandas DataFrame of the full joined result is ever
    created. The weighted-blend formula and the meta-learner's fitted
    logistic regression (just an intercept + 2 coefficients + sigmoid)
    are both simple enough to express directly in SQL."""
    weighted_expr = (
        f"({weight_lgbm}*l.lgbm_prob + (1-{weight_lgbm})*"
        f"GREATEST(0.0, LEAST(1.0, ({horizon_days}-r.lstm_rul_pred)/{horizon_days})))"
    )
    if strategy == "meta" and meta_coef is not None:
        meta_expr = (
            f"(1.0 / (1.0 + EXP(-({meta_intercept} + {meta_coef[0]}*l.lgbm_prob "
            f"+ {meta_coef[1]}*r.lstm_rul_pred))))"
        )
        chosen_expr = meta_expr
    else:
        meta_expr = "NULL"
        chosen_expr = weighted_expr

    con = duckdb.connect()
    query = f"""
        COPY (
            SELECT
                l.serial_number AS serial_number,
                l.date AS date,
                l.lgbm_prob AS lgbm_prob,
                r.lstm_rul_pred AS lstm_rul_pred,
                {weighted_expr} AS hybrid_weighted,
                {meta_expr} AS hybrid_meta,
                {chosen_expr} AS hybrid_score,
                l."{target}" AS target
            FROM read_csv_auto('{lgbm_path}') AS l
            INNER JOIN read_csv_auto('{lstm_path}') AS r
                ON l.serial_number = r.serial_number AND l.date = r.date
        ) TO '{out_path.as_posix()}' (HEADER, DELIMITER ',')
    """
    con.execute(query)
    con.close()


def horizon_bounded_risk(rul: np.ndarray, horizon_days: float) -> np.ndarray:
    """Converts RUL into a [0, 1] urgency score tied to the actual
    operational decision horizon, instead of a dataset-dependent min/max.
    RUL >= horizon -> 0 urgency, RUL <= 0 -> 1 (max urgency), linear
    between. Needs no fitting, so val and test share the exact same scale."""
    return np.clip((horizon_days - rul) / horizon_days, 0.0, 1.0)


def weighted_blend(lgbm_prob: np.ndarray, lstm_rul: np.ndarray, horizon_days: float,
                    weight_lgbm: float = 0.6) -> np.ndarray:
    risk_from_lgbm = lgbm_prob
    risk_from_lstm = horizon_bounded_risk(lstm_rul, horizon_days)
    return weight_lgbm * risk_from_lgbm + (1 - weight_lgbm) * risk_from_lstm


def safe_auprc(y, p):
    return average_precision_score(y, p) if y.sum() > 0 else float("nan")


def safe_auc(y, p):
    return roc_auc_score(y, p) if y.sum() > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse LightGBM and LSTM outputs into a hybrid risk score")
    parser.add_argument("--lgbm_val", type=str, required=True)
    parser.add_argument("--lgbm_test", type=str, required=True)
    parser.add_argument("--lstm_val", type=str, required=True)
    parser.add_argument("--lstm_test", type=str, required=True)
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--horizon_days", type=float, default=14.0)
    parser.add_argument("--weight_lgbm", type=float, default=0.6)
    parser.add_argument("--out_dir", nargs=2, default=["models", "results"], metavar=("MODELS_DIR", "RESULTS_DIR"))
    args = parser.parse_args()

    models_dir, results_dir = Path(args.out_dir[0]), Path(args.out_dir[1])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] Joining Stage 6 (lgbm) and Stage 7 (lstm) outputs on (serial_number, date)...")
    val_lgbm_prob, val_lstm_rul, y_val = fetch_numeric_columns(args.lgbm_val, args.lstm_val, args.target)
    test_lgbm_prob, test_lstm_rul, y_test = fetch_numeric_columns(args.lgbm_test, args.lstm_test, args.target)

    # Strategy 1: weighted blend (no fitting, so no leakage question at all)
    val_weighted = weighted_blend(val_lgbm_prob, val_lstm_rul, args.horizon_days, args.weight_lgbm)
    test_weighted = weighted_blend(test_lgbm_prob, test_lstm_rul, args.horizon_days, args.weight_lgbm)

    # Strategy 2: meta-learner, selected via a genuine held-out half of
    # validation (fit half trains it, select half is the ONLY data whose
    # AUPRC decides weighted-vs-meta), then refit on the full validation
    # set for the model actually used on test.
    meta_X_val_full = np.column_stack([val_lgbm_prob, val_lstm_rul])
    meta_model = None
    val_meta_select_auprc = -1.0
    val_meta = np.zeros_like(val_weighted)
    test_meta = np.zeros_like(test_weighted)

    can_split = y_val.sum() >= 4 and len(np.unique(y_val)) > 1
    if can_split:
        from sklearn.model_selection import train_test_split
        idx_fit, idx_select = train_test_split(
            np.arange(len(y_val)), test_size=0.5, stratify=y_val, random_state=42
        )
        fit_model = LogisticRegression(class_weight="balanced")
        fit_model.fit(meta_X_val_full[idx_fit], y_val[idx_fit])
        select_preds = fit_model.predict_proba(meta_X_val_full[idx_select])[:, 1]
        if y_val[idx_select].sum() > 0:
            val_meta_select_auprc = average_precision_score(y_val[idx_select], select_preds)

        meta_model = LogisticRegression(class_weight="balanced")
        meta_model.fit(meta_X_val_full, y_val)
        val_meta = meta_model.predict_proba(meta_X_val_full)[:, 1]
        test_meta = meta_model.predict_proba(np.column_stack([test_lgbm_prob, test_lstm_rul]))[:, 1]
    else:
        print("WARNING: validation set doesn't have enough positives to hold out a genuine "
              "fit/select split for the meta-learner -- skipping the meta strategy entirely "
              "rather than risk a leaked, artificially inflated selection. Using weighted blend.")

    weighted_val_auprc = safe_auprc(y_val, val_weighted)
    chosen_strategy = "meta" if (meta_model is not None and val_meta_select_auprc > weighted_val_auprc) else "weighted"
    test_hybrid_score = test_meta if chosen_strategy == "meta" else test_weighted

    print(f"[COMPARE] weighted-blend val AUPRC (full val):        {weighted_val_auprc:.4f}")
    print(f"[COMPARE] meta-learner val AUPRC (held-out half only): "
          f"{val_meta_select_auprc if meta_model is not None else 'N/A (skipped)'}")
    print(f"[CHOSEN] strategy = {chosen_strategy}")

    metrics = {
        "chosen_strategy": chosen_strategy,
        "test_auprc": safe_auprc(y_test, test_hybrid_score),
        "test_roc_auc": safe_auc(y_test, test_hybrid_score),
        "val_auprc_used_for_selection": weighted_val_auprc if chosen_strategy == "weighted" else val_meta_select_auprc,
    }
    print(f"[TEST] hybrid AUPRC={metrics['test_auprc']:.4f}  ROC-AUC={metrics['test_roc_auc']:.4f}")

    meta_coef = meta_intercept = None
    if meta_model is not None:
        joblib.dump(meta_model, models_dir / "hybrid_meta_model.pkl")
        meta_coef = meta_model.coef_[0].tolist()
        meta_intercept = float(meta_model.intercept_[0])
    with open(models_dir / "hybrid_config.json", "w") as f:
        json.dump({
            "chosen_strategy": chosen_strategy, "weight_lgbm": args.weight_lgbm,
            "horizon_days": args.horizon_days,
            "meta_coef": meta_coef, "meta_intercept": meta_intercept,
        }, f, indent=2)

    # Write the final CSVs directly via DuckDB (no pandas involved for the
    # full string-laden result) -- this is the step that used to crash.
    print("[WRITE] Exporting val scores directly to CSV via DuckDB...")
    export_hybrid_csv(args.lgbm_val, args.lstm_val, args.target, results_dir / "hybrid_val_scores.csv",
                       args.weight_lgbm, args.horizon_days, chosen_strategy, meta_coef, meta_intercept)
    print("[WRITE] Exporting test scores directly to CSV via DuckDB...")
    export_hybrid_csv(args.lgbm_test, args.lstm_test, args.target, results_dir / "hybrid_test_scores.csv",
                       args.weight_lgbm, args.horizon_days, chosen_strategy, meta_coef, meta_intercept)

    import pandas as pd
    pd.DataFrame([metrics]).to_csv(results_dir / "hybrid_metrics.csv", index=False)

    print(f"[SAVE] {results_dir/'hybrid_val_scores.csv'}")
    print(f"[SAVE] {results_dir/'hybrid_test_scores.csv'}")
    print(f"[SAVE] {results_dir/'hybrid_metrics.csv'}")
    print(f"[SAVE] {models_dir/'hybrid_config.json'}")


if __name__ == "__main__":
    main()
