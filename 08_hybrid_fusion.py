"""
Stage 8 — Hybrid Fusion (LightGBM + LSTM)

Combines the LightGBM failure-probability output (Stage 6) and the LSTM
RUL estimate (Stage 7's full, unfiltered predictions) into a single
blended risk score per drive. Tries two combination strategies and keeps
whichever validates better, as planned:

  1. weighted  — a simple fixed-weight linear blend of the two normalized
     signals (interpretable, no extra training).
  2. meta      — a small logistic regression trained on [lgbm_prob,
     lstm_rul_pred] -> failure_within_{horizon}d, letting the data decide
     the combination weights.

Usage:
    python3 08_hybrid_fusion.py \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --lgbm_model models/lgbm_model.txt \
        --lgbm_threshold models/lgbm_threshold.json \
        --lstm_model models/lstm_model.pt \
        --lstm_config models/lstm_config.json \
        --target failure_within_14d \
        --out_dir models results
"""
import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

lstm_module = import_module("07_train_lstm".replace("-", "_")) if False else None


def load_lstm_module():
    """Loads 07_train_lstm.py as a module so we can reuse its architecture
    class and helper functions instead of duplicating them here."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lstm_stage7", Path(__file__).parent / "07_train_lstm.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_lstm_predictions(mod, df: pd.DataFrame, model_path: str, config_path: str, device: str) -> np.ndarray:
    with open(config_path) as f:
        config = json.load(f)
    feature_cols = config["feature_cols"]
    feat_mean = pd.Series(config["feat_mean"])
    feat_std = pd.Series(config["feat_std"])

    model = mod.AttentionLSTM(n_features=len(feature_cols), hidden_size=config["hidden_size"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    df_norm = df.copy()
    df_norm[feature_cols] = (df_norm[feature_cols] - feat_mean) / feat_std
    df_norm[feature_cols] = df_norm[feature_cols].fillna(0.0)

    X, _ = mod.build_sequences(df_norm.assign(rul_days=0.0), feature_cols, config["lookback"])
    loader = torch.utils.data.DataLoader(
        mod.RULDataset(X, np.zeros(len(X), dtype=np.float32)), batch_size=128, shuffle=False,
    )
    _, preds = mod.evaluate_rmse_mae(model, loader, device)
    return preds


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]. Kept for reference/backward compatibility but NOT
    used for the RUL signal anymore — see horizon_bounded_risk() below."""
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def horizon_bounded_risk(rul: np.ndarray, horizon_days: float) -> np.ndarray:
    """Converts RUL into a [0, 1] urgency score tied to the actual
    operational decision horizon, instead of a dataset-dependent min/max.
    This deliberately avoids the earlier bug (fitting min/max separately
    on val and test, producing two different scales) by using a fixed
    formula that needs no fitting at all: RUL >= horizon -> 0 urgency,
    RUL <= 0 -> 1 (maximum urgency), linear in between. A 500-day RUL and
    a 50-day RUL are both just 'not urgent' (0), rather than a huge raw
    RUL value dominating the scale the way plain min-max scaling would let
    it."""
    return np.clip((horizon_days - rul) / horizon_days, 0.0, 1.0)


def weighted_blend(lgbm_prob: np.ndarray, lstm_rul: np.ndarray, horizon_days: float,
                    weight_lgbm: float = 0.6) -> np.ndarray:
    """lgbm_prob is already a valid [0,1] probability — no rescaling needed
    or wanted. lstm_rul is converted via the fixed, horizon-bounded
    transform above, which needs no per-split fitting, sidestepping the
    val/test normalization mismatch entirely rather than just patching it."""
    risk_from_lgbm = lgbm_prob
    risk_from_lstm = horizon_bounded_risk(lstm_rul, horizon_days)
    return weight_lgbm * risk_from_lgbm + (1 - weight_lgbm) * risk_from_lstm


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse LightGBM and LSTM outputs into a hybrid risk score")
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--lgbm_model", type=str, required=True)
    parser.add_argument("--lgbm_threshold", type=str, required=True)
    parser.add_argument("--lstm_model", type=str, required=True)
    parser.add_argument("--lstm_config", type=str, required=True)
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--horizon_days", type=float, default=14.0)
    parser.add_argument("--out_dir", nargs=2, default=["models", "results"], metavar=("MODELS_DIR", "RESULTS_DIR"))
    args = parser.parse_args()

    models_dir, results_dir = Path(args.out_dir[0]), Path(args.out_dir[1])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lstm_mod = load_lstm_module()

    with open(args.lgbm_threshold) as f:
        lgbm_meta = json.load(f)
    feature_cols = lgbm_meta["feature_cols"]

    booster = lgb.Booster(model_file=args.lgbm_model)

    val_df = pd.read_parquet(args.val)
    test_df = pd.read_parquet(args.test)

    val_lgbm_prob = booster.predict(val_df[feature_cols])
    test_lgbm_prob = booster.predict(test_df[feature_cols])

    print("[LSTM] running inference for fusion inputs...")
    val_lstm_rul = get_lstm_predictions(lstm_mod, val_df, args.lstm_model, args.lstm_config, device)
    test_lstm_rul = get_lstm_predictions(lstm_mod, test_df, args.lstm_model, args.lstm_config, device)

    y_val = val_df[args.target].values
    y_test = test_df[args.target].values

    # Strategy 1: weighted blend (no fitting, so no leakage question at all)
    val_weighted = weighted_blend(val_lgbm_prob, val_lstm_rul, args.horizon_days)
    test_weighted = weighted_blend(test_lgbm_prob, test_lstm_rul, args.horizon_days)

    # Strategy 2: meta-learner. To select between strategies fairly, we
    # need a GENUINE held-out estimate of the meta-learner's performance —
    # training and evaluating it on the exact same validation rows (the
    # earlier version's bug) lets it look artificially good purely by
    # memorizing validation. Fix: split validation itself into a fit half
    # (trains the meta-learner) and a select half (the ONLY rows whose
    # AUPRC is used for the weighted-vs-meta decision). The final model
    # used on test is then refit on the FULL validation set — standard
    # practice: use a held-out split to pick a strategy, then refit on all
    # available data once picked.
    meta_X_val_full = np.column_stack([val_lgbm_prob, val_lstm_rul])
    meta_model = None
    val_meta_select_auprc = -1.0
    val_meta = np.zeros_like(val_weighted)
    test_meta = np.zeros_like(test_weighted)

    can_split = y_val.sum() >= 4 and len(np.unique(y_val)) > 1  # need enough positives in BOTH halves
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

        # Refit on the FULL validation set for the model actually used on test
        meta_model = LogisticRegression(class_weight="balanced")
        meta_model.fit(meta_X_val_full, y_val)
        val_meta = meta_model.predict_proba(meta_X_val_full)[:, 1]
        test_meta = meta_model.predict_proba(np.column_stack([test_lgbm_prob, test_lstm_rul]))[:, 1]
    else:
        print("WARNING: validation set doesn't have enough positives to hold out a genuine "
              "fit/select split for the meta-learner — skipping the meta strategy entirely "
              "rather than risk a leaked, artificially inflated selection. Using weighted blend.")

    def safe_auprc(y, p):
        return average_precision_score(y, p) if y.sum() > 0 else float("nan")

    def safe_auc(y, p):
        return roc_auc_score(y, p) if y.sum() > 0 else float("nan")

    weighted_val_auprc = safe_auprc(y_val, val_weighted)

    chosen_strategy = "meta" if (meta_model is not None and val_meta_select_auprc > weighted_val_auprc) else "weighted"
    test_hybrid_score = test_meta if chosen_strategy == "meta" else test_weighted
    val_hybrid_score = val_meta if chosen_strategy == "meta" else val_weighted

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

    if meta_model is not None:
        joblib.dump(meta_model, models_dir / "hybrid_meta_model.pkl")
    with open(models_dir / "hybrid_config.json", "w") as f:
        json.dump({"chosen_strategy": chosen_strategy, "weight_lgbm": 0.6, "horizon_days": args.horizon_days}, f, indent=2)

    pd.DataFrame({
        "lgbm_prob": test_lgbm_prob, "lstm_rul_pred": test_lstm_rul,
        "hybrid_weighted": test_weighted, "hybrid_meta": test_meta,
        "hybrid_score": test_hybrid_score, "target": y_test,
    }).to_csv(results_dir / "hybrid_test_scores.csv", index=False)
    pd.DataFrame([metrics]).to_csv(results_dir / "hybrid_metrics.csv", index=False)

    print(f"[SAVE] {results_dir/'hybrid_test_scores.csv'}")
    print(f"[SAVE] {results_dir/'hybrid_metrics.csv'}")
    print(f"[SAVE] {models_dir/'hybrid_config.json'}")


if __name__ == "__main__":
    main()
