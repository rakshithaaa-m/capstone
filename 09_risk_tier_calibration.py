"""
Stage 9 — Risk-Tier Calibration

Calibrates Low/Medium/High tier boundaries against ACTUAL detection rate
on the validation set, instead of fixed/arbitrary probability cutoffs
(the root cause of the original project's tiering bug, where 84% of real
failures ended up in "Low"). Boundaries are chosen so the High tier
captures at least `--high_target_recall` of validation failures, then the
same cutoffs are applied to test and reported honestly.

Usage:
    python3 09_risk_tier_calibration.py \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --lgbm_model models/lgbm_model.txt \
        --lgbm_threshold models/lgbm_threshold.json \
        --lstm_model models/lstm_model.pt \
        --lstm_config models/lstm_config.json \
        --hybrid_config models/hybrid_config.json \
        --target failure_within_14d \
        --high_target_recall 0.5 \
        --medium_target_recall 0.8 \
        --out_dir results
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

import importlib.util


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_hybrid_score(df: pd.DataFrame, lgbm_prob: np.ndarray, lstm_rul: np.ndarray,
                          hybrid_config: dict, meta_model=None) -> np.ndarray:
    if hybrid_config["chosen_strategy"] == "meta" and meta_model is not None:
        X = np.column_stack([lgbm_prob, lstm_rul])
        return meta_model.predict_proba(X)[:, 1]
    fusion_mod = load_module("fusion_helper", Path(__file__).parent / "08_hybrid_fusion.py")
    return fusion_mod.weighted_blend(
        lgbm_prob, lstm_rul,
        horizon_days=hybrid_config.get("horizon_days", 14.0),
        weight_lgbm=hybrid_config.get("weight_lgbm", 0.6),
    )


def find_tier_cutoff(scores: np.ndarray, y_true: np.ndarray, target_recall: float,
                      strict: bool = True) -> float:
    """Finds the LOWEST score cutoff such that all drives scoring at or
    above it capture at least `target_recall` of the true positives. Lower
    cutoff = larger tier = easier to hit the recall target, so we search
    from high scores downward and stop at the first cutoff that qualifies.
    If validation has zero positives, there's no real basis to calibrate
    against — with strict=True (default) this raises rather than silently
    falling back to an arbitrary percentile, matching Stage 6's behavior.
    Only pass strict=False for quick smoke-testing on tiny/synthetic data."""
    if y_true.sum() == 0:
        if strict:
            raise ValueError(
                "Validation set has zero positive examples — there is no genuine basis to "
                "calibrate a tier cutoff against. Push --train_end/--val_end so validation "
                "actually contains failures, or pass a non-default (non-strict) mode explicitly "
                "if this is intentional smoke-testing on tiny data."
            )
        return float(np.percentile(scores, 90))

    order = np.argsort(-scores)  # highest score first
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    cum_positives = np.cumsum(sorted_labels)
    total_positives = y_true.sum()
    recall_at_each_cutoff = cum_positives / total_positives

    qualifying = np.where(recall_at_each_cutoff >= target_recall)[0]
    if len(qualifying) == 0:
        return float(np.min(scores))  # even including everyone doesn't hit the target
    cutoff_idx = qualifying[0]
    return float(sorted_scores[cutoff_idx])


def assign_tiers(scores: np.ndarray, high_cutoff: float, medium_cutoff: float) -> np.ndarray:
    tiers = np.full(len(scores), "LOW", dtype=object)
    tiers[scores >= medium_cutoff] = "MEDIUM"
    tiers[scores >= high_cutoff] = "HIGH"
    return tiers


def tier_report(df: pd.DataFrame, tiers: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Reports BOTH the per-tier-alone breakdown AND the cumulative
    detection rate, since the MEDIUM cutoff was calibrated against a
    CUMULATIVE HIGH+MEDIUM target, not MEDIUM in isolation — reporting
    only the per-tier share without the cumulative column would make it
    look like MEDIUM alone hits the target_recall you calibrated for,
    which it doesn't and was never meant to."""
    out = pd.DataFrame({"tier": tiers, "target": y_true})
    total_positives = y_true.sum()

    n_high = int(out.loc[out["tier"] == "HIGH", "target"].sum())
    n_medium = int(out.loc[out["tier"] == "MEDIUM", "target"].sum())
    n_low = int(out.loc[out["tier"] == "LOW", "target"].sum())

    def rate(n):
        return n / total_positives if total_positives > 0 else float("nan")

    rows = [
        {"tier": "HIGH", "total_drives": int((tiers == "HIGH").sum()),
         "true_failures_in_tier_alone": n_high, "share_of_failures_in_tier_alone": rate(n_high),
         "cumulative_failures_captured": n_high, "cumulative_share_captured": rate(n_high)},
        {"tier": "MEDIUM", "total_drives": int((tiers == "MEDIUM").sum()),
         "true_failures_in_tier_alone": n_medium, "share_of_failures_in_tier_alone": rate(n_medium),
         "cumulative_failures_captured": n_high + n_medium, "cumulative_share_captured": rate(n_high + n_medium)},
        {"tier": "LOW", "total_drives": int((tiers == "LOW").sum()),
         "true_failures_in_tier_alone": n_low, "share_of_failures_in_tier_alone": rate(n_low),
         "cumulative_failures_captured": n_high + n_medium + n_low,
         "cumulative_share_captured": rate(n_high + n_medium + n_low)},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate risk tiers against real detection rate")
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--lgbm_model", type=str, required=True)
    parser.add_argument("--lgbm_threshold", type=str, required=True)
    parser.add_argument("--lstm_model", type=str, required=True)
    parser.add_argument("--lstm_config", type=str, required=True)
    parser.add_argument("--hybrid_config", type=str, required=True)
    parser.add_argument("--hybrid_meta_model", type=str, default=None)
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--high_target_recall", type=float, default=0.5,
                         help="HIGH tier must capture at least this fraction of real failures")
    parser.add_argument("--medium_target_recall", type=float, default=0.8,
                         help="HIGH+MEDIUM combined must capture at least this fraction")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--allow_fallback", action="store_true",
                         help="Allow falling back to an arbitrary percentile cutoff if validation "
                              "has zero positives, instead of raising. Only use for smoke-testing "
                              "on tiny/synthetic data — never for a real experiment.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lstm_mod = load_module("lstm_stage7", Path(__file__).parent / "07_train_lstm.py")
    fusion_mod = load_module("fusion_stage8", Path(__file__).parent / "08_hybrid_fusion.py")

    with open(args.lgbm_threshold) as f:
        lgbm_meta = json.load(f)
    feature_cols = lgbm_meta["feature_cols"]
    booster = lgb.Booster(model_file=args.lgbm_model)

    with open(args.hybrid_config) as f:
        hybrid_config = json.load(f)
    meta_model = None
    if args.hybrid_meta_model and Path(args.hybrid_meta_model).exists():
        import joblib
        meta_model = joblib.load(args.hybrid_meta_model)

    val_df = pd.read_parquet(args.val)
    test_df = pd.read_parquet(args.test)

    val_lgbm_prob = booster.predict(val_df[feature_cols])
    test_lgbm_prob = booster.predict(test_df[feature_cols])
    val_lstm_rul = fusion_mod.get_lstm_predictions(lstm_mod, val_df, args.lstm_model, args.lstm_config, device)
    test_lstm_rul = fusion_mod.get_lstm_predictions(lstm_mod, test_df, args.lstm_model, args.lstm_config, device)

    val_hybrid = compute_hybrid_score(val_df, val_lgbm_prob, val_lstm_rul, hybrid_config, meta_model)
    test_hybrid = compute_hybrid_score(test_df, test_lgbm_prob, test_lstm_rul, hybrid_config, meta_model)

    y_val = val_df[args.target].values
    y_test = test_df[args.target].values

    high_cutoff = find_tier_cutoff(val_hybrid, y_val, args.high_target_recall, strict=not args.allow_fallback)
    medium_cutoff = find_tier_cutoff(val_hybrid, y_val, args.medium_target_recall, strict=not args.allow_fallback)
    medium_cutoff = min(medium_cutoff, high_cutoff)  # MEDIUM cutoff can't be above HIGH's

    print(f"[CALIBRATED] HIGH cutoff = {high_cutoff:.4f} (target recall {args.high_target_recall:.0%})")
    print(f"[CALIBRATED] MEDIUM cutoff = {medium_cutoff:.4f} (target recall {args.medium_target_recall:.0%})")

    val_tiers = assign_tiers(val_hybrid, high_cutoff, medium_cutoff)
    test_tiers = assign_tiers(test_hybrid, high_cutoff, medium_cutoff)

    val_report = tier_report(val_df, val_tiers, y_val)
    test_report = tier_report(test_df, test_tiers, y_test)

    print("\n[VALIDATION tier report]")
    print(val_report.to_string(index=False))
    print("\n[TEST tier report — the honest, held-out check]")
    print(test_report.to_string(index=False))

    test_report.to_csv(out_dir / "tier_accuracy_test.csv", index=False)
    val_report.to_csv(out_dir / "tier_accuracy_val.csv", index=False)
    pd.DataFrame({
        "serial_number": test_df["serial_number"], "date": test_df["date"],
        "hybrid_score": test_hybrid, "risk_tier": test_tiers, "target": y_test,
    }).to_csv(out_dir / "risk_tiers_test.csv", index=False)
    with open(out_dir / "tier_cutoffs.json", "w") as f:
        json.dump({"high_cutoff": high_cutoff, "medium_cutoff": medium_cutoff}, f, indent=2)

    print(f"\n[SAVE] {out_dir/'tier_accuracy_test.csv'}")
    print(f"[SAVE] {out_dir/'risk_tiers_test.csv'}")
    print(f"[SAVE] {out_dir/'tier_cutoffs.json'}")


if __name__ == "__main__":
    main()
