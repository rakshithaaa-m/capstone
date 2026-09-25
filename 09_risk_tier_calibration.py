"""
Stage 9 — Risk-Tier Calibration, memory-safe version

CHANGE FROM THE PREVIOUS VERSION: this script no longer reloads the LGBM
and LSTM models or recomputes any predictions. The previous version
called pd.read_parquet() on the FULL val/test files and re-ran LSTM
inference on the entire fleet from scratch -- exactly the pattern that
caused OOM crashes earlier in Stages 6-8, reintroduced here. It also
loaded "07_train_lstm.py" (the old, pre-rewrite script) instead of
"07_train_lstm_v2.py" -- the same silent architecture-mismatch risk we
already fixed once in Stage 8.

Stage 8 already computes the full hybrid score for every drive-day in
val and test, streamed/joined safely via DuckDB, and now saves BOTH:
    - results/hybrid_val_scores.csv
    - results/hybrid_test_scores.csv
(each with serial_number, date, hybrid_score, target)

This script's only job is to calibrate tier cutoffs against val's
hybrid_score/target columns, then apply those cutoffs to test. No model
loading, no parquet reads, no re-inference, no GPU/CPU device needed.

Calibrates Low/Medium/High tier boundaries against ACTUAL detection rate
on the validation set, instead of fixed/arbitrary probability cutoffs.
Boundaries are chosen so the HIGH tier captures at least
`--high_target_recall` of validation failures, HIGH+MEDIUM combined
capture at least `--medium_target_recall`, then the same cutoffs are
applied to test and reported honestly.

Usage:
    python3 09_risk_tier_calibration.py \
        --val_scores results/hybrid_val_scores.csv \
        --test_scores results/hybrid_test_scores.csv \
        --high_target_recall 0.5 \
        --medium_target_recall 0.8 \
        --out_dir results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_tier_cutoff(scores: np.ndarray, y_true: np.ndarray, target_recall: float,
                      strict: bool = True) -> float:
    """Finds the LOWEST score cutoff such that all drives scoring at or
    above it capture at least `target_recall` of the true positives. Lower
    cutoff = larger tier = easier to hit the recall target, so we search
    from high scores downward and stop at the first cutoff that qualifies.
    If validation has zero positives, there's no real basis to calibrate
    against -- with strict=True (default) this raises rather than silently
    falling back to an arbitrary percentile."""
    if y_true.sum() == 0:
        if strict:
            raise ValueError(
                "Validation set has zero positive examples -- there is no genuine basis to "
                "calibrate a tier cutoff against. Check --val_scores points to the right file, "
                "or pass --allow_fallback explicitly if this is intentional smoke-testing on "
                "tiny/synthetic data."
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


def tier_report(tiers: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Reports BOTH the per-tier-alone breakdown AND the cumulative
    detection rate, since the MEDIUM cutoff was calibrated against a
    CUMULATIVE HIGH+MEDIUM target, not MEDIUM in isolation -- reporting
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
    parser.add_argument("--val_scores", type=str, required=True,
                         help="Stage 8's results/hybrid_val_scores.csv")
    parser.add_argument("--test_scores", type=str, required=True,
                         help="Stage 8's results/hybrid_test_scores.csv")
    parser.add_argument("--target", type=str, default="target",
                         help="Target column name in the hybrid scores CSVs (Stage 8 saves it as 'target')")
    parser.add_argument("--high_target_recall", type=float, default=0.5,
                         help="HIGH tier must capture at least this fraction of real failures")
    parser.add_argument("--medium_target_recall", type=float, default=0.8,
                         help="HIGH+MEDIUM combined must capture at least this fraction")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--allow_fallback", action="store_true",
                         help="Allow falling back to an arbitrary percentile cutoff if validation "
                              "has zero positives, instead of raising. Only use for smoke-testing "
                              "on tiny/synthetic data -- never for a real experiment.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {args.val_scores}")
    val_df = pd.read_csv(args.val_scores)
    print(f"[LOAD] {args.test_scores}")
    test_df = pd.read_csv(args.test_scores)

    val_hybrid = val_df["hybrid_score"].to_numpy()
    test_hybrid = test_df["hybrid_score"].to_numpy()
    y_val = val_df[args.target].to_numpy()
    y_test = test_df[args.target].to_numpy()

    high_cutoff = find_tier_cutoff(val_hybrid, y_val, args.high_target_recall, strict=not args.allow_fallback)
    medium_cutoff = find_tier_cutoff(val_hybrid, y_val, args.medium_target_recall, strict=not args.allow_fallback)
    medium_cutoff = min(medium_cutoff, high_cutoff)  # MEDIUM cutoff can't be above HIGH's

    print(f"[CALIBRATED] HIGH cutoff = {high_cutoff:.4f} (target recall {args.high_target_recall:.0%})")
    print(f"[CALIBRATED] MEDIUM cutoff = {medium_cutoff:.4f} (target recall {args.medium_target_recall:.0%})")

    val_tiers = assign_tiers(val_hybrid, high_cutoff, medium_cutoff)
    test_tiers = assign_tiers(test_hybrid, high_cutoff, medium_cutoff)

    val_report = tier_report(val_tiers, y_val)
    test_report = tier_report(test_tiers, y_test)

    print("\n[VALIDATION tier report]")
    print(val_report.to_string(index=False))
    print("\n[TEST tier report -- the honest, held-out check]")
    print(test_report.to_string(index=False))

    test_report.to_csv(out_dir / "tier_accuracy_test.csv", index=False)
    val_report.to_csv(out_dir / "tier_accuracy_val.csv", index=False)
    pd.DataFrame({
        "serial_number": test_df["serial_number"], "date": test_df["date"],
        "hybrid_score": test_hybrid, "risk_tier": test_tiers, "target": y_test,
    }).to_csv(out_dir / "risk_tiers_test.csv", index=False)
    with open(out_dir / "tier_cutoffs.json", "w") as f:
        json.dump({"high_cutoff": high_cutoff, "medium_cutoff": medium_cutoff}, f, indent=2)

    print(f"\n[SAVE] {out_dir/'tier_accuracy_val.csv'}")
    print(f"[SAVE] {out_dir/'tier_accuracy_test.csv'}")
    print(f"[SAVE] {out_dir/'risk_tiers_test.csv'}")
    print(f"[SAVE] {out_dir/'tier_cutoffs.json'}")


if __name__ == "__main__":
    main()
