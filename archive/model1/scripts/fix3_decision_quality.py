"""
Fix 3 — Decision-quality comparison: does Hybrid Fusion actually make
BETTER decisions than LightGBM alone, under your own cost matrix?

Your AUPRC comparison shows Hybrid (0.073) very slightly below LightGBM
alone (0.077) -- but AUPRC ranks ALL possible thresholds; it doesn't
measure what actually happens at the ONE threshold each model is
calibrated to operate at. This script computes actual expected cost
(false_negative_cost * FN + false_positive_cost * FP) for both models at
their own cost-optimal thresholds, on the same test rows -- the metric
that matters for "should you deploy the hybrid score."

No retraining needed -- this reads Stage 6/8's already-computed outputs.

Usage:
    python3 fix3_decision_quality.py \
        --lgbm_test_probs results/lgbm_test_probs.csv \
        --lgbm_threshold models/lgbm_threshold.json \
        --hybrid_test_scores results/hybrid_test_scores.csv \
        --cost_matrix data/processed/cost_matrix.json \
        --target failure_within_14d \
        --out results/decision_quality.csv
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def cost_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float,
                       fn_cost: float, fp_cost: float) -> dict:
    pred_positive = probs >= threshold
    fn = int(((~pred_positive) & (y_true == 1)).sum())
    fp = int((pred_positive & (y_true == 0)).sum())
    tp = int((pred_positive & (y_true == 1)).sum())
    tn = int(((~pred_positive) & (y_true == 0)).sum())
    total_cost = fn * fn_cost + fp * fp_cost
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_cost": total_cost, "cost_per_1000_drives": total_cost / len(y_true) * 1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm_test_probs", type=str, required=True)
    parser.add_argument("--lgbm_val_probs", type=str, required=True,
                        help="Stage 6 validation probabilities. Cost-optimal thresholds are "
                             "selected HERE, never on test.")
    parser.add_argument("--lgbm_threshold", type=str, required=True)
    parser.add_argument("--hybrid_test_scores", type=str, required=True)
    parser.add_argument("--hybrid_val_scores", type=str, required=True,
                        help="Stage 8 validation scores, used for threshold selection only.")
    parser.add_argument("--cost_matrix", type=str, required=True)
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--hybrid_config", type=str, required=True, help="models/hybrid_config.json")
    parser.add_argument("--out", type=str, default="results/decision_quality.csv")
    args = parser.parse_args()

    with open(args.cost_matrix) as f:
        cost_matrix = json.load(f)
    with open(args.lgbm_threshold) as f:
        lgbm_threshold = json.load(f)["threshold"]
    with open(args.hybrid_config) as f:
        hybrid_config = json.load(f)

    # Both files share (serial_number, date) -- fetch only the small
    # numeric columns needed, same pattern used throughout this pipeline.
    con = duckdb.connect()

    def join_scores(a_path, b_path):
        return con.execute(f"""
            SELECT l.lgbm_prob AS lgbm_prob, h.hybrid_score AS hybrid_score,
                   l."{args.target}" AS target
            FROM read_csv_auto('{a_path}') AS l
            INNER JOIN read_csv_auto('{b_path}') AS h
                ON l.serial_number = h.serial_number AND l.date = h.date
        """).fetchdf()

    test = join_scores(args.lgbm_test_probs, args.hybrid_test_scores)
    val = join_scores(args.lgbm_val_probs, args.hybrid_val_scores)
    con.close()

    y_test = test["target"].to_numpy()
    y_val = val["target"].to_numpy()
    fn_cost, fp_cost = cost_matrix["false_negative_cost"], cost_matrix["false_positive_cost"]

    import lightgbm  # noqa -- ensures the same environment as the rest of the pipeline
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("lgbm_stage6", Path(__file__).parent / "06_train_lightgbm.py")
    lgbm_mod = module_from_spec(spec)
    spec.loader.exec_module(lgbm_mod)

    # THRESHOLDS ARE SELECTED ON VALIDATION AND FROZEN.
    # The previous version selected the second threshold on the TEST set and
    # then reported test cost at it, which is threshold tuning on the
    # evaluation data.  That inflates the second row and, because Stage 8
    # chose lgbm_only (so both scores are literally lgbm_prob), it produced a
    # spurious "-7.9% fusion benefit" that was purely a threshold artefact.
    lgbm_test_scores = test["lgbm_prob"].to_numpy()
    hybrid_test_scores = test["hybrid_score"].to_numpy()
    scores_identical = bool(np.allclose(lgbm_test_scores, hybrid_test_scores, atol=1e-12))

    strategy = hybrid_config.get("chosen_strategy", "unknown")
    lgbm_val_optimal = lgbm_mod.select_cost_optimal_threshold(
        y_val, val["lgbm_prob"].to_numpy(), fn_cost, fp_cost)
    hybrid_val_optimal = lgbm_mod.select_cost_optimal_threshold(
        y_val, val["hybrid_score"].to_numpy(), fn_cost, fp_cost)

    lgbm_shipped = cost_at_threshold(y_test, lgbm_test_scores, lgbm_threshold, fn_cost, fp_cost)
    lgbm_shipped["threshold_source"] = "shipped (Stage 6, selected on validation)"
    lgbm_opt = cost_at_threshold(y_test, lgbm_test_scores, lgbm_val_optimal, fn_cost, fp_cost)
    lgbm_opt["threshold_source"] = f"cost-optimal on validation ({lgbm_val_optimal:.4f})"
    hybrid_opt = cost_at_threshold(y_test, hybrid_test_scores, hybrid_val_optimal, fn_cost, fp_cost)
    hybrid_opt["threshold_source"] = f"cost-optimal on validation ({hybrid_val_optimal:.4f})"

    strategy_label = (f"Selected strategy ({strategy})" if not scores_identical
                      else f"Selected strategy ({strategy}) -- IDENTICAL to LightGBM")
    rows = [
        {"model": "LightGBM only", **lgbm_shipped},
        {"model": "LightGBM only", **lgbm_opt},
        {"model": strategy_label, **hybrid_opt},
    ]
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))

    if scores_identical:
        print("\n[CHECK] The shipped strategy is 'lgbm_only', so hybrid_score == lgbm_prob on "
              "every test row (verified, max abs diff < 1e-12). The third row is therefore NOT "
              "an independent model: it is LightGBM at its own cost-optimal threshold.")
        print("[CHECK] Any cost difference between rows 2 and 3 is caused by the THRESHOLD, "
              "not by the model architecture.")

    base = lgbm_opt["total_cost"]
    cost_diff_pct = 100 * (hybrid_opt["total_cost"] - base) / base
    print(f"\n[VERDICT] Shipped strategy total cost is {cost_diff_pct:+.1f}% vs LightGBM, "
          "both at validation-selected thresholds.")
    if scores_identical:
        print("[VERDICT] Because the shipped strategy IS LightGBM, this is not a fusion "
              "result at all. Report it as a THRESHOLD-CALIBRATION finding: the "
              "decision threshold, not the model architecture, drove the cost change.")
    elif cost_diff_pct < 0:
        print("[VERDICT] The shipped strategy produces LOWER expected cost at a threshold "
              "chosen on validation. That is a defensible benefit of the fusion, and it is "
              "the metric to lead with.")
    else:
        print("[VERDICT] The shipped strategy does not reduce expected cost. Report the "
              "fusion honestly -- do not claim a cost benefit that is not supported.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.out, index=False)
    print(f"\n[SAVE] {args.out}")


if __name__ == "__main__":
    main()
