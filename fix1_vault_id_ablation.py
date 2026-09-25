"""
Fix 1 — Infrastructure-identifier ablation (vault_id, pod_id, pod_slot_num,
cluster_id, is_legacy_format)

Directly answers the reviewer question "why is your top SHAP feature a
location ID, not a SMART attribute?" by retraining LightGBM WITHOUT these
columns and comparing test AUPRC/ROC-AUC against the original. Reuses
Stage 10's exact training/streaming harness so the comparison is
apples-to-apples (same tuning budget, same threshold-selection logic).

Two outcomes, both publishable:
  - AUPRC barely changes  -> the model doesn't actually depend on these
    identifiers much; vault_id's SHAP rank may reflect a real but small
    effect, and the paper can state robustness to their exclusion.
  - AUPRC drops meaningfully -> the model WAS leaning on infrastructure
    identifiers; report this honestly as a limitation/confound, and use
    the no-infra model as your reported headline result instead.

Usage:
    python3 fix1_vault_id_ablation.py \
        --train_resampled data/processed/train_resampled.parquet \
        --val data/processed/features_val.parquet \
        --test data/processed/features_test.parquet \
        --cost_matrix data/processed/cost_matrix.json \
        --lgbm_metrics results/lgbm_metrics.csv \
        --target failure_within_14d \
        --n_trials 10 \
        --out_dir results
"""
import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from importlib.util import spec_from_file_location, module_from_spec


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INFRA_COLS = {"vault_id", "pod_id", "pod_slot_num", "cluster_id", "is_legacy_format"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_resampled", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--cost_matrix", type=str, required=True)
    parser.add_argument("--lgbm_metrics", type=str, required=True, help="Original (with infra cols) metrics")
    parser.add_argument("--target", type=str, default="failure_within_14d")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lgbm_mod = load_module("lgbm_stage6", Path(__file__).parent / "06_train_lightgbm.py")
    eval_mod = load_module("eval_stage10", Path(__file__).parent / "10_evaluation.py")
    with open(args.cost_matrix) as f:
        cost_matrix = json.load(f)

    with_infra = pd.read_csv(args.lgbm_metrics).iloc[0].to_dict()

    train_df, all_feature_cols = lgbm_mod.load_train_chunked(Path(args.train_resampled), args.target)
    no_infra_cols = [c for c in all_feature_cols if c not in INFRA_COLS]
    removed = [c for c in all_feature_cols if c in INFRA_COLS]
    print(f"[ABLATION] Removing {len(removed)} infrastructure columns: {removed}")
    print(f"[ABLATION] Training on {len(no_infra_cols)}/{len(all_feature_cols)} remaining feature columns")

    no_infra_metrics = eval_mod.train_and_stream_evaluate(
        train_df, no_infra_cols, args.target, Path(args.val), Path(args.test),
        cost_matrix, args.n_trials, lgbm_mod
    )
    del train_df
    gc.collect()

    result = pd.DataFrame([
        {"variant": "with infrastructure identifiers (original)",
         **{k: with_infra[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
        {"variant": "without infrastructure identifiers",
         **{k: no_infra_metrics[k] for k in ["roc_auc", "auprc", "f1", "precision", "recall"]}},
    ])
    print("\n" + result.to_string(index=False))

    auprc_pct_change = 100 * (no_infra_metrics["auprc"] - with_infra["auprc"]) / with_infra["auprc"]
    print(f"\n[VERDICT] AUPRC change from removing infrastructure identifiers: {auprc_pct_change:+.1f}%")
    if abs(auprc_pct_change) < 10:
        print("[VERDICT] Small change -- model is largely robust to excluding location identifiers.")
    else:
        print("[VERDICT] Meaningful change -- the model WAS leaning on infrastructure identifiers. "
              "Report this honestly; consider using the no-infra model as your headline result.")

    result.to_csv(out_dir / "ablation_infra_identifiers.csv", index=False)
    print(f"\n[SAVE] {out_dir/'ablation_infra_identifiers.csv'}")


if __name__ == "__main__":
    main()
