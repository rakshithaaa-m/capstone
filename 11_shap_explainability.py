"""
Stage 11 — SHAP Explainability

Runs SHAP TreeExplainer on the trained LightGBM model, saves per-row SHAP
values (needed by Stage 12's root-cause mapping) and a top-features
summary table.

Usage:
    python3 11_shap_explainability.py \
        --test data/processed/features_test.parquet \
        --lgbm_model models/lgbm_model.txt \
        --lgbm_threshold models/lgbm_threshold.json \
        --n_samples 3000 \
        --out_dir results
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # headless-safe, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP explainability for the LightGBM model")
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--lgbm_model", type=str, required=True)
    parser.add_argument("--lgbm_threshold", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=3000,
                         help="Max rows to run SHAP on (TreeExplainer is fast, but this caps memory/time)")
    parser.add_argument("--n_dependence_plots", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.lgbm_threshold) as f:
        lgbm_meta = json.load(f)
    feature_cols = lgbm_meta["feature_cols"]

    booster = lgb.Booster(model_file=args.lgbm_model)
    test_df = pd.read_parquet(args.test)

    n_samples = min(args.n_samples, len(test_df))
    sample_df = test_df.sample(n=n_samples, random_state=42) if len(test_df) > n_samples else test_df
    print(f"[SHAP] Computing TreeExplainer SHAP values on {len(sample_df)} rows...")

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(sample_df[feature_cols])

    # LightGBM binary classifiers can return either a single array or a
    # [class0, class1] list depending on version — normalize to the
    # positive-class array either way.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"[SHAP] Done — shape: {shap_values.shape}")

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    top_features = mean_abs_shap.sort_values(ascending=False).head(15)
    print("\n[TOP 15 features by mean |SHAP|]")
    print(top_features.to_string())

    top_features.reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    ).to_csv(out_dir / "shap_top_features.csv", index=False)

    # Dependence plots for the top-N features: shows how the feature's own
    # value relates to its SHAP contribution, not just its overall
    # importance rank — this is what actually goes in the report.
    plots_dir = out_dir / "shap" / "dependence"
    plots_dir.mkdir(parents=True, exist_ok=True)
    top_n_names = top_features.head(args.n_dependence_plots).index.tolist()
    for rank, feat_name in enumerate(top_n_names, start=1):
        feat_idx = feature_cols.index(feat_name)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sample_df[feat_name].to_numpy(), shap_values[:, feat_idx], s=10, alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel(feat_name)
        ax.set_ylabel(f"SHAP value for {feat_name}")
        ax.set_title(f"SHAP dependence — {feat_name}")
        fig.tight_layout()
        out_path = plots_dir / f"dep_{rank}_{feat_name}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[SHAP] saved dependence plot: {out_path}")

    # Save per-row SHAP values keyed by (serial_number, date) so Stage 12
    # can look up "what drove this specific drive's risk score" per row.
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    shap_df["serial_number"] = sample_df["serial_number"].values
    shap_df["date"] = sample_df["date"].values
    shap_df.to_parquet(out_dir / "shap_values.parquet", index=False)

    print(f"\n[SAVE] {out_dir/'shap_top_features.csv'}")
    print(f"[SAVE] {out_dir/'shap_values.parquet'}")


if __name__ == "__main__":
    main()
