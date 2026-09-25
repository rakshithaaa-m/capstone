"""
Stage 11 — SHAP Explainability (MEMORY-SAFE SAMPLING)

The original script's pd.read_parquet(args.test) loads the ENTIRE test set
before sampling n_samples rows from it -- at 144.7M rows, that would OOM
before SHAP ever runs, the same failure mode every prior stage hit. Since
SHAP only actually needs a few thousand rows (n_samples, default 3000),
this version samples directly at the row-group level -- reading only
enough row groups to gather n_samples rows, never the full file.

Usage:
    python3 11_shap_explainability.py \
        --test data/processed/features_test.parquet \
        --lgbm_model models/lgbm_model.txt \
        --lgbm_feature_cols models/lgbm_feature_cols.json \
        --n_samples 3000 \
        --out_dir results
"""
import argparse
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap


def coerce_legacy_format(df: pd.DataFrame) -> pd.DataFrame:
    if "is_legacy_format" in df.columns and not pd.api.types.is_numeric_dtype(df["is_legacy_format"]):
        df["is_legacy_format"] = df["is_legacy_format"].fillna(False).astype(int)
    return df


def sample_rows(path: Path, n_samples: int, extra_cols: list[str], seed: int = 42) -> pd.DataFrame:
    """Row-group-level sampling: reads whole row groups (in random order)
    until n_samples rows are gathered, then stops -- never loads the full
    144.7M-row file. Same approach used for Stage 7's normalization
    sampling, applied here since SHAP only needs a few thousand rows."""
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    row_group_order = rng.permutation(pf.metadata.num_row_groups)

    parts = []
    total = 0
    for rg in row_group_order:
        tbl = pf.read_row_group(int(rg), columns=extra_cols)
        part = tbl.to_pandas()
        part = coerce_legacy_format(part)
        parts.append(part)
        total += len(part)
        if total >= n_samples:
            break
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) > n_samples:
        sample = sample.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    print(f"[SAMPLE] Gathered {len(sample):,} row(s) from {len(parts)} row group(s) "
          f"(target was {n_samples:,})")
    return sample


def sample_at_risk_rows(test_path: Path, risk_tiers_path: Path, feature_cols: list[str], max_rows: int) -> pd.DataFrame:
    """Instead of a RANDOM sample of the whole fleet (which rarely
    intersects the ~30-40K currently at-risk drives at all), this fetches
    EVERY currently HIGH/MEDIUM drive's most recent row, joined directly
    against the test parquet via DuckDB. This is what makes Stage 12's
    mitigation mapping actually cover the population it's meant to serve,
    instead of the ~0.3% overlap a random 3,000-row sample gives you.
    Bounded by the at-risk population size (tens of thousands), not by
    the full ~145M-row test file -- the join and fetch are both cheap."""
    con = duckdb.connect()
    feature_col_list = ", ".join(f't."{c}"' for c in feature_cols)
    query = f"""
        WITH latest AS (
            SELECT serial_number, date, risk_tier,
                   ROW_NUMBER() OVER (PARTITION BY serial_number ORDER BY date DESC) AS rn
            FROM read_csv_auto('{risk_tiers_path.as_posix()}')
        ),
        at_risk AS (
            SELECT serial_number, date FROM latest WHERE rn = 1 AND risk_tier IN ('HIGH', 'MEDIUM')
        )
        SELECT t.serial_number AS serial_number, t.date AS date, {feature_col_list}
        FROM read_parquet('{test_path.as_posix()}') AS t
        INNER JOIN at_risk AS a
            ON t.serial_number = a.serial_number AND t.date = a.date
    """
    sample = con.execute(query).fetchdf()
    con.close()
    sample = coerce_legacy_format(sample)

    if len(sample) > max_rows:
        print(f"[AT-RISK SAMPLE] {len(sample):,} at-risk drives found, capping to {max_rows:,} "
              f"for SHAP runtime -- increase --max_at_risk_rows to cover more.")
        sample = sample.sample(n=max_rows, random_state=42).reset_index(drop=True)
    print(f"[AT-RISK SAMPLE] Computing SHAP for {len(sample):,} currently at-risk (HIGH/MEDIUM) drives "
          f"-- this is what Stage 12's mitigation mapping should be pointed at for full coverage.")
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP explainability for the LightGBM model (memory-safe sampling)")
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--lgbm_model", type=str, required=True)
    parser.add_argument("--lgbm_feature_cols", type=str, required=True,
                         help="models/lgbm_feature_cols.json from Stage 6")
    parser.add_argument("--n_samples", type=int, default=3000,
                         help="Rows to run SHAP on -- TreeExplainer is fast, this caps memory/time")
    parser.add_argument("--n_dependence_plots", type=int, default=5)
    parser.add_argument("--risk_tiers", type=str, default=None,
                         help="results/risk_tiers_test.csv from Stage 9 -- if provided, ALSO computes "
                              "SHAP for every currently at-risk (HIGH/MEDIUM) drive's most recent row, "
                              "saved separately as shap_values_at_risk.parquet. Point Stage 12 at THIS "
                              "file (not shap_values.parquet) for full mitigation coverage instead of "
                              "a random sample's lucky intersection with at-risk drives.")
    parser.add_argument("--max_at_risk_rows", type=int, default=50_000,
                         help="Safety cap on the at-risk sample size, in case the at-risk population "
                              "is very large")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.lgbm_feature_cols) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    booster = lgb.Booster(model_file=args.lgbm_model)

    extra_cols = list(set(feature_cols) | {"serial_number", "date"})
    sample_df = sample_rows(Path(args.test), args.n_samples, extra_cols)

    print(f"[SHAP] Computing TreeExplainer SHAP values on {len(sample_df)} rows...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(sample_df[feature_cols])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"[SHAP] Done -- shape: {shap_values.shape}")

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    top_features = mean_abs_shap.sort_values(ascending=False).head(15)
    print("\n[TOP 15 features by mean |SHAP|]")
    print(top_features.to_string())

    top_features.reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    ).to_csv(out_dir / "shap_top_features.csv", index=False)

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
        ax.set_title(f"SHAP dependence -- {feat_name}")
        fig.tight_layout()
        out_path = plots_dir / f"dep_{rank}_{feat_name}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[SHAP] saved dependence plot: {out_path}")

    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
    shap_df["serial_number"] = sample_df["serial_number"].values
    shap_df["date"] = sample_df["date"].values
    shap_df.to_parquet(out_dir / "shap_values.parquet", index=False)

    print(f"\n[SAVE] {out_dir/'shap_top_features.csv'}")
    print(f"[SAVE] {out_dir/'shap_values.parquet'}")

    if args.risk_tiers:
        at_risk_df = sample_at_risk_rows(
            Path(args.test), Path(args.risk_tiers), feature_cols, args.max_at_risk_rows
        )
        if len(at_risk_df) == 0:
            print("[AT-RISK SAMPLE] No at-risk drives found -- skipping.")
        else:
            print(f"[SHAP] Computing TreeExplainer SHAP values on {len(at_risk_df)} at-risk rows...")
            at_risk_shap_values = explainer.shap_values(at_risk_df[feature_cols])
            if isinstance(at_risk_shap_values, list):
                at_risk_shap_values = at_risk_shap_values[1]

            at_risk_shap_df = pd.DataFrame(at_risk_shap_values, columns=[f"shap_{c}" for c in feature_cols])
            at_risk_shap_df["serial_number"] = at_risk_df["serial_number"].values
            at_risk_shap_df["date"] = at_risk_df["date"].values
            at_risk_shap_df.to_parquet(out_dir / "shap_values_at_risk.parquet", index=False)
            print(f"[SAVE] {out_dir/'shap_values_at_risk.parquet'} "
                  f"({len(at_risk_shap_df):,} at-risk drives, full coverage -- point Stage 12 at this file)")


if __name__ == "__main__":
    main()