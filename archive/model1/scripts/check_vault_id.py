"""
Quick diagnostic: does vault_id's SHAP signal come from a few specific
vaults (suggesting a batch/deployment effect worth investigating) or
spread broadly across many vaults (more consistent with a general,
plausible physical effect)?

Usage:
    python check_vault_id.py --shap_values results/shap_values.parquet --test data/processed/features_test.parquet
"""
import argparse
import duckdb
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--shap_values", default="results/shap_values.parquet")
parser.add_argument("--test", default="data/processed/features_test.parquet")
args = parser.parse_args()

shap_df = pd.read_parquet(args.shap_values, columns=["serial_number", "date", "shap_vault_id"])
shap_df.to_csv("_tmp_shap_keys.csv", index=False)

con = duckdb.connect()
joined = con.execute(f"""
    SELECT s.serial_number, s.date, s.shap_vault_id, t.vault_id
    FROM read_csv_auto('_tmp_shap_keys.csv') AS s
    INNER JOIN read_parquet('{args.test}') AS t
        ON s.serial_number = t.serial_number AND s.date = t.date
""").fetchdf()
con.close()

print(f"[JOINED] {len(joined)} of {len(shap_df)} sampled rows matched\n")

# Per-vault summary: how many sampled rows come from each vault, and what
# their SHAP contribution looks like there
summary = (
    joined.groupby("vault_id")["shap_vault_id"]
    .agg(n_rows="count", mean_abs_shap=lambda x: x.abs().mean(), mean_shap="mean")
    .sort_values("mean_abs_shap", ascending=False)
)
print("[TOP 15 vaults by mean |SHAP contribution|]")
print(summary.head(15).to_string())

print(f"\n[SPREAD CHECK] {joined['vault_id'].nunique()} distinct vaults present in this sample")
top5_share = summary.head(5)["n_rows"].sum() / len(joined)
print(f"[SPREAD CHECK] Top 5 vaults account for {top5_share:.1%} of sampled rows "
      f"(if this is close to the overall n_vaults share, no single vault dominates)")

import os
os.remove("_tmp_shap_keys.csv")
