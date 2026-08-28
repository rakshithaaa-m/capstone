"""
Stage 12 — Root Cause and Mitigation Mapping

For each Medium/High-risk drive, looks up its single strongest SHAP-driving
feature (from Stage 11's per-row values) and maps it to a plain-language
root cause and a recommended action, using a rule-based lookup table (not
a model — built once, reused by the dashboard).

Usage:
    python3 12_mitigation_mapping.py \
        --risk_tiers results/risk_tiers_test.csv \
        --shap_values results/shap_values.parquet \
        --out results/mitigation_mapping.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Root-cause / action lookup, keyed by the base SMART attribute (rolling/
# delta suffixes are stripped before lookup, so any *_roll7_mean,
# *_roll30_std, *_delta variant of a given attribute maps to the same
# underlying physical cause).
ROOT_CAUSE_LOOKUP = {
    "smart_5_raw": {
        "root_cause": "Reallocated sector count rising — the drive is remapping bad sectors to spare storage.",
        "action": "Schedule replacement within 7 days; increase SMART polling to daily in the meantime.",
    },
    "smart_187_raw": {
        "root_cause": "Reported uncorrectable errors climbing — data integrity is being compromised on read/write.",
        "action": "Schedule replacement within 7 days; verify backups for data hosted on this drive.",
    },
    "smart_188_raw": {
        "root_cause": "Command timeouts increasing — the drive is failing to respond within the expected time.",
        "action": "Increase monitoring frequency; plan replacement within 14 days.",
    },
    "smart_194_raw": {
        "root_cause": "Operating temperature elevated or unstable — thermal stress accelerates mechanical wear.",
        "action": "Check airflow/cooling around this drive; monitor closely, no immediate replacement required.",
    },
    "smart_197_raw": {
        "root_cause": "Current pending sector count rising — sectors are awaiting remapping, an early failure signal.",
        "action": "Increase SMART polling to daily; flag for replacement if the trend continues.",
    },
    "smart_198_raw": {
        "root_cause": "Offline uncorrectable sector count rising — sectors have failed and could not be remapped.",
        "action": "Schedule replacement within 7 days; this is a strong, direct failure indicator.",
    },
    "smart_9_raw": {
        "root_cause": "Power-on hours are high relative to the fleet — the drive is nearing typical end-of-life age.",
        "action": "Add to the standard replacement rotation; not urgent on its own.",
    },
    "days_since_install": {
        "root_cause": "Drive tenure is high relative to the fleet.",
        "action": "Add to the standard replacement rotation; combine with other signals before acting.",
    },
    "error_total": {
        "root_cause": "Combined error-attribute total is elevated across multiple SMART signals at once.",
        "action": "Treat as a strong composite warning; schedule replacement within 7 days.",
    },
    "temp_stress": {
        "root_cause": "Sustained operation above the normal temperature band.",
        "action": "Check airflow/cooling around this drive; monitor closely.",
    },
}

DEFAULT_ENTRY = {
    "root_cause": "Elevated risk driven by a SMART attribute outside the standard mapping table.",
    "action": "Review the drive's individual SHAP explanation and escalate to standard monitoring rotation.",
}


def strip_suffix(feature_name: str) -> str:
    for suffix in ["_roll7_mean", "_roll7_std", "_roll14_mean", "_roll14_std",
                   "_roll30_mean", "_roll30_std", "_delta"]:
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return feature_name


def top_driver_per_row(shap_df: pd.DataFrame) -> pd.DataFrame:
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
    values = shap_df[shap_cols].values
    # Most INFLUENTIAL feature by magnitude, not just "most positive" — the
    # earlier version's argmax() would pick the least-negative feature if
    # every SHAP value for a row happened to be negative, which isn't
    # actually the strongest explanation for that row. Direction (did it
    # push risk up or down) is recorded separately alongside magnitude.
    top_idx = np.abs(values).argmax(axis=1)
    top_feature_names = [shap_cols[i][len("shap_"):] for i in top_idx]
    top_values = values[np.arange(len(values)), top_idx]
    top_direction = np.where(top_values > 0, "increased risk", "decreased risk")

    out = shap_df[["serial_number", "date"]].copy()
    out["top_driver_raw"] = top_feature_names
    out["top_driver_base"] = [strip_suffix(f) for f in top_feature_names]
    out["top_driver_shap_value"] = top_values
    out["top_driver_direction"] = top_direction
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Map SHAP top drivers to root cause and mitigation actions")
    parser.add_argument("--risk_tiers", type=str, required=True)
    parser.add_argument("--shap_values", type=str, required=True)
    parser.add_argument("--out", type=str, default="results/mitigation_mapping.json")
    args = parser.parse_args()

    tiers_df = pd.read_csv(args.risk_tiers)
    shap_df = pd.read_parquet(args.shap_values)

    tiers_df["date"] = pd.to_datetime(tiers_df["date"])
    shap_df["date"] = pd.to_datetime(shap_df["date"])

    drivers = top_driver_per_row(shap_df)
    merged = tiers_df.merge(drivers, on=["serial_number", "date"], how="left")

    at_risk = merged[merged["risk_tier"].isin(["HIGH", "MEDIUM"])].copy()
    print(f"[MAPPING] {len(at_risk)} Medium/High risk rows to map "
          f"(out of {len(merged)} total, {len(merged) - len(at_risk)} had no SHAP value available)")

    records = []
    for _, row in at_risk.iterrows():
        base = row["top_driver_base"]
        lookup = ROOT_CAUSE_LOOKUP.get(base, DEFAULT_ENTRY)
        records.append({
            "serial_number": row["serial_number"],
            "date": str(row["date"].date()) if pd.notna(row["date"]) else None,
            "risk_tier": row["risk_tier"],
            "top_driver_feature": row.get("top_driver_raw"),
            "top_driver_direction": row.get("top_driver_direction"),
            "root_cause": lookup["root_cause"],
            "recommended_action": lookup["action"],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    if records:
        print("\n[EXAMPLE] first mapped drive:")
        print(json.dumps(records[0], indent=2))
    else:
        print("[NOTE] No Medium/High risk rows had a matching SHAP value in this test run "
              "(expected on very small synthetic data — check with your real dataset).")

    print(f"\n[SAVE] {out_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
