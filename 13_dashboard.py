"""
Stage 13 — Dashboard

Streamlit app reading Stage 9-12's outputs: a fleet-wide risk overview and
a per-drive detail view (risk tier, time-to-failure, root cause,
recommended action). Run with:

    streamlit run 13_dashboard.py -- --results_dir results

On ARC Labs, view it via an SSH tunnel back to your laptop:
    ssh -R 8501:localhost:8501 <your_arc_connection>
then open http://localhost:8501 in your laptop's browser.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


def parse_args():
    # Streamlit passes its own args before "--", so we only parse what
    # comes after it; falls back to a default if run without any.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args(argv)


@st.cache_data
def load_data(results_dir: str):
    results = Path(results_dir)
    risk_tiers = pd.read_csv(results / "risk_tiers_test.csv")
    tier_accuracy = pd.read_csv(results / "tier_accuracy_test.csv")

    mitigation_path = results / "mitigation_mapping.json"
    mitigation = json.loads(mitigation_path.read_text()) if mitigation_path.exists() else []
    mitigation_df = pd.DataFrame(mitigation) if mitigation else pd.DataFrame(
        columns=["serial_number", "date", "risk_tier", "top_driver_feature", "root_cause", "recommended_action"]
    )

    hybrid_scores_path = results / "hybrid_test_scores.csv"
    hybrid_scores = pd.read_csv(hybrid_scores_path) if hybrid_scores_path.exists() else None

    return risk_tiers, tier_accuracy, mitigation_df, hybrid_scores


def main():
    args = parse_args()
    st.set_page_config(page_title="HDD Failure Risk Dashboard", layout="wide")
    st.title("Predictive System Health Monitoring")
    st.caption("Fleet-wide HDD failure risk, powered by the LightGBM + LSTM hybrid model")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        st.error(f"Results directory '{results_dir}' not found. Run Stages 6-12 first.")
        return

    risk_tiers, tier_accuracy, mitigation_df, hybrid_scores = load_data(args.results_dir)

    # --- Fleet-wide overview ---
    st.header("Fleet Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total drives monitored", len(risk_tiers))
    col2.metric("HIGH risk", int((risk_tiers["risk_tier"] == "HIGH").sum()))
    col3.metric("MEDIUM risk", int((risk_tiers["risk_tier"] == "MEDIUM").sum()))
    col4.metric("LOW risk", int((risk_tiers["risk_tier"] == "LOW").sum()))

    st.subheader("Detection rate by tier")
    st.dataframe(tier_accuracy, use_container_width=True)

    tier_counts = risk_tiers["risk_tier"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0)
    st.bar_chart(tier_counts)

    # --- Per-drive detail ---
    st.header("Per-Drive Detail")
    at_risk_serials = risk_tiers[risk_tiers["risk_tier"].isin(["HIGH", "MEDIUM"])]["serial_number"].unique()

    if len(at_risk_serials) == 0:
        st.info("No drives currently in HIGH or MEDIUM risk tiers.")
    else:
        selected = st.selectbox("Select a drive to inspect", sorted(at_risk_serials))
        drive_row = risk_tiers[risk_tiers["serial_number"] == selected].iloc[0]
        drive_mitigation = mitigation_df[mitigation_df["serial_number"] == selected]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Risk tier", drive_row["risk_tier"])
            st.metric("Hybrid risk score", f"{drive_row['hybrid_score']:.3f}")
        with c2:
            if hybrid_scores is not None:
                match = hybrid_scores[hybrid_scores.index == risk_tiers[risk_tiers["serial_number"] == selected].index[0]]
                if not match.empty and "lstm_rul_pred" in match.columns:
                    st.metric("Estimated time-to-failure (days)", f"{match['lstm_rul_pred'].iloc[0]:.1f}")

        if not drive_mitigation.empty:
            st.subheader("Root cause")
            st.write(drive_mitigation.iloc[0]["root_cause"])
            st.subheader("Recommended action")
            st.write(drive_mitigation.iloc[0]["recommended_action"])
        else:
            st.info("No SHAP-derived explanation available for this drive in the current sample.")

    st.header("All Flagged Drives")
    st.dataframe(
        risk_tiers[risk_tiers["risk_tier"].isin(["HIGH", "MEDIUM"])]
        .sort_values("hybrid_score", ascending=False),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
