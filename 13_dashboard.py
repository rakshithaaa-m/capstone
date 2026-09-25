"""
Stage 13 — Dashboard (memory-safe, correctness-fixed version)

CHANGES FROM THE PREVIOUS VERSION:
  1. No longer loads risk_tiers_test.csv (~144M rows) or
     hybrid_test_scores.csv (~144M rows) fully into pandas at startup.
     Both are queried on-demand via DuckDB directly against the CSV files
     on disk -- fleet-wide stats and per-drive lookups return small
     results regardless of file size, and the dashboard starts instantly
     instead of loading 20+GB into memory before showing anything.
  2. Fixed a real correctness bug: the previous version matched a drive's
     LSTM time-to-failure estimate by comparing INTEGER ROW INDEX between
     two separately-written CSV files -- there's no guarantee those two
     files share row ordering, so this could silently show the WRONG
     drive's estimate with no error at all. Now joined properly by
     (serial_number, date).
  3. Fixed a second bug: the previous version picked whichever row
     happened to be first (`.iloc[0]`) for a selected drive, which could
     be a stale reading from early in the test period rather than its
     current status. Now explicitly queries each drive's MOST RECENT
     date.
  4. Fixed a misleading metric: "Total drives monitored" previously
     showed len(risk_tiers), which counts drive-DAY rows (~144M), not
     unique physical drives (~300K). Now uses COUNT(DISTINCT
     serial_number), and fleet tier counts reflect each drive's most
     recent status, not a raw count of every drive-day row.

Run with:
    streamlit run 13_dashboard.py -- --results_dir results

On ARC Labs, view it via an SSH tunnel back to your laptop:
    ssh -R 8501:localhost:8501 <your_arc_connection>
then open http://localhost:8501 in your laptop's browser.
"""
import argparse
import importlib.util as _ilu
import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

_agent_spec = _ilu.spec_from_file_location("agent14", Path(__file__).parent / "14_agentic_mitigation.py")
_agent_mod = _ilu.module_from_spec(_agent_spec)
_agent_spec.loader.exec_module(_agent_mod)

TIER_COLORS = {"HIGH": "#E5484D", "MEDIUM": "#F5A623", "LOW": "#3FB950"}
TIER_ICONS = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}

CUSTOM_CSS = """
<style>
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.75; }
    .tier-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        color: white;
    }
    .info-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #4F8BFF;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .info-card h4 { margin: 0 0 6px 0; font-size: 0.95rem; opacity: 0.85; }
    .info-card p { margin: 0; font-size: 1.02rem; }
</style>
"""


def tier_badge_html(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#888")
    icon = TIER_ICONS.get(tier, "")
    return f'<span class="tier-badge" style="background-color:{color};">{icon} {tier}</span>'


def parse_args():
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    return parser.parse_args(argv)


@st.cache_data
def load_small_tables(results_dir: str):
    """These are genuinely small (a few hundred rows at most) -- safe to
    load fully and cache in memory, unlike risk_tiers/hybrid_scores."""
    results = Path(results_dir)
    tier_accuracy = pd.read_csv(results / "tier_accuracy_test.csv")

    mitigation_path = results / "mitigation_mapping.json"
    mitigation = json.loads(mitigation_path.read_text()) if mitigation_path.exists() else []
    mitigation_df = pd.DataFrame(mitigation) if mitigation else pd.DataFrame(
        columns=["serial_number", "date", "risk_tier", "top_driver_feature", "root_cause", "recommended_action"]
    )
    return tier_accuracy, mitigation_df


def get_source_connection(results_dir: str):
    """Returns (connection, risk_tiers_source, hybrid_source) -- uses the
    indexed dashboard.duckdb database if build_dashboard_db.py has been
    run (near-instant lookups), otherwise falls back to scanning the raw
    CSVs directly (slow, but works without any setup step)."""
    db_path = Path(results_dir) / "dashboard.duckdb"
    if db_path.exists():
        con = duckdb.connect(str(db_path), read_only=True)
        return con, "risk_tiers", "hybrid_scores"
    con = duckdb.connect()
    risk_tiers_csv = str(Path(results_dir) / "risk_tiers_test.csv")
    hybrid_csv = str(Path(results_dir) / "hybrid_test_scores.csv")
    return con, f"read_csv_auto(\'{risk_tiers_csv}\')", f"read_csv_auto(\'{hybrid_csv}\')"


@st.cache_data
def fleet_overview(results_dir: str):
    """Each drive's MOST RECENT row determines its current tier, so a
    drive isn't counted once per day it appears in the test period --
    it's counted once, by its latest known status. Computed entirely in
    DuckDB; only 4 small numbers come back to Python."""
    con, risk_tiers_src, _ = get_source_connection(results_dir)
    result = con.execute(f"""
        WITH latest AS (
            SELECT serial_number, risk_tier,
                   ROW_NUMBER() OVER (PARTITION BY serial_number ORDER BY date DESC) AS rn
            FROM {risk_tiers_src}
        )
        SELECT risk_tier, COUNT(*) AS n
        FROM latest
        WHERE rn = 1
        GROUP BY risk_tier
    """).fetchdf()
    con.close()
    counts = {row["risk_tier"]: int(row["n"]) for _, row in result.iterrows()}
    total = sum(counts.values())
    return total, counts.get("HIGH", 0), counts.get("MEDIUM", 0), counts.get("LOW", 0)


@st.cache_data
def at_risk_drive_list(results_dir: str):
    """Only drives whose MOST RECENT status is HIGH or MEDIUM -- a small,
    bounded list regardless of how many total drive-day rows exist."""
    con, risk_tiers_src, _ = get_source_connection(results_dir)
    result = con.execute(f"""
        WITH latest AS (
            SELECT serial_number, risk_tier,
                   ROW_NUMBER() OVER (PARTITION BY serial_number ORDER BY date DESC) AS rn
            FROM {risk_tiers_src}
        )
        SELECT serial_number FROM latest WHERE rn = 1 AND risk_tier IN ('HIGH', 'MEDIUM')
        ORDER BY serial_number
    """).fetchdf()
    con.close()
    return result["serial_number"].tolist()


def get_drive_detail(results_dir: str, serial_number: str):
    """Queries just ONE drive's most recent row, properly joined by
    (serial_number, date) against hybrid_test_scores.csv for its RUL
    estimate -- not an index-position guess. Cheap regardless of file
    size since only one drive is ever fetched."""
    con, risk_tiers_src, hybrid_src = get_source_connection(results_dir)
    hybrid_path = Path(results_dir) / "hybrid_test_scores.csv"
    db_path = Path(results_dir) / "dashboard.duckdb"

    row = con.execute(f"""
        SELECT serial_number, date, hybrid_score, risk_tier
        FROM {risk_tiers_src}
        WHERE serial_number = ?
        ORDER BY date DESC
        LIMIT 1
    """, [serial_number]).fetchdf()
    if row.empty:
        con.close()
        return None

    if db_path.exists() or hybrid_path.exists():
        lstm_row = con.execute(f"""
            SELECT lstm_rul_pred
            FROM {hybrid_src}
            WHERE serial_number = ? AND date = ?
            LIMIT 1
        """, [serial_number, row.iloc[0]["date"]]).fetchdf()
        rul_pred = float(lstm_row.iloc[0]["lstm_rul_pred"]) if not lstm_row.empty else None
    else:
        rul_pred = None

    con.close()
    return row.iloc[0], rul_pred


def get_flagged_drives_preview(results_dir: str, limit: int = 200):
    """A bounded preview table (top N by hybrid score) instead of
    materializing every HIGH/MEDIUM drive-day row -- the table below was
    previously unbounded and could try to render hundreds of thousands
    of rows in the browser."""
    con, risk_tiers_src, _ = get_source_connection(results_dir)
    result = con.execute(f"""
        WITH latest AS (
            SELECT serial_number, date, hybrid_score, risk_tier,
                   ROW_NUMBER() OVER (PARTITION BY serial_number ORDER BY date DESC) AS rn
            FROM {risk_tiers_src}
        )
        SELECT serial_number, date, hybrid_score, risk_tier
        FROM latest
        WHERE rn = 1 AND risk_tier IN ('HIGH', 'MEDIUM')
        ORDER BY hybrid_score DESC
        LIMIT {limit}
    """).fetchdf()
    con.close()
    return result


def main():
    args = parse_args()
    st.set_page_config(page_title="HDD Failure Risk Dashboard", layout="wide", page_icon="💽")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        st.error(f"Results directory '{results_dir}' not found. Run Stages 6-12 first.")
        return

    tier_accuracy, mitigation_df = load_small_tables(args.results_dir)
    total, n_high, n_medium, n_low = fleet_overview(args.results_dir)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 💽 System Health Monitoring")
        st.caption("LightGBM + Attention-LSTM hybrid risk model")
        st.divider()
        st.markdown("**Fleet snapshot**")
        st.markdown(f"Total drives: **{total:,}**")
        st.markdown(
            f"{tier_badge_html('HIGH')} &nbsp; {n_high:,}<br>"
            f"{tier_badge_html('MEDIUM')} &nbsp; {n_medium:,}<br>"
            f"{tier_badge_html('LOW')} &nbsp; {n_low:,}",
            unsafe_allow_html=True,
        )
        st.divider()
        explained_count = mitigation_df["serial_number"].nunique() if not mitigation_df.empty else 0
        at_risk_count = n_high + n_medium
        st.markdown("**Explainability coverage**")
        st.progress(explained_count / at_risk_count if at_risk_count else 0)
        st.caption(f"{explained_count:,} / {at_risk_count:,} at-risk drives have a root-cause explanation")
        st.divider()
        st.caption(f"Data source: `{results_dir}`")

    # --- Header ---
    st.title("Predictive System Health Monitoring")
    st.caption("Fleet-wide HDD failure risk, powered by the LightGBM + LSTM hybrid model")

    tab_overview, tab_detail, tab_flagged, tab_agent = st.tabs(
        ["📊 Fleet Overview", "🔍 Per-Drive Detail", "🚩 Top Flagged Drives", "🤖 Agentic Assistant"]
    )

    # =========================== TAB 1: OVERVIEW ===========================
    with tab_overview:
        st.caption("Each drive counted once, by its most recent risk assessment -- not once per "
                   "day it appears in the test period.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total drives monitored", f"{total:,}")
        col2.metric("🔴 HIGH risk", f"{n_high:,}")
        col3.metric("🟠 MEDIUM risk", f"{n_medium:,}")
        col4.metric("🟢 LOW risk", f"{n_low:,}")

        st.markdown("")
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Drives by risk tier")
            tier_counts = pd.Series({"HIGH": n_high, "MEDIUM": n_medium, "LOW": n_low})
            st.bar_chart(tier_counts, color="#4F8BFF")
        with right:
            st.subheader("Detection rate by tier")
            st.caption("Drive-DAY observations across the full test period (not unique drives) -- "
                       "this is what determines the calibrated recall targets.")
            tier_accuracy_display = tier_accuracy.rename(columns={"total_drives": "total_drive_day_rows"})
            st.dataframe(tier_accuracy_display, width="stretch", hide_index=True)

    # ======================== TAB 2: PER-DRIVE DETAIL =======================
    with tab_detail:
        at_risk_serials = at_risk_drive_list(args.results_dir)
        explained_serials = sorted(mitigation_df["serial_number"].unique().tolist()) if not mitigation_df.empty else []

        if len(at_risk_serials) == 0:
            st.info("No drives currently in HIGH or MEDIUM risk tiers.")
        else:
            cov_col, toggle_col = st.columns([3, 2])
            with cov_col:
                st.caption(f"**{len(explained_serials):,}** of **{len(at_risk_serials):,}** at-risk "
                           f"drives have a root-cause explanation available.")
            with toggle_col:
                show_all = st.checkbox("Show all at-risk drives", value=False)
            drive_pool = at_risk_serials if show_all else explained_serials

            if len(drive_pool) == 0:
                st.warning("No drives with a root-cause explanation available. Check 'show all' above.")
            else:
                selected = st.selectbox("Select a drive to inspect", drive_pool)
                detail = get_drive_detail(args.results_dir, selected)

                if detail is None:
                    st.warning("No data found for this drive.")
                else:
                    drive_row, rul_pred = detail
                    st.markdown("#### " + selected)
                    st.markdown(tier_badge_html(drive_row["risk_tier"]) +
                               f"&nbsp;&nbsp;<span style='opacity:0.7'>as of {drive_row['date']}</span>",
                               unsafe_allow_html=True)
                    st.markdown("")

                    m1, m2 = st.columns(2)
                    m1.metric("Hybrid risk score", f"{drive_row['hybrid_score']:.3f}")
                    if rul_pred is not None:
                        m2.metric("Estimated time-to-failure", f"{rul_pred:.1f} days")

                    gauge_color = TIER_COLORS.get(drive_row["risk_tier"], "#4F8BFF")
                    pct = max(0.0, min(1.0, float(drive_row["hybrid_score"]))) * 100
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.08); border-radius:6px; height:10px; margin-top:-8px;">'
                        f'<div style="width:{pct:.1f}%; background:{gauge_color}; height:10px; border-radius:6px;"></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("")
                    drive_mitigation = mitigation_df[mitigation_df["serial_number"] == selected] if not mitigation_df.empty else mitigation_df
                    if not drive_mitigation.empty:
                        st.markdown(
                            f'<div class="info-card"><h4>🔎 Root Cause</h4>'
                            f'<p>{drive_mitigation.iloc[0]["root_cause"]}</p></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="info-card"><h4>🛠️ Recommended Action</h4>'
                            f'<p>{drive_mitigation.iloc[0]["recommended_action"]}</p></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No SHAP-derived explanation available for this drive in the current sample.")

    # ======================== TAB 3: TOP FLAGGED DRIVES ======================
    with tab_flagged:
        st.caption("Top 200 by hybrid risk score, among drives currently HIGH or MEDIUM (by most recent status).")
        flagged = get_flagged_drives_preview(args.results_dir)

        def _highlight_tier(row):
            color = TIER_COLORS.get(row["risk_tier"], "#888")
            return [f"background-color: {color}22; color: {color}" if col == "risk_tier" else "" for col in row.index]

        styled = flagged.style.apply(_highlight_tier, axis=1).format({"hybrid_score": "{:.3f}"})
        st.dataframe(styled, width="stretch", hide_index=True)

    # ======================== TAB 4: AGENTIC ASSISTANT ========================
    with tab_agent:
        st.markdown(
            "Give Claude real tools -- spare-parts inventory, technician scheduling, ticket "
            "drafting -- for a selected drive, and let it decide for itself which to use. "
            "All tool data is **simulated** (proof-of-concept, no real inventory/scheduling "
            "system connected)."
        )
        at_risk_serials_agent = at_risk_drive_list(args.results_dir)
        if not at_risk_serials_agent:
            st.info("No at-risk drives to run the agent on.")
        else:
            agent_selected = st.selectbox("Select a drive", at_risk_serials_agent, key="agent_drive_select")
            mode_col, run_col = st.columns([2, 1])
            with mode_col:
                use_dry_run = st.toggle(
                    "Dry-run mode (rule-based stand-in, no API key/cost needed)",
                    value=(not os.environ.get("ANTHROPIC_API_KEY")),
                )
            with run_col:
                run_clicked = st.button("▶ Run agent", type="primary", width="stretch")

            if run_clicked:
                try:
                    context = _agent_mod.build_drive_context(Path(args.results_dir), agent_selected)
                    with st.spinner("Agent is deciding which tools to use..."):
                        result = _agent_mod.run_agent(context, dry_run=use_dry_run)

                    st.subheader("Tool calls made by the agent")
                    if not result["tool_trace"]:
                        st.caption("The agent made no tool calls and answered directly.")
                    for step in result["tool_trace"]:
                        with st.expander(f"🔧 {step['tool']}"):
                            st.json({"input": step["input"], "output": step["output"]})

                    st.subheader("Agent summary")
                    st.markdown(f'<div class="info-card"><p>{result["summary"]}</p></div>', unsafe_allow_html=True)

                    if result.get("ticket"):
                        st.subheader("Drafted ticket")
                        st.json(result["ticket"])
                except Exception as e:
                    st.error(f"Agent run failed: {e}")

    st.divider()
    st.caption(
        "Pipeline: Data Ingestion → Cleaning → Feature Engineering → Temporal Split → "
        "Imbalance Handling → LightGBM + Attention-LSTM → Hybrid Fusion → Risk-Tier Calibration → "
        "Evaluation → SHAP Explainability → Mitigation Mapping → Agentic Assistant → Dashboard"
    )


if __name__ == "__main__":
    main()
