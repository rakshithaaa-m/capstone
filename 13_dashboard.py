"""
Stage 13 - predictive maintenance dashboard.

The dashboard is a serving view, not another training step.  It keeps the
important operational boundary visible:

    ML models -> frozen prediction -> action agent -> operator review

The large drive-day exports are queried through the compact ``drive_status``
table built by ``build_dashboard_db.py``.  The app fails fast with setup
instructions when that serving table is absent, rather than silently scanning
multi-gigabyte CSVs on every interaction.

Run from the project directory::

    streamlit run 13_dashboard.py -- --results_dir results

The only optional live-agent dependency is Anthropic.  Simulation mode is the
default when no API key is configured and exercises the same action/tool
result shape without making a network call.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st


TIER_COLORS = {"HIGH": "#ef6b73", "MEDIUM": "#f3b562", "LOW": "#43c59e"}
TIER_ICONS = {"HIGH": "HIGH", "MEDIUM": "MED", "LOW": "LOW"}


def _load_agent_module():
    path = Path(__file__).with_name("14_agentic_mitigation.py")
    spec = importlib.util.spec_from_file_location("capstone_agent14", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_AGENT = _load_agent_module()


CUSTOM_CSS = """
<style>
    :root {
        --ink: #e8eef7;
        --muted: #91a0b5;
        --panel: #111a2a;
        --panel-2: #162238;
        --line: rgba(170, 190, 220, .14);
        --blue: #6ea8fe;
        --cyan: #63d6d0;
        --red: #ef6b73;
        --amber: #f3b562;
        --green: #43c59e;
    }
    .stApp {
        background:
            radial-gradient(circle at 80% -10%, rgba(74, 125, 220, .16), transparent 34rem),
            #0b1220;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #0d1728;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] hr { border-color: var(--line); }
    h1, h2, h3, h4 { letter-spacing: -.02em; }
    .brand {
        font-size: 1.03rem;
        font-weight: 800;
        letter-spacing: .13em;
        color: var(--ink);
        margin: 0 0 .2rem 0;
    }
    .brand span { color: var(--cyan); }
    .brand-sub {
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.35;
        margin-bottom: 1rem;
    }
    .kicker {
        color: var(--cyan);
        font-size: .72rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: .1rem 0 .45rem 0;
    }
    .page-subtitle { color: var(--muted); margin-top: -.55rem; margin-bottom: 1.15rem; }
    .hero {
        padding: 1.15rem 1.3rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: linear-gradient(120deg, rgba(22, 34, 56, .95), rgba(17, 26, 42, .82));
        margin-bottom: 1rem;
    }
    .hero-title { font-size: 1.65rem; font-weight: 760; margin: 0 0 .35rem 0; }
    .hero-copy { color: var(--muted); max-width: 850px; line-height: 1.45; }
    .flow {
        display: flex;
        align-items: center;
        gap: .55rem;
        flex-wrap: wrap;
        margin-top: .9rem;
    }
    .flow-step {
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: .35rem .7rem;
        color: var(--ink);
        font-size: .78rem;
        background: rgba(255,255,255,.035);
    }
    .flow-step.active { border-color: rgba(110,168,254,.7); color: #cfe0ff; }
    .flow-arrow { color: var(--muted); }
    .panel {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(17, 26, 42, .72);
        padding: 1rem 1.1rem;
        margin-bottom: .85rem;
    }
    .panel-title {
        color: var(--ink);
        font-size: .98rem;
        font-weight: 720;
        margin-bottom: .55rem;
    }
    .panel-copy { color: var(--muted); line-height: 1.42; }
    .metric-card {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(17, 26, 42, .82);
        padding: .85rem 1rem;
        min-height: 104px;
    }
    .metric-label { color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value {
        color: var(--ink);
        font-size: 1.38rem;
        font-weight: 760;
        letter-spacing: -.035em;
        margin-top: .28rem;
        white-space: nowrap;
    }
    .metric-note { color: var(--muted); font-size: .74rem; margin-top: .15rem; }
    .metric-high .metric-value { color: var(--red); }
    .metric-medium .metric-value { color: var(--amber); }
    .metric-low .metric-value { color: var(--green); }
    .tier-badge {
        display: inline-block;
        padding: .26rem .62rem;
        border-radius: 999px;
        color: #fff;
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .08em;
    }
    .score-track {
        height: 9px;
        border-radius: 999px;
        background: rgba(255,255,255,.09);
        overflow: hidden;
        margin: .45rem 0 .1rem 0;
    }
    .score-fill { height: 100%; border-radius: 999px; }
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: .35rem; }
    .muted { color: var(--muted); }
    .small { font-size: .78rem; }
    .tight { margin-top: -.45rem; }
    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: rgba(17, 26, 42, .82);
        padding: .8rem 1rem;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: .78rem; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .stButton > button { border-radius: 10px; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: .25rem; }
    .stTabs [data-baseweb="tab"] { padding: .55rem .85rem; }
    @media (max-width: 760px) {
        .hero-title { font-size: 1.35rem; }
        .flow-step { font-size: .7rem; }
    }
</style>
"""


def parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Predictive HDD maintenance dashboard")
    # results_deploy2026: the retrained model on 2026 data (31_deploy_2026.py);
    # results: the frozen model on the 2024-25 test period (paper evaluation)
    parser.add_argument("--results_dir", type=str, default="results_deploy2026")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Cached serving-data functions
# ---------------------------------------------------------------------------


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _source_signature(results: Path) -> str:
    """Fingerprint every artifact that can change agent-visible context."""
    paths = [
        results / "risk_tiers_test.csv",
        results / "hybrid_test_scores.csv",
        results / "mitigation_mapping.json",
        results / "tier_cutoffs.json",
        results.parent / "models" / "hybrid_config.json",
        # Must match build_dashboard_db.py's signature order exactly.
        results / "pa_urgency_test.csv"
        if (results / "pa_urgency_test.csv").exists()
        else results.parent / "results_fusion" / "pa_urgency_test.csv",
    ]
    parts: list[str] = []
    for path in paths:
        if not path.exists():
            parts.append("missing")
        else:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _table_names(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def get_source_connection(results_dir: str):
    """Return connection, prediction source, and optional hybrid source.

    The compact ``drive_status`` table is the only indexed source.  CSV is
    used only when no database exists at all; if a database is present but
    locked or incomplete, fail visibly instead of silently scanning a 6.8GB
    export on every interaction.  The three-value shape is retained for
    compatibility with the original dashboard.
    """
    results = Path(results_dir)
    db_path = results / "dashboard.duckdb"
    if db_path.exists():
        con = None
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            tables = _table_names(con)
            if "drive_status" in tables:
                return con, "drive_status", "drive_status"
            con.close()
        except Exception as exc:
            if con is not None:
                con.close()
            raise RuntimeError(
                f"Cannot open indexed dashboard data at {db_path}. Close any "
                "build process and retry; CSV fallback is disabled while a "
                "database exists."
            ) from exc
        raise RuntimeError(
            f"{db_path} has no drive_status serving table. Run "
            "build_dashboard_db.py --results_dir results once."
        )

    con = duckdb.connect()
    risk_source = f"read_csv_auto('{_sql_path(results / 'risk_tiers_test.csv')}')"
    hybrid_path = results / "hybrid_test_scores.csv"
    hybrid_source = (
        f"read_csv_auto('{_sql_path(hybrid_path)}')" if hybrid_path.exists() else None
    )
    return con, risk_source, hybrid_source


@st.cache_data(ttl=300, show_spinner=False)
def serving_status(results_dir: str, source_version: str | None = None) -> dict[str, Any]:
    results = Path(results_dir)
    if not results.exists():
        return {"ok": False, "indexed": False, "last_date": None, "count": None}
    con, source, _ = get_source_connection(results_dir)
    try:
        is_status = source == "drive_status"
        if is_status:
            row = con.execute(
                "SELECT MAX(date), COUNT(*) FROM drive_status"
            ).fetchone()
        else:
            # Do not run COUNT(DISTINCT serial_number) on a raw export at app
            # startup; max(date) is cheap and sufficient for the status card.
            row = con.execute(f"SELECT MAX(date) FROM {source}").fetchone()
        last_date = row[0] if row else None
        count = int(row[1]) if is_status and row and len(row) > 1 else None
        stale = None
        if is_status:
            try:
                meta = con.execute(
                    "SELECT source_version FROM dashboard_meta LIMIT 1"
                ).fetchone()
                expected_version = source_version or _source_signature(results)
                stale = None if meta is None else bool(meta[0] != expected_version)
            except Exception:
                # Older serving databases do not have metadata; they remain
                # usable, but the UI simply does not claim freshness.
                stale = None
        return {
            "ok": True,
            "indexed": is_status,
            "last_date": str(last_date)[:10] if last_date is not None else None,
            "count": count,
            "stale": stale,
        }
    finally:
        con.close()


@st.cache_data(ttl=600, show_spinner=False)
def load_small_tables(
    results_dir: str, source_version: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only the small, overview-level metrics.

    The mitigation mapping is intentionally not loaded here.  The agent loads
    its date-matched record lazily for the selected drive, so the dashboard
    does not spend startup time parsing a multi-megabyte JSON file.
    """
    del source_version  # cache-key only; invalidates when metric/config files change
    results = Path(results_dir)
    tier_path = results / "tier_accuracy_test.csv"
    tier_accuracy = (
        pd.read_csv(tier_path)
        if tier_path.exists()
        else pd.DataFrame(
            columns=[
                "tier",
                "total_drives",
                "cumulative_share_captured",
            ]
        )
    )
    metrics_path = results / "hybrid_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    lgbm_path = results / "lgbm_metrics.csv"
    lgbm_metrics = pd.read_csv(lgbm_path) if lgbm_path.exists() else pd.DataFrame()
    return tier_accuracy, metrics, lgbm_metrics


@st.cache_data(ttl=300, show_spinner=False)
def fleet_overview(results_dir: str, source_version: str = "") -> dict[str, int]:
    del source_version  # cache-key only; invalidates when serving inputs change
    con, source, _ = get_source_connection(results_dir)
    try:
        if source == "drive_status":
            result = con.execute(
                """
                SELECT risk_tier, COUNT(*) AS n
                FROM drive_status
                GROUP BY risk_tier
                """
            ).fetchdf()
        else:
            result = con.execute(
                f"""
                WITH latest AS (
                    SELECT serial_number, risk_tier,
                           ROW_NUMBER() OVER (
                               PARTITION BY serial_number ORDER BY date DESC
                           ) AS rn
                    FROM {source}
                )
                SELECT risk_tier, COUNT(*) AS n
                FROM latest
                WHERE rn = 1
                GROUP BY risk_tier
                """
            ).fetchdf()
    finally:
        con.close()
    counts = {str(row["risk_tier"]): int(row["n"]) for _, row in result.iterrows()}
    return {
        "total": sum(counts.values()),
        "HIGH": counts.get("HIGH", 0),
        "MEDIUM": counts.get("MEDIUM", 0),
        "LOW": counts.get("LOW", 0),
    }


@st.cache_data(ttl=300, show_spinner=False)
def at_risk_drive_list(results_dir: str, source_version: str = "") -> list[str]:
    del source_version  # cache-key only; invalidates when serving inputs change
    con, source, _ = get_source_connection(results_dir)
    try:
        if source == "drive_status":
            result = con.execute(
                """
                SELECT serial_number
                FROM drive_status
                WHERE risk_tier IN ('HIGH', 'MEDIUM')
                ORDER BY serial_number
                """
            ).fetchdf()
        else:
            result = con.execute(
                f"""
                WITH latest AS (
                    SELECT serial_number, risk_tier,
                           ROW_NUMBER() OVER (
                               PARTITION BY serial_number ORDER BY date DESC
                           ) AS rn
                    FROM {source}
                )
                SELECT serial_number
                FROM latest
                WHERE rn = 1 AND risk_tier IN ('HIGH', 'MEDIUM')
                ORDER BY serial_number
                """
            ).fetchdf()
    finally:
        con.close()
    return result["serial_number"].astype(str).tolist()


@st.cache_data(ttl=180, show_spinner=False)
def get_flagged_drives_preview(
    results_dir: str, limit: int = 100, source_version: str = ""
) -> pd.DataFrame:
    del source_version  # cache-key only; invalidates when serving inputs change
    con, source, _ = get_source_connection(results_dir)
    try:
        if source == "drive_status":
            result = con.execute(
                f"""
                SELECT serial_number,
                       date,
                       hybrid_score AS risk_score,
                       risk_tier,
                       fleet_urgency_pct,
                       CAST(date_diff('day', date, (SELECT MAX(date) FROM drive_status)) AS INTEGER)
                           AS observation_age_days
                FROM drive_status
                WHERE risk_tier IN ('HIGH', 'MEDIUM')
                ORDER BY risk_score DESC
                LIMIT {int(limit)}
                """
            ).fetchdf()
        else:
            result = con.execute(
                f"""
                WITH bounds AS (
                    SELECT MAX(date) AS latest_date FROM {source}
                ),
                latest AS (
                    SELECT serial_number, date, hybrid_score, risk_tier,
                           ROW_NUMBER() OVER (
                               PARTITION BY serial_number ORDER BY date DESC
                           ) AS rn
                    FROM {source}
                )
                SELECT l.serial_number,
                       l.date,
                       l.hybrid_score AS risk_score,
                       l.risk_tier,
                       CAST(NULL AS DOUBLE) AS fleet_urgency_pct,
                       CAST(date_diff('day', l.date, b.latest_date) AS INTEGER)
                           AS observation_age_days
                FROM latest l CROSS JOIN bounds b
                WHERE l.rn = 1 AND l.risk_tier IN ('HIGH', 'MEDIUM')
                ORDER BY l.hybrid_score DESC
                LIMIT {int(limit)}
                """
            ).fetchdf()
    finally:
        con.close()
    return result


@st.cache_data(ttl=180, show_spinner=False)
def get_drive_context_cached(
    results_dir: str,
    serial_number: str,
    source_version: str = "",
) -> dict[str, Any]:
    del source_version  # included in the cache key for source-change invalidation
    return _AGENT.build_drive_context(Path(results_dir), serial_number)


@st.cache_data(ttl=180, show_spinner=False)
def get_drive_detail(results_dir: str, serial_number: str) -> dict[str, Any] | None:
    """Return one drive's latest ML row using the compact serving table."""
    con, source, hybrid_source = get_source_connection(results_dir)
    try:
        if source == "drive_status":
            row = con.execute(
                """
                SELECT *
                FROM drive_status
                WHERE serial_number = ?
                LIMIT 1
                """,
                [str(serial_number)],
            ).fetchdf()
            if row.empty:
                return None
            record = row.iloc[0].to_dict()
            record.pop("target", None)
            return _json_safe_record(record)  # includes fleet_urgency_pct

        row = con.execute(
            f"""
            SELECT serial_number, date, hybrid_score, risk_tier
            FROM {source}
            WHERE serial_number = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            [str(serial_number)],
        ).fetchdf()
        if row.empty:
            return None
        record = row.iloc[0].to_dict()
        return _json_safe_record(record)
    finally:
        con.close()


@st.cache_data(ttl=300, show_spinner=False)
def run_simulation_cached(
    results_dir: str,
    serial_number: str,
    prediction_version: str = "",
) -> dict[str, Any]:
    context = _AGENT.build_drive_context(Path(results_dir), serial_number)
    return _AGENT.run_agent(context, dry_run=True)


def run_agent_cached(
    results_dir: str,
    serial_number: str,
    dry_run: bool,
    model: str | None,
    prediction_version: str = "",
) -> dict[str, Any]:
    """Run an action request.

    Simulation is cached by Streamlit. Live results stay in the current
    session only, so a paid API response is never persisted in a shared
    on-disk cache or silently reused for a different prediction.
    """
    context = _AGENT.build_drive_context(Path(results_dir), serial_number)
    return _AGENT.run_agent(context, dry_run=dry_run, model=model)


def _json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            safe[key] = None
        elif hasattr(value, "isoformat"):
            safe[key] = value.isoformat()[:10]
        elif isinstance(value, (int, float, str, bool)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def tier_badge_html(tier: str) -> str:
    normalized = str(tier or "UNKNOWN").upper()
    color = TIER_COLORS.get(normalized, "#8391a8")
    label = TIER_ICONS.get(normalized, normalized[:3])
    return f'<span class="tier-badge" style="background:{color}">{esc(label)}</span>'


def metric_card(label: str, value: str, note: str = "", tone: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card {esc(tone)}">
          <div class="metric-label">{esc(label)}</div>
          <div class="metric-value">{esc(value)}</div>
          <div class="metric-note">{esc(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, body_html: str) -> None:
    """Render one self-contained HTML panel.

    Keeping the opening and closing tags in the same Streamlit block avoids
    the empty rounded rectangle produced when Streamlit splits separate
    markdown calls.
    """
    st.markdown(
        f'<div class="panel"><div class="panel-title">{esc(title)}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def score_bar(score: float, tier: str, strategy: str = "hybrid") -> None:
    value = max(0.0, min(1.0, float(score)))
    color = TIER_COLORS.get(tier, "#6ea8fe")
    # Name the model that actually produced the score. Stage 8 may have
    # selected lgbm_only, in which case calling it a "hybrid score" is wrong.
    label = "LightGBM risk score" if strategy == "lgbm_only" else "ML hybrid score"
    st.markdown(
        f"""
        <div class="score-track"><div class="score-fill" style="width:{value * 100:.1f}%;background:{color}"></div></div>
        <div class="muted small">{esc(label)} · {value:.4f} (0–1 model scale)</div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">Predictive maintenance control room</div>
          <div class="hero-copy">ML models estimate failure risk. The action agent decides which operational checks and response are appropriate. A human approves every maintenance action.</div>
          <div class="flow">
            <span class="flow-step active">ML prediction</span><span class="flow-arrow">→</span>
            <span class="flow-step">Agent tools</span><span class="flow-arrow">→</span>
            <span class="flow-step">Operator approval</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_drive_picker(prefix: str, options: list[str]) -> str | None:
    if not options:
        return None
    st.markdown('<div class="kicker">Drive selection</div>', unsafe_allow_html=True)
    search = st.text_input(
        "Search serial number",
        value="",
        placeholder="Type part of a serial number",
        key=f"{prefix}_search",
        label_visibility="collapsed",
    )
    query = search.strip().lower()
    matches = [serial for serial in options if not query or query in serial.lower()]
    # Streamlit selectboxes become slow when given tens of thousands of
    # options.  Search narrows the list; this cap keeps the widget responsive.
    filtered = matches[:500]
    if not filtered:
        st.info("No serial number matches that search. Clear the search to restore the queue.")
        return None

    # One shared widget key keeps Explorer -> Agent navigation on the same
    # drive.  Normalize the persisted value before constructing the widget so
    # Streamlit never has to resolve an out-of-options session value.
    select_key = "drive_select"
    canonical = st.session_state.get("selected_drive")
    persisted = st.session_state.get(select_key)
    current = canonical or persisted
    if current and not query and current not in filtered:
        # Keep a cross-page selection visible even when it is outside the
        # alphabetical first-500 window.
        filtered = [current] + filtered[:499]
    if current and current in filtered:
        st.session_state[select_key] = current
    else:
        st.session_state.pop(select_key, None)
        if query and current:
            st.session_state.pop("selected_drive", None)

    selected = st.selectbox(
        "Select a drive",
        filtered,
        key=select_key,
        label_visibility="collapsed",
    )
    st.session_state["selected_drive"] = selected
    truncated = len(matches) > len(filtered)
    suffix = f" · showing the first {len(filtered):,}" if truncated else ""
    st.caption(
        f"{len(matches):,} matching at-risk drive{'s' if len(matches) != 1 else ''}{suffix}"
    )
    return selected


def render_prediction(context: dict[str, Any], compact: bool = False) -> None:
    prediction = context.get("prediction", {})
    tier = str(prediction.get("risk_tier") or context.get("risk_tier") or "UNKNOWN").upper()
    score = float(prediction.get("risk_score", context.get("hybrid_score", 0.0)))
    urgency = prediction.get("fleet_urgency_pct")
    as_of = prediction.get("as_of", context.get("date"))
    observation_age = prediction.get("observation_age_days")
    top_driver = prediction.get("top_driver_feature") or "Not available"
    direction = prediction.get("top_driver_direction") or "Not available"
    tier_source = prediction.get("tier_from_model") or "hybrid"

    st.markdown('<div class="kicker">ML prediction · read only</div>', unsafe_allow_html=True)
    st.markdown(tier_badge_html(tier), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("LightGBM risk score" if tier_source == "lgbm_only" else "Hybrid score",
              f"{score:.4f}")
    c2.metric("Calibrated tier", tier.title())
    c3.metric(
        "Fleet urgency",
        "—" if urgency is None else f"sooner than {float(urgency):.0f}%",
        help=(
            "Share of the fleet the survival model expects to outlive this drive "
            "(test C-index 0.75). A ranking for ordering work, not a day count."
        ),
    )
    score_bar(score, tier, tier_source)
    cutoffs = prediction.get("tier_cutoffs") or {}
    cutoff_text = ""
    if cutoffs.get("high") is not None and cutoffs.get("medium") is not None:
        cutoff_text = (
            f" · HIGH ≥ {float(cutoffs['high']):.3f} · MEDIUM ≥ {float(cutoffs['medium']):.3f}"
        )
    age_text = (
        f" ({int(observation_age)} days before dataset end)"
        if observation_age is not None and int(observation_age) > 0
        else ""
    )
    st.caption(
        f"{esc(context.get('drive_model', 'UNKNOWN'))} · last observed {esc(as_of or 'unknown')}{esc(age_text)} · "
        f"tier from {esc(tier_source)} · top LightGBM driver: {esc(top_driver)} ({esc(direction)}){esc(cutoff_text)}"
    )
    if observation_age is not None and int(observation_age) > 30:
        st.caption("This is a last-known retrospective assessment, not live telemetry.")

    if not compact:
        panel(
            "Model explanation",
            f'<div class="panel-copy">{esc(context.get("root_cause", "No explanation available."))}</div>'
            f'<div class="panel-copy" style="margin-top:.6rem"><strong>Model mapping:</strong> {esc(context.get("recommended_action", "Continue standard monitoring."))}</div>',
        )


def model_description(results_dir: str) -> str:
    """Name the scoring model; a deployment dir records it in deployment.json."""
    path = Path(results_dir) / "deployment.json"
    if not path.exists():
        return "LightGBM (5-seed ensemble)"
    info = json.loads(path.read_text())
    return f"{info['model']}, trained to {info['train_window'][1]}, scoring {' to '.join(info['scored_window'])}"


def render_overview(
    results_dir: str,
    overview: dict[str, int],
    tier_accuracy: pd.DataFrame,
    metrics: pd.DataFrame,
    lgbm_metrics: pd.DataFrame,
    status: dict[str, Any],
    source_version: str = "",
) -> None:
    st.markdown('<div class="kicker">Fleet overview</div>', unsafe_allow_html=True)
    st.markdown("## Latest-known fleet position")
    st.markdown(
        '<div class="page-subtitle">The queue is based on each physical drive\'s most recent ML assessment in the held-out results.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        metric_card("Monitored drives", f"{overview['total']:,}", "latest row per drive")
    with cols[1]:
        metric_card("HIGH", f"{overview['HIGH']:,}", "priority queue", "metric-high")
    with cols[2]:
        metric_card("MEDIUM", f"{overview['MEDIUM']:,}", "watch closely", "metric-medium")
    with cols[3]:
        metric_card("LOW", f"{overview['LOW']:,}", "standard monitoring", "metric-low")

    st.markdown("### Operational queue")
    st.caption(
        "Highest latest-known risk scores (6 decimals). Fleet urgency orders work "
        "within a tier and never changes it. Select a drive in Drive Explorer "
        "for the full prediction."
    )
    queue = get_flagged_drives_preview(results_dir, limit=50, source_version=source_version)
    if queue.empty:
        st.success("No HIGH or MEDIUM drives in the current queue.")
    else:
        display = queue.rename(
            columns={
                "serial_number": "Drive",
                "date": "As of",
                "risk_score": "Risk score",
                "risk_tier": "Tier",
                "fleet_urgency_pct": "Fleet urgency (%)",
                "observation_age_days": "Age (days)",
            }
        )
        display["As of"] = pd.to_datetime(display["As of"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "Risk score": st.column_config.NumberColumn(format="%.6f"),
                "Fleet urgency (%)": st.column_config.NumberColumn(
                    format="%.0f",
                    help="Share of the fleet expected to outlive this drive; not a day count.",
                ),
                "Age (days)": st.column_config.NumberColumn(format="%d"),
            },
        )
        max_age = pd.to_numeric(display["Age (days)"], errors="coerce").max()
        if pd.notna(max_age) and float(max_age) > 30:
            st.caption(
                "Age is relative to the latest assessment date; older rows are "
                "last-known retrospective results, not live telemetry."
            )

    st.markdown("### Model health")
    health_cols = st.columns(3, gap="medium")
    if not metrics.empty and str(metrics.iloc[0].get("chosen_strategy")) == "lgbm_only":
        # Nothing fused is shipped, so no card is labelled "hybrid".
        row = metrics.iloc[0]
        with health_cols[0]:
            metric_card("LightGBM AUPRC", f"{float(row['test_auprc']):.3f}", "held-out test")
        with health_cols[1]:
            metric_card("LightGBM ROC-AUC", f"{float(row['test_roc_auc']):.3f}", "held-out test")
        with health_cols[2]:
            if not tier_accuracy.empty and "cumulative_share_captured" in tier_accuracy:
                high = tier_accuracy.loc[tier_accuracy["tier"] == "HIGH"]
                recall = (
                    float(high.iloc[0]["cumulative_share_captured"])
                    if not high.empty
                    else float("nan")
                )
                metric_card("HIGH-tier capture", f"{recall:.1%}", "held-out test")
        st.caption(
            f"Predictor: {model_description(results_dir)}. More complex combinations "
            "were evaluated and did not beat it (archive/fusion_study/)."
        )
        if not tier_accuracy.empty and "cumulative_share_captured" in tier_accuracy:
            st.caption("Cutoffs were calibrated on validation; the capture shown here is held-out test.")
    elif not metrics.empty:
        row = metrics.iloc[0]
        strategy = str(row.get("chosen_strategy", "hybrid"))
        hybrid_auprc = float(row.get("test_auprc", float("nan")))
        with health_cols[0]:
            metric_card("Hybrid AUPRC", f"{hybrid_auprc:.3f}", "held-out test")
        if not lgbm_metrics.empty:
            lgbm_auprc = float(lgbm_metrics.iloc[0].get("auprc", float("nan")))
            with health_cols[1]:
                metric_card("LightGBM AUPRC", f"{lgbm_auprc:.3f}", "held-out test")
            with health_cols[2]:
                if not tier_accuracy.empty and "cumulative_share_captured" in tier_accuracy:
                    high = tier_accuracy.loc[tier_accuracy["tier"] == "HIGH"]
                    recall = (
                        float(high.iloc[0]["cumulative_share_captured"])
                        if not high.empty
                        else float("nan")
                    )
                    metric_card("HIGH-tier capture", f"{recall:.1%}", "held-out test")
                else:
                    metric_card("Fusion", strategy.title(), "selected strategy")
        else:
            with health_cols[1]:
                metric_card(
                    "Hybrid ROC-AUC",
                    f"{float(row.get('test_roc_auc', float('nan'))):.3f}",
                    "held-out test",
                )
        if not lgbm_metrics.empty:
            delta = hybrid_auprc - lgbm_auprc
            st.caption(
                f"Hybrid − LightGBM AUPRC: {delta:+.3f} · fusion strategy: {strategy.title()}"
            )
        else:
            st.caption(f"Selected fusion strategy: {strategy.title()}")
        if not tier_accuracy.empty and "cumulative_share_captured" in tier_accuracy:
            st.caption("Cutoffs were calibrated on validation; the capture shown here is held-out test.")
    else:
        st.info("Model metrics are not available yet.")

    source_text = (
        "Indexed drive-status table"
        if status.get("indexed")
        else "CSV fallback (build the serving DB for speed)"
    )
    last_text = status.get("last_date") or "unknown"
    panel(
        "Serving data",
        f'<div class="panel-copy">{esc(source_text)}<br>Latest assessment: {esc(last_text)}</div>',
    )

    st.caption(
        "Definitions: HIGH and MEDIUM counts are latest-known physical-drive statuses, not drive-day rows. "
        "Model metrics are held-out evaluation metrics and do not change the current queue."
    )


def render_explorer(
    results_dir: str, options: list[str], source_version: str = ""
) -> None:
    st.markdown('<div class="kicker">Drive explorer</div>', unsafe_allow_html=True)
    st.markdown("## Inspect one prediction")
    st.markdown(
        '<div class="page-subtitle">Search is local to the current at-risk queue; selecting a drive reuses the cached serving lookup.</div>',
        unsafe_allow_html=True,
    )
    selected = render_drive_picker("explorer", options)
    if not selected:
        return
    try:
        with st.spinner("Loading the latest prediction...", show_time=True):
            context = get_drive_context_cached(
                results_dir, selected, source_version
            )
    except Exception as exc:
        st.error(f"Could not load drive {selected}: {exc}")
        return

    render_prediction(context)
    st.caption(
        "The explanation is LightGBM SHAP guidance; the action agent evaluates it "
        "together with operational constraints."
    )


def render_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.caption("The agent returned without calling a tool.")
        return
    for step in trace:
        tool = step.get("tool", "tool")
        with st.expander(f"{step.get('turn', '')} · {tool}"):
            st.json({"input": step.get("input", {}), "output": step.get("output", {})})


def render_agent_workspace(
    results_dir: str, options: list[str], source_version: str = ""
) -> None:
    st.markdown('<div class="kicker">Agent workspace</div>', unsafe_allow_html=True)
    st.markdown("## Turn a prediction into a next step")
    st.markdown(
        '<div class="page-subtitle">The agent receives the frozen ML output, chooses operational tools, and drafts a human-reviewable action. It does not change the prediction.</div>',
        unsafe_allow_html=True,
    )
    selected = render_drive_picker("agent", options)
    if not selected:
        return
    try:
        with st.spinner("Loading prediction...", show_time=True):
            context = get_drive_context_cached(
                results_dir, selected, source_version
            )
    except Exception as exc:
        st.error(f"Could not load drive {selected}: {exc}")
        return

    render_prediction(context, compact=True)
    panel(
        "ML explanation and rule hint",
        f'<div class="panel-copy">{esc(context.get("root_cause", "No explanation available."))}</div>'
        f'<div class="panel-copy" style="margin-top:.6rem"><strong>Rule hint:</strong> {esc(context.get("recommended_action", "Continue standard monitoring."))}</div>',
    )
    with st.container(border=True):
        st.markdown('<div class="panel-title">Agent controls</div>', unsafe_allow_html=True)
        has_key = _AGENT.live_provider() is not None
        mode_label = st.radio(
            "Decision mode",
            ["Simulation", "Live agent"],
            horizontal=True,
            key="agent_mode",
            help="Simulation is deterministic and has no API cost. Live mode requires "
            "ANTHROPIC_API_KEY or GROQ_API_KEY (free tier).",
        )
        if mode_label == "Live agent" and not has_key:
            st.warning("No ANTHROPIC_API_KEY or GROQ_API_KEY is configured; run in Simulation mode.")
        if mode_label == "Live agent" and has_key:
            st.caption(
                f"Configured live model: {_AGENT.live_provider()} / {_AGENT.live_model()}"
            )
        st.caption("Inventory, scheduling, and ticket tools are simulated. No external action is executed.")

    dry_run = mode_label == "Simulation"
    model = _AGENT.live_model() if not dry_run and has_key else None
    prediction_version = (
        f"{context.get('prediction', {}).get('as_of')}:"
        f"{float(context.get('prediction', {}).get('risk_score', 0.0)):.6f}"
    )
    cache_version = f"{source_version}|{prediction_version}"
    if st.button("Run action agent", type="primary", width="stretch"):
        if not dry_run and not has_key:
            st.error("Live mode is unavailable until ANTHROPIC_API_KEY is set.")
        else:
            result_key = (selected, dry_run, model, cache_version)
            try:
                with st.spinner("Agent is selecting operational tools...", show_time=True):
                    if dry_run:
                        result = run_simulation_cached(
                            results_dir, selected, cache_version
                        )
                    else:
                        result = run_agent_cached(
                            results_dir, selected, dry_run, model, cache_version
                        )
                st.session_state["agent_result_key"] = result_key
                st.session_state["agent_result"] = result
            except Exception as exc:
                st.session_state.pop("agent_result", None)
                st.session_state.pop("agent_result_key", None)
                st.error(f"Agent run failed: {exc}")

    result_key = st.session_state.get("agent_result_key")
    result = st.session_state.get("agent_result")
    if not result or result_key != (selected, dry_run, model, cache_version):
        st.info("Run the agent to see the action decision and tool trace for this drive.")
        return

    decision = result.get("decision", {})
    if result.get("status") != "complete":
        st.warning(f"Agent run ended with status: {result.get('status', 'unknown')}.")
    st.markdown('<div class="kicker">Agent action result</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("Priority", str(decision.get("priority", "—")).title())
    a2.metric("Mode", str(result.get("mode", "—")).title())
    a3.metric("Tools used", str(len(result.get("tool_trace", []))))
    st.caption(
        f"Decision source: {esc(result.get('decision_source', 'unknown'))} · "
        "the ML prediction above remains unchanged."
    )
    panel(
        "Recommended response",
        f'<div class="panel-copy">{esc(result.get("summary", "No summary available."))}</div>',
    )

    if result.get("ticket"):
        st.markdown("### Draft maintenance ticket")
        st.dataframe(
            pd.DataFrame([result["ticket"]]),
            hide_index=True,
            width="stretch",
        )
    st.markdown("### Tool trace")
    render_trace(result.get("tool_trace", []))
    st.caption("Operator approval is required before any ticket, replacement, or scheduling action is executed.")


# ---------------------------------------------------------------------------
# Application shell
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    st.set_page_config(
        page_title="HDD Risk Control Room",
        page_icon="💽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        st.error(f"Results directory '{results_dir}' was not found. Run Stages 6-12 first.")
        st.stop()
    if not (results_dir / "dashboard.duckdb").exists():
        st.error(
            "The indexed dashboard data is missing. Run "
            "`python build_dashboard_db.py --results_dir results` once, then reload."
        )
        st.stop()

    try:
        source_version = _source_signature(results_dir)
        status = serving_status(args.results_dir, source_version)
        overview = fleet_overview(args.results_dir, source_version)
        tier_accuracy, metrics, lgbm_metrics = load_small_tables(
            args.results_dir, source_version
        )
        at_risk = at_risk_drive_list(args.results_dir, source_version)
    except Exception as exc:
        st.error(f"Serving data could not be loaded: {exc}")
        st.stop()

    with st.sidebar:
        st.markdown('<div class="brand">DISK<span>GUARD</span></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-sub">Predictive maintenance<br>control room</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Workspace",
            ["Fleet overview", "Drive explorer", "Agent workspace"],
            key="workspace",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("#### System status")
        if status.get("indexed"):
            st.markdown('<span class="status-dot" style="background:#43c59e"></span>Indexed serving data', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot" style="background:#f3b562"></span>CSV fallback', unsafe_allow_html=True)
        st.caption(f"Latest assessment: {status.get('last_date') or 'unknown'}")
        st.caption(f"At-risk queue: {len(at_risk):,} drives")
        if status.get("stale") is True:
            st.warning("Source exports changed; rebuild the serving DB.", icon="⚠️")
        st.divider()
        st.markdown("#### Decision boundary")
        st.caption("ML supplies the prediction. The agent chooses the response. A human approves execution.")
        if not status.get("indexed"):
            st.info("Run build_dashboard_db.py once for fast indexed lookups.", icon="⚡")

    render_hero()

    if page == "Fleet overview":
        render_overview(
            args.results_dir,
            overview,
            tier_accuracy,
            metrics,
            lgbm_metrics,
            status,
            source_version,
        )
    elif page == "Drive explorer":
        render_explorer(args.results_dir, at_risk, source_version)
    else:
        render_agent_workspace(args.results_dir, at_risk, source_version)

    st.divider()
    st.caption(
        "ML prediction → action agent → operator approval · Source tables are queried on demand; "
        "raw drive-day exports are never loaded into the browser."
    )


if __name__ == "__main__":
    main()
