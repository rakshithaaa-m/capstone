"""
Build the small, indexed data layer used by the Streamlit dashboard.

The raw prediction exports are intentionally kept as CSV files because they
are useful pipeline artefacts.  They are not, however, a good serving data
source: ``risk_tiers_test.csv`` and ``hybrid_test_scores.csv`` contain roughly
145 million drive-day rows, so a lookup for one serial number can scan several
gigabytes.  This script builds a persistent DuckDB serving database once and
adds compact tables for the dashboard:

* ``drive_status``  - one latest row per physical drive, including the ML
  LightGBM score, tier, and P-A fleet urgency (Stage 23).
* ``drive_models``  - one drive model per serial number (when the test feature
  parquet is available).
* ``mitigation``    - the small SHAP-to-action mapping, indexed by serial/date.

The large source tables are not required after the serving tables exist.  By
default the builder keeps only the compact serving tables and drops the copied
risk/hybrid tables after validation.  Pass ``--keep-source-tables`` when a
local offline rebuild is more important than disk space, and pass
``--rebuild-sources`` after a new Stage 8/9/10 run.

Usage::

    python build_dashboard_db.py --results_dir results

For a completely fresh source load::

    python build_dashboard_db.py --results_dir results --rebuild-sources

To rebuild only the compact serving tables from native source tables already
inside the serving database (if you kept them with ``--keep-source-tables``)::

    python build_dashboard_db.py --results_dir results --refresh-serving
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb


RISK_TIERS = "risk_tiers"
HYBRID_SCORES = "hybrid_scores"
DRIVE_STATUS = "drive_status"
DRIVE_MODELS = "drive_models"
MITIGATION = "mitigation"


def sql_path(path: Path) -> str:
    """Return a path safe to embed in a DuckDB SQL string literal."""
    return path.resolve().as_posix().replace("'", "''")


def table_names(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def timed(label: str, started: float) -> None:
    print(f"[BUILD] {label}: {time.perf_counter() - started:.1f}s")


def source_signature(*paths: Path | None) -> str:
    """Return a cheap fingerprint of the source exports used by the build."""
    parts: list[str] = []
    for path in paths:
        if path is None or not path.exists():
            parts.append("missing")
        else:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def build_serving_tables(
    con: duckdb.DuckDBPyConnection,
    risk_source: str,
    hybrid_source: str | None,
    models_available: bool,
    urgency_source: str | None = None,
) -> None:
    """Build the compact latest-per-drive table from table or CSV sources."""
    # One row per serial number is all the current dashboard needs.  arg_max
    # is considerably cheaper than joining the 145M-row hybrid export back to
    # every historical row before reducing it to the latest reading.  The
    # latest hybrid date is joined back to the risk date so a missing/misaligned
    # source row becomes NULL rather than silently attaching the wrong score.
    if hybrid_source:
        hybrid_select = f"""
            SELECT serial_number,
                   arg_max(lgbm_prob, date) AS lgbm_prob,
                   arg_max(date, date) AS latest_date
            FROM {hybrid_source}
            GROUP BY serial_number
        """
        lgbm_expr = "h.lgbm_prob"
        hybrid_join = (
            f"LEFT JOIN ({hybrid_select}) h "
            "ON h.serial_number = l.serial_number AND h.latest_date = l.date"
        )
    else:
        lgbm_expr = "CAST(NULL AS DOUBLE)"
        hybrid_join = ""

    if models_available:
        model_expr = "m.drive_model"
        model_join = "LEFT JOIN drive_models m ON m.serial_number = l.serial_number"
    else:
        model_expr = "CAST(NULL AS VARCHAR)"
        model_join = ""

    # Fleet urgency (Stage 23) is computed on each drive's latest test date;
    # joining on the date too leaves it NULL rather than misattributed if the
    # two exports ever disagree about which row is latest.
    if urgency_source:
        urgency_expr = "u.fleet_urgency_pct"
        urgency_join = (
            f"LEFT JOIN {urgency_source} u "
            "ON u.serial_number = l.serial_number AND CAST(u.date AS DATE) = CAST(l.date AS DATE)"
        )
    else:
        urgency_expr = "CAST(NULL AS DOUBLE)"
        urgency_join = ""

    started = time.perf_counter()
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {DRIVE_STATUS} AS
            WITH latest AS (
                SELECT serial_number, date, hybrid_score, risk_tier
                FROM {risk_source}
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY serial_number ORDER BY date DESC
                ) = 1
            )
            SELECT l.serial_number,
                   l.date,
                   l.hybrid_score,
                   l.risk_tier,
                   {lgbm_expr} AS lgbm_prob,
                   {urgency_expr} AS fleet_urgency_pct,
                   {model_expr} AS drive_model
            FROM latest l
            {hybrid_join}
            {urgency_join}
            {model_join}
            """
        )
        row_count, distinct_serials, invalid_rows = con.execute(
            f"""
            SELECT COUNT(*),
                   COUNT(DISTINCT serial_number),
                   COUNT(*) FILTER (
                       WHERE serial_number IS NULL OR risk_tier IS NULL
                   )
            FROM {DRIVE_STATUS}
            """
        ).fetchone()
        if row_count != distinct_serials or invalid_rows:
            raise RuntimeError(
                f"Invalid {DRIVE_STATUS} table: rows={row_count}, "
                f"distinct_serials={distinct_serials}, invalid_rows={invalid_rows}"
            )
        timed(f"built {DRIVE_STATUS}", started)

        started = time.perf_counter()
        con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{DRIVE_STATUS}_serial "
            f"ON {DRIVE_STATUS} (serial_number)"
        )
        con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{DRIVE_STATUS}_tier_score "
            f"ON {DRIVE_STATUS} (risk_tier, hybrid_score)"
        )
        timed("indexed drive_status", started)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=str, default="results_deploy2026")
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Optional features_test.parquet path used to populate drive_models",
    )
    parser.add_argument(
        "--rebuild-sources",
        action="store_true",
        help="Reload the large risk/hybrid CSV exports before rebuilding serving tables",
    )
    parser.add_argument(
        "--refresh-serving",
        action="store_true",
        help="Rebuild compact tables; reuse native source tables when retained, otherwise read CSVs",
    )
    parser.add_argument(
        "--keep-source-tables",
        action="store_true",
        help="Keep the large risk/hybrid tables in dashboard.duckdb for faster offline rebuilds",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not scan features_test.parquet for serial-to-model metadata",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    risk_path = results_dir / "risk_tiers_test.csv"
    hybrid_path = results_dir / "hybrid_test_scores.csv"
    mitigation_path = results_dir / "mitigation_mapping.json"
    cutoffs_path = results_dir / "tier_cutoffs.json"
    hybrid_config_path = results_dir.parent / "models" / "hybrid_config.json"
    # the deployment results dir (31_deploy_2026.py) carries its own urgency file
    urgency_path = results_dir / "pa_urgency_test.csv"
    if not risk_path.exists():
        raise FileNotFoundError(
            f"Missing {risk_path}. Run Stage 9 (risk-tier calibration) first."
        )

    features_path = None
    if not args.skip_models:
        features_path = (
            Path(args.features_path).resolve()
            if args.features_path
            else results_dir.parent / "data" / "processed" / "features_test.parquet"
        )
        if not features_path.exists():
            features_path = None

    db_path = results_dir / "dashboard.duckdb"
    print(f"[BUILD] dashboard database: {db_path}")
    con = duckdb.connect(str(db_path))
    try:
        tables = table_names(con)
        previous_version = None
        if "dashboard_meta" in tables:
            try:
                previous_version = con.execute(
                    "SELECT source_version FROM dashboard_meta LIMIT 1"
                ).fetchone()[0]
            except Exception:
                previous_version = None
        current_version = source_signature(
            risk_path,
            hybrid_path,
            mitigation_path,
            cutoffs_path,
            hybrid_config_path,
            urgency_path,
        )
        source_changed = bool(
            previous_version and current_version != previous_version
        )
        if source_changed and not args.rebuild_sources:
            if args.keep_source_tables:
                print(
                    "[BUILD] warning: source exports changed; retained source tables "
                    "were reused. Re-run with --rebuild-sources to refresh predictions."
                )

        # Use existing native source tables when they are available.  On a
        # fresh compact build, read the CSVs directly in the serving query
        # instead of duplicating 20GB inside dashboard.duckdb.
        have_risk_table = RISK_TIERS in tables
        have_hybrid_table = HYBRID_SCORES in tables
        if args.rebuild_sources or (args.keep_source_tables and not have_risk_table):
            started = time.perf_counter()
            print(f"[BUILD] loading {risk_path.name} ...")
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {RISK_TIERS} AS
                SELECT * FROM read_csv_auto('{sql_path(risk_path)}')
                """
            )
            have_risk_table = True
            timed("loaded risk tiers", started)
        elif have_risk_table:
            print(f"[BUILD] reusing existing {RISK_TIERS} table")
        else:
            print(f"[BUILD] reading {risk_path.name} directly for compact build")

        if hybrid_path.exists():
            if args.rebuild_sources or (args.keep_source_tables and not have_hybrid_table):
                started = time.perf_counter()
                print(f"[BUILD] loading {hybrid_path.name} ...")
                con.execute(
                    f"""
                    CREATE OR REPLACE TABLE {HYBRID_SCORES} AS
                    SELECT * FROM read_csv_auto('{sql_path(hybrid_path)}')
                    """
                )
                have_hybrid_table = True
                timed("loaded hybrid scores", started)
            elif have_hybrid_table:
                print(f"[BUILD] reusing existing {HYBRID_SCORES} table")
            else:
                print(f"[BUILD] reading {hybrid_path.name} directly for compact build")
        else:
            if args.rebuild_sources and have_hybrid_table:
                con.execute(f"DROP TABLE {HYBRID_SCORES}")
                have_hybrid_table = False
                print(f"[BUILD] removed stale {HYBRID_SCORES} table")
            print(f"[BUILD] warning: {hybrid_path.name} not found; lgbm_prob will be unavailable")

        if source_changed and not args.rebuild_sources and not args.keep_source_tables:
            print(
                "[BUILD] source exports changed; rebuilding the compact layer "
                "directly from the current CSVs."
            )
            have_risk_table = False
            have_hybrid_table = False

        tables = table_names(con)
        risk_source = (
            RISK_TIERS
            if have_risk_table
            else f"read_csv_auto('{sql_path(risk_path)}')"
        )
        hybrid_source = (
            HYBRID_SCORES
            if have_hybrid_table and hybrid_path.exists()
            else (
                f"read_csv_auto('{sql_path(hybrid_path)}')"
                if hybrid_path.exists()
                else None
            )
        )
        had_drive_models = DRIVE_MODELS in tables
        models_rebuilt = False
        if features_path is not None and (
            args.rebuild_sources or not had_drive_models
        ):
            models_rebuilt = True
            started = time.perf_counter()
            print(f"[BUILD] scanning model metadata from {features_path.name} ...")
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {DRIVE_MODELS} AS
                SELECT serial_number, arg_max(model, date) AS drive_model
                FROM read_parquet('{sql_path(features_path)}')
                WHERE model IS NOT NULL
                GROUP BY serial_number
                """
            )
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{DRIVE_MODELS}_serial "
                f"ON {DRIVE_MODELS} (serial_number)"
            )
            timed("built drive_models", started)
        elif features_path is None:
            print("[BUILD] model metadata unavailable; using UNKNOWN in agent context")

        # Refresh the table catalogue after the optional model table creation.
        # ``models_rebuilt`` is captured before creation so a newly-populated
        # model table always triggers a compact-status rebuild.
        tables = table_names(con)
        needs_serving_rebuild = (
            args.rebuild_sources
            or args.refresh_serving
            or DRIVE_STATUS not in tables
            or (not have_risk_table and previous_version != current_version)
            or models_rebuilt
        )
        if needs_serving_rebuild:
            build_serving_tables(
                con,
                risk_source,
                hybrid_source,
                models_available=DRIVE_MODELS in table_names(con),
                urgency_source=(
                    f"read_csv_auto('{sql_path(urgency_path)}')"
                    if urgency_path.exists()
                    else None
                ),
            )
        else:
            print(f"[BUILD] reusing current {DRIVE_STATUS} table")

        tables = table_names(con)
        if mitigation_path.exists():
            if (
                args.rebuild_sources
                or MITIGATION not in tables
                or (source_changed and not have_risk_table)
            ):
                started = time.perf_counter()
                con.execute(
                    f"""
                    CREATE OR REPLACE TABLE {MITIGATION} AS
                    SELECT * FROM read_json_auto('{sql_path(mitigation_path)}')
                    """
                )
                timed("built mitigation table", started)
            # Always ensure the compact lookup index exists, including when the
            # table was reused from an older build.
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{MITIGATION}_serial_date "
                f"ON {MITIGATION} (serial_number, date)"
            )
        else:
            print(f"[BUILD] warning: {mitigation_path.name} not found")

        tables = table_names(con)
        mitigation_rows = (
            int(con.execute(f"SELECT COUNT(*) FROM {MITIGATION}").fetchone()[0])
            if MITIGATION in tables
            else 0
        )
        risk_rows = (
            int(con.execute(f"SELECT COUNT(*) FROM {RISK_TIERS}").fetchone()[0])
            if have_risk_table
            else -1
        )
        monitored_drives = int(
            con.execute(f"SELECT COUNT(*) FROM {DRIVE_STATUS}").fetchone()[0]
        )
        # If source files changed but the user intentionally reused the
        # existing tables, preserve the old fingerprint so the dashboard can
        # surface the stale serving layer instead of claiming it is fresh.
        version = (
            current_version
            if (
                args.rebuild_sources
                or previous_version is None
                or (source_changed and not args.keep_source_tables)
            )
            else previous_version
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE dashboard_meta AS
            SELECT ? AS source_version,
                   ?::BIGINT AS risk_rows,
                   ?::BIGINT AS monitored_drives,
                   ?::BIGINT AS mitigation_rows,
                   CAST(CURRENT_TIMESTAMP AS TIMESTAMP) AS built_at
            """,
            [version, risk_rows, monitored_drives, mitigation_rows],
        )

        if not args.keep_source_tables:
            # The CSVs remain the source-of-truth artefacts.  Keeping copied
            # 145M-row tables in the serving database wastes tens of gigabytes
            # and is not needed by the dashboard or agent.
            dropped_source = False
            current_tables = table_names(con)
            for table in (HYBRID_SCORES, RISK_TIERS):
                if table in current_tables:
                    con.execute(f"DROP TABLE {table}")
                    dropped_source = True
            if dropped_source:
                try:
                    con.execute("VACUUM")
                    print("[BUILD] dropped large source tables and compacted dashboard.duckdb")
                except Exception as exc:
                    print(f"[BUILD] source tables dropped; VACUUM skipped: {exc}")

        print("[BUILD] serving tables are ready")
    finally:
        con.close()


if __name__ == "__main__":
    main()
