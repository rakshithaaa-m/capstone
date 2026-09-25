"""
Dashboard speed fix — build an indexed DuckDB database, once.

Every dashboard click currently triggers a full linear scan through
risk_tiers_test.csv and hybrid_test_scores.csv (~14GB each) via
read_csv_auto(), because plain CSVs have no index -- DuckDB has to read
the whole file to find one matching serial_number. This script loads
both files into a persistent .duckdb database file ONE TIME, as native
tables with an index on serial_number, so every future lookup is
near-instant regardless of file size.

Usage (run once, or whenever risk_tiers_test.csv / hybrid_test_scores.csv
change):
    python3 build_dashboard_db.py --results_dir results
"""
import argparse
from pathlib import Path

import duckdb

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results")
args = parser.parse_args()

results_dir = Path(args.results_dir)
db_path = results_dir / "dashboard.duckdb"

print(f"[BUILD] Creating indexed database at {db_path} ...")
con = duckdb.connect(str(db_path))

print("[BUILD] Loading risk_tiers_test.csv as a native table...")
con.execute(f"""
    CREATE OR REPLACE TABLE risk_tiers AS
    SELECT * FROM read_csv_auto('{(results_dir / "risk_tiers_test.csv").as_posix()}')
""")
con.execute("CREATE INDEX IF NOT EXISTS idx_risk_tiers_serial ON risk_tiers (serial_number)")

hybrid_path = results_dir / "hybrid_test_scores.csv"
if hybrid_path.exists():
    print("[BUILD] Loading hybrid_test_scores.csv as a native table...")
    con.execute(f"""
        CREATE OR REPLACE TABLE hybrid_scores AS
        SELECT * FROM read_csv_auto('{hybrid_path.as_posix()}')
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_hybrid_serial ON hybrid_scores (serial_number)")

con.close()
print(f"[BUILD] Done -- {db_path} is ready. Point the dashboard at this file instead of the raw CSVs.")
