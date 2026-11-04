"""MEDIUM-tier sizing study for the shipped model (results_tiers2026/PREREGISTRATION.md).

R80: MEDIUM cutoff from the 80%-recall rule (results_retrain2026, fitted on validation).
S9 : MEDIUM cutoff = score quantile at which HIGH + MEDIUM covers 9.3% of validation
     drive-days. HIGH is unchanged. Decided on validation; 2026 reported once.

    python 34_tier_sizing.py
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

OUT = Path("results_tiers2026")
DECISION = Path("results_retrain2026/results.json")
SCORES = {"val": "SELECT serial_number s, CAST(date AS DATE) d, p_A p, y "
                 "FROM read_parquet('data/processed/retrain2026/preds_val_A.parquet')",
          # deployment scores run to 2026-06-30; labels are complete to 06-16
          "test": "SELECT serial_number s, date d, p, y "
                  "FROM read_parquet('results_deploy2026/_work/scores/*.parquet')"}
LABEL_END = {"val": "2025-12-17", "test": "2026-06-16"}
AT_RISK_SHARE = 0.093        # 32,794 / 352,383: HIGH + MEDIUM share of the original design
MIN_REDUCTION, MAX_CATCH_DROP = 0.30, 0.05


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(Path.home() / 'duck_spool').as_posix()}'")
    return con


def measure(con, split: str, high: float, med: float) -> dict:
    lab = f"d <= DATE '{LABEL_END[split]}'"
    rows, at_risk_rows, pos, pos_in = con.execute(f"""
        SELECT count(*), sum(CASE WHEN p >= {med} THEN 1 ELSE 0 END),
               sum(CASE WHEN {lab} THEN y ELSE 0 END),
               sum(CASE WHEN {lab} AND p >= {med} THEN y ELSE 0 END)
        FROM {split}""").fetchone()
    med_rows, med_pos = con.execute(f"""
        SELECT count(*), sum(y) FROM {split} WHERE {lab} AND p >= {med} AND p < {high}""").fetchone()
    catch = con.execute(f"""
        SELECT avg(c) FROM (SELECT max(CASE WHEN y = 1 AND p >= {med} THEN 1.0 ELSE 0 END) c
                            FROM {split} WHERE {lab} GROUP BY s HAVING max(y) = 1)""").fetchone()[0]
    latest = con.execute(f"""
        SELECT sum(CASE WHEN p >= {high} THEN 1 ELSE 0 END), sum(CASE WHEN p >= {med} AND p < {high} THEN 1 ELSE 0 END),
               count(*)
        FROM {split} WHERE d = (SELECT max(d) FROM {split})""").fetchone()
    return {"medium_cutoff": med, "at_risk_row_share": at_risk_rows / rows,
            "positive_rows_captured": pos_in / pos, "failing_drive_capture": catch,
            "medium_row_precision": med_pos / med_rows,
            "latest_day_high": int(latest[0]), "latest_day_medium": int(latest[1]),
            "latest_day_drives": int(latest[2])}


def main() -> None:
    cut = json.loads(DECISION.read_text())["fitted_on_validation"]["A"]["cut"]
    high, med_r80 = cut["high_cutoff"], cut["medium_cutoff"]
    con = _duck()
    for split, sql in SCORES.items():
        con.execute(f"CREATE TEMP TABLE {split} AS {sql}")
    med_s9 = con.execute(f"SELECT quantile_cont(p, {1 - AT_RISK_SHARE}) FROM val").fetchone()[0]

    res = {"preregistration": str(OUT / "PREREGISTRATION.md"), "high_cutoff": high,
           "at_risk_share_target": AT_RISK_SHARE}
    for split in ("val", "test"):
        res[split] = {"R80": measure(con, split, high, med_r80), "S9": measure(con, split, high, med_s9)}
        print(f"\n[{split.upper()}]\n{pd.DataFrame(res[split]).to_string()}", flush=True)
    v = res["val"]
    reduction = 1 - v["S9"]["at_risk_row_share"] / v["R80"]["at_risk_row_share"]
    catch_drop = v["R80"]["failing_drive_capture"] - v["S9"]["failing_drive_capture"]
    res["decision"] = {"reduction": reduction, "catch_drop": catch_drop,
                       "criterion_1": reduction >= MIN_REDUCTION, "criterion_2": catch_drop <= MAX_CATCH_DROP}
    res["adopted"] = "S9" if res["decision"]["criterion_1"] and res["decision"]["criterion_2"] else "R80"
    res["adopted_cut"] = {"high_cutoff": high, "medium_cutoff": med_s9 if res["adopted"] == "S9" else med_r80}
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[DECISION] at-risk reduction {reduction:.1%}, failing-drive capture drop {catch_drop * 100:.1f} pts "
          f"-> adopted {res['adopted']} (MEDIUM cutoff {res['adopted_cut']['medium_cutoff']:.4f})", flush=True)


if __name__ == "__main__":
    main()
