"""Alert-policy study for the shipped model: persistence and daily top-k.

Pre-registration: results_alerts2026/PREREGISTRATION.md. Uses model A's scores already
written by 29_retrain_2026.py (validation 2025-07-01..12-17, test 2026-01-01..06-16).

    python 32_alert_policies.py fit       # k* on validation, descriptive k grid
    python 32_alert_policies.py evaluate  # per-drive tables, paired bootstrap, verdicts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PREDS = Path("data/processed/retrain2026")
OUT = Path("results_alerts2026")
THRESHOLD = 0.73          # shipped threshold of model A (results_retrain2026/results.json)
K_GRID = [50, 100, 200, 300, 500, 1000, 2000]
MAX_CATCH_DROP = 0.05
SEED = 20261104


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=4; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(Path.home() / 'duck_spool').as_posix()}'")
    return con


def _base(split: str) -> str:
    return (f"SELECT serial_number AS s, CAST(date AS DATE) AS d, CAST(y AS INTEGER) AS y, p_A AS p "
            f"FROM read_parquet('{(PREDS / f'preds_{split}_A.parquet').as_posix()}')")


# ------------------------------------------------------------------------ fit
def fit() -> None:
    """k* = smallest k whose validation catch rate reaches the baseline's."""
    con = _duck()
    con.execute(f"CREATE TEMP TABLE v AS SELECT *, row_number() OVER (PARTITION BY d ORDER BY p DESC) AS rk "
                f"FROM ({_base('val')})")
    fail = con.execute("""SELECT s, min(CASE WHEN y = 1 THEN rk END) AS best_rank,
                                 max(CASE WHEN y = 1 AND p >= ? THEN 1 ELSE 0 END) AS caught_b
                          FROM v GROUP BY s HAVING max(y) = 1""", [THRESHOLD]).fetchdf()
    n_days, n_rows, n_pos = con.execute("SELECT count(DISTINCT d), count(*), sum(y) FROM v").fetchone()
    catch_b = fail["caught_b"].mean()
    ranks = np.sort(fail["best_rank"].to_numpy())
    # catch(k) = share of failing drives whose best positive-row rank is <= k
    k_star = int(ranks[int(np.ceil(catch_b * len(ranks))) - 1])
    grid = []
    for k in sorted(set(K_GRID + [k_star])):
        tp = con.execute("SELECT sum(y) FROM v WHERE rk <= ?", [k]).fetchone()[0]
        grid.append({"k": k, "catch_rate": float((ranks <= k).mean()), "row_precision": tp / (k * n_days),
                     "alerts_per_day": k})
    b_rows, b_tp = con.execute("SELECT count(*), sum(y) FROM v WHERE p >= ?", [THRESHOLD]).fetchone()
    fitted = {"threshold": THRESHOLD, "k_star": k_star, "val_catch_rate_B": float(catch_b),
              "val_alerts_per_day_B": b_rows / n_days, "val_row_precision_B": b_tp / b_rows,
              "val_failing_drives": int(len(fail)), "val_days": int(n_days), "val_rows": int(n_rows),
              "val_positive_rows": int(n_pos)}
    OUT.mkdir(exist_ok=True)
    (OUT / "fitted_on_validation.json").write_text(json.dumps(fitted, indent=2))
    pd.DataFrame(grid).to_csv(OUT / "topk_grid_val.csv", index=False)
    print(f"[FIT] baseline catch {catch_b:.3f} at {b_rows / n_days:,.0f} alerts/day -> k* = {k_star}", flush=True)
    print(pd.DataFrame(grid).to_string(index=False), flush=True)


# ------------------------------------------------------------------- per drive
POLICIES = ["B", "P2", "P3", "K"]


def per_drive(split: str, k: int) -> pd.DataFrame:
    """One row per drive: episodes, true episodes, alerted rows, true alerted rows,
    caught, failing -- for every policy."""
    t0 = time.time()
    con = _duck()
    t = THRESHOLD
    con.execute(f"""
        CREATE TEMP TABLE a AS
        SELECT s, d, y,
               p >= {t} AS a_B,
               p >= {t} AND l1p >= {t} AND d - l1d = 1 AS a_P2,
               p >= {t} AND l1p >= {t} AND d - l1d = 1 AND l2p >= {t} AND d - l2d = 2 AS a_P3,
               rk <= {k} AS a_K
        FROM (SELECT *, lag(p, 1) OVER w AS l1p, lag(d, 1) OVER w AS l1d,
                        lag(p, 2) OVER w AS l2p, lag(d, 2) OVER w AS l2d,
                        row_number() OVER (PARTITION BY d ORDER BY p DESC) AS rk
              FROM ({_base(split)}) WINDOW w AS (PARTITION BY s ORDER BY d))""")
    starts = ", ".join(
        f"a_{x} AND NOT coalesce(lag(a_{x}) OVER w AND d - lag(d) OVER w = 1, false) AS st_{x}" for x in POLICIES)
    aggs = ", ".join(
        f"sum(CAST(st_{x} AS INTEGER)) AS ep_{x}, sum(CASE WHEN st_{x} AND y = 1 THEN 1 ELSE 0 END) AS tep_{x}, "
        f"sum(CAST(a_{x} AS INTEGER)) AS al_{x}, sum(CASE WHEN a_{x} AND y = 1 THEN 1 ELSE 0 END) AS tal_{x}, "
        f"max(CASE WHEN a_{x} AND y = 1 THEN 1 ELSE 0 END) AS cau_{x}" for x in POLICIES)
    df = con.execute(f"""
        SELECT s, max(y) AS failing, {aggs}
        FROM (SELECT *, {starts} FROM a WINDOW w AS (PARTITION BY s ORDER BY d))
        GROUP BY s ORDER BY s""").fetchdf()
    # a drive with fewer rows than the persistence window has no P2/P3 flags (NULL sums)
    df = df.fillna(0)
    n_days = con.execute("SELECT count(DISTINCT d) FROM a").fetchone()[0]
    df.attrs["n_days"] = n_days
    print(f"[DRIVES] {split}: {len(df):,} drives, {int(df.failing.sum()):,} failing, {n_days} days "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    n_days, fail = df.attrs["n_days"], df["failing"] == 1
    rows = []
    for x in POLICIES:
        rows.append({"policy": x,
                     "new_alert_precision": df[f"tep_{x}"].sum() / df[f"ep_{x}"].sum(),
                     "catch_rate": df.loc[fail, f"cau_{x}"].mean(),
                     "new_alerts_per_day": df[f"ep_{x}"].sum() / n_days,
                     "alerted_rows_per_day": df[f"al_{x}"].sum() / n_days,
                     "row_precision": df[f"tal_{x}"].sum() / df[f"al_{x}"].sum()})
    return pd.DataFrame(rows)


def paired_bootstrap(df: pd.DataFrame, x: str, base: str = "B", n_boot: int = 1000) -> dict:
    """Resample drives; ratio estimators recomputed per replicate."""
    rng = np.random.default_rng(SEED)
    n = len(df)
    ep_x, tep_x, ep_b, tep_b = (df[c].to_numpy(float) for c in (f"ep_{x}", f"tep_{x}", f"ep_{base}", f"tep_{base}"))
    f = df["failing"].to_numpy(float)
    cau_x, cau_b = df[f"cau_{x}"].to_numpy(float) * f, df[f"cau_{base}"].to_numpy(float) * f
    dp, dc = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        dp[b] = (w @ tep_x) / (w @ ep_x) - (w @ tep_b) / (w @ ep_b)
        dc[b] = ((w @ cau_x) - (w @ cau_b)) / (w @ f)
    point_p = tep_x.sum() / ep_x.sum() - tep_b.sum() / ep_b.sum()
    point_c = (cau_x.sum() - cau_b.sum()) / f.sum()
    return {"delta_precision": point_p, "precision_ci": list(np.percentile(dp, [2.5, 97.5])),
            "delta_catch": point_c, "catch_ci": list(np.percentile(dc, [2.5, 97.5])), "n_boot": n_boot}


# -------------------------------------------------------------------- evaluate
def evaluate(n_boot: int) -> None:
    fitted = json.loads((OUT / "fitted_on_validation.json").read_text())
    k = fitted["k_star"]
    res = {"preregistration": str(OUT / "PREREGISTRATION.md"), "k_star": k, "threshold": THRESHOLD}
    for split in ("val", "test"):
        df = per_drive(split, k)
        s = summarise(df)
        s.to_csv(OUT / f"policies_{split}.csv", index=False)
        print(f"\n[{split.upper()}]\n{s.to_string(index=False)}", flush=True)
        res[f"{split}_summary"] = s.set_index("policy").to_dict(orient="index")
        if split == "test":
            for x in ("P2", "K"):
                bt = paired_bootstrap(df, x, n_boot=n_boot)
                bt["improves_precision"] = bool(bt["precision_ci"][0] > 0)
                bt["adoptable"] = bool(bt["improves_precision"] and bt["delta_catch"] >= -MAX_CATCH_DROP)
                res[f"test_{x}_vs_B"] = bt
                print(f"[TEST] {x} vs B: {bt}", flush=True)
    ok = [x for x in ("P2", "K") if res[f"test_{x}_vs_B"]["adoptable"]]
    res["adopted"] = (max(ok, key=lambda x: res["val_summary"][x]["new_alert_precision"]) if ok else "B")
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[DECISION] adopted policy: {res['adopted']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fit", "evaluate"])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    fit() if args.stage == "fit" else evaluate(args.boot)


if __name__ == "__main__":
    main()
