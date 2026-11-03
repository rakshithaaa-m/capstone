"""Relabel the last two weeks of the test set (2025-12-18 to 2025-12-31).

When the test labels were built, failures after 2025-12-31 were not yet visible, so a
drive that failed in early January 2026 had its December rows labelled 0 although they
fall inside the 14-day horizon. The 2026 data (28_fresh_2026.py partitions) now shows
those failures. Same rule as Stage 3: positive when the failure date is 1-14 days ahead.

Only labels change (0 -> 1); scores, features and rows are untouched. The flipped
(serial_number, date) keys are saved, so the change is reversible.

    python 30_relabel_dec2025.py            # build keys, rewrite the three label files
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb

KEYS = Path("data/processed/relabel_dec2025_keys.parquet")
PARTS = "data/processed/fresh2026/partitions/*/*.parquet"
START = "2025-12-18"
# file -> (reader, label column)
FILES = {
    Path("results/lgbm_test_probs.csv"): ("read_csv", "failure_within_14d"),
    Path("results/risk_tiers_test.csv"): ("read_csv", "target"),
    Path("data/processed/features_test.parquet"): ("read_parquet", "failure_within_14d"),
    Path("results/hybrid_test_scores.csv"): ("read_csv", "target"),
}
# bootstrap cache used by 26/27: drive_id = rank of serial_number (sorted) - 1
CACHE_ROWS = Path("results_fusion/_pb_confirm_joined.parquet")
CACHE_DRIVES = Path("results_fusion/_pb_confirm_drives.parquet")


def _duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'; SET threads=4")
    con.execute(f"SET temp_directory='{(Path.home() / 'duck_spool').as_posix()}'")
    return con


def build_keys(con) -> None:
    con.execute(f"""
        COPY (
          WITH f26 AS (SELECT serial_number, max(CAST(date AS DATE)) AS fd FROM read_parquet('{PARTS}')
                       WHERE CAST(failure AS DOUBLE) = 1 AND CAST(date AS DATE) >= DATE '2026-01-01' GROUP BY 1)
          SELECT t.serial_number, CAST(t.date AS DATE) AS date
          FROM read_csv('results/lgbm_test_probs.csv') t JOIN f26 USING (serial_number)
          WHERE CAST(t.date AS DATE) >= DATE '{START}' AND t.failure_within_14d = 0
            AND date_diff('day', CAST(t.date AS DATE), f26.fd) BETWEEN 1 AND 14
          ORDER BY 1, 2
        ) TO '{KEYS.as_posix()}' (FORMAT parquet)""")
    n, d = con.execute(f"SELECT count(*), count(DISTINCT serial_number) FROM '{KEYS.as_posix()}'").fetchone()
    print(f"[KEYS] {n:,} rows on {d:,} drives -> {KEYS}", flush=True)


def rewrite(con, path: Path, reader: str, label: str) -> None:
    t0 = time.time()
    src = f"{reader}('{path.as_posix()}')"
    done = con.execute(f"""SELECT count(*) FROM {src} t JOIN read_parquet('{KEYS.as_posix()}') k
                           ON k.serial_number = t.serial_number AND k.date = CAST(t.date AS DATE)
                           WHERE t.{label} = 1""").fetchone()[0]
    if done:
        print(f"[SKIP] {path}: already relabelled ({done:,} key rows positive)", flush=True)
        return
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    sel = ", ".join(
        f"CASE WHEN k.serial_number IS NOT NULL THEN CAST(1 AS {con.execute(f'SELECT typeof({label}) FROM {src} LIMIT 1').fetchone()[0]}) "
        f"ELSE t.{label} END AS {label}" if c == label else f't."{c}"' for c in cols)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fmt = "(FORMAT parquet)" if reader == "read_parquet" else "(HEADER, DELIMITER ',')"
    con.execute(f"""
        COPY (SELECT {sel} FROM {src} t
              LEFT JOIN read_parquet('{KEYS.as_posix()}') k
                ON k.serial_number = t.serial_number AND k.date = CAST(t.date AS DATE))
        TO '{tmp.as_posix()}' {fmt}""")
    before = con.execute(f"SELECT count(*), sum({label}) FROM {src}").fetchone()
    after = con.execute(f"SELECT count(*), sum({label}) FROM {reader}('{tmp.as_posix()}')").fetchone()
    n_keys = con.execute(f"SELECT count(*) FROM '{KEYS.as_posix()}'").fetchone()[0]
    assert after[0] == before[0], f"{path}: row count changed {before[0]} -> {after[0]}"
    assert after[1] - before[1] == n_keys, f"{path}: positives {before[1]} -> {after[1]}, expected +{n_keys}"
    tmp.replace(path)
    print(f"[REWRITE] {path}: {before[0]:,} rows, positives {int(before[1]):,} -> {int(after[1]):,} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


def rewrite_cache(con) -> None:
    con.execute(f"""CREATE TEMP TABLE kid AS
        SELECT r.drive_id, k.date FROM read_parquet('{KEYS.as_posix()}') k JOIN (
          SELECT serial_number, CAST(dense_rank() OVER (ORDER BY serial_number) - 1 AS INTEGER) AS drive_id
          FROM (SELECT DISTINCT serial_number FROM read_csv('results/lgbm_test_probs.csv'))) r USING (serial_number)""")
    n_keys = con.execute("SELECT count(*) FROM kid").fetchone()[0]
    rows = f"read_parquet('{CACHE_ROWS.as_posix()}')"
    if con.execute(f"SELECT count(*) FROM {rows} t JOIN kid USING (drive_id, date) WHERE t.y = 1").fetchone()[0]:
        print(f"[SKIP] {CACHE_ROWS}: already relabelled", flush=True)
        return
    before = con.execute(f"SELECT count(*), sum(y) FROM {rows}").fetchone()
    tmp = CACHE_ROWS.with_suffix(".parquet.tmp")
    con.execute(f"""COPY (SELECT t.drive_id, t.date, t.prob, t.partner,
                         CAST(CASE WHEN kid.drive_id IS NOT NULL THEN 1 ELSE t.y END AS TINYINT) AS y
                         FROM {rows} t LEFT JOIN kid USING (drive_id, date) ORDER BY t.drive_id, t.date)
                    TO '{tmp.as_posix()}' (FORMAT parquet)""")
    after = con.execute(f"SELECT count(*), sum(y) FROM read_parquet('{tmp.as_posix()}')").fetchone()
    assert after[0] == before[0] and after[1] - before[1] == n_keys, (before, after, n_keys)
    tmp.replace(CACHE_ROWS)
    print(f"[REWRITE] {CACHE_ROWS}: positives {int(before[1]):,} -> {int(after[1]):,}", flush=True)

    drives = f"read_parquet('{CACHE_DRIVES.as_posix()}')"
    b = con.execute(f"SELECT count(*), sum(ever_fail) FROM {drives}").fetchone()
    tmp = CACHE_DRIVES.with_suffix(".parquet.tmp")
    con.execute(f"""COPY (SELECT d.drive_id, d.prob, d.partner,
                         CAST(CASE WHEN d.drive_id IN (SELECT drive_id FROM kid) THEN 1 ELSE d.ever_fail END AS TINYINT) AS ever_fail
                         FROM {drives} d ORDER BY d.drive_id) TO '{tmp.as_posix()}' (FORMAT parquet)""")
    a = con.execute(f"SELECT count(*), sum(ever_fail) FROM read_parquet('{tmp.as_posix()}')").fetchone()
    assert a[0] == b[0]
    tmp.replace(CACHE_DRIVES)
    print(f"[REWRITE] {CACHE_DRIVES}: failing drives {int(b[1]):,} -> {int(a[1]):,}", flush=True)


def recompute_lgbm_metrics() -> None:
    """results/lgbm_metrics.csv (Stage 6 test metrics) on the relabelled test set; the
    pre-relabel row is kept as results/lgbm_metrics_prerelabel.csv."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    path, old = Path("results/lgbm_metrics.csv"), Path("results/lgbm_metrics_prerelabel.csv")
    if not old.exists():
        path.replace(old)
    prev = pd.read_csv(old).iloc[0]
    thr = float(prev["threshold"])
    con = _duck()
    r = con.execute("SELECT CAST(failure_within_14d AS TINYINT) y, CAST(lgbm_prob AS FLOAT) p "
                    "FROM read_csv('results/lgbm_test_probs.csv')").fetchnumpy()
    y, p = np.asarray(r["y"]), np.asarray(r["p"])
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int(y.sum()) - tp, int((y == 0).sum()) - fp
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    pd.DataFrame([{"roc_auc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
                   "f1": 2 * prec * rec / (prec + rec), "precision": prec, "recall": rec,
                   "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": thr}]).to_csv(path, index=False)
    print(f"[METRICS] {path}: AUPRC {prev['auprc']:.5f} -> {average_precision_score(y, p):.5f}, "
          f"precision {prec:.4f}, recall {rec:.4f}", flush=True)


def main() -> None:
    con = _duck()
    if not KEYS.exists():
        build_keys(con)
    for path, (reader, label) in FILES.items():
        rewrite(con, path, reader, label)
    rewrite_cache(con)
    recompute_lgbm_metrics()


if __name__ == "__main__":
    main()
