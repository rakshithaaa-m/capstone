"""
Stage 2 — Data Cleaning (RAM-chunked, resumable)

Processes raw data ONE CHUNK AT A TIME (one quarter, matching Stage 1's
output layout of data/raw/<quarter_label>/*.csv) instead of loading every
raw CSV into a single DataFrame at once — the original all-at-once version
is the main RAM bottleneck on a 15.2GB laptop when working with several
years of Backblaze data.

Two-pass design, because one cleaning decision genuinely needs to see
every quarter before it can be made correctly:
  Pass 1 (cheap, streams through files, only accumulates per-column null
          counts — never holds a full quarter in memory): decides which
          SMART columns are near-empty ACROSS THE WHOLE DATASET. Deciding
          this per-quarter would be wrong — a column that's 95% null in
          2013 but well-populated by 2020 (attributes some manufacturers
          added later) would be incorrectly dropped if judged quarter by
          quarter. This pass has to run in full every time (it's read-only,
          so a rerun just costs time, not correctness).
  Pass 2: cleans each quarter fully (types, dedup, per-drive forward-fill,
          drop the globally-decided near-empty columns) and writes ONE
          cleaned parquet file per quarter to disk. Only one quarter's
          data is ever in memory at a time. Quarters whose output file
          already exists AND opens cleanly are skipped, so an interrupted
          run (crash, power loss) can be resumed without reprocessing
          everything. Writes are atomic (.tmp then rename) so a crash
          mid-write can never leave a corrupted parquet behind.

This does NOT combine the quarters into one dataframe — that's a separate,
deliberate decision left to Stage 3, since combining years of data into
one in-memory frame is exactly the next RAM question to test empirically
before committing to a strategy for it (see the pipeline discussion on
this — don't assume this alone makes the full 2013-2025 range RAM-safe).

Usage:
    python3 02_clean_data.py --raw_dir data/raw --out_dir data/processed/cleaned_by_quarter
"""
import argparse
import os
from pathlib import Path

import pandas as pd

REQUIRED_COLS = ["date", "serial_number", "model", "failure"]
NULL_DROP_THRESHOLD = 0.90  # drop a SMART column if more than this fraction is null, GLOBALLY


def discover_chunks(raw_dir: Path) -> dict[str, list[Path]]:
    """Groups raw CSVs by quarter. Matches Stage 1's actual output layout
    (data/raw/<quarter_label>/*.csv); falls back to treating each top-level
    CSV as its own chunk if no subfolders exist."""
    raw_dir = Path(raw_dir)
    subdirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    chunks: dict[str, list[Path]] = {}

    if subdirs:
        for d in subdirs:
            csvs = sorted(d.rglob("*.csv"))
            if csvs:
                chunks[d.name] = csvs
    else:
        for p in sorted(raw_dir.glob("*.csv")):
            chunks[p.stem] = [p]

    if not chunks:
        raise FileNotFoundError(
            f"No CSVs found under {raw_dir}, either directly or in subfolders. "
            f"Did Stage 1 run and extract successfully?"
        )
    return chunks


def is_valid_output(path: Path) -> bool:
    """A quarter is considered already-done only if its parquet file
    exists AND actually opens (reads its metadata) without error. A
    truncated file left behind by a crash mid-write will fail this check
    and get reprocessed, not silently skipped."""
    if not path.exists():
        return False
    try:
        # Reading just the schema/footer is enough to catch truncated/
        # corrupt files without loading the whole quarter into memory.
        pd.read_parquet(path, columns=[]).shape
        return True
    except Exception:
        return False


def scan_null_fractions(chunks: dict[str, list[Path]]) -> pd.Series:
    """Pass 1: accumulates null counts and row counts per SMART column
    across ALL chunks without ever holding a full chunk's data — reads
    each file, immediately reduces it to scalar counts, discards it."""
    null_counts: dict[str, int] = {}
    total_rows = 0
    smart_cols_seen: set[str] = set()

    n_files_total = sum(len(paths) for paths in chunks.values())
    n_files_done = 0

    for label, paths in chunks.items():
        for p in paths:
            df = pd.read_csv(p, low_memory=False)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{p} is missing required column(s): {missing}")
            smart_cols = [c for c in df.columns if c.startswith("smart_")]
            smart_cols_seen.update(smart_cols)
            for c in smart_cols:
                null_counts[c] = null_counts.get(c, 0) + int(df[c].isna().sum())
            total_rows += len(df)
            del df  # explicit, this loop must never accumulate memory across files

            n_files_done += 1
            if n_files_done % 25 == 0 or n_files_done == n_files_total:
                pct = 100 * n_files_done / n_files_total
                print(f"[SCAN] {n_files_done}/{n_files_total} files ({pct:.1f}%) | "
                      f"{total_rows:,} rows so far", flush=True)

    null_frac = pd.Series({c: null_counts.get(c, 0) / total_rows for c in smart_cols_seen})
    print(f"[SCAN] {total_rows:,} rows scanned across {len(chunks)} chunk(s) for null-fraction decisions")
    return null_frac


def clean_chunk(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    # format="mixed" lets pandas infer the date format PER ROW instead of
    # assuming the whole column matches one format. The raw CSVs mix
    # "YYYY-MM-DD" with occasional "M/D/YY" style strings (e.g. "2/25/18"),
    # which crashes the old fixed-format parsing the moment it hits one.
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["serial_number"] = df["serial_number"].astype(str)
    df["failure"] = df["failure"].fillna(0).astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=["serial_number", "date"], keep="last")
    n_dupes = before - len(df)

    if drop_cols:
        present = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=present)

    smart_cols = [c for c in df.columns if c.startswith("smart_")]
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    # Per-drive forward fill within THIS chunk only. A drive's very first
    # rows in the next chunk won't see this chunk's tail value — that's a
    # deliberate, acceptable simplification: Stage 3 already applies a
    # train-derived imputation statistic that mops up any leftover NaNs,
    # so a small number of un-forward-filled boundary rows here is not a
    # correctness problem, just a minor completeness one.
    df[smart_cols] = df.groupby("serial_number")[smart_cols].ffill()

    return df, n_dupes


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Backblaze SMART data, one chunk (quarter) at a time")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed/cleaned_by_quarter")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = discover_chunks(raw_dir)
    print(f"[DISCOVER] {len(chunks)} chunk(s): {list(chunks.keys())}")

    # ---- Resume check: which quarters are already validly written? ----
    already_done = []
    remaining = {}
    for label, paths in chunks.items():
        out_path = out_dir / f"cleaned_{label}.parquet"
        if is_valid_output(out_path):
            already_done.append(label)
        else:
            remaining[label] = paths

    print(f"[RESUME] {len(already_done)}/{len(chunks)} chunk(s) already cleaned and valid, "
          f"will be skipped: {already_done}")
    print(f"[RESUME] {len(remaining)} chunk(s) remaining to process: {list(remaining.keys())}")

    if not remaining:
        print("[DONE] Nothing left to do — all chunks already cleaned.")
        return

    # Pass 1 still scans ALL chunks (including already-done ones), because
    # the near-empty-column decision is global across the whole dataset,
    # not just the remaining quarters. This is read-only and doesn't write
    # anything, so it's just time spent, not risk.
    null_frac = scan_null_fractions(chunks)
    drop_cols = null_frac[null_frac > NULL_DROP_THRESHOLD].index.tolist()
    print(f"[SCAN] Dropping {len(drop_cols)} near-empty SMART column(s) globally "
          f"(>{NULL_DROP_THRESHOLD:.0%} null across ALL chunks): {drop_cols}")

    written_paths = []
    total_rows_out = 0
    total_dupes = 0
    total_failures = 0

    n_remaining = len(remaining)
    for i, (label, paths) in enumerate(remaining.items(), start=1):
        print(f"[{i}/{n_remaining}] Processing {label}...", flush=True)

        frames = [pd.read_csv(p, low_memory=False) for p in paths]
        chunk_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        del frames

        cleaned, n_dupes = clean_chunk(chunk_df, drop_cols)
        del chunk_df

        out_path = out_dir / f"cleaned_{label}.parquet"
        tmp_path = out_path.with_suffix(".parquet.tmp")
        cleaned.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)  # atomic on Windows and POSIX
        written_paths.append(out_path)

        total_rows_out += len(cleaned)
        total_dupes += n_dupes
        total_failures += int(cleaned["failure"].sum())
        print(f"[{label}] {len(cleaned):,} rows written -> {out_path} "
              f"({n_dupes} dupes dropped, {int(cleaned['failure'].sum())} failures)")
        del cleaned

    print(f"\n[DONE] {len(written_paths)} chunk file(s) written this run to {out_dir} "
          f"({len(already_done)} more were already done and skipped)")
    print(f"       total rows (this run): {total_rows_out:,} | total dupes dropped: {total_dupes:,} | "
          f"total failure rows: {total_failures:,}")
    print(f"       NOTE: these are still SEPARATE per-quarter files — Stage 3 combines them, "
          f"see that stage's own memory-handling notes before running it on a large date range.")


if __name__ == "__main__":
    main()